"""Full R3 material domains, sum links, styles and atomic publication."""
from copy import deepcopy
import numpy as np
import pytest
from kika.nuclear_data.model import (ReactionSuite,Reaction,ReactionId,CrossSection,CrossSectionSum,
    Summands,Add,Evaluated,PhysicalQuantity,RangeQuantity,Background,ResonancesWithBackground,
    Reference,XYs1d,Regions1d,CrossSectionReconstructed,ConversionReport)
from kika.processing.resonances import (NeutronContext,reconstruct_suite,attach_reconstruction,
    ReactionKey,ReconstructionOptions,ReconstructionConvergenceError,UnsupportedResonanceError)
from test_resonance_foundations import model
from test_resonance_tabulation import AXES


def curve(lo,hi,value):return XYs1d([lo,hi],[value,value],axes=AXES,label='eval')


def href(label):return f"/reactionSuite/reactions/reaction[@label='{label}']/crossSection"


def suite_model():
    suite=ReactionSuite('R3 unit','n','Fe56')
    suite.styles.add(Evaluated('eval',temperature=PhysicalQuantity(0.,'K'),
        projectileEnergyDomain=RangeQuantity(10.,1000.,'eV')))
    suite.resonances=model()
    suite.resonances.resolved[0].domainMin,suite.resonances.resolved[0].domainMax=10.,300.
    for label,mt,fast in (('elastic',2,2.),('capture',102,1.)):
        form=ResonancesWithBackground(Background(resolvedRegion=curve(10.,300.,0.),
            fastRegion=curve(300.,1000.,fast)),resonanceRegionHref='/reactionSuite/resonances/resolved',label='eval')
        suite.reactions.append(Reaction(ReactionId(label,ENDF_MT=mt),CrossSection({'eval':form})))
    suite.reactions.append(Reaction(ReactionId('extra',ENDF_MT=16),CrossSection({'eval':curve(10.,1000.,.5)})))
    total=ResonancesWithBackground(Background(resolvedRegion=curve(10.,300.,.5),fastRegion=curve(300.,1000.,3.5)),label='eval')
    suite.sums.append(CrossSectionSum(ReactionId('total',ENDF_MT=1),CrossSection({'eval':total}),
        summands=Summands([Add(href(label)) for label in ('elastic','capture','extra')])))
    return suite


def run(suite=None,**kwargs):return reconstruct_suite(suite_model() if suite is None else suite,NeutronContext(56.,0.),**kwargs)


def test_constant_projection_preserves_full_tables_and_checks(monkeypatch):
    import kika.processing.resonances.suite as module
    reference=run()
    actual_linearize=module.linearize
    declarations=[]
    def projected(*args,**kwargs):
        declarations.append(kwargs.get('constants',{}))
        return actual_linearize(*args,**kwargs)
    monkeypatch.setattr(module,'linearize',projected)
    optimized=run()
    assert any(ReactionKey('reactions','extra') in c for c in declarations)
    def original(*args,**kwargs):
        kwargs.pop('constants',None)
        return actual_linearize(*args,**kwargs)
    monkeypatch.setattr(module,'linearize',original)
    baseline=run()
    for key,form in optimized.forms.items():
        for a,b in zip(getattr(form,'function1ds',[form]),
                       getattr(baseline.forms[key],'function1ds',[baseline.forms[key]])):
            np.testing.assert_array_equal(a.xs,b.xs)
            np.testing.assert_array_equal(a.ys,b.ys)
    for a,b in zip(optimized.report['regions'],baseline.report['regions']):
        assert a['evaluations']==b['evaluations']
        assert a['refinement_maxima']==b['refinement_maxima']
        assert a['verification_maxima']==b['verification_maxima']
    assert optimized.report['points']==reference.report['points']


def test_linear_background_deferral_keeps_source_knots_and_physical_sums(monkeypatch):
    import kika.processing.resonances.suite as module
    suite=suite_model()
    suite.reactions['extra'].crossSection['eval']=XYs1d(
        [10.,37.,110.,300.,1000.],[.3,1.,.1,.9,1.8],axes=AXES,label='eval')
    actual_linearize=module.linearize;declarations=[]
    def track(*args,**kwargs):
        declarations.append(kwargs.get('deferred',{}))
        return actual_linearize(*args,**kwargs)
    monkeypatch.setattr(module,'linearize',track)
    optimized=run(suite)
    extra=ReactionKey('reactions','extra')
    assert any(extra in d for d in declarations)
    assert all(ReactionKey('reactions','elastic') not in d for d in declarations[:1])
    def ordinary(*args,**kwargs):
        kwargs.pop('deferred',None)
        return actual_linearize(*args,**kwargs)
    monkeypatch.setattr(module,'linearize',ordinary)
    baseline=run(suite)
    for key,form in optimized.forms.items():
        for a,b in zip(form.function1ds,baseline.forms[key].function1ds):
            np.testing.assert_array_equal(a.xs,b.xs)
            np.testing.assert_array_equal(a.ys,b.ys)
    attach_reconstruction(suite,optimized)
    assert max(optimized.verify_suite(suite).values())<=1


def test_shared_backgrounds_keep_sums_and_independent_readers(monkeypatch):
    from kika.nuclear_data.model.enums import Interpolation
    from kika.algebra.prepared import _SharedLinearEvaluator
    suite=suite_model()
    def mixed(values):
        return Regions1d([XYs1d([10.,100.],values[:2],axes=AXES,
                                interpolation=Interpolation.loglog),
                          XYs1d([100.,1000.],values[1:],axes=AXES)],label='eval')
    suite.reactions['extra'].crossSection['eval']=mixed([.5,.7,1.2])
    suite.reactions.append(Reaction(ReactionId('second-extra',ENDF_MT=17),
        CrossSection({'eval':mixed([.4,.6,.9])})))
    suite.sums[1].summands.append(Add(href('second-extra')))
    result=run(suite)
    segment=result._segments[0]
    calls=[];original=_SharedLinearEvaluator.__call__
    def track(self,q,**kwargs):
        value=original(self,q,**kwargs)
        if value is not None:calls.append(len(np.atleast_1d(q)))
        return value
    monkeypatch.setattr(_SharedLinearEvaluator,'__call__',track)
    q=np.array([110.,200.,250.])
    fast=result._evaluate_segment(segment,q)
    assert calls
    reference=result._evaluate_segment(segment,q,shared_linear=False)
    for key in fast:np.testing.assert_array_equal(fast[key],reference[key])
    calls.clear()
    result._evaluate_segment(segment,q,physical_witness={})
    assert not calls


def test_complete_domains_sums_and_nonresonant_data():
    suite=suite_model();result=run(suite)
    assert suite.styles.labels==['eval']
    assert result.report['domain']==(10.,1000.)
    e=np.array([20.,100.,np.nextafter(300.,10.),300.,500.,1000.])
    values=result.evaluate(e)
    el,cap,extra=(values[ReactionKey('reactions',k)] for k in ('elastic','capture','extra'))
    np.testing.assert_allclose(values[ReactionKey('sums','total')],el+cap+extra)
    np.testing.assert_allclose(el[3:],[2.,2.,2.])
    np.testing.assert_allclose(cap[3:],[1.,1.,1.])
    np.testing.assert_allclose(extra,.5)
    assert el[2]!=el[3]  # Boundary left limit and fast-region ownership.
    assert set(result.forms_by_mt)=={1,2,16,102}


def test_attach_keeps_eval_adds_style_and_stores_independent_forms():
    suite=suite_model();evaluated=suite.reactions['elastic'].crossSection['eval']
    result=run(suite)
    attach_reconstruction(suite,result)
    assert suite.styles.labels==['eval','recon']
    assert isinstance(suite.styles['recon'],CrossSectionReconstructed)
    assert suite.styles['recon'].derivedFrom=='eval'
    assert max(result.verify_suite(suite).values())<=1
    assert suite.reactions['elastic'].crossSection['eval'] is evaluated
    np.testing.assert_array_equal(suite.reactions['elastic'].crossSection['eval'].background.fastRegion.ys,evaluated.background.fastRegion.ys)
    result.forms_by_mt[2].function1ds[0].ys*=2
    assert max(result.verify_suite(suite).values())<=1  # Published copy is independent.


@pytest.mark.parametrize('edit',[
    lambda s:s.resonances.resolved[0].formalism.resonanceParameters.spinGroups[0].resonances.__setitem__(0,
        type(s.resonances.resolved[0].formalism.resonanceParameters.spinGroups[0].resonances[0])(101.,.5,.3,.1,.2,0.)),
    lambda s:s.reactions['elastic'].crossSection['eval'].background.fastRegion.ys.__setitem__(0,9.),
    lambda s:s.sums[1].summands.append(Add(href('elastic'))),
])
def test_stale_input_rejected_without_partial_attachment(edit):
    suite=suite_model();result=run(suite);edit(suite)
    with pytest.raises(ValueError,match='changed'):attach_reconstruction(suite,result)
    assert suite.styles.labels==['eval']
    assert all('recon' not in r.crossSection for r in list(suite.reactions)+list(suite.sums))


def test_corrupted_result_fails_before_any_publication():
    suite=suite_model();result=run(suite)
    result.forms_by_mt[102].function1ds[0].ys*=1.1
    with pytest.raises(ReconstructionConvergenceError,match='budget'):attach_reconstruction(suite,result)
    assert suite.styles.labels==['eval']
    assert all('recon' not in r.crossSection for r in suite.reactions)


def test_reference_resolution_and_cycle_failure():
    suite=suite_model()
    suite.reactions['extra'].crossSection['eval']=Reference(href('elastic'),label='eval')
    suite.sums[1].crossSection['eval']=ResonancesWithBackground(Background(
        resolvedRegion=curve(10.,300.,0.),fastRegion=curve(300.,1000.,5.)),label='eval')
    result=run(suite)
    np.testing.assert_allclose(result.evaluate([100.,500.])[ReactionKey('reactions','extra')],[0.,2.])
    suite.reactions['elastic'].crossSection['eval']=Reference(href('extra'))
    with pytest.raises(ValueError,match='cycle'):run(suite)


def test_inconsistent_source_sum_is_diagnosed_and_missing_graph_rejects():
    suite=suite_model()
    suite.sums[1].summands.summands.pop()
    result=run(suite)
    key=ReactionKey('sums','total')
    assert result.report['source_sum_error_ratios'][key]>1
    assert result.report['source_sum_policy']=='derive-from-leaves'
    values=result.evaluate([500.])
    np.testing.assert_allclose(values[key],values[ReactionKey('reactions','elastic')]+values[ReactionKey('reactions','capture')])
    suite=suite_model();suite.sums[1].summands=Summands()
    with pytest.raises(UnsupportedResonanceError,match='component graph'):run(suite)


def test_missing_source_region_and_conversion_loss_reject():
    suite=suite_model()
    suite.reactions['capture'].crossSection['eval'].background.fastRegion=curve(400.,1000.,1.)
    with pytest.raises(UnsupportedResonanceError,match='gap'):run(suite)
    suite=suite_model();suite.report=ConversionReport();suite.report.lost('missing channel')
    with pytest.raises(UnsupportedResonanceError,match='conversion'):run(suite)


def test_processed_style_cannot_be_reconstructed_twice():
    suite=suite_model();result=run(suite);attach_reconstruction(suite,result)
    with pytest.raises(ValueError,match='already includes'):run(suite,source_style='recon',label='recon2')


def test_full_gnds_written_read_and_verified(tmp_path):
    from kika.gnds.encode import writeReactionSuite
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    suite=suite_model();result=run(suite);attach_reconstruction(suite,result)
    xml,report=writeReactionSuite(suite)
    path=tmp_path/'r3.xml'
    xml.write(path)
    reloaded,report=readReactionSuite(Document.parse(path))
    assert max(result.verify_suite(reloaded).values())<=1


def test_endf_rounding_resamples_steep_flanks_without_mutating_source():
    from kika.endf.writers.assemble import _resample_reconstructed_endf
    from kika._records import format_endf_number_precise, parse_number
    from kika.algebra import evaluate
    suite=ReactionSuite('rounding','n','U238')
    x=np.array([16689.,16689.06538420718,16689.065603114654,16690.])
    y=200*(x-16689.065)+.8
    original=XYs1d(x,y,axes=AXES,label='recon')
    suite.reactions.append(Reaction(ReactionId('elastic',ENDF_MT=2),CrossSection({'recon':original})))
    rounded=_resample_reconstructed_endf(suite,'recon').reactions['elastic'].crossSection['recon']
    expected_x=np.array([parse_number(format_endf_number_precise(v)) for v in x])
    np.testing.assert_array_equal(rounded.xs,expected_x)
    np.testing.assert_allclose(rounded.ys,200*(expected_x-16689.065)+.8,atol=1e-12)
    probe=np.array([16689.065467822395])
    np.testing.assert_allclose(evaluate(rounded.xs,rounded.ys,2,probe),evaluate(x,y,2,probe),atol=1e-12)
    np.testing.assert_array_equal(original.xs,x)
    np.testing.assert_array_equal(original.ys,y)


@pytest.mark.parametrize('x',[[16689.,16689.0656001,16689.0656002,16690.],
                              [16689.0000001,16689.0656001,16690.]])
def test_endf_resampling_keeps_collisions_and_unrepresentable_boundaries(x):
    from kika.endf.writers.assemble import _resample_reconstructed_endf
    suite=ReactionSuite('rounding','n','U238')
    y=np.arange(len(x),dtype=float)
    form=XYs1d(x,y,axes=AXES,label='recon')
    suite.reactions.append(Reaction(ReactionId('elastic',ENDF_MT=2),CrossSection({'recon':form})))
    output=_resample_reconstructed_endf(suite,'recon').reactions['elastic'].crossSection['recon']
    np.testing.assert_array_equal(output.xs,x)
    np.testing.assert_array_equal(output.ys,y)


@pytest.mark.parametrize('mode',['cycle','duplicate','external','other_style'])
def test_native_sum_links_reject_invalid_graphs(mode):
    suite=suite_model()
    total=suite.sums[1]
    if mode=='cycle':total.summands=Summands([Add("/reactionSuite/sums/crossSectionSums/crossSectionSum[@label='total']/crossSection")])
    elif mode=='duplicate':total.summands.append(Add(href('elastic')))
    elif mode=='external':total.summands=Summands([Add('other.xml#'+href('elastic'))])
    else:total.summands=Summands([Add(href('elastic')+"/XYs1d[@label='processed']")])
    with pytest.raises((ValueError,UnsupportedResonanceError)):run(suite)


def test_competition_already_in_background_is_not_added_twice():
    from kika.nuclear_data.model.resonances import CompetitiveChannel
    suite=suite_model()
    group=suite.resonances.resolved[0].formalism.resonanceParameters.spinGroups[0]
    group.resonances[0].totalWidth=.7
    group.competitiveChannel=CompetitiveChannel(-20.,L=0,inEvaluatedBackground=True)
    suite.reactions.append(Reaction(ReactionId('competition',ENDF_MT=51),CrossSection({'eval':curve(10.,1000.,.6)})))
    suite.sums[1].summands.append(Add(href('competition')))
    total=suite.sums[1].crossSection['eval'].background
    total.resolvedRegion.ys+=.6;total.fastRegion.ys+=.6
    result=run(suite)
    np.testing.assert_allclose(result.evaluate([20.,100.,500.])[ReactionKey('reactions','competition')],.6)
    attach_reconstruction(suite,result)
    assert max(result.verify_suite(suite).values())<=1


def test_header_marks_complete_reconstructed_style_and_rejects_mixed_export():
    from pathlib import Path
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.endf.writers.assemble import _mf1Sections
    data=Path(__file__).resolve().parents[2]/'endf/tests/data/micro_fe56_structural.endf'
    source,_=decodeReactionSuite(read_endf(str(data),mf_numbers=[1,3]))
    suite=suite_model();result=run(suite);attach_reconstruction(suite,result)
    suite.provenance=source.provenance
    suite.resonances=None  # This unit test exercises header + MF3 assembly only.
    sections,_=_mf1Sections(suite,2631,ConversionReport(),'recon')
    header=next(section for mf,mt,section in sections if (mf,mt)==(1,451))
    assert header._lrp==2 and header._ldrv==1
    assert suite.provenance.headerFields['lrp']==1
    suite.reactions['capture'].crossSection.forms.pop('recon')
    with pytest.raises(ValueError,match='complete|missing|every|recon'):
        _mf1Sections(suite,2631,ConversionReport(),'recon')


def test_serialized_nonfinite_table_is_rejected():
    suite=suite_model();result=run(suite);attach_reconstruction(suite,result)
    suite.reactions['capture'].crossSection['recon'].function1ds[0].ys[0]=np.nan
    with pytest.raises(ReconstructionConvergenceError,match='invalid'):result.verify_suite(suite)



def test_relative_and_fragment_references_resolve_in_model():
    suite=suite_model()
    suite.reactions['extra'].crossSection['eval']=Reference("../../../reaction[@label='elastic']/crossSection",label='eval')
    suite.sums[1].crossSection['eval']=ResonancesWithBackground(Background(
        resolvedRegion=curve(10.,300.,0.),fastRegion=curve(300.,1000.,5.)),label='eval')
    suite.sums[1].summands.summands[0]=Add('#'+href('elastic'))
    result=run(suite)
    np.testing.assert_allclose(result.evaluate([100.,500.])[ReactionKey('reactions','extra')],[0.,2.])


def test_material_identity_and_options_are_part_of_certification():
    suite=suite_model();result=run(suite)
    tighter=run(suite,options=ReconstructionOptions(rtol=5e-4))
    assert tighter.report['normalized_sha256']!=result.report['normalized_sha256']
    assert tighter.report['source_sha256']==result.report['source_sha256']
    attach_reconstruction(suite,result)
    suite.target='Ni58'
    with pytest.raises(ReconstructionConvergenceError,match='identity'):result.verify_suite(suite)



def test_individually_accurate_exported_reactions_must_also_close_the_sum():
    suite=suite_model();result=run(suite);attach_reconstruction(suite,result)
    for reaction in suite.reactions:
        reaction.crossSection['recon'].function1ds[-1].ys*=.9994
    suite.sums[1].crossSection['recon'].function1ds[-1].ys*=1.0006
    # Each constant fast-region form is within its individual budget, but
    # the total and its components disagree by more than the requested budget.
    with pytest.raises(ReconstructionConvergenceError,match='sum does not close'):
        result.verify_suite(suite)
