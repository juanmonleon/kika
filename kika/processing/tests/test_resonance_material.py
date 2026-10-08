"""R7 complete materials: redundant diagnostics, empty MF2 and publication."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from kika.nuclear_data.model import (ReactionSuite,Reaction,ReactionId,CrossSection,
    CrossSectionSum,Summands,Add,Evaluated,RangeQuantity,XYs1d,Regions1d)
from kika.processing.resonances import (reconstruct_suite,attach_reconstruction,
    ReconstructionOptions,ReconstructionConvergenceError,UnsupportedResonanceError,ReactionKey)
from test_resonance_suite import AXES,suite_model,run,href


def nonresonant_suite():
    suite=ReactionSuite('R7 complete MF3','n','H1')
    suite.styles.add(Evaluated('eval',projectileEnergyDomain=RangeQuantity(1.,100.,'eV')))
    for label,mt,ys in [('elastic',2,[2.,4.]),('capture',102,[1.,2.])]:
        suite.reactions.append(Reaction(ReactionId(label,ENDF_MT=mt),CrossSection({
            'eval':XYs1d([1.,100.],ys,axes=AXES,label='eval')})))
    suite.sums.append(CrossSectionSum(ReactionId('total',ENDF_MT=1),CrossSection({
        'eval':XYs1d([1.,100.],[30.,60.],axes=AXES,label='eval')}),
        summands=Summands([Add(href('elastic')),Add(href('capture'))])))
    return suite


def test_mf3_without_resonances_needs_no_invented_mass_or_spin():
    suite=nonresonant_suite();before=deepcopy(suite.sums[1].crossSection['eval'].ys)
    result=reconstruct_suite(suite)
    assert result.report['context'] is None
    assert result.report['source_sum_policy']=='derive-from-leaves'
    assert result.report['source_sum_error_ratios'][ReactionKey('sums','total')]>1
    np.testing.assert_allclose(result.forms_by_mt[1].ys,[3.,6.])
    attach_reconstruction(suite,result)
    np.testing.assert_array_equal(suite.sums[1].crossSection['eval'].ys,before)
    assert max(result.verify_suite(suite).values())<=1


def test_scoped_conversion_is_retained_without_exempting_unknown_losses():
    from kika.nuclear_data.model import ConversionReport
    suite=nonresonant_suite();suite.report=ConversionReport()
    suite.report.unsupportedNode('energy metadata',unaffectedScopes=('cross-sections',))
    result=reconstruct_suite(suite)
    assert not result.report['source_conversion_report'].isClean
    assert result.report['source_conversion_report'] is not suite.report
    suite.report.lost('unknown cross-section effect')
    with pytest.raises(UnsupportedResonanceError) as failure:reconstruct_suite(suite)
    assert failure.value.category=='conversion-not-clean'


def test_resonant_preparation_uses_the_same_scoped_conversion_gate():
    from kika.nuclear_data.model import ConversionReport
    suite=suite_model();suite.report=ConversionReport()
    suite.report.unsupportedNode('energy metadata',unaffectedScopes=('cross-sections',))
    result=run(suite);attach_reconstruction(suite,result)
    assert max(result.verify_suite(suite).values())<=1


def test_nested_source_aggregates_are_compared_to_exclusive_leaves():
    suite=nonresonant_suite();suite.sums.append(CrossSectionSum(
        ReactionId('nonelastic',ENDF_MT=3),CrossSection({'eval':XYs1d([1.,100.],[10.,20.],axes=AXES,label='eval')}),
        summands=Summands([Add(href('capture'))])))
    suite.sums[1].summands=Summands([Add(href('elastic')),Add(
        "/reactionSuite/sums/crossSectionSums/crossSectionSum[@label='nonelastic']/crossSection")])
    result=reconstruct_suite(suite)
    assert len(result.report['source_sum_discrepancies'])==2
    np.testing.assert_allclose(result.forms_by_mt[1].ys,[3.,6.])
    np.testing.assert_allclose(result.forms_by_mt[3].ys,[1.,2.])
    worst=result.report['source_sum_discrepancies'][ReactionKey('sums','total')]
    assert worst['stated_b']==10*worst['components_b']


def test_total_graph_cannot_omit_an_active_resonance_partial():
    suite=suite_model()
    suite.sums[1].summands=Summands([Add(href('elastic')),Add(href('extra'))])
    with pytest.raises(UnsupportedResonanceError) as failure:run(suite)
    assert failure.value.category=='incomplete-total-graph'


def test_redundant_aggregate_grid_drives_diagnostics_but_not_refinement():
    suite=nonresonant_suite();x=np.geomspace(1.,100.,1001);y=np.full(len(x),30.)
    y[417]=10000.
    suite.sums[1].crossSection['eval']=XYs1d(x,y,axes=AXES,label='eval')
    result=reconstruct_suite(suite)
    assert result.report['points']==2
    worst=result.report['source_sum_discrepancies'][ReactionKey('sums','total')]
    assert worst['energy_eV']==x[417] and worst['stated_b']==10000.
    assert worst['side']=='point'


def test_nonresonant_partial_threshold_is_a_step_in_total():
    suite=nonresonant_suite()
    suite.reactions['capture'].crossSection['eval']=XYs1d([20.,100.],[1.,1.],axes=AXES,label='eval')
    result=reconstruct_suite(suite);attach_reconstruction(suite,result)
    assert isinstance(result.forms_by_mt[1],Regions1d)
    e=[np.nextafter(20.,1.),20.]
    values=result.evaluate(e)
    np.testing.assert_array_equal(values[ReactionKey('reactions','capture')],[0.,1.])
    assert max(result.verify_suite(suite).values())<=1


def test_repeated_identical_threshold_copies_do_not_reject_the_material():
    suite=nonresonant_suite()
    suite.reactions['capture'].crossSection['eval']=XYs1d([1.,20.,20.,20.,20.,100.],
        [0.,0.,0.,0.,0.,2.],axes=AXES,label='eval')
    result=reconstruct_suite(suite);attach_reconstruction(suite,result)
    assert max(result.verify_suite(suite).values())<=1
    assert result.evaluate([20.])[ReactionKey('reactions','capture')][0]==0.


@pytest.mark.parametrize('xs,ys',[
    ([20.,20.,100.],[0.,1.,2.]),
    ([1.,20.,20.,20.,100.],[0.,0.,9.,2.,3.]),
    ([1.,100.,100.],[0.,0.,2.]),
])
def test_raw_repeated_nodes_keep_algebra_values_and_one_sided_limits(xs,ys):
    suite=nonresonant_suite();source=XYs1d(xs,ys,axes=AXES,label='eval')
    suite.reactions['capture'].crossSection['eval']=source
    result=reconstruct_suite(suite);attach_reconstruction(suite,result)
    energies=np.unique(np.r_[xs,np.nextafter(xs,np.inf),np.nextafter(xs,0.)])
    energies=energies[(energies>=1.)&(energies<=100.)]
    np.testing.assert_array_equal(result.evaluate(energies)[ReactionKey('reactions','capture')],source.evaluate(energies))
    assert max(result.verify_suite(suite).values())<=1


@pytest.mark.parametrize('include_total_fission',[False,True])
def test_mf2_fission_is_owned_by_first_chance_and_summed_once(include_total_fission):
    from kika.nuclear_data.model import Resonance
    suite=suite_model()
    group=suite.resonances.resolved[0].formalism.resonanceParameters.spinGroups[0]
    group.resonances[0]=Resonance(100.,.5,.4,.1,.2,.1)
    for label,mt,values in [('first',19,[0.,0.]),('second',20,[.3,.3])]:
        suite.reactions.append(Reaction(ReactionId(label,ENDF_MT=mt),CrossSection({
            'eval':XYs1d([10.,1000.],values,axes=AXES,label='eval')})))
    total=suite.sums[1]
    if include_total_fission:
        suite.sums.append(CrossSectionSum(ReactionId('fission',ENDF_MT=18),CrossSection({
            'eval':XYs1d([10.,1000.],[.3,.3],axes=AXES,label='eval')}),
            summands=Summands([Add(href('first')),Add(href('second'))])))
        total.summands.append(Add("/reactionSuite/sums/crossSectionSums/crossSectionSum[@label='fission']/crossSection"))
    else:
        for label in ('first','second'):total.summands.append(Add(href(label)))
    result=run(suite);attach_reconstruction(suite,result)
    assert result.report['resonance_partial_owners'][18]==ReactionKey('reactions','first')
    values=result.evaluate([100.,500.])
    assert values[ReactionKey('reactions','first')][0]>0
    np.testing.assert_array_equal(values[ReactionKey('reactions','first')][1:],0.)
    np.testing.assert_array_equal(values[ReactionKey('reactions','second')],.3)
    if include_total_fission:
        np.testing.assert_allclose(values[ReactionKey('sums','fission')],values[ReactionKey('reactions','first')]+.3)
    assert max(result.verify_suite(suite).values())<=1


def test_empty_mf2_does_not_hide_a_resonance_background():
    suite=suite_model();suite.resonances=None
    with pytest.raises(UnsupportedResonanceError) as failure:run(suite)
    assert failure.value.category=='missing-resonance-regions'


def test_omitting_required_mf2_does_not_turn_raw_mf3_into_complete_data():
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite
    path=Path(__file__).resolve().parents[2]/'endf/tests/data/micro_fe56_structural.endf'
    suite,report=decodeReactionSuite(read_endf(str(path),mf_numbers=[1,3]));suite.report=report
    with pytest.raises(UnsupportedResonanceError) as failure:run(suite)
    assert failure.value.category=='missing-resonance-regions'


def test_numerical_budget_category_does_not_depend_on_error_wording():
    with pytest.raises(ReconstructionConvergenceError) as failure:
        run(options=ReconstructionOptions(max_points=2))
    assert failure.value.category=='budget-exhausted'
    assert UnsupportedResonanceError('arbitrary wording',category='missing-radius-policy').category=='missing-radius-policy'


def test_reloaded_material_cannot_silently_gain_a_reaction():
    suite=nonresonant_suite();result=reconstruct_suite(suite);attach_reconstruction(suite,result)
    extra=deepcopy(suite.reactions['capture']);extra.id=ReactionId('extra',ENDF_MT=16)
    suite.reactions.append(extra)
    with pytest.raises(ReconstructionConvergenceError,match='coverage'):result.verify_suite(suite)


def test_empty_region_still_rejects_dropped_source_physics():
    from kika.nuclear_data.model import Resonances,EndfProvenance
    suite=nonresonant_suite();suite.resonances=Resonances(provenance=EndfProvenance(
        headerFields={'regions':[{'kind':'unsupported','el':1.,'eh':100.}]}))
    with pytest.raises(UnsupportedResonanceError,match='dropped'):reconstruct_suite(suite)


def test_radius_only_mf2_reconstructs_and_roundtrips_processed_tape(tmp_path):
    from kika.endf.classes.mf2.mf2mt151 import MF2MT151,Isotope,EnergyRange,ResolvedResonanceRange
    from kika.endf.model_adapter import decodeMF2MT151,decodeReactionSuite
    from kika.endf import read_endf
    from kika.endf.writers.assemble import writeReconstructedEndfTape
    from test_resonance_publication import writable_suite
    suite=writable_suite()
    section=MF2MT151(number=151);section._za,section._awr,section._mat,section._nis=26056.,56.,2631,1
    section._isotopes=[Isotope(26056.,1.,0,1,[EnergyRange(10.,1000.,0,0,0,0,ResolvedResonanceRange(0.,.5,0,0,[]))])]
    suite.resonances,provenance,report=decodeMF2MT151(section)
    suite.resonances.provenance=provenance;suite.report=report
    assert report.isClean,vars(report)
    for r in list(suite.reactions)+list(suite.sums):
        if hasattr(r.crossSection['eval'],'background'):
            r.crossSection['eval']=XYs1d([10.,1000.],[1.,2.],axes=AXES,label='eval')
    result=reconstruct_suite(suite);attach_reconstruction(suite,result)
    path=tmp_path/'radius.endf';assert writeReconstructedEndfTape(suite,result,path).isClean
    back,report=decodeReactionSuite(read_endf(str(path)));assert report.isClean,vars(report)
    assert max(result.verify_suite(back,label='eval').values())<=1


def test_full_material_shared_histogram_urr_jump_is_preserved():
    from test_unresolved import writable_urr_suite
    from kika.nuclear_data.model.enums import Interpolation
    from kika.processing.resonances import NeutronContext
    suite=writable_urr_suite();w=suite.resonances.unresolved.tabulatedWidths
    w.potentialScatteringInterpolation=Interpolation.flat
    w.potentialScatteringEnergies=np.array([300.,500.,700.])
    result=reconstruct_suite(suite,NeutronContext(56.,0.));attach_reconstruction(suite,result)
    assert 500. in {s.low for s in result._segments}
    assert max(result.verify_suite(suite).values())<=1


def test_repeated_endpoint_is_joined_across_background_ownership():
    suite=suite_model()
    background=suite.reactions['capture'].crossSection['eval'].background
    background.resolvedRegion=XYs1d([10.,300.,300.],[0.,0.,1.],axes=AXES)
    result=run(suite);attach_reconstruction(suite,result)
    assert max(result.verify_suite(suite).values())<=1
    values=result.evaluate([np.nextafter(300.,10.),300.])[ReactionKey('reactions','capture')]
    assert values[0]!=values[1] and values[1]==1.


def test_log_background_zero_endpoint_uses_algebra_limit():
    from kika.nuclear_data.model.enums import Interpolation
    suite=nonresonant_suite()
    suite.reactions['capture'].crossSection['eval']=XYs1d([1.,100.],[0.,2.],
        interpolation=Interpolation.loglog,axes=AXES,label='eval')
    result=reconstruct_suite(suite);attach_reconstruction(suite,result)
    np.testing.assert_array_equal(result.evaluate([1.,50.,100.])[ReactionKey('reactions','capture')],[0.,0.,2.])
    assert max(result.verify_suite(suite).values())<=1


def test_nonresonant_full_suite_gnds_publication_and_selected_style(tmp_path):
    import kika
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    suite=nonresonant_suite()
    from kika.nuclear_data.model import Q
    for reaction in list(suite.reactions)+list(suite.sums):reaction.outputChannel.Q=Q(value=0.)
    result=reconstruct_suite(suite);attach_reconstruction(suite,result)
    path=tmp_path/'material.xml';assert kika.write(suite,path).isClean
    back,report=readReactionSuite(Document.parse(path));assert report.isClean,vars(report)
    assert max(result.verify_suite(back).values())<=1
    np.testing.assert_allclose(back.sums[1].crossSection['recon'].evaluate([1.,100.]),[3.,6.])
    from kika.algebra import group_averages,interval_laws
    for weight in (None,'1/x'):
        reconstructed=back.sums[1].crossSection['recon'];evaluated=back.sums[1].crossSection['eval']
        x,y,pairs=reconstructed.toEndfRegions();average=group_averages(x,y,interval_laws(len(x),pairs),[1.,100.],weight)
        ex,ey,ep=evaluated.toEndfRegions();old=group_averages(ex,ey,interval_laws(len(ex),ep),[1.,100.],weight)
        np.testing.assert_allclose(old,10*average)
