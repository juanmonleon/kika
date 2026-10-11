"""D3 budgets, semantic boundaries, source declarations and atomic publication."""
from copy import deepcopy
import numpy as np
import pytest
from kika.nuclear_data.model import (
    ReactionSuite, Reaction, ReactionId, CrossSection, CrossSectionSum,
    Summands, Add, Evaluated, Heated, CrossSectionReconstructed, PhysicalQuantity,
    RangeQuantity, XYs1d, Regions1d, Reference, Resonances, UnresolvedRegion,
    Particle, crossSectionAxes,
)
from kika.processing.doppler import (
    DopplerContext, DopplerOptions, SupportPolicy, ReactionKey, DopplerInputAccuracy,
    prepare_curve, broaden_curve, broaden_suite, attach_broadening,
    DopplerInputError, DopplerConvergenceError, UnsupportedDopplerError,
)


def form(x,y,label='eval'):
    return XYs1d(x,y,axes=crossSectionAxes(),label=label)


def fixture():
    s=ReactionSuite('D3 unit','n','Fe56')
    s.styles.add(Evaluated('eval',temperature=PhysicalQuantity(0.,'K'),
                           projectileEnergyDomain=RangeQuantity(.1,4.,'eV')))
    x=np.array([.1,1.,4.]); a=np.array([1.,1.,1.]); b=np.array([0.,2.,0.])
    for label,mt,y in [('elastic',2,a),('capture',102,b)]:
        s.reactions.append(Reaction(ReactionId(label,ENDF_MT=mt),CrossSection({'eval':form(x,y)})))
    s.sums.append(CrossSectionSum(ReactionId('total',ENDF_MT=1),CrossSection({'eval':form(x,a+b)}),
        summands=Summands([Add(f"/reactionSuite/reactions/reaction[@label='{v}']/crossSection") for v in ('elastic','capture')])))
    s.PoPs.add(Particle('n',mass=PhysicalQuantity(1.,'amu')))
    s.PoPs.add(Particle('Fe56',mass=PhysicalQuantity(56.,'amu')))
    return s


def support():return {ReactionKey('reactions',v):SupportPolicy('zero','constant') for v in ('elastic','capture')}


def run(s=None,**kw):
    return broaden_suite(fixture() if s is None else s,source_style='eval',temperature=600.,support=support(),**kw)


@pytest.fixture(scope='module')
def result():return run()


def test_tabulation_and_aggregate_reference(result):
    assert max(result.verify_table().values())<=1
    values=result.evaluate(np.array([.15,.8,1.,3.]))
    assert np.array_equal(values[ReactionKey('sums','total')],
                          values[ReactionKey('reactions','elastic')]+values[ReactionKey('reactions','capture')])
    assert result.report['tabulation']['points']<200000
    assert not result.report['global_error_bound']


def test_attachment_preserves_input_and_publishes_copies(result):
    s=fixture(); before=s.reactions['elastic'].crossSection['eval']; style=s.styles['eval']
    attach_broadening(s,result)
    assert s.reactions['elastic'].crossSection['eval'] is before
    assert s.styles['eval'] is style
    assert s.styles['heated'].temperature==PhysicalQuantity(600.,'K')
    assert s.styles['heated'].derivedFrom=='eval'
    assert max(result.verify_suite(s).values())<=1
    assert s.reactions['elastic'].crossSection['heated'] is not result.forms_by_mt[2]
    with pytest.raises(DopplerInputError):attach_broadening(s,result)


@pytest.mark.parametrize('edit',[
    lambda s:s.reactions['elastic'].crossSection['eval'].ys.__setitem__(0,9.),
    lambda s:s.sums[1].summands.append(Add("/reactionSuite/reactions/reaction[@label='elastic']/crossSection")),
    lambda s:s.PoPs.add(Particle('Fe56',mass=PhysicalQuantity(57.,'amu'))),
    lambda s:setattr(s.styles['eval'],'temperature',PhysicalQuantity(1.,'K')),
])
def test_stale_source_no_partial_attachment(result,edit):
    s=fixture();edit(s)
    with pytest.raises(DopplerInputError,match='changed'):attach_broadening(s,result)
    assert s.styles.labels==['eval']
    assert all('heated' not in r.crossSection for r in s.reactions)


def test_mutated_result_forms_and_report_rejected(result):
    key=ReactionKey('reactions','elastic'); old=result.forms[key].ys.copy()
    try:
        result.forms[key].ys[:]*=2
        s=fixture()
        with pytest.raises(DopplerConvergenceError):attach_broadening(s,result)
        assert s.styles.labels==['eval'] and 'heated' not in s.reactions['capture'].crossSection
    finally:result.forms[key].ys[:]=old
    old=result.report['mass']['atomic_weight_ratio']
    try:
        result.report['mass']['atomic_weight_ratio']=99
        with pytest.raises(DopplerInputError,match='edited'):attach_broadening(fixture(),result)
    finally:result.report['mass']['atomic_weight_ratio']=old


def test_verify_suite_rejects_temperature_and_graph_changes(result):
    s=fixture();attach_broadening(s,result)
    s.styles['heated'].temperature=PhysicalQuantity(601.,'K')
    with pytest.raises(DopplerInputError):result.verify_suite(s)
    s.styles['heated'].temperature=PhysicalQuantity(600.,'K')
    s.sums[1].summands.append(Add("/reactionSuite/reactions/reaction[@label='capture']/crossSection"))
    with pytest.raises(DopplerInputError):result.verify_suite(s)


def test_identity_exact_regions_and_source_zero_domain():
    source=Regions1d([form([0.,1.],[1.,3.]),form([1.,2.],[7.,9.])],axes=crossSectionAxes())
    p=prepare_curve(source,context=DopplerContext(1.),source_temperature=0.,support=SupportPolicy('zero','zero'))
    with pytest.raises(DopplerInputError):broaden_curve(p,temperature=0.)
    r=broaden_curve(p,temperature=0.,output_domain=(.1,2.))
    assert isinstance(r.form,Regions1d)
    assert np.array_equal(r.form.function1ds[0].xs,[.1,1.])
    assert np.array_equal(r.form.function1ds[0].ys,[1.2,3.])
    assert np.array_equal(r.form.function1ds[1].ys,[7.,9.])
    assert r.report['regions'][0]['iterations']==0


def test_threshold_sides_and_mixed_leaf_aggregate():
    s=fixture(); key=ReactionKey('reactions','capture')
    r=run(s,thresholds_by_reaction={key:1.})
    assert isinstance(r.forms_by_mt[102],Regions1d)
    assert r.forms_by_mt[102].function1ds[0].ys[-1]==0
    assert r.forms_by_mt[102].function1ds[1].ys[0]>0
    assert max(r.verify_table().values())<=1
    attach_broadening(s,r)
    with pytest.raises(UnsupportedDopplerError,match='threshold'):
        broaden_suite(s,source_style='heated',label='hotter',temperature=900.,support=support())
    same=broaden_suite(s,source_style='heated',label='same',temperature=600.,support=support())
    assert same.report['temperature_model']=='free-gas-plus-threshold-convention'
    attach_broadening(s,same)
    with pytest.raises(UnsupportedDopplerError,match='threshold'):
        broaden_suite(s,source_style='same',label='hotter',temperature=900.,support=support())


def test_positive_increment_heated_source(result):
    s=fixture();attach_broadening(s,result)
    r=broaden_suite(s,source_style='heated',label='hotter',temperature=900.,support=support())
    assert r.report['temperature_source']==600.
    attach_broadening(s,r)
    assert s.styles['hotter'].derivedFrom=='heated'


@pytest.mark.parametrize('kwargs',[
    {'context':DopplerContext(55.)},
    {'source_temperature':1.},
    {'input_accuracy':DopplerInputAccuracy(infinite_dilution=False)},
    {'options':DopplerOptions(max_points=2)},
    {'options':DopplerOptions(max_iterations=1,rtol=1e-9)},
    {'options':DopplerOptions(max_work_bytes=100)},
])
def test_declared_inputs_and_resource_rejections(kwargs):
    s=fixture()
    with pytest.raises(DopplerInputError):run(s,**kwargs)
    assert s.styles.labels==['eval']


@pytest.mark.parametrize('edit',[
    lambda s:setattr(s,'projectile','photon'),
    lambda s:setattr(s,'target','Fe'),
    lambda s:setattr(s.styles['eval'],'targetInfo',object()),
    lambda s:setattr(s.styles['eval'],'temperature',None),
    lambda s:setattr(s.styles['eval'],'temperature',PhysicalQuantity(True,'K')),
    lambda s:s.sums[1].summands.append(Add("/reactionSuite/reactions/reaction[@label='elastic']/crossSection")),
    lambda s:s.sums[1].summands.summands.clear(),
    lambda s:s.reactions['capture'].crossSection.__setitem__('eval',Reference('https://example.invalid/x',label='eval')),
    lambda s:s.reactions['elastic'].crossSection.__setitem__('eval',Reference("/reactionSuite/reactions/reaction[@label='elastic']/crossSection",label='eval')),
    lambda s:s.reactions['capture'].crossSection.__setitem__('eval',form([.2,4.],[0.,0.])),
    lambda s:s.sums[1].crossSection['eval'].ys.__setitem__(1,99.),
])
def test_unsupported_suite_contracts(edit):
    s=fixture();edit(s)
    with pytest.raises((DopplerInputError,UnsupportedDopplerError)):run(s)
    assert s.styles.labels==['eval']


def test_urr_explicit_mean_no_double_counting(result):
    s=fixture();s.resonances=Resonances(unresolved=UnresolvedRegion(.1,4.))
    with pytest.raises(UnsupportedDopplerError,match='infinite'):run(s)
    r=run(s,urr_policy='dilute_mean',input_accuracy=DopplerInputAccuracy(infinite_dilution=True))
    q=[.2,1.,3.]
    for k,y in result.evaluate(q).items():assert np.array_equal(y,r.evaluate(q)[k])
    assert r.report['urr_treatment']=='broadened-dilute-mean'
    assert not r.report['finite_dilution_supported']


def test_reconstruction_source_requires_temperature_assertion():
    s=fixture();s.styles.add(CrossSectionReconstructed('recon',derivedFrom='eval'))
    for reaction in [*s.reactions,*s.sums]:
        reaction.crossSection['recon']=deepcopy(reaction.crossSection['eval'])
        reaction.crossSection['recon'].label='recon'
    with pytest.raises(DopplerInputError,match='explicit'):
        broaden_suite(s,temperature=0.,source_style='recon',support=support())
    r=broaden_suite(s,temperature=0.,source_style='recon',support=support(),source_temperature=0.)
    assert r.report['temperature_source']==0.
    s.styles['eval'].temperature=PhysicalQuantity(293.6,'K')
    with pytest.raises(UnsupportedDopplerError,match='background'):
        broaden_suite(s,temperature=600.,source_style='recon',support=support(),source_temperature=0.)


def test_cancellation_progress_no_mutation():
    s=fixture()
    def stop(_):raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):run(s,progress=stop)
    assert s.styles.labels==['eval']


def test_source_snapshot_and_narrow_peak_seeding():
    width=1e-9
    x=np.array([.1,1-width,1.,1+width,2.]); y=np.array([0.,0.,1/width,0.,0.])
    p=prepare_curve(((x,y),),context=DopplerContext(1.),source_temperature=0.,support=SupportPolicy('zero','zero'))
    y[:]=0.
    r=broaden_curve(p,temperature=600.,options=DopplerOptions(rtol=1e-3))
    assert r.verify_table()<=1 and max(r.form.ys)>0
    assert not r.report['thinning']


def test_external_reference_method_and_uncertainty(result):
    q=np.array([.2,1.,3.]); values=result.evaluate(q)
    assert result.verify_reference(q,values)['method']=='external-reference'
    bad={k:v*2 for k,v in values.items()}
    with pytest.raises(DopplerConvergenceError):result.verify_reference(q,bad)
    eta={k:np.full(len(q),np.inf) for k in values}
    with pytest.raises(DopplerInputError):result.verify_reference(q,values,uncertainty=eta)


def test_external_heated_source_requires_convolution_declaration():
    s=fixture();s.styles.add(Heated('foreign',derivedFrom='eval',temperature=PhysicalQuantity(300.,'K')))
    for r in [*s.reactions,*s.sums]:
        r.crossSection['foreign']=deepcopy(r.crossSection['eval']);r.crossSection['foreign'].label='foreign'
    with pytest.raises(UnsupportedDopplerError,match='semantics'):
        broaden_suite(s,source_style='foreign',temperature=600.,support=support())
    # Equal-temperature copying does not require a semigroup assertion.
    r=broaden_suite(s,source_style='foreign',temperature=300.,support=support())
    assert r.report['temperature_source']==300.
    assert r.report['temperature_model']=='unknown'
    attach_broadening(s,r)
    with pytest.raises(UnsupportedDopplerError):
        broaden_suite(s,source_style='heated',label='hotter',temperature=600.,support=support())


def test_shared_table_optimization_is_not_duplicate_reaction():
    s=fixture()
    shared=form([.1,1.,4.],[1.,1.,1.])
    s.reactions['elastic'].crossSection['eval']=shared
    s.reactions['capture'].crossSection['eval']=shared
    s.sums[1].crossSection['eval']=form([.1,1.,4.],[2.,2.,2.])
    r=broaden_suite(s,source_style='eval',temperature=0.,support=support())
    assert np.array_equal(r.forms_by_mt[1].ys,[2.,2.,2.])


def test_reference_alias_and_nonlin_rejections():
    s=fixture()
    s.reactions['capture'].crossSection['eval']=Reference("/reactionSuite/reactions/reaction[@label='elastic']/crossSection",label='eval')
    with pytest.raises(UnsupportedDopplerError,match='aliased'):run(s)
    s=fixture();s.reactions['capture'].crossSection['eval'].interpolation='log-log'
    with pytest.raises((DopplerInputError,UnsupportedDopplerError)):run(s)


def test_incomplete_total_and_overlapping_aggregate():
    s=fixture()
    s.reactions.append(Reaction(ReactionId('extra',ENDF_MT=16),CrossSection({'eval':form([.1,4.],[0.,0.])})))
    extra=support();extra[ReactionKey('reactions','extra')]=SupportPolicy('zero','zero')
    with pytest.raises(UnsupportedDopplerError,match='all exclusive'):
        broaden_suite(s,source_style='eval',temperature=0.,support=extra)
    s=fixture()
    for label,mt in [('proton-total',103),('proton-level',600)]:
        s.reactions.append(Reaction(ReactionId(label,ENDF_MT=mt),CrossSection({'eval':form([.1,4.],[0.,0.])})))
    with pytest.raises(UnsupportedDopplerError,match='overlapping'):
        broaden_suite(s,source_style='eval',temperature=0.,support=support())


def test_unreconstructed_source_and_missing_mass():
    from kika.nuclear_data.model import ResonancesWithBackground,Background
    s=fixture();s.reactions['capture'].crossSection['eval']=ResonancesWithBackground(Background(),label='eval')
    with pytest.raises(UnsupportedDopplerError):run(s)
    s=fixture();s.PoPs.particles.clear()
    with pytest.raises(DopplerInputError,match='DopplerContext'):run(s)


def test_numpy_scalar_quantities_have_serializable_provenance():
    import json
    s=fixture()
    r=broaden_suite(s,DopplerContext(np.int64(56)),source_style='eval',temperature=np.float64(0),
                    support=support(),options=DopplerOptions(max_points=np.int64(1000)),
                    input_accuracy=DopplerInputAccuracy(source_temperature=np.int64(0)))
    json.dumps(dict(r.report))
    attach_broadening(s,r)
