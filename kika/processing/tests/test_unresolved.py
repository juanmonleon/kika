"""Native dilute URR physics, interpolation semantics and MF3 ownership."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma
from kika.nuclear_data.model import Axis,Axes,XYs1d,Regions1d
from kika.nuclear_data.model.enums import Interpolation
from kika.nuclear_data.model.resonances import (
    Resonances,UnresolvedRegion,TabulatedWidths,UnresolvedSpinGroup,
    UnresolvedChannel,RadiusPolicy,ScatteringRadius)
from kika.processing.resonances import (
    NeutronContext,prepare_resonances,tabulate_resonances,UnsupportedResonanceError)


def model(*,law=None,grid=None,shielding=False,nu=1.):
    group=UnresolvedSpinGroup(0,.5,levelSpacing=[100.],channels=[
        UnresolvedChannel('neutron',nu,constantWidth=.02),
        UnresolvedChannel('capture',0.,constantWidth=.1),
        UnresolvedChannel('fission',0.,constantWidth=.2),
        UnresolvedChannel('competitive',0.,constantWidth=.3)],
        crossSectionInterpolation=law)
    widths=TabulatedWidths([group],energyGrid=grid,scatteringRadius=6.,radiusUnit='fm',
        radiusPolicy=RadiusPolicy(channelMode='mass'),selfShieldingOnly=shielding)
    return Resonances(unresolved=UnresolvedRegion(100.,200.,tabulatedWidths=widths))


CONTEXT=NeutronContext(100.,0.)


@pytest.mark.parametrize('nu',[.75,1.,1.5,2.4])
def test_cross_sections_against_direct_gamma_average(nu):
    energy=140.;m=model(nu=nu);got=prepare_resonances(m,CONTEXT).evaluate(energy)
    neutron=.02*np.sqrt(energy)*nu;fixed=.6;shape=nu/2;theta=neutron/shape
    # Independent integral over physical neutron widths, not a Laplace transform.
    def density(x):return x**(shape-1)*np.exp(-x/theta)/(gamma(shape)*theta**shape)
    qng=quad(lambda x:x*.1/(x+fixed)*density(x),0,np.inf,epsabs=1e-13)[0]
    qnn=quad(lambda x:x*x/(x+fixed)*density(x),0,np.inf,epsabs=1e-13)[0]
    k2=CONTEXT.k_squared_per_ev*energy;phi=np.sqrt(k2)*6.
    factor=2*np.pi**2*.01/k2/100.
    expected_el=factor*(qnn-2*neutron*np.sin(phi)**2)+4*np.pi*.01/k2*np.sin(phi)**2
    assert got[2]==pytest.approx(expected_el,rel=3e-10)
    assert got[102]==pytest.approx(factor*qng,rel=3e-10)
    assert got[18]==pytest.approx(2*factor*qng,rel=3e-10)
    assert got[1]==got[2]+got[18]+got[102]
    assert set(got)=={1,2,18,102}  # GX affects widths, no extra competitive cross section.


@pytest.mark.parametrize('law',[Interpolation.linlin,Interpolation.loglog])
def test_interpolate_cross_sections_not_parameters(law):
    from kika.algebra import evaluate
    m=model(law=law,grid=[100.,200.]);prepared=prepare_resonances(m,CONTEXT)
    endpoints=prepare_resonances(model(),CONTEXT).evaluate([100.,200.])
    got=prepared.evaluate([125.,150.,175.])
    for mt in got:
        if mt==1:continue
        expected=evaluate([100.,200.],endpoints[mt],2 if law==Interpolation.linlin else 5,[125.,150.,175.])
        np.testing.assert_allclose(got[mt],expected,rtol=1e-14)
    physical=prepare_resonances(model(),CONTEXT).evaluate([125.,150.,175.])
    assert np.max(abs(physical[102]/got[102]-1))>1e-3


def test_wide_panel_is_filled_before_sigma_interpolation():
    m=model(law=Interpolation.linlin,grid=[100.,10000.]);m.unresolved.domainMax=10000.
    prepared=prepare_resonances(m,CONTEXT);grid=np.array(prepared.regions[0].unresolved.grid)
    assert len(grid)==21
    assert np.max(grid[1:]/grid[:-1])<=10**.1*(1+1e-14)
    physical=deepcopy(m);physical.unresolved.tabulatedWidths.spinGroups[0].crossSectionInterpolation=None
    expected=prepare_resonances(physical,CONTEXT).evaluate(grid)
    for mt,values in prepared.evaluate(grid).items():np.testing.assert_allclose(values,expected[mt],rtol=2e-14)


def test_self_shielding_only_never_reconstructs_again():
    m=model(shielding=True);w=m.unresolved.tabulatedWidths
    w.radiusPolicy=None;w.spinGroups[0].J=123.;w.spinGroups[0].channels=[]
    prepared=prepare_resonances(m,CONTEXT)
    for value in prepared.evaluate([100.,150.,200.]).values():np.testing.assert_array_equal(value,0.)
    assert any('LSSF=1' in note for note in prepared.preparation_notes)


def test_preparation_is_an_independent_snapshot_and_tabulation_verifies():
    m=model();prepared=prepare_resonances(m,CONTEXT);expected=prepared.evaluate([120.,180.])
    m.unresolved.tabulatedWidths.spinGroups[0].channels[0].constantWidth=50.
    for mt,values in prepared.evaluate([120.,180.]).items():np.testing.assert_array_equal(values,expected[mt])
    result=tabulate_resonances(prepared)
    assert max(result.verify_forms(result.forms).values())<=1


def test_parameter_function_owns_its_grid_and_interpolation():
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'width','eV')])
    m=model();c=m.unresolved.tabulatedWidths.spinGroups[0].channels[1]
    c.averageFunction=XYs1d([100.,130.,200.],[.1,.3,.5],axes=axes,interpolation=Interpolation.loglog)
    c.constantWidth=99.  # Compatibility fields never override the canonical function.
    prepared=prepare_resonances(m,CONTEXT)
    np.testing.assert_allclose(prepared.regions[0].groups[0].averages[1].evaluate([130.,200.]),[.3,.5])
    assert 130. in prepared.regions[0].groups[0].averages[1].knots


def test_mf3_step_at_int_boundary_keeps_its_following_interval():
    from test_resonance_tabulation import AXES
    background=Regions1d([
        XYs1d([100.,150.,150.],[1.,1.,4.],axes=AXES),
        XYs1d([150.,200.],[4.,16.],axes=AXES,interpolation=Interpolation.loglog)],axes=AXES)
    prepared=prepare_resonances(model(shielding=True),CONTEXT)
    result=tabulate_resonances(prepared,backgrounds={102:background})
    probes=np.array([np.nextafter(150.,100.),150.,175.,200.])
    np.testing.assert_allclose(result.evaluate(probes)[102],background.evaluate(probes),rtol=1e-14)
    assert result.evaluate(probes)[102][0]==1.
    assert result.evaluate(probes)[102][1]==4.
    assert max(result.verify_forms(result.forms).values())<=1


@pytest.mark.parametrize('edit,match',[
    (lambda m:setattr(m.unresolved.tabulatedWidths,'radiusPolicy',None),'radius policy'),
    (lambda m:setattr(m.unresolved.tabulatedWidths.spinGroups[0],'J',0.),'incompatible'),
    (lambda m:setattr(m.unresolved.tabulatedWidths.spinGroups[0],'levelSpacing',[-1.]),'spacings'),
    (lambda m:setattr(m.unresolved.tabulatedWidths.spinGroups[0].channels[0],'degreesOfFreedom',0.),'GN0'),
])
def test_ambiguous_or_invalid_inputs_reject_before_evaluation(edit,match):
    m=model();edit(m)
    with pytest.raises(ValueError,match=match):prepare_resonances(m,CONTEXT)


def test_global_radius_is_never_a_urr_fallback():
    m=model();m.scatteringRadius=ScatteringRadius(constant=99.,unit='fm')
    prepared=prepare_resonances(m,CONTEXT)
    assert prepared.regions[0].groups[0].phase_radius.constant==6.
    m.unresolved.tabulatedWidths.scatteringRadius=None
    with pytest.raises(ValueError,match='radius'):prepare_resonances(m,CONTEXT)


def test_mixed_sigma_conventions_are_not_resolved_by_the_last_group():
    m=model(law=Interpolation.linlin,grid=[100.,200.]);w=m.unresolved.tabulatedWidths
    w.spinGroups[0].J=0.
    other=deepcopy(w.spinGroups[0]);other.J=1.;other.crossSectionInterpolation=Interpolation.loglog
    w.spinGroups.append(other)
    with pytest.raises(UnsupportedResonanceError,match='mixed URR sigma'):
        prepare_resonances(m,NeutronContext(100.,.5))


def test_zero_capture_and_no_non_neutron_width_has_exact_elastic_limit():
    m=model();g=m.unresolved.tabulatedWidths.spinGroups[0]
    g.channels=g.channels[:1]
    energy=np.array([100.,150.,200.]);got=prepare_resonances(m,CONTEXT).evaluate(energy)
    k2=CONTEXT.k_squared_per_ev*energy;phi=np.sqrt(k2)*6.;gn=.02*np.sqrt(energy)
    expected=2*np.pi**2*.01/k2/100.*gn*(1-2*np.sin(phi)**2)+4*np.pi*.01/k2*np.sin(phi)**2
    np.testing.assert_allclose(got[2],expected,rtol=1e-14)
    np.testing.assert_array_equal(got[102],0.);np.testing.assert_array_equal(got[18],0.)


def test_negative_values_are_retained_and_log_sigma_rejects_them():
    m=model();w=m.unresolved.tabulatedWidths;w.scatteringRadius=600.
    g=w.spinGroups[0];g.channels=g.channels[:1];g.levelSpacing=np.array([.001])
    assert prepare_resonances(m,CONTEXT).evaluate(150.)[2]<0
    g.crossSectionInterpolation=Interpolation.loglog;w.energyGrid=np.array([100.,200.])
    with pytest.raises(ValueError,match='positive'):prepare_resonances(m,CONTEXT)


def test_underflow_is_an_explicit_failure_instead_of_a_dropped_channel():
    m=model();g=m.unresolved.tabulatedWidths.spinGroups[0];g.L=64;g.J=64.5
    m.unresolved.domainMin,m.unresolved.domainMax=1e-8,2e-8
    prepared=prepare_resonances(m,CONTEXT)
    with pytest.raises(FloatingPointError,match='penetrability underflows'):prepared.evaluate(1e-8)


@pytest.mark.parametrize('sigma_law',[None,Interpolation.linlin])
def test_parameter_steps_retain_both_sides(sigma_law):
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'width','eV')])
    m=model(law=sigma_law,grid=[100.,200.]);g=m.unresolved.tabulatedWidths.spinGroups[0]
    g.channels[1].averageFunction=Regions1d([
        XYs1d([100.,150.],[.1,.1],axes=axes),XYs1d([150.,200.],[.9,.9],axes=axes)],axes=axes)
    prepared=prepare_resonances(m,CONTEXT)
    energy=np.array([np.nextafter(150.,100.),150.]);values=prepared.evaluate(energy)
    assert values[102][1]>3*values[102][0]
    result=tabulate_resonances(prepared)
    assert isinstance(result.forms[102],Regions1d)
    assert max(result.verify_forms(result.forms).values())<=1


def test_case_b_reader_declares_sigma_law_and_kernel_uses_it():
    from kika.endf.classes.mf2.mf2mt151 import (
        MF2MT151,Isotope,EnergyRange,UnresolvedCaseB,URR_LValue_CaseB,URR_JState_CaseB)
    from kika.endf.model_adapter import decodeMF2MT151
    section=MF2MT151(number=151)
    section._za,section._awr,section._mat,section._nis=100100.,100.,1001,1
    parameters=UnresolvedCaseB(0.,.6,0,2,1,[100.,200.],[URR_LValue_CaseB(100.,0,[
        URR_JState_CaseB(d=100.,aj=.5,amun=1.,gn0=.02,gg=.1,muf=2,gf=[.1,.9])])])
    section._isotopes=[Isotope(100100.,1.,1,1,[EnergyRange(100.,200.,2,1,0,0,parameters)])]
    m,_,report=decodeMF2MT151(section);assert report.isClean
    g=m.unresolved.tabulatedWidths.spinGroups[0]
    assert g.crossSectionInterpolation==Interpolation.linlin
    prepared=prepare_resonances(m,CONTEXT);v=prepared.evaluate([100.,150.,200.])
    for mt in v:assert v[mt][1]==pytest.approx((v[mt][0]+v[mt][2])*.5,rel=1e-14)


def writable_urr_suite(shielding=False):
    from test_resonance_publication import writable_suite
    from test_resonance_suite import curve
    from kika.endf.model_adapter import decodeMF2MT151,encodeMF2MT151
    from kika.endf.classes.mf2.mf2mt151 import (
        EnergyRange,UnresolvedCaseC,URR_LValue_CaseC,URR_JState_CaseC,URR_EnergyPoint)
    from kika.nuclear_data.model.resonances import ResonanceReaction
    suite=writable_suite()
    section=encodeMF2MT151(suite.resonances,suite.resonances.provenance)
    points=[URR_EnergyPoint(e,100.,0.,.02,.1,0.) for e in (300.,700.)]
    parameters=UnresolvedCaseC(0.,.6,int(shielding),1,[URR_LValue_CaseC(56.,0,[
        URR_JState_CaseC(.5,2,0.,1.,0.,0.,points)])])
    isotope=section._isotopes[0];isotope.num_energy_ranges=2
    isotope.energy_ranges.append(EnergyRange(300.,700.,2,2,0,0,parameters))
    suite.resonances,provenance,report=decodeMF2MT151(section)
    suite.resonances.provenance=provenance;suite.report=report
    widths=suite.resonances.unresolved.tabulatedWidths
    widths.resonanceReactions=[ResonanceReaction('neutron',reactionMT=2,
        href="/reactionSuite/reactions/reaction[@label='elastic']"),
        ResonanceReaction('capture',reactionMT=102,href="/reactionSuite/reactions/reaction[@label='capture']")]
    for label,value in [('elastic',2.),('capture',1.)]:
        b=suite.reactions[label].crossSection['eval'].background
        b.unresolvedRegion=curve(300.,700.,0.);b.fastRegion=curve(700.,1000.,value)
    b=suite.sums[1].crossSection['eval'].background
    b.unresolvedRegion=curve(300.,700.,.5);b.fastRegion=curve(700.,1000.,3.5)
    return suite


@pytest.mark.parametrize('shielding',[False,True])
def test_rrr_urr_boundaries_and_native_publication(tmp_path,shielding):
    import kika
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction,ReactionKey
    from kika.endf.writers.assemble import writeReconstructedEndfTape
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    suite=writable_urr_suite(shielding);ctx=NeutronContext(56.,0.)
    result=reconstruct_suite(suite,ctx);attach_reconstruction(suite,result)
    e=np.array([np.nextafter(300.,10.),300.,500.,np.nextafter(700.,300.),700.])
    v=result.evaluate(e)[ReactionKey('reactions','elastic')]
    assert v[0]!=v[1];assert v[-1]==2.
    xml=tmp_path/'urr.xml';report=kika.write(suite,xml,resonance_extensions=True)
    assert report.isClean,vars(report)
    loaded,report=readReactionSuite(Document.parse(xml));assert report.isClean,vars(report)
    assert max(result.verify_suite(loaded).values())<=1.
    before=prepare_resonances(suite.resonances,ctx).evaluate([300.,500.,700.])
    after=prepare_resonances(loaded.resonances,ctx).evaluate([300.,500.,700.])
    for mt in before:np.testing.assert_allclose(after[mt],before[mt],rtol=3e-14,atol=1e-14)
    endf=tmp_path/'processed.endf';report=writeReconstructedEndfTape(suite,result,endf)
    assert report.isClean,vars(report)
    loaded,report=decodeReactionSuite(read_endf(str(endf)));assert report.isClean,vars(report)
    assert max(result.verify_suite(loaded,label='eval').values())<=1.


@pytest.mark.parametrize('shielding',[False,True])
def test_suite_assembly_retains_mf3_competition_and_lssf(shielding):
    from kika.nuclear_data.model import (ReactionSuite,Reaction,ReactionId,CrossSection,
        CrossSectionSum,Summands,Add,Evaluated,RangeQuantity,PhysicalQuantity,
        Background,ResonancesWithBackground)
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction,ReactionKey
    from test_resonance_tabulation import AXES
    suite=ReactionSuite('URR','n','Fm100')
    suite.styles.add(Evaluated('eval',temperature=PhysicalQuantity(0.,'K'),
        projectileEnergyDomain=RangeQuantity(100.,200.,'eV')))
    suite.resonances=model(shielding=shielding)
    from kika.nuclear_data.model.resonances import ResonanceReaction
    widths=suite.resonances.unresolved.tabulatedWidths
    widths.spinGroups[0].channels[0].label='incident neutron'
    widths.resonanceReactions=[ResonanceReaction('incident neutron',
        href="/reactionSuite/reactions/reaction[@label='elastic']")]
    def curve(value):return XYs1d([100.,200.],[value,value],axes=AXES,label='eval')
    labels=[]
    for label,mt,value in [('elastic',2,1.),('capture',102,.2),('fission',18,.3),('extra',16,.4)]:
        labels.append(label)
        form=ResonancesWithBackground(Background(unresolvedRegion=curve(value)),
            resonanceRegionHref='/reactionSuite/resonances/unresolved',label='eval')
        suite.reactions.append(Reaction(ReactionId(label,ENDF_MT=mt),CrossSection({'eval':form})))
    suite.sums.append(CrossSectionSum(ReactionId('total',ENDF_MT=1),CrossSection({'eval':curve(1.9)}),
        summands=Summands([Add(f"/reactionSuite/reactions/reaction[@label='{label}']/crossSection") for label in labels])))
    result=reconstruct_suite(suite,CONTEXT)
    energy=np.array([100.,125.,150.,200.]);values=result.evaluate(energy)
    physics=prepare_resonances(model(shielding=shielding),CONTEXT).evaluate(energy)
    for label,mt,value in [('elastic',2,1.),('capture',102,.2),('fission',18,.3),('extra',16,.4)]:
        np.testing.assert_allclose(values[ReactionKey('reactions',label)],value+physics.get(mt,0.),rtol=1e-14)
    np.testing.assert_allclose(values[ReactionKey('sums','total')],1.9+physics[1],rtol=1e-14)
    attach_reconstruction(suite,result)
    assert suite.resonances.unresolved.tabulatedWidths.resonanceReactions[0].reactionMT is None
    assert max(result.verify_suite(suite).values())<=1
    for reaction in suite.reactions:assert reaction.crossSection['eval'].background.unresolvedRegion.ys[0] in (1.,.2,.3,.4)
