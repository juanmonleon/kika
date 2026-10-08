"""URR policy completion: canonical widths, laws, potential and persistence."""
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from kika.algebra import evaluate
from kika.nuclear_data.model import Axis,Axes,XYs1d,Regions1d
from kika.nuclear_data.model.enums import Interpolation
from kika.nuclear_data.model.resonances import RadiusPolicy
from kika.processing.resonances import prepare_resonances,tabulate_resonances,NeutronContext
from test_unresolved import model,CONTEXT,writable_urr_suite


@pytest.mark.parametrize('law',[Interpolation.flat,Interpolation.linlin,Interpolation.linlog,Interpolation.loglin,Interpolation.loglog])
def test_all_scalar_sigma_laws_follow_declared_algebra(law):
    m=model(law=law,grid=[100.,150.,200.]);p=prepare_resonances(m,CONTEXT)
    nodes=prepare_resonances(model(),CONTEXT).evaluate([100.,150.,200.])
    e=np.array([100.,125.,150.,175.,200.]);v=p.evaluate(e)
    from kika.nuclear_data.model.enums import INTERPOLATION_TO_ENDF_INT
    for mt in (2,18,102):np.testing.assert_allclose(v[mt],evaluate([100.,150.,200.],nodes[mt],INTERPOLATION_TO_ENDF_INT[law],e),rtol=2e-14)
    table=tabulate_resonances(p)
    assert max(table.verify_forms(table.forms).values())<=1
    np.testing.assert_allclose(table.evaluate(e)[102],v[102],rtol=2e-3)


def mixed_model(potential='continuous'):
    m=model(law=Interpolation.linlin,grid=[100.,200.]);w=m.unresolved.tabulatedWidths
    g=w.spinGroups[0];g.J=0.
    other=deepcopy(g);other.J=1.;other.levelSpacingEnergies=np.array([100.,140.,200.])
    other.crossSectionInterpolation=Interpolation.loglog;w.spinGroups.append(other)
    w.potentialScatteringInterpolation=potential
    return m,NeutronContext(100.,.5)


@pytest.mark.parametrize('potential',['continuous',Interpolation.linlin,Interpolation.loglog])
def test_mixed_group_policies_sum_resonance_terms_and_one_potential(potential):
    m,ctx=mixed_model(potential);w=m.unresolved.tabulatedWidths
    if potential!='continuous':w.potentialScatteringEnergies=np.array([100.,160.,200.])
    p=prepare_resonances(m,ctx);e=np.array([100.,125.,140.,160.,190.,200.]);got=p.evaluate(e)
    # Independent formula for each single-group result minus its potential:
    # in l=0 with fixed radius the potential is 4*pi/k^2 sin^2(k*R).
    def potential_at(x):
        k2=ctx.k_squared_per_ev*np.asarray(x);return 4*np.pi*.01/k2*np.sin(np.sqrt(k2)*6.)**2
    expected={mt:np.zeros_like(e) for mt in (2,18,102)}
    for group in w.spinGroups:
        isolated=deepcopy(m);iw=isolated.unresolved.tabulatedWidths;iw.spinGroups=[deepcopy(group)]
        iw.potentialScatteringInterpolation=None;iw.potentialScatteringEnergies=None
        iw.spinGroups[0].crossSectionInterpolation=None
        nodes=group.levelSpacingEnergies if group.levelSpacingEnergies is not None else w.energyGrid
        values=prepare_resonances(isolated,ctx).evaluate(nodes)
        values[2]-=potential_at(nodes)
        law=2 if group.crossSectionInterpolation==Interpolation.linlin else 5
        for mt in expected:expected[mt]+=evaluate(nodes,values[mt],law,e)
    term=(potential_at(e) if potential=='continuous' else
          evaluate(w.potentialScatteringEnergies,potential_at(w.potentialScatteringEnergies),2 if potential==Interpolation.linlin else 5,e))
    expected[2]+=term
    for mt in expected:np.testing.assert_allclose(got[mt],expected[mt],rtol=2e-14)
    np.testing.assert_array_equal(got[1],got[2]+got[18]+got[102])
    table=tabulate_resonances(p);assert max(table.verify_forms(table.forms).values())<=1
    # Group order does not assign or multiply the potential term.
    w.spinGroups.reverse();reverse=prepare_resonances(m,ctx).evaluate(e)
    for mt in got:np.testing.assert_allclose(reverse[mt],got[mt],rtol=2e-14)


@pytest.mark.parametrize('mode',['mass','phase','constant'])
def test_missing_radius_can_be_resolved_by_explicit_canonical_policy(mode):
    m=model();w=m.unresolved.tabulatedWidths;w.radiusPolicy=None
    with pytest.raises(ValueError,match='radius policy'):prepare_resonances(m,CONTEXT)
    w.radiusPolicy=RadiusPolicy(channelMode=mode,channelRadius=5. if mode=='constant' else None)
    assert np.isfinite(prepare_resonances(m,CONTEXT).evaluate(150.)[102])


@pytest.mark.parametrize('nu',[0.,1.,1.75])
def test_explicit_physical_neutron_width_is_not_converted_again(nu):
    m=model(nu=nu);c=m.unresolved.tabulatedWidths.spinGroups[0].channels[0]
    c.neutronWidthConvention='physical';c.constantWidth=.25
    e=np.array([100.,140.,200.]);v=prepare_resonances(m,CONTEXT).evaluate(e)
    if nu==0:
        k2=CONTEXT.k_squared_per_ev*e;factor=2*np.pi**2*.01/k2/100.
        np.testing.assert_allclose(v[102],factor*.25*.1/.85,rtol=1e-14)
    else:
        reduced=deepcopy(m);c=reduced.unresolved.tabulatedWidths.spinGroups[0].channels[0]
        c.neutronWidthConvention='reduced';c.constantWidth=None;c.widths=.25/(np.sqrt(e)*nu);c.energies=e
        at_nodes=prepare_resonances(reduced,CONTEXT).evaluate(e)
        for mt in v:np.testing.assert_allclose(v[mt],at_nodes[mt],rtol=2e-14)


def test_parameter_step_at_change_of_interpolation_keeps_both_sides():
    m=model();g=m.unresolved.tabulatedWidths.spinGroups[0];g.channels[1].constantWidth=None
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'width','eV')])
    g.channels[1].averageFunction=Regions1d([XYs1d([100.,150.,150.],[.1,.1,.4],axes=axes),
        XYs1d([150.,200.],[.4,.8],axes=axes,interpolation=Interpolation.loglog)],axes=axes)
    p=prepare_resonances(m,CONTEXT)
    assert p.evaluate(np.nextafter(150.,100.))[102]!=p.evaluate(150.)[102]
    table=tabulate_resonances(p);assert max(table.verify_forms(table.forms).values())<=1


@pytest.mark.parametrize('potential',['continuous',Interpolation.linlog])
def test_new_conventions_persist_opt_in_and_reject_lossy_endf(tmp_path,potential):
    import kika
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    from kika.endf.model_adapter import encodeMF2MT151
    suite=writable_urr_suite();w=suite.resonances.unresolved.tabulatedWidths
    w.potentialScatteringInterpolation=potential
    if potential!='continuous':w.potentialScatteringEnergies=np.array([300.,500.,700.])
    c=w.spinGroups[0].channels[0];c.neutronWidthConvention='physical';c.degreesOfFreedom=0.
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction
    ctx=NeutronContext(56.,0.)
    reconstructed=reconstruct_suite(suite,ctx);attach_reconstruction(suite,reconstructed)
    path=tmp_path/'explicit.xml';report=kika.write(suite,path,resonance_extensions=True)
    assert report.isClean,vars(report)
    loaded,report=readReactionSuite(Document.parse(path));assert report.isClean,vars(report)
    assert max(reconstructed.verify_suite(loaded).values())<=1
    lw=loaded.resonances.unresolved.tabulatedWidths
    assert lw.potentialScatteringInterpolation==potential
    assert lw.spinGroups[0].channels[0].neutronWidthConvention=='physical'
    ctx=NeutronContext(56.,0.);e=[300.,400.,500.,700.]
    before=prepare_resonances(suite.resonances,ctx).evaluate(e);after=prepare_resonances(loaded.resonances,ctx).evaluate(e)
    for mt in before:np.testing.assert_allclose(after[mt],before[mt],rtol=3e-14)
    with pytest.raises(ValueError,match='physical neutron widths'):encodeMF2MT151(suite.resonances,suite.resonances.provenance)
    c.neutronWidthConvention='reduced';c.degreesOfFreedom=1.
    with pytest.raises(ValueError,match='potential-scattering'):encodeMF2MT151(suite.resonances,suite.resonances.provenance)
    standard=kika.write(suite,tmp_path/'standard.xml');assert not standard.isClean
    root=ET.parse(path).getroot();block=root.find("applicationData/institution/unresolvedPotentialScattering")
    block.set('parameterFingerprint','invalid');ET.ElementTree(root).write(path)
    with pytest.raises(ValueError,match='does not match'):readReactionSuite(Document.parse(path))


def test_mixed_continuous_and_tabulated_groups_keep_own_conventions():
    m,ctx=mixed_model();w=m.unresolved.tabulatedWidths
    w.spinGroups[0].crossSectionInterpolation=None
    e=np.array([110.,150.,190.]);got=prepare_resonances(m,ctx).evaluate(e)
    continuous=deepcopy(m);cw=continuous.unresolved.tabulatedWidths;cw.spinGroups=cw.spinGroups[:1]
    first=prepare_resonances(continuous,ctx).evaluate(e)
    second=deepcopy(m);sw=second.unresolved.tabulatedWidths;sw.spinGroups=sw.spinGroups[1:]
    latter=prepare_resonances(second,ctx).evaluate(e)
    k2=ctx.k_squared_per_ev*e;potential=4*np.pi*.01/k2*np.sin(np.sqrt(k2)*6.)**2
    np.testing.assert_allclose(got[102],first[102]+latter[102],rtol=2e-14)
    np.testing.assert_allclose(got[2],first[2]+latter[2]-potential,rtol=2e-14)


def test_histogram_potential_has_right_owned_jumps():
    m=model();w=m.unresolved.tabulatedWidths
    w.potentialScatteringInterpolation=Interpolation.flat;w.potentialScatteringEnergies=np.array([100.,150.,200.])
    p=prepare_resonances(m,CONTEXT);e=np.array([np.nextafter(150.,100.),150.])
    direct=p.evaluate(e)[2];assert direct[0]!=direct[1]
    table=tabulate_resonances(p);assert max(table.verify_forms(table.forms).values())<=1
    np.testing.assert_allclose(table.evaluate(e)[2],direct,rtol=2e-3)


def test_zero_reduced_neutron_width_with_zero_nu_is_potential_only():
    m=model(nu=0.);m.unresolved.tabulatedWidths.spinGroups[0].channels[0].constantWidth=0.
    e=np.array([100.,150.,200.]);values=prepare_resonances(m,CONTEXT).evaluate(e)
    k2=CONTEXT.k_squared_per_ev*e
    np.testing.assert_allclose(values[2],4*np.pi*.01/k2*np.sin(np.sqrt(k2)*6.)**2,rtol=1e-14)
    np.testing.assert_array_equal(values[102],0.)


def test_parameter_jump_does_not_redefine_the_independent_potential_grid():
    m=model();w=m.unresolved.tabulatedWidths
    w.spinGroups[0].channels[0].constantWidth=0.
    capture=w.spinGroups[0].channels[1];capture.constantWidth=None
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'width','eV')])
    capture.averageFunction=XYs1d([100.,150.,150.,200.],[.1,.1,.4,.4],axes=axes)
    w.potentialScatteringInterpolation=Interpolation.linlin;w.potentialScatteringEnergies=np.array([100.,200.])
    e=np.array([125.,150.,175.]);p=prepare_resonances(m,CONTEXT)
    k2=CONTEXT.k_squared_per_ev*np.array([100.,200.])
    values=4*np.pi*.01/k2*np.sin(np.sqrt(k2)*6.)**2
    np.testing.assert_allclose(p.evaluate(e)[2],evaluate([100.,200.],values,2,e),rtol=1e-14)


@pytest.mark.parametrize('table',[False,True])
def test_distinct_constant_channel_and_phase_radii_survive_gnds(tmp_path,table):
    import kika
    from kika.nuclear_data.model.resonances import ScatteringRadius
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    suite=writable_urr_suite();w=suite.resonances.unresolved.tabulatedWidths
    w.spinGroups[0].L=1
    phase=(ScatteringRadius(energies=[300.,700.],values=[6.,8.],interpolation=[(2,2)],unit='fm') if table
           else ScatteringRadius(constant=6.,unit='fm'))
    w.radiusPolicy=RadiusPolicy(channelMode='constant',channelRadius=5.,phaseRadius=phase)
    path=tmp_path/'radii.xml';report=kika.write(suite,path,resonance_extensions=True);assert report.isClean,vars(report)
    loaded,report=readReactionSuite(Document.parse(path));assert report.isClean,vars(report)
    policy=loaded.resonances.unresolved.tabulatedWidths.radiusPolicy
    assert policy.channelMode=='constant' and policy.channelRadius==5.
    ctx=NeutronContext(56.,0.);e=[300.,500.,700.]
    before=prepare_resonances(suite.resonances,ctx).evaluate(e);after=prepare_resonances(loaded.resonances,ctx).evaluate(e)
    for mt in before:np.testing.assert_allclose(after[mt],before[mt],rtol=3e-14)


@pytest.mark.parametrize('canonical_spacing',[False,True])
def test_mixed_sigma_grids_survive_constant_spacing_gnds(tmp_path,canonical_spacing):
    import kika
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    from kika.endf.model_adapter import encodeMF2MT151
    suite=writable_urr_suite();w=suite.resonances.unresolved.tabulatedWidths
    w.potentialScatteringInterpolation='continuous'
    other=deepcopy(w.spinGroups[0]);other.L=1;other.levelSpacing=np.array([100.])
    if canonical_spacing:
        from kika.nuclear_data.model import Constant1d
        other.levelSpacing=None
        other.levelSpacingFunction=Constant1d(100.,300.,700.,axes=Axes([Axis(0,'spacing','eV'),Axis(1,'energy','eV')]))
    other.crossSectionInterpolation=Interpolation.loglog;other.crossSectionEnergies=np.array([300.,450.,700.])
    w.spinGroups.append(other)
    ctx=NeutronContext(56.,0.);before=prepare_resonances(suite.resonances,ctx).evaluate([300.,400.,450.,600.,700.])
    path=tmp_path/'mixed.xml';report=kika.write(suite,path,resonance_extensions=True);assert report.isClean,vars(report)
    loaded,report=readReactionSuite(Document.parse(path));assert report.isClean,vars(report)
    np.testing.assert_array_equal(loaded.resonances.unresolved.tabulatedWidths.spinGroups[1].crossSectionEnergies,[300.,450.,700.])
    root=ET.parse(path).getroot()
    assert root.find('applicationData/institution/unresolvedCrossSectionInterpolation').get('version')=='2'
    after=prepare_resonances(loaded.resonances,ctx).evaluate([300.,400.,450.,600.,700.])
    for mt in before:np.testing.assert_allclose(after[mt],before[mt],rtol=3e-14)
    # An independent sigma mesh is representable in the canonical model, but
    # cannot be written as ENDF's one parameter/sigma mesh without resampling.
    w.potentialScatteringInterpolation=None
    with pytest.raises(ValueError,match='common parameter/sigma grid'):encodeMF2MT151(suite.resonances,suite.resonances.provenance)


def test_self_shielding_sigma_grid_preserves_repeated_nodes_gnds(tmp_path):
    import kika
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    suite=writable_urr_suite();w=suite.resonances.unresolved.tabulatedWidths
    w.selfShieldingOnly=True
    w.spinGroups[0].crossSectionEnergies=np.array([300.,500.,500.,700.])
    path=tmp_path/'repeated.xml'
    report=kika.write(suite,path,resonance_extensions=True);assert report.isClean,vars(report)
    loaded,report=readReactionSuite(Document.parse(path));assert report.isClean,vars(report)
    np.testing.assert_array_equal(loaded.resonances.unresolved.tabulatedWidths.spinGroups[0].crossSectionEnergies,[300.,500.,500.,700.])


@pytest.mark.parametrize('edit,match',[
    (lambda w:setattr(w,'potentialScatteringEnergies',[100.,200.]),'requires a policy'),
    (lambda w:(setattr(w,'potentialScatteringInterpolation','continuous'),setattr(w,'potentialScatteringEnergies',[100.,200.])), 'does not use'),
    (lambda w:setattr(w,'potentialScatteringInterpolation',Interpolation.chargedParticle),'INT1-5'),
    (lambda w:setattr(w.spinGroups[0],'crossSectionInterpolation',Interpolation.chargedParticle),'INT1-5'),
    (lambda w:setattr(w.spinGroups[0].channels[1],'neutronWidthConvention','physical'),'only to the neutron'),
])
def test_undeclared_or_invalid_policies_do_not_silently_fall_back(edit,match):
    m=model();edit(m.unresolved.tabulatedWidths)
    with pytest.raises(ValueError,match=match):prepare_resonances(m,CONTEXT)
