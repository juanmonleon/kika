"""Independent analytic and level-space gates for Reich-Moore reconstruction."""
from copy import deepcopy
import numpy as np
import pytest
from kika.nuclear_data.model.resonances import RMatrix,RMatrixSpinGroup,Channel,ResonanceReaction,Resonances,ResolvedRegion,ScatteringRadius
from kika.nuclear_data.model.resonances.radius_policy import RadiusPolicy
from kika.processing.resonances import NeutronContext,prepare_resonances,tabulate_resonances,UnsupportedResonanceError
from kika.processing.resonances.reich_moore import solve_collision
from kika.processing.resonances.channel_functions import neutral_channel_functions


def rm_model(energies=(100.,),widths=((.1,.2,0.,0.),),spins=None,l=0,spin=.5,channel_spin=.5,amplitudes=False):
    channels=[Channel(name,name,L=l,channelSpin=channel_spin if name=='elastic' else 0.,columnIndex=i, radiusUnit='fm') for i,name in enumerate(('elastic','capture','fissionA','fissionB'))]
    reactions=[ResonanceReaction('elastic',reactionMT=2),ResonanceReaction('capture',reactionMT=102,eliminated=True),ResonanceReaction('fissionA',reactionMT=18),ResonanceReaction('fissionB',reactionMT=18)]
    f=RMatrix(approximation='ReichMoore',boundaryCondition='EliminateShiftFunction',scatteringRadius=5.,radiusUnit='fm',reducedWidthAmplitudes=amplitudes,resonanceReactions=reactions,spinGroups=[RMatrixSpinGroup('group',spin=spin,channels=channels,energies=list(energies),widths=[list(row) for row in widths],spins=[] if spins is None else list(spins))])
    return Resonances(resolved=[ResolvedRegion(1e-5,1000.,formalism=f)])


@pytest.mark.parametrize('l',[0,1,2,3])
def test_isolated_level_against_rm_closed_form_without_bw_shift(l):
    ctx=NeutronContext(56.,0.);model=rm_model(l=l,spin=l+.5)
    energy=np.array([25.,99.85,100.,100.15,400.])
    result=prepare_resonances(model,ctx).evaluate(energy)
    rho=np.sqrt(ctx.k_squared_per_ev*energy)*5.
    p,_,phi=neutral_channel_functions(l,rho)
    pr=neutral_channel_functions(l,np.sqrt(ctx.k_squared_per_ev*100.)*5.)[0]
    gn=.1*p/pr;beta=np.pi*.01/(ctx.k_squared_per_ev*energy);g=(2*(l+.5)+1)/2
    den=(energy-100.)**2+(.5*(gn+.2))**2
    np.testing.assert_allclose(result[102],beta*g*gn*.2/den,rtol=3e-14)
    collision=np.exp(-2j*phi)*(1-1j*gn/(energy-100.+.5j*(gn+.2)))
    represented=beta*g*abs(1-collision)**2
    absent=beta*(2*l+1-g)*4*np.sin(phi)**2
    np.testing.assert_allclose(result[2],represented+absent,rtol=3e-12,atol=1e-20)


@pytest.mark.parametrize('capture',[0.,1e-30,1e-16,.2])
def test_capture_is_positive_and_accurate_even_below_flux_subtraction(capture):
    ctx=NeutronContext(56.,0.);model=rm_model(widths=((.1,capture,.3,-.2),))
    prepared=prepare_resonances(model,ctx);diag={};value=prepared.evaluate(np.array([100.]),diagnostics=diag)
    beta=np.pi*.01/(ctx.k_squared_per_ev*100.)
    assert value[102][0]==pytest.approx(beta*4*.1*capture/(.6+capture)**2,rel=1e-13,abs=0.)
    assert value[18][0]==pytest.approx(beta*4*.1*.5/(.6+capture)**2,rel=1e-13)
    assert diag['rm_max_solver_residual']<1e-12
    assert diag['rm_max_absolute_flux_error']<2e-14


def test_common_kernel_matches_independent_dense_level_space():
    e=np.array([99.7,100.,100.1,100.4]);er=np.array([100.,100.2]);gamma=np.array([.02,.3])
    amplitudes=np.broadcast_to(np.array([[.3,.4,-.2],[.2,-.1,.5]]),(len(e),2,3)).copy()
    w,x=solve_collision(e,er,gamma,amplitudes)
    for i,energy in enumerate(e):
        a=amplitudes[i];matrix=np.diag(er-energy-.5j*gamma)-1j*a@a.T
        independent=np.linalg.solve(matrix,a[:,0])
        np.testing.assert_allclose(x[i],independent,rtol=2e-14,atol=1e-14)
        np.testing.assert_allclose(w[i],a.T@independent,rtol=2e-14,atol=1e-14)


@pytest.mark.parametrize('degenerate',[False,True])
def test_exact_undamped_pole_uses_limit_and_preserves_unitarity(degenerate):
    n=2 if degenerate else 1;a=np.broadcast_to(np.array([[[.2,.3,-.1]]]),(3,n,3)).copy()
    er=np.full(n,100.);e=np.array([np.nextafter(100.,0.),100.,np.nextafter(100.,np.inf)])
    diag={};w,x=solve_collision(e,er,np.zeros(n),a,diag)
    expected=1j*np.array([.2,.3,-.1])*.2/.14
    np.testing.assert_allclose(w[1],expected,rtol=1e-13)
    np.testing.assert_allclose(abs(1+2j*w[:,0])**2+4*np.sum(abs(w[:,1:])**2,axis=1),1.,rtol=1e-13)
    assert np.isfinite(x).all()
    if degenerate:assert diag['rm_singular_pole_limits']>=1


def test_relative_fission_signs_control_interference():
    model=rm_model(energies=(100.,100.2),widths=((.1,.05,.2,.3),(.2,.05,-.3,.2)))
    ctx=NeutronContext(56.,0.);e=np.linspace(99.8,100.4,33)
    signed=prepare_resonances(model,ctx).evaluate(e)
    positive=deepcopy(model);positive.resolved[0].formalism.spinGroups[0].widths[1][2]=.3
    assert np.max(abs(signed[18]-prepare_resonances(positive,ctx).evaluate(e)[18]))>100.
    # A global sign of a channel is an unobservable channel-basis phase.
    flipped=deepcopy(model)
    for row in flipped.resolved[0].formalism.spinGroups[0].widths:row[2]*=-1
    np.testing.assert_allclose(prepare_resonances(flipped,ctx).evaluate(e)[18],signed[18],rtol=2e-13)


def test_signed_aj_channels_are_incoherent_and_absent_sectors_complete_potential():
    ctx=NeutronContext(56.,1.);model=rm_model(energies=(100.,100.1),widths=((.1,.2,0.,0.),(.2,.1,0.,0.)),spins=(1.5,-1.5),l=1)
    prepared=prepare_resonances(model,ctx)
    occupied=[g for g in prepared.regions[0].groups if g.levels]
    assert {g.channel_spin for g in occupied}=={.5,1.5}
    assert sum((2*g.spin+1)/(2*(2*ctx.target_spin+1)) for g in prepared.regions[0].groups)==pytest.approx(3.)
    empty=rm_model(energies=(),widths=(),l=1);p=prepare_resonances(empty,ctx)
    e=np.array([25.,100.,400.]);phi=neutral_channel_functions(1,np.sqrt(ctx.k_squared_per_ev*e)*5.)[2]
    np.testing.assert_allclose(p.evaluate(e)[2],np.pi*.01/(ctx.k_squared_per_ev*e)*12*np.sin(phi)**2,rtol=2e-14)


@pytest.mark.parametrize('mode',['phase','mass','constant'])
def test_independent_rm_radii_and_angular_metadata(mode):
    model=rm_model();f=model.resolved[0].formalism
    f.radiusPolicy=RadiusPolicy(channelMode=mode,channelRadius=7. if mode=='constant' else None)
    f.angularLCount=9;f.spinGroups[0].channels[0].scatteringRadius=6.
    ctx=NeutronContext(56.,0.);group=prepare_resonances(model,ctx).regions[0].groups[0]
    assert group.phase_radius.constant==6.
    assert group.channel_radius.constant==pytest.approx(ctx.mass_channel_radius_fm if mode=='mass' else 7. if mode=='constant' else 6.)
    assert len(prepare_resonances(model,ctx).regions[0].groups)==1


def test_widths_and_amplitudes_have_equivalent_physics():
    source=rm_model(widths=((.1,.2,-.3,.4),));ctx=NeutronContext(56.,0.)
    converted=deepcopy(source);f=converted.resolved[0].formalism;f.reducedWidthAmplitudes=True
    pr=neutral_channel_functions(0,np.sqrt(ctx.k_squared_per_ev*100.)*5.)[0]
    f.spinGroups[0].widths=[[np.sqrt(.1/(2*pr)),np.sqrt(.2/2),-np.sqrt(.3/2),np.sqrt(.4/2)]]
    e=np.linspace(99.,101.,30)
    for mt,v in prepare_resonances(source,ctx).evaluate(e).items():np.testing.assert_allclose(prepare_resonances(converted,ctx).evaluate(e)[mt],v,rtol=2e-14)


def test_rm_tabulates_with_strict_existing_multireaction_budget():
    prepared=prepare_resonances(rm_model(energies=(100.,100.2),widths=((.1,.05,.2,.3),(.2,.05,-.3,.2))),NeutronContext(56.,0.))
    table=tabulate_resonances(prepared)
    e=np.r_[np.geomspace(1e-5,1000.,3021),np.linspace(99.,101.,1999)]
    actual=prepared.evaluate(e)
    for mt,form in table.forms.items():
        values=form.evaluate(e)
        ratio=abs(values-actual[mt])/(table.options.atol+table.options.rtol*np.maximum(abs(values),abs(actual[mt])))
        assert np.max(ratio)<=1.


@pytest.mark.parametrize('field,value',[('boundaryCondition','Brune'),('relativisticKinematics',True),('approximation','RMatrixLimited')])
def test_unimplemented_rm_conventions_reject(field,value):
    model=rm_model();setattr(model.resolved[0].formalism,field,value)
    with pytest.raises(UnsupportedResonanceError):prepare_resonances(model,NeutronContext(56.,0.))


def test_common_core_supports_distinct_incoherent_entrances():
    e=np.array([99.9,100.,100.1]);er=np.array([100.,100.2]);gamma=np.array([.01,.2])
    a=np.broadcast_to(np.array([[.3,-.2,.1],[.2,.4,-.3]]),(len(e),2,3)).copy()
    for entrance in (0,1):
        w,x=solve_collision(e,er,gamma,a,entrance=entrance)
        for i,energy in enumerate(e):
            independent=np.linalg.solve(np.diag(er-energy-.5j*gamma)-1j*a[i]@a[i].T,a[i,:,entrance])
            np.testing.assert_allclose(w[i],a[i].T@independent,rtol=2e-14,atol=1e-14)


def test_native_gnds_rm_roundtrip_preserves_kernel_and_whole_suite(tmp_path):
    from test_resonance_publication import writable_suite
    from kika.gnds.encode import writeReactionSuite
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction
    suite=writable_suite();suite.resonances=rm_model();region=suite.resonances.resolved[0]
    region.domainMin,region.domainMax=10.,300.;f=region.formalism
    f.resonanceReactions=f.resonanceReactions[:2];group=f.spinGroups[0]
    group.channels=group.channels[:2];group.widths=[[.1,.2]];group.parity=1
    for i,ch in enumerate(group.channels):ch.columnIndex=i+1
    for rr in f.resonanceReactions:rr.href=f"/reactionSuite/reactions/reaction[@label='{rr.label}']"
    ctx=NeutronContext(56.,0.);result=reconstruct_suite(suite,ctx);attach_reconstruction(suite,result)
    tree,report=writeReactionSuite(suite);assert report.isClean,vars(report)
    path=tmp_path/'native.xml';tree.write(path)
    reloaded,report=readReactionSuite(Document.parse(path))
    assert report.isClean,vars(report)
    assert reloaded.resonances.resolved[0].formalism.resonanceReactions[0].reactionMT==2
    energy=np.linspace(10.,300.,301)
    for mt,value in prepare_resonances(suite.resonances,ctx).evaluate(energy).items():
        np.testing.assert_allclose(prepare_resonances(reloaded.resonances,ctx).evaluate(energy)[mt],value,rtol=2e-14)
    assert max(result.verify_suite(reloaded).values())<=1.


def test_rm_whole_tape_publication_and_source_metadata(tmp_path):
    from test_resonance_publication import writable_suite
    from kika.endf.classes.mf2.mf2mt151 import MF2MT151,Isotope,EnergyRange,ResolvedResonanceRange,LValueBlock,Resonance
    from kika.endf.model_adapter import decodeMF2MT151,decodeReactionSuite,encodeMF2MT151
    from kika.endf.read_endf import read_endf
    from kika.endf.writers.assemble import writeReconstructedEndfTape
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction
    suite=writable_suite();section=MF2MT151(number=151)
    section._za,section._awr,section._mat,section._nis=26056.,56.,2631,1
    params=ResolvedResonanceRange(0.,.5,1,4,[LValueBlock(56.,0,1,[Resonance(100.,.5,.1,.2,0.,0.)],.65)])
    section._isotopes=[Isotope(26056.,1.,0,1,[EnergyRange(10.,300.,1,3,0,0,params)])]
    suite.resonances,provenance,report=decodeMF2MT151(section);suite.resonances.provenance=provenance;suite.report=report
    f=suite.resonances.resolved[0].formalism
    assert f.calculateChannelRadius and f.radiusPolicy.channelMode=='mass' and f.angularLCount==4
    assert str(encodeMF2MT151(suite.resonances,provenance))==str(section)
    result=reconstruct_suite(suite,NeutronContext(56.,0.));attach_reconstruction(suite,result)
    target=tmp_path/'rm.endf';assert writeReconstructedEndfTape(suite,result,target).isClean
    reloaded,report=decodeReactionSuite(read_endf(str(target)))
    assert report.isClean and max(result.verify_suite(reloaded,label='eval').values())<=1.
    assert reloaded.resonances.resolved[0].formalism.radiusPolicy.channelMode=='mass'


def test_tabulation_at_undamped_bare_pole_does_not_move_the_energy():
    prepared=prepare_resonances(rm_model(widths=((.1,0.,0.,0.),)),NeutronContext(56.,0.))
    table=tabulate_resonances(prepared)
    assert np.isfinite(table.forms[2].evaluate(100.))
    assert table.forms[102].evaluate(100.)==0.


def test_lrf3_and_equivalent_lrf7_decode_to_identical_collision_physics():
    from kika.endf.classes.mf2.mf2mt151 import MF2MT151,Isotope,EnergyRange,ResolvedResonanceRange,LValueBlock,Resonance,RMatrixLimited,RML_ParticlePair,RML_Channel,RML_SpinGroup,RML_Resonance
    from kika.endf.model_adapter import decodeMF2MT151
    def decode(parameters,lrf):
        section=MF2MT151(number=151);section._za,section._awr,section._mat,section._nis=26056.,56.,2631,1
        section._isotopes=[Isotope(26056.,1.,0,1,[EnergyRange(10.,300.,1,lrf,0,1,parameters)])]
        model,provenance,report=decodeMF2MT151(section);assert report.isClean
        return prepare_resonances(model,NeutronContext(56.,0.))
    rows=[(100.,.5,.1,.2,-.3,.4),(100.2,.5,.2,.05,.2,-.1)]
    legacy=ResolvedResonanceRange(0.,.5,1,0,[LValueBlock(56.,0,2,[Resonance(*row) for row in rows])])
    pairs=[RML_ParticlePair(0.,0.,0.,0.,0.,0.,1e6,-1,0,102,0.,0.),RML_ParticlePair(1.,56.,0.,26.,.5,0.,0.,1,0,2,0.,1.),RML_ParticlePair(0.,0.,0.,0.,0.,0.,0.,-1,0,18,0.,0.),RML_ParticlePair(0.,0.,0.,0.,0.,0.,0.,-1,0,18,0.,0.)]
    channels=[RML_Channel(1,0,0.,0.,0.,0.),RML_Channel(2,0,.5,0.,.5,.5),RML_Channel(3,0,0.,0.,0.,0.),RML_Channel(4,0,0.,0.,0.,0.)]
    group=RML_SpinGroup(.5,1.,0,0,channels,[RML_Resonance(row[0],[row[3],row[2],row[4],row[5]]) for row in rows])
    limited=RMatrixLimited(0,3,0,.0,.5,pairs,[group])
    a,b=decode(legacy,3),decode(limited,7);e=np.r_[np.linspace(99.,101.,203),100.,100.2]
    for mt,value in a.evaluate(e).items():np.testing.assert_allclose(b.evaluate(e)[mt],value,rtol=3e-14)


def test_explicit_zero_hard_sphere_radius_is_valid_with_positive_true_radius():
    model=rm_model();channel=model.resolved[0].formalism.spinGroups[0].channels[0]
    channel.scatteringRadius=5.;channel.hardSphereRadius=0.
    ctx=NeutronContext(56.,0.);value=prepare_resonances(model,ctx).evaluate(np.array([100.]))
    beta=np.pi*.01/(ctx.k_squared_per_ev*100.)
    assert value[2][0]==pytest.approx(beta*4*.1**2/.3**2,rel=1e-14)


def test_legacy_rm_named_channels_and_gnds_projection_preserve_rows():
    from kika.endf.classes.mf2.mf2mt151 import MF2MT151,Isotope,EnergyRange,ResolvedResonanceRange,LValueBlock,Resonance
    from kika.endf.model_adapter import decodeMF2MT151,encodeMF2MT151
    from kika.gnds.encode_resonances import _rmGroupsForGNDS
    from kika.nuclear_data.model import ConversionReport
    section=MF2MT151(number=151);section._za,section._awr,section._mat,section._nis=26056.,56.,2631,1
    rows=[Resonance(100.,1.5,.1,.2,-.3,.4),Resonance(100.2,-1.5,.2,.1,.4,-.3)]
    params=ResolvedResonanceRange(1.,.5,1,0,[LValueBlock(56.,1,2,rows)])
    section._isotopes=[Isotope(26056.,1.,1,1,[EnergyRange(10.,300.,1,3,0,1,params)])]
    model,provenance,report=decodeMF2MT151(section);f=model.resolved[0].formalism
    f.PoPs.particles['Fe56'].parity=1
    report=ConversionReport();projected=_rmGroupsForGNDS(f,report)
    assert {(g.spin,g.channels[0].channelSpin) for g in projected}=={(1.5,.5),(1.5,1.5)}
    assert all([c.columnIndex for c in g.channels]==[1,2,3,4] for g in projected)
    assert all(g.parity==-1 and not g.spins for g in projected)
    normalized=deepcopy(model);normalized.resolved[0].formalism.spinGroups=projected
    ctx=NeutronContext(56.,1.);e=np.linspace(99.,101.,53)
    for mt,value in prepare_resonances(model,ctx).evaluate(e).items():np.testing.assert_allclose(prepare_resonances(normalized,ctx).evaluate(e)[mt],value,rtol=2e-14)
    group=f.spinGroups[0];order=[2,0,3,1]
    group.channels=[group.channels[i] for i in order];group.widths=[[row[i] for i in order] for row in group.widths]
    assert str(encodeMF2MT151(model,provenance))==str(section)


def test_channel_local_mass_has_precedence_and_invalid_projectile_rejects():
    from dataclasses import replace
    from kika.nuclear_data.model.resonances import ChannelParticle,ChannelKinematics
    source=rm_model();f=source.resolved[0].formalism
    pair=ChannelKinematics(ChannelParticle(1.,0.,.5,1),ChannelParticle(55.,26.,0.,1),'calculate','zero')
    f.resonanceReactions[0].kinematics=pair
    prepared=prepare_resonances(source,NeutronContext(56.,0.))
    assert prepared.regions[0].groups[0].context.atomic_weight_ratio==55.
    assert any('declared neutron-pair' in note for note in prepared.preparation_notes)
    f.resonanceReactions[0].kinematics=replace(pair,particleA=replace(pair.particleA,charge=1.))
    with pytest.raises(UnsupportedResonanceError,match='physical neutron'):prepare_resonances(source,NeutronContext(56.,0.))


def test_explicit_neutron_pair_parity_is_checked():
    from kika.nuclear_data.model.resonances import ChannelParticle,ChannelKinematics
    source=rm_model();f=source.resolved[0].formalism
    f.resonanceReactions[0].kinematics=ChannelKinematics(ChannelParticle(1.,0.,.5,1),ChannelParticle(56.,26.,0.,1),'calculate','zero')
    f.spinGroups[0].parity=-1
    with pytest.raises(ValueError,match='parity'):prepare_resonances(source,NeutronContext(56.,0.))


def test_typed_zero_phase_policy_keeps_mass_derived_true_radius():
    model=rm_model();f=model.resolved[0].formalism
    f.radiusPolicy=RadiusPolicy(channelMode='mass',phaseRadius=ScatteringRadius(constant=0.,unit='fm'))
    ctx=NeutronContext(56.,0.);prepared=prepare_resonances(model,ctx)
    group=prepared.regions[0].groups[0]
    assert group.phase_radius.constant==0.
    assert group.channel_radius.constant==ctx.mass_channel_radius_fm
    assert np.isfinite(prepared.evaluate(np.array([100.]))[2]).all()
