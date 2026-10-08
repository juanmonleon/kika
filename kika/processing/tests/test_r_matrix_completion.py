"""Independent gates for the remaining R5 capabilities."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.linalg import eigh
from kika.processing.resonances import prepare_resonances,tabulate_resonances
from kika.processing.tests.test_r_matrix import rml_model


@pytest.mark.parametrize('l,alpha,expected',[
    (0,1e-12,-5.1566885214004485546550669546e-11),
    (1,1e-12,-1.000000000000999999999999),
    (8,1e-12,-8.000000000000124999999999999),
    (64,1e-12,-64.000000000000015625),
    (64,5.,-64.07807700032623459188),
    (0,1e14,-14142135.37373095711714),
    (64,1e14,-14142135.37387803532243)])
def test_exact_threshold_shift_against_70_digit_bessel_reference(l,alpha,expected):
    from kika.processing.resonances.coulomb import charged_threshold_shift
    assert charged_threshold_shift(l,alpha)==pytest.approx(expected,rel=5e-15)
    assert np.isfinite(charged_threshold_shift(l,[np.nextafter(0.,1.),np.finfo(float).max])).all()


@pytest.mark.parametrize('krm4',[False,True])
@pytest.mark.parametrize('amplitudes',[False,True])
def test_brune_collision_against_independent_equation_33_and_formal_transform(krm4,amplitudes):
    source,ctx=rml_model(multiple=True,threshold=150.,amplitudes=amplitudes)
    f=source.resolved[0].formalism;f.boundaryCondition='Brune'
    if not amplitudes:
        for row in f.spinGroups[0].widths:row[-1]=1e-13
    for ch in f.spinGroups[0].channels:ch.boundaryConditionValue=None
    if krm4:
        f.approximation='RMatrixLimited';f.resonanceReactions[-1].eliminated=False
    p=prepare_resonances(source,ctx);g=p.regions[0].groups[0]
    er=np.array([v.energy for v in g.levels]);a=np.asarray(g.reduced);gg=np.asarray(g.radiation)
    metric=np.asarray(g.level_metric);nmat=np.asarray(g.level_energy)
    eig,b=eigh(nmat,metric);eig+=g.level_origin;af=b.T@a;gf=b.T@np.diag(gg)@b
    energies=np.array([50.,99.8,100.,100.2,125.,150.,150.0001,200.])
    actual=p.evaluate(energies);expected={mt:np.zeros(len(energies)) for mt in actual}
    for k,e in enumerate(energies):
        values=[ch.functions(np.array([e]),logarithmic=True) for ch in g.channels]
        pen=np.array([v[0][0] for v in values]);log=np.array([v[1][0] for v in values]);phase=np.array([v[2][0] for v in values])
        ss=np.array([[ch.functions(np.array([r]),logarithmic=True)[1][0].real for ch in g.channels] for r in er])
        inverse=np.diag(er-e-.5j*gg)-(a*log)@a.T
        for i in range(len(er)):
            for j in range(len(er)):
                correction=ss[i] if i==j else (ss[i]*(e-er[j])-ss[j]*(e-er[i]))/(er[i]-er[j])
                inverse[i,j]+=np.sum(a[i]*a[j]*correction)
        for entrance in g.entrances:
            z=np.linalg.solve(inverse,a[:,entrance]);w=a.T@z
            formal_inverse=np.diag(eig-e)-.5j*gf-(af*log)@af.T
            wf=af.T@np.linalg.solve(formal_inverse,af[:,entrance])
            np.testing.assert_allclose(w,wf,rtol=2e-11,atol=2e-12)
            beta=np.pi*.01/g.channels[entrance].k_squared(g.channels[entrance].channel_energy(e))*3/(2*(2*ctx.target_spin+1))
            u=2j*np.sqrt(pen*pen[entrance])*w*np.exp(-1j*(phase+phase[entrance]));u[entrance]+=np.exp(-2j*phase[entrance])
            for c,ch in enumerate(g.channels):expected[ch.mt][k]+=beta*abs((1-u[c]) if c==entrance else u[c])**2
            expected[102][k]+=beta*2*pen[entrance]*np.sum(gg*abs(z)**2)
    expected[1]=sum(expected[mt] for mt in expected if mt not in (1,18))
    for mt in actual:np.testing.assert_allclose(actual[mt],expected[mt],rtol=3e-11,atol=1e-12)
    assert p.evaluate([100.])[102][0]>0
    assert source.resolved[0].formalism.boundaryCondition=='Brune'


def test_brune_invalid_metric_and_degenerate_coordinates_reject():
    source,ctx=rml_model(threshold=150.)
    f=source.resolved[0].formalism;f.boundaryCondition='Brune'
    for ch in f.spinGroups[0].channels:ch.boundaryConditionValue=None
    f.spinGroups[0].energies=[100.,100.]
    with pytest.raises(ValueError,match='degenerate Brune'):prepare_resonances(source,ctx)
    f.spinGroups[0].energies=[100.,100.2]
    f.spinGroups[0].widths=[[.4,1e5,.2],[.3,1e5,.1]]
    with pytest.raises(ValueError,match='positive definite'):prepare_resonances(source,ctx)


def test_brune_no_boundary_constants_and_snapshot():
    source,ctx=rml_model();f=source.resolved[0].formalism;f.boundaryCondition='Brune'
    with pytest.raises(ValueError,match='boundary constants'):prepare_resonances(source,ctx)
    for ch in f.spinGroups[0].channels:ch.boundaryConditionValue=None
    p=prepare_resonances(source,ctx);before=p.evaluate([99.,100.,101.])
    f.spinGroups[0].widths[0][0]=20.
    for mt,v in before.items():np.testing.assert_array_equal(v,p.evaluate([99.,100.,101.])[mt])


def test_brune_preserves_explicitly_disabled_shift():
    from dataclasses import replace
    source,ctx=rml_model();formalism=source.resolved[0].formalism
    for reaction in formalism.resonanceReactions:reaction.kinematics=replace(reaction.kinematics,shift='zero')
    for channel in formalism.spinGroups[0].channels:channel.boundaryConditionValue=0.
    formalism.boundaryCondition='Given'
    expected=prepare_resonances(source,ctx).evaluate([50.,99.9,100.,100.2,200.])
    formalism.boundaryCondition='Brune'
    for channel in formalism.spinGroups[0].channels:channel.boundaryConditionValue=None
    actual=prepare_resonances(source,ctx).evaluate([50.,99.9,100.,100.2,200.])
    for mt,v in expected.items():np.testing.assert_allclose(actual[mt],v,rtol=3e-13,atol=1e-13)


@pytest.mark.parametrize('function_name',['additionalPhaseShift','tabulatedBackground'])
def test_opt_in_gnds_complex_channel_extension_roundtrip_and_tamper_guard(tmp_path,function_name):
    from test_resonance_publication import writable_suite
    from kika.processing.tests.test_r_matrix_advanced import _complex_phase
    from kika.gnds.encode import writeReactionSuite
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    suite=writable_suite();source,ctx=rml_model();suite.resonances=source
    f=source.resolved[0].formalism
    for rr in f.resonanceReactions:rr.href=f"/reactionSuite/reactions/reaction[@label='{rr.label}']"
    ch=f.spinGroups[0].channels[1];setattr(ch,function_name,_complex_phase(.1,-.2 if function_name=='additionalPhaseShift' else .2))
    if function_name=='additionalPhaseShift':ch.phaseAbsorptionReaction='capture';ch.phaseShiftMode=1
    baseline=prepare_resonances(source,ctx).evaluate([50.,99.9,100.,100.2,150.,300.])
    ordinary,report=writeReactionSuite(suite)
    assert report.unsupported
    tree,report=writeReactionSuite(suite,resonance_extensions=True)
    assert not report.unsupported,vars(report)
    assert tree.getroot().find('applicationData/institution').get('label')=='KIKA::resonance_channel_functions'
    path=tmp_path/'extension.xml';tree.write(path)
    loaded,report=readReactionSuite(Document.parse(path))
    assert not report.unsupported,vars(report)
    # Pair properties in this hand-built suite are channel-local. Standard
    # GNDS reconstructs them from PoPs; copy them explicitly for this unit gate.
    for old,new in zip(f.resonanceReactions,loaded.resonances.resolved[0].formalism.resonanceReactions):
        new.kinematics=old.kinematics;new.reactionMT=old.reactionMT
    observed=prepare_resonances(loaded.resonances,ctx).evaluate([50.,99.9,100.,100.2,150.,300.])
    for mt,v in baseline.items():np.testing.assert_array_equal(v,observed[mt])
    assert getattr(ch,function_name) is not None
    invalid=deepcopy(tree)
    data=invalid.getroot().find('applicationData/institution/resonanceChannelFunctions')
    data.set('version','2');invalid.write(path)
    _,unknown_report=readReactionSuite(Document.parse(path))
    assert any('version' in v for v in unknown_report.unsupported)
    data.set('version','1');data.append(deepcopy(data[0]));invalid.write(path)
    with pytest.raises(ValueError,match='duplicate'):readReactionSuite(Document.parse(path))
    data.remove(data[-1]);data[0].append(deepcopy(data[0][0]));invalid.write(path)
    with pytest.raises(ValueError,match='duplicate'):readReactionSuite(Document.parse(path))
    tree.getroot().find('applicationData/institution/resonanceChannelFunctions/channel').set('parameterFingerprint','changed')
    tree.write(path)
    with pytest.raises(ValueError,match='parameter table'):readReactionSuite(Document.parse(path))


@pytest.mark.parametrize('l,eta,rho,lp,shift',[
    (0,10.,.1,-57.064392075625010203,-1.209224449914662464),
    (2,10.,.1,-63.596722274528217620,-2.437877546497109625),
    (0,1000.,.001,-6277.4104491167915903,-1.214076717814102066),
    (0,1e5,1e-5,-628312.7558591579007,-1.214077201455532560),
    (2,1e5,1e-5,-628319.3420954909182,-2.440318014797690786),
    (0,1e5,5e-5,-628304.8406132766256,-2.938010710574678194),
    (2,1e5,5e-5,-628308.2529906558051,-3.688623281134071056),
    (8,1e5,5e-5,-628338.3603773938266,-8.601041276670421011),
    (64,1e5,5e-5,-629013.0072571265155,-64.078077000306573918),
    (0,1e6,5e-6,-6283171.617074903805,-2.938010711029082284)])
def test_threshold_coulomb_against_independent_70_and_90_digit_whittaker(l,eta,rho,lp,shift):
    from kika.processing.resonances.coulomb import charged_channel_log_functions
    actual=charged_channel_log_functions(l,eta,rho)
    assert abs(actual[0]-lp)<2e-8+3e-15*abs(lp)
    assert actual[1]==pytest.approx(shift,rel=2e-10)


@pytest.mark.parametrize('l',[0,2,8,64])
@pytest.mark.parametrize('eta,rho',[(1.,.25),(8.,.25),(1e5,2e-5)])
def test_coulomb_overlap_series_integral_and_differential_identity(l,eta,rho):
    from kika.processing.resonances.coulomb import _small_radius_outgoing,_outgoing_laplace,charged_channel_log_functions
    series=np.asarray(_small_radius_outgoing(l,eta,rho));integral=np.asarray(_outgoing_laplace(l,eta,rho))
    np.testing.assert_allclose(series,integral,rtol=2e-10,atol=2e-11)
    step=1e-4
    minus=charged_channel_log_functions(l,eta,rho*np.exp(-step))
    plus=charged_channel_log_functions(l,eta,rho*np.exp(step))
    lp,s,_=charged_channel_log_functions(l,eta,rho)
    assert (plus[0]-minus[0])/(2*step)==pytest.approx(1-2*s,rel=3e-6)


@pytest.mark.parametrize('brune',[False,True])
def test_ifg0_true_underflow_reference_normalizes_without_inventing_zero_width(brune):
    from scipy.constants import alpha
    from kika.nuclear_data.model.resonances import ChannelKinematics,ChannelParticle
    from kika.processing.resonances.coulomb import charged_channel_log_functions
    source,ctx=rml_model(amplitudes=False);f=source.resolved[0].formalism;g=f.spinGroups[0]
    if brune:
        f.boundaryCondition='Brune'
        for channel in g.channels:channel.boundaryConditionValue=None
    rr=f.resonanceReactions[1];rr.reactionMT=103
    rr.kinematics=ChannelKinematics(ChannelParticle(1.,1.,.5,1),ChannelParticle(56.,1.,.5,1),'calculate','calculate')
    mu=ctx.neutron_mass_mev*56/57;strength=alpha*mu/ctx.hbar_c_mev_fm;k=strength/120.
    rr.Q=k*k/(2*mu*1e-6/ctx.hbar_c_mev_fm**2)-100*56/57
    g.channels[1].scatteringRadius=.001/k
    for row in g.widths:row[1]=-np.nextafter(0.,1.)
    prepared=prepare_resonances(source,ctx);ch=prepared.regions[0].groups[0].channels[1]
    ecm=abs(float(ch.channel_energy(100.)));rho=np.sqrt(ch.k_squared(ecm))*ch.radius.evaluate(100.)
    lp=charged_channel_log_functions(0,ch.charge_strength/np.sqrt(ch.k_squared(ecm)),rho)[0]
    assert np.exp(lp)==0
    amplitude=prepared.regions[0].groups[0].reduced[0][1]
    expected=-np.exp(.5*(np.log(np.nextafter(0.,1.))-np.log(2.)-lp))
    assert amplitude==pytest.approx(expected,rel=3e-14)
    assert prepared.regions[0].groups[0].levels[0].competitive==np.nextafter(0.,1.)
    diagnostics={}
    assert all(np.isfinite(v).all() for v in prepared.evaluate([99.9,100.,100.1],diagnostics=diagnostics).values())
    assert diagnostics['rml_underflow_bounded_solves']>0


def test_brune_kps_lbk_native_publication_readback_and_high_level_write(tmp_path):
    import kika
    from test_resonance_publication import writable_suite
    from kika.processing.tests.test_r_matrix_advanced import _complex_phase
    from kika.nuclear_data.model import Particle,Nuclide,PhysicalQuantity
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction
    from kika.processing.resonances.prepare_r_matrix import normalize_suite_pairs
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    suite=writable_suite();source,ctx=rml_model();suite.resonances=source
    f=source.resolved[0].formalism;group=f.spinGroups[0];f.boundaryCondition='Brune'
    f.resonanceReactions=[f.resonanceReactions[i] for i in (0,2)]
    group.channels=[group.channels[i] for i in (0,2)];group.widths=[[row[i] for i in (0,2)] for row in group.widths]
    group.parity=-1;group.channels[0].L=1
    for index,ch in enumerate(group.channels):ch.columnIndex=index+1;ch.boundaryConditionValue=None
    group.channels[0].additionalPhaseShift=_complex_phase(.1,-.05)
    group.channels[0].phaseShiftMode=1;group.channels[0].phaseAbsorptionReaction='capture'
    group.channels[0].tabulatedBackground=_complex_phase(.02,.001)
    for label,j,s in [('J0',0.,1.),('J2',2.,1.),('J1s0',1.,0.)]:
        potential=deepcopy(group);potential.label=label;potential.spin=j
        potential.channels[0].channelSpin=s;potential.energies=[];potential.widths=[]
        f.spinGroups.append(potential)
    for rr in f.resonanceReactions:
        rr.kinematics=None;rr.href=f"/reactionSuite/reactions/reaction[@label='{rr.label}']"
    suite.PoPs.add(Particle('n',mass=PhysicalQuantity(ctx.neutron_mass_amu,'amu'),spin=PhysicalQuantity(.5,'hbar'),parity=1,charge=0,halflife='stable'))
    suite.PoPs.add(Nuclide(suite.target,Z=26,A=56,mass=PhysicalQuantity(56*ctx.neutron_mass_amu,'amu'),spin=PhysicalQuantity(.5,'hbar'),parity=1,charge=26))
    result=reconstruct_suite(suite,ctx);attach_reconstruction(suite,result)
    path=tmp_path/'brune-kps-lbk.xml'
    report=kika.write(suite,path,resonance_extensions=True)
    assert report.isClean,vars(report)
    loaded,report=readReactionSuite(Document.parse(path));assert report.isClean,vars(report)
    assert max(result.verify_suite(loaded).values())<=1.
    e=np.unique(np.r_[np.geomspace(10.,300.,71),99.99,100.,100.01,100.2])
    before=prepare_resonances(normalize_suite_pairs(suite,ctx),ctx).evaluate(e)
    after=prepare_resonances(normalize_suite_pairs(loaded,ctx),ctx).evaluate(e)
    for mt,v in before.items():np.testing.assert_array_equal(v,after[mt])
    from kika.endf.model_adapter.resonances import _encodeResolved
    from kika.nuclear_data.model import ConversionReport
    with pytest.raises(ValueError,match='Brune'):_encodeResolved(f,{'lrf':7},ConversionReport())
