"""R5b gates against independent level space and high-precision Coulomb values."""
from copy import deepcopy
import numpy as np
import pytest
from test_r_matrix import rml_model
from kika.nuclear_data.model.resonances import ChannelParticle,ChannelKinematics,Channel
from kika.processing.resonances import prepare_resonances,tabulate_resonances
from kika.processing.resonances.coulomb import charged_channel_functions,closed_charged_shift,charged_threshold_shift


def test_relativistic_two_body_momentum_against_invariant_four_vector():
    from kika.processing.resonances.kinematics import RelativisticPair
    # MeV masses, eV kinetic energy. Independent Kallen invariant expression.
    pair=RelativisticPair(939.,1800.,900.,1841.,-2e6,197.)
    e=np.array([1e7,5e7,1e8])
    invariant=(939.+1800.)**2+2*1800.*e*1e-6
    reference=((invariant-(900.+1841.)**2)*(invariant-(900.-1841.)**2)/(4*invariant*197.**2))
    np.testing.assert_allclose(pair.k_squared(pair.energy(e)),reference,rtol=5e-14)
    assert pair.energy(pair.threshold)==0.
    assert pair.k_squared(pair.energy(np.nextafter(pair.threshold,0.)))<0
    assert pair.k_squared(pair.energy(np.nextafter(pair.threshold,np.inf)))>0
    with pytest.raises(ValueError,match='conserve rest energy'):
        RelativisticPair(939.,1800.,900.,1840.,-2e6,197.)


def test_relativistic_rml_uses_incident_momentum_and_declared_threshold():
    source,ctx=rml_model()
    f=source.resolved[0].formalism;f.relativisticKinematics=True
    source.resolved[0].domainMax=1e8;f.spinGroups[0].energies=[5e7,5.1e7]
    rr=f.resonanceReactions[1];a,b=rr.kinematics.particleA,rr.kinematics.particleB
    # Declare the small residual mass increment; do not repair it in the kernel.
    rr.kinematics=ChannelKinematics(a,ChannelParticle(b.massRatio-rr.Q*1e-6/ctx.neutron_mass_mev,b.charge,b.spin,b.parity),'calculate','calculate')
    prepared=prepare_resonances(source,ctx);group=prepared.regions[0].groups[0]
    channel=group.channels[1];threshold=channel.threshold
    e=np.array([30.,threshold,np.nextafter(threshold,np.inf),5e7,5.1e7,8e7])
    diagnostics={};result=prepared.evaluate(e,diagnostics=diagnostics)
    assert np.all(result[51][:2]==0) and result[51][2]>0
    assert diagnostics['rml_max_absolute_flux_error']<1e-11
    assert threshold != -rr.Q/(56/57)
    incident=group.channels[0];kin=incident.kinematics
    invariant=(kin.incident_mass+kin.target_mass)**2+2*kin.target_mass*e[-1]*1e-6
    k2=((invariant-(kin.incident_mass+kin.target_mass)**2)*(invariant-(kin.incident_mass-kin.target_mass)**2)/(4*invariant*ctx.hbar_c_mev_fm**2))
    assert incident.k_squared(incident.channel_energy(e[-1]))==pytest.approx(k2,rel=3e-14)


def test_per_channel_phase_replaces_hard_sphere_and_preserves_jump():
    from kika.nuclear_data.model import XYs1d,Axes,Axis,Regions1d
    from kika.nuclear_data.model.resonances import ComplexChannelFunction
    source,ctx=rml_model();f=source.resolved[0].formalism
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'phase','')])
    phase=ComplexChannelFunction(XYs1d([10.,150.,150.,300.],[.1,.1,.3,.3],axes=axes),XYs1d([10.,300.],[0.,0.],axes=axes))
    f.spinGroups[0].channels[0].additionalPhaseShift=phase
    potential=deepcopy(f.spinGroups[0]);potential.spin=0.;potential.energies=[];potential.widths=[]
    for ch in potential.channels:ch.channelSpin=0.
    f.spinGroups.append(potential)
    prepared=prepare_resonances(source,ctx)
    c=prepared.regions[0].groups[0].channels[0]
    e=np.array([np.nextafter(150.,0.),150.,np.nextafter(150.,np.inf)])
    np.testing.assert_array_equal(c.functions(e)[2],[.1,.3,.3])
    table=tabulate_resonances(prepared)
    assert isinstance(table.forms[2],Regions1d)
    assert max(table.verify_forms(table.forms).values())<=1.
    phase.imaginary.ys[:]=.01
    with pytest.raises(ValueError,match='reaction ownership'):prepare_resonances(source,ctx)


@pytest.mark.parametrize('amplitudes',[False,True])
def test_endf_kps_records_are_per_channel_and_survive_parse_and_model(amplitudes):
    from kika.endf.classes.mf2.mf2mt151 import (MF2MT151,Isotope,EnergyRange,RMatrixLimited,
        RML_ParticlePair,RML_Channel,RML_SpinGroup,RML_Resonance,TabulatedPhaseShift)
    from kika.endf.parsers.parse_mf2 import parse_mf2_mt151
    from kika.endf.model_adapter import decodeMF2MT151,encodeMF2MT151
    phase=TabulatedPhaseShift([(2,2)],[10.,300.],[.1,.2],[(2,2)],[10.,300.],[0.,0.])
    p=RMatrixLimited(int(amplitudes),3,0,.5,.5,
        [RML_ParticlePair(1.,56.,0.,26.,.5,.5,0.,1,1,2,1.,1.),RML_ParticlePair(0.,0.,0.,0.,0.,0.,1e6,-1,0,102,1.,1.)],
        [RML_SpinGroup(1.,1.,0,1,[RML_Channel(1,0,1.,.2,0.,.5),RML_Channel(2,0,0.,0.,0.,0.)],
          [RML_Resonance(100.,[.1,.2])],phase_shifts=[phase,None])])
    section=MF2MT151(number=151);section._za,section._awr,section._mat,section._nis=26056,56.,2631,1
    section._isotopes=[Isotope(26056,1.,0,1,[EnergyRange(10.,300.,1,7,0,1,p)])]
    loaded=parse_mf2_mt151(str(section).splitlines(),151)
    phases=loaded.isotopes[0].energy_ranges[0].parameters.spin_groups[0].phase_shifts
    assert len(phases)==2 and phases[1] is None and phases[0].psr_values==[.1,.2]
    model,provenance,report=decodeMF2MT151(loaded);assert report.isClean
    assert model.resolved[0].formalism.spinGroups[0].channels[0].phaseShiftMode==1
    assert model.resolved[0].formalism.spinGroups[0].channels[1].phaseShiftMode==0
    assert str(encodeMF2MT151(model,provenance))==str(loaded)


@pytest.mark.parametrize('capture_explicit',[False,True])
def test_ifg1_gnds_mixed_energy_and_amplitude_units_roundtrip(capture_explicit):
    import xml.etree.ElementTree as ET
    from kika.gnds.resonances import readResonances
    from kika.gnds.encode_resonances import writeResonances
    from kika.nuclear_data.model import ConversionReport
    source,ctx=rml_model();f=source.resolved[0].formalism
    if capture_explicit:
        f.approximation='RMatrixLimited';f.resonanceReactions[2].eliminated=False
        f.spinGroups[0].channels[2].boundaryConditionValue=0.
    root=ET.Element('reactionSuite');report=ConversionReport()
    represented=deepcopy(source)
    for rr in represented.resolved[0].formalism.resonanceReactions:
        rr.kinematics=None
        rr.href=f"/reactionSuite/reactions/reaction[@label='{rr.label}']"
    writeResonances(root,represented,report,('10','300'));assert report.isClean,vars(report)
    table=root.find('.//spinGroup//table');headers=table.findall('columnHeaders/column')
    data=np.fromstring(table.find('data').text,sep=' ').reshape(2,4)
    headers[0].set('unit','keV');data[:,0]/=1000
    for header in headers[1:]:header.set('unit','keV**0.5')
    data[:,1:]/=np.sqrt(1000)
    table.find('data').text=' '.join(format(v,'.17g') for v in data.ravel())
    report=ConversionReport();loaded=readResonances(root.find('resonances'),'/reactionSuite',None,report,lambda element:None)
    assert report.isClean,vars(report)
    lf=loaded.resolved[0].formalism
    for original,rr in zip(f.resonanceReactions,lf.resonanceReactions):
        rr.kinematics=original.kinematics;rr.reactionMT=original.reactionMT
    e=np.array([30.,75.,99.9,100.,100.2,180.])
    a=prepare_resonances(source,ctx).evaluate(e);b=prepare_resonances(loaded,ctx).evaluate(e)
    for mt,value in a.items():np.testing.assert_allclose(b[mt],value,rtol=2e-12,atol=1e-12)


@pytest.mark.parametrize('l,eta,rho,p,s,phase',[
    (0,1.,1.,.2366295887750756,-.5921199941554618,.11090640906288389),
    (1,.1,3.,2.597665156054475,-.12294956893763151,1.6171883325630718),
    (2,3.,.2,2.4905129540826183e-10,-2.2652273622312054,4.5661182578926835e-11),
    (0,20.,.1,1.3345761746765487e-51,-1.784760715582078,3.2558859077767346e-52),
    (4,10.,20.,3.5170766030452665,-3.3556220942628876,.32692251992373517),
    (20,1.,1.,4.334026843725795e-49,-20.024344595824733,1.0558257721449437e-50),
    (0,1e-8,10.,9.9999999900486,-4.928412781075167e-10,-2.5663706501132943),
    (0,3.,100.,96.95376824135805,-.015954651821891402,-2.7605850613947154)])
def test_coulomb_against_60_digit_reference(l,eta,rho,p,s,phase):
    actual=charged_channel_functions(l,eta,rho)
    np.testing.assert_allclose(actual[:2],[p,s],rtol=5e-10,atol=3e-12)
    assert abs(np.sin(actual[2]-phase))<2e-11
    assert actual[0]/p==pytest.approx(1.,rel=5e-10)


@pytest.mark.parametrize('l,eta,rho,s',[(0,1.,.1,-.3258617807019725),(2,3.,.2,-2.2867501236408074),(4,20.,10.,-22.596637283230084)])
def test_closed_charged_shift_against_whittaker_reference(l,eta,rho,s):
    assert closed_charged_shift(l,eta,rho)==pytest.approx(s,rel=2e-11)


def test_coulomb_neutral_limit_and_zero_energy_limit():
    from kika.processing.resonances.channel_functions import neutral_channel_functions
    rho=np.array([.01,.2,1.,3.,20.])
    for l in (0,1,2,8):
        for a,b in zip(charged_channel_functions(l,0.,rho),neutral_channel_functions(l,rho)):
            np.testing.assert_array_equal(a,b)
    constant=.4
    threshold=charged_threshold_shift(1,constant)
    for kappa in (1e-3,1e-4):
        assert closed_charged_shift(1,constant/kappa,kappa)==pytest.approx(threshold,abs=2e-6)


def test_coulomb_unrepresentable_flux_is_explicit_error():
    with pytest.raises(FloatingPointError,match='not representable'):
        charged_channel_functions(0,200.,.1)


def test_extreme_barrier_and_uncertified_relativistic_coulomb_are_explicit():
    from kika.processing.resonances import UnsupportedResonanceError
    with pytest.raises(FloatingPointError,match='verified computational range'):
        charged_channel_functions(0,201.,.1)
    source,ctx=rml_model();f=source.resolved[0].formalism
    rr=f.resonanceReactions[1];rr.reactionMT=103;f.relativisticKinematics=True
    rr.kinematics=ChannelKinematics(ChannelParticle(1.,1.,.5,1),ChannelParticle(56.,1.,.5,1),'calculate','calculate')
    with pytest.raises(UnsupportedResonanceError,match='relativistic Coulomb'):
        prepare_resonances(source,ctx)


def test_krm4_ifg1_native_suite_publication(tmp_path):
    from test_resonance_publication import writable_suite
    from kika.nuclear_data.model import Particle,Nuclide,PhysicalQuantity
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction
    from kika.processing.resonances.prepare_r_matrix import normalize_suite_pairs
    from kika.gnds.encode import writeReactionSuite
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    suite=writable_suite();source,ctx=rml_model();suite.resonances=source
    f=source.resolved[0].formalism;g=f.spinGroups[0]
    f.approximation='RMatrixLimited';f.resonanceReactions[2].eliminated=False
    f.resonanceReactions=[f.resonanceReactions[i] for i in (0,2)]
    g.channels=[g.channels[i] for i in (0,2)];g.widths=[[row[i] for i in (0,2)] for row in g.widths]
    g.channels[1].boundaryConditionValue=0.
    for index,ch in enumerate(g.channels):ch.columnIndex=index+1
    for rr in f.resonanceReactions:
        rr.kinematics=None;rr.href=f"/reactionSuite/reactions/reaction[@label='{rr.label}']"
    suite.PoPs.add(Particle('n',mass=PhysicalQuantity(ctx.neutron_mass_amu,'amu'),spin=PhysicalQuantity(.5,'hbar'),parity=1,charge=0,halflife='stable'))
    suite.PoPs.add(Nuclide(suite.target,Z=26,A=56,mass=PhysicalQuantity(56*ctx.neutron_mass_amu,'amu'),spin=PhysicalQuantity(.5,'hbar'),parity=1,charge=26))
    result=reconstruct_suite(suite,ctx);attach_reconstruction(suite,result)
    tree,report=writeReactionSuite(suite);assert report.isClean,vars(report)
    path=tmp_path/'krm4.xml';tree.write(path)
    loaded,report=readReactionSuite(Document.parse(path));assert report.isClean,vars(report)
    assert max(result.verify_suite(loaded).values())<=1.
    e=np.linspace(10.,300.,301)
    a=prepare_resonances(normalize_suite_pairs(suite,ctx),ctx).evaluate(e)
    b=prepare_resonances(normalize_suite_pairs(loaded,ctx),ctx).evaluate(e)
    for mt,value in a.items():np.testing.assert_allclose(b[mt],value,rtol=3e-13)


def test_exothermic_charged_exit_survives_tabulation_budget():
    source,ctx=rml_model();f=source.resolved[0].formalism
    rr=f.resonanceReactions[1];rr.reactionMT=103;rr.Q=1e6
    rr.kinematics=ChannelKinematics(ChannelParticle(1.,1.,.5,1),ChannelParticle(56.,1.,.5,1),'calculate','calculate')
    prepared=prepare_resonances(source,ctx)
    table=tabulate_resonances(prepared)
    assert max(table.verify_forms(table.forms).values())<=1.
    assert np.all(prepared.evaluate(np.array([10.,100.,200.,300.]))[103]>0)


def test_split_coherent_sectors_preserve_shared_channel_interference():
    source,ctx=rml_model();f=source.resolved[0].formalism
    expected=prepare_resonances(source,ctx).evaluate(np.array([30.,75.,99.9,100.,100.2,180.]))
    first=f.spinGroups[0];second=deepcopy(first)
    first.energies=first.energies[:1];first.widths=first.widths[:1]
    second.energies=second.energies[1:];second.widths=second.widths[1:]
    f.spinGroups.append(second)
    prepared=prepare_resonances(source,ctx)
    assert len(prepared.regions[0].groups[0].levels)==2
    for mt,value in prepared.evaluate(np.array([30.,75.,99.9,100.,100.2,180.])).items():
        np.testing.assert_array_equal(value,expected[mt])
    second.channels[0].scatteringRadius=5.1
    with pytest.raises(ValueError,match='radii|incompatible channel'):
        prepare_resonances(source,ctx)


@pytest.mark.parametrize('amplitudes',[False,True])
def test_explicit_capture_krm4_against_independent_level_matrix(amplitudes):
    source,ctx=rml_model(amplitudes=amplitudes)
    f=source.resolved[0].formalism;f.approximation='RMatrixLimited'
    f.resonanceReactions[2].eliminated=False
    f.spinGroups[0].channels[2].boundaryConditionValue=0.
    # A second explicit photon channel preserves coherent off-diagonal level
    # coupling, which eliminated-capture Reich-Moore does not contain.
    group=f.spinGroups[0]
    photon2=deepcopy(f.resonanceReactions[2]);photon2.label='capture2'
    f.resonanceReactions.append(photon2)
    group.channels.append(Channel('g2','capture2',L=0,channelSpin=0.,columnIndex=4,boundaryConditionValue=0.))
    for index,row in enumerate(group.widths):row.append((-.12 if index==0 else .1) if amplitudes else (-.0288 if index==0 else .02))
    prepared=prepare_resonances(source,ctx);g=prepared.regions[0].groups[0]
    e=np.array([50.,75.,99.9,100.,100.2,180.]);actual=prepared.evaluate(e,diagnostics={})
    gamma=np.asarray(g.reduced)
    for index,energy in enumerate(e):
        functions=[c.functions(np.array([energy])) for c in g.channels]
        p=np.array([v[0][0] for v in functions]);log=np.array([v[1][0] for v in functions])
        matrix=np.diag(np.array([100.,100.2])-energy)-(gamma*log)@gamma.T
        x=np.linalg.solve(matrix,gamma[:,0]);w=gamma.T@x
        beta=np.pi*.01/(ctx.k_squared_per_ev*energy)*.75
        expected={2:0.,51:0.,102:0.}
        for c,channel in enumerate(g.channels):expected[channel.mt]+=beta*4*p[0]*p[c]*abs(w[c])**2
        for mt,value in expected.items():assert actual[mt][index]==pytest.approx(value,rel=3e-11,abs=1e-11)


def test_single_level_explicit_capture_matches_eliminated_capture():
    source,ctx=rml_model();g=source.resolved[0].formalism.spinGroups[0]
    g.energies=g.energies[:1];g.widths=g.widths[:1]
    explicit=deepcopy(source);f=explicit.resolved[0].formalism
    f.approximation='RMatrixLimited';f.resonanceReactions[2].eliminated=False
    f.spinGroups[0].channels[2].boundaryConditionValue=0.
    e=np.array([30.,75.,99.9,100.,100.1,180.])
    a=prepare_resonances(source,ctx).evaluate(e);b=prepare_resonances(explicit,ctx).evaluate(e)
    for mt,value in a.items():np.testing.assert_allclose(b[mt],value,rtol=2e-12,atol=1e-12)


def test_charged_exit_model_retains_closed_shift_and_open_flux():
    source,ctx=rml_model();f=source.resolved[0].formalism
    f.resonanceReactions[1].reactionMT=103;f.resonanceReactions[1].Q=-75*56/57
    f.resonanceReactions[1].kinematics=ChannelKinematics(ChannelParticle(1.,1.,.5,1),ChannelParticle(56.,1.,.5,1),'calculate','calculate')
    # Above threshold use MeV-scale energies to avoid an unrepresentable
    # charged penetrability; below threshold can still be evaluated at low E.
    source.resolved[0].domainMax=2e6
    f.spinGroups[0].energies=[1e6,1.001e6]
    prepared=prepare_resonances(source,ctx);diagnostics={}
    e=np.array([30.,75.,1e5,1e6,1.001e6,1.5e6])
    result=prepared.evaluate(e,diagnostics=diagnostics)
    assert np.all(result[103][:2]==0) and np.all(result[103][2:]>0)
    assert diagnostics['rml_max_absolute_flux_error']<1e-11
    c=prepared.regions[0].groups[0].channels[1]
    assert c.functions(e[:2])[1][0].real<-.2
    assert np.allclose(result[1],result[2]+result[103]+result[102])


def test_legacy_rm_fallback_never_drops_channel_phase():
    from kika.processing.resonances import UnsupportedResonanceError
    source,ctx=rml_model();f=source.resolved[0].formalism
    f.boundaryCondition='EliminateShiftFunction'
    f.resonanceReactions=[f.resonanceReactions[i] for i in (0,2)]
    g=f.spinGroups[0];g.channels=[g.channels[i] for i in (0,2)]
    g.widths=[[row[i] for i in (0,2)] for row in g.widths]
    for rr in f.resonanceReactions:rr.kinematics=None
    for ch in g.channels:ch.boundaryConditionValue=None
    g.channels[0].phaseShiftMode=1
    with pytest.raises(UnsupportedResonanceError,match='channel phase'):
        prepare_resonances(source,ctx)
