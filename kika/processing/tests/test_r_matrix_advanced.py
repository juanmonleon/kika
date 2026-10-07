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
@pytest.mark.parametrize('imaginary',[0.,-.1])
def test_endf_kps_records_are_per_channel_and_survive_parse_and_model(amplitudes,imaginary):
    from kika.endf.classes.mf2.mf2mt151 import (MF2MT151,Isotope,EnergyRange,RMatrixLimited,
        RML_ParticlePair,RML_Channel,RML_SpinGroup,RML_Resonance,TabulatedPhaseShift)
    from kika.endf.parsers.parse_mf2 import parse_mf2_mt151
    from kika.endf.model_adapter import decodeMF2MT151,encodeMF2MT151
    phase=TabulatedPhaseShift([(2,2)],[10.,300.],[.1,.2],[(2,2)],[10.,300.],[imaginary,imaginary])
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
    ch=model.resolved[0].formalism.spinGroups[0].channels[0]
    assert ch.phaseAbsorptionReaction is None
    ch.phaseAbsorptionReaction=model.resolved[0].formalism.resonanceReactions[1].label
    with pytest.raises(ValueError,match='cannot preserve phase absorption'):
        encodeMF2MT151(model,provenance)


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


def test_extreme_barrier_and_inconsistent_relativistic_pair_are_explicit():
    from kika.processing.resonances.coulomb import charged_channel_log_functions
    log_p,shift,_=charged_channel_log_functions(0,1000.,1000.)
    assert log_p < -1000. and np.isfinite(shift)
    with pytest.raises(FloatingPointError,match='not representable'):
        charged_channel_functions(0,1000.,1000.)
    source,ctx=rml_model();f=source.resolved[0].formalism
    rr=f.resonanceReactions[1];rr.reactionMT=103;f.relativisticKinematics=True
    rr.kinematics=ChannelKinematics(ChannelParticle(1.,1.,.5,1),ChannelParticle(56.,1.,.5,1),'calculate','calculate')
    with pytest.raises(ValueError,match='conserve rest energy'):
        prepare_resonances(source,ctx)


@pytest.mark.parametrize('amplitudes',[False,True])
def test_relativistic_charged_exit_uses_one_momentum_for_eta_and_rho(amplitudes):
    from scipy.constants import alpha
    source,ctx=rml_model(amplitudes=amplitudes);f=source.resolved[0].formalism
    f.relativisticKinematics=True;rr=f.resonanceReactions[1]
    rr.reactionMT=103;rr.Q=1e6
    a=ChannelParticle(1.,1.,.5,1)
    b=ChannelParticle(56.-rr.Q*1e-6/ctx.neutron_mass_mev,1.,.5,1)
    rr.kinematics=ChannelKinematics(a,b,'calculate','calculate')
    f.spinGroups[0].energies=[1e7,1.001e7];source.resolved[0].domainMax=2e8
    prepared=prepare_resonances(source,ctx);c=prepared.regions[0].groups[0].channels[1]
    e=np.array([30.,1e5,1e7,1.001e7,5e7,1e8]);t=c.channel_energy(e)
    invariant=(57*ctx.neutron_mass_mev)**2+2*56*ctx.neutron_mass_mev*e*1e-6
    ma,mb=a.massRatio*ctx.neutron_mass_mev,b.massRatio*ctx.neutron_mass_mev
    k=np.sqrt((invariant-(ma+mb)**2)*(invariant-(ma-mb)**2)/(4*invariant))/ctx.hbar_c_mev_fm
    eta=alpha*(ma*mb/(ma+mb))/(ctx.hbar_c_mev_fm*k)
    p,s,phase=charged_channel_functions(c.l,eta,k*6.)
    actual=c.functions(e)
    np.testing.assert_allclose(actual[0],p,rtol=2e-10)
    np.testing.assert_allclose(actual[1].real,s-c.boundary,rtol=2e-10)
    # Nonzero APE must use the same eta and momentum as APT.
    from dataclasses import replace
    from kika.processing.resonances.radii import RadiusFunction
    with_phase=replace(c,phase_radius=RadiusFunction(constant=4.))
    np.testing.assert_allclose(np.sin(with_phase.functions(e)[2]-charged_channel_functions(c.l,eta,k*4.)[2]),0.,atol=2e-12)
    diagnostics={};values=prepared.evaluate(e,diagnostics=diagnostics)
    np.testing.assert_allclose(values[1],values[2]+values[103]+values[102],rtol=2e-14)
    assert diagnostics['rml_max_absolute_flux_error']<1e-11


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


@pytest.mark.parametrize('field',['phaseShiftMode','phaseAbsorptionReaction'])
def test_legacy_rm_fallback_never_drops_channel_phase(field):
    from kika.processing.resonances import UnsupportedResonanceError
    source,ctx=rml_model();f=source.resolved[0].formalism
    f.boundaryCondition='EliminateShiftFunction'
    f.resonanceReactions=[f.resonanceReactions[i] for i in (0,2)]
    g=f.spinGroups[0];g.channels=[g.channels[i] for i in (0,2)]
    g.widths=[[row[i] for i in (0,2)] for row in g.widths]
    for rr in f.resonanceReactions:rr.kinematics=None
    for ch in g.channels:ch.boundaryConditionValue=None
    setattr(g.channels[0],field,1 if field=='phaseShiftMode' else 'capture')
    with pytest.raises(UnsupportedResonanceError,match='channel phase'):
        prepare_resonances(source,ctx)


def test_relativistic_charged_native_suite_publication(tmp_path):
    from test_resonance_publication import writable_suite
    from test_resonance_suite import curve,href
    from kika.nuclear_data.model import (Particle,Nuclide,PhysicalQuantity,Reaction,ReactionId,
        CrossSection,ResonancesWithBackground,Background,Add,Product,Products,Q,
        Multiplicity,Constant1d,Axes,Axis,Distribution,Unspecified)
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction
    from kika.processing.resonances.prepare_r_matrix import normalize_suite_pairs
    from kika.gnds.encode import writeReactionSuite
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    suite=writable_suite();source,ctx=rml_model();suite.resonances=source
    f=source.resolved[0].formalism;f.relativisticKinematics=True
    f.resonanceReactions[1].label='charged';f.resonanceReactions[1].reactionMT=103
    f.resonanceReactions[1].Q=None;f.resonanceReactions[1].ejectile='H1'
    f.spinGroups[0].channels[1].resonanceReaction='charged'
    for rr in f.resonanceReactions:
        rr.kinematics=None;rr.href=f"/reactionSuite/reactions/reaction[@label='{rr.label}']"
    form=ResonancesWithBackground(Background(resolvedRegion=curve(10.,300.,0.),fastRegion=curve(300.,1000.,0.)),resonanceRegionHref='/reactionSuite/resonances/resolved',label='eval')
    reaction=Reaction(ReactionId('charged',ENDF_MT=103),CrossSection({'eval':form}))
    reaction.outputChannel.products=Products([Product(pid,label=pid) for pid in ('H1','Mn56')])
    for p in reaction.outputChannel.products:
        p.multiplicity=Multiplicity(form=Constant1d(1.,10.,1000.,axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'multiplicity','')]),label='eval'))
        p.distribution=Distribution({'eval':Unspecified(label='eval')})
    reaction.outputChannel.Q=Q(value=1e6,unit='eV',label='eval',domainMin=10.,domainMax=1000.)
    suite.reactions.append(reaction);suite.sums[1].summands.summands.append(Add(href('charged')))
    suite.PoPs.add(Particle('n',mass=PhysicalQuantity(ctx.neutron_mass_amu,'amu'),spin=PhysicalQuantity(.5,'hbar'),parity=1,charge=0,halflife='stable'))
    # Explicit synthetic masses conserve the stated Q. They are not a fit or
    # an alteration of any evaluated material.
    for pid,z,a,mass in ((suite.target,26,56,56.),('H1',1,1,1.),('Mn56',25,56,56.-1./ctx.neutron_mass_mev)):
        suite.PoPs.add(Nuclide(pid,Z=z,A=a,mass=PhysicalQuantity(mass*ctx.neutron_mass_amu,'amu'),spin=PhysicalQuantity(.5,'hbar'),parity=1,charge=z))
    result=reconstruct_suite(suite,ctx);attach_reconstruction(suite,result)
    tree,report=writeReactionSuite(suite);assert report.isClean,vars(report)
    path=tmp_path/'charged_krl.xml';tree.write(path)
    loaded,report=readReactionSuite(Document.parse(path));assert report.isClean,vars(report)
    assert loaded.resonances.resolved[0].formalism.relativisticKinematics
    assert max(result.verify_suite(loaded).values())<=1.
    e=np.array([10.,30.,100.,100.2,250.,300.])
    before=prepare_resonances(normalize_suite_pairs(suite,ctx),ctx).evaluate(e)
    after=prepare_resonances(normalize_suite_pairs(loaded,ctx),ctx).evaluate(e)
    for mt,v in before.items():np.testing.assert_allclose(after[mt],v,rtol=4e-13)
    assert np.all(after[103]>0)


@pytest.mark.parametrize('q,explicit,amplitudes,reference',[
    (1e6,False,False,[.013919645971544983,.027839291943089976,.011135716777197869]),
    (1e6,False,True,[.06379374985094877,.04472193848987714,.004661976399958515]),
    (1e6,True,False,[.01391964663830124,.027839293276602475,.011135717310640957]),
    (1e6,True,True,[.06379375038486433,.04472193886417304,.004661976439012291]),
    (-1e6,False,False,[.013916170340474206,.027832340680948354,.01113293627190646]),
    (-1e6,False,True,[.06912105045643953,.04362222568653895,.005051289612558657]),
    (-1e6,True,False,[.013916171055364763,.027832342110729498,.011132936844291886]),
    (-1e6,True,True,[.06912105108452296,.043622226082921774,.005051289658497106]),
])
def test_relativistic_charged_against_independent_60_digit_level_matrix(q,explicit,amplitudes,reference):
    # Constants from the independent invariant/Coulomb/level-space gate,
    # archived in workspace evidence. No production kernel generates them.
    source,ctx=rml_model(amplitudes=amplitudes);f=source.resolved[0].formalism
    f.relativisticKinematics=True;f.approximation='RMatrixLimited' if explicit else 'ReichMoore'
    f.resonanceReactions[2].eliminated=not explicit
    f.spinGroups[0].channels[2].boundaryConditionValue=0.
    rr=f.resonanceReactions[1];rr.reactionMT=103;rr.Q=q
    rr.kinematics=ChannelKinematics(ChannelParticle(1.,1.,.5,1),
        ChannelParticle(56.-q*1e-6/ctx.neutron_mass_mev,1.,.5,1),'calculate','calculate')
    f.spinGroups[0].energies=[1e7,1.001e7];source.resolved[0].domainMax=2e8
    prepared=prepare_resonances(source,ctx)
    result=prepared.evaluate(np.array([1e7]))
    np.testing.assert_allclose([result[mt][0] for mt in (2,103,102)],reference,rtol=2e-10,atol=0.)
    if q<0:
        c=prepared.regions[0].groups[0].channels[1]
        e=np.array([np.nextafter(c.threshold,0.),c.threshold])
        p,log,_=c.functions(e)
        np.testing.assert_array_equal(p,0.)
        np.testing.assert_allclose(log.real[0],log.real[1],rtol=2e-11)


def _complex_phase(real,imaginary):
    from kika.nuclear_data.model import XYs1d,Axes,Axis
    from kika.nuclear_data.model.resonances import ComplexChannelFunction
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'phase','')])
    return ComplexChannelFunction(XYs1d([10.,300.],[real,real],axes=axes),
        XYs1d([10.,300.],[imaginary,imaginary],axes=axes))


@pytest.mark.parametrize('imaginary',[-1e-16,-.2,-1000.])
def test_passive_kps_potential_only_direct_absorption_and_optical_theorem(imaginary):
    source,ctx=rml_model();f=source.resolved[0].formalism;g=f.spinGroups[0]
    g.energies=[];g.widths=[]
    g.channels[0].additionalPhaseShift=_complex_phase(.12,imaginary)
    g.channels[0].phaseAbsorptionReaction='capture'
    potential=deepcopy(g);potential.spin=0.
    for ch in potential.channels:ch.channelSpin=0.
    f.spinGroups.append(potential)
    e=np.array([10.,30.,100.,300.]);result=prepare_resonances(source,ctx).evaluate(e)
    beta=np.pi*.01/(ctx.k_squared_per_ev*e)
    s=np.exp(-2j*(.12+1j*imaginary))
    np.testing.assert_allclose(result[2],beta*abs(1-s)**2,rtol=3e-14)
    np.testing.assert_allclose(result[102],beta*(-np.expm1(4*imaginary)),rtol=3e-14)
    np.testing.assert_allclose(result[1],2*beta*(1-s.real),rtol=3e-14)
    assert np.all(result[102]>0)


@pytest.mark.parametrize('explicit',[False,True])
def test_passive_kps_level_matrix_and_separate_reaction_ownership(explicit):
    from kika.nuclear_data.model.resonances import ResonanceReaction
    source,ctx=rml_model();f=source.resolved[0].formalism;g=f.spinGroups[0]
    if explicit:
        f.approximation='RMatrixLimited';f.resonanceReactions[2].eliminated=False
        g.channels[2].boundaryConditionValue=0.
    f.resonanceReactions.append(ResonanceReaction('opticalLoss',reactionMT=18,
        kinematics=deepcopy(f.resonanceReactions[2].kinematics)))
    g.channels[0].additionalPhaseShift=_complex_phase(.12,-.2)
    g.channels[0].phaseAbsorptionReaction='opticalLoss'
    g.channels[1].additionalPhaseShift=_complex_phase(-.04,-.1)
    g.channels[1].phaseAbsorptionReaction='capture'
    potential=deepcopy(g);potential.spin=0.;potential.energies=[];potential.widths=[]
    for ch in potential.channels:ch.channelSpin=0.
    f.spinGroups.append(potential)
    prepared=prepare_resonances(source,ctx)
    e=np.array([30.,99.8,100.,100.1,100.2,101.,250.])
    diagnostics={};actual=prepared.evaluate(e,diagnostics=diagnostics)
    expected={mt:np.zeros(len(e)) for mt in (2,51,102,18)}
    for group in prepared.regions[0].groups:
        gamma=np.asarray(group.reduced).reshape(len(group.levels),len(group.channels))
        for i,energy in enumerate(e):
            values=[c.functions(np.array([energy])) for c in group.channels]
            p=np.array([v[0][0] for v in values]);logs=np.array([v[1][0] for v in values]);ph=np.array([v[2][0] for v in values])
            d=np.diag([lv.energy-energy-.5j*gg for lv,gg in zip(group.levels,group.radiation)])-(gamma*logs)@gamma.T
            x=np.linalg.solve(d,gamma[:,0]) if len(gamma) else np.zeros(0,complex)
            core=2j*np.sqrt(p*p[0])*(gamma.T@x);core[0]+=1.
            observed=np.exp(-1j*ph)*core*np.exp(-1j*ph[0])
            beta=np.pi*.01/(ctx.k_squared_per_ev*energy)*(2*group.spin+1)/4
            for out,c in enumerate(group.channels):expected[c.mt][i]+=beta*abs((1 if out==0 else 0)-observed[out])**2
            core_capture=2*p[0]*sum(gg*abs(xx)**2 for gg,xx in zip(group.radiation,x))
            expected[102][i]+=beta*np.exp(2*ph[0].imag)*core_capture
            for out,c in enumerate(group.channels):
                loss=np.exp(2*ph[0].imag)*(-np.expm1(2*ph[out].imag))*abs(core[out])**2
                if out==0:loss+=-np.expm1(2*ph[0].imag)
                if c.phase_absorption_mt is not None:expected[c.phase_absorption_mt][i]+=beta*loss
    for mt in expected:np.testing.assert_allclose(actual[mt],expected[mt],rtol=8e-13,atol=1e-15)
    assert diagnostics['rml_max_absolute_flux_error']<1e-12


@pytest.mark.parametrize('capture,elastic,radiative',[
    (True,7.4452865664117607e-6,1.6929247875343792e-4),
    (False,7.4459126596722876e-6,0.)])
def test_actual_extreme_coulomb_kernel_against_70_digit_level_reference(capture,elastic,radiative):
    from scipy.constants import alpha
    source,ctx=rml_model();f=source.resolved[0].formalism;g=f.spinGroups[0]
    rr=f.resonanceReactions[1];rr.reactionMT=103
    rr.kinematics=ChannelKinematics(ChannelParticle(1.,1.,.5,1),ChannelParticle(56.,25.,.5,1),'calculate','calculate')
    mu=ctx.neutron_mass_mev*56/57;k=alpha*25*mu/ctx.hbar_c_mev_fm/1000
    rr.Q=k*k/(2*mu*1e-6/ctx.hbar_c_mev_fm**2)-100*56/57
    g.channels[1].scatteringRadius=1000/k
    if not capture:
        for row in g.widths:row[2]=0.
    diagnostics={};result=prepare_resonances(source,ctx).evaluate(np.array([100.]),diagnostics=diagnostics)
    assert result[2][0]==pytest.approx(elastic,rel=3e-11)
    assert result[102][0]==pytest.approx(radiative,rel=3e-11,abs=0.)
    assert result[103][0]==0.  # The final ~6e-497 b observable rounds to zero.
    assert diagnostics['rml_underflow_bounded_solves']==1
    assert diagnostics['rml_underflow_max_log_perturbation_bound'] < -1100.


@pytest.mark.parametrize('owner,imaginary,message',[(None,-.1,'reaction ownership'),('missing',-.1,'reaction ownership'),('elastic',-.1,'reaction ownership'),('capture',.1,'passive KPS')])
def test_kps_absorption_rejects_missing_ownership_and_gain(owner,imaginary,message):
    source,ctx=rml_model();ch=source.resolved[0].formalism.spinGroups[0].channels[1]
    ch.additionalPhaseShift=_complex_phase(.1,imaginary);ch.phaseAbsorptionReaction=owner
    with pytest.raises(ValueError,match=message):prepare_resonances(source,ctx)


def test_kps_absorption_model_snapshot_and_tabulation_preserve_imaginary_step():
    from kika.nuclear_data.model import XYs1d,Axes,Axis,Regions1d
    source,ctx=rml_model();f=source.resolved[0].formalism;g=f.spinGroups[0]
    ch=g.channels[1];ch.additionalPhaseShift=_complex_phase(.1,-.1);ch.phaseAbsorptionReaction='capture'
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'phase','')])
    ch.additionalPhaseShift.imaginary=XYs1d([10.,150.,150.,300.],[-.1,-.1,-.3,-.3],axes=axes)
    prepared=prepare_resonances(source,ctx)
    points=np.array([np.nextafter(150.,0.),150.,np.nextafter(150.,np.inf)])
    before=prepared.evaluate(points)
    ch.additionalPhaseShift.imaginary.ys[:]=-.8;ch.phaseAbsorptionReaction='missing'
    after=prepared.evaluate(points)
    for mt in before:np.testing.assert_array_equal(before[mt],after[mt])
    table=tabulate_resonances(prepared)
    assert isinstance(table.forms[102],Regions1d)
    assert max(table.verify_forms(table.forms).values())<=1.


def test_kps_absorption_gnds_reports_unrepresentable_processing_metadata():
    import xml.etree.ElementTree as ET
    from kika.gnds.encode_resonances import writeResonances
    from kika.nuclear_data.model import ConversionReport
    source,_=rml_model();ch=source.resolved[0].formalism.spinGroups[0].channels[1]
    ch.additionalPhaseShift=_complex_phase(.1,-.1);ch.phaseAbsorptionReaction='capture'
    report=ConversionReport()
    writeResonances(ET.Element('reactionSuite'),source,report,('10','300'))
    assert not report.isClean
    assert any('KPS' in str(value) for value in report.unsupported)


def test_kps_absorption_reconstructs_and_attaches_native_suite():
    from test_resonance_publication import writable_suite
    from test_resonance_suite import curve,href
    from kika.nuclear_data.model import (Reaction,ReactionId,CrossSection,
        ResonancesWithBackground,Background,Add)
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction
    suite=writable_suite();source,ctx=rml_model();suite.resonances=source
    ch=source.resolved[0].formalism.spinGroups[0].channels[1]
    ch.additionalPhaseShift=_complex_phase(.1,-.2);ch.phaseAbsorptionReaction='capture'
    form=ResonancesWithBackground(Background(resolvedRegion=curve(10.,300.,0.),
        fastRegion=curve(300.,1000.,0.)),resonanceRegionHref='/reactionSuite/resonances/resolved',label='eval')
    suite.reactions.append(Reaction(ReactionId('inelastic',ENDF_MT=51),CrossSection({'eval':form})))
    suite.sums[1].summands.summands.append(Add(href('inelastic')))
    result=reconstruct_suite(suite,ctx);attach_reconstruction(suite,result)
    assert max(result.verify_suite(suite).values())<=1.
    assert suite.resonances.resolved[0].formalism.spinGroups[0].channels[1].phaseAbsorptionReaction=='capture'


@pytest.mark.parametrize('l,constant',[(1,1.),(2,9.)])
def test_logarithmic_neutral_limit_retains_unrepresentable_penetrability(l,constant):
    from kika.processing.resonances.coulomb import charged_channel_log_functions
    rho=1e-200
    log_p,shift,phase=charged_channel_log_functions(l,0.,rho)
    assert log_p==pytest.approx((2*l+1)*np.log(rho)-np.log(constant),abs=1e-12)
    assert shift==-l and phase==0.


@pytest.mark.parametrize('log_p',[-740.,-746.,-1200.])
@pytest.mark.parametrize('capture,phase',[ (True,False),(False,False),(False,True)])
def test_logarithmic_exit_preserves_rescued_observables_and_exact_poles(monkeypatch,log_p,capture,phase):
    from kika.processing.resonances import coulomb
    source,ctx=rml_model();f=source.resolved[0].formalism;g=f.spinGroups[0]
    source.resolved[0].domainMin=99.8;source.resolved[0].domainMax=101.
    rr=f.resonanceReactions[1];rr.reactionMT=103
    rr.kinematics=ChannelKinematics(ChannelParticle(1.,1.,.5,1),ChannelParticle(56.,1.,.5,1),'calculate','calculate')
    if not capture:
        for row in g.widths:row[2]=0.
    if phase:
        g.channels[1].additionalPhaseShift=_complex_phase(.1,-.2)
        g.channels[1].phaseAbsorptionReaction='capture'
    def channel(l,eta,rho):
        return np.full_like(rho,log_p),np.full_like(rho,-.7),np.zeros_like(rho)
    monkeypatch.setattr(coulomb,'charged_channel_log_functions',channel)
    prepared=prepare_resonances(source,ctx);group=prepared.regions[0].groups[0]
    e=np.array([99.8,100.,100.1,100.2,101.]);diagnostics={}
    actual=prepared.evaluate(e,diagnostics=diagnostics)
    gamma=np.asarray(group.reduced);radiation=np.asarray(group.radiation)
    for i,energy in enumerate(e):
        pn=5*np.sqrt(ctx.k_squared_per_ev*energy)
        logs=np.array([-.2+1j*pn,-.9+0j])
        d=np.diag(np.array([100.,100.2])-energy-.5j*radiation)-(gamma*logs)@gamma.T
        x=np.linalg.solve(d,gamma[:,0]);w=gamma.T@x
        beta=np.pi*.01/(ctx.k_squared_per_ev*energy)*.75
        expected=np.exp(np.log(beta)+np.log(4*pn)+log_p+2*np.log(abs(w[1]))-(.4 if phase else 0.))
        assert abs(actual[103][i]-expected)<=4*np.nextafter(0.,1.)
        if log_p>=-746.:assert actual[103][i]>0
        if phase:
            loss=np.exp(np.log(beta)+np.log(4*pn)+log_p+2*np.log(abs(w[1]))+np.log(-np.expm1(-.4)))
            assert abs(actual[102][i]-loss)<=4*np.nextafter(0.,1.)
        else:
            expected_capture=beta*2*pn*np.sum(radiation*abs(x)**2)
            np.testing.assert_allclose(actual[102][i],expected_capture,rtol=2e-12,atol=0.)
    assert diagnostics['rml_underflow_bounded_solves']==len(e)
    assert diagnostics['rml_underflow_max_log_perturbation_bound'] < -700.
    assert diagnostics['rml_max_absolute_flux_error']<1e-12
    if capture and log_p==-746.:
        table=tabulate_resonances(prepared)
        assert max(table.verify_forms(table.forms).values())<=1.


def test_logarithmic_exit_rejects_an_unbounded_singular_limit():
    from kika.processing.resonances.r_matrix import solve_rml
    with pytest.raises(FloatingPointError,match='unrepresentable penetrability'):
        solve_rml([1.],[1.],[0.],[[1e-160,1.]],[[.2j,0j]],
            log_penetrability=np.array([[np.log(.2),-750.]]),error_bounds={})


@pytest.mark.parametrize('l,eta,rho,log_p,shift,phase',[
    (0,201.,300.,-69.4117667370198578,-173.9347792081184676,2.0463621957271838e-33),
    (0,500.,800.,-121.3061245022519847,-398.7432497719208222,2.5961915583029375e-56),
    (2,500.,800.,-121.3121056686043086,-398.7507783562332885,2.5806612597095178e-56),
    (0,1000.,1900.,-24.2059855936139224,-430.7482270259691634,3.5253284460346487e-14),
    (0,1000.,1000.,-1134.6853151028429608,-999.4998746240512031,0.),
    (0,200.,.1,-1229.5526784282512425,-6.0874603576934330,0.)])
def test_extreme_barrier_against_independent_whittaker_limit(l,eta,rho,log_p,shift,phase):
    from kika.processing.resonances.coulomb import charged_channel_log_functions
    actual=charged_channel_log_functions(l,eta,rho)
    assert actual[0]==pytest.approx(log_p,abs=2e-8)
    assert actual[1]==pytest.approx(shift,rel=2e-9)
    if phase:assert actual[2]==pytest.approx(phase,rel=3e-8,abs=0.)
    else:assert actual[2]==0.


def test_integer_level_energies_do_not_truncate_reference_widths():
    source,ctx=rml_model();source.resolved[0].formalism.spinGroups[0].energies=[100,101]
    group=prepare_resonances(source,ctx).regions[0].groups[0]
    expected=2*.4**2*5*np.sqrt(ctx.k_squared_per_ev*100)
    assert group.levels[0].neutron==pytest.approx(expected,rel=2e-14)


def test_kps_first_chance_fission_ownership_uses_the_kernel_fission_total():
    from kika.nuclear_data.model.resonances import ResonanceReaction
    source,ctx=rml_model();f=source.resolved[0].formalism
    f.resonanceReactions.append(ResonanceReaction('fissionLoss',reactionMT=19,
        kinematics=deepcopy(f.resonanceReactions[2].kinematics)))
    ch=f.spinGroups[0].channels[1]
    ch.additionalPhaseShift=_complex_phase(.1,-.2);ch.phaseAbsorptionReaction='fissionLoss'
    actual=prepare_resonances(source,ctx).evaluate(np.array([100.1]))
    assert 19 not in actual and actual[18][0]>0
    np.testing.assert_allclose(actual[1],actual[2]+actual[51]+actual[102]+actual[18],rtol=2e-14)
