"""Independent level algebra and neutral multichannel reconstruction gates."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.special import spherical_kn
from kika.nuclear_data.model.resonances import (RMatrix, RMatrixSpinGroup, Channel,
    ResonanceReaction, Resonances, ResolvedRegion, ChannelParticle, ChannelKinematics,
    ExternalRMatrix, ComplexChannelFunction)
from kika.nuclear_data.model.quantities import PhysicalQuantity
from kika.processing.resonances import (NeutronContext, prepare_resonances,
    tabulate_resonances, UnsupportedResonanceError)
from kika.processing.resonances.r_matrix import solve_rml, closed_neutral_shift


def rml_model(*, multiple=False, shift='calculate', boundary=.2, threshold=75., amplitudes=True):
    ctx = NeutronContext(56.,.5)
    neutron = ChannelParticle(1.,0.,.5,1); target = ChannelParticle(56.,26.,.5,1)
    physical = ChannelKinematics(neutron,target,'calculate',shift)
    zero = ChannelParticle(0.,0.,0.,1)
    capture = ChannelKinematics(zero,zero,'unity','zero',effective=True)
    rr = [ResonanceReaction('elastic',reactionMT=2,Q=0.,kinematics=physical),
          ResonanceReaction('inelastic',reactionMT=51,Q=-threshold*56/57,kinematics=physical),
          ResonanceReaction('capture',reactionMT=102,Q=1e6,eliminated=True,kinematics=capture)]
    channels = [Channel('n0','elastic',L=0,channelSpin=1.,columnIndex=1,
                        scatteringRadius=5.,hardSphereRadius=0.,radiusUnit='fm',boundaryConditionValue=boundary),
                Channel('out','inelastic',L=0,channelSpin=1.,columnIndex=2,
                        scatteringRadius=6.,hardSphereRadius=0.,radiusUnit='fm',boundaryConditionValue=boundary),
                Channel('g','capture',L=0,channelSpin=0.,columnIndex=3)]
    row = [.4,-.3,.2] if amplitudes else [.1,-.2,.08]
    if multiple:
        channels.append(Channel('n2','elastic',L=2,channelSpin=1.,columnIndex=4,
                                scatteringRadius=5.,hardSphereRadius=0.,radiusUnit='fm',boundaryConditionValue=boundary))
        row.append(.5 if amplitudes else .3)
    sg = RMatrixSpinGroup('J1',spin=1.,parity=1,channels=channels,energies=[100.,100.2],widths=[row,[v*.7 for v in row]])
    f = RMatrix(approximation='ReichMoore',boundaryCondition='Given',reducedWidthAmplitudes=amplitudes,
                resonanceReactions=rr,spinGroups=[sg])
    return Resonances(resolved=[ResolvedRegion(10.,300.,formalism=f)]), ctx


@pytest.mark.parametrize('l',[0,1,2,3,8,20,64])
def test_closed_shift_against_independent_modified_bessel_derivative(l):
    rho = np.array([.05,.4,2.,20.,100.])
    expected = 1+rho*spherical_kn(l,rho,derivative=True)/spherical_kn(l,rho)
    np.testing.assert_allclose(closed_neutral_shift(l,rho),expected,rtol=2e-14,atol=2e-13)
    assert closed_neutral_shift(l,np.array([0.]))[0] == -l
    assert np.isfinite(closed_neutral_shift(l,np.array([1e-200]))).all()


@pytest.mark.parametrize('entrance',[0,1,2])
def test_channel_solve_against_independent_dense_level_solve(entrance):
    e = np.array([99.8,100.,100.1,101.])
    er = np.array([100.,100.2,100.4]); gg = np.array([.01,.05,.003])
    a = np.array([[.4,-.2,.3],[.1,.3,-.4],[.2,-.1,.1]])
    log = np.broadcast_to(np.array([-.2+.3j,-.4+0j,.1+1j]),(len(e),3))
    w,x,y = solve_rml(e,er,gg,a,log,entrance=entrance)
    for i,energy in enumerate(e):
        matrix = np.diag(er-energy-.5j*gg)-(a*log[i])@a.T
        direct = np.linalg.solve(matrix,a[:,entrance])
        np.testing.assert_allclose(x[i],direct,rtol=8e-14,atol=1e-14)
        np.testing.assert_allclose(w[i],a.T@direct,rtol=8e-14,atol=1e-14)


@pytest.mark.parametrize('degenerate',[False,True])
def test_exact_zero_capture_pole_and_dark_state(degenerate):
    count = 2 if degenerate else 1
    a = np.tile([.2,.3],(count,1));e = np.array([np.nextafter(100.,0.),100.,np.nextafter(100.,np.inf)])
    log = np.tile([-.3+.4j,.1+.6j],(len(e),1));diag = {}
    w,x,y = solve_rml(e,[100.]*count,[0.]*count,a,log,diagnostics=diag)
    p = log.imag
    u = 2j*np.sqrt(p*p[:,0,None])*w;u[:,0] += 1
    np.testing.assert_allclose(np.sum(abs(u)**2,axis=1),1.,rtol=4e-14)
    np.testing.assert_allclose(w[0],w[1],rtol=2e-12)
    assert diag['rml_max_solver_residual'] < 1e-12


@pytest.mark.parametrize('multiple',[False,True])
@pytest.mark.parametrize('shift',['zero','calculate'])
def test_model_cross_sections_against_level_space_including_closed_channels(multiple,shift):
    model,ctx = rml_model(multiple=multiple,shift=shift)
    prepared = prepare_resonances(model,ctx);group = prepared.regions[0].groups[0]
    e = np.array([30.,74.999,75.,75.001,99.99,100.,100.2,180.]);diag = {}
    result = prepared.evaluate(e,diagnostics=diag)
    reference = {mt:np.zeros(len(e)) for mt in (2,51,102)}
    a = np.asarray(group.reduced)
    for i,energy in enumerate(e):
        functions = [c.functions(np.array([energy])) for c in group.channels]
        p = np.array([v[0][0] for v in functions]);log = np.array([v[1][0] for v in functions])
        matrix = np.diag(np.array([100.,100.2])-energy-.5j*np.asarray(group.radiation))-(a*log)@a.T
        beta = np.pi*.01/(ctx.k_squared_per_ev*energy)*3/4
        for entrance in group.entrances:
            x = np.linalg.solve(matrix,a[:,entrance]);w = a.T@x
            for out,c in enumerate(group.channels):reference[c.mt][i] += beta*4*p[entrance]*p[out]*abs(w[out])**2
            reference[102][i] += beta*2*p[entrance]*np.sum(np.asarray(group.radiation)*abs(x)**2)
    for mt,value in reference.items():np.testing.assert_allclose(result[mt],value,rtol=3e-12,atol=1e-11)
    assert np.all(result[51][:3] == 0)
    assert diag['rml_max_absolute_flux_error'] < 1e-12


def test_closed_channel_virtual_shift_is_not_discarded():
    source,ctx = rml_model();e = np.array([50.])
    shifted = prepare_resonances(source,ctx).evaluate(e)[102]
    source.resolved[0].formalism.resonanceReactions[1].kinematics = ChannelKinematics(
        source.resolved[0].formalism.resonanceReactions[1].kinematics.particleA,
        source.resolved[0].formalism.resonanceReactions[1].kinematics.particleB,'calculate','zero')
    removed = prepare_resonances(source,ctx).evaluate(e)[102]
    assert abs(shifted[0]-removed[0])/shifted[0] > 1e-5


def effective_fission_model():
    model,ctx=rml_model(shift='zero')
    f=model.resolved[0].formalism;sg=f.spinGroups[0]
    zero=ChannelParticle(0.,0.,0.,1)
    f.resonanceReactions.append(ResonanceReaction('fission',reactionMT=18,Q=0.,
        kinematics=ChannelKinematics(zero,zero,'unity','zero',effective=True)))
    sg.channels += [Channel('f1','fission',L=0,channelSpin=0.,columnIndex=4,boundaryConditionValue=0.),
                    Channel('f2','fission',L=0,channelSpin=0.,columnIndex=5,boundaryConditionValue=0.)]
    sg.widths[0] += [.25,-.15];sg.widths[1] += [-.1,.3]
    return model,ctx


def test_effective_fission_exits_with_same_pair_and_quantum_numbers_are_orthogonal():
    model,ctx=effective_fission_model();sg=model.resolved[0].formalism.spinGroups[0]
    prepared=prepare_resonances(model,ctx);g=prepared.regions[0].groups[0]
    e=np.array([30.,99.9,100.,100.2,180.]);actual=prepared.evaluate(e)
    a=np.asarray(g.reduced);expected=[]
    for energy in e:
        functions=[c.functions(np.array([energy])) for c in g.channels]
        p=np.array([v[0][0] for v in functions]);log=np.array([v[1][0] for v in functions])
        matrix=np.diag(np.asarray(sg.energies)-energy-.5j*np.asarray(g.radiation))-(a*log)@a.T
        x=np.linalg.solve(matrix,a[:,g.entrances[0]]);w=a.T@x
        beta=np.pi*.01/(ctx.k_squared_per_ev*energy)*3/4
        expected.append(beta*4*p[g.entrances[0]]*sum(p[i]*abs(w[i])**2
            for i,c in enumerate(g.channels) if c.mt==18))
    np.testing.assert_allclose(actual[18],expected,rtol=3e-12,atol=1e-11)
    np.testing.assert_allclose(actual[1],actual[2]+actual[51]+actual[102]+actual[18],rtol=3e-13)


def test_duplicate_physical_pair_sector_is_still_rejected():
    model,ctx=rml_model();sg=model.resolved[0].formalism.spinGroups[0]
    duplicate=deepcopy(sg.channels[0]);duplicate.label='duplicate';duplicate.columnIndex=4
    sg.channels.append(duplicate)
    for row in sg.widths:row.append(row[0])
    with pytest.raises(ValueError,match='duplicate coherent physical channel'):
        prepare_resonances(model,ctx)


def test_effective_fission_columns_survive_complete_suite_and_publication(tmp_path):
    from test_resonance_publication import writable_suite
    from test_resonance_suite import curve,href
    from kika.nuclear_data.model import (Reaction,ReactionId,CrossSection,Background,
        ResonancesWithBackground,Add,Q,EndfProvenance)
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction
    from kika.endf.writers.assemble import writeReconstructedEndfTape
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    from kika.endf.classes.mf2.mf2mt151 import (MF2MT151,Isotope,EnergyRange,
        RMatrixLimited,RML_ParticlePair,RML_Channel,RML_SpinGroup,RML_Resonance)
    from kika.endf.model_adapter import decodeMF2MT151
    import kika
    suite=writable_suite();ctx=NeutronContext(56.,.5)
    # A small ENDF source supplies the bookkeeping required by its writer.
    pairs=[RML_ParticlePair(0.,0.,0.,0.,0.,0.,1e6,-1,0,102,1.,1.),
           RML_ParticlePair(1.,56.,0.,26.,.5,.5,0.,1,0,2,1.,1.),
           RML_ParticlePair(0.,0.,0.,0.,0.,0.,0.,-1,0,18,1.,1.)]
    channels=[RML_Channel(2,0,1.,.2,0.,.5),RML_Channel(1,0,0.,0.,0.,0.),
              RML_Channel(3,0,0.,0.,0.,0.),RML_Channel(3,0,0.,0.,0.,0.)]
    group=RML_SpinGroup(1.,1.,0,0,channels,
        [RML_Resonance(100.,[.4,.2,.25,-.15]),RML_Resonance(100.2,[.28,.14,-.1,.3])])
    section=MF2MT151(number=151);section._za,section._awr,section._mat,section._nis=26056.,56.,2631,1
    section._isotopes=[Isotope(26056.,1.,0,1,[EnergyRange(10.,300.,1,7,0,0,
        RMatrixLimited(1,3,0,.5,.5,pairs,[group]))])]
    model,provenance,report=decodeMF2MT151(section);model.provenance=provenance
    assert report.isClean
    suite.resonances=model
    form=ResonancesWithBackground(Background(resolvedRegion=curve(10.,300.,0.),
        fastRegion=curve(300.,1000.,0.)),resonanceRegionHref='/reactionSuite/resonances/resolved',label='eval')
    reaction=Reaction(ReactionId('fission',ENDF_MT=18),CrossSection({'eval':form}))
    reaction.provenance=EndfProvenance(qm=0.,lr=0)
    reaction.outputChannel.Q=Q(value=0.);suite.reactions.append(reaction)
    suite.sums[1].summands.append(Add(href('fission')))
    result=reconstruct_suite(suite,ctx);attach_reconstruction(suite,result)
    assert writeReconstructedEndfTape(suite,result,tmp_path/'effective.endf').isClean
    path=tmp_path/'effective.xml';kika.write(suite,path,resonance_extensions=True)
    back,_=readReactionSuite(Document.parse(path))
    assert max(result.verify_suite(back).values())<=1
    restored=back.resonances.resolved[0].formalism.spinGroups[0]
    original=model.resolved[0].formalism.spinGroups[0]
    assert sum(c.resonanceReaction=='MT18' for c in restored.channels)==2
    np.testing.assert_array_equal(restored.widths,original.widths)


def test_channel_permutation_preserves_physics_and_snapshot():
    source,ctx = rml_model(multiple=True);clone = deepcopy(source)
    g = clone.resolved[0].formalism.spinGroups[0];order = [3,2,0,1]
    g.channels = [g.channels[i] for i in order];g.widths = [[row[i] for i in order] for row in g.widths]
    prepared = prepare_resonances(clone,ctx);e = np.linspace(60.,140.,103)
    for mt,value in prepare_resonances(source,ctx).evaluate(e).items():
        np.testing.assert_allclose(prepared.evaluate(e)[mt],value,rtol=3e-13,atol=1e-12)
    before = prepared.evaluate(e);g.widths[0][0] *= 100
    for mt,value in before.items():np.testing.assert_array_equal(prepared.evaluate(e)[mt],value)


@pytest.mark.parametrize('kind',['SAMMY','Froehner'])
def test_external_functions_and_direct_absorption_against_channel_inverse(kind):
    source,ctx = rml_model();ch = source.resolved[0].formalism.spinGroups[0].channels[0]
    terms = [('singularityEnergyBelow',0.,'eV'),('singularityEnergyAbove',400.,'eV'),('constantExternalR',.02,'')]
    terms += [('poleStrength',.05,''),('averageRadiationWidth',.4,'eV')] if kind == 'Froehner' else [('constantLogarithmicCoefficient',.04,'')]
    ch.externalRMatrix = ExternalRMatrix(kind,[PhysicalQuantity(v,u,label=n) for n,v,u in terms])
    prepared = prepare_resonances(source,ctx);g = prepared.regions[0].groups[0]
    e = np.array([20.,70.,99.5,100.,100.2,210.]);value = prepared.evaluate(e)
    t = (2*e-400)/400
    z = .02+.1*np.arctanh(t)+1j*.4/400/(1-t*t) if kind == 'Froehner' else .02-.04*np.log((400-e)/e)
    np.testing.assert_allclose(g.channels[0].external.evaluate(e),z,rtol=2e-14)
    # Deliberately use an explicit channel inverse as an independent small reference.
    a = np.asarray(g.reduced)
    for i,energy in enumerate(e):
        f = [c.functions(np.array([energy])) for c in g.channels]
        p = np.array([v[0][0] for v in f]);log = np.array([v[1][0] for v in f])
        d = np.array([100.,100.2])-energy-.5j*np.asarray(g.radiation)
        r = (a.T/d)@a+np.diag([z[i],0.])
        y = np.linalg.inv(np.eye(2)-np.diag(log)@r)[:,0]
        capture = 2*p[0]*np.sum(np.asarray(g.radiation)*abs(a@y/d)**2)+4*p[0]*z[i].imag*abs(y[0])**2
        beta = np.pi*.01/(ctx.k_squared_per_ev*energy)*3/4
        assert value[102][i] == pytest.approx(beta*capture,rel=3e-12)


def test_tabulator_includes_threshold_and_passes_independent_points():
    source,ctx = rml_model(amplitudes=False);prepared = prepare_resonances(source,ctx)
    table = tabulate_resonances(prepared)
    e = np.unique(np.r_[np.linspace(10.,300.,2003),75.,75.+np.geomspace(1e-8,1.,40),100.,100.2])
    from kika.processing.resonances.grid import error_ratio
    for mt,value in prepared.evaluate(e).items():
        assert np.max(error_ratio(value,table.forms[mt].evaluate(e),table.options)) <= 1.


@pytest.mark.parametrize('capability',['brune','charged','KRM4','KPS'])
def test_unsupported_capabilities_are_not_silently_replaced(capability):
    source,ctx = rml_model();f = source.resolved[0].formalism
    if capability == 'brune':f.boundaryCondition = 'Brune'
    elif capability == 'KRM4':f.approximation = 'RMatrixLimited'
    elif capability == 'KPS':f.spinGroups[0].phaseShiftMode = 1
    else:
        rr = f.resonanceReactions[1];p = rr.kinematics
        rr.kinematics = ChannelKinematics(ChannelParticle(1.,1.,.5,1),p.particleB,p.penetrability,p.shift)
    if capability=='brune':
        with pytest.raises(ValueError,match='boundary constants'):prepare_resonances(source,ctx)
    else:
        with pytest.raises(UnsupportedResonanceError):prepare_resonances(source,ctx)


def test_tabulated_external_snapshot_and_passive_absorption():
    from kika.nuclear_data.model import XYs1d,Axes,Axis
    source,ctx = rml_model();ch=source.resolved[0].formalism.spinGroups[0].channels[0]
    axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'R','')])
    ch.tabulatedBackground=ComplexChannelFunction(XYs1d([10.,300.],[.01,.03],axes=axes),
                                                  XYs1d([10.,300.],[.001,.002],axes=axes))
    prepared=prepare_resonances(source,ctx);e=np.array([20.,75.,100.,200.]);diag={}
    values=prepared.evaluate(e,diagnostics=diag)
    assert diag['rml_max_absolute_flux_error']<1e-12
    assert np.all(values[102]>0)
    ch.tabulatedBackground.imaginary.ys[:]=-1.
    np.testing.assert_array_equal(prepared.evaluate(e)[102],values[102])
    with pytest.raises(ValueError,match='negative external absorption'):prepare_resonances(source,ctx).evaluate(e)


def test_native_gnds_global_boundary_and_suite_pair_normalization(tmp_path):
    from test_resonance_publication import writable_suite
    from kika.nuclear_data.model import Particle,Nuclide,PhysicalQuantity
    from kika.processing.resonances import reconstruct_suite,attach_reconstruction
    from kika.processing.resonances.prepare_r_matrix import normalize_suite_pairs
    from kika.gnds.encode import writeReactionSuite
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document
    suite=writable_suite();source,ctx=rml_model();suite.resonances=source
    f=source.resolved[0].formalism;g=f.spinGroups[0]
    f.resonanceReactions=[f.resonanceReactions[i] for i in (0,2)]
    g.channels=[g.channels[i] for i in (0,2)];g.widths=[[row[i] for i in (0,2)] for row in g.widths]
    f.boundaryConditionValue=.2
    for index,ch in enumerate(g.channels):ch.columnIndex=index+1;ch.boundaryConditionValue=None
    for rr in f.resonanceReactions:
        rr.kinematics=None;rr.href=f"/reactionSuite/reactions/reaction[@label='{rr.label}']"
    suite.PoPs.add(Particle('n',mass=PhysicalQuantity(ctx.neutron_mass_amu,'amu'),spin=PhysicalQuantity(.5,'hbar'),parity=1,charge=0,halflife='stable'))
    suite.PoPs.add(Nuclide(suite.target,Z=26,A=56,mass=PhysicalQuantity(56*ctx.neutron_mass_amu,'amu'),spin=PhysicalQuantity(.5,'hbar'),parity=1,charge=26))
    before=deepcopy(source)
    result=reconstruct_suite(suite,ctx);attach_reconstruction(suite,result)
    tree,report=writeReactionSuite(suite);assert report.isClean,vars(report)
    path=tmp_path/'native_rml.xml';tree.write(path)
    reloaded,report=readReactionSuite(Document.parse(path));assert report.isClean,vars(report)
    assert reloaded.resonances.resolved[0].formalism.boundaryConditionValue==.2
    assert max(result.verify_suite(reloaded).values())<=1.
    e=np.linspace(10.,300.,301)
    a=prepare_resonances(normalize_suite_pairs(suite,ctx),ctx).evaluate(e)
    b=prepare_resonances(normalize_suite_pairs(reloaded,ctx),ctx).evaluate(e)
    for mt,value in a.items():np.testing.assert_allclose(b[mt],value,rtol=3e-13)
    assert f.resonanceReactions[0].kinematics is None
    assert before.resolved[0].formalism.resonanceReactions[0].kinematics is None


def test_global_given_boundary_requires_value_and_channel_has_precedence():
    source,ctx=rml_model();f=source.resolved[0].formalism;f.boundaryConditionValue=.9
    g=prepare_resonances(source,ctx).regions[0].groups[0]
    assert g.channels[0].boundary==.2
    f.spinGroups[0].channels[0].boundaryConditionValue=None
    assert prepare_resonances(source,ctx).regions[0].groups[0].channels[0].boundary==.9
    f.boundaryConditionValue=None
    with pytest.raises(ValueError,match='Given boundary requires'):prepare_resonances(source,ctx)


def test_native_gnds_inelastic_threshold_resolves_declared_residual_and_q(tmp_path):
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
    f=source.resolved[0].formalism
    for i,ch in enumerate(f.spinGroups[0].channels):ch.columnIndex=i+1
    for rr in f.resonanceReactions:
        rr.kinematics=None;rr.href=f"/reactionSuite/reactions/reaction[@label='{rr.label}']"
    form=ResonancesWithBackground(Background(resolvedRegion=curve(10.,300.,0.),fastRegion=curve(300.,1000.,0.)),resonanceRegionHref='/reactionSuite/resonances/resolved',label='eval')
    reaction=Reaction(ReactionId('inelastic',ENDF_MT=51),CrossSection({'eval':form}))
    reaction.outputChannel.products=Products([Product('n',label='n'),Product('Fe56_e1',label='Fe56_e1')])
    for product in reaction.outputChannel.products:
        product.multiplicity=Multiplicity(form=Constant1d(1.,10.,1000.,axes=Axes([Axis(1,'energy_in','eV'),Axis(0,'multiplicity','')]),label='eval'))
        product.distribution=Distribution({'eval':Unspecified(label='eval')})
    reaction.outputChannel.Q=Q(value=-75*56/57,unit='eV',label='eval',domainMin=10.,domainMax=1000.)
    suite.reactions.append(reaction);suite.sums[1].summands.summands.append(Add(href('inelastic')))
    f.resonanceReactions[1].Q=None  # Q is resolved from the linked output channel.
    f.resonanceReactions[1].ejectile='n'
    suite.PoPs.add(Particle('n',mass=PhysicalQuantity(ctx.neutron_mass_amu,'amu'),spin=PhysicalQuantity(.5,'hbar'),parity=1,charge=0,halflife='stable'))
    for pid,level in ((suite.target,0),('Fe56_e1',1)):
        suite.PoPs.add(Nuclide(pid,Z=26,A=56,nuclearLevel=level,mass=PhysicalQuantity(56*ctx.neutron_mass_amu,'amu'),spin=PhysicalQuantity(.5,'hbar'),parity=1,charge=26))
    result=reconstruct_suite(suite,ctx);attach_reconstruction(suite,result)
    tree,report=writeReactionSuite(suite)
    if not report.isClean:print(vars(report))
    assert report.isClean,vars(report)
    path=tmp_path/'native_inelastic.xml';tree.write(path)
    reloaded,report=readReactionSuite(Document.parse(path));assert report.isClean,vars(report)
    assert max(result.verify_suite(reloaded).values())<=1.
    normalized=normalize_suite_pairs(reloaded,ctx)
    rr=normalized.resolved[0].formalism.resonanceReactions[1]
    assert rr.Q==pytest.approx(-75*56/57)
    e=np.array([30.,74.999,75.,75.001,100.,100.2,250.])
    before=prepare_resonances(normalize_suite_pairs(suite,ctx),ctx).evaluate(e)
    for mt,value in prepare_resonances(normalized,ctx).evaluate(e).items():np.testing.assert_allclose(value,before[mt],rtol=4e-13)


def test_signed_capture_width_is_a_sign_of_amplitude_and_negative_dummy_uses_absolute_lab_reference():
    source,ctx=rml_model(amplitudes=False);source.resolved[0].formalism.spinGroups[0].energies[0]=-50.
    expected=prepare_resonances(source,ctx).evaluate(np.array([30.,100.]))
    source.resolved[0].formalism.spinGroups[0].widths[0][2]*=-1
    prepared=prepare_resonances(source,ctx)
    for mt,value in expected.items():np.testing.assert_array_equal(value,prepared.evaluate(np.array([30.,100.]))[mt])
    ch=prepared.regions[0].groups[0].channels[1]
    from kika.processing.resonances.channel_functions import neutral_channel_functions
    reference=abs(50*56/57-75*56/57)
    p=neutral_channel_functions(0,np.sqrt(ch.k2_cm*reference)*6.)[0]
    assert abs(prepared.regions[0].groups[0].reduced[0][1])==pytest.approx(np.sqrt(.2/(2*p)),rel=3e-15)


def test_missing_rml_parity_is_rejected():
    source,ctx = rml_model()
    source.resolved[0].formalism.spinGroups[0].parity = None
    with pytest.raises(ValueError,match='requires declared parity'):
        prepare_resonances(source,ctx)


def test_active_open_channel_underflow_is_rejected_but_closed_channel_is_retained(monkeypatch):
    from kika.processing.resonances import r_matrix
    original = r_matrix.neutral_channel_functions
    def lost_penetrability(l,rho):
        p,s,phase = original(l,rho)
        return np.zeros_like(p),s,phase
    source,ctx = rml_model()
    prepared = prepare_resonances(source,ctx)
    monkeypatch.setattr(r_matrix,'neutral_channel_functions',lost_penetrability)
    with pytest.raises(FloatingPointError,match='open RML penetrability underflows'):
        prepared.evaluate(np.array([100.]))
    channel = prepared.regions[0].groups[0].channels[1]
    p,log,phase = channel.functions(np.array([30.]))
    assert p[0] == 0 and log[0].real < 0 and phase[0] == 0


def test_external_repeated_energy_preserves_both_limits_and_tabulation_segments():
    from kika.nuclear_data.model import XYs1d,Axes,Axis,Regions1d
    from kika.processing.resonances.prepare import group_breaks
    source,ctx = rml_model()
    channel = source.resolved[0].formalism.spinGroups[0].channels[0]
    axes = Axes([Axis(1,'energy_in','eV'),Axis(0,'R','')])
    channel.tabulatedBackground = ComplexChannelFunction(
        XYs1d([10.,150.,150.,300.],[.01,.01,.08,.08],axes=axes),
        XYs1d([10.,300.],[.001,.001],axes=axes))
    prepared = prepare_resonances(source,ctx)
    group = prepared.regions[0].groups[0]
    e = np.array([np.nextafter(150.,0.),150.,np.nextafter(150.,np.inf)])
    np.testing.assert_allclose(group.channels[0].external.evaluate(e).real,[.01,.08,.08],rtol=0,atol=0)
    assert 150. in group_breaks(group)
    result = tabulate_resonances(prepared)
    assert isinstance(result.forms[2],Regions1d)
    left,right = result.forms[2].function1ds
    assert left.domainMax == right.domainMin == 150.
    np.testing.assert_allclose(left.ys[-1],prepared.evaluate(e[:1])[2][0],rtol=1e-13)
    np.testing.assert_allclose(right.ys[0],prepared.evaluate(e[1:2])[2][0],rtol=1e-13)
    assert max(result.verify_forms(result.forms).values()) <= 1.
