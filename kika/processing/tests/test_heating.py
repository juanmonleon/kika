"""kika's own KERMA and damage (roadmap H8): the pieces against closed forms.

The whole calculation is checked against HEATR on real evaluations in
``kika-workspace/kika_dev/processing/heating`` (Fe-56, ENDF/B-VIII.1: elastic
and capture to < 0.1 %, the total per collision to the medians listed in
``docs/library/heating_kerma_damage_plan.md``).
"""
import numpy as np
import pytest

from kika.nuclear_data import model as m
from kika.processing import heating as H


def _table(xs, pdf, E, flat=False, discrete=0):
    from kika.processing.tabulation.tables import pdfTable

    table, _ = pdfTable(np.asarray(xs, float), np.asarray(pdf, float), histogram=flat,
                        discrete=discrete, outer=E)
    return table


def test_displacement_energy_is_heatrs_table():
    assert H.displacementEnergy(26) == 40.0
    assert H.displacementEnergy(92) == 25.0
    assert H.displacementEnergy(6) == 31.0
    assert H.displacementEnergy(74) == 90.0


def test_lindhard_has_a_hard_step_and_no_damage_without_charge():
    E = np.array([10.0, 39.9, 40.0, 1e5])
    d = H.lindhard(E, 26, 55.85, 26, 55.85, 40.0)
    assert d[0] == 0 and d[1] == 0 and d[2] > 0 and 0 < d[3] < 1e5
    assert np.all(H.lindhard(E, 0, 1.0, 26, 55.85, 40.0) == 0)
    # Robinson's formula by hand at 100 keV, Fe in Fe.
    z, a = 26.0, 55.85
    el = 30.724 * z * z * np.sqrt(2 * z ** (2 / 3)) * 2
    kl = 0.0793 * z ** (2 / 3) * np.sqrt(z) * (2 * a) ** 1.5 / ((2 * z ** (2 / 3)) ** 0.75 * a ** 1.5 * np.sqrt(a))
    eps = 1e5 / el
    assert H.lindhard(1e5, z, a, z, a, 40.0) == pytest.approx(
        1e5 / (1 + kl * (3.4008 * eps ** (1 / 6) + 0.40244 * eps ** 0.75 + eps)))


def test_expectations_over_a_table_are_exact_for_polynomials():
    t = _table([0.0, 1.0, 3.0], [1.0, 1.0, 0.0], 1.0)
    # density 1 on [0,1], 1 -> 0 on [1,3]: normalised by 2.
    assert H._expect(t, lambda x: np.ones_like(x)) == pytest.approx(1.0)
    mean = (0.5 + 5.0 / 3.0) / 2.0
    assert H._expect(t, lambda x: x) == pytest.approx(mean)


def test_a_line_counts_with_its_weight():
    # A line of weight 0.5 at 2, and a continuum of integral 1 on [0, 1].
    t = _table([2.0, 0.0, 1.0], [0.5, 1.0, 1.0], 1.0, discrete=1)
    assert H._expect(t, lambda x: x) == pytest.approx((0.5 * 2.0 + 1.0 * 0.5) / 1.5)


def test_isotropic_elastic_deposits_two_a_over_a_plus_one_squared():
    A = 55.454
    mu = _table([-1.0, 1.0], [0.5, 0.5], 1.0)
    angular = m.XYs2d(function1ds=[mu, _table([-1.0, 1.0], [0.5, 0.5], 2e7)])
    form = m.AngularTwoBody(angular=angular, productFrame=m.Frame.centerOfMass)
    E = np.array([1.0, 1e3, 1e6])
    mean = H._meanEnergyOf(form, E, H._Kinematics(A=A, Q=0.0))
    assert np.allclose(E - mean, E * 2 * A / (A + 1) ** 2)


def test_a_level_takes_the_excitation_out_of_the_neutron():
    A, Q = 55.454, -846778.0
    mu = _table([-1.0, 1.0], [0.5, 0.5], 1e6)
    angular = m.XYs2d(function1ds=[mu, _table([-1.0, 1.0], [0.5, 0.5], 2e7)])
    level = m.LevelEnergy(threshold=(A + 1) / A * -Q, massRatio=(A / (A + 1)) ** 2)
    form = m.Uncorrelated(angular=angular, energy=level, productFrame=m.Frame.centerOfMass)
    E = np.array([2e6])
    mean = H._meanEnergyOf(form, E, H._Kinematics(A=A, Q=Q))
    r2 = 1 - (A + 1) / A * -Q / E
    assert mean == pytest.approx(E * (1 + A * A * r2) / (A + 1) ** 2)


def test_photon_lines_ignore_the_frame_and_primaries_take_the_neutron_energy():
    line = m.Uncorrelated(energy=m.DiscretePhotonEnergy(primaryFlag=0, energy=846778.0,
                                                        atomicWeightRatio=55.454),
                          productFrame=m.Frame.centerOfMass)
    E = np.array([1e6, 1e7])
    k = H._Kinematics(A=55.454, Q=0.0, awp=0.0)
    assert np.allclose(H._meanEnergyOf(line, E, k), 846778.0)
    primary = m.Uncorrelated(energy=m.DiscretePhotonEnergy(primaryFlag=2, energy=7.6e6,
                                                           atomicWeightRatio=55.454))
    assert np.allclose(H._meanEnergyOf(primary, E, k), 7.6e6 + 55.454 / 56.454 * E)


def test_mf6_continuum_photon_balance_integrates_density_instead_of_trapezoids(monkeypatch):
    # p(E') = 2(1-E') on [0,1]: <E'> = 1/3, while endpoint trapezoids give zero.
    # This exercises the whole heating path with MF6 provenance and conservation.
    from types import SimpleNamespace as NS
    spectrum = m.XYs2d(function1ds=[_table([0., 1.], [2., 0.], 2.)])
    form = m.Uncorrelated(energy=spectrum, productFrame=m.Frame.lab)
    photon = NS(pid="photon", distribution=NS(forms={"MonteCarlo": form}))
    rx = NS(ENDF_MT=91, outputChannel=NS(Q=NS(value=0.), products=[photon]))
    suite = NS(reactions=[rx])
    monkeypatch.setattr(H, "_yield", lambda product, grid: np.ones(len(grid)))
    monkeypatch.setattr(H, "_statedInMF6", lambda evaluation, mt: True)
    result = H.heatingOf(suite, np.array([2.]), {91: np.array([3.])}, awr=55., Z=26)
    assert result.kerma[0] == pytest.approx(5.)
    assert result.kermaLocal[0] == pytest.approx(6.)
    assert result.perReaction[91][0][0] == pytest.approx(5./3.)


def test_unit_base_cm_mean_comes_from_the_joint_law_not_blended_lab_moments():
    energy=m.XYs2d(function1ds=[_table([0.,1.],[1.,1.],10.),_table([0.,4.],[.5,0.],20.)],
                    interpolationQualifier=m.InterpolationQualifier.unitBase)
    angular=[]
    for E,width,mean_mu in ((10.,1.,.2),(20.,4.,.1)):
        children=[_table([-1.,1.],[.5-1.5*mean_mu,.5+1.5*mean_mu],x) for x in (0.,width)]
        angular.append(m.XYs2d(function1ds=children,outerDomainValue=E))
    form=m.EnergyAngularMC(energy=energy,energyAngular=m.XYs3d(function2ds=angular),productFrame=m.Frame.centerOfMass)
    # At E=15 the unit support has width 2.5. Its density is a half-mixture
    # of a uniform and a triangular law; the joint angular first moments
    # travel with each source component under that mapping.
    cm=2.5*(.5+1./3.)/2
    cross=np.sqrt(2.5)*(.2*2./3.+.1*8./15.)/2
    expected=cm+15./9.+2*np.sqrt(15./9.)*cross
    got=H._meanEnergyOf(form,np.array([15.]),H._Kinematics(A=2.,Q=0.))
    assert got[0]==pytest.approx(expected,abs=1e-12)


def test_unit_base_photon_moments_map_both_support_and_density():
    energy=m.XYs2d(function1ds=[_table([0.,1.],[1.,1.],10.),_table([0.,4.],[.5,0.],20.)],
                    interpolationQualifier=m.InterpolationQualifier.unitBase)
    form=m.Uncorrelated(energy=energy,productFrame=m.Frame.lab)
    k=H._Kinematics(A=2.,Q=0.,awp=0.)
    assert H._meanEnergyOf(form,np.array([15.]),k)[0]==pytest.approx(2.5*(.5+1./3.)/2)
    assert H._meanEnergyOf(form,np.array([15.]),k,moment=2)[0]==pytest.approx(2.5**2*(1./3.+1./6.)/2)


def test_two_body_capture_includes_compound_recoil_and_fixed_photon_kick(monkeypatch):
    from types import SimpleNamespace as NS
    A, binding = .9991673, 2223300.
    grid = np.array([.0253, 1e6, 2e7])
    line = m.Uncorrelated(energy=m.DiscretePhotonEnergy(primaryFlag=2,
                            energy=binding, atomicWeightRatio=A), productFrame=m.Frame.lab)
    photon = NS(pid="photon", distribution=NS(forms={"MonteCarlo": line}))
    sourcePhoton = NS(pid="photon", distribution=NS(forms={"eval": m.AngularTwoBody(angular=m.Isotropic2d())}))
    rx = NS(ENDF_MT=102, outputChannel=NS(Q=NS(value=2224648.), products=[photon]))
    source = NS(outputChannel=NS(products=[sourcePhoton]))
    evaluation = NS(findReactionByENDF_MT=lambda mt: source)
    monkeypatch.setattr(H, "_yield", lambda product, grid: np.ones(len(grid)))
    monkeypatch.setattr(H, "_statedInMF6", lambda evaluation, mt: True)
    result = H.heatingOf(NS(reactions=[rx]), grid, {102: np.ones(len(grid))}, awr=A, Z=1, evaluation=evaluation)
    recoil = grid/(A+1)+binding**2/(2*H.NEUTRON_REST_ENERGY*(A+1))
    assert np.allclose(result.kerma, recoil, rtol=1e-14)
    assert np.allclose(result.kermaLocal-result.kerma, binding+A/(A+1)*grid)
    assert np.allclose(result.damage, H.lindhard(recoil, 1, A+1, 1, A, 25.))


def test_capture_cascade_does_not_take_two_body_recoil_branch():
    from types import SimpleNamespace as NS
    photon = NS(pid="photon", distribution=NS(forms={"eval": m.Uncorrelated()}))
    evaluation = NS(findReactionByENDF_MT=lambda mt: NS(outputChannel=NS(products=[photon])))
    assert H._twoBodyCapture([(photon,m.Uncorrelated())], evaluation, np.array([1.]), 55.) is None


def test_histogram_joint_cm_mean_retains_constant_density_and_angle_in_each_bin():
    spectra=[];angular=[]
    for E,scale in ((100.,1.),(200.,2.)):
        x=np.array([0.,10.,20.])*scale
        spectra.append(m.Xs_pdf_cdf1d(xs=x,pdf=np.array([2.,1.,0.])/(30*scale),
            cdf=np.array([0.,2./3.,1.]),interpolation=m.Interpolation.flat,outerDomainValue=E))
        children=[_table([-1.,1.],[.5-1.5*mu,.5+1.5*mu],e) for e,mu in zip(x,(.2,-.3,0.))]
        angular.append(m.XYs2d(function1ds=children,interpolation=m.Interpolation.flat,outerDomainValue=E))
    form=m.EnergyAngularMC(energy=m.XYs2d(function1ds=spectra,interpolationQualifier=m.InterpolationQualifier.unitBase),
        energyAngular=m.XYs3d(function2ds=angular),productFrame=m.Frame.centerOfMass)
    grid=np.array([100.,150.,200.]);scale=grid/100.
    cm=(2*50+150)/30*scale
    cross=(2*.2*10**1.5-.3*(20**1.5-10**1.5))*2/3/30*np.sqrt(scale)
    expected=cm+grid/9+2*np.sqrt(grid/9)*cross
    actual=H._meanEnergyOf(form,grid,H._Kinematics(A=2.,Q=0.))
    np.testing.assert_allclose(actual,expected,rtol=0.,atol=1e-12)
    from kika.hdf5.distributions import encodeCorrelated,decodeCorrelated
    decoded=decodeCorrelated(encodeCorrelated(form),m.Frame.centerOfMass)
    assert all(str(t.interpolation)=='flat' for t in decoded.energyAngular.function2ds)
    np.testing.assert_allclose(H._meanEnergyOf(decoded,grid,H._Kinematics(A=2.,Q=0.)),expected,rtol=0.,atol=1e-12)


def test_mf12_capture_uses_gheat_compound_recoil_and_fixed_primary_kick(monkeypatch):
    from types import SimpleNamespace as NS
    A,binding=29.,6e6
    E=np.array([.0253,1e6,2e7])
    form=m.Uncorrelated(energy=m.DiscretePhotonEnergy(primaryFlag=2,energy=binding,atomicWeightRatio=A),productFrame=m.Frame.lab)
    p=NS(pid='photon',distribution=NS(forms={'MonteCarlo':form}))
    rx=NS(ENDF_MT=102,outputChannel=NS(Q=NS(value=binding),products=[p]))
    source=NS(provenance=NS(headerFields={'mf12':{'lo':1}}),outputChannel=NS(products=[]))
    evaluation=NS(findReactionByENDF_MT=lambda mt:source)
    monkeypatch.setattr(H,'_yield',lambda p,E:np.ones(len(E)))
    result=H.heatingOf(NS(reactions=[rx]),E,{102:np.ones(len(E))},awr=A,Z=14,evaluation=evaluation,kinematic=True)
    expected=E/(A+1)+binding**2/(2*H.NEUTRON_REST_ENERGY*(A+1))
    np.testing.assert_allclose(result.kerma,expected,rtol=1e-14)
    np.testing.assert_allclose(result.kinematicUpper[102],expected,rtol=1e-14)
    np.testing.assert_allclose(result.damage,H.lindhard(binding**2/(2*H.NEUTRON_REST_ENERGY*(A+1)),14,A+1,14,A,25.),rtol=1e-14)


def test_mf12_capture_retains_neutron_balance_outside_photon_domain(monkeypatch):
    from types import SimpleNamespace as NS
    A,binding=2.,6e6;E=np.array([1e6,2e7+10])
    form=m.Uncorrelated(energy=m.DiscretePhotonEnergy(primaryFlag=2,energy=binding,atomicWeightRatio=A),productFrame=m.Frame.lab)
    multiplicity=NS(forms={},form=NS(domainMin=1e-5,domainMax=2e7))
    p=NS(pid='photon',multiplicity=multiplicity,distribution=NS(forms={'MonteCarlo':form}))
    rx=NS(ENDF_MT=102,outputChannel=NS(Q=NS(value=binding),products=[p]))
    source=NS(provenance=NS(headerFields={'mf12':{'lo':1}}),outputChannel=NS(products=[]))
    monkeypatch.setattr(H,'_yield',lambda p,E:(E<=2e7).astype(float))
    result=H.heatingOf(NS(reactions=[rx]),E,{102:np.ones(len(E))},awr=A,Z=1,
        evaluation=NS(findReactionByENDF_MT=lambda mt:source),kinematic=True)
    assert result.kerma[1]==E[1]+binding
    expected=E[1]/(A+1)+(binding+A*E[1]/(A+1))**2/(2*H.NEUTRON_REST_ENERGY*(A+1))
    assert result.kinematicUpper[102][1]==pytest.approx(expected,rel=1e-14)


def test_mf12_zero_tail_keeps_its_shared_support_endpoint_in_the_correction():
    from types import SimpleNamespace as NS
    A,binding=2.,6e6;E=np.array([2e7,2e7+10.])
    function=m.XYs1d(xs=[1e-5,2e7,2e7,2e8],ys=[1.,1.,0.,0.])
    multiplicity=NS(forms={'eval':function},form=function)
    form=m.Uncorrelated(energy=m.DiscretePhotonEnergy(primaryFlag=2,energy=binding,atomicWeightRatio=A),productFrame=m.Frame.lab)
    p=NS(pid='photon',multiplicity=multiplicity,distribution=NS(forms={'MonteCarlo':form}))
    rx=NS(ENDF_MT=102,outputChannel=NS(Q=NS(value=binding),products=[p]))
    source=NS(provenance=NS(headerFields={'mf12':{'lo':1}}),outputChannel=NS(products=[]))
    result=H.heatingOf(NS(reactions=[rx]),E,{102:np.ones(len(E))},awr=A,Z=1,evaluation=NS(findReactionByENDF_MT=lambda mt:source))
    np.testing.assert_array_equal(H._yield(p,E),[0.,0.])
    np.testing.assert_allclose(result.kerma,[E[0]/(A+1),E[1]+binding],rtol=1e-14)
