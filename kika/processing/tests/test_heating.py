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


@pytest.mark.parametrize("explicit_primary", [False, True])
def test_two_body_capture_includes_compound_recoil_and_fixed_photon_kick(monkeypatch, explicit_primary):
    from types import SimpleNamespace as NS
    A, binding = .9991673, 2223300.
    grid = np.array([.0253, 1e6, 2e7])
    line = m.Uncorrelated(energy=m.DiscretePhotonEnergy(primaryFlag=2,
                            energy=binding, atomicWeightRatio=A), productFrame=m.Frame.lab)
    photon = NS(pid="photon", distribution=NS(forms={"MonteCarlo": line}))
    sourceForm = (m.Uncorrelated(angular=m.Isotropic2d(),
                      energy=m.PrimaryGamma(value=binding, domainMin=1e-5, domainMax=2e7),
                      productFrame=m.Frame.centerOfMass) if explicit_primary
                  else m.AngularTwoBody(angular=m.Isotropic2d()))
    sourcePhoton = NS(pid="photon", distribution=NS(forms={"eval": sourceForm}))
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


@pytest.mark.parametrize("name,shape", [("SimpleMaxwellianFission", 1.5), ("Evaporation", 2.)])
def test_analytic_emission_damage_retains_logarithmic_parameter_law(name, shape, monkeypatch):
    from scipy.special import gammainc
    theta = m.XYs1d(xs=[1e-5, 1.5e7, 3e7], ys=[1295500., 1492300., 1499835.],
                    interpolation=m.Interpolation.linlog)
    law = getattr(m, name)(U=m.PhysicalQuantity(-3e7, "eV"), theta=theta)
    form = m.Uncorrelated(energy=law, productFrame=m.Frame.lab)
    E = np.array([1e-5, 786.4939, 1e4, 1e6, 2e7, 3e7]); A = 238.
    # With an identity partition the angular cross term integrates to zero,
    # and the spectrum's first moment has a closed incomplete-gamma form.
    monkeypatch.setattr(H, "lindhard", lambda recoil, *args: recoil); monkeypatch.setattr(H, "_heating_native", None)  # the NumPy reference uses the stub
    t = theta.evaluate(E); upper = (E + 3e7) / t
    expected = (E + shape * t * gammainc(shape+1, upper) / gammainc(shape, upper)) / A
    actual = H._emissionDamage(form, E, A, 92, 0.)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=.1)


def test_analytic_emission_damage_refuses_invalid_parameter_instead_of_zero():
    law = m.SimpleMaxwellianFission(U=m.PhysicalQuantity(-3e7, "eV"),
        theta=m.XYs1d(xs=[1., 2.], ys=[-1., -1.]))
    with pytest.raises(ValueError, match="positive theta"):
        H._emissionDamage(m.Uncorrelated(energy=law), np.array([1., 2.]), 238., 92, 25.)


def test_analytic_emission_damage_does_not_interpolate_across_zero_support(monkeypatch):
    from scipy.special import gammainc
    law = m.SimpleMaxwellianFission(U=m.PhysicalQuantity(1e6, "eV"),
        theta=m.XYs1d(xs=[1., 2e6], ys=[1e6, 1e6]))
    E = np.array([1e-5, 1e6, 1e6+1., 2e6])
    monkeypatch.setattr(H, "lindhard", lambda recoil, *args: recoil); monkeypatch.setattr(H, "_heating_native", None)  # the NumPy reference uses the stub
    actual = H._emissionDamage(m.Uncorrelated(energy=law), E, 238., 92, 0.)
    np.testing.assert_array_equal(actual[:2], [0., 0.])
    upper = (E[2:]-1e6)/1e6
    expected = (E[2:] + 1.5e6*gammainc(2.5, upper)/gammainc(1.5, upper))/238.
    np.testing.assert_allclose(actual[2:], expected, rtol=2e-5, atol=.1)


@pytest.mark.parametrize("qualifier", [None, m.InterpolationQualifier.unitBase])
def test_photon_unit_base_maps_only_continuum_and_retains_line_mass(qualifier):
    first = m.Xs_pdf_cdf1d(xs=[10., 1., 3.], pdf=[.2, .4, .4], cdf=[.2, .2, 1.],
        numberOfDiscrete=1, outerDomainValue=10.)
    last = m.Xs_pdf_cdf1d(xs=[20., 2., 6.], pdf=[.4, .15, .15], cdf=[.4, .4, 1.],
        numberOfDiscrete=1, outerDomainValue=20.)
    form = m.Uncorrelated(energy=m.XYs2d(function1ds=[first, last],
        interpolationQualifier=qualifier), productFrame=m.Frame.lab)
    E = np.array([15.]); k = H._Kinematics(A=55., Q=0., awp=0.)
    # At the midpoint the continuum is uniform on [1.5, 4.5], mass .7;
    # the two unchanged lines carry mean mass .1 at 10 and .2 at 20.
    assert H._meanEnergyOf(form, E, k)[0] == pytest.approx(7.1, abs=1e-12)
    assert H._meanEnergyOf(form, E, k, moment=2)[0] == pytest.approx(96.825, abs=1e-12)
    from kika.hdf5.distributions import encodeContinuous, decodeContinuous
    decoded = decodeContinuous(encodeContinuous(form.energy))
    roundtrip = m.Uncorrelated(energy=decoded, productFrame=m.Frame.lab)
    assert H._meanEnergyOf(roundtrip, E, k)[0] == pytest.approx(7.1, abs=1e-12)


@pytest.mark.parametrize("name,shape", [("SimpleMaxwellianFission", 1.5), ("Evaporation", 2.)])
@pytest.mark.parametrize("power", [.5, 1, 2])
def test_analytic_energy_moments_against_normalized_adaptive_quadrature(name, shape, power):
    from scipy.integrate import quad
    theta = m.XYs1d(xs=[1., 1e8], ys=[1e5, 3e6], interpolation=m.Interpolation.linlog)
    law = getattr(m, name)(U=m.PhysicalQuantity(1., "eV"), theta=theta)
    E = np.array([1., 1.+1e-9, 2., 1e4, 1e6, 1e8])
    expected = [0.]
    for e in E[1:]:
        t = float(theta.evaluate(e)); top = law.upperBound(e); ratio = top/t
        # Scale to [0,1] for tiny support; otherwise split the independent
        # sqrt-energy quadrature where the exponential density is appreciable.
        if ratio < 1:
            norm = quad(lambda u: u**(shape-1)*np.exp(-ratio*u), 0, 1, epsabs=1e-12)[0]
            moment = top**power*quad(lambda u: u**(shape-1+power)*np.exp(-ratio*u), 0, 1, epsabs=1e-12)[0]/norm
        else:
            cuts = np.unique(np.r_[0., np.sqrt(ratio), 1., 2., 4., 8.])
            cuts = cuts[cuts <= np.sqrt(ratio)]
            def integral(p):
                return sum(quad(lambda v: v**(2*shape-1+2*p)*np.exp(-v*v), a, b, epsabs=1e-12)[0]
                           for a,b in zip(cuts[:-1], cuts[1:]))
            moment = t**power*integral(power)/integral(0)
        expected.append(moment)
    np.testing.assert_allclose(H._analyticSpectrumMoment(law, E, power), expected, rtol=2e-9, atol=1e-12)
    if power in (1,2):
        form = m.Uncorrelated(energy=law, productFrame=m.Frame.lab)
        np.testing.assert_allclose(H._meanEnergyOf(form, E, H._Kinematics(A=238., Q=0.), power), expected, rtol=2e-9, atol=1e-12)


@pytest.mark.parametrize("theta", [-1., float("nan"), float("inf")])
def test_analytic_energy_moments_refuse_invalid_active_theta(theta):
    law = m.Evaporation(U=m.PhysicalQuantity(0., "eV"),
                       theta=m.XYs1d(xs=[1., 2.], ys=[theta, theta]))
    with pytest.raises(ValueError, match="positive theta|non-finite"):
        H._analyticSpectrumMoment(law, np.array([1., 2.]), 1)


def test_analytic_energy_cm_isotropic_shift_and_unsupported_second_moment():
    law = m.Evaporation(U=m.PhysicalQuantity(0., "eV"),
                       theta=m.XYs1d(xs=[1., 1e8], ys=[1e6, 1e6]))
    form = m.Uncorrelated(energy=law, angular=m.Isotropic2d(), productFrame=m.Frame.centerOfMass)
    E = np.array([1., 1e5, 1e8]); k = H._Kinematics(A=55., Q=0.)
    expected = H._analyticSpectrumMoment(law, E, 1) + E/(k.A+1)**2
    np.testing.assert_allclose(H._meanEnergyOf(form, E, k), expected, rtol=1e-14)
    with pytest.raises(NotImplementedError, match="second moment"):
        H._meanEnergyOf(form, E, k, 2)


def test_mf6_capture_443_uses_independent_photon_kicks_without_compound_recoil():
    from types import SimpleNamespace as NS
    A = 55.; E = np.array([1., 1e5, 1e7]); binding = 2e6
    form = m.Uncorrelated(energy=m.DiscretePhotonEnergy(primaryFlag=0, energy=binding, atomicWeightRatio=A), productFrame=m.Frame.lab)
    product = NS(pid="photon", multiplicity=m.Multiplicity(form=m.Constant1d(constant=2.,domainMin_=0.,domainMax_=2e7)),
                 distribution=NS(forms={"MonteCarlo":form}))
    source = NS(provenance=NS(headerFields={"mf6":{"lct":1}}), outputChannel=NS(products=[]))
    reaction = NS(ENDF_MT=102,outputChannel=NS(Q=NS(value=5e6),products=[product]))
    result = H.heatingOf(NS(reactions=[reaction]),E,{102:np.ones(len(E))},awr=A,Z=26,
                        evaluation=NS(findReactionByENDF_MT=lambda mt:source),kinematic=True)
    expected = np.full(len(E),2*binding**2/(2*(A+1)*H.NEUTRON_REST_ENERGY))
    np.testing.assert_allclose(result.kinematicUpper[102],expected,rtol=1e-14)


@pytest.mark.parametrize("A,ed", [(1.,25.), (55.,40.), (238.,25.)])
def test_two_body_damage_splits_displacement_threshold_and_controls_interpolation(A,ed,monkeypatch):
    # An identity partition above ed gives an exact truncated uniform-recoil
    # integral. This checks threshold clipping without copying quadrature.
    monkeypatch.setattr(H,"lindhard",lambda recoil,*args:np.where(recoil>=ed,recoil,0.)); monkeypatch.setattr(H, "_heating_native", None)  # the NumPy reference uses the stub
    form=m.AngularTwoBody(angular=m.Isotropic2d(),productFrame=m.Frame.centerOfMass)
    activation=25*(A+1)**2/(4*A)
    E=np.unique(np.r_[activation*np.array([.999999,1.,1.000001,1.4]),np.geomspace(activation,1e8,17)])
    maximum=4*A/(A+1)**2*E
    expected=np.where((E>=activation)&(maximum>=ed),(maximum**2-ed**2)/(2*maximum),0.)
    actual=H._twoBodyDamage(form,E,A,0.,26,ed,elastic=True)
    np.testing.assert_allclose(actual,expected,atol=.02,rtol=2e-5)


@pytest.mark.parametrize("extent", ["below","above","mixed"])
def test_unit_base_held_spectrum_keeps_exact_incident_boost(extent):
    energy=m.XYs2d(function1ds=[_table([0.,1.],[1.,1.],10.),_table([0.,4.],[.5,0.],20.)],
                    interpolationQualifier=m.InterpolationQualifier.unitBase)
    angular=[]
    for E,width,mu in ((10.,1.,.2),(20.,4.,.1)):
        angular.append(m.XYs2d(function1ds=[_table([-1.,1.],[.5-1.5*mu,.5+1.5*mu],x) for x in (0.,width)],outerDomainValue=E))
    form=m.EnergyAngularMC(energy=energy,energyAngular=m.XYs3d(function2ds=angular),productFrame=m.Frame.centerOfMass)
    E={"below":np.array([1e-5,1.,9.]),"above":np.array([21.,100.,1e6]),"mixed":np.array([1e-5,1.,9.,21.,100.,1e6])}[extent]
    low=E<10.;mean=np.where(low,.5,4./3.);cross=np.where(low,.2*2./3.,.1*2.*8./15.)
    expected=mean+E/9.+2*np.sqrt(E/9.)*cross
    np.testing.assert_allclose(H._meanEnergyOf(form,E,H._Kinematics(A=2.,Q=0.)),expected,rtol=1e-14,atol=1e-12)


def test_fission_chances_use_common_aggregate_spectrum_once_without_double_counting(monkeypatch):
    from types import SimpleNamespace as NS
    form=object();neutron=NS(pid="n",distribution=NS(forms={"MonteCarlo":form}))
    def reaction(mt,products):return NS(ENDF_MT=mt,outputChannel=NS(Q=NS(value=0.),products=products))
    suite=NS(reactions=[reaction(19,[]),reaction(20,[])],sums=[reaction(18,[neutron])])
    calls=[]
    def emission(g,*args):
        assert g is form;calls.append(g);return np.array([123.,234.])
    monkeypatch.setattr(H,"_emissionDamage",emission)
    E=np.array([1.,2.]);xs={18:np.array([4.,2.]),19:np.array([1.,2.]),20:np.array([3.,0.])}
    result=H.heatingOf(suite,E,xs,awr=238.,Z=92)
    np.testing.assert_array_equal(result.damage,[4*123.,2*234.])
    assert len(calls)==1
    assert 18 not in result.perReaction


def test_inelastic_damage_keeps_both_sides_of_reaction_threshold(monkeypatch):
    A=55.;Q=-1e6;threshold=(A+1)/A*(-Q)
    monkeypatch.setattr(H,"lindhard",lambda recoil,*args:recoil); monkeypatch.setattr(H, "_heating_native", None)  # the NumPy reference uses the stub
    form=m.AngularTwoBody(angular=m.Isotropic2d(),productFrame=m.Frame.centerOfMass)
    E=np.array([threshold/10,np.nextafter(threshold,-np.inf),threshold,threshold+1.,2*threshold])
    r2=np.maximum(1-threshold/E,0.);expected=np.where(E>=threshold,E*A/(A+1)**2*(1+r2),0.)
    np.testing.assert_allclose(H._twoBodyDamage(form,E,A,Q,26,0.,elastic=False),expected,atol=.02,rtol=2e-5)



def test_kalbach_mean_maps_continuum_and_keeps_sampled_parameters_on_source_support():
    energy=m.XYs2d(function1ds=[_table([0.,1.],[1.,1.],10.),_table([0.,4.],[.5,0.],20.)],interpolationQualifier=m.InterpolationQualifier.unitBase)
    rs=m.XYs2d(function1ds=[m.XYs1d(xs=[0.,1.],ys=[.2,.2],outerDomainValue=10.),m.XYs1d(xs=[0.,4.],ys=[.8,.8],outerDomainValue=20.)])
    aa=m.XYs2d(function1ds=[m.XYs1d(xs=[0.,1.],ys=[1.,1.],outerDomainValue=10.),m.XYs1d(xs=[0.,4.],ys=[2.,2.],outerDomainValue=20.)])
    form=m.KalbachMann(f=energy,r=rs,a=aa,productFrame=m.Frame.centerOfMass)
    E=np.array([15.]);cm=2.5*(.5+1/3)/2
    mu1=.2*(1/np.tanh(1.)-1.);mu2=.8*(1/np.tanh(2.)-.5)
    cross=np.sqrt(2.5)*(mu1*2/3+mu2*8/15)/2
    expected=cm+15/9+2*np.sqrt(15/9)*cross
    np.testing.assert_allclose(H._meanEnergyOf(form,E,H._Kinematics(A=2.,Q=0.)),[expected],atol=1e-10,rtol=1e-12)
    np.testing.assert_allclose(H._meanEnergyOf(form,E,H._Kinematics(A=2.,Q=0.,awp=0.)),[cm],atol=1e-12,rtol=1e-14)



def test_generated_mf6_recoil_uses_canonical_cm_moments_and_requires_missing_residual():
    from types import SimpleNamespace as NS
    energy=m.XYs2d(function1ds=[_table([0.,4.],[.25,.25],E) for E in (10.,20.)],interpolationQualifier=m.InterpolationQualifier.unitBase)
    angles=m.XYs3d(function2ds=[m.XYs2d(function1ds=[_table([-1.,1.],[.2,.8],x) for x in (0.,4.)],outerDomainValue=E) for E in (10.,20.)])
    form=m.EnergyAngularMC(energy=energy,energyAngular=angles,productFrame=m.Frame.centerOfMass)
    records=[{"zap":1,"law":1},{"zap":0,"law":1}]
    source=NS(provenance=NS(headerFields={"mf6":{"products":records}}))
    evaluation=NS(findReactionByENDF_MT=lambda mt:source)
    E=np.array([5.,10.,15.,20.,25.]);A=55.;k=H._Kinematics(A=A,Q=0.)
    cross=.2*2/3*2
    expected=E*A/(A+1)**2+2/A-2*np.sqrt(E)/(A+1)*cross
    np.testing.assert_allclose(H._generatedMF6Recoil(evaluation,16,form,E,k),expected,rtol=1e-12,atol=1e-14)
    records.append({"zap":2004,"law":0})
    assert H._generatedMF6Recoil(evaluation,16,form,E,k) is None



def test_zero_lower_unit_base_support_has_closed_incident_dependence():
    energy=m.XYs2d(function1ds=[_table([0.,1.],[1.,1.],10.),_table([0.,4.],[.5,0.],20.)],interpolationQualifier=m.InterpolationQualifier.unitBase)
    angular=[]
    for E,width,mu in ((10.,1.,.2),(20.,4.,.1)):
        angular.append(m.XYs2d(function1ds=[_table([-1.,1.],[.5-1.5*mu,.5+1.5*mu],x) for x in (0.,width)],outerDomainValue=E))
    form=m.EnergyAngularMC(energy=energy,energyAngular=m.XYs3d(function2ds=angular),productFrame=m.Frame.centerOfMass)
    E=np.linspace(10.,20.,101);fraction=(E-10)/10;width=1+3*fraction
    expected=width*((1-fraction)*.5+fraction/3)+E/9+2*np.sqrt(E/9*width)*((1-fraction)*.2*2/3+fraction*.1*8/15)
    np.testing.assert_allclose(H._meanEnergyOf(form,E,H._Kinematics(A=2.,Q=0.)),expected,rtol=2e-15,atol=1e-14)


def test_fission_photons_are_taken_out_once_against_the_release():
    # MT18 as the sum of its chances carries fission's photons; MT3 carries others.
    # heatingOf takes MT3's out of MT301 and leaves MT18's to fissionPhotonKerma,
    # which the release's EGP balances; delayed photons are not EGP's.
    from types import SimpleNamespace as NS
    E = np.array([1., 1e5, 1e7]); one = np.ones(len(E))

    def line(energy, count, mode="prompt"):
        form = m.Uncorrelated(energy=m.DiscretePhotonEnergy(primaryFlag=0, energy=energy, atomicWeightRatio=239.),
                              productFrame=m.Frame.lab)
        return NS(pid="photon", provenance=NS(attributes={"emission_mode": mode}),
                  multiplicity=m.Multiplicity(form=m.Constant1d(constant=count, domainMin_=0., domainMax_=2e7)),
                  distribution=NS(forms={"MonteCarlo": form}))

    chance = NS(ENDF_MT=19, outputChannel=NS(Q=NS(value=2e8), products=[]))
    fission = NS(ENDF_MT=18, outputChannel=NS(Q=NS(value=2e8), products=[line(1e6, 7.), line(4e5, 3., "delayed")]))
    nonelastic = NS(ENDF_MT=3, outputChannel=NS(Q=NS(value=0.), products=[line(5e5, 2.)]))
    suite = NS(reactions=[], fissionComponents=[chance], sums=[fission, nonelastic])
    sigma = {19: 2 * one, 18: 2 * one, 3: 5 * one}
    result = H.heatingOf(suite, E, sigma, awr=239., Z=94)
    np.testing.assert_allclose(result.kerma, -5 * 2 * 5e5 * one, rtol=1e-14)
    assert 18 not in result.photonEnergy
    np.testing.assert_allclose(H.fissionPhotonEnergy(suite, E, awr=239.)[18], 7e6 * one, rtol=1e-14)
    np.testing.assert_allclose(H.fissionPhotonKerma(suite, E, sigma, awr=239.), 2 * 7e6 * one, rtol=1e-14)
    fission.outputChannel.products = []
    assert H.fissionPhotonEnergy(suite, E, awr=239.) == {}
    np.testing.assert_array_equal(H.fissionPhotonKerma(suite, E, sigma, awr=239.), 0 * one)


def test_heavy_tabulation_is_done_once_per_form_and_forgotten_with_it(monkeypatch):
    import gc
    from kika.processing import tabulation as T
    from kika.processing.tabulation import TabulationContext
    calls = []

    def tabulate(form, context):
        calls.append(context.mt)
        if context.mt == 2:
            raise ValueError("no such law")
        return ("table", context.mt)

    monkeypatch.setattr(T, "tabulateDistribution", tabulate)
    form = m.Uncorrelated()
    first = TabulationContext(awr=55., zaTarget=26056, mt=103, Q=0., pid="H1", zaEmitted=1001)
    assert H._tabulateHeavy(form, first) == H._tabulateHeavy(form, first) == ("table", 103)
    assert calls == [103]
    failing = TabulationContext(awr=55., zaTarget=26056, mt=2, Q=0., pid="H1", zaEmitted=1001)
    for _ in range(2):
        with pytest.raises(ValueError, match="no such law"):
            H._tabulateHeavy(form, failing)
    assert calls == [103, 2]
    key = id(form)
    assert key in H._HEAVY_TABULATIONS
    del form
    gc.collect()
    assert key not in H._HEAVY_TABULATIONS
