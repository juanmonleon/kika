"""The ACER-like tabulation (roadmap H3): normalised, monotone, accurate tables."""
from __future__ import annotations

import numpy as np
import pytest

from kika.nuclear_data import model as m
from kika.processing.tabulation import (TabulationContext, TabulationOptions, kalbachSlope,
                                        legendreDensity, muTable, pdfTable, refine,
                                        tabulateAngular, tabulateDistribution, tabulateEnergy)


def _assertTable(table):
    assert isinstance(table, m.Xs_pdf_cdf1d)
    assert table.cdf[0] == pytest.approx(0.0, abs=1e-15) or table.numberOfDiscrete
    assert table.cdf[-1] == 1.0
    assert np.all(np.diff(table.cdf) >= -1e-15)
    assert np.all(table.pdf >= 0)


def test_refine_reaches_the_tolerance():
    options = TabulationOptions(rtol=1e-4, atol=0.0)
    x, y = refine(np.exp, [0.0, 5.0], options)
    q = np.linspace(0, 5, 10001)
    assert np.max(np.abs(np.interp(q, x, y) - np.exp(q)) / np.exp(q)) < 2e-4


def test_legendre_isotropic_is_three_points():
    table, clipped = muTable(m.Legendre(coefficients=np.array([1.0])), TabulationOptions())
    assert len(table) == 3 and not clipped
    _assertTable(table)
    assert np.allclose(table.pdf, 0.5)


def test_forward_peaked_legendre_is_refined_and_reproduced():
    coefficients = np.array([1.0, 0.5, 0.2, 0.05])  # positive on [-1, 1]
    table, _ = muTable(m.Legendre(coefficients=coefficients), TabulationOptions())
    _assertTable(table)
    density = legendreDensity(coefficients)
    q = np.linspace(-1, 1, 4001)
    exact = density(q)
    assert np.max(np.abs(np.interp(q, table.xs, table.pdf) - exact) / exact.max()) < 2e-3


def test_negative_legendre_is_clipped_and_renormalised():
    table, clipped = muTable(m.Legendre(coefficients=np.array([1.0, 0.9, 0.0, -0.9])),
                             TabulationOptions())
    assert clipped
    _assertTable(table)
    with pytest.raises(ValueError):
        muTable(m.Legendre(coefficients=np.array([1.0, 0.9, 0.0, -0.9])),
                TabulationOptions(negative="raise"))


def test_histogram_cdf_and_discrete_lines():
    table, _ = pdfTable([0.5, 1.0, 2.0, 3.0], [0.2, 0.3, 0.5, 0.5], histogram=True, discrete=2)
    _assertTable(table)
    assert table.numberOfDiscrete == 2
    total = 0.2 + 0.3 + 0.5 * 1.0
    assert table.pdf[0] == pytest.approx(0.2 / total)
    assert table.cdf[1] == pytest.approx(0.5 / total)


def test_angular_regions_retain_both_sides_of_a_boundary():
    low = m.XYs2d(function1ds=[m.Legendre(coefficients=np.array([1.0]), outerDomainValue=1.0),
                               m.Legendre(coefficients=np.array([1.0, 0.1]), outerDomainValue=2.0)])
    high = m.XYs2d(function1ds=[m.XYs1d(xs=[-1, 1], ys=[0.4, 0.6], outerDomainValue=2.0),
                                m.XYs1d(xs=[-1, 1], ys=[0.3, 0.7], outerDomainValue=3.0)])
    form, _ = tabulateAngular(m.Regions2d(function2ds=[low, high]))
    assert [t.outerDomainValue for t in form.function1ds] == [1.0, np.nextafter(2., -np.inf), 2.0, 3.0]
    assert form.function1ds[2].pdf[-1] == pytest.approx(0.6)
    from kika.processing.heating import _meanMu
    from kika.hdf5.distributions import encodeAngle, decodeAngle
    E = np.array([1.5, np.nextafter(2., -np.inf), 2., 2.5])
    expected = [0.05, 0.1, 1./15., 0.1]
    np.testing.assert_allclose(_meanMu(form, E), expected, rtol=0., atol=1e-6)
    decoded = decodeAngle(encodeAngle(form))
    np.testing.assert_array_equal([t.outerDomainValue for t in decoded.function1ds], [1., np.nextafter(2., -np.inf), 2., 3.])
    np.testing.assert_allclose(_meanMu(decoded, E), expected, rtol=0., atol=1e-6)


def test_evaporation_table_from_a_law():
    law = m.GeneralEvaporation(U=m.PhysicalQuantity(1e5, "eV"),
                               theta=m.XYs1d(xs=[1e5, 2e7], ys=[1e6, 1e6]),
                               g=m.XYs1d(xs=[0, 1, 5, 20], ys=[0, 1, 0.1, 0]))
    form, _ = tabulateEnergy(law)
    for table in form.function1ds:
        _assertTable(table)


def test_level_reaction_gets_openmcs_kinematics():
    form = m.AngularTwoBody(angular=m.Isotropic2d(), productFrame=m.Frame.centerOfMass)
    out = tabulateDistribution(form, TabulationContext(awr=55.45, zaTarget=26056, mt=51, Q=-846778.0))
    assert isinstance(out.energy, m.LevelEnergy)
    A = 55.45
    assert out.energy.threshold == pytest.approx((A + 1) / A * 846778.0)
    assert out.energy.massRatio == pytest.approx((A / (A + 1)) ** 2)


def test_kalbach_slope_is_of_order_one_and_grows_with_the_emitted_energy():
    """Kalbach's systematics for n + Fe-56 at 14 MeV: finite, of order one, and
    steeper (more forward) the more energy the neutron takes out. Not a
    comparison with OpenMC's ``kalbach_slope``, which is not a dependency."""
    value = float(kalbachSlope(14e6, 5e6, 1, 26056))
    assert value == pytest.approx(0.0, abs=10)  # finite and of order unity
    assert np.all(np.diff(kalbachSlope(14e6, np.array([1e5, 1e6, 5e6]), 1, 26056)) > 0)


def test_kalbach_mann_gets_a_slope_on_f_grid():
    f = m.XYs2d(function1ds=[m.XYs1d(xs=[0, 1e6, 2e6], ys=[0, 1e-6, 0], outerDomainValue=1e7)])
    r = m.XYs2d(function1ds=[m.XYs1d(xs=[0, 1e6, 2e6], ys=[0.1, 0.2, 0.3], outerDomainValue=1e7)])
    out = tabulateDistribution(m.KalbachMann(f=f, r=r), TabulationContext(
        awr=55.45, zaTarget=26056, mt=91, zaEmitted=1))
    assert len(out.a.function1ds[0]) == len(out.f.function1ds[0]) == 3
    _assertTable(out.f.function1ds[0])


def test_energy_angular_legendre_becomes_mc():
    surface = m.XYs2d(function1ds=[
        m.Legendre(coefficients=np.array([0.0, 0.0]), outerDomainValue=0.0),
        m.Legendre(coefficients=np.array([2e-6, 1e-6]), outerDomainValue=1e6),
        m.Legendre(coefficients=np.array([0.0, 0.0]), outerDomainValue=2e6)], outerDomainValue=1e7)
    out = tabulateDistribution(m.EnergyAngular(xys3d=m.XYs3d(function2ds=[surface])),
                               TabulationContext(awr=55.45, zaTarget=26056, mt=91))
    assert isinstance(out, m.EnergyAngularMC)
    _assertTable(out.energy.function1ds[0])
    for t in out.energyAngular.function2ds[0].function1ds:
        _assertTable(t)


@pytest.mark.parametrize('field', ['correlatedRtol', 'correlatedAtol'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.])
def test_correlated_accuracy_cannot_be_silently_disabled_by_nonfinite_or_negative_values(field, value):
    from kika.processing.tabulation.tables import TabulationOptions
    with pytest.raises(ValueError, match='finite and nonnegative'):
        TabulationOptions(**{field: value})



def test_angular_moment_control_matches_clipped_polynomial_integrals():
    from kika.processing.tabulation.angular import muTable, _tableMoments
    # p(mu)=1.5*mu, clipped positive on [0,1], with moments 2/3 and 1/2.
    form = m.Legendre(coefficients=[0., 1.])
    table, clipped = muTable(form, TabulationOptions(angularMomentAtol=1e-6))
    assert clipped
    assert np.allclose(_tableMoments(table), [2/3, 1/2], rtol=0, atol=1e-6)


def test_angular_moment_control_detects_hidden_negative_interval():
    from kika.processing.tabulation.angular import muTable
    # p(mu)=.5 + 1.2 P2(mu), positive at +/-1, negative near zero.
    with pytest.raises(ValueError, match="negative interval"):
        muTable(m.Legendre(coefficients=[1., 0., .48]), TabulationOptions(negative="raise"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_angular_moment_tolerance_is_validated(value):
    with pytest.raises(ValueError, match="angular moment tolerance"):
        TabulationOptions(angularMomentAtol=value)



def test_angular_moments_catch_midpoint_cancellation_within_capacity():
    from kika.processing.tabulation.angular import muTable, _tableMoments
    c=[1.,.0455164863,.00692508399,.000996352270,-.00285652874,.00111578676,.00174842263]
    table,_=muTable(m.Legendre(coefficients=c), TabulationOptions(rtol=1e-4,atol=1e-6))
    assert len(table.xs)<500
    assert np.allclose(_tableMoments(table), [c[1],1/3+2*c[2]/3], rtol=0,atol=1e-5)


def test_legendre_table_keeps_the_mean_cosine_where_a_midpoint_sits_on_the_chord():
    """Fe-56 elastic at 1.438 MeV (ENDF/B-VIII.1): with three seeds and a midpoint-only
    test, [-1, 0] passed as one segment (mu = -0.5 lies on the chord) and the table's
    mean cosine came out 0.025 instead of a1 = -0.073."""
    from kika.nuclear_data.model import Legendre
    from kika.processing.tabulation.angular import muTable
    from kika.processing.tabulation.tables import TabulationOptions

    a = [1.0, -0.07297533, 0.2537618, 0.06656012, 0.005421167, 0.01914622, -0.01104767]
    table, _ = muTable(Legendre(coefficients=a, outerDomainValue=1.438e6), TabulationOptions())
    mu, cdf = np.asarray(table.xs), np.asarray(table.cdf)
    mean = float(np.sum(np.diff(cdf) * 0.5 * (mu[1:] + mu[:-1])))
    assert mean == pytest.approx(a[1], abs=2e-3)


def test_angular_boundary_coordinates_match_openmc_lower_bound_without_source_edits():
    low = m.XYs2d(function1ds=[m.Legendre(coefficients=[1.], outerDomainValue=1.),
                              m.Legendre(coefficients=[1., .1], outerDomainValue=2.)])
    high = m.XYs2d(function1ds=[m.Legendre(coefficients=[1., .2], outerDomainValue=2.),
                               m.Legendre(coefficients=[1., .3], outerDomainValue=3.)])
    source = m.Regions2d(function2ds=[low, high])
    converted, _ = tabulateAngular(source)
    energies = np.array([t.outerDomainValue for t in converted.function1ds])
    assert np.all(np.diff(energies) > 0)
    assert low.function1ds[-1].outerDomainValue == 2.
    assert high.function1ds[0].outerDomainValue == 2.
    from kika.processing.heating import _meanMu
    means = _meanMu(converted, energies)
    probes = np.array([1.5, np.nextafter(2., -np.inf), 2., np.nextafter(2., np.inf), 2.5])
    # OpenMC 0.16 get_energy_index / lower_bound_index, including exact-node
    # selection; this arithmetic differs from np.interp on repeated energies.
    ids = np.maximum(np.searchsorted(energies, probes, side='left') - 1, 0)
    fraction = (probes - energies[ids]) / (energies[ids + 1] - energies[ids])
    sampled_mean = (1-fraction)*means[ids] + fraction*means[ids+1]
    np.testing.assert_allclose(sampled_mean, _meanMu(converted, probes), atol=1e-14, rtol=0.)
    np.testing.assert_allclose(sampled_mean, [.05, .1, .2, .2, .25], atol=1e-5, rtol=0.)


def test_mixed_histogram_retains_lines_mass_and_two_continuum_moments():
    source = m.XYs2d(function1ds=[m.XYs1d(xs=[10., 0., 2.], ys=[.2, 0., .8], outerDomainValue=15.)])
    converted, _ = tabulateEnergy(source, discrete=[1])
    table = converted.function1ds[0]
    assert str(table.interpolation) == "flat" and table.numberOfDiscrete == 1
    assert table.pdf[0] == pytest.approx(.2, abs=1e-14)
    from kika.processing.heating import _expect
    assert _expect(table, lambda x: x) == pytest.approx(2.+.8*4/3, abs=1e-12)
    assert _expect(table, lambda x: x*x) == pytest.approx(20.+.8*2., abs=1e-12)
    q = np.linspace(0., 2., 10001)
    assert np.max(abs(np.interp(q, table.xs[1:], table.cdf[1:]) - (.2+.2*q*q))) <= TabulationOptions().mixedCdfAtol
    assert len(table.xs) <= TabulationOptions().maxPoints


def test_mixed_histogram_is_a_projection_maxpoints_does_not_drop():
    # The projection's size is set by mixedCdfAtol; capping it at maxPoints
    # dropped whole products (Pu-242's MF6 capture photons, ENDF/B-VIII.1).
    source = m.XYs2d(function1ds=[m.XYs1d(xs=[10., 0., 2.], ys=[.2, 0., .8], outerDomainValue=15.)])
    options = TabulationOptions(maxPoints=100)
    converted, _ = tabulateEnergy(source, options, discrete=[1])
    table = converted.function1ds[0]
    assert len(table.xs) > options.maxPoints
    q = np.linspace(0., 2., 10001)
    assert np.max(abs(np.interp(q, table.xs[1:], table.cdf[1:]) - (.2+.2*q*q))) <= options.mixedCdfAtol
    reference, _ = tabulateEnergy(source, discrete=[1])
    np.testing.assert_array_equal(table.xs, reference.function1ds[0].xs)


def test_mixed_single_histogram_panel_gets_a_transport_safe_second_panel():
    source = m.XYs2d(function1ds=[m.XYs1d(xs=[10., 1., 3.], ys=[.2, .4, .4],
        interpolation=m.Interpolation.flat, outerDomainValue=15.)])
    converted, _ = tabulateEnergy(source, discrete=[1])
    table = converted.function1ds[0]
    np.testing.assert_array_equal(table.xs, [10., 1., 2., 3.])
    np.testing.assert_allclose(table.cdf, [.2, .2, .6, 1.], rtol=0., atol=1e-14)


def test_pure_lines_do_not_enter_the_continuum_interpolation_branch():
    source = m.XYs2d(function1ds=[m.XYs1d(xs=[10., 5.], ys=[.25, .75], outerDomainValue=15.)])
    converted, _ = tabulateEnergy(source, discrete=[2])
    table = converted.function1ds[0]
    assert str(table.interpolation) == "flat" and table.numberOfDiscrete == 2
    np.testing.assert_array_equal(table.xs, [10., 5.])
    np.testing.assert_allclose(table.cdf, [.25, 1.], rtol=0., atol=1e-14)


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_mixed_cdf_tolerance_is_validated(value):
    with pytest.raises(ValueError, match="mixed CDF tolerance"):
        TabulationOptions(mixedCdfAtol=value)



def test_angular_moments_positive_proof_and_clipping_keep_exact_integrals():
    from kika.processing.tabulation.angular import _legendreMoments
    np.testing.assert_allclose(_legendreMoments([1.,.02,.04,.01],"clip"),[.02,1/3+2*.04/3],rtol=1e-14,atol=1e-15)
    np.testing.assert_allclose(_legendreMoments([1.,1.],"clip"),[5/9,11/27],rtol=1e-14,atol=1e-15)
    with pytest.raises(ValueError,match="negative"):
        _legendreMoments([1.,1.],"raise")


def test_repeated_joint_angles_are_cached_with_independent_mutable_arrays(monkeypatch):
    from kika.processing.tabulation import correlated as C
    from kika.processing.tabulation.tables import TabulationOptions
    from kika.nuclear_data import model as m
    real=C.muTable;calls=[]
    def spy(*args,**kwargs):calls.append(1);return real(*args,**kwargs)
    monkeypatch.setattr(C,"muTable",spy)
    surfaces=[]
    for E in (10.,20.):
        surfaces.append(m.XYs2d(function1ds=[m.Legendre(coefficients=np.array([1.,.1]),outerDomainValue=x) for x in (0.,1.)],outerDomainValue=E))
    form=m.EnergyAngular(xys3d=m.XYs3d(function2ds=surfaces),productFrame=m.Frame.centerOfMass)
    result=C.tabulateEnergyAngular(form,TabulationOptions())
    tables=[t for surface in result.energyAngular.function2ds for t in surface.function1ds]
    assert len(calls)==1
    for t in tables[1:]:
        np.testing.assert_array_equal(t.pdf,tables[0].pdf)
        assert not np.shares_memory(t.pdf,tables[0].pdf)
    old=tables[1].pdf.copy();tables[0].pdf[0]+=1
    np.testing.assert_array_equal(tables[1].pdf,old)



def test_legendre_density_native_and_reference_keep_shape_values_and_fallback(monkeypatch):
    from kika.processing.tabulation import angular as A
    coefficients=np.array([1.,.3,-.1,.01])
    for points in (np.array(.7),np.array([-1.,0.,1.]),np.arange(6.).reshape(2,3)/5):
        expected=np.polynomial.legendre.legval(points,coefficients*(2*np.arange(4)+1)/2)
        actual=A.legendreDensity(coefficients)(points)
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),np.asarray(expected).view(np.uint64))
        assert np.shape(actual)==np.shape(points)
    outside=np.array([-2.,1.1])
    before=A.legendreDensity(coefficients)(outside)
    monkeypatch.setattr(A,"_polynomial_native",None)
    np.testing.assert_array_equal(A.legendreDensity(coefficients)(outside),before)
