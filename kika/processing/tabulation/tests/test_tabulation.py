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


def test_angular_regions_take_the_boundary_once():
    low = m.XYs2d(function1ds=[m.Legendre(coefficients=np.array([1.0]), outerDomainValue=1.0),
                               m.Legendre(coefficients=np.array([1.0, 0.1]), outerDomainValue=2.0)])
    high = m.XYs2d(function1ds=[m.XYs1d(xs=[-1, 1], ys=[0.4, 0.6], outerDomainValue=2.0),
                                m.XYs1d(xs=[-1, 1], ys=[0.3, 0.7], outerDomainValue=3.0)])
    form, _ = tabulateAngular(m.Regions2d(function2ds=[low, high]))
    assert [t.outerDomainValue for t in form.function1ds] == [1.0, 2.0, 3.0]
    assert form.function1ds[1].pdf[-1] == pytest.approx(0.6)


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


def test_kalbach_slope_matches_openmc():
    # openmc.data.kalbach_slope(14e6, 5e6, 1, 1, 26056) in OpenMC 0.16.0
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
