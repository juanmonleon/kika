"""The forward operator of an elastic measurement (roadmap G4NDL §0bis, Fase 12).

What is pinned: the bin-through-resolution kernel reduces to the two readings
that already exist (a box, a Gaussian); with no resolution and no bin the
forward reading is the pointwise dσ/dΩ; the PDF read from the model is the flat
class's; a suite read back from G4NDL gives the same reading as the ENDF suite it
was written from; and the product fold differs from the factor fold only where σ
varies inside the kernel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf.dcs import TofResolution
from kika.nuclear_data.forward import (
    ElasticView, ForwardSetup, forward_dcs, forward_sigma, kernel_sigma_ev,
)
from kika.algebra import (
    box_gaussian_fold_nodes, gaussian_fold_nodes, group_averages,
)

DATA = Path(kika.__file__).parent / "endf" / "tests" / "data"
FE56 = DATA / "micro_fe56_xs_and_angular.endf"


# ------------------------------------------------------------------ the kernel

def _avg(nodes, weights, x, y):
    return float(weights @ np.interp(nodes, x, y))


def test_box_gaussian_weights_are_normalised_and_keep_a_line():
    x = np.linspace(0.0, 10.0, 51)
    y = 2.0 + 0.3 * x
    for s in (0.0, 0.05, 0.4, 2.0):
        n, w = box_gaussian_fold_nodes(4.0, 6.0, s, (x,))
        assert w.sum() == pytest.approx(1.0)
        assert _avg(n, w, x, y) == pytest.approx(2.0 + 0.3 * 5.0, rel=1e-12)


def test_without_resolution_it_is_the_bin_average():
    x = np.linspace(0.0, 10.0, 2001)
    y = 1.0 + np.sin(7 * x) ** 2 * np.exp(-x / 4)
    n, w = box_gaussian_fold_nodes(2.3, 3.1, 0.0, (x,))
    ref = group_averages(x, y, 2, np.array([2.3, 3.1]))[0]
    assert _avg(n, w, x, y) == pytest.approx(ref, rel=1e-6)


def test_a_zero_width_bin_is_the_gaussian_fold():
    x = np.linspace(0.0, 10.0, 2001)
    y = 1.0 + np.sin(7 * x) ** 2
    n1, w1 = box_gaussian_fold_nodes(5.0, 5.0, 0.3, (x,))
    n2, w2 = gaussian_fold_nodes(5.0, 0.3, (x,))
    assert np.array_equal(n1, n2) and np.array_equal(w1, w2)


def test_the_two_limits_are_continuous():
    x = np.linspace(0.0, 10.0, 4001)
    y = 1.0 + np.sin(7 * x) ** 2
    box = _avg(*box_gaussian_fold_nodes(4.0, 6.0, 0.0, (x,)), x, y)
    near_box = _avg(*box_gaussian_fold_nodes(4.0, 6.0, 1e-4, (x,)), x, y)
    assert near_box == pytest.approx(box, rel=1e-4)
    gauss = _avg(*gaussian_fold_nodes(5.0, 0.5, (x,)), x, y)
    near_gauss = _avg(*box_gaussian_fold_nodes(5.0 - 1e-4, 5.0 + 1e-4, 0.5, (x,)), x, y)
    assert near_gauss == pytest.approx(gauss, rel=1e-5)


def _trapezoid_box_gaussian(lo, hi, sigma, x, y, n_uniform=101):
    """The quadrature the exact weights replaced: K times the trapezoid rule on
    the table's points, the bin edges and ``n_uniform`` uniform points within
    5 sigma of the bin."""
    from scipy.special import ndtr

    a, b = lo - 5.0 * sigma, hi + 5.0 * sigma
    t = np.unique(np.concatenate([np.linspace(a, b, n_uniform), [lo, hi],
                                  x[(x > a) & (x < b)]]))
    trap = np.zeros_like(t)
    trap[:-1] += 0.5 * np.diff(t)
    trap[1:] += 0.5 * np.diff(t)
    w = (ndtr((hi - t) / sigma) - ndtr((lo - t) / sigma)) * trap
    return float(w @ np.interp(t, x, y) / w.sum())


def test_the_exact_weights_agree_with_the_trapezoid_they_replaced():
    x = np.linspace(0.0, 10.0, 2001)
    y = 1.0 + np.sin(7 * x) ** 2 * np.exp(-x / 4)
    for lo, hi, s in ((4.0, 6.0, 0.3), (2.3, 3.1, 0.05), (4.0, 6.0, 2.0), (4.9, 5.1, 0.4)):
        exact = _avg(*box_gaussian_fold_nodes(lo, hi, s, (x,)), x, y)
        assert exact == pytest.approx(_trapezoid_box_gaussian(lo, hi, s, x, y), rel=1e-4)


def test_the_box_gaussian_weights_are_exact_for_the_interpolant():
    """Against adaptive quadrature of the interpolant times K, panel by panel.

    The window edge 2.3 - 6 * 0.05 lands an ulp from the grid point 2.0, so this
    also pins the panel of width 2e-16 the closed-form moment cannot resolve."""
    from scipy.integrate import quad
    from scipy.special import ndtr

    x = np.linspace(0.0, 10.0, 2001)
    y = 1.0 + np.sin(7 * x) ** 2 * np.exp(-x / 4)
    lo, hi, s = 2.3, 3.1, 0.05
    n, w = box_gaussian_fold_nodes(lo, hi, s, (x,))
    assert np.all(w > 0.0) and w.sum() == pytest.approx(1.0, abs=1e-14)
    k = lambda t: np.interp(t, x, y) * (ndtr((hi - t) / s) - ndtr((lo - t) / s))
    edges = np.concatenate([[lo - 12 * s], n, [hi + 12 * s]])
    ref = sum(quad(k, p, q, epsabs=1e-15, epsrel=1e-13)[0]
              for p, q in zip(edges[:-1], edges[1:])) / (hi - lo)
    assert _avg(n, w, x, y) == pytest.approx(ref, rel=1e-10)


def test_doppler_adds_in_quadrature():
    tof = TofResolution(flight_path_m=27.037, delta_t_ns=1.5, min_sigma_e_kev=0.0)
    e = 1.0e6
    s_e = float(tof.sigma_e_mev(1.0)) * 1e6
    s_d2 = 2 * e * 8.617333262e-5 * 300.0 / 206.0
    assert kernel_sigma_ev(ForwardSetup(tof=tof, temperature_k=300.0), e, 206.0) == \
        pytest.approx(np.sqrt(s_e**2 + s_d2))
    assert kernel_sigma_ev(ForwardSetup(), e, 206.0) == 0.0


# ------------------------------------------------------------- from the model

@pytest.fixture(scope="module")
def suite():
    from kika.endf.model_adapter.pendf import attachReconstruction

    s = kika.read(str(FE56), format="endf")
    attachReconstruction(s, FE56)   # the micro-tape stands in for its own PENDF
    return s


def test_the_background_is_refused(suite):
    with pytest.raises(ValueError, match="resonance region"):
        ElasticView.from_suite(suite, cross_section_label="eval")


def test_the_pdf_is_the_flat_class_one(suite):
    """Against the flat class read from the tape itself, Legendre and table halves both.

    Not through ``interop.flatAngularDistribution``: that one fails on this mixed
    (LTT=3) section, handing the table's ``XYs1d`` to the Legendre densifier.
    """
    from kika.nuclear_data.angular_distribution import AngularDistribution

    view = ElasticView.from_suite(suite)
    flat = AngularDistribution.from_endf(kika.read_endf(str(FE56), mf_numbers=[4]).files[4].mt[2])
    assert flat.representation == "mixed"
    energies = np.concatenate([np.geomspace(1e-5, 2e7, 801), view.energy_grids()[1]])
    for mu in (-0.96, -0.3, 0.0, 0.52, 0.96):
        np.testing.assert_allclose(view.pdf_native(mu)(energies),
                                   flat.evaluate_pdf_vs_energy(mu, energies),
                                   rtol=1e-12, atol=1e-14)


def test_no_resolution_and_no_bin_is_the_pointwise_dcs(suite):
    view = ElasticView.from_suite(suite)
    e = np.array([0.9e6, 1.3e6, 2.5e6, 7.0e6])
    mu = np.array([-0.8, 0.1, 0.96])
    got = forward_dcs(view, e, mu, ForwardSetup())["dcs"]
    for j, m in enumerate(mu):
        np.testing.assert_allclose(got[:, j], view.dcs_lab(m, e), rtol=1e-12)
    # A vanishing acceptance is the point detector.
    narrow = forward_dcs(view, e, mu, ForwardSetup(angular_half_width_deg=1e-6))["dcs"]
    np.testing.assert_allclose(narrow, got, rtol=1e-9)


def test_the_cm_to_lab_step_preserves_the_integral(suite):
    view = ElasticView.from_suite(suite)
    x, w = np.polynomial.legendre.leggauss(96)
    e = np.array([3.0e6])
    dcs = forward_dcs(view, e, x, ForwardSetup())["dcs"][0]
    assert 2 * np.pi * float(w @ dcs) == pytest.approx(float(view.sigma(e)[0]), rel=1e-6)


def test_a_g4ndl_round_trip_reads_the_same(suite, tmp_path):
    import kika.g4ndl as g4ndl

    kika.write(suite, tmp_path, format="g4ndl")
    back = g4ndl.open(tmp_path).read("Fe56")
    a, b = ElasticView.from_suite(suite), ElasticView.from_suite(back)
    tof = TofResolution(flight_path_m=27.037, delta_t_ns=1.5, min_sigma_e_kev=0.0)
    setup = ForwardSetup(tof=tof, temperature_k=293.6)
    e = np.geomspace(0.9e6, 5e6, 25)
    ra = forward_dcs(a, e, [0.96, -0.5], setup)
    rb = forward_dcs(b, e, [0.96, -0.5], setup)
    np.testing.assert_allclose(rb["dcs"], ra["dcs"], rtol=1e-6)


def test_with_flat_sigma_the_product_fold_is_sigma_times_the_folded_pdf(suite):
    """σ constant: ⟨σ f⟩ = σ ⟨f⟩, with ⟨f⟩ folded independently of the operator."""
    from kika.algebra import fold_tabulated

    real = ElasticView.from_suite(suite)
    flat = ElasticView(real.xs_energies, np.full_like(real.xs_values, 3.0), real.angular,
                       real.frame, real.awr)
    tof = TofResolution(flight_path_m=27.037, delta_t_ns=5.0, min_sigma_e_kev=0.0)
    setup = ForwardSetup(tof=tof)
    e = np.geomspace(0.9e6, 1.5e6, 15)
    s_e = np.array([kernel_sigma_ev(setup, x, real.awr) for x in e])
    grid = np.union1d(np.linspace(0.5e6, 2.0e6, 300001), real.energy_grids()[1])
    p = forward_dcs(flat, e, [0.96, -0.5], setup, fold="product")["dcs"]
    for j, mu in enumerate((0.96, -0.5)):
        mu_n, jac = real.native_cosine(mu)
        f_avg = fold_tabulated(grid, real.pdf_native(mu_n)(grid), e, s_e)
        np.testing.assert_allclose(p[:, j], 3.0 * f_avg * jac / (2 * np.pi), rtol=1e-4)


def test_the_factor_fold_misses_where_sigma_is_resonant(suite):
    """Fe-56 near 1 MeV: σ and f both move inside a 5 ns kernel, and the readings part."""
    view = ElasticView.from_suite(suite)
    tof = TofResolution(flight_path_m=27.037, delta_t_ns=5.0, min_sigma_e_kev=0.0)
    e = np.geomspace(0.9e6, 1.5e6, 15)
    p = forward_dcs(view, e, [0.96, -0.5], ForwardSetup(tof=tof), fold="product")
    s = forward_dcs(view, e, [0.96, -0.5], ForwardSetup(tof=tof), fold="sigma_only")
    np.testing.assert_array_equal(p["sigma"], s["sigma"])
    assert np.max(np.abs(p["dcs"] / s["dcs"] - 1)) > 0.05


def test_forward_sigma_is_the_dcs_integral(suite):
    view = ElasticView.from_suite(suite)
    tof = TofResolution(flight_path_m=27.037, delta_t_ns=1.5, min_sigma_e_kev=0.0)
    setup = ForwardSetup(tof=tof)
    e = np.array([1.0e6, 2.0e6])
    bins = [(0.998e6, 1.002e6), (1.99e6, 2.01e6)]
    x, w = np.polynomial.legendre.leggauss(64)
    dcs = forward_dcs(view, e, x, setup, bins)["dcs"]
    np.testing.assert_allclose(2 * np.pi * dcs @ w, forward_sigma(view, e, setup, bins),
                               rtol=1e-6)
