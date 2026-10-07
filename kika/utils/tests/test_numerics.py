"""Tests for the shared numerical primitives and the unified TOF resolution.

These pin the two properties the folding/TOF unification was done for:

* every entry point computes the *same* sigma_E, under both conventions;
* the Gaussian fold is the exact integral of the interpolant, so its answer
  does not depend on how densely the input happens to be sampled — which is
  exactly what the old point-weighted implementation got wrong — and carries
  no quadrature error at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika._constants import FWHM_TO_SIGMA
from kika.utils.numerics import (
    average_over_intervals,
    fold_tabulated,
    gaussian_fold_nodes,
)
from kika.utils import numerics
from kika.utils.energy_folding import tof_energy_resolution


# --- fold_tabulated --------------------------------------------------------

def test_fold_flat_is_identity():
    x = np.linspace(0.0, 10.0, 101)
    y = np.full_like(x, 3.0)
    assert fold_tabulated(x, y, 5.0, 1.0) == pytest.approx(3.0, rel=1e-12)


def test_fold_linear_returns_value_at_centroid():
    """Odd moments of a symmetric kernel vanish, so a line folds to y(x0)."""
    x = np.linspace(0.0, 10.0, 1001)
    y = 2.0 + 0.5 * x
    assert fold_tabulated(x, y, 5.0, 0.7) == pytest.approx(2.0 + 0.5 * 5.0, rel=1e-9)


def test_fold_quadratic_matches_analytic_second_moment():
    """For y = x^2 the exact fold is x0^2 + sigma^2."""
    x = np.linspace(-20.0, 20.0, 40001)
    y = x ** 2
    got = fold_tabulated(x, y, 1.0, 2.0)
    assert got == pytest.approx(1.0 ** 2 + 2.0 ** 2, rel=1e-4)


def test_fold_is_insensitive_to_input_sampling_density():
    """The regression the unification exists to prevent.

    A Gaussian-weighted average of tabulated *points* changes when the same
    function is resampled more densely near its peak.  Integrating the
    interpolant does not.
    """
    peak = lambda t: 1.0 + 40.0 * np.exp(-0.5 * ((t - 5.0) / 0.10) ** 2)
    uniform = np.linspace(0.0, 10.0, 4001)
    # Same function, but heavily oversampled around the peak.
    clustered = np.unique(np.concatenate([uniform, np.linspace(4.5, 5.5, 8000)]))

    def point_weighted(t):
        w = np.exp(-0.5 * ((t - 5.0) / 1.0) ** 2)
        return float(np.sum(w * peak(t)) / np.sum(w))

    quad_a = fold_tabulated(uniform, peak(uniform), 5.0, 1.0)
    quad_b = fold_tabulated(clustered, peak(clustered), 5.0, 1.0)
    pw_a, pw_b = point_weighted(uniform), point_weighted(clustered)

    quad_shift = abs(quad_b - quad_a) / quad_a
    pw_shift = abs(pw_b - pw_a) / pw_a

    # The quadrature is not perfectly invariant — the interpolant itself
    # improves with sampling — but it is orders of magnitude steadier than
    # weighting the points, which is the property that matters.
    assert quad_shift < 1e-4
    assert pw_shift > 0.05
    assert pw_shift / quad_shift > 1000


def test_fold_zero_sigma_is_interpolation():
    x = np.linspace(0.0, 10.0, 101)
    y = x ** 2
    assert fold_tabulated(x, y, 3.3, 0.0) == pytest.approx(np.interp(3.3, x, y))


def test_fold_vectorises_over_centroids():
    x = np.linspace(0.0, 10.0, 501)
    y = np.sin(x)
    centroids = np.array([2.0, 4.0, 6.0])
    got = fold_tabulated(x, y, centroids, 0.3)
    assert got.shape == (3,)
    for i, c in enumerate(centroids):
        assert got[i] == pytest.approx(fold_tabulated(x, y, float(c), 0.3))


# --- exactness ----------------------------------------------------------------

def _reference_fold(x, y, x0, s):
    """The same integral by adaptive quadrature, panel by panel between the
    table's own points, over +-12 sigma -- an independent route to the number.
    A uniform trapezoid is not one: it straddles every kink of the table and
    is off by ~1e-7 at two million points."""
    from scipy.integrate import quad

    def f(t):
        return np.interp(t, x, y) * np.exp(-0.5 * ((t - x0) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    lo, hi = x0 - 12 * s, x0 + 12 * s
    cuts = np.r_[lo, x[(x > lo) & (x < hi)], hi]
    return sum(quad(f, a, b, epsabs=0.0, epsrel=1e-13)[0] for a, b in zip(cuts[:-1], cuts[1:]))


def test_fold_is_the_exact_integral_of_the_interpolant():
    """A coarse, jagged table: no quadrature error left to see."""
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0.0, 10.0, 40))
    y = rng.uniform(0.0, 5.0, x.size)
    for x0, s in [(5.0, 0.05), (5.0, 1.0), (2.3, 3.0)]:
        assert fold_tabulated(x, y, x0, s) == pytest.approx(_reference_fold(x, y, x0, s), rel=1e-8)


def test_a_step_folds_to_the_normal_cdf():
    """A repeated abscissa is a step, and it is integrated as one."""
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert fold_tabulated(x, y, 1.0, 0.1) == pytest.approx(0.5, abs=1e-12)
    assert fold_tabulated(x, y, 1.1, 0.1) == pytest.approx(0.8413447460685429, rel=1e-12)


def test_past_the_table_the_end_value_is_held():
    x = np.array([0.0, 1.0])
    y = np.array([2.0, 2.0])
    assert fold_tabulated(x, y, 1.0, 0.5) == pytest.approx(2.0, rel=1e-12)
    assert fold_tabulated(x, y, 50.0, 0.5) == pytest.approx(2.0, rel=1e-12)


def test_batches_give_the_answer_one_batch_gives(monkeypatch):
    x = np.linspace(0.0, 10.0, 2001)
    y = np.sin(3 * x) ** 2
    centroids = np.linspace(1.0, 9.0, 300)
    whole = fold_tabulated(x, y, centroids, 0.2)
    monkeypatch.setattr(numerics, "_FOLD_BATCH_PAIRS", 500)
    np.testing.assert_allclose(fold_tabulated(x, y, centroids, 0.2), whole, rtol=1e-14)


# --- gaussian_fold_nodes ---------------------------------------------------

def test_fold_nodes_are_the_same_integral_as_the_table_fold():
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(0.0, 10.0, 300))
    y = rng.uniform(0.0, 5.0, x.size)
    nodes, w = gaussian_fold_nodes(5.0, 0.4, [x])
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    assert 5.0 in nodes
    # Equal up to where each holds the 1e-9 of mass past 6 sigma.
    assert float(w @ np.interp(nodes, x, y)) == pytest.approx(
        fold_tabulated(x, y, 5.0, 0.4), rel=1e-8)


def test_fold_nodes_need_the_integrand_breakpoints():
    with pytest.raises(ValueError):
        gaussian_fold_nodes(5.0, 1.0, [])


def test_nodes_degenerate_at_zero_sigma():
    pts, w = gaussian_fold_nodes(1.5, 0.0, [np.linspace(0.0, 3.0, 4)])
    np.testing.assert_allclose(pts, [1.5])
    np.testing.assert_allclose(w, [1.0])


# --- average_over_intervals ------------------------------------------------

def test_interval_average_of_a_line_is_the_midpoint_value():
    x = np.linspace(0.0, 10.0, 1001)
    y = 3.0 * x + 1.0
    edges = np.array([0.0, 2.0, 5.0, 10.0])
    got = average_over_intervals(x, y, edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    np.testing.assert_allclose(got, 3.0 * mids + 1.0, rtol=1e-9)


def test_interval_average_does_not_step_over_a_narrow_feature():
    """The native points are unioned into the sub-grid, so spikes survive."""
    x = np.unique(np.concatenate([
        np.linspace(0.0, 10.0, 101), np.linspace(4.999, 5.001, 201),
    ]))
    y = np.where(np.abs(x - 5.0) < 0.001, 100.0, 1.0)
    got = average_over_intervals(x, y, np.array([0.0, 10.0]), n_sub=5)
    assert got[0] > 1.0, "narrow spike was skipped by the uniform sub-grid"


# --- TOF resolution --------------------------------------------------------

def test_fwhm_and_sigma_conventions_differ_by_the_gaussian_factor():
    kw = dict(flight_path_m=27.037, delta_t_ns=5.0)
    as_fwhm = tof_energy_resolution(2.0, delta_t_is_fwhm=True, **kw)
    as_sigma = tof_energy_resolution(2.0, delta_t_is_fwhm=False, **kw)
    assert as_sigma / as_fwhm == pytest.approx(FWHM_TO_SIGMA, rel=1e-12)


def test_resolution_scales_as_sqrt_energy():
    """sigma_E / E = 2 dt / t and t ~ 1/sqrt(E), so sigma_E ~ E^{3/2}."""
    kw = dict(flight_path_m=27.037, delta_t_ns=5.0, delta_t_is_fwhm=True)
    s1 = tof_energy_resolution(1.0, **kw)
    s4 = tof_energy_resolution(4.0, **kw)
    assert s4 / s1 == pytest.approx(4.0 ** 1.5, rel=1e-9)


def test_all_entry_points_agree():
    """One formula, five call sites — the point of the unification."""
    from kika.ace.classes.angular_distribution.container import (
        _compute_energy_resolution_tof as ace_fn,
    )
    from scripts.resample_AD import compute_energy_resolution_tof as pipeline_fn
    from scripts.tof_parameters import compute_sigma_E_direct

    for energy in (0.847, 2.0, 4.0):
        for fwhm in (True, False):
            core = tof_energy_resolution(
                energy, flight_path_m=27.037, delta_t_ns=5.0, delta_t_is_fwhm=fwhm,
            )
            assert ace_fn(energy, 27.037, 5.0, delta_t_is_fwhm=fwhm) == core
            assert pipeline_fn(
                E_mev=energy, delta_t_ns=5.0, flight_path_m=27.037,
                delta_t_is_fwhm=fwhm,
            ) == core
            # compute_sigma_E_direct applies a 1 keV floor; compare above it.
            assert compute_sigma_E_direct(
                energy, 27.037, 5.0, delta_t_is_fwhm=fwhm,
            ) == pytest.approx(max(core, 1e-3))


def test_zero_energy_is_zero():
    assert tof_energy_resolution(0.0, flight_path_m=27.0, delta_t_ns=5.0) == 0.0


# ─── Structure finer than the kernel ────────────────────────────────────────

def test_the_fold_resolves_structure_finer_than_the_kernel():
    """A table with structure finer than the kernel, the case a fixed set of
    nodes (Gauss-Hermite, a uniform window) gets wrong."""
    rng = np.random.default_rng(11)
    x = np.sort(rng.uniform(0.9e6, 1.1e6, 4000))
    y = 0.5 + 7.5 * rng.random(x.size)
    x0, s = 1.0e6, 2.2e3
    assert fold_tabulated(x, y, x0, s) == pytest.approx(_reference_fold(x, y, x0, s), rel=1e-8)


def test_fold_nodes_include_the_table_inside_the_window():
    x = np.linspace(0.0, 10.0, 1001)
    nodes, w = gaussian_fold_nodes(5.0, 0.3, [x])
    assert w.sum() == pytest.approx(1.0)
    inside = x[np.abs(x - 5.0) < 1.5]
    assert np.isin(inside, nodes).all()
