"""The ratio method (roadmap G4NDL §0bis, Fase 13): closure on a synthetic truth.

The truth is the Fe-56 micro-tape with a known smooth r(E) on σ and δa_l(E) added
to f's Legendre coefficients;
the pseudo-data are the truth read through the forward operator. Correcting the
untouched tape must give the truth back where the data are, leave it alone far
from them, and never touch the input suite.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf.dcs import TofResolution
from kika.nuclear_data import ratio_correction as rc
from kika.nuclear_data.forward import ElasticView, ForwardSetup, forward_dcs, forward_sigma

FE56 = Path(kika.__file__).parent / "endf" / "tests" / "data" / "micro_fe56_xs_and_angular.endf"
E = np.linspace(2.0e6, 3.0e6, 48)
BINS = [(x - 4e3, x + 4e3) for x in E]
MU = np.polynomial.legendre.leggauss(8)[0]
SETUP = ForwardSetup(tof=TofResolution(flight_path_m=27.037, delta_t_ns=3.5,
                                       delta_t_is_fwhm=True, min_sigma_e_kev=0.0),
                     temperature_k=293.6)
INSIDE = np.linspace(2.1e6, 2.9e6, 1601)
FAR = np.array([1.0e6, 1.5e6, 4.0e6, 6.0e6])


def r_true(q):
    return 1.0 + 0.08 * np.sin((np.asarray(q, float) - 2.0e6) / 2.0e5)


def d_true(q):
    return np.tile([-0.02, 0.01, 0.01], (np.size(q), 1))


@pytest.fixture(scope="module")
def suite():
    from kika.endf.model_adapter.pendf import attachReconstruction

    s = kika.read(str(FE56), format="endf")
    attachReconstruction(s, FE56)
    return s


@pytest.fixture(scope="module")
def truth(suite):
    t = copy.deepcopy(suite)
    rc._multiply_cross_section(t, "recon", r_true, np.linspace(1.9e6, 3.1e6, 400), 1e-5)
    problems, _, _ = rc._add_angular(t, d_true, np.linspace(1.9e6, 3.1e6, 40), 1e-5, 2001)
    assert not problems, "the synthetic truth itself must be a distribution"
    return t


def _sigma_arrays(s):
    v = ElasticView.from_suite(s)
    return v.xs_energies.copy(), v.xs_values.copy()


# ------------------------------------------------------------------ the smoother

def test_the_kernel_is_never_narrower_than_the_floor():
    e = np.linspace(1.0, 2.0, 50)
    width = lambda x: np.full(np.size(x), 0.01)                       # noqa: E731
    fit = rc.fit_smooth_ratio(e, np.ones_like(e), np.full_like(e, 0.01), width, min_width=1.5)
    assert fit.factor >= 1.5 * rc.FWHM_PER_SIGMA
    with pytest.raises(ValueError, match="no candidate factor"):
        rc.fit_smooth_ratio(e, np.ones_like(e), np.full_like(e, 0.01), width,
                            min_width=1.0, factors=[0.5, 1.0])


def test_a_flat_ratio_ramps_back_to_one_outside():
    e = np.linspace(10.0, 20.0, 30)
    width = lambda x: np.full(np.size(x), 0.1)                        # noqa: E731
    fit = rc.fit_smooth_ratio(e, np.full_like(e, 1.2), np.full_like(e, 0.01), width)
    assert fit(np.array([15.0]))[0] == pytest.approx(1.2)
    lo, hi = fit.support
    assert fit(np.array([lo - 1e-9, hi + 1e-9])) == pytest.approx([1.0, 1.0])
    assert fit(np.array([10.0 - fit.hold(10.0)]))[0] == pytest.approx(1.2)


def test_the_legendre_sum_is_exact_and_stays_normalised():
    a = np.array([1.0, 0.3])
    delta = np.array([0.05, -0.02, 0.01])
    a_new = rc._legendre_plus(a, delta)
    assert a_new == pytest.approx([1.0, 0.35, -0.02, 0.01])
    mu, w = np.polynomial.legendre.leggauss(20)
    f = lambda c: np.polynomial.legendre.legval(mu, c * (2 * np.arange(c.size) + 1) / 2)  # noqa: E731
    assert float(w @ f(a_new)) == pytest.approx(1.0, rel=1e-14)
    assert np.allclose(f(a_new) - f(np.pad(a, (0, 2))),
                       np.polynomial.legendre.legval(mu, rc._shape_series(delta)), atol=1e-14)


def test_the_shape_basis_is_the_response_to_one_coefficient(suite):
    v = ElasticView.from_suite(suite)
    e = np.array([2.5e6])
    base = forward_dcs(v, e, MU, SETUP)["dcs"]
    bumped = copy.deepcopy(suite)
    step = np.array([0.0, 0.01])
    rc._add_angular(bumped, lambda q: np.tile(step, (np.size(q), 1)),
                    np.linspace(2.0e6, 3.0e6, 5), 1e-6, 201)
    fwd = forward_dcs(ElasticView.from_suite(bumped), e, MU, SETUP)
    basis = rc._shape_basis(v, MU, SETUP, fwd["sigma"], 2)
    assert np.allclose(fwd["dcs"] - base, 0.01 * basis[:, :, 1], rtol=1e-6, atol=1e-12)


# ------------------------------------------------------------------ closure

def test_the_cross_section_closes_on_the_truth(suite, truth):
    before = _sigma_arrays(suite)
    sig = forward_sigma(ElasticView.from_suite(truth), E, SETUP, BINS)
    new, rep = rc.correct_cross_section(suite, E, sig, 0.01 * sig, SETUP, bins_ev=BINS)
    assert rep.converged and rep.chi2[-1] < 0.05 * rep.chi2[0]
    vn, vt, v0 = (ElasticView.from_suite(x) for x in (new, truth, suite))
    assert np.max(np.abs(vn.sigma(INSIDE) / vt.sigma(INSIDE) - 1.0)) < 2e-3
    assert np.array_equal(vn.sigma(FAR), v0.sigma(FAR))
    # the evaluation's own points are all kept: the structure is not resampled away
    assert np.all(np.isin(v0.xs_energies, vn.xs_energies))
    after = _sigma_arrays(suite)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])


def test_the_angular_shape_closes_and_leaves_sigma_alone(suite, truth):
    vt = ElasticView.from_suite(truth)
    dcs = forward_dcs(vt, E, MU, SETUP, BINS)["dcs"]
    # σ already right, so the residual is the shape alone
    fixed = copy.deepcopy(suite)
    rc._multiply_cross_section(fixed, "recon", r_true, np.linspace(1.9e6, 3.1e6, 400), 1e-5)
    new, rep = rc.correct_angular(fixed, E, MU, dcs, 0.02 * dcs, SETUP, bins_ev=BINS)
    assert rep.converged and rep.level == "separate" and rep.excluded == 0
    vn = ElasticView.from_suite(new)
    worst = max(np.max(np.abs(vn.pdf_native(m)(INSIDE) / vt.pdf_native(m)(INSIDE) - 1.0))
                for m in np.linspace(-1, 1, 21))
    assert worst < 3e-3
    assert np.array_equal(vn.xs_values, ElasticView.from_suite(fixed).xs_values)
    v0 = ElasticView.from_suite(suite)
    for m in (-0.9, 0.0, 0.9):
        assert np.array_equal(vn.pdf_native(m)(FAR), v0.pdf_native(m)(FAR))
    # the micro-tape's own negative record at 2.414 MeV is never blamed on the
    # correction; this truth happens to lift it above zero, so it may not be listed
    assert {x[0] for x in rep.inherited_negative} <= {2.414e6}


def test_level_from_dcs_scales_sigma_by_the_fitted_level(suite, truth):
    vt = ElasticView.from_suite(truth)
    dcs = forward_dcs(vt, E, MU, SETUP, BINS)["dcs"]
    new, rep = rc.correct_angular(suite, E, MU, dcs, 0.02 * dcs, SETUP, bins_ev=BINS,
                                  level="from_dcs", max_iterations=1)
    assert rep.cross_section is not None
    v0 = ElasticView.from_suite(suite)
    q = v0.xs_energies[(v0.xs_energies > 2.2e6) & (v0.xs_energies < 2.8e6)]
    ratio = ElasticView.from_suite(new).sigma(q) / v0.sigma(q)
    assert np.allclose(ratio, rep.level_factor(q), rtol=1e-6)
    # and the level it found is the r(E) the truth was built with, at the kernel's width
    assert np.max(np.abs(rep.level_factor(q) / r_true(q) - 1.0)) < 0.02


def test_nothing_is_corrected_below_emin(suite, truth):
    vt = ElasticView.from_suite(truth)
    dcs = forward_dcs(vt, E, MU, SETUP, BINS)["dcs"]
    new, rep = rc.correct_angular(suite, E, MU, dcs, 0.02 * dcs, SETUP, bins_ev=BINS,
                                  emin_ev=2.5e6, max_iterations=1)
    assert rep.emin == 2.5e6 and 0 < rep.excluded < E.size
    assert rep.iterations[0].support[0] >= 2.5e6
    vn, v0 = ElasticView.from_suite(new), ElasticView.from_suite(suite)
    below = np.linspace(2.0e6, 2.49e6, 7)
    assert np.array_equal(vn.pdf_native(0.3)(below), v0.pdf_native(0.3)(below))
    with pytest.raises(ValueError, match="clear of emin"):
        rc.correct_angular(suite, E, MU, dcs, 0.02 * dcs, SETUP, bins_ev=BINS, emin_ev=3.5e6)


def test_one_measured_energy_touches_only_its_neighbourhood(suite, truth):
    vt = ElasticView.from_suite(truth)
    e0 = np.array([2.5e6])
    dcs = forward_dcs(vt, e0, MU, SETUP)["dcs"]
    new, rep = rc.correct_angular(suite, e0, MU, dcs, 0.02 * dcs, SETUP, degree=3)
    lo, hi = rep.iterations[0].support
    assert lo > 2.45e6 and hi < 2.55e6
    vn, v0 = ElasticView.from_suite(new), ElasticView.from_suite(suite)
    away = np.array([2.3e6, 2.7e6])
    assert np.array_equal(vn.pdf_native(0.3)(away), v0.pdf_native(0.3)(away))


def test_a_correction_that_makes_f_negative_raises(suite):
    v0 = ElasticView.from_suite(suite)
    dcs = forward_dcs(v0, E, MU, SETUP, BINS)["dcs"]
    mu_cm = np.array([v0.native_cosine(m)[0] for m in MU])
    with pytest.raises(ValueError, match="negative"):
        rc.correct_angular(suite, E, MU, dcs * (1.0 + 1.6 * mu_cm), 0.02 * dcs, SETUP,
                           bins_ev=BINS, degree=1)
