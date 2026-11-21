"""Response cells against integrals taken independently, a fine prior shape, and a rank-deficient prior."""
import warnings

import numpy as np
import pytest

from kika.algebra import group_integrals, product_group_integrals
from kika.dosimetry import BARN_CM2, Dosimeter, SpectrumPrior, adjust_spectrum, response_cells

EDGES = np.geomspace(1e5, 2e7, 31)
FINE = np.geomspace(1e5, 2e7, 400)
SIGMA = 0.5 / (1 + np.exp(-(FINE - 2.8e6) / 3.2e5))
COV_EDGES = np.array([5e5, 1e6, 2.5e6, 4e6, 7e6, 1.2e7, 3e7])


def _dosimeter():
    k = COV_EDGES.size - 1
    return Dosimeter("thr", FINE, SIGMA, 2, COV_EDGES, 0.03**2 * (0.5 * np.eye(k) + 0.5))


def _flux():
    mid = np.sqrt(EDGES[:-1] * EDGES[1:]) / 1e6
    return 1e10 * np.sqrt(mid) * np.exp(-mid / 1.3) * np.diff(EDGES) / 1e6


def test_rows_are_the_group_rates_and_columns_the_interval_rates():
    flux = _flux()
    prior = SpectrumPrior(EDGES, flux, 0.04 * np.eye(30))
    cells = response_cells(prior, _dosimeter())
    sigma_in_group = product_group_integrals((FINE, SIGMA, 2), (EDGES[[0, -1]], [1.0, 1.0], 2), EDGES)
    np.testing.assert_allclose(cells.sum(axis=1), BARN_CM2 * flux * sigma_in_group / np.diff(EDGES), rtol=1e-12)
    density = (EDGES, np.r_[flux / np.diff(EDGES), 0.0], 1)
    by_interval = BARN_CM2 * product_group_integrals((FINE, SIGMA, 2), density, COV_EDGES)
    np.testing.assert_allclose(cells[:, :-1].sum(axis=0), by_interval, rtol=1e-12)
    below = BARN_CM2 * product_group_integrals((FINE, SIGMA, 2), density, [EDGES[0], COV_EDGES[0]])[0]
    assert cells[:, -1].sum() == pytest.approx(below, rel=1e-12)


def test_with_a_fine_shape_the_rate_is_the_integral_over_the_fine_spectrum():
    phi = (FINE, 1e4 * np.sqrt(FINE / 1e6) * np.exp(-FINE / 1.3e6), 2)
    prior = SpectrumPrior(EDGES, group_integrals(*phi, EDGES), 0.04 * np.eye(30), fine_shape=phi)
    cells = response_cells(prior, _dosimeter())
    exact = BARN_CM2 * product_group_integrals((FINE, SIGMA, 2), phi, EDGES)
    np.testing.assert_allclose(cells.sum(axis=1), exact, rtol=1e-12)


def test_a_sample_covariance_of_fewer_replicas_than_groups_adjusts_inside_its_span():
    rng = np.random.default_rng(3)
    replicas = rng.normal(scale=0.1, size=(10, 30)) @ np.triu(np.ones((30, 30))) / 5
    deviations = replicas - replicas.mean(axis=0)
    C = deviations.T @ deviations / 9 + 0.03**2
    prior = SpectrumPrior(EDGES, _flux(), C)
    d = _dosimeter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = adjust_spectrum(prior, [d], [1.2 * response_cells(prior, d).sum()], [[0.04**2]])
    x = np.log(result.flux / prior.flux)
    w, q = np.linalg.eigh(C)
    span = q[:, w > 1e-12 * w.max()]
    assert span.shape[1] <= 10
    assert np.linalg.norm(x - span @ (span.T @ x)) < 1e-10 * np.linalg.norm(x)
