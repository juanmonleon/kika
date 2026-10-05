"""The arithmetic of perturbing a tabulated f(mu) by Legendre-order factors.

T1 of ``docs/library/mf4_tabulated_perturbation_roadmap.md``: one table, one
incident energy, no model nodes. The energy dimension and the gate against the
Legendre applier are further down the roadmap and further down this file.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import legendre as L

from kika.nuclear_data.model.angular_tables import (legendreMoments,
                                                    perturbTabulatedAngular,
                                                    tableIntegral)


def _pdf(coefficients, mu):
    """f(mu) = sum (2l+1)/2 a_l P_l(mu)."""
    series = [0.5 * (2 * l + 1) * a for l, a in enumerate(coefficients)]
    return L.legval(mu, series)


def _forwardPeaked(n=91):
    """A real-looking table: smooth plus a forward peak no low order holds."""
    mu = np.concatenate([np.linspace(-1.0, 0.9, n - 20), np.linspace(0.905, 1.0, 20)])
    p = 0.3 + 0.2 * mu + 5.0 * np.exp(-((1.0 - mu) / 0.03))
    return mu, p / tableIntegral(mu, p)


def test_every_factor_one_gives_the_table_back_bit_for_bit():
    mu, p = _forwardPeaked()
    pPrime, info = perturbTabulatedAngular(mu, p, {1: 1.0, 2: 1.0, 3: 1.0})
    assert np.array_equal(pPrime, p)
    assert pPrime is not p
    assert all(delta == 0.0 for delta in info["deltas"].values())


def test_the_projection_of_a_linlin_table_is_exact():
    """f linear between nodes: each panel integrates to rounding, at any order.

    Checked against the closed form for a single panel spanning [-1, 1], where
    f = a + b mu gives a_0 = 2a, a_1 = 2b/3 and nothing above.
    """
    mu = np.array([-1.0, 1.0])
    p = np.array([0.2, 0.8])                   # a = 0.5, b = 0.3
    moments = legendreMoments(mu, p, range(0, 7))
    assert moments[0] == pytest.approx(1.0, abs=1e-15)
    assert moments[1] == pytest.approx(0.2, abs=1e-15)
    for order in range(2, 7):
        assert moments[order] == pytest.approx(0.0, abs=1e-15)


def test_the_moments_shift_by_exactly_the_deltas_and_nothing_else_moves():
    """Project ``f'`` and ``f``: the named orders differ by delta, others by ~0.

    Exact up to the lin-lin interpolation of the correction between nodes,
    which is what D3 measures; at 401 nodes it is below 1e-5.
    """
    mu = np.cos(np.linspace(np.pi, 0.0, 401))
    p = _pdf([1.0, 0.4, 0.2, 0.1], mu)
    factors = {1: 1.10, 2: 0.95, 3: 1.20}
    pPrime, info = perturbTabulatedAngular(mu, p, factors)
    before = legendreMoments(mu, p, range(0, 8))
    after = legendreMoments(mu, pPrime, range(0, 8))
    for order in range(0, 8):
        expected = info["deltas"].get(order, 0.0)
        assert after[order] - before[order] == pytest.approx(expected, abs=1e-5), order


def test_the_integral_is_kept_to_the_linearisation():
    mu, p = _forwardPeaked()
    pPrime, info = perturbTabulatedAngular(mu, p, {1: 1.1, 2: 0.9})
    assert info["integral_before"] == pytest.approx(1.0, abs=1e-12)
    assert abs(info["integral_after"] - info["integral_before"]) < 1e-3


def test_an_analytic_case_comes_out_exact():
    """f = 1/2 (1 + 3 a1 mu), 201 nodes, c_1 = 1.1: f' = 1/2 (1 + 3.3 a1 mu)."""
    a1 = 0.3
    mu = np.linspace(-1.0, 1.0, 201)
    p = 0.5 * (1.0 + 3.0 * a1 * mu)
    pPrime, info = perturbTabulatedAngular(mu, p, {1: 1.1})
    np.testing.assert_allclose(pPrime, 0.5 * (1.0 + 3.0 * 1.1 * a1 * mu),
                               rtol=0, atol=1e-14)
    assert info["integral_after"] == pytest.approx(1.0, abs=1e-14)


def test_the_forward_peak_survives_a_low_order_perturbation():
    """The point of D-B: the naive resum would flatten the peak; this keeps it."""
    mu, p = _forwardPeaked()
    pPrime, _info = perturbTabulatedAngular(mu, p, {1: 1.05, 2: 1.05})
    peak = mu > 0.95
    assert np.max(np.abs(pPrime[peak] / p[peak] - 1.0)) < 0.05


def test_a_negative_node_is_reported_and_left_alone():
    mu = np.linspace(-1.0, 1.0, 51)
    p = _pdf([1.0, 0.3], mu)                   # 0.05 at mu = -1
    pPrime, info = perturbTabulatedAngular(mu, p, {1: 1.5})
    assert info["n_negative"] > 0 and info["min_p"] < 0.0
    assert pPrime[0] == pytest.approx(0.5 * (1.0 - 3.0 * 0.45))


def test_the_magnitude_order_and_unknown_laws_are_refused():
    mu = np.linspace(-1.0, 1.0, 5)
    p = np.full(5, 0.5)
    with pytest.raises(ValueError, match="magnitude"):
        perturbTabulatedAngular(mu, p, {0: 1.1})
    with pytest.raises(NotImplementedError, match="transcendental"):
        perturbTabulatedAngular(mu, p, {1: 1.1}, pairs=[(5, 5)])
    with pytest.raises(ValueError, match=r"not \[-1, 1\]"):
        perturbTabulatedAngular(np.linspace(-0.5, 1.0, 5), p, {1: 1.1})


def test_a_histogram_table_is_projected_exactly():
    """INT=1: f = 0.25 on [-1, 0), 0.75 on [0, 1]; a_1 = int mu f = 0.25."""
    mu = np.array([-1.0, 0.0, 1.0])
    p = np.array([0.25, 0.75, 0.75])
    moments = legendreMoments(mu, p, [0, 1], pairs=[(3, 1)])
    assert moments[0] == pytest.approx(1.0, abs=1e-15)
    assert moments[1] == pytest.approx(0.25, abs=1e-15)
