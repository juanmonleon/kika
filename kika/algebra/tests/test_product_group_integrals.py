"""Group integrals of a product of two tables against adaptive quadrature of the functions themselves."""
import numpy as np
import pytest
from scipy.integrate import quad

from kika.algebra import group_integrals, product_group_integrals


def _linlin(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return lambda t: float(np.interp(t, x, y)) if x[0] <= t <= x[-1] else 0.0


def _quad_groups(fun, edges, breaks):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inner = sorted(b for b in breaks if lo < b < hi)
        out.append(quad(fun, lo, hi, points=inner or None, limit=500, epsabs=0, epsrel=1e-13)[0] if hi > lo else 0.0)
    return np.array(out)


def test_two_linlin_tables_on_different_grids_are_exact():
    xf, yf = [1.0, 2.5, 4.0, 7.0], [3.0, 1.0, 5.0, 2.0]
    xg, yg = [0.5, 3.0, 6.0, 9.0], [1.0, 4.0, 2.0, 6.0]
    edges = np.array([0.0, 1.7, 3.0, 3.3, 6.5, 10.0])
    got = product_group_integrals((xf, yf, 2), (xg, yg, 2), edges)
    f, g = _linlin(xf, yf), _linlin(xg, yg)
    want = _quad_groups(lambda t: f(t) * g(t), edges, xf + xg)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_a_histogram_keeps_its_steps():
    xh, yh = [1.0, 2.0, 3.0, 5.0], [2.0, 7.0, 1.0, 1.0]
    xg, yg = [0.0, 6.0], [1.0, 13.0]
    edges = np.array([0.0, 1.5, 2.0, 2.2, 4.0, 6.0])
    got = product_group_integrals((xh, yh, 1), (xg, yg, 2), edges)
    g = _linlin(xg, yg)

    def h(t):
        if not 1.0 <= t < 5.0:
            return 0.0
        return yh[np.searchsorted(xh, t, side="right") - 1]

    want = _quad_groups(lambda t: h(t) * g(t), edges, xh)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_log_laws_are_good_to_the_tolerance():
    x = np.geomspace(1e-3, 20.0, 9)
    sigma = 2.0 * x ** -0.5
    phi = np.sqrt(x) * np.exp(-x / 1.3)
    edges = np.geomspace(1e-3, 20.0, 14)
    got = product_group_integrals((x, sigma, 5), (x, phi, 2), edges, tol=1e-7)
    phi_t = _linlin(x, phi)
    want = _quad_groups(lambda t: 2.0 * t ** -0.5 * phi_t(t), edges, list(x))
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_a_constant_second_table_gives_the_group_integrals_of_the_first():
    x, y = np.array([1.0, 2.0, 2.0, 4.0, 8.0]), np.array([1.0, 3.0, 0.5, 2.0, 1.0])
    edges = np.array([0.5, 1.5, 2.0, 3.0, 9.0])
    got = product_group_integrals((x, y, 2), ([0.0, 10.0], [1.0, 1.0], 2), edges)
    np.testing.assert_allclose(got, group_integrals(x, y, 2, edges), rtol=1e-14, atol=1e-15)


def test_groups_outside_both_tables_and_of_zero_width_are_zero():
    edges = np.array([-5.0, -1.0, -1.0, 0.5, 2.0, 20.0, 30.0])
    got = product_group_integrals(([0.0, 1.0, 3.0], [1.0, 2.0, 1.0], 2),
                                  ([0.0, 3.0], [1.0, 1.0], 2), edges)
    assert got.shape == (6,)
    assert got[0] == got[1] == got[5] == 0.0
    assert got.sum() == pytest.approx(1.5 + 3.0, rel=1e-14)


def test_bad_edges_raise():
    with pytest.raises(ValueError):
        product_group_integrals(([0.0, 1.0], [1.0, 1.0], 2), ([0.0, 1.0], [1.0, 1.0], 2), [1.0, 0.0])
