"""Overlapping-interval integrals and the common grid two tables are compared on."""
import numpy as np
import pytest
from scipy.integrate import quad

from kika import algebra as A


X = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
Y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
LAWS = np.array([2, 5, 3, 1])  # every law with a closed 1/x integral


@pytest.mark.parametrize("weight", [None, "1/x"])
def test_interval_integrals_match_quadrature_on_overlapping_windows(weight):
    lo = np.array([1.1, 1.5, 3.0, 0.5, 7.9, 6.0])
    hi = np.array([3.9, 15.0, 3.0, 2.5, 8.1, 20.0])
    got = A.interval_integrals(X, Y, LAWS, lo, hi, weight)
    w = (lambda t: 1.0) if weight is None else (lambda t: 1.0 / t)
    for g, a, b in zip(got, lo, hi):
        if b <= a:
            assert g == 0.0
            continue
        ref = quad(lambda t: A.evaluate(X, Y, LAWS, t) * w(t), a, b, points=X, limit=200)[0]
        assert g == pytest.approx(ref, rel=1e-10)


def test_interval_averages_equal_group_averages_on_a_tiling():
    edges = np.array([1.0, 1.7, 3.0, 9.0, 16.0])
    group = A.group_averages(X, Y, LAWS, edges, "1/x")
    window = A.interval_averages(X, Y, LAWS, edges[:-1], edges[1:], "1/x")
    np.testing.assert_array_equal(window, group)


def test_interval_integral_keeps_a_step():
    x, y = np.array([1.0, 2.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0, 5.0])
    assert A.interval_integrals(x, y, 2, [1.5, 1.0], [2.5, 3.0]).tolist() == [3.0, 6.0]


def test_interval_integral_does_not_cancel_against_a_large_running_total():
    # A 1/x-weighted integral dominated by its low end, then a tiny window far up.
    x = np.geomspace(1e-5, 1e4, 4001)
    y = 1e3 / np.sqrt(x / 1e-5)
    lo, hi = np.array([1000.0]), np.array([1000.0 * np.exp(0.05)])
    got = A.interval_integrals(x, y, 2, lo, hi, "1/x")[0]
    ref = A.integral(x, y, 2, lo[0], hi[0], "1/x")
    assert got == pytest.approx(ref, rel=1e-13)


def test_on_common_grid_union_keeps_every_node_and_step():
    a = (np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 3.0]), 2)
    b = (np.array([1.5, 2.0, 2.0, 5.0]), np.array([1.0, 1.0, 5.0, 5.0]), 2)
    u, v = A.on_common_grid([a, b])
    assert u.tolist() == [1.5, 2.0, 2.0, 4.0]
    assert v[0].tolist() == [2.0, 3.0, 3.0, 3.0]
    assert v[1].tolist() == [1.0, 1.0, 5.0, 5.0]


def test_on_common_grid_reference_misses_the_other_tables_peak():
    ref = (np.array([1.0, 3.0]), np.array([1.0, 1.0]), 2)
    peaked = (np.array([1.0, 2.0, 3.0]), np.array([1.0, 9.0, 1.0]), 2)
    _, on_ref = A.on_common_grid([ref, peaked], grid=0)
    _, on_union = A.on_common_grid([ref, peaked])
    assert np.max(on_ref[1] - on_ref[0]) == 0.0
    assert np.max(on_union[1] - on_union[0]) == 8.0


def test_on_common_grid_reads_each_table_under_its_own_law():
    loglog = (np.array([1.0, 4.0]), np.array([1.0, 16.0]), 5)
    linlin = (np.array([1.0, 2.0, 4.0]), np.array([0.0, 0.0, 0.0]), 2)
    u, v = A.on_common_grid([loglog, linlin], grid=1)
    assert v[0][1] == pytest.approx(4.0)  # x**2 at 2, not the lin-lin 6


def test_on_common_grid_without_a_shared_span_raises():
    with pytest.raises(ValueError, match="share no span"):
        A.on_common_grid([(np.array([1.0, 2.0]), np.ones(2), 2),
                          (np.array([3.0, 4.0]), np.ones(2), 2)])
