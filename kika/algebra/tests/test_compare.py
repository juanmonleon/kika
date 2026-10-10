"""A table against another, and against its own window and group averages."""
import numpy as np
import pytest

import kika.algebra as A


X = np.geomspace(1.0, 1000.0, 200)


def test_window_against_a_constant_table_is_zero_both_ways():
    centres = np.geomspace(10.0, 100.0, 9)
    for side in ("reference", "comparison"):
        got = A.window_difference(centres, np.full(9, 3.0), X, np.full(X.size, 3.0),
                                  "log-log", average_is=side)
        assert got == pytest.approx(np.zeros(9), abs=1e-14)


def test_window_relative_divides_by_the_reference_whichever_it_is():
    centres = np.array([10.0, 20.0])
    avg, flat = np.full(2, 2.0), np.full(X.size, 2.2)
    as_ref = A.window_difference(centres, avg, X, flat, "lin-lin",
                                 average_is="reference", percent=True)
    as_cmp = A.window_difference(centres, avg, X, flat, "lin-lin",
                                 average_is="comparison", percent=True)
    assert as_ref == pytest.approx([10.0, 10.0])
    assert as_cmp == pytest.approx([100 * (2.0 - 2.2) / 2.2] * 2)


def test_window_reads_the_table_by_its_law_and_is_nan_off_it():
    x, y = np.array([1.0, 100.0]), np.array([1.0, 100.0])
    centres = np.array([10.0, 1000.0])
    got = A.window_difference(centres, np.array([10.0, 5.0]), x, y, "log-log",
                              mode="absolute")
    assert got[0] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(got[1])


def test_steps_against_a_constant_table_is_zero():
    edges = np.geomspace(10.0, 100.0, 4)
    u, d = A.steps_difference(edges, np.full(3, 3.0), X, np.full(X.size, 3.0), "log-log")
    assert u[0] == 10.0 and u[-1] == 100.0
    assert d == pytest.approx(np.zeros(u.size), abs=1e-14)


def test_steps_step_at_each_inner_edge():
    x, y = np.array([0.0, 4.0]), np.array([1.0, 1.0])
    u, d = A.steps_difference([1.0, 2.0, 3.0], [1.0, 2.0], x, y, "lin-lin",
                              average_is="comparison", mode="absolute")
    # 2.0 twice: the group below (avg 1), then the group above (avg 2).
    assert u.tolist() == [1.0, 2.0, 2.0, 3.0]
    assert d.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_steps_diff_of_a_table_from_its_exact_average_integrates_to_zero():
    # Lin-lin table, flat weight: the mean of (y - mean) over each group is 0.
    x = np.linspace(0.0, 10.0, 41)
    y = np.sin(x) + 2.0
    edges = np.array([1.0, 3.3, 7.0, 9.5])
    avg = A.group_averages(x, y, 2, edges)
    u, d = A.steps_difference(edges, avg, x, y, "lin-lin", average_is="reference",
                              mode="absolute")
    assert np.abs(d).max() > 0.5          # spiky: the table strays from its mean
    per_group = A.group_integrals(*_pieces(u, d), edges)
    assert per_group == pytest.approx(np.zeros(3), abs=1e-12)


def _pieces(u, d):
    return u, d, 2


def test_steps_relative_and_nan_groups():
    x, y = np.array([1.0, 3.0]), np.array([2.0, 2.0])
    u, d = A.steps_difference([1.0, 2.0, 3.0], [4.0, np.nan], x, y, "lin-lin",
                              average_is="reference")
    assert d[u < 2.0] == pytest.approx([-0.5])
    assert np.isnan(d[-1])


def test_a_hole_in_the_table_is_nan_around_it():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 1.0, np.nan, 1.0, 1.0])
    got = A.interpolate_to_grid([1.5, 2.5, 3.5, 4.5], x, y, "lin-lin")
    assert got[0] == 1.0 and got[3] == 1.0
    assert np.isnan(got[1]) and np.isnan(got[2])
    u, d = A.steps_difference([1.0, 5.0], [1.0], x, y, "lin-lin")
    assert np.isnan(d[(u > 2.0) & (u < 4.0)]).all()
    assert d[u <= 2.0] == pytest.approx(0.0)


def test_difference_modes():
    ref, cmp_ = np.array([2.0, 0.0, np.nan]), np.array([3.0, 1.0, 1.0])
    rel = A.difference(ref, cmp_, "relative", percent=True)
    assert rel[0] == pytest.approx(50.0) and np.isnan(rel[1]) and np.isnan(rel[2])
    assert A.difference(ref, cmp_, "absolute")[:2].tolist() == [1.0, 1.0]
    with pytest.raises(ValueError):
        A.difference(ref, cmp_, "ratio")


def test_log_window_average_is_the_group_average_at_a_group_centre():
    x = np.geomspace(1.0, 1e4, 300)
    y = 1.0 + 1.0 / (1.0 + ((x - 50.0) / 2.0) ** 2)
    edges = np.geomspace(10.0, 1000.0, 11)
    width = np.log(edges[1] / edges[0])
    centres = np.sqrt(edges[:-1] * edges[1:])
    window = A.log_window_averages(x, y, 2, centres, width, "1/x")
    assert window == pytest.approx(A.group_averages(x, y, 2, edges, "1/x"), rel=1e-12)


def test_log_window_average_clips_to_the_bounds():
    x = np.array([1.0, 100.0])
    y = np.array([1.0, 100.0])
    # Window [5, 20] clipped to [10, 20]: the flat mean of y = x there is 15.
    got = A.log_window_averages(x, y, 2, [10.0], 2 * np.log(2.0), None, bounds=(10.0, 50.0))
    assert got == pytest.approx([15.0])
