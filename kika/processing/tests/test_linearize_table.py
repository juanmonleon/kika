"""Tests for re-expressing a table on any ENDF law as lin-lin.

:func:`~kika.processing.linearization.linearize_table` is judged against the
interpolator that defines what a table means,
:func:`~kika.processing.interpolation.interpolate_1d`: lin-lin on the output must
reproduce it to the stated tolerance everywhere, and every original point must
come through untouched.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.processing.interpolation import interpolate_1d, interval_codes
from kika.processing.linearization import linearize_table


def _probe(x):
    """Points inside every panel, avoiding the abscissae themselves."""
    edges = np.unique(x)
    inner = [np.geomspace(a, b, 203)[1:-1] if a > 0 else np.linspace(a, b, 203)[1:-1]
             for a, b in zip(edges[:-1], edges[1:])]
    return np.concatenate(inner)


def test_interval_codes_follow_the_interpolator():
    """The interval across a region boundary belongs to the region after it."""
    codes = interval_codes(6, [(3, 2), (6, 5)])
    np.testing.assert_array_equal(codes, [2, 2, 5, 5, 5])


def test_interval_codes_hold_the_last_law_when_nbt_falls_short():
    np.testing.assert_array_equal(interval_codes(5, [(3, 4)]), [4, 4, 4, 4])


def test_interval_codes_default_to_lin_lin():
    np.testing.assert_array_equal(interval_codes(4, []), [2, 2, 2])


@pytest.mark.parametrize("law", [3, 4, 5])
def test_every_log_law_is_reproduced_to_the_tolerance(law):
    x = np.array([1.0, 3.0, 100.0, 1.0e4])
    y = np.array([5.0, 0.2, 40.0, 0.01])
    xs, ys = linearize_table(x, y, [(4, law)], tol=1e-4)

    probe = _probe(x)
    exact = interpolate_1d(x, y, [(4, law)], probe)
    error = np.abs(np.interp(probe, xs, ys) - exact) / np.abs(exact)
    assert error.max() < 1e-4


def test_the_original_points_come_through_untouched():
    x = np.array([1.0, 10.0, 100.0])
    y = np.array([2.0, 0.5, 9.0])
    xs, ys = linearize_table(x, y, [(3, 5)], tol=1e-4)

    keep = np.isin(xs, x)
    np.testing.assert_array_equal(xs[keep], x)
    np.testing.assert_array_equal(ys[keep], y)
    assert np.all(np.diff(xs) > 0)


def test_lin_lin_is_returned_as_it_is():
    x = np.array([1.0, 2.0, 2.0, 5.0])
    y = np.array([1.0, 3.0, 4.0, 0.0])
    xs, ys = linearize_table(x, y, [(4, 2)])

    np.testing.assert_array_equal(xs, x)
    np.testing.assert_array_equal(ys, y)


def test_a_histogram_becomes_steps_at_repeated_abscissae():
    xs, ys = linearize_table([1.0, 2.0, 3.0], [1.0, 4.0, 4.0], [(3, 1)])

    np.testing.assert_array_equal(xs, [1.0, 2.0, 2.0, 3.0])
    np.testing.assert_array_equal(ys, [1.0, 1.0, 4.0, 4.0])


def test_mixed_regions_keep_a_discontinuity_and_refine_only_the_log_ones():
    """lin-lin, then a log-log panel, with a step written as a repeated energy."""
    x = np.array([1.0, 2.0, 2.0, 200.0])
    y = np.array([1.0, 1.0, 50.0, 0.5])
    pairs = [(2, 2), (4, 5)]
    xs, ys = linearize_table(x, y, pairs, tol=1e-4)

    assert np.count_nonzero(xs == 2.0) == 2
    assert not np.any((xs > 1.0) & (xs < 2.0))
    probe = np.geomspace(2.0, 200.0, 4001)[1:-1]
    exact = interpolate_1d(x, y, pairs, probe)
    assert np.max(np.abs(np.interp(probe, xs, ys) - exact) / exact) < 1e-4


def test_snapped_nodes_are_valued_where_they_land():
    """Every added node is on the snap grid, and carries the law's value there."""
    x = np.array([1.0, 100.0])
    y = np.array([10.0, 0.1])
    xs, ys = linearize_table(x, y, [(2, 5)], tol=1e-4,
                             snap=lambda v: np.round(v, 1))

    added = ~np.isin(xs, x)
    assert added.any()
    np.testing.assert_array_equal(xs[added], np.round(xs[added], 1))
    np.testing.assert_allclose(ys[added], 10.0 / xs[added], rtol=1e-12)


def test_a_panel_the_snap_grid_cannot_split_is_left_as_it_is():
    xs, ys = linearize_table([1.0, 2.0], [10.0, 0.1], [(2, 5)],
                             snap=lambda v: np.round(v))

    np.testing.assert_array_equal(xs, [1.0, 2.0])
    np.testing.assert_array_equal(ys, [10.0, 0.1])


def test_a_log_law_the_interpolator_reads_lin_lin_is_left_alone():
    """A zero on a log-y law: interpolate_1d falls back to lin-lin there."""
    x = np.array([1.0, 10.0])
    y = np.array([0.0, 3.0])
    xs, ys = linearize_table(x, y, [(2, 5)])

    np.testing.assert_array_equal(xs, x)
    np.testing.assert_array_equal(ys, y)
