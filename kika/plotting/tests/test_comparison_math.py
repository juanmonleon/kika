"""The comparison's numbers come from kika.algebra: laws per interval, steps, holes."""
import numpy as np
import pytest

from kika.plotting.comparison import compute_difference, interpolate_to_grid, method_laws
from kika.plotting.plot_data import PlotData


def test_log_y_falls_back_only_on_the_intervals_without_a_log():
    # Before: one non-positive value turned the whole series lin-lin.
    x, y = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, -1.0, 2.0, 8.0])
    assert method_laws(y, "lin-log").tolist() == [2, 2, 4]
    got = interpolate_to_grid([1.5, 3.5], x, y, method="lin-log")
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(4.0)  # geometric mean of 2 and 8, not 5


def test_a_repeated_source_abscissa_is_a_step():
    x, y = np.array([1.0, 2.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0, 5.0])
    got = interpolate_to_grid([1.5, 2.0, 2.0, 2.5], x, y, method="lin-lin")
    assert got.tolist() == [1.0, 1.0, 5.0, 5.0]


def test_union_sees_a_peak_the_reference_grid_misses():
    ref = PlotData(x=np.array([1.0, 3.0]), y=np.array([1.0, 1.0]))
    cmp = PlotData(x=np.array([1.0, 2.0, 3.0]), y=np.array([1.0, 9.0, 1.0]))
    on_ref = compute_difference(ref, cmp, mode="absolute", interpolation="lin-lin", grid="reference")
    on_union = compute_difference(ref, cmp, mode="absolute", interpolation="lin-lin", grid="union")
    assert np.nanmax(on_ref.difference.y) == 0.0
    assert np.nanmax(on_union.difference.y) == 8.0


def test_a_nan_in_a_series_is_a_hole_not_a_bridge():
    ref = PlotData(x=np.arange(1.0, 6.0), y=np.ones(5))
    y = np.ones(5)
    y[2] = np.nan
    cmp = PlotData(x=np.arange(1.0, 6.0), y=y)
    res = compute_difference(ref, cmp, mode="absolute", interpolation="lin-lin", grid="union")
    assert np.isnan(res.difference.y).tolist() == [False, False, True, False, False]
