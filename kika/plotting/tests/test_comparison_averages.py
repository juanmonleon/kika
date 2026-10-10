"""The averaged layers of a comparison: window curve, group steps, or both."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kika.plotting.comparison import ComparisonBuilder
from kika.plotting.plot_data import PlotData

BOUNDS = [10.0, 100.0]
CENTRES = np.geomspace(10.0, 100.0, 9)
EDGES = np.geomspace(10.0, 100.0, 4)


def _series(label, level, *, window=True, steps=False, centres=CENTRES):
    x = np.geomspace(1.0, 1000.0, 50)
    overlay = {"bounds_used": BOUNDS, "weighting": "lethargy"}
    if window:
        overlay["centres"] = list(centres)
        overlay["values"] = [level] * len(centres)
    if steps:
        overlay["edges"] = list(EDGES)
        overlay["xs"] = [level] * (len(EDGES) - 1)
    data = PlotData(x=x, y=np.full_like(x, level), label=label, color="C0")
    data.metadata["group_average_overlay"] = overlay
    return data


def _build(ref, cmp, main_display="average", shade=False):
    builder = ComparisonBuilder(interpolation="lin-lin")
    builder.set_reference(ref).add_comparison(cmp)
    builder.set_scales(log_x=True, log_y=False)
    builder.set_difference_panel(mode="relative")
    builder.set_group_average(main_display=main_display, shade_range=shade)
    fig = builder.build()
    return fig, fig.axes[0], fig.axes[1]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _lines_at(ax, x):
    """The lines on ``ax`` whose x data is exactly ``x``."""
    return [ln for ln in ax.get_lines()
            if len(ln.get_xdata()) == len(x) and np.allclose(ln.get_xdata(), x)]


def test_window_diff_is_taken_centre_by_centre():
    _, _, ax_diff = _build(_series("ref", 2.0), _series("cmp", 2.2))
    (line,) = _lines_at(ax_diff, CENTRES)
    assert np.allclose(line.get_ydata(), 10.0)
    assert line.get_drawstyle() == "default"


def test_both_layers_diff_and_steps_are_dashed_beside_the_window():
    _, ax_main, ax_diff = _build(
        _series("ref", 2.0, steps=True), _series("cmp", 2.2, steps=True),
    )
    (window,) = _lines_at(ax_diff, CENTRES)
    (steps,) = _lines_at(ax_diff, EDGES)
    assert window.get_linestyle() == "-"
    assert steps.get_linestyle() == "--" and steps.get_drawstyle() == "steps-post"
    assert np.allclose(steps.get_ydata(), 10.0)
    labels = [ln.get_label() for ln in ax_main.get_lines()]
    # 'average': the first layer carries the series' own label.
    assert "ref" in labels and "ref (group avg)" in labels


def test_a_layer_on_different_centres_is_not_diffed():
    _, _, ax_diff = _build(
        _series("ref", 2.0), _series("cmp", 2.2, centres=CENTRES * 1.01),
    )
    assert not _lines_at(ax_diff, CENTRES)


def test_average_masks_the_pointwise_curve_inside_the_range():
    _, ax_main, _ = _build(_series("ref", 2.0), _series("cmp", 2.2))
    pointwise = [ln for ln in ax_main.get_lines() if len(ln.get_xdata()) == 50]
    for line in pointwise:
        x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        inside = (x > BOUNDS[0]) & (x < BOUNDS[1])
        assert np.isnan(y[inside][1:-1]).all()
        assert np.isfinite(y[~inside]).all()


def test_both_fades_the_pointwise_curve_and_the_diff_follows_the_averages():
    _, ax_main, ax_diff = _build(_series("ref", 2.0), _series("cmp", 2.2), main_display="both")
    pointwise = [ln for ln in ax_main.get_lines() if len(ln.get_xdata()) == 50]
    assert pointwise and all(ln.get_alpha() == pytest.approx(0.35) for ln in pointwise)
    assert all(np.isfinite(ln.get_ydata()).all() for ln in pointwise)
    # Through resonances a faded pointwise diff is a solid band, so the diff
    # panel masks it inside the range in this mode too.
    (pointwise_diff,) = [ln for ln in ax_diff.get_lines() if len(ln.get_xdata()) > len(CENTRES)]
    x, y = np.asarray(pointwise_diff.get_xdata()), np.asarray(pointwise_diff.get_ydata())
    assert np.isnan(y[(x > BOUNDS[0]) & (x < BOUNDS[1])][1:-1]).all()


def test_the_y_axis_covers_the_averages_when_the_pointwise_curve_is_masked():
    ref, cmp = _series("ref", 2.0), _series("cmp", 2.2)
    # A peak inside the range that only the averages show once it is masked.
    for data, level in ((ref, 2.0), (cmp, 2.2)):
        data.metadata["group_average_overlay"]["values"][4] = 50.0 * level
    _, ax_main, _ = _build(ref, cmp, main_display="average")
    assert ax_main.get_ylim()[1] >= 110.0


@pytest.mark.parametrize("shade, expected", [(False, 0), (True, 1)])
def test_shade_range_tints_the_range_on_each_panel(shade, expected):
    _, ax_main, ax_diff = _build(_series("ref", 2.0), _series("cmp", 2.2), shade=shade)
    assert len(ax_main.patches) == expected
    assert len(ax_diff.patches) == expected


def test_the_primary_layer_carries_the_label_and_the_bridge():
    ref, cmp = _series("ref", 2.0, steps=True), _series("cmp", 2.2, steps=True)
    for data in (ref, cmp):
        data.metadata["group_average_overlay"]["primary"] = "steps"
    _, ax_main, _ = _build(ref, cmp, main_display="average")
    (steps,) = [ln for ln in _lines_at(ax_main, EDGES) if ln.get_color() == "C0"][:1]
    assert steps.get_label() == "ref"
    labels = [ln.get_label() for ln in ax_main.get_lines()]
    assert "ref (window avg)" in labels
