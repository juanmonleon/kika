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


# ---------------------------------------------------------------------------
# One side averaged: the averages against the other's pointwise curve
# ---------------------------------------------------------------------------

def _pointwise(label, level):
    x = np.geomspace(1.0, 1000.0, 50)
    return PlotData(x=x, y=np.full_like(x, level), label=label, color="C1")


def test_window_against_pointwise_at_the_same_constant_is_zero():
    for ref, cmp in ((_series("ref", 2.0), _pointwise("cmp", 2.0)),
                     (_pointwise("ref", 2.0), _series("cmp", 2.0))):
        _, _, ax_diff = _build(ref, cmp)
        (line,) = _lines_at(ax_diff, CENTRES)
        assert np.allclose(line.get_ydata(), 0.0)
        plt.close("all")


@pytest.mark.parametrize("averaged_is_ref, expected", [(True, 10.0), (False, 100 * (2.0 - 2.2) / 2.2)])
def test_one_sided_relative_divides_by_the_reference(averaged_is_ref, expected):
    if averaged_is_ref:
        ref, cmp = _series("ref", 2.0), _pointwise("cmp", 2.2)
    else:
        ref, cmp = _pointwise("ref", 2.2), _series("cmp", 2.0)
    _, _, ax_diff = _build(ref, cmp)
    (line,) = _lines_at(ax_diff, CENTRES)
    assert np.allclose(line.get_ydata(), expected)


def test_steps_against_pointwise_is_a_curve_on_the_pointwise_grid():
    ref = _series("ref", 2.0, window=False, steps=True)
    _, _, ax_diff = _build(ref, _pointwise("cmp", 2.2))
    lines = [ln for ln in ax_diff.get_lines()
             if np.asarray(ln.get_xdata()).min() == BOUNDS[0]
             and np.asarray(ln.get_xdata()).max() == BOUNDS[1]]
    (steps,) = lines
    x = np.asarray(steps.get_xdata())
    assert steps.get_drawstyle() == "default"
    # Each inner edge twice, the pointwise nodes in between.
    for edge in EDGES[1:-1]:
        assert np.count_nonzero(x == edge) == 2
    assert x.size > len(EDGES) + 2
    assert np.allclose(steps.get_ydata(), 10.0)


def test_the_pointwise_diff_stands_outside_the_averaged_span():
    _, _, ax_diff = _build(_series("ref", 2.0), _pointwise("cmp", 2.2))
    (pointwise,) = [ln for ln in ax_diff.get_lines() if len(ln.get_xdata()) > len(CENTRES)
                    and not np.allclose(ln.get_ydata(), 0.0)]
    x, y = np.asarray(pointwise.get_xdata()), np.asarray(pointwise.get_ydata())
    outside = (x < BOUNDS[0]) | (x > BOUNDS[1])
    assert np.allclose(y[outside], 10.0)
    assert np.isnan(y[(x > BOUNDS[0]) & (x < BOUNDS[1])][1:-1]).all()


def test_a_series_compared_with_its_own_average():
    # The use case behind per-series averaging: duplicate a curve, average one.
    x = np.geomspace(1.0, 1000.0, 400)
    y = 2.0 + 1.0 / (1.0 + ((x - 30.0) / 1.0) ** 2)
    from kika.processing import resonance_window_average
    plain = PlotData(x=x, y=y, label="pointwise", color="C0")
    averaged = PlotData(x=x, y=y.copy(), label="averaged", color="C1")
    values = resonance_window_average(x, y, CENTRES, 0.3, bounds=BOUNDS)
    averaged.metadata["group_average_overlay"] = {
        "bounds_used": BOUNDS, "weighting": "lethargy", "centres": list(CENTRES),
        "values": list(values)}
    _, _, ax_diff = _build(plain, averaged)
    (line,) = _lines_at(ax_diff, CENTRES)
    # The average against the curve it averages, at each centre (lin-lin here).
    at_centres = np.interp(CENTRES, x, y)
    assert np.allclose(line.get_ydata(), 100 * (values - at_centres) / at_centres)


def test_no_average_anywhere_leaves_the_comparison_as_it_was():
    _, ax_main, ax_diff = _build(_pointwise("ref", 2.0), _pointwise("cmp", 2.2))
    assert not _lines_at(ax_diff, CENTRES)
    assert all(np.isfinite(ln.get_ydata()).all() for ln in ax_diff.get_lines()
               if len(ln.get_xdata()) == 50)


# ---------------------------------------------------------------------------
# A plain plot draws the averages too
# ---------------------------------------------------------------------------

def _plain(*series, display="average", shade=False):
    from kika.plotting import PlotBuilder
    builder = PlotBuilder()
    for data in series:
        builder.add_data(data)
    builder.set_scales(log_x=True)
    builder.set_group_average(main_display=display, shade_range=shade)
    return builder.build().axes[0]


def test_a_plain_plot_draws_the_layers_of_an_averaged_series():
    ax = _plain(_series("xs", 2.0, steps=True), _pointwise("other", 3.0), shade=True)
    (window,) = _lines_at(ax, CENTRES)
    (steps,) = _lines_at(ax, EDGES)
    assert window.get_label() == "xs" and steps.get_label() == "xs (group avg)"
    assert steps.get_drawstyle() == "steps-post"
    # Drawn in the series' own color, the pointwise curve masked inside the span.
    assert window.get_color() == "C0"
    (masked,) = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == 50
                 and np.isnan(ln.get_ydata()).any()]
    assert len(ax.patches) == 1
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels.count("xs") == 1 and "other" in labels


def test_a_plain_plot_in_pointwise_mode_draws_no_layer():
    ax = _plain(_series("xs", 2.0, steps=True), display="pointwise")
    assert not _lines_at(ax, CENTRES) and not _lines_at(ax, EDGES)


def test_a_plain_plot_fits_y_to_the_layers():
    data = _series("xs", 2.0)
    data.metadata["group_average_overlay"]["values"][4] = 100.0
    ax = _plain(data)
    assert ax.get_ylim()[1] >= 100.0


def test_an_unknown_display_is_refused():
    from kika.plotting import PlotBuilder
    with pytest.raises(ValueError):
        PlotBuilder().set_group_average(main_display="avg")
    with pytest.raises(ValueError):
        ComparisonBuilder().set_group_average(main_display="avg")


# ---------------------------------------------------------------------------
# The 'ref: <label>' annotation, in a corner of choice
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("only", [False, True])
@pytest.mark.parametrize("loc, xy, ha, va", [
    (None, (0.02, 0.97), "left", "top"),
    ("upper right", (0.98, 0.97), "right", "top"),
    ("lower left", (0.02, 0.03), "left", "bottom"),
    ("lower right", (0.98, 0.03), "right", "bottom"),
])
def test_the_reference_label_sits_in_the_corner_asked_for(only, loc, xy, ha, va):
    builder = ComparisonBuilder(interpolation="lin-lin")
    builder.set_reference(_pointwise("ref", 2.0)).add_comparison(_pointwise("cmp", 2.2))
    builder.set_difference_panel(only=only)
    builder.set_reference_label(loc=loc)
    ax = builder.build().axes[-1]
    (text,) = [t for t in ax.texts if t.get_text() == "ref: ref"]
    assert text.get_position() == pytest.approx(xy)
    assert (text.get_ha(), text.get_va()) == (ha, va)


def test_an_unknown_reference_label_corner_is_refused():
    with pytest.raises(ValueError):
        ComparisonBuilder().set_reference_label(loc="best")
