"""The style registry, and the promise that rendering never touches global state."""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kika.plotting import (
    ComparisonBuilder,
    CovarianceHeatmapData,
    HeatmapBuilder,
    PlotBuilder,
    PlotData,
    Style,
    get_style,
    list_styles,
    register_style,
    style_names,
)
from kika.plotting.styles import CLASSIC_PALETTE


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _line(label="a", scale=1.0):
    x = np.logspace(0, 6, 50)
    return PlotData(x=x, y=scale / np.sqrt(x), label=label)


# ---------------------------------------------------------------- registry

def test_registry_has_the_documented_styles_in_display_order():
    assert style_names() == [
        "classic", "signature", "journal", "isotope", "mineral", "classic-dark", "signature-dark",
    ]


@pytest.mark.parametrize("alias,name", [
    ("light", "classic"), ("dark", "classic-dark"), ("LIGHT", "classic"),
    ("default", "classic"), ("publication", "classic"), (None, "classic"),
])
def test_aliases(alias, name):
    assert get_style(alias).name == name


def test_unknown_style_lists_the_alternatives():
    with pytest.raises(ValueError, match="signature"):
        get_style("does-not-exist")
    with pytest.raises(ValueError):
        PlotBuilder(style="does-not-exist")


def test_a_style_instance_passes_through():
    style = get_style("journal")
    assert get_style(style) is style


def test_register_refuses_duplicates_and_aliases():
    with pytest.raises(ValueError):
        register_style(get_style("classic"))
    with pytest.raises(ValueError):
        register_style(Style(name="light", label="x", description="x",
                             palette=("#000000",), sequential="viridis", diverging="RdBu"))


@pytest.mark.parametrize("style", list_styles(), ids=lambda s: s.name)
def test_to_dict_is_json_and_complete(style):
    data = json.loads(json.dumps(style.to_dict()))
    assert data["name"] == style.name
    assert data["palette"] == list(style.palette)
    assert len(data["sequential"]) == 11 and len(data["diverging"]) == 11
    assert all(c.startswith("#") for c in data["sequential"] + data["diverging"])
    assert data["dark"] is style.dark


@pytest.mark.parametrize("style", list_styles(), ids=lambda s: s.name)
def test_colour_maps_resolve_by_name(style):
    for kind in ("sequential", "diverging"):
        assert mpl.colormaps[style.cmap_name(kind)].N > 0


def test_dark_styles_name_a_light_sibling():
    for style in list_styles():
        if style.dark:
            assert not get_style(style.light_variant).dark


# ---------------------------------------------------------------- classic is the old look

def test_classic_keeps_the_old_palette_and_font():
    style = get_style("classic")
    assert style.palette == CLASSIC_PALETTE
    rc = style.rc_params()
    assert rc["font.family"] == "serif"
    assert rc["grid.linestyle"] == "--" and rc["grid.alpha"] == 0.3
    assert rc["legend.edgecolor"] == "black" and rc["legend.fancybox"] is False


def test_classic_line_plot_draws_like_before():
    fig = (PlotBuilder(style="light").add_data(_line("a")).add_data(_line("b", 2))
           .set_scales(log_x=True, log_y=True).build())
    ax = fig.axes[0]
    colors = [mpl.colors.to_hex(l.get_color()) for l in ax.get_lines()[:2]]
    assert colors == [c.lower() for c in CLASSIC_PALETTE[:2]]
    legend = ax.get_legend()
    assert legend.get_frame_on()
    assert mpl.colors.to_hex(legend.get_frame().get_edgecolor()) == "#000000"
    gridline = ax.xaxis.get_gridlines()[0]
    assert gridline.get_visible() and gridline.get_linestyle() == "--"
    assert ax.spines["top"].get_visible()


def test_classic_dark_keeps_the_black_background():
    fig = PlotBuilder(style="dark").add_data(_line()).build()
    assert mpl.colors.to_hex(fig.axes[0].get_facecolor()) == "#000000"


# ---------------------------------------------------------------- new styles render their look

def test_signature_open_axes_and_palette():
    fig = PlotBuilder(style="signature").add_data(_line("a")).add_data(_line("b", 2)).build()
    ax = fig.axes[0]
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    colors = [mpl.colors.to_hex(l.get_color()) for l in ax.get_lines()[:2]]
    assert colors == list(get_style("signature").palette[:2])
    assert not ax.get_legend().get_frame_on()


def test_journal_has_no_grid_unless_asked():
    ax = PlotBuilder(style="journal").add_data(_line()).build().axes[0]
    assert not any(g.get_visible() for g in ax.xaxis.get_gridlines())
    ax = PlotBuilder(style="journal").add_data(_line()).set_grid(True).build().axes[0]
    assert any(g.get_visible() for g in ax.xaxis.get_gridlines())


def test_explicit_font_family_overrides_the_style():
    fig = PlotBuilder(style="signature", font_family="serif").add_data(_line()).set_labels(x_label="E").build()
    assert fig.axes[0].xaxis.label.get_fontfamily() == ["serif"]


@pytest.mark.parametrize("style", list_styles(), ids=lambda s: s.name)
def test_every_style_builds_line_comparison_and_heatmap(style):
    PlotBuilder(style=style).add_data(_line("a")).add_data(_line("b", 2)).set_labels(title="t").build()
    (ComparisonBuilder(style=style.name, interpolation="log-log")
     .set_reference(_line("ref")).add_comparison(_line("cmp", 1.1))
     .set_difference_panel().build())
    corr = np.eye(4) * 0.5 + 0.5
    heat = CovarianceHeatmapData(matrix_data=corr, matrix_type="corr")
    fig = HeatmapBuilder(style=style).add_heatmap(heat, show_uncertainties=False).build()
    assert fig.axes


# ---------------------------------------------------------------- no global side effects

def test_building_does_not_touch_global_state():
    before = plt.figure()
    mpl.rcParams["lines.linewidth"] = 7.0
    try:
        for style in ("classic", "signature-dark", "journal"):
            PlotBuilder(style=style).add_data(_line()).build()
            (ComparisonBuilder(style=style, interpolation="log-log")
             .set_reference(_line("ref")).add_comparison(_line("cmp", 1.1))
             .set_difference_panel().build())
            heat = CovarianceHeatmapData(matrix_data=np.eye(3), matrix_type="corr")
            HeatmapBuilder(style=style).add_heatmap(heat, show_uncertainties=False).build()
        assert plt.fignum_exists(before.number), "a builder closed the caller's figure"
        assert mpl.rcParams["lines.linewidth"] == 7.0, "a builder reset the caller's rcParams"
    finally:
        mpl.rcParams["lines.linewidth"] = mpl.rcParamsDefault["lines.linewidth"]


def test_style_context_restores_rcparams():
    before = dict(mpl.rcParams)
    with get_style("mineral").context():
        assert mpl.rcParams["axes.facecolor"] == "#fdfcf9"
    assert mpl.rcParams["axes.facecolor"] == before["axes.facecolor"]


# ---------------------------------------------------------------- heatmap colour maps

def test_heatmap_colour_map_follows_the_style_only_when_nobody_chose_one():
    corr = CovarianceHeatmapData(matrix_data=np.eye(3) * 2 - 1, matrix_type="corr")
    assert corr.metadata["auto_cmap"] is True

    def mesh_cmap(fig):
        return next(c for ax in fig.axes for c in ax.collections + ax.images).get_cmap()

    classic = HeatmapBuilder(style="classic").add_heatmap(corr, show_uncertainties=False).build()
    assert mesh_cmap(classic).name.startswith("RdYlGn")

    signature = HeatmapBuilder(style="signature").add_heatmap(corr, show_uncertainties=False).build()
    assert mesh_cmap(signature).name.startswith("kika-signature-diverging")

    chosen = HeatmapBuilder(style="signature").add_heatmap(corr, show_uncertainties=False, cmap="RdBu_r").build()
    assert mesh_cmap(chosen).name.startswith("RdBu_r")

    # A dark style keeps its own map and its own page. Its diverging map is
    # centred on the page colour, so zero correlation reads as background
    # instead of as the brightest band in the figure.
    dark = HeatmapBuilder(style="signature-dark").add_heatmap(corr, show_uncertainties=False).build()
    assert mesh_cmap(dark).name.startswith("kika-signature-dark-diverging")
    assert dark.get_facecolor()[:3] == mpl.colors.to_rgb("#07151f")
    # The heatmap axes itself is painted the masked-cell colour (what shows
    # through where there is no data): the page, nudged towards the ink.
    assert dark.axes[0].get_facecolor()[:3] == mpl.colors.to_rgb("#1d2f3a")

    light = HeatmapBuilder(style="signature").add_heatmap(corr, show_uncertainties=False).build()
    assert light.get_facecolor()[:3] == mpl.colors.to_rgb("white")


def test_vector_text_is_embedded_truetype(tmp_path):
    fig = PlotBuilder(style="signature").add_data(_line()).set_labels(title="Embedded").build()
    with get_style("signature").context():
        fig.savefig(tmp_path / "f.pdf")
    content = (tmp_path / "f.pdf").read_bytes()
    assert b"/FontFile2" in content and b"/Subtype /Type3" not in content


# ---------------------------------------------------------------- energy axis ticks

def _xtick_labels(x, xlim):
    fig = (PlotBuilder().add_data(PlotData(x=np.asarray(x), y=np.ones(len(x))))
           .set_scales(log_x=True).set_limits(x_lim=xlim).build())
    ax = fig.axes[0]
    fig.canvas.draw()
    lo, hi = ax.get_xlim()
    return [t.get_text() for t, pos in zip(ax.get_xticklabels(), ax.get_xticks())
            if t.get_text() and lo <= pos <= hi]


def test_a_wide_energy_axis_is_labelled_by_decades():
    labels = _xtick_labels(np.logspace(2, 7.3, 50), (1e2, 2e7))
    assert all("10^" in l or "$" in l for l in labels), labels
    assert 4 <= len(labels) <= 8, labels  # 5.3 decades: one per decade, no more


def test_a_narrow_energy_axis_keeps_numbers_inside_the_decade():
    labels = _xtick_labels(np.logspace(4.3, 5, 50), (2e4, 1e5))
    assert len(labels) <= 5, labels
    assert all("$" not in l for l in labels), labels


def test_ticks_survive_an_empty_or_invalid_axis():
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    from kika.plotting.styles import format_energy_axis_ticks
    format_energy_axis_ticks(ax)  # must not raise on the default (0.1, 1) limits
    plt.close(fig)


# ---------------------------------------------------------------- minor grid

def _subdivision(set_grid=True, **grid_kwargs):
    """(marks, lines) per axis of a wide log-energy figure: X first, then Y."""
    x = np.logspace(-5, 7, 400)
    builder = (PlotBuilder(style="signature-dark")
               .add_data(PlotData(x=x, y=1.0 / np.sqrt(x), x_unit="eV", y_unit="b"))
               .set_scales(log_x=True, log_y=True))
    if set_grid:
        builder.set_grid(grid=True, alpha=0.3, minor_alpha=0.15, **grid_kwargs)
    fig = builder.build()
    ax = fig.axes[0]
    counts = tuple(
        (sum(1 for t in axis.get_minor_ticks() if t.tick1line.get_visible()),
         sum(1 for t in axis.get_minor_ticks() if t.gridline.get_visible()))
        for axis in (ax.xaxis, ax.yaxis)
    )
    plt.close(fig)
    return counts


def test_the_minor_grid_reaches_a_wide_log_axis():
    """A twelve-decade energy axis must gain minor gridlines, not lose its ticks.

    `Axes.minorticks_on()` used to be called here. On a log axis it asks for
    `subs='auto'`, which over this many decades lands on the major ticks and is
    filtered out — so turning the minor grid on wiped the 120 minor ticks that
    `format_energy_axis_ticks` had put on X and drew the grid only on Y, the
    exact opposite of the request.
    """
    (x_marks, x_lines), (y_marks, y_lines) = _subdivision(show_minor=True)
    assert x_lines > 50, f"the wide X axis got {x_lines} minor gridlines"
    assert x_lines == x_marks
    assert y_lines > 0 and y_lines == y_marks


def test_leaving_the_flag_unset_changes_nothing():
    """`None` is the default and must draw exactly what the style and scale do."""
    untouched = _subdivision(set_grid=False)
    assert untouched == _subdivision()          # set_grid(grid=True), minor unset
    (x_marks, x_lines), (_, y_lines) = untouched
    assert x_marks > 50 and x_lines == 0 and y_lines == 0


def test_switching_an_axis_off_strips_its_minor_ticks_too():
    """Off means off. A log axis arrives subdivided, so leaving the marks behind
    is what made the setting look like it did nothing."""
    assert _subdivision(show_minor=False) == ((0, 0), (0, 0))


def test_a_per_axis_flag_overrides_the_shorthand():
    (x, y) = _subdivision(show_minor=True, show_minor_x=False)
    assert x == (0, 0)
    assert y[1] > 0
