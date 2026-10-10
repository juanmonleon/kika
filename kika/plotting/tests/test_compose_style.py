"""Composed styles: one style built from the parts of the registered ones."""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kika.plotting import (
    CovarianceHeatmapData,
    HeatmapBuilder,
    PlotBuilder,
    PlotData,
    compose_style,
    get_style,
    register_style,
    style_names,
    style_parts,
)
from kika.plotting import styles as styles_module


@pytest.fixture(autouse=True)
def _clean():
    names = list(styles_module._REGISTRY)
    yield
    plt.close("all")
    # A test that registered a composed style must not leak it into the others.
    for key in list(styles_module._REGISTRY):
        if key not in names:
            del styles_module._REGISTRY[key]


def _corr():
    return CovarianceHeatmapData(matrix_data=np.eye(3) * 2 - 1, matrix_type="corr")


def _mesh_cmap(fig):
    return next(c for ax in fig.axes for c in ax.collections + ax.images).get_cmap()


# ---------------------------------------------------------------- composition

def test_nothing_chosen_keeps_the_base():
    base = get_style("signature")
    style = compose_style("signature")
    assert style.name == "custom" and style.label == "Custom"
    assert style.palette == base.palette
    assert style.sequential == base.sequential and style.diverging == base.diverging
    assert dict(style.rc) == dict(base.rc)
    assert style.font_family == base.font_family


def test_composed_style_is_not_registered():
    compose_style("signature", palette="isotope")
    assert "custom" not in style_names()


def test_parts_come_from_their_styles():
    style = compose_style("mineral", palette="isotope", sequential="journal",
                          diverging="signature", font="stix", grid=False)
    assert style.palette == get_style("isotope").palette
    assert style.sequential == get_style("journal").sequential
    assert style.diverging == get_style("signature").diverging
    assert style.rc["axes.facecolor"] == get_style("mineral").rc["axes.facecolor"]
    assert style.font_family == "serif"
    assert style.rc["font.serif"][0] == "STIX Two Text"
    assert style.rc["mathtext.fontset"] == "stix"
    assert style.rc["axes.grid"] is False
    assert style.to_dict()["axesGrid"] is False


def test_extra_palettes_and_explicit_colours():
    assert compose_style("classic", palette="okabe-ito").palette[0] == "#E69F00"
    assert len(compose_style("classic", palette="tab10").palette) == 10
    assert compose_style("classic", palette=["red", "#00ff00"]).palette == ("#ff0000", "#00ff00")


def test_matplotlib_colormaps_stay_names():
    style = compose_style("signature", sequential="cividis", diverging="RdBu_r")
    assert style.sequential == "cividis" and style.diverging == "RdBu_r"
    assert style.cmap_name("diverging") == "RdBu_r"


def test_a_non_stix_font_drops_the_base_stix_maths():
    style = compose_style("journal", font="sans")
    assert style.font_family == "sans-serif"
    assert "mathtext.fontset" not in style.rc
    assert style.rc_params()["font.family"] == "sans-serif"


def test_dark_base_carries_its_light_variant():
    for base, light in (("classic-dark", "classic"), ("signature-dark", "signature")):
        style = compose_style(base, name="custom-dark", label="Custom dark", palette="isotope")
        assert style.dark is True and style.light_variant == light


@pytest.mark.parametrize("kwargs", [
    {"base": "nope"},
    {"base": "classic", "palette": "nope"},
    {"base": "classic", "palette": []},
    {"base": "classic", "palette": ["not-a-colour"]},
    {"base": "classic", "sequential": "nope"},
    {"base": "classic", "diverging": "nope"},
    {"base": "classic", "font": "comic"},
    {"base": "classic", "name": "signature"},
    {"base": "classic", "name": "dark"},
])
def test_unknown_parts_raise(kwargs):
    base = kwargs.pop("base")
    with pytest.raises(ValueError):
        compose_style(base, **kwargs)


# ---------------------------------------------------------------- rendering

def test_to_dict_matches_a_registered_style():
    composed = compose_style("signature-dark", name="custom-dark", palette="mineral")
    data = json.loads(json.dumps(composed.to_dict()))
    assert set(data) == set(get_style("signature-dark").to_dict())
    assert data["name"] == "custom-dark" and data["dark"] is True
    assert data["palette"] == list(get_style("mineral").palette)
    assert len(data["sequential"]) == 11


def test_composed_cmaps_resolve_in_matplotlib_and_recompose_replaces_them():
    first = compose_style("classic", diverging="signature")
    assert first.cmap_name("diverging") == "kika-custom-diverging"
    assert mpl.colors.to_hex(mpl.colormaps["kika-custom-diverging"](0.0)) == \
        mpl.colors.to_hex(get_style("signature").cmap("diverging")(0.0))
    compose_style("classic", diverging="isotope")
    assert mpl.colors.to_hex(mpl.colormaps["kika-custom-diverging"](0.0)) == \
        mpl.colors.to_hex(get_style("isotope").cmap("diverging")(0.0))


def test_heatmap_and_line_plot_render_with_a_composed_style():
    style = compose_style("signature", diverging="isotope", font="humanist")
    fig = HeatmapBuilder(style=style).add_heatmap(_corr(), show_uncertainties=False).build()
    assert _mesh_cmap(fig).name.startswith("kika-custom-diverging")

    x = np.logspace(0, 6, 20)
    fig = PlotBuilder(style=style).add_data(PlotData(x=x, y=1 / x, label="a")).build()
    line = fig.axes[0].get_lines()[0]
    assert mpl.colors.to_hex(line.get_color()) == style.palette[0]


def test_registered_with_replace_works_by_name_and_can_be_recomposed():
    register_style(compose_style("signature", diverging="isotope"), replace=True)
    assert get_style("custom").diverging == get_style("isotope").diverging
    fig = HeatmapBuilder(style="custom").add_heatmap(_corr(), show_uncertainties=False).build()
    assert _mesh_cmap(fig).name.startswith("kika-custom-diverging")

    # The API recomposes on every change: same name, new parts.
    register_style(compose_style("classic", diverging="RdBu_r"), replace=True)
    assert get_style("custom").diverging == "RdBu_r"
    fig = HeatmapBuilder(style="custom").add_heatmap(_corr(), show_uncertainties=False).build()
    assert _mesh_cmap(fig).name.startswith("RdBu_r")


# ---------------------------------------------------------------- parts for the UI

def test_style_parts_is_json_and_complete():
    parts = json.loads(json.dumps(style_parts()))
    palettes = {p["name"] for p in parts["palettes"]}
    assert palettes == set(style_names()) | {"okabe-ito", "tab10"}

    seq = {c["name"] for c in parts["colormaps"] if c["kind"] == "sequential"}
    div = {c["name"] for c in parts["colormaps"] if c["kind"] == "diverging"}
    assert seq == set(style_names()) | {"viridis", "cividis", "magma"}
    assert div == set(style_names()) | {"RdBu_r", "coolwarm"}
    assert all(len(c["stops"]) == 11 for c in parts["colormaps"])

    fonts = {f["name"]: f for f in parts["fonts"]}
    assert set(fonts) == {"sans", "humanist", "serif", "stix", "mono"}
    assert {f["family"] for f in fonts.values()} == {"sans-serif", "serif", "monospace"}


def test_every_part_offered_is_accepted():
    parts = style_parts()
    for p in parts["palettes"]:
        compose_style("classic", palette=p["name"])
    for c in parts["colormaps"]:
        compose_style("classic", **{c["kind"]: c["name"]})
    for f in parts["fonts"]:
        compose_style("classic", font=f["name"])


def test_a_registered_composed_style_is_not_offered_as_a_part():
    register_style(compose_style("signature"), replace=True)
    assert "custom" not in {p["name"] for p in style_parts()["palettes"]}
