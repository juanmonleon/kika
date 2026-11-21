"""The energy axis names only a unit the data carry; overlays and styles fail out loud."""

import dataclasses
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kika.plotting import ComparisonBuilder, PlotBuilder, PlotData, get_style, register_style
from kika.plotting import comparison as comparison_module
from kika.plotting.units import QUANTITIES

X = np.logspace(0, 6, 20)
Y = 1.0 / np.sqrt(X)


def _labels(figure):
    try:
        return [axis.get_xlabel() for axis in figure.axes]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("unit", ["eV", "MeV"])
def test_a_curve_without_quantity_labels_the_energy_axis_with_its_unit(unit):
    figure = (PlotBuilder().add_data(PlotData(x=X, y=Y, label="c", x_unit=unit))
              .set_scales(log_x=True).build())

    assert _labels(figure) == [f"Energy ({unit})"]


def test_a_curve_without_unit_claims_none_on_the_energy_axis():
    figure = PlotBuilder().add_data(PlotData(x=X, y=Y, label="c")).set_scales(log_x=True).build()

    assert _labels(figure) == ["Energy"]


@pytest.mark.parametrize("unit", ["eV", "MeV"])
def test_the_difference_panel_names_the_reference_energy_unit(unit):
    reference = PlotData(x=X, y=Y, label="ref", quantity="cross_section", x_unit=unit, y_unit="b")
    other = PlotData(x=X, y=1.1 * Y, label="cmp", quantity="cross_section", x_unit=unit, y_unit="b")
    figure = (ComparisonBuilder().set_reference(reference).add_comparison(other)
              .set_difference_panel().set_scales(log_x=True).build())

    assert _labels(figure)[-1] == QUANTITIES["cross_section"].axis_label("x", unit)


def test_an_empty_x_label_stays_empty_on_the_difference_panel():
    reference = PlotData(x=X, y=Y, label="ref", quantity="cross_section", x_unit="eV", y_unit="b")
    figure = (ComparisonBuilder().set_reference(reference)
              .add_comparison(PlotData(x=X, y=1.1 * Y, label="cmp", quantity="cross_section",
                                       x_unit="eV", y_unit="b"))
              .set_difference_panel().set_scales(log_x=True).set_labels(x_label="").build())

    assert _labels(figure)[-1] == ""


def _comparison_with_overlay(overlay):
    reference = PlotData(x=X, y=Y, label="ref")
    return (ComparisonBuilder().set_reference(reference)
            .add_comparison(PlotData(x=X, y=1.1 * Y, label="cmp"))
            .add_scatter_overlay(overlay).set_difference_panel().set_scales(log_x=True))


def test_an_overlay_left_out_of_the_difference_panel_says_why():
    outside = PlotData(x=np.array([1.0e8, 2.0e8]), y=np.array([1.0, 1.0]), label="EXFOR")

    with pytest.warns(UserWarning, match="EXFOR"):
        plt.close(_comparison_with_overlay(outside).build())


def test_an_unexpected_error_in_an_overlay_difference_is_not_swallowed(monkeypatch):
    real = comparison_module.compute_difference

    def broken(*args, **kwargs):
        if kwargs.get("grid") == "comparison":
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(comparison_module, "compute_difference", broken)
    inside = PlotData(x=X[2:5], y=Y[2:5], label="EXFOR")

    with pytest.raises(RuntimeError, match="boom"):
        _comparison_with_overlay(inside).build()


def test_a_style_cannot_take_the_name_of_an_alias_even_with_replace():
    alias_named = dataclasses.replace(get_style("classic"), name="light")

    with pytest.raises(ValueError, match="alias"):
        register_style(alias_named, replace=True)
    assert get_style("light").name == "classic"
