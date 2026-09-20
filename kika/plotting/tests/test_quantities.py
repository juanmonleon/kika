"""kika.plotting.plottable: one verb, any source, canonical units, honest provenance.

The unit and label tests build their data by hand. The acceptance tests read real
tapes (Fe-56: JEFF-4.0 ENDF and PENDF, ENDF/B-VIII.1 ENDF and ACE) and skip where
those are not reachable, like every other tape-backed test in kika.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from kika.plotting import (
    ComparisonBuilder,
    MixedQuantityWarning,
    NotPlottable,
    PlotBuilder,
    PlotData,
    PlotItem,
    Provenance,
    ReconstructionRequired,
    UncertaintyBand,
    UnitError,
    plottable,
    supported_quantities,
)
from kika.plotting.quantities import auto_label, fold_in_energy, library_from_text
from kika.plotting.units import convert, data_to_units, get_quantity


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------- units

def test_energy_and_area_conversions():
    assert convert([1.0, 2.0], "MeV", "eV").tolist() == [1e6, 2e6]
    assert convert([1.0], "b", "mb").tolist() == [1000.0]
    assert convert([50.0], "%", "1").tolist() == [0.5]
    with pytest.raises(UnitError):
        convert([1.0], "eV", "b")


def test_axis_labels_follow_the_display_unit():
    q = get_quantity("cross_section")
    assert q.axis_label("x") == "Incident energy (eV)"
    assert q.axis_label("x", "MeV") == "Incident energy (MeV)"
    assert get_quantity("angular_distribution").axis_label("y") == r"$f(\mu)$"


def test_data_without_units_is_left_alone():
    data = PlotData(x=[1, 2], y=[3, 4])
    assert data_to_units(data, "MeV", "b") is data


# ---------------------------------------------------------------- labels and provenance

@pytest.mark.parametrize("text,library", [
    ("FE56 - 293.6 K - ENDFB-8.0 with covariances (NJOY 2016.34)", "ENDF/B-VIII.0"),
    ("FE56 - 293.6 K - ENDF/B-VIII.1 GAIA1 (NJOY 2016.77)", "ENDF/B-VIII.1"),
    ("10010 ENDF/B-VII.1 library at 293.6 K", "ENDF/B-VII.1"),
    ("U238 - 600 K - JEFF-3.3 (NJOY 2016.35)", "JEFF-3.3"),
    ("JEFF-4.0 Incident Neutron File", "JEFF-4.0"),
    ("JENDL-5 neutron sublibrary", "JENDL-5"),
    ("kika reconr reconstruction", None),
])
def test_library_names_are_read_from_free_text(text, library):
    assert library_from_text(text) == library


def test_labels_say_what_state_the_numbers_are_in():
    prov = Provenance(evaluation="JEFF-4.0", nuclide="Fe56", reaction=2, state="background")
    assert auto_label("cross_section", prov) == "JEFF-4.0 Fe56 (n,el) · MF3 background"
    ace = Provenance(evaluation="ENDF/B-VIII.1", nuclide="Fe56", reaction=2, state="heated",
                     format="ace", temperature=293.6)
    assert auto_label("cross_section", ace) == "ENDF/B-VIII.1 Fe56 (n,el) · ACE 293.6 K"
    assert auto_label("angular_distribution", ace, energy=2e6).endswith("@ 2 MeV · ACE 293.6 K")
    projected = Provenance(nuclide="Fe56", reaction=2, state="projected", format="ace", temperature=293.6)
    assert auto_label("legendre_coefficient", projected, order=1) == "Fe56 (n,el) L=1 · projected, ACE 293.6 K"


# ---------------------------------------------------------------- PlotBuilder with units

def _xs(x_ev, y_b, label, **kw):
    return PlotItem(PlotData(x=np.asarray(x_ev), y=np.asarray(y_b), label=label,
                             quantity="cross_section", x_unit="eV", y_unit="b", **kw))


def test_builder_converts_every_curve_to_the_display_unit():
    ev = _xs([1e6, 2e6], [1.0, 2.0], "eV curve")
    mev = PlotItem(PlotData(x=[1.0, 2.0], y=[1.0, 2.0], label="MeV curve",
                            quantity="cross_section", x_unit="MeV", y_unit="b"))
    ax = PlotBuilder().add_data(ev).add_data(mev).set_units(x="MeV").build().axes[0]
    for line in ax.get_lines()[:2]:
        assert line.get_xdata().tolist() == [1.0, 2.0]
    assert ax.get_xlabel() == "Incident energy (MeV)"
    assert ax.get_ylabel() == "Cross section (b)"


def test_first_curve_sets_the_unit_when_none_is_chosen():
    mev = PlotItem(PlotData(x=[1.0, 2.0], y=[1.0, 2.0], label="MeV",
                            quantity="cross_section", x_unit="MeV", y_unit="b"))
    ev = _xs([1e6, 2e6], [1000.0, 2000.0], "eV, mb", )
    ev.data.y_unit = "mb"
    ax = PlotBuilder().add_data(mev).add_data(ev).build().axes[0]
    assert ax.get_lines()[1].get_xdata().tolist() == [1.0, 2.0]
    assert ax.get_lines()[1].get_ydata().tolist() == [1.0, 2.0]


def test_absolute_bands_follow_the_conversion_relative_ones_do_not_need_to():
    item = _xs([1e6, 2e6], [10.0, 20.0], "with band")
    item.band = UncertaintyBand(x=[1e6, 2e6], y_lower=[9.0, 18.0], y_upper=[11.0, 22.0])
    ax = PlotBuilder().add_data(item).set_units(x="MeV", y="mb").build().axes[0]
    band = ax.collections[0].get_paths()[0].vertices
    assert band[:, 0].min() == pytest.approx(1.0)
    assert band[:, 1].max() == pytest.approx(22000.0)


def test_mixing_quantities_warns():
    xs = _xs([1, 2], [1, 2], "sigma")
    legendre = PlotItem(PlotData(x=[1, 2], y=[0.1, 0.2], quantity="legendre_coefficient",
                                 x_unit="eV", y_unit="1"))
    with pytest.warns(MixedQuantityWarning):
        PlotBuilder().add_data(xs).add_data(legendre).build()


def test_mixing_frames_warns():
    def ad(frame):
        return PlotItem(PlotData(x=[-1, 1], y=[0.5, 0.5], quantity="angular_distribution",
                                 x_unit="mu", y_unit="1/mu", provenance=Provenance(frame=frame)))
    with pytest.warns(MixedQuantityWarning, match="frames"):
        PlotBuilder().add_data(ad("CM")).add_data(ad("LAB")).build()


def test_legacy_data_keeps_the_old_axis_guess():
    ax = PlotBuilder().add_data(PlotData(x=[1, 10], y=[1, 2])).set_scales(log_x=True).build().axes[0]
    assert ax.get_xlabel() == "Energy (MeV)"


def test_comparison_is_computed_in_the_reference_units():
    ref = _xs([1e6, 2e6, 3e6], [1.0, 2.0, 3.0], "ref")
    other = PlotItem(PlotData(x=[1.0, 2.0, 3.0], y=[1.1, 2.2, 3.3], label="other",
                              quantity="cross_section", x_unit="MeV", y_unit="b"))
    fig = (ComparisonBuilder(interpolation="lin-lin").set_reference(ref).add_comparison(other)
           .set_difference_panel(mode="relative").build())
    diff = fig.axes[-1].get_lines()[0].get_ydata()
    assert np.allclose(diff[np.isfinite(diff)], 10.0)


def test_comparison_takes_the_interpolation_law_from_the_data():
    data = PlotData(x=[1, 2], y=[1, 2], interpolation="lin-lin")
    assert ComparisonBuilder()._infer_interpolation(data) == "lin-lin"


# ---------------------------------------------------------------- registry and errors

def test_an_unknown_source_says_so():
    with pytest.raises(NotPlottable):
        plottable(object(), "cross_section", mt=2)
    with pytest.raises(ValueError, match="Unknown quantity"):
        plottable(object(), "flux")


def test_folding_averages_over_the_kernel():
    # a linear function is its own Gaussian average
    assert fold_in_energy(lambda e: np.array([2.0 * e]), 5.0, 0.5)[0] == pytest.approx(10.0)
    # a quadratic picks up sigma^2
    assert fold_in_energy(lambda e: np.array([e * e]), 5.0, 0.5)[0] == pytest.approx(25.25)


# ---------------------------------------------------------------- EXFOR (synthetic set)

def _exfor():
    from kika.exfor.cross_section import ExforCrossSection

    df = pd.DataFrame({"energy": [1.0, 2.0, 3.0], "cross_section": [3.0, 2.5, 2.0],
                       "error": [0.1, 0.1, 0.2]})
    return ExforCrossSection(entry="22316", subentry="002", quantity="SIG",
                             citation={"authors": ["W.E.Kinney"], "year": 1970},
                             reaction={"target": "Fe56"}, _data=df)


def test_exfor_points_arrive_in_ev_with_error_bars():
    item = plottable(_exfor(), "cross_section")
    assert item.data.x.tolist() == [1e6, 2e6, 3e6]
    assert item.data.provenance.state == "measured"
    assert item.data.label.startswith("Kinney")
    assert item.band.style == "errorbar"
    assert item.band.y_upper.tolist() == pytest.approx([3.1, 2.6, 2.2])


# ---------------------------------------------------------------- tapes: the acceptance tests

@pytest.fixture(scope="module")
def fe56_jeff(fe56_host_tape):
    from kika.endf.read_endf import read_endf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_endf(str(fe56_host_tape), mf_numbers=[1, 2, 3, 4, 33])


@pytest.fixture(scope="module")
def fe56_b81(fe56_b81_tape):
    from kika.endf.read_endf import read_endf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_endf(str(fe56_b81_tape), mf_numbers=[1, 4])


@pytest.fixture(scope="module")
def fe56_ace(fe56_ace_b81_tape):
    from kika.ace.parsers.parse_ace import read_ace
    return read_ace(str(fe56_ace_b81_tape))


@pytest.mark.tape
def test_endf_pendf_ace_and_exfor_overlay_with_no_manual_units(fe56_jeff, fe56_pendf_tape, fe56_ace):
    items = [
        plottable(fe56_jeff, "cross_section", mt=2, reconstructed=False),
        plottable(fe56_pendf_tape, "cross_section", mt=2, evaluation="JEFF-4.0"),
        plottable(fe56_ace, "cross_section", mt=2),
        plottable(_exfor(), "cross_section"),
    ]
    assert all(i.data.x_unit == "eV" and i.data.y_unit == "b" for i in items)
    states = [i.data.provenance.state for i in items]
    assert states == ["background", "reconstructed", "heated", "measured"]
    labels = [i.label for i in items]
    assert "MF3 background" in labels[0] and "reconstructed" in labels[1] and "293.6 K" in labels[2]

    # Above the resonance range all three evaluations must sit on top of each
    # other once they share a unit (the ACE curve used to be 1e6 away).
    at = 5.0e6
    values = [np.interp(at, i.data.x, i.data.y) for i in items[:3]]
    assert max(values) / min(values) < 1.15

    builder = PlotBuilder(style="signature").set_units(x="MeV")
    for item in items:
        builder.add_data(item)
    ax = builder.set_scales(log_x=True, log_y=True).build().axes[0]
    assert ax.get_xlabel() == "Incident energy (MeV)"


@pytest.mark.tape
def test_reconstruction_is_required_only_when_asked(fe56_jeff):
    with pytest.raises(ReconstructionRequired):
        plottable(fe56_jeff, "cross_section", mt=2, reconstructed=True)
    with pytest.raises(ReconstructionRequired):
        plottable(fe56_jeff, "differential_cross_section", mt=2, energy=2e6)


@pytest.mark.tape
def test_endf_pendf_overlay_uses_endf_pendf_when_set(fe56_jeff, fe56_pendf_tape):
    from kika.endf.read_endf import read_endf
    pendf = read_endf(str(fe56_pendf_tape), mf_numbers=[1, 3])
    fe56_jeff.pendf = {2: pendf.mf[3].mt[2]}
    try:
        item = plottable(fe56_jeff, "cross_section", mt=2)
        assert item.data.provenance.state == "reconstructed"
        dsigma = plottable(fe56_jeff, "differential_cross_section", mt=2, energy=2e6)
        assert dsigma.data.y.min() > 0
    finally:
        fe56_jeff.pendf = None


@pytest.mark.tape
def test_mf33_band_is_relative_and_on_the_curve_grid(fe56_jeff):
    item = plottable(fe56_jeff, "cross_section", mt=2, reconstructed=False, uncertainty=True)
    assert item.band is not None and item.band.is_relative()
    assert len(item.band.x) == len(item.data.x)
    assert 0 < np.nanmax(item.band.relative_uncertainty) < 1


@pytest.mark.tape
def test_ace_projected_legendre_moments_match_the_evaluation(fe56_ace, fe56_b81):
    """The ACE file and the ENDF tape are the same ENDF/B-VIII.1 evaluation, so the
    Legendre moments of ACE's tabulated f(mu) must reproduce MF4's coefficients."""
    for order in (1, 2, 3):
        ace = plottable(fe56_ace, "legendre_coefficient", mt=2, order=order)
        endf = plottable(fe56_b81, "legendre_coefficient", mt=2, order=order)
        assert ace.data.provenance.state == "projected"
        for energy in (1e6, 5e6, 1.4e7):
            assert np.interp(energy, ace.data.x, ace.data.y) == pytest.approx(
                np.interp(energy, endf.data.x, endf.data.y), abs=2e-3)


@pytest.mark.tape
def test_angular_distributions_agree_and_share_a_frame(fe56_ace, fe56_b81):
    ace = plottable(fe56_ace, "angular_distribution", mt=2, energy=2e6)
    endf = plottable(fe56_b81, "angular_distribution", mt=2, energy=2e6)
    assert ace.data.provenance.frame == endf.data.provenance.frame == "CM"
    assert np.trapezoid(ace.data.y, ace.data.x) == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(ace.data.y, endf.data.y, rtol=0.05, atol=0.01)


@pytest.mark.tape
def test_ace_differential_cross_section_and_its_folding(fe56_ace):
    plain = plottable(fe56_ace, "differential_cross_section", mt=2, energy=2e6)
    folded = plottable(fe56_ace, "differential_cross_section", mt=2, energy=2e6,
                       resolution=(27.037, 5.0))
    # 2 pi int dsigma/dOmega dmu = sigma
    total = 2 * np.pi * np.trapezoid(plain.data.y, plain.data.x)
    xs = plottable(fe56_ace, "cross_section", mt=2)
    assert total == pytest.approx(np.interp(2e6, xs.data.x, xs.data.y), rel=2e-3)
    assert "folded" in folded.label
    assert not np.allclose(plain.data.y, folded.data.y)


@pytest.mark.tape
def test_supported_quantities(fe56_jeff, fe56_ace):
    assert "relative_uncertainty" in supported_quantities(fe56_jeff)
    assert "relative_uncertainty" not in supported_quantities(fe56_ace)
