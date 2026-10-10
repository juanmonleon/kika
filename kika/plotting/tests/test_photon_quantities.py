"""The photon quantities of kika.plotting: production, multiplicity, gamma energy.

The numbers are kika.nuclear_data.photon_production's; what is tested here is
the plotting contract around them: canonical units, the axis each quantity
labels, provenance that says where a reaction was left out, and that ENDF tapes
and ReactionSuites both answer.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kika.plotting import NotPlottable, PlotBuilder, plottable, supported_quantities
from kika.plotting.quantities import ReconstructionRequired, photon_line_text
from kika.plotting.units import convert, get_quantity

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
N14 = DATA / "micro_n14_photons.endf"
S36 = DATA / "micro_s36_b81_photons.endf"
FE56_ENDF = Path(r"C:\Users\Usuario\.kika\endf_cache\endfb8.1\n\26056.endf")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def s36():
    from kika.endf import read_endf
    return read_endf(str(S36))


@pytest.fixture(scope="module")
def n14():
    from kika.endf import read_endf
    return read_endf(str(N14))


def test_the_three_quantities_are_registered_with_their_units():
    assert get_quantity("photon_production").y_unit == "b"
    assert get_quantity("photon_multiplicity").y_unit == "1"
    assert get_quantity("photon_energy").y_unit == "b*eV"
    assert convert([1.0], "b*MeV", "b*eV").tolist() == [1e6]
    assert get_quantity("photon_energy").axis_label("y", "b*MeV").endswith("(b·MeV)")


def test_an_endf_tape_supplies_them(s36):
    assert {"photon_production", "photon_multiplicity", "photon_energy"} <= set(
        supported_quantities(s36))


def test_a_cascade_line_production_is_sigma_times_its_photons(s36):
    from kika.nuclear_data.photon_production import PhotonProduction

    item = plottable(s36, "photon_production", mt=51, line="cascade:S36_e1:S36")
    data = item.data
    assert data.quantity == "photon_production" and data.y_unit == "b" and data.x_unit == "eV"
    assert data.provenance.reaction == 51
    assert "S36_e1 → S36" in data.label
    suite = s36._kika_photon_suite
    x, y, _ = PhotonProduction(suite).curve("cascade:S36_e1:S36", mt=51)
    assert np.allclose(data.x, x) and np.allclose(data.y, y, equal_nan=True)


def test_multiplicity_is_per_reaction(s36):
    data = plottable(s36, "photon_multiplicity", mt=51).data
    assert data.y_unit == "1"
    assert np.nanmax(data.y) == pytest.approx(1.0, rel=1e-6)


def test_the_gamma_energy_is_its_own_quantity(s36):
    with pytest.raises(NotPlottable, match="photon_energy"):
        plottable(s36, "photon_production", line="energy")
    summed = plottable(s36, "photon_energy").data
    assert summed.y_unit == "b*eV"
    per = plottable(s36, "photon_energy", mt=51, per_reaction=True).data
    assert per.y_unit == "eV"


def test_an_unknown_line_is_not_plottable(n14):
    with pytest.raises(NotPlottable):
        plottable(n14, "photon_production", line="line:1")
    with pytest.raises(NotPlottable):
        plottable(n14, "photon_multiplicity", mt=2)


def test_a_reaction_suite_supplies_them(s36):
    from kika.endf.model_adapter import decodeReactionSuite

    suite, _ = decodeReactionSuite(s36)
    a = plottable(suite, "photon_production", mt=51).data
    b = plottable(s36, "photon_production", mt=51).data
    assert np.allclose(a.y, b.y, equal_nan=True)
    assert a.provenance.format == "gnds"


def test_the_builder_labels_the_axis_from_the_quantity(n14):
    builder = PlotBuilder()
    builder.add_data(plottable(n14, "photon_production", mt=102, line="primary:102:10829100"))
    fig = builder.set_units(x="MeV", y="mb").build()
    ax = fig.axes[0]
    assert ax.get_ylabel().endswith("(mb)") and "Photon production" in ax.get_ylabel()
    assert ax.get_xlabel() == "Incident energy (MeV)"


def test_line_keys_read_as_legends():
    assert photon_line_text("line:846778") == "846.8 keV γ"
    assert photon_line_text("cascade:Fe56_e1:Fe56") == "γ Fe56_e1 → Fe56"
    assert photon_line_text(None) == "all photons"


@pytest.mark.skipif(not FE56_ENDF.exists(), reason=f"{FE56_ENDF} not available")
def test_fe56_background_reactions_are_left_out_and_said_so():
    from kika.endf import read_endf

    endf = read_endf(str(FE56_ENDF))
    data = plottable(endf, "photon_production").data
    assert "MT102 left out below" in data.label
    assert np.isnan(data.y[data.x < 1.0e3]).all()
    with pytest.raises(ReconstructionRequired):
        plottable(endf, "photon_production", reconstructed=True)
    line = plottable(endf, "photon_production", mt=51, line="cascade:Fe56_e1:Fe56").data
    assert np.interp(2.0e6, line.x, line.y) == pytest.approx(0.528, rel=0.01)
    raw = plottable(endf, "photon_production", reconstructed=False).data
    assert raw.provenance.state == "background" and not np.isnan(raw.y).any()
