"""Tests for reading a SINBAD benchmark in the v0.3 XML format.

The fixture is ``data/mini/``: a benchmark and one calculations file, small
enough to read in one screen and complete enough to exercise every structure
the format has -- both table bodies, a blank cell, a correction applied and one
declared and not applied, a correction whose size is not reported, the three
correlation scopes, a linked grid across files, materials, geometry and a
factor chain. Its header says which numbers were chosen to make an assertion
exact.

What is asserted here is mostly **not** that the reader returns something, but
that it returns the thing the specification says. The two places worth naming:

* the covariance is the sum of the shared components, so every off-diagonal
  entry is a number this file can state in closed form, and does;
* a comparison's C/E divides by the denominator *in the declared convention*,
  which for the fixture means times 0.98. A reader that forgets it is out by
  2 %, and ``test_comparison_recompute_matches_published`` is what notices.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import kika.sinbad as sinbad
from kika.sinbad.exceptions import (
    AmbiguousLabelError,
    ContentTypeError,
    LabelNotFoundError,
    SinbadError,
    SinbadFormatError,
)

DATA = Path(__file__).parent / "data" / "mini"
BENCHMARK = DATA / "mini.xml"


@pytest.fixture
def b():
    return sinbad.read(BENCHMARK)


# -- opening ---------------------------------------------------------------


def test_read_a_file_and_a_directory_are_the_same(b):
    from_directory = sinbad.read(DATA)
    assert from_directory.id == b.id
    assert from_directory.data.labels == b.data.labels


def test_a_calculations_file_is_not_a_benchmark():
    with pytest.raises(SinbadFormatError, match="calculations file"):
        sinbad.read(DATA / "calculations" / "lab-code-1.0.xml")


def test_not_a_sinbad_file(tmp_path):
    path = tmp_path / "other.xml"
    path.write_text("<something/>")
    with pytest.raises(SinbadFormatError, match="root <something>"):
        sinbad.read(path)


def test_calculations_are_opened_with_the_benchmark(b):
    assert b.calculations.labels == ["lab-code-1.0"]
    assert sinbad.read(BENCHMARK, calculations=False).calculations.labels == []


# -- the entry blocks ------------------------------------------------------


def test_identity(b):
    assert b.short_code == "mini"
    assert b.id == "TST-ATN-BLK-STR-SUR-001-R"
    assert b.title == "Mini fixture benchmark"
    assert b.format_version == "0.3"
    assert b.identification.domain == "TST"
    assert b.identification.related[0].short_code == "mini2"


def test_documentation(b):
    assert [a.name for a in b.documentation.authors] == ["A. Tester"]
    assert b.documentation.authors[0].affiliations == ("Nowhere",)
    assert b.documentation.contributors[0].contributor_type == "Other"
    assert b.year == 1990
    assert b.documentation.bibitem("1").text.startswith("A. Tester")


def test_status_issues_and_absences(b):
    assert b.status.availability == "unrestricted"
    assert b.status.quality.rating == "fixture"
    assert b.status.quality.reservations == ("everything",)
    assert [i.label for i in b.issues] == ["F1"]
    assert b.issues[0].about == ("reactionRate-Al27",)
    assert len(b.absences) == 2
    assert b.absences[0].kind == "not stated"


def test_detectors_and_positions(b):
    assert b.detectors.labels == ["det-Al27", "det-Au197"]
    gold = b.detectors["det-Au197"]
    assert gold.target == "Au197"
    assert gold.dimensions["diameter"].value == 12.7
    assert gold.dimensions["mass"].min == 0.12  # a range, not a double
    assert gold.cover.material == "Cd"
    assert gold.cover.dimensions["thickness"].unit == "inch"
    assert b.positions["A3"].shield_thickness == 10.0
    assert b.positions["A3"].layer == "gap-A3"


def test_definitions_resolve_from_a_data_object(b):
    obj = b["reactionRate-Al27"]
    assert obj.normalisation.label == "power30kW"
    assert obj.normalisation.value.value == 30
    assert obj.normalisation.value.unit == "kW"
    assert obj.convention.basis == "includes the background"
    assert [d.label for d in obj.detectors] == ["det-Al27"]


# -- tables ----------------------------------------------------------------


def test_table_with_td_rows_and_a_blank_cell(b):
    table = b["reactionRate-Al27"].table
    assert table.nrows == 4
    assert table.names == [
        "position", "shieldThickness", "reactionRate", "countingStatistics", "totalSd",
    ]
    assert list(table["position"]) == ["A1", "A2", "A3", "A4"]
    assert table["reactionRate"][0] == pytest.approx(1.0e-20)
    assert np.isnan(table["reactionRate"][3])  # the blank cell, not a zero
    assert np.isnan(table["totalSd"][3])


def test_table_with_a_whitespace_body(b):
    table = b.calculations["lab-code-1.0"]["selfShieldingFactors"].table
    assert table.nrows == 3
    assert table["shieldThickness"][2] == 10.0
    assert table["factor"][2] == pytest.approx(0.980)


def test_column_roles(b):
    table = b["reactionRate-Al27"].table
    assert [c.name for c in table.independent] == ["position", "shieldThickness"]
    assert [c.name for c in table.value_columns] == ["reactionRate"]
    assert table.column("totalSd").kind == "total"
    assert table.column("position").is_text
    assert table.units()["reactionRate"] == "1/s"


def test_a_table_with_several_value_columns_refuses_to_guess(b):
    calculated = b.calculations["lab-code-1.0"]["calculated-Al27"].table
    with pytest.raises(ContentTypeError, match="2 value columns"):
        _ = calculated.values


def test_asking_a_geometry_for_its_table(b):
    with pytest.raises(ContentTypeError, match="not a table"):
        _ = b["assemblyGeometry"].table


# -- corrections and conventions -------------------------------------------


def test_correction_declared_and_not_applied(b):
    obj = b["reactionRate-Al27"]
    correction = obj.corrections[0]
    assert correction.kind == "background"
    assert correction.applied is False
    assert correction.relative == pytest.approx(0.02)
    assert correction.factor() == pytest.approx(0.98)
    assert obj.corrected("backgroundSubtracted")[0] == pytest.approx(0.98e-20)
    assert obj.values[0] == pytest.approx(1.0e-20)  # the stored values are untouched


def test_corrected_to_its_own_convention_is_the_stored_values(b):
    obj = b["reactionRate-Au197"]
    assert obj.convention_label == "backgroundSubtracted"
    assert obj.corrected("backgroundSubtracted")[0] == pytest.approx(5.0e-15)


def test_a_correction_without_a_magnitude_cannot_be_applied(b):
    with pytest.raises(SinbadError, match="not reported"):
        b["reactionRate-Au197"].corrected("selfShielded")


def test_a_convention_the_object_cannot_reach(b):
    with pytest.raises(SinbadError, match="declares no correction"):
        b["reactionRate-Al27"].corrected("selfShielded")


# -- uncertainty -----------------------------------------------------------


def test_budget_components_and_their_scopes(b):
    budget = b["reactionRate-Al27"].uncertainty_budget
    assert budget.names == ["countingStatistics", "calibration", "platePower"]
    assert budget["countingStatistics"].is_per_point
    assert budget["platePower"].coverage_factor == 2.0
    # 8 % at 2 s.d. is 4 % at 1 s.d.
    assert budget["platePower"].relative()[0] == pytest.approx(0.04)
    assert budget["calibration"].relative()[0] == pytest.approx(0.03)


def test_budget_total_reproduces_the_published_column(b):
    obj = b["reactionRate-Al27"]
    total = obj.uncertainty_budget.total(obj.table)
    # 1 % / 2 % / 4 % counting, 3 % calibration, 4 % plate power, in quadrature
    assert total[:3] == pytest.approx([0.05099, 0.05385, 0.06403], abs=5e-5)
    assert obj.uncertainty[:3] == pytest.approx([0.051, 0.054, 0.064])


def test_covariance_is_the_sum_of_the_shared_components(b):
    matrix, index = b.covariance(relative=True)
    # four measured points: Al27 A1-A3 (A4 is blank) and Au197 A1-A2
    assert [label for label, _ in index] == (
        ["reactionRate-Al27"] * 3 + ["reactionRate-Au197"] * 2
    )
    assert [position for _, position in index] == ["A1", "A2", "A3", "A1", "A2"]
    assert matrix.shape == (5, 5)
    assert np.allclose(matrix, matrix.T)
    # diagonal: the total, squared
    assert matrix[0, 0] == pytest.approx(0.01**2 + 0.03**2 + 0.04**2)
    # same detector: calibration and plate power, no counting statistics
    assert matrix[0, 1] == pytest.approx(0.03**2 + 0.04**2)
    # different detectors: only what is within-entry
    assert matrix[0, 3] == pytest.approx(0.04**2)


def test_absolute_covariance_scales_by_the_values(b):
    relative, _ = b.covariance(relative=True)
    absolute, index = b.covariance()
    values = np.array([1.0e-20, 2.0e-21, 3.0e-22, 5.0e-15, 4.0e-15])
    assert np.allclose(absolute, relative * np.outer(values, values))


def test_correlation_has_a_unit_diagonal(b):
    matrix, _ = b.correlation()
    assert np.allclose(np.diag(matrix), 1.0)
    assert matrix[0, 3] == pytest.approx(
        0.04**2 / np.sqrt((0.01**2 + 0.03**2 + 0.04**2) * (0.01**2 + 0.02**2 + 0.04**2))
    )


def test_a_blank_point_carries_no_uncertainty(b):
    _, index = b["reactionRate-Al27"].covariance()
    assert [position for _, position in index] == ["A1", "A2", "A3"]


# -- other containers ------------------------------------------------------


def test_materials(b):
    materials = b["materials"].content
    assert materials.names == ["mildSteel1", "fuel2"]
    assert materials["mildSteel1"].density == 7.850
    assert materials["mildSteel1"].total == pytest.approx(1.0)
    assert materials["fuel2"].nuclides[1].name == "U235"
    frame = materials.to_dataframe()
    assert len(frame) == 4
    assert set(frame["units"]) == {"weight fraction"}


def test_geometry(b):
    geometry = b["assemblyGeometry"].content
    assert len(geometry.layers) == 8
    assert geometry.layer("slab-01").thickness == pytest.approx(4.5)
    assert geometry.layer("slab-01").material == "mildSteel1"
    assert geometry.shapes[0].lengths["length_z"].value == 16.5
    assert len(geometry.to_dataframe()) == 8


def test_factor_chain_recomputes_its_published_result(b):
    chain = b["sourceStrength"].content
    assert len(chain) == 3
    assert chain["platePower"].value.uncertainty.standard == pytest.approx(2.0e-5)
    assert chain.recompute() == pytest.approx(chain.result.value)


def test_gridded2d_shape_and_grids(b):
    gridded = b["sourceDistribution-xy"].content
    assert gridded.shape == (3, 2)
    assert gridded.values[2, 1] == 6.0        # highest grid index first
    assert gridded.grid("y").size == 3
    assert gridded.grid("x").centers == pytest.approx([-1.0, 1.0])
    assert gridded.unit == "1/(cm**3*s)"
    assert len(gridded.to_dataframe()) == 6


def test_a_grid_is_a_group_structure(b):
    content = b.calculations["lab-code-1.0"]["groupStructure-MINI3"].content
    assert content.size == 3                  # four boundaries, three groups
    assert content.grid.unit == "eV"
    assert len(content.to_dataframe()) == 3


# -- the join between the files --------------------------------------------


def test_a_linked_grid_resolves_across_files(b):
    response = b.calculations["lab-code-1.0"]["response-Al27"].content
    assert response.values.shape == (3,)
    assert response.grid("energy").values[0] == 1.0e7
    assert response.unit == "b"


def test_a_calculations_file_borrows_the_benchmarks_labels(b):
    calculations = b.calculations["lab-code-1.0"]
    assert calculations["reactionRate-Al27"] is b["reactionRate-Al27"]
    assert calculations["power30kW"].basis == "reactor power"


def test_a_benchmark_reaches_into_its_calculations(b):
    assert b["calculated-Al27"].nature == "calculated"
    assert b["CE-Al27"].operator == "ratio"


def test_an_unknown_label(b):
    with pytest.raises(LabelNotFoundError):
        b["no-such-thing"]
    assert "no-such-thing" not in b


def test_comparison_recompute_matches_published(b):
    comparison = b.calculations["lab-code-1.0"].comparisons["CE-Al27"]
    assert comparison.denominator_label == "reactionRate-Al27"
    frame = comparison.recompute()
    assert len(frame) == 6                    # three positions, two libraries
    assert frame["ratio"].tolist() == pytest.approx([1.0] * 6)
    published = comparison.to_dataframe()
    assert list(published["CE-LIBA"]) == [1.0, 0.9, 0.8]


def test_ce_collects_every_comparison(b):
    frame = b.ce()
    assert set(frame["series"]) == {"CE-LIBA", "CE-LIBB"}
    assert set(frame["reaction"]) == {"Al27(n,a)Na24"}
    assert len(frame) == 6
    wide = b.ce(wide=True)
    assert wide.shape == (3, 2)


def test_a_changed_benchmark_is_reported(b, tmp_path):
    calculations = b.calculations["lab-code-1.0"]
    assert calculations.matches_benchmark is False  # the fixture cites a wrong sha1
    assert not b.check()["ok"].all()

    copy = tmp_path / "mini"
    shutil.copytree(DATA, copy)
    real = b.document.checksum()
    path = copy / "calculations" / "lab-code-1.0.xml"
    text = path.read_text().replace("0" * 40, real)
    assert real in text
    path.write_text(text)
    fixed = sinbad.read(copy / "mini.xml")
    assert fixed.calculations["lab-code-1.0"].matches_benchmark is True
    assert fixed.check()["ok"].all()


# -- external files --------------------------------------------------------


def test_external_files_are_verified_against_the_entry(b):
    frame = b.verify_files(DATA).set_index("label")
    assert frame.loc["file-source", "ok"] is True
    assert frame.loc["file-stale", "ok"] is False
    assert "archive" in frame.loc["file-inside", "note"]


def test_resolving_a_file_inside_an_archive(b):
    with pytest.raises(SinbadError, match="archive"):
        b.files["file-inside"].resolve(DATA)
    assert b.files["file-source"].resolve(DATA).exists()


# -- frames ----------------------------------------------------------------


def test_to_dataframe_stacks_the_measured_tables(b):
    frame = b.to_dataframe()
    assert set(frame["object"]) == {"reactionRate-Al27", "reactionRate-Au197"}
    assert len(frame) == 6                    # 4 + 2 rows, blanks included

    corrected = b.to_dataframe(convention="backgroundSubtracted")
    al27 = corrected[corrected["object"] == "reactionRate-Al27"]
    assert al27["reactionRate"].iloc[0] == pytest.approx(0.98e-20)
    assert set(corrected["convention"]) == {"backgroundSubtracted"}


def test_filtering_the_data_objects(b):
    assert len(b.data(quantity="reaction rate")) == 2
    # the materials and the geometry are measured too -- nature is not "is a table"
    assert len(b.data(nature="measured")) == 4
    assert b.data(detector="Al27").labels == ["reactionRate-Al27"]
    assert b.data(kind="geometry").labels == ["assemblyGeometry"]
    assert b.data.quantities[0] == "material composition"


def test_summary_mentions_what_the_entry_holds(b):
    text = b.summary()
    assert "Mini fixture benchmark" in text
    assert "det-Al27" in text
    assert "measured points 5" in text
    assert "lab-code-1.0" in text
    assert "[benchmark has changed]" in text


# -- interop ---------------------------------------------------------------


def test_gridded_to_gnds_axes(b):
    from kika.nuclear_data.model import Axes as ModelAxes

    axes, values = b.calculations["lab-code-1.0"]["response-Al27"].content.to_gnds()
    assert isinstance(axes, ModelAxes)
    assert axes.dependent.label == "crossSection"
    assert axes.dependent.unit == "b"
    assert values.shape == (3,)


def test_a_unit_gnds_cannot_spell_is_dropped_not_raised(b):
    # 1/(cm**3*s) does not parse today; building the axes must still work.
    axes, _ = b["sourceDistribution-xy"].content.to_gnds()
    assert axes.dependent.label == "sourceDensity"
    assert axes.dependent.unit == ""


def test_check_units_reports_what_would_be_refused(b):
    from kika.sinbad.gnds import check_units

    frame = check_units(b).set_index("unit")
    assert frame.loc["1/s", "parses"] is np.True_ or frame.loc["1/s", "parses"] is True
    assert not frame.loc["%", "parses"]
    assert not frame.loc["1/(cm**3*s)", "parses"]


def test_materials_to_kika(b):
    collection = b["materials"].content.to_kika_materials()
    assert len(collection) == 2
    steel = collection.by_id[1]
    assert steel.name == "mildSteel1"
    assert steel.fraction_type == "wo"
    assert steel.density == 7.850
    assert "26000" in steel.to_mcnp()          # Fe0 became natural iron


# -- import cost -----------------------------------------------------------


def test_reading_an_entry_does_not_wake_the_gnds_model():
    """Opening a benchmark must not import :mod:`kika.nuclear_data.model`.

    The model is thirty modules describing an evaluated nuclear-data file, and
    ``kika/nuclear_data/model/tests/test_dormancy.py`` keeps it unreachable from
    a plain ``import kika``. A shielding entry has no business waking it: the
    translation lives in :mod:`kika.sinbad.gnds` and happens when it is asked
    for. It is also what lets this reader stand on numpy alone.

    Run in a subprocess, because by the time this module is collected pytest
    has imported everything. pandas and matplotlib are not asserted on here:
    ``kika/__init__.py`` imports them itself, so this test could say nothing
    about ``kika.sinbad``'s own cost.
    """
    code = textwrap.dedent(
        """
        import sys
        import kika.sinbad as sinbad
        b = sinbad.read(sys.argv[1])
        assert b.short_code == "mini"
        assert b["reactionRate-Al27"].values[0] > 0
        assert b["reactionRate-Al27"].covariance()[0].shape == (3, 3)
        print(",".join(sorted(m for m in sys.modules
                              if m.startswith("kika.nuclear_data.model"))))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(BENCHMARK)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"model woken by reading: {result.stdout.strip()}"


# -- what a wrong answer would look like -----------------------------------


def test_an_ambiguous_fragment_says_so_rather_than_no_such_thing(b):
    """A fragment matching several objects is ambiguous, not missing.

    Look-up accepts a fragment as a convenience, and the failure that costs
    time is the one that answers "no detector 'Al'" when the entry holds two
    of them.
    """
    with pytest.raises(AmbiguousLabelError, match="matches 2"):
        _ = b.data["reactionRate-A"]
    with pytest.raises(LabelNotFoundError, match="no data object"):
        _ = b.data["reactionRate-Xx"]
    assert isinstance(AmbiguousLabelError("x"), KeyError)


def test_a_multi_column_table_is_not_stamped_with_a_convention_it_is_not_in(b):
    """The trap: a frame labelled ``backgroundSubtracted`` that never was.

    ``corrected()`` needs one value column to know what to correct, so a
    calculated table of several libraries cannot be brought to another
    convention as one frame. Writing the column anyway would hand back numbers
    in one convention under the name of another.
    """
    calculations = b.calculations["lab-code-1.0"]
    stored = calculations["calculated-Al27"].table["reactionRate-LIBA"].copy()

    # its own convention is a no-op, and is allowed
    same = calculations.data(kind="table").to_dataframe(convention="backgroundSubtracted")
    assert (same[same["object"] == "calculated-Al27"]["reactionRate-LIBA"].to_numpy()
            == pytest.approx(stored))

    with pytest.raises(SinbadError, match="value columns"):
        calculations.data(kind="table").to_dataframe(convention="asMeasured")


def test_a_per_point_component_falls_back_to_its_own_confidence_level():
    """§2.3 puts the level on the column; a file that puts it on the component
    must not be read at 1 s.d. when it says 2."""
    import xml.etree.ElementTree as ET

    from kika.sinbad.content import Table
    from kika.sinbad.uncertainty import UncertaintyComponent

    table = Table(ET.fromstring(
        '<table rows="1" columns="2">'
        '  <columnHeaders>'
        '    <column index="0" name="rate" unit="1/s" role="value"/>'
        '    <column index="1" name="stat" unit="%" role="uncertainty" kind="component"/>'
        '  </columnHeaders>'
        '  <data sep="tr"><tr sep="td"><td>1.0</td><td>8.0</td></tr></data>'
        '</table>'
    ))
    component = UncertaintyComponent(name="stat", column="stat", confidence_level="2 s.d.")
    assert component.relative(table)[0] == pytest.approx(0.04)


def test_the_subpackage_is_reachable_from_a_plain_import_kika():
    """``import kika`` must find it without importing it -- the reason anyone
    who does not already know the subpackage exists will ever find it."""
    code = textwrap.dedent(
        """
        import sys
        import kika
        assert "kika.sinbad" not in sys.modules, "imported eagerly"
        assert "sinbad" in dir(kika)
        assert kika.sinbad.read is not None
        assert "kika.sinbad" in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
