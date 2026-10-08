"""MF1/451 derived from the model, for a suite no ENDF read kept a header for.

The gate is the one FUDGE's ``toENDF6`` never had: take a suite decoded from a
real tape, hide its provenance, derive the nineteen fields from the model alone
and compare them with what the file states. Measured 2026-10-08 on JEFF-4.0
Fe-56/He-4/U-235, JENDL-5 Fe-56 and ENDF/B-VIII.1 B-10/He-4/U-238/Pu-239: equal
on every field, ZA and AWR included.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from kika.endf.model_adapter import decodeReactionSuite, encodeMF1MT451
from kika.endf.model_adapter.mf1_header import synthesiseMF1Header
from kika.endf.read_endf import read_endf
from kika.nuclear_data.model import ConversionReport

REAL_TAPES = ["fe56_host_tape", "fe57_host_tape", "fe56_jendl_tape",
              "u235_tape", "th232_tape", "pu241_tape", "u238_tape"]


def _derivedAgainstFile(path):
    suite, _ = decodeReactionSuite(read_endf(str(path)))
    report = ConversionReport()
    fields, za, awr = synthesiseMF1Header(suite, report)
    stated = suite.provenance.headerFields
    differ = {name: (stated.get(name), value) for name, value in fields.items()
              if stated.get(name) != value}
    return differ, (za, suite.provenance.za), (awr, suite.provenance.awr), report


def test_the_derived_header_is_the_files_header(micro_tape):
    differ, za, awr, report = _derivedAgainstFile(micro_tape)
    assert differ == {}
    assert za[0] == za[1]
    assert awr[0] == pytest.approx(awr[1], rel=1e-12)
    assert report.approximations == []


@pytest.mark.parametrize("tape", REAL_TAPES)
def test_the_same_holds_on_real_tapes(request, tape):
    differ, za, awr, _ = _derivedAgainstFile(request.getfixturevalue(tape))
    assert differ == {}, f"{tape}: derived header differs from the file"
    assert za[0] == za[1]
    assert awr[0] == pytest.approx(awr[1], rel=1e-12)


def test_the_library_is_spelled_as_fudge_spells_it(micro_tape):
    """The NLIB name and ``"NVER.LREL.NMOD"``: the number alone lost LREL and NMOD.

    The JEFF-4.0 Fe-56 slice says NLIB=32, INDL/A — its Fe-56 is INDEN's — which
    is the file's statement, not a misreading.
    """
    suite, _ = decodeReactionSuite(read_endf(str(micro_tape)))
    style = suite.styles.evaluatedFor("eval")
    fields = suite.provenance.headerFields
    assert fields["nlib"] == 32 and style.library == "INDL/A"
    assert style.version == f"{fields['nver']}.{fields['lrel']}.{fields['nmod']}"


def test_a_header_goes_through_a_gnds_file(micro_tape, tmp_path):
    """ENDF -> GNDS XML -> model -> MF1/451 with nothing lent from the ENDF read.

    The directory is the one thing a GNDS file cannot hold — it counts lines of
    the tape it came from — and the whole-file writer rebuilds it after writing,
    so it is set from the original before comparing.
    """
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.encode import writeReactionSuite
    from kika.gnds.xpath import Document

    endf = read_endf(str(micro_tape))
    original = endf.mf[1].mt[451]
    suite, _ = decodeReactionSuite(endf)
    tree, _ = writeReactionSuite(suite)
    path = tmp_path / "suite.xml"
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    reread, _ = readReactionSuite(Document.parse(path))
    assert getattr(reread.provenance, "sourceFormat", None) == "gnds"

    rebuilt, report = encodeMF1MT451(reread, mat=original._mat)
    rebuilt._directory = [tuple(entry) for entry in original._directory]
    rebuilt._nxc = len(rebuilt._directory)
    assert str(rebuilt) == str(original)
    assert report.approximations == [] and report.losses == []


def test_a_fudge_written_gnds_file_gives_its_header(h2_gnds):
    """ENDF/B-VIII.1 H-2 as FUDGE wrote it: every field derived, none assumed."""
    import kika

    suite = kika.read(h2_gnds, covariances=False)
    rebuilt, report = encodeMF1MT451(suite, mat=128)
    assert report.approximations == [] and report.losses == []
    assert int(rebuilt._za) == 1002
    assert rebuilt._awr == pytest.approx(1.9968, abs=5e-5)
    assert (rebuilt._nlib, rebuilt._nver, rebuilt._lrel, rebuilt._nmod) == (0, 8, 1, 0)
    assert (rebuilt._nsub, rebuilt._nfor, rebuilt._ldrv) == (10, 6, 0)
    assert rebuilt._emax == 1.5e8
    assert rebuilt.descriptive_text[0].startswith("  1-H -  2 LANL       EVAL-FEB97")
    assert rebuilt.authors.startswith("P.G.Young")


def test_with_no_comment_block_the_identification_records_are_written(micro_tape, tmp_path):
    """And the report says the block is missing rather than the tape looking whole."""
    suite, _ = decodeReactionSuite(read_endf(str(micro_tape)))
    suite.provenance = None
    suite.styles.evaluatedFor("eval").documentation = None

    rebuilt, report = encodeMF1MT451(suite, mat=2631)
    text = rebuilt.descriptive_text
    assert len(text) == 5
    assert text[0].startswith(" 26-Fe- 56 ")
    assert "INDL/A-" in text[2] and "MATERIAL 2631" in text[2]
    assert text[3].startswith("----- INCIDENT-NEUTRON DATA")
    assert any("no ENDF comment block" in loss for loss in report.losses)
    assert rebuilt.material_id == "26-Fe- 56"


def test_what_could_not_be_derived_is_reported(micro_tape):
    """A library outside ENDF-102's table and a version that is not N.N.N."""
    suite, _ = decodeReactionSuite(read_endf(str(micro_tape)))
    suite.provenance = None
    style = suite.styles.evaluatedFor("eval")
    style.library, style.version = "my-own-evaluation", "draft"

    rebuilt, report = encodeMF1MT451(suite, mat=2631)
    assert rebuilt._nlib == -1
    assert (rebuilt._nver, rebuilt._lrel, rebuilt._nmod) == (0, 0, 0)
    assert any("NLIB" in line for line in report.approximations)
    assert any("NVER" in line for line in report.approximations)


def test_an_endf_suite_still_writes_back_what_it_read(micro_tape):
    """Derivation is for suites with no header to keep, never instead of one."""
    endf = read_endf(str(micro_tape))
    suite, _ = decodeReactionSuite(endf)
    suite.provenance = replace(suite.provenance,
                               headerFields={**suite.provenance.headerFields, "nmod": 7})
    rebuilt, _ = encodeMF1MT451(suite)
    assert rebuilt._nmod == 7
