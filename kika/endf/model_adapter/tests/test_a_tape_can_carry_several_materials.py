"""Several materials on one tape: ``writeEndfTapes`` (ENDF-coverage roadmap T2).

``assembleTape`` writes one material, MEND and TEND. A tape may carry many
(§0.6.3): each material closed by its MEND, one TPID before them all and one
TEND after. ``writeEndfTapes`` writes each material with ``writeEndfTape`` and
splices the records, so the gate is exactly that: the multi-material tape is
the single-material tapes' bodies, in order, between one TPID and one TEND.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kika.endf.model_adapter import decodeCovarianceSuite, decodeReactionSuite
from kika.endf.read_endf import read_endf
from kika.endf.writers.assemble import writeEndfTape, writeEndfTapes

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_TAPES = ("micro_na23_mf32.endf", "micro_th232_mf32.endf")


def _suite(name):
    endf = read_endf(str(_DATA / name))
    suite, _ = decodeReactionSuite(endf)
    suite.covarianceSuite, _ = decodeCovarianceSuite(endf)
    return suite


@pytest.fixture
def suites():
    return [_suite(name) for name in _TAPES]


def test_the_tape_is_the_materials_between_one_tpid_and_one_tend(suites, tmp_path):
    out = tmp_path / "two.endf"
    reports = writeEndfTapes(suites, out)
    assert len(reports) == 2

    singles = []
    for index, suite in enumerate(suites):
        one = tmp_path / f"one{index}.endf"
        writeEndfTape(suite, one)
        singles.append(one.read_text().rstrip("\n").split("\n"))

    lines = out.read_text().rstrip("\n").split("\n")
    assert lines[0] == singles[0][0], "the first suite's own TPID"
    assert lines[1:-1] == singles[0][1:-1] + singles[1][1:-1]
    assert lines[-1] == singles[0][-1], "one TEND"


def test_each_material_ends_with_its_mend(suites, tmp_path):
    out = tmp_path / "two.endf"
    writeEndfTapes(suites, out)
    lines = out.read_text().rstrip("\n").split("\n")
    mends = [i for i, line in enumerate(lines) if line[66:75] == "   0 0  0"]
    assert len(mends) == 2
    mats = [int(lines[i - 1][66:70]) for i in mends]   # the FEND before each MEND
    assert mats == [int(s.provenance.mat) for s in suites]


def test_two_suites_with_one_mat_are_refused(tmp_path):
    suite = _suite(_TAPES[0])
    with pytest.raises(ValueError, match="appears more than once"):
        writeEndfTapes([suite, _suite(_TAPES[0])], tmp_path / "x.endf")


def test_a_given_tape_id_wins(suites, tmp_path):
    out = tmp_path / "two.endf"
    reports = writeEndfTapes(suites, out, tapeId="TWO MATERIALS")
    assert out.read_text().split("\n")[0].startswith("TWO MATERIALS")
    assert not any("tape identification" in l for r in reports for l in r.losses)
