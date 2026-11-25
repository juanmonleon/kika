"""The multigroup readers report, through ``diagnostics=``, what they drop and assume.

Two ERRORR outputs of the NJOY2016 test suite are the real files (``data/njoy2016``);
the rest are files kika writes, changed in one place to make the event happen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.cov import read_boxer, read_covfil, read_coverx, write_boxer, write_coverx
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.parse_covmat import READ_EVENT_KINDS, ReadEvent

NJOY = Path(__file__).parent / "data" / "njoy2016"
MF33 = NJOY / "test04_errorr_mf33.gendf"
MF34 = NJOY / "test15_errorr_mf34.gendf"
FE56 = 26056
GRID = [1.0e-5, 1.0, 1.0e3, 1.0e6, 2.0e7]


def _covariance() -> CrossSectionCovariance:
    cov = CrossSectionCovariance(num_groups=4, energy_grid=list(GRID))
    cov.add_matrix(FE56, 2, FE56, 2, np.diag([0.01, 0.02, 0.03, 0.04]))
    cov.add_matrix(FE56, 102, FE56, 102, np.diag([0.04, 0.03, 0.02, 0.01]))
    cov.add_matrix(FE56, 2, FE56, 102, np.full((4, 4), 0.001))
    cov.cross_sections[(FE56, 2)] = np.array([1.0, 2.0, 3.0, 4.0])
    cov.cross_sections[(FE56, 102)] = np.array([0.5, 0.6, 0.7, 0.8])
    return cov


def _same(a, b) -> bool:
    lists = ("num_groups", "energy_grid", "isotope_rows", "reaction_rows", "isotope_cols",
             "reaction_cols", "is_relative", "metadata")
    if any(getattr(a, n) != getattr(b, n) for n in lists if hasattr(a, n)):
        return False
    if len(a.matrices) != len(b.matrices):
        return False
    if any(not np.array_equal(x, y) for x, y in zip(a.matrices, b.matrices)):
        return False
    for name in ("cross_sections", "legendre_coefficients", "l_rows", "l_cols", "energy_grids"):
        x, y = getattr(a, name, None), getattr(b, name, None)
        if isinstance(x, dict):
            if set(x) != set(y) or any(not np.array_equal(x[k], y[k]) for k in x):
                return False
        elif x != y:
            return False
    return True


def _kinds(events):
    return [e.kind for e in events]


def _edited(src: Path, dst: Path, line: int, old: str, new: str) -> Path:
    lines = src.read_text().split("\n")
    assert old in lines[line - 1], lines[line - 1]
    lines[line - 1] = lines[line - 1].replace(old, new, 1)
    dst.write_text("\n".join(lines))
    return dst


def _written(tmp_path, kind):
    path = tmp_path / f"cov.{kind}"
    if kind == "boxer":
        write_boxer(_covariance(), path, nvf=14)
    else:
        write_coverx(_covariance(), path, fmt=kind)
    return path


READERS = [
    ("errorr-mf33", lambda tmp: (read_covfil, MF33)),
    ("errorr-mf34", lambda tmp: (read_covfil, MF34)),
    ("coverx-binary", lambda tmp: (read_coverx, _written(tmp, "binary"))),
    ("coverx-text", lambda tmp: (read_coverx, _written(tmp, "text"))),
    ("boxer", lambda tmp: (read_boxer, _written(tmp, "boxer"))),
]


@pytest.mark.parametrize("make", [m for _, m in READERS], ids=[n for n, _ in READERS])
def test_the_object_read_does_not_depend_on_diagnostics(tmp_path, make):
    reader, path = make(tmp_path)
    events = []
    assert _same(reader(path), reader(path, diagnostics=events))
    assert events and all(isinstance(e, ReadEvent) and e.kind in READ_EVENT_KINDS for e in events)


def test_errorr_mf33_says_which_zero_pair_it_dropped_and_which_unit_it_assumed():
    events = []
    read_covfil(MF33, diagnostics=events)
    dropped = [e for e in events if e.kind == "dropped_block"]
    assert [(e.where["mt"], e.where["mt1"], e.evidence["reason"]) for e in dropped] == [(18, 102, "zero")]
    (unit,) = [e for e in events if e.kind == "unit_assumed"]
    assert unit.evidence["is_relative"] is True
    assert [e.evidence["ngroup"] for e in events if e.kind == "declared" and e.where.get("mf") == 1] == [9]


def test_errorr_mf34_says_the_reaction_is_not_in_the_file():
    events = []
    read_covfil(MF34, diagnostics=events)
    (event,) = [e for e in events if e.kind == "reaction_not_stated"]
    assert event.where == {"mat": 9237, "mf": 34, "mt": 251}


def test_a_row_numbered_zero_is_out_of_range_though_the_reader_puts_it_on_row_one(tmp_path):
    path = _edited(MF33, tmp_path / "irow0.gendf", 31, "          5139533", "          0139533")
    events = []
    read_covfil(path, diagnostics=events)
    (event,) = [e for e in events if e.kind == "index_out_of_range"]
    assert (event.where["mt"], event.evidence["irow"]) == (18, 0)


def test_a_row_given_twice_is_a_row_order_event(tmp_path):
    path = _edited(MF33, tmp_path / "irow4.gendf", 31, "          5139533", "          4139533")
    events = []
    read_covfil(path, diagnostics=events)
    (event,) = [e for e in events if e.kind == "row_order"]
    assert event.evidence == {"previous": 4, "repeated": True}


def test_coverx_binary_declares_ntype_and_gives_the_standard_deviations_it_does_not_keep(tmp_path):
    events = []
    read_coverx(_written(tmp_path, "binary"), diagnostics=events)
    (head,) = [e for e in events if e.kind == "declared" and "ntype" in e.evidence]
    assert head.evidence["ntype"] == 2
    assert {(e.where["mat"], e.where["mt"]) for e in events if e.kind == "raw_stddev"} == {
        (FE56, 2), (FE56, 102)}
    assert "unit_assumed" in _kinds(events)


def test_coverx_text_says_it_skipped_an_unknown_mt(tmp_path):
    path = _written(tmp_path, "text")
    text = path.read_text().replace(f"{FE56:12d}{102:12d}{FE56:12d}{102:12d}",
                                    f"{FE56:12d}{999:12d}{FE56:12d}{999:12d}", 1)
    path.write_text(text)
    events = []
    cov = read_coverx(path, diagnostics=events)
    (event,) = [e for e in events if e.kind == "dropped_block"]
    assert (event.where["mt"], event.evidence["reason"]) == (999, "unknown_mt")
    assert 999 not in cov.reactions


def test_coverx_says_a_pair_stored_in_both_orientations(tmp_path):
    cov = _covariance()
    cov.add_matrix(FE56, 102, FE56, 2, np.full((4, 4), 0.002))
    path = tmp_path / "both.coverx"
    write_coverx(cov, path, fmt="text")
    events = []
    read_coverx(path, diagnostics=events)
    (event,) = [e for e in events if e.kind == "repeated_block"]
    assert event.evidence["same_orientation"] is False


def test_boxer_gives_the_standard_deviations_and_says_a_matrix_cut_short(tmp_path):
    path = _written(tmp_path, "boxer")
    lines = path.read_text().split("\n")
    last = max(i for i, ln in enumerate(lines) if len(ln) >= 80 and ln[0] == "3")
    lines[last] = lines[last][:68] + f"{1:4d}" + lines[last][72:]  # NROWM=1: rows still to come
    path.write_text("\n".join(lines))
    events = []
    read_boxer(path, diagnostics=events)
    assert {(e.where["mt"], e.evidence["order"]) for e in events if e.kind == "raw_stddev"} == {
        (2, "file"), (102, "file")}
    assert [e.evidence["reason"] for e in events if e.kind == "dropped_block"] == ["unfinished"]
