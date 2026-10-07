"""Cutting sections out of a tape has to leave a tape.

:func:`~kika.endf.writers.section_ops.remove_sections` and
:func:`~kika.endf.writers.mf34_writer.remove_mf34_from_file` both delete data
lines, and both used to leave the *bookkeeping* records of what they deleted
behind: the SEND that closes a removed MT, and the FEND that closes an MF with
nothing left in it. The result parses far enough to look fine — kika's own
reader skips what it does not recognise — and is not valid ENDF: two FENDs in a
row say the previous file ended twice.

The invariant these tests pin is structural, not textual: **one SEND per
surviving MT section, one FEND per surviving MF block**, with the TPID, MEND and
TEND records — which wear MF=0/MT=0 in the same columns as a FEND — untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import pytest

from kika.endf.utils import (
    format_endf_data_line,
    format_endf_fend_record,
    format_endf_mend_record,
    format_endf_send_record,
    format_endf_tend_record,
    parse_endf_id,
)
from kika.endf.writers import remove_mf34_from_file, remove_sections

DATA = Path(__file__).resolve().parent / "data"
STRUCTURAL = DATA / "micro_fe56_structural.endf"

MAT = 2631


# --------------------------------------------------------------------------
# A tape small enough to read by eye
# --------------------------------------------------------------------------

def _data_line(mf: int, mt: int, seq: int) -> str:
    return format_endf_data_line([0.0, 0.0, 0, 0, 0, 0], MAT, mf, mt, seq) + "\n"


def _tape(sections: Sequence[Tuple[int, Sequence[int]]], n_lines: int = 2) -> str:
    """A whole tape: TPID, the given (MF, [MT…]) blocks, MEND, TEND."""
    out = [format_endf_data_line([0.0, 0.0, 0, 0, 0, 0], 1, 0, 0, 0) + "\n"]
    for mf, mts in sections:
        for mt in mts:
            out += [_data_line(mf, mt, i + 1) for i in range(n_lines)]
            out.append(format_endf_send_record(MAT, mf) + "\n")
        out.append(format_endf_fend_record(MAT) + "\n")
    out.append(format_endf_mend_record() + "\n")
    out.append(format_endf_tend_record() + "\n")
    return "".join(out)


#: MF1/451 first because :func:`remove_sections` rebuilds it, and it must be
#: long enough for the directory parser to find its four CONT records.
FULL = _tape(
    [(1, [451]), (3, [1, 2, 102]), (4, [2]), (33, [1, 2]), (34, [2])],
    n_lines=8,
)


def _ids(content: str) -> List[Tuple[int, int, int]]:
    return [parse_endf_id(line) for line in content.splitlines()]


def _structure(content: str) -> List[Tuple[int, int, int]]:
    """The record sequence with runs of identical data lines collapsed."""
    out: List[Tuple[int, int, int]] = []
    for ident in _ids(content):
        if not out or out[-1] != ident:
            out.append(ident)
    return out


def assert_valid_tape(content: str) -> None:
    """One SEND per MT section, one FEND per MF block, terminators intact."""
    ids = _ids(content)
    assert ids[0][1:] == (0, 0), "a tape opens with its TPID"
    assert ids[-2] == (0, 0, 0), "MEND is the penultimate record"
    assert ids[-1] == (-1, 0, 0), "TEND closes the tape"

    open_mt = None          # the (mf, mt) whose data lines are running
    mfs_in_block: set = set()
    seen_mfs: List[int] = []
    for i, (mat, mf, mt) in enumerate(ids[1:-2], start=1):
        if mf > 0 and mt > 0:
            if open_mt is not None and open_mt != (mf, mt):
                raise AssertionError(
                    f"line {i}: MF{mf}/MT{mt} starts before MF{open_mt[0]}"
                    f"/MT{open_mt[1]} was closed by a SEND")
            open_mt = (mf, mt)
            mfs_in_block.add(mf)
        elif mf > 0 and mt == 0:
            assert open_mt is not None, f"line {i}: SEND closing nothing"
            assert open_mt[0] == mf, (
                f"line {i}: MF{mf} SEND closing an MF{open_mt[0]} section")
            open_mt = None
        else:
            assert open_mt is None, f"line {i}: FEND before the SEND of {open_mt}"
            assert mfs_in_block, f"line {i}: FEND closing an empty MF block"
            assert len(mfs_in_block) == 1, (
                f"line {i}: one FEND for several MFs {sorted(mfs_in_block)}")
            seen_mfs += sorted(mfs_in_block)
            mfs_in_block = set()
    assert not mfs_in_block, f"MF{sorted(mfs_in_block)} never got its FEND"
    assert seen_mfs == sorted(set(seen_mfs)), (
        f"MF blocks out of order or split: {seen_mfs}")


def mfs_mts(content: str) -> List[Tuple[int, int]]:
    return sorted({(mf, mt) for _, mf, mt in _ids(content) if mf > 0 and mt > 0})


# --------------------------------------------------------------------------
# The fixture itself has to be valid, or nothing below means anything
# --------------------------------------------------------------------------

def test_the_synthetic_tape_is_valid_to_begin_with():
    assert_valid_tape(FULL)
    assert mfs_mts(FULL) == [
        (1, 451), (3, 1), (3, 2), (3, 102), (4, 2), (33, 1), (33, 2), (34, 2)]


# --------------------------------------------------------------------------
# FEND: a whole MF removed takes the record that closed it
# --------------------------------------------------------------------------

def test_removing_a_whole_mf_takes_its_fend():
    out, n = remove_sections(FULL, [(34, None)])
    assert n == 1
    assert_valid_tape(out)
    assert (34, 2) not in mfs_mts(out)
    assert _ids(out).count((MAT, 0, 0)) == _ids(FULL).count((MAT, 0, 0)) - 1


def test_removing_the_last_two_mfs_takes_both_fends():
    out, n = remove_sections(FULL, [(33, None), (34, None)])
    assert n == 3  # (33,1), (33,2), (34,2)
    assert_valid_tape(out)
    assert mfs_mts(out) == [(1, 451), (3, 1), (3, 2), (3, 102), (4, 2)]


def test_removing_a_middle_mf_takes_its_fend():
    out, _ = remove_sections(FULL, [(4, None)])
    assert_valid_tape(out)
    assert mfs_mts(out) == [
        (1, 451), (3, 1), (3, 2), (3, 102), (33, 1), (33, 2), (34, 2)]


# --------------------------------------------------------------------------
# SEND: a removed MT takes the record that closed it, wherever it sat
# --------------------------------------------------------------------------

def test_removing_the_first_mt_of_an_mf_takes_its_send():
    out, n = remove_sections(FULL, [(3, 1)])
    assert n == 1
    assert_valid_tape(out)
    assert mfs_mts(out) == [
        (1, 451), (3, 2), (3, 102), (4, 2), (33, 1), (33, 2), (34, 2)]


def test_removing_a_middle_mt_takes_its_send():
    """The case the first-survivor rule got wrong: MT2 sits after MT1."""
    out, n = remove_sections(FULL, [(3, 2)])
    assert n == 1
    assert_valid_tape(out)
    assert mfs_mts(out) == [
        (1, 451), (3, 1), (3, 102), (4, 2), (33, 1), (33, 2), (34, 2)]


def test_removing_the_last_mt_of_an_mf_takes_its_send():
    out, n = remove_sections(FULL, [(3, 102)])
    assert n == 1
    assert_valid_tape(out)
    assert mfs_mts(out) == [
        (1, 451), (3, 1), (3, 2), (4, 2), (33, 1), (33, 2), (34, 2)]


def test_emptying_an_mf_one_mt_at_a_time_takes_the_fend_too():
    out, n = remove_sections(FULL, [(3, 1), (3, 2), (3, 102)])
    assert n == 3
    assert_valid_tape(out)
    assert mfs_mts(out) == [(1, 451), (4, 2), (33, 1), (33, 2), (34, 2)]


# --------------------------------------------------------------------------
# What must never go
# --------------------------------------------------------------------------

def test_tpid_mend_and_tend_are_never_cut():
    """All three wear MF=0/MT=0, and only one of them is a FEND."""
    out, _ = remove_sections(FULL, [(3, None), (4, None), (33, None), (34, None)])
    ids = _ids(out)
    assert ids[0][1:] == (0, 0) and ids[0][0] > 0   # TPID, MAT > 0 like a FEND
    assert ids[-2] == (0, 0, 0) and ids[-1] == (-1, 0, 0)
    assert_valid_tape(out)
    assert mfs_mts(out) == [(1, 451)]


def test_mf1_451_survives_a_whole_mf1_removal_with_its_send_and_fend():
    out, _ = remove_sections(FULL, [(1, None), (34, None)])
    assert_valid_tape(out)
    assert (1, 451) in mfs_mts(out)


def test_removing_nothing_changes_nothing():
    out, n = remove_sections(FULL, [(5, None), (35, 18)])
    assert n == 0
    assert out == FULL


# --------------------------------------------------------------------------
# The two removal routes are one behaviour
# --------------------------------------------------------------------------

def test_remove_mf34_from_file_agrees_with_remove_sections(tmp_path):
    """The line-slice route exists so an 800 MB tape need not be one string."""
    path = tmp_path / "tape.endf"
    path.write_text(FULL)
    assert remove_mf34_from_file(str(path), update_directory=False) is True
    sliced = path.read_text()

    filtered, _ = remove_sections(FULL, [(34, None)])
    assert_valid_tape(sliced)
    assert _structure(sliced) == _structure(filtered)


def test_remove_mf34_from_file_is_a_no_op_without_mf34(tmp_path):
    before, _ = remove_sections(FULL, [(34, None)])
    path = tmp_path / "tape.endf"
    path.write_text(before)
    assert remove_mf34_from_file(str(path), update_directory=False) is False
    assert path.read_text() == before


# --------------------------------------------------------------------------
# On a real JEFF-4.0 cut, not a tape we wrote
# --------------------------------------------------------------------------

@pytest.mark.skipif(not STRUCTURAL.exists(), reason="micro-tape not committed")
def test_real_tape_loses_exactly_the_fend_of_the_mf_it_loses():
    """The committed micro-tapes predate this fix and carry stranded records
    of their own; what is asserted here is the *delta*, which is exact."""
    content = STRUCTURAL.read_text()
    out, n = remove_sections(content, [(34, None)])
    assert n >= 1
    assert not [1 for _, mf, _ in _ids(out) if mf == 34]
    assert (_ids(content).count((MAT, 0, 0))
            - _ids(out).count((MAT, 0, 0))) == 1
