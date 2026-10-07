"""``endf-delta`` replaces the sections it perturbed, and not one line more (PF-5).

The emitter used to replace the whole MF it touched, so every *sibling* MT in
that MF was re-encoded although the realisation never moved it: MF5/MT455 beside
a perturbed MT18 came back with new sequence numbers, zero-filled SEND records
and, on ENDF/B-VIII.1 U-233, its ZA spelt ``9.223300+4`` instead of
``92233.0000``. The values were the same on every tape measured, which is why
the Q1 gates (``test_the_model_writes_a_fission_spectrum_as_the_legacy_does.py``)
compare numbers; this file compares the text.

The property: take the delta tape and the tape it came from, drop the sections
the realisation perturbed (with their SEND records) and MF1/MT451, whose
directory legitimately changes with the line counts -- and what is left is the
same lines, control records included, in the same order. FEND and MEND are
*not* exempt: the Fe-56 micro-tape carries 78 copies of MF3's FEND, and a
per-MT replacement has no business normalising them.

Lines, not raw bytes: ``ENDFWriter`` writes in text mode, so on Windows the
whole output gets CRLF endings whatever the source had. That is the writer's,
not the emitter's, and it touches every line equally.

The plan is ``kika-workspace/docs/pfns/pfns_mf5_mf35_roadmap.md``, PF-5 / PD-7.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kika.sampling.model_perturbation import perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"


def _outside(path, perturbed):
    """The tape's lines with *perturbed* sections, their SENDs and MF1/451 dropped."""
    dropped = set(perturbed) | {(1, 451)}
    kept, inDropped = [], False
    for line in Path(path).read_text().splitlines():
        mf, mt = int(line[70:72]), int(line[72:75])
        if (mf, mt) in dropped:
            inDropped = True
            continue
        if inDropped and mt == 0 and mf != 0:
            inDropped = False
            continue  # the SEND that closes a replaced section
        inDropped = False
        kept.append(line)
    return kept


def _sections(path):
    """``{(mf, mt): [line, ...]}`` of the data records."""
    out = {}
    for line in Path(path).read_text().splitlines():
        mf, mt = int(line[70:72]), int(line[72:75])
        if mt:
            out.setdefault((mf, mt), []).append(line)
    return out


def _assertOnlyPerturbedMoved(tape, request, perturbed, tmp_path):
    run = perturbFromModel(str(tape), request, nSamples=1, seed=11,
                           outputDir=tmp_path, formats=("endf-delta",))
    delta = run.paths("endf-delta")[0]

    before, after = _outside(tape, perturbed), _outside(delta, perturbed)
    assert len(after) == len(before), (
        f"{len(after)} lines outside the perturbed sections, "
        f"{len(before)} in the source")
    for number, (old, new) in enumerate(zip(before, after)):
        assert new == old, (
            f"line {number} outside the perturbed sections was rewritten:\n"
            f"  source: {old!r}\n  delta:  {new!r}")

    source, written = _sections(tape), _sections(delta)
    for key in perturbed:
        assert written[key] != source[key], f"MF{key[0]}/MT{key[1]} did not move"


# ----------------------------------------------------------------------
# The spectrum: MF5 has siblings
# ----------------------------------------------------------------------

def test_a_spectrum_delta_leaves_the_other_mf5_sections_alone_on_the_micro_tape(
        tmp_path):
    _assertOnlyPerturbedMoved(DATA / "micro_cf252_pfns.endf", {35: None},
                              {(5, 18)}, tmp_path)


def test_a_spectrum_delta_leaves_mt455_alone_on_endfb81_u233(u233_b81_tape,
                                                              tmp_path):
    """The tape PF-5 was found on: MT455's ZA is ``92233.0000`` in the source."""
    source = _sections(u233_b81_tape)
    assert (5, 455) in source, "this evaluation is meant to carry MF5/MT455"
    assert source[(5, 455)][0].startswith(" 92233.0000"), (
        "MT455's HEAD no longer spells its ZA the way that exposed PF-5")
    _assertOnlyPerturbedMoved(u233_b81_tape, {35: None}, {(5, 18)}, tmp_path)


# ----------------------------------------------------------------------
# The cross section and the angular distribution: MF3/MF4 have siblings
# ----------------------------------------------------------------------

FE56 = DATA / "micro_fe56_xs_and_angular.endf"


def test_a_cross_section_delta_leaves_the_other_mf3_sections_alone(tmp_path):
    """MT1 and MT102 sit in the MF3 that MT2's perturbation rewrites."""
    _assertOnlyPerturbedMoved(FE56, {33: None}, {(3, 2)}, tmp_path)


def test_a_joint_delta_moves_mf3_and_mf4_and_nothing_else(tmp_path):
    _assertOnlyPerturbedMoved(
        FE56, {33: None, 34: {"mt": [2], "index": [1, 2, 3]}},
        {(3, 2), (4, 2)}, tmp_path)


# ----------------------------------------------------------------------
# The multiplicity: MF1 is shared with the directory
# ----------------------------------------------------------------------

def test_a_multiplicity_delta_moves_the_three_nubars_and_nothing_else(tmp_path):
    """MT452 is moved although only 455 and 456 were asked for (the sum rule)."""
    _assertOnlyPerturbedMoved(DATA / "micro_u235_nubar.endf", {31: [455, 456]},
                              {(1, 452), (1, 455), (1, 456)}, tmp_path)
