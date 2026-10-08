"""Capture against the two whole libraries and the JEFF-4.0 tapes. ``tape``-marked.

Every ``Capture/`` file is read to its last token and is a text fixed point,
and every capture isotope is read into the model, written back from it and
compared record for record (the model fixed point). Then the other direction:
from the JEFF-4.0 ENDF tapes, the final state kika writes is the IAEA
translation's (Mendoza & Cano-Ott), record for record but for what the
translation does (the AWR cut to 6 digits, roadmap Fase 9); the σ needs NJOY
and is the ``njoy``-marked test. The counts are the census of 2026-10-08.
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

import kika
import kika.g4ndl as g4ndl
from kika.g4ndl.capture import (
    CaptureMF6Record, captureFinalState, encodeCapture, finalStateDifferences, formatCaptureFS,
    parseFinalState,
)
from kika.g4ndl.encode import recordDifferences

pytestmark = pytest.mark.tape

#: library -> (capture isotopes, FSMF6 files, FS files, product laws of the FSMF6 files).
CENSUS = {
    "g4ndl_jeff40_library": (592, 564, 22, {(1,): 562, (2, 4): 1, (1, 0): 1}),
    "g4ndl_g4ndl471_library": (559, 508, 48, {(1,): 507, (2, 4): 1}),
}


@pytest.mark.parametrize("fixture", sorted(CENSUS))
def test_every_capture_isotope_reads_and_is_a_fixed_point(request, fixture):
    lib = g4ndl.open(request.getfixturevalue(fixture))
    n, n_mf6, n_fs, laws = CENSUS[fixture]
    keys = lib.isotopes("capture")
    assert len(keys) == n
    kinds, seen = collections.Counter(), collections.Counter()
    for key in keys:
        record = lib.captureFinalState(key)
        cs, fs, _ = encodeCapture(lib.read(key, processes=["capture"]))
        assert recordDifferences(lib.captureCrossSection(key), cs) == [], key
        if record is None:
            assert fs is None, key
            continue
        kind = "FSMF6" if isinstance(record, CaptureMF6Record) else "FS"
        kinds[kind] += 1
        if kind == "FSMF6":
            seen[tuple(p.distLaw for p in record.body.products)] += 1
        text = formatCaptureFS(record)
        assert finalStateDifferences(record, parseFinalState(text, kind)) == [], key
        assert finalStateDifferences(record, fs) == [], key
    assert kinds == {"FSMF6": n_mf6, "FS": n_fs} and dict(seen) == laws


def _jeff40(tape_dir, name):
    path = tape_dir / name
    if not path.is_file():
        pytest.skip(f"{path} not found")
    return kika.read(str(path), format="endf")


@pytest.mark.parametrize("tape,target,diffs", [
    ("n_82-Pb-208g.jeff", "Pb208", []),
    ("n_29-Cu-063g.jeff", "Cu63", []),            # a LAW=0 Cu-64 besides the photon
    ("n_26-Fe-056g.jeff", "Fe56", ["CaptureMF6Record.body.targetMass: 55.4544 != 55.45443"]),
    ("n_1-H-001g.jeff", "H1", ["CaptureMF6Record.body.targetMass: 0.999167 != 0.9991673"]),
])
def test_the_final_state_from_an_endf_tape_is_the_iaea_translation(
        fe56_jeff40_tape, g4ndl_jeff40_library, tape, target, diffs):
    suite = _jeff40(fe56_jeff40_tape.parent, tape)
    ours = captureFinalState(suite)
    theirs = g4ndl.open(g4ndl_jeff40_library).captureFinalState(target)
    assert finalStateDifferences(theirs, ours) == diffs


def test_an_endf_tape_with_its_capture_photons_in_mf12_writes_no_final_state(
        fe56_jeff40_tape, g4ndl_jeff40_library):
    """N-14: MF12-15 photons, which the ENDF adapter does not read (D10-2)."""
    from kika.nuclear_data.model import ConversionReport

    report = ConversionReport()
    assert captureFinalState(_jeff40(fe56_jeff40_tape.parent, "n_7-N-014g.jeff"),
                             report=report) is None
    assert any("G4PhotonEvaporation" in m for m in report.warnings)
    assert g4ndl.open(g4ndl_jeff40_library).captureFinalStateDirectory("N14") == "Capture/FS"


def _groupAverages(x, y, edges, extra):
    grid = np.union1d(np.union1d(x, edges), extra)
    values = np.interp(grid, x, y)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(grid))])
    return np.diff(np.interp(edges, grid, cumulative)) / np.diff(edges)


@pytest.mark.njoy
def test_fe56_jeff40_capture_against_the_iaea_translation(tmp_path, fe56_jeff40_tape, njoy_exe,
                                                          g4ndl_jeff40_library):
    from kika.endf.model_adapter.pendf import readReconstructed

    suite, _ = readReconstructed(fe56_jeff40_tape, njoy=njoy_exe, tolerance=1e-3,
                                 cache_dir=tmp_path)
    cs, fs, _ = encodeCapture(suite)
    iaea = g4ndl.open(g4ndl_jeff40_library)
    ref = iaea.captureCrossSection("Fe56")
    # The bookkeeping is the tape's Q, rounded to the eV, as the IAEA writes it.
    assert cs.bookkeeping == ref.bookkeeping
    assert finalStateDifferences(iaea.captureFinalState("Fe56"), fs) == [
        "CaptureMF6Record.body.targetMass: 55.4544 != 55.45443"]
    # σ as for the elastic (test_from_endf.py): pointwise above the resonance
    # region, 1 000 log-group averages inside it.
    top = max(r.domainMax for r in suite.resonances.resolved)
    above = ref.energy > top
    rel = np.interp(ref.energy[above], cs.energy, cs.sigma) / ref.sigma[above] - 1
    assert np.abs(rel).max() < 1e-4
    edges = np.logspace(-5, np.log10(2e7), 1001)
    ours = _groupAverages(cs.energy, cs.sigma, edges, ref.energy)
    theirs = _groupAverages(ref.energy, ref.sigma, edges, cs.energy)
    # Looser than the elastic's 2e-3, and measured (2026-10-08): the worst
    # groups are at 0.15-0.4 MeV, between resonances, where capture is 2-7 mb
    # (1e-4 of the total) and the IAEA grid is twice as dense as ours at
    # ERR=1e-3. Rerun at ERR=1e-4, our grid is denser than theirs and the
    # maximum only falls from 3.6e-3 to 2.2e-3: the rest is the IAEA's own
    # reconstruction there, not ours.
    assert np.abs(ours / theirs - 1).max() < 4e-3
