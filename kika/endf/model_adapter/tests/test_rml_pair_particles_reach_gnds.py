"""An RML pair's particles survive kika's GNDS: the GNDS route rebuilds the
pairs the ENDF route reads.

ENDF states each LRF=7 pair's two particles (MA, ZA, IA, PA; MB, ...), and the
ENDF decoder keeps them as ``kinematics``, which GNDS has no node for. The
GNDS route rebuilds the pair from the formalism's PoPs and the ejectile, so the
decoder writes them there (``_pairParticlesFromMF2``). Before (2026-10-16) kika
could not process its own GNDS of Cl-35, and with the suite's atomic masses its
(n,p) came 1.6e-3 off; now all 163 MF3 sections of the two routes' PENDFs are
identical.

``KIKA_LIB_TAPES`` points at ``NuclearData/endf`` (``endfb81/`` inside).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import kika

pytestmark = pytest.mark.tape


def _tape() -> Path:
    tapes = os.environ.get("KIKA_LIB_TAPES")
    return Path(tapes) / "endfb81" / "n-017_Cl_035.endf" if tapes else Path("missing")


def _pairs(suite):
    from kika.endf.process import _context_from_model
    from kika.processing.resonances.prepare_r_matrix import normalize_suite_pairs

    (region,) = normalize_suite_pairs(suite, _context_from_model(suite)[0]).resolved
    return {rr.reactionMT: rr for rr in region.formalism.resonanceReactions
            if not rr.kinematics.effective}


def test_cl35_pairs_come_back_from_kikas_gnds(tmp_path):
    tape = _tape()
    if not tape.is_file():
        pytest.skip(f"{tape} not found (set KIKA_LIB_TAPES)")
    from kika.gnds.decode import readReactionSuite
    from kika.gnds.xpath import Document

    suite = kika.read(str(tape), format="endf", covariances=False)
    path = tmp_path / "cl35.gnds.xml"
    kika.write(suite, path, format="gnds")
    back, _ = readReactionSuite(Document.parse(path))

    before, after = _pairs(suite), _pairs(back)
    assert sorted(after) == sorted(before) == [2, 600]
    assert after[600].ejectile == "H1"
    for mt, pair in before.items():
        for a, b in ((pair.kinematics.particleA, after[mt].kinematics.particleA),
                     (pair.kinematics.particleB, after[mt].kinematics.particleB)):
            assert (b.charge, b.spin, b.parity) == (a.charge, a.spin, a.parity)
            assert b.massRatio == pytest.approx(a.massRatio, rel=1e-14)
