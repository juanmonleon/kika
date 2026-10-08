"""Phase 9: an ENDF evaluation → RECONR at 0 K → the model → G4NDL.

Acceptance (roadmap §5 Fase 9): the processed σ agrees with an independent
processor within a declared tolerance, the thermal treatment is stated, and the
exported data have one coherent origin. The independent processing is the IAEA
G4NDL translation of JEFF-4.0 (Mendoza & Cano-Ott): from the JEFF-4.0 tape,
kika must write the IAEA's final state record for record, and σ that agrees
with theirs where both are resolvable (the ``njoy``-marked test).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kika
import kika.g4ndl as g4ndl
from kika.endf.model_adapter.pendf import attachReconstruction, readReconstructed
from kika.g4ndl import G4NDLUnsupportedError
from kika.g4ndl.encode import encodeElastic, recordDifferences

DATA = Path(kika.__file__).parent / "endf" / "tests" / "data"
FE56 = DATA / "micro_fe56_xs_and_angular.endf"


def _micro():
    return kika.read(str(FE56), format="endf")


def test_the_background_is_never_written_as_the_cross_section():
    suite = _micro()
    with pytest.raises(G4NDLUnsupportedError, match="readReconstructed"):
        encodeElastic(suite)
    with pytest.raises(G4NDLUnsupportedError, match="resonance region"):
        encodeElastic(suite, crossSectionLabel="eval")


def test_an_endf_suite_with_its_reconstruction_is_written(tmp_path):
    """The micro-tape stands in for its own PENDF (see test_pendf.py)."""
    suite = _micro()
    attachReconstruction(suite, FE56)
    report = kika.write(suite, tmp_path, format="g4ndl")
    lib = g4ndl.open(tmp_path)
    assert lib.isotopes() == [g4ndl.IsotopeKey(26, 56)]
    fs = lib.elasticFinalState("Fe56")
    assert fs.targetMass == 55.45443 and fs.frameFlag == 2    # AWR and LCT of the tape
    assert fs.header is None and report.warnings == []
    # The angular distribution is the tape's MF4, record for record.
    mf4 = suite.reactions[2].outputChannel.products.byPid("n")[0].distribution["eval"]
    assert [r.energy for b in (fs.legendre, fs.tabulated) if b for r in b.records] == mf4.energies
    back = lib.read("Fe56")
    assert np.array_equal(back.reactions[2].crossSection["recon"].ys,
                          suite.reactions[2].crossSection["recon"].ys)


# ---------------------------------------------------------- RECONR, real tape

def _groupAverages(x, y, edges, extra):
    grid = np.union1d(np.union1d(x, edges), extra)
    values = np.interp(grid, x, y)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(grid))])
    return np.diff(np.interp(edges, grid, cumulative)) / np.diff(edges)


@pytest.mark.njoy
@pytest.mark.tape
def test_fe56_jeff40_against_the_iaea_translation(tmp_path, fe56_jeff40_tape, njoy_exe,
                                                  g4ndl_jeff40_library):
    suite, report = readReconstructed(fe56_jeff40_tape, njoy=njoy_exe, tolerance=1e-3,
                                      cache_dir=tmp_path)
    assert any("ERR=0.001" in a for a in report.approximations)
    cs, fs, _ = encodeElastic(suite)
    iaea = g4ndl.open(g4ndl_jeff40_library)
    ref_cs, ref_fs = iaea.crossSection("Fe56"), iaea.elasticFinalState("Fe56")

    # Final state: identical but for two things the translation did, not kika
    # (roadmap §0, Phase 4): AWR cut to 6 digits, and the Legendre record at the
    # 45 MeV transition cut from NL=32 to NL=30.
    assert recordDifferences(ref_fs, fs) == [
        "targetMass: 55.4544 != 55.45443", "legendre[3959].coefficients differ"]

    # Cross section. Above the resonance region both are smooth: pointwise.
    top = max(r.domainMax for r in suite.resonances.resolved)
    above = ref_cs.energy > top
    rel = np.interp(ref_cs.energy[above], cs.energy, cs.sigma) / ref_cs.sigma[above] - 1
    assert np.abs(rel).max() < 1e-4
    # Inside it, narrow resonances sit between the IAEA's 7-digit print
    # collisions, so pointwise comparison measures their printing. Averages
    # over 1 000 log groups measure the reconstruction: within RECONR's ERR.
    edges = np.logspace(-5, np.log10(2e7), 1001)
    ours = _groupAverages(cs.energy, cs.sigma, edges, ref_cs.energy)
    theirs = _groupAverages(ref_cs.energy, ref_cs.sigma, edges, cs.energy)
    assert np.abs(ours / theirs - 1).max() < 2e-3


# ---------------------------------------------------------- Phase 10, inelastic

def _sameUpToTheTranslation(ours, theirs):
    """MF4/MF6 bodies equal but for what the IAEA translation does (roadmap Fase 10).

    It cuts Legendre series at NL = 30 (MF4, and the MF6 LANG=1 rows), prints
    incident energies of MF6 to 6 digits and the AWR to 6. Anything else is a
    difference.
    """
    if hasattr(theirs, "legendre"):            # dataType 4
        for block in ("legendre", "tabulated"):
            a, b = getattr(theirs, block), getattr(ours, block)
            assert (a is None) == (b is None)
            for ra, rb in zip(a.records if a else (), b.records if b else ()):
                assert ra.energy == rb.energy
                if block == "legendre":
                    n = len(ra.coefficients)
                    assert n == len(rb.coefficients) or n == 30
                    assert np.array_equal(ra.coefficients, rb.coefficients[:n])
                else:
                    assert np.array_equal(ra.mu, rb.mu)
                    assert np.array_equal(ra.probability, rb.probability)
        return
    assert len(theirs.products) == len(ours.products)
    for pa, pb in zip(theirs.products, ours.products):
        assert (pa.massCode, pa.distLaw) == (pb.massCode, pb.distLaw)
        assert np.array_equal(pa.yield_.y, pb.yield_.y)
        if pa.distLaw != 1:
            continue
        assert len(pa.body.energies) == len(pb.body.energies)
        for ea, eb in zip(pa.body.energies, pb.body.energies):
            assert ea.energy == pytest.approx(eb.energy, rel=5e-6)
            k = ea.rows.shape[1]
            assert k == eb.rows.shape[1] or k == 31
            assert np.array_equal(ea.rows, eb.rows[:, :k])


@pytest.mark.njoy
@pytest.mark.tape
def test_fe56_jeff40_inelastic_against_the_iaea_translation(tmp_path, fe56_jeff40_tape,
                                                            njoy_exe, g4ndl_jeff40_library):
    from kika.g4ndl.inelastic_encode import encodeInelastic

    suite, _ = readReconstructed(fe56_jeff40_tape, njoy=njoy_exe, tolerance=1e-3,
                                 cache_dir=tmp_path)
    # A tape states no G4NDL total; Geant4 needs it, so the encoder makes it.
    _, files, report = encodeInelastic(suite)
    assert any(m.startswith("inelastic: not in the suite") for m in report.warnings)
    iaea = g4ndl.open(g4ndl_jeff40_library)
    assert sorted(files) == iaea.inelasticChannels("Fe56")
    # MT600-649's MF4 is the proton's, and the adapter puts it on the proton.
    assert [p.pid for p in suite.findReactionByENDF_MT(600).outputChannel.products] == ["H1"]
    missing = set()
    for ch, ours in files.items():
        theirs = iaea.inelasticFinalState("Fe56", ch)
        mine = {(s.sfType, s.dataType): s.body for s in ours.sections}
        for s in theirs.sections:
            key = (s.sfType, s.dataType)
            if s.dataType >= 12:
                missing.add(s.dataType)       # photon production: not in the model
                assert key not in mine
                continue
            if s.dataType == 3:
                a, b = s.body.points, mine[key].points
                grid = a.x[(a.x >= b.x[0]) & (a.x <= b.x[-1])]
                assert np.abs(np.interp(grid, b.x, b.y) - np.interp(grid, a.x, a.y)).max() \
                    <= 1.5e-3 * a.y.max(), (ch, key)
                continue
            _sameUpToTheTranslation(mine[key], s.body)
    assert missing == {12, 14}
