"""Fission against the two whole libraries and the JEFF-4.0 tapes. ``tape``-marked.

Every ``Fission/`` file is read to its last token and is a text fixed point,
and every fission isotope is read into the model, written back from it and
compared record for record (the model fixed point). Then the model itself is
compared with the ENDF adapter's on the same JEFF-4.0 evaluation: the IAEA
translation and the tape land on the same nodes with the same numbers, to the
translation's print rounding. The counts are the census of 2026-10-08.
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

import kika
import kika.g4ndl as g4ndl
from kika.g4ndl.encode import recordDifferences
from kika.g4ndl.fission import (
    CHANCES, fissionDifferences, formatChanceFission, formatFissionFS, formatFragmentYields,
    parse_chance_fission, parse_fission_fs, parse_fragment_yields,
)
from kika.g4ndl.fission_model import encodeFission
from kika.g4ndl.tokens import TokenStream

pytestmark = pytest.mark.tape

#: library -> (fission isotopes, chance files {dir: (files, with a final state)},
#: FF files, sections kept as text {(infoType, dataType): count}).
CENSUS = {
    # The photons (1, 12/14/15) reach the model since D10-2 (2026-10-09).
    "g4ndl_g4ndl471_library": (70, {"FC": (38, 5), "SC": (38, 5), "TC": (36, 3), "LC": (35, 2)},
                               11, {(3, 5): 1}),
    "g4ndl_jeff40_library": (75, {"FC": (33, 1), "SC": (33, 1), "TC": (33, 1), "LC": (33, 1)},
                             0, {}),
}


@pytest.mark.parametrize("fixture", sorted(CENSUS))
def test_every_fission_isotope_reads_and_is_a_fixed_point(request, fixture):
    lib = g4ndl.open(request.getfixturevalue(fixture))
    n, chanceCounts, nff, keptCounts = CENSUS[fixture]
    keys = lib.isotopes("fission")
    assert len(keys) == n
    chances = collections.Counter()
    finalStates = collections.Counter()
    kept = collections.Counter()
    nFF = 0
    for key in keys:
        fs = lib.fissionFinalState(key)
        assert fissionDifferences(fs, parse_fission_fs(TokenStream(formatFissionFS(fs)))) == [], key
        suite = lib.read(key, processes=["fission"])
        cs, fsOut, chanceOut, ffOut, _ = encodeFission(suite)
        assert recordDifferences(lib.fissionCrossSection(key), cs) == [], key
        assert fissionDifferences(fs, fsOut) == [], key
        assert list(chanceOut) == lib.fissionChances(key), key
        for c in lib.fissionChances(key):
            record = lib.chanceFission(key, c)
            chances[c] += 1
            finalStates[c] += record.hasFinalState
            text = formatChanceFission(record)
            assert fissionDifferences(record, parse_chance_fission(TokenStream(text), c)) == []
            assert fissionDifferences(record, chanceOut[c]) == [], (key, c)
        ff = lib.fragmentYields(key)
        assert (ff is None) == (ffOut is None), key
        if ff is not None:
            nFF += 1
            text = formatFragmentYields(ff)
            assert fissionDifferences(ff, parse_fragment_yields(TokenStream(text))) == []
            assert fissionDifferences(ff, ffOut) == [], key
        for e in suite.findReactionByENDF_MT(18).provenance.sections:
            if e.get("verbatim") is not None:
                kept[(e["infoType"], e["dataType"])] += 1
    assert {c: (chances[c], finalStates[c]) for c in CHANCES} == chanceCounts
    assert nFF == nff and dict(kept) == keptCounts


def _jeff40(tape_dir, name):
    path = tape_dir / name
    if not path.is_file():
        pytest.skip(f"{path} not found")
    return kika.read(str(path), format="endf")


def _nodes(suite):
    """The fission nodes both adapters fill, as numbers at a few energies."""
    e = np.array([0.0253, 1.0e6, 1.4e7])
    r = suite.findReactionByENDF_MT(18)
    (n,) = [p for p in r.outputChannel.products if p.pid == "n"]
    data = r.outputChannel.fissionFragmentData
    sums = {m.ENDF_MT: np.asarray(m.multiplicity.form.evaluate(e)) for m in
            suite.sums.multiplicitySums}
    families = [(f.label, f.rate.value, f.rate.unit,
                 type(f.product.distribution["eval"].energy).__name__,
                 np.asarray(f.product.multiplicity.form.evaluate(e)))
                for f in data.delayedNeutrons]
    release = data.fissionEnergyReleases[0]
    return dict(prompt=np.asarray(n.multiplicity.form.evaluate(e)), sums=sums,
                families=families,
                spectrum=type(n.distribution["eval"].energy).__name__,
                angular=type(n.distribution["eval"].angular).__name__,
                release={name: float(np.asarray(f.evaluate(0.0253)))
                         for name, f in release.terms()},
                chances=[x.id.ENDF_MT for x in suite.reactions if x.id.ENDF_MT in (19, 20, 21, 38)])


@pytest.mark.parametrize("tape,target", [
    ("n_92-U-235g.jeff", "U235"),    # LF=1 prompt, eight LF=1 families
    ("n_90-Th-230g.jeff", "Th230"),  # six LF=5 families, four chances
    ("n_94-Pu-239g.jeff", "Pu239"),
])
def test_the_model_is_the_endf_adapters(fe56_jeff40_tape, g4ndl_jeff40_library, tape, target):
    ours = _nodes(g4ndl.open(g4ndl_jeff40_library).read(target, processes=["fission"]))
    theirs = _nodes(_jeff40(fe56_jeff40_tape.parent, tape))
    # The same tables, but for one point the translation adds to some grids
    # (Th-230's prompt nu-bar has 12 to the tape's 11): ~1e-9 at thermal.
    assert np.allclose(ours["prompt"], theirs["prompt"], rtol=1e-8, atol=0)
    assert sorted(ours["sums"]) == sorted(theirs["sums"]) == [452, 455]
    for mt in (452, 455):
        assert np.allclose(ours["sums"][mt], theirs["sums"][mt], rtol=1e-8, atol=0)
    assert [f[:4] for f in ours["families"]] == [f[:4] for f in theirs["families"]]
    # Above 1 eV only: Pu-239's p_k(E) declares INT=1 from 1e-5 eV to 5 MeV and
    # steps at 1 eV. G4NDL reads it lin-lin, as Geant4 evaluates it (roadmap
    # D10-3), ENDF a histogram, so at thermal the two differ by 0.25 %, by design.
    for a, b in zip(ours["families"], theirs["families"]):
        assert np.allclose(a[4][1:], b[4][1:], rtol=1e-6, atol=0)
    assert (ours["spectrum"], ours["angular"]) == (theirs["spectrum"], theirs["angular"])
    assert ours["chances"] == theirs["chances"]
    # The translation prints the energy release to 6 digits; the tape has 9 to 11.
    for name, value in theirs["release"].items():
        assert ours["release"][name] == pytest.approx(value, rel=5e-6, abs=1.0), name
