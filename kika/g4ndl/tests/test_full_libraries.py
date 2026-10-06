"""The index and the parsers against the two whole libraries. ``tape``-marked: needs ``KIKA_G4NDL``.

The counts are the ones ``kika-workspace/myworkspace/G4NDL/inventory.py``
measured on 2026-10-05 with code that shares nothing with ``kika.g4ndl``.
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

import kika.g4ndl as g4ndl
from kika.g4ndl import IsotopeKey, IsotopeNotFoundError

#: Explicit: two tests reach their library through ``request.getfixturevalue``,
#: which the conftest auto-marker cannot see.
pytestmark = pytest.mark.tape


@pytest.mark.parametrize("fixture,n_iso", [
    ("g4ndl_jeff40_library", 593),
    ("g4ndl_g4ndl471_library", 560),
])
def test_index_counts(request, fixture, n_iso):
    lib = g4ndl.open(request.getfixturevalue(fixture))
    assert len(lib) == n_iso
    assert lib.duplicates == [] and lib.unindexed == []


def test_g4ndl_ships_jendl_he_and_it_stays_out(g4ndl_g4ndl471_library):
    lib = g4ndl.open(g4ndl_g4ndl471_library)
    assert "JENDL_HE" in lib.presentTopLevel()
    assert all(f.path.parent.parent.parent == lib.root
               for f in (lib.locate(k, "Elastic/FS") for k in lib.isotopes()))


@pytest.mark.parametrize("fixture,n_pairs", [
    ("g4ndl_jeff40_library", 37706),
    ("g4ndl_g4ndl471_library", 40199),
])
def test_fe56_cross_section(request, fixture, n_pairs):
    stream = g4ndl.open(request.getfixturevalue(fixture)).tokens("Fe56", "Elastic/CrossSection")
    assert stream.header is None
    assert (stream.int("a"), stream.int("b")) == (0, 0)
    n = stream.int("N")
    stream.pairs(n, "pairs")
    stream.expectEnd()
    assert n == n_pairs


def test_a_removed_isotope_is_not_substituted(g4ndl_g4ndl471_library):
    lib = g4ndl.open(g4ndl_g4ndl471_library)
    # Argon-36/38 were removed in 4.7.1 (README); natural argon is not handed out.
    assert IsotopeKey(18, 36) not in lib.isotopes()
    with pytest.raises(IsotopeNotFoundError):
        lib.locate("Ar36", "Elastic/FS")


# ---------------------------------------------------------- Phases 2 and 3

@pytest.mark.parametrize("fixture,reps,cs_repeat,fs_repeat", [
    ("g4ndl_jeff40_library", {1: 61, 2: 6, 3: 526}, 580, 11),
    ("g4ndl_g4ndl471_library", {1: 192, 2: 15, 3: 353}, 543, 7),
])
def test_every_elastic_file_parses_to_its_last_token(request, fixture, reps,
                                                     cs_repeat, fs_repeat):
    """Every check the parsers make holds on every real file (spec §7 counts)."""
    lib = g4ndl.open(request.getfixturevalue(fixture))
    seen = collections.Counter()
    n_cs = n_fs = 0
    for key in lib.isotopes():
        cs = lib.crossSection(key)
        fs = lib.elasticFinalState(key)
        seen[fs.repFlag] += 1
        n_cs += bool(np.any(np.diff(cs.energy) == 0))
        n_fs += any(np.any(np.diff(b.energies) == 0)
                    for b in (fs.legendre, fs.tabulated) if b is not None)
        assert fs.frameFlag == 2 and not fs.tempdeps.any() and not fs.temperatures.any()
    assert dict(seen) == reps
    assert (n_cs, n_fs) == (cs_repeat, fs_repeat)


@pytest.mark.parametrize("fixture,n_pairs,n_leg,n_tab,e_trans", [
    ("g4ndl_jeff40_library", 37706, 3960, 19, 45.0e6),
    ("g4ndl_g4ndl471_library", 40199, 1782, 28, 20.0e6),
])
def test_fe56_records(request, fixture, n_pairs, n_leg, n_tab, e_trans):
    lib = g4ndl.open(request.getfixturevalue(fixture))
    cs = lib.crossSection("Fe56")
    assert len(cs) == n_pairs and cs.bookkeeping == (0, 0)
    fs = lib.elasticFinalState("Fe56")
    assert fs.repFlag == 3 and (len(fs.legendre), len(fs.tabulated)) == (n_leg, n_tab)
    assert fs.transitionEnergy == pytest.approx(e_trans, rel=1e-12)
    # The mass ratio, not A.
    assert 55 < fs.targetMass < 56


# ----------------------------------------------------------------- Phase 4

@pytest.mark.parametrize("fixture", ["g4ndl_jeff40_library", "g4ndl_g4ndl471_library"])
def test_every_isotope_decodes_into_the_model(request, fixture):
    lib = g4ndl.open(request.getfixturevalue(fixture))
    for key in lib.isotopes():
        suite = lib.read(key)
        assert suite.reactions.ENDF_MTs == [2]
        # The library root holds more than Elastic/, and the report says so.
        assert suite.report.unsupported


def _records(angular):
    """``(kind, E, values)`` per incident energy, Regions2d flattened."""
    out = []
    stack = [angular]
    while stack:
        node = stack.pop(0)
        if hasattr(node, "function2ds"):
            stack[:0] = list(node.function2ds)
            continue
        for f in node.function1ds:
            if hasattr(f, "coefficients"):
                out.append(("L", f.outerDomainValue, np.asarray(f.coefficients)))
            else:
                mu, p, _ = f.toEndfRegions()
                out.append(("T", f.outerDomainValue, np.concatenate([mu, p])))
    return out


@pytest.mark.parametrize("target,tape,truncated", [
    # The translation cuts Fe-56's last Legendre record (45 MeV, the
    # transition) from NL=32 to NL=30: a_31 = 8.5e-9 and a_32 = 0 are dropped.
    # Everything else, 3 978 records, is bit-for-bit the tape.
    ("Fe56", "fe56_jeff40_tape", {3959: 30}),
    ("U238", "u238_tape", {}),
])
def test_jeff40_angular_is_the_tape_mf4(request, g4ndl_jeff40_library, target, tape,
                                        truncated):
    """Two independent roads to one object: G4NDL through kika.g4ndl, and the
    evaluation it was translated from through the ENDF adapter."""
    from kika.endf.model_adapter.angular import decodeMF4MT
    from kika.endf.read_endf import read_endf

    suite = g4ndl.open(g4ndl_jeff40_library).read(target)
    g = suite.reactions[2].outputChannel.products.byPid("n")[0].distribution["eval"]
    endf = read_endf(str(request.getfixturevalue(tape)), mf_numbers=[4])
    e, _, _ = decodeMF4MT(endf.mf[4].mt[2])
    assert g.productFrame == e.productFrame
    rg, re_ = _records(g.angular), _records(e.angular)
    assert len(rg) == len(re_)
    for i, (a, b) in enumerate(zip(rg, re_)):
        assert a[0] == b[0] and a[1] == b[1]
        if i in truncated:
            n = truncated[i] + 1
            assert len(a[2]) == n and a[2].tolist() == b[2][:n].tolist()
        else:
            assert a[2].tolist() == b[2].tolist()


# ----------------------------------------------------------------- Phase 5

@pytest.mark.parametrize("fixture,negative,unnormalised", [
    # (records, isotopes). Library anomalies, measured 2026-10-06, reported and
    # never corrected. |int p - 1| > 1e-6 on tables is mostly the evaluations'
    # own normalisation at the 1e-5..1e-4 level, except where noted below.
    ("g4ndl_jeff40_library", (75, 12), (284, 98)),
    ("g4ndl_g4ndl471_library", (36, 6), (895, 65)),
])
def test_physics_anomalies_of_each_library(request, fixture, negative, unnormalised):
    from kika.g4ndl.physics import checkElastic

    lib = g4ndl.open(request.getfixturevalue(fixture))
    neg, norm = [], []
    for key in lib.isotopes():
        check = checkElastic(lib.read(key))
        assert not check.byKind("moments") and not check.byKind("nonfinite")
        neg += [(check.target, f) for f in check.byKind("negative")]
        norm += [(check.target, f) for f in check.byKind("normalisation")]
        # Every Legendre record has a_0 = 1: the import adds no factor.
        assert all(f.block == "table" for f in check.byKind("normalisation"))
    assert (len(neg), len({t for t, _ in neg})) == negative
    assert (len(norm), len({t for t, _ in norm})) == unnormalised


def test_jeff40_hf178m2_ships_empty_tables(g4ndl_jeff40_library):
    """A translation defect: all 15 tables above 30 MeV are p = 0 at mu = -1, +1."""
    from kika.g4ndl.physics import checkElastic

    check = checkElastic(g4ndl.open(g4ndl_jeff40_library).read("Hf178m2"))
    empty = [f for f in check.byKind("normalisation") if f.value == -1.0]
    assert len(empty) == 15 and empty[0].energy == 30.0e6


def test_jeff40_fe56_negative_lobe(g4ndl_jeff40_library):
    """Fe-56 JEFF-4.0's Legendre density dips below zero near mu = -0.2 at
    1.557-1.560 MeV, and at mu = -1 at 2.414 MeV. The file's, not kika's."""
    from kika.g4ndl.physics import checkElastic

    neg = checkElastic(g4ndl.open(g4ndl_jeff40_library).read("Fe56")).byKind("negative")
    assert [f.energy for f in neg] == [1.557e6, 1.558e6, 1.559e6, 1.560e6, 2.414e6]
    assert min(f.value for f in neg) == pytest.approx(-0.0261, abs=1e-4)


# ----------------------------------------------------------------- Phase 6

@pytest.mark.parametrize("fixture", ["g4ndl_jeff40_library", "g4ndl_g4ndl471_library"])
def test_every_pair_is_a_fixed_point_of_read_model_write_read(request, fixture):
    """Roadmap §5 Fase 6: values, regions, repeated energies and their order.

    Through the text, which is what Geant4 reads: records → model → records →
    text → strict parser must give the records the library has, exactly.
    """
    from kika.g4ndl.decode import decodeElastic
    from kika.g4ndl.encode import (encodeElastic, formatCrossSection,
                                   formatElasticFS, recordDifferences)
    from kika.g4ndl.parse import parse_cross_section, parse_elastic_fs
    from kika.g4ndl.tokens import TokenStream

    lib = g4ndl.open(request.getfixturevalue(fixture))
    bad = {}
    for key in lib.isotopes():
        cs, fs = lib.crossSection(key), lib.elasticFinalState(key)
        suite, _ = decodeElastic(cs, fs, key, library=lib)
        cs2, fs2, report = encodeElastic(suite)
        back_cs = parse_cross_section(TokenStream(formatCrossSection(cs2)))
        back_fs = parse_elastic_fs(TokenStream(formatElasticFS(fs2)))
        diffs = recordDifferences(cs, back_cs) + recordDifferences(fs, back_fs)
        if diffs or report.warnings:
            bad[str(key)] = (diffs + report.warnings)[:3]
    assert bad == {}


def test_patch_a_whole_library(tmp_path, g4ndl_g4ndl471_library):
    """The collaborator's loop on the real G4NDL 4.7.1: Pb208, hard-linked copy."""
    import os

    base = g4ndl_g4ndl471_library
    lib = g4ndl.open(base)
    targets = [lib.locate("Pb208", s).path for s in ("Elastic/CrossSection", "Elastic/FS")]
    before = {p: p.read_bytes() for p in targets}
    suite = lib.read("Pb208")
    suite.reactions[2].crossSection["recon"].ys[:] *= 0.97
    out = tmp_path / "G4NDL4.7.1-Pb208"
    result = g4ndl.patch_elastic(base, suite, out, share="hardlink")
    # Removed here, not by pytest: its cleanup clears the read-only bit to delete
    # a file, and on a hard link that bit is the base library's.
    from kika.g4ndl.patch import _removeTree
    try:
        _checkPatched(base, lib, out, result, targets, before)
    finally:
        _removeTree(out, base)
    assert all(not os.access(p, os.W_OK) for p in targets)  # still read-only


def _checkPatched(base, lib, out, result, targets, before):
    import os

    assert {p: p.read_bytes() for p in targets} == before   # the base is untouched
    files = [p for p in base.rglob("*") if p.is_file()]
    written = {out / r for r in result.replaced}
    for p in files:
        mirror = out / p.relative_to(base)
        if mirror in written:
            assert not os.path.samefile(p, mirror)
        else:                                               # identical: same file
            assert os.path.samefile(p, mirror), p
    patched = g4ndl.open(out)
    assert np.array_equal(patched.crossSection("Pb208").sigma,
                          lib.crossSection("Pb208").sigma * 0.97)
