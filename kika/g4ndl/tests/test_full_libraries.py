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
