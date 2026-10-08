"""The committed G4NDL fixtures are what they claim to be.

Phase 0 of the G4NDL roadmap: no reader exists yet, and these files are what it
will be written against. Each claim ``data/build_fixtures.py`` makes about a
fixture — "this one is tabulated", "this one has a repeated energy", "this
one's ``.z`` disagrees with its text" — is asserted here, with a token walker
written for this file alone. It deliberately shares no code with
``kika.g4ndl``: a fixture checked by the code under test proves nothing.

The numbers pinned below come from ``kika-workspace/myworkspace/G4NDL/
inventory.py`` run over the full libraries on 2026-10-05. Fe-56 is not a
committed fixture (size); ``test_full_libraries.py`` pins it on the real tree.
"""
from __future__ import annotations

import hashlib
import zlib
from pathlib import Path

import pytest

from kika.g4ndl.tests.data.build_fixtures import (
    CLAIMS, GAMMAS, INELASTIC, INELASTIC_ISOTOPES, REAL, SHA256, SUBDIRS, SYNTHETIC,
    SYNTHETIC_COMPRESSED,
)

DATA = Path(__file__).parent / "data"


def _tokens(path: Path) -> list[str]:
    raw = path.read_bytes()
    return (zlib.decompress(raw) if path.suffix == ".z" else raw).decode("ascii").split()


def _walk_fs(t: list[str]) -> dict:
    """Consume an elastic FS stream as G4ParticleHPElasticFS::Init does."""
    i = 2 if t[0] == "G4NDL" else 0
    out = {"header": t[:2] if i else None, "repFlag": int(t[i]),
           "frameFlag": int(t[i + 2]), "temps": [], "tempdep": [], "codes": []}
    i += 3

    def interp(i, key):
        nr = int(t[i])
        regions = [(int(t[i + 1 + 2 * k]), int(t[i + 2 + 2 * k])) for k in range(nr)]
        out["codes"] += [c for _, c in regions]
        out.setdefault(key, regions)
        return i + 1 + 2 * nr

    def block(i, tabulated, tag):
        ne = int(t[i])
        i = interp(i + 1, f"{tag}_regions")
        energies = []
        for _ in range(ne):
            out["temps"].append(float(t[i]))
            energies.append(float(t[i + 1]))
            out["tempdep"].append(int(t[i + 2]))
            n = int(t[i + 3])
            i += 4
            if tabulated:
                nr = int(t[i])
                out.setdefault("mu_codes", set()).update(
                    int(t[i + 2 + 2 * k]) for k in range(nr))
                i = interp(i, "mu_regions")
                i += 2 * n
            else:
                i += n
        out[f"{tag}_E"] = energies
        return i

    if out["repFlag"] == 0:
        out["frameFlag2"] = int(t[i])
        i += 1
    elif out["repFlag"] == 1:
        i = block(i, False, "leg")
    elif out["repFlag"] == 2:
        i = block(i, True, "tab")
    elif out["repFlag"] == 3:
        i = block(i, False, "leg")
        i = block(i, True, "tab")
    out["trailing"] = len(t) - i
    return out


def _walk_cs(t: list[str]) -> dict:
    i = 2 if t[0] == "G4NDL" else 0
    n = int(t[i + 2])
    pairs = t[i + 3:]
    return {"header": t[:2] if i else None, "bookkeeping": t[i:i + 2], "n": n,
            "values": len(pairs), "sigma": [float(x) for x in pairs[1::2]]}


def _fs(lib, stem, z=True):
    return _walk_fs(_tokens(DATA / lib / "Elastic/FS" / (stem + (".z" if z else ""))))


def _repeats(e):
    return sum(1 for a, b in zip(e, e[1:]) if a == b)


# ---------------------------------------------------------------- real files

@pytest.mark.parametrize("key", sorted(SHA256))
def test_real_files_are_byte_for_byte_the_recorded_ones(key):
    assert hashlib.sha256((DATA / key).read_bytes()).hexdigest() == SHA256[key]


def test_every_real_isotope_has_both_files_and_a_hash():
    expected = {f"{lib}/{sub}/{stem}.z" for lib, stem in REAL for sub in SUBDIRS}
    assert expected == {k for k in SHA256 if not k.startswith("inelastic/")}


def test_every_inelastic_fixture_has_a_hash():
    expected = {f"inelastic/{lib}/Inelastic/{sub}/{name}" for lib, sub, name in INELASTIC}
    expected |= {f"inelastic/{lib}/Inelastic/Gammas/{name}" for lib, name in GAMMAS}
    for lib, stem in INELASTIC_ISOTOPES:
        expected |= {k for k in SHA256 if k.startswith(f"inelastic/{lib}/") and stem in k}
    assert expected == {k for k in SHA256 if k.startswith("inelastic/")}
    assert sum((DATA / k).stat().st_size for k in expected) < 200_000


@pytest.mark.parametrize("lib,stem", sorted(REAL))
def test_real_files_parse_to_the_last_token_and_carry_no_header(lib, stem):
    fs = _fs(lib, stem)
    cs = _walk_cs(_tokens(DATA / lib / "Elastic/CrossSection" / f"{stem}.z"))
    assert fs["trailing"] == 0
    assert cs["values"] == 2 * cs["n"]
    # Neither library writes the optional header; it is tested synthetically.
    assert fs["header"] is None and cs["header"] is None
    # Fe-56's "0 0" is the norm, not an exception: no MT in the bookkeeping.
    assert cs["bookkeeping"] == ["0", "0"]
    assert fs["frameFlag"] == 2
    assert set(fs["tempdep"]) == {0} and set(fs["temps"]) == {0.0}


def test_each_representation_has_a_real_case():
    assert _fs("JEFF-4.0", "1_1_Hydrogen")["repFlag"] == 1
    assert _fs("JEFF-4.0", "2_3_Helium")["repFlag"] == 2
    assert _fs("JEFF-4.0", "27_58m1_Cobalt")["repFlag"] == 3
    co = _fs("JEFF-4.0", "27_58m1_Cobalt")
    # The boundary energy appears in both blocks: last Legendre == first table.
    assert co["leg_E"][-1] == co["tab_E"][0]


def test_hydrogen_repeats_an_incident_energy():
    assert _repeats(_fs("JEFF-4.0", "1_1_Hydrogen")["leg_E"]) == 1


def test_nitrogen_has_two_energy_regions_linlog_then_linlin():
    assert _fs("JEFF-4.0", "7_14_Nitrogen")["leg_regions"] == [(14, 3), (614, 2)]


def test_carbon_interpolates_mu_loglin():
    fs = _fs("JEFF-4.0", "6_12_Carbon")
    # LOGLIN in mu means ln p linear in mu: it needs p > 0 at every node.
    assert fs["mu_codes"] == {4} and fs["tab_regions"] == [(17, 2)]


# ----------------------------------------------------------- synthetic files

def test_every_synthetic_isotope_has_a_claim():
    assert set(CLAIMS) == set(SYNTHETIC)


def test_synthetic_files_on_disk_are_the_builder_text():
    root = DATA / "synthetic"
    for stem, (fs, cs) in SYNTHETIC.items():
        assert (root / "Elastic/FS" / stem).read_text("ascii") == fs
        assert (root / "Elastic/CrossSection" / stem).read_text("ascii") == cs
    for stem, text in SYNTHETIC_COMPRESSED.items():
        assert zlib.decompress((root / "Elastic/FS" / f"{stem}.z").read_bytes()).decode() == text


def _synthetic(claim):
    stem = next(s for s, c in CLAIMS.items() if c == claim)
    return stem, DATA / "synthetic" / "Elastic"


def test_repflag0_with_header():
    stem, root = _synthetic("repflag0_with_header")
    fs = _walk_fs(_tokens(root / "FS" / stem))
    assert fs["header"] == ["G4NDL", "synthetic"]
    assert fs["repFlag"] == 0 and fs["frameFlag2"] == 2 and fs["trailing"] == 0
    assert _walk_cs(_tokens(root / "CrossSection" / stem))["header"] == ["G4NDL", "synthetic"]


def test_lab_frame_tempdep():
    stem, root = _synthetic("lab_frame_tempdep")
    fs = _walk_fs(_tokens(root / "FS" / stem))
    assert fs["frameFlag"] == 1 and 1 in fs["tempdep"] and fs["temps"][0] == 293.6


def test_z_and_text_disagree():
    stem, root = _synthetic("z_and_text_disagree")
    assert _walk_fs(_tokens(root / "FS" / stem))["repFlag"] == 1
    assert _walk_fs(_tokens(root / "FS" / f"{stem}.z"))["repFlag"] == 0


def test_unknown_interpolation():
    stem, root = _synthetic("unknown_interpolation")
    assert 6 in _walk_fs(_tokens(root / "FS" / stem))["codes"]


def test_truncated_fs():
    stem, root = _synthetic("truncated_fs")
    assert _walk_fs(_tokens(root / "FS" / stem))["trailing"] < 0


def test_negative_cross_section():
    stem, root = _synthetic("negative_cross_section")
    assert min(_walk_cs(_tokens(root / "CrossSection" / stem))["sigma"]) < 0


def test_integer_as_float():
    stem, root = _synthetic("integer_as_float")
    assert _tokens(root / "FS" / stem)[3] == "2.0"


def test_trailing_tokens():
    stem, root = _synthetic("trailing_tokens")
    assert _walk_fs(_tokens(root / "FS" / stem))["trailing"] == 3


def test_transition_mismatch():
    stem, root = _synthetic("transition_mismatch")
    fs = _walk_fs(_tokens(root / "FS" / stem))
    assert fs["repFlag"] == 3 and fs["leg_E"][-1] != fs["tab_E"][0]


def test_cs_short_count():
    stem, root = _synthetic("cs_short_count")
    cs = _walk_cs(_tokens(root / "CrossSection" / stem))
    assert cs["values"] < 2 * cs["n"]
