"""Phase 6: the model back to G4NDL, and one isotope replaced in a library copy.

Acceptance (roadmap §5 Fase 6): ``read → model → write → read`` keeps values,
regions, repeated energies and their order — checked here on every real
fixture as **record equality**, array for array — and replacing an isotope
changes exactly its two files and never the base. The same gates on all
1 152 real pairs are in ``test_full_libraries.py``; Geant4 reading the output
is in the harness (``kika-workspace/myworkspace/G4NDL/harness``).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zlib
from pathlib import Path

import numpy as np
import pytest

import kika
import kika.g4ndl as g4ndl
from kika.g4ndl import G4NDLUnsupportedError
from kika.g4ndl.encode import (
    KEEP, encodeElastic, formatCrossSection, formatElasticFS, recordDifferences,
    targetKey, writeElastic,
)
from kika.g4ndl.names import file_name
from kika.g4ndl.parse import parse_cross_section, parse_elastic_fs
from kika.g4ndl.patch import MANIFEST_NAME
from kika.g4ndl.tokens import TokenStream
from kika.nuclear_data.model import (
    AngularTwoBody, CrossSection, Frame, Isotropic2d, Legendre, XYs1d, XYs2d, angularAxes,
)
from kika.nuclear_data.model.enums import Interpolation

DATA = Path(__file__).parent / "data"
JEFF = g4ndl.open(DATA / "JEFF-4.0")
G4 = g4ndl.open(DATA / "G4NDL-4.7.1")
SYNTH = g4ndl.open(DATA / "synthetic")

#: Every fixture the reader accepts: (library, target).
READABLE = ([(JEFF, k) for k in JEFF.isotopes()] + [(G4, k) for k in G4.isotopes()]
            + [(SYNTH, t) for t in ("H1", "Li6", "Be9")])


def _distribution(suite):
    return suite.reactions[2].outputChannel.products.byPid("n")[0].distribution


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(root: Path):
    return {p.relative_to(root).as_posix(): _sha(p) for p in root.rglob("*") if p.is_file()}


def _modes(root: Path):
    return {p.relative_to(root).as_posix(): p.stat().st_mode for p in root.rglob("*") if p.is_file()}


# ------------------------------------------------------------- fixed point

@pytest.mark.parametrize("lib,key", READABLE, ids=lambda x: str(x) if not hasattr(x, "root") else x.root.name)
def test_read_model_write_read_is_a_fixed_point(lib, key):
    cs, fs = lib.crossSection(key), lib.elasticFinalState(key)
    cs2, fs2, report = encodeElastic(lib.read(key))
    assert recordDifferences(cs, cs2) == [] and recordDifferences(fs, fs2) == []
    # ... and through the text, which is what Geant4 reads.
    back_cs = parse_cross_section(TokenStream(formatCrossSection(cs2)))
    back_fs = parse_elastic_fs(TokenStream(formatElasticFS(fs2)))
    assert recordDifferences(cs, back_cs) == [] and recordDifferences(fs, back_fs) == []
    assert report.warnings == []


@pytest.mark.parametrize("compressed", [False, True])
def test_kika_write_door_and_read_back(tmp_path, compressed):
    suite = JEFF.read("N14")
    report = kika.write(suite, tmp_path, format="g4ndl", compressed=compressed)
    assert report.isEmpty
    suffix = ".z" if compressed else ""
    for sub in ("Elastic/CrossSection", "Elastic/FS"):
        path = tmp_path / sub / f"7_14_Nitrogen{suffix}"
        assert path.is_file()
        raw = path.read_bytes()
        (zlib.decompress(raw) if compressed else raw).decode("ascii")
    again = g4ndl.open(tmp_path)
    assert recordDifferences(JEFF.crossSection("N14"), again.crossSection("N14")) == []
    assert recordDifferences(JEFF.elasticFinalState("N14"), again.elasticFinalState("N14")) == []


def test_the_reader_gives_the_same_suite_back(tmp_path):
    kika.write(JEFF.read("C12"), tmp_path, format="g4ndl")
    suite = kika.read(tmp_path, format="g4ndl", target="C12")
    xs = suite.reactions[2].crossSection["recon"]
    assert np.array_equal(xs.xs, JEFF.crossSection("C12").energy)


# ------------------------------------------------------------- the numbers

def test_floats_never_round_and_integers_stay_integers():
    suite = JEFF.read("He3")
    xs = suite.reactions[2].crossSection["recon"]
    xs.ys[:] = xs.ys / 3.0          # values with 17 significant digits
    cs, fs, _ = encodeElastic(suite, targetMass=1.0 / 3.0)
    text_cs, text_fs = formatCrossSection(cs), formatElasticFS(fs)
    back = parse_cross_section(TokenStream(text_cs))
    assert np.array_equal(back.sigma, xs.ys)
    assert parse_elastic_fs(TokenStream(text_fs)).targetMass == 1.0 / 3.0
    tokens = text_fs.split()
    # repFlag, then frameFlag and NE/NR/NBT/INT: written as integers, never "2.0".
    assert tokens[0] == "2" and tokens[2] == "2" and "." not in tokens[3]


def test_repeated_energies_and_their_order_survive():
    cs = JEFF.crossSection("Co58m1")
    cs2, _, _ = encodeElastic(JEFF.read("Co58m1"))
    assert np.array_equal(cs.energy, cs2.energy)
    fs = JEFF.elasticFinalState("H1")       # one repeated incident energy
    _, fs2, _ = encodeElastic(JEFF.read("H1"))
    assert np.array_equal(fs.legendre.energies, fs2.legendre.energies)
    assert np.any(np.diff(fs2.legendre.energies) == 0)


# ------------------------------------------------- what the model decides

def _tabulatedFrom(suite, energies, mu, p):
    axes = angularAxes()
    functions = []
    for i, e in enumerate(energies):
        f = XYs1d(xs=np.asarray(mu, float), ys=np.asarray(p, float),
                  interpolation=Interpolation.linlin)
        f.outerDomainValue, f.index = float(e), i
        functions.append(f)
    return AngularTwoBody(angular=XYs2d(function1ds=functions, axes=axes),
                          productFrame=Frame.centerOfMass)


def test_repflag_comes_from_the_shape_not_the_provenance():
    """The collaborator's case: a measured p(mu) table in place of Legendre."""
    suite = JEFF.read("H1")
    assert suite.provenance.repFlag == 1
    _distribution(suite)["eval"] = _tabulatedFrom(
        suite, [1.0e-5, 2.0e7], [-1.0, 0.0, 1.0], [0.4, 0.5, 0.6])
    _, fs, report = encodeElastic(suite)
    assert fs.repFlag == 2 and fs.legendre is None
    assert len(fs.tabulated.records) == 2
    # The Legendre block's T/tempdep no longer apply; the table gets zeros.
    assert all(r.temperature == 0.0 and r.tempdep == 0 for r in fs.tabulated.records)

    _distribution(suite)["eval"] = Isotropic2d(productFrame=Frame.lab)
    _, fs, _ = encodeElastic(suite)
    assert (fs.repFlag, fs.frameFlag, fs.frameFlag2) == (0, 1, 1)


def test_isotropic_keeps_the_first_frame_flag_while_the_second_agrees():
    suite = SYNTH.read("H1")          # repFlag=0, frame flags as read
    fs = SYNTH.elasticFinalState("H1")
    _, fs2, _ = encodeElastic(suite)
    assert (fs2.frameFlag, fs2.frameFlag2) == (fs.frameFlag, fs.frameFlag2)


def test_histogram_is_refused_in_mu_and_in_energy():
    suite = JEFF.read("H1")
    _distribution(suite)["eval"] = _tabulatedFrom(suite, [1.0e-5, 2.0e7], [-1.0, 1.0], [0.5, 0.5])
    form = _distribution(suite)["eval"].angular
    form.function1ds[0].interpolation = Interpolation.flat
    with pytest.raises(G4NDLUnsupportedError, match="code 1"):
        encodeElastic(suite)
    form.function1ds[0].interpolation = Interpolation.linlin
    form.interpolation = Interpolation.flat
    with pytest.raises(G4NDLUnsupportedError, match="code 1"):
        encodeElastic(suite)


def test_a_legendre_a0_other_than_one_is_refused():
    suite = JEFF.read("H1")
    first = _distribution(suite)["eval"].angular.function1ds[0]
    assert isinstance(first, Legendre)
    first.coefficients[0] = 2.0
    with pytest.raises(G4NDLUnsupportedError, match="a_0"):
        encodeElastic(suite)


def test_what_the_grammar_refuses_is_not_written(tmp_path):
    suite = JEFF.read("He3")
    xs = suite.reactions[2].crossSection["recon"]
    xs.ys[3] = -1.0
    with pytest.raises(G4NDLUnsupportedError, match="negative cross section"):
        writeElastic(suite, tmp_path)
    assert not any(tmp_path.rglob("*.*")) and not (tmp_path / "Elastic").exists()


def test_cross_section_and_mass_are_never_guessed():
    suite = JEFF.read("He3")
    only_eval = CrossSection()
    only_eval["eval"] = suite.reactions[2].crossSection["recon"]
    suite.reactions[2].crossSection = only_eval
    with pytest.raises(G4NDLUnsupportedError, match="crossSectionLabel='eval'"):
        encodeElastic(suite)
    cs, _, _ = encodeElastic(suite, crossSectionLabel="eval")
    assert len(cs) == len(JEFF.crossSection("He3"))

    suite = JEFF.read("He3")
    suite.provenance.targetMass = None
    suite.reactions[2].provenance = suite.provenance
    with pytest.raises(G4NDLUnsupportedError, match="targetMass"):
        encodeElastic(suite)
    assert encodeElastic(suite, targetMass=2.99)[1].targetMass == 2.99


def test_headers():
    suite = SYNTH.read("Be9")         # the fixture with a G4NDL header
    fs = SYNTH.elasticFinalState("Be9")
    assert encodeElastic(suite)[1].header == fs.header
    assert encodeElastic(suite, header=None)[1].header is None
    cs, fs2, _ = encodeElastic(suite, header="kika-test")
    assert cs.header == fs2.header == ("G4NDL", "kika-test")
    assert formatElasticFS(fs2).startswith("G4NDL kika-test\n")
    with pytest.raises(ValueError):
        encodeElastic(suite, header="two words")


def test_names():
    assert targetKey(JEFF.read("Co58m1")) == g4ndl.IsotopeKey(27, 58, 1)
    assert file_name(targetKey(G4.read("Cnat"))) == "6_nat_Carbon"
    assert file_name(g4ndl.IsotopeKey(4, 9)) == "4_9_Berylium"   # Geant4's spelling


def test_a_stale_twin_is_removed_and_reported(tmp_path):
    suite = JEFF.read("He3")
    writeElastic(suite, tmp_path, compressed=True)
    report = writeElastic(suite, tmp_path, compressed=False)
    assert not list(tmp_path.rglob("*.z"))
    assert len(report.warnings) == 2 and "shadowed" in report.warnings[0]


# ------------------------------------------------------------- patch_elastic

@pytest.fixture
def base(tmp_path):
    """A copy of the JEFF fixtures, read-only like the distributed libraries.

    Read-only is the case that matters: Windows refuses to replace such a file,
    and its read-only bit is shared by every hard link to it.
    """
    root = tmp_path / "base"
    shutil.copytree(DATA / "JEFF-4.0", root)
    for p in root.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    return root


@pytest.mark.parametrize("share", ["copy", "hardlink"])
def test_patch_changes_exactly_two_files_and_never_the_base(tmp_path, base, share):
    before, modes = _tree(base), _modes(base)
    suite = g4ndl.open(base).read("C12")
    suite.reactions[2].crossSection["recon"].ys[:] *= 1.05
    out = tmp_path / "patched"
    result = g4ndl.patch_elastic(base, suite, out, share=share)

    assert _tree(base) == before and _modes(base) == modes
    after = _tree(out)
    changed = sorted(p for p in before if after.get(p) != before[p])
    assert changed == ["Elastic/CrossSection/6_12_Carbon.z", "Elastic/FS/6_12_Carbon.z"]
    assert set(after) - set(before) == {MANIFEST_NAME}
    assert sorted(result.replaced) == changed
    # The FS is rewritten at 17 digits but says the same thing.
    lib = g4ndl.open(out)
    assert recordDifferences(g4ndl.open(base).elasticFinalState("C12"),
                             lib.elasticFinalState("C12")) == []
    assert np.array_equal(lib.crossSection("C12").sigma,
                          g4ndl.open(base).crossSection("C12").sigma * 1.05)
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["target"] == "C12" and manifest["share"] == share
    assert {f["replaces_sha256"] for f in manifest["files"]} == {before[p] for p in changed}
    assert not list(tmp_path.glob(".patched.kika-*"))


def test_patch_matches_the_base_variant_and_drops_the_twin(tmp_path, base):
    out = tmp_path / "plain"
    result = g4ndl.patch_elastic(base, JEFF.read("H1"), out, compressed=False)
    assert (out / "Elastic/FS/1_1_Hydrogen").is_file()
    assert not (out / "Elastic/FS/1_1_Hydrogen.z").exists()
    assert sorted(result.removed) == ["Elastic/CrossSection/1_1_Hydrogen.z",
                                      "Elastic/FS/1_1_Hydrogen.z"]


def test_patch_refuses_to_touch_what_it_did_not_make(tmp_path, base):
    suite = JEFF.read("H1")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    with pytest.raises(FileExistsError):
        g4ndl.patch_elastic(base, suite, foreign)
    with pytest.raises(FileExistsError, match="no kika_manifest"):
        g4ndl.patch_elastic(base, suite, foreign, overwrite=True)
    with pytest.raises(ValueError, match="separate trees"):
        g4ndl.patch_elastic(base, suite, base / "inside")

    out = tmp_path / "loop"
    g4ndl.patch_elastic(base, suite, out)
    suite.reactions[2].crossSection["recon"].ys[:] *= 2.0
    with pytest.raises(FileExistsError):
        g4ndl.patch_elastic(base, suite, out)
    modes = _modes(base)
    g4ndl.patch_elastic(base, suite, out, overwrite=True, share="hardlink")  # the loop
    g4ndl.patch_elastic(base, suite, out, overwrite=True, share="hardlink")
    assert _modes(base) == modes        # removing the old links kept the base read-only
    assert np.array_equal(g4ndl.open(out).crossSection("H1").sigma,
                          JEFF.crossSection("H1").sigma * 2.0)


def test_a_refused_suite_leaves_nothing_behind(tmp_path, base):
    suite = JEFF.read("H1")
    suite.reactions[2].crossSection["recon"].ys[0] = -1.0
    with pytest.raises(G4NDLUnsupportedError):
        g4ndl.patch_elastic(base, suite, tmp_path / "out")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["base"]
