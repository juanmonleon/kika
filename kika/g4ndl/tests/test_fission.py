"""Fission (MT18 and its chances): the grammar, its model mapping, the writer and the patch.

Against the committed real fixtures under ``data/fission/`` (four whole
isotopes of G4NDL 4.7.1 and one ``Fission/FF``, ``build_fixtures.FISSION``)
and hand-written token streams for what the grammar must refuse. The two whole
libraries and the comparison with the ENDF adapter on the JEFF-4.0 tapes are
``test_fission_full_libraries.py`` (``tape``).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import kika
import kika.g4ndl as g4ndl
from kika.g4ndl import G4NDLFormatError, G4NDLUnsupportedError
from kika.g4ndl.encode import recordDifferences, writeSuite
from kika.g4ndl.fission import (
    fissionDifferences, formatChanceFission, formatFissionFS, formatFragmentYields,
    parse_chance_fission, parse_fission_fs, parse_fragment_yields,
)
from kika.g4ndl.fission_model import encodeFission, writeFission
from kika.g4ndl.patch import MANIFEST_NAME
from kika.g4ndl.tests.data.build_fixtures import FISSION
from kika.g4ndl.tokens import TokenStream
from kika.nuclear_data.model import (
    G4NDLFissionProvenance, GeneralEvaporation, SimpleMaxwellianFission, Uncorrelated, XYs2d,
)

DATA = Path(__file__).parent / "data" / "fission" / "G4NDL-4.7.1"
CAPTURE = Path(__file__).parent / "data" / "capture" / "G4NDL-4.7.1"
G4 = g4ndl.open(DATA)
TARGETS = ["Ra223", "Th230", "Pu244", "Fm255"]


def _check(target):
    """Read ``target`` into the model, encode it, and compare every file."""
    cs, fs, chances, ff, report = encodeFission(G4.read(target))
    diffs = recordDifferences(G4.fissionCrossSection(target), cs)
    diffs += fissionDifferences(G4.fissionFinalState(target), fs)
    assert list(chances) == G4.fissionChances(target)
    for c, record in chances.items():
        diffs += fissionDifferences(G4.chanceFission(target, c), record)
    return diffs, ff, report


# ------------------------------------------------------------------ index and grammar

def test_the_index_holds_mt18_its_chances_and_the_yields():
    assert [str(k) for k in G4.isotopes("fission")] == TARGETS
    assert {t: G4.fissionChances(t) for t in TARGETS} == {
        "Ra223": [], "Th230": [], "Pu244": ["FC", "SC"], "Fm255": ["FC", "SC", "TC", "LC"]}
    d = G4.describe()
    assert d["processes"] == ["fission"] and "Fission" not in d["unread"]
    assert sorted(FISSION) == sorted(("G4NDL-4.7.1", G4.locate(t, "Fission/FS").path.name[:-2])
                                     for t in TARGETS)
    assert G4.fragmentYields("Th230") is None
    assert not G4.has("Th227", "fission") and G4.fragmentYields("Th227") is not None


@pytest.mark.parametrize("target", TARGETS)
def test_every_fixture_reads_to_its_last_token_and_back_as_text(target):
    fs = G4.fissionFinalState(target)
    assert fissionDifferences(fs, parse_fission_fs(TokenStream(formatFissionFS(fs)))) == []
    for c in G4.fissionChances(target):
        r = G4.chanceFission(target, c)
        back = parse_chance_fission(TokenStream(formatChanceFission(r)), c)
        assert fissionDifferences(r, back) == []


def test_the_fragment_yields_read_and_come_back_as_text():
    ff = G4.fragmentYields("Th227")
    assert ff.header == ("G4NDL", "ENDF/B-VII.1")
    assert [(b.MT, b.MF, b.imax) for b in ff.blocks] == [(454, 8, 0), (459, 8, 0)]
    at = ff.blocks[0].energies[0]
    assert len(at) == 1220 and at.fsp.dtype == np.int64
    assert fissionDifferences(ff, parse_fragment_yields(TokenStream(formatFragmentYields(ff)))) == []


def test_the_consumer_structures():
    th = G4.fissionFinalState("Th230")
    assert [(s.infoType, s.dataType) for s in th.sections] == [(2, 1), (5, 1), (1, 4), (1, 5)]
    assert [p.law for p in th.section(1, 5).body.partials] == [7]
    assert len(th.section(5, 1).body.values) == 10
    fm = G4.fissionFinalState("Fm255")
    assert [p.law for p in fm.section(3, 5).body.partials] == [5] * 6
    assert len(fm.section(3, 1).body.decayConstants) == 6
    assert G4.fissionFinalState("Ra223").section(1, 4).body.repFlag == 2
    fc = G4.chanceFission("Pu244", "FC")
    assert fc.hasFinalState and (fc.angularHead, fc.energyHead) == ((1, 4), (3, 5))
    assert fc.crossSection.bookkeeping == (207654000, 0)
    assert not G4.chanceFission("Fm255", "LC").hasFinalState


@pytest.mark.parametrize("text,parser,match", [
    ("", parse_fission_fs, "no section"),
    ("6 1 1.0", parse_fission_fs, "infoType=6"),
    ("2 5 1.0", parse_fission_fs, "dataType=5 under infoType=2"),
    ("2 1 230.0 3 1 2.0", parse_fission_fs, "iflag=3"),
    ("1 14 0", parse_fission_fs, "before the dataType 12"),
    ("3 1 230.0 1 1 -0.1 1 0.01", parse_fission_fs, "negative"),
    ("", parse_fragment_yields, "no block"),
    ("454 8 0.0 0 1.0 1 2 23066.0 0 0.5", parse_fragment_yields, "expected an integer"),
])
def test_the_grammar_refuses_what_would_misalign(text, parser, match):
    with pytest.raises(G4NDLFormatError, match=match):
        parser(TokenStream(text))


# ------------------------------------------------------------------ the model

@pytest.mark.parametrize("target", TARGETS)
def test_every_fixture_is_a_fixed_point_through_the_model(target):
    diffs, ff, _ = _check(target)
    assert diffs == [] and ff is None


def test_mt18_lands_where_the_endf_adapter_puts_it():
    from kika.endf.model_adapter.multiplicity import (
        DELAYED_NUBAR_LABEL, TOTAL_NUBAR_LABEL, nubarNode,
    )

    suite = G4.read("Fm255")
    r = suite.findReactionByENDF_MT(18)
    assert isinstance(r.provenance, G4NDLFissionProvenance)
    assert r.outputChannel.Q.value is None   # the file states 0 0, not the fission Q
    assert np.array_equal(r.crossSection["recon"].ys, G4.fissionCrossSection("Fm255").sigma)
    (n,) = r.outputChannel.products
    d = n.distribution["eval"]
    assert isinstance(d, Uncorrelated) and isinstance(d.energy, XYs2d)
    # Prompt on the product, total and delayed as §21.3 sums, as attachNubar places them.
    assert float(n.multiplicity.form.evaluate(0.0253)) == 4.0
    sums = suite.sums.multiplicitySums
    assert sums.byENDF_MT(452).label == TOTAL_NUBAR_LABEL
    assert sums.byENDF_MT(455).label == DELAYED_NUBAR_LABEL
    assert len(sums.byENDF_MT(452).summands) == 2
    assert nubarNode(suite, 456) is n.multiplicity
    families = r.outputChannel.fissionFragmentData.delayedNeutrons
    assert len(families) == 6 and families[0].rate.unit == "1/s"
    assert all(isinstance(f.product.distribution["eval"].energy, GeneralEvaporation)
               for f in families)
    # The families' multiplicities add up to the delayed nu-bar.
    e = np.array([0.0253, 1e6])
    total = sum(np.asarray(f.product.multiplicity.form.evaluate(e)) for f in families)
    assert np.allclose(total, sums.byENDF_MT(455).multiplicity.form.evaluate(e), rtol=1e-6)
    release = r.outputChannel.fissionFragmentData.fissionEnergyReleases[0]
    assert release.totalEnergy.coefficients.tolist() == \
        [G4.fissionFinalState("Fm255").section(5, 1).body.values[9]]
    # Only the photons are kept as text, and the report says so.
    kept = [(e["infoType"], e["dataType"]) for e in r.provenance.sections if "verbatim" in e]
    assert kept == [(1, 12), (1, 14), (1, 15)]
    assert any("photon" in m for m in suite.report.unsupported)


def test_a_total_only_nubar_is_the_product_multiplicity_and_lf7_is_a_maxwellian():
    suite = G4.read("Th230")
    r = suite.findReactionByENDF_MT(18)
    (n,) = r.outputChannel.products
    assert float(n.multiplicity.form.evaluate(0.0253)) == pytest.approx(2.01)
    assert len(suite.sums.multiplicitySums) == 0
    d = n.distribution["eval"]
    assert isinstance(d.energy, SimpleMaxwellianFission) and d.energy.U is None


def test_the_chances_are_reactions_with_their_own_neutron():
    suite = G4.read("Pu244")
    fc = suite.findReactionByENDF_MT(19)
    assert fc.outputChannel.Q.value == 207654000.0
    (n,) = fc.outputChannel.products
    assert n.multiplicity is None   # Geant4 takes nu-bar from FS
    assert isinstance(n.distribution["eval"].energy, SimpleMaxwellianFission)
    assert [r.id.ENDF_MT for r in G4.read("Fm255").reactions] == [18, 19, 20, 21, 38]
    assert list(G4.read("Fm255").findReactionByENDF_MT(38).outputChannel.products) == []


def test_fragment_yields_travel_in_the_provenance(tmp_path):
    root = tmp_path / "lib"
    shutil.copytree(DATA, root)
    ff = root / "Fission/FF/90_227_Thorium.z"
    shutil.move(ff, root / "Fission/FF/90_230_Thorium.z")
    lib = g4ndl.open(root)
    suite = lib.read("Th230")
    assert suite.findReactionByENDF_MT(18).provenance.fragmentYields
    assert any("productYields" in m for m in suite.report.unsupported)
    _, _, _, ff, _ = encodeFission(suite)
    assert fissionDifferences(lib.fragmentYields("Th230"), ff) == []
    writeFission(suite, tmp_path / "out")
    assert fissionDifferences(lib.fragmentYields("Th230"),
                              g4ndl.open(tmp_path / "out").fragmentYields("Th230")) == []


def test_model_edits_are_what_is_written(tmp_path):
    suite = G4.read("Fm255")
    r = suite.findReactionByENDF_MT(18)
    r.crossSection["recon"].ys[:] *= 0.5
    n = r.outputChannel.products[0]
    n.multiplicity.form.ys[:] *= 1.1
    writeFission(suite, tmp_path, compressed=True)
    back = g4ndl.open(tmp_path)
    assert np.array_equal(back.fissionCrossSection("Fm255").sigma,
                          0.5 * G4.fissionCrossSection("Fm255").sigma)
    prompt = back.fissionFinalState("Fm255").section(4, 1).body
    assert np.allclose(prompt.table.y, 1.1 * G4.fissionFinalState("Fm255").section(4, 1).body.table.y,
                       rtol=1e-15)
    # Everything else, photons included, is what it was.
    was = G4.fissionFinalState("Fm255")
    for s in was.sections:
        if (s.infoType, s.dataType) != (4, 1):
            assert fissionDifferences(s, back.fissionFinalState("Fm255").section(
                s.infoType, s.dataType)) == []
    for c in G4.fissionChances("Fm255"):
        assert fissionDifferences(G4.chanceFission("Fm255", c), back.chanceFission("Fm255", c)) == []


def test_an_edited_family_multiplicity_rewrites_its_weight():
    from kika.nuclear_data.model import ConversionReport

    suite = G4.read("Fm255")
    families = suite.findReactionByENDF_MT(18).outputChannel.fissionFragmentData.delayedNeutrons
    families[0].product.multiplicity.form.ys[:] *= 2.0
    report = ConversionReport()
    _, fs, _, _, _ = encodeFission(suite, report=report)
    before = G4.fissionFinalState("Fm255").section(3, 5).body.partials
    after = fs.section(3, 5).body.partials
    assert np.allclose(after[0].probability.y[:2], 2.0 * before[0].probability.y[:2])
    assert any("p_k(E)" in m for m in report.approximations)


def test_a_suite_without_chances_removes_the_stale_ones(tmp_path):
    suite = G4.read("Pu244")
    writeFission(suite, tmp_path)
    assert (tmp_path / "Fission/SC/94_244_Plutonium").is_file()
    suite.reactions.reactions[:] = [r for r in suite.reactions if r.id.ENDF_MT != 20]
    report = writeFission(suite, tmp_path)
    assert not (tmp_path / "Fission/SC/94_244_Plutonium").exists()
    assert g4ndl.open(tmp_path).fissionChances("Pu244") == ["FC"]
    assert any("Fission/SC" in m for m in report.warnings)


def test_a_histogram_in_the_model_is_refused():
    suite = G4.read("Th230")
    form = suite.findReactionByENDF_MT(18).outputChannel.products[0].multiplicity.form
    from kika.nuclear_data.model.enums import Interpolation as ModelInterpolation
    form.interpolation = ModelInterpolation.flat
    with pytest.raises(G4NDLUnsupportedError, match="code 1"):
        encodeFission(suite)


def test_the_front_door_reads_and_writes_the_fission(tmp_path):
    suite = kika.read(DATA, format="g4ndl", target="Pu244")
    assert [r.id.ENDF_MT for r in suite.reactions] == [18, 19, 20]
    kika.write(suite, tmp_path, format="g4ndl", compressed=True)
    lib = g4ndl.open(tmp_path)
    assert recordDifferences(G4.fissionCrossSection("Pu244"), lib.fissionCrossSection("Pu244")) == []
    assert fissionDifferences(G4.fissionFinalState("Pu244"), lib.fissionFinalState("Pu244")) == []
    with pytest.raises(G4NDLUnsupportedError, match="no MT18"):
        writeSuite(kika.read(CAPTURE, format="g4ndl", target="H1"), tmp_path,
                   processes=["fission"])


# ------------------------------------------------------------------ patch

@pytest.fixture
def base(tmp_path):
    root = tmp_path / "base"
    shutil.copytree(DATA, root)
    for p in root.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    return root


def _tree(root):
    import hashlib
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("share", ["copy", "hardlink"])
def test_patch_isotope_changes_only_that_isotope(tmp_path, base, share):
    before = _tree(base)
    suite = g4ndl.open(base).read("Pu244")
    suite.findReactionByENDF_MT(19).crossSection["recon"].ys[:] *= 1.1
    result = g4ndl.patch_isotope(base, suite, tmp_path / "out", share=share)
    assert _tree(base) == before
    after = _tree(tmp_path / "out")
    changed = sorted(p for p in before if after.get(p) != before[p])
    # The isotope's files are written fresh, so their bytes change; nothing else does.
    assert "Fission/FC/94_244_Plutonium.z" in changed
    assert all(p.startswith("Fission/") and "94_244_Plutonium" in p for p in changed)
    assert set(after) - set(before) == {MANIFEST_NAME}
    manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    assert manifest["processes"] == ["fission"]
    lib = g4ndl.open(tmp_path / "out")
    assert np.allclose(lib.chanceFission("Pu244", "FC").crossSection.sigma,
                       1.1 * G4.chanceFission("Pu244", "FC").crossSection.sigma,
                       rtol=1e-15, atol=0)
    assert fissionDifferences(G4.fissionFinalState("Pu244"),
                              lib.fissionFinalState("Pu244")) == []
    assert fissionDifferences(G4.chanceFission("Pu244", "SC"),
                              lib.chanceFission("Pu244", "SC")) == []
    assert result.removed == []


def test_patch_removes_a_chance_the_suite_no_longer_has(tmp_path, base):
    suite = g4ndl.open(base).read("Pu244")
    suite.reactions.reactions[:] = [r for r in suite.reactions if r.id.ENDF_MT != 20]
    result = g4ndl.patch_isotope(base, suite, tmp_path / "out")
    assert result.removed == ["Fission/SC/94_244_Plutonium.z"]
    assert g4ndl.open(tmp_path / "out").fissionChances("Pu244") == ["FC"]


def test_parsing_fission_does_not_wake_the_model():
    code = textwrap.dedent("""
        import sys
        import kika.g4ndl, kika.g4ndl.fission
        lib = kika.g4ndl.open(sys.argv[1])
        lib.fissionCrossSection("Fm255"); lib.fissionFinalState("Fm255")
        lib.chanceFission("Pu244", "FC"); lib.fragmentYields("Th227")
        print(any(m.startswith("kika.nuclear_data.model") for m in sys.modules))
    """)
    out = subprocess.run([sys.executable, "-c", code, str(DATA)],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ------------------------------------------------------------------ what a viewer shows

def test_the_fission_summary():
    from kika.g4ndl.tables import _reactionRows, fissionSummary

    fm = fissionSummary(G4.read("Fm255"))
    assert (fm["nubar_prompt"], fm["delayed_families"]) == (4.0, 6)
    assert fm["nubar_total"] == pytest.approx(fm["nubar_prompt"] + fm["nubar_delayed"])
    assert fm["delayed_spectra"] == "LF=5 general evaporation"
    assert fm["prompt_spectrum"] == "LF=1 table" and fm["photons"] == "verbatim"
    assert [c["directory"] for c in fm["chances"]] == ["FC", "SC", "TC", "LC"]
    assert not any(c["final_state"] for c in fm["chances"])
    assert fm["energy_release"]["totalEnergy"] > 1.9e8 and fm["verbatim_sections"] == []
    th = fissionSummary(G4.read("Th230"))
    assert (th["nubar_prompt"], th["nubar_delayed"], th["photons"]) == (None, None, None)
    assert th["prompt_spectrum"] == "LF=7 Maxwellian"
    pu = fissionSummary(G4.read("Pu244"))
    assert [c["final_state"] for c in pu["chances"]] == [True, True]
    rows = {row["mt"]: row for row in _reactionRows(G4.read("Pu244"))}
    assert sorted(rows) == [18, 19, 20] and rows[18]["q_value"] is None
    assert rows[19]["q_value"] == 207654000.0
    assert fissionSummary(kika.read(CAPTURE, format="g4ndl", target="H1")) is None
