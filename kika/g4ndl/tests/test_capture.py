"""Capture (MT102): the grammar, its model mapping, the writer and the patch.

Against the committed real fixtures under ``data/capture/`` (four whole
isotopes of G4NDL 4.7.1, ``build_fixtures.CAPTURE``) and hand-written token
streams for what the grammar must refuse. The whole libraries and the
comparison with the IAEA translation of the JEFF-4.0 tapes are
``test_capture_full_libraries.py`` (``tape``).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
import zlib
from pathlib import Path

import numpy as np
import pytest

import kika
import kika.g4ndl as g4ndl
from kika.g4ndl import G4NDLError, G4NDLFormatError
from kika.g4ndl.capture import (
    CaptureMF6Record, CapturePhotonsRecord, captureFinalState, encodeCapture, finalStateDifferences,
    formatCaptureFS, parse_capture_mf6, parse_capture_photons, writeCapture,
)
from kika.g4ndl.encode import formatCrossSection, recordDifferences, writeSuite
from kika.g4ndl.parse import parse_cross_section
from kika.g4ndl.patch import MANIFEST_NAME
from kika.g4ndl.tests.data.build_fixtures import CAPTURE
from kika.g4ndl.tokens import TokenStream
from kika.nuclear_data.model import AngularTwoBody, G4NDLCaptureProvenance, Uncorrelated

DATA = Path(__file__).parent / "data" / "capture" / "G4NDL-4.7.1"
G4 = g4ndl.open(DATA)
TARGETS = ["H1", "H2", "N14", "N15"]


def _record(target):
    return G4.captureFinalState(target)


# ------------------------------------------------------------------ index and grammar

def test_the_index_holds_the_cross_section_and_one_final_state_each():
    assert [str(k) for k in G4.isotopes("capture")] == TARGETS
    assert {t: G4.captureFinalStateDirectory(t) for t in TARGETS} == {
        "H1": "Capture/FSMF6", "H2": "Capture/FS", "N14": "Capture/FS",
        "N15": "Capture/FSMF6"}
    assert G4.describe()["processes"] == ["capture"]
    assert "Capture" not in G4.describe()["unread"]
    assert sorted(CAPTURE) == sorted(("G4NDL-4.7.1", G4.locate(t, "Capture/CrossSection")
                                      .path.name[:-2]) for t in TARGETS)


@pytest.mark.parametrize("target", TARGETS)
def test_every_fixture_reads_to_its_last_token_and_back_as_text(target):
    record = _record(target)
    text = formatCaptureFS(record)
    parser = parse_capture_mf6 if isinstance(record, CaptureMF6Record) else parse_capture_photons
    assert finalStateDifferences(record, parser(TokenStream(text))) == []
    cs = G4.captureCrossSection(target)
    assert recordDifferences(cs, parse_cross_section(TokenStream(formatCrossSection(cs)))) == []


def test_the_final_states_are_the_consumer_structures():
    h2, n14 = _record("H2"), _record("N14")
    assert isinstance(h2, CapturePhotonsRecord)
    # One 6.257 MeV line (disType 2): MF14 isotropic and InitEnergies reads nothing.
    assert (h2.mean.repFlag, h2.mean.nDiscrete) == (1, 1)
    assert h2.mean.lines[0].disType == 2 and h2.mean.lines[0].energy == 6.25725e6
    assert h2.angular.isoFlag == 1 and not h2.energies.needed
    assert n14.mean.nDiscrete == 59 and n14.angular.isoFlag == 0 and n14.energies.needed
    h1, n15 = _record("H1"), _record("N15")
    assert [(p.massCode, p.distLaw) for p in h1.body.products] == [(0.0, 2), (1002.0, 4)]
    assert [(p.massCode, p.distLaw) for p in n15.body.products] == [(0.0, 1)]


def test_the_cross_section_bookkeeping_is_the_q_value():
    """Geant4 discards the two integers; in Capture/ they are Q (eV, rounded) and 0."""
    for t in TARGETS:
        b0, b1 = G4.captureCrossSection(t).bookkeeping
        assert b1 == 0 and b0 > 1e6
    for t in ("H1", "N15"):
        assert {p.actualStateQ for p in _record(t).body.products} == \
            {float(G4.captureCrossSection(t).bookkeeping[0])}


@pytest.mark.parametrize("text,parser,match", [
    ("", parse_capture_photons, "no photon data"),
    ("1 1.9968 1 2 6.25725e+06 2 1 2 2 1.0e-5 1.0 3.0e7 1.0 1 7", parse_capture_photons,
     "unread tokens"),
    ("1 1.9968 1 2 6.25725e+06 2 1 2 2 1.0e-5 1.0 3.0e7 1.0", parse_capture_photons,
     "MF14"),
    ("1.0 2 1 0 0 0 5 0 0 2 1 2 2 1.0e-5 1.0 3.0e7 1.0", parse_capture_mf6, "LAW|law|distLaw"),
])
def test_the_grammar_refuses_what_would_misalign(text, parser, match):
    with pytest.raises(G4NDLFormatError, match=match):
        parser(TokenStream(text))


def test_a_library_with_both_final_states_is_refused(tmp_path):
    root = tmp_path / "lib"
    shutil.copytree(DATA, root)
    shutil.copy(root / "Capture/FS/1_2_Hydrogen.z", root / "Capture/FSMF6/1_2_Hydrogen.z")
    with pytest.raises(G4NDLError, match="reads the first and ignores the second"):
        g4ndl.open(root).captureFinalState("H2")


# ------------------------------------------------------------------ the model

@pytest.mark.parametrize("target", TARGETS)
def test_every_fixture_is_a_fixed_point_through_the_model(target):
    suite = G4.read(target)
    cs, fs, _ = encodeCapture(suite)
    assert recordDifferences(G4.captureCrossSection(target), cs) == []
    assert finalStateDifferences(_record(target), fs) == []


def test_mt102_and_what_reaches_the_model():
    h1 = G4.read("H1").findReactionByENDF_MT(102)
    assert h1.outputChannel.Q.value == 2224631.0
    assert np.array_equal(h1.crossSection["recon"].ys, G4.captureCrossSection("H1").sigma)
    photon, deuteron = h1.outputChannel.products
    assert (photon.pid, deuteron.pid) == ("photon", "H2")
    assert isinstance(photon.distribution["eval"], AngularTwoBody)
    assert deuteron.distribution["eval"].recoilHref is not None
    n15 = G4.read("N15").findReactionByENDF_MT(102)
    (photon,) = n15.outputChannel.products
    assert isinstance(photon.distribution["eval"], Uncorrelated)
    assert photon.multiplicity is not None
    p = n15.provenance
    assert isinstance(p, G4NDLCaptureProvenance) and p.finalState == "FSMF6"
    assert p.finalStateEntry["products"][0]["distLaw"] == 1


def test_capture_photons_reach_the_model():
    """D10-2: an ``FS`` body is ENDF's MF12/14/15 and reaches the model through
    the ENDF photon adapter: one photon product per InitMean line."""
    suite = G4.read("N14")
    r = suite.findReactionByENDF_MT(102)
    record = _record("N14")
    photons = list(r.outputChannel.products)
    assert len(photons) == len(record.mean.lines) and all(p.pid == "photon" for p in photons)
    assert all(isinstance(p.distribution["eval"], Uncorrelated) for p in photons)
    first, line = photons[0], record.mean.lines[0]
    assert np.array_equal(first.multiplicity.form.ys, line.yield_.y)
    entry = r.provenance.finalStateEntry
    assert r.provenance.finalState == "FS" and "verbatim" not in entry
    assert {"endf", "g4ndl"} <= set(entry)
    assert r.outputChannel.Q.value == float(G4.captureCrossSection("N14").bookkeeping[0])
    assert not any("MF12-15" in m for m in suite.report.unsupported)


def test_an_edit_to_a_capture_photon_is_what_is_written():
    suite = G4.read("N14")
    r = suite.findReactionByENDF_MT(102)
    r.outputChannel.products[0].multiplicity.form.ys[:] *= 2.0
    _, fs, _ = encodeCapture(suite)
    assert np.array_equal(fs.mean.lines[0].yield_.y, 2.0 * _record("N14").mean.lines[0].yield_.y)
    assert [d for d in finalStateDifferences(_record("N14"), fs)
            if "lines[0].yield_" not in d] == []


def test_model_edits_are_what_is_written(tmp_path):
    suite = G4.read("N15")
    r = suite.findReactionByENDF_MT(102)
    r.crossSection["recon"].ys[:] *= 0.5
    writeCapture(suite, tmp_path, compressed=True)
    back = g4ndl.open(tmp_path)
    assert np.array_equal(back.captureCrossSection("N15").sigma,
                          0.5 * G4.captureCrossSection("N15").sigma)
    assert back.captureCrossSection("N15").bookkeeping == G4.captureCrossSection("N15").bookkeeping
    assert finalStateDifferences(_record("N15"), back.captureFinalState("N15")) == []
    # A new Q gives new bookkeeping: the integer follows the model, not the file.
    r.outputChannel.Q.value += 1000.0
    cs, _, _ = encodeCapture(suite)
    assert cs.bookkeeping == (G4.captureCrossSection("N15").bookkeeping[0] + 1000, 0)


def test_a_suite_without_a_final_state_writes_none_and_removes_the_stale_one(tmp_path):
    suite = G4.read("N15")
    writeCapture(suite, tmp_path)
    assert (tmp_path / "Capture/FSMF6/7_15_Nitrogen").is_file()
    r = suite.findReactionByENDF_MT(102)
    r.outputChannel.products.products[:] = []
    r.provenance.finalStateEntry.clear()
    report = writeCapture(suite, tmp_path)
    assert any("G4PhotonEvaporation" in m for m in report.warnings)
    assert not (tmp_path / "Capture/FSMF6/7_15_Nitrogen").exists()
    assert g4ndl.open(tmp_path).captureFinalState("N15") is None


def test_model_products_replace_the_photon_file():
    """Other products in the model win over the FS the provenance describes."""
    donor = G4.read("N15").findReactionByENDF_MT(102)
    suite = G4.read("N14")
    r = suite.findReactionByENDF_MT(102)
    r.outputChannel.products.products[:] = list(donor.outputChannel.products)
    r.provenance.finalStateEntry.clear()
    r.provenance.finalStateEntry.update(
        {k: v for k, v in donor.provenance.finalStateEntry.items()})
    from kika.nuclear_data.model import ConversionReport
    fs = captureFinalState(suite, report=ConversionReport())
    assert isinstance(fs, CaptureMF6Record)
    assert finalStateDifferences(_record("N15"), fs) == []


def test_write_replaces_a_twin_of_the_other_variant(tmp_path):
    suite = G4.read("H2")
    writeCapture(suite, tmp_path, compressed=False)
    report = writeCapture(suite, tmp_path, compressed=True)
    assert (tmp_path / "Capture/FS/1_2_Hydrogen.z").is_file()
    assert not (tmp_path / "Capture/FS/1_2_Hydrogen").exists()
    assert any("reads the .z" in m for m in report.warnings)
    raw = zlib.decompress((tmp_path / "Capture/FS/1_2_Hydrogen.z").read_bytes()).decode()
    assert finalStateDifferences(_record("H2"), parse_capture_photons(TokenStream(raw))) == []


def test_the_front_door_reads_and_writes_the_capture(tmp_path):
    suite = kika.read(DATA, format="g4ndl", target="H1")
    assert [r.id.ENDF_MT for r in suite.reactions] == [102]
    kika.write(suite, tmp_path, format="g4ndl", compressed=True)
    lib = g4ndl.open(tmp_path)
    assert recordDifferences(G4.captureCrossSection("H1"), lib.captureCrossSection("H1")) == []
    assert finalStateDifferences(_record("H1"), lib.captureFinalState("H1")) == []
    with pytest.raises(ValueError, match="'capture'"):
        writeSuite(suite, tmp_path, processes=["thermal"])


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
    suite = g4ndl.open(base).read("H1")
    suite.findReactionByENDF_MT(102).crossSection["recon"].ys[:] *= 1.1
    result = g4ndl.patch_isotope(base, suite, tmp_path / "out", share=share)
    assert _tree(base) == before
    after = _tree(tmp_path / "out")
    changed = sorted(p for p in before if after.get(p) != before[p])
    assert changed[0] == "Capture/CrossSection/1_1_Hydrogen.z"
    assert all("1_1_Hydrogen" in p for p in changed)
    assert set(after) - set(before) == {MANIFEST_NAME}
    manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    assert manifest["processes"] == ["capture"]
    lib = g4ndl.open(tmp_path / "out")
    assert np.allclose(lib.captureCrossSection("H1").sigma,
                       1.1 * G4.captureCrossSection("H1").sigma, rtol=1e-15, atol=0)
    assert finalStateDifferences(_record("H1"), lib.captureFinalState("H1")) == []
    assert result.removed == []


def test_patch_removes_a_final_state_the_suite_no_longer_has(tmp_path, base):
    suite = g4ndl.open(base).read("N15")
    r = suite.findReactionByENDF_MT(102)
    r.outputChannel.products.products[:] = []
    r.provenance.finalStateEntry.clear()
    result = g4ndl.patch_isotope(base, suite, tmp_path / "out")
    assert result.removed == ["Capture/FSMF6/7_15_Nitrogen.z"]
    assert g4ndl.open(tmp_path / "out").captureFinalState("N15") is None


def test_parsing_capture_does_not_wake_the_model():
    code = textwrap.dedent("""
        import sys
        import kika.g4ndl, kika.g4ndl.capture
        lib = kika.g4ndl.open(sys.argv[1])
        lib.captureCrossSection("N14"); lib.captureFinalState("N14"); lib.captureFinalState("H1")
        print(any(m.startswith("kika.nuclear_data.model") for m in sys.modules))
    """)
    out = subprocess.run([sys.executable, "-c", code, str(DATA)],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ------------------------------------------------------------------ what a viewer shows

def test_the_capture_summary_says_where_the_photons_live():
    from kika.g4ndl.tables import _reactionRows, captureSummary, crossSections

    n15 = captureSummary(G4.read("N15"))
    assert (n15["final_state"], n15["photons"], n15["products"]) == ("FSMF6", "model", ["photon"])
    assert n15["q_value"] == float(G4.captureCrossSection("N15").bookkeeping[0])
    assert n15["cross_section"]["count"] == len(G4.captureCrossSection("N15").energy)
    h1 = captureSummary(G4.read("H1"))
    assert h1["products"] == ["photon", "H2"]
    n14 = captureSummary(G4.read("N14"))
    assert (n14["final_state"], n14["photons"]) == ("FS", "model")
    assert n14["products"] and all(label.startswith("photon") for label in n14["products"])
    suite = G4.read("N15")
    r = suite.findReactionByENDF_MT(102)
    r.outputChannel.products.products[:] = []
    r.provenance.finalStateEntry.clear()
    assert captureSummary(suite)["photons"] is None
    # MT102 is one more row of the reactions table, with its Q, and no angular.
    assert 102 in crossSections(suite)
    (row,) = [row for row in _reactionRows(suite) if row["mt"] == 102]
    assert row["q_value"] == r.outputChannel.Q.value and not row["sum"] and not row["angular"]
