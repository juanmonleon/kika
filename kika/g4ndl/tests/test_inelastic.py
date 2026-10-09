"""Phase 10: the inelastic grammar, its model mapping, the writer and the patch.

Against the committed real fixtures under ``data/inelastic/`` (one small file
per section type and law, ``build_fixtures.INELASTIC``; the whole of Pu-244
from G4NDL 4.7.1; two ``Gammas`` level schemes) and hand-written token
streams for what no small real file shows. The whole libraries are
``test_inelastic_full_libraries.py`` (``tape``).
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
from kika.g4ndl.inelastic_decode import INELASTIC_SUM_LABEL, decodeInelastic
from kika.g4ndl.inelastic_encode import (
    SUM_RTOL, channelOf, encodeInelastic, partialSumCheck, rebuildInelasticSums, writeInelastic,
)
from kika.g4ndl.inelastic_format import formatGammas, formatInelasticFS, inelasticDifferences
from kika.g4ndl.inelastic_parse import parse_gammas, parse_inelastic_fs
from kika.g4ndl.inelastic_records import (
    AngularBody, ContinuumBody, EnergyAngleBody, PhotonCascadeBody, PhotonEnergyBody,
    PhotonPartialsBody,
)
from kika.g4ndl.patch import MANIFEST_NAME
from kika.g4ndl.tests.data.build_fixtures import INELASTIC, INELASTIC_ISOTOPES
from kika.g4ndl.tokens import TokenStream
from kika.nuclear_data.model import (
    AngularEnergy, AngularTwoBody, CrossSectionSum, EnergyAngular, G4NDLInelasticProvenance,
    Isotropic2d, KalbachMann, Uncorrelated, Unspecified, XYs1d, XYs2d,
)

DATA = Path(__file__).parent / "data" / "inelastic"
JEFF = g4ndl.open(DATA / "JEFF-4.0")
G4 = g4ndl.open(DATA / "G4NDL-4.7.1")
LIBS = {"JEFF-4.0": JEFF, "G4NDL-4.7.1": G4}


def _files():
    out = [(lib, sub, name) for (lib, sub, name) in INELASTIC]
    for (lib, stem) in INELASTIC_ISOTOPES:
        for ch in LIBS[lib].inelasticChannels(_key(stem)):
            out.append((lib, ch, stem + ".z"))
    return sorted(set(out))


def _key(stem):
    from kika.g4ndl.names import parse_file_name
    return parse_file_name(stem.replace(".z", ""))[0]


def _parse(lib, channel, name):
    return LIBS[lib].inelasticFinalState(_key(name), channel)


def _text(channel, text):
    return parse_inelastic_fs(TokenStream(text), channel)


# ------------------------------------------------------------------ index

def test_the_index_holds_channels_and_level_schemes():
    assert G4.inelasticChannels("Pu244") == ["F01", "F04", "F05", "F18"]
    assert G4.has("Pu244", "inelastic") and not G4.has("Ni64", "inelastic")
    assert G4.isotopes("inelastic") == [_key("94_244_Plutonium")]
    assert G4.gammaNuclei() == [(6, 15), (55, 120)]
    assert JEFF.gammaNuclei() == []   # JEFF-4.0 ships no Inelastic/Gammas
    d = G4.describe()
    assert d["processes"] == ["inelastic"] and "Inelastic" not in d["unread"]


# ------------------------------------------------------------------ grammar

@pytest.mark.parametrize("lib,channel,name", _files())
def test_every_fixture_reads_to_its_last_token_and_back(lib, channel, name):
    record = _parse(lib, channel, name)
    assert record.channel == channel and record.sections
    back = _text(channel, formatInelasticFS(record))
    assert inelasticDifferences(record, back) == []


def test_composite_and_base_headers():
    ni = _parse("G4NDL-4.7.1", "F01", "28_64_Nickel.z")
    assert ni.composite and ni.Qvalue is None
    assert ni.reactionMTs == (4, 51, 52, 91)
    assert all(s.sfType is not None and s.dummy == 0 for s in ni.sections)
    gd = _parse("G4NDL-4.7.1", "F04", "64_156_Gadolinium.z")
    assert not gd.composite and gd.Qvalue == -8526900.0 and gd.Qdummy == 0
    assert all(s.sfType is None for s in gd.sections) and gd.reactionMTs == (16,)
    # A base file states Q once; the consumer reads no second (Q, dummy).
    text = formatInelasticFS(gd).split("\n")
    assert text[0] == "1 3 -8526900.0 0" and text[0] != text[1]


def test_the_section_bodies_are_the_consumer_structures():
    hg = _parse("G4NDL-4.7.1", "F05", "80_196_Mercury.z")
    kinds = [type(s.body).__name__ for s in hg.sections]
    assert kinds == ["CrossSectionBody", "AngularBody", "EnergyBody",
                     "PhotonMultiplicityBody", "PhotonAngularBody", "PhotonEnergyBody"]
    assert hg.sections[5].body.needed
    eu = _parse("G4NDL-4.7.1", "F01", "63_151_Europium.z")
    assert any(isinstance(s.body, PhotonCascadeBody) for s in eu.sections)
    au = _parse("JEFF-4.0", "F18", "79_197_Gold.z")
    assert any(isinstance(s.body, PhotonPartialsBody) for s in au.sections)
    km = _parse("G4NDL-4.7.1", "F22", "13_27_Aluminum.z").sections[1].body
    assert isinstance(km, EnergyAngleBody) and km.frameFlag in (1, 2, 3)
    assert {p.distLaw for p in km.products} >= {1}
    assert any(isinstance(p.body, ContinuumBody) and p.body.angularRep == 2 for p in km.products)


def test_dataType_15_reads_nothing_without_a_continuum_photon():
    # InitEnergies reads only if some disType == 1 among the slot's photons.
    head = "1 3 -1.0e6 0\n2\n1.0e6 0.0 2.0e7 1.0\n"
    mean = "1 12\n1 9.0\n1\n0 5.0e5\n2\n1\n2 2\n1.0e6 1.0 2.0e7 1.0\n"
    rec = _text("F04", head + mean + "1 15\n")
    body = rec.sections[-1].body
    assert isinstance(body, PhotonEnergyBody) and body.needed is False and body.spectra == ()
    with pytest.raises(G4NDLFormatError, match="dataType=14 before"):
        _text("F04", head + "1 14\n1\n")


@pytest.mark.parametrize("text,match", [
    ("1 3 -1.0e6 0\n2.0\n1.0e6 0.0 2.0e7 1.0\n", "expected an integer"),
    ("1 7 -1.0e6 0\n", "dataType=7"),
    ("1 3 -1.0e6 0\n2\n1.0e6 0.0 2.0e7 -1.0\n", "negative value"),
    ("1 3 -1.0e6 0\n2\n2.0e7 0.0 1.0e6 1.0\n", "energy decreases"),
    ("1 3 -1.0e6 0\n2\n1.0e6 0.0 2.0e7 1.0\n1 4\n3 9.0 2\n", "repFlag=3"),
    ("1 3 -1.0e6 0\n2\n1.0e6 0.0 2.0e7 1.0\n1 5\n0.0 1\n2\n2\n1\n2 2\n1.0e6 1.0 2.0e7 1.0\n",
     "energy law 2"),
    ("1 3 -1.0e6 0\n2\n1.0e6 0.0 2.0e7 1.0\n1 6\n9.0 1 1\n1.0 1.0 0 5 0.0 0.0\n"
     "2\n1\n2 2\n1.0e6 1.0 2.0e7 1.0\n", "distLaw=5"),
    ("1 3 -1.0e6 0\n2\n1.0e6 0.0 2.0e7 1.0\n1 6\n9.0 4 1\n", "frameFlag=4"),
    ("1 3 -1.0e6 0\n2\n1.0e6 0.0 2.0e7\n", "unexpected end of data"),
    ("", "holds no section"),
])
def test_the_grammar_refuses_what_would_misalign(text, match):
    with pytest.raises(G4NDLFormatError, match=match):
        _text("F04", text)


def test_qualified_interpolation_is_inelastic_only():
    # Unit base (22) on an incident-energy table: accepted in an energy law.
    body = ("1 3 -1.0e6 0\n2\n1.0e6 0.0 2.0e7 1.0\n1 5\n0.0 1\n1\n"
            "2\n1\n2 2\n1.0e6 1.0 2.0e7 1.0\n"
            "2\n1\n2 22\n"
            "1.0e6 2\n1\n2 2\n0.0 0.0 1.0e5 1.0\n"
            "2.0e7 2\n1\n2 2\n0.0 0.0 1.0e6 1.0\n")
    rec = _text("F04", body)
    assert rec.sections[1].body.partials[0].interpolation.codes == (22,)


# ------------------------------------------------------------------ Gammas

def test_level_schemes():
    g = G4.gammas(55, 120)
    assert len(g) and np.any(np.diff(g.levels) < 0)   # out of order, as shipped
    assert inelasticDifferences(g, parse_gammas(TokenStream(formatGammas(g)), 55, 120)) == []
    with pytest.raises(G4NDLFormatError, match="whole number"):
        parse_gammas(TokenStream("100.0 100.0"), 1, 2)
    with pytest.raises(G4NDLFormatError, match="ifstream"):
        parse_gammas(TokenStream("G4NDL x 100.0 100.0 1.0"), 1, 2)
    with pytest.raises(g4ndl.IsotopeNotFoundError):
        G4.gammas(1, 1)


# ------------------------------------------------------------------ the model

@pytest.mark.parametrize("lib,channel,name", _files())
def test_every_fixture_is_a_fixed_point_through_the_model(lib, channel, name):
    key = _key(name)
    suite = LIBS[lib].read(key, processes=["inelastic"])
    _, files, _ = encodeInelastic(suite)
    assert inelasticDifferences(_parse(lib, channel, name), files[channel]) == []


def test_reactions_sums_and_provenance():
    suite = G4.read("Pu244")
    labels = [r.id.label for r in suite.reactions]
    assert labels[:6] == ["MT51", "MT52", "MT53", "MT54", "MT55", "MT91"]
    assert {"MT16", "MT17", "MT37"} <= set(labels)
    mt4 = list(suite.sums)[0]
    assert isinstance(mt4, CrossSectionSum) and mt4.id.ENDF_MT == 4
    assert len(mt4.summands) == 6
    total = [s for s in suite.sums if s.id.label == INELASTIC_SUM_LABEL][0]
    assert total.id.ENDF_MT is None and len(total.summands) == len(suite.reactions)
    r = suite.reactions[51]
    p = r.provenance
    assert isinstance(p, G4NDLInelasticProvenance) and p.channel == "F01" and p.composite
    qi = [s.body.QI for s in G4.inelasticFinalState("Pu244", "F01").sections
          if s.sfType == 51 and s.dataType == 3][0]
    assert r.outputChannel.Q.value == qi
    assert suite.reactions[16].outputChannel.Q.value == G4.inelasticFinalState("Pu244", "F04").Qvalue
    assert r.crossSection["recon"].interpolation.value == "lin-lin"
    for row in partialSumCheck(suite):
        assert row["maxRel"] < SUM_RTOL, row


def test_the_law_mapping_is_the_endf_adapters():
    def forms(lib, target):
        out = {}
        for r in LIBS[lib].read(target).reactions:
            for pr in r.outputChannel.products:
                if pr.distribution is not None:
                    out[(r.id.label, pr.label or pr.pid)] = pr.distribution["eval"]
        return out

    al = forms("G4NDL-4.7.1", "Al27")
    assert isinstance(al[("MT45", "n")], KalbachMann)
    assert isinstance(al[("MT45", "photon")], Uncorrelated)
    assert isinstance(al[("MT45", "photon")].angular, Isotropic2d)
    pu = forms("JEFF-4.0", "Pu240")
    assert isinstance(pu[("MT5", "U236")], Unspecified)
    bk = forms("JEFF-4.0", "Bk247")
    assert isinstance(bk[("MT37", "n")], EnergyAngular)
    b = forms("JEFF-4.0", "B10")
    assert b[("MT700", "Be8")].recoilHref.endswith("product[@label='H3']"
                                                  "/distribution/angularTwoBody[@label='eval']")
    fe = forms("G4NDL-4.7.1", "Fe58")
    assert isinstance(fe[("MT750", "He3")], AngularTwoBody)
    ba = forms("G4NDL-4.7.1", "Ba132")
    u = ba[("MT17", "n")]
    assert isinstance(u, Uncorrelated) and isinstance(u.energy, XYs2d)
    gd = forms("G4NDL-4.7.1", "Gd156")[("MT16", "n")]
    assert isinstance(gd, AngularTwoBody)    # MF5 NK=2 stays verbatim; MF4 is modelled
    assert isinstance(gd.angular, XYs2d) and isinstance(gd.angular.function1ds[0], XYs1d)


def test_the_photons_of_a_channel_reach_the_model():
    """D10-2: Hg-196 MT17's dataType 12, 14 and 15 are its photons' MF12-15."""
    suite = G4.read("Hg196")
    r = suite.reactions[17]
    p = r.provenance
    assert [e["dataType"] for e in p.sections if e.get("verbatim") is not None] == []
    photons = [x for x in r.outputChannel.products if x.pid == "photon"]
    assert photons and all(x.multiplicity is not None for x in photons)
    mean = next(e for e in p.sections if e["dataType"] in (12, 13))
    assert {"endf", "g4ndl", "targetMass"} <= set(mean["photons"])
    assert [e.get("photonsOf") for e in p.sections if e["dataType"] in (14, 15)] == [12, 12]


def test_code_1_is_read_lin_lin_and_written_back(tmp_path):
    """D10-3: Hg-196 MT17's MF5 spectra state INT=1, which Geant4 evaluates lin-lin.

    The model holds lin-lin, the report says it is an approximation of what
    ENDF would read, and the declared code goes back into the file.
    """
    suite = G4.read("Hg196")
    p = suite.reactions[17].provenance
    mf5 = [e for e in p.sections if e["dataType"] == 5][0]
    assert mf5.get("verbatim") is None and mf5["code1"]
    assert any("interpolation code 1" in m and "read lin-lin" in m
               for m in suite.report.approximations)
    energy = suite.reactions[17].outputChannel.products[0].distribution["eval"].energy
    for f in energy.function1ds if hasattr(energy, "function1ds") else []:
        assert f.interpolation.value != "flat"
    writeInelastic(suite, tmp_path)
    ch = channelOf(suite.reactions[17])
    assert inelasticDifferences(G4.inelasticFinalState("Hg196", ch),
                                g4ndl.open(tmp_path).inelasticFinalState("Hg196", ch)) == []


def test_a_histogram_in_the_model_is_not_written():
    """The other half of D10-3: Geant4 would read a written code 1 lin-lin."""
    from kika.g4ndl.inelastic_encode import _tab1
    from kika.nuclear_data.model.enums import Interpolation
    f = XYs1d(xs=np.array([1.0, 2.0, 3.0]), ys=np.array([1.0, 2.0, 0.0]),
              interpolation=Interpolation.flat)
    with pytest.raises(G4NDLUnsupportedError, match="code 1"):
        _tab1(f, "a histogram")


LAW7 = """\
1 3 -1665000.0 0
2
1.0e6 0.1 2.0e7 0.2
1 6
8.9348 1 1
1.0 1.0 0 7 -1665000.0 -1665000.0
2
1
2 2
1.0e6 2.0 2.0e7 2.0
2
1
2 2
1.0e6 2
1
2 2
-1.0
2
1
2 2
0.0 0.0 1.0e5 1.0
1.0
2
1
2 2
0.0 0.0 1.0e5 1.0
2.0e7 2
1
2 2
-1.0
2
1
2 3
1.0 0.0 1.0e7 1.0
1.0
2
1
2 2
0.0 0.0 1.0e7 1.0
"""


def test_a_lab_angle_energy_law_is_angularEnergy_and_comes_back():
    record = _text("F04", LAW7)
    from kika.g4ndl.decode import newSuite
    from kika.g4ndl.names import IsotopeKey

    suite, report = newSuite(IsotopeKey(4, 9))
    decodeInelastic(None, [record], suite, report=report)
    form = suite.reactions[16].outputChannel.products[0].distribution["eval"]
    assert isinstance(form, AngularEnergy) and len(form.xys3d) == 2
    _, files, _ = encodeInelastic(suite)
    assert inelasticDifferences(record, files["F04"]) == []


# ------------------------------------------------------------------ writing

def test_model_edits_are_what_is_written(tmp_path):
    suite = G4.read("Pu244")
    mt16 = suite.reactions[16]
    mt16.crossSection["recon"].ys[:] *= 1.10
    # Replace MT51's Legendre by an isotropic distribution: repFlag 0.
    mt51 = suite.reactions[51]
    frame = mt51.outputChannel.products[0].distribution["eval"].productFrame
    mt51.outputChannel.products[0].distribution["eval"] = AngularTwoBody(
        angular=Isotropic2d(productFrame=frame), productFrame=frame)
    report = writeInelastic(suite, tmp_path)
    back = g4ndl.open(tmp_path)
    f04 = back.inelasticFinalState("Pu244", "F04")
    orig = G4.inelasticFinalState("Pu244", "F04")
    assert np.array_equal(f04.sections[0].body.points.y, orig.sections[0].body.points.y * 1.10)
    f01 = back.inelasticFinalState("Pu244", "F01")
    angular = [s for s in f01.sections if s.sfType == 51 and s.dataType == 4][0].body
    assert isinstance(angular, AngularBody) and angular.repFlag == 0
    # The total followed MT16 (D10-1), the report says so, and the caller's
    # suite still holds the total it was read with.
    assert any("inelastic differs from the sum" in m and "rebuilt from its parts" in m
               for m in report.warnings)
    assert [row["maxRel"] < 1e-12 for row in partialSumCheck(back.read("Pu244"))
            if row["sum"] == INELASTIC_SUM_LABEL] == [True]
    assert [row["maxRel"] > SUM_RTOL for row in partialSumCheck(suite)
            if row["sum"] == INELASTIC_SUM_LABEL] == [True]


def test_a_sum_follows_its_partials_and_nothing_else_moves(tmp_path):
    """MT51 × 2: MT4 and the total are rebuilt, every other channel is untouched."""
    suite = G4.read("Pu244")
    suite.reactions[51].crossSection["recon"].ys[:] *= 2.0
    report = writeInelastic(suite, tmp_path)
    rebuilt = sorted(m.split()[0] for m in report.warnings if "rebuilt from its parts" in m)
    assert rebuilt == ["MT4", "inelastic"]
    back = g4ndl.open(tmp_path)
    assert all(row["maxRel"] < 1e-12 for row in partialSumCheck(back.read("Pu244")))
    for ch in ("F04", "F05", "F18"):
        assert inelasticDifferences(G4.inelasticFinalState("Pu244", ch),
                                    back.inelasticFinalState("Pu244", ch)) == []
    # The same rebuild, in the model: what was written is what it gives.
    assert rebuildInelasticSums(suite, only={"MT4"}) == ["MT4"]
    mt4 = [s for s in suite.sums if s.id.ENDF_MT == 4][0].crossSection["recon"]
    written = [s for s in back.inelasticFinalState("Pu244", "F01").sections
               if s.sfType == 4 and s.dataType == 3][0].body.points
    assert np.array_equal(mt4.xs, written.x) and np.array_equal(mt4.ys, written.y)


def test_editing_a_sum_its_partials_define_is_refused(tmp_path):
    suite = G4.read("Pu244")
    mt4 = [s for s in suite.sums if s.id.ENDF_MT == 4][0]
    mt4.crossSection["recon"].ys[:] *= 1.5
    with pytest.raises(G4NDLUnsupportedError, match="it is the sum that was edited"):
        writeInelastic(suite, tmp_path)
    assert not (tmp_path / "Inelastic").exists()


def test_a_sum_with_no_record_of_what_it_was_is_rebuilt(tmp_path):
    """A suite from an ENDF tape carries no digests: an inconsistent sum is derived."""
    suite = G4.read("Pu244")
    for s in suite.sums:
        s.provenance.crossSectionDigest = s.provenance.partsDigest = None
    mt4 = [s for s in suite.sums if s.id.ENDF_MT == 4][0]
    mt4.crossSection["recon"].ys[:] *= 1.5
    report = writeInelastic(suite, tmp_path)
    assert any(m.startswith("MT4 ") and "rebuilt" in m for m in report.warnings)


def test_write_removes_stale_channels_and_twins(tmp_path):
    suite = G4.read("Pu244")
    writeInelastic(suite, tmp_path, compressed=True)
    assert (tmp_path / "Inelastic/F18/94_244_Plutonium.z").is_file()
    suite.reactions.reactions[:] = [r for r in suite.reactions if r.id.ENDF_MT != 37]
    suite.sums.reactions[:] = [s for s in suite.sums if s.id.label != INELASTIC_SUM_LABEL]
    report = writeInelastic(suite, tmp_path, compressed=False)
    # The total is G4NDL's and the suite lost it: it is made from the parts.
    assert any(m.startswith("inelastic: not in the suite") for m in report.warnings)
    assert (tmp_path / "Inelastic/CrossSection/94_244_Plutonium").is_file()
    assert not (tmp_path / "Inelastic/F18/94_244_Plutonium.z").exists()
    assert not (tmp_path / "Inelastic/F04/94_244_Plutonium.z").exists()
    assert (tmp_path / "Inelastic/F04/94_244_Plutonium").is_file()
    assert any("no reaction for F18" in m for m in report.warnings)


def test_the_front_door_reads_and_writes_the_inelastic(tmp_path):
    suite = kika.read(DATA / "G4NDL-4.7.1", format="g4ndl", target="Pu244")
    assert any(r.id.ENDF_MT == 16 for r in suite.reactions)
    kika.write(suite, tmp_path, format="g4ndl", compressed=True)
    lib = g4ndl.open(tmp_path)
    for ch in G4.inelasticChannels("Pu244"):
        assert inelasticDifferences(G4.inelasticFinalState("Pu244", ch),
                                    lib.inelasticFinalState("Pu244", ch)) == []
    assert recordDifferences(G4.inelasticCrossSection("Pu244"),
                             lib.inelasticCrossSection("Pu244")) == []


def test_elastic_and_inelastic_together(tmp_path):
    """An elastic library and an inelastic one merged: both processes, one suite."""
    root = tmp_path / "lib"
    shutil.copytree(DATA.parent / "G4NDL-4.7.1", root)   # Elastic: 6_nat_Carbon
    shutil.copytree(DATA / "G4NDL-4.7.1" / "Inelastic", root / "Inelastic")
    lib = g4ndl.open(root)
    assert lib.read("Cnat").reactions[2].id.ENDF_MT == 2    # no inelastic for it
    suite = lib.read("Pu244")
    assert not any("Elastic/" in m for m in suite.report.unsupported)   # it has none
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        writeSuite(suite, out, processes=["thermal"])
    writeSuite(suite, out, processes=["inelastic"])
    assert sorted(p.name for p in (out / "Inelastic").iterdir()) == ["CrossSection", "F01",
                                                                    "F04", "F05", "F18"]


# ------------------------------------------------------------------ patch

@pytest.fixture
def base(tmp_path):
    root = tmp_path / "base"
    shutil.copytree(DATA / "G4NDL-4.7.1", root)
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
    suite.reactions[17].crossSection["recon"].ys[:] *= 0.9
    gammas = G4.gammas(6, 15)
    result = g4ndl.patch_isotope(base, suite, tmp_path / "out", share=share,
                                 gammas=[gammas])
    assert _tree(base) == before
    after = _tree(tmp_path / "out")
    changed = sorted(p for p in before if after.get(p) != before[p])
    # F05 (the edit), the total (rebuilt), the other channels rewritten at 17
    # digits, and the level scheme rewritten as triples.
    assert "Inelastic/F05/94_244_Plutonium.z" in changed
    assert "Inelastic/CrossSection/94_244_Plutonium.z" in changed
    assert all("94_244_Plutonium" in p or p.startswith("Inelastic/Gammas/z6.a15")
               for p in changed)
    assert set(after) - set(before) == {MANIFEST_NAME}
    manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    assert manifest["processes"] == ["inelastic"]
    assert manifest["channels"] == ["F01", "F04", "F05", "F18"]
    lib = g4ndl.open(tmp_path / "out")
    # MT17 is no part of MT4, so F01 says exactly what it said.
    assert inelasticDifferences(G4.inelasticFinalState("Pu244", "F01"),
                                lib.inelasticFinalState("Pu244", "F01")) == []
    rows = {row["sum"]: row["maxRel"] for row in partialSumCheck(lib.read("Pu244"))}
    assert rows[INELASTIC_SUM_LABEL] < 1e-12 and rows["MT4"] < SUM_RTOL
    assert inelasticDifferences(gammas, lib.gammas(6, 15)) == []
    assert result.removed == []


def test_patch_removes_a_dropped_channel(tmp_path, base):
    suite = g4ndl.open(base).read("Pu244")
    suite.reactions.reactions[:] = [r for r in suite.reactions if r.id.ENDF_MT != 37]
    result = g4ndl.patch_isotope(base, suite, tmp_path / "out")
    assert result.removed == ["Inelastic/F18/94_244_Plutonium.z"]
    assert not (tmp_path / "out/Inelastic/F18/94_244_Plutonium.z").exists()


def test_the_inelastic_modules_do_not_wake_the_model():
    code = textwrap.dedent("""
        import sys
        import kika.g4ndl.inelastic_parse, kika.g4ndl.inelastic_format
        import kika.g4ndl
        lib = kika.g4ndl.open(sys.argv[1])
        lib.inelasticFinalState("Pu244", "F01")
        print(any(m.startswith("kika.nuclear_data.model") for m in sys.modules))
    """)
    out = subprocess.run([sys.executable, "-c", code, str(DATA / "G4NDL-4.7.1")],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
