"""What an ENDF tape keeps only in provenance is declared lost when the suite goes to GNDS."""
from __future__ import annotations

from pathlib import Path

import pytest

import kika
from kika.endf.model_adapter.fission_energy import MF1MT460_KEY
from kika.endf.model_adapter.productions import (ACTIVATION_MF8_OPAQUE_KEY,
                                                 ACTIVATION_VERBATIM_KEY)
from kika.endf.model_adapter.tape_only import tapeOnlyData

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
PREFIX = "not in the GNDS file, kept only in the suite's ENDF provenance"


def _tapeOnlyLosses(report):
    return [message for message in report.losses if message.startswith(PREFIX)]


def _writeGnds(suite, tmp_path):
    return kika.write(suite, tmp_path / "out.xml", format="gnds")


@pytest.mark.parametrize("tape, expected", [
    ("micro_u235_mf6.endf", "MF6: P(nu), the probability that one fission emits nu "
                            "particles (MT18: 14 neutron and 40 photon subsections, and the "
                            "y(E) of its 2 LAW=0 subsections)"),
    ("micro_u235_fission_energy.endf", "MF1/458 (LFC=1): the thermal values, their "
                                       "uncertainties and the LDRV of promptProductKE"),
    ("micro_fe56_photons.endf", "photons: the sections kika could not model (MF12/52, "
                                "MF14/52"),
    ("micro_am241_activation.endf", "MF8 (NO=0): the decay data stated with the product "
                                    "states (MT102)"),
])
def test_what_the_reader_kept_for_the_tape_is_declared_once(tape, expected, tmp_path):
    suite = kika.read(DATA / tape)
    report = _writeGnds(suite, tmp_path)
    declared = _tapeOnlyLosses(report)
    assert len(declared) == 1, declared
    assert expected in declared[0]


def test_the_declarations_do_not_block_the_cross_section_scope(tmp_path):
    suite = kika.read(DATA / "micro_u235_mf6.endf")
    report = _writeGnds(suite, tmp_path)
    excluded = {entry["message"] for entry in report.scopeExclusions
                if "cross-sections" in entry["scopes"]}
    assert set(_tapeOnlyLosses(report)) <= excluded


def test_a_tape_with_nothing_kept_aside_declares_nothing(tmp_path):
    suite = kika.read(DATA / "micro_fe56_xs_and_angular.endf")
    assert _tapeOnlyLosses(_writeGnds(suite, tmp_path)) == []


def test_a_suite_read_back_from_gnds_declares_nothing(tmp_path):
    suite = kika.read(DATA / "micro_u235_mf6.endf")
    _writeGnds(suite, tmp_path)
    again = kika.read(tmp_path / "out.xml")
    assert _tapeOnlyLosses(kika.write(again, tmp_path / "again.xml", format="gnds")) == []


def test_the_suite_level_stores_and_ldg1_are_each_named(tmp_path):
    suite = kika.read(DATA / "micro_u235_mf6.endf")
    fields = suite.provenance.headerFields
    fields[MF1MT460_KEY] = ["record"]
    fields[ACTIVATION_MF8_OPAQUE_KEY] = {102: [[92236, 1], [92236, 2]]}
    fields[ACTIVATION_VERBATIM_KEY] = {"10/102": ["record"]}
    suite.reactions[18].provenance.headerFields["ldg"] = 1
    found = [sentence[sentence.index("back: ") + 6:] for sentence in tapeOnlyData(suite)]
    assert [sentence.split(" ")[0].rstrip(":") for sentence in found] == [
        "MF1/455", "MF1/460", "MF6", "MF8", "MF9/MF10"]
    assert found[3] == "MF8: 2 product state(s) that no MF9 or MF10 describes (MT102)"
    assert found[4] == "MF9/MF10: the sections kika could not model (MF10/102)"


def test_another_format_names_itself():
    from kika._write import _declareTapeOnly
    from kika.nuclear_data.model import ConversionReport

    suite = kika.read(DATA / "micro_u235_mf6.endf")
    report = _declareTapeOnly(suite, ConversionReport(), "the ACE table")
    assert [m.split(",")[0] for m in report.losses] == ["not in the ACE table"]


@pytest.mark.parametrize("tape, expected", [
    ("dec-026_Fe_056.endf", []),
    ("dec-055_Cs_137.endf", ["MF8/MT457: each spectrum's normalisation (FD, FC), its ERAV "
                             "and the order of its lines"]),
    ("dec-094_Pu_244.endf", ["MF8/MT457: each spectrum's normalisation (FD, FC), its ERAV "
                             "and the order of its lines",
                             "the sections a decay PoPs has no node for (MF1/MT452)"]),
])
def test_a_decay_pops_declares_what_it_keeps_for_the_tape(tape, expected, tmp_path):
    report = kika.write(kika.read(DATA / tape), tmp_path / "out.xml", format="gnds")
    declared = [m.split("back: ", 1)[1] for m in report.losses
                if m.startswith("not in the GNDS file, kept only in the PoPs' ENDF provenance")]
    assert declared == expected


def test_a_level_qm_off_its_pops_energy_is_declared(zr90_b81_tape):
    """B-VIII.1 Zr-90 MT651: MF3's QM - QI is 908970 eV, MF8 puts Y89_e1 at 909000 eV."""
    found = [s.split("back: ", 1)[1] for s in tapeOnlyData(kika.read(zr90_b81_tape, covariances=False))]
    assert [s for s in found if s.startswith("MF3/")] == [
        "MF3/MT651: QM -6128816 eV; QI plus Y89_e1's energy gives -6128786 eV"]
