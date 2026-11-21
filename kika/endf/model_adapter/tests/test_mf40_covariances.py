"""MF40 ↔ ``covarianceSuite``: the covariances of the MF10 productions.

On two ENDF/B-VIII.1 cuts: Nb-93 MT4, one final state (Nb-93m, a dosimetry
reaction), and Mn-54 MT17, two final states (Mn-52 and Mn-52m). Pinned: one
section per state block, linked to the production it is about; the matrix the
MF33 code reads from the same records; the tape fixed point of MF40 (byte for
byte where the source spells its numbers canonically); and the
cases that cannot be modelled, reported and not invented.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf.classes.mf33.mf33 import MF33MT
from kika.endf.model_adapter import decodeCovarianceSuite, decodeMF40MT, encodeMF40MT
from kika.endf.read_endf import read_endf
from kika.endf.utils import parse_endf_id
from kika.endf.writers.assemble import writeEndfTape

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
NB93 = DATA / "micro_nb93_b81_mf40.endf"
MN54 = DATA / "micro_mn54_b81_mf40.endf"


def _mf40_lines(path):
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if len(line) >= 75:
            _, mf, mt = parse_endf_id(line)
            if mf == 40 and mt:
                out[mt].append(line[:66])
    return dict(out)


@pytest.mark.parametrize("path, labels", [
    (NB93, {"MF40-MT4-LFS1": "MT4 Nb93_e1"}),
    (MN54, {"MF40-MT17-LFS0": "MT17 Mn52", "MF40-MT17-LFS1": "MT17 Mn52_e1"}),
], ids=["nb93", "mn54"])
def test_each_state_block_is_a_section_linked_to_its_production(path, labels):
    suite = kika.read(path)
    sections = {s.label: s for s in suite.covarianceSuite.covarianceSections if s.rowData.ENDF_MF == 40}
    assert set(sections) == set(labels)
    productions = {p.label for p in suite.productions}
    for label, production in labels.items():
        section = sections[label]
        assert section.rowData.href.split("'")[1] == production and production in productions
        assert section.columnData is None and section.form.isRelative


def test_the_matrix_is_what_the_mf33_code_reads_from_the_same_records():
    endf = read_endf(str(NB93))
    state = endf.mf[40].mt[4].states[0]
    section40 = endf.mf[40].mt[4]
    expected = state.as_mf33(4, section40._mat, za=section40._za, awr=section40._awr).to_xs_covmat()
    section = next(s for s in kika.read(NB93).covarianceSuite.covarianceSections if s.rowData.ENDF_MF == 40)
    np.testing.assert_array_equal(section.form.matrix, expected.matrices[0])
    np.testing.assert_array_equal(section.form.rowGrid, expected.energy_grids[0])


@pytest.mark.parametrize("path", [NB93, MN54], ids=["nb93", "mn54"])
def test_mf40_is_written_back_as_read(path, tmp_path):
    suite = kika.read(path)
    out = tmp_path / "written.endf"
    writeEndfTape(suite, out)
    # The records as read, spelled canonically: Mn-54 writes its HEAD as 25054.0000, not 2.505400+4.
    written, source = read_endf(str(out)).mf[40], read_endf(str(path)).mf[40]
    assert sorted(written.mt) == sorted(source.mt)
    assert all(str(written.mt[mt]) == str(source.mt[mt]) for mt in source.mt)
    if path == NB93:
        assert _mf40_lines(out) == _mf40_lines(path)


def test_a_state_that_is_not_a_nuclide_is_reported_not_invented():
    endf = read_endf(str(MN54))
    section = endf.mf[40].mt[17]
    section.states[1].izap = -1
    sections, report = decodeMF40MT(section)
    assert [s.label for s in sections] == ["MF40-MT17-LFS0"]
    assert any("not a nuclide" in m for m in report.losses)


def test_a_state_mf10_does_not_give_is_warned_about():
    endf = read_endf(str(NB93))
    endf.mf[40].mt[4].states[0].lfs = 21
    for state in endf.mf[40].mt[4].states:
        for sub in state.subsections:
            sub.xlfs1 = 21.0
    suite, report = decodeCovarianceSuite(endf, checks=False, target="Nb93")
    assert any("MF10/MT4 has no state" in m for m in report.warnings)


def test_a_section_without_its_ens_header_is_not_written():
    suite = kika.read(NB93)
    section = next(s for s in suite.covarianceSuite.covarianceSections if s.rowData.ENDF_MF == 40)
    section.provenance.headerFields.clear()
    built, report = encodeMF40MT(suite.covarianceSuite, 4)
    assert built is None and any("MF40/MT4" in m for m in report.unsupported)


def test_a_view_of_one_subsection_is_an_mf33_section():
    endf = read_endf(str(MN54))
    sub = endf.mf[40].mt[17].states[1].subsections[0]
    view = MF33MT(number=17, _za=endf.mf[40].mt[17]._za, _awr=endf.mf[40].mt[17]._awr, _mtl=0, _nl=1, _mf=40)
    view._subsections = [sub]
    assert view.to_xs_covmat().matrices[0].shape == (3, 3)
