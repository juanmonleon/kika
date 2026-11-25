"""kika's GNDS carries what the tape states and GNDS cannot, and its ENDF states it again."""
from __future__ import annotations

from pathlib import Path

import pytest

import kika
from kika.nuclear_data.model.endf_conversion import EndfTapeData

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
PREFIX = "not in the GNDS file, kept only in the suite's ENDF provenance"


def _sections(path, mf):
    """MF's sections as their records' numbers: a blank field and a 0 state the same."""
    from kika._records import parse_number

    def record(line):
        return [parse_number(line[k:k + 11]) or 0 for k in range(0, 66, 11)]

    out = {}
    for line in open(path, encoding="latin-1"):
        if line[70:72].strip() == str(mf) and int(line[72:75] or 0):
            out.setdefault(int(line[72:75]), []).append(record(line))
    return out


def _roundTrip(tape, tmp_path, covariances=False):
    suite = kika.read(str(tape), covariances=covariances)
    report = kika.write(suite, tmp_path / "out.xml", format="gnds")
    back = kika.read(tmp_path / "out.xml", covariances=covariances)
    kika.write(back, tmp_path / "back.endf", format="endf", mat=suite.provenance.mat)
    return report, back, tmp_path / "back.endf"


def _stripped(gnds, tmp_path):
    import re

    text = re.sub(r'\n\s*<institution label="KIKA::[^"]*">.*?</institution>', "",
                  Path(gnds).read_text(encoding="utf-8"), flags=re.S)
    (tmp_path / "stripped.xml").write_text(text, encoding="utf-8")
    return tmp_path / "stripped.xml"


@pytest.mark.tape
def test_a_level_qm_off_its_pops_energy_comes_back(zr90_b81_tape, tmp_path):
    """B-VIII.1 Zr-90 MT651: QM -6128816 eV, where QI plus Y89_e1's PoPs energy gives -6128786."""
    report, back, endf = _roundTrip(zr90_b81_tape, tmp_path)
    assert not [m for m in report.losses if m.startswith(PREFIX) and "MF3/" in m]
    assert EndfTapeData.of(back).nodes["suite.reactions[MT651]"] == {"qm": -6128816.0}
    assert _sections(endf, 3)[651] == _sections(zr90_b81_tape, 3)[651]

    bare = kika.read(_stripped(tmp_path / "out.xml", tmp_path), covariances=False)
    assert EndfTapeData.of(bare) is None
    kika.write(bare, tmp_path / "bare.endf", format="endf", mat=4025)
    assert _sections(tmp_path / "bare.endf", 3)[651][1][0] == -6128786.0


def test_the_block_reads_back_what_it_wrote():
    import xml.etree.ElementTree as ET

    import numpy as np

    from kika.gnds import tape_data

    data = EndfTapeData(nodes={"suite": {"headerFields": {16: [" 4.0+4", (1, 2.5)], "x": None}}},
                        mf32={"awri": {0: 102.021, 1: 0.1 + 0.2}},
                        background={"MT1": np.array([1e-5, 862047.8])})
    institution = ET.Element("institution")
    tape_data.write(institution, data)
    back = tape_data.read(ET.fromstring(ET.tostring(institution))[0])
    assert back.nodes == data.nodes and back.mf32 == data.mf32
    assert back.background["MT1"].tolist() == [1e-5, 862047.8]


@pytest.mark.parametrize("tape, sections", [
    ("micro_u235_fission_energy.endf", [(1, 458)]),
    ("micro_u235_mf6.endf", [(6, 18)]),
    ("micro_am241_activation.endf", [(8, 102)]),
    ("micro_fe56_photons.endf", [(12, 52), (12, 55), (14, 52), (14, 55)]),
])
def test_the_sections_the_reader_kept_for_the_tape_come_back(tape, sections, tmp_path):
    report, back, endf = _roundTrip(DATA / tape, tmp_path)
    assert not [m for m in report.losses if m.startswith(PREFIX)]
    for mf, mt in sections:
        assert _sections(endf, mf).get(mt) == _sections(DATA / tape, mf)[mt], (mf, mt)
