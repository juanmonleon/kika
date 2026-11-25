"""MF32 written from a GNDS parameter covariance, as FUDGE's toENDF6 writes it.

From NNDC's ENDF/B-VIII.1 GNDS (FUDGE's translation) and from kika's own, the
MF32 of Na-23 (LRF=2, LCOMP=2), Am-241 (LRF=2, LCOMP=0) and Mn-55 (LRF=3,
LCOMP=1) is the tape's, record for record. Cl-35 (LRF=7, LCOMP=2) states the
tape's covariance; its records differ where no GNDS node holds the tape's
choice: NJSX, the capture residual's ZB, and how the INTG records split a row.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf import read_endf
from kika.endf.model_adapter.parameter_covariances import decodeMF32MT


def _paths(name):
    tapes = Path(os.environ.get("KIKA_LIB_TAPES", "/share_snc/lib/endf"))
    gnds = Path(os.environ.get("KIKA_GNDS_LIB", tapes.parent / "gnds" / "endfb81"))
    endf, xml = tapes / "endfb81" / f"{name}.endf", gnds / f"{name}.endf.gnds.xml"
    if not (endf.is_file() and xml.is_file()):
        pytest.skip(f"needs {endf} and {xml}")
    return endf, xml


def _mf32(path):
    lines = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            if line[70:75] == "32151":
                lines.append(line[:66])
    return lines


def _written(xml, tmp_path):
    out = tmp_path / f"{Path(xml).stem}.endf"
    report = kika.write(kika.read(str(xml), covariances=True), str(out), format="endf")
    assert not any("MF32" in message for message in report.losses)
    return out


def _matrix(path):
    (covariance,) = decodeMF32MT(read_endf(str(path), mf_numbers=[32]).mf[32].mt[151])[0]
    return np.asarray(covariance.form.matrix)


@pytest.mark.parametrize("name", ["n-011_Na_023", "n-095_Am_241", "n-025_Mn_055"])
def test_nndc_gnds_gives_the_tapes_mf32(name, tmp_path):
    endf, xml = _paths(name)
    assert _mf32(_written(xml, tmp_path)) == _mf32(endf)


def test_cl35_states_the_tapes_covariance(tmp_path):
    endf, xml = _paths("n-017_Cl_035")
    out = _written(xml, tmp_path)
    assert np.array_equal(_matrix(out), _matrix(endf))


def test_kikas_own_gnds_of_na23_gives_the_tapes_mf32(tmp_path):
    endf, _ = _paths("n-011_Na_023")
    gnds = tmp_path / "na23.gnds.xml"
    kika.write(kika.read(str(endf), covariances=True), str(gnds), format="gnds")
    assert _mf32(_written(gnds, tmp_path)) == _mf32(endf)
