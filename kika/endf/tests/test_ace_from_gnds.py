"""An ACE table from a GNDS evaluation is the one its ENDF-6 tape gives.

Two differences in how GNDS states the same physics broke that, both found
on ENDF/B-VIII.1 H-1 (2026-10-26):

* the capture photon's multiplicity is a ``constant1d``, where the tape has a
  two-point MF12 yield, and the photon block could not encode it;
* elastic lists the target's recoil among its products (H1 on H-1, H2 on
  H-2), which the gas production counted as an emitted proton, so MT203 came
  out equal to the elastic cross section. GASPR counts no gas from elastic.
"""
from pathlib import Path

import pytest

import kika
from kika.ace.parsers.parse_ace import read_ace

DATA = Path(__file__).resolve().parents[2] / "gnds/tests/data"


@pytest.mark.parametrize("name, recoil", [("n-001_H_001", 203), ("n-001_H_002", 204)])
def test_hydrogen_from_gnds_writes_ace_without_elastic_recoil_gas(tmp_path, name, recoil):
    processing = kika.process(DATA / f"{name}.endf.gnds.xml", temperatures=[293.6])
    ace = read_ace(str(processing.write(tmp_path, "ace")[0]))
    assert 102 in ace.mt_numbers
    assert ace.photon_production_total() is not None
    if recoil in ace.mt_numbers:
        gas = ace.get_cross_section(recoil)[f"MT={recoil}"].to_numpy()
        elastic = ace.get_cross_section(2)["MT=2"].to_numpy()
        assert gas.max() < 0.01 * elastic.max()
