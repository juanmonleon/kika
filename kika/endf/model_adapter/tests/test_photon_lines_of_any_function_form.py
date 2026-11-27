"""MF6 photon lines are derived whatever form a multiplicity or spectrum node takes.

A photon of a GNDS file can state its multiplicity as a constant1d or a
regions1d. The line-spectrum writer read ``.xs`` off each one and failed on
the first constant (NNDC's Ca-40, Cu-63, Cu-65, V-51). Every table is now read
through one helper that gives a constant its two ends and a step its two
values; V-51's photons come back as the tape states them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kika._nuclear_data import nuclear_data

import kika
from kika.endf import read_endf
from kika.endf.model_adapter.photon_lines import _points
from kika.nuclear_data.model.functions.simple import Constant1d


def test_a_constant_is_read_as_its_two_ends():
    constant = Constant1d(constant=2.0, domainMin_=1e-5, domainMax_=2e7)
    assert [list(v) for v in _points(constant)] == [[1e-5, 2e7], [2.0, 2.0]]


def _nndc(name):
    tapes = nuclear_data("endf") or Path("missing")
    gnds = tapes.parent / "gnds" / "endfb81"
    endf, xml = tapes / "endfb81" / f"{name}.endf", gnds / f"{name}.endf.gnds.xml"
    if not (endf.is_file() and xml.is_file()):
        pytest.skip(f"needs {endf} and {xml}")
    return endf, xml


def _photons(path):
    tape = read_endf(str(path), mf_numbers=[6, 12, 13, 14, 15])
    mf6 = {mt: [repr(p) for p in section.products if p.zap == 0]
           for mt, section in tape.mf[6].mt.items()}
    rest = {(mf, mt): repr(section) for mf in (12, 13, 14, 15) if mf in tape.mf
            for mt, section in tape.mf[mf].mt.items()}
    return {mt: v for mt, v in mf6.items() if v}, rest


def test_v51_photons_come_back_as_the_tape_states_them(tmp_path):
    endf, xml = _nndc("n-023_V_051")
    out = tmp_path / "v51.endf"
    kika.write(kika.read(str(xml), covariances=False), str(out), format="endf")
    assert _photons(out) == _photons(endf)
