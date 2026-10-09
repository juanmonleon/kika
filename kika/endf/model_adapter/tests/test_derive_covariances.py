"""G6: MF31-35 from a covariance suite read from GNDS.

NNDC's ENDF/B-VIII.1 GNDS covariance files were written by FUDGE from the ENDF
tapes, so the tape is the reference: the sections kika writes from the GNDS are
compared with the tape's, number for number (blank and zero read alike).

Measured 2026-10-09: U-235 -- MF31 3/3, MF33 87/87 (the 76 lumped components
MT52+ among them, from the summands of `lump0`/`lump1`, and the cross-material
blocks with Li-6, Au-197, Pu-239 and U-238), MF34 1/1 (LTT=3), MF35 1/1 --
and Fe-56 MF33 7/7 identical.

What it took, all in the writer and none of it guessed: the sections' header
from the suite (``derive/covariances.py``); each ``mixed`` component as its own
NI sub-subsection with FUDGE's LB rule, a ``sum`` as an NC record; the array's
storage and whether the column axis links the row axis, which the GNDS reader
now keeps on ``CovarianceMatrix``; MAT1 from the external file a column points
into; MF34's LTT=3 when an order-0 row is stated.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import kika
from kika.endf import read_endf
from kika.endf.writers.assemble import writeEndfTape

pytestmark = pytest.mark.slow


def _gnds() -> Path:
    stated = os.environ.get("KIKA_GNDS_LIB")
    if stated:
        return Path(stated)
    tapes = os.environ.get("KIKA_LIB_TAPES")
    return Path(tapes).parent / "gnds" / "endfb81" if tapes else Path("missing")


def _fields(line):
    text = line[:66].ljust(66)
    return [text[k:k + 11].strip() or "0" for k in range(0, 66, 11)]


@pytest.mark.parametrize("name, counts", [
    ("n-092_U_235", {31: 3, 33: 87, 34: 1, 35: 1}),
    ("n-026_Fe_056", {33: 7}),
])
def test_covariances_from_nndcs_gnds_are_the_tapes(name, counts, neutron_libraries, tmp_path):
    source = _gnds() / f"{name}.endf.gnds.xml"
    tape = neutron_libraries["endfb81"] / f"{name}.endf"
    if not source.is_file() or not tape.is_file():
        pytest.skip(f"{source} or {tape} is not here")
    out = tmp_path / "from_gnds.endf"
    writeEndfTape(kika.read(source), out)
    mine = read_endf(str(out), mf_numbers=list(counts))
    theirs = read_endf(str(tape), mf_numbers=list(counts))
    for mf, expected in counts.items():
        a, b = theirs.mf[mf].mt, mine.mf[mf].mt
        assert sorted(a) == sorted(b) and len(a) == expected, mf
        for mt in a:
            assert [_fields(l) for l in str(b[mt]).splitlines()] == \
                   [_fields(l) for l in str(a[mt]).splitlines()], f"{name} MF{mf}/MT{mt}"


def test_the_gnds_reader_keeps_the_array_storage(neutron_libraries):
    source = _gnds() / "n-026_Fe_056.endf.gnds.xml"
    if not source.is_file():
        pytest.skip(f"{source} is not here")
    suite = kika.read(source)
    storages = set()
    for section in suite.covarianceSuite.covarianceSections:
        forms = getattr(section.form, "components", [section.form])
        storages.update(getattr(f, "arrayStorage", None) for f in forms)
    assert storages == {"symmetric"}
