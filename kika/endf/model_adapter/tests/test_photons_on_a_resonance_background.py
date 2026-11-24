"""MF13 on a reaction whose GNDS σ is a resonancesWithBackground comes back as the tape has it.

The GNDS written from a tape wraps MF3 of the resonance MTs as resonancesWithBackground,
so the σ an MF13 section is stated against is that background table, not a function.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import kika
from kika.endf import read_endf


def _tape(name):
    tapes = Path(os.environ.get("KIKA_LIB_TAPES", "/share_snc/lib/endf"))
    path = tapes / "endfb81" / f"{name}.endf"
    if not path.exists():
        pytest.skip(f"{path} not available")
    return path


def _rows(path, mf, mt):
    with open(path, encoding="latin-1") as f:
        return [line[:66] for line in f if line[70:72].strip() == str(mf) and line[72:75].strip() == str(mt)]


def test_u237_mf13_on_mt3_through_kika_gnds(tmp_path):
    tape = _tape("n-092_U_237")
    gnds, back = tmp_path / "u237.gnds.xml", tmp_path / "u237.endf"
    kika.write(kika.read(str(tape)), str(gnds), format="gnds")
    kika.write(kika.read(str(gnds)), str(back), format="endf")

    assert read_endf(str(back), mf_numbers=[13]).mf[13].mt[3] is not None
    for mt in (1, 2, 18, 102):
        assert _rows(back, 3, mt) == _rows(tape, 3, mt), f"MF3/MT{mt}"
