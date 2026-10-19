"""Dosimetry ACE tables (``.NNy``): encoder, decoder and writer against NJOY's layout.

The synthetic table is built from known reactions (a lin-lin one, one with two
interpolation regions, an MF 10 production) and must come back through
``write_ace`` and ``read_ace`` word for word, with the layout NJOY2016's
``dosout`` writes (``acedo.f90``): JXS(1) = JXS(3) = 1, LSIG right after MTR,
SIGD after LSIG, JXS(22) the length, NBT and INT as two arrays.

Tables written by NJOY2016 ACER (``iopt=3``) are looked for in
``$KIKA_DOSIMETRY`` or ``../kika-files/dosimetry`` next to the repo; each must be
read and written back byte for byte (line ends aside), and re-encoded from its decoded reactions
byte for byte too.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.classes.dosimetry import DosimetryAce, DosimetryReaction, production_mt, split_production_mt
from kika.ace.encoders.dosimetry import encode_dosimetry
from kika.ace.parsers.parse_dosimetry import is_dosimetry_extension, parse_dosimetry
from kika.ace.writers.write_ace import write_ace, write_ace_checked

REACTIONS = [
    DosimetryReaction(102, np.array([1e-5, 0.0253, 1e3, 2e7]), np.array([30.0, 1.5, 0.01, 1e-4])),
    DosimetryReaction(103, np.array([1e6, 2e6, 5e6, 1e7, 2e7]), np.array([0.0, 0.01, 0.2, 0.3, 0.1]),
                      breakpoints=[2, 5], interpolation=[2, 5]),
    DosimetryReaction(production_mt(4, 1), np.array([2e5, 1e6, 2e7]), np.array([0.0, 0.05, 0.02])),
]


def _table():
    return encode_dosimetry(REACTIONS, zaid=41093, extension=".80y", za=41093, awr=92.1051,
                            temperature_mev=2.53e-8, comment="synthetic", date="10/20/26", matid=4125)


def test_production_numbers_follow_acer():
    assert production_mt(4, 1) == 11004
    assert production_mt(5, 0) == 10005
    assert split_production_mt(11004) == (4, 1)
    assert split_production_mt(102) is None
    assert is_dosimetry_extension(".80y") and not is_dosimetry_extension(".80c")


def test_layout_is_dosout():
    t = _table()
    nxs, jxs, xss = t.header.nxs_array, t.header.jxs_array, t.xss_data
    ntr = len(REACTIONS)
    assert (nxs[1], nxs[2], nxs[3], nxs[4]) == (xss.size - 1, 41093, 0, ntr)
    assert (jxs[1], jxs[3], jxs[6], jxs[7], jxs[22]) == (1, 1, 1 + ntr, 1 + 2 * ntr, xss.size - 1)
    assert list(xss[1:1 + ntr]) == [102, 103, 11004]
    assert xss[jxs[6]] == 1.0                                   # the first reaction starts SIGD
    second = jxs[7] + int(xss[jxs[6] + 1]) - 1
    assert list(xss[second:second + 6]) == [2, 2, 5, 2, 5, 5]   # NR, NBT(2), INT(2), NE


def test_round_trip_through_the_file(tmp_path):
    path = tmp_path / "41093.80y"
    write_ace_checked(_table(), path)
    back = read_ace(str(path))
    assert isinstance(back, DosimetryAce)
    assert back.mt_numbers == [102, 103, 11004] and back.repeated_numbers == []
    for want, got in zip(REACTIONS, back.reactions):
        np.testing.assert_allclose(got.energies, want.energies, rtol=1e-12)
        np.testing.assert_allclose(got.cross_section, want.cross_section, rtol=1e-12)
        assert (got.breakpoints, got.interpolation) == (want.breakpoints, want.interpolation)
    assert back.reaction(11004).production == (4, 1)
    again = tmp_path / "again.80y"
    write_ace(back, str(again))
    assert again.read_bytes() == path.read_bytes()


def test_repeated_numbers_are_kept_and_named(tmp_path):
    twice = REACTIONS + [DosimetryReaction(production_mt(4, 1), np.array([3e5, 2e7]), np.array([0.0, 0.01]))]
    t = encode_dosimetry(twice, zaid=41093, extension=".80y", za=41093, awr=92.1051, temperature_mev=0.0)
    path = tmp_path / "t.80y"
    write_ace(t, str(path))
    back = read_ace(str(path))
    assert back.repeated_numbers == [11004]
    with pytest.raises(ValueError, match="several MF 10 products"):
        back.reaction(11004)


def test_interleaved_nbt_int_is_refused():
    t = _table()
    xss = t.xss_data.copy()
    jxs = t.header.jxs_array
    second = jxs[7] + int(xss[jxs[6] + 1]) - 1
    xss[second + 1:second + 5] = [2, 2, 5, 5]                    # NJOY's MF 10 spelling: (NBT, INT) pairs
    with pytest.raises(ValueError, match="interleaved"):
        parse_dosimetry(t.header, xss)


def _njoy_tables():
    root = os.environ.get("KIKA_DOSIMETRY")
    root = Path(root) if root else Path(__file__).resolve().parents[3] / ".." / "kika-files" / "dosimetry"
    return sorted(root.glob("*.80y")) if root.is_dir() else []


@pytest.mark.parametrize("path", _njoy_tables() or [None], ids=lambda p: p.name if p else "none")
def test_njoy_tables_are_written_back_byte_for_byte(path, tmp_path):
    if path is None:
        pytest.skip("no NJOY dosimetry tables ($KIKA_DOSIMETRY or ../kika-files/dosimetry)")
    t = read_ace(str(path))
    h = t.header
    # NJOY built with gfortran on Windows ends its lines in CRLF; ACE is LF.
    reference = path.read_bytes().replace(b"\r\n", b"\n")
    out = tmp_path / "out.80y"
    write_ace(t, str(out))
    assert out.read_bytes() == reference
    again = encode_dosimetry(t.reactions, zaid=h.zaid, extension=h.extension, za=t.za, awr=h.atomic_weight_ratio,
                             temperature_mev=h.temperature, comment=h.comment, date=h.date, matid=h.matid)
    out2 = tmp_path / "again.80y"
    write_ace(again, str(out2))
    assert out2.read_bytes() == reference
