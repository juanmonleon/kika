"""The XSS is one float64 array, and every block is a window onto it.

``read_xss`` used to build one ``XssEntry(index, value)`` object per number:
~145 bytes each against 8 for a float64, which made a parsed ACE table ~8x
its size on disk (U-238 from ENDF/B-VIII.0: 1.7 M numbers, 249 of 266 MB held).
Blocks shared those objects with the XSS list, and that sharing is what let a
perturbation written into a block reach the file. With an array the same job
falls to numpy views, and these tests pin the three ways that can go wrong:
the reader disagreeing with the fixed-width format, a block that is a copy,
and ``Ace.copy()`` detaching the blocks of the copy from its own array.

Tape-free, so they run in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.ace.classes.ace import Ace
from kika.ace.classes.header import Header
from kika.ace.classes.xss import xss_position
from kika.ace.parsers.parse_ace import read_xss
from kika.ace.parsers.parse_esz import read_esz_block


def _lines(fields):
    """4E20 lines, as an ACE file lays them out."""
    return ["".join(fields[i:i + 4]) + "\n" for i in range(0, len(fields), 4)]


VALUES = [1.0e-11, 2.5, -3.25e-100, 4.0, 1.23456789012e5, 6.0, -7.5]
FIELDS = [f"{v:20.11E}" for v in VALUES]


def test_the_xss_is_float64_with_a_placeholder_at_zero():
    xss = read_xss(_lines(FIELDS), len(VALUES))
    assert xss.dtype == np.float64
    assert xss[0] == 0.0
    np.testing.assert_array_equal(xss[1:], VALUES)


def test_integers_are_read_as_floats():
    fields = [f"{v:20d}" for v in (11, 2, 3)] + [FIELDS[1]]
    np.testing.assert_array_equal(read_xss(_lines(fields), 4)[1:], [11, 2, 3, 2.5])


def test_glued_fields_fall_back_to_the_fixed_width_reading():
    """Two 20-character fields with no blank between them split as one token."""
    glued = "-1.000000000000E-100"
    assert len(glued) == 20
    fields = [glued, glued, FIELDS[1], FIELDS[3]]
    np.testing.assert_array_equal(
        read_xss(_lines(fields), 4)[1:], [-1e-100, -1e-100, 2.5, 4.0]
    )


def test_a_short_xss_is_an_error():
    with pytest.raises(ValueError, match="NXS"):
        read_xss(_lines(FIELDS), len(VALUES) + 1)


def test_lines_past_nxs1_are_not_read():
    """A second table concatenated in the same file is not part of this XSS."""
    lines = _lines(FIELDS) + ["  92238.80c  236.005800  2.5301E-08   12/12/12\n"]
    np.testing.assert_array_equal(read_xss(lines, len(VALUES))[1:], VALUES)


def _ace_with_esz(n_energy=3):
    """An Ace whose XSS is just an ESZ block: E, total, abs, elastic, heating."""
    esz = np.arange(1.0, 5 * n_energy + 1)
    xss = np.concatenate([[0.0], esz])
    header = Header(
        format_version="legacy",
        zaid=26056,
        extension=".02c",
        atomic_weight_ratio=55.454,
        temperature=2.5301e-08,
        nxs_array=[0, len(esz), 0, n_energy] + [0] * 13,
        jxs_array=[0, 1] + [0] * 31,
    )
    ace = Ace(filename=None, header=header, xss_data=xss)
    ace.esz_block = read_esz_block(ace)
    return ace


def test_a_block_is_a_view_and_writing_it_writes_the_xss():
    ace = _ace_with_esz()
    total = ace.cross_section.reaction[1]._xs_entries
    assert np.shares_memory(total, ace.xss_data)
    assert xss_position(total) == 4  # XSS(JXS(1) + NE)

    total *= 2.0
    np.testing.assert_array_equal(ace.xss_data[4:7], [8.0, 10.0, 12.0])


def test_a_copy_writes_its_own_xss_and_not_the_original():
    ace = _ace_with_esz()
    twin = ace.copy()
    twin_total = twin.cross_section.reaction[1]._xs_entries

    assert np.shares_memory(twin_total, twin.xss_data)
    assert not np.shares_memory(twin.xss_data, ace.xss_data)
    # The ESZ block and the reaction still name one window, as in the original.
    assert np.shares_memory(twin.esz_block.total_xs, twin_total)

    twin_total[:] = -1.0
    np.testing.assert_array_equal(twin.xss_data[4:7], [-1.0, -1.0, -1.0])
    np.testing.assert_array_equal(ace.xss_data[4:7], [4.0, 5.0, 6.0])
