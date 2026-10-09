"""``write_ace`` must not rewrite the object it was handed.

It used to. When the XSS was a list of ``XssEntry`` objects, a guard that read
element 0 (the bare ``0`` placeholder) re-wrapped every entry into
``XssEntry(index=i, value=XssEntry(...))`` — on the caller's object, not a
copy. The XSS is now one float64 array and the writer only reads it, but the
contract the fix established still holds and is pinned here: the caller's
array comes back as the same object with the same values, writing is
idempotent, and a plain list of numbers is still accepted.

These tests are deliberately free of any ACE tape, so they run in CI where the
rest of ``kika/ace`` cannot.
"""
from __future__ import annotations

import numpy as np

from kika.ace.classes.ace import Ace
from kika.ace.classes.header import Header
from kika.ace.writers.write_ace import write_ace

VALUES = [11.0, 2.5, 3.0, 4.25, 5.0, 6.5, 7.0]


def _tiny_ace() -> Ace:
    """An Ace shaped like a parsed one: 0.0 placeholder, then the XSS."""
    xss = np.array([0.0] + VALUES)
    header = Header(
        format_version="legacy",
        zaid=26056,
        extension=".02c",
        atomic_weight_ratio=55.454,
        temperature=2.5301e-08,
        date="01/01/26",
        comment="synthetic fixture",
        matid=2631,
        izaw_array=[(0, 0.0)] * 16,
        nxs_array=[len(xss) - 1] + [0] * 15,
        jxs_array=[1] + [0] * 31,
    )
    return Ace(filename=None, header=header, xss_data=xss)


def test_write_ace_leaves_its_input_alone(tmp_path):
    ace = _tiny_ace()
    xss = ace.xss_data
    before = xss.copy()

    write_ace(ace, str(tmp_path / "out.02c"), overwrite=True)

    assert ace.xss_data is xss, "write_ace replaced the caller's XSS array"
    assert ace.xss_data.dtype == np.float64
    np.testing.assert_array_equal(ace.xss_data, before)


def test_writing_twice_gives_the_same_file(tmp_path):
    """A mutating writer is not idempotent; this is what that looked like."""
    ace = _tiny_ace()
    first = tmp_path / "first.02c"
    second = tmp_path / "second.02c"

    write_ace(ace, str(first), overwrite=True)
    write_ace(ace, str(second), overwrite=True)

    assert first.read_text() == second.read_text()


def test_raw_float_input_is_still_accepted(tmp_path):
    """A caller may pass a plain list of numbers."""
    ace = _tiny_ace()
    ace.xss_data = [0.0, 1.0, 2.0, 3.0]

    write_ace(ace, str(tmp_path / "out.02c"), overwrite=True)

    assert ace.xss_data == [0.0, 1.0, 2.0, 3.0], "raw input was rewritten in place"


def test_values_reach_the_file(tmp_path):
    """Guard against the payload being dropped.

    The XSS block is the tail of the file, after the header's IZAW/NXS/JXS
    arrays — which are numeric too, hence the tail slice rather than a scan.
    Index 0 is the placeholder and is not written.
    """
    ace = _tiny_ace()
    out = tmp_path / "out.02c"
    write_ace(ace, str(out), overwrite=True)

    numeric = []
    for line in reversed(out.read_text().splitlines()):
        try:
            row = [float(tok) for tok in line.split()]
        except ValueError:
            break
        numeric = row + numeric

    np.testing.assert_allclose(numeric[-len(VALUES):], VALUES)
