"""Tier-1 golden: the ACE read-write spine.

The ENDF half of this gate lives in ``kika/endf/tests/test_roundtrip_golden.py``.
This is the ACE half, and it splits cleanly in two.

**The numbers survive exactly.** ACE keeps everything in one flat XSS array.
Reading a real 16 MB Fe-56 file and writing it straight back returns all
821 670 entries bitwise identical — not ``allclose``, identical. That is the
gate, and it is what the perturbation pipeline depends on: it edits a handful
of XSS entries and writes the rest back untouched.

**The text survives too.** ACE has no types -- MT numbers, block lengths
and pointers live in the same REAL array as the cross sections -- and only the
spelling tells them apart: NJOY writes them as I20, the rest as E20.11. Until
2026-10-14 kika wrote every word as a real,
and 6 903 of 205 430 lines of this file differed. The reader now keeps the
spelling as one bit per word (``Ace.xss_int_words``) and the writer gives the
same bytes back, header and unpadded last line included.

No fixture is committed: a real ACE file is 16 MB at best, and truncating one
means recomputing the internal pointer blocks, which would make the fixture
kika's arithmetic rather than the library's. These tests run against the shared
tree, under ``tape``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.writers.write_ace import write_ace


def xss_array(ace) -> np.ndarray:
    """The XSS block, index 0 being the FORTRAN 1-based placeholder."""
    return np.asarray(ace.xss_data, dtype=float)


@pytest.fixture(scope="module")
def roundtripped(fe56_ace, tmp_path_factory):
    """Read a real ACE file and write it straight back, unmodified."""
    out = tmp_path_factory.mktemp("ace") / "roundtrip.02c"
    write_ace(read_ace(str(fe56_ace)), str(out), overwrite=True)
    return Path(fe56_ace), out


@pytest.mark.slow
def test_xss_survives_the_roundtrip_bitwise(roundtripped, fe56_ace):
    """Every XSS entry comes back with the same bits. The gate.

    Both sides are parsed from files, which keeps this honest about what the
    writer put on disk rather than about what it left in memory.
    """
    _, out = roundtripped
    before = xss_array(read_ace(str(fe56_ace)))
    after = xss_array(read_ace(str(out)))

    assert after.size == before.size, f"XSS length {before.size} -> {after.size}"
    n_diff = int(np.count_nonzero(before != after))
    assert n_diff == 0, (
        f"{n_diff} of {before.size} XSS entries changed; "
        f"first at {int(np.flatnonzero(before != after)[0])}"
    )


@pytest.mark.slow
def test_roundtrip_preserves_the_line_count(roundtripped):
    """The layout is unchanged even though individual fields are respelled.

    A differing line count would mean entries per line changed, which would
    break MCNP's reader. Only the rendering of each field moves.
    """
    src, out = roundtripped
    assert len(out.read_text().splitlines()) == len(src.read_text().splitlines())


@pytest.mark.slow
def test_ace_roundtrip_is_byte_identical(roundtripped):
    src, out = roundtripped
    assert (
        hashlib.sha256(out.read_bytes()).hexdigest()
        == hashlib.sha256(src.read_bytes()).hexdigest()
    )
