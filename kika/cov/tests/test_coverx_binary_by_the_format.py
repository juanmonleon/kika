"""read_coverx on binary files built record by record from the COVERX format.

The format is appendix A of ORNL/TM-7181 (Drischler, "The COVERX Service Module of
the FORSS System"). The files here are built from it directly, not by kika's writer,
so a reader and a writer that agree on a wrong layout cannot pass together.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from kika.cov import read_coverx
from kika.cov.parse_covmat import InvalidDataFormatError

E = ">"
BOUNDS = [2.0e7, 1.0e6, 1.0e3, 1.0e-5]  # three groups, descending, eV (maxima, then EMIN)
MAT, MT = 26056, 2
XS = [1.0, 2.0, 3.0]
SD = [0.1, 0.2, 0.3]
MATRIX = np.array([[0.04, 0.01, 0.00],
                   [0.01, 0.09, 0.02],
                   [0.00, 0.02, 0.16]])


def _record(payload: bytes) -> bytes:
    return struct.pack(E + "i", len(payload)) + payload + struct.pack(E + "i", len(payload))


def _coverx(path, *, ntype: int, matrix: np.ndarray, lgrp=(3,)):
    n = len(BOUNDS) - 1
    out = [
        _record(b"COVERXtest  " + struct.pack(E + "i", 1)),
        _record(struct.pack(E + "7i", n, n, 0, ntype, 1, 1, 1)),
        _record(b"descr "),
        _record(struct.pack(E + f"{n + 1}f", *BOUNDS)),
        _record(struct.pack(E + "3i", MAT, MT, 2)),
        _record(struct.pack(E + f"{2 * n}f", *XS, *SD)),
        _record(struct.pack(E + "5i", MAT, MT, MAT, MT, len(lgrp))),
    ]
    band = []
    for j in range(n):
        band += [n, j + 1]  # full rows: band from column 1, diagonal at position j+1
    out.append(_record(struct.pack(E + f"{2 * n + len(lgrp)}i", *band, *lgrp)))
    row = 0
    for rows in lgrp:
        vals = matrix[row:row + rows].ravel()
        out.append(_record(struct.pack(E + f"{vals.size}f", *vals)))
        row += rows
    path.write_bytes(b"".join(out))
    return path


def _read(path):
    return read_coverx(path, ascending=False)


def test_a_relative_covariance_is_flagged_relative(tmp_path):
    cov = _read(_coverx(tmp_path / "rel.cx", ntype=2, matrix=MATRIX))
    assert cov.is_relative == [True]
    np.testing.assert_allclose(cov.matrices[0], MATRIX, rtol=1e-6)


def test_an_absolute_covariance_is_flagged_absolute(tmp_path):
    assert _read(_coverx(tmp_path / "abs.cx", ntype=1, matrix=MATRIX)).is_relative == [False]


def test_a_correlation_is_turned_into_a_covariance_with_the_standard_deviations(tmp_path):
    corr = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, -0.2], [0.0, -0.2, 1.0]])
    cov = _read(_coverx(tmp_path / "corr.cx", ntype=3, matrix=corr))
    np.testing.assert_allclose(cov.matrices[0], corr * np.outer(SD, SD), rtol=1e-6)
    assert cov.is_relative == [False]


def test_a_matrix_split_in_blocks_of_groups_comes_back_whole(tmp_path):
    cov = _read(_coverx(tmp_path / "split.cx", ntype=2, matrix=MATRIX, lgrp=(2, 1)))
    np.testing.assert_allclose(cov.matrices[0], MATRIX, rtol=1e-6)


def test_blocks_that_do_not_add_up_to_the_groups_are_refused(tmp_path):
    path = _coverx(tmp_path / "bad.cx", ntype=2, matrix=MATRIX, lgrp=(2, 2))
    with pytest.raises(InvalidDataFormatError, match="blocks hold 4 groups"):
        _read(path)
