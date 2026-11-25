"""read_covfil reads ERRORR's output by ERRORR's own conventions (NJOY2016 ``errorr.f90``).

The files are two ERRORR outputs of the NJOY2016 test suite (``data/njoy2016``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kika.cov import read_covfil, write_covfil

NJOY = Path(__file__).parent / "data" / "njoy2016"
MF33 = NJOY / "test04_errorr_mf33.gendf"
MF34 = NJOY / "test15_errorr_mf34.gendf"


def test_a_mat_kika_does_not_know_takes_the_isotope_from_the_za_of_the_file():
    # MAT 1395 is U-235 of ENDF/B-V; the HEAD of MF1 gives ZA = 92235.
    cov = read_covfil(MF33)
    assert cov.isotopes == [92235]
    assert set(cov.cross_sections) == {(92235, 18), (92235, 102)}


def test_errorr_mf34_reads_its_pair_and_takes_the_unit_from_irelco(tmp_path):
    cov = read_covfil(MF34)
    assert (cov.num_matrices, cov.is_relative, cov.l_rows, cov.l_cols) == (1, [True], [1], [1])

    absolute = tmp_path / "irelco0.gendf"
    lines = MF34.read_text().split("\n")
    assert lines[19][33:44] == "          1"
    lines[19] = lines[19][:33] + "          0" + lines[19][44:]
    absolute.write_text("\n".join(lines))
    cov0 = read_covfil(absolute)
    assert cov0.is_relative == [False]
    np.testing.assert_array_equal(cov0.matrices[0], cov.matrices[0])


def test_an_mf34_file_kika_writes_still_reads_nmt1_from_the_head(tmp_path):
    cov = read_covfil(MF34)
    path = tmp_path / "rewritten.gendf"
    write_covfil(cov, path)
    again = read_covfil(path)
    assert again.num_matrices == cov.num_matrices
    np.testing.assert_array_equal(again.matrices[0], cov.matrices[0])
