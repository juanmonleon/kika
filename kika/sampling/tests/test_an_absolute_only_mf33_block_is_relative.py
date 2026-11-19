"""An MF33 block stated only in barns² comes out of ``build_mf33_covariance`` relative.

``MF33MT.to_xs_covmat`` returns such a block absolute, and a caller that took
√C as a fraction showed 20 % where the file states 2000 %. The expectation here
is the manual's arithmetic on a hand-computable section, not the function's own.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from kika.endf.classes.mf33.mf33 import MF33MT, NISubSubsectionRecord, Subsection
from kika.sampling.mf33_sampling import build_mf33_covariance

GRID = [1.0e5, 1.0e6, 2.0e7]
F_ABS = [0.04, 0.01]  # b², one variance per interval (LB=0)
SIGMA = 0.01  # b, constant, so its 1/E bin average is exactly SIGMA


def _mf33_file(mt: int = 102):
    record = NISubSubsectionRecord(lb=0, lt=0, e_table_k=GRID, f_table_k=F_ABS + [0.0])
    sub = Subsection(mat1=0, mt1=mt, nc=0, ni=1, ni_records=[record])
    section = MF33MT(number=mt, _za=26056.0, _awr=55.4, _mat=2631, _subsections=[sub])
    return SimpleNamespace(sections={mt: section})


def _flat_sigma(mt: int = 102):
    energies = np.array([GRID[0], GRID[-1]])
    values = np.full(2, SIGMA)
    return {mt: SimpleNamespace(energies=energies, cross_sections=values, values=values)}


def test_the_absolute_block_is_divided_by_the_cross_section():
    cov, grid, present = build_mf33_covariance(_mf33_file(), [102], _flat_sigma())

    assert present == [102]
    assert list(grid) == GRID
    assert all(cov.is_relative)
    relative_sd = np.sqrt(np.diag(np.asarray(cov.matrices[0])))
    np.testing.assert_allclose(relative_sd, np.sqrt(F_ABS) / SIGMA, rtol=1e-12)


def test_without_the_cross_section_the_block_is_dropped_not_misread():
    cov, _grid, _present = build_mf33_covariance(_mf33_file(), [102], {})

    assert cov.matrices == []
