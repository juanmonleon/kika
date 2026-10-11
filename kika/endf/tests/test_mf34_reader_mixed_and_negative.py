"""``MF34MT.to_ang_covmat`` on an LB=0 component and a negative variance.

It used to add LB=0 records (absolute) to the relative ones as they stood and
then label the sum absolute, and to clip negative variances to zero. Found
while documenting the MF34 reader (11-oct-2026).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from kika.endf.classes.mf33 import NISubSubsectionRecord
from kika.endf.classes.mf34 import MF34MT, Subsection, SubSubsection, SubSubsectionRecord

GRID = [1.0e5, 1.0e6, 1.0e7]


def _lb(lb, values, cls=SubSubsectionRecord):
    return cls(lb=lb, np=len(GRID), ne=len(GRID), nt=2 * len(GRID),
               e_table_k=list(GRID), f_table_k=list(values) + [0.0])


def _lb5(matrix):
    m = np.asarray(matrix, dtype=float)
    tri = list(m[np.triu_indices(m.shape[0])])
    return SubSubsectionRecord(lb=5, ls=1, ne=len(GRID), nt=len(GRID) + len(tri),
                               energies=list(GRID), matrix=tri)


def _section(records):
    sec = MF34MT(number=2, _za=26056.0, _awr=55.45, _ltt=1, _nmt1=1, _mat=2631)
    sec.add_subsection(Subsection(mt1=2, nl=1, nl1=1, mat1=0, sub_subsections=[
        SubSubsection(l=1, l1=1, lct=1, ni=len(records), records=records)]))
    return sec


class _MF4Section:
    def legendre_cell_averages(self, edges, max_order):
        return {max_order: np.array([0.2, 0.5])}


MF4 = SimpleNamespace(mt={2: _MF4Section()})


def test_an_absolute_component_is_divided_by_a_l_before_the_sum():
    sec = _section([_lb(1, [0.01, 0.04]), _lb(0, [0.0004, 0.0025])])
    cov = sec.to_ang_covmat(mf4_data=MF4)
    assert cov.is_relative[0]
    # 0.0004 / 0.2^2 = 0.01 and 0.0025 / 0.5^2 = 0.01, added to the relative 0.01, 0.04.
    assert np.allclose(np.diag(cov.matrices[0]), [0.02, 0.05])


def test_without_mf4_a_mixed_block_is_refused():
    sec = _section([_lb(1, [0.01, 0.04]), _lb(0, [0.0004, 0.0025])])
    with pytest.raises(ValueError, match="mixes absolute"):
        sec.to_ang_covmat()


def test_a_wholly_absolute_block_stays_absolute():
    cov = _section([_lb(0, [0.0004, 0.0025])]).to_ang_covmat()
    assert not cov.is_relative[0]
    assert np.allclose(np.diag(cov.matrices[0]), [0.0004, 0.0025])


def test_a_negative_variance_is_kept_as_stated():
    cov = _section([_lb5([[-0.01, 0.0], [0.0, 0.04]])]).to_ang_covmat()
    assert np.allclose(np.diag(cov.matrices[0]), [-0.01, 0.04])
