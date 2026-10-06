"""Layer 1 for MF35 and MF40, one planted fault per test.

As in ``test_covariance_checks.py``: synthetic sections built from the parser's
own classes, nothing wrong but the fault. MF32 is pinned on real cuts in
``test_covariance_checks_on_real_cuts.py``: its records are packed text, and a
synthetic one would test the builder more than the check.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf import check_covariances
from kika.endf.classes.endf import ENDF
from kika.endf.classes.mf import MF
from kika.endf.classes.mf33 import NISubSubsectionRecord, Subsection
from kika.endf.classes.mf35 import MF35MT, MF35SubSection
from kika.endf.classes.mf40 import MF40MT, MF40State
from kika.endf.parsers.parse_mf40 import parse_mf40_mt

GRID = [1.0e5, 1.0e6, 2.0e6, 5.0e6, 2.0e7]


def _tape(mf_number, *sections):
    endf = ENDF()
    mf = MF(number=mf_number)
    for sec in sections:
        mf.add_section(sec)
    endf.add_file(mf)
    return endf


def _faults(report):
    return sorted((f.level, f.check) for f in report.at_least("warn"))


# ---------------------------------------------------------------------------
# MF35
# ---------------------------------------------------------------------------


def _normalised(sigma=0.01, p=(0.1, 0.4, 0.3, 0.2), seed=0):
    """A covariance of group probabilities: C = A S A^T with A = I - p 1^T, so C.1 = 0."""
    n = len(p)
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n, n))
    s = sigma ** 2 * (b @ b.T / n + np.eye(n))
    a = np.eye(n) - np.outer(p, np.ones(n))
    return a @ s @ a.T


def _band(matrix, e1, e2, grid=GRID):
    m = np.asarray(matrix, dtype=float)
    tri = list(m[np.triu_indices(m.shape[0])])
    ne = len(grid)
    return MF35SubSection(e1=e1, e2=e2, ls=1, lb=7, nt=ne + len(tri), ne=ne,
                          boundaries=list(grid), upper_triangle=tri)


def _mf35(*bands, nk=None):
    sec = MF35MT(number=18, _za=92235.0, _awr=233.0, _nk=len(bands) if nk is None else nk,
                 _mat=9228)
    sec.subsections.extend(bands)
    return sec


def test_a_normalised_spectrum_covariance_passes():
    sec = _mf35(_band(_normalised(), 1e-5, 1e6), _band(_normalised(seed=1), 1e6, 2e7))
    report = check_covariances(_tape(35, sec))
    assert _faults(report) == []
    # MF5 was not read: the probability bound ran without central values, and says so.
    assert [f.check for f in report.notes] == ["central_values_unavailable"]


def test_rows_that_do_not_sum_to_zero_break_the_sum_rule():
    report = check_covariances(_tape(35, _mf35(_band(np.diag([1e-4, 4e-4, 1e-4, 2e-4]), 0, 2e7))))
    assert _faults(report) == [("defect", "sum_rule_violated")]
    found = report.by_check("sum_rule_violated")[0]
    assert found.location.band == 0 and found.evidence["row_sum_residual"] == pytest.approx(1.0)


def test_a_sigma_no_probability_can_have_is_a_defect():
    # sigma = 0.6 on a quantity confined to [0, 1]: impossible whatever MF5 says.
    c = _normalised()
    c[1, :] = c[:, 1] = 0.0
    c[1, 1] = 0.36
    report = check_covariances(_tape(35, _mf35(_band(c, 0, 2e7))))
    assert ("defect", "variance_exceeds_physical_bound") in _faults(report)
    bound = report.by_check("variance_exceeds_physical_bound")[0].evidence
    assert bound["worst_sigma"] == pytest.approx(0.6) and bound["worst_bound"] == 0.5


def test_bands_must_be_non_empty_contiguous_and_counted():
    c = _normalised()
    report = check_covariances(_tape(35, _mf35(
        _band(c, 1e-5, 1e-5),          # empty
        _band(c, 1e-5, 1e6),
        _band(c, 2e6, 5e6),            # gap 1e6-2e6
        _band(c, 4e6, 2e7),            # overlaps the previous band
        nk=5)))
    assert _faults(report) == [("defect", "band_overlap"), ("defect", "count_mismatch"),
                               ("warn", "band_empty"), ("warn", "band_gap")]
    assert report.by_check("band_gap")[0].location.band == 2


def _indefinite(depth, seed=3):
    """A normalised band with one negative eigenvalue of -depth x lambda_max."""
    c = _normalised(seed=seed)
    w, v = np.linalg.eigh(c)
    w[1] = -depth * w[-1]             # w[0] is the sum-rule null direction
    return (v * w) @ v.T


def test_a_light_psd_warning_is_kept_only_if_clipping_moves_sigma():
    # The same ratio class (1e-6..1e-3) on both; only the consequence differs.
    shallow = check_covariances(_tape(35, _mf35(_band(_indefinite(1e-5), 0, 2e7))))
    # lambda_min = -2e-4 of lambda_max, all of it in two groups with sigma 0.02:
    # clipping adds 1e-4 to a variance of 4e-4, a quarter.
    small = np.array([[4e-4, 6e-4], [6e-4, 4e-4]])
    deep_matrix = np.zeros((4, 4))
    deep_matrix[:2, :2] = np.eye(2)
    deep_matrix[2:, 2:] = small
    deep = check_covariances(_tape(35, _mf35(_band(deep_matrix, 0, 2e7))))
    f_shallow = shallow.by_check("not_positive_semidefinite")[0]
    f_deep = deep.by_check("not_positive_semidefinite")[0]
    assert 1e-6 < f_shallow.evidence["ratio"] < 1e-3 and 1e-6 < f_deep.evidence["ratio"] < 1e-3
    assert f_shallow.level == "note" and f_shallow.evidence["sigma_change_if_clipped"] < 0.02
    assert f_deep.level == "warn" and f_deep.evidence["sigma_change_if_clipped"] >= 0.02


def test_mf35_without_its_mf5_section_is_a_covariance_of_nothing():
    endf = _tape(35, _mf35(_band(_normalised(), 0, 2e7)))
    endf.add_file(MF(number=5))
    report = check_covariances(endf)
    assert _faults(report) == [("defect", "missing_distribution")]


# ---------------------------------------------------------------------------
# MF40
# ---------------------------------------------------------------------------

G40 = [1.0e6, 5.0e6, 1.0e7, 2.0e7]


def _lb5(matrix, ls=1, grid=G40):
    m = np.asarray(matrix, dtype=float)
    values = list(m[np.triu_indices(m.shape[0])]) if ls == 1 else list(m.ravel())
    return NISubSubsectionRecord(lb=5, ls=ls, ne=len(grid), nt=len(grid) + len(values),
                                 energies=list(grid), matrix=values)


def _sub(mt1, lfs1, *records):
    return Subsection(xmf1=10.0, xlfs1=float(lfs1), mat1=0, mt1=mt1, nc=0, ni=len(records),
                      ni_records=list(records))


def _mf40(mt, *states, ns=None):
    sec = MF40MT(number=mt, _za=13027.0, _awr=26.75, _lis=0,
                 _ns=len(states) if ns is None else ns, _mat=1325)
    for lfs, subs in states:
        sec.states.append(MF40State(qm=-3e6, qi=-3e6, izap=13026, lfs=lfs, nl=len(subs),
                                    subsections=list(subs)))
    return sec


SELF = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.01]])


def test_a_clean_mf40_says_only_that_mf10_is_not_read():
    report = check_covariances(_tape(40, _mf40(16, (0, [_sub(16, 0, _lb5(SELF))]),
                                               (1, [_sub(16, 1, _lb5(SELF))]))))
    assert _faults(report) == []
    assert [f.check for f in report] == ["central_values_unavailable"]


def test_mf40_blocks_are_named_by_mt_and_final_state():
    cross = np.full((3, 3), 0.5)                 # |rho| = 0.5 / sqrt(0.04 * 0.01) = 25
    report = check_covariances(_tape(40, _mf40(
        16,
        (0, [_sub(16, 0, _lb5(SELF)), _sub(16, 1, _lb5(cross, ls=0))]),
        (1, [_sub(16, 1, _lb5(SELF))]),
        (2, [_sub(16, 0, _lb5(cross, ls=0))]),          # no self block
        (3, [_sub(16, 3, _lb5(SELF)), _sub(17, 0, _lb5(cross, ls=0))]),  # MT17 absent
        ns=5)))
    assert _faults(report) == [("defect", "correlation_out_of_bounds"),
                               ("defect", "count_mismatch"),
                               ("defect", "missing_partner"),
                               ("defect", "missing_self_block")]
    rho = report.by_check("correlation_out_of_bounds")[0]
    assert (rho.location.mt, rho.location.lfs, rho.location.mt1, rho.location.lfs1) == (16, 0, 16, 1)
    assert str(rho.location).startswith("MF40 MT16/LFS0xMT16/LFS1")
    assert report.by_check("missing_self_block")[0].location.lfs == 2
    assert report.by_check("missing_partner")[0].location.mt1 == 17


def test_an_ls1_triangle_between_two_final_states_is_flagged_where_it_is():
    report = check_covariances(_tape(40, _mf40(
        16, (0, [_sub(16, 0, _lb5(SELF)), _sub(16, 1, _lb5(np.full((3, 3), 0.001)))]),
        (1, [_sub(16, 1, _lb5(SELF))]))))
    ls1 = report.by_check("ls1_in_cross_block")
    assert len(ls1) == 1 and (ls1[0].location.lfs, ls1[0].location.lfs1, ls1[0].location.ni) == (0, 1, 0)


def test_mf40_writes_back_what_it_reads():
    sec = _mf40(16, (0, [_sub(16, 0, _lb5(SELF))]), (1, [_sub(16, 1, _lb5(SELF))]))
    text = str(sec)
    again = parse_mf40_mt(text.split("\n")[:-1], 16)
    assert str(again) == text
    assert [s.lfs for s in again.states] == [0, 1]
    assert again.states[1].subsections[0].ni_records[0].matrix == pytest.approx(
        list(SELF[np.triu_indices(3)]))
