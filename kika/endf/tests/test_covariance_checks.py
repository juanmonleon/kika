"""Layer 1 of the covariance checks: ``kika.endf.check_covariances``.

Each check gets a synthetic section built from the parser's own record classes,
with the fault planted and nothing else wrong, so a test pins one rule. The
real-tape cases (O-16 of ENDF/B-VIII.1, B-10/B-11 of JENDL-5, the MF34 defect A
of JEFF-4.0) are the validation of phase C7, against the census in
kika-workspace ``docs/library/cov_checks_roadmap.md``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kika.endf import check_covariances
from kika.endf.checks import (DEFECT, NOTE, WARN, CovarianceCheckReport,
                              CovarianceFinding, CovarianceLocation)
from kika.endf.classes.endf import ENDF
from kika.endf.classes.mf import MF
from kika.endf.classes.mf33 import MF33MT, NISubSubsectionRecord, Subsection
from kika.endf.classes.mf34 import MF34MT
from kika.endf.classes.mf34 import Subsection as Subsection34
from kika.endf.classes.mf34 import SubSubsection, SubSubsectionRecord

GRID = [1.0, 10.0, 100.0, 1000.0]


def _triangle(matrix):
    m = np.asarray(matrix, dtype=float)
    return list(m[np.triu_indices(m.shape[0])])


def _lb5(matrix, grid=GRID, ls=1, cls=NISubSubsectionRecord):
    """An LB=5 record holding ``matrix`` (upper triangle if LS=1)."""
    m = np.asarray(matrix, dtype=float)
    values = _triangle(m) if ls == 1 else list(m.ravel())
    return cls(lb=5, ls=ls, ne=len(grid), nt=len(grid) + len(values),
               energies=list(grid), matrix=values)


def _lb1(variances, grid=GRID, lb=1):
    return NISubSubsectionRecord(lb=lb, np=len(grid), ne=len(grid), nt=2 * len(grid),
                                 e_table_k=list(grid), f_table_k=list(variances) + [0.0])


def _section(blocks, mt=1, nl=None):
    """An MF33 section; ``blocks`` maps MT1 -> list of NI records."""
    sec = MF33MT(number=mt, _za=26056.0, _awr=55.45, _mat=2631,
                 _nl=len(blocks) if nl is None else nl)
    for mt1, records in blocks.items():
        sec.add_subsection(Subsection(mt1=mt1, mat1=0, nc=0, ni=len(records),
                                      ni_records=list(records)))
    return sec


def _tape(*sections, mf_number=33, files=()):
    endf = ENDF()
    mf = MF(number=mf_number)
    for sec in sections:
        mf.add_section(sec)
    endf.add_file(mf)
    for extra in files:
        endf.add_file(extra)
    return endf


def _only(report, check):
    found = report.by_check(check)
    assert len(found) == 1, [str(f) for f in report]
    return found[0]


def _spectrum(eigenvalues, seed=0):
    """A symmetric matrix with exactly these eigenvalues."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(len(eigenvalues), len(eigenvalues))))
    return q @ np.diag(eigenvalues) @ q.T


def _grid(n):
    return list(np.geomspace(1.0, 1e6, n + 1))


# --------------------------------------------------------------------------
# A clean section, and the report
# --------------------------------------------------------------------------

CLEAN = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.01]]


def test_a_clean_block_has_nothing_above_a_note():
    report = check_covariances(_tape(_section({1: [_lb5(CLEAN)]})))
    assert report.at_least(WARN) == ()
    # No MF2/MF3 was read, so the large-variance check says it could not run.
    assert [f.check for f in report] == ["central_values_unavailable"]


def test_the_report_tabulates_and_prints_without_unicode():
    loc = CovarianceLocation(mat=1, mf=33, mt=2, mat1=0, mt1=102, ni=0, lb=5, ls=1)
    report = CovarianceCheckReport((
        CovarianceFinding("ls1_in_cross_block", DEFECT, "x", loc),
        CovarianceFinding("inert_rows", NOTE, "y", loc),
    ), source="tape", mat=1)
    assert str(loc) == "MF33 MT2xMT102 NI[0] LB=5 LS=1"
    text = str(report)
    text.encode("cp1252")  # a Windows console must be able to print it
    assert "1 defect" in text and "1 note" in text
    df = report.to_dataframe()
    assert list(df["check"]) == ["ls1_in_cross_block", "inert_rows"]
    assert df.loc[0, "mt1"] == 102
    assert report.counts() == {("defect", "ls1_in_cross_block"): 1, ("note", "inert_rows"): 1}
    with pytest.raises(ValueError):
        CovarianceFinding("x", "fatal", "y")


# --------------------------------------------------------------------------
# C2 -- structure
# --------------------------------------------------------------------------

def test_a_section_the_parser_dropped_is_a_parse_error():
    endf = _tape(_section({1: [_lb5(CLEAN)]}))
    endf.files[33].parse_errors[102] = "ValueError: LB=7 unsupported"
    f = _only(check_covariances(endf), "parse_error")
    assert f.level == DEFECT and f.location.mt == 102 and "LB=7" in f.summary


def test_counts_that_do_not_match_what_was_read():
    sec = _section({1: [_lb5(CLEAN)]}, nl=2)
    sec.subsections[0].ni = 3
    found = check_covariances(_tape(sec)).by_check("count_mismatch")
    assert {f.evidence["field"] for f in found} == {"NL", "NI"}


def test_nt_that_does_not_match_ne_and_ls():
    rec = _lb5(CLEAN)
    rec.nt += 1
    f = _only(check_covariances(_tape(_section({1: [rec]}))), "nt_mismatch")
    assert f.evidence == {"nt": rec.nt, "expected": rec.nt - 1}
    assert f.location.ni == 0 and f.location.lb == 5 and f.location.ls == 1


def test_a_decreasing_grid_is_a_defect_and_a_repeated_point_a_warning():
    down = _lb1([0.01, 0.01, 0.01], grid=[1.0, 100.0, 10.0, 1000.0])
    f = _only(check_covariances(_tape(_section({1: [down]}))), "grid_not_increasing")
    assert f.level == DEFECT and f.evidence["index"] == 2
    flat = _lb1([0.01, 0.01, 0.01], grid=[1.0, 10.0, 10.0, 1000.0])
    f = _only(check_covariances(_tape(_section({1: [flat]}))), "grid_repeated_point")
    assert f.level == WARN and f.evidence["energies"] == [10.0]


def test_ls1_in_a_cross_block_is_defect_a_and_the_mirror_is_blamed():
    # The stored triangle's only off-diagonal is rho = 0.009/sqrt(0.01*0.01) = 0.9;
    # mirrored into (1, 0) it meets the two small variances: 0.009/1e-4 = 90.
    grid = [1.0, 10.0, 100.0]
    row = _section({1: [_lb5([[0.01, 0.0], [0.0, 1e-4]], grid)],
                    2: [_lb5([[5e-4, 0.009], [0.009, 5e-4]], grid)]}, mt=1)
    col = _section({2: [_lb5([[1e-4, 0.0], [0.0, 0.01]], grid)]}, mt=2)
    report = check_covariances(_tape(row, col))
    a = _only(report, "ls1_in_cross_block")
    assert a.location.mt == 1 and a.location.mt1 == 2 and a.location.ni == 0
    assert a.evidence["stored_triangle_max_abs_rho"] == pytest.approx(0.9)
    rho = _only(report, "correlation_out_of_bounds")
    assert rho.level == DEFECT and rho.evidence["max_abs_rho"] == pytest.approx(90.0)
    assert "creates" in rho.summary


def test_an_unknown_lty_and_an_l_outside_nl():
    from kika.endf.classes.mf33 import NCSubSubsection

    sec = _section({1: [_lb5(CLEAN)]})
    sec.subsections[0].nc = 1
    sec.subsections[0].nc_records.append(NCSubSubsection(lty=7, e1=1.0, e2=10.0))
    assert _only(check_covariances(_tape(sec)), "lty_invalid").location.nc == 0

    sec34 = MF34MT(number=2, _za=26056.0, _awr=55.45, _ltt=1, _nmt1=1, _mat=2631)
    rec = _lb5(CLEAN, cls=SubSubsectionRecord)
    sub = Subsection34(mt1=2, nl=1, nl1=1, mat1=0,
                       sub_subsections=[SubSubsection(l=2, l1=2, lct=1, ni=1, records=[rec])])
    sec34.add_subsection(sub)
    f = _only(check_covariances(_tape(sec34, mf_number=34)), "l_out_of_range")
    assert (f.location.l, f.location.l1) == (2, 2)


# --------------------------------------------------------------------------
# C3 -- values
# --------------------------------------------------------------------------

def test_rho_above_one_in_a_self_block_and_rounding_below_the_line():
    bad = [[0.01, 0.02, 0.0], [0.02, 0.01, 0.0], [0.0, 0.0, 0.01]]  # rho = 2
    f = _only(check_covariances(_tape(_section({1: [_lb5(bad)]}))), "correlation_out_of_bounds")
    assert f.level == DEFECT and f.evidence["max_abs_rho"] == pytest.approx(2.0)
    edge = [[0.01, 0.01 * (1 + 1e-5), 0.0], [0.01 * (1 + 1e-5), 0.01, 0.0], [0.0, 0.0, 0.01]]
    report = check_covariances(_tape(_section({1: [_lb5(edge)]})))
    assert _only(report, "correlation_out_of_bounds").level == NOTE


def test_negative_variance_inert_rows_and_covariance_without_variance():
    m = [[-0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.04]]
    report = check_covariances(_tape(_section({1: [_lb5(m)]})))
    assert _only(report, "negative_variance").evidence["n"] == 1
    assert _only(report, "inert_rows").evidence["n"] == 1
    lonely = [[0.0, 0.001, 0.0], [0.001, 0.04, 0.0], [0.0, 0.0, 0.04]]
    report = check_covariances(_tape(_section({1: [_lb5(lonely)]})))
    assert _only(report, "covariance_without_variance").level == DEFECT


def test_large_variance_is_judged_in_absolute_against_sigma():
    # sigma_rel = 2 on a flat 1 b cross section: sigma_abs 2 b > the 1 b scale.
    big = np.diag([4.0, 0.01, 0.01])
    xs = {1: SimpleNamespace(energies=np.array([0.1, 1e4]), values=np.array([1.0, 1.0]))}
    f = _only(check_covariances(_tape(_section({1: [_lb5(big)]})), xs_sections=xs),
              "large_variance")
    assert f.level == WARN and f.evidence["n_above_scale"] == 1
    assert f.evidence["worst_sigma_abs"] == pytest.approx(2.0)
    # The same 200 % where the cross section is ~0 (a threshold) is legitimate.
    ramp = {1: SimpleNamespace(energies=np.array([1.0, 10.0, 1e4]),
                               values=np.array([0.0, 0.0, 1.0]))}
    report = check_covariances(_tape(_section({1: [_lb5(big)]})), xs_sections=ramp)
    assert report.by_check("large_variance") == ()


def test_a_mixed_block_without_sigma_is_checked_on_its_relative_part():
    recs = [_lb5(CLEAN), _lb1([1e-3, 1e-3, 1e-3], lb=8)]
    f = _only(check_covariances(_tape(_section({1: recs}))), "mixed_needs_cross_sections")
    assert f.level == NOTE and f.evidence["lb"] == [5, 8]
    xs = {1: SimpleNamespace(energies=np.array([0.1, 1e4]), values=np.array([1.0, 1.0]))}
    report = check_covariances(_tape(_section({1: recs})), xs_sections=xs)
    assert report.by_check("mixed_needs_cross_sections") == ()


def test_mf34_sigma_wider_than_the_physical_range_of_a_l():
    sec34 = MF34MT(number=2, _za=26056.0, _awr=55.45, _ltt=1, _nmt1=1, _mat=2631)
    rec = _lb5(np.diag([400.0, 0.01, 0.01]), cls=SubSubsectionRecord)  # sigma_rel 20
    sec34.add_subsection(Subsection34(
        mt1=2, nl=1, nl1=1, mat1=0,
        sub_subsections=[SubSubsection(l=1, l1=1, lct=1, ni=1, records=[rec])]))
    mf4 = MF(number=4)
    mf4.add_section(SimpleNamespace(
        number=2,
        extract_legendre_coefficients=lambda e, max_legendre_order, out_of_range:
            {1: np.full(np.size(e), 0.1)}))
    f = _only(check_covariances(_tape(sec34, mf_number=34, files=[mf4])), "large_variance")
    assert f.evidence["worst_sigma_abs"] == pytest.approx(2.0)  # 20 x |a_1| = 0.1


# --------------------------------------------------------------------------
# C4 -- positive semi-definiteness, in three levels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ratio, level", [(1e-4, WARN), (1e-2, DEFECT)])
def test_psd_levels(ratio, level):
    n = 6
    m = _spectrum([1.0, 0.8, 0.5, 0.3, 0.2, -ratio]) * 0.01
    f = _only(check_covariances(_tape(_section({1: [_lb5(m, _grid(n))]}))),
              "not_positive_semidefinite")
    assert f.level == level and f.evidence["ratio"] == pytest.approx(ratio, rel=1e-6)


def test_an_eigenvalue_inside_endf_rounding_is_a_note():
    m = _spectrum([1.0, 0.8, 0.5, -1e-7]) * 0.01
    f = _only(check_covariances(_tape(_section({1: [_lb5(m, _grid(4))]}))),
              "not_positive_semidefinite")
    assert f.level == NOTE and "rounding" in f.summary


def test_correlations_rounded_to_a_lattice_explain_a_small_negative_eigenvalue():
    # JENDL-5 writes sigma_i sigma_j rho_ij with rho rounded to 0.001: a rank-3
    # correlation rounded that way has small negative eigenvalues of its own.
    rng = np.random.default_rng(3)
    n = 20
    a = rng.normal(size=(n, 3))
    r = a @ a.T
    r /= np.sqrt(np.outer(np.diag(r), np.diag(r)))
    r = np.round(r, 3)
    np.fill_diagonal(r, 1.0)
    sigma = np.linspace(0.02, 0.08, n)
    m = r * np.outer(sigma, sigma)
    lam = np.linalg.eigvalsh(m)
    assert PSD_WARN_TIER(lam), "the fixture must land in the warning tier"
    report = check_covariances(_tape(_section({1: [_lb5(m, _grid(n))]})))
    f = _only(report, "not_positive_semidefinite")
    assert f.level == NOTE and f.evidence["rho_quantum"] == [0.001]
    # Summed with a diagonal LB=1 too small to lift it, still explained.
    recs = [_lb5(m, _grid(n)), _lb1([1e-9] * n, grid=_grid(n))]
    f = _only(check_covariances(_tape(_section({1: recs}))), "not_positive_semidefinite")
    assert f.level == NOTE


def PSD_WARN_TIER(lam):
    return lam[0] < 0 and 1e-6 <= -lam[0] / lam[-1] <= 1e-3


def test_a_defect_names_the_record_that_is_already_indefinite_alone():
    n = 4
    bad = _spectrum([1.0, 0.5, 0.2, -0.05]) * 0.01
    recs = [_lb1([1e-4] * n, grid=_grid(n)), _lb5(bad, _grid(n))]
    f = _only(check_covariances(_tape(_section({1: recs}))), "not_positive_semidefinite")
    assert f.level == DEFECT
    assert [a["ni"] for a in f.evidence["records_indefinite_alone"]] == [1]
    assert "NI[1]" in f.summary


def test_mf31_is_judged_against_nubar_from_mf1():
    big = np.diag([4.0, 0.01, 0.01])  # sigma_rel 2 on nu-bar 2.4
    mf1 = MF(number=1)
    mf1.add_section(SimpleNamespace(
        number=452, get_nubar=lambda e, out_of_range: np.full(np.size(e), 2.4)))
    endf = _tape(_section({452: [_lb5(big)]}, mt=452), mf_number=31, files=[mf1])
    f = _only(check_covariances(endf), "large_variance")
    assert f.location.mf == 31 and f.evidence["worst_sigma_abs"] == pytest.approx(4.8)
    assert "nu-bar" in f.evidence["scale_name"]


def test_a_summation_mt_missing_from_mf3_is_summed_from_its_partials():
    # MF33 MT3 with no MF3 MT3: its sigma is the sum of the nonelastic partials
    # the tape gives (writers.redundant rules), here MT4 + MT102.
    mf3 = MF(number=3)
    for mt, value in ((4, 1.5), (102, 0.5)):
        mf3.add_section(SimpleNamespace(number=mt, energies=np.array([0.1, 1e4]),
                                        cross_sections=np.array([value, value])))
    mf2 = MF(number=2)
    mf2.add_section(SimpleNamespace(number=151, isotopes=[]))
    big = np.diag([4.0, 0.01, 0.01])
    endf = _tape(_section({3: [_lb5(big)]}, mt=3), files=[mf2, mf3])
    report = check_covariances(endf)
    assert report.by_check("central_values_unavailable") == ()
    f = _only(report, "large_variance")
    assert f.evidence["scale"] == pytest.approx(2.0)


def test_an_all_zero_ls1_triangle_in_a_cross_block_is_only_a_note():
    # ENDF/B-VIII.1 La-139 MF34 MT2 (1, 2): mirroring zeros changes nothing.
    grid = [1.0, 10.0, 100.0]
    row = _section({1: [_lb5([[0.01, 0.0], [0.0, 0.01]], grid)],
                    2: [_lb5([[0.0, 0.0], [0.0, 0.0]], grid)]}, mt=1)
    col = _section({2: [_lb5([[0.01, 0.0], [0.0, 0.01]], grid)]}, mt=2)
    f = _only(check_covariances(_tape(row, col)), "ls1_in_cross_block")
    assert f.level == NOTE and "all zeros" in f.summary


# --------------------------------------------------------------------------
# Real tapes: what the census measured
# --------------------------------------------------------------------------

def test_o16_b81_mt2_is_indefinite_already_in_its_lb5(o16_b81_tape):
    from kika.endf import read_endf

    report = check_covariances(read_endf(str(o16_b81_tape), mf_numbers=[1, 2, 3, 4, 33, 34]))
    psd = [f for f in report.by_check("not_positive_semidefinite")
           if f.location.mf == 33 and f.location.mt == 2]
    assert len(psd) == 1 and psd[0].level == DEFECT
    assert psd[0].evidence["ratio"] == pytest.approx(1.0, abs=1e-3)
    assert [a["ni"] for a in psd[0].evidence["records_indefinite_alone"]] == [0]
    rho = [f for f in report.by_check("correlation_out_of_bounds")
           if f.location.mf == 33 and f.location.mt == 2]
    assert rho and rho[0].level == DEFECT


def test_jendl5_fe56_has_no_defect_and_its_negative_eigenvalues_are_rounding(fe56_jendl_tape):
    # JENDL-5 writes sigma_i sigma_j rho_ij with rho rounded to 0.001; Fe-56 is the
    # evaluation the thesis compares against, and the NC LTY=0 case of MF33 MT2.
    from kika.endf import read_endf

    report = check_covariances(read_endf(str(fe56_jendl_tape), mf_numbers=[1, 2, 3, 4, 33, 34]))
    assert report.defects == ()
    psd = report.by_check("not_positive_semidefinite")
    assert psd and all(f.level == NOTE for f in psd)
