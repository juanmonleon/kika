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


def test_sigma_rel_above_one_is_a_note_with_the_chance_of_a_negative_draw():
    # 200 % on a flat 1 b cross section: allowed (a lognormal carries it), but a
    # normal draw goes negative Phi(-0.5) = 31 % of the time.
    big = np.diag([4.0, 0.01, 0.01])
    xs = {1: SimpleNamespace(energies=np.array([0.1, 1e4]), values=np.array([1.0, 1.0]))}
    report = check_covariances(_tape(_section({1: [_lb5(big)]})), xs_sections=xs)
    f = _only(report, "relative_uncertainty_above_one")
    assert f.level == NOTE and f.evidence["worst_rel"] == pytest.approx(2.0)
    assert f.evidence["p_negative_if_normal"] == pytest.approx(0.3085, abs=1e-4)
    assert report.at_least(WARN) == ()


def test_sigma_rel_above_ten_warns_only_outside_a_threshold_region():
    huge = np.diag([400.0, 0.01, 0.01])  # sigma_rel 20 in the first bin
    flat = {1: SimpleNamespace(energies=np.array([0.1, 1e4]), values=np.array([1.0, 1.0]))}
    f = _only(check_covariances(_tape(_section({1: [_lb5(huge)]})), xs_sections=flat),
              "implausible_relative_uncertainty")
    assert f.level == WARN and f.evidence["worst_rel"] == pytest.approx(20.0)
    # The same 2000 % where sigma_bar is under 1 % of the maximum (a threshold).
    ramp = {1: SimpleNamespace(energies=np.array([1.0, 10.0, 1e4]),
                               values=np.array([0.0, 0.0, 1.0]))}
    report = check_covariances(_tape(_section({1: [_lb5(huge)]})), xs_sections=ramp)
    assert report.by_check("implausible_relative_uncertainty") == ()


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
    mf4.add_section(SimpleNamespace(  # a_1 = 0.1 at every energy
        number=2,
        legendre_cell_averages=lambda edges, order: {order: np.full(len(edges) - 1, 0.1)},
        legendre_cell_min_abs=lambda edges, order: np.full(len(edges) - 1, 0.1)))
    f = _only(check_covariances(_tape(sec34, mf_number=34, files=[mf4])),
              "variance_exceeds_physical_bound")
    assert f.level == DEFECT  # |a_l| <= 1 caps sigma at 1 (Popoviciu)
    assert f.evidence["worst_sigma_abs"] == pytest.approx(2.0)  # 20 x |a_1| = 0.1


# --------------------------------------------------------------------------
# C4 -- positive semi-definiteness, in three levels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ratio, level", [(1e-4, NOTE), (1e-2, DEFECT)])
def test_psd_levels(ratio, level):
    # Spread over every row, a negative eigenvalue of 1e-4 lambda_max moves no
    # sigma by 2 %: the light warning is regraded a note. 1e-2 is a defect by ratio.
    n = 6
    m = _spectrum([1.0, 0.8, 0.5, 0.3, 0.2, -ratio]) * 0.01
    f = _only(check_covariances(_tape(_section({1: [_lb5(m, _grid(n))]}))),
              "not_positive_semidefinite")
    assert f.level == level and f.evidence["ratio"] == pytest.approx(ratio, rel=1e-6)


@pytest.mark.parametrize("a, depth, level", [(4e-6, 1.68e-6, WARN), (2e-6, 5e-6, DEFECT)])
def test_a_light_warning_is_graded_by_what_clipping_does_to_sigma(a, depth, level):
    # A negative eigenvalue -depth confined to two rows of variance a: clipping
    # adds depth/2 to each, sigma grows by ~10 % (warning) or ~50 % (defect), and
    # the ratio is in the light band (1.7e-4 and 5e-4 of lambda_max = 0.01) both times.
    m = np.zeros((4, 4))
    m[:2, :2] = 0.01 * np.eye(2)
    m[2:, 2:] = [[a, a + depth], [a + depth, a]]
    f = _only(check_covariances(_tape(_section({1: [_lb5(m, _grid(4))]}))),
              "not_positive_semidefinite")
    assert 1e-6 < f.evidence["ratio"] <= 1e-3
    assert f.level == level
    assert f.evidence["sigma_change_if_clipped"] == pytest.approx(
        np.sqrt(1 + depth / (2 * a)) - 1, rel=1e-6)


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
    f = _only(check_covariances(endf), "relative_uncertainty_above_one")
    assert f.location.mf == 31 and f.evidence["worst_rel"] == pytest.approx(2.0)


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
    assert _only(report, "relative_uncertainty_above_one").evidence["worst_rel"] ==         pytest.approx(2.0)


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


def _mf34_case(l, a_of_e, variance=400.0):
    sec34 = MF34MT(number=2, _za=26056.0, _awr=55.45, _ltt=3, _nmt1=1, _mat=2631)
    rec = _lb5(np.diag([variance, 0.01, 0.01]), cls=SubSubsectionRecord)
    sec34.add_subsection(Subsection34(
        mt1=2, nl=2, nl1=2, mat1=0,
        sub_subsections=[SubSubsection(l=l, l1=l, lct=1, ni=1, records=[rec])]))
    mf4 = MF(number=4)
    mf4.add_section(SimpleNamespace(
        number=2,
        extract_legendre_coefficients=lambda e, max_legendre_order, out_of_range:
            {l: a_of_e(np.asarray(e, dtype=float))}))
    return check_covariances(_tape(sec34, mf_number=34, files=[mf4]))


def test_the_legendre_bound_skips_a0_and_uses_the_smallest_a_l_in_the_bin():
    # a_0 = 1 by normalisation: its covariance is not bounded by |a_l| <= 1.
    report = _mf34_case(0, lambda e: np.ones_like(e))
    assert report.by_check("variance_exceeds_physical_bound") == ()
    # a_1 crossing zero inside the first bin (1-10 eV): its average is 0.1, but the
    # file's sigma_rel 20 may be relative to a value near 0 -- not a certain breach.
    report = _mf34_case(1, lambda e: np.where(e <= 10.0, (e - 5.5) / 22.5 + 0.1, 0.1))
    assert report.by_check("variance_exceeds_physical_bound") == ()


# --------------------------------------------------------------------------
# C5 -- completeness: the blocks and references a section implies
# --------------------------------------------------------------------------

G2 = [1.0, 10.0, 100.0]
SELF2 = [[0.01, 0.002], [0.002, 0.01]]


def _lb6(matrix, rows=G2, cols=G2):
    m = np.asarray(matrix, dtype=float)
    return NISubSubsectionRecord(lb=6, ne=len(rows), nt=1 + len(rows) * len(cols),
                                 row_energies=list(rows), col_energies=list(cols),
                                 rect_matrix=list(m.ravel()))


def _nc0(refs, e1=1.0, e2=100.0):
    from kika.endf.classes.mf33 import NCSubSubsection

    return NCSubSubsection(lty=0, e1=e1, e2=e2, nci=len(refs),
                           ci=[c for c, _ in refs], xmti=[float(m) for _, m in refs])


def _with_nc(sec, *ncs):
    sub = next(s for s in sec.subsections if s.mt1 == sec.number)
    sub.nc_records.extend(ncs)
    sub.nc = len(sub.nc_records)
    return sec


def test_a_section_without_its_own_variance_and_a_partner_with_no_section():
    sec = _section({2: [_lb6([[0.001, 0.0], [0.0, 0.001]])]}, mt=1)
    report = check_covariances(_tape(sec))
    assert _only(report, "missing_self_block").location.mt == 1
    f = _only(report, "missing_partner")
    assert f.level == DEFECT and (f.location.mt, f.location.mt1) == (1, 2)


def test_a_cross_block_below_the_diagonal_is_a_note_when_stated_once():
    low = _section({2: [_lb5(SELF2, G2)], 1: [_lb6([[0.001, 0.0], [0.0, 0.001]])]}, mt=2)
    report = check_covariances(_tape(_section({1: [_lb5(SELF2, G2)]}, mt=1), low))
    f = _only(report, "cross_block_below_diagonal")
    assert f.level == NOTE and (f.location.mt, f.location.mt1) == (2, 1)
    assert report.at_least(WARN) == ()


def test_a_cross_block_stated_both_ways_is_compared():
    c = [[0.001, 0.0005], [0.0, 0.001]]
    up = _section({1: [_lb5(SELF2, G2)], 2: [_lb6(c)]}, mt=1)
    agree = _section({2: [_lb5(SELF2, G2)], 1: [_lb6(np.transpose(c))]}, mt=2)
    f = _only(check_covariances(_tape(up, agree)), "symmetric_block_repeated")
    assert f.level == NOTE and f.evidence["relative_difference"] == 0.0
    clash = _section({2: [_lb5(SELF2, G2)], 1: [_lb6(c)]}, mt=2)  # not transposed
    f = _only(check_covariances(_tape(up, clash)), "symmetric_block_conflict")
    assert f.level == DEFECT and f.evidence["relative_difference"] == pytest.approx(0.5)


def test_mat1_written_as_the_own_mat_is_read_as_the_same_material():
    sec = _section({1: [_lb5(SELF2, G2)], 2: [_lb6([[0.001, 0.0], [0.0, 0.001]])]}, mt=1)
    for sub in sec.subsections:
        sub.mat1 = sec._mat
    report = check_covariances(_tape(sec, _section({2: [_lb5(SELF2, G2)]}, mt=2)))
    assert _only(report, "mat1_is_own_mat").evidence["mt1"] == [1, 2]
    assert report.by_check("missing_self_block") == ()
    assert report.by_check("external_material") == ()


def test_nc_lty0_references_that_do_not_resolve():
    mt4 = _with_nc(_section({4: []}, mt=4), _nc0([(1.0, 51), (1.0, 52), (1.0, 4)]))
    mt51 = _section({51: [_lb5(SELF2, G2)]}, mt=51)
    mt52 = MF33MT(number=52, _za=26056.0, _awr=55.45, _mat=2631, _nl=0, _mtl=851)
    lump = _section({851: [_lb5(SELF2, G2)]}, mt=851)
    report = check_covariances(_tape(mt4, mt51, mt52, lump))
    refs = report.by_check("unresolved_reference")
    assert {f.evidence["xmti"] for f in refs} == {52, 4}
    assert all(f.level == DEFECT and f.location.nc == 0 for f in refs)


def test_nc_lty0_out_of_place_and_with_overlapping_ranges():
    mt1 = _section({1: [_lb5(SELF2, G2)], 2: []}, mt=1)
    cross = next(s for s in mt1.subsections if s.mt1 == 2)
    cross.nc_records.append(_nc0([(1.0, 2)]))
    cross.nc = 1
    mt2 = _section({2: [_lb5(SELF2, G2)]}, mt=2)
    assert _only(check_covariances(_tape(mt1, mt2)), "nc_misplaced").location.mt1 == 2

    mt3 = _with_nc(_section({3: []}, mt=3), _nc0([(1.0, 1)], 1.0, 50.0),
                   _nc0([(1.0, 2)], 20.0, 100.0))
    report = check_covariances(_tape(_section({1: [_lb5(SELF2, G2)]}, mt=1), mt2, mt3))
    assert _only(report, "nc_ranges_overlap").location.nc == 1
    assert _only(report, "nc_only_first_resolved").level == NOTE


def test_a_constituent_derived_in_the_same_range_and_a_cycle_in_another():
    mt1 = _with_nc(_section({1: [_lb5(SELF2, G2)]}, mt=1), _nc0([(1.0, 2), (1.0, 3)], 1.0, 10.0))
    mt2 = _with_nc(_section({2: [_lb5(SELF2, G2)]}, mt=2),
                   _nc0([(1.0, 1), (-1.0, 3)], 10.0, 100.0))
    mt3 = _section({3: [_lb5(SELF2, G2)]}, mt=3)
    chained = check_covariances(_tape(mt1, mt2, mt3)).by_check("nc_chained")
    # As in Si-28 of ENDF/B-VIII.1: MT1 from MT2 below 10 eV, MT2 from MT1 above.
    assert {(f.location.mt, f.level) for f in chained} == {(1, NOTE), (2, NOTE)}
    mt2.subsections[0].nc_records[0].e1 = 5.0  # now overlapping the range of MT1
    chained = check_covariances(_tape(mt1, mt2, mt3)).by_check("nc_chained")
    assert {f.level for f in chained} == {DEFECT}


def test_lumped_reactions_and_ratios_to_standards():
    from kika.endf.classes.mf33 import NCSubSubsection

    comp = MF33MT(number=51, _za=26056.0, _awr=55.45, _mat=2631, _nl=0, _mtl=852)
    orphan = _section({851: [_lb5(SELF2, G2)]}, mt=851)
    report = check_covariances(_tape(comp, orphan))
    assert _only(report, "unresolved_reference").evidence["mtl"] == 852
    assert _only(report, "lumped_without_components").location.mt == 851

    ratio = NCSubSubsection(lty=1, e1=1.0, e2=100.0, mats=2631, mts=1)
    report = check_covariances(_tape(_with_nc(_section({1: [_lb5(SELF2, G2)]}), ratio)))
    assert _only(report, "unresolved_reference").evidence["mats"] == 2631
    assert _only(report, "ratio_to_standard").level == NOTE


def _mf34(mt, blocks, nl=2, ltt=1):
    """An MF34 section; ``blocks`` maps MT1 -> {(L, L1): [records]}."""
    sec = MF34MT(number=mt, _za=26056.0, _awr=55.45, _ltt=ltt, _nmt1=len(blocks), _mat=2631)
    for mt1, subs in blocks.items():
        sec.add_subsection(Subsection34(
            mt1=mt1, nl=nl, nl1=nl, mat1=0,
            sub_subsections=[SubSubsection(l=l, l1=l1, lct=1, ni=len(recs), records=recs)
                             for (l, l1), recs in subs.items()]))
    return sec


def _lb5_34(matrix, ls=1):
    return _lb5(matrix, G2, ls=ls, cls=SubSubsectionRecord)


def test_mf34_a_cross_order_block_without_the_variance_of_one_order():
    # NL=2 declares a_1 and a_2; only a_1 has a variance block, and (1, 2) is stated.
    cross = _lb5_34([[0.001, 0.0005], [0.0002, 0.001]], ls=0)
    sec = _mf34(2, {2: {(1, 1): [_lb5_34(SELF2)], (1, 2): [cross], (2, 2): []}})
    report = check_covariances(_tape(sec, mf_number=34))
    f = [f for f in report.by_check("missing_partner") if f.location.l1 == 2]
    assert len(f) == 1 and f[0].level == DEFECT and f[0].evidence["absent"] == ["a_2 of MT2"]
    assert _only(report, "order_without_variance").evidence["l"] == [2]
    # The same block with all zeros only says so.
    zero = _lb5_34([[0.0, 0.0], [0.0, 0.0]], ls=0)
    sec = _mf34(2, {2: {(1, 1): [_lb5_34(SELF2)], (1, 2): [zero], (2, 2): []}})
    f = _only(check_covariances(_tape(sec, mf_number=34)), "missing_partner")
    assert f.level == NOTE


def test_mf34_partner_sections_and_blocks_stated_both_ways():
    c = [[0.001, 0.0005], [0.0, 0.001]]
    up = _mf34(2, {2: {(1, 1): [_lb5_34(SELF2)]}, 51: {(1, 1): [_lb5_34(c, ls=0)]}}, nl=1)
    report = check_covariances(_tape(up, mf_number=34))
    assert _only(report, "missing_partner").location.mt1 == 51
    low = _mf34(51, {51: {(1, 1): [_lb5_34(SELF2)]},
                     2: {(1, 1): [_lb5_34(np.transpose(c), ls=0)]}}, nl=1)
    report = check_covariances(_tape(up, low, mf_number=34))
    assert report.by_check("missing_partner") == ()
    f = _only(report, "symmetric_block_repeated")
    assert (f.location.mt, f.location.mt1, f.location.l, f.location.l1) == (51, 2, 1, 1)


# --------------------------------------------------------------------------
# C9 -- the comparison with FUDGE, and what neither had
# --------------------------------------------------------------------------

def _mf3(*mts, energies=(1e-5, 2e7), value=1.0):
    mf3 = MF(number=3)
    for mt in mts:
        mf3.add_section(SimpleNamespace(number=mt, energies=np.array(energies, dtype=float),
                                        cross_sections=np.full(len(energies), value)))
    return mf3


def test_a_non_zero_last_f_is_a_warning():
    rec = _lb1([0.01, 0.01, 0.01])
    rec.f_table_k[-1] = 0.02
    f = _only(check_covariances(_tape(_section({1: [rec]}))), "trailing_value_not_zero")
    assert f.level == WARN and f.evidence["value"] == pytest.approx(0.02)
    clean = check_covariances(_tape(_section({1: [_lb1([0.01, 0.01, 0.01])]})))
    assert clean.by_check("trailing_value_not_zero") == ()
    # In LB=8 the major libraries do it as a habit: a note.
    rec = _lb1([1e-4, 1e-4, 1e-4], lb=8)
    rec.f_table_k[-1] = 1e-4
    f = _only(check_covariances(_tape(_section({1: [_lb5(CLEAN), rec]}))),
              "trailing_value_not_zero")
    assert f.level == NOTE


def test_a_short_range_variance_between_two_reactions_is_a_defect():
    sec = _section({2: [_lb5(CLEAN)], 102: [_lb1([1e-3, 1e-3, 1e-3], lb=8)]}, mt=2)
    f = _only(check_covariances(_tape(sec, _section({102: [_lb5(CLEAN)]}, mt=102))),
              "short_range_in_cross_block")
    assert f.level == DEFECT and (f.location.mt, f.location.mt1, f.location.lb) == (2, 102, 8)


def test_a_short_range_variance_just_above_a_threshold_is_a_note():
    grid = [1e6, 1.5e6, 5e6, 2e7]
    mf3 = MF(number=3)
    mf3.add_section(SimpleNamespace(number=16, energies=np.array([1e6, 2e6, 2e7]),
                                    cross_sections=np.array([0.0, 0.1, 0.5])))
    sec = _section({16: [_lb5(CLEAN, grid=grid), _lb1([1e-4, 1e-4, 1e-4], grid=grid, lb=8)]},
                   mt=16)
    f = _only(check_covariances(_tape(sec, files=[mf3])), "short_range_near_threshold")
    assert f.level == NOTE and f.evidence["threshold"] == pytest.approx(1e6)


def test_a_covariance_of_a_reaction_the_tape_does_not_give():
    report = check_covariances(_tape(_section({16: [_lb5(CLEAN)]}, mt=16),
                                     files=[_mf3(2, 102)]))
    f = _only(report, "missing_central_values")
    assert f.level == DEFECT and f.location.mt == 16
    assert report.by_check("central_values_unavailable") == ()
    # MT3 is not in MF3 but its partials are.
    report = check_covariances(_tape(_section({3: [_lb5(CLEAN)]}, mt=3), files=[_mf3(4, 102)]))
    assert report.by_check("missing_central_values") == ()
    # Without MF3 read nothing is judged.
    report = check_covariances(_tape(_section({16: [_lb5(CLEAN)]}, mt=16)))
    assert report.by_check("missing_central_values") == ()


def test_mf31_nubar_452_may_come_from_455_and_456():
    mf1 = MF(number=1)
    for mt in (455, 456):
        mf1.add_section(SimpleNamespace(
            number=mt, get_nubar=lambda e, out_of_range: np.full(np.size(e), 2.4)))
    endf = _tape(_section({452: [_lb5(CLEAN)]}, mt=452), mf_number=31, files=[mf1])
    assert check_covariances(endf).by_check("missing_central_values") == ()


def _mf4(mt, lct=2, a=0.1):
    return SimpleNamespace(
        number=mt, _lct=lct,
        legendre_cell_averages=lambda edges, order: {order: np.full(len(edges) - 1, a)},
        legendre_cell_min_abs=lambda edges, order: np.full(len(edges) - 1, abs(a)))


def test_mf34_without_mf4_is_a_defect_and_a_note_when_mf6_has_it():
    sec = _mf34(2, {2: {(1, 1): [_lb5_34(SELF2)]}}, nl=1)
    mf4 = MF(number=4)
    mf4.add_section(_mf4(51))
    f = _only(check_covariances(_tape(sec, mf_number=34, files=[mf4])), "missing_central_values")
    assert f.level == DEFECT
    mf6 = MF(number=6)
    mf6.add_section(SimpleNamespace(number=2))
    f = _only(check_covariances(_tape(sec, mf_number=34, files=[mf4, mf6])),
              "missing_central_values")
    assert f.level == NOTE and f.evidence["in_mf6"]


def test_mf34_frame_against_mf4():
    sec = _mf34(2, {2: {(1, 1): [_lb5_34(SELF2)]}}, nl=1)  # LCT=1, LAB
    mf4 = MF(number=4)
    mf4.add_section(_mf4(2, lct=2))
    f = _only(check_covariances(_tape(sec, mf_number=34, files=[mf4])), "frame_differs_from_mf4")
    assert f.level == WARN and f.evidence["lct"] == 1
    sec.subsections[0].sub_subsections[0].lct = 0  # the same frame as MF4
    report = check_covariances(_tape(sec, mf_number=34, files=[mf4]))
    assert report.by_check("frame_differs_from_mf4") == ()


def test_mf34_a_non_null_l0_block_counts_the_magnitude_twice():
    sec = _mf34(2, {2: {(0, 0): [_lb5_34(SELF2)], (1, 1): [_lb5_34(SELF2)]}}, ltt=3)
    f = _only(check_covariances(_tape(sec, mf_number=34)), "magnitude_covariance_in_mf34")
    assert f.level == WARN and (f.location.l, f.location.l1) == (0, 0)


def test_variance_where_the_central_value_is_zero():
    ramp = {1: SimpleNamespace(energies=np.array([1.0, 10.0, 1e4]),
                               values=np.array([0.0, 0.0, 1.0]))}
    f = _only(check_covariances(_tape(_section({1: [_lb5(CLEAN)]})), xs_sections=ramp),
              "variance_where_central_value_is_zero")
    assert f.level == NOTE and f.evidence["bins"] == [[1.0, 10.0]]
    # MF34: an order MF4 gives as zero everywhere.
    sec = _mf34(2, {2: {(1, 1): [_lb5_34(SELF2)]}}, nl=1)
    mf4 = MF(number=4)
    mf4.add_section(_mf4(2, lct=1, a=0.0))
    f = _only(check_covariances(_tape(sec, mf_number=34, files=[mf4])),
              "variance_where_central_value_is_zero")
    assert f.evidence["n"] == 2


ONE = [1.0, 10.0]


def _joint_tape(r12, r13, r23):
    """Three one-bin reactions, unit relative variances, the given correlations."""
    def lb5(v):
        return _lb5([[v]], grid=ONE, ls=0)
    return _tape(_section({2: [_lb5([[1.0]], grid=ONE)], 4: [lb5(r12)], 102: [lb5(r13)]}, mt=2),
                 _section({4: [_lb5([[1.0]], grid=ONE)], 102: [lb5(r23)]}, mt=4),
                 _section({102: [_lb5([[1.0]], grid=ONE)]}, mt=102))


def test_pairwise_compatible_blocks_can_make_an_indefinite_joint():
    # |rho| <= 0.9 everywhere and no pair alone is indefinite, but x = (1, -1, -1)
    # has variance 3 - 5.4 < 0.
    report = check_covariances(_joint_tape(0.9, 0.9, -0.9))
    assert report.by_check("correlation_out_of_bounds") == ()
    assert report.by_check("not_positive_semidefinite") == ()
    f = _only(report, "joint_not_positive_semidefinite")
    assert f.level == DEFECT and f.location.mt is None
    assert f.evidence["nodes"] == ["MT2", "MT4", "MT102"]
    assert f.evidence["pairs_indefinite_alone"] == []
    expected = np.linalg.eigvalsh([[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1]])[0]
    assert f.evidence["lambda_min"] == pytest.approx(expected)
    # The same correlations with a consistent sign are a valid joint matrix.
    report = check_covariances(_joint_tape(0.9, 0.9, 0.9))
    assert report.by_check("joint_not_positive_semidefinite") == ()


def test_the_joint_names_a_cross_block_that_is_indefinite_alone():
    f = _only(check_covariances(_joint_tape(1.5, 0.0, 0.0)), "joint_not_positive_semidefinite")
    assert [p["pair"] for p in f.evidence["pairs_indefinite_alone"]] == [["MT2", "MT4"]]


def test_mf34_joint_across_orders():
    def cross(v):
        return _lb5_34([[v, 0.0], [0.0, v]], ls=0)
    sec = _mf34(2, {2: {(1, 1): [_lb5_34(np.eye(2))], (1, 2): [cross(0.9)],
                        (1, 3): [cross(0.9)], (2, 2): [_lb5_34(np.eye(2))],
                        (2, 3): [cross(-0.9)], (3, 3): [_lb5_34(np.eye(2))]}}, nl=3)
    f = _only(check_covariances(_tape(sec, mf_number=34)), "joint_not_positive_semidefinite")
    assert f.location.mt == 2 and f.evidence["nodes"] == ["MT2 a_1", "MT2 a_2", "MT2 a_3"]


def _mf32_lcomp0(rows):
    """MF32 LCOMP=0, LRF=2: rows of (ER, GN, GG, DJ2)."""
    from kika.endf.classes.mf32.mf32mt151 import (CovEnergyRange, CovIsotope, LCOMP0Body,
                                                  MF32MT151)
    from kika.endf.classes.mf32.records import PackedList, Record

    values = []
    for er, gn, gg, dj2 in rows:
        values += [er, 0.5, gn + gg, gn, gg, 0.0,
                   1e-4, 1e-4, 0.0, 1e-4, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, dj2, 0.0]
    block = Record(raw="", l2=0, n1=18 * len(rows), n2=len(rows), body=PackedList())
    block.body.set_values(values)
    rng = CovEnergyRange(el=1e-5, eh=1e3, lru=1, lrf=2,
                         body=LCOMP0Body(control=Record(raw="", n1=1), l_blocks=[block]))
    return MF32MT151(number=151, _za=26056.0, _awr=55.45, _nis=1, _mat=2631,
                     isotopes=[CovIsotope(zai=26056.0, abn=1.0, energy_ranges=[rng])])


def _mf2_mlbw(rows):
    """MF2/MT151 with one MLBW range: rows of (ER, GN, GG)."""
    from kika.endf.classes.mf2.mf2mt151 import (EnergyRange, LValueBlock, Resonance,
                                                ResolvedResonanceRange)

    res = [Resonance(energy=er, spin=0.5, c3=gn + gg, c4=gn, c5=gg, c6=0.0) for er, gn, gg in rows]
    params = ResolvedResonanceRange(spi=0.0, ap=0.5, nls=1, nlsc=1, l_values=[
        LValueBlock(awri=55.45, l=0, num_resonances=len(res), resonances=res)])
    rng = EnergyRange(el=1e-5, eh=1e3, lru=1, lrf=2, nro=0, naps=0, parameters=params)
    mf2 = MF(number=2)
    mf2.add_section(SimpleNamespace(number=151, isotopes=[SimpleNamespace(energy_ranges=[rng])]))
    return mf2


def test_mf32_widths_and_spin_against_mf2():
    # The 20 eV resonance has GN = 2.0 in MF32 and 2.5 in MF2, and an uncertainty on J.
    endf = _tape(_mf32_lcomp0([(10.0, 1.0, 0.5, 0.0), (20.0, 2.0, 0.5, 0.3)]), mf_number=32,
                 files=[_mf2_mlbw([(10.0, 1.0, 0.5), (20.0, 2.5, 0.5)])])
    report = check_covariances(endf, mf=(32,))
    f = _only(report, "widths_differ_from_mf2")
    assert f.level == WARN and (f.evidence["n"], f.evidence["of"]) == (1, 2)
    assert f.evidence["examples"] == [{"er": 20.0, "parameter": "GN", "mf32": 2.0, "mf2": 2.5}]
    assert _only(report, "spin_uncertainty").evidence["n"] == 1
    # A width copied with fewer digits, far inside its sigma (0.01): a note.
    endf = _tape(_mf32_lcomp0([(10.0, 1.0, 0.5, 0.0), (20.0, 2.0, 0.5, 0.0)]), mf_number=32,
                 files=[_mf2_mlbw([(10.0, 1.0, 0.5), (20.0, 2.0004, 0.5)])])
    f = _only(check_covariances(endf, mf=(32,)), "widths_differ_from_mf2")
    assert f.level == NOTE and f.evidence["max_in_sigmas"] == pytest.approx(0.04, rel=1e-3)
    # The same widths within the 1e-5 that ENDF-6 §32.3 allows say nothing.
    endf = _tape(_mf32_lcomp0([(10.0, 1.0, 0.5, 0.0), (20.0, 2.0, 0.5, 0.0)]), mf_number=32,
                 files=[_mf2_mlbw([(10.0, 1.0, 0.5), (20.000001, 2.000001, 0.5)])])
    report = check_covariances(endf, mf=(32,))
    assert report.by_check("widths_differ_from_mf2") == ()
    assert report.by_check("parameters_not_in_mf2") == ()
    assert report.by_check("spin_uncertainty") == ()


# a_0 stands for the integrated cross section (ENDF-6 §34.1, §34.3): its variance
# is MF33's, its own (0, 0) block is null by convention, and the blocks (0, L1)
# carry the magnitude-shape correlation. This is the layout of the Fe-56
# `_a0cross` deliverable, built here synthetically.

NULL2 = [[0.0, 0.0], [0.0, 0.0]]


def _a0cross(cross, mf33_mt=2, l0=NULL2):
    """MF34 MT2 (LTT=3): (0, 0), (0, 1) and (1, 1); MF33 MT ``mf33_mt``.

    (0, 0) is written null by default, as §34.3 asks: NSS = NL(NL+1)/2 is what
    tells a reader how many sub-subsections follow.
    """
    blocks = {(0, 0): [_lb5_34(l0)], (1, 1): [_lb5_34(SELF2)], (0, 1): [_lb5_34(cross, ls=0)]}
    sec34 = _mf34(2, {2: blocks}, ltt=3)
    mf33 = MF(number=33)
    mf33.add_section(_section({mf33_mt: [_lb5(SELF2, grid=G2)]}, mt=mf33_mt))
    return _tape(sec34, mf_number=34, files=[mf33])


def test_a0_takes_its_variance_from_mf33():
    report = check_covariances(_a0cross([[0.005, 0.0], [0.0, 0.005]]))
    assert report.at_least(WARN) == ()
    # Nothing about the null (0, 0) block: not inert rows, not an order without variance.
    assert report.by_check("inert_rows") == () and report.by_check("order_without_variance") == ()
    # rho of (0, 1) is measured against MF33's variance: 0.015 / 0.01 = 1.5.
    f = _only(check_covariances(_a0cross([[0.015, 0.0], [0.0, 0.005]])),
              "correlation_out_of_bounds")
    assert (f.location.l, f.location.l1) == (0, 1)
    assert f.evidence["max_abs_rho"] == pytest.approx(1.5)


def test_a0_without_mf33_for_its_mt_has_no_partner():
    f = _only(check_covariances(_a0cross([[0.005, 0.0], [0.0, 0.005]], mf33_mt=102)),
              "missing_partner")
    assert f.level == DEFECT and f.evidence["absent"] == ["a_0 of MT2 (nor MF33)"]


def test_the_joint_of_mf33_and_mf34_through_a0():
    # Each block is fine alone, but sigma_0 and a_1 correlated at 0.9 in both bins
    # while (1, 1) anti-correlates the two bins of a_1 and MF33 correlates sigma_0's.
    anti = [[0.01, -0.009], [-0.009, 0.01]]
    sec34 = _mf34(2, {2: {(0, 0): [_lb5_34(NULL2)], (1, 1): [_lb5_34(anti)],
                          (0, 1): [_lb5_34([[0.009, 0.0], [0.0, 0.009]], ls=0)]}}, ltt=3)
    mf33 = MF(number=33)
    mf33.add_section(_section({2: [_lb5([[0.01, 0.009], [0.009, 0.01]], grid=G2)]}, mt=2))
    f = _only(check_covariances(_tape(sec34, mf_number=34, files=[mf33])),
              "joint_not_positive_semidefinite")
    assert f.evidence["nodes"] == ["MT2 a_0", "MT2 a_1"]
    assert f.evidence["pairs_indefinite_alone"][0]["pair"] == ["MT2 a_0", "MT2 a_1"]


def test_a_non_null_l0_block_is_compared_with_mf33():
    same = _only(check_covariances(_a0cross([[0.005, 0.0], [0.0, 0.005]], l0=SELF2)),
                 "magnitude_covariance_in_mf34")
    assert same.level == NOTE and same.evidence["mf33"] == "same"
    other = _only(check_covariances(_a0cross([[0.005, 0.0], [0.0, 0.005]],
                                             l0=np.multiply(SELF2, 2.0))),
                  "magnitude_covariance_in_mf34")
    assert other.level == WARN and other.evidence["mf33"] == "different"
