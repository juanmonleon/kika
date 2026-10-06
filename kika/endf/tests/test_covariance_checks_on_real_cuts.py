"""Layer 1 on real evaluations: one committed micro-tape per fault (phase C7).

The synthetic tests in ``test_covariance_checks.py`` pin each rule; these pin
that the rules find what the evaluators actually wrote. Each fixture is a
verbatim, closed cut of a real tape -- ``COV_CHECK_FIXTURES`` in
``test_micro_tape_regen.py`` says what was kept and why -- and each was checked
to give the same findings as the full tape on the same sections. The numbers
asserted here are the ones the library-wide validation measured.
Plan: kika-workspace ``docs/library/cov_checks_roadmap.md``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from kika.endf import check_covariances, read_endf

DATA = Path(__file__).resolve().parent / "data"


def _check(key: str):
    return check_covariances(read_endf(str(DATA / f"micro_{key}_covcheck.endf")))


def _faults(report):
    return Counter((f.level, f.check) for f in report.at_least("warn"))


def test_ne20_mf34_the_mirror_of_an_ls1_triangle_is_what_breaks_rho():
    """JEFF-4.0 Ne-20 MT2, defect A: seven cross-order blocks stored as LS=1."""
    report = _check("ne20")
    assert _faults(report) == {("defect", "ls1_in_cross_block"): 10,
                               ("defect", "correlation_out_of_bounds"): 7}
    rho = {(f.location.l, f.location.l1): f for f in report.by_check("correlation_out_of_bounds")}
    assert set(rho) == {(1, 2), (1, 3), (1, 5), (2, 4), (2, 5), (3, 5), (4, 5)}
    worst = rho[(1, 5)]
    # Valid as stored, a correlation of 4.63 once kika mirrors the triangle.
    assert worst.evidence["stored_triangle_max_abs_rho"] == pytest.approx(0.877, abs=1e-3)
    assert worst.evidence["max_abs_rho"] == pytest.approx(4.634, abs=1e-3)
    assert "creates it" in worst.summary
    # Every |rho| > 1 sits in a block with an LS=1 triangle; three LS=1 blocks
    # (L1xL4, L2xL3, L3xL4) stay within bounds even mirrored.
    ls1 = {(f.location.l, f.location.l1) for f in report.by_check("ls1_in_cross_block")
           if f.level == "defect"}
    assert set(rho) <= ls1 and len(ls1 - set(rho)) == 3


def test_w186_mf34_rho_49_a_negative_variance_and_an_indefinite_block():
    """JEFF-4.0 W-186 MT51 L=1, defect C: all three faults in one LB=5."""
    report = _check("w186")
    assert _faults(report) == {("defect", "negative_variance"): 1,
                               ("defect", "correlation_out_of_bounds"): 1,
                               ("defect", "not_positive_semidefinite"): 1}
    assert all((f.location.mt, f.location.l, f.location.l1) == (51, 1, 1)
               for f in report.at_least("warn"))
    rho = report.by_check("correlation_out_of_bounds")[0].evidence
    assert rho["max_abs_rho"] == pytest.approx(49.05, abs=0.01) and rho["n_above"] == 224
    assert report.by_check("negative_variance")[0].evidence["n"] == 10
    psd = report.by_check("not_positive_semidefinite")[0].evidence
    assert psd["ratio"] == pytest.approx(0.414, abs=1e-3)
    assert [r["ni"] for r in psd["records_indefinite_alone"]] == [0]


def test_fe57_mf33_sum_rules_that_name_reactions_with_no_section():
    """JEFF-4.0 Fe-57: MT2 and MT4 are derived (NC LTY=0) partly from MTs the
    file does not have, so what kika assembles is smaller than what is stated."""
    report = _check("fe57")
    assert _faults(report) == {("defect", "unresolved_reference"): 5}
    named = sorted((f.location.mt, f.location.nc, f.evidence["xmti"])
                   for f in report.by_check("unresolved_reference"))
    assert named == [(2, 1, 3), (4, 0, 3), (4, 0, 22), (4, 0, 28), (4, 0, 107)]
    assert _only(report, "nc_only_first_resolved").location.mt == 2


def test_si28_mf33_a_constituent_derived_over_the_same_range():
    """ENDF/B-VIII.1 Si-28: MT1 from MT4 below 2.75 MeV, MT4 itself derived above
    1.8431 MeV -- the two ranges overlap, which §33.3.1 does not allow."""
    report = _check("si28")
    assert _faults(report) == {("defect", "nc_chained"): 1}
    f = report.by_check("nc_chained")
    defect = [x for x in f if x.level == "defect"][0]
    assert (defect.location.mt, defect.evidence["xmti"]) == (1, 4)
    assert defect.evidence["range"] == pytest.approx([1e-5, 2.75e6])
    assert defect.evidence["constituent_range"] == pytest.approx([1.8431e6, 2.0e7])
    # The same chain through a range that does not overlap is only a note.
    assert {x.level for x in f} == {"defect", "note"}


def test_hf176_mf33_a_repeated_point_and_an_implausible_sigma_rel():
    """JEFF-4.0 Hf-176 MT107: with MF2 and MF3 on the tape, the magnitude check
    has its central values, and sigma_rel 10.7 at 9-10 MeV is past the measured
    edge of all three libraries while sigma is 1.3 % of its maximum."""
    report = _check("hf176")
    assert _faults(report) == {("warn", "grid_repeated_point"): 1,
                               ("warn", "implausible_relative_uncertainty"): 1}
    assert report.by_check("grid_repeated_point")[0].evidence["energies"] == [2.0e7]
    big = report.by_check("implausible_relative_uncertainty")[0].evidence
    assert big["worst_rel"] == pytest.approx(10.74, abs=0.01)
    assert big["worst_bin"] == pytest.approx([9.0e6, 1.0e7])
    assert big["central_over_max"] == pytest.approx(0.0133, abs=1e-4)
    assert report.by_check("central_values_unavailable") == ()
    # The larger sigma_rel, 14.2 at 5-5.5 MeV, is where sigma is 0.3 % of its
    # maximum -- the threshold, where a large sigma_rel is legitimate. It stays a
    # note: the false warning the absolute criterion exists to avoid.
    near = report.by_check("relative_uncertainty_above_one")[0].evidence
    assert near["worst_rel"] == pytest.approx(14.16, abs=0.01)
    assert near["worst_rel"] > big["worst_rel"]
    assert near["worst_bin"] == pytest.approx([5.0e6, 5.5e6])


def test_fe56_jendl5_the_clean_case_has_only_notes():
    """JENDL-5 Fe-56 MF33 and MF34: no fault at all. Its 50 indefinite blocks are
    rounding -- JENDL-5 writes rho rounded to 0.001 -- and must stay notes."""
    report = _check("fe56_jendl")
    assert report.at_least("warn") == ()
    assert Counter(f.check for f in report) == {"inert_rows": 57,
                                                "not_positive_semidefinite": 50,
                                                "central_values_unavailable": 2}


def _only(report, check):
    found = report.by_check(check)
    assert len(found) == 1, [str(f) for f in found]
    return found[0]
