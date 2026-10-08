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
                               ("defect", "correlation_out_of_bounds"): 7,
                               ("defect", "joint_not_positive_semidefinite"): 1}
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
    # The joint of a_1..a_6 is indefinite, and the cross block that does most of it
    # on its own is L2xL3: one of those three, |rho| <= 1 everywhere and still not
    # compatible with its two self blocks (C9, 8-oct-2026).
    joint = report.by_check("joint_not_positive_semidefinite")[0].evidence
    assert joint["excess_ratio"] == pytest.approx(0.168, abs=1e-3)
    assert joint["pairs_indefinite_alone"][0]["pair"] == ["MT2 a_2", "MT2 a_3"]


def test_w186_mf34_rho_49_a_negative_variance_and_an_indefinite_block():
    """JEFF-4.0 W-186 MT51 L=1, defect C: all three faults in one LB=5."""
    report = _check("w186")
    assert _faults(report) == {("defect", "negative_variance"): 1,
                               ("defect", "correlation_out_of_bounds"): 1,
                               ("defect", "not_positive_semidefinite"): 1}
    # LAB covariances (LCT=1) for an MF4 given in CM (C9). A note since the
    # samplers convert between frames for MT51 (8-oct-2026).
    frame = report.by_check("frame_differs_from_mf4")[0]
    assert frame.level == "note" and frame.evidence["mf4_lct"] == {"51": 2}
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


# ---------------------------------------------------------------------------
# MF32 and MF35 (phase C8). The MF32 cuts of ENDF/B-VIII.1 were committed for
# the MF32 parser (``test_mf32_roundtrip.py``) and carry MF2 with them.
# ---------------------------------------------------------------------------


def _check_file(name: str):
    return check_covariances(read_endf(str(DATA / name)))


def test_k41_mf32_an_intg_record_outside_the_matrix():
    """JEFF-4.0 K-41: NNN = 93, and an INTG record at row 201. The correlations it
    carries belong to no parameter, so the range is reported and not decoded --
    by the check and by the model's decoder, which used to raise IndexError."""
    from kika.endf.model_adapter.parameter_covariances import decodeMF32MT

    report = _check("k41")
    assert _faults(report) == {("defect", "correlation_index_out_of_range"): 1}
    found = report.by_check("correlation_index_out_of_range")[0]
    assert found.location.range_index == 0
    assert found.evidence["nnn"] == 93 and found.evidence["ii_jj"] == [[201, 200]]
    # The unresolved range writes LRF=1 against MF2's LRF=2: one §32.2.4 format
    # serves every LRF, so that is not a formalism mismatch.
    assert report.by_check("formalism_mismatch") == ()
    endf = read_endf(str(DATA / "micro_k41_covcheck.endf"))
    covariances, conversion = decodeMF32MT(endf.files[32].mt[151])
    assert any("outside the NNN=93 matrix" in m for m in conversion.losses)
    assert len(covariances) == 1        # the unresolved range still converts


def test_th232_mf32_a_correlation_matrix_that_cannot_be_sampled():
    """ENDF/B-VIII.1 Th-232: lambda_min of the resolved range's correlation
    matrix is -0.66, past what its NDIGIT=2 rounding can do. On the covariance it
    would read -3e-5 of lambda_max, a warning: ER and the widths sit on scales
    that far apart."""
    report = _check_file("micro_th232_mf32.endf")
    assert _faults(report) == {("defect", "not_positive_semidefinite"): 1}
    psd = report.by_check("not_positive_semidefinite")
    defect = next(f for f in psd if f.level == "defect")
    assert defect.location.range_index == 0
    assert defect.evidence["lambda_min"] == pytest.approx(-0.661, abs=1e-3)
    assert defect.evidence["lambda_min"] < -defect.evidence["quantised_rounding_bound"]
    # The unresolved range is indefinite too, within rounding.
    assert next(f for f in psd if f.level == "note").location.range_index == 1


@pytest.mark.parametrize("name", ["micro_cl35_mf32.endf", "micro_cm244_mf32.endf",
                                  "micro_na23_mf32.endf"])
def test_mf32_clean_evaluations_give_notes_only(name):
    """LRF=7 (Cl-35), LCOMP=0 (Cm-244) and LCOMP=2 with NM=0 (Na-23): nothing
    above a note, and every resonance energy found in MF2."""
    report = _check_file(name)
    assert report.at_least("warn") == ()
    assert set(f.check for f in report) <= {"inert_rows", "relative_uncertainty_above_one"}


def test_u239_mf35_not_the_covariance_of_a_normalised_spectrum():
    """JEFF-4.0 U-239: 21 bands whose rows do not sum to zero, the first of them
    empty, and sigma(P) up to 2919 for a probability."""
    report = _check("u239")
    assert _faults(report) == {("defect", "sum_rule_violated"): 21,
                               ("defect", "variance_exceeds_physical_bound"): 19,
                               ("warn", "band_empty"): 1}
    assert report.by_check("band_empty")[0].location.band == 0
    worst = max(report.by_check("variance_exceeds_physical_bound"),
                key=lambda f: f.evidence["worst_sigma"])
    assert worst.evidence["worst_sigma"] == pytest.approx(2919.3, abs=0.1)
    assert worst.evidence["worst_bound"] == 0.5          # MF5 left out of the cut
    assert all(f.evidence["row_sum_residual"] > 0.9
               for f in report.by_check("sum_rule_violated"))


def test_cf252_mf35_the_reference_spectrum_keeps_its_sum_rule_and_bound():
    """ENDF/B-VIII.1 Cf-252 with its MF5: the sum rule holds, sigma(P) is inside
    P(1-P) with MF5's own P, and nothing is above a note. Its bands are slightly
    indefinite (-1.4e-4 of lambda_max, not 6-figure rounding), which MF33 and
    MF34 would grade a light warning; for MF35 what is graded is what forcing
    the band to PSD would do to sigma, and that is 0.2-0.3 %."""
    report = _check_file("micro_cf252_pfns.endf")
    assert report.at_least("warn") == ()
    assert report.by_check("central_values_unavailable") == ()
    psd = report.by_check("not_positive_semidefinite")
    assert len(psd) == 4
    assert max(f.evidence["ratio"] for f in psd) == pytest.approx(1.40e-4, abs=0.01e-4)
    assert max(f.evidence["sigma_change_if_clipped"] for f in psd) < 0.004
