"""Q0 of the PFNS migration: the model draws MF35 the way the shipped pipeline does.

``perturbFromModel`` could already apply a fission-spectrum perturbation, and
``test_perturbing_a_fission_spectrum_on_the_model.py`` gates that applier
against ``perturb_pfns_partial`` point for point -- **on deltas both sides were
handed**. Nothing gated the *draw*, and the draw was where the two differed:

* the model drew in every direction (``nullTol=None``), so decomposition debris
  of ~1e-10 reached groups whose stated variance is exactly zero. Divided by a
  group probability of ~1e-17 that is the ×2.6e+7 spike
  ``core._draw_one_block`` documents;
* the model had no 5σ clamp, which removes what ``clip`` adds to the diagonal
  (``mf35_sampling.generate_pfns_samples``);
* a hand-written conditioning plan could put a congruence on an MF35 band,
  which destroys ``C·1 ≈ 0`` -- the property the normalisation rests on;
* and the budgets the legacy run reports were computed and dropped.

kika-app runs this path, so these are fixes to a live pipeline. The plan is
``kika-workspace/docs/pfns/pfns_mf5_mf35_roadmap.md``, "Phase 2", Q0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.sampling.joint_blocks import (assembleRequest, collectEntries,
                                        componentDomains)
from kika.sampling.mf35_sampling import (NULL_TOL, SIGMA_CLAMP, build_pfns_covariance,
                                         generate_pfns_samples)
from kika.sampling.model_perturbation import perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
CF252 = DATA / "micro_cf252_pfns.endf"
#: Only Cf-252: a run needs an MT18 reaction to hang the spectrum on, and the
#: synthetic ``micro_pfns_cov.endf`` carries MF5 and MF35 without MF3.
TAPES = ("micro_cf252_pfns.endf",)


def _stated(name):
    """``{band: stated matrix}`` straight from the file, before any repair."""
    endf = read_endf(str(DATA / name), mf_numbers=[5, 35])
    suite, _mf5, _bands = build_pfns_covariance(endf, mt=18)
    return suite, {index: np.asarray(section.form.matrix, dtype=float)
                   for index, section in enumerate(suite)}


def _modelDeltas(run):
    """``{band: (nSamples, nGroups)}`` of what the model actually applied."""
    bySample = [{component.index: np.asarray(values, dtype=float)
                 for component, values in sample["set"].factors.items()}
                for sample in run.samples]
    return {band: np.vstack([sample[band] for sample in bySample])
            for band in bySample[0]}


# ======================================================================
# Q0.1 + Q0.2 -- what may be drawn at all
# ======================================================================

@pytest.mark.parametrize("name", TAPES)
def test_no_group_is_drawn_beyond_what_the_file_states(name):
    """Zero stated variance draws zero; nothing goes past 5 stated σ.

    The two failures this replaces are the ones the legacy docstrings measured
    on Cf-252: an untruncated draw puts ~1e-10 on groups with σ = 0, and
    ``clip`` lets a handful of groups reach far past their own marginal.
    """
    _suite, stated = _stated(name)
    run = perturbFromModel(str(DATA / name), {35: None}, nSamples=16, seed=11,
                           dryRun=True)
    drawn = _modelDeltas(run)
    assert set(drawn) == set(stated)

    for band, matrix in stated.items():
        sigma = np.sqrt(np.clip(np.diag(matrix), 0.0, None))
        deltas = drawn[band]
        silent = sigma == 0.0
        assert np.all(deltas[:, silent] == 0.0), (
            f"band {band}: a group the file gives no variance was perturbed")
        assert np.all(np.abs(deltas) <= SIGMA_CLAMP * sigma * (1 + 1e-12)), (
            f"band {band}: a draw exceeds {SIGMA_CLAMP}σ of its stated marginal")


@pytest.mark.parametrize("name", TAPES)
def test_the_model_draw_is_the_legacy_draw(name):
    """Bit for bit, for an MF35-only request and one seed.

    The applier gate feeds both sides the same deltas, so until this test the
    claim "the model path is the shipped path" stopped at the draw. It also
    pins the band order and the seed ladder, which were inferred to agree.
    """
    suite, _stated_ = _stated(name)
    legacy, _info = generate_pfns_samples(suite, 8, seed=11, verbose=False)
    legacy = {key[-1]: value for key, value in legacy.items()}

    run = perturbFromModel(str(DATA / name), {35: None}, nSamples=8, seed=11,
                           dryRun=True)
    model = _modelDeltas(run)

    assert set(model) == set(legacy)
    for band in legacy:
        assert np.array_equal(model[band], legacy[band]), (
            f"band {band}: max |Δ| = "
            f"{np.max(np.abs(model[band] - legacy[band])):.3e}")


def test_an_explicit_null_tolerance_still_wins():
    """The default changes for absolute blocks; a caller's own value does not."""
    run = perturbFromModel(str(CF252), {35: None}, nSamples=4, seed=11,
                           dryRun=True, nullTol=1e-6)
    for diag in run.diagnostics.values():
        assert diag["null_tol"] == 1e-6


# ======================================================================
# Q0.3 -- the conditioning a band may receive
# ======================================================================

def test_the_automatic_plan_clips_a_spectrum_band():
    run = perturbFromModel(str(CF252), {35: None}, nSamples=1, seed=11,
                           dryRun=True)
    remedies = {step.remedy for step in run.conditioningPlan.steps}
    assert remedies <= {"clip", "none"}


@pytest.mark.parametrize("remedy", ["clip_rescale", "higham"])
def test_a_plan_that_would_break_the_sum_rule_is_refused(remedy):
    """A congruence maps 1 to D·1, and Higham degrades C·1 ~40×: both refused."""
    from kika.cov.conditioning import ConditioningPlan, PlanStep

    endf = read_endf(str(CF252))
    from kika.endf.model_adapter import decodeCovarianceSuite

    covariances, _report = decodeCovarianceSuite(endf)
    entries = collectEntries(covariances, {35: None})
    blocks, _index = assembleRequest(
        entries, domains=componentDomains(covariances, {35: None}))
    plan = ConditioningPlan(steps=tuple(
        PlanStep(key=key, remedy=remedy, reason="test") for key, _m in blocks))

    with pytest.raises(ValueError, match="sum rule"):
        perturbFromModel(str(CF252), {35: None}, nSamples=1, seed=11,
                         dryRun=True, conditioningPlan=plan)


# ======================================================================
# Q0.4 -- the run says what the legacy run says
# ======================================================================

#: Per band, from the draw. The names are the legacy run summary's.
DRAW_KEYS = ("null_tol", "sigma_clamp", "n_clamped", "clamped_fraction",
             "row_sum_residual", "normalisation_drift",
             "spectral_fidelity_median", "gate_tolerance",
             "passes_spectral_gate", "null_leakage")

#: Per band, per sample, from the applier and the ratio rule.
APPLIED_KEYS = ("max_projection_shift", "max_sum_error_before_projection",
                "max_sum_error_after_projection", "total_clipped",
                "max_clipped_mass_fraction", "total_groups_frozen",
                "max_renormalisation_error", "max_group_mass_error",
                "input_normalisation_max_abs")


def test_the_draw_reports_what_the_legacy_draw_reports():
    run = perturbFromModel(str(CF252), {35: None}, nSamples=8, seed=11,
                           dryRun=True)
    assert run.diagnostics
    for diag in run.diagnostics.values():
        missing = [key for key in DRAW_KEYS if key not in diag]
        assert not missing, missing
        assert diag["null_tol"] == NULL_TOL
        assert diag["sigma_clamp"] == SIGMA_CLAMP


def test_every_sample_reports_its_projection_budgets():
    run = perturbFromModel(str(CF252), {35: None}, nSamples=2, seed=11,
                           dryRun=True)
    for sample in run.samples:
        for component, applied in sample["applied"].items():
            missing = [key for key in APPLIED_KEYS if key not in applied]
            assert not missing, (component, missing)
            assert applied["max_sum_error_after_projection"] < 1e-12


def test_the_run_says_the_covariance_was_not_moved(tmp_path):
    perturbFromModel(str(CF252), {35: None}, nSamples=1, seed=11,
                     outputDir=tmp_path, formats=("endf-delta",))
    payload = json.loads((tmp_path / "run_metadata.json").read_text("utf-8"))
    assert payload["mf35_unchanged"] is True
    assert any("MF35" in note for note in payload["notes"])


def test_a_run_without_a_spectrum_does_not_claim_one(tmp_path):
    # NJOY cannot read the section-sliced micro-tape, so its resonance region
    # is perturbed as stated; this test is about MF35's metadata.
    perturbFromModel(str(DATA / "micro_fe56_xs_and_angular.endf"), {33: None}, nSamples=1,
                     seed=5, outputDir=tmp_path, formats=("endf-delta",),
                     resonanceRegion="evaluated")
    payload = json.loads((tmp_path / "run_metadata.json").read_text("utf-8"))
    assert "mf35_unchanged" not in payload


# ======================================================================
# Q0.5 -- the point cap reaches the applier
# ======================================================================

def test_the_outgoing_point_cap_reaches_the_applier():
    run = perturbFromModel(str(CF252), {35: None}, nSamples=1, seed=11,
                           dryRun=True, maxOutgoingPoints=50)
    dropped = sum(applied["total_steps_dropped"]
                  for applied in run.samples[0]["applied"].values())
    assert dropped > 0
    assert any("dropped" in str(event.get("message", ""))
               for event in run.log.to_dicts())
