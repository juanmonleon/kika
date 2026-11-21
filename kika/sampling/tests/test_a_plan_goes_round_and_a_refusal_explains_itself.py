"""The plan a run writes can be handed back, and a refused pre-flight says why.

A relative covariance of -1 or below has no logarithm, so a log-normal draw
cannot take it and no repair applies. The micro-tape states none, so one entry
of its MF33 block is set to -2 on the way out of the assembly.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest

from kika.cov.conditioning import ConditioningPlan
from kika.sampling import model_perturbation
from kika.sampling.model_perturbation import perturbFromModel
from kika.sampling.run_report import PerturbationReport

perturbFromModel = partial(perturbFromModel, resonanceRegion="evaluated")

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = str(DATA / "micro_fe56_xs_and_angular.endf")
REQUEST = {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}
SEED = 20261121


def _factors(run):
    return {c: run.samples[0]["set"].factors[c] for c in run.samples[0]["set"].components()}


@pytest.fixture
def below_minus_one(monkeypatch):
    """Rows 0 and 1 of the first MF33 block get a relative covariance of -2."""
    assemble = model_perturbation.assembleRequest
    placed = {}

    def assembleWithANegativeEntry(*args, **kwargs):
        blocks, index = assemble(*args, **kwargs)
        out = []
        for key, matrix in blocks:
            meta = index[key]
            if not placed and all(c.mf == 33 for c in meta["components"]):
                matrix = np.array(matrix, dtype=float)
                matrix[0, 1] = matrix[1, 0] = -2.0
                placed["component"] = meta["components"][0]
            out.append((key, matrix))
        return out, index

    monkeypatch.setattr(model_perturbation, "assembleRequest", assembleWithANegativeEntry)
    return placed


def test_a_refusal_names_the_entry_and_leaves_the_plan(below_minus_one, tmp_path):
    with pytest.raises(ValueError) as refused:
        perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path)
    mt = below_minus_one["component"].mt
    message = str(refused.value)
    assert f"MF33 MT{mt}" in message and "relative covariance -2" in message
    assert "space={33: 'linear'}" in message and f"leave MT{mt} out" in message
    assert "conditioning_plan.json" in message

    plan = ConditioningPlan.read(tmp_path / "conditioning_plan.json")
    assert plan.steps and plan.notes, "one step per block and the refusal in the notes"

    data = PerturbationReport.read(tmp_path).to_dict()
    failure = data["failure"]
    assert failure["stage"] == "inspected"
    assert "relative covariance -2" in failure["blocking"][0]["message"]
    assert not any("non-finite" in b["message"] for b in failure["blocking"]
                   if b["subject"] == failure["blocking"][0]["subject"])
    fixes = {f["title"]: f["text"] for f in failure["fixes"]}
    assert f"MT{mt}" in fixes["Leave those reactions out"]
    assert 'space={33: "linear"}' in fixes["Draw that quantity in linear space"]
    assert "conditioning_plan.json" in failure["lead"]
    assert "refusal in its notes" in {f["name"]: f["what"] for f in data["files"]}[
        "conditioning_plan.json"]


def test_without_a_folder_the_refusal_points_to_the_inspection(below_minus_one):
    with pytest.raises(ValueError, match="recommended_plan"):
        perturbFromModel(TAPE, REQUEST, 1, seed=SEED)


@pytest.mark.parametrize("handBack", ["path", "str", "dict"])
def test_the_written_plan_goes_back_in_as_a_path_or_a_dict(tmp_path, handBack):
    first = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path / "first")
    written = tmp_path / "first" / "conditioning_plan.json"
    plan = {"path": written, "str": str(written),
            "dict": ConditioningPlan.read(written).to_dict()}[handBack]

    again = perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path / "again",
                             conditioningPlan=plan)
    assert again.conditioningMode == "explicit"
    for component, factors in _factors(first).items():
        np.testing.assert_array_equal(factors, _factors(again)[component])
    drawing = PerturbationReport.read(tmp_path / "again").to_dict()["drawing"]
    assert any("taken from the plan passed to the run" in line for line in drawing)


def test_a_run_without_repairs_says_so_in_the_report(tmp_path):
    perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path, conditioningPlan=False)
    drawing = PerturbationReport.read(tmp_path).to_dict()["drawing"]
    assert any("No repairs to the covariance" in line for line in drawing)


def test_anything_else_as_a_plan_is_refused(tmp_path):
    with pytest.raises(TypeError, match="the path of its JSON"):
        perturbFromModel(TAPE, REQUEST, 1, seed=SEED, outputDir=tmp_path, conditioningPlan=3)


def test_a_plan_file_that_is_not_json_is_refused(tmp_path):
    bad = tmp_path / "plan.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a conditioning plan"):
        perturbFromModel(TAPE, REQUEST, 1, seed=SEED, conditioningPlan=bad)
