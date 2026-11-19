"""A perturbation run leaves ``report.html`` however it ends.

A run that finishes, one that fails and one that is stopped each write their
events and the report built from them, so the report a person opens is never
missing when it is needed most. The report is built from the events alone, so
the file read back gives the same page.
"""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path

import pytest

from kika.sampling.model_perturbation import perturbFromModel
from kika.sampling.run_log import RunCancelled, RunLog
from kika.sampling.run_report import PerturbationReport

perturbFromModel = partial(perturbFromModel, resonanceRegion="evaluated")

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = str(DATA / "micro_fe56_xs_and_angular.endf")
REQUEST = {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}


def _events(directory):
    return [json.loads(line) for line in
            (directory / "run.log.jsonl").read_text(encoding="utf-8").splitlines()]


def test_a_finished_run_reports_its_samples(tmp_path):
    run = perturbFromModel(TAPE, REQUEST, 2, seed=5, outputDir=tmp_path)
    report = PerturbationReport.read(tmp_path)
    data = report.to_dict()
    assert report.status in ("pass", "warn")
    assert report.headline.startswith("Completed: 2 samples written")
    assert [s["status"] for s in data["stages"] if s["id"] == "files"] == ["pass"]
    assert {c["title"]: c["status"] for c in data["checks"]}[
        "Every sample was written"] == "pass"
    assert len(data["samples"]["rows"]) == 2
    assert data["header"]["file"] == Path(TAPE).name
    assert data["header"]["kika"]
    assert run.files["report"] == tmp_path / "report.html"


def test_a_failed_run_leaves_its_report_and_says_where(tmp_path):
    with pytest.raises(ValueError, match="states no section"):
        perturbFromModel(TAPE, {33: [999]}, 1, outputDir=tmp_path)
    assert (tmp_path / "report.html").is_file()
    errors = [e for e in _events(tmp_path) if e["kind"] == "error"]
    assert len(errors) == 1 and errors[0]["payload"]["stage"] == "assembled"
    report = PerturbationReport.read(tmp_path)
    assert report.status == "fail"
    assert report.headline.startswith("Failed at covariance")
    assert "states no section" in report.to_html()


def test_a_failure_outside_a_timed_stage_is_recorded_once(tmp_path):
    # Nothing the file states is asked for: refused while the request is read,
    # outside any timed block, so the wrapper records the error itself.
    with pytest.raises(ValueError, match="nothing to perturb"):
        perturbFromModel(TAPE, {35: None}, 1, outputDir=tmp_path, onMissing="skip")
    errors = [e for e in _events(tmp_path) if e["kind"] == "error"]
    assert len(errors) == 1
    assert errors[0]["payload"]["stage"] == "request"
    assert errors[0]["payload"]["exception"] == "ValueError"


def test_a_stopped_run_is_cancelled_not_failed(tmp_path):
    class StopAtFirstSample(RunLog):
        def event(self, kind, message, **kwargs):
            recorded = super().event(kind, message, **kwargs)
            if kind == "applied" and kwargs.get("sample") == 0:
                raise RunCancelled("stopped by the user")
            return recorded

    with pytest.raises(RunCancelled):
        perturbFromModel(TAPE, REQUEST, 3, seed=5, outputDir=tmp_path,
                         runLog=StopAtFirstSample())
    kinds = [e["kind"] for e in _events(tmp_path)]
    assert "cancelled" in kinds and "error" not in kinds
    report = PerturbationReport.read(tmp_path)
    assert report.headline.startswith("Stopped:")


def test_a_call_refused_before_it_starts_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match="unknown emitter"):
        perturbFromModel(TAPE, REQUEST, 1, formats=("no-such-format",), outputDir=tmp_path)
    assert not any(tmp_path.iterdir())
