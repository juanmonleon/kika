"""PD-9: the run keeps MF35 as stated and reports C10 beside it.

Juan's decision (2026-10-08): production draws the library's MF35 as it is --
no constraint, no clipping to a band, no rejection of replicas -- and every
run that perturbs a fission spectrum reports what that MF35 states for the
mean outgoing energy and the tail fractions, against independent knowledge
where the roadmap has read some (``kika.sampling.pfns_moments``). The GLS
constraint through P5 is pending and not built; the record says so.

The numbers gated on JEFF-4.0 U-235 are the ones the C10 first pass measured
on 2026-10-07 with ``kika_dev/sampling/checks/pfns_c10_stated_moments.py``
(band 0: <E'> 2.0007 MeV, sigma 1.998 %, F(>1 MeV) 0.73 %, F(>3 MeV) 3.91 %),
now computed by the run on the model node.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kika.nuclear_data.model.enums import Interpolation
from kika.nuclear_data.model.functions import XYs1d, XYs2d
from kika.sampling.model_perturbation import perturbFromModel
from kika.sampling.pfns_moments import (C10_CONSTRAINT, spectrumFunctionals,
                                        statedSpectrumMoments)

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"


def test_the_functionals_are_exact_on_a_lin_lin_table():
    """``<E'>`` of a triangle on [0, 2] peaking at 1 is 1; F(>1) is 1/2."""
    xs = np.array([0.0, 1.0, 2.0]) * 1e6
    ys = np.array([0.0, 1.0, 0.0]) / 1e6
    edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0]) * 1e6
    out = spectrumFunctionals(xs, ys, 2, edges, thresholds=(1e6,))
    assert out["P"].sum() == pytest.approx(1.0, rel=1e-15)
    assert out["meanEnergy"][1] == pytest.approx(1e6, rel=1e-14)
    assert out["F>1MeV"][1] == pytest.approx(0.5, rel=1e-14)
    # Group 0 is y = x on [0, 0.5 MeV]: its mean is 2/3 of the way up.
    assert out["meanEnergy"][0][0] == pytest.approx(2 / 3 * 0.5e6, rel=1e-14)


def test_the_stated_sigma_is_the_sandwich_and_the_drawn_one_follows_it():
    """``sqrt(w^T C w)`` against the spread of ``w . delta`` over the draws."""
    xs = np.linspace(0.0, 4.0, 41) * 1e6
    ys = xs * np.exp(-xs / 1e6)
    ys /= np.trapezoid(ys, xs)
    child = XYs1d(xs=xs, ys=ys, interpolation=Interpolation.linlin)
    child.outerDomainValue = 1.0
    form = XYs2d(function1ds=[child])
    edges = np.linspace(0.0, 4.0, 9) * 1e6
    n = edges.size - 1
    A = np.random.default_rng(0).normal(size=(n, n)) * 1e-3
    C = A @ A.T
    centre = np.eye(n) - np.ones((n, n)) / n              # rows sum to zero
    C = centre @ C @ centre
    draws = np.random.default_rng(1).multivariate_normal(np.zeros(n), C, 20000)

    record = statedSpectrumMoments(form, {0: (1.0, 1.0)}, {0: edges}, {0: C},
                                   drawn={0: draws})[0]
    stated = record["sigma(meanEnergy)_stated_rel"]
    drawn = record["sigma(meanEnergy)_drawn_rel"]
    assert stated > 0
    assert drawn == pytest.approx(stated, rel=3 / np.sqrt(2 * 20000))
    assert record["constraint"] == C10_CONSTRAINT
    assert "P5" in C10_CONSTRAINT and "not built" in C10_CONSTRAINT


def test_every_run_writes_c10_beside_its_samples(tmp_path):
    """The micro Cf-252 tape: four bands, four records, in the metadata."""
    run = perturbFromModel(str(DATA / "micro_cf252_pfns.endf"), {35: None},
                           nSamples=8, seed=11, outputDir=tmp_path)
    assert set(run.spectrumMoments) == {18}
    assert len(run.spectrumMoments[18]) == 4
    payload = json.loads((tmp_path / "run_metadata.json").read_text("utf-8"))
    moments = payload["mf35StatedMoments"]["18"]
    assert set(moments) == {"0", "1", "2", "3"}
    for record in moments.values():
        assert record["sigma(meanEnergy)_stated_rel"] > 0
        assert "sigma(meanEnergy)_drawn_rel" in record
        assert record["constraint"] == C10_CONSTRAINT
        assert "reference" not in record        # no reference read for Cf-252
    assert not any(note.startswith("C10") for note in run.notes)


def test_jeff40_u235_reports_its_thermal_sigma_against_the_standard(u235_tape):
    """2.00 % stated against the IAEA 0.45 %: x4.4, said in the notes."""
    run = perturbFromModel(str(u235_tape), {35: {"index": [0]}}, nSamples=2,
                           seed=11, dryRun=True)
    record = run.spectrumMoments[18][0]
    assert record["probe"] == pytest.approx(1e-5)
    assert record["meanEnergy"] == pytest.approx(2.0007e6, rel=1e-4)
    assert record["sigma(meanEnergy)_stated_rel"] == pytest.approx(0.01998, rel=2e-3)
    assert record["sigma(F>1MeV)_stated_rel"] == pytest.approx(0.0073, rel=1e-2)
    assert record["sigma(F>3MeV)_stated_rel"] == pytest.approx(0.0391, rel=1e-2)
    assert record["reference"]["stated_over_reference"] == pytest.approx(4.44, rel=1e-2)
    assert any(note.startswith("C10") and "x4.4" in note for note in run.notes)
