"""MT2's cross section and angular distribution drawn in one run, on a real tape.

``micro_u238_mf34_l0.endf`` is U-238 of ENDF/B-VIII.1 cut to MF3, MF4, MF33 and
MF34 of MT2 (``kika-workspace/kika_dev/sampling/checks/build_micro_u238_mf34_l0.py``).
It is one of two evaluations to hand whose MF34 states L=0 (U-235 is the other),
and that L=0 is a placeholder: L0xL0 is one bin of zero variance and L0xL1,
L0xL2 are noise at 2e-19. So the file gives sigma's variance through MF33 and
the shape's through L=1,2, with no correlation between them -- and a joint
request has to reproduce exactly that.

Before ``resolveMagnitudeOrder`` this request was refused: MF33 and the L=0
placeholder both claimed sigma.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.model_adapter import decodeCovarianceSuite
from kika.sampling.joint_blocks import (ComponentKey, assembleRequest,
                                        collectEntries, resolveMagnitudeOrder)
from kika.sampling.model_perturbation import _touchedFiles, perturbFromModel

TAPE = (Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
        / "micro_u238_mf34_l0.endf")
ZA, MT = 92238, 2
REQUEST = {33: [MT], 34: [MT]}
N = 1024

pytestmark = pytest.mark.skipif(not TAPE.is_file(), reason=f"{TAPE} missing")


def _draw(samplingMethod):
    return perturbFromModel(TAPE, REQUEST, N, seed=11, dryRun=True,
                            samplingMethod=samplingMethod)


@pytest.fixture(scope="module")
def run():
    return _draw("random")


@pytest.fixture(scope="module")
def stated():
    """Per component, the variance the file states for each of its bins."""
    suite, _report = decodeCovarianceSuite(read_endf(str(TAPE)))
    request, _notes = resolveMagnitudeOrder(suite, REQUEST)
    blocks, index = assembleRequest(collectEntries(suite, request))
    out = {}
    for key, matrix in blocks:
        meta = index[key]
        for position, component in enumerate(meta["components"]):
            start = position * meta["stride"]
            width = meta["widths"][component]
            out[component] = np.diag(matrix)[start:start + width]
    return out


def _factors(run):
    components = run.samples[0]["set"].components()
    return {c: np.array([s["set"].factors[c] for s in run.samples])
            for c in components}


def _correlation(a, b):
    """Pairwise correlation of the columns that move. A constant column is
    tested with max == min, never a sigma threshold (memory
    *constant-column-test-is-max-eq-min*)."""
    a = a[:, a.max(0) != a.min(0)]
    b = b[:, b.max(0) != b.min(0)]
    za = (a - a.mean(0)) / a.std(0)
    zb = (b - b.mean(0)) / b.std(0)
    return za.T @ zb / a.shape[0]


def test_the_request_draws_sigma_from_mf33_and_the_shape_from_l1_l2(run):
    sample = run.samples[0]
    assert sample["set"].components() == (ComponentKey(ZA, 33, MT, 0),
                                           ComponentKey(ZA, 34, MT, 1),
                                           ComponentKey(ZA, 34, MT, 2))
    assert _touchedFiles(sample["set"], tuple(sample["applied"])) == {3: [MT], 4: [MT]}
    (note,) = [n for n in run.notes if "states L=0" in n]
    assert "null" in note and "nothing is lost" in note
    # MF4/MT2 is LTT=3 with tables from 17 MeV and the MF34 runs to 30 MeV, so
    # the tabulated half is perturbed too -- decision D1, here with the
    # evaluation's own, well-formed covariance.
    for component, info in sample["applied"].items():
        if component.mf == 34:
            assert info["tables"]["max_integral_change"] < 1e-3


def test_each_component_carries_the_variance_the_file_states(run, stated):
    """Per bin, sample variance over stated variance, at N=1024.

    The sampling error of a variance is sqrt(2/N) = 4.4 %; the bound is three
    of those.
    """
    for component, values in _factors(run).items():
        variance = stated[component]
        live = variance > 1e-10
        ratio = values.var(axis=0, ddof=1)[live] / variance[live]
        assert np.all(np.abs(ratio - 1.0) < 3 * np.sqrt(2.0 / N)), (
            f"{component.describe()}: variance ratio "
            f"{ratio.min():.3f}..{ratio.max():.3f}")
        # perturbFromModel's default nullTol=None draws the null directions
        # too, so a bin of zero stated variance carries decomposition debris:
        # 4e-9 measured on the first bin of L=1 and L=2 here.
        assert np.all(np.abs(values[:, ~live] - 1.0) < 1e-6), (
            f"{component.describe()}: a bin with no stated variance moved")


def test_sigma_and_shape_are_drawn_independently(run):
    """The file states no sigma <-> a_l correlation, so the draw has none.

    Over the 75 x 44 bin pairs the largest |corr| of independent columns at
    N=1024 is about 4/sqrt(N) = 0.125; the 95th percentile about 2/sqrt(N).
    """
    factors = _factors(run)
    sigma = factors[ComponentKey(ZA, 33, MT, 0)]
    for order in (1, 2):
        corr = np.abs(_correlation(sigma, factors[ComponentKey(ZA, 34, MT, order)]))
        assert np.quantile(corr, 0.95) < 2.5 / np.sqrt(N), (
            f"L={order}: 95th percentile |corr| {np.quantile(corr, 0.95):.3f}")
        assert corr.max() < 5.0 / np.sqrt(N), f"L={order}: max |corr| {corr.max():.3f}"


@pytest.mark.xfail(strict=True, reason=(
    "draw_samples gives each block its own Sobol sequence and pairs blocks by "
    "replica index (core.py:205), so two blocks the file leaves independent "
    "come out correlated -- 95th percentile |corr| 0.20 (L=1) and 0.39 (L=2) "
    "measured 2026-10-05, against 0.06 for random sampling. The same defect as "
    "memory sobol-blocks-collide-across-materials, inside one call. Not fixed "
    "here: core.py is shared with the frozen v2 pipeline."))
def test_sigma_and_shape_are_independent_under_sobol_too():
    factors = _factors(_draw("sobol"))
    sigma = factors[ComponentKey(ZA, 33, MT, 0)]
    for order in (1, 2):
        corr = np.abs(_correlation(sigma, factors[ComponentKey(ZA, 34, MT, order)]))
        assert np.quantile(corr, 0.95) < 2.5 / np.sqrt(N)
