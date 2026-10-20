"""J3: a request for MF33 *and* MF34 draws the sigma <-> a_l cross term the file states.

MF34's L0xLl sections are the covariance between the cross section and the a_l.
Decided 2026-10-04: when cross section and angular distribution are asked for together,
that term is drawn **by default**, in one matrix with MF33's sigma and the
L>=1 shape. The design:

* sigma is **one** component, MF33's; the L=0 side of every L0xLl section is
  re-keyed to it, and L0xL0 is not used;
* with the cross term null (ENDF/B-VIII.x's placeholders) nothing changes, bit
  for bit;
* in log space the joint is ``log1p(Σ)``; with ``{33: "log", 34: "linear"}``
  the block is mixed and the cross block is drawn untransformed, which is exact
  in second moments (Stein's lemma).

The covariance is :mod:`l0_fixture`, whose every number is known: MF33 5/4/6 %,
L0 3 %, L1 10/8/12 %, L2 15 %, order correlations 0.4 (L0-L1) and 0.2 (L0-L2).
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pytest

import l0_fixture
from kika.endf import read_endf
from kika.endf.model_adapter import decodeCovarianceSuite
from kika.sampling.joint_blocks import (ComponentKey, assembleRequest,
                                        collectEntries, resolveMagnitudeOrder)
from kika.sampling.model_perturbation import (_drawEverything, _touchedFiles,
                                              perturbFromModel)
from test_the_magnitude_order_of_mf34 import FE56

perturbFromModel = partial(perturbFromModel, resonanceRegion="evaluated")

ZA, MT = l0_fixture.ZA, l0_fixture.MT
SIGMA = ComponentKey(ZA, 33, MT, 0)
A1, A2 = ComponentKey(ZA, 34, MT, 1), ComponentKey(ZA, 34, MT, 2)
REQUEST = {33: None, 34: None}

pytestmark = pytest.mark.skipif(not FE56.is_file(), reason=f"{FE56} missing")


@pytest.fixture(scope="module")
def covariance(tmp_path_factory):
    return l0_fixture.write(tmp_path_factory.mktemp("l0") / "fe56_l0.endf")


@pytest.fixture(scope="module")
def nullCross(tmp_path_factory):
    """The same tape with the L0xLl sections written as zeros."""
    saved = l0_fixture.C_L.copy()
    try:
        l0_fixture.C_L[0, 1:] = l0_fixture.C_L[1:, 0] = 0.0
        return l0_fixture.write(tmp_path_factory.mktemp("l0null") / "fe56_l0_null.endf")
    finally:
        l0_fixture.C_L[...] = saved


def _suite(path):
    suite, _report = decodeCovarianceSuite(read_endf(str(path)))
    return suite


def _assembled(path, request=REQUEST):
    suite = _suite(path)
    resolved, notes = resolveMagnitudeOrder(suite, request)
    blocks, index = assembleRequest(collectEntries(suite, resolved))
    return blocks, index, notes


def _expected():
    """The joint the file states, sigma's variance from MF33: rows sigma, a1, a2."""
    b = l0_fixture.block
    return np.block([[l0_fixture.mf33Block(), b(0, 1), b(0, 2)],
                     [b(0, 1).T, b(1, 1), b(1, 2)],
                     [b(0, 2).T, b(1, 2).T, b(2, 2)]])


def test_sigma_and_the_shape_are_one_block_with_the_files_cross_term(covariance):
    blocks, index, notes = _assembled(covariance)
    ((key, matrix),) = blocks
    assert index[key]["components"] == [SIGMA, A1, A2]
    assert index[key]["quantities"] == ["angularDistribution", "crossSection"]
    # sigma x sigma is MF33's (5/4/6 %), not L0xL0's (3 %): option (a) of J2.
    np.testing.assert_allclose(matrix, _expected(), rtol=1e-6)
    (note,) = notes
    assert "is drawn with them in one matrix" in note
    # L0xL0 states 3 % where MF33 states 5/4/6 %: the run says how far apart.
    assert "up to 50% away from MF33's" in note


def test_the_run_draws_it_and_says_so(covariance):
    run = perturbFromModel(FE56, REQUEST, 2, seed=3, dryRun=True,
                           covarianceSource=covariance)
    ((key, meta),) = run.index.items()
    assert meta["components"] == [SIGMA, A1, A2]
    sample = run.samples[0]
    assert sample["set"].components() == (SIGMA, A1, A2)
    assert _touchedFiles(sample["set"], tuple(sample["applied"])) == {3: [1, MT], 4: [MT]}
    (note,) = [n for n in run.notes if "states L=0" in n]
    assert "is drawn with them in one matrix" in note


@pytest.mark.parametrize("space", ["log", {33: "log", 34: "linear"}],
                         ids=["log", "mixed"])
def test_a_large_draw_reproduces_the_joint_covariance(covariance, space):
    """Cov(f) = Σ, cross block included, within Monte Carlo error.

    In log space Cov(f_i, f_j) = exp(M_ij) - 1 with M = log1p(Σ), so Σ comes
    back exactly in expectation. In the mixed block the a_l are drawn linearly
    and the cross block untransformed -- Stein's lemma makes that exact too.
    """
    blocks, index, _notes = _assembled(covariance)
    n = 40_000
    samples, diagnostics = _drawEverything(
        blocks, index, n, seed=7, space=space, decompositionMethod="eigen",
        samplingMethod="random", psdMethod="none", nullTol=None, logger=None)
    ((key, factors),) = samples.items()
    want = _expected()
    got = np.cov(factors, rowvar=False)
    tolerance = 5 * np.sqrt((np.outer(np.diag(want), np.diag(want)) + want ** 2) / n)
    assert np.all(np.abs(got - want) <= tolerance), np.max(np.abs(got - want) / tolerance)
    np.testing.assert_allclose(factors.mean(axis=0), 1.0, atol=5e-3)
    if isinstance(space, dict):
        assert diagnostics[key]["space"] == "mixed"
        # sigma's rows are lognormal (positive); the a_l rows are not.
        assert np.all(factors[:, :3] > 0)


def test_a_mixed_run_draws_the_a_l_linearly(covariance):
    run = perturbFromModel(FE56, REQUEST, 2, seed=3, dryRun=True,
                           covarianceSource=covariance,
                           space={33: "log", 34: "linear"})
    (diagnostic,) = run.diagnostics.values()
    assert diagnostic["space"] == "mixed"
    assert diagnostic["log_rows"] == 3


def test_a_null_cross_term_leaves_the_run_as_it_was(nullCross):
    """Bit for bit the run that names the shape orders and never sees L=0."""
    _blocks, index, notes = _assembled(nullCross)
    assert [meta["components"] for meta in index.values()] == [[SIGMA], [A1, A2]]
    (note,) = notes
    assert "nothing is lost" in note

    default = perturbFromModel(FE56, REQUEST, 4, seed=5, dryRun=True,
                               covarianceSource=nullCross)
    named = perturbFromModel(FE56, {33: None, 34: {"index": [1, 2]}}, 4, seed=5,
                             dryRun=True, covarianceSource=nullCross)
    for a, b in zip(default.samples, named.samples):
        assert a["set"].components() == b["set"].components()
        for component in a["set"].components():
            np.testing.assert_array_equal(a["set"].factors[component],
                                          b["set"].factors[component])


def _onto(fine, coarse):
    """``P[i, j] = 1`` where bin *i* of *fine* lies inside bin *j* of *coarse*."""
    mids = 0.5 * (fine[1:] + fine[:-1])
    return (np.searchsorted(coarse, mids)[:, None] - 1
            == np.arange(coarse.size - 1)[None, :]).astype(float)


def test_sigma_on_mf33s_own_grid_takes_the_union_with_the_l0_side(tmp_path):
    """MF33 and MF34 on different grids, as an evaluation's usually are.

    sigma is one component on the union of MF33's grid and the L=0 side's;
    sigma x sigma is MF33's and the cross block the file's L0xLl, each read
    piecewise constant onto it. The a_l keep MF34's grid.
    """
    grid33 = np.array([8.5e5, 2.0e6, 4.0e6])
    mf33 = np.outer([0.05, 0.06], [0.05, 0.06]) * np.array([[1.0, 0.5], [0.5, 1.0]])
    path = l0_fixture.write(tmp_path / "fe56_l0_grid33.endf", mf33=(grid33, mf33))
    blocks, index, notes = _assembled(path)
    ((key, matrix),) = blocks
    meta = index[key]
    assert meta["components"] == [SIGMA, A1, A2]
    union = np.array([8.5e5, 1.5e6, 2.0e6, 2.5e6, 4.0e6])
    np.testing.assert_array_equal(meta["grids"][SIGMA], union)
    np.testing.assert_array_equal(meta["grids"][A1], l0_fixture.GRID)
    stride = meta["stride"]

    def sub(i, j):
        rows, cols = (meta["widths"][c] for c in (meta["components"][i],
                                                  meta["components"][j]))
        return np.asarray(matrix)[i * stride:i * stride + rows,
                                  j * stride:j * stride + cols]

    to33, toL0 = _onto(union, grid33), _onto(union, l0_fixture.GRID)
    b = l0_fixture.block
    np.testing.assert_allclose(sub(0, 0), to33 @ mf33 @ to33.T, rtol=1e-6)
    np.testing.assert_allclose(sub(0, 1), toL0 @ b(0, 1), rtol=1e-6)
    np.testing.assert_allclose(sub(0, 2), toL0 @ b(0, 2), rtol=1e-6)
    np.testing.assert_allclose(sub(1, 2), b(1, 2), rtol=1e-6)
    (note,) = notes
    assert "is drawn with them in one matrix" in note

    run = perturbFromModel(FE56, REQUEST, 2, seed=3, dryRun=True,
                           covarianceSource=path)
    assert run.samples[0]["set"].components() == (SIGMA, A1, A2)


def test_a_null_l0xl0_is_not_compared_with_mf33():
    """This project's ``_a0cross`` tapes write L0xL0 as a one-bin zero block."""
    from kika.sampling.joint_blocks import _sigmaDiscrepancy

    grid = l0_fixture.GRID
    sigma = [((ZA, MT), (ZA, MT), l0_fixture.mf33Block(), grid, grid)]
    null = [((ZA, MT, 0), (ZA, MT, 0), np.zeros((1, 1)), grid[[0, -1]], grid[[0, -1]])]
    stated = [((ZA, MT, 0), (ZA, MT, 0), l0_fixture.block(0, 0), grid, grid)]
    assert _sigmaDiscrepancy(null, sigma, MT) is None
    assert _sigmaDiscrepancy(stated, sigma, MT) == pytest.approx(0.5)
    assert _sigmaDiscrepancy(stated, [], MT) is None


def test_named_orders_are_left_alone(covariance):
    """Orders named explicitly are the caller's choice: no cross term added."""
    _blocks, index, notes = _assembled(covariance, {33: None, 34: {"index": [1, 2]}})
    assert [meta["components"] for meta in index.values()] == [[SIGMA], [A1, A2]]
    assert notes == []
