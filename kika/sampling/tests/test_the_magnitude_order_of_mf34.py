"""MF34's L=0 -- the magnitude order -- in a request, today.

In MF4 a_0 is identically 1, so an MF34 section with L=0 is not about the shape:
L0xL0 is the variance of sigma(E) of that MT (a second estimate beside MF33, on
MF34's own grid) and L0xLl is the correlation between sigma and a_l. Pista J of
``docs/library/mf4_tabulated_perturbation_roadmap.md``.

The fixture is :mod:`l0_fixture`, written to ``tmp_path`` and used as the
``covarianceSource`` of ``micro_fe56_xs_and_angular.endf``; no committed or
evaluated tape carries an L=0 section small enough to read in a test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import l0_fixture
from kika.endf import read_endf
from kika.endf.model_adapter import decodeCovarianceSuite
from kika.sampling.joint_blocks import ComponentKey, collectEntries, samplingGroups
from kika.sampling.model_perturbation import _touchedFiles, perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
FE56 = DATA / "micro_fe56_xs_and_angular.endf"
ZA, MT = l0_fixture.ZA, l0_fixture.MT

pytestmark = pytest.mark.skipif(not FE56.is_file(), reason=f"{FE56} missing")


@pytest.fixture(scope="module")
def covariance(tmp_path_factory):
    return l0_fixture.write(tmp_path_factory.mktemp("l0") / "fe56_l0.endf")


@pytest.fixture(scope="module")
def covarianceSuite(covariance):
    suite, _report = decodeCovarianceSuite(read_endf(str(covariance)))
    return suite


def _run(covariance, request, n=2, seed=3):
    return perturbFromModel(FE56, request, n, seed=seed, dryRun=True,
                            covarianceSource=covariance)


def test_the_fixture_reads_back_as_written(covarianceSuite):
    """Every section the builder wrote, with the numbers it wrote."""
    entries = collectEntries(covarianceSuite, {33: None, 34: None})
    found = {(row.mf, row.index, col.mf, col.index): matrix
             for row, col, matrix, *_ in entries}
    assert set(found) == {(33, 0, 33, 0), (34, 0, 34, 0), (34, 0, 34, 1),
                          (34, 0, 34, 2), (34, 1, 34, 1), (34, 1, 34, 2),
                          (34, 2, 34, 2)}
    np.testing.assert_allclose(found[(33, 0, 33, 0)], l0_fixture.mf33Block(),
                               rtol=1e-6)
    for l, l1 in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]:
        np.testing.assert_allclose(found[(34, l, 34, l1)],
                                   l0_fixture.block(l, l1), rtol=1e-6)


# ----------------------------------------------------------------------
# J0: what happens today, pinned before it is changed
# ----------------------------------------------------------------------

def test_an_angular_request_perturbs_the_cross_section_today(covariance):
    """``{34: None}`` lets L=0 in, and it lands on sigma -- MF3 is rewritten.

    A request that names only the angular distribution changes the cross
    section, and nothing in the run says so.
    """
    run = _run(covariance, {34: None})
    sample = run.samples[0]
    components = sample["set"].components()
    assert ComponentKey(ZA, 34, MT, 0) in components
    assert _touchedFiles(sample["set"], tuple(sample["applied"])) == {3: [MT], 4: [MT]}
    assert run.notes == []


def test_both_files_unordered_are_refused(covariance):
    """MF33 and MF34's L=0 claim one sigma, and the set refuses both."""
    with pytest.raises(ValueError, match="Ask for one of them"):
        _run(covariance, {33: None, 34: None})


def test_orders_from_one_drop_the_sigma_shape_cross_term(covariance,
                                                         covarianceSuite):
    """``{33, 34: L>=1}`` draws two independent groups: L0xLl is not used.

    The cross term is in the file and nowhere in the draw; the run does not say
    so. That is the state J2/J3 exist to change -- until then, a request for
    both quantities samples them as uncorrelated.
    """
    request = {33: None, 34: {"index": [1, 2]}}
    groups = samplingGroups(collectEntries(covarianceSuite, request))
    assert groups == [[ComponentKey(ZA, 33, MT, 0)],
                      [ComponentKey(ZA, 34, MT, 1), ComponentKey(ZA, 34, MT, 2)]]
    run = _run(covariance, request)
    assert {c.index for c in run.samples[0]["set"].components() if c.mf == 34} == {1, 2}
