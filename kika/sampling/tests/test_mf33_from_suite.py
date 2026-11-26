"""MF33/MF31 assembled from the model equals the tape path, whichever format filled the model."""
from __future__ import annotations

import types
import warnings
from pathlib import Path

import numpy as np
import pytest

from kika.cov.group_arithmetic import (
    CovarianceSumNotResolved,
    MixedKindsNeedCrossSections,
    collapse_form,
    short_range_variance,
)
from kika.endf.read_endf import read_endf
from kika.sampling.mf33_sampling import (
    build_mf33_covariance,
    build_mf33_covariance_from_suite,
)

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
GNDS_COV = Path(__file__).resolve().parents[2] / "gnds" / "tests" / "data" / "Covariances"


def _sigma(mts):
    """A smooth, MT-dependent σ(E) for every MT, so absolute and relative blocks can meet."""
    e = np.geomspace(1e-5, 2.1e7, 400)
    return {mt: types.SimpleNamespace(energies=e, values=1.0 + 0.01 * mt + np.log1p(e) * 1e-3)
            for mt in mts}


def _all_mts():
    return list(range(1, 900))


def _same(a, b):
    A, B = np.nan_to_num(a.covariance_matrix), np.nan_to_num(b.covariance_matrix)
    assert A.shape == B.shape
    np.testing.assert_array_equal(A, B)


def _model_suite(endf, xs):
    from kika.endf.model_adapter.covariances import decodeCovarianceSuite

    endf.pendf = xs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suite, _ = decodeCovarianceSuite(endf, checks=False)
    return suite


@pytest.mark.parametrize("tape", [
    "micro_si28_covcheck.endf",
    "micro_fe56_jendl_covcheck.endf",
    "micro_f19_mixed_mf33.endf",
    "micro_fe56_mf33.endf",
])
def test_a_suite_decoded_from_a_tape_assembles_as_the_tape_does(tape):
    endf = read_endf(str(DATA / tape))
    xs = _sigma(_all_mts())
    mf33 = endf.mf[33]
    suite = _model_suite(endf, xs)

    fromSuite, _, present, blocks = build_mf33_covariance_from_suite(suite, None, xs)
    assert not blocks.declined
    fromTape, _, _ = build_mf33_covariance(mf33, present, xs)
    _same(fromTape, fromSuite)


def test_the_nubar_covariance_assembles_as_the_tape_does():
    from kika.endf.model_adapter.covariances import nubarShims

    endf = read_endf(str(DATA / "micro_u235_nubar.endf"))
    xs = nubarShims(endf.mf.get(1))
    assert xs, "the fixture carries MF1 nu-bar"
    mf31 = endf.mf[31]
    suite = _model_suite(endf, xs)

    fromSuite, _, present, _ = build_mf33_covariance_from_suite(suite, None, xs, mf=31)
    fromTape, _, _ = build_mf33_covariance(mf31, present, xs)
    _same(fromTape, fromSuite)


def test_a_gnds_mixed_with_a_short_range_term_and_a_sum_assembles_as_its_endf_records():
    """NNDC's F-19: ``mixed`` of matrices with LB=8 terms, and a stated ``sum``.

    The oracle is the same suite written back to ENDF records component by
    component (FUDGE's rule) and assembled on the tape path.
    """
    from kika.endf.model_adapter.covariances import _faithfulSection
    from kika.gnds.covariances import readCovarianceSuite
    from kika.gnds.xpath import Document
    from kika.nuclear_data.model.conversion import ConversionReport

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suite, _ = readCovarianceSuite(Document.parse(GNDS_COV / "n-009_F_019.endf.gnds-covar.xml"))
    forms = [s.form for s in suite.covarianceSections]
    assert any(type(f).__name__ == "Mixed" for f in forms)
    assert any("ShortRange" in type(c).__name__
               for f in forms for c in getattr(f, "components", []) or [])

    xs = _sigma(_all_mts())
    rows = {}
    for s in suite.covarianceSections:
        if s.rowData is not None and s.rowData.ENDF_MF == 33:
            rows.setdefault(s.rowData.ENDF_MT, []).append(s)
    report = ConversionReport()
    sections = {mt: _faithfulSection(ss, 33, mt, 9019.0, 18.835, 925, report)
                for mt, ss in rows.items()}
    mts = sorted(sections)

    fromSuite, _, present, blocks = build_mf33_covariance_from_suite(suite, mts, xs)
    assert present == mts and not blocks.declined
    fromRecords, _, _ = build_mf33_covariance(types.SimpleNamespace(sections=sections), mts, xs)
    _same(fromRecords, fromSuite)


def _matrix(values, grid, relative=True):
    from kika.nuclear_data.model import CovarianceMatrix

    return CovarianceMatrix(label="eval", matrix=np.asarray(values, dtype=float),
                            rowGrid=list(grid), isRelative=relative)


def _short_range(f, grid, direct=False):
    from kika.nuclear_data.model import ShortRangeSelfScalingVariance

    return ShortRangeSelfScalingVariance(
        matrix=_matrix(np.diag(f), grid, relative=False),
        dependenceOnProcessedGroupWidth="direct" if direct else "inverse")


def test_a_mixed_of_kinds_needs_cross_sections():
    from kika.nuclear_data.model import Mixed

    form = Mixed(components=[_matrix(np.eye(2) * 0.01, [1.0, 2.0, 3.0]),
                             _short_range([0.5, 0.5], [1.0, 2.0, 3.0])], label="eval")
    with pytest.raises(MixedKindsNeedCrossSections):
        collapse_form(form)
    sigma = (np.array([1.0, 3.0]), np.array([2.0, 2.0]))
    matrix, grid, relative = collapse_form(form, xs_row=sigma, xs_col=sigma)
    assert relative and grid == [1.0, 2.0, 3.0]
    np.testing.assert_allclose(np.diag(matrix), 0.01 + 0.5 / 4.0)


def test_a_short_range_variance_grows_as_the_group_narrows_and_lb9_vanishes_on_its_grid():
    grid = [1.0, 3.0]
    np.testing.assert_allclose(short_range_variance(grid, [1.0, 0.0], [1.0, 2.0, 3.0], 8), [2.0, 2.0])
    np.testing.assert_allclose(short_range_variance(grid, [1.0, 0.0], grid, 9), [0.0])


def test_a_sum_is_not_collapsed_on_its_own():
    from kika.nuclear_data.model import Sum, Summand

    form = Sum(summands=[Summand(href="x", coefficient=1.0, ENDF_MFMT="33,2")], label="eval")
    with pytest.raises(CovarianceSumNotResolved):
        collapse_form(form)
