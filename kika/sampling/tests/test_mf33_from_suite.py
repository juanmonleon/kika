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


def test_the_nubar_covariance_of_a_model_suite_is_build_mf31_covariance_s():
    """MF31 weighted by the suite's own nu-bar, against the tape path the app uses."""
    import kika
    from kika.sampling.mf31_sampling import build_mf31_covariance, build_mf31_covariance_from_suite

    tape = DATA / "micro_u235_nubar.endf"
    endf = read_endf(str(tape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suite = kika.read(str(tape))
    fromTape, _, gridTape, presentTape = build_mf31_covariance(endf)
    fromSuite, _, gridSuite, presentSuite, blocks = build_mf31_covariance_from_suite(suite)
    assert presentSuite == presentTape and not blocks.declined
    assert gridSuite == gridTape
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


def test_a_metastable_target_keys_its_blocks_by_its_ground_state_za():
    from kika.sampling.model_blocks import _za_of

    bare = types.SimpleNamespace(provenance=None)
    assert _za_of(types.SimpleNamespace(target="Am242_m1"), bare) == 95242
    assert _za_of(types.SimpleNamespace(target="U235_e1"), bare) == 92235


def test_a_sum_is_not_collapsed_on_its_own():
    from kika.nuclear_data.model import Sum, Summand

    form = Sum(summands=[Summand(href="x", coefficient=1.0, ENDF_MFMT="33,2")], label="eval")
    with pytest.raises(CovarianceSumNotResolved):
        collapse_form(form)


def test_without_cross_sections_a_block_that_needs_them_is_declined_by_code():
    from kika.sampling.mf33_sampling import NEEDS_CROSS_SECTIONS, suite_needs_cross_sections

    endf = read_endf(str(DATA / "micro_si28_covcheck.endf"))
    suite = _model_suite(endf, _sigma(_all_mts()))
    assert suite_needs_cross_sections(suite)
    _, _, present, blocks = build_mf33_covariance_from_suite(suite, None, None)
    assert blocks.declined and {d.code for d in blocks.declined.values()} == {NEEDS_CROSS_SECTIONS}
    assert not set(blocks.declined) & set(present)


def test_a_suite_written_to_gnds_by_kika_assembles_as_it_did(tmp_path):
    """NNDC's F-19 covariances, written by kika's GNDS writer and read back: the same MF33."""
    from kika.gnds.covariances import readCovarianceSuite
    from kika.gnds.encode import writeCovarianceSuite
    from kika.gnds.xpath import Document

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suite, _ = readCovarianceSuite(Document.parse(GNDS_COV / "n-009_F_019.endf.gnds-covar.xml"))
        tree, _ = writeCovarianceSuite(suite, "2.0")
        path = tmp_path / "f19-covar.xml"
        tree.write(str(path), encoding="utf-8", xml_declaration=True)
        back, _ = readCovarianceSuite(Document.parse(path))
    xs = _sigma(_all_mts())
    before, _, present, _ = build_mf33_covariance_from_suite(suite, None, xs)
    after, _, presentAfter, _ = build_mf33_covariance_from_suite(back, None, xs)
    assert presentAfter == present
    _same(before, after)


_C2_102 = np.arange(1.0, 10.0).reshape(3, 3) * 1e-5


def _cross_nc_suite():
    """MT1 = MT2 + MT102, stated as a sum in its own block and in its blocks with MT2 and MT102.

    Only Cov(2, 102) is stated, and it is not symmetric: Cov(1, 2) needs its transpose.
    """
    from kika.nuclear_data.model import CovarianceSection, DataLink, Sum, Summand

    grid = [1.0e-5, 1.0e3, 1.0e6, 2.0e7]
    rel = lambda values: _matrix(values, grid)
    link = lambda mt: DataLink(href=f"/reactionSuite/reactions/reaction[@label='MT{mt}']", ENDF_MFMT=f"33,{mt}")
    total = lambda: Sum(summands=[Summand(href="MF33-MT2", coefficient=1.0, ENDF_MFMT="33,2"),
                                  Summand(href="MF33-MT102", coefficient=1.0, ENDF_MFMT="33,102")],
                        label="eval", domainMin=1.0e-5, domainMax=2.0e7, domainUnit="eV")
    sections = [
        CovarianceSection(label="MF33-MT1", rowData=link(1), form=total()),
        CovarianceSection(label="MF33-MT1-MT2", rowData=link(1), columnData=link(2), form=total()),
        CovarianceSection(label="MF33-MT1-MT102", rowData=link(1), columnData=link(102), form=total()),
        CovarianceSection(label="MF33-MT2", rowData=link(2), form=rel(np.diag([4e-4, 9e-4, 1e-3]))),
        CovarianceSection(label="MF33-MT2-MT102", rowData=link(2), columnData=link(102),
                          form=rel(_C2_102)),
        CovarianceSection(label="MF33-MT102", rowData=link(102), form=rel(np.diag([1e-2, 4e-3, 2.5e-3]))),
    ]
    return types.SimpleNamespace(covarianceSections=sections, target="Fe56")


def test_a_sum_stated_in_a_cross_block_is_the_sum_of_its_parts_blocks():
    """Cov(1, j) = Cov(2, j) + Cov(102, j) in absolute terms, the same on the tape and on the model."""
    from kika.endf.model_adapter.covariances import _faithfulSection
    from kika.nuclear_data.model.conversion import ConversionReport

    suite = _cross_nc_suite()
    e = np.geomspace(1e-5, 2e7, 50)
    xs = {2: types.SimpleNamespace(energies=e, values=np.full(e.size, 3.0)),
          102: types.SimpleNamespace(energies=e, values=np.full(e.size, 1.0)),
          1: types.SimpleNamespace(energies=e, values=np.full(e.size, 4.0))}

    fromSuite, grid, present, blocks = build_mf33_covariance_from_suite(suite, [1, 2, 102], xs)
    assert not blocks.declined
    rows = {}
    for s in suite.covarianceSections:
        rows.setdefault(s.rowData.ENDF_MT, []).append(s)
    sections = {mt: _faithfulSection(ss, 33, mt, 26056.0, 55.45, 2631, ConversionReport())
                for mt, ss in rows.items()}
    fromTape, _, _ = build_mf33_covariance(types.SimpleNamespace(sections=sections), [1, 2, 102], xs)
    _same(fromTape, fromSuite)

    cov = fromSuite.covariance_matrix
    n = len(grid) - 1
    c22, c102 = np.diag([4e-4, 9e-4, 1e-3]), np.diag([1e-2, 4e-3, 2.5e-3])
    with_102 = (_C2_102 * 3.0 * 1.0 + c102 * 1.0 * 1.0) / (4.0 * 1.0)
    with_2 = (c22 * 3.0 * 3.0 + _C2_102.T * 1.0 * 3.0) / (4.0 * 3.0)
    np.testing.assert_allclose(cov[0:n, 2 * n:3 * n], with_102, rtol=1e-12)   # block order 1, 2, 102
    np.testing.assert_allclose(cov[0:n, n:2 * n], with_2, rtol=1e-12)
