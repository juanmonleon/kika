"""Dosimeters read from a real evaluation (JEFF-4.0 Fe-56 MT4 and MT16 with their MF33), and the adjustment on them."""
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import kika
from kika.algebra import group_averages, linear_update
from kika.cov.decomposition import clip_projection
from kika.dosimetry import SpectrumPrior, adjust_spectrum, dosimeter_from_suite, response_cells

TAPE = Path(kika.__file__).parent / "endf" / "tests" / "data" / "micro_fe56_mf33.endf"
EDGES = np.geomspace(1e5, 2e7, 41)


@pytest.fixture(scope="module")
def suite():
    return kika.read(TAPE)


def _prior():
    mid = np.sqrt(EDGES[:-1] * EDGES[1:]) / 1e6
    flux = 1e10 * np.sqrt(mid) * np.exp(-mid / 1.3) * np.diff(EDGES) / 1e6
    ell = np.log(mid)
    return SpectrumPrior(EDGES, flux, 0.15**2 * np.exp(-0.5 * (ell[:, None] - ell[None, :]) ** 2) + 0.03**2)


def _section(suite, mt):
    return next(s for s in suite.covarianceSuite.covarianceSections if s.rowData.ENDF_MFMT == f"33/{mt}")


def test_the_cross_section_and_its_covariance_are_the_evaluations(suite):
    d = dosimeter_from_suite(suite, 16)
    e, sigma = suite.cross_section(16)
    np.testing.assert_array_equal(d.energies, e)
    np.testing.assert_array_equal(d.cross_section, sigma)
    form = _section(suite, 16).form
    np.testing.assert_array_equal(d.covariance_edges, form.rowGrid)
    np.testing.assert_array_equal(d.relative_covariance, form.matrix)
    assert d.name == "Fe56 MT16"


def test_an_absolute_covariance_is_made_relative_with_the_interval_averages(suite):
    form = _section(suite, 16).form
    d = dosimeter_from_suite(suite, 16)
    average = group_averages(d.energies, d.cross_section, d.laws, form.rowGrid)
    absolute = dataclasses.replace(form, matrix=form.matrix * np.outer(average, average), isRelative=False)
    section = SimpleNamespace(rowData=SimpleNamespace(ENDF_MFMT="33/16"), columnData=None, form=absolute, label="x")
    fake = SimpleNamespace(cross_section=suite.cross_section, reactionByENDF_MT=suite.reactionByENDF_MT,
                           covarianceSuite=SimpleNamespace(covarianceSections=[section]), target="Fe56")
    back = dosimeter_from_suite(fake, 16).relative_covariance
    seen = np.outer(average > 0, average > 0)
    np.testing.assert_allclose(back[seen], form.matrix[seen], rtol=1e-12)
    assert np.all(back[~seen] == 0.0)


def _clipped(d):
    clipped, _ = clip_projection(d.relative_covariance, verbose=False)
    return dataclasses.replace(d, relative_covariance=clipped)


def test_an_evaluation_covariance_that_is_not_positive_semi_definite_is_refused(suite):
    prior = _prior()
    d = dosimeter_from_suite(suite, 4)
    # JEFF-4.0's MF33 of MT4 has 41 negative eigenvalues, down to -4.4e-6 against 7.7.
    with pytest.raises(ValueError, match="pre-flight"):
        adjust_spectrum(prior, [d], [response_cells(prior, d).sum()], [[0.03**2]])


def test_with_real_covariances_the_first_iteration_equals_folding_them_into_the_data(suite):
    prior = _prior()
    dosimeters = [_clipped(dosimeter_from_suite(suite, 4)), _clipped(dosimeter_from_suite(suite, 16))]
    cells = [response_cells(prior, d) for d in dosimeters]
    rates = np.array([c.sum() for c in cells])
    measured = rates * np.array([1.08, 0.93])
    V = np.diag([0.03**2, 0.05**2])
    result = adjust_spectrum(prior, dosimeters, measured, V)

    G = np.vstack([c.sum(axis=1) for c in cells]) / rates[:, None]
    shares = [c.sum(axis=0) / r for c, r in zip(cells, rates)]
    folded = np.diag([s @ d.cell_covariance @ s for s, d in zip(shares, dosimeters)])
    one = linear_update(np.zeros(40), prior.relative_covariance, G, np.log(measured / rates), V + folded)
    np.testing.assert_allclose(result.one_step_flux, prior.flux * np.exp(one.mean), rtol=1e-12)
    assert set(result.cross_section_factors) == {"Fe56 MT4", "Fe56 MT16"}
    assert result.cross_section_factors["Fe56 MT16"].shape == (124,)
