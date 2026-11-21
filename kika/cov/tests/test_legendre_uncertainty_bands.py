"""Legendre uncertainty bands are relative where they say per cent, absolute where they say absolute."""

import warnings

import numpy as np
import pytest

from kika.cov import LegendreCovariance
from kika.cov.multigroup.mg_legendre_covariance import MultigroupLegendreCovariance
from kika.cov.parse_covmat import read_covfil, write_covfil
from kika.cov import CrossSectionCovariance

FE56 = 26056
GRID_FROM_ZERO = [0.0, 1.0, 10.0, 100.0]
GRID = [1.0e-5, 1.0e5, 1.0e6, 2.0e7]
VARIANCE = np.diag([0.01, 0.04, 0.09])
A1 = np.array([0.5, -0.5, 0.25])


def _legendre(is_relative: bool, grid=GRID, coefficients=True) -> LegendreCovariance:
    covariance = LegendreCovariance()
    covariance.add_matrix(FE56, 2, 1, FE56, 2, 1, VARIANCE, list(grid), is_relative, "cm")
    if coefficients:
        covariance.legendre_coefficients[(FE56, 2, 1)] = A1.copy()
    return covariance


def _multigroup(grid=GRID) -> MultigroupLegendreCovariance:
    covariance = MultigroupLegendreCovariance(energy_grid=np.array(grid))
    covariance.add_matrix(
        FE56, 2, 1, FE56, 2, 1,
        VARIANCE, VARIANCE * np.outer(A1, A1), A1, A1, "cm",
    )
    return covariance


def test_the_multigroup_heatmap_band_is_the_relative_uncertainty_in_percent():
    heatmap = _multigroup().to_heatmap_data(FE56, 2, 1, matrix_type="cov")

    assert np.allclose(heatmap.uncertainty_data[1], [10.0, 20.0, 30.0])


def test_the_heatmap_band_of_an_absolute_block_is_divided_by_the_coefficient():
    covariance = _legendre(is_relative=False)
    heatmap = covariance.to_heatmap_data(FE56, 2, 1, matrix_type="cov")

    assert np.allclose(heatmap.uncertainty_data[1], 100.0 * np.sqrt(np.diag(VARIANCE)) / np.abs(A1))


def test_an_absolute_block_without_its_coefficient_has_no_heatmap_band():
    covariance = _legendre(is_relative=False, coefficients=False)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        heatmap = covariance.to_heatmap_data(FE56, 2, 1, matrix_type="cov")

    assert not heatmap.uncertainty_data


@pytest.mark.parametrize("build", [_legendre, _multigroup], ids=["mf34", "multigroup"])
def test_a_grid_starting_at_zero_gives_its_first_bin_one_decade_on_a_log_axis(build):
    covariance = build(grid=GRID_FROM_ZERO) if build is _multigroup else build(True, GRID_FROM_ZERO)
    heatmap = covariance.to_heatmap_data(FE56, 2, 1, matrix_type="cov", scale="log")

    assert np.allclose(heatmap.x_edges, [0.0, 1.0, 2.0, 3.0])


def test_the_absolute_uncertainty_of_a_relative_block_is_multiplied_by_the_coefficient():
    _, band = _legendre(is_relative=True).to_plot_data(FE56, 2, 1, uncertainty_type="absolute")

    expected = np.sqrt(np.diag(VARIANCE)) * np.abs(A1)
    assert np.allclose(band.y, np.append(expected, expected[-1]))


def test_the_absolute_uncertainty_of_a_relative_block_needs_the_coefficient():
    covariance = _legendre(is_relative=True, coefficients=False)

    with pytest.raises(ValueError, match="a_L"):
        covariance.to_plot_data(FE56, 2, 1, uncertainty_type="absolute")


def _mf3_lines(tmp_path, values) -> list:
    carrier = CrossSectionCovariance(num_groups=3, energy_grid=list(GRID))
    carrier.add_matrix(FE56, 2, FE56, 2, VARIANCE, is_relative=True)
    carrier.cross_sections[(FE56, 251)] = np.asarray(values, dtype=float)
    path = tmp_path / "carrier.gendf"
    write_covfil(carrier, str(path))
    lines = path.read_text().splitlines(keepends=True)
    mf3 = [line for line in lines if line[70:72] == " 3" and line[72:75] == "251"]
    return mf3 + [line for line in lines if line[70:75] == " 3  0"][:1]


def test_covfil_coefficients_are_filed_under_the_reaction_of_the_mf34_blocks(tmp_path):
    path = tmp_path / "mf34.gendf"
    write_covfil(_legendre(is_relative=True, coefficients=False), str(path))
    lines = path.read_text().splitlines(keepends=True)
    end_of_mf1 = next(i for i, line in enumerate(lines) if i > 0 and line[70:75] == " 0  0") + 1
    lines[end_of_mf1:end_of_mf1] = _mf3_lines(tmp_path, A1) + [lines[end_of_mf1 - 1]]
    path.write_text("".join(lines))

    covariance = read_covfil(str(path))

    assert set(covariance.legendre_coefficients) == {(FE56, 2, 1)}
    coefficients, _ = covariance.to_plot_data(FE56, 2, 1)
    assert np.allclose(coefficients.y[:-1], A1)
