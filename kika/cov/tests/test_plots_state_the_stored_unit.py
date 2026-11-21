"""Covariance plots draw the grid as stored and name its unit on the axis."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kika.cov import CrossSectionCovariance

FE56 = 26056


def _covariance(unit: str) -> CrossSectionCovariance:
    scale = 1.0e6 if unit == "eV" else 1.0
    covariance = CrossSectionCovariance(
        num_groups=3,
        energy_grid=[e * scale for e in (1.0e-11, 0.1, 1.0, 20.0)],
        energy_unit=unit,
    )
    covariance.add_matrix(FE56, 2, FE56, 2, np.diag([0.01, 0.04, 0.09]), is_relative=True)
    covariance.cross_sections[(FE56, 2)] = np.array([1.0, 2.0, 3.0])
    return covariance


@pytest.mark.parametrize("unit", ["eV", "MeV"])
@pytest.mark.parametrize("method", ["plot_uncertainties", "plot_multigroup_xs"])
def test_the_energy_axis_is_labelled_in_the_stored_unit(method, unit):
    covariance = _covariance(unit)
    figure = getattr(covariance, method)(FE56, 2)
    try:
        assert figure.axes[0].get_xlabel() == f"Energy ({unit})"
        assert figure.axes[0].get_xlim()[1] >= covariance.energy_grid[-1] * 0.99
    finally:
        plt.close(figure)
