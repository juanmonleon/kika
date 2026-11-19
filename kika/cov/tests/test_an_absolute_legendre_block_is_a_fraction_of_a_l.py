"""An LB=0 MF34 block is shown as a per cent of a_L, not as √C·100.

A relative block's √C already is the fraction; an absolute one is in units of
a_L and has to be divided by it. Hand-computable case, no tape.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.cov.legendre_covariance import LegendreCovariance

ISO, MT, L = 26056, 2, 1
GRID = [1.0e5, 1.0e6, 2.0e7]
VARIANCE = np.diag([4.0e-4, 1.0e-4])  # σ(a_1) = 0.02, 0.01
A1 = np.array([0.1, -0.05])  # cell means of a_1, one of them negative


def _covariance(is_relative: bool, with_coefficients: bool = True) -> LegendreCovariance:
    cov = LegendreCovariance(energy_unit="eV")
    cov.add_matrix(ISO, MT, L, ISO, MT, L, VARIANCE, GRID,
                   is_relative=is_relative, frame="same-as-MF4")
    if with_coefficients:
        cov.legendre_coefficients[(ISO, MT, L)] = A1
    return cov


def _percent(cov: LegendreCovariance) -> np.ndarray:
    _, data = cov.to_plot_data(nuclide=ISO, mt=MT, order=L)
    return np.asarray(data.y)[:-1]  # the last value repeats for the step plot


def test_an_absolute_block_is_divided_by_the_coefficient():
    np.testing.assert_allclose(_percent(_covariance(is_relative=False)),
                               [20.0, 20.0], rtol=1e-12)


def test_a_relative_block_is_already_the_fraction():
    np.testing.assert_allclose(_percent(_covariance(is_relative=True)),
                               [2.0, 1.0], rtol=1e-12)


def test_an_absolute_block_without_a_l_is_refused():
    with pytest.raises(ValueError, match="LB=0"):
        _percent(_covariance(is_relative=False, with_coefficients=False))
