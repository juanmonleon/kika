"""Neutron spectrum adjustment with dosimeters, by generalized least squares.

A calculated group spectrum with its covariance, dosimetry cross sections with
theirs (IRDFF-II, for example) and measured reaction rates with theirs give the
adjusted spectrum, its covariance and the consistency of the measurements.
:func:`adjust_spectrum` does it; the linear algebra is
:func:`kika.algebra.linear_update`.
"""
from .response import BARN_CM2, Dosimeter, SpectrumPrior, response_cells
from .adjust import AdjustmentNotConverged, SpectrumAdjustment, adjust_spectrum
from .model import dosimeter_from_suite
from .prior import nuclear_data_covariance, prior_covariance

__all__ = [
    "BARN_CM2", "Dosimeter", "SpectrumPrior", "response_cells",
    "AdjustmentNotConverged", "SpectrumAdjustment", "adjust_spectrum",
    "dosimeter_from_suite", "nuclear_data_covariance", "prior_covariance",
]
