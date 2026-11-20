"""Sensitivities of a response to nuclear data, from MCNP PERT tallies or SDF files.

Sensitivities are relative coefficients, ``(dR/R) / (d sigma / sigma)`` for a response
R and a cross-section sigma.
:func:`~kika.sensitivities.sensitivity_processing.compute_sensitivity` builds them
from the METHOD=2 tallies of PERT cards into a
:class:`~kika.sensitivities.sensitivity.SensitivityData`; the METHOD=3 tallies only
give the second-order Taylor coefficients (c2).
:class:`~kika.sensitivities.sdf.SDFData` holds them per group in the layout of a SCALE
Sensitivity Data File (SDF) and writes it;
:func:`~kika.sensitivities.sdf_parser.read_sdf` reads one.
:class:`~kika.sensitivities.profile.SensitivityProfile` is the format-neutral form that
:func:`~kika.sensitivities.condensation.condense_sensitivity_profile` sums onto a
coarser grid.
Energies are in MeV unless a function says otherwise.
"""
from .sensitivity_processing import compute_sensitivity, compute_total_sensitivity, create_sdf_data, plot_sens_comparison
from .sdf import SDFData, SDFReactionData, sensitivity_to_plot_data
from .sensitivity import SensitivityData, TaylorCoefficients, Coefficients
from .profile import SensitivityProfile, SensitivityReaction
from .condensation import (
    CondensationError,
    CondensationReport,
    CondensationResult,
    condense_sensitivity_profile,
)

__all__ = [
    # Core processing
    'compute_sensitivity', 'compute_total_sensitivity', 'plot_sens_comparison',
    'create_sdf_data',

    # Data classes
    'SDFData', 'SDFReactionData', 'SensitivityProfile', 'SensitivityReaction',
    'SensitivityData', 'TaylorCoefficients', 'Coefficients',

    # Explicit multigroup condensation
    'CondensationError', 'CondensationReport', 'CondensationResult',
    'condense_sensitivity_profile',

    # Plotting adapter
    'sensitivity_to_plot_data',
]
