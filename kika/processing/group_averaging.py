"""Flux-weighted energy-group averaging of pointwise cross sections.

Pointwise comparisons in the resonance region are dominated by narrow
peaks whose positions differ slightly between evaluations, producing
large pointwise discrepancies with small integral impact. Group
averaging with a 1/E spectrum is the standard way to compare
evaluations in that region.

For each group ``g`` with boundaries ``[E_lo, E_hi]`` the average is:

    sigma_bar_g = integral sigma(E) * phi(E) dE / integral phi(E) dE

evaluated on the overlap of the group with the pointwise energy range.
The pointwise cross section is lin-lin between tabulated points, and both
integrals are closed form for either weight (:func:`kika.algebra.group_averages`):
for ``phi = 1/E`` a panel ``sigma = a + bE`` gives ``a ln(E2/E1) + b (E2 - E1)``.

Until October 2026 the 1/E integral was a trapezoid in ``u = ln E``, which is
exact only for a constant sigma: on a table of three points
``[1e5, 1e6, 2e7] -> [1, 10, 1]`` it was 7 % low, and 8e-7 off on a fine one.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from kika.algebra import group_averages

__all__ = ["resonance_group_average"]


Weighting = Literal["lethargy", "constant"]


def resonance_group_average(
    energies: np.ndarray,
    cross_sections: np.ndarray,
    group_boundaries: np.ndarray,
    weighting: Weighting = "lethargy",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute flux-weighted group averages of a pointwise cross section.

    Parameters
    ----------
    energies : np.ndarray
        Pointwise energies in eV, non-decreasing (a repeated energy is a
        step).
    cross_sections : np.ndarray
        Pointwise cross section values (e.g., barns), same shape as
        ``energies``.
    group_boundaries : np.ndarray
        Group bin edges (``n_bins + 1`` values), strictly increasing.
    weighting : {"lethargy", "constant"}
        Spectrum ``phi(E)`` used for weighting. ``"lethargy"`` (default)
        uses ``phi(E) = 1/E``.

    Returns
    -------
    group_edges : np.ndarray
        The input ``group_boundaries`` (returned for convenience, so
        callers can treat the function as the source of both edges and
        values).
    group_averages : np.ndarray
        Group-averaged cross sections, length ``n_bins``. Groups that
        fall entirely outside the pointwise energy range, or that have
        a degenerate weight integral, are set to ``NaN``.
    """
    energies = np.asarray(energies, dtype=float)
    cross_sections = np.asarray(cross_sections, dtype=float)
    group_boundaries = np.asarray(group_boundaries, dtype=float)

    if energies.ndim != 1 or cross_sections.ndim != 1:
        raise ValueError("energies and cross_sections must be 1-D arrays")
    if energies.shape != cross_sections.shape:
        raise ValueError("energies and cross_sections must have the same shape")
    if energies.size < 2:
        raise ValueError("need at least 2 pointwise energies")
    if np.any(np.diff(energies) < 0):
        raise ValueError("energies must be non-decreasing")
    if group_boundaries.ndim != 1 or group_boundaries.size < 2:
        raise ValueError("group_boundaries must be a 1-D array with >=2 entries")
    if not np.all(np.diff(group_boundaries) > 0):
        raise ValueError("group_boundaries must be strictly increasing")
    if weighting not in ("lethargy", "constant"):
        raise ValueError(f"unknown weighting: {weighting!r}")
    if weighting == "lethargy" and energies[0] <= 0.0:
        raise ValueError("lethargy weighting requires strictly positive energies")

    # Each group is averaged over its overlap with the table, so clip the
    # edges to the table's range; a group left with no width is NaN.
    clipped = np.clip(group_boundaries, energies[0], energies[-1])
    weight = "1/x" if weighting == "lethargy" else None
    averages = group_averages(energies, cross_sections, 2, clipped, weight)

    return group_boundaries.copy(), averages
