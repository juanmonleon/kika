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

Groups are steps, and where a step falls is arbitrary: a resonance that sits
on an edge is split between two groups, and moving the edges moves the steps.
:func:`resonance_window_average` slides one window of fixed lethargy width
along the energy axis instead, giving a continuous curve; at the centre of each
group of an equal-lethargy grid of that width it *is* the group average.

Until October 2026 the 1/E integral was a trapezoid in ``u = ln E``, which is
exact only for a constant sigma: on a table of three points
``[1e5, 1e6, 2e7] -> [1, 10, 1]`` it was 7 % low, and 8e-7 off on a fine one.

Both functions are adapters, not implementations: they name the weight in
physical terms (lethargy, ``1/E``) and hand the table to :mod:`kika.algebra`,
whose :func:`~kika.algebra.group_averages` and
:func:`~kika.algebra.log_window_averages` do the integrals. The comparison of a
series with these averages is :mod:`kika.algebra.compare`.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from kika.algebra import group_averages, log_window_averages

__all__ = ["resonance_group_average", "resonance_window_average"]


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


def resonance_window_average(
    energies: np.ndarray,
    cross_sections: np.ndarray,
    centres: np.ndarray,
    width: float,
    weighting: Weighting = "lethargy",
    bounds: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Flux-weighted average over a sliding window of fixed lethargy width.

    At each centre ``E`` the window is ``[E e^(-width/2), E e^(width/2)]``,
    clipped to the pointwise range as :func:`resonance_group_average` clips
    its groups, and the average is the same exact integral. The result is a
    smooth function of ``E`` with no group edges to place: at the geometric
    centre of a group of an equal-lethargy grid with ``ln(E_hi/E_lo) ==
    width`` it equals that group's average. Near either end of the table the
    clipped window is one-sided.

    Parameters
    ----------
    energies, cross_sections : np.ndarray
        Pointwise table, lin-lin, energies in eV and non-decreasing.
    centres : np.ndarray
        Energies (eV) at which to evaluate the average -- a display grid,
        not the table's own grid: the cost grows with the number of centres
        that fall in each window.
    width : float
        Window width in lethargy, ``ln(E_hi/E_lo)``; 0.01 is about 1 % in
        energy.
    weighting : {"lethargy", "constant"}
        ``phi(E) = 1/E`` (default) or a flat spectrum.
    bounds : (float, float), optional
        Energies (eV) the windows may not cross -- the span being compared,
        so that a window near its top does not average in the region above
        it. The pointwise range always bounds them.

    Returns
    -------
    np.ndarray
        The average at each centre; ``nan`` where the clipped window is empty.
    """
    energies = np.asarray(energies, dtype=float)
    cross_sections = np.asarray(cross_sections, dtype=float)
    centres = np.asarray(centres, dtype=float)
    if energies.ndim != 1 or energies.shape != cross_sections.shape:
        raise ValueError("energies and cross_sections must be 1-D arrays of one shape")
    if energies.size < 2:
        raise ValueError("need at least 2 pointwise energies")
    if np.any(np.diff(energies) < 0):
        raise ValueError("energies must be non-decreasing")
    if centres.ndim != 1 or np.any(centres <= 0):
        raise ValueError("centres must be a 1-D array of positive energies")
    if not (np.isfinite(width) and width > 0):
        raise ValueError("width must be a positive lethargy interval")
    if weighting not in ("lethargy", "constant"):
        raise ValueError(f"unknown weighting: {weighting!r}")
    if weighting == "lethargy" and energies[0] <= 0.0:
        raise ValueError("lethargy weighting requires strictly positive energies")
    weight = "1/x" if weighting == "lethargy" else None
    return log_window_averages(energies, cross_sections, 2, centres, width,
                               weight, bounds)
