"""Dosimeter responses to a group spectrum, cell by cell.

A dosimeter's rate in a spectrum of group fluxes ``phi_g`` is
``R = sum_g phi_g sigma_g`` with ``sigma_g`` the cross section averaged over the
group with the prior's own fine spectrum inside it. Spectrum adjustment needs that rate
split twice: by flux group (how the rate moves when a group flux moves) and by
the intervals of the cross section's covariance (how much of the rate each
covariance interval carries). :func:`response_cells` gives the rate of every
**cell**, a flux group intersected with a covariance interval; both splits are
its row and column sums. Every integral is exact (:func:`kika.algebra.product_group_integrals`).

Energies are in eV, cross sections in barns. A rate is per target atom and per
unit time of the flux: ``1e-24 cm2/b x sigma x phi``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from kika.algebra import group_integrals, product_group_integrals, validate

__all__ = ["BARN_CM2", "Dosimeter", "SpectrumPrior", "response_cells"]

#: One barn in cm2.
BARN_CM2 = 1.0e-24


def _symmetric(name: str, matrix: np.ndarray, size: int) -> np.ndarray:
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must be {size} x {size}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} has a value that is not finite")
    if np.max(np.abs(matrix - matrix.T), initial=0.0) > 1e-10 * np.max(np.abs(matrix), initial=0.0):
        raise ValueError(f"{name} is not symmetric")
    return matrix


def _increasing(name: str, edges) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must be a strictly increasing 1-d array of two values or more")
    return edges


@dataclass(frozen=True)
class SpectrumPrior:
    """A group spectrum from a calculation, with its relative covariance.

    Parameters
    ----------
    edges : array_like
        Group boundaries in eV, strictly increasing: ``n + 1`` values.
    flux : array_like
        The flux integrated over each group, ``n`` values, in the unit the rates
        will be in (for example cm-2 s-1). A group of zero flux (above the source,
        or one a tally never scored) is not adjusted and stays zero; at least one
        group must be positive.
    relative_covariance : array_like
        ``n x n``, symmetric: the covariance of ``phi_g / phi_g`` (to first order,
        of ``ln phi_g``). It may be singular. It holds every uncertainty of the
        calculation, the normalisation included.
    fine_shape : tuple, optional
        A fine spectrum per unit energy ``(x, y, laws)``, x in eV, used only as the
        weight inside each group when averaging a cross section; its scale does
        not matter. None weights each group flat in energy.

    Raises
    ------
    ValueError
        If a shape does not match, a flux is negative or not finite, no flux is
        positive, the covariance is not symmetric, or *fine_shape* is not a valid
        table.
    """

    edges: np.ndarray
    flux: np.ndarray
    relative_covariance: np.ndarray
    fine_shape: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def __post_init__(self):
        edges = _increasing("edges", self.edges)
        flux = np.asarray(self.flux, dtype=float)
        if flux.shape != (edges.size - 1,):
            raise ValueError(f"flux must have {edges.size - 1} values, got shape {flux.shape}")
        if not np.all(np.isfinite(flux)) or np.any(flux < 0):
            raise ValueError("flux must be finite and not negative in every group")
        if not np.any(flux > 0):
            raise ValueError("flux is zero in every group")
        cov = _symmetric("relative_covariance", np.asarray(self.relative_covariance, dtype=float), flux.size)
        fine_shape = None if self.fine_shape is None else validate(*self.fine_shape)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "relative_covariance", cov)
        object.__setattr__(self, "fine_shape", fine_shape)


@dataclass(frozen=True)
class Dosimeter:
    """A dosimetry reaction: its cross section and the relative covariance of it.

    Parameters
    ----------
    name : str
        A label for reports.
    energies, cross_section : array_like
        The cross section table: energies in eV (non-decreasing) and values in barns.
    laws : array_like or int
        One ENDF interpolation law per interval (1-5), or one for all.
    covariance_edges : array_like, optional
        The boundaries in eV of the intervals of *relative_covariance*, as the
        evaluation states them. None means the cross section carries no
        uncertainty.
    relative_covariance : array_like, optional
        The relative covariance of the cross section averaged over each interval,
        ``len(covariance_edges) - 1`` square, symmetric. Given with
        *covariance_edges* or not at all.
    reaction : str, optional
        The reaction whose cross-section parameters the adjustment uses for this
        dosimeter; *name* by default. Two dosimeters of one reaction (a foil bare
        and one covered, :func:`covered`) share those parameters, so they must
        state the same covariance.

    Raises
    ------
    ValueError
        If the table is not valid, or only one of *covariance_edges* and
        *relative_covariance* is given, or their shapes do not match.
    """

    name: str
    energies: np.ndarray
    cross_section: np.ndarray
    laws: np.ndarray = 2
    covariance_edges: Optional[np.ndarray] = None
    relative_covariance: Optional[np.ndarray] = None
    reaction: Optional[str] = None

    def __post_init__(self):
        x, y, laws = validate(self.energies, self.cross_section, self.laws)
        if self.reaction is None:
            object.__setattr__(self, "reaction", self.name)
        if (self.covariance_edges is None) != (self.relative_covariance is None):
            raise ValueError("covariance_edges and relative_covariance go together")
        edges, cov = None, None
        if self.covariance_edges is not None:
            edges = _increasing("covariance_edges", self.covariance_edges)
            cov = _symmetric("relative_covariance", np.asarray(self.relative_covariance, dtype=float),
                             edges.size - 1)
        object.__setattr__(self, "energies", x)
        object.__setattr__(self, "cross_section", y)
        object.__setattr__(self, "laws", laws)
        object.__setattr__(self, "covariance_edges", edges)
        object.__setattr__(self, "relative_covariance", cov)

    @property
    def cell_covariance(self) -> np.ndarray:
        """The relative covariance of the cell columns of :func:`response_cells`: the intervals, then a column of no uncertainty."""
        k = 0 if self.covariance_edges is None else self.covariance_edges.size - 1
        out = np.zeros((k + 1, k + 1))
        if k:
            out[:k, :k] = self.relative_covariance
        return out


def response_cells(prior: SpectrumPrior, dosimeter: Dosimeter) -> np.ndarray:
    """The rate of a dosimeter in the prior spectrum, split by flux group and covariance interval.

    Cell ``(g, k)`` is ``1e-24 phi_g int_{g cap k} sigma w dE / int_g w dE``, with
    ``w`` the prior's fine spectrum (flat when it has none). The last column holds the
    part of the rate that falls outside the covariance intervals (all of it for
    a dosimeter with no covariance), which carries no uncertainty.

    Parameters
    ----------
    prior : SpectrumPrior
        The group spectrum and the fine spectrum inside each group.
    dosimeter : Dosimeter
        The cross section and its covariance intervals.

    Returns
    -------
    numpy.ndarray
        ``n_groups x (n_intervals + 1)``, per target atom and per unit time of the
        flux. Its sum is the rate.

    Raises
    ------
    ValueError
        If the prior's fine spectrum has no weight in a group of positive flux.
    """
    groups = prior.edges
    cov_edges = np.zeros(0) if dosimeter.covariance_edges is None else dosimeter.covariance_edges
    cuts = np.union1d(groups, cov_edges[(cov_edges > groups[0]) & (cov_edges < groups[-1])])
    sigma = (dosimeter.energies, dosimeter.cross_section, dosimeter.laws)
    if prior.fine_shape is None:
        weight = (groups[[0, -1]], np.ones(2), 2)
    else:
        weight = prior.fine_shape
    pieces = product_group_integrals(sigma, weight, cuts)
    norm = group_integrals(*weight, groups)
    empty = (norm <= 0) & (prior.flux > 0)
    if np.any(empty):
        raise ValueError(f"the prior's fine spectrum has no weight in group {int(np.argmax(empty))}, "
                         f"whose flux is positive")
    norm = np.where(norm > 0, norm, 1.0)

    lower = cuts[:-1]
    g = np.searchsorted(groups, lower, side="right") - 1
    n_intervals = cov_edges.size - 1 if cov_edges.size else 0
    k = np.searchsorted(cov_edges, lower, side="right") - 1 if n_intervals else np.full(lower.size, -1)
    k = np.where((k >= 0) & (k < n_intervals), k, n_intervals)
    cells = np.zeros((groups.size - 1, n_intervals + 1))
    np.add.at(cells, (g, k), pieces * (BARN_CM2 * prior.flux / norm)[g])
    return cells
