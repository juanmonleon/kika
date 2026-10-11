"""GNDS ``xs_pdf_cdf1d``: a probability density and its cumulative, side by side.

The form processed Monte Carlo data takes. GNDS defines it for the
``MonteCarlo_cdf`` style (§9.4.3), whose forms are what a transport code samples
from directly: FUDGE writes ``xs_pdf_cdf1d`` children under ``energyAngularMC``,
``angularEnergyMC`` and the Monte Carlo ``KalbachMann``. In kika it is what an
OpenMC HDF5 table becomes on read, and what the ACER-like tabulation of
:mod:`kika.processing.tabulation` produces for the HDF5 writer (roadmap D6, D9).

**The cumulative is data, not a derived view.** OpenMC's C++ samples the stored
``c`` row as it is -- the PDF→CDF rebuild is dead code
(``distribution_energy.cpp:111-117``) -- and the official tables come from ACER,
whose integration rule is not one kika reproduces bit for bit. Recomputing the
CDF on write would make a read → write of an official file differ in every
table, which is why the node keeps both arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..axes import Axes
from ..enums import INTERPOLATION_TO_ENDF_INT, Interpolation
from .base import Function1d, OutOfRange, _interpolate_1d

__all__ = ["Xs_pdf_cdf1d", "cumulativeOf"]


def cumulativeOf(xs, pdf, interpolation: Interpolation = Interpolation.linlin) -> np.ndarray:
    """The cumulative of a tabulated density, starting at 0.

    Histogram densities integrate as rectangles, lin-lin ones as trapezoids --
    the rule ACER and OpenMC's ``Tabular`` use. Not normalised: a caller that
    wants a CDF ending at 1 divides by the last value and decides what a zero
    integral means.
    """
    xs, pdf = np.asarray(xs, dtype=float), np.asarray(pdf, dtype=float)
    out = np.zeros_like(xs)
    if xs.size < 2:
        return out
    dx = np.diff(xs)
    if Interpolation(interpolation) == Interpolation.flat:
        pieces = pdf[:-1] * dx
    else:
        pieces = 0.5 * (pdf[:-1] + pdf[1:]) * dx
    out[1:] = np.cumsum(pieces)
    return out


@dataclass
class Xs_pdf_cdf1d(Function1d):  # noqa: N801 - GNDS node name
    """``xs_pdf_cdf1d``: abscissae, density and cumulative on the same points.

    ``interpolation`` is the density's rule between points: ``flat``
    (histogram) or ``lin-lin``, the only two OpenMC tables use. ``evaluate``
    returns the density.

    ``numberOfDiscrete`` is **not GNDS**. OpenMC stores the discrete lines of a
    secondary-energy table as its first points (``n_discrete_lines``, ACE law
    4's ND), each a delta of weight ``pdf`` at ``xs``, followed by the
    continuum. GNDS would split them into a separate form; the one-table layout
    is kept here because the C++ samples it as one table with one cumulative.
    """

    xs: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray
    interpolation: Interpolation = Interpolation.linlin
    axes: Optional[Axes] = None
    label: Optional[str] = None
    outerDomainValue: Optional[float] = None
    index: Optional[int] = None
    numberOfDiscrete: int = 0

    def __post_init__(self) -> None:
        self.xs = np.asarray(self.xs, dtype=float)
        self.pdf = np.asarray(self.pdf, dtype=float)
        self.cdf = np.asarray(self.cdf, dtype=float)
        if not (self.xs.shape == self.pdf.shape == self.cdf.shape) or self.xs.ndim != 1:
            raise ValueError(f"xs, pdf and cdf must be 1-d of one length, got "
                             f"{self.xs.shape}, {self.pdf.shape}, {self.cdf.shape}")
        self.interpolation = Interpolation(self.interpolation)
        self.numberOfDiscrete = int(self.numberOfDiscrete)

    @classmethod
    def fromPdf(cls, xs, pdf, interpolation: Interpolation = Interpolation.linlin,
                normalise: bool = True, **kwargs) -> "Xs_pdf_cdf1d":
        """The node for a density alone, its cumulative integrated here."""
        xs = np.asarray(xs, dtype=float)
        pdf = np.asarray(pdf, dtype=float)
        cdf = cumulativeOf(xs, pdf, interpolation)
        if normalise:
            total = cdf[-1] if cdf.size else 0.0
            if not total > 0:
                raise ValueError("the density integrates to zero; there is nothing to normalise")
            pdf, cdf = pdf / total, cdf / total
        return cls(xs=xs, pdf=pdf, cdf=cdf, interpolation=interpolation, **kwargs)

    def __len__(self) -> int:
        return int(self.xs.size)

    @property
    def domainMin(self) -> float:
        return float(self.xs[0]) if self.xs.size else float("nan")

    @property
    def domainMax(self) -> float:
        return float(self.xs[-1]) if self.xs.size else float("nan")

    @property
    def endfInterpolationCode(self) -> int:
        return INTERPOLATION_TO_ENDF_INT[self.interpolation]

    def evaluate(self, x, outOfRange: OutOfRange = "zero"):
        start = self.numberOfDiscrete
        xs, pdf = self.xs[start:], self.pdf[start:]
        return _interpolate_1d()(xs, pdf, [(len(xs), self.endfInterpolationCode)], x, outOfRange)

    def __repr__(self) -> str:
        lines = f", {self.numberOfDiscrete} discrete" if self.numberOfDiscrete else ""
        return f"Xs_pdf_cdf1d(n={len(self)}, {self.interpolation.value}{lines})"
