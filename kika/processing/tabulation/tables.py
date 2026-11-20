"""Building blocks of the ACER-like tabulation: refine, clip, normalise, integrate.

A Monte Carlo table is a density on a lin-lin (or histogram) grid fine enough
that interpolation between its nodes reproduces the function, with the
cumulative at the nodes (``xs_pdf_cdf1d``). These helpers turn a function
evaluable at any point into such a table, and a stated table into a normalised
one with its cumulative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["TabulationOptions", "refine", "pdfTable", "normalisedTable", "NEGATIVE_POLICIES"]

#: What to do with a density that dips below zero (a truncated Legendre series
#: does): ``"clip"`` sets the negative nodes to 0 and renormalises -- the rule
#: ACER applies when it tabulates Legendre data; ``"raise"`` refuses.
NEGATIVE_POLICIES = ("clip", "raise")


@dataclass(frozen=True)
class TabulationOptions:
    """Tolerances of the tabulation.

    ``rtol``/``atol`` bound the lin-lin interpolation error of a refined table
    against the function it tabulates, ``|f - lin| <= rtol*|f| + atol*max|f|``
    at a quarter, the half and three quarters of every interval.
    0.1 % is the tolerance the official libraries are processed to (RECONR,
    BROADR, ACER). ``maxPoints`` caps one table; reaching it is an error, not a
    silent truncation.

    Attributes
    ----------
    rtol : float
        Relative tolerance of the interpolation, a fraction. Default 1e-3.
    atol : float
        Absolute tolerance as a fraction of the table's largest value. Default
        1e-3.
    maxPoints : int
        Most points of one refined table. Default 2000. Reaching it raises
        ``ValueError``.
    negative : {"clip", "raise"}
        What to do with a density that goes negative: ``"clip"`` (default) sets
        it to zero and renormalises, ``"raise"`` refuses.
    angularSeeds : int
        Initial points of an angular table refined from a Legendre series. Default
        3 (a high order starts from more).
    energySeeds : int
        Initial points of an outgoing-energy table refined from a law. Default 64.
    angularMomentAtol : float
        Absolute error allowed in the first and second moments of a Legendre
        angular law, not negative. Default 1e-4.
    mixedCdfAtol : float
        Absolute cumulative error of the histogram that replaces a mixed
        discrete and continuous energy law, positive and finite. Default 1e-6; it
        is not capped by ``maxPoints``.

    Raises
    ------
    ValueError
        On construction, if ``mixedCdfAtol`` is not positive and finite,
        ``angularMomentAtol`` is negative or not finite, or ``negative`` is not
        one of the two above.
    """

    rtol: float = 1.0e-3
    #: Relative to the table's largest value. 1e-3 puts Fe-56's 3 979 elastic
    #: tables (ENDF/B-VIII.1) at 120 809 points where ACER puts 153 265 and
    #: 1e-6 would put 271 771 (measured 2026-10-11): the error stays bounded at
    #: every point, 0.1 % of the value or of the table's peak.
    atol: float = 1.0e-3
    maxPoints: int = 2000
    negative: str = "clip"
    #: Initial points of an angular table refined from a Legendre series.
    angularSeeds: int = 3
    #: Initial points of an outgoing-energy table refined from a law.
    energySeeds: int = 64
    #: Absolute error in normalized first and second moments of a Legendre
    #: angular law (after the explicitly selected negative-density policy). A
    #: density read to rtol moves <mu> by at most rtol (1e-3); 1e-4 keeps it ten
    #: times inside. 1e-5 kept 241 points of an L = 60 law that 10-40 hold.
    angularMomentAtol: float = 1.0e-4
    #: Absolute CDF error of the transport-safe histogram projection of a
    #: mixed discrete/continuous energy law. Its mass, first and second
    #: moments are retained exactly per panel. The projection's size is bounded
    #: by this tolerance and the panel count, so maxPoints does not cap it.
    mixedCdfAtol: float = 1.0e-6

    def __post_init__(self):
        if not np.isfinite(self.mixedCdfAtol) or self.mixedCdfAtol <= 0:
            raise ValueError("mixed CDF tolerance must be positive and finite")
        if not np.isfinite(self.angularMomentAtol) or self.angularMomentAtol < 0:
            raise ValueError("angular moment tolerance must be finite and nonnegative")
        if self.negative not in NEGATIVE_POLICIES:
            raise ValueError(f"negative must be one of {NEGATIVE_POLICIES}")


def refine(function: Callable[[np.ndarray], np.ndarray], grid: Sequence[float],
           options: TabulationOptions) -> Tuple[np.ndarray, np.ndarray]:
    """``(x, f(x))`` on ``grid`` bisected until lin-lin reproduces ``function``.

    Each pass tests every interval still open at a quarter, the half and three
    quarters of its width, all at once, and splits at the half those where any
    of the three misses the chord by more than the tolerance. A midpoint alone
    can sit on the chord by coincidence while the function bulges on both
    sides of it (Fe-56's elastic at 1.438 MeV: [-1, 0] passed as one segment,
    and the mean cosine came out +0.10 off). Raises when ``maxPoints`` would be
    exceeded.

    Parameters
    ----------
    function
        Vectorised: takes an array of abscissae and returns an array of values.
    grid
        Starting abscissae (sorted and made unique here).
    options : TabulationOptions
        ``rtol``, ``atol`` and ``maxPoints`` are used. The error allowed at a probe
        is ``rtol * |f| + atol * max|f|``, the maximum over the values seen so far.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)``: the abscissae, increasing, and the function's values there.
        An interval too narrow to split in floating point is left as it is.

    Raises
    ------
    ValueError
        If the table would need more than ``options.maxPoints`` points.
    """
    x = np.unique(np.asarray(grid, dtype=float))
    y = np.asarray(function(x), dtype=float)
    scale = max(float(np.max(np.abs(y))) if y.size else 0.0, 1e-300)
    open_ = np.ones(len(x) - 1, dtype=bool)
    while open_.any():
        lo = np.flatnonzero(open_)
        mid = 0.5 * (x[lo] + x[lo + 1])
        fm = np.asarray(function(mid), dtype=float)
        bad = np.zeros(len(lo), dtype=bool)
        for t, ft in ((0.5, fm),
                      (0.25, np.asarray(function(x[lo] + 0.25 * (x[lo + 1] - x[lo])), dtype=float)),
                      (0.75, np.asarray(function(x[lo] + 0.75 * (x[lo + 1] - x[lo])), dtype=float))):
            chord = y[lo] + t * (y[lo + 1] - y[lo])
            bad |= np.abs(ft - chord) > options.rtol * np.abs(ft) + options.atol * scale
        # Intervals already at floating-point resolution cannot split further.
        bad &= (x[lo + 1] - x[lo]) > 4 * np.spacing(np.maximum(np.abs(x[lo]), np.abs(x[lo + 1])))
        if not bad.any():
            break
        if len(x) + int(bad.sum()) > options.maxPoints:
            raise ValueError(f"tabulation needs more than {options.maxPoints} points to reach "
                             f"rtol={options.rtol}")
        newX = np.concatenate([x, mid[bad]])
        newY = np.concatenate([y, fm[bad]])
        order = np.argsort(newX, kind="stable")
        x, y = newX[order], newY[order]
        scale = max(scale, float(np.max(np.abs(fm[bad]))))
        # Re-open the two halves of every split interval, close the rest.
        isNew = np.zeros(len(x), dtype=bool)
        isNew[np.searchsorted(x, mid[bad])] = True
        open_ = isNew[1:] | isNew[:-1]
    return x, y


def _cumulative(x, p, histogram: bool) -> np.ndarray:
    c = np.zeros_like(x)
    if len(x) > 1:
        dx = np.diff(x)
        c[1:] = np.cumsum(p[:-1] * dx if histogram else 0.5 * (p[:-1] + p[1:]) * dx)
    return c


def normalisedTable(x, p, *, histogram: bool = False, options: Optional[TabulationOptions] = None,
                    discrete: int = 0, what: str = "table"):
    """``(x, pdf, cdf, clipped)``: the density normalised to 1, with its cumulative.

    The first ``discrete`` points are lines (weights in ``p``), followed by the
    continuum; the cumulative runs through the lines and then the continuum,
    as ACE law 4 and OpenMC's ``Mixture`` lay them out. ``clipped`` says whether
    negative densities were set to zero.

    Parameters
    ----------
    x, p
        Abscissae and density (weights for the first *discrete* points), of one
        length. The continuum is integrated lin-lin, or as a histogram.
    histogram
        If True the density is constant from each point to the next.
    options : TabulationOptions, optional
        Only ``negative`` is used. None is ``TabulationOptions()``.
    discrete
        Number of leading points that are lines.
    what
        Name of the table, used in error messages.

    Returns
    -------
    tuple
        ``(x, pdf, cdf, clipped)``: *x* as given, the density divided by the total
        (lines plus continuum), the cumulative at each point (ending at exactly 1;
        a line's own weight is included at its point) and a bool.

    Raises
    ------
    ValueError
        If a density is negative and ``options.negative`` is ``"raise"``, or the
        density integrates to zero.
    """
    options = options or TabulationOptions()
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float).copy()
    clipped = bool(np.any(p < 0))
    if clipped:
        if options.negative == "raise":
            raise ValueError(f"{what}: negative density {p.min():.3e}")
        p[p < 0] = 0.0
    lines = p[:discrete]
    cx, cp = x[discrete:], p[discrete:]
    continuum = _cumulative(cx, cp, histogram)
    total = float(lines.sum()) + (float(continuum[-1]) if continuum.size else 0.0)
    if not total > 0:
        raise ValueError(f"{what}: the density integrates to zero")
    cdf = np.concatenate([np.cumsum(lines), lines.sum() + continuum]) / total
    if discrete:
        # A line's cumulative includes its own weight; the continuum starts there.
        cdf[discrete:] = (lines.sum() + continuum) / total
    p = p / total
    cdf[-1] = 1.0
    return x, p, cdf, clipped


def pdfTable(xs, pdf, *, histogram=False, options=None, discrete=0, outer=None, what="table"):
    """An ``Xs_pdf_cdf1d`` from a density, normalised, with its cumulative.

    Parameters
    ----------
    xs, pdf, histogram, options, discrete, what
        As in :func:`normalisedTable`.
    outer
        The outer coordinate of the table (the incident energy in eV, say), or
        None.

    Returns
    -------
    tuple
        ``(table, clipped)``: the model table, linear-linear or flat (histogram)
        and with ``numberOfDiscrete`` set to *discrete*; and whether negative
        densities were set to zero.

    Raises
    ------
    ValueError
        As :func:`normalisedTable`.
    """
    from kika.nuclear_data.model import Interpolation, Xs_pdf_cdf1d

    x, p, c, clipped = normalisedTable(xs, pdf, histogram=histogram, options=options,
                                       discrete=discrete, what=what)
    table = Xs_pdf_cdf1d(xs=x, pdf=p, cdf=c,
                         interpolation=Interpolation.flat if histogram else Interpolation.linlin,
                         numberOfDiscrete=discrete,
                         outerDomainValue=None if outer is None else float(outer))
    return table, clipped
