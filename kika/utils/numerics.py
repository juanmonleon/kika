"""Format-agnostic numerical primitives.

Pure mathematics on tabulated data: no nuclear physics, no file formats, no MT
numbers.  Anything in the library or in ``scripts/`` that needs to convolve a
tabulated function with a Gaussian, or average it over intervals, should call
these rather than growing its own copy.

**The Gaussian fold is exact.**  A tabulated function is a straight line on
every panel, and the integral of a straight line against a Gaussian has a
closed form in the normal CDF :math:`\\Phi` and density :math:`\\varphi`.  So
:func:`fold_tabulated` and :func:`gaussian_fold_nodes` compute
:math:`\\int y(t)\\,N(t; x_0, \\sigma^2)\\,dt` with no quadrature error at all,
for the piecewise-linear interpolant of the nodes they are given.

That replaced, in October 2026, three quadratures that had each been "the" fold
at some point and disagreed with one another: twelve (and, for plots, 21)
Gauss-Hermite nodes, which do not know where the table has structure and missed
Fe-56 elastic by 4 % median and up to 55-78 %; a Gaussian-weighted average of
the tabulated points with no ``dx`` measure; and a Gaussian x trapezoid rule on
the table's points plus 101 uniform ones, which was the best of them and still
off by up to 6e-4.  Only the exact one is kept: two rules for the same integral
is how the three came to disagree in the first place.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from scipy.special import ndtr

__all__ = [
    "gaussian_fold_nodes",
    "fold_tabulated",
    "average_over_intervals",
]

#: Half-width of a fold window, in kernel standard deviations.  The Gaussian
#: beyond 6 sigma carries 1e-9 of its weight, and even that is not dropped:
#: it is given to the outermost node, i.e. the function is held constant past
#: the window, which is also what happens past the end of the table.
FOLD_HALF_WIDTH_SIGMAS = 6.0

#: Centres folded per vectorised batch by :func:`fold_tabulated`, bounded by the
#: number of (centre, node) pairs rather than by centres, so a dense table and a
#: wide kernel cannot build a multi-gigabyte array.
_FOLD_BATCH_PAIRS = 2_000_000

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _panel_terms(nodes, centre, sigma):
    r"""Exact Gaussian integrals over every panel between consecutive *nodes*.

    Returns ``(mass, to_right, lower, upper)``. On a panel ``[a, b]`` of width
    *h*, with ``z = (t - c)/s``,

    .. math::
        \text{mass} = \int_a^b N = \Delta\Phi, \qquad
        \text{to\_right} = \int_a^b \frac{t-a}{h} N
            = \frac{(c-a)\,\Delta\Phi + s\,(\varphi(z_a)-\varphi(z_b))}{h},

    so a straight line from ``y_a`` to ``y_b`` integrates to
    ``y_a (mass - to_right) + y_b to_right``. A panel of zero width (a repeated
    abscissa, ENDF's step) has neither, so a discontinuity is exact. ``lower``
    and ``upper`` are the Gaussian mass left and right of each node, for the
    tails. *centre* and *sigma* are per node, so one call can cover several
    centres laid end to end; the panels that straddle two of them are garbage
    and are the caller's to discard.
    """
    z = (nodes - centre) / sigma
    lower = ndtr(z)
    upper = 1.0 - lower
    tail = z > 0
    upper[tail] = ndtr(-z[tail])
    phi = np.exp(-0.5 * z * z)
    mass = np.diff(lower)
    # Right of the centre, difference the upper tails instead: two CDF values
    # next to one would lose the far panels to cancellation.
    right = np.flatnonzero(tail[:-1] & tail[1:])
    mass[right] = upper[right] - upper[right + 1]
    width = np.diff(nodes)
    numerator = ((centre[:-1] - nodes[:-1]) * mass
                 + (sigma[:-1] * _INV_SQRT_2PI) * (phi[:-1] - phi[1:]))
    to_right = np.divide(numerator, width, out=np.zeros_like(numerator),
                         where=width > 0)
    return mass, to_right, lower, upper


def gaussian_fold_nodes(
    x0: float,
    sigma: float,
    grids: Sequence[Sequence[float]],
    *,
    half_width_sigmas: float = FOLD_HALF_WIDTH_SIGMAS,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Nodes and exact weights for averaging a function over a Gaussian.

    For callers whose integrand is not one table but something evaluated at
    each energy -- an angular distribution, the product :math:`\sigma f(\mu)` --
    so the fold has to be a weighted sum of evaluations.

    The nodes are every point of every grid in ``grids`` strictly inside
    :math:`x_0 \pm` ``half_width_sigmas`` :math:`\sigma`, plus the two window
    edges and ``x0`` itself, so a caller can read the unfolded value off the
    same evaluations.  The weights are exact for the integrand's piecewise-linear
    interpolant on those nodes (see :func:`_panel_terms`) and sum to one, so
    :math:`\langle y\rangle = \sum_i w_i\,y(x_i)` has no quadrature error for
    any integrand that is linear between the breakpoints passed in.

    ``grids`` must therefore hold the breakpoints of everything the integrand
    is made of -- the cross-section grid *and* the angular-distribution grid
    for :math:`\sigma f`.  For a product of two lin-lin tables that is still an
    approximation, of the quadratic term inside each panel of the union grid,
    which is second order in the panel width.  An empty ``grids`` would make
    the integrand a single straight line across the window and is refused.

    ``sigma <= 0`` yields the single point ``x0`` with weight 1.
    """
    x0 = float(x0)
    s = float(sigma)
    if not (s > 0.0):
        return np.array([x0]), np.array([1.0])
    if len(grids) == 0:
        raise ValueError("gaussian_fold_nodes needs the breakpoints of the integrand")

    lo = x0 - half_width_sigmas * s
    hi = x0 + half_width_sigmas * s
    parts = [np.array([lo, x0, hi])]
    for grid in grids:
        g = np.asarray(grid, dtype=float)
        if g.size:
            i0 = int(np.searchsorted(g, lo, side="right"))
            i1 = int(np.searchsorted(g, hi, side="left"))
            parts.append(g[i0:i1])
    nodes = np.unique(np.concatenate(parts))
    mass, to_right, lower, upper = _panel_terms(
        nodes, np.full(nodes.size, x0), np.full(nodes.size, s))
    weights = np.zeros_like(nodes)
    weights[:-1] += mass - to_right
    weights[1:] += to_right
    weights[0] += lower[0]
    weights[-1] += upper[-1]
    return nodes, weights


def fold_tabulated(
    x: np.ndarray,
    y: np.ndarray,
    x0: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""Average a lin-lin table over a Gaussian kernel, exactly.

    Computes :math:`\langle y \rangle = \int y(t)\,N(t; x_0, \sigma^2)\,dt`
    with ``y`` the lin-lin interpolant of ``(x, y)``, held at its end values
    outside the table (``numpy.interp`` semantics: a threshold reaction that
    starts at zero folds to zero below it).  The integral is the closed form of
    :func:`_panel_terms` on the table's own points inside the window, so it
    does not depend on how finely the table is sampled, and a step written as
    a repeated abscissa is integrated as a step.

    Vectorised over centroids, in batches bounded by the number of
    (centroid, table point) pairs.

    Parameters
    ----------
    x, y : np.ndarray
        Tabulated function, ``x`` ascending (repeated values allowed: a step).
        Units are the caller's; ``x0`` and ``sigma`` must be in the same units.
    x0 : float or np.ndarray
        Kernel centroid(s).
    sigma : float or np.ndarray
        Kernel standard deviation(s), broadcast against ``x0``.  Where
        ``sigma <= 0`` the kernel is a delta and ``y(x0)`` is returned -- for a
        sharply peaked ``y`` a point sample is usually *not* what you want;
        consider :func:`average_over_intervals` instead.

    Returns
    -------
    float or np.ndarray
        Folded value(s); scalar in, scalar out.

    Examples
    --------
    A straight line folds to its value at the centroid, whatever the width:

    >>> xs = np.linspace(0.0, 10.0, 11)
    >>> round(float(fold_tabulated(xs, 2.0 + 0.5 * xs, 5.0, 1.0)), 12)
    4.5
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    scalar_in = np.ndim(x0) == 0
    x0_arr = np.atleast_1d(np.asarray(x0, dtype=float))
    sigma_arr = np.broadcast_to(
        np.atleast_1d(np.asarray(sigma, dtype=float)), x0_arr.shape,
    ).astype(float)

    out = np.interp(x0_arr, x, y)          # the delta kernel, and the fallback
    live = np.flatnonzero(sigma_arr > 0.0)
    if x.size < 2 or live.size == 0:
        return float(out[0]) if scalar_in else out
    c, s = x0_arr[live], sigma_arr[live]
    # The table points whose panels reach into +-FOLD_HALF_WIDTH_SIGMAS, at
    # least one panel, so a centre off the end of the table folds to the
    # held end value through the tails below.
    first = np.clip(np.searchsorted(x, c - FOLD_HALF_WIDTH_SIGMAS * s, side="right") - 1,
                    0, x.size - 2)
    last = np.clip(np.searchsorted(x, c + FOLD_HALF_WIDTH_SIGMAS * s, side="left"),
                   first + 1, x.size - 1)
    count = last - first + 1
    running = np.cumsum(count)
    start = 0
    while start < live.size:
        done = int(running[start - 1]) if start else 0
        stop = int(np.searchsorted(running, done + _FOLD_BATCH_PAIRS, side="right"))
        stop = min(max(stop, start + 1), live.size)
        n = count[start:stop]
        begin = np.cumsum(n) - n                       # first node of each centre
        index = np.arange(n.sum()) - np.repeat(begin, n) + np.repeat(first[start:stop], n)
        mass, to_right, lower, upper = _panel_terms(
            x[index], np.repeat(c[start:stop], n), np.repeat(s[start:stop], n))
        yi = y[index]
        panels = yi[:-1] * (mass - to_right) + yi[1:] * to_right
        panels[begin[1:] - 1] = 0.0                    # straddles two centres
        end = begin + n - 1
        value = np.add.reduceat(panels, begin) if panels.size else np.zeros(n.size)
        out[live[start:stop]] = value + yi[begin] * lower[begin] + yi[end] * upper[end]
        start = stop
    return float(out[0]) if scalar_in else out


def average_over_intervals(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    *,
    n_sub: int = 200,
) -> np.ndarray:
    """Average a tabulated function over each interval of ``edges``.

    Trapezoid integral of the linearly-interpolated ``y(x)`` over
    ``[edges[i], edges[i+1]]``, divided by the interval width.

    Each interval is sampled on the union of a uniform sub-grid and the native
    ``x`` points falling inside it.  The union matters: with uniform sampling
    alone, a feature narrower than ``(hi - lo) / n_sub`` — a resonance on a
    finely tabulated cross section, say — can be stepped over entirely.

    Parameters
    ----------
    x, y : np.ndarray
        Tabulated function, ``x`` ascending.
    edges : np.ndarray
        Interval boundaries, ``N + 1`` ascending values.
    n_sub : int, default 200
        Uniform sub-samples per interval, before the union with native points.

    Returns
    -------
    np.ndarray
        Interval averages, shape ``(N,)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.asarray(edges, dtype=float)

    out = np.empty(len(edges) - 1, dtype=float)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        inside = x[(x > lo) & (x < hi)]
        x_sub = np.unique(np.concatenate([np.linspace(lo, hi, n_sub), inside]))
        y_sub = np.interp(x_sub, x, y)
        out[i] = np.trapezoid(y_sub, x_sub) / (hi - lo)
    return out
