"""Format-agnostic numerical primitives.

Pure mathematics on tabulated data: no nuclear physics, no file formats, no MT
numbers.  Anything in the library or in ``scripts/`` that needs to convolve a
tabulated function with a Gaussian, or average it over intervals, should call
these rather than growing its own copy.

That had happened three times over for the Gaussian convolution alone, with
three different quadratures — Gauss-Hermite on the interpolant, a
Gaussian-weighted average of the *tabulated points*, and 21 uniform samples —
which disagreed by up to 21% on a resonant cross section.  The point-weighted
variant is the one to avoid: it carries no ``dx`` measure, so on an adaptively
refined grid (dense exactly where the function is large) it over-weights the
peaks.  :func:`fold_tabulated` integrates the interpolant instead and is
therefore insensitive to how the input happens to be sampled.
"""
from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np

__all__ = [
    "gauss_hermite_nodes",
    "gaussian_fold_nodes",
    "fold_tabulated",
    "average_over_intervals",
]

#: Half-width of a fold window, in kernel standard deviations.  The Gaussian
#: beyond 5 sigma carries 6e-7 of its weight.
FOLD_HALF_WIDTH_SIGMAS = 5.0

#: Uniform points laid across a fold window on top of the data's own points, so
#: a stretch where the table is sparse is still sampled finely enough for the
#: Gaussian's own curvature.
FOLD_UNIFORM_POINTS = 101


def gauss_hermite_nodes(
    x0: float,
    sigma: float,
    n_nodes: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Sample points and normalised weights for averaging over a Gaussian.

    Returns ``(points, weights)`` with ``weights`` summing to 1, such that
    ``sum(weights * f(points))`` approximates
    :math:`\int f(x)\,N(x; x_0, \sigma^2)\,dx`.

    Exposed separately from :func:`fold_tabulated` for callers whose integrand
    is not a tabulated array — folding an angular distribution, say, where each
    node requires re-evaluating a whole curve.  Sharing the nodes is what keeps
    those callers on the same quadrature instead of reinventing a uniform-sample
    Riemann sum.

    Parameters
    ----------
    x0 : float
        Kernel centroid.
    sigma : float
        Kernel standard deviation.  ``sigma <= 0`` yields the single point
        ``x0`` with weight 1.
    n_nodes : int, default 12
        Number of nodes.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Points and weights, both shape ``(n_nodes,)``.
    """
    if sigma <= 0.0 or n_nodes < 1:
        return np.array([float(x0)]), np.array([1.0])
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    return x0 + np.sqrt(2.0) * sigma * nodes, weights / np.sqrt(np.pi)


def gaussian_fold_nodes(
    x0: float,
    sigma: float,
    grids: Sequence[Sequence[float]] = (),
    *,
    half_width_sigmas: float = FOLD_HALF_WIDTH_SIGMAS,
    n_uniform: int = FOLD_UNIFORM_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Nodes and normalised weights for averaging tabulated data over a Gaussian.

    The nodes are every point of every grid in ``grids`` that falls strictly
    inside :math:`x_0 \pm` ``half_width_sigmas`` :math:`\sigma`, the two window
    edges, and ``n_uniform`` evenly spaced points across it.  The weights are the
    Gaussian times the trapezoid rule on those nodes, normalised to sum to one,
    so :math:`\langle y\rangle \approx \sum_i w_i\,y(x_i)`.

    Why the data's own points.  A tabulated function is piecewise linear between
    its points, so with a node on every breakpoint the rule is exact for it up to
    the Gaussian's curvature inside one interval.  Gauss-Hermite nodes
    (:func:`gauss_hermite_nodes`) do not know where the breakpoints are: above
    the resolved range an evaluated cross section has structure on a scale finer
    than a TOF kernel -- Fe-56 elastic in JEFF-4.0 has ~120 MF3 points inside
    :math:`\pm 3\sigma_E` at 1 MeV for a 27 m / 5 ns geometry -- and twelve
    Gauss-Hermite nodes miss that average by 28 %, ninety-six by 2 %.

    ``sigma <= 0`` yields the single point ``x0`` with weight 1.
    """
    x0 = float(x0)
    s = float(sigma)
    if not (s > 0.0):
        return np.array([x0]), np.array([1.0])

    lo = x0 - half_width_sigmas * s
    hi = x0 + half_width_sigmas * s
    parts = [np.linspace(lo, hi, max(int(n_uniform), 2))]
    for grid in grids:
        g = np.asarray(grid, dtype=float)
        if g.size:
            i0 = int(np.searchsorted(g, lo, side="right"))
            i1 = int(np.searchsorted(g, hi, side="left"))
            parts.append(g[i0:i1])
    nodes = np.unique(np.concatenate(parts))

    gauss = np.exp(-0.5 * ((nodes - x0) / s) ** 2)
    spans = np.diff(nodes)
    trap = np.zeros_like(nodes)
    trap[:-1] += 0.5 * spans
    trap[1:] += 0.5 * spans
    weights = gauss * trap
    return nodes, weights / weights.sum()


def fold_tabulated(
    x: np.ndarray,
    y: np.ndarray,
    x0: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    *,
    method: Literal["grid", "gauss-hermite"] = "grid",
    n_nodes: int = 12,
) -> Union[float, np.ndarray]:
    r"""Average a tabulated function over a Gaussian kernel.

    Computes :math:`\langle y \rangle = \int y(x')\,N(x'; x_0, \sigma^2)\,dx'`
    with ``y`` evaluated by linear interpolation on ``(x, y)`` and clamped to
    the table endpoints outside its coverage (``numpy.interp`` default).

    ``method="grid"`` (the default since September 2026) integrates on the
    table's own points inside the window, see :func:`gaussian_fold_nodes`.
    ``method="gauss-hermite"`` is the previous rule,

    .. math::
        \langle y \rangle = \frac{1}{\sqrt{\pi}} \sum_i w_i\,
                            y\!\left(x_0 + \sqrt{2}\,\sigma\,t_i\right),

    exact for the Gaussian weight but blind to where the table has structure:
    it is kept only so results computed with it can be reproduced.

    Parameters
    ----------
    x, y : np.ndarray
        Tabulated function, ``x`` ascending.  Units are the caller's; ``x0`` and
        ``sigma`` must be in the same units as ``x``.
    x0 : float or np.ndarray
        Kernel centroid(s).
    sigma : float or np.ndarray
        Kernel standard deviation(s).  Broadcast against ``x0``.  Where
        ``sigma <= 0`` the kernel collapses to a delta and the interpolated
        value ``y(x0)`` is returned — note that for a sharply peaked ``y`` a
        point sample is usually *not* what you want; consider
        :func:`average_over_intervals` instead.
    method : {"grid", "gauss-hermite"}, default "grid"
        The quadrature.  See above.
    n_nodes : int, default 12
        Number of Gauss-Hermite nodes; read only with ``method="gauss-hermite"``.

    Returns
    -------
    float or np.ndarray
        Folded value(s); scalar in, scalar out.

    Examples
    --------
    A flat function folds to itself whatever the resolution:

    >>> xs = np.linspace(0.0, 10.0, 101)
    >>> float(fold_tabulated(xs, np.full_like(xs, 3.0), 5.0, 1.0))
    3.0
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0_arr = np.atleast_1d(np.asarray(x0, dtype=float))
    sigma_arr = np.broadcast_to(
        np.atleast_1d(np.asarray(sigma, dtype=float)), x0_arr.shape,
    )
    scalar_in = np.isscalar(x0) or np.ndim(x0) == 0

    out = np.empty(x0_arr.shape, dtype=float)
    if method == "grid":
        for i, (c, s) in enumerate(zip(x0_arr, sigma_arr)):
            nodes, weights = gaussian_fold_nodes(c, s, (x,))
            out[i] = weights @ np.interp(nodes, x, y)
        return float(out[0]) if scalar_in else out
    if method != "gauss-hermite":
        raise ValueError(f"unknown fold method {method!r}")

    if n_nodes < 1:
        out[:] = np.interp(x0_arr, x, y)
        return float(out[0]) if scalar_in else out

    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    norm = np.sqrt(np.pi)
    for i, (c, s) in enumerate(zip(x0_arr, sigma_arr)):
        if s <= 0.0:
            out[i] = np.interp(c, x, y)
        else:
            out[i] = np.sum(weights * np.interp(c + np.sqrt(2.0) * s * nodes, x, y)) / norm

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
