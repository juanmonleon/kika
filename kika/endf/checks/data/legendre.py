"""Is a Legendre series a probability density? Vectorised over many series at once.

``f(mu) = sum_l (2l+1)/2 a_l P_l(mu)`` with ``a_0 = 1`` integrates to 1 over
[-1, 1]; it is a density only if it is nowhere negative. ``|a_l| <= 1`` is
necessary (|P_l| <= 1) and checked exactly; positivity is checked on a grid of
cosines dense enough for the order -- the Chebyshev points of degree
``max(128, 8 L)`` plus both ends -- and measured by the **negative area**,
``integral of max(-f, 0) d mu``: the probability a sampler would have to
throw away or clip. That is what PSYCHE's Sturm test and LEGEND's scan look
for, said as a number.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["negative_area", "mu_grid"]

_GRIDS = {}


def mu_grid(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(mu, w)``: cosines from -1 to 1 and trapezoid weights on them."""
    n = max(128, 8 * int(order))
    if n not in _GRIDS:
        k = np.arange(n + 1)
        mu = -np.cos(np.pi * k / n)          # Chebyshev-Lobatto, ends included
        w = np.zeros_like(mu)
        d = np.diff(mu)
        w[:-1] += d / 2
        w[1:] += d / 2
        _GRIDS[n] = (mu, w)
    return _GRIDS[n]


def negative_area(coeffs: np.ndarray, *, chunk: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """``(area, f_min)`` per row of *coeffs* (``a_1..a_L``, ``a_0 = 1`` implied).

    *area* is the integral of the negative part of each series, *f_min* its
    lowest value on the grid (both 0 for a non-negative series).
    """
    from numpy.polynomial import legendre as L

    a = np.atleast_2d(np.asarray(coeffs, dtype=float))
    n, order = a.shape
    if n == 0:
        return np.zeros(0), np.zeros(0)
    mu, w = mu_grid(order)
    scale = (2 * np.arange(order + 1) + 1) / 2.0
    vander = L.legvander(mu, order) * scale          # (n_mu, order+1)
    area = np.empty(n)
    fmin = np.empty(n)
    for s in range(0, n, chunk):
        block = a[s:s + chunk]
        full = np.hstack([np.ones((block.shape[0], 1)), block])
        f = full @ vander.T                          # (rows, n_mu)
        fmin[s:s + chunk] = f.min(axis=1)
        area[s:s + chunk] = np.maximum(-f, 0.0) @ w
    return area, np.minimum(fmin, 0.0)
