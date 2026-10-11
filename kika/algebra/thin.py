"""Thinning several tables that share one grid, to a stated error.

The inverse of refinement: given rows ``y[k, :]`` tabulated on one
non-decreasing grid ``x`` and read lin-lin, drop grid points while every row,
interpolated between the points kept, still reproduces **every** point it
loses to ``|y_interp - y| <= rtol*|y| + atol``. The values at the points kept
are never changed, so the thinned table is a subset of the original one.

The sweep is greedy from the left: from each kept point the farthest next
point is found by doubling the step and then bisecting, each candidate tested
on all rows at once. Points named in ``keep`` (and the two ends) are never
dropped, and no interval spans one.

This is what NJOY's BROADR/RECONR thinning does with ``errthn``; the error
here is measured at every dropped point of every row rather than estimated.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

__all__ = ["thin_common_grid"]


def _fits(x, ys, i, j, rtol, atol):
    """Rows read lin-lin from ``i`` to ``j`` reproduce every point strictly between."""
    if j <= i + 1:
        return True
    xi, xj = x[i], x[j]
    inner = x[i + 1:j]
    if xj == xi:
        return False
    t = (inner - xi) / (xj - xi)
    chord = ys[:, i:i + 1] + (ys[:, j:j + 1] - ys[:, i:i + 1]) * t
    actual = ys[:, i + 1:j]
    return bool(np.all(np.abs(chord - actual) <= rtol * np.abs(actual) + atol))


def thin_common_grid(x, ys, rtol: float = 1e-3, atol: float = 0.0,
                     keep: Optional[Iterable[int]] = None) -> np.ndarray:
    """Indices of the points of ``x`` to keep. ``ys`` is ``(rows, len(x))``.

    ``atol`` is in the units of ``y`` and may be an array with one value per
    row (a floor for each reaction).
    """
    x = np.asarray(x, dtype=float)
    ys = np.atleast_2d(np.asarray(ys, dtype=float))
    n = x.size
    if ys.shape[1] != n:
        raise ValueError(f"rows have {ys.shape[1]} points and the grid {n}")
    atol = np.asarray(atol, dtype=float)
    if atol.ndim == 1:
        atol = atol[:, None]
    if n <= 2:
        return np.arange(n)
    mandatory = np.zeros(n, dtype=bool)
    mandatory[[0, n - 1]] = True
    if keep is not None:
        mandatory[np.asarray(list(keep), dtype=int)] = True
    nextMandatory = np.empty(n, dtype=int)
    upcoming = n - 1
    for k in range(n - 1, -1, -1):
        nextMandatory[k] = upcoming
        if mandatory[k]:
            upcoming = k
    kept = [0]
    i = 0
    while i < n - 1:
        limit = nextMandatory[i]
        step = 1
        good = i + 1
        while True:
            candidate = min(i + 2 * step, limit)
            if candidate <= good:
                break
            if _fits(x, ys, i, candidate, rtol, atol):
                good = candidate
                if candidate == limit:
                    break
                step *= 2
            else:
                lo, hi = good, candidate
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if _fits(x, ys, i, mid, rtol, atol):
                        lo = mid
                    else:
                        hi = mid
                good = lo
                break
        kept.append(good)
        i = good
    return np.asarray(kept, dtype=int)
