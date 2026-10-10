"""Two tables compared point by point, and a table against its own averages.

A comparison reads both operands on one set of abscissae and takes their
difference there. Here that is done three ways:

* two tables against each other (:func:`read_pair`, then :func:`difference`),
  on the union of their grids or on one of them
  (:func:`~kika.algebra.arithmetic.on_common_grid`);
* a table against window averages (:func:`window_difference`): the table is
  read at each window's centre, by its own law, and compared value by value;
* a table against group averages (:func:`steps_difference`): inside each group
  the table, on its own grid, against that group's constant. The result is as
  spiky as the table is -- it says how far the function strays from its mean.
  Each group edge is a step of the averages, so it appears twice in the result,
  once with the group below and once with the group above.

A table here may have holes: a non-finite ordinate is a point the table does
not have, not a number. It is read without it, and every result in an interval
that touches it is ``nan`` -- what ``numpy.interp`` gave the comparison before
it moved onto this package.

*method* names the law a table is read with, from the reader's side
(:func:`~kika.algebra.laws.method_laws`). A relative difference divides by the
reference, and is ``nan`` where the reference is zero.
"""
from __future__ import annotations

from typing import Literal, Tuple, Union

import numpy as np

from .arithmetic import on_common_grid
from .evaluate import evaluate, sample_on_union
from .grid import occurrences, union
from .laws import LOG_X, METHODS, method_laws

__all__ = ["interpolate_to_grid", "read_pair", "difference",
           "window_difference", "steps_difference"]

Mode = Literal["relative", "absolute"]
Side = Literal["reference", "comparison"]


def _finite_table(x, y, method: str):
    """The table without its holes, read under *method*; and the mask of what was kept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x and y must be 1-d arrays of one length, got {x.shape} and {y.shape}")
    finite = np.isfinite(y)
    xf, yf = x[finite], y[finite]
    if METHODS.get(method) in LOG_X and np.any(xf <= 0):
        raise ValueError(f"method={method!r} requires positive x values. "
                         f"Source x range: [{x.min()}, {x.max()}]")
    return (xf, yf, method_laws(yf, method)), finite


def _blank_holes(values: np.ndarray, x: np.ndarray, finite: np.ndarray, u: np.ndarray) -> None:
    """Set to ``nan`` every value of *values* (read at *u*) in an interval that touches a hole."""
    if finite.all() or u.size == 0:
        return
    bad = ~finite
    # The interval [x_k, x_k+1] around each point of u, by the table's own nodes.
    k = np.clip(np.searchsorted(x, u, side='right') - 1, 0, x.size - 1)
    touching = bad[k] | bad[np.minimum(k + 1, x.size - 1)]
    values[touching & (u != x[k])] = np.nan
    values[bad[k] & (u == x[k])] = np.nan


def interpolate_to_grid(x_target, x_source, y_source, method: str = 'log-log',
                        fill_value: float = np.nan) -> np.ndarray:
    """The table ``(x_source, y_source)`` read at *x_target* under *method*.

    A repeated source abscissa is a step, and a repeated target abscissa in a
    non-decreasing *x_target* reads its left then its right limit
    (:func:`~kika.algebra.evaluate.sample_on_union`). Targets outside the
    source range get *fill_value*, ``nan`` by default so they are seen; a
    target in an interval that touches a hole of the source is ``nan``.
    *method* is ``'log-log'``, ``'lin-lin'``, ``'log-lin'`` (log x, linear y)
    or ``'lin-log'`` (linear x, log y); intervals where a log y has no value
    are read linear in y (:func:`~kika.algebra.laws.method_laws`).
    """
    x_src = np.asarray(x_source, dtype=float)
    y_src = np.asarray(y_source, dtype=float)
    x_tgt = np.asarray(x_target, dtype=float)
    if len(x_src) != len(y_src):
        raise ValueError(f"x_source and y_source must have the same length. "
                         f"Got {len(x_src)} and {len(y_src)}")
    result = np.full_like(x_tgt, fill_value, dtype=float)
    if x_src.size == 0:
        return result
    in_range = (x_tgt >= x_src[0]) & (x_tgt <= x_src[-1])
    if not np.any(in_range):
        return result
    if METHODS.get(method) in LOG_X and np.any(x_tgt[in_range] <= 0):
        raise ValueError(f"method={method!r} requires positive x values. "
                         f"Source x range: [{x_src.min()}, {x_src.max()}]")
    (xf, yf, laws), finite = _finite_table(x_src, y_src, method)
    q = x_tgt[in_range]
    if xf.size == 0:
        values = np.full(q.size, np.nan)
    elif np.all(np.diff(q) >= 0):
        values = sample_on_union(xf, yf, laws, q)
    else:
        values = evaluate(xf, yf, laws, q)
    _blank_holes(values, x_src, finite, q)
    result[in_range] = values
    return result


def read_pair(first: Tuple, second: Tuple, method: str = 'lin-lin',
              grid: Union[str, int] = 'union') -> Tuple[np.ndarray, np.ndarray]:
    """Two tables ``(x, y)`` on one grid, over the span they share.

    *grid* is ``'union'`` or the index (0 or 1) of the table whose abscissae
    are used (:func:`~kika.algebra.arithmetic.on_common_grid`, which says what
    each choice can miss). Returns ``(u, values)``, ``values`` of shape
    ``(2, u.size)``, with ``nan`` wherever a table has a hole.
    """
    tables, holes = [], []
    for x, y in (first, second):
        table, finite = _finite_table(x, y, method)
        tables.append(table)
        holes.append((np.asarray(x, dtype=float), finite))
    u, values = on_common_grid(tables, grid)
    for row, (x, finite) in zip(values, holes):
        _blank_holes(row, x, finite, u)
    return u, values


def difference(reference, comparison, mode: Mode = 'relative',
               percent: bool = False) -> np.ndarray:
    """``comparison - reference``, or that over ``reference``, element by element.

    ``nan`` where either is not finite, and for a relative difference where the
    reference is zero. *percent* multiplies a relative difference by 100.
    """
    ref = np.asarray(reference, dtype=float)
    cmp_ = np.asarray(comparison, dtype=float)
    if ref.shape != cmp_.shape:
        raise ValueError(f"operands of different shapes: {ref.shape} and {cmp_.shape}")
    valid = np.isfinite(ref) & np.isfinite(cmp_)
    out = np.full(ref.shape, np.nan)
    if mode == 'relative':
        ok = valid & (ref != 0)
        out[ok] = (cmp_[ok] - ref[ok]) / ref[ok]
        if percent:
            out *= 100.0
    elif mode == 'absolute':
        out[valid] = cmp_[valid] - ref[valid]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return out


def _ordered(averaged, pointwise, average_is: Side):
    if average_is == 'reference':
        return averaged, pointwise
    if average_is == 'comparison':
        return pointwise, averaged
    raise ValueError(f"average_is must be 'reference' or 'comparison', got {average_is!r}")


def window_difference(centres, averages, x, y, method: str = 'lin-lin', *,
                      average_is: Side = 'reference', mode: Mode = 'relative',
                      percent: bool = False) -> np.ndarray:
    """Window averages against a table, at the windows' centres.

    The table ``(x, y)`` is read at each centre under *method*
    (:func:`interpolate_to_grid`) and the difference is taken value by value;
    *average_is* says which operand is the reference. One value per centre,
    ``nan`` at a centre off the table. On a constant table, whose average is
    itself, it is zero.
    """
    centres = np.asarray(centres, dtype=float)
    averages = np.asarray(averages, dtype=float)
    if centres.shape != averages.shape:
        raise ValueError("one average per centre")
    pointwise = interpolate_to_grid(centres, x, y, method)
    return difference(*_ordered(averages, pointwise, average_is), mode, percent)


def steps_difference(edges, averages, x, y, method: str = 'lin-lin', *,
                     average_is: Side = 'reference', mode: Mode = 'relative',
                     percent: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Group averages against a table, on the table's own grid.

    Inside each group ``[edges[g], edges[g+1]]`` the table is read at its own
    abscissae (and at the group's ends) and compared with that group's
    constant ``averages[g]``. Returns ``(u, diff)`` over the span the groups
    and the table share; an inner edge appears twice in ``u``, its first copy
    against the group below and the second against the group above, so the
    difference steps there as the averages do. A group whose average is not
    finite gives ``nan``.
    """
    edges = np.asarray(edges, dtype=float)
    averages = np.asarray(averages, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or averages.shape != (edges.size - 1,):
        raise ValueError("need n + 1 edges for n averages")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be strictly increasing")
    x = np.asarray(x, dtype=float)
    (xf, yf, laws), finite = _finite_table(x, y, method)
    empty = np.zeros(0), np.zeros(0)
    if xf.size < 2:
        return empty
    lo, hi = max(edges[0], xf[0]), min(edges[-1], xf[-1])
    if not lo < hi:
        return empty
    inner = edges[(edges > lo) & (edges < hi)]
    u = union([xf[(xf > lo) & (xf < hi)], [lo, hi]], steps=inner)
    pointwise = sample_on_union(xf, yf, laws, u)
    _blank_holes(pointwise, x, finite, u)
    group = np.searchsorted(edges, u, side='right') - 1
    # The first copy of an inner edge closes the group below it.
    group[np.isin(u, inner) & (occurrences(u) == 0)] -= 1
    steps = averages[np.clip(group, 0, averages.size - 1)]
    return u, difference(*_ordered(steps, pointwise, average_is), mode, percent)
