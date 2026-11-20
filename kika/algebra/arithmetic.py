"""Sums of tables, exact between the nodes and at every step.

A sum of tables is lin-lin on the union of their grids only once every table
*is* lin-lin: two log-log partials summed at their union nodes and read lin-lin
are right at the nodes and up to 10 % wrong between them (8.1 % on B-10 of
ENDF/B-VIII.1). So each table goes through :func:`~kika.algebra.to_linlin`
first, and the sum is then exact up to that tolerance.

Each table is zero outside its own domain. One that starts or ends at a
non-zero value inside the range of the sum puts a step in the sum there; the
union grid carries that abscissa twice (:func:`~kika.algebra.union`'s
*steps*), and :func:`~kika.algebra.sample_on_union` gives its first
copy the left limit and its second the right one. Leaving it out turned the
step into a ramp across the neighbouring interval: +66 % in the case that found
it. FUDGE refuses the same situation (``nfu_domainsNotMutual``) unless told to
replace the step by a ramp of width ``eps``; a repeated abscissa needs no
width.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np

from .evaluate import sample_on_union
from .grid import union
from .laws import validate
from .refine import LINEARIZATION_TOLERANCE, to_linlin

__all__ = ["add", "domain_steps", "on_common_grid"]


def domain_steps(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> list:
    """The domain edges of a table that are steps inside ``[lo, hi]``.

    An edge is a step when the table's value there is not zero and the range
    carries on past it.
    """
    out = []
    if x.size:
        if x[0] > lo and y[0] != 0.0:
            out.append(float(x[0]))
        if x[-1] < hi and y[-1] != 0.0:
            out.append(float(x[-1]))
    return out


def add(tables: Sequence, coefficients: Optional[Sequence[float]] = None, *,
        tol: float = LINEARIZATION_TOLERANCE, snap: Optional[Callable] = None,
        grids: Sequence[np.ndarray] = ()):
    """``sum_i c_i f_i`` as one lin-lin table ``(x, y)``.

    *tables* are ``(x, y, laws)`` triples; *coefficients* default to one each.
    Every table is re-expressed lin-lin to *tol* (with *snap*, see
    :func:`~kika.algebra.refine`), and the result lives on the union of
    those grids, plus any extra *grids* the caller wants carried. Its range is
    the union of the tables' domains and of the extra grids' ranges; each table
    is zero off its own.

    Parameters
    ----------
    tables : sequence of (x, y, laws)
        The tables to sum: each with abscissae (non-decreasing), ordinates and
        one interpolation law per interval (codes 1-5, or one code for all).
    coefficients : sequence of float, optional
        One coefficient per table. None is 1 for every table.
    tol : float
        Relative tolerance of the lin-lin re-expression of each table
        (:func:`~kika.algebra.to_linlin`).
    snap : callable, optional
        As in :func:`~kika.algebra.refine`.
    grids : sequence of array_like
        Extra abscissae the result must carry; empty grids are ignored. The
        sum is zero where such a grid reaches past every table.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)``: the abscissae, non-decreasing, with a repeated abscissa at
        each step, and the sum there. Both are empty when no table or grid has
        a point.

    Raises
    ------
    ValueError
        If a table is not valid (:func:`~kika.algebra.validate`), or the
        number of *coefficients* differs from the number of tables.
    RefinementError
        If the lin-lin re-expression of a table does not converge
        (:func:`~kika.algebra.to_linlin`).
    """
    tables = [validate(*t) for t in tables]
    if coefficients is None:
        coefficients = [1.0] * len(tables)
    if len(coefficients) != len(tables):
        raise ValueError("one coefficient per table")
    linear = [to_linlin(x, y, laws, tol, snap=snap) for x, y, laws in tables]
    extra = [np.asarray(g, dtype=float) for g in grids if len(g)]
    spans = [lx for lx, _ in linear if lx.size] + extra
    if not spans:
        return np.zeros(0), np.zeros(0)
    lo = min(float(g[0]) for g in spans)
    hi = max(float(g[-1]) for g in spans)
    steps = [e for lx, ly in linear for e in domain_steps(lx, ly, lo, hi)]
    u = union([lx for lx, _ in linear] + extra, steps=steps)
    total = np.zeros(u.size)
    for c, (lx, ly) in zip(coefficients, linear):
        total += float(c) * sample_on_union(lx, ly, 2, u)
    return u, total


def on_common_grid(tables: Sequence, grid: Union[str, int] = "union"):
    """Several tables read on one grid, over the span they all cover.

    *tables* are ``(x, y, laws)`` triples, each read under its own laws.
    *grid* is ``"union"`` -- every abscissa of every table, steps kept
    (:func:`~kika.algebra.union`) -- or the index of the table whose own
    abscissae are used. Returns ``(u, values)`` with ``values`` of shape
    ``(len(tables), u.size)``; a repeated abscissa of *u* carries the left
    limit on its first copy and the right limit on its last
    (:func:`~kika.algebra.sample_on_union`).

    This is how two tables are compared point by point. On the union no
    table loses a node: the difference of two lin-lin tables is itself lin-lin
    there, so its extremes are among the returned points. On one table's grid
    the other is only sampled, and a peak of it that falls between the chosen
    nodes is not in the result at all.

    Parameters
    ----------
    tables : sequence of (x, y, laws)
        The tables to read: each with abscissae (non-decreasing), ordinates and
        one interpolation law per interval (codes 1-5, or one code for all).
        Each must have at least one point.
    grid : {"union"} or int
        ``"union"`` or the index of the table whose abscissae are used. Only
        the part inside the span shared by every table is kept.

    Returns
    -------
    tuple of numpy.ndarray
        ``(u, values)``: the abscissae, non-decreasing, and the readings, one
        row per table. A step at either end of the shared span appears once, as
        a value.

    Raises
    ------
    ValueError
        If a table is not valid (:func:`~kika.algebra.validate`); if
        *tables* is empty or holds an empty table; if the tables share no span
        of positive width; or if *grid* is a string other than ``"union"``.
    """
    tables = [validate(*t) for t in tables]
    if not tables:
        raise ValueError("no tables to read")
    if any(x.size == 0 for x, _, _ in tables):
        raise ValueError("an empty table has no domain to share")
    lo = max(float(x[0]) for x, _, _ in tables)
    hi = min(float(x[-1]) for x, _, _ in tables)
    if not lo < hi:
        raise ValueError(f"the tables share no span: [{lo!r}, {hi!r}]")
    if isinstance(grid, str):
        if grid != "union":
            raise ValueError(f"grid must be 'union' or a table index, got {grid!r}")
        u = union([x for x, _, _ in tables])
    else:
        u = tables[int(grid)][0]
    u = u[(u >= lo) & (u <= hi)]
    # A step at an end of the shared span would read its outer limit there --
    # zero, off some table's domain -- so the ends are kept once, as values.
    u = u[(np.r_[True, u[1:] != u[:-1]] | (u != lo)) & (np.r_[u[1:] != u[:-1], True] | (u != hi))]
    values = np.vstack([sample_on_union(x, y, laws, u) for x, y, laws in tables])
    return u, values
