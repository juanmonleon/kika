"""Sums of tables, exact between the nodes and at every step.

A sum of tables is lin-lin on the union of their grids only once every table
*is* lin-lin: two log-log partials summed at their union nodes and read lin-lin
are right at the nodes and up to 10 % wrong between them (8.1 % on B-10 of
ENDF/B-VIII.1). So each table goes through :func:`~kika.algebra.refine.to_linlin`
first, and the sum is then exact up to that tolerance.

Each table is zero outside its own domain. One that starts or ends at a
non-zero value inside the range of the sum puts a step in the sum there; the
union grid carries that abscissa twice (:func:`~kika.algebra.grid.union`'s
*steps*), and :func:`~kika.algebra.evaluate.sample_on_union` gives its first
copy the left limit and its second the right one. Leaving it out turned the
step into a ramp across the neighbouring interval: +66 % in the case that found
it. FUDGE refuses the same situation (``nfu_domainsNotMutual``) unless told to
replace the step by a ramp of width ``eps``; a repeated abscissa needs no
width.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .evaluate import sample_on_union
from .grid import union
from .laws import validate
from .refine import LINEARIZATION_TOLERANCE, to_linlin

__all__ = ["add", "domain_steps"]


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
    :func:`~kika.algebra.refine.refine`), and the result lives on the union of
    those grids, plus any extra *grids* the caller wants carried. Its range is
    the union of the tables' domains; each table is zero off its own.
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
