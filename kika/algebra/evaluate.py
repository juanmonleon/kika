"""A table's value, and its one-sided limits, at any abscissa.

Three readings of the same table, because a step needs all three:

* :func:`evaluate` -- the value. Right-continuous inside the domain (at a
  repeated abscissa it is the value on the right), and closed at both ends:
  ``evaluate(x[-1]) == y[-1]``.
* :func:`left_limit` / :func:`right_limit` -- the limits from either side.
  Outside the domain the table is zero (or whatever *outside* says), so the
  left limit at ``x[0]`` and the right limit at ``x[-1]`` are that outside
  value: a table that starts at a non-zero value starts with a step.

At a tabulated abscissa the value is the tabulated one, bit for bit, whatever
the law -- no ``exp(log(y))`` round trip. Between abscissae each law is
evaluated in closed form. A log law over a non-positive value is refused when
the table is checked (:func:`~kika.algebra.laws.validate`), not quietly read
lin-lin.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from .laws import HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG, validate

__all__ = ["evaluate", "left_limit", "right_limit", "sample_on_union",
           "panel_value", "OUTSIDE"]

#: What a table is outside its own domain.
OUTSIDE = ("zero", "hold", "raise")


def panel_value(x1, y1, x2, y2, law, q) -> np.ndarray:
    """The law of each panel ``(x1, y1)-(x2, y2)`` evaluated at *q*.

    Arrays broadcast together; *law* may be one code or one per element.
    Panels must have positive width, and *q* is taken to lie inside them.
    The endpoints are not special-cased here -- the callers do that, so the
    tabulated value comes back untouched.
    """
    x1, y1, x2, y2, q = np.broadcast_arrays(*(np.asarray(a, dtype=float)
                                              for a in (x1, y1, x2, y2, q)))
    law = np.broadcast_to(np.asarray(law, dtype=np.int64), q.shape)
    out = np.empty(q.shape, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for code in np.unique(law):
            m = law == code
            a1, b1, a2, b2, t = x1[m], y1[m], x2[m], y2[m], q[m]
            if code == HISTOGRAM:
                out[m] = b1
            elif code == LINLIN:
                # np.interp's own arithmetic, so a lin-lin table reads the
                # same bits it did when it went through np.interp.
                out[m] = (b2 - b1) / (a2 - a1) * (t - a1) + b1
            elif code == LINLOG:
                out[m] = b1 + (b2 - b1) * (np.log(t / a1) / np.log(a2 / a1))
            elif code == LOGLIN:
                out[m] = b1 * np.exp((t - a1) / (a2 - a1) * np.log(b2 / b1))
            elif code == LOGLOG:
                out[m] = b1 * np.exp(np.log(t / a1) / np.log(a2 / a1) * np.log(b2 / b1))
            else:  # pragma: no cover - validate() refuses it first
                raise ValueError(f"interpolation law {code} is not 1-5")
    return out


def _outside_value(y: np.ndarray, outside: str, at_low: np.ndarray) -> np.ndarray:
    if outside == "zero":
        return np.zeros(at_low.shape)
    if outside == "hold":
        return np.where(at_low, y[0], y[-1])
    raise ValueError("abscissa outside the table's domain")


def _read(x, y, laws, q, side: str, outside: str):
    if outside not in OUTSIDE:
        raise ValueError(f"outside must be one of {OUTSIDE}, got {outside!r}")
    x, y, laws = validate(x, y, laws)
    scalar = np.ndim(q) == 0
    q = np.asarray(q, dtype=float)
    flat = q.reshape(-1)
    out = np.zeros(flat.shape, dtype=float)
    if x.size == 0:
        if outside == "raise" and flat.size:
            raise ValueError("an empty table has no domain")
        return float(out[0]) if scalar else out.reshape(q.shape)

    if side == "left":
        inside = (flat > x[0]) & (flat <= x[-1])
    elif side == "right":
        inside = (flat >= x[0]) & (flat < x[-1])
    else:
        inside = (flat >= x[0]) & (flat <= x[-1])
    if not inside.all():
        off = ~inside
        out[off] = _outside_value(y, outside, flat[off] <= x[0])

    if inside.any() and x.size == 1:
        out[inside] = y[0]
    elif inside.any():
        p = flat[inside]
        if side == "left":
            # The panel (x[k], x[k+1]] that ends at or after p, never a
            # zero-width one: x[k] < p.
            k = np.searchsorted(x, p, side="left") - 1
        else:
            # The panel [x[k], x[k+1]) that starts at or before p.
            k = np.minimum(np.searchsorted(x, p, side="right") - 1, x.size - 2)
        x1, x2, y1, y2, law = x[k], x[k + 1], y[k], y[k + 1], laws[k]
        wide = x2 > x1
        value = np.empty(p.shape)
        value[wide] = panel_value(x1[wide], y1[wide], x2[wide], y2[wide],
                                  law[wide], p[wide])
        value[~wide] = y2[~wide]
        # Tabulated abscissae read their tabulated value, untouched.
        at_start = p == x1
        value[at_start] = y1[at_start]
        at_end = (p == x2) & ((law != HISTOGRAM) | (side != "left"))
        value[at_end] = y2[at_end]
        out[inside] = value
    return float(out[0]) if scalar else out.reshape(q.shape)


def evaluate(x: ArrayLike, y: ArrayLike, laws, q: Union[float, ArrayLike],
             outside: str = "zero") -> Union[float, np.ndarray]:
    """The table's value at *q*: right-continuous, closed at both ends.

    *laws* is one code or one per interval (:func:`~kika.algebra.laws.interval_laws`
    builds it from ENDF pairs). *outside* is ``"zero"`` (the default: the
    function is zero off its own domain), ``"hold"`` (the end value continues)
    or ``"raise"``.
    """
    return _read(x, y, laws, q, "point", outside)


def left_limit(x, y, laws, q, outside: str = "zero"):
    """``lim y(t)`` as ``t -> q`` from below; the outside value at ``x[0]``."""
    return _read(x, y, laws, q, "left", outside)


def right_limit(x, y, laws, q, outside: str = "zero"):
    """``lim y(t)`` as ``t -> q`` from above; the outside value at ``x[-1]``."""
    return _read(x, y, laws, q, "right", outside)


def sample_on_union(x, y, laws, u: ArrayLike) -> np.ndarray:
    """The table read at every point of a union grid *u* that may repeat.

    *u* is non-decreasing, and an abscissa it repeats is a step of the result
    (:func:`~kika.algebra.grid.union` builds such a grid). At an abscissa that
    appears once the table gives its value; at one that appears ``m >= 2``
    times the first copy gets the left limit and the last the right limit, so
    every step of every table -- a repeated abscissa of its own, or the edge of
    its domain -- comes through as a step. Copies in between (``m >= 3``) take
    the table's own entry of that rank when it has one, and its value
    otherwise.
    """
    x, y, laws = validate(x, y, laws)
    u = np.asarray(u, dtype=float)
    if u.size == 0:
        return np.zeros(0)
    first = np.r_[True, u[1:] != u[:-1]]
    start = np.flatnonzero(first)
    count = np.diff(np.r_[start, u.size])
    run = np.cumsum(first) - 1
    occurrence = np.arange(u.size) - start[run]
    multiplicity = count[run]

    out = evaluate(x, y, laws, u)
    lone = multiplicity == 1
    lead = (~lone) & (occurrence == 0)
    tail = (~lone) & (occurrence == multiplicity - 1)
    if lead.any():
        out[lead] = left_limit(x, y, laws, u[lead])
    if tail.any():
        out[tail] = right_limit(x, y, laws, u[tail])
    middle = ~(lone | lead | tail)
    if middle.any() and x.size:
        lo = np.searchsorted(x, u[middle], side="left")
        hi = np.searchsorted(x, u[middle], side="right")
        own = occurrence[middle] < hi - lo
        idx = np.flatnonzero(middle)[own]
        out[idx] = y[lo[own] + occurrence[middle][own]]
    return out
