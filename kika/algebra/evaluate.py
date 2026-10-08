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

from .laws import HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG, validate, vanishing_panels

__all__ = ["evaluate", "left_limit", "right_limit", "sample_on_union",
           "panel_value", "interpolate_between", "OUTSIDE"]

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
    law = np.asarray(law, dtype=np.int64)
    codes = (np.unique(law) if law.ndim else [int(law)])
    law = np.broadcast_to(law, q.shape)
    out = np.empty(q.shape, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for code in codes:
            m = slice(None) if len(codes) == 1 else law == code
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
                v = b1 * np.exp((t - a1) / (a2 - a1) * np.log(b2 / b1))
                out[m] = np.where(vanishing_panels(b1, b2), 0.0, v)
            elif code == LOGLOG:
                v = b1 * np.exp(np.log(t / a1) / np.log(a2 / a1) * np.log(b2 / b1))
                out[m] = np.where(vanishing_panels(b1, b2), 0.0, v)
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
    return _read_checked(x, y, laws, q, side, outside)


def _read_checked(x, y, laws, q, side: str, outside: str):
    """:func:`_read` on a table :func:`validate` has already passed."""
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
    elif inside.any() and side == "point" and laws.min() == laws.max() == LINLIN:
        # np.interp is this same arithmetic (see panel_value), right-continuous
        # at a repeated abscissa and exact at the nodes, and it binary-searches
        # with a hint -- several times faster on the sorted queries most
        # callers make.
        out[inside] = np.interp(flat[inside], x, y)
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
        # A log-y panel with an end at 0 is 0 inside (laws.vanishing_panels):
        # its jump is at the other end, which a one-sided limit sees.
        if side != "point":
            logy = (law == LOGLIN) | (law == LOGLOG)
            jump = wide & logy & vanishing_panels(y1, y2)
            if jump.any():
                inner = (at_start & (side == "right")) | (at_end & (side == "left"))
                value[jump & inner] = 0.0
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
    out = _read_checked(x, y, laws, u, "point", "zero")
    # Only the repeated abscissae need more than the value; they are few, so
    # the bookkeeping is done on them alone.
    first = np.r_[True, u[1:] != u[:-1]]
    lone = first & np.r_[first[1:], True]
    if lone.all():
        return out
    rep = np.flatnonzero(~lone)
    ur, fr = u[rep], first[rep]
    start = np.flatnonzero(fr)
    count = np.diff(np.r_[start, rep.size])
    run = np.cumsum(fr) - 1
    occurrence = np.arange(rep.size) - start[run]
    multiplicity = count[run]
    lead = occurrence == 0
    tail = occurrence == multiplicity - 1
    out[rep[lead]] = _read_checked(x, y, laws, ur[lead], "left", "zero")
    out[rep[tail]] = _read_checked(x, y, laws, ur[tail], "right", "zero")
    middle = ~(lead | tail)
    if middle.any() and x.size:
        lo = np.searchsorted(x, ur[middle], side="left")
        hi = np.searchsorted(x, ur[middle], side="right")
        own = occurrence[middle] < hi - lo
        idx = rep[np.flatnonzero(middle)[own]]
        out[idx] = y[lo[own] + occurrence[middle][own]]
    return out


def interpolate_between(x1: float, y1: ArrayLike, x2: float, y2: ArrayLike,
                        law: int, q: float) -> np.ndarray:
    """The value at outer coordinate *q* between two whole functions.

    ``y1`` and ``y2`` are arrays of the same shape -- two tables already read
    on one inner grid, say -- and each element is interpolated on its own
    under *law*, the way a TAB2 interpolates between incident energies. At
    ``q == x1`` the result is ``y1`` and at ``q == x2`` it is ``y2`` (``y1``
    for a histogram), bit for bit.

    A log law with a non-positive value or abscissa raises: the element has no
    value under that law, and reading it lin-lin instead is how a wrong number
    used to come back silently. An element with an end at exactly 0 and none
    below is read as the limit of the law, 0 between the ends
    (:func:`~kika.algebra.laws.vanishing_panels`).
    """
    law = int(law)
    if law not in (HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG):
        raise ValueError(f"interpolation law {law} is not 1-5")
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    if x1 == x2 or q == x1 or law == HISTOGRAM:
        return y1.copy()
    if q == x2:
        return y2.copy()
    if law in (LINLOG, LOGLOG) and (x1 <= 0 or x2 <= 0 or q <= 0):
        raise ValueError(f"law {law} interpolates in ln x but the outer "
                         f"coordinates are {x1!r}, {x2!r}, {q!r}")
    if law in (LOGLIN, LOGLOG) and (np.any(y1 < 0) or np.any(y2 < 0)):
        raise ValueError(f"law {law} interpolates in ln y but a value is "
                         f"negative")
    return panel_value(x1, y1, x2, y2, law, np.full(y1.shape, float(q)))
