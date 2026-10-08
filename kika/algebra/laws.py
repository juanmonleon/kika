"""Interpolation laws, one per interval, and their ENDF ``(NBT, INT)`` spelling.

The five codes are ENDF-6's (manual §0.5.2) and GNDS adopted them wholesale
(§3.4.4), so they are the vocabulary of the table, not of a format:

====  ==========  =====================================
code  name        y between (x1, y1) and (x2, y2)
====  ==========  =====================================
1     histogram   y1
2     lin-lin     linear in x
3     lin-log     linear in ln x
4     log-lin     ln y linear in x
5     log-log     ln y linear in ln x
====  ==========  =====================================

Inside this package a table carries its laws per interval: ``laws[i]`` joins
points ``i`` and ``i + 1``. That is easier to compute with than ENDF's
cumulative NBT, and :func:`interval_laws` / :func:`pairs_from_laws` translate.

Anything else is refused. INT=6 (the charged-particle Coulomb law) has no
implementation here, and 11-15/21-25 are *two-dimensional* codes -- the law
between incident energies of a TAB2, with a unit-base or corresponding-point
flag in the tens digit. Reading one as the 1-d law of its units digit drops
that flag; it was done silently before this package existed, and it is the
caller's decision to make, with the flag in hand.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

__all__ = ["HISTOGRAM", "LINLIN", "LINLOG", "LOGLIN", "LOGLOG", "LAWS",
           "interval_laws", "pairs_from_laws", "laws_on_refinement", "validate",
           "vanishing_panels", "LOG_X", "LOG_Y"]

HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG = 1, 2, 3, 4, 5
LAWS = (HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG)

#: Laws that interpolate in ln x, and in ln y.
LOG_X = (LINLOG, LOGLOG)
LOG_Y = (LOGLIN, LOGLOG)


def _check_codes(codes: np.ndarray) -> None:
    bad = sorted(set(np.unique(codes).tolist()) - set(LAWS))
    if bad:
        two_d = [c for c in bad if 11 <= c <= 15 or 21 <= c <= 25]
        hint = (f"; {two_d} are two-dimensional (TAB2) codes, whose tens digit "
                f"is a unit-base/corresponding-point flag a 1-d law cannot "
                f"carry -- map them explicitly" if two_d else "")
        raise ValueError(f"interpolation law(s) {bad} are not 1-5{hint}")


def interval_laws(n_points: int, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """The law of every interval of a table of *n_points*, from ``(NBT, INT)``.

    NBT is cumulative and one-based: region *r* owns intervals ``NBT[r-1] - 1``
    up to ``NBT[r] - 2``, and the interval that straddles a region boundary
    belongs to the region *after* it. An empty pair list is one lin-lin region.
    A last NBT short of the table holds its law to the end.
    """
    laws = np.full(max(n_points - 1, 0), LINLIN, dtype=np.int64)
    pairs = list(pairs) or [(n_points, LINLIN)]
    start = 0
    for nbt, code in pairs:
        stop = min(int(nbt) - 1, n_points - 1)
        if stop > start:
            laws[start:stop] = int(code)
        start = max(start, stop)
    if start < n_points - 1:
        laws[start:] = int(pairs[-1][1])
    _check_codes(laws)
    return laws


def pairs_from_laws(laws: np.ndarray) -> list:
    """The shortest ``(NBT, INT)`` list that states *laws*.

    The inverse of :func:`interval_laws` up to merging adjacent regions that
    share a law. A table of one point, or none, is one lin-lin region.
    """
    laws = np.asarray(laws, dtype=np.int64)
    if laws.size == 0:
        return [(1, LINLIN)]
    change = np.flatnonzero(np.diff(laws)) + 1
    ends = np.r_[change, laws.size]
    return [(int(e) + 1, int(laws[e - 1])) for e in ends]


def laws_on_refinement(x_old, laws_old, x_new) -> np.ndarray:
    """The law of every interval of *x_new*, a refinement of *x_old*.

    A point inserted inside a panel cuts it into two panels of the **same**
    law -- that is what makes the insertion exact -- so a refined table states
    no law of its own: each new interval inherits the law of the old panel it
    lies in. Spelled out here so a caller that inserts points into a
    multi-region table does not have to choose between relabelling it (one
    lin-lin region, what ``MF5PartialTabulated.replace_table`` does by default)
    and refusing it.

    Raises ``ValueError`` when *x_new* is not a refinement: a different
    domain, or an interval that straddles an old node. A zero-width interval
    (a repeated abscissa, a step) has no law; it takes the old one at the same
    place when the old table had the repeat too, and the law of the panel it
    starts otherwise -- which no reader consults, but which keeps the regions
    contiguous.
    """
    x_old = np.asarray(x_old, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    laws_old = np.asarray(laws_old, dtype=np.int64)
    if laws_old.ndim == 0:
        laws_old = np.full(max(x_old.size - 1, 0), int(laws_old), dtype=np.int64)
    if x_new.size < 2 or x_old.size < 2:
        return np.full(max(x_new.size - 1, 0), LINLIN, dtype=np.int64)
    if x_new[0] != x_old[0] or x_new[-1] != x_old[-1]:
        raise ValueError(
            f"[{x_new[0]:.6e}, {x_new[-1]:.6e}] is not the domain "
            f"[{x_old[0]:.6e}, {x_old[-1]:.6e}] of the table it is meant to "
            f"refine, so its intervals have no law to inherit")
    a, b = x_new[:-1], x_new[1:]
    wide = b > a
    middle = np.where(wide, 0.5 * (a + b), a)
    k = np.clip(np.searchsorted(x_old, middle, side="right") - 1, 0, x_old.size - 2)
    straddles = wide & ((a < x_old[k]) | (b > x_old[k + 1]))
    if np.any(straddles):
        first = int(np.flatnonzero(straddles)[0])
        raise ValueError(
            f"interval [{a[first]:.6e}, {b[first]:.6e}] crosses a node of the "
            f"table it is meant to refine, so it is not one panel of one law")
    return laws_old[k].copy()


def vanishing_panels(y1, y2) -> np.ndarray:
    """Where a log-y panel has an end at exactly 0 and none below it.

    ``y1 (x / x1)**p`` (log-log) or ``y1 exp(p (x - x1))`` (log-lin) with ``y2 -> 0``
    sends ``p`` to minus infinity, and with ``y1 -> 0`` to plus infinity: the
    law's limit is 0 everywhere inside the panel, with the jump at the end that
    is not 0 (both ends 0: the zero function). That is what every operation in
    this package reads, exactly and in closed form -- and what NJOY's ``terp1``
    returns -- rather than refusing the table or reading it lin-lin.
    """
    y1, y2 = np.asarray(y1), np.asarray(y2)
    return (y1 >= 0) & (y2 >= 0) & ((y1 == 0) | (y2 == 0))


def validate(x, y, laws) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x, y, laws)`` as float/int arrays, or a ``ValueError`` saying why not.

    *laws* may be one code for the whole table or one per interval. Checked:
    matching shapes, finite values, non-decreasing ``x``, codes 1-5, and that
    every log law has positive values on its own axis. A zero-width interval
    (a repeated abscissa) has no law and is not checked. A log-y interval with
    an end at exactly 0 is let through (:func:`vanishing_panels`); one below 0
    is refused.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape[:1] != x.shape:
        raise ValueError(f"x must be 1-d and y must have it as first axis, "
                         f"got {x.shape} and {y.shape}")
    n_intervals = max(x.size - 1, 0)
    laws = np.asarray(laws, dtype=np.int64)
    if laws.ndim == 0:
        laws = np.full(n_intervals, int(laws), dtype=np.int64)
    if laws.shape != (n_intervals,):
        raise ValueError(f"{x.size} points need {n_intervals} interval laws, "
                         f"got {laws.shape}")
    _check_codes(laws)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("table has non-finite values")
    if np.any(np.diff(x) < 0):
        raise ValueError("abscissae must be non-decreasing")
    if n_intervals:
        wide = np.diff(x) > 0
        logx = wide & np.isin(laws, LOG_X)
        if np.any(logx & (x[:-1] <= 0)):
            i = int(np.flatnonzero(logx & (x[:-1] <= 0))[0])
            raise ValueError(f"law {laws[i]} interpolates in ln x but interval "
                             f"{i} starts at x = {x[i]!r}")
        logy = wide & np.isin(laws, LOG_Y)
        if logy.any():
            y1, y2 = y[:-1], y[1:]
            nonpositive = (y1 < 0) | (y2 < 0)
            if y.ndim > 1:
                bad = logy & np.any(nonpositive, axis=tuple(range(1, y.ndim)))
            else:
                bad = logy & nonpositive
            if bad.any():
                i = int(np.flatnonzero(bad)[0])
                raise ValueError(
                    f"law {laws[i]} interpolates in ln y but interval {i} "
                    f"([{x[i]!r}, {x[i + 1]!r}]) has a negative value")
    return x, y, laws
