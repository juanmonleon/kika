"""The checks every TAB1 gets, whatever it tabulates (ENDF-6 §0.5.2, §0.5.3).

A TAB1 is ``NP`` pairs ``(x, y)`` and ``NR`` interpolation regions
``(NBT, INT)``. The format pins down:

* ``NBT`` strictly increasing, the last one equal to ``NP``, every ``INT`` a
  defined law (1-5; 6 is the charged-particle law of MF3);
* ``x`` non-decreasing; a value repeated twice is a discontinuity, three times
  has no meaning;
* a log law needs positive values on its own axis.

A table read past its last real pair -- ``NP`` larger than what the evaluator
wrote, the rest of the record blank and read as zeros -- shows up as pairs at
``x = 0`` after a grid that was increasing: that is told apart as
``table_blank_pairs`` because it says what happened.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..findings import DEFECT, NOTE, WARN
from .context import Out, Table, fmt_e

__all__ = ["check_table", "VALID_INT"]

#: The interpolation laws of a TAB1; 6 is MF3's charged-particle law.
VALID_INT = frozenset({1, 2, 3, 4, 5, 6})


def check_table(table: Table, out: Out, *, mf: int, mt: Optional[int], what: str,
                x_name: str = "E", log_y_zero_ok: bool = True, **where) -> bool:
    """Report what is wrong with *table*'s structure; True when its values can be read.

    *what* names the table in the summaries ("the cross section", "nu-bar").
    A table whose grid or laws are broken is reported and the value checks of
    the caller should skip it (the return is False).
    """
    x, y, pairs = table.x, table.y, table.pairs
    n = x.size
    ok = True
    if n == 0:
        out.add("table_empty", DEFECT, f"{what} has no points", mf=mf, mt=mt, **where)
        return False
    if y.size != n:
        out.add("table_interpolation", DEFECT,
                f"{what} has {n} abscissae and {y.size} values", mf=mf, mt=mt, **where)
        return False

    # ---- the interpolation regions ------------------------------------------
    problems = []
    if not pairs and n > 1:
        problems.append("no interpolation region (NR = 0)")
    nbt = [p[0] for p in pairs]
    ints = [p[1] for p in pairs]
    if any(b <= a for a, b in zip(nbt, nbt[1:])):
        problems.append(f"NBT not increasing ({nbt[:8]}{'...' if len(nbt) > 8 else ''})")
    if nbt and nbt[-1] != n:
        problems.append(f"last NBT = {nbt[-1]}, NP = {n}")
    bad_int = sorted({i for i in ints if i not in VALID_INT})
    if bad_int:
        problems.append(f"undefined interpolation law INT = {bad_int}")
    if problems:
        out.add("table_interpolation", DEFECT,
                f"{what}: " + "; ".join(problems), mf=mf, mt=mt, nr=len(pairs), np=n, **where)
        ok = not bad_int

    # ---- the grid -------------------------------------------------------------
    dx = np.diff(x)
    down = np.flatnonzero(dx < 0)
    if down.size:
        blank = _blank_pairs(x, y)
        if blank.any():
            i = int(np.flatnonzero(blank)[0])
            k = int(blank.sum())
            out.add("table_blank_pairs", DEFECT,
                    f"{what} has {k} blank pair{'s' if k != 1 else ''} read as ({x_name} = 0, 0) after "
                    f"{x_name} = {_fx(x[i - 1], x_name)} (point {i + 1} of {n}): fields the record left "
                    "empty, counted by NP", mf=mf, mt=mt, np=n, blank=k, after=float(x[i - 1]), **where)
            keep = ~blank
            down = np.flatnonzero(np.diff(x[keep]) < 0)
        if down.size:
            i = int(down[0])
            out.add("table_grid_order", DEFECT,
                    f"{what}: {x_name} decreases at {down.size} point{'s' if down.size != 1 else ''}, "
                    f"first from {_fx(x[i], x_name)} to {_fx(x[i + 1], x_name)} (point {i + 1})",
                    mf=mf, mt=mt, count=int(down.size), index=i + 1, **where)
        return False
    if n >= 3:
        # A run of three or more equal abscissae whose values are not all the
        # same: the step is from the first to the last, the ones between are
        # read by no code the same way. Equal values are harmless.
        triple = np.flatnonzero((dx[:-1] == 0) & (dx[1:] == 0))
        ambiguous = [int(i) for i in triple if not (y[i] == y[i + 1] == y[i + 2])]
        if ambiguous:
            i = ambiguous[0]
            out.add("table_grid_repeat", NOTE,
                    f"{what}: {x_name} = {_fx(x[i], x_name)} appears three or more times with "
                    f"different values ({len(ambiguous)} such run{'s' if len(ambiguous) != 1 else ''}); "
                    "the step is from the first to the last, the values between are not used",
                    mf=mf, mt=mt, count=len(ambiguous), at=float(x[i]), **where)

    # ---- log laws on non-positive values --------------------------------------
    if ok and n > 1 and pairs:
        laws = table.laws()
        wide = dx > 0
        logx = wide & np.isin(laws, (3, 5))
        logy = wide & np.isin(laws, (4, 5))
        bad_x = logx & ((x[:-1] <= 0) | (x[1:] <= 0))
        neg_y = logy & ((y[:-1] < 0) | (y[1:] < 0))
        zero_y = logy & ~neg_y & ((y[:-1] == 0) | (y[1:] == 0))
        if bad_x.any() or neg_y.any():
            i = int(np.flatnonzero(bad_x | neg_y)[0])
            axis = x_name if bad_x[i] else "value"
            out.add("table_log_nonpositive", DEFECT,
                    f"{what}: logarithmic interpolation (INT = {int(laws[i])}) over a "
                    f"{'non-positive' if bad_x[i] else 'negative'} {axis} between "
                    f"{_fx(x[i], x_name)} and {_fx(x[i + 1], x_name)} "
                    f"({int((bad_x | neg_y).sum())} interval{'s' if (bad_x | neg_y).sum() != 1 else ''})",
                    mf=mf, mt=mt, count=int((bad_x | neg_y).sum()), at=float(x[i]), **where)
        if zero_y.any() and not log_y_zero_ok:
            i = int(np.flatnonzero(zero_y)[0])
            out.add("table_log_zero", NOTE,
                    f"{what}: logarithmic interpolation (INT = {int(laws[i])}) towards a zero value "
                    f"between {_fx(x[i], x_name)} and {_fx(x[i + 1], x_name)} "
                    f"({int(zero_y.sum())} interval{'s' if zero_y.sum() != 1 else ''}); the law is "
                    "undefined there and codes differ in how they read it",
                    mf=mf, mt=mt, count=int(zero_y.sum()), at=float(x[i]), **where)
    return ok


def _blank_pairs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairs that are (0, 0) after a positive abscissa: blank fields read as numbers."""
    before = np.maximum.accumulate(np.concatenate([[-np.inf], x[:-1]]))
    return (x == 0) & (y == 0) & (before > 0)


def _fx(v: float, x_name: str) -> str:
    return fmt_e(float(v)) if x_name in ("E", "E'") else f"{float(v):.6g}"
