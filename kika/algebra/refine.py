"""Adaptive refinement, pass by pass, and the lin-lin form of any table.

:func:`refine` is the one engine behind every "add points until a chord is
good enough" in kika: re-expressing a log-law table as lin-lin
(:func:`to_linlin`), and tabulating a resonance cross section that is only
known as a function (``kika.processing.resonances.grid.linearize``). What
differs between them is passed in -- what the function is, what "good enough"
means, where a panel is probed, which probes are kept -- and nothing else.

**Pass by pass.** Each pass probes every panel still being refined in one
vectorised call, and splits the ones that fail. **Only panels created in the
previous pass are probed**: a panel that passed keeps its probes, and probing
it again would ask the same question with the same numbers. That is what makes
the engine's cost the number of panels it *creates* rather than panels times
passes.

**Exact, or it raises.** A pass budget or point budget that runs out raises
:class:`RefinementError`. A failing panel that cannot be split -- every probe
rounds (or *snaps*) onto its own ends -- raises too, unless the caller says
that is acceptable, which is the case when the representation itself cannot
resolve finer (an ENDF float at 1 MeV resolves 1 eV).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .evaluate import panel_value
from .laws import HISTOGRAM, LINLIN, LOG_X, validate

__all__ = ["RefinementError", "RefineResult", "refine", "to_linlin",
           "LINEARIZATION_TOLERANCE", "PANEL_TEST_FRACTIONS"]

#: Default relative tolerance of :func:`to_linlin`. 1e-4 sits between FUDGE's
#: two settings -- 1e-3 as the ``XYs1d`` default, 1e-5 for its own processing
#: (``heat``, ``processMultiGroup``) -- and an order of magnitude above the
#: round-off of a seven-digit ENDF float.
LINEARIZATION_TOLERANCE = 1e-4

#: Where :func:`to_linlin` tests a chord, as fractions of the panel in the
#: law's own variable. One midpoint is not enough: on a steep log-log panel the
#: largest deviation sits well off the middle.
PANEL_TEST_FRACTIONS = np.array([0.25, 0.5, 0.75])


class RefinementError(RuntimeError):
    """The refinement could not meet its tolerance within its budget.

    ``reason`` says which budget: ``"points"``, ``"passes"`` or
    ``"unresolvable"`` (a failing panel float64 cannot split).
    """

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class RefineResult:
    """What :func:`refine` returns.

    ``x``, ``y`` are the refined nodes (``y`` keeps the column axis of the
    input). ``passes`` counts probing passes and ``evaluations`` the points
    the function was asked for. ``unresolved`` is the number of failing panels
    accepted because they could not be split. With ``keep_probes``,
    ``probe_x`` and ``probe_y`` hold, for every final panel in order, the
    probes it passed with -- ``(panels, fractions)`` and
    ``(panels, fractions[, columns])`` -- so a caller can report on them
    without asking the function again.
    """

    x: np.ndarray
    y: np.ndarray
    passes: int
    evaluations: int
    unresolved: int = 0
    probe_x: Optional[np.ndarray] = None
    probe_y: Optional[np.ndarray] = None


def refine(x, y, evaluate: Callable, exceeds: Callable, *,
           active=None, log_x=None, fractions=PANEL_TEST_FRACTIONS,
           snap: Optional[Callable] = None, insert: str = "worst",
           max_passes: int = 40, max_points: Optional[int] = None,
           min_width_ulps: float = 0.0, unresolvable: str = "raise",
           keep_probes: bool = False) -> RefineResult:
    """Add nodes to ``(x, y)`` until every chord passes *exceeds*.

    Parameters
    ----------
    x, y
        Starting nodes: ``x`` non-decreasing (a repeated value is a step and
        its zero-width interval is never refined), ``y`` of shape ``(n,)`` or
        ``(n, k)`` for *k* functions refined on one shared grid.
    evaluate
        ``evaluate(q, owner) -> values`` at abscissae *q* (1-d), with the same
        trailing shape as *y*. *owner* is the index of the starting interval
        each point falls in, for an evaluator that is a table's own law; a
        function of x alone ignores it.
    exceeds
        ``exceeds(actual, chord) -> ratio``, elementwise. A probe fails when
        its ratio is above one; with *k* columns a probe fails when any column
        does.
    active
        Which starting intervals to refine (default: every one of positive
        width).
    log_x
        Per starting interval, probe in ``ln x`` rather than ``x``.
    fractions
        Probe positions as fractions of a panel, in its probing variable.
    snap
        Maps candidate abscissae to representable ones *before* they are
        evaluated, so that a stored value belongs to the abscissa that will be
        written.
    insert
        ``"worst"``: a failing panel gains its worst probe. ``"all"``: it gains
        every distinct probe strictly inside it.
    max_passes, max_points
        Budgets; running out of either raises :class:`RefinementError`.
    min_width_ulps
        A failing panel narrower than this many ulps of its ends is not split.
    unresolvable
        ``"raise"`` or ``"accept"`` a failing panel that cannot be split.
    keep_probes
        Return the probes of every final panel (requires every interval to be
        active).
    """
    if insert not in ("worst", "all"):
        raise ValueError(f"insert must be 'worst' or 'all', got {insert!r}")
    if unresolvable not in ("raise", "accept"):
        raise ValueError(f"unresolvable must be 'raise' or 'accept', got {unresolvable!r}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    n = x.size
    width = np.diff(x)
    if active is None:
        active = width > 0
    active = np.asarray(active, dtype=bool) & (width > 0)
    if keep_probes and not active.all():
        raise ValueError("keep_probes needs every interval refined")
    log_x = (np.zeros(width.size, dtype=bool) if log_x is None
             else np.asarray(log_x, dtype=bool))
    if max_points is not None and n > max_points:
        raise RefinementError(f"{n} starting points exceed max_points={max_points}",
                              "points")

    # The panels being refined, as flat arrays.
    idx = np.flatnonzero(active)
    x1, x2, y1, y2 = x[idx], x[idx + 1], y[idx], y[idx + 1]
    owner, lg = idx, log_x[idx]

    added_x, added_y, added_at = [], [], []
    done_x1, done_px, done_py = [], [], []
    n_points, evaluations, unresolved = n, 0, 0
    passes = 0
    tiny = np.finfo(float).eps

    while owner.size:
        if passes == max_passes:
            raise RefinementError(f"refinement did not converge in {max_passes} passes",
                                  "passes")
        passes += 1
        f = fractions[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(lg[:, None],
                         x1[:, None] * (x2 / np.where(lg, x1, 1.0))[:, None] ** f,
                         x1[:, None] + (x2 - x1)[:, None] * f)
        if snap is not None:
            q = np.asarray(snap(q.ravel()), dtype=float).reshape(q.shape)
        q = np.maximum(x1[:, None], np.minimum(x2[:, None], q))
        usable = (q > x1[:, None]) & (q < x2[:, None])
        actual = np.asarray(evaluate(q.ravel(), np.repeat(owner, q.shape[1])),
                            dtype=float).reshape(q.shape + y.shape[1:])
        evaluations += q.size
        frac = (q - x1[:, None]) / (x2 - x1)[:, None]
        if y.ndim > 1:
            frac_k = frac[..., None]
            chord = y1[:, None, :] + (y2 - y1)[:, None, :] * frac_k
            ratio = np.max(exceeds(actual, chord), axis=-1)
        else:
            chord = y1[:, None] + (y2 - y1)[:, None] * frac
            ratio = exceeds(actual, chord)
        ratio = np.where(usable, ratio, 0.0)
        bad = np.any(ratio > 1.0, axis=1)

        narrow = (x2 - x1) <= min_width_ulps * tiny * (np.abs(x1) + np.abs(x2))
        stuck = bad & (narrow | ~usable.any(axis=1))
        if stuck.any():
            if unresolvable == "raise":
                i = int(np.flatnonzero(stuck)[0])
                raise RefinementError(
                    f"panel [{x1[i]!r}, {x2[i]!r}] fails its tolerance and cannot "
                    f"be split further in float64", "unresolvable")
            unresolved += int(stuck.sum())
        settle = ~bad | stuck
        if keep_probes and settle.any():
            done_x1.append(x1[settle])
            done_px.append(q[settle])
            done_py.append(actual[settle])
        split = bad & ~stuck
        if not split.any():
            break

        # The nodes each failing panel gains.
        rows = np.flatnonzero(split)
        if insert == "worst":
            pick = np.zeros(q.shape, dtype=bool)
            worst = np.argmax(np.where(usable, ratio, -1.0), axis=1)
            pick[rows, worst[rows]] = True
        else:
            pick = usable.copy()
        pick &= split[:, None]
        order = np.argsort(q, axis=1, kind="stable")
        qs = np.take_along_axis(q, order, axis=1)
        ps = np.take_along_axis(pick, order, axis=1)
        vs = actual[np.arange(q.shape[0])[:, None], order]
        # A snapped or rounded probe may coincide with another of its panel.
        ps[:, 1:] &= ~((qs[:, 1:] == qs[:, :-1]) & ps[:, :-1])
        ps &= split[:, None]
        r, c = np.nonzero(ps)
        new_x, new_y, new_owner = qs[r, c], vs[r, c], owner[r]
        n_points += new_x.size
        if max_points is not None and n_points > max_points:
            raise RefinementError(f"refinement exceeds max_points={max_points}", "points")
        added_x.append(new_x), added_y.append(new_y), added_at.append(new_owner)

        # Children: consecutive nodes of [x1, new..., x2] within each panel.
        seq_x = np.concatenate([x1[rows][:, None], np.where(ps[rows], qs[rows], np.nan),
                                x2[rows][:, None]], axis=1)
        keep = ~np.isnan(seq_x)
        seq_r = np.broadcast_to(rows[:, None], seq_x.shape)[keep]
        seq_x = seq_x[keep]
        if y.ndim > 1:
            seq_y = np.concatenate([y1[rows][:, None], vs[rows], y2[rows][:, None]], axis=1)
            seq_y = seq_y[keep]
        else:
            seq_y = np.concatenate([y1[rows][:, None], vs[rows], y2[rows][:, None]], axis=1)[keep]
        same = seq_r[1:] == seq_r[:-1]
        x1, x2 = seq_x[:-1][same], seq_x[1:][same]
        y1, y2 = seq_y[:-1][same], seq_y[1:][same]
        owner, lg = owner[seq_r[:-1][same]], lg[seq_r[:-1][same]]

    if added_x:
        all_x = np.concatenate([x, np.concatenate(added_x)])
        all_y = np.concatenate([y, np.concatenate(added_y)])
        # Original node j ranks 2j; a node added inside interval i ranks 2i+1,
        # so a merge never has to decide how a new node ranks against a
        # repeated original abscissa.
        rank = np.concatenate([2 * np.arange(n), 2 * np.concatenate(added_at) + 1])
        order = np.lexsort((all_x, rank))
        out_x, out_y = all_x[order], all_y[order]
    else:
        out_x, out_y = x.copy(), y.copy()

    result = RefineResult(out_x, out_y, passes, evaluations, unresolved)
    if keep_probes:
        if done_x1:
            dx1 = np.concatenate(done_x1)
            o = np.argsort(dx1, kind="stable")
            result.probe_x = np.concatenate(done_px)[o]
            result.probe_y = np.concatenate(done_py)[o]
        else:
            result.probe_x = np.zeros((0, fractions.size))
            result.probe_y = np.zeros((0, fractions.size) + y.shape[1:])
    return result


def to_linlin(x, y, laws, tol: float = LINEARIZATION_TOLERANCE, *,
              snap: Optional[Callable] = None, max_passes: int = 40):
    """``(x, y)`` that, read lin-lin, reproduce the table's own laws to *tol*.

    Every original point is kept with its value untouched; points are only
    added.

    * **lin-lin** panels are copied as they are.
    * **histogram** panels gain the left limit of their right end as a repeated
      abscissa, ``(x_{i+1}, y_i)`` -- the exact lin-lin form of a step.
    * **lin-log, log-lin, log-log** panels are refined until the chord agrees
      with the law to a relative *tol* at a quarter, a half and three quarters
      of every panel, in ``ln x`` for the log-x laws. Each added point is valued
      by the law of the panel it falls in, from that panel's original ends.

    A panel stops splitting at a width of ten ulps; with *snap* (see
    :func:`refine`) it also stops when every candidate snaps onto its own ends.
    Both are limits of the representation, not of the method, and are
    accepted.
    """
    x, y, laws = validate(x, y, laws)
    if x.size < 2:
        return x.copy(), y.copy()
    # Histogram panels first: a constant panel followed by a zero-width step
    # is already lin-lin, exactly.
    flat = (laws == HISTOGRAM) & (np.diff(x) > 0)
    step = np.flatnonzero(flat & (y[:-1] != y[1:]))
    laws = np.where(flat, LINLIN, laws)
    if step.size:
        x = np.insert(x, step + 1, x[step + 1])
        y = np.insert(y, step + 1, y[step])
        laws = np.insert(laws, step + 1, LINLIN)
    curved = (laws > LINLIN) & (np.diff(x) > 0)
    if not curved.any():
        return x.copy(), y.copy()

    def law_of(q, owner):
        return panel_value(x[owner], y[owner], x[owner + 1], y[owner + 1],
                           laws[owner], q)

    def relative(actual, chord):
        scale = np.maximum(np.maximum(np.abs(actual), np.abs(chord)), 1e-300)
        return np.abs(actual - chord) / scale / tol

    res = refine(x, y, law_of, relative, active=curved, log_x=np.isin(laws, LOG_X),
                 snap=snap, insert="worst", max_passes=max_passes,
                 min_width_ulps=10.0, unresolvable="accept")
    return res.x, res.y
