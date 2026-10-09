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
from .laws import HISTOGRAM, LINLIN, LOGLIN, LOGLOG, LOG_X, validate, vanishing_panels

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
           keep_probes: bool = False, precheck: int = 0,
           reuse_probes: bool = False) -> RefineResult:
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
        ``"worst"``: a failing panel gains its worst probe. ``"balanced"``:
        it gains the worst and the usable probe nearest the midpoint, keeping
        progress on both sides of a narrow feature. ``"all"``: it gains
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
    precheck
        Check this prefix of fractions first. Panels that already fail need
        not evaluate the remaining fractions until their children are tested.
        Accepted panels always pass all fractions. Zero uses one evaluation.
    reuse_probes
        Reuse exact (abscissa, starting interval) values from the previous
        pass; requires a deterministic evaluator. No rounded cache keys.
    """
    if insert not in ("worst", "balanced", "all"):
        raise ValueError(f"insert must be 'worst', 'balanced' or 'all', got {insert!r}")
    if unresolvable not in ("raise", "accept"):
        raise ValueError(f"unresolvable must be 'raise' or 'accept', got {unresolvable!r}")
    if not isinstance(precheck,int) or isinstance(precheck,bool) or not 0<=precheck<=len(fractions):
        raise ValueError('precheck must be an integer prefix length of fractions')
    if precheck and insert=='all':
        raise ValueError("precheck cannot skip fractions with insert='all'")
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
    cache_keys=cache_values=cache_x=cache_owner=None
    key_dtype=np.dtype([('x',np.float64),('owner',np.int64)])

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
        flat=q.ravel();flat_owner=np.repeat(owner,q.shape[1])
        actual=np.empty((flat.size,)+y.shape[1:])
        available=np.zeros(flat.size,dtype=bool)
        keys=None
        if reuse_probes:
            keys=np.empty(flat.size,dtype=key_dtype)
            keys['owner']=flat_owner;keys['x']=flat
        def fetch(indices):
            nonlocal evaluations
            missing=indices
            if reuse_probes and cache_keys is not None and len(cache_keys):
                # Search the numeric abscissa first. Most probes have only one
                # starting owner at this energy; compound comparisons are only
                # needed at repeated/snapped boundaries shared by owners.
                query=keys[indices]
                at=np.searchsorted(cache_x,query['x'])
                at=np.minimum(at,len(cache_keys)-1)
                ambiguous=(cache_x[at]==query['x']) & (cache_owner[at]!=query['owner'])
                if np.any(ambiguous):
                    at[ambiguous]=np.minimum(np.searchsorted(cache_keys,query[ambiguous]),len(cache_keys)-1)
                hit=(cache_x[at]==query['x']) & (cache_owner[at]==query['owner'])
                actual[indices[hit]]=cache_values[at[hit]]
                missing=indices[~hit]
            if len(missing):
                actual[missing]=np.asarray(evaluate(flat[missing],flat_owner[missing]),dtype=float)
                evaluations+=len(missing)
            available[indices]=True
        primary=precheck or q.shape[1]
        first=np.flatnonzero(np.tile(np.arange(q.shape[1])<primary,len(owner)))
        fetch(first)
        actual=actual.reshape(q.shape+y.shape[1:])
        frac = (q - x1[:, None]) / (x2 - x1)[:, None]
        chord=(y1[:,None,:]+(y2-y1)[:,None,:]*frac[...,None] if y.ndim>1
               else y1[:,None]+(y2-y1)[:,None]*frac)
        ratio=np.zeros(q.shape)
        def ratios(indices):
            if not len(indices):return
            a=actual.reshape((flat.size,)+y.shape[1:])[indices]
            b=chord.reshape((flat.size,)+y.shape[1:])[indices]
            r=exceeds(a,b)
            if y.ndim>1:r=np.max(r,axis=-1)
            ratio.ravel()[indices]=np.where(usable.ravel()[indices],r,0.)
        ratios(first)
        if primary<q.shape[1]:
            remaining=np.flatnonzero(np.repeat(~np.any(ratio>1.,axis=1),q.shape[1])&~available)
            # fetch writes the flattened array even after the shaped view is made.
            actual=actual.reshape((flat.size,)+y.shape[1:])
            fetch(remaining)
            actual=actual.reshape(q.shape+y.shape[1:])
            ratios(remaining)
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
            if keep_probes:
                remaining=np.flatnonzero(np.repeat(stuck,q.shape[1])&~available)
                actual=actual.reshape((flat.size,)+y.shape[1:])
                fetch(remaining)
                actual=actual.reshape(q.shape+y.shape[1:])
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
        if insert in ("worst","balanced"):
            pick = np.zeros(q.shape, dtype=bool)
            worst = np.argmax(np.where(usable, ratio, -1.0), axis=1)
            pick[rows, worst[rows]] = True
            if insert == "balanced":
                middle=np.argmin(np.where(usable,abs(frac-.5),np.inf),axis=1)
                pick[rows,middle[rows]]=True
        else:
            pick = usable.copy()
        pick &= split[:, None]
        # A balanced midpoint can be outside a caller's precheck prefix.
        # Every inserted node needs its own evaluated value.
        missing=np.flatnonzero(pick.ravel()&~available)
        actual=actual.reshape((flat.size,)+y.shape[1:])
        fetch(missing)
        actual=actual.reshape(q.shape+y.shape[1:])
        if reuse_probes:
            retain=available&np.repeat(split,q.shape[1])
            retained_keys=keys[retain]
            order=np.lexsort((retained_keys['owner'],retained_keys['x']))
            cache_keys=retained_keys[order]
            cache_x=np.ascontiguousarray(cache_keys['x'])
            cache_owner=np.ascontiguousarray(cache_keys['owner'])
            cache_values=actual.reshape((flat.size,)+y.shape[1:])[retain][order]
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
    * **log-lin, log-log** panels with an end at 0 are 0 inside, with the jump
      at the other end (:func:`~kika.algebra.laws.vanishing_panels`): written as
      that step, exactly, by repeating the non-zero end's abscissa at 0.
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
    wide = np.diff(x) > 0
    vanishing = wide & np.isin(laws, (LOGLIN, LOGLOG)) & vanishing_panels(y[:-1], y[1:])
    if vanishing.any():
        # (x1, y1 > 0) -> (x2, 0) becomes (x1, y1), (x1, 0), (x2, 0); and
        # (x1, 0) -> (x2, y2 > 0) becomes (x1, 0), (x2, 0), (x2, y2).
        k = np.flatnonzero(vanishing)
        falls, rises = k[y[k] > 0], k[y[k + 1] > 0]
        laws = np.where(vanishing, LINLIN, laws)
        at = np.r_[falls + 1, rises + 1]
        order = np.argsort(at, kind="stable")
        x = np.insert(x, at[order], np.r_[x[falls], x[rises + 1]][order])
        y = np.insert(y, at[order], np.zeros(at.size))
        laws = np.insert(laws, at[order], LINLIN)
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
