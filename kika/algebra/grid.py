"""Grids: unions that keep steps, and the step <-> region correspondence.

A step is written two ways in practice. A TAB1 record and this package repeat
the abscissa -- ``(x, y_left), (x, y_right)``. GNDS ``regions1d`` (and kika's
:class:`~kika.nuclear_data.model.functions.regions1d.Regions1d`) start a new
region at ``x`` instead. :func:`split_at_discontinuities` goes from the first
to the second and :func:`join_pieces` back; neither ever drops one of the two
values or interpolates across the zero-width interval between them.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .laws import LINLIN, validate

__all__ = ["union", "occurrences", "discontinuities", "split_at_discontinuities",
           "join_pieces", "compress_flat"]


def compress_flat(x, y, laws=LINLIN):
    """Remove interior nodes of exactly constant lin-lin/histogram spans.

    No tolerance or approximation is involved. Keep law boundaries, repeated
    abscissae and signed-zero ordinates. Return independent arrays, including
    when no node can be removed.
    """
    x, y, laws = validate(x, y, laws)
    if y.ndim != 1:
        raise ValueError('flat compression requires one-dimensional ordinates')
    if len(x)>16 and laws[0] in (1,2) and np.all(laws==laws[0]):
        value=y[0]
        if (np.all(y==value) and (value!=0 or not np.any(np.signbit(y)))
                and np.all(np.diff(x)>0)):
            with np.errstate(over='ignore'):
                if np.isfinite(x[-1]-x[0]):
                    # A proven constant needs two endpoints. Retain the
                    # generic path for steps, law boundaries and signed zero.
                    return x[[0,-1]],y[[0,-1]],laws[[0]]
    keep = np.ones(len(x), dtype=bool)
    if len(x) > 2:
        keep[1:-1] = ~((y[:-2] == y[1:-1]) & (y[1:-1] == y[2:])
            & (laws[:-1] == laws[1:]) & ((laws[:-1] == 1) | (laws[:-1] == 2))
            & (x[:-2] < x[1:-1]) & (x[1:-1] < x[2:])
            & ~((y[:-2] == 0) & np.signbit(y[:-2]))
            & ~((y[1:-1] == 0) & np.signbit(y[1:-1]))
            & ~((y[2:] == 0) & np.signbit(y[2:])))
    indices = np.flatnonzero(keep)
    with np.errstate(over='ignore'):
        if np.any(~np.isfinite(np.diff(x[indices]))):
            return x.copy(), y.copy(), laws.copy()
    # Advanced indexing already owns its result; a second copy is redundant.
    return x[indices], y[indices], laws[indices[:-1]]


def union(grids: Iterable[np.ndarray], steps: Iterable[float] = ()) -> np.ndarray:
    """The sorted union of *grids*, keeping repeated abscissae.

    A repeated abscissa is a step, not a duplicate to be tidied away: each
    value appears as many times as the grid that repeats it most (``np.unique``
    would collapse the pair and turn the step into a ramp across the whole
    neighbouring interval). *steps* are abscissae that must appear at least
    twice whatever the grids say -- the edge of a table that ends at a non-zero
    value inside the union's range, say, where the sum steps.
    """
    grids = [np.asarray(g, dtype=float).ravel() for g in grids]
    steps = np.asarray(list(steps), dtype=float)
    pool = [g for g in grids if g.size] + ([steps] if steps.size else [])
    if not pool:
        return np.zeros(0)
    values = np.unique(np.concatenate(pool))
    multiplicity = np.ones(values.size, dtype=np.int64)
    for g in grids:
        if g.size:
            seen, counts = np.unique(g, return_counts=True)
            np.maximum.at(multiplicity, np.searchsorted(values, seen), counts)
    if steps.size:
        np.maximum.at(multiplicity, np.searchsorted(values, np.unique(steps)), 2)
    return np.repeat(values, multiplicity)


def occurrences(u: np.ndarray) -> np.ndarray:
    """For each point of *u*, its 0-based rank among the repeats of its value."""
    u = np.asarray(u, dtype=float)
    if u.size == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.r_[True, u[1:] != u[:-1]]
    start = np.flatnonzero(first)
    return np.arange(u.size) - start[np.cumsum(first) - 1]


def discontinuities(x: np.ndarray) -> np.ndarray:
    """Indices ``i`` with ``x[i] == x[i + 1]``: the zero-width intervals."""
    x = np.asarray(x, dtype=float)
    return np.flatnonzero(np.diff(x) == 0)


def split_at_discontinuities(x, y, laws) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The continuous pieces of a table, cut at every repeated abscissa.

    Each piece is ``(x, y, laws)`` with strictly increasing abscissae. A step's left value ends one piece and its right value
    starts the next. A piece of a single point -- a value that is both the
    right limit of one step and the left limit of the next -- has no interval
    and is refused, since it states a value no interval owns.
    """
    x, y, laws = validate(x, y, laws)
    # ENDF can repeat an identical threshold point more than twice. Such
    # extra copies state no extra one-sided value; remove only consecutive
    # equal values within those runs. Distinct intermediate values remain
    # ambiguous and are still refused below. Keep an ordinary two-point
    # repeat, including a continuous region boundary, unchanged.
    repeated=np.diff(x)==0
    transitions=np.flatnonzero(np.diff(np.r_[False,repeated,False]))
    keep=np.ones(len(x),dtype=bool)
    for a,b in zip(transitions[::2],transitions[1::2]):
        if b-a<2:continue
        retained=[a]+[i for i in range(a+1,b+1) if not np.array_equal(y[i],y[i-1])]
        if len(retained)==1:retained=[a,b]
        else:retained[-1]=b
        keep[a:b+1]=False;keep[retained]=True
    if not np.all(keep):
        indices=np.flatnonzero(keep)
        x,y,laws=x[indices],y[indices],laws[indices[1:]-1]
    cuts = np.r_[0, discontinuities(x) + 1, x.size]
    pieces = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 2:
            raise ValueError(f"isolated point at x = {x[a]!r}: a value between "
                             f"two steps has no interval")
        pieces.append((x[a:b], y[a:b], laws[a:b - 1]))
    return pieces


def join_pieces(pieces: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    """The inverse of :func:`split_at_discontinuities`: one table with steps.

    Consecutive pieces must touch (``prev.x[-1] == next.x[0]``). Where they
    agree in value the shared point is written once; where they do not it is
    written twice, which is the step. A gap between pieces would need values
    nobody stated and is refused.
    """
    if not pieces:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
    xs, ys, ls = [], [], []
    for i, (x, y, laws) in enumerate(pieces):
        x, y, laws = validate(x, y, laws)
        if i == 0:
            xs.append(x), ys.append(y), ls.append(laws)
            continue
        if x[0] != xs[-1][-1]:
            raise ValueError(f"pieces {i - 1} and {i} do not touch: "
                             f"{xs[-1][-1]!r} != {x[0]!r}")
        if y[0] == ys[-1][-1]:
            xs.append(x[1:]), ys.append(y[1:]), ls.append(laws)
        else:
            # The zero-width interval of the step has no law; lin-lin is the
            # placeholder every reader ignores.
            xs.append(x), ys.append(y), ls.append(np.r_[LINLIN, laws])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ls).astype(np.int64)
