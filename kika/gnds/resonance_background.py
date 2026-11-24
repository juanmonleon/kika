"""ENDF's MF3 background written as GNDS's ``resonancesWithBackground`` (§16.1.1).

A suite decoded from ENDF keeps MF3 as the table it is; only the written file says
"add the resonances". Which MTs, and how the background is cut at the resonance
edges, follow FUDGE's ENDF->GNDS (``ENDF_ITYPE_0.py`` after ``readMF2``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

import numpy as np

RESONANCES_HREF = "/reactionSuite/resonances"
MERGE = "mergeWithNextRegion"
DELETE = "deletePoint"


@dataclass(frozen=True)
class Plan:
    mts: FrozenSet[int]
    resolvedMax: Optional[float]
    unresolvedMax: Optional[float]
    fissionInResolved: bool


def plan(suite) -> Optional[Plan]:
    """What to wrap in *suite*, or ``None`` when its resonances are not added to MF3."""
    from kika.nuclear_data.model import RMatrix

    resonances = getattr(suite, "resonances", None)
    provenance = getattr(suite, "provenance", None)
    if resonances is None or getattr(provenance, "sourceFormat", None) != "endf":
        return None
    # LRP=2: the resonances are for self-shielding only and MF3 is the whole σ.
    if (getattr(provenance, "headerFields", None) or {}).get("lrp") == 2:
        return None
    resolved = [r for r in resonances.resolved if r.formalism is not None]
    unresolved = resonances.unresolved
    widths = getattr(unresolved, "tabulatedWidths", None)
    haveUnresolved = widths is not None and not widths.selfShieldingOnly
    if not resolved and not haveUnresolved:
        return None
    mts = {1, 2, 3, 18, 19, 102}
    fission = False
    for region in resolved:
        formalism = region.formalism
        if isinstance(formalism, RMatrix):
            for reaction in formalism.resonanceReactions:
                if reaction.reactionMT is not None:
                    mts.add(int(reaction.reactionMT))
                    fission = fission or int(reaction.reactionMT) == 18
        else:
            groups = getattr(getattr(formalism, "resonanceParameters", None), "spinGroups", [])
            fission = fission or any(r.fissionWidth for g in groups for r in g.resonances)
    if 51 in mts:
        mts.add(4)
    if 600 in mts:
        mts.add(103)
    return Plan(mts=frozenset(mts),
                resolvedMax=max(r.domainMax for r in resolved) if resolved else None,
                unresolvedMax=unresolved.domainMax if haveUnresolved else None,
                fissionInResolved=fission)


def wrap(how: Plan, mt: Optional[int], form, label: str
         ) -> Optional[Tuple[object, Dict[str, str]]]:
    """*form* as a ``resonancesWithBackground``, with FUDGE's flags by background region."""
    from kika.nuclear_data.model import (Background, Regions1d, ResonancesWithBackground,
                                         XYs1d)

    if mt is None or int(mt) not in how.mts:
        return None
    mt = int(mt)
    if mt in (18, 19) and how.unresolvedMax is None and not how.fissionInResolved:
        return None
    if isinstance(form, XYs1d):
        pieces, axes = [form], form.axes
    elif isinstance(form, Regions1d) and form.function1ds:
        pieces, axes = list(form.function1ds), form.axes or form.function1ds[0].axes
    else:
        return None
    top = how.unresolvedMax if how.unresolvedMax is not None else how.resolvedMax
    if pieces[0].domainMin > top:
        return None
    fast = pieces[-1].domainMax > top
    edges = []
    if how.resolvedMax is not None and (how.unresolvedMax is not None or fast):
        edges.append(("resolvedRegion", how.resolvedMax))
    if how.unresolvedMax is not None and fast:
        edges.append(("unresolvedRegion", how.unresolvedMax))

    flags: Dict[str, str] = {}
    for name, edge in edges:
        if any(p.domainMax == edge for p in pieces[:-1]):
            continue
        index = next((k for k, p in enumerate(pieces) if p.domainMin < edge < p.domainMax), None)
        if index is None:
            continue
        piece = pieces[index]
        left, right, stated = _split(piece, edge)
        pieces[index:index + 1] = [left, right]
        if stated is not None:
            flags[name] = MERGE if stated else f"{MERGE},{DELETE}"

    def take(upper):
        group = [p for p in pieces if p.domainMax <= upper]
        del pieces[:len(group)]
        if not group:
            return None
        return group[0] if len(group) == 1 else Regions1d(function1ds=group, axes=axes)

    resolvedRegion = take(how.resolvedMax) if how.resolvedMax is not None else None
    unresolvedRegion = take(how.unresolvedMax) if how.unresolvedMax is not None else None
    fastRegion = (pieces[0] if len(pieces) == 1
                  else Regions1d(function1ds=pieces, axes=axes) if pieces else None)
    background = Background(resolvedRegion=resolvedRegion,
                            unresolvedRegion=unresolvedRegion, fastRegion=fastRegion)
    return (ResonancesWithBackground(background=background,
                                     resonanceRegionHref=RESONANCES_HREF, label=label),
            flags)


def _split(piece, edge: float):
    """``(left, right, stated)``: *piece* cut at *edge*; ``stated`` says whether the tape had the point.

    ``stated`` is ``None`` for a jump at the edge (the energy twice): each side keeps
    its own value and the join needs no flag.
    """
    from kika.nuclear_data.model import XYs1d

    xs, ys = np.asarray(piece.xs, float), np.asarray(piece.ys, float)
    at = np.flatnonzero(xs == edge)
    if at.size > 1:
        cut, stated = at[0] + 1, None
        lx, ly, rx, ry = xs[:cut], ys[:cut], xs[cut:], ys[cut:]
    elif at.size == 1:
        i, stated = at[0], True
        lx, ly, rx, ry = xs[:i + 1], ys[:i + 1], xs[i:], ys[i:]
    else:
        y = float(piece.evaluate(edge))
        below = xs < edge
        lx, ly = np.append(xs[below], edge), np.append(ys[below], y)
        rx, ry = np.insert(xs[~below], 0, edge), np.insert(ys[~below], 0, y)
        stated = False

    def part(x, y):
        return XYs1d(xs=x, ys=y, interpolation=piece.interpolation, axes=piece.axes)

    return part(lx, ly), part(rx, ry), stated
