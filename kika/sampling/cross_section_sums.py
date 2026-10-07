"""MF3's sum rules on a realisation: the partials govern, the sums follow.

ENDF-6 states a summed cross section twice -- MT4 *and* MT51-91, MT1 *and* MT2
plus everything non-elastic -- and requires the two statements to agree. A
realisation that moves MT51 alone leaves MT4, MT3 and MT1 stating the old sum,
and a realisation that perturbs MT1 *and* MT2 from their own blocks leaves a
total that is not the sum of anything. Both tapes parse; neither is an
evaluation.

The rule this module applies is decision 3 of
``kika-workspace/docs/library/perturbation_model_roadmap.md``, completed on
2026-10-07, with its casuistry in ``mf3_perturbation_casuistry.md`` beside it.
Its principle: **a covariance block perturbs the cross section it was stated
for, and no other.** A block applied to a section it was not given for pairs a
central value with an uncertainty nobody stated, and the realisation stops
being a draw from the evaluation.

1. **A partial with its own block is perturbed by it.** Partials govern: when a
   request names a sum and some of its parts, the parts keep their own factors.
2. **A partial without one is not perturbed.** MT4's block says nothing about
   MT53 on its own -- only about MT51 + ... + MT91 together -- so carrying it to
   MT53 would invent a covariance (same relative uncertainty, fully correlated
   with the sum) the file never states.
3. **Every sum with a moved partial under it is re-derived**, and a sum's own
   block is never applied to the sum itself: the sum has to equal its parts,
   so it moves by what they moved. Its block is *discarded*, and the run says
   so.
4. **A sum's block with no partial of its own under it has nowhere to go.**
   Applying it to the sum alone breaks (3), and NJOY's RECONR would discard it
   anyway (it rebuilds MT1 and MT4 from the partials). Such a block is refused
   -- see :func:`screenSumClaims` -- unless the caller opts into
   ``distribute=True``.

``distribute=True`` is the explicit, recorded assumption that a sum's block may
be carried to its partials: each partial without a block of its own then rides
the nearest perturbed sum above it, by the same factor. That keeps the sum
consistent with its own covariance and gives the partials one the file does not
state. It is what ``apply_factors_to_pendf_mf3`` does for a composite the PENDF
lacks, and it is off by default.

**Re-derived as a delta, not as a fresh sum.** ``S' = S + sum_p (p' - p)`` over
the leaf partials under ``S``. Summing the partials afresh would also "repair"
whatever the evaluation's own sum residual was -- JEFF-4.0 U-235 MT1 sits 2.3 %
off its partials, JENDL-5 Fe-56 MT1 0.88 % -- and move the total everywhere,
including where nothing was perturbed; on a cut-down tape
(``micro_fe56_xs_and_angular.endf`` keeps MT1, MT2 and MT102 of a full Fe-56)
it would replace the total by the sum of three survivors. The delta moves the
sum by exactly what its parts moved and nowhere else, so outside the
covariance's energy range the section is the evaluation's, value for value.

Which MT sums which is :data:`kika._constants.MF3_SUM_RULES`, resolved against
what the tape carries by
:func:`~kika.endf.writers.redundant.resolve_sum_components` -- the same table
the decoder uses to put an MT in ``suite.sums`` and the writer uses to resum.
**What a sum is made of is MF3's question, not MF33's**: MT4 is the sum of every
MT51-91 the tape states, whichever of them carry a covariance. The rules form a
tree (no MT has two parents; checked in the tests), which is what makes "the
partials under a sum" and "the nearest perturbed sum above it" definite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from kika.sampling.joint_blocks import ComponentKey

__all__ = ["SumPlan", "SumScreen", "planCrossSectionSums", "rederiveSum",
           "screenSumClaims", "suiteSumLayout", "sumLeaves", "sumTree"]


def suiteSumLayout(suite) -> Tuple[set, set]:
    """``(present, sums)``: every MF3 MT the suite carries, and its sums."""
    present = {int(r.ENDF_MT) for container in (suite.reactions, suite.sums)
               for r in container if getattr(r, "ENDF_MT", None) is not None}
    sums = {int(r.ENDF_MT) for r in suite.sums
            if getattr(r, "ENDF_MT", None) is not None}
    return present, sums


def sumTree(present: Iterable[int], sums: Iterable[int]
            ) -> Tuple[Dict[int, Tuple[int, ...]], Dict[int, int]]:
    """``(children, parent)`` of the sums the tape carries.

    *children* of a sum are its partials **as this tape states them**: MT1's
    are ``(2, 3)`` on a tape that writes MT3 and ``(2, 4, 16, 102, ...)`` on one
    that does not. A partial that is itself a sum appears as one node, not as
    its own partials, so the tree has every MT the tape carries exactly once.
    """
    from kika.endf.writers.redundant import resolve_sum_components

    available = {int(mt) for mt in present}
    children = {int(s): tuple(int(c) for c in resolve_sum_components(int(s), available))
                for s in sums}
    parent: Dict[int, int] = {}
    for total, parts in children.items():
        for part in parts:
            if part in parent and parent[part] != total:
                raise ValueError(
                    f"MT{part} is a partial of both MT{parent[part]} and "
                    f"MT{total}; the sum rules are meant to be a tree, and "
                    f"which sum governs it would be a guess")
            parent[part] = total
    return children, parent


def _mtList(mts) -> str:
    """``[51, 52, 53, 91]`` as ``MT51-53, 91``."""
    mts = sorted(int(mt) for mt in mts)
    if not mts:
        return "no partial"
    runs, start = [], None
    for index, mt in enumerate(mts):
        if start is None:
            start = mt
        if index + 1 == len(mts) or mts[index + 1] != mt + 1:
            runs.append(f"{start}" if start == mt else f"{start}-{mt}")
            start = None
    return "MT" + ", ".join(runs)


def _leavesUnder(total: int, children: Mapping[int, Tuple[int, ...]]
                 ) -> Tuple[int, ...]:
    out: List[int] = []
    for part in children.get(total, ()):
        if part in children:
            out.extend(_leavesUnder(part, children))
        else:
            out.append(part)
    return tuple(out)


def _depth(mt: int, parent: Mapping[int, int]) -> int:
    depth = 0
    while mt in parent:
        mt = parent[mt]
        depth += 1
    return depth


@dataclass(frozen=True)
class SumPlan:
    """What a realisation does to MF3 once the sum rules are applied."""

    #: Leaf partial -> the component whose block it is perturbed by: its own,
    #: or, only with ``distribute=True``, the nearest perturbed sum above it.
    leafControl: Dict[int, ComponentKey] = field(default_factory=dict)
    #: The sums to re-derive, deepest first (MT4 before MT3 before MT1).
    rederive: Tuple[int, ...] = ()
    #: Sum -> the leaf partials that moved under it.
    movedUnder: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    #: Perturbed sum -> the leaves its own block reached. Empty means the block
    #: was discarded: every partial under it had a block of its own.
    ownBlockReached: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    #: Perturbed sums the tape does not state in MF3, only their partials:
    #: ENDF/B-VIII.1 Fe-56 gives MF33 for MT103 and MT107 and MF3 for
    #: MT600-649 and MT800-849 alone (NJOY builds MT103 itself). Their block
    #: reaches the partials and there is no section of theirs to rebuild.
    virtual: Tuple[int, ...] = ()


def _virtualClaims(claims: Iterable[int], present: set) -> set:
    """Claimed MTs the tape states no MF3 section for, only partials of."""
    from kika.endf.writers.redundant import resolve_sum_components

    virtual = set()
    for mt in claims:
        if int(mt) in present:
            continue
        if not resolve_sum_components(int(mt), present):
            raise KeyError(
                f"no reaction with ENDF_MT {mt} anywhere in this suite, and none "
                f"of its partials either: the realisation perturbs a cross "
                f"section the tape does not state")
        virtual.add(int(mt))
    return virtual


@dataclass(frozen=True)
class SumScreen:
    """Which claimed sums the strict rule leaves without a use."""

    #: Sum -> the leaf partials under it that carry a block of their own. The
    #: sum's block is not applied; the sum is re-derived from those partials.
    discarded: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    #: Sum -> every leaf partial under it, none of which carries a block. The
    #: sum's block could only reach the tape by being carried to them.
    unreachable: Dict[int, Tuple[int, ...]] = field(default_factory=dict)


def sumLeaves(present: Iterable[int], sums: Optional[Iterable[int]] = None,
              claims: Iterable[int] = ()) -> Dict[int, Tuple[int, ...]]:
    """``{sum: leaf partials}`` for every sum the tape states, as MF3 has them.

    *present* is every MF3 MT; *sums* the ones that are sums, or ``None`` to
    derive them the way the decoder does (an MT the tape states *and* gives
    the partials of), so a caller holding only the parsed tape's MF3 listing
    gets the tree a run will use. *claims* adds the virtual sums -- MTs a
    covariance is stated for and MF3 holds only the partials of. A claim that
    is neither (ENDF/B-VIII.1 U-238's lumped MT851) is left out rather than
    refused: this is a question about the tape's layout, asked by tools that
    describe it.
    """
    from kika.endf.writers.redundant import resolve_sum_components

    present = {int(mt) for mt in present}
    if sums is None:
        sums = {mt for mt in present if resolve_sum_components(mt, present)}
    virtual = {int(mt) for mt in claims
               if int(mt) not in present
               and resolve_sum_components(int(mt), present)}
    allSums = {int(s) for s in sums} | virtual
    children, _parent = sumTree(present | virtual, allSums)
    return {total: _leavesUnder(total, children) for total in sorted(allSums)}


def screenSumClaims(claims: Iterable[int], present: Iterable[int],
                    sums: Iterable[int]) -> SumScreen:
    """Sort the claimed sums by what the strict rule does with their blocks.

    *claims* is every MT a block is drawn for (MF33, or MF34's L=0 magnitude);
    *present* and *sums* as :func:`suiteSumLayout` gives them. A claimed sum
    lands in exactly one of :class:`SumScreen`'s two maps: some partial under
    it is claimed too (its block is discarded and the sum re-derived), or none
    is (its block has nowhere to go). Which partials a sum has is read from
    MF3, so a virtual sum -- a covariance for MT103 on a tape that states only
    MT600-649 -- counts its MF3 partials like any other.
    """
    claims = {int(mt) for mt in claims}
    present = {int(mt) for mt in present}
    sums = {int(s) for s in sums}
    virtual = _virtualClaims(claims, present)
    children, _parent = sumTree(present | virtual, sums | virtual)
    discarded: Dict[int, Tuple[int, ...]] = {}
    unreachable: Dict[int, Tuple[int, ...]] = {}
    for total in sorted((sums | virtual) & claims):
        leaves = _leavesUnder(total, children)
        own = tuple(leaf for leaf in leaves if leaf in claims)
        if own:
            discarded[total] = own
        else:
            unreachable[total] = leaves
    return SumScreen(discarded=discarded, unreachable=unreachable)


def planCrossSectionSums(claims: Mapping[int, ComponentKey],
                         present: Iterable[int], sums: Iterable[int], *,
                         distribute: bool = False) -> SumPlan:
    """Decide, per MT, which block moves it and which sums are rebuilt.

    *claims* is ``MT -> component`` for every cross section the realisation
    perturbs (MF33, or MF34's L=0 magnitude). *present* is every MF3 MT of the
    suite and *sums* the ones the decoder recognised as sums.

    By default a leaf moves only by its own block, and a claimed sum with no
    claimed partial under it raises: its block has nowhere it may go (rule 4
    of the module docstring). With *distribute*, a leaf without a block rides
    the nearest claimed sum above it instead.
    """
    sums = {int(s) for s in sums}
    present = {int(mt) for mt in present}
    virtual = _virtualClaims(claims, present)
    children, parent = sumTree(present | virtual, sums | virtual)

    if not distribute:
        screen = screenSumClaims(claims, present, sums)
        if screen.unreachable:
            named = "; ".join(
                f"MT{total} over {_mtList(leaves)}"
                for total, leaves in screen.unreachable.items())
            raise ValueError(
                f"{named}: a block is drawn for the sum and for none of its "
                f"partials, and a covariance perturbs only the cross section "
                f"it was stated for. Applied to the sum alone it would leave "
                f"the sum unequal to its parts (and NJOY rebuilds MT1 and MT4 "
                f"from the parts anyway); carried to the partials it would "
                f"give them an uncertainty the file does not state. Ask for "
                f"partials that carry a covariance, or pass "
                f"distributeSums=True to carry the sum's block to them as a "
                f"stated assumption")

    leafControl: Dict[int, ComponentKey] = {}
    for leaf in sorted(present - sums):
        if not distribute:
            if leaf in claims:
                leafControl[leaf] = claims[leaf]
            continue
        node: Optional[int] = leaf
        while node is not None and node not in claims:
            node = parent.get(node)
        if node is not None:
            leafControl[leaf] = claims[node]

    movedUnder = {}
    for total in sums:  # not the virtual ones: they have no section to rebuild
        moved = tuple(leaf for leaf in _leavesUnder(total, children)
                      if leaf in leafControl)
        if moved:
            movedUnder[total] = moved
    rederive = tuple(sorted(movedUnder, key=lambda s: (-_depth(s, parent), s)))

    ownBlockReached = {}
    for total in sorted((sums | virtual) & set(claims)):
        ownBlockReached[total] = tuple(
            leaf for leaf, component in leafControl.items()
            if component == claims[total])
    return SumPlan(leafControl=leafControl, rederive=rederive,
                   movedUnder=movedUnder, ownBlockReached=ownBlockReached,
                   virtual=tuple(sorted(virtual)))


# ----------------------------------------------------------------------
# The arithmetic: S' = S + sum (p' - p)
# ----------------------------------------------------------------------

def _limits(function1d, energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Left and right limits of a tabulated cross section at *energies*.

    A repeated abscissa is ENDF's step, so a single value per energy cannot
    describe it: the first copy is the limit from below and the last the limit
    from above. Elsewhere the two agree and come from the node's own
    interpolation. Outside its table a cross section is zero, so at its first
    abscissa the left limit is zero and at its last the right one is.
    """
    xs, ys, _pairs = function1d.toEndfRegions()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    values = np.asarray(function1d.evaluate(energies, outOfRange="zero"),
                        dtype=float).copy()
    left, right = values, values.copy()
    if xs.size == 0:
        return np.zeros_like(left), np.zeros_like(right)
    first = np.searchsorted(xs, energies, side="left")
    last = np.searchsorted(xs, energies, side="right")
    hit = last > first
    left[hit] = ys[first[hit]]
    right[hit] = ys[last[hit] - 1]
    left[energies <= xs[0]] = 0.0
    right[energies >= xs[-1]] = 0.0
    return left, right


def rederiveSum(total, moves: Sequence[Tuple[object, object, float, float]]):
    """``total`` moved by what its partials moved: ``S' = S + sum (p' - p)``.

    Parameters
    ----------
    total
        The sum's evaluated form (``XYs1d`` or ``Regions1d``). Not mutated.
    moves
        One ``(before, after, lo, hi)`` per leaf partial that moved: its
        evaluated form, its perturbed form, and the energy span of the factor
        block that moved it. Outside ``[lo, hi]`` the factor is one, so the
        partial did not move there and the sum must not either -- this is
        enforced rather than left to round-off, because two tables of the same
        line interpolated over different sub-segments agree to 1e-16 and not
        exactly, and "untouched" here means the evaluation's own value.

    Returns
    -------
    (form, info)
        A node of the same kind as *total*, keeping its interpolation regions
        and laws; points are added where a partial moved (its grid, and the
        steps at the factor block's edges). *info* carries ``n_inserted``,
        ``max_rel_change`` and ``n_negative`` -- a negative total can only come
        from an evaluation whose sum sits below its own parts, and the caller
        must say so rather than write it. A negative point where a moved
        partial is itself negative (a background inside the resolved range,
        which MF3 may state below zero) or where the evaluation's own sum
        already was is not counted.
    """
    from kika.nuclear_data.model.functions import Regions1d, XYs1d

    xs, ys, pairs = total.toEndfRegions()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return total, {"n_inserted": 0, "max_rel_change": 0.0, "n_negative": 0}
    lo, hi = float(xs[0]), float(xs[-1])

    extra = []
    for _before, after, spanLo, spanHi in moves:
        grid = np.asarray(after.toEndfRegions()[0], dtype=float)
        keep = ((grid >= max(lo, spanLo)) & (grid <= min(hi, spanHi)))
        extra.append(grid[keep])
    energies = np.unique(np.concatenate([xs] + extra))

    deltaLeft = np.zeros(energies.size)
    deltaRight = np.zeros(energies.size)
    # Where a partial is itself negative -- a background inside the resolved
    # range, which MF3 is allowed to state below zero -- a negative sum is not
    # evidence of anything, so it is not counted as one.
    signedPart = np.zeros(energies.size, dtype=bool)
    for before, after, spanLo, spanHi in moves:
        aL, aR = _limits(after, energies)
        bL, bR = _limits(before, energies)
        signedPart |= (aL < 0) | (aR < 0) | (bL < 0) | (bR < 0)
        dL, dR = aL - bL, aR - bR
        # The block steps from 1 to its first factor at spanLo and back at
        # spanHi, so the left limit at spanLo and the right one at spanHi are
        # outside it.
        dL[(energies <= spanLo) | (energies > spanHi)] = 0.0
        dR[(energies < spanLo) | (energies >= spanHi)] = 0.0
        deltaLeft += dL
        deltaRight += dR

    sL, sR = _limits(total, energies)
    # The sum's own end points have one side only, and it is the inside one.
    sL[0], sR[-1] = sR[0], sL[-1]
    deltaLeft[0], deltaRight[-1] = deltaRight[0], deltaLeft[-1]

    newXs: List[float] = []
    newYs: List[float] = []
    newAt: List[int] = []                   # new point -> its index in energies
    lastOf = np.empty(xs.size, dtype=int)   # old point -> its index in the new table
    first = np.searchsorted(xs, energies, side="left")
    last = np.searchsorted(xs, energies, side="right")
    for k, energy in enumerate(energies):
        copies = int(last[k] - first[k])
        valueLeft = sL[k] + deltaLeft[k]
        valueRight = sR[k] + deltaRight[k]
        if copies >= 2:
            # The evaluation steps here already: keep every copy, the first as
            # the limit from below and the rest from above.
            for offset in range(copies):
                old = int(first[k]) + offset
                if offset == 0:
                    value = ys[old] + deltaLeft[k]
                else:
                    value = ys[old] + deltaRight[k]
                newXs.append(energy)
                newYs.append(value)
                newAt.append(k)
                lastOf[old] = len(newXs) - 1
            continue
        if valueLeft == valueRight:
            newXs.append(energy)
            newYs.append(valueLeft)
            newAt.append(k)
        else:
            newXs.extend((energy, energy))
            newYs.extend((valueLeft, valueRight))
            newAt.extend((k, k))
        if copies == 1:
            lastOf[int(first[k])] = len(newXs) - 1

    newXs = np.asarray(newXs, dtype=float)
    newYs = np.asarray(newYs, dtype=float)
    # An old point keeps its own value where nothing moved -- the delta is
    # exactly zero there -- so this is the evaluation's number, not a copy of it
    # through arithmetic that could round.
    same = (deltaLeft == 0.0) & (deltaRight == 0.0)
    for k in np.flatnonzero(same):
        if last[k] - first[k] == 1:
            newYs[lastOf[int(first[k])]] = ys[int(first[k])]

    newPairs = [(int(lastOf[int(nbt) - 1]) + 1, int(code)) for nbt, code in pairs]
    if isinstance(total, Regions1d):
        out = Regions1d.fromEndfRegions(newXs, newYs, newPairs, axes=total.axes,
                                        label=total.label)
        out.outerDomainValue = total.outerDomainValue
        out.index = total.index
    else:
        out = XYs1d(xs=newXs, ys=newYs, interpolation=total.interpolation,
                    axes=total.axes, label=total.label,
                    outerDomainValue=total.outerDomainValue, index=total.index)

    old = np.asarray(total.evaluate(xs, outOfRange="zero"), dtype=float)
    moved = np.asarray(out.evaluate(xs, outOfRange="zero"), dtype=float)
    nonzero = old != 0.0
    maxRel = float(np.max(np.abs(moved[nonzero] / old[nonzero] - 1.0))) \
        if nonzero.any() else 0.0
    newAt = np.asarray(newAt, dtype=int)
    ownNegative = (sL < 0) | (sR < 0)
    suspicious = (newYs < 0.0) & ~signedPart[newAt] & ~ownNegative[newAt]
    info = {"n_inserted": int(newXs.size - xs.size),
            # Relative to the evaluation's value, so it is large wherever that
            # is near zero -- a background in the resolved range, typically.
            # The absolute move is the one to read there.
            "max_rel_change": maxRel,
            "max_abs_change": float(np.max(np.abs(moved - old))) if xs.size else 0.0,
            "n_negative": int(np.count_nonzero(suspicious))}
    return out, info
