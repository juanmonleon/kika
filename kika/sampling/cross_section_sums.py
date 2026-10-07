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
Its principle: **the file says with which covariance each section moves, and a
sum's covariance is used only where the file does not break the sum down.**

1. **A partial with its own block is perturbed by it.** Partials govern: when a
   request names a sum and some of its parts, the parts keep their own factors.
2. **Every sum with a moved partial under it is re-derived**, and a sum's own
   block is never applied to the sum itself: the sum has to equal its parts,
   so it moves by what they moved.
3. **A sum the file decomposes** -- some section under it (a partial, or a
   smaller sum) carries a covariance of its own -- is perturbed through those
   sections and nothing else. Its own block is *discarded*, and the partials
   under it that carry none stay as evaluated: MT4's block is the uncertainty
   of MT51 + ... + MT91 together, and with MT51 and MT52 stated apart the file
   no longer says what the rest share of it.
4. **A sum the file does not decompose** -- nothing under it carries a
   covariance -- moves its partials by its own factor, all of them alike. The
   covariance exists; it is just not broken down, and this is the only reading
   of it that reaches the tape (NJOY's RECONR rebuilds MT1 and MT4 from the
   partials, so a factor left on the sum is lost).

The three ``mode`` values of :func:`planCrossSectionSums` (and ``sumBlocks`` of
``perturbFromModel``) are ``"undecomposed"`` -- rules 1-4, the default --
``"fill"``, which in case 3 also carries the sum's block to the partials that
carry none (an assumption the file does not make; what
``apply_factors_to_pendf_mf3`` does for a composite the PENDF lacks), and
``"never"``, which drops rule 4 and uses no sum's block at all. SANDY's
``Samples.iterate_xs_samples`` is ``"undecomposed"`` one level deep.

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

__all__ = ["SUM_BLOCK_MODES", "SumPlan", "SumScreen", "planCrossSectionSums",
           "rederiveSum", "screenSumClaims", "suiteSumLayout", "sumLeaves",
           "sumMembers", "sumTree"]

#: What a sum's own covariance block may do -- see the module docstring.
SUM_BLOCK_MODES = ("undecomposed", "fill", "never")


def checkSumBlockMode(mode: str) -> str:
    if mode not in SUM_BLOCK_MODES:
        raise ValueError(f"sumBlocks must be one of {SUM_BLOCK_MODES}, got {mode!r}")
    return mode


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
    #: or the nearest perturbed sum above it (see ``mode``).
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
    #: Partial -> the remainder statements that move it (its anchor is drawn).
    remainders: Dict[int, Tuple["Remainder", ...]] = field(default_factory=dict)
    #: Lump member -> its lumped reaction (MTL), for the ones a lump moves.
    lumpOf: Dict[int, int] = field(default_factory=dict)
    #: Index into ``rederive`` at which the remainders are taken: the sums
    #: before it contain none, the ones from it on contain one.
    remainderAt: int = 0
    #: Virtual sum -> its leaf partials: a remainder may subtract one.
    virtualLeaves: Dict[int, Tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Remainder:
    """MF33 states this partial's covariance as a total's minus the others'.

    ENDF-6 §33.2's NC-type LTY=0 with one coefficient +1 (*total*) and the
    rest -1 (*minus*), over ``[lo, hi]`` eV: JENDL-5 Fe-56's MT2 is
    MT1 - (MT16 + MT22 + ... + MT115), its O-16's MT2 the same above 6 MeV,
    and B-VIII.1/JEFF-4.0 O-16's MT5 is MT1 - MT2. The covariance of the
    partial is then *defined* by the total's and the others', so the
    realisation that honours it is the arithmetic itself: the total moves by
    its own block, the others by theirs, and this partial takes the
    difference -- ``p' = p + (T' - T) - sum (X' - X)`` -- after which the
    total, re-derived from its partials, is ``T'`` again. No covariance is
    invented and none is approximated.
    """

    total: int
    minus: Tuple[int, ...]
    lo: float
    hi: float


def readSumStatements(endfObj, present: Iterable[int], sums: Iterable[int]
                      ) -> Tuple[Dict[int, Tuple[Remainder, ...]],
                                 Dict[int, Tuple[int, ...]], List[str]]:
    """``(remainders, lumped, notes)`` from a parsed tape's MF33.

    *lumped* is ``{MTL: members}``: ENDF-6 §33.2's lumped reactions
    (MT851-870), whose covariance the members' own sections point to with a
    non-zero MTL instead of stating one -- B-VIII.1 U-235's MT851 = MT52-91.
    The file states the lump's block *for* the members, so they move by it.

    *remainders* is ``{partial: (Remainder, ...)}`` (see :class:`Remainder`).
    An NC statement whose coefficients are all +1 is a derived *sum* and needs
    nothing here (the sum rules already re-derive it); one that fits neither
    shape, or whose partial is itself a sum in MF3, is left out and named in
    *notes* rather than guessed at.
    """
    from kika.endf.writers.redundant import resolve_sum_components  # noqa: F401

    present = {int(mt) for mt in present}
    sums = {int(s) for s in sums}
    remainders: Dict[int, List[Remainder]] = {}
    lumped: Dict[int, List[int]] = {}
    notes: List[str] = []
    mf = getattr(endfObj, "mf", None) or {}
    if 33 not in mf:
        return {}, {}, []
    children, parent = sumTree(present, sums)
    for mt, section in sorted(mf[33].mt.items()):
        mtl = int(getattr(section, "_mtl", 0) or 0)
        if mtl:
            lumped.setdefault(mtl, []).append(int(mt))
            continue
        for sub in getattr(section, "subsections", ()):
            if int(sub.mt1) != int(mt):
                continue
            for nc in sub.nc_records:
                if nc.lty != 0:
                    continue
                terms = [(float(c), int(round(x))) for c, x in zip(nc.ci, nc.xmti)]
                plus = [x for c, x in terms if c > 0]
                minus = [x for c, x in terms if c < 0]
                if not minus:
                    continue                      # a derived sum: nothing to do
                if (len(plus) != 1 or any(abs(abs(c) - 1.0) > 1e-9 for c, _ in terms)):
                    notes.append(
                        f"MT{mt}'s MF33 is stated as {terms[:6]}..., which is "
                        f"neither a sum nor a total minus the rest; that part of "
                        f"its covariance is not applied")
                    continue
                total = plus[0]
                node, ancestors = int(mt), []
                while node in parent:
                    node = parent[node]
                    ancestors.append(node)
                if int(mt) in sums:
                    notes.append(
                        f"MT{mt}'s MF33 states it as MT{total} minus the rest, "
                        f"and MT{mt} is itself a sum in MF3; that part of its "
                        f"covariance is not applied (its own NI part is)")
                    continue
                if int(mt) not in present or total not in ancestors:
                    notes.append(
                        f"MT{mt}'s MF33 states it as MT{total} minus the rest, "
                        f"and MF3 does not put MT{mt} under MT{total}; not applied")
                    continue
                remainders.setdefault(int(mt), []).append(Remainder(
                    total=total, minus=tuple(minus), lo=float(nc.e1),
                    hi=float(nc.e2)))
    return ({mt: tuple(rs) for mt, rs in remainders.items()},
            {mtl: tuple(sorted(m)) for mtl, m in lumped.items()}, notes)


def _effectiveClaims(claims: Mapping[int, object], present: set,
                     lumped: Optional[Mapping[int, Tuple[int, ...]]]
                     ) -> Tuple[Dict[int, object], Dict[int, int]]:
    """Claims with every claimed lump replaced by its members in MF3.

    A member that carries a block of its own keeps it; the others take the
    lump's. ``lumpOf`` maps each member a lump moves to the lump.
    """
    out = {int(mt): value for mt, value in claims.items()
           if not lumped or int(mt) not in lumped}
    lumpOf: Dict[int, int] = {}
    for lump, members in (lumped or {}).items():
        if lump not in claims:
            continue
        for member in members:
            if member in present and member not in out:
                out[member] = claims[lump]
                lumpOf[member] = int(lump)
    return out, lumpOf


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
    """The claimed sums, sorted by whether the file decomposes them."""

    #: Decomposed sum -> the sections under it (partials or smaller sums, any
    #: depth) that carry a block of their own. The sum's block is not applied;
    #: the sum is re-derived from what those move.
    discarded: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    #: Undecomposed sum -> every leaf partial under it, none of which (nor any
    #: smaller sum between) carries a block. Its block reaches the tape only by
    #: moving them.
    undecomposed: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    #: Decomposed sum whose block is still drawn, because MF33 states a partial
    #: under it as the remainder (:class:`Remainder`) -> those partials.
    anchored: Dict[int, Tuple[int, ...]] = field(default_factory=dict)


def _descendants(total: int, children: Mapping[int, Tuple[int, ...]]
                 ) -> Tuple[int, ...]:
    out: List[int] = []
    for part in children.get(total, ()):
        out.append(part)
        if part in children:
            out.extend(_descendants(part, children))
    return tuple(out)


def sumMembers(present: Iterable[int], sums: Optional[Iterable[int]] = None,
               claims: Iterable[int] = ()) -> Dict[int, Tuple[int, ...]]:
    """``{sum: every section under it}``, smaller sums included, any depth.

    The question "does the file decompose this sum?" is whether any of these
    carries a covariance. Arguments as for :func:`sumLeaves`.
    """
    children, allSums = _layout(present, sums, claims)
    return {total: _descendants(total, children) for total in sorted(allSums)}


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
    children, allSums = _layout(present, sums, claims)
    return {total: _leavesUnder(total, children) for total in sorted(allSums)}


def _layout(present, sums, claims):
    from kika.endf.writers.redundant import resolve_sum_components

    present = {int(mt) for mt in present}
    if sums is None:
        sums = {mt for mt in present if resolve_sum_components(mt, present)}
    virtual = {int(mt) for mt in claims
               if int(mt) not in present
               and resolve_sum_components(int(mt), present)}
    allSums = {int(s) for s in sums} | virtual
    children, _parent = sumTree(present | virtual, allSums)
    return children, allSums


def screenSumClaims(claims: Iterable[int], present: Iterable[int],
                    sums: Iterable[int], *,
                    lumped: Optional[Mapping[int, Tuple[int, ...]]] = None,
                    remainders: Optional[Mapping[int, Tuple[Remainder, ...]]] = None
                    ) -> SumScreen:
    """Sort the claimed sums by whether the file decomposes them.

    *claims* is every MT a block is drawn for (MF33, or MF34's L=0 magnitude);
    *present* and *sums* as :func:`suiteSumLayout` gives them. A claimed sum
    lands in exactly one of :class:`SumScreen`'s two maps: some section under
    it -- a partial, or a smaller sum -- is claimed too (decomposed: its block
    is discarded and the sum re-derived), or none is (undecomposed). Which
    sections a sum has is read from MF3, so a virtual sum -- a covariance for
    MT103 on a tape that states only MT600-649 -- counts its MF3 partials like
    any other.

    A claimed lump (*lumped*, MT851...) counts as claims on its members. A
    decomposed sum that is the total of a remainder statement on a partial
    present in MF3 (*remainders*) is *anchored* instead of discarded: its block
    is what the remainder is made of.
    """
    present = {int(mt) for mt in present}
    claims, _lumpOf = _effectiveClaims({int(mt): mt for mt in claims}, present,
                                       lumped)
    claims = set(claims)
    sums = {int(s) for s in sums}
    virtual = _virtualClaims(claims, present)
    children, _parent = sumTree(present | virtual, sums | virtual)
    anchorsOf: Dict[int, List[int]] = {}
    for partial, statements in (remainders or {}).items():
        if partial in present:
            for statement in statements:
                anchorsOf.setdefault(statement.total, []).append(int(partial))
    discarded: Dict[int, Tuple[int, ...]] = {}
    undecomposed: Dict[int, Tuple[int, ...]] = {}
    anchored: Dict[int, Tuple[int, ...]] = {}
    for total in sorted((sums | virtual) & claims):
        own = tuple(mt for mt in _descendants(total, children) if mt in claims)
        if total in anchorsOf:
            anchored[total] = tuple(sorted(set(anchorsOf[total])))
        elif own:
            discarded[total] = own
        else:
            undecomposed[total] = _leavesUnder(total, children)
    return SumScreen(discarded=discarded, undecomposed=undecomposed,
                     anchored=anchored)


def planCrossSectionSums(claims: Mapping[int, ComponentKey],
                         present: Iterable[int], sums: Iterable[int], *,
                         mode: str = "undecomposed",
                         lumped: Optional[Mapping[int, Tuple[int, ...]]] = None,
                         remainders: Optional[Mapping[int, Tuple[Remainder, ...]]] = None
                         ) -> SumPlan:
    """Decide, per MT, which block moves it and which sums are rebuilt.

    *claims* is ``MT -> component`` for every cross section the realisation
    perturbs (MF33, or MF34's L=0 magnitude). *present* is every MF3 MT of the
    suite and *sums* the ones the decoder recognised as sums.

    A leaf with a block of its own moves by it. One without moves by the
    nearest claimed sum above it when *mode* is ``"undecomposed"`` and that sum
    is undecomposed (nothing under it is claimed), always when it is
    ``"fill"``, and never when it is ``"never"`` -- where an undecomposed sum
    raises instead, since its block would have nowhere to go.

    In every mode a claimed lump (*lumped*) moves its members as if the block
    were theirs -- the file says it is -- and a partial MF33 states as a
    remainder (*remainders*) takes the difference when its total is claimed.
    """
    checkSumBlockMode(mode)
    sums = {int(s) for s in sums}
    present = {int(mt) for mt in present}
    claims, lumpOf = _effectiveClaims(claims, present, lumped)
    virtual = _virtualClaims(claims, present)
    children, parent = sumTree(present | virtual, sums | virtual)
    active = {int(partial): tuple(r for r in statements if r.total in claims)
              for partial, statements in (remainders or {}).items()
              if int(partial) in present and int(partial) not in sums}
    active = {partial: rs for partial, rs in active.items() if rs}
    screen = screenSumClaims(claims, present, sums, remainders=active)

    if mode == "never" and screen.undecomposed:
        named = "; ".join(
            f"MT{total} over {_mtList(leaves)}"
            for total, leaves in screen.undecomposed.items())
        raise ValueError(
            f"{named}: a block is drawn for the sum and for nothing under it, "
            f"and sumBlocks='never' uses no sum's block. Applied to the sum "
            f"alone it would leave the sum unequal to its parts (and NJOY "
            f"rebuilds MT1 and MT4 from the parts anyway). Use "
            f"sumBlocks='undecomposed' to move the partials by it")

    leafControl: Dict[int, ComponentKey] = {}
    for leaf in sorted(present - sums):
        if leaf in claims:
            leafControl[leaf] = claims[leaf]
            continue
        if mode == "never" or leaf in active:
            continue
        node: Optional[int] = parent.get(leaf)
        while node is not None and node not in claims:
            node = parent.get(node)
        if node is None:
            continue
        if mode == "fill" or node in screen.undecomposed:
            leafControl[leaf] = claims[node]

    movedUnder = {}
    for total in sums:  # not the virtual ones: they have no section to rebuild
        moved = tuple(leaf for leaf in _leavesUnder(total, children)
                      if leaf in leafControl or leaf in active)
        if moved:
            movedUnder[total] = moved
    # A remainder needs the others final and moves the sums above it, so the
    # sums that do not contain one are rebuilt first, then the remainders are
    # taken (``remainderAt``), then the sums above them.
    above = set()
    for partial in active:
        node = partial
        while node in parent:
            node = parent[node]
            above.add(node)
    order = sorted(movedUnder, key=lambda s: (-_depth(s, parent), s))
    first = [s for s in order if s not in above]
    rederive = tuple(first + [s for s in order if s in above])

    ownBlockReached = {}
    for total in sorted((sums | virtual) & set(claims)):
        ownBlockReached[total] = tuple(
            leaf for leaf, component in leafControl.items()
            if component == claims[total])
    return SumPlan(leafControl=leafControl, rederive=rederive,
                   movedUnder=movedUnder, ownBlockReached=ownBlockReached,
                   virtual=tuple(sorted(virtual)), remainders=active,
                   lumpOf=lumpOf, remainderAt=len(first),
                   virtualLeaves={v: _leavesUnder(v, children) for v in virtual})


# ----------------------------------------------------------------------
# The arithmetic: S' = S + sum (p' - p)
# ----------------------------------------------------------------------

def _table(function1d) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x, y, laws)`` of a cross section, one law per interval."""
    from kika.algebra import interval_laws

    xs, ys, pairs = function1d.toEndfRegions()
    xs = np.asarray(xs, dtype=float)
    return xs, np.asarray(ys, dtype=float), interval_laws(xs.size, pairs)


def _limits(table, energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Left and right limits of a tabulated cross section at *energies*.

    A repeated abscissa is ENDF's step, so a single value per energy cannot
    describe it. Outside its table a cross section is zero, so at its first
    abscissa the left limit is zero and at its last the right one is. See
    :func:`kika.algebra.left_limit`.
    """
    from kika.algebra import left_limit, right_limit

    return left_limit(*table, energies), right_limit(*table, energies)


def rederiveSum(total, moves: Sequence[Tuple[object, object, float, float]], *,
                keepZeros: bool = False):
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
        steps at the factor block's edges). A sum of tables under different
        laws has no law of its own, so wherever the sum moved and its own law
        is not lin-lin, the result is lin-lin there, on the points that make
        every form in the sum lin-lin to
        :data:`~kika.algebra.refine.LINEARIZATION_TOLERANCE`
        (:func:`kika.algebra.to_linlin`, each added energy valued where the
        tape will put it). Between nodes it is then the sum to that tolerance,
        not only at them. *info* carries ``n_inserted``,
        ``max_rel_change`` and ``n_negative`` -- a negative total can only come
        from an evaluation whose sum sits below its own parts, and the caller
        must say so rather than write it. A negative point where a moved
        partial is itself negative (a background inside the resolved range,
        which MF3 may state below zero) or where the evaluation's own sum
        already was is not counted.

    A move may be passed reversed, ``(after, before, lo, hi)``, to *subtract*
    what a section moved -- which is how a remainder (:class:`Remainder`) is
    taken -- so the inserted points come from both forms. With *keepZeros*,
    wherever *total* is exactly zero it stays zero: a remainder statement
    covers an energy range, and the partial it names may not exist over all
    of it (B-VIII.1 O-16's MT5 = MT1 - MT2 is stated from 1e-5 eV, and MT5
    only has values above its threshold).
    """
    from kika.algebra import LINLIN, interval_laws, pairs_from_laws, to_linlin
    from kika.endf.writers.redundant import _as_written
    from kika.nuclear_data.model.enums import ENDF_INT_TO_INTERPOLATION
    from kika.nuclear_data.model.functions import Regions1d, XYs1d

    xs, ys, pairs = total.toEndfRegions()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return total, {"n_inserted": 0, "max_rel_change": 0.0, "n_negative": 0}
    lo, hi = float(xs[0]), float(xs[-1])
    laws = interval_laws(xs.size, pairs)
    # Where anything moved: the blocks' spans, merged (they are usually one).
    starts, ends = [], []
    for a, b in sorted((max(lo, s0), min(hi, s1)) for _, _, s0, s1 in moves):
        if starts and a <= ends[-1]:
            ends[-1] = max(ends[-1], b)
        else:
            starts.append(a)
            ends.append(b)
    starts, ends = np.asarray(starts), np.asarray(ends)

    def inSpans(grid):
        j = np.searchsorted(starts, grid, side="right") - 1
        return grid[(j >= 0) & (grid <= ends[np.maximum(j, 0)])]

    # Every form's own points where it moved, and the ones that make it
    # lin-lin there -- none for a lin-lin form, so a lin-lin tape gets exactly
    # the grid it always did.
    tables = [(_table(before), _table(after)) for before, after, _, _ in moves]
    curved = [t for pair in tables for t in pair if np.any(t[2] != LINLIN)]
    if np.any(laws != LINLIN):
        curved.append((xs, ys, laws))
    extra = [inSpans(t[0]) for pair in tables for t in pair]
    extra += [inSpans(to_linlin(*t, snap=_as_written)[0]) for t in curved]
    energies = np.unique(np.concatenate([xs] + extra))

    deltaLeft = np.zeros(energies.size)
    deltaRight = np.zeros(energies.size)
    # Where a partial is itself negative -- a background inside the resolved
    # range, which MF3 is allowed to state below zero -- a negative sum is not
    # evidence of anything, so it is not counted as one.
    signedPart = np.zeros(energies.size, dtype=bool)
    for (bTable, aTable), (_b, _a, spanLo, spanHi) in zip(tables, moves):
        # Off its block and off both forms' domains a move is zero: read only
        # the window in between, ends included (a limit there is not zero).
        reach = [t[0] for t in (bTable, aTable) if t[0].size]
        if not reach:
            continue
        first = np.searchsorted(energies, max(spanLo, min(r[0] for r in reach)), "left")
        last = np.searchsorted(energies, min(spanHi, max(r[-1] for r in reach)), "right")
        if last <= first:
            continue
        window = energies[first:last]
        aL, aR = _limits(aTable, window)
        bL, bR = _limits(bTable, window)
        signedPart[first:last] |= (aL < 0) | (aR < 0) | (bL < 0) | (bR < 0)
        dL, dR = aL - bL, aR - bR
        # The block steps from 1 to its first factor at spanLo and back at
        # spanHi, so the left limit at spanLo and the right one at spanHi are
        # outside it.
        dL[(window <= spanLo) | (window > spanHi)] = 0.0
        dR[(window < spanLo) | (window >= spanHi)] = 0.0
        deltaLeft[first:last] += dL
        deltaRight[first:last] += dR

    sL, sR = _limits((xs, ys, laws), energies)
    # The sum's own end points have one side only, and it is the inside one.
    sL[0], sR[-1] = sR[0], sL[-1]
    deltaLeft[0], deltaRight[-1] = deltaRight[0], deltaLeft[-1]
    if keepZeros:
        deltaLeft[sL == 0.0] = 0.0
        deltaRight[sR == 0.0] = 0.0

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
        if valueLeft == valueRight or (deltaLeft[k] == 0.0
                                       and deltaRight[k] == 0.0):
            # Unmoved, a histogram's node keeps the one point it had.
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
    # Where the sum moved under a law of its own that is not lin-lin, the
    # interval is lin-lin now: the points above make it the sum there.
    newAt = np.asarray(newAt, dtype=int)
    if np.any(laws != LINLIN) and newXs.size > 1:
        newLaws = interval_laws(newXs.size, newPairs)
        # A panel moved if it did just inside either of its ends.
        a, b = newAt[:-1], newAt[1:]
        wide = newXs[1:] > newXs[:-1]
        relaw = (wide & (newLaws != LINLIN)
                 & ((deltaRight[a] != 0.0) | (deltaLeft[b] != 0.0)))
        if relaw.any():
            newLaws[relaw] = LINLIN
            # A step's zero-width interval keeps the law before it, as the
            # tape had it, so the regions do not fragment at every step.
            index = np.where(wide, np.arange(wide.size), 0)
            np.maximum.accumulate(index, out=index)
            newLaws = newLaws[index]
            newPairs = pairs_from_laws(newLaws)
    # A log-law XYs1d that moved in part has two laws now: a Regions1d.
    if isinstance(total, Regions1d) or len(newPairs) > 1:
        out = Regions1d.fromEndfRegions(newXs, newYs, newPairs, axes=total.axes,
                                        label=total.label)
        out.outerDomainValue = total.outerDomainValue
        out.index = total.index
    else:
        out = XYs1d(xs=newXs, ys=newYs,
                    interpolation=ENDF_INT_TO_INTERPOLATION[int(newPairs[0][1])],
                    axes=total.axes, label=total.label,
                    outerDomainValue=total.outerDomainValue, index=total.index)

    old = np.asarray(total.evaluate(xs, outOfRange="zero"), dtype=float)
    moved = np.asarray(out.evaluate(xs, outOfRange="zero"), dtype=float)
    nonzero = old != 0.0
    maxRel = float(np.max(np.abs(moved[nonzero] / old[nonzero] - 1.0))) \
        if nonzero.any() else 0.0
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
