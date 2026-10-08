"""What a drawn perturbation *is*, written down once -- for every quantity.

A draw produces a flat vector of factors. On its own that vector says nothing:
which reaction each stretch of it belongs to, which *quantity* of that reaction,
what energies it is stated on, whether it multiplies or adds, and what happens
at a bin edge are all carried somewhere else -- in the shape of the code that
made it and the shape of the code that consumes it, which is how a perturbation
comes to mean two different things at its two ends.

:class:`PerturbationSet` is that meaning as data. It sits in a run directory
beside the :class:`~kika.cov.conditioning.ConditioningPlan`, and between them
they say everything that was done to a covariance and everything that was drawn
from it.

**One realisation, not one covariance.** A request may cover cross sections,
angular distributions and multiplicities at once, and
:mod:`kika.sampling.joint_blocks` decides which of them are drawn together and
which apart. What comes out of that is several blocks and therefore several
draws -- but *one* realisation of the evaluation, and it has to be applied and
written as one. So this object holds every block of one realisation, keyed by
:class:`~kika.sampling.joint_blocks.ComponentKey`, and records which components
were drawn together in :attr:`groups`, because "these two were independent" is a
claim about the evaluation that a file has to carry if anyone is to check it
later.

**Its index already exists.** :func:`kika.sampling.joint_blocks.requestIndex`
returns ``{blockKey: {components, stride, grids, widths, dimension, ...}}`` --
the ``*_index`` this repository has always passed around, only as data rather
than as a convention. What this class adds on top is the part that was never
written down anywhere: the semantics.

**Why the semantics is not a free-form string.** ``"relative"`` versus
``"absolute"`` is the difference between multiplying a cross section and adding
to it, and a file that does not say which cannot be read back safely a year
later. The values here are a closed set, checked on construction, and the edge
convention travels with them because a factor block is piecewise-constant and
therefore says nothing at all about its own discontinuities -- that rule lives
in :func:`kika.nuclear_data.model.perturbation.applyFactors` and is named here
so a reader of the file knows which applier produced it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Dict, Hashable, List, Mapping, Optional, Sequence,
                    Tuple)

import numpy as np

from kika.sampling.joint_blocks import ComponentKey

__all__ = ["PerturbationSet", "SEMANTICS", "EDGE_RULE", "FACTORS_STEM",
           "writeFactorsTable", "readFactorsTable", "readFactorsIndex"]

#: The ways a factor block can act on the quantity it perturbs. A closed set,
#: because "the file says ``relative`` and the code assumed ``absolute``" is a
#: failure that produces plausible numbers.
SEMANTICS = ("multiplicative-relative", "additive-absolute")

#: Which of :data:`SEMANTICS` a block of each file acts under, read off the
#: file rather than chosen. MF31, MF33 and MF34 state relative covariances --
#: MF33's absolute sections are converted before they reach a draw, and MF34's
#: are dropped -- so their blocks are factors and the applier multiplies. **MF35
#: is the exception and it is not a variant of the same thing**: LB=7 is the
#: covariance of group-integrated *probabilities*, which are dimensionless and
#: sum to one, so a draw of it is a vector of absolute deltas on quantities the
#: node does not even store. Multiplying a spectrum by one of those would be
#: arithmetic on the wrong object, and it would produce a plausible file.
SEMANTICS_OF_MF = {31: SEMANTICS[0], 33: SEMANTICS[0], 34: SEMANTICS[0],
                   35: SEMANTICS[1]}

#: Which discontinuity convention the factors were drawn to be applied under.
#: Named rather than implied: a piecewise-constant block is silent about its own
#: steps, so the rule lives in the applier and the set records which one.
EDGE_RULE = "endf-step-duplicate"

#: The stem of the run-level factors table, ``<stem>.parquet``: every drawn
#: value of every sample, with the index that says what they are in the
#: file's own metadata. See :func:`writeFactorsTable`.
FACTORS_STEM = "factors"

#: Bumped to 3 when a block learned to state its own semantics and its own
#: outer domain, which is what a fission spectrum needs and the other three
#: quantities do not. A version-2 file is still read: every block in one is
#: ``multiplicative-relative`` with no outer domain, which is exactly what the
#: defaults give, so nothing about it has to be guessed.
_FORMAT_VERSION = 3
_READABLE_VERSIONS = (2, 3)


@dataclass(frozen=True)
class PerturbationSet:
    """One drawn realisation of an evaluation, over one or more quantities.

    Parameters
    ----------
    label
        The §9.3 style label this realisation will be written under --
        ``'realization-0007'``. It is the name the perturbed forms take inside a
        :class:`~kika.nuclear_data.model.suite.ReactionSuite`, so it belongs to
        the perturbation and not to whoever applies it.
    factors
        ``ComponentKey -> one factor per bin``.
    binEdges
        ``ComponentKey -> the bin boundaries those factors are stated on``, one
        longer than the factors.
    groups
        The components that were drawn together, one tuple per block. Two
        components in different tuples were drawn independently, which is a
        statement about what the evaluation correlates and is worth keeping.
    """

    label: str
    factors: Dict[ComponentKey, np.ndarray]
    binEdges: Dict[ComponentKey, np.ndarray]
    groups: Tuple[Tuple[ComponentKey, ...], ...] = ()
    semantics: str = SEMANTICS[0]
    #: Where one component acts under something other than :attr:`semantics`.
    #: A realisation may cover several quantities at once and they need not
    #: agree -- a request for cross sections and a fission spectrum draws
    #: factors for one and absolute deltas for the other -- so the set-wide
    #: field is the default and this is what a block says about itself.
    componentSemantics: Dict[ComponentKey, str] = field(default_factory=dict)
    #: The coordinate a block is stated *over* but not *on*: MF35's
    #: incident-energy band. Empty for every quantity whose factors already
    #: live on the axis they apply to.
    outerDomains: Dict[ComponentKey, Tuple[float, float]] = field(
        default_factory=dict)
    #: The frame an MF34 block's coefficients are stated in, ``"LAB"`` or
    #: ``"CM"``, where it is not the distribution's own (MF34 LCT 1 or 2 against
    #: a different MF4 LCT). Absent means "the same as the distribution". The
    #: factors act on that frame's coefficients, and the applier converts.
    componentFrames: Dict[ComponentKey, str] = field(default_factory=dict)
    edgeRule: str = EDGE_RULE
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value in (self.semantics, *self.componentSemantics.values()):
            if value not in SEMANTICS:
                raise ValueError(
                    f"semantics {value!r} is not one of {list(SEMANTICS)}. A "
                    f"perturbation that does not say how it acts cannot be "
                    f"applied"
                )
        stranded = [component for component in self.factors
                    if component.mf == 35 and component not in self.outerDomains]
        if stranded:
            raise ValueError(
                f"{[c.describe() for c in stranded]} carry factors on an "
                f"outgoing-energy grid and no incident-energy band. An MF35 "
                f"block says how the spectrum moves; without its band nothing "
                f"says at which incident energies it moves that way, and an "
                f"applier would have to guess"
            )
        if set(self.factors) != set(self.binEdges):
            missing = set(self.factors) ^ set(self.binEdges)
            raise ValueError(
                f"{sorted(missing)} has factors without a grid or a grid "
                f"without factors; a block and its bins are one object"
            )
        for component, values in self.factors.items():
            edges = self.binEdges[component]
            if len(values) != len(edges) - 1:
                raise ValueError(
                    f"{component.describe()}: {len(values)} factor(s) on "
                    f"{len(edges) - 1} bin(s)"
                )
        for component, frame in self.componentFrames.items():
            if component.mf != 34 or frame not in ("LAB", "CM"):
                raise ValueError(
                    f"{component.describe()}: frame {frame!r}. A frame is "
                    f"stated for MF34 components only, as 'LAB' or 'CM'")
        grouped = {component for group in self.groups for component in group}
        if self.groups and grouped != set(self.factors):
            raise ValueError(
                f"the groups cover {len(grouped)} component(s) and the factors "
                f"{len(self.factors)}; a realisation cannot say two different "
                f"things about what it perturbs"
            )

    # ------------------------------------------------------------------
    # From a draw
    # ------------------------------------------------------------------

    @classmethod
    def fromDraw(cls, drawn, index: Mapping[Hashable, Mapping[str, Any]], *,
                 label: str, provenance: Optional[Dict[str, Any]] = None
                 ) -> "PerturbationSet":
        """Cut one realisation out of what a draw returned.

        *index* is what :func:`~kika.sampling.joint_blocks.requestIndex`
        returned. *drawn* is either ``{blockKey: one row of factors}`` -- sample
        *i* of what :func:`~kika.sampling.core.draw_samples` produced, which is
        the normal case -- or, when the index has exactly one block, that row on
        its own.

        ``widths`` is read rather than assumed. Under the ``global`` union every
        component is ``stride`` wide and the two are the same number; under
        ``per-component`` they are not, and the tail of each component's stride
        is the zero padding a uniform stride implies. Slicing by ``stride``
        there would hand a component its neighbour's padding as if it were its
        own factors -- and a factor of zero is not a small perturbation, it is a
        deleted cross section.
        """
        if not isinstance(drawn, Mapping):
            if len(index) != 1:
                raise ValueError(
                    f"a bare factor vector needs an index of one block, got "
                    f"{len(index)}. With several blocks the rows have to say "
                    f"which block they came from"
                )
            (onlyKey,) = index
            drawn = {onlyKey: drawn}

        missing = set(index) - set(drawn)
        if missing:
            raise ValueError(
                f"{len(missing)} block(s) of the index were not drawn: a "
                f"realisation that silently omits a block is perturbed in fewer "
                f"places than its metadata claims"
            )

        factors: Dict[ComponentKey, np.ndarray] = {}
        binEdges: Dict[ComponentKey, np.ndarray] = {}
        outerDomains: Dict[ComponentKey, Tuple[float, float]] = {}
        componentFrames: Dict[ComponentKey, str] = {}
        componentSemantics: Dict[ComponentKey, str] = {}
        groups: List[Tuple[ComponentKey, ...]] = []
        for blockKey, meta in index.items():
            components = list(meta.get("components") or meta.get("pairs")
                              or meta.get("triplets"))
            stride = int(meta["stride"])
            widths, grids = meta["widths"], meta["grids"]
            values = np.asarray(drawn[blockKey], dtype=float).ravel()
            expected = len(components) * stride
            if values.size != expected:
                raise ValueError(
                    f"block {blockKey}: {values.size} factor(s) for "
                    f"{len(components)} component(s) x stride {stride} = "
                    f"{expected}"
                )
            group: List[ComponentKey] = []
            for position, component in enumerate(components):
                component = cls._asComponentKey(component, blockKey)
                lookup = components[position]
                width = int(widths[lookup] if isinstance(widths, Mapping)
                            else widths[position])
                start = position * stride
                factors[component] = values[start:start + width].copy()
                grid = grids[lookup] if isinstance(grids, Mapping) else grids[position]
                binEdges[component] = np.asarray(grid, dtype=float)
                # The semantics is read off the MF, which is to say off the
                # file: `SEMANTICS_OF_MF` records what each covariance file
                # states its numbers to be. Recorded per block rather than
                # inferred again at apply time, so the JSON beside the sample
                # says it and a reader can check it.
                semantics = SEMANTICS_OF_MF.get(component.mf)
                if semantics is not None and semantics != cls.semantics:
                    componentSemantics[component] = semantics
                domain = (meta.get("domains") or {}).get(lookup)
                if domain is not None:
                    outerDomains[component] = (float(domain[0]), float(domain[1]))
                frame = (meta.get("frames") or {}).get(lookup)
                if frame is not None:
                    componentFrames[component] = frame
                group.append(component)
            groups.append(tuple(group))

        return cls(label=label, factors=factors, binEdges=binEdges,
                   groups=tuple(groups), componentSemantics=componentSemantics,
                   outerDomains=outerDomains, componentFrames=componentFrames,
                   provenance=dict(provenance or {}))

    @staticmethod
    def _asComponentKey(component, blockKey) -> ComponentKey:
        """A component of any of the three index shapes, as one key.

        ``requestIndex`` already gives :class:`ComponentKey`. The two
        single-quantity indices predate it and give ``(ZA, MT)`` for MF33/MF31
        and ``(ZA, MT, L)`` for MF34, so the MF has to come from the block key,
        which is where those two put it.
        """
        if isinstance(component, ComponentKey):
            return component
        label = blockKey[1] if isinstance(blockKey, tuple) and len(blockKey) > 1 else ""
        mf = int(str(label).replace("MF", "")) if str(label).startswith("MF") else 0
        if len(component) == 3:
            return ComponentKey(int(component[0]), 34, int(component[1]),
                                int(component[2]))
        if not mf:
            raise ValueError(
                f"block key {blockKey!r} does not name an MF, so the component "
                f"{component!r} cannot be placed. Use requestIndex, whose "
                f"components carry it"
            )
        return ComponentKey(int(component[0]), mf, int(component[1]), 0)

    # ------------------------------------------------------------------
    # What is in it
    # ------------------------------------------------------------------

    def components(self) -> Tuple[ComponentKey, ...]:
        """Every component this set perturbs, sorted."""
        return tuple(sorted(self.factors))

    def reactions(self) -> Tuple[int, ...]:
        """The MTs this set perturbs, ascending, whatever the quantity."""
        return tuple(sorted({component.mt for component in self.factors}))

    def semanticsOf(self, component: ComponentKey) -> str:
        """How this component's block acts -- its own statement, or the set's."""
        return self.componentSemantics.get(component, self.semantics)

    def quantities(self) -> Tuple[str, ...]:
        """The model nodes this set touches, e.g. ``('crossSection',)``."""
        return tuple(sorted({component.quantity for component in self.factors}))

    def block(self, mt: int, *, mf: int = 33, order: int = 0, za: Optional[int] = None):
        """``(factors, binEdges)`` for one component, addressed the readable way.

        A convenience for callers and tests that know which reaction they mean
        but not which ZA the tape stated it under.
        """
        matches = [component for component in self.factors
                   if component.mt == mt and component.mf == mf
                   and component.index == order
                   and (za is None or component.za == za)]
        if not matches:
            raise KeyError(
                f"this PerturbationSet perturbs "
                f"{[c.describe() for c in self.components()]}, not "
                f"MF{mf}/MT{mt} index {order}"
            )
        if len(matches) > 1:
            raise KeyError(
                f"{len(matches)} components match MF{mf}/MT{mt} index {order} "
                f"(different ZA); name the ZA"
            )
        return self.factors[matches[0]], self.binEdges[matches[0]]

    def describe(self) -> str:
        """One line per group, saying what was drawn with what."""
        lines = [f"{self.label}: {len(self.factors)} component(s) in "
                 f"{len(self.groups) or 1} draw(s), {', '.join(self.quantities())}"]
        for number, group in enumerate(self.groups):
            lines.append(f"  [{number}] " + ", ".join(c.describe() for c in group))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------------

    def apply(self, function1d, component: ComponentKey):
        """Perturb *function1d* with this set's block for *component*.

        Returns ``(perturbed, diagnostics)``, the pair
        :func:`~kika.nuclear_data.model.perturbation.applyFactors` returns.
        Raises :class:`KeyError` for a component this set does not carry, rather
        than returning the function unchanged: "no perturbation for this one"
        and "a perturbation of one" are different answers, and silently
        conflating them is how an ensemble comes to be narrower than it claims.
        """
        from kika.nuclear_data.model.perturbation import applyFactors

        if component not in self.factors:
            raise KeyError(
                f"this PerturbationSet perturbs "
                f"{[c.describe() for c in self.components()]}, not "
                f"{component.describe()}"
            )
        return applyFactors(function1d, self.factors[component],
                            self.binEdges[component])

    def _crossSectionBlocks(self) -> Dict[Tuple[int, int], ComponentKey]:
        """Which component perturbs each reaction's ``sigma(E)``, and the refusal.

        Two components can claim the same cross section: MF33's own block, and
        MF34's Legendre order 0, which is the *magnitude* -- in MF4 ``a_0`` is
        identically 1 and the size lives in MF3, so an L=0 covariance is a
        statement about ``sigma(E)`` on MF34's grid. Applying both would
        multiply the cross section twice and neither factor would be what the
        file said.

        Refused rather than reconciled. Which of the two a run wants is a
        modelling decision -- they are two estimates of the same uncertainty on
        two grids -- and it belongs in the request, not in an applier picking
        one silently.
        """
        claims: Dict[Tuple[int, int], List[ComponentKey]] = {}
        for component in self.components():
            isMagnitude = component.mf == 34 and component.index == 0
            if component.mf == 33 or isMagnitude:
                claims.setdefault((component.za, component.mt), []).append(component)

        resolved = {}
        for reaction, components in claims.items():
            if len(components) > 1:
                raise ValueError(
                    f"ZA {reaction[0]} MT{reaction[1]}: "
                    f"{[c.describe() for c in components]} all perturb the same "
                    f"cross section -- MF33 states its uncertainty and MF34's "
                    f"L=0 states the magnitude's, and applying both multiplies "
                    f"sigma twice. Ask for one of them"
                )
            resolved[reaction] = components[0]
        return resolved

    def _applyCrossSectionsWithSums(self, suite, claims, mode="undecomposed",
                                    lumped=None, remainders=None):
        """Perturb the cross sections with MF3's sum rules held.

        :func:`~kika.sampling.cross_section_sums.planCrossSectionSums` decides
        which block moves each leaf partial -- its own, a lump's it belongs to,
        or the nearest perturbed sum above it where *mode* allows -- and which
        sums are rebuilt; this applies the blocks to the leaves, takes the
        remainders (*remainders*: a partial MF33 states as a total minus the
        others) and re-derives the sums as ``S + sum (p' - p)``. See the module
        docstring of :mod:`~kika.sampling.cross_section_sums`.

        Diagnostics, keyed so the emitter writes everything that moved and the
        run can say why:

        * a leaf with its own block -- under its component, as before;
        * a leaf moved by another block (a sum's it rides, or a lump's it
          belongs to, ``lumped: True``) -- under ``ComponentKey(za, 33, leaf)``
          with ``factor_from`` naming the block. It was not drawn, and it was
          moved, so it has to be written;
        * a remainder -- under its own component if it has one, else
          ``ComponentKey(za, 33, leaf)``, with ``remainder_of`` naming the
          total;
        * a rebuilt sum -- under its own component if the request named it,
          else ``ComponentKey(za, 33, sum)``, with ``rederived_from`` and, for
          a named one, ``own_block``: the leaves its block reached,
          ``"discarded"``, or ``"remainder"`` when a remainder carries it.
        """
        from kika.nuclear_data.model import EVAL_LABEL
        from kika.sampling.cross_section_sums import (planCrossSectionSums,
                                                      rederiveSum,
                                                      suiteSumLayout)

        for component in claims.values():
            if self.semanticsOf(component) != SEMANTICS[0]:
                raise ValueError(
                    f"{component.describe()} acts as {self.semanticsOf(component)}"
                    f"; MF3's sum rules carry a factor from a sum to its parts, "
                    f"which only means something for a multiplicative block")

        byMT = {mt: component for (_za, mt), component in claims.items()}
        za = next(iter(claims))[0]
        present, sums = suiteSumLayout(suite)
        plan = planCrossSectionSums(byMT, present, sums, mode=mode,
                                    lumped=lumped, remainders=remainders)

        def span(component):
            edges = np.asarray(self.binEdges[component], dtype=float)
            return float(edges[0]), float(edges[-1])

        diagnostics: Dict[ComponentKey, Dict[str, Any]] = {}
        moved = {}
        for leaf, component in plan.leafControl.items():
            reaction = suite.reactionByENDF_MT(leaf)
            before = reaction.crossSection[EVAL_LABEL]
            after, info = self.apply(before, component)
            reaction.crossSection[self.label] = self._labelled(after)
            moved[leaf] = (before, after, *span(component))
            if component.mt == leaf:
                diagnostics[component] = info
            else:
                extra = {"factor_from": component.mt}
                if leaf in plan.lumpOf:
                    extra["lumped"] = True
                diagnostics[ComponentKey(za, 33, leaf)] = {**info, **extra}

        anchors = {r.total: [] for rs in plan.remainders.values() for r in rs}
        for partial, statements in plan.remainders.items():
            for statement in statements:
                anchors[statement.total].append(partial)

        def takeRemainders():
            for partial, statements in plan.remainders.items():
                reaction = suite.reactionByENDF_MT(partial)
                before = reaction.crossSection[EVAL_LABEL]
                moves = [moved[partial]] if partial in moved else []
                for statement in statements:
                    totalForm = suite.reactionByENDF_MT(statement.total
                                                        ).crossSection[EVAL_LABEL]
                    component = byMT[statement.total]
                    totalAfter, _ = self.apply(totalForm, component)
                    lo, hi = span(component)
                    lo, hi = max(lo, statement.lo), min(hi, statement.hi)
                    if lo < hi:
                        moves.append((totalForm, totalAfter, lo, hi))
                    for other in statement.minus:
                        # A virtual sum has no section of its own to have
                        # moved; its partials did.
                        for piece in (plan.virtualLeaves.get(other) or (other,)):
                            if piece not in moved:
                                continue
                            oBefore, oAfter, oLo, oHi = moved[piece]
                            lo, hi = max(oLo, statement.lo), min(oHi, statement.hi)
                            if lo < hi:
                                # Reversed: subtract what the other moved.
                                moves.append((oAfter, oBefore, lo, hi))
                if not moves:
                    continue
                after, info = rederiveSum(before, moves, keepZeros=True)
                reaction.crossSection[self.label] = self._labelled(after)
                moved[partial] = (before, after, min(m[2] for m in moves),
                                  max(m[3] for m in moves))
                key = (plan.leafControl[partial]
                       if partial in plan.leafControl
                       and plan.leafControl[partial].mt == partial
                       else ComponentKey(za, 33, partial))
                diagnostics[key] = {**diagnostics.get(key, {}), **info,
                                    "remainder_of": statements[0].total}

        for index, total in enumerate(plan.rederive):
            if index == plan.remainderAt:
                takeRemainders()
            reaction = suite.reactionByENDF_MT(total)
            before = reaction.crossSection[EVAL_LABEL]
            moves = [moved[leaf] for leaf in plan.movedUnder[total] if leaf in moved]
            rebuilt, info = rederiveSum(before, moves)
            reaction.crossSection[self.label] = self._labelled(rebuilt)
            # A sum is itself a section others may be subtracted from: a
            # remainder's "minus" list names MT103-107 beside the partials.
            moved[total] = (before, rebuilt, min(m[2] for m in moves),
                            max(m[3] for m in moves))
            info = {**info, "rederived_from": plan.movedUnder[total]}
            if total in byMT:
                if total in anchors:
                    info["own_block"] = "remainder"
                    info["anchors"] = tuple(sorted(anchors[total]))
                else:
                    info["own_block"] = (plan.ownBlockReached.get(total)
                                         or "discarded")
                diagnostics[byMT[total]] = info
            else:
                diagnostics[ComponentKey(za, 33, total)] = info
        if plan.remainderAt >= len(plan.rederive):
            takeRemainders()
        for total in plan.virtual:
            diagnostics[byMT[total]] = {
                "virtual": True,
                "own_block": plan.ownBlockReached.get(total) or "discarded"}
        return diagnostics

    def applyToSuite(self, suite, *, multiplicityResolver=None,
                     maxOutgoingPoints: Optional[int] = None,
                     crossSectionSums: bool = True,
                     sumBlocks: str = "undecomposed",
                     lumped=None, remainders=None
                     ) -> Dict[ComponentKey, Dict[str, Any]]:
        """Put a perturbed form under :attr:`label` on every node this set covers.

        The evaluated form is left where it is and the realisation goes beside
        it, which is what §9.1's multi-form container and §9.3's ``realization``
        style are for -- and what ``encodeMF3MT(..., label=)`` then writes out.
        Nodes this set does not cover are not touched and not reported as
        perturbed.

        Three dispatches, by what the component *is* rather than by which file
        it came from:

        * ``crossSection`` -- MF33, and MF34's L=0 magnitude, which lands on the
          same node. See :meth:`_crossSectionBlocks` for why both at once is
          refused. With *crossSectionSums* (the default) MF3's sum rules hold
          on the realisation: partials govern, every sum with a moved partial
          is re-derived, and a sum's own block moves its partials only where
          the file does not decompose it (*sumBlocks*, ``"undecomposed"`` by
          default; ``"fill"`` and ``"never"`` as in
          :mod:`~kika.sampling.cross_section_sums`) -- see
          :meth:`_applyCrossSectionsWithSums`. *lumped* and *remainders* are
          what :func:`~kika.sampling.cross_section_sums.readSumStatements`
          reads off the tape's MF33: lumped reactions move their members, and
          a partial stated as a total minus the rest takes the difference.
          ``crossSectionSums=False`` scales each named MT by its own block and
          nothing else, which leaves the sums stale; it exists for the
          equivalence gate against ``perturb_PENDF_files``, which does not
          re-derive.
        * ``angularDistribution`` -- MF34's L>=1, all orders of one reaction in
          one call, because a Legendre vector is perturbed once and not once per
          order.
        * ``energyDistribution`` -- MF35, every band of one reaction in one
          call, because the bands are drawn apart and written together. See
          :meth:`_applyEnergyDistribution`; *maxOutgoingPoints* caps how large
          one perturbed table may grow and is reported when it bites.
        * ``multiplicity`` -- MF31, beside the evaluation like the other two
          since ``Multiplicity`` became a
          :class:`~kika.nuclear_data.model.component.Component`. It needs
          *multiplicityResolver*, and without one it raises rather than skipping:
          a nu-bar silently left unperturbed inside a realisation that claims to
          carry it is exactly the failure this class exists to prevent.

        Parameters
        ----------
        multiplicityResolver
            ``callable(suite, mt) -> Multiplicity``, for MF31 only. The model
            does not know which node an ENDF MT names -- that is the adapter's
            question, and ``kika.endf.model_adapter.multiplicity.nubarNode`` is
            its answer, the same one the MF1 encoders use. Passing it in rather
            than importing it is what keeps the sampling layer off the adapter.
        """
        from kika.nuclear_data.model import EVAL_LABEL

        diagnostics: Dict[ComponentKey, Dict[str, Any]] = {}

        multiplicities = [c for c in self.components() if c.mf == 31]
        if multiplicities:
            diagnostics.update(self._applyMultiplicity(
                suite, multiplicities, multiplicityResolver))

        # `reactionByENDF_MT` raises for an MT the suite does not have, and
        # that is what should happen: a request named the reaction, so its
        # absence is a mistake in the request or a tape that does not carry it,
        # not something to skip past. It searches `sums` too, which is where the
        # ENDF adapter now puts MT1 and MT4.
        claims = self._crossSectionBlocks()
        if crossSectionSums and claims:
            diagnostics.update(self._applyCrossSectionsWithSums(
                suite, claims, mode=sumBlocks, lumped=lumped,
                remainders=remainders))
        else:
            for (_za, mt), component in claims.items():
                reaction = suite.reactionByENDF_MT(mt)
                perturbed, info = self.apply(reaction.crossSection[EVAL_LABEL],
                                             component)
                reaction.crossSection[self.label] = self._labelled(perturbed)
                diagnostics[component] = info

        byReaction: Dict[Tuple[int, int], Dict[int, ComponentKey]] = {}
        for component in self.components():
            if component.mf == 34 and component.index != 0:
                byReaction.setdefault((component.za, component.mt), {})[
                    component.index] = component
        for (_za, mt), orders in byReaction.items():
            reaction = suite.reactionByENDF_MT(mt)
            product, angular = self._angularOf(reaction, mt)
            conversion = self._frameConversion(suite, reaction, product, mt,
                                               orders)
            perturbed, info = self._applyAngular(angular, orders, conversion)
            self._putRealisation(product, perturbed)
            for order, component in orders.items():
                diagnostics[component] = {
                    "n_inserted": info["n_inserted"],
                    **info["per_order"].get(order, {}),
                    **({"frame": info["frame"]} if "frame" in info else {}),
                }

        spectra = [c for c in self.components() if c.mf == 35]
        if spectra:
            diagnostics.update(self._applyEnergyDistribution(
                suite, spectra, maxOutgoingPoints=maxOutgoingPoints))
        return diagnostics

    def _applyEnergyDistribution(self, suite, components,
                                 maxOutgoingPoints=None):
        """Put a perturbed fission spectrum beside the evaluated one.

        **The third dispatch, and the one that is not a scaling.** MF35 states
        the covariance of the *group-integrated probabilities* of chi(E'|E), so
        a block is a vector of absolute deltas on quantities the node does not
        store, over one band of incident energy. Applying it means integrating
        the node over the band's groups, turning each delta into a ratio,
        refining the table where the ratio steps, scaling, and putting the
        integral back where it was --
        :func:`~kika.nuclear_data.model.perturbation.applySpectrumFactors`.

        **Every band of one reaction goes in one call.** The bands are drawn
        independently -- the file states no block between two of them, which is
        why :func:`~kika.sampling.joint_blocks.samplingGroups` never merges them
        -- but they act on **one** node, and an applier run once per band would
        integrate a table a previous band had already rewritten. The realisation
        is one function of two variables and it is written once.

        What a *sample* may be -- positivity, the projection onto
        ``{r >= 0, sum P0 r = sum P0}``, freezing a group with no probability to
        scale -- arrives as
        :func:`~kika.sampling.mf35_sampling.pfns_ratio_rule`. It stays in the
        sampling layer on purpose (roadmap decision 4): it is a statement about
        the draw, not about the function.
        """
        from kika.nuclear_data.model.perturbation import (applySpectrumFactors,
                                                          summariseSpectrumNodes)
        from kika.sampling.mf35_sampling import pfns_ratio_rule

        diagnostics: Dict[ComponentKey, Dict[str, Any]] = {}
        byReaction: Dict[Tuple[int, int], List[ComponentKey]] = {}
        for component in components:
            byReaction.setdefault((component.za, component.mt), []).append(
                component)

        for (_za, mt), bandComponents in byReaction.items():
            reaction = suite.reactionByENDF_MT(mt)
            product, energyForm = self._energyOf(reaction, mt)
            bands = {c.index: self.outerDomains[c] for c in bandComponents}
            boundaries = {c.index: self.binEdges[c] for c in bandComponents}
            deltas = {c.index: self.factors[c] for c in bandComponents}

            perturbed, info = applySpectrumFactors(
                energyForm, bands, boundaries, pfns_ratio_rule(deltas),
                maxOutgoingPoints=maxOutgoingPoints)
            self._putEnergyRealisation(product, perturbed)

            # The evaluation's own departure from int chi = 1, measured on the
            # node the realisation started from. It is how a budget below is
            # read: a renormalisation of 1e-7 means nothing against an input
            # that was already off by 3e-7 (JEFF-4.0 Pu-241).
            inputResidual = _inputNormalisationResidual(energyForm, bands)

            for component in bandComponents:
                # Per band, because that is what was drawn: a band's own nodes
                # carry its renormalisation and its group self-check, and the
                # summary over every band would hide one bad band behind three
                # good ones.
                nodes = [entry for entry in info["per_node"]
                         if entry["band"] == component.index]
                diagnostics[component] = {
                    "n_outer_inserted": info["n_outer_inserted"],
                    **summariseSpectrumNodes(nodes),
                    "input_normalisation_max_abs":
                        inputResidual.get(component.index, 0.0),
                }
        return diagnostics

    @staticmethod
    def _energyOf(reaction, mt: int):
        """The product whose evaluated distribution carries chi(E'|E).

        The energy twin of :meth:`_angularOf`, and it refuses in the same two
        places. **§18.3's ``uncorrelated`` holds both halves of one
        distribution**, so MF4's angular and MF5's energy hang on the same node
        and this differs from that method only in which half it asks for.

        A product whose ``energy`` is a ``discreteGamma``, a ``primaryGamma`` or
        anything else that is not a table of chi against outgoing energy is not
        a candidate: MF35 is the covariance of an integral over E', and a
        discrete line has no such integral.
        """
        from kika.nuclear_data.model import EVAL_LABEL
        from kika.nuclear_data.model import ANALYTIC_SPECTRA
        from kika.nuclear_data.model.functions.higher import Function2d

        channel = getattr(reaction, "outputChannel", None)
        candidates, refused = [], []
        for product in (getattr(channel, "products", None) or ()):
            distribution = getattr(product, "distribution", None)
            if distribution is None:
                continue
            form = (distribution.get(EVAL_LABEL)
                    if hasattr(distribution, "get") else distribution)
            energy = getattr(form, "energy", None)
            if isinstance(energy, ANALYTIC_SPECTRA):
                refused.append(type(energy).__name__)
            elif isinstance(energy, Function2d):
                candidates.append((product, energy))
        if not candidates and refused:
            # PD-3 (decided 2026-10-08): refused by name.
            # A parametrised spectrum is a formula, MF35 is a covariance of the group
            # integrals of a *table*, and perturbing the formula would need a
            # covariance of its parameters, which ENDF does not carry.
            # Tabulating it and applying an external covariance is roadmap P5.
            raise ValueError(
                f"MT{mt}: the energy distribution is a parametrised spectrum "
                f"({', '.join(refused)}), and an MF35 perturbation is refused "
                f"for it (decision PD-3): MF35 covers group integrals of a "
                f"tabulated chi(E'|E), and a formula has no such table. "
                f"Tabulating it with toPointwise() and supplying a covariance "
                f"is the external-covariance route (P5), not built yet"
            )
        if not candidates:
            raise ValueError(
                f"MT{mt} has no product carrying an evaluated energy "
                f"distribution as a table of chi(E'|E), so an MF35 "
                f"perturbation has nothing to act on. A law the reader keeps as "
                f"bytes reaches the model as provenance and not as a node -- "
                f"the decoder's report says so"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"MT{mt} has {len(candidates)} products carrying an energy "
                f"distribution and MF35 states one covariance; which one it is "
                f"about is not in the file, so it is not for this to guess"
            )
        return candidates[0]

    def _putEnergyRealisation(self, product, energyForm) -> None:
        """The perturbed spectrum, beside the evaluated one, under the label.

        The same move as :meth:`_putRealisation` makes for the angular half, and
        deliberately built from the **evaluated** distribution rather than from
        whatever this realisation may already have put there: a request that
        perturbs MF34 and MF35 of one reaction writes two halves of one node,
        and each has to start from the evaluation. Where a labelled form already
        exists -- the angular half was written first -- it is that form that
        gets the new energy, so neither half is dropped.
        """
        import dataclasses

        from kika.nuclear_data.model import EVAL_LABEL

        distribution = product.distribution
        form = distribution.get(self.label) or distribution[EVAL_LABEL]
        distribution[self.label] = self._labelled(
            dataclasses.replace(form, energy=energyForm))

    def _applyMultiplicity(self, suite, components, resolver):
        """Put a nu-bar realisation on the node an MT names, and say what it cost.

        **The three MTs are three different nodes, measured rather than assumed**
        (``micro_u235_nubar.endf``, 2026-09-06): MT456, the prompt nu-bar, is the
        fission product's own ``multiplicity`` -- a ``Regions1d`` of 95 points --
        while MT452 and MT455 are ``multiplicitySum`` nodes under ``suite.sums``,
        "total fission neutron multiplicity" and "delayed fission neutron
        multiplicity", because §18.4 puts delayed neutrons on precursor families
        and the aggregate is a sum over them. All three carry a real
        ``Regions1d``, so the *arithmetic* is :func:`applyFactors` unchanged.

        **And where the realisation goes is no longer special.** It is
        ``multiplicity[label] = form``, beside the evaluated one, exactly as for
        a cross section or a distribution: ``Multiplicity`` became a
        :class:`~kika.nuclear_data.model.component.Component` on 2026-09-06 for
        this. What it replaced was a realisation that *displaced* the evaluated
        form and had to be put back afterwards -- which worked, and produced a
        GNDS document carrying the perturbed nu-bar instead of the evaluation it
        was drawn from, while the same document's cross sections carried both.

        **The sum rule is enforced, not checked.** ENDF-6 requires
        nu_452 = nu_455 + nu_456, so the family is not perturbed member by
        member: it goes through
        :func:`~kika.sampling.mf31_sampling.perturbNubarFamilyOnModel`, which is
        the model-side twin of the rule this project already runs -- perturb the
        components, **derive** the redundant member from them, and discard the
        derived member's own factor block. A member with no block of its own
        rides the total's when there is one, so "perturb everything with the
        total" still holds the rule; a family that is not derivable, because the
        tape does not carry every contributor, is perturbed member by member, as
        it is there.

        A consequence worth stating: rebuilding the derived member also repairs
        whatever residual the input evaluation carried, which moves the central
        value by something the perturbation did not ask for.
        :func:`~kika.sampling.mf31_sampling.sum_rule_residual` is what puts a
        number on that.
        """
        if resolver is None:
            raise ValueError(
                f"{[c.describe() for c in components]}: which model node an ENDF "
                f"MT names is the adapter's question, not the model's, so a "
                f"multiplicity needs a resolver. Pass "
                f"kika.endf.model_adapter.multiplicity.nubarNode -- the same "
                f"lookup the MF1 encoders use, so what is perturbed is what gets "
                f"written"
            )

        from kika.sampling.mf31_sampling import perturbNubarFamilyOnModel

        nodes = {}
        for mt in (452, 455, 456):
            try:
                node = resolver(suite, mt)
            except (KeyError, ValueError):
                node = None
            if node is not None and getattr(node, "form", None) is not None:
                nodes[mt] = node
        for component in components:
            if component.mt not in nodes:
                raise ValueError(
                    f"{component.describe()}: this suite carries no such nu-bar, "
                    f"so there is nothing to perturb"
                )

        grid = self._familyGrid(components, self.binEdges)
        blocks = {component.mt: self.factors[component] for component in components}
        forms = {mt: node.form for mt, node in nodes.items()}
        perturbed, info = perturbNubarFamilyOnModel(forms, blocks, grid)

        byMT = {component.mt: component for component in components}
        za = components[0].za
        diagnostics = {}
        for mt, form in perturbed.items():
            if form is forms[mt]:
                continue                      # untouched, and not reported as
            # The derived member has no component in the request -- its own
            # factor block is discarded by the rule -- and it is still part of
            # this realisation: it was rewritten, it has to be emitted, and its
            # diagnostics carry the residual the rebuild repaired. So it gets a
            # key of its own rather than being dropped for not having been asked
            # for.
            component = byMT.get(mt) or ComponentKey(za, 31, mt)
            nodes[mt][self.label] = self._labelled(form)
            diagnostics[component] = info.get(mt, {})
        return diagnostics


    @staticmethod
    def _familyGrid(components, binEdges):
        """The one grid the family is perturbed on, or a refusal.

        :func:`~kika.sampling.mf31_sampling.perturbNubarFamilyOnModel` takes a
        single ``bins`` array for the whole family, and it has to: the derived
        member is rebuilt from the others, so "which bin is this factor for" must
        mean the same thing for all of them. MF31 assembles under the ``global``
        union, where every component is stated on the pooled grid, so this holds
        by construction -- and it is checked rather than assumed, because a
        per-component assembly would break it silently and the result would be a
        total derived from parts perturbed on grids that do not line up.
        """
        grids = [np.asarray(binEdges[component], dtype=float)
                 for component in components]
        for grid in grids[1:]:
            if grid.shape != grids[0].shape or not np.array_equal(grid, grids[0]):
                raise ValueError(
                    "the nu-bar family is perturbed on one grid and these blocks "
                    "are stated on different ones; MF31 assembles under the "
                    "'global' union for exactly this reason"
                )
        return grids[0]


    def _applyAngular(self, angular, orders: Mapping[int, ComponentKey],
                      frameConversion=None):
        from kika.nuclear_data.model.perturbation import applyLegendreFactors

        return applyLegendreFactors(
            angular,
            {order: self.factors[component] for order, component in orders.items()},
            {order: self.binEdges[component] for order, component in orders.items()},
            frameConversion=frameConversion,
        )

    def _frameConversion(self, suite, reaction, product, mt: int,
                         orders: Mapping[int, ComponentKey]):
        """The CM/LAB change these blocks need, or ``None`` when there is none.

        MF34 may state the coefficients in another frame than MF4's (ENDF-6
        §34.1). Then the factors act on that frame's coefficients, and
        :mod:`kika._legendre_frames` brings the change back. The kinematics are
        the evaluation's: AWR off the reaction's provenance (or the suite's),
        and the reaction's Q, which is QI for an ENDF-built suite.
        """
        from kika._legendre_frames import FrameConversion
        from kika.nuclear_data.model import EVAL_LABEL

        stated = {self.componentFrames.get(component)
                  for component in orders.values()}
        if len(stated) > 1:
            raise ValueError(
                f"MT{mt}: the orders of one reaction carry different frames "
                f"{sorted(map(str, stated))}; one drawn vector is in one frame")
        (covarianceFrame,) = stated
        if covarianceFrame is None:
            return None
        form = product.distribution[EVAL_LABEL]
        awr = None
        for holder in (reaction, suite):
            awr = getattr(getattr(holder, "provenance", None), "awr", None)
            if awr is not None:
                break
        q = 0.0
        if int(mt) != 2:
            q = getattr(getattr(reaction.outputChannel, "Q", None), "value", None)
            if q is None:
                raise ValueError(
                    f"MT{mt}: the covariance is in {covarianceFrame} and the "
                    f"distribution in {form.productFrame}; the frame change "
                    f"needs the reaction's Q, and this suite does not know it")
        return FrameConversion.between(form.productFrame, covarianceFrame,
                                       awr=awr, q=q, mt=mt)

    @staticmethod
    def _angularOf(reaction, mt: int):
        """The product whose evaluated distribution carries Legendre coefficients.

        MF34 is a covariance of *the* angular distribution of the reaction, and
        ENDF has one per MT; §17.2.1 hangs it on a product, so the product has
        to be found. Raises rather than guessing when a channel has two
        candidates -- a recoil and its partner both carry an angular
        distribution, and perturbing the wrong one is invisible.
        """
        from kika.nuclear_data.model import EVAL_LABEL

        channel = getattr(reaction, "outputChannel", None)
        candidates = []
        for product in (getattr(channel, "products", None) or ()):
            distribution = getattr(product, "distribution", None)
            if distribution is None:
                continue
            form = (distribution.get(EVAL_LABEL)
                    if hasattr(distribution, "get") else distribution)
            if form is not None and getattr(form, "angular", None) is not None:
                candidates.append((product, form.angular))
        if not candidates:
            raise ValueError(
                f"MT{mt} has no product carrying an evaluated angular "
                f"distribution, so an MF34 perturbation has nothing to act on"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"MT{mt} has {len(candidates)} products carrying an angular "
                f"distribution and MF34 states one covariance; which one it is "
                f"about is not in the file, so it is not for this to guess"
            )
        return candidates[0]

    def _labelled(self, form):
        """*form* carrying this realisation's label as its own.

        **Not a formality, and the GNDS writer is what proves it.** A §9.1
        container keys its forms by label, and a form also carries one; the
        applier returns a node that kept the evaluated form's, so a realisation
        stored under ``realization-0007`` still said ``eval`` about itself.
        ``gnds/encode.py``'s ``crossSection`` writes ``_function(element, form)``
        and that takes the label off the *form*, so both cross sections came out
        as ``label="eval"`` -- two forms with one label, which no reader can
        tell apart and which the schema does not allow. The ENDF path never saw
        it: ``encodeMF3MT(..., label=)`` looks the form up by key and writes one
        section. So the container and the form have to agree, and this is where
        they are made to.
        """
        import dataclasses

        if getattr(form, "label", None) == self.label:
            return form
        try:
            return dataclasses.replace(form, label=self.label)
        except TypeError:
            return form

    def _putRealisation(self, product, angular) -> None:
        """The perturbed distribution, beside the evaluated one, under the label."""
        import dataclasses

        from kika.nuclear_data.model import EVAL_LABEL

        distribution = product.distribution
        form = distribution[EVAL_LABEL]
        distribution[self.label] = self._labelled(
            dataclasses.replace(form, angular=angular))

    # ------------------------------------------------------------------
    # On disk
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-ready mapping. Grids and factors become lists of floats."""
        return {
            "format": _FORMAT_VERSION,
            "label": self.label,
            "semantics": self.semantics,
            "edgeRule": self.edgeRule,
            "provenance": dict(self.provenance),
            "groups": [[list(component) for component in group]
                       for group in self.groups],
            "blocks": [
                {
                    "component": list(component),
                    "quantity": component.quantity,
                    "semantics": self.semanticsOf(component),
                    "factors": [float(v) for v in self.factors[component]],
                    "binEdges": [float(e) for e in self.binEdges[component]],
                    **({"outerDomain": list(self.outerDomains[component])}
                       if component in self.outerDomains else {}),
                    **({"frame": self.componentFrames[component]}
                       if component in self.componentFrames else {}),
                }
                for component in self.components()
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PerturbationSet":
        version = int(data.get("format", 0))
        if version not in _READABLE_VERSIONS:
            raise ValueError(
                f"perturbation set format {version}, this kika writes "
                f"{_FORMAT_VERSION} and reads {list(_READABLE_VERSIONS)}"
            )
        setSemantics = data.get("semantics", SEMANTICS[0])
        factors, binEdges, outerDomains, componentSemantics = {}, {}, {}, {}
        componentFrames = {}
        for block in data["blocks"]:
            component = ComponentKey(*(int(v) for v in block["component"]))
            factors[component] = np.asarray(block["factors"], dtype=float)
            binEdges[component] = np.asarray(block["binEdges"], dtype=float)
            semantics = block.get("semantics")
            if semantics is not None and semantics != setSemantics:
                componentSemantics[component] = semantics
            domain = block.get("outerDomain")
            if domain is not None:
                outerDomains[component] = (float(domain[0]), float(domain[1]))
            if block.get("frame") is not None:
                componentFrames[component] = block["frame"]
        groups = tuple(tuple(ComponentKey(*(int(v) for v in component))
                             for component in group)
                       for group in data.get("groups", ()))
        return cls(label=data["label"], factors=factors, binEdges=binEdges,
                   groups=groups, semantics=setSemantics,
                   componentSemantics=componentSemantics,
                   outerDomains=outerDomains, componentFrames=componentFrames,
                   edgeRule=data.get("edgeRule", EDGE_RULE),
                   provenance=dict(data.get("provenance", {})))

    def write(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path) -> "PerturbationSet":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def fromRun(cls, directory, sample: int, *, name: str = FACTORS_STEM
                ) -> "PerturbationSet":
        """Realisation *sample* of a run, back from its factors table.

        The one-file-per-run counterpart of :meth:`read`; see
        :func:`writeFactorsTable`.
        """
        return readFactorsTable(directory, sample, name=name)

    def __repr__(self) -> str:
        return (f"PerturbationSet({self.label!r}, "
                f"{len(self.factors)} component(s), "
                f"{'+'.join(self.quantities())})")


# ---------------------------------------------------------------------------
# One file per run
# ---------------------------------------------------------------------------

_TABLE_FORMAT_VERSION = 1


def _factorsIndex(sets: Sequence["PerturbationSet"]) -> Dict[str, Any]:
    """What is true of every sample, written once.

    Checks that it *is* true of every sample -- same components, same grids,
    same semantics -- because a table whose rows mean different things in
    different samples is exactly the file this format exists to prevent.
    """
    first = sets[0]
    components = first.components()
    for number, pset in enumerate(sets):
        if pset.components() != components:
            raise ValueError(
                f"sample {number} ({pset.label}) perturbs "
                f"{[c.describe() for c in pset.components()]}, sample 0 "
                f"{[c.describe() for c in components]}: one table cannot hold "
                f"realisations of different requests"
            )
        for component in components:
            if not np.array_equal(pset.binEdges[component], first.binEdges[component]):
                raise ValueError(
                    f"sample {number}: {component.describe()} is stated on a "
                    f"different grid from sample 0"
                )
            if pset.semanticsOf(component) != first.semanticsOf(component):
                raise ValueError(
                    f"sample {number}: {component.describe()} acts under "
                    f"{pset.semanticsOf(component)!r}, sample 0 under "
                    f"{first.semanticsOf(component)!r}"
                )
    shared = {key: value for key, value in first.provenance.items()
              if key != "sample"}
    for pset in sets[1:]:
        for key, value in pset.provenance.items():
            if key != "sample" and shared.get(key) != value:
                shared.pop(key, None)
    return {
        "format": _TABLE_FORMAT_VERSION,
        "setFormat": _FORMAT_VERSION,
        "nSamples": len(sets),
        "labels": [pset.label for pset in sets],
        "semantics": first.semantics,
        "edgeRule": first.edgeRule,
        "provenance": shared,
        "groups": [[list(component) for component in group]
                   for group in first.groups],
        "blocks": [
            {
                "component": list(component),
                "quantity": component.quantity,
                "describe": component.describe(),
                "semantics": first.semanticsOf(component),
                "nBins": int(first.factors[component].size),
                "binEdges": [float(e) for e in first.binEdges[component]],
                **({"outerDomain": list(first.outerDomains[component])}
                   if component in first.outerDomains else {}),
                **({"frame": first.componentFrames[component]}
                   if component in first.componentFrames else {}),
            }
            for component in components
        ],
        "columns": {
            "sample": "0-based sample number, the row of `labels`",
            "za": "ZA of the component", "mf": "ENDF file the covariance came from",
            "mt": "reaction", "index": "Legendre order (MF34) or band (MF35), else 0",
            "bin": "0-based bin on `binEdges` of that component",
            "value": "the drawn factor (multiplicative-relative) or delta (additive-absolute)",
        },
    }


def writeFactorsTable(sets: Sequence["PerturbationSet"], directory, *,
                      name: str = FACTORS_STEM) -> Tuple[Path, Path]:
    """Every realisation of a run in **one** self-describing table.

    ``<name>.parquet`` has one row per ``(sample, component, bin)`` with the
    drawn value, zstd-compressed, and carries its **index** in the parquet
    schema metadata: what a row means -- the bin edges, the semantics and the
    outer domain per component, the groups, the shared provenance, and the
    label of every sample. That is what :meth:`PerturbationSet.write` wrote
    once per sample, with the part that never changes between samples written
    once, inside the one file it describes. :func:`readFactorsIndex` reads it
    back without touching a row.

    Returns the table's path twice, for callers written when the index was a
    sidecar file.

    Why a table and not a thousand JSON files: a 1 000-sample run of the
    Fe-56 MF33 request is 1 000 files of 13 KB each, and every one of them
    repeats the same 125 bin edges and the same provenance. The shipped
    pipelines wrote a single factors parquet per run for the same reason.
    """
    import pandas as pd

    sets = list(sets)
    if not sets:
        raise ValueError("no realisations to write")
    index = _factorsIndex(sets)
    components = sets[0].components()

    samples, zas, mfs, mts, indices, bins, values = [], [], [], [], [], [], []
    for number, pset in enumerate(sets):
        for component in components:
            block = np.asarray(pset.factors[component], dtype=float)
            n = block.size
            samples.append(np.full(n, number, dtype=np.int32))
            zas.append(np.full(n, component.za, dtype=np.int32))
            mfs.append(np.full(n, component.mf, dtype=np.int16))
            mts.append(np.full(n, component.mt, dtype=np.int16))
            indices.append(np.full(n, component.index, dtype=np.int16))
            bins.append(np.arange(n, dtype=np.int32))
            values.append(block)
    frame = pd.DataFrame({
        "sample": np.concatenate(samples), "za": np.concatenate(zas),
        "mf": np.concatenate(mfs), "mt": np.concatenate(mts),
        "index": np.concatenate(indices), "bin": np.concatenate(bins),
        "value": np.concatenate(values),
    })

    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table = directory / f"{name}.parquet"
    # The index travels *inside* the parquet, as schema metadata, so the table
    # is one self-describing file rather than a file and a sidecar that can be
    # separated. `pq.read_schema` gets it back without reading a row.
    arrow = pa.Table.from_pandas(frame, preserve_index=False)
    arrow = arrow.replace_schema_metadata({
        **(arrow.schema.metadata or {}),
        _INDEX_METADATA_KEY: json.dumps(index).encode("utf-8"),
    })
    pq.write_table(arrow, table, compression="zstd")
    return table, table


#: The parquet schema-metadata key under which the index is stored.
_INDEX_METADATA_KEY = b"kika.factors_index"


def readFactorsIndex(directory, *, name: str = FACTORS_STEM) -> Dict[str, Any]:
    """The index of a run's factors table, read from the parquet's metadata.

    *directory* may also be the path of the parquet itself.
    """
    import pyarrow.parquet as pq

    path = Path(directory)
    if path.is_dir():
        path = path / f"{name}.parquet"
    metadata = pq.read_schema(path).metadata or {}
    if _INDEX_METADATA_KEY not in metadata:
        raise ValueError(
            f"{path.name} carries no kika factors index in its schema metadata; "
            f"it was not written by writeFactorsTable")
    data = json.loads(metadata[_INDEX_METADATA_KEY].decode("utf-8"))
    version = int(data.get("format", 0))
    if version != _TABLE_FORMAT_VERSION:
        raise ValueError(
            f"factors table format {version}; this kika reads "
            f"{_TABLE_FORMAT_VERSION}"
        )
    return data


def readFactorsTable(directory, sample: int, *, name: str = FACTORS_STEM
                     ) -> "PerturbationSet":
    """One realisation back from the table, as the set that was applied.

    Reads only the rows of *sample* -- the parquet filter pushes down to the
    row groups -- so picking sample 731 of a thousand does not load the other
    999.
    """
    import pandas as pd

    directory = Path(directory)
    table = directory / f"{name}.parquet" if directory.is_dir() else directory
    index = readFactorsIndex(table, name=name)
    if not 0 <= int(sample) < int(index["nSamples"]):
        raise IndexError(
            f"sample {sample} of a run with {index['nSamples']} sample(s)")
    frame = pd.read_parquet(table, filters=[("sample", "==", int(sample))])
    if frame.empty:
        raise ValueError(f"the table holds no rows for sample {sample}")

    factors, binEdges, outerDomains, componentSemantics = {}, {}, {}, {}
    componentFrames = {}
    setSemantics = index.get("semantics", SEMANTICS[0])
    for block in index["blocks"]:
        component = ComponentKey(*(int(v) for v in block["component"]))
        rows = frame[(frame["za"] == component.za) & (frame["mf"] == component.mf)
                     & (frame["mt"] == component.mt)
                     & (frame["index"] == component.index)].sort_values("bin")
        values = rows["value"].to_numpy(dtype=float)
        if values.size != int(block["nBins"]):
            raise ValueError(
                f"sample {sample}: {component.describe()} has {values.size} "
                f"value(s) in the table and {block['nBins']} in the index"
            )
        if not np.array_equal(rows["bin"].to_numpy(), np.arange(values.size)):
            raise ValueError(
                f"sample {sample}: {component.describe()} has bins missing or "
                f"repeated in the table"
            )
        factors[component] = values
        binEdges[component] = np.asarray(block["binEdges"], dtype=float)
        semantics = block.get("semantics")
        if semantics is not None and semantics != setSemantics:
            componentSemantics[component] = semantics
        domain = block.get("outerDomain")
        if domain is not None:
            outerDomains[component] = (float(domain[0]), float(domain[1]))
        if block.get("frame") is not None:
            componentFrames[component] = block["frame"]
    groups = tuple(tuple(ComponentKey(*(int(v) for v in component))
                         for component in group)
                   for group in index.get("groups", ()))
    provenance = dict(index.get("provenance", {}))
    provenance["sample"] = int(sample)
    return PerturbationSet(label=index["labels"][int(sample)], factors=factors,
                           binEdges=binEdges, groups=groups,
                           semantics=setSemantics,
                           componentSemantics=componentSemantics,
                           outerDomains=outerDomains,
                           componentFrames=componentFrames,
                           edgeRule=index.get("edgeRule", EDGE_RULE),
                           provenance=provenance)


def _inputNormalisationResidual(energyForm, bands) -> Dict[Any, float]:
    """``{band: max |int chi - 1|}`` over the evaluated nodes inside each band.

    The model twin of ``mf35_sampling.normalisation_residual``, kept per band so
    it sits beside the band's own budgets. A band with no node inside it gets 0.
    """
    worst: Dict[Any, float] = {key: 0.0 for key in bands}
    for k, value in enumerate(list(energyForm.outerDomainValues)):
        for key, (lo, hi) in bands.items():
            if lo <= float(value) <= hi:
                residual = abs(float(energyForm.normalisation(k)) - 1.0)
                worst[key] = max(worst[key], residual)
    return worst

