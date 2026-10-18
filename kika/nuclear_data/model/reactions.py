"""GNDS-2.1 §15.1.1 ``reaction``, and the containers that hold reactions."""
from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Union

from .cross_section_forms import CrossSection
from .output_channel import OutputChannel
from .reaction_id import ReactionId
from .sums import MultiplicitySum, MultiplicitySums, Summands

__all__ = ["Reaction", "CrossSectionSum", "Reactions", "Sums", "OrphanProducts",
           "FissionComponents", "Productions", "IncompleteReactions",
           "isRadionuclideProduction"]


@dataclass
class Reaction:
    """§15.1.1. One reaction: its identity, its cross section, its products."""

    id: ReactionId
    crossSection: CrossSection = field(default_factory=CrossSection)
    outputChannel: OutputChannel = field(default_factory=OutputChannel)
    doubleDifferentialCrossSection: Optional[object] = None
    availableEnergy: Optional[object] = None
    availableMomentum: Optional[object] = None
    #: Not a GNDS node. Where the section came from, in its own format's terms,
    #: so an encoder can write it back without recomputing anything. GNDS keeps
    #: this kind of thing in `documentation`; kika keeps it typed and separate
    #: because the encoders need it and `documentation` is free text.
    provenance: Optional[object] = None

    @property
    def label(self) -> str:
        return self.id.label

    @property
    def ENDF_MT(self) -> Optional[int]:
        """Present because §15.1.1 requires it, and *derived* because §15.1.1
        deprecates it in the same breath. Nothing in the model keys off it."""
        return self.id.ENDF_MT

    def __repr__(self) -> str:
        return f"Reaction({self.id}, forms={sorted(self.crossSection)})"


@dataclass
class CrossSectionSum(Reaction):
    """§21.2 ``crossSectionSum``: MT1, MT3, MT4 — a σ **and** what it sums.

    A subclass of :class:`Reaction` rather than a node of its own, because
    :class:`Sums` has always been a list of reactions and
    ``ReactionSuite.reactionByENDF_MT`` searches it — asking for MT1 and being
    told "that is not a reaction" would be pedantically right and useless. It
    *is* a reaction-shaped thing: it has an ENDF_MT, a cross section in several
    forms, and a Q. What it does not have is products, so its inherited
    ``outputChannel`` stays empty, which §21.2 agrees with — the node has no
    ``outputChannel`` child at all.

    ``summands`` is the addition. Without it a reader keeps MT1's σ and throws
    away the statement of *which* partials it is the sum of, and that statement
    is the only thing distinguishing a sum from an ordinary reaction that
    happens to be labelled "total". §21.2 makes ``summands`` mandatory.

    The evaluated σ is **not** recomputed from the summands, for the reason
    :class:`~kika.nuclear_data.model.sums.MultiplicitySum` gives: the evaluation
    states it, and it need not equal the sum of the parts to the last digit.
    """

    summands: Summands = field(default_factory=Summands)

    def __repr__(self) -> str:
        return (
            f"CrossSectionSum({self.id}, {len(self.summands)} summands, "
            f"forms={sorted(self.crossSection)})"
        )


class _ReactionList:
    """Shared behaviour for the several §14.1.1 children that are lists of reactions."""

    def __init__(self, reactions: Optional[List[Reaction]] = None) -> None:
        self.reactions: List[Reaction] = list(reactions or [])

    def __len__(self) -> int:
        return len(self.reactions)

    def __iter__(self) -> Iterator[Reaction]:
        return iter(self.reactions)

    def __bool__(self) -> bool:
        # An empty container is still *present*, which is the roadmap's rule.
        # Without this, `if suite.reactions:` would read an empty-but-declared
        # container as absent, which is exactly the distinction being made.
        return True

    def append(self, reaction: Reaction) -> None:
        self.reactions.append(reaction)

    def remove(self, reaction: Reaction) -> None:
        """Drop ``reaction`` (the object itself, not an equal one)."""
        for i, held in enumerate(self.reactions):
            if held is reaction:
                del self.reactions[i]
                return
        raise ValueError("that reaction is not in this container")

    def byLabel(self, label: str) -> Reaction:
        for reaction in self.reactions:
            if reaction.label == label:
                return reaction
        raise KeyError(f"no reaction labelled {label!r}")

    def byENDF_MT(self, mt: int) -> Reaction:
        """Look-up by MT, for the ENDF adapters. Deliberately not the primary key."""
        for reaction in self.reactions:
            if reaction.ENDF_MT == mt:
                return reaction
        raise KeyError(f"no reaction with ENDF_MT {mt}")

    @property
    def ENDF_MTs(self) -> List[int]:
        """The MTs this container holds, sorted. ``None`` — a reaction with no
        MT equivalent, which §15.1.1 says future evaluations may carry — is
        skipped rather than sorted against the integers."""
        return sorted(r.ENDF_MT for r in self.reactions if r.ENDF_MT is not None)

    def __contains__(self, key: Union[int, str]) -> bool:
        try:
            self[key]
        except (KeyError, TypeError):
            return False
        return True

    def __getitem__(self, key: Union[int, str]) -> Reaction:
        """``reactions[102]`` is capture; ``reactions['capture']`` is the same object.

        **An integer key is an MT, not a position.** This breaks the list
        convention on purpose. The position of a reaction is an artefact of the
        order the file happened to store its sections in, and nobody has ever
        wanted "the seventh reaction"; MT is the number users actually hold in
        their heads. Iteration stays ordered, so ``list(reactions)[0]`` is still
        there for the rare caller who genuinely means position.

        ``numbers.Integral`` rather than ``int`` because an MT that arrives out
        of a numpy array is an ``np.int64``, which is not an ``int``, and
        failing on it would be a papercut with no defensible reason.
        """
        if isinstance(key, str):
            return self.byLabel(key)
        if isinstance(key, numbers.Integral):
            mt = int(key)
            try:
                return self.byENDF_MT(mt)
            except KeyError:
                raise KeyError(
                    f"no reaction with MT{mt} in {type(self).__name__}; it holds "
                    f"{self.ENDF_MTs}. Indexing here is by MT, not by position. "
                    f"Summed quantities such as MT1 and MT4 are not exclusive "
                    f"reactions and belong in `suite.sums` rather than here; "
                    f"`suite.reactionByENDF_MT()` searches every container. "
                    f"(Which container a given decoder actually fills is its own "
                    f"business -- the ENDF one puts every MF3 section here.)"
                ) from None
        raise TypeError(
            f"a reaction is looked up by MT (int) or by label (str), not by "
            f"{type(key).__name__}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self.reactions)})"


class Reactions(_ReactionList):
    """§14.1.1. The exclusive reactions. Their cross sections sum to the total."""


class Sums(_ReactionList):
    """§21.1. ``crossSectionSums`` (MT1, MT4, ...) and ``multiplicitySums``.

    The reaction list this inherits **is** ``crossSectionSums``: iterating a
    ``Sums`` yields reactions, which is what every existing caller expects and
    what the ENDF decoder fills. ``multiplicitySums`` is the second §21.1 child,
    and it is a separate attribute rather than more entries in the same list
    because a multiplicity sum is not a reaction — it has no cross section and
    no output channel, and putting one in the list would break
    ``reactionByENDF_MT``.
    """

    def __init__(self, reactions: Optional[List[Reaction]] = None,
                 multiplicitySums: Optional[List["MultiplicitySum"]] = None) -> None:
        super().__init__(reactions)
        self.multiplicitySums = MultiplicitySums(list(multiplicitySums or []))


class OrphanProducts(_ReactionList):
    """§14.1.1. Products with no identified parent reaction."""


class FissionComponents(_ReactionList):
    """§14.1.1. First-chance, second-chance, ... fission."""


class Productions(_ReactionList):
    """§14.1.1. Production reactions."""


def isRadionuclideProduction(reaction) -> bool:
    """A ``production`` of a radioactive nuclide (ENDF MF8-10, roadmap E6).

    Told apart from the other responses ``suite.productions`` holds (gas
    production, MT203-207) by its channel's ``genre``, which the ENDF and GNDS
    decoders both set to ``production``. It shares its MT with the reaction that
    makes it -- MT102's isomer is MT102 -- so it is not a cross section of the
    material: whatever projects reactions by MT (the resonance reconstruction,
    Doppler broadening, OpenMC's format) leaves it out and treats it on its own.
    """
    return getattr(getattr(reaction, "outputChannel", None), "genre", None) == "production"


#: Outgoing neutrons by MT, for the reactions whose MT fixes the count (ENDF-6
#: Appendix B). Only what an MF4 or MF5 section can leave without a
#: multiplicity; fission is absent on purpose, its count is the nu-bar.
_NEUTRONS_BY_MT = {2: 1, 4: 1, 11: 2, 16: 2, 17: 3, 22: 1, 23: 1, 24: 2, 25: 3,
                   28: 1, 29: 1, 30: 2, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1,
                   37: 4, 41: 2, 42: 3, 44: 1, 45: 1,
                   **{mt: 1 for mt in range(50, 92)},
                   **{mt: 2 for mt in range(875, 892)}}

#: The one light particle each charged-particle level series emits.
_CHARGED_SERIES = ((600, "H1"), (650, "H2"), (700, "H3"), (750, "He3"), (800, "He4"))


def impliedMultiplicity(mt, pid: str):
    """How many ``pid`` the reaction ``mt`` emits when its MT alone says so, else ``None``.

    ENDF writes no multiplicity for the neutron of an MF4 or MF5 section, nor for the
    particle of an MF4 charged level: the MT fixes it (ENDF-6 Appendix B), two for
    (n,2n), three for (n,3n). FUDGE writes that count as the multiplicity, and so must
    whatever leaves the model without one.
    """
    if mt is None:
        return None
    if pid == "n":
        return _NEUTRONS_BY_MT.get(mt)
    for first, light in _CHARGED_SERIES:
        if first <= mt < first + 50 and pid == light:
            return 1
    return None


class IncompleteReactions(_ReactionList):
    """§14.1.1. Reactions with only some data available."""
