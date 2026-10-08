"""GNDS-2.1 §19.4.1 ``tabulatedWidths``: the unresolved region.

Average widths and level spacings as functions of energy, with a degrees-of-
freedom count per channel. ENDF's LSSF flag — whether the URR cross sections are
already in MF3 or must be computed from these parameters — is kept, because
getting it wrong double-counts the unresolved region.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from ..functions import Function1d
from ..enums import Interpolation

import numpy as np

__all__ = ["UnresolvedChannel", "UnresolvedSpinGroup", "TabulatedWidths"]


@dataclass
class UnresolvedChannel:
    """Average width for one channel, constant or tabulated against energy.

    **What the numbers are.** By default the ``"neutron"`` channel holds ENDF's GN0, the
    average *reduced* neutron width, exactly as ENDF and GNDS store it (FUDGE
    copies it verbatim into the ``elastic`` width). The physical average is
    ⟨Γn⟩(E) = GN0 · √E · ν_n · V_l(ρ), with V_l = P_l/ρ (V_0 = 1) — NJOY
    ``unfac`` (reconr.f90:4473-4495) and FUDGE (reconstructResonances.py:
    3273-3281) agree. That conversion belongs to the URR kernel; the model
    never stores the converted width on that path, so ν_n is not applied twice.
    A canonical input explicitly declaring ``neutronWidthConvention="physical"``
    supplies the physical mean in eV instead; neither sqrt(E), nu nor V_l is
    applied again. That additional convention requires KIKA's GNDS extension.

    The other channels are physical average widths in eV. The ``"competitive"``
    width only enters the total width: ENDF-6 §2.1 (LRP=1) puts the competing
    cross sections entirely in File 3 and lets Γx carry their effect on the
    resonance reactions, so the channel needs no exit-reaction descriptor of
    its own (unlike BW's :class:`~.breit_wigner.CompetitiveChannel`).
    """

    label: str
    #: ν of the χ² distribution of this width. **Zero means the width does not
    #: fluctuate** — ENDF's AMUG/AMUF/AMUX=0, and what cases A and B imply for
    #: capture. Non-integer values are real data (TALYS-derived AMUX such as
    #: 1.0123 in ~25 B-VIII.1/JENDL-5 evaluations) and must not be truncated.
    #: The 1.0 default only serves a model built by hand; every reader states
    #: the value it read.
    degreesOfFreedom: float = 1.0
    widths: Optional[np.ndarray] = None
    constantWidth: Optional[float] = None
    #: The energies ``widths`` is tabulated against, when they are **not** the
    #: block's :attr:`TabulatedWidths.energyGrid`. ENDF's URR puts every average
    #: on one grid per range, which is why this field did not exist; GNDS gives
    #: each width its own ``XYs1d``, and **66 of the library's 351 unresolved
    #: blocks use more than one grid** — up to seven of them. Without this,
    #: those 66 come out with one grid's energies attached to another grid's
    #: values, which is a *wrong* average width rather than a missing one.
    #:
    #: ``None`` means "the block's grid", which is the common case and what an
    #: ENDF-decoded evaluation always says. Same two-field shape as
    #: :class:`~kika.nuclear_data.model.resonances.ScatteringRadius`.
    energies: Optional[np.ndarray] = None
    #: Canonical function preserves interpolation and independent regions.
    #: Arrays above remain a compatibility view; processing must use this
    #: function when supplied, without inventing a common grid.
    averageFunction: Optional[Function1d] = None
    #: ENDF/GNDS elastic averages are reduced GN0 by default. A canonical
    #: model may instead explicitly supply physical mean neutron widths in eV,
    #: including deterministic widths (nu=0). This declaration is carried by
    #: the KIKA opt-in extension and cannot be exported as ENDF GN0.
    neutronWidthConvention: Literal["reduced", "physical"] = "reduced"

    def __post_init__(self) -> None:
        if self.widths is not None:
            self.widths = np.asarray(self.widths, dtype=float)
        if self.energies is not None:
            self.energies = np.asarray(self.energies, dtype=float)
            size = 0 if self.widths is None else self.widths.size
            if self.energies.size != size:
                raise ValueError(
                    f"channel {self.label!r} has {self.energies.size} energies "
                    f"and {size} widths"
                )


@dataclass
class UnresolvedSpinGroup:
    """Averages for one (L, J)."""

    L: int
    J: float
    levelSpacing: Optional[np.ndarray] = None
    channels: List[UnresolvedChannel] = field(default_factory=list)
    atomicWeightRatio: Optional[float] = None
    #: The energies ``levelSpacing`` is tabulated against, when they are not the
    #: block's grid. See :attr:`UnresolvedChannel.energies`.
    levelSpacingEnergies: Optional[np.ndarray] = None
    levelSpacingFunction: Optional[Function1d] = None
    #: ENDF URR INT interpolates cross sections, not the average parameters.
    crossSectionInterpolation: Optional[Interpolation] = None
    #: Sigma interpolation nodes, independent of the level-spacing function.
    #: None retains the compatibility fallback to levelSpacingEnergies/block
    #: energyGrid. KIKA's GNDS extension saves the effective sigma grid even
    #: when D and all widths serialize as constant functions.
    crossSectionEnergies: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.levelSpacing is not None:
            self.levelSpacing = np.asarray(self.levelSpacing, dtype=float)
        if self.levelSpacingEnergies is not None:
            self.levelSpacingEnergies = np.asarray(
                self.levelSpacingEnergies, dtype=float
            )
        if self.crossSectionEnergies is not None:
            self.crossSectionEnergies = np.asarray(self.crossSectionEnergies,dtype=float)


@dataclass
class TabulatedWidths:
    """§19.4.1. The unresolved resonance region."""

    spinGroups: List[UnresolvedSpinGroup] = field(default_factory=list)
    energyGrid: Optional[np.ndarray] = None
    scatteringRadius: Optional[float] = None
    #: The unit the radius above was read with. Same field, same reason as
    #: :attr:`~kika.nuclear_data.model.resonances.r_matrix.Channel.radiusUnit`.
    radiusUnit: Optional[str] = None
    #: §19.4.1's ``resonanceReactions`` — the channels the averages are for,
    #: each with the link to the reaction it is. All 351 unresolved blocks in
    #: ENDF/B-VIII.1-GNDS carry them, and the schema makes the ``<link>`` inside
    #: each one mandatory, so a writer that reconstructed the list from the
    #: channel labels alone could not produce a valid file. Typed as the
    #: resolved region's :class:`~.r_matrix.ResonanceReaction` because it is the
    #: same node; an ENDF-decoded evaluation leaves it empty, as ENDF states
    #: the URR channels only by position.
    resonanceReactions: List[object] = field(default_factory=list)
    #: ENDF LSSF: 0 = compute the URR cross sections from these parameters,
    #: 1 = they are already in MF3 and these are for self-shielding only.
    selfShieldingOnly: bool = False
    label: Optional[str] = None
    #: §19.4.1's local ``PoPs``, holding the target and — the part that matters
    #: for a calculation — its **spin**. The URR cross sections carry the same
    #: ``g_J = (2J+1) / (2(2I+1))`` factor the resolved region does, so a
    #: reconstructor needs I here as much as there. The resolved formalisms
    #: (:class:`~.breit_wigner.BreitWigner`, :class:`~.r_matrix.RMatrix`) have
    #: carried this field since 3b; the unresolved one did not, which made the
    #: two halves of the same calculation ask for the same number in different
    #: ways. Added in phase 4.
    PoPs: Optional[object] = None
    #: The region's own radius policy, the same
    #: :class:`~.radius_policy.RadiusPolicy` the resolved formalisms carry.
    #: ``channelMode`` says what P and S use (ENDF NAPS: ``"mass"`` = 0,
    #: ``"phase"`` = 1); ``phaseRadius`` is the region's energy-dependent AP(E)
    #: (ENDF NRO=1), and ``None`` there means the constant
    #: :attr:`scatteringRadius`. GNDS states the same three facts on
    #: ``tabulatedWidths`` itself: ``calculateChannelRadius``,
    #: ``hardSphereRadius`` and ``scatteringRadius``.
    #:
    #: Before this field the URR's NAPS lived only in ENDF provenance, and its
    #: AP(E) was copied to :attr:`Resonances.scatteringRadius`, the radius of
    #: the **whole evaluation** — Au-197 (B-VIII.1, JEFF-4.0) handed its URR
    #: table to every consumer that read the global radius. ``None`` means the
    #: source stated no policy (a GNDS file without ``calculateChannelRadius``).
    radiusPolicy: Optional[object] = None

    #: Explicit convention for the potential-scattering term when spin groups
    #: have different cross-section interpolation policies. ``None`` retains
    #: the historical common-grid aggregate convention and requires all groups
    #: to agree. ``"continuous"`` evaluates the potential once per L at E;
    #: an Interpolation evaluates it on potentialScatteringEnergies, then
    #: interpolates it independently of each group's resonance contribution.
    #: These explicit policies require KIKA's opt-in GNDS extension; ENDF
    #: cannot represent this additional convention.
    potentialScatteringInterpolation: Optional[Interpolation | Literal["continuous"]] = None
    potentialScatteringEnergies: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.energyGrid is not None:
            self.energyGrid = np.asarray(self.energyGrid, dtype=float)
        if self.potentialScatteringEnergies is not None:
            self.potentialScatteringEnergies = np.asarray(self.potentialScatteringEnergies, dtype=float)
