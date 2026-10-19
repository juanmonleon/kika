"""Photo-atomic and atomic-relaxation data in the model (GNDS §14, §12 ``atomic``).

**Photo-atomic reactions** (ENDF sublibrary NSUB=3, MF23 + MF27) are ordinary
reactions of a ``reactionSuite`` whose projectile is the photon, whose target is
the element (``Fe``) and whose ``interaction`` is ``atomic``: each MT's σ(E) is a
cross section, the totals (MT501 total, 516 pair, 522 photoelectric) are
``crossSectionSum`` nodes over their parts. Coherent and incoherent scattering
add a ``doubleDifferentialCrossSection`` (:class:`~kika.nuclear_data.model.thermal_scattering.DoubleDifferentialCrossSection`)
holding one of the two forms below, as FUDGE writes them:

- :class:`CoherentPhotonScattering`: the form factor F(x) (MF27/MT502) and the
  real and imaginary anomalous scattering factors f'(E), f''(E) (MT505, 506);
- :class:`IncoherentPhotonScattering`: the incoherent scattering function S(x)
  (MF27/MT504).

``x = sin(θ/2)/λ`` in 1/Å (ENDF-6 §27; GNDS stores the same abscissa).

**Atomic relaxation** (NSUB=6, MF28) is a property of the element, so it lives
in PoPs: a :class:`ChemicalElement` with its :class:`Atomic` data, one
:class:`Configuration` per subshell (label, electron number, binding energy)
whose ``decayData`` lists the transitions that fill a vacancy there, as
``decayMode`` nodes of mode ``atomicRelaxation`` with the transition's
probability: a radiative transition emits a ``photon`` and moves the vacancy to
``Fe{2p1/2}``, a non-radiative one emits an ``e-`` and leaves ``Fe{2s1/2,2s1/2}``
(FUDGE's ``ENDF_ITYPE_6``). GNDS has no node for a transition's energy: it is
the difference of binding energies, and an ENDF tape's ETR is kept in the
provenance for the way back.

Subshells are named as FUDGE names them (:data:`SUBSHELL_LABELS`), by ENDF's
designator, which is ``MT - 533`` in MF23 and ``SUBI`` in MF28.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Union

from .functions import Regions1d, XYs1d
from .quantities import PhysicalQuantity

__all__ = ["ATOMIC_INTERACTION", "SUBSHELL_LABELS", "subshellLabel", "subshellDesignator",
           "CoherentPhotonScattering", "IncoherentPhotonScattering",
           "Configuration", "Atomic", "ChemicalElement"]

#: ``reactionSuite/@interaction`` of a photo-atomic or electro-atomic evaluation.
ATOMIC_INTERACTION = "atomic"

#: ENDF subshell designator (1 = K, ``MT - 533``) → the GNDS label, FUDGE's
#: ``MT_AtomicConfigurations``.
SUBSHELL_LABELS = {
    1: "1s1/2", 2: "2s1/2", 3: "2p1/2", 4: "2p3/2", 5: "3s1/2", 6: "3p1/2", 7: "3p3/2",
    8: "3d3/2", 9: "3d5/2", 10: "4s1/2", 11: "4p1/2", 12: "4p3/2", 13: "4d3/2", 14: "4d5/2",
    15: "4f5/2", 16: "4f7/2", 17: "5s1/2", 18: "5p1/2", 19: "5p3/2", 20: "5d3/2", 21: "5d5/2",
    22: "5f5/2", 23: "5f7/2", 24: "5g7/2", 25: "5g9/2", 26: "6s1/2", 27: "6p1/2", 28: "6p3/2",
    29: "6d3/2", 30: "6d5/2", 31: "6f5/2", 32: "6f7/2", 33: "6g7/2", 34: "6g9/2", 35: "6h9/2",
    36: "6h11/2", 37: "7s1/2", 38: "7p1/2", 39: "7p3/2",
}
_DESIGNATORS = {label: designator for designator, label in SUBSHELL_LABELS.items()}


def subshellLabel(designator: int) -> str:
    """``1 → "1s1/2"``; KeyError for a designator ENDF does not define."""
    try:
        return SUBSHELL_LABELS[int(designator)]
    except KeyError:
        raise KeyError(f"ENDF subshell designator {designator} is not defined (1-39)") from None


def subshellDesignator(label: str) -> int:
    """``"1s1/2" → 1``; KeyError for a label that is not one of :data:`SUBSHELL_LABELS`."""
    try:
        return _DESIGNATORS[label]
    except KeyError:
        raise KeyError(f"{label!r} is not an atomic subshell label") from None


Function1d = Union[XYs1d, Regions1d]


@dataclass
class CoherentPhotonScattering:
    """``coherentPhotonScattering``: Rayleigh scattering's form and anomalous factors.

    ``formFactor`` is F(x) against x in 1/Å; ``realAnomalousFactor`` and
    ``imaginaryAnomalousFactor`` are f'(E) and f''(E) against the photon energy
    in eV. Either anomalous factor may be absent.
    """

    formFactor: Function1d
    realAnomalousFactor: Optional[Function1d] = None
    imaginaryAnomalousFactor: Optional[Function1d] = None
    label: Optional[str] = None
    pid: str = "photon"
    productFrame: str = "lab"
    gndsNodeName: ClassVar[str] = "coherentPhotonScattering"


@dataclass
class IncoherentPhotonScattering:
    """``incoherentPhotonScattering``: Compton scattering's scattering function S(x)."""

    scatteringFactor: Function1d
    label: Optional[str] = None
    pid: str = "photon"
    productFrame: str = "lab"
    gndsNodeName: ClassVar[str] = "incoherentPhotonScattering"


@dataclass
class Configuration:
    """§12 ``configuration``: one subshell, its occupancy, binding energy and relaxation.

    ``decayData`` (:class:`~kika.nuclear_data.model.decay.DecayData`) holds the
    transitions that fill a vacancy in this subshell; ``None`` when none is
    stated (an outer subshell).
    """

    subshell: str
    electronNumber: float
    bindingEnergy: PhysicalQuantity
    decayData: Optional[object] = None

    @property
    def designator(self) -> int:
        return subshellDesignator(self.subshell)


@dataclass
class Atomic:
    """§12 ``atomic``: the element's subshell configurations, in the order stated."""

    configurations: List[Configuration] = field(default_factory=list)

    def __getitem__(self, subshell: str) -> Configuration:
        for c in self.configurations:
            if c.subshell == subshell:
                return c
        raise KeyError(f"no configuration {subshell!r}; have {[c.subshell for c in self.configurations]}")


@dataclass
class ChemicalElement:
    """§12 ``chemicalElement``: an element, with its atomic data when it carries some."""

    symbol: str
    Z: int
    name: Optional[str] = None
    atomic: Optional[Atomic] = None
