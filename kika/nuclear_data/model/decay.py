"""§12's ``decayData``, the electromagnetic part: how an excited level decays.

ENDF states a level's gamma cascade in MF12 with LO=2 -- the transition
probability array (TP, and GP when LG=2) of each level to the levels below it.
GNDS puts the same physics on the **particle**, in PoPs: the level ``Fe56_e3``
carries ``decayData/decayModes``, one ``decayMode mode="electroMagnetic"`` per
transition, whose ``decayPath/decay/products`` name the photon and the level it
ends on. The reaction that leaves the residual in that level only points at it
(``branching1d``/``branching3d``). This is how FUDGE converts an evaluation and
how the 558 distributed GNDS files read (``n-016_S_036.endf.gnds.xml``).

This module is the **subset roadmap E5c needs**, with the names of §12 and of
``gnds.xsd:539-575``: ``decayData > decayModes > decayMode(label, mode) >
probability, photonEmissionProbabilities?, decayPath > decay(index) > products``.
The rest of §12's decay data -- ``spectra``, ``averageEnergies``,
``internalConversionCoefficients``… -- belongs to the decay sublibrary (roadmap
E7) and is still reported, not read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .output_channel import Product

__all__ = ["Shell", "PhotonEmissionProbabilities", "Decay", "DecayPath",
           "DecayMode", "DecayModes", "DecayData", "ELECTROMAGNETIC"]

#: The ``mode`` of a gamma transition, as §12 and FUDGE spell it.
ELECTROMAGNETIC = "electroMagnetic"


@dataclass
class Shell:
    """``photonEmissionProbabilities/shell``: ``label="total"`` holds ENDF's GP."""

    label: str
    value: float


@dataclass
class PhotonEmissionProbabilities:
    """The probability that the transition emits a photon rather than converting.

    ENDF's GP (MF12 LO=2 with LG=2). Absent when LG=1, where every transition
    is a photon.
    """

    shells: List[Shell] = field(default_factory=list)

    def total(self) -> Optional[float]:
        return next((s.value for s in self.shells if s.label == "total"), None)


@dataclass
class Decay:
    """One step of a decay path: what comes out (the photon and the final level)."""

    index: int
    products: List[Product] = field(default_factory=list)
    mode: Optional[str] = None


@dataclass
class DecayPath:
    decays: List[Decay] = field(default_factory=list)

    def __iter__(self):
        return iter(self.decays)

    def __len__(self) -> int:
        return len(self.decays)


@dataclass
class DecayMode:
    """One transition of a level: its probability (ENDF's TP) and where it goes.

    ``probability`` is a float rather than a node: GNDS writes it as a single
    labelled ``<double>`` and kika keeps one evaluation per suite, the same
    flattening ``Shell.value`` is.
    """

    label: str
    mode: str
    probability: float
    decayPath: DecayPath = field(default_factory=DecayPath)
    photonEmissionProbabilities: Optional[PhotonEmissionProbabilities] = None

    def finalState(self) -> Optional[str]:
        """The pid the transition ends on: the non-photon product of its last decay."""
        for decay in reversed(self.decayPath.decays):
            for product in decay.products:
                if product.pid != "photon":
                    return product.pid
        return None


@dataclass
class DecayModes:
    decayModes: List[DecayMode] = field(default_factory=list)

    def __iter__(self):
        return iter(self.decayModes)

    def __len__(self) -> int:
        return len(self.decayModes)

    def __bool__(self) -> bool:
        return True


@dataclass
class DecayData:
    """§12 ``decayData`` on a particle. Only ``decayModes`` is modelled (E5c)."""

    decayModes: DecayModes = field(default_factory=DecayModes)

    def __repr__(self) -> str:
        return f"DecayData({len(self.decayModes)} decayMode(s))"
