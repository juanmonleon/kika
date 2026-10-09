"""§12's ``decayData``, the electromagnetic part: how an excited level decays.

ENDF states a level's gamma cascade in MF12 with LO=2 -- the transition
probability array (TP, and GP when LG=2) of each level to the levels below it.
GNDS puts the same physics on the **particle**, in PoPs: the level ``Fe56_e3``
carries ``decayData/decayModes``, one ``decayMode mode="electroMagnetic"`` per
transition, whose ``decayPath/decay/products`` name the photon and the level it
ends on. The reaction that leaves the residual in that level only points at it
(``branching1d``/``branching3d``). This is how FUDGE converts an evaluation and
how the 558 distributed GNDS files read (``n-016_S_036.endf.gnds.xml``).

Roadmap E5c modelled the electromagnetic subset, with the names of §12 and of
``gnds.xsd:539-575``: ``decayData > decayModes > decayMode(label, mode) >
probability, photonEmissionProbabilities?, decayPath > decay(index) > products``.

Roadmap E7b adds the rest of §12's decay data, which is what a radioactive
decay sublibrary (ENDF MF8/MT457) states: a mode's ``Q`` and the uncertainty of
its probability, its ``spectra`` (``spectrum(label, pid)`` of ``discrete``
lines -- intensity, energy, transition type, internal conversion coefficients,
positron emission intensity, internal pair formation -- and ``continuum``
tables), and the ``averageEnergies`` of the decay. The labels are the ones
FUDGE writes (``PoPs/decays/spectrum.py``, ``averageEnergy.py``), so a file
kika writes and one FUDGE writes name the same things. Only a mode-level
``internalConversionCoefficients``, which no evaluation kika has read states,
is still reported rather than read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .output_channel import Product
from .quantities import PhysicalQuantity

__all__ = ["Shell", "PhotonEmissionProbabilities", "Decay", "DecayPath",
           "DecayMode", "DecayModes", "DecayData", "ELECTROMAGNETIC",
           "Spectrum", "Discrete", "Continuum", "TRANSITION_TYPES",
           "SPECTRUM_LABELS", "AVERAGE_ENERGY_LABELS"]

#: The ``mode`` of a gamma transition, as §12 and FUDGE spell it.
ELECTROMAGNETIC = "electroMagnetic"

#: ``discrete/@type``: the beta transition's type (ENDF's TYPE 1, 2, 3).
TRANSITION_TYPES = ("allowed", "first-forbidden", "second-forbidden")

#: ``spectrum/@label`` by ENDF's STYP, as FUDGE spells them.
SPECTRUM_LABELS = {0: "gamma", 1: "beta-", 2: "beta+ or electronCapture", 4: "alpha",
                   5: "neutron", 6: "spontaneous fission fragments", 7: "proton",
                   8: "discrete electron", 9: "x-ray", 10: "anti-neutrino",
                   11: "neutrino"}

#: ``averageEnergy/@label`` in ENDF's order: the first three when NC=3, all 17
#: when the evaluation breaks the energy down by radiation.
AVERAGE_ENERGY_LABELS = ("lightParticles", "electroMagneticRadiation", "heavyParticles",
                         "betaMinus", "betaPlus", "AugerElectron", "conversionElectron",
                         "gamma", "xRay", "internalBremsstrahlung", "annihilation",
                         "alpha", "recoil", "spontaneousFission", "fissionNeutrons",
                         "proton", "neutrino")


@dataclass
class Shell:
    """A ``shell``: in ``photonEmissionProbabilities`` ``label="total"`` holds
    ENDF's GP; in ``internalConversionCoefficients`` the labels are ``total``,
    ``K`` and ``L`` (ENDF's RICC, RICK, RICL) and each may carry a standard
    uncertainty."""

    label: str
    value: float
    uncertainty: Optional[float] = None


@dataclass
class Discrete:
    """``spectrum/discrete``: one line of a decay spectrum.

    ``intensity`` is ENDF's relative intensity RI, as FUDGE writes it: the
    absolute one is RI times the spectrum's FD, and GNDS has no node for FD (it
    is 1 everywhere in ENDF/B-VIII.1 but Cf-252 and Sm-164; the ENDF route keeps
    it, see :mod:`kika.endf.model_adapter.decay_sublibrary`).
    """

    intensity: PhysicalQuantity
    energy: PhysicalQuantity
    type: Optional[str] = None
    internalConversionCoefficients: List[Shell] = field(default_factory=list)
    photonEmissionProbabilities: Optional[PhotonEmissionProbabilities] = None
    positronEmissionIntensity: Optional[PhysicalQuantity] = None
    internalPairFormationCoefficient: Optional[PhysicalQuantity] = None


@dataclass
class Continuum:
    """``spectrum/continuum``: the continuous part of a spectrum, an ``XYs1d``
    of the emission probability per unit energy."""

    spectrum: object


@dataclass
class Spectrum:
    """``spectra/spectrum``: what one kind of radiation the mode emits."""

    label: str
    pid: str
    emissions: List[object] = field(default_factory=list)   # Discrete | Continuum

    @property
    def discretes(self) -> List[Discrete]:
        return [e for e in self.emissions if isinstance(e, Discrete)]

    @property
    def continua(self) -> List[Continuum]:
        return [e for e in self.emissions if isinstance(e, Continuum)]


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
    #: ``decay/@complete``: false when the step does not list every product
    #: (FUDGE: a beta+/EC or a spontaneous fission step). ``None`` = not stated.
    complete: Optional[bool] = None


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
    #: The decay sublibrary's part (E7b): the mode's Q, the probability's
    #: standard uncertainty, and the spectra of what it emits.
    Q: Optional[PhysicalQuantity] = None
    probabilityUncertainty: Optional[float] = None
    spectra: List[Spectrum] = field(default_factory=list)
    #: The label of the probability's ``<double>``: ``eval`` for a level's
    #: cascade, ``BR`` (FUDGE's) for a decay sublibrary's branching ratio.
    probabilityLabel: str = "eval"

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
    """§12 ``decayData`` on a particle: its decay modes and average energies."""

    decayModes: DecayModes = field(default_factory=DecayModes)
    #: ``averageEnergies/averageEnergy``: each a quantity whose ``label`` is one
    #: of :data:`AVERAGE_ENERGY_LABELS`.
    averageEnergies: List[PhysicalQuantity] = field(default_factory=list)

    def averageEnergy(self, label: str) -> Optional[PhysicalQuantity]:
        return next((e for e in self.averageEnergies if e.label == label), None)

    def __repr__(self) -> str:
        spectra = sum(len(m.spectra) for m in self.decayModes)
        tail = f", {spectra} spectrum(a)" if spectra else ""
        return f"DecayData({len(self.decayModes)} decayMode(s){tail})"
