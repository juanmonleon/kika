"""What one inelastic G4NDL file says, field by field, before any physics.

The inelastic counterpart of :mod:`kika.g4ndl.records`, with the same rule:
every token the consumer reads is kept, in file order, and none is
interpreted. The grammar is Geant4 v11.4.3's, written down with line
references in ``G4NDL_token_spec.md`` §10:

* ``Inelastic/CrossSection/<name>`` is the elastic cross-section grammar
  (``G4ParticleHPIsoData::Init``) and reuses
  :class:`~kika.g4ndl.records.CrossSectionRecord`.
* ``Inelastic/Fxx/<name>`` is a sequence of **sections**, each opened by
  ``infoType dataType``. ``dataType`` is the ENDF file number of what follows
  (3 cross section, 4 angular, 5 energy, 6 energy-angle, 12-15 photons). Two
  consumers read the sections, and they disagree on the header:

  - ``G4ParticleHPInelasticCompFS`` (``F01`` n, ``F23`` p, ``F24`` d, ``F25``
    t, ``F26`` He-3, ``F27`` α, :data:`COMPOSITE_CHANNELS`) reads
    ``infoType dataType sfType dummy`` for **every** section, and a
    ``dataType=3`` section adds ``QI LR`` before its points. ``sfType`` is the
    ENDF MT (4, 51-91, 600-649, ...): one file holds several reactions;
  - ``G4ParticleHPInelasticBaseFS`` (every other ``Fxx``) reads
    ``Qvalue dummy`` after ``infoType dataType`` of the **first** section
    only (``dummy == INT_MAX`` guard), and nothing of the kind afterwards: one
    file is one reaction, whose MT is the directory's.

* ``Inelastic/Gammas/z<Z>.a<A>`` is a different thing altogether: the level
  scheme of a *residual* nucleus, ``(E_level, E_gamma, probability)``
  triples in keV, read with ``std::ifstream`` (plain text only, no
  ``G4NDL`` header, no ``.z``) by ``G4ParticleHPDeExGammas::Init``.

**Repeated-count quirks are kept as data.** Several consumer loops allocate
``max(n, 1)`` items and read that many, so a count of 0 still reads one
item (``G4ParticleHPDiscreteTwoBody``, ``G4ParticleHPArbitaryTab``,
``G4ParticleHPEnAngCorrelation``, ``G4ParticleHPPhotonDist``); others read
exactly ``n``. The records hold the count *as written* beside the items
actually read, so that writing them back reproduces the file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from kika.g4ndl.records import AngularBlock, Interpolation

__all__ = [
    "COMPOSITE_CHANNELS", "CHANNEL_MT", "CHANNEL_NAMES",
    "DT_CROSS_SECTION", "DT_ANGULAR", "DT_ENERGY", "DT_ENERGY_ANGLE",
    "DT_PHOTON_MULTIPLICITY", "DT_PHOTON_PARTIALS", "DT_PHOTON_ANGULAR",
    "DT_PHOTON_ENERGY",
    "Tab1", "Pairs",
    "CrossSectionBody", "AngularBody",
    "EnergyLaw", "EnergyBody",
    "ContinuumEnergy", "ContinuumBody", "DiscreteTwoBodyEnergy", "DiscreteTwoBodyBody",
    "NBodyBody", "LabAngleEnergyAngle", "LabAngleEnergyEnergy", "LabAngleEnergyBody",
    "ProductRecord", "EnergyAngleBody",
    "PhotonLine", "PhotonMultiplicityBody", "PhotonTransition", "PhotonCascadeBody",
    "PhotonPartial", "PhotonPartialsBody",
    "PhotonLegendre", "PhotonTabulated", "PhotonAngularLine", "PhotonAngularBody",
    "PhotonSpectrumAt", "PhotonSpectrum", "PhotonEnergyBody",
    "Section", "InelasticFSRecord", "GammasRecord",
]

# ------------------------------------------------------------------ channels

#: ``G4ParticleHPInelastic.cc:261-296``: the channel classes that derive from
#: ``G4ParticleHPInelasticCompFS`` (checked in the class headers); every other
#: ``Fxx`` derives from ``G4ParticleHPInelasticBaseFS``.
COMPOSITE_CHANNELS = frozenset({"F01", "F23", "F24", "F25", "F26", "F27"})

#: ``Fxx`` → (Geant4 class, ENDF MT of the reaction, as the class name spells
#: it). For a composite channel the MT is that of the lumped reaction
#: (MT4 = (n,n'), MT103-107); the file's ``sfType`` gives each partial's own.
CHANNEL_MT = {
    "F01": 4, "F02": 5, "F03": 11, "F04": 16, "F05": 17, "F06": 22, "F07": 23,
    "F08": 24, "F09": 25, "F10": 28, "F11": 29, "F12": 30, "F13": 32, "F14": 33,
    "F15": 34, "F16": 35, "F17": 36, "F18": 37, "F19": 41, "F20": 42, "F21": 44,
    "F22": 45, "F23": 103, "F24": 104, "F25": 105, "F26": 106, "F27": 107,
    "F28": 108, "F29": 109, "F30": 111, "F31": 112, "F32": 114, "F33": 113,
    "F34": 115, "F35": 116, "F36": 117,
}

#: ``Fxx`` → the Geant4 final-state class that reads it.
CHANNEL_NAMES = {
    "F01": "NInelasticFS", "F02": "NXInelasticFS", "F03": "2NDInelasticFS",
    "F04": "2NInelasticFS", "F05": "3NInelasticFS", "F06": "NAInelasticFS",
    "F07": "N3AInelasticFS", "F08": "2NAInelasticFS", "F09": "3NAInelasticFS",
    "F10": "NPInelasticFS", "F11": "N2AInelasticFS", "F12": "2N2AInelasticFS",
    "F13": "NDInelasticFS", "F14": "NTInelasticFS", "F15": "NHe3InelasticFS",
    "F16": "ND2AInelasticFS", "F17": "NT2AInelasticFS", "F18": "4NInelasticFS",
    "F19": "2NPInelasticFS", "F20": "3NPInelasticFS", "F21": "N2PInelasticFS",
    "F22": "NPAInelasticFS", "F23": "PInelasticFS", "F24": "DInelasticFS",
    "F25": "TInelasticFS", "F26": "He3InelasticFS", "F27": "AInelasticFS",
    "F28": "2AInelasticFS", "F29": "3AInelasticFS", "F30": "2PInelasticFS",
    "F31": "PAInelasticFS", "F32": "D2AInelasticFS", "F33": "T2AInelasticFS",
    "F34": "PDInelasticFS", "F35": "PTInelasticFS", "F36": "DAInelasticFS",
}

DT_CROSS_SECTION, DT_ANGULAR, DT_ENERGY, DT_ENERGY_ANGLE = 3, 4, 5, 6
DT_PHOTON_MULTIPLICITY, DT_PHOTON_PARTIALS = 12, 13
DT_PHOTON_ANGULAR, DT_PHOTON_ENERGY = 14, 15


# ------------------------------------------------------------------ primitives

@dataclass(frozen=True, eq=False)
class Tab1:
    """``G4ParticleHPVector::Init(stream)``: ``N``, an interpolation record, ``N`` pairs."""

    interpolation: Interpolation
    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


@dataclass(frozen=True, eq=False)
class Pairs:
    """``N`` pairs whose count was read by the caller and with no interpolation record."""

    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


# ------------------------------------------------------------------ dataType 3

@dataclass(frozen=True, eq=False)
class CrossSectionBody:
    """``dataType=3``: σ(E) of the section's reaction, eV and barn, lin-lin.

    ``QI`` and ``LR`` are read by the composite consumer only
    (``G4ParticleHPInelasticCompFS.cc:184``); a base-FS section has neither,
    and its Q is the file's first-section ``Qvalue``.
    """

    QI: Optional[float]
    LR: Optional[int]
    points: Pairs


# ------------------------------------------------------------------ dataType 4

@dataclass(frozen=True, eq=False)
class AngularBody:
    """``dataType=4``, ``G4ParticleHPAngular::Init``: ENDF MF4 for the outgoing particle.

    ``repFlag`` 0 isotropic (no block, and — unlike the elastic FS — no
    second frame flag), 1 Legendre, 2 tabulated. There is no ``repFlag=3``:
    the consumer throws on it.
    """

    repFlag: int
    targetMass: float
    frameFlag: int
    legendre: Optional[AngularBlock] = None
    tabulated: Optional[AngularBlock] = None


# ------------------------------------------------------------------ dataType 5

@dataclass(frozen=True, eq=False)
class EnergyLaw:
    """One partial of ``G4ParticleHPEnergyDistribution``: ENDF MF5 LF and its records.

    ``law`` is the representation type the consumer switches on (1 tabulated,
    5 general evaporation, 7 simple fission (Maxwellian), 9 evaporation,
    11 Watt, 12 Madland-Nix). ``probability`` is ``p(E)``. The rest by law:

    * 1 — ``nDistFunc`` (as written), its ``interpolation`` in incident
      energy, and ``spectra``: ``(E, Tab1 g(E'))`` for ``max(nDistFunc, 1)``
      energies;
    * 5 — ``parameters = (θ(E), g(x))``; 7, 9 — ``(θ(E),)``; 11 — ``(a(E), b(E))``;
    * 12 — ``scalars = (EFL, EFH)`` and ``parameters = (T_M(E),)``.
    """

    law: int
    probability: Tab1
    nDistFunc: Optional[int] = None
    interpolation: Optional[Interpolation] = None
    spectra: Tuple[Tuple[float, Tab1], ...] = ()
    parameters: Tuple[Tab1, ...] = ()
    scalars: Tuple[float, ...] = ()


@dataclass(frozen=True, eq=False)
class EnergyBody:
    """``dataType=5``: ``dummy nPartials`` and the partials. ``dummy`` is read as a double."""

    dummy: float
    partials: Tuple[EnergyLaw, ...]


# ------------------------------------------------------------------ dataType 6

@dataclass(frozen=True, eq=False)
class ContinuumEnergy:
    """One incident energy of ``G4ParticleHPContAngularPar``.

    ``nEnergies`` outgoing energies, of which the first ``nDiscrete`` are
    discrete lines (ENDF ND), each a row of ``nAngularParameters`` values
    after its ``E'``: ``rows`` has shape ``(nEnergies, 1 + nAngularParameters)``.
    """

    energy: float
    nDiscrete: int
    nAngularParameters: int
    rows: np.ndarray


@dataclass(frozen=True, eq=False)
class ContinuumBody:
    """``distLaw=1``, ``G4ParticleHPContEnergyAngular``: ENDF MF6 LAW=1.

    ``targetCode`` (read as a double), ``angularRep`` (LANG), the single
    outgoing-energy interpolation code (LEP), the incident-energy
    interpolation record and the energies.
    """

    targetCode: float
    angularRep: int
    secondaryInterpolation: int
    interpolation: Interpolation
    energies: Tuple[ContinuumEnergy, ...]


@dataclass(frozen=True, eq=False)
class DiscreteTwoBodyEnergy:
    """One incident energy of ``G4ParticleHPDiscreteTwoBody``.

    ``representation`` 0 Legendre (``nCoefficients`` values ``a_1 … a_NL``),
    >0 a table of ``nCoefficients`` ``(μ, p)`` pairs flattened in ``values``.
    """

    energy: float
    representation: int
    nCoefficients: int
    values: np.ndarray


@dataclass(frozen=True, eq=False)
class DiscreteTwoBodyBody:
    """``distLaw=2``: ENDF MF6 LAW=2. ``nEnergy`` as written; ``max(nEnergy, 1)`` read."""

    nEnergy: int
    interpolation: Interpolation
    energies: Tuple[DiscreteTwoBodyEnergy, ...]


@dataclass(frozen=True, eq=False)
class NBodyBody:
    """``distLaw=6``, ``G4ParticleHPNBodyPhaseSpace``: ENDF MF6 LAW=6 (APSX, NPSX)."""

    totalMass: float
    totalCount: int


@dataclass(frozen=True, eq=False)
class LabAngleEnergyAngle:
    """One ``μ`` of a lab angle-energy record: ``μ`` and its ``Tab1 p(E'|μ)``."""

    mu: float
    spectrum: Tab1


@dataclass(frozen=True, eq=False)
class LabAngleEnergyEnergy:
    """One incident energy: ``nCosTh`` as written, the ``μ`` interpolation, the ``μ`` records."""

    energy: float
    nCosTh: int
    interpolation: Interpolation
    angles: Tuple[LabAngleEnergyAngle, ...]


@dataclass(frozen=True, eq=False)
class LabAngleEnergyBody:
    """``distLaw=7``, ``G4ParticleHPLabAngularEnergy``: ENDF MF6 LAW=7."""

    interpolation: Interpolation
    energies: Tuple[LabAngleEnergyEnergy, ...]


@dataclass(frozen=True, eq=False)
class ProductRecord:
    """One product of ``G4ParticleHPProduct::Init``.

    ``massCode`` is ZAP (read as a double), ``mass`` AWP, ``isomerFlag`` LIP,
    ``distLaw`` LAW, the two Q values in eV, the yield ``Tab1`` and the law's
    ``body`` (``None`` for laws 0, 3 and 4, which read nothing more).
    """

    massCode: float
    mass: float
    isomerFlag: int
    distLaw: int
    groundStateQ: float
    actualStateQ: float
    yield_: Tab1
    body: object = None


@dataclass(frozen=True, eq=False)
class EnergyAngleBody:
    """``dataType=6``, ``G4ParticleHPEnAngCorrelation::Init``: ``targetMass frameFlag nProducts``.

    ``nProducts`` as written; ``max(nProducts, 1)`` products are read.
    """

    targetMass: float
    frameFlag: int
    nProducts: int
    products: Tuple[ProductRecord, ...]


# ------------------------------------------------------------------ photons

@dataclass(frozen=True, eq=False)
class PhotonLine:
    """``InitMean`` repFlag=1: ``disType energy`` and the multiplicity ``Tab1``."""

    disType: int
    energy: float
    yield_: Tab1


@dataclass(frozen=True, eq=False)
class PhotonMultiplicityBody:
    """``dataType=12`` with ``repFlag=1`` (ENDF MF12 LO=1). ``nDiscrete`` as written."""

    repFlag: int
    targetMass: float
    nDiscrete: int
    lines: Tuple[PhotonLine, ...]


@dataclass(frozen=True, eq=False)
class PhotonTransition:
    """One ``(level energy, transition probability[, photon fraction])`` of a cascade."""

    levelEnergy: float
    probability: float
    photonFraction: Optional[float] = None


@dataclass(frozen=True, eq=False)
class PhotonCascadeBody:
    """``dataType=12`` with ``repFlag=2`` (ENDF MF12 LO=2).

    The consumer reads the internal-conversion flag **twice**, before and
    after the base energy, and keeps the second; both are kept here.
    """

    repFlag: int
    targetMass: float
    conversionFlag: int
    baseEnergy: float
    conversionFlag2: int
    nGammaEnergies: int
    transitions: Tuple[PhotonTransition, ...]


@dataclass(frozen=True, eq=False)
class PhotonPartial:
    """One discrete photon of ``InitPartials``: ``Eγ ES LP LF`` and its σ ``Tab1``."""

    energy: float
    shell: float
    isPrimary: int
    disType: int
    sigma: Tab1


@dataclass(frozen=True, eq=False)
class PhotonPartialsBody:
    """``dataType=13`` (ENDF MF13). ``total`` is present only when ``nDiscrete != 1``."""

    nDiscrete: int
    targetMass: float
    total: Optional[Tab1]
    partials: Tuple[PhotonPartial, ...]


@dataclass(frozen=True, eq=False)
class PhotonLegendre:
    """``G4ParticleHPLegendreTable::Init``: ``E nPoly`` and ``a_1 … a_nPoly``."""

    energy: float
    coefficients: np.ndarray


@dataclass(frozen=True, eq=False)
class PhotonTabulated:
    """``G4ParticleHPAngularP::Init``: ``E nProb``, a μ interpolation record, ``(μ, p)``."""

    energy: float
    interpolation: Interpolation
    mu: np.ndarray
    probability: np.ndarray


@dataclass(frozen=True, eq=False)
class PhotonAngularLine:
    """One anisotropic photon: ``Eγ ES nNeu``, then (Legendre) an interpolation record."""

    energy: float
    shell: float
    nNeu: int
    interpolation: Optional[Interpolation]
    records: Tuple = ()


@dataclass(frozen=True, eq=False)
class PhotonAngularBody:
    """``dataType=14`` (ENDF MF14). ``isoFlag=1`` reads nothing else."""

    isoFlag: int
    tabulationType: Optional[int] = None
    nDiscrete2: Optional[int] = None
    nIso: Optional[int] = None
    isotropic: Tuple[Tuple[float, float], ...] = ()
    lines: Tuple[PhotonAngularLine, ...] = ()


@dataclass(frozen=True, eq=False)
class PhotonSpectrumAt:
    """One incident energy of a photon spectrum: ``E`` and its ``(E', p)`` ``Tab1``."""

    energy: float
    spectrum: Tab1


@dataclass(frozen=True, eq=False)
class PhotonSpectrum:
    """One partial of ``InitEnergies``: ``dummy``, ``p(E)``, ``nen``, interpolation, spectra."""

    dummy: int
    probability: Tab1
    interpolation: Interpolation
    spectra: Tuple[PhotonSpectrumAt, ...]


@dataclass(frozen=True, eq=False)
class PhotonEnergyBody:
    """``dataType=15`` (ENDF MF15).

    ``needed`` records whether the consumer read anything: ``InitEnergies``
    reads ``nPartials`` and the spectra **only** if a discrete photon of the
    same reaction has ``disType == 1``, and nothing otherwise.
    """

    needed: bool
    spectra: Tuple[PhotonSpectrum, ...] = ()


# ------------------------------------------------------------------ the file

@dataclass(frozen=True, eq=False)
class Section:
    """One ``infoType dataType [sfType dummy]`` section and its body.

    ``sfType`` and ``dummy`` are ``None`` in a base-FS file, where the
    header has neither.
    """

    infoType: int
    dataType: int
    sfType: Optional[int]
    dummy: Optional[int]
    body: object


@dataclass(frozen=True, eq=False)
class InelasticFSRecord:
    """``Inelastic/Fxx/<name>``, as the channel's ``Init`` reads it.

    ``channel`` is the directory (``"F01"``); ``composite`` whether it is read
    by ``G4ParticleHPInelasticCompFS``. A base-FS file carries ``Qvalue`` and
    ``dummy`` once, after the first section's ``dataType``.
    """

    path: Optional[Path]
    header: Optional[Tuple[str, str]]
    channel: str
    composite: bool
    sections: Tuple[Section, ...]
    Qvalue: Optional[float] = None
    Qdummy: Optional[int] = None

    @property
    def reactionMTs(self) -> Tuple[int, ...]:
        """The MTs this file states, in order of first appearance."""
        if not self.composite:
            return (CHANNEL_MT[self.channel],)
        seen = []
        for s in self.sections:
            if s.sfType not in seen:
                seen.append(s.sfType)
        return tuple(seen)


@dataclass(frozen=True, eq=False)
class GammasRecord:
    """``Inelastic/Gammas/z<Z>.a<A>``: the residual nucleus's level scheme.

    ``levels``, ``gammas`` and ``probabilities`` are the three columns, in
    file order (the consumer groups consecutive rows of one level).
    """

    path: Optional[Path]
    Z: int
    A: int
    levels: np.ndarray
    gammas: np.ndarray
    probabilities: np.ndarray

    def __len__(self) -> int:
        return len(self.levels)
