"""Processed Monte Carlo forms: what a transport library samples from.

GNDS §9.4.3's ``MonteCarlo_cdf`` style holds forms a Monte Carlo code samples
directly. Most have GNDS nodes -- ``xs_pdf_cdf1d``
(:class:`~kika.nuclear_data.model.functions.Xs_pdf_cdf1d`), ``energyAngularMC``
(:class:`EnergyAngularMC`), the Monte Carlo ``KalbachMann`` -- and two of the
laws ACE and OpenMC's HDF5 use have none, because in an evaluation they are not
data but consequences of kinematics:

- :class:`LevelEnergy`, ACE law 3 (OpenMC ``level``): the outgoing energy of a
  two-body level reaction, ``E' = massRatio * (E - threshold)``. An evaluation
  states Q and the masses; ACER computes the two constants from them, and the
  processed file stores the constants, not the Q.
- :class:`DiscretePhotonEnergy`, ACE law 2 (OpenMC ``discrete_photon``): a line,
  or a primary photon whose energy follows the incident one. GNDS states these
  as ``discreteGamma`` / ``primaryGamma`` with an incident-energy domain the
  processed form does not carry, and ACE's ``LP`` flag (0 or 1 for a line, 2 for
  a primary) has no GNDS spelling at all.

They are kept as their own classes rather than forced into the GNDS nodes they
resemble so that a read of a processed file holds exactly what it said, and the
writer emits exactly that back (roadmap H2, gate 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .enums import Frame
from .functions import Function2d, XYs2d, XYs3d

__all__ = ["LevelEnergy", "DiscretePhotonEnergy", "EnergyAngularMC", "DistributionMixture",
           "ProbabilityTables", "ProbabilityTableStatistics", "PROBABILITY_TABLE_ROWS", "CoherentElasticXS",
           "IncoherentElasticXS", "SumOfFunctions", "CoherentElasticMC", "IncoherentElasticMC",
           "IncoherentElasticDiscreteMC", "IncoherentInelasticDiscreteMC", "MixedElasticMC"]


@dataclass
class LevelEnergy:
    """ACE law 3 / OpenMC ``level``: ``E' = massRatio * (E - threshold)``.

    For a neutron leaving a level of a nucleus of mass ratio ``A`` in the
    centre-of-mass frame, ``threshold = (A + 1)/A * |Q|`` and
    ``massRatio = (A/(A + 1))**2`` (``energy_distribution.py:870-890``).
    """

    threshold: float
    massRatio: float

    def outgoingEnergy(self, energy):
        return self.massRatio * (np.asarray(energy, dtype=float) - self.threshold)


@dataclass
class DiscretePhotonEnergy:
    """ACE law 2 / OpenMC ``discrete_photon``.

    ``primaryFlag`` is ACE's ``LP``: 0 or 1, a line at ``energy``; 2, a primary
    photon at ``energy + A/(A+1) * E`` with ``A`` the ``atomicWeightRatio``
    (``energy_distribution.py:800-860``).
    """

    primaryFlag: int
    energy: float
    atomicWeightRatio: float

    def outgoingEnergy(self, incident):
        incident = np.asarray(incident, dtype=float)
        if self.primaryFlag == 2:
            A = self.atomicWeightRatio
            return self.energy + A / (A + 1.0) * incident
        return np.full(incident.shape, self.energy) if incident.ndim else self.energy


@dataclass
class EnergyAngularMC:
    """GNDS ``energyAngularMC``: P(E'|E) and P(mu|E,E'), each with its cumulative.

    The sampled shape of a correlated energy-angle distribution: draw E' from
    ``energy`` (an :class:`~kika.nuclear_data.model.functions.XYs2d` -- or a
    ``regions2d`` -- of ``xs_pdf_cdf1d`` over the incident energy), then mu from
    the ``energyAngular`` table at that (E, E') (an
    :class:`~kika.nuclear_data.model.functions.XYs3d` whose inner functions are
    ``xs_pdf_cdf1d`` in mu). OpenMC's ``correlated`` layout, and the
    ``incoherent_inelastic`` of a thermal file, are this node.
    """

    energy: Optional[Function2d] = None
    energyAngular: Optional[XYs3d] = None
    label: Optional[str] = None
    productFrame: Frame = Frame.lab

    def __post_init__(self) -> None:
        self.productFrame = Frame(self.productFrame)

    @property
    def isComplete(self) -> bool:
        return self.energy is not None and self.energyAngular is not None


@dataclass
class DistributionMixture:
    """Several distributions of one product, one drawn per collision by ``p_k(E)``.

    ACE chains a product's energy laws with a probability each (the ``LNW``
    list) and OpenMC keeps them as ``distribution_0..k`` with an
    ``applicability`` function per distribution, sampling distribution ``k``
    with probability ``p_k(E)`` (``reaction_product.cpp``). GNDS states a
    weighted sum only for energy spectra (``weightedFunctionals``); this is the
    same statement one level up, for whole angle-energy forms. Each
    :class:`~kika.nuclear_data.model.energy_spectra.Weighted` holds ``p_k`` as
    ``weight`` and the form as ``functional``.

    OpenMC writes an ``applicability`` even for a product with **one**
    distribution, and the official libraries carry thousands of them; a mixture
    of one keeps that dataset through a read and a write.
    """

    weighted: List[object] = field(default_factory=list)
    label: Optional[str] = None

    def __len__(self) -> int:
        return len(self.weighted)


#: Rows of OpenMC's ``urr/<T>/table`` (``urr.py:17-24``).
PROBABILITY_TABLE_ROWS = ("cdf", "total", "elastic", "fission", "capture", "heating")


@dataclass
class ProbabilityTableStatistics:
    """The statistical error of a :class:`ProbabilityTables` (kika's PURR).

    Over the N energies and M bands: the ``ladders`` drawn at each energy,
    and delete-one-ladder jackknife standard errors -- the ladder is the
    statistical unit -- of each band's probability (``probabilitySigma``,
    (N, M)), of its total/elastic/fission/capture factors (``factorSigma``,
    (N, 4, M)) and of its share ``p · factor`` of the dilute average
    (``contributionSigma``, FRENDY's per-band statistic made relative). The
    Bondarenko factors ``f(σ0)`` of the four rows at the backgrounds
    ``sigma0`` (b), from the bands' cross sections, and their errors:
    ``shielding``/``shieldingSigma``, (N, 4, K). With an ``errorTarget``,
    ``converged`` says per energy whether it was reached before the cap.
    Band errors condition on the realized probability-bin boundaries. They
    do not include variation of those boundaries between independent pilots.

    """

    ladders: np.ndarray
    sigma0: np.ndarray
    probabilitySigma: np.ndarray
    factorSigma: np.ndarray
    contributionSigma: np.ndarray
    shielding: np.ndarray
    shieldingSigma: np.ndarray
    errorTarget: Optional[float] = None
    converged: Optional[np.ndarray] = None
    shieldingBasis: str = "ladder_cross_sections"
    #: Absolute per-band cross-section standard errors in barns, (N, 4, M).
    #: With transportXS, includes smooth partial scaling and reconstructed
    #: total; otherwise uses the sampled ladder cross sections. Unlike
    #: factorSigma this retains the uncertainty of the dilute normalization.
    crossSectionSigma: Optional[np.ndarray] = None

    def worst(self) -> dict:
        """The largest errors over energies and bands, per row: the band share
        and the relative error of ``f`` at each ``sigma0``. A row without an
        average (its factors exactly 1, without error) reports 0."""
        rows = PROBABILITY_TABLE_ROWS[1:5]
        f = np.where(self.shielding != 0, self.shielding, 1.0)
        rel = self.shieldingSigma / np.abs(f)
        out = {}
        for r, row in enumerate(rows):
            live = bool(np.any(self.factorSigma[:, r] > 0))
            out[row] = dict(contribution=float(np.max(self.contributionSigma[:, r])) if live else 0.0,
                            shielding={float(s0): float(np.max(rel[:, r, k])) if live else 0.0
                                       for k, s0 in enumerate(self.sigma0)})
        return out


@dataclass
class ProbabilityTables:
    """Unresolved-resonance probability tables at one temperature (PURR's output).

    ``table`` is ``(N, 6, M)``: for each of the N incident ``energies``, the M
    bands' cumulative probability and their total, elastic, fission, capture
    and heating values (:data:`PROBABILITY_TABLE_ROWS`). With
    ``multiplySmooth`` the values are factors on the smooth cross sections;
    otherwise they are cross sections in barns. ``inelastic`` and
    ``absorption`` name the MT whose smooth cross section stands for the
    competition (``<= 0``: none); ``interpolation`` is 2 (lin-lin) or 5
    (log-log) between incident energies (``urr.py:132-162``).

    OpenMC's C++ uses the cdf, elastic, fission and capture rows and the
    reaction ``inelastic`` names; the total and heating rows are stored and
    not used in transport (``nuclide.cpp:963-987``).
    """

    energies: np.ndarray
    table: np.ndarray
    interpolation: int = 2
    inelastic: int = 0
    absorption: int = 0
    multiplySmooth: bool = True
    #: The tables' statistical error when kika computed them; not part of the
    #: tables (two tables are equal whatever their statistics).
    statistics: Optional[ProbabilityTableStatistics] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        self.energies = np.asarray(self.energies, dtype=float)
        self.table = np.asarray(self.table, dtype=float)
        if self.table.ndim != 3 or self.table.shape[:2] != (len(self.energies), 6):
            raise ValueError(f"a probability table is (N, 6, M) over N energies; got "
                             f"{self.table.shape} for {len(self.energies)} energies")

    @property
    def bands(self) -> int:
        return int(self.table.shape[2])

    def row(self, name: str) -> np.ndarray:
        """``(N, M)`` values of one row, by name."""
        return self.table[:, PROBABILITY_TABLE_ROWS.index(name), :]


# ----------------------------------------------------------------------
# Thermal scattering, processed (one temperature per form)
# ----------------------------------------------------------------------
# An evaluated TSL is a law: S(alpha, beta, T), Bragg edges with S(E, T), a
# Debye-Waller integral W'(T) (:mod:`~kika.nuclear_data.model.thermal_scattering`).
# A processed library states what a transport code samples at each
# temperature, and these are those forms -- OpenMC's ``data_thermal`` objects
# (``thermal.py``, ``thermal_angle_energy.py``), each filed under the
# ``heated_<T>`` label of its temperature.


@dataclass
class CoherentElasticXS:
    """σ_coh(E) = s_i / E between Bragg edges, at one temperature (OpenMC ``CoherentElastic``)."""

    braggEdges: np.ndarray
    structureFactors: np.ndarray

    def __post_init__(self) -> None:
        self.braggEdges = np.asarray(self.braggEdges, dtype=float)
        self.structureFactors = np.asarray(self.structureFactors, dtype=float)

    def evaluate(self, energies):
        e = np.asarray(energies, dtype=float)
        index = np.searchsorted(self.braggEdges, e, side="right") - 1
        held = np.where(index >= 0, self.structureFactors[np.clip(index, 0, None)], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(e > 0, held / np.where(e > 0, e, 1.0), 0.0)


@dataclass
class IncoherentElasticXS:
    """σ_inc(E) = σ_b/2 (1 - e^(-4EW'))/(2EW') at one temperature (OpenMC ``IncoherentElastic``)."""

    boundCrossSection: float
    debyeWaller: float

    def evaluate(self, energies):
        e = np.asarray(energies, dtype=float)
        x = 2.0 * e * self.debyeWaller
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(x > 0, -np.expm1(-2.0 * x) / np.where(x > 0, x, 1.0), 2.0)
        return 0.5 * self.boundCrossSection * ratio


@dataclass
class SumOfFunctions:
    """A cross section stated as a sum (OpenMC ``Sum``): a mixed elastic's coherent + incoherent."""

    functions: List[object] = field(default_factory=list)

    def evaluate(self, energies):
        return sum(np.asarray(f.evaluate(energies), dtype=float) for f in self.functions)


@dataclass
class CoherentElasticMC:
    """Outgoing angle of Bragg scattering: from the edges of the reaction's cross section."""


@dataclass
class IncoherentElasticMC:
    """Outgoing angle of incoherent elastic scattering, from the Debye-Waller integral (1/eV)."""

    debyeWaller: float


@dataclass
class IncoherentElasticDiscreteMC:
    """Equiprobable outgoing cosines per incident energy, ``muOut`` (NE, n)."""

    muOut: np.ndarray


@dataclass
class IncoherentInelasticDiscreteMC:
    """Equiprobable outgoing energies and cosines per incident energy (ACE ``nxs[7] != 2``).

    ``energyOut`` (NE, n_E'), ``muOut`` (NE, n_E', n_mu); ``skewed`` is ACE's
    flag for the skewed outgoing-energy weighting.
    """

    energyOut: np.ndarray
    muOut: np.ndarray
    skewed: bool = False


@dataclass
class MixedElasticMC:
    """A mixed elastic's two angular laws, coherent and incoherent."""

    coherent: object = None
    incoherent: object = None
