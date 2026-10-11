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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .enums import Frame
from .functions import Function2d, XYs2d, XYs3d

__all__ = ["LevelEnergy", "DiscretePhotonEnergy", "EnergyAngularMC"]


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
