"""§18.4's ``productYields``: fission product yields (ENDF MF8/MT454 and MT459).

The names are GNDS's (``gnds.xsd`` ``FissionProductYieldType`` and below):

``productYield(label) > nuclides?, elapsedTimes > elapsedTime(label) > time,
(yields | incidentEnergies > incidentEnergy(label) > energy, yields)``,
``yields > nuclides, values, uncertainty?``.

ENDF's independent yields (MT454) are FUDGE's ``elapsedTime label="initial"``
at time 0, the cumulative ones (MT459) ``label="unspecified"`` with the string
time ``unspecified``. A neutron-induced evaluation (NSUB=11) lists one
``incidentEnergy`` per ENDF energy; a spontaneous-fission one (NSUB=5, or the
MT454/459 of a decay tape) puts the ``yields`` straight under the
``elapsedTime``. ENDF states only each yield's standard uncertainty, which GNDS
writes as the diagonal of a covariance, so :attr:`Yields.uncertainty` is that
diagonal: the **variances**.

``nuclides`` on a :class:`Yields` is ``None`` when it is the product yield's own
list -- GNDS's ``href`` to it, which FUDGE writes whenever every energy lists the
same products in the same order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from .quantities import PhysicalQuantity

__all__ = ["Yields", "IncidentEnergy", "ElapsedTime", "ProductYield",
           "INDEPENDENT", "CUMULATIVE"]

#: The ``elapsedTime`` labels FUDGE gives ENDF's MT454 and MT459.
INDEPENDENT = "initial"
CUMULATIVE = "unspecified"


@dataclass
class Yields:
    """``yields``: one value per product, and the variance of each."""

    values: List[float] = field(default_factory=list)
    nuclides: Optional[List[str]] = None
    uncertainty: Optional[List[float]] = None


@dataclass
class IncidentEnergy:
    """``incidentEnergy``: the yields at one incident neutron energy."""

    label: str
    energy: PhysicalQuantity
    yields: Yields = field(default_factory=Yields)


@dataclass
class ElapsedTime:
    """``elapsedTime``: the yields at one time after fission.

    ``time`` is a quantity (0 s for the independent yields) or the string
    ``"unspecified"`` (the cumulative ones), §12's two spellings.
    """

    label: str
    time: Union[PhysicalQuantity, str]
    yields: Optional[Yields] = None
    incidentEnergies: List[IncidentEnergy] = field(default_factory=list)


@dataclass
class ProductYield:
    """``productYield``: the products and their yields at each elapsed time."""

    label: str
    nuclides: Optional[List[str]] = None
    elapsedTimes: List[ElapsedTime] = field(default_factory=list)

    def elapsedTime(self, label: str) -> Optional[ElapsedTime]:
        return next((e for e in self.elapsedTimes if e.label == label), None)

    def __repr__(self) -> str:
        n = len(self.nuclides) if self.nuclides is not None else "per-energy"
        return (f"ProductYield({self.label!r}, {n} nuclides, "
                f"{[e.label for e in self.elapsedTimes]})")
