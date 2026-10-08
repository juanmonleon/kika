"""GNDS-2.1 §18.3: the parametrised energy spectra (``gnds.xsd:1697-1770``).

Six of the eleven forms ``uncorrelated/energy`` admits are not tables but
formulae with named parameters -- ENDF-6 §5.1.1's LF=5, 7, 9, 11 and 12, and
the weighted sum of several (NK > 1):

==========================  ====  ==============================================
GNDS node                   LF    parameters (each a function of E, in eV)
==========================  ====  ==============================================
generalEvaporation          5     ``U``, ``theta``, ``g`` (``g`` of E'/theta)
simpleMaxwellianFission     7     ``U``, ``theta``
evaporation                 9     ``U``, ``theta``
Watt                        11    ``U``, ``a``, ``b`` (``b`` in 1/eV)
MadlandNix                  12    ``EFL``, ``EFH``, ``T_M`` -- and **no** ``U``
weightedFunctionals         NK>1  ``weighted`` = (weight ``p_k(E)``, functional)
==========================  ====  ==============================================

The names, the fields and the units are FUDGE's
(``fudge/productData/distributions/energy.py``), with the same simplification
:class:`~kika.nuclear_data.model.output_channel.FissionEnergyRelease` makes:
FUDGE wraps each parameter in a node (``theta``, ``g``, ...) whose one child is
the table, and here the attribute *is* the table.

**None of these is a** :class:`~kika.nuclear_data.model.functions.Function2d`,
**on purpose.** The MF35 sampler picks its candidates with
``isinstance(energy, Function2d)`` (``kika/sampling/perturbation_set.py``), and
a node that inherited from it would be perturbed as if it were a table it is
not. A parametrised spectrum has no covariance in ENDF -- MF35 is stated for
tables -- and perturbing it would need a covariance of its *parameters*, which
no library carries. PD-3 as recommended on 2026-10-08, not yet confirmed by
Juan: refuse it by name. What these nodes offer instead is :meth:`toPointwise`:
the spectrum tabulated, as an ``XYs2d`` the caller asked for, never filed under
the evaluated label.

**The arithmetic is** :mod:`kika.algebra.spectra`'s, the same functions the
ENDF reader evaluates with, so the two cannot disagree. Every form is
normalised over its own support: ``[0, E - U]`` for the four with a ``U``,
``[0, inf)`` for Madland-Nix, which the ``E - U`` rule would truncate to
nothing at thermal incidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from kika.algebra import integral, interval_laws, spectra

from .axes import energyAxes
from .functions import Function1d, Regions1d, XYs1d, XYs2d
from .quantities import PhysicalQuantity

__all__ = [
    "Evaporation", "GeneralEvaporation", "SimpleMaxwellianFission", "Watt",
    "MadlandNix", "Weighted", "WeightedFunctionals", "ANALYTIC_SPECTRA",
]

#: Points in a default outgoing grid. A rendering choice only.
DEFAULT_POINTS = 400
_DECADES = 9.0


def _at(function: Function1d, energy: float) -> float:
    """A parameter table at one incident energy, its end values held outside."""
    return float(np.asarray(function.evaluate(float(energy), outOfRange="hold")))


def _abscissae(*functions: Function1d) -> np.ndarray:
    """The union of the parameter tables' incident energies -- their own grid."""
    points = []
    for function in functions:
        if function is None:
            continue
        xs = function.toEndfRegions()[0] if isinstance(function, Regions1d) else function.xs
        points.extend(np.asarray(xs, dtype=float).tolist())
    return np.unique(np.asarray(points, dtype=float))


def _energyValue(quantity: Optional[PhysicalQuantity]) -> float:
    """A ``U``/``EFL``/``EFH`` in eV. The model keeps the unit beside the number."""
    if quantity is None:
        return 0.0
    if getattr(quantity, "unit", "eV") in ("eV", "", None):
        return float(quantity.value)
    return float(quantity.convertedTo("eV").value)


class _Spectrum:
    """What every parametrised form answers: a normalised P(E'|E), and a table of it."""

    def upperBound(self, energy: float) -> float:
        """The largest outgoing energy the law admits: ``E - U``."""
        return float(energy) - _energyValue(self.U)

    def _shape(self, energy: float, eOut: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _norm(self, energy: float) -> float:
        raise NotImplementedError

    def evaluate(self, energy: float, energyOut) -> np.ndarray:
        """P(E'|E) at outgoing energies *energyOut*; zero outside ``[0, upperBound]``."""
        points = np.asarray(energyOut, dtype=float)
        out = np.zeros(points.shape)
        hi = self.upperBound(energy)
        norm = self._norm(energy)
        if hi <= 0.0 or not np.isfinite(norm) or norm <= 0.0:
            return out
        inside = (points >= 0.0) & (points <= hi)
        if np.any(inside):
            out[inside] = self._shape(energy, points[inside]) / norm
        return out

    def outgoingGrid(self, energy: float, points: int = DEFAULT_POINTS) -> np.ndarray:
        """A log grid over the support, 0 kept. Only a rendering choice."""
        hi = self.upperBound(energy)
        if not np.isfinite(hi):
            hi = self._displayTop(energy)
        if hi <= 0.0:
            return np.zeros(0)
        return np.concatenate(([0.0], np.geomspace(hi * 10.0 ** -_DECADES, hi,
                                                   max(points - 1, 2))))

    def _displayTop(self, energy: float) -> float:  # pragma: no cover - finite laws
        raise NotImplementedError

    def incidentEnergies(self) -> np.ndarray:
        """The incident energies the parameters are tabulated at."""
        return _abscissae(*self._parameters())

    def _parameters(self):
        raise NotImplementedError

    def toPointwise(self, incidentEnergies: Optional[Sequence[float]] = None,
                    points: int = DEFAULT_POINTS,
                    outgoing: Optional[Sequence[float]] = None) -> XYs2d:
        """The spectrum tabulated: an ``XYs2d`` of lin-lin ``XYs1d`` in E'.

        At the parameters' own incident energies unless *incidentEnergies* is
        given; on :meth:`outgoingGrid` unless *outgoing* is. A **derived
        object**, not a style: the caller holds it and nothing files it under
        the evaluated label, which is what keeps PD-3's refusal meaningful.
        Lin-lin between the outgoing points is an approximation of the law and
        the grid decides how good -- ``outgoing`` is there for a caller that
        needs a specific one.
        """
        energies = (self.incidentEnergies() if incidentEnergies is None
                    else np.asarray(incidentEnergies, dtype=float))
        axes = energyAxes()
        table = XYs2d(axes=axes)
        for energy in energies:
            grid = (self.outgoingGrid(float(energy), points) if outgoing is None
                    else np.asarray(outgoing, dtype=float))
            table.function1ds.append(XYs1d(
                xs=grid, ys=self.evaluate(float(energy), grid),
                outerDomainValue=float(energy), axes=axes,
            ))
        return table


@dataclass
class SimpleMaxwellianFission(_Spectrum):
    """LF=7: ``sqrt(E') exp(-E'/theta(E))`` on ``[0, E - U]`` (``gnds.xsd:1748``)."""

    U: Optional[PhysicalQuantity] = None
    theta: Optional[Function1d] = None

    def _parameters(self):
        return (self.theta,)

    def _shape(self, energy, eOut):
        return spectra.maxwellian(eOut, _at(self.theta, energy))

    def _norm(self, energy):
        return spectra.maxwellian_integral(self.upperBound(energy), _at(self.theta, energy))


@dataclass
class Evaporation(_Spectrum):
    """LF=9: ``E' exp(-E'/theta(E))`` on ``[0, E - U]`` (``gnds.xsd:1714``)."""

    U: Optional[PhysicalQuantity] = None
    theta: Optional[Function1d] = None

    def _parameters(self):
        return (self.theta,)

    def _shape(self, energy, eOut):
        return spectra.evaporation(eOut, _at(self.theta, energy))

    def _norm(self, energy):
        return spectra.evaporation_integral(self.upperBound(energy), _at(self.theta, energy))


@dataclass
class Watt(_Spectrum):
    """LF=11: ``exp(-E'/a(E)) sinh(sqrt(b(E) E'))`` on ``[0, E - U]`` (``gnds.xsd:1731``)."""

    U: Optional[PhysicalQuantity] = None
    a: Optional[Function1d] = None
    b: Optional[Function1d] = None

    def _parameters(self):
        return (self.a, self.b)

    def _shape(self, energy, eOut):
        return spectra.watt(eOut, _at(self.a, energy), _at(self.b, energy))

    def _norm(self, energy):
        return spectra.watt_integral(self.upperBound(energy), _at(self.a, energy),
                                     _at(self.b, energy))


@dataclass
class GeneralEvaporation(_Spectrum):
    """LF=5: a table ``g(x)``, ``x = E'/theta(E)`` (``gnds.xsd:1740``).

    ``f = g(E'/theta)/theta`` normalised by its own integral, the reading that
    keeps ``int f dE' = 1`` with an energy-dependent theta -- see
    :mod:`kika.endf.classes.mf5.analytic`, which reads it the same way.
    """

    U: Optional[PhysicalQuantity] = None
    theta: Optional[Function1d] = None
    g: Optional[Function1d] = None

    def _parameters(self):
        return (self.theta,)

    def _g(self):
        g = self.g if isinstance(self.g, Regions1d) else Regions1d(function1ds=[self.g])
        xs, ys, pairs = g.toEndfRegions()
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), interval_laws(len(xs), pairs)

    def _shape(self, energy, eOut):
        theta = _at(self.theta, energy)
        if theta <= 0.0:
            return np.zeros(np.shape(eOut))
        return np.asarray(self.g.evaluate(np.asarray(eOut) / theta), dtype=float) / theta

    def _norm(self, energy):
        theta = _at(self.theta, energy)
        hi = self.upperBound(energy)
        if theta <= 0.0 or hi <= 0.0:
            return 0.0
        xs, ys, laws = self._g()
        return integral(xs, ys, laws, xs[0], min(hi / theta, xs[-1]))

    def outgoingGrid(self, energy: float, points: int = DEFAULT_POINTS) -> np.ndarray:
        """``theta * x`` over g's own abscissae -- the law's exact break points."""
        theta = _at(self.theta, energy)
        hi = self.upperBound(energy)
        if theta <= 0.0 or hi <= 0.0:
            return np.zeros(0)
        full = self._g()[0] * theta
        grid = full[full <= hi]
        if full.size and full[-1] > hi:
            grid = np.concatenate((grid, [hi]))
        return grid


@dataclass
class MadlandNix(_Spectrum):
    """LF=12: ``EFL``, ``EFH`` and ``T_M(E)`` (``gnds.xsd:1722``). No ``U``.

    Normalised over ``[0, inf)`` analytically; see :func:`kika.algebra.spectra.madland_nix`.
    """

    EFL: Optional[PhysicalQuantity] = None
    EFH: Optional[PhysicalQuantity] = None
    T_M: Optional[Function1d] = None

    U = None  # the law has none; ``upperBound`` is infinite

    def _parameters(self):
        return (self.T_M,)

    def upperBound(self, energy: float) -> float:
        return float("inf")

    def _shape(self, energy, eOut):
        return spectra.madland_nix(eOut, _energyValue(self.EFL), _energyValue(self.EFH),
                                   _at(self.T_M, energy))

    def _norm(self, energy):
        return 1.0 if _at(self.T_M, energy) > 0.0 else 0.0

    def _displayTop(self, energy):
        return spectra.madland_nix_upper(_energyValue(self.EFH), _at(self.T_M, energy))

    def averageEnergy(self, energy: float) -> float:
        """``<E'>`` in closed form, ``(EFL + EFH)/2 + 4/3 T_M(E)``."""
        return spectra.madland_nix_mean(_energyValue(self.EFL), _energyValue(self.EFH),
                                        _at(self.T_M, energy))


@dataclass
class Weighted:
    """One term of a ``weightedFunctionals``: ``p_k(E)`` and its functional."""

    weight: Optional[Function1d] = None
    functional: Optional[object] = None


@dataclass
class WeightedFunctionals(_Spectrum):
    """NK > 1: ``P(E'|E) = sum_k p_k(E) f_k(E'|E)`` (``gnds.xsd:1756``).

    **Not for MT455.** There ENDF's NK subsections are one spectrum per
    delayed-neutron precursor family, not a sum: each goes on its family's
    product (§18.4), with ``p_k`` scaling the family's multiplicity -- FUDGE's
    reading too. The ENDF adapter routes MT455 there and never builds this.
    """

    weighted: List[Weighted] = field(default_factory=list)

    U = None

    def _parameters(self):
        return tuple(w.weight for w in self.weighted)

    def incidentEnergies(self) -> np.ndarray:
        own = [_abscissae(*w.functional._parameters()) for w in self.weighted
               if hasattr(w.functional, "_parameters")]
        return np.unique(np.concatenate([_abscissae(*self._parameters()), *own]))

    def upperBound(self, energy: float) -> float:
        return max(term.functional.upperBound(energy) for term in self.weighted)

    def evaluate(self, energy: float, energyOut) -> np.ndarray:
        points = np.asarray(energyOut, dtype=float)
        total = np.zeros(points.shape)
        for term in self.weighted:
            p = _at(term.weight, energy)
            if p:
                total = total + p * term.functional.evaluate(energy, points)
        return total

    def outgoingGrid(self, energy: float, points: int = DEFAULT_POINTS) -> np.ndarray:
        grids = [term.functional.outgoingGrid(energy, points) for term in self.weighted]
        return np.unique(np.concatenate(grids)) if grids else np.zeros(0)


#: The six §18.3 parametrised forms, for ``isinstance`` checks.
ANALYTIC_SPECTRA = (Evaporation, GeneralEvaporation, SimpleMaxwellianFission,
                    Watt, MadlandNix, WeightedFunctionals)
