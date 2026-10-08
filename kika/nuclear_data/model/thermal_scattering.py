"""GNDS-2.1: the thermal neutron scattering law (``gnds.xsd`` TNSL types).

ENDF-coverage roadmap E4. A TSL evaluation is a ``reactionSuite`` whose
``interaction`` is ``thermalNeutronScatteringLaw``. Each of its reactions holds
one of the three forms below in ``doubleDifferentialCrossSection``, and points
at that form from two places: its ``crossSection`` (a
:class:`~kika.nuclear_data.model.cross_section_forms.ThermalNeutronScatteringLaw1d`)
and its neutron's ``distribution`` (a :class:`ThermalNeutronScatteringLaw`).
The double-differential cross section *is* the data. The other two are links,
because neither σ(E) nor the angular distribution of a bound scatterer can be
stated apart from the law.

==================================================  =======  ==========================
GNDS node                                           ENDF     here
==================================================  =======  ==========================
thermalNeutronScatteringLaw_coherentElastic         7/2      :class:`CoherentElastic`
thermalNeutronScatteringLaw_incoherentElastic       7/2      :class:`IncoherentElastic`
thermalNeutronScatteringLaw_incoherentInelastic     7/4      :class:`IncoherentInelastic`
==================================================  =======  ==========================

**Names, fields and units are FUDGE's** where FUDGE follows the specification
(``fudge/reactionData/doubleDifferentialCrossSection/thermalNeutronScatteringLaw/``):

- ``S_table`` is a ``gridded2d`` over (``temperature`` K, ``energy_in`` eV)
  with dependent axis ``S_cumulative`` in ``eV*b``;
- the Debye-Waller integral is an ``XYs1d`` of ``temperature`` (K), in ``1/eV``;
- the kernel S(alpha, beta, T) is a ``gridded3d`` over (``temperature``,
  ``beta``, ``alpha``);
- masses are in amu and cross sections in b, **bound**.

Two deliberate departures from FUDGE, both declared:

- **No identity is guessed from a file name.** FUDGE infers the scattering
  atoms' pids from the tape's name (``HinH2O`` gives H x2 and O x1). The adapter
  matches them by mass against MF7/MT451 instead, and says when it cannot.
- **``symmetric`` is never recomputed on read.** FUDGE's reader turns a missing
  attribute into False, while its constructor computes the value. Here it is
  what the source said: LASYM on the way in from ENDF, the attribute from GNDS.

What ENDF states and GNDS does not (MF7/451's NAS, the B array's B(5), LLN, the
padding of the records) is not modelled. It travels in the reactions'
:class:`~kika.nuclear_data.model.provenance.EndfProvenance`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Sequence, Union

import numpy as np

from .component import Component
from .functions import Gridded2d, Gridded3d, Regions1d, XYs1d
from .quantities import PhysicalQuantity
from .enums import Interpolation
from .units import conversion_factor

#: Energies a caller may pass to the cross-section methods, in eV.
Energies = Union[float, Sequence[float], np.ndarray]

__all__ = [
    "TNSL_INTERACTION", "TNSL_PROCESSES", "DoubleDifferentialCrossSection",
    "ThermalNeutronScatteringLaw",
    "CoherentElastic", "IncoherentElastic", "IncoherentInelastic",
    "ScatteringAtom", "SelfScatteringKernel", "DistinctScatteringKernel",
    "SCTApproximation", "FreeGasApproximation", "GaussianApproximation",
    "TNSL_FORMS",
]

#: ``reactionSuite/@interaction`` for a TSL evaluation (``gnds.xsd`` InteractionEnum).
TNSL_INTERACTION = "thermalNeutronScatteringLaw"


@dataclass(repr=False)
class DoubleDifferentialCrossSection(Component):
    """``doubleDifferentialCrossSection``: the reaction's d²σ/dΩdE, by style label.

    ``Reaction.doubleDifferentialCrossSection`` was a declared slot that nothing
    filled. Now a TSL reaction fills it with one of :data:`TNSL_FORMS` under
    ``'eval'``.
    """

    gndsNodeName: ClassVar[str] = "doubleDifferentialCrossSection"

    def __repr__(self) -> str:
        return f"DoubleDifferentialCrossSection({ {k: type(v).__name__ for k, v in self.forms.items()} })"


@dataclass
class ThermalNeutronScatteringLaw:
    """§18 ``distribution`` child: a link to the reaction's double-differential form.

    The product-side twin of
    :class:`~kika.nuclear_data.model.cross_section_forms.ThermalNeutronScatteringLaw1d`.
    The outgoing neutron of a bound scatterer has no distribution of its own;
    it has the law.
    """

    href: Optional[str] = None
    label: Optional[str] = None
    gndsNodeName = "thermalNeutronScatteringLaw"


@dataclass
class CoherentElastic:
    """``thermalNeutronScatteringLaw_coherentElastic``: Bragg scattering.

    ``S_table`` is the cumulative structure factor S(E, T) on the Bragg edges,
    a staircase in E (the energy grid's interpolation is ``flat``). The
    schema's alternative ``BraggEdges`` has no ENDF counterpart and is not read
    by FUDGE either; it is not modelled.
    """

    S_table: Gridded2d
    label: Optional[str] = None
    pid: str = "n"
    productFrame: str = "lab"
    gndsNodeName = "thermalNeutronScatteringLaw_coherentElastic"

    @property
    def temperatures(self) -> np.ndarray:
        """The tabulated temperatures, in K."""
        grid = self.S_table.grids[0]
        return np.asarray(grid.values, dtype=float) * conversion_factor(grid.unit, "K")

    def braggEdges(self) -> np.ndarray:
        """The Bragg-edge energies, in eV."""
        grid = self.S_table.grids[1]
        return np.asarray(grid.values, dtype=float) * conversion_factor(grid.unit, "eV")

    def cumulativeS(self, temperature: float) -> np.ndarray:
        """S(E_i, T) on the Bragg edges, in eV*b, at a tabulated *temperature* (K)."""
        temperatures = self.temperatures
        matches = np.flatnonzero(temperatures == float(temperature))
        if matches.size == 0:
            raise KeyError(
                f"{temperature} K is not tabulated; have {[float(t) for t in temperatures]}"
            )
        factor = conversion_factor(self.S_table.dependentAxis.unit, "eV*b")
        return np.asarray(self.S_table.values[matches[0]], dtype=float) * factor

    def crossSection(self, energies: Energies, temperature: float) -> np.ndarray:
        """σ_coh(E, T) = S(E, T) / E in b, at incident *energies* in eV (ENDF-102 eq. 7.3).

        S is cumulative, so between edge *i* and edge *i+1* it is the value at
        edge *i*, and below the first edge σ is zero. The temperature must be
        tabulated: how two temperatures combine is the temperature grid's law,
        and applying it here would hand back an approximation as if it were data.
        Refused when the energy grid is not ``flat``, because S/E reads S as a
        staircase and on any other law it would be the wrong function.
        """
        law = self.S_table.grids[1].interpolation
        if law is not Interpolation.flat:
            raise ValueError(
                f"coherent elastic S(E) is interpolated {law.value}; sigma = S/E "
                "needs the flat (histogram) staircase every evaluation writes"
            )
        s = self.cumulativeS(temperature)
        edges = self.braggEdges()
        e = np.asarray(energies, dtype=float)
        index = np.searchsorted(edges, e, side="right") - 1
        held = np.where(index >= 0, s[np.clip(index, 0, None)], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(e > 0, held / np.where(e > 0, e, 1.0), 0.0)


@dataclass
class IncoherentElastic:
    """``thermalNeutronScatteringLaw_incoherentElastic``: the bound atom and W'(T)."""

    boundAtomCrossSection: PhysicalQuantity
    DebyeWallerIntegral: Union[XYs1d, Regions1d]
    label: Optional[str] = None
    pid: str = "n"
    productFrame: str = "lab"
    gndsNodeName = "thermalNeutronScatteringLaw_incoherentElastic"

    @property
    def temperatures(self) -> np.ndarray:
        """The temperatures W'(T) is tabulated at, in K."""
        w = self.DebyeWallerIntegral
        xs, _, _ = w.toEndfRegions()
        return np.asarray(xs, dtype=float) * conversion_factor(w.domainUnit or "K", "K")

    def debyeWaller(self, temperature: float) -> float:
        """W'(T) in 1/eV at *temperature* (K), under the function's own law.

        Interpolated: temperature *is* the abscissa here, so a value between
        two nodes is what the evaluator's law defines. Outside the table it is
        refused, because a held end value would invent a lattice.
        """
        w = self.DebyeWallerIntegral
        toKelvin = conversion_factor(w.domainUnit or "K", "K")
        low, high = w.domainMin * toKelvin, w.domainMax * toKelvin
        if not low <= temperature <= high:
            raise KeyError(f"{temperature} K is outside the W'(T) table [{low}, {high}] K")
        value = w.evaluate(float(temperature) / toKelvin)
        return float(value) * conversion_factor(w.rangeUnit or "1/eV", "1/eV")

    def crossSection(self, energies: Energies, temperature: float) -> np.ndarray:
        """σ_inc(E, T) in b at incident *energies* in eV (ENDF-102 eq. 7.5).

        σ = σ_b/2 · (1 − e^(−4EW′)) / (2EW′), with σ_b the bound cross section.
        It tends to σ_b as E → 0, written with ``expm1`` so that limit is not a
        cancellation of two numbers near one.
        """
        w = self.debyeWaller(temperature)
        sigmaBound = self.boundAtomCrossSection.convertedTo("b").value
        e = np.asarray(energies, dtype=float)
        x = 2.0 * e * w
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(x > 0, -np.expm1(-2.0 * x) / np.where(x > 0, x, 1.0), 2.0)
        return 0.5 * sigmaBound * ratio


@dataclass
class SCTApproximation:
    """The short-collision-time kernel (ENDF eq. 7.8), parametrised by ``T_effective``."""

    gndsNodeName = "SCTApproximation"


@dataclass
class FreeGasApproximation:
    """The free-gas kernel, for a secondary scatterer ENDF marks with B(6i+1) = 1."""

    gndsNodeName = "freeGasApproximation"


@dataclass
class GaussianApproximation:
    """The Gaussian-approximation kernel, built from a phonon spectrum.

    In the schema and in FUDGE. ENDF has no way to state it, so it can only
    arrive from GNDS.
    """

    phononSpectrum: Union[XYs1d, Regions1d]
    gndsNodeName = "GaussianApproximation"


@dataclass
class SelfScatteringKernel:
    """``selfScatteringKernel``: S(alpha, beta, T), tabulated or approximated.

    ``symmetric`` is None when the source did not say. An ENDF tape always
    says, through LASYM; ``symmetric = (LASYM == 0)``.
    """

    kernel: Union[Gridded3d, SCTApproximation, FreeGasApproximation, GaussianApproximation]
    symmetric: Optional[bool] = None
    gndsNodeName = "selfScatteringKernel"


@dataclass
class DistinctScatteringKernel:
    """``distinctScatteringKernel``: the distinct part of S. No ENDF counterpart."""

    kernel: Gridded3d
    gndsNodeName = "distinctScatteringKernel"


@dataclass
class ScatteringAtom:
    """``scatteringAtom``: one atom type of the scattering molecule.

    ``boundAtomCrossSection`` is the bound cross section. ENDF's B array
    carries the *free* one, σ_free x M, and the adapter converts with
    ``((A + 1) / A)**2``, as FUDGE does.
    """

    pid: str
    numberPerMolecule: int
    mass: PhysicalQuantity
    e_max: PhysicalQuantity
    boundAtomCrossSection: PhysicalQuantity
    selfScatteringKernel: SelfScatteringKernel
    primaryScatterer: bool = False
    e_critical: Optional[PhysicalQuantity] = None
    T_effective: Optional[Union[XYs1d, Regions1d]] = None
    #: ``boundAtomCrossSection`` per nuclide, keyed by pid. GNDS 2.1 and later.
    boundAtomCrossSectionByNuclide: dict = field(default_factory=dict)
    coherentAtomCrossSection: Optional[PhysicalQuantity] = None
    distinctScatteringKernel: Optional[DistinctScatteringKernel] = None
    gndsNodeName = "scatteringAtom"


@dataclass
class IncoherentInelastic:
    """``thermalNeutronScatteringLaw_incoherentInelastic``: the S(alpha, beta, T) law."""

    scatteringAtoms: List[ScatteringAtom]
    primaryScatterer: str
    label: Optional[str] = None
    pid: str = "n"
    productFrame: str = "lab"
    #: ENDF LAT = 1: alpha and beta are scaled by kT0 = 0.0253 eV rather than kT.
    calculatedAtThermal: bool = False
    incoherentApproximation: bool = True
    gndsNodeName = "thermalNeutronScatteringLaw_incoherentInelastic"

    @property
    def principal(self) -> ScatteringAtom:
        """The atom whose kernel is tabulated, the one ``primaryScatterer`` names."""
        for atom in self.scatteringAtoms:
            if atom.primaryScatterer:
                return atom
        raise LookupError("no scattering atom is marked primaryScatterer")


TNSL_FORMS = (CoherentElastic, IncoherentElastic, IncoherentInelastic)

#: ``outputChannel/@process`` per form, FUDGE's spelling.
TNSL_PROCESSES = {
    CoherentElastic: "thermalNeutronScatteringLaw coherent-elastic",
    IncoherentElastic: "thermalNeutronScatteringLaw incoherent-elastic",
    IncoherentInelastic: "thermalNeutronScatteringLaw incoherent-inelastic",
}
