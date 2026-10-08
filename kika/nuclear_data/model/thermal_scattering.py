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
from typing import ClassVar, List, Optional, Union

from .component import Component
from .functions import Gridded2d, Gridded3d, Regions1d, XYs1d
from .quantities import PhysicalQuantity

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


@dataclass
class IncoherentElastic:
    """``thermalNeutronScatteringLaw_incoherentElastic``: the bound atom and W'(T)."""

    boundAtomCrossSection: PhysicalQuantity
    DebyeWallerIntegral: Union[XYs1d, Regions1d]
    label: Optional[str] = None
    pid: str = "n"
    productFrame: str = "lab"
    gndsNodeName = "thermalNeutronScatteringLaw_incoherentElastic"


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
