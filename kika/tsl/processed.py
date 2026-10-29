"""Processed thermal-scattering data, and how it enters kika's model.

ACE ``t`` tables, OpenMC's HDF5 ``thermal/`` files and G4NDL's
``ThermalScattering`` all hold the same thing: what NJOY's THERMR (and ACER)
made of an ENDF MF7 evaluation at a set of temperatures. Per temperature:

- **coherent elastic** as the Bragg edges and the cumulative S(E_i) on them.
  That is the evaluation's own table, so it becomes a model
  :class:`~kika.nuclear_data.model.CoherentElastic`, exact;
- **incoherent elastic** as σ(E) on a grid with equiprobable cosines (ACE,
  G4NDL), or as the bound σ and W'(T) themselves (OpenMC). The second becomes a
  model :class:`~kika.nuclear_data.model.IncoherentElastic`, the first a
  cross section;
- **incoherent inelastic** as σ(E) on a grid and a table of outgoing energies
  and cosines per incident energy. S(α,β) is gone: it was integrated away.

**What reaches the model.** A ``reactionSuite`` with the TNSL interaction, an
``evaluated`` style the file refers to and does not contain (as
:func:`kika.ace.model_adapter.aceStyles` declares it) and one ``heated`` style
per temperature. Each reaction carries its σ(E) under the heated labels; the
Bragg table and the exact incoherent parameters are double-differential forms
under the same labels.

**What does not.** The outgoing-neutron tables are sampling tables
(equiprobable cosines, discrete or skewed energy lines, pdf/cdf pairs). GNDS
has no node for an equiprobable cosine set, and turning one into a pdf would
invent the bin edges. They stay here, as :class:`OutgoingTable`, read on
demand, and the conversion report says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["BraggTable", "IncoherentElasticTable", "OutgoingTable", "ProcessedTemperature",
           "ProcessedTSL", "processed_suite", "heated_label"]

#: OpenMC and NJOY name the outgoing tables by these.
OUTGOING_KINDS = ("continuous", "discrete", "skewed")


@dataclass
class BraggTable:
    """Coherent elastic at one temperature: Bragg-edge energies (eV), cumulative S (eV·b).

    ``energies`` and ``xs`` are THERMR's σ_coh(E) tabulated on its own grid,
    when the format ships one beside the edges (G4NDL's
    ``Coherent/CrossSection``): the physics is the edges, the table is what
    that consumer interpolates.
    """

    edges: np.ndarray
    s: np.ndarray
    energies: Optional[np.ndarray] = None
    xs: Optional[np.ndarray] = None

    def cross_section(self, energies) -> np.ndarray:
        """σ_coh = S(E_i)/E for E between edge *i* and the next, 0 below the first."""
        e = np.asarray(energies, dtype=float)
        index = np.searchsorted(self.edges, e, side="right") - 1
        held = np.where(index >= 0, self.s[np.clip(index, 0, None)], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(e > 0, held / np.where(e > 0, e, 1.0), 0.0)


@dataclass
class IncoherentElasticTable:
    """Incoherent elastic at one temperature.

    Either ``bound_xs`` and ``debye_waller`` (the evaluation's own numbers, as
    OpenMC keeps them) or ``energies`` and ``xs`` (THERMR's σ(E), as ACE and
    G4NDL keep it), and the equiprobable cosines per incident energy when the
    format has them.
    """

    energies: Optional[np.ndarray] = None
    xs: Optional[np.ndarray] = None
    bound_xs: Optional[float] = None
    debye_waller: Optional[float] = None
    cosine_energies: Optional[np.ndarray] = None
    cosines: Optional[np.ndarray] = None

    @property
    def exact(self) -> bool:
        return self.bound_xs is not None and self.debye_waller is not None

    def cross_section(self, energies) -> np.ndarray:
        """σ_inc(E): ENDF-102 eq. 7.5 when exact, else the table, lin-lin, 0 outside it."""
        e = np.asarray(energies, dtype=float)
        if self.exact:
            x = 2.0 * e * self.debye_waller
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(x > 0, -np.expm1(-2.0 * x) / np.where(x > 0, x, 1.0), 2.0)
            return 0.5 * self.bound_xs * ratio
        return np.interp(e, self.energies, self.xs, left=np.nan, right=0.0)


@dataclass
class OutgoingTable:
    """The outgoing neutron of incoherent inelastic scattering, at one temperature.

    For incident energy ``incident[i]``: outgoing energies ``energies[i]`` (eV),
    with ``pdf[i]`` (1/eV) and ``cdf[i]`` when the table is continuous, or
    ``weights[i]`` when it is discrete (equal weights) or skewed; and
    ``cosines[i]``, one row of equiprobable cosines per outgoing energy.
    """

    kind: str
    incident: np.ndarray
    energies: List[np.ndarray]
    cosines: List[np.ndarray]
    pdf: Optional[List[np.ndarray]] = None
    cdf: Optional[List[np.ndarray]] = None
    weights: Optional[List[np.ndarray]] = None

    def nearest(self, energy: float) -> int:
        """The index of the tabulated incident energy closest to *energy* (on a log scale)."""
        e = max(float(energy), 1e-30)
        return int(np.argmin(np.abs(np.log(np.maximum(self.incident, 1e-30)) - np.log(e))))

    def probabilities(self, index: int) -> np.ndarray:
        """The probability of each outgoing energy line or bin at incident ``index``."""
        if self.kind == "continuous":
            cdf = self.cdf[index]
            return np.diff(np.concatenate([[0.0], cdf]))
        if self.weights is not None and self.weights[index] is not None:
            w = np.asarray(self.weights[index], dtype=float)
            return w / w.sum() if w.sum() > 0 else w
        n = len(self.energies[index])
        return np.full(n, 1.0 / n) if n else np.zeros(0)


@dataclass
class ProcessedTemperature:
    """Everything a processed format holds at one temperature (K)."""

    temperature: float
    coherent: Optional[BraggTable] = None
    incoherent: Optional[IncoherentElasticTable] = None
    inelastic_energies: Optional[np.ndarray] = None
    inelastic_xs: Optional[np.ndarray] = None
    #: Reads the outgoing table when first asked; the tables are the bulk of every format.
    outgoing_loader: Optional[Callable[[], Optional[OutgoingTable]]] = field(default=None, repr=False)
    _outgoing: Optional[OutgoingTable] = field(default=None, repr=False)
    outgoing_kind: Optional[str] = None

    def outgoing(self) -> Optional[OutgoingTable]:
        if self._outgoing is None and self.outgoing_loader is not None:
            self._outgoing = self.outgoing_loader()
        return self._outgoing


@dataclass
class ProcessedTSL:
    """A processed TSL material: name, mass, nuclides and its temperatures."""

    format: str
    name: str
    awr: Optional[float]
    temperatures: List[ProcessedTemperature]
    nuclides: List[str] = field(default_factory=list)
    title: Optional[str] = None
    library: Optional[str] = None
    emax: Optional[float] = None
    paths: List[str] = field(default_factory=list)
    #: Format-specific fields worth showing (ZAID, NXS modes, the G4NDL source tag).
    details: Dict[str, object] = field(default_factory=dict)

    def at(self, temperature: float) -> ProcessedTemperature:
        for block in self.temperatures:
            if abs(block.temperature - float(temperature)) < 1e-6 * max(1.0, block.temperature):
                return block
        raise KeyError(f"{temperature} K is not in this file; have "
                       f"{[b.temperature for b in self.temperatures]}")


def heated_label(index: int) -> str:
    """The style label of the *index*-th temperature (in the file's order)."""
    return f"heated_{index:03d}"


def _bragg_form(table: BraggTable, temperature: float, label: str):
    from kika.nuclear_data.model import CoherentElastic, Gridded2d, Interpolation
    from kika.nuclear_data.model.axes import Axes, Axis, Grid
    from kika.nuclear_data.model.enums import GridStyle

    axes = Axes([
        Grid(2, "temperature", "K", style=GridStyle.points, interpolation=Interpolation.linlin,
             values=np.asarray([temperature], dtype=float)),
        Grid(1, "energy_in", "eV", style=GridStyle.points, interpolation=Interpolation.flat,
             values=np.asarray(table.edges, dtype=float)),
        Axis(0, "S_cumulative", "eV*b"),
    ])
    return CoherentElastic(S_table=Gridded2d(values=np.asarray([table.s], dtype=float), axes=axes),
                           label=label)


def _incoherent_form(table: IncoherentElasticTable, temperature: float, label: str):
    from kika.nuclear_data.model import IncoherentElastic, PhysicalQuantity, XYs1d
    from kika.nuclear_data.model.axes import Axes, Axis

    w = XYs1d(xs=np.asarray([temperature]), ys=np.asarray([table.debye_waller]),
              axes=Axes([Axis(1, "temperature", "K"), Axis(0, "DebyeWallerIntegral", "1/eV")]))
    return IncoherentElastic(boundAtomCrossSection=PhysicalQuantity(float(table.bound_xs), "b"),
                             DebyeWallerIntegral=w, label=label)


def _xs(energies, values, label: str):
    from kika.nuclear_data.model import XYs1d, crossSectionAxes

    return XYs1d(xs=np.asarray(energies, dtype=float), ys=np.asarray(values, dtype=float),
                 axes=crossSectionAxes(), label=label)


def processed_suite(data: ProcessedTSL):
    """*data* as a model ``reactionSuite`` and its conversion report."""
    from kika.nuclear_data.model import (TNSL_INTERACTION, TNSL_PROCESSES, CoherentElastic,
                                         ConversionReport, DoubleDifferentialCrossSection,
                                         Evaluated, Frame, Heated, IncoherentElastic,
                                         IncoherentInelastic, PhysicalQuantity, PoPs, Reaction,
                                         ReactionSuite, ThermalNeutronScatteringLaw1d, Unorthodox)
    from kika.nuclear_data.model.reaction_id import ReactionId

    report = ConversionReport()
    pid = "tnsl-" + "_".join((data.name or "unknown").split())
    pops = PoPs()
    mass = PhysicalQuantity(float(data.awr) * 1.00866491595, "amu") if data.awr else None
    pops.add(Unorthodox(id=pid, mass=mass))
    suite = ReactionSuite(evaluation=data.title or data.name, projectile="n", target=pid,
                          projectileFrame=Frame.lab, interaction=TNSL_INTERACTION)
    suite.PoPs = pops
    suite.styles.add(Evaluated(label="eval", library=data.library))
    report.lost(f"{data.format} refers to an evaluation it does not contain: the 'eval' style "
                "is declared with no data behind it, and S(α,β) is not in the file")

    reactions: Dict[str, Reaction] = {}

    def reaction(key: str, mt: int, process_of) -> Reaction:
        if key not in reactions:
            r = Reaction(id=ReactionId(label=key, ENDF_MT=mt))
            r.doubleDifferentialCrossSection = DoubleDifferentialCrossSection()
            channel = r.outputChannel
            channel.genre = "twoBody"
            channel.process = TNSL_PROCESSES[process_of]
            channel.ensureProduct("n", label="n")
            reactions[key] = r
        return reactions[key]

    for index, block in enumerate(data.temperatures):
        label = heated_label(index)
        suite.styles.add(Heated(label=label, derivedFrom="eval",
                                temperature=PhysicalQuantity(block.temperature, "K")))
        if block.coherent is not None:
            r = reaction("coherent-elastic", 2, CoherentElastic)
            form = _bragg_form(block.coherent, block.temperature, label)
            r.doubleDifferentialCrossSection[label] = form
            r.crossSection[label] = ThermalNeutronScatteringLaw1d(
                href=f"/reactionSuite/reactions/reaction[@label='coherent-elastic']"
                     f"/doubleDifferentialCrossSection/{form.gndsNodeName}[@label='{label}']",
                label=label)
        if block.incoherent is not None:
            r = reaction("incoherent-elastic", 2, IncoherentElastic)
            if block.incoherent.exact:
                form = _incoherent_form(block.incoherent, block.temperature, label)
                r.doubleDifferentialCrossSection[label] = form
                r.crossSection[label] = ThermalNeutronScatteringLaw1d(
                    href=f"/reactionSuite/reactions/reaction[@label='incoherent-elastic']"
                         f"/doubleDifferentialCrossSection/{form.gndsNodeName}[@label='{label}']",
                    label=label)
            else:
                r.crossSection[label] = _xs(block.incoherent.energies, block.incoherent.xs, label)
        if block.inelastic_xs is not None:
            r = reaction("incoherent-inelastic", 4, IncoherentInelastic)
            r.crossSection[label] = _xs(block.inelastic_energies, block.inelastic_xs, label)

    for key in ("coherent-elastic", "incoherent-elastic", "incoherent-inelastic"):
        if key in reactions:
            suite.reactions.append(reactions[key])
    if any(b.outgoing_kind for b in data.temperatures):
        report.unsupportedNode(
            f"{data.format}'s outgoing-neutron tables (equiprobable cosines and outgoing "
            "energies) have no GNDS node; they are kept beside the model, read on demand",
            unaffectedScopes=("cross-sections",))
    suite.report = report
    return suite, report


def merge_temperatures(blocks: Sequence[ProcessedTemperature]) -> List[ProcessedTemperature]:
    """Sorted by temperature, one block per temperature (the first one read wins)."""
    seen: Dict[float, ProcessedTemperature] = {}
    for block in blocks:
        seen.setdefault(round(block.temperature, 3), block)
    return [seen[t] for t in sorted(seen)]


def _pairs(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    return arr[0::2], arr[1::2]
