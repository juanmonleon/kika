"""ENDF MF7 ↔ the thermal neutron scattering law (ENDF-coverage roadmap E4).

The flat path (:mod:`kika.endf.classes.mf7`) has read and written File 7 byte
for byte since 2026-08-13; this is the step to the model, which until now no TSL
tape could reach -- a TSL tape decoded to an empty ``nuclear`` suite that said
MF7 was not covered.

**The mapping is FUDGE's** (``brownies/legacy/converting/ENDFToGNDS/ENDF_ITYPE_2.py``)
wherever FUDGE follows the specification:

=====================================  ==================================================
ENDF                                   model
=====================================  ==================================================
MF7/MT2 LTHR=1|3, TAB1 + LT LISTs      ``CoherentElastic.S_table``: ``gridded2d``
                                       (temperature K, energy_in eV, S_cumulative eV*b);
                                       energy INT=1 → ``flat``, LI → temperature law
MF7/MT2 LTHR=2|3, TAB1 (T, W')         ``IncoherentElastic``: SB as ``boundAtomCrossSection``
                                       (b, used as is), W' as ``DebyeWallerIntegral`` (1/eV)
MF7/MT4 B(1)/B(6), B(3)                principal atom's σ_free and A; bound σ =
                                       σ_free·((A+1)/A)², mass = A·m_n (amu)
B(2), B(4), B(6)                       ``e_critical``, ``e_max`` (eV), ``numberPerMolecule``
B(6i+1) = 0 / 1                        secondary kernel ``SCTApproximation`` /
                                       ``freeGasApproximation``
S(α, β, T) TAB2 over β                 ``gridded3d`` (temperature, beta, alpha), ENDF's
                                       (β, T, α) transposed
Teff TAB1s                             ``T_effective``: principal, then each SCT secondary
LAT, LASYM                             ``calculatedAtThermal``, ``symmetric = LASYM == 0``
=====================================  ==================================================

**Three departures from FUDGE, declared.** (1) Atom identities come from
MF7/MT451 matched by mass, never from the tape's file name. (2) LI = 0 is not
rewritten to 2 (FUDGE's JENDL-5 workaround); it is kept, because kika
reproduces the tape. (3) LLN = 1 (ln S stored) is **refused by name** rather
than exponentiated with its interpolation laws relabelled -- FUDGE's own FIXME
calls that unverified, and no tape measured uses LLN = 1. A diffusive secondary
(B(6i+1) = 2), and α grids that differ between β, are refused the same way.
FUDGE refuses all three too.

**Unchanged means byte for byte.** The bookkeeping GNDS has no node for -- the
B array as written (B(5), and the exact digits of the scalars converted to bound
σ and amu), LLN, the (NBT, INT) regions of every TAB1 and TAB2, each
temperature's LI, and the padding dialect -- travels in each reaction's
provenance under :data:`MF7_KEY`. The encoder rebuilds the flat sections from
the model and takes from the provenance only what the model does not hold. A
scalar the model still holds unchanged is written with the digits it was read
with. MF7/MT451, the composition, is stated in the model since E4b (the
evaluated style's ``targetInfo``, PoPs nuclides, and the principal atom's
``boundAtomCrossSectionByNuclide``, as FUDGE maps it) and still goes back out
verbatim from the suite's provenance (:data:`MF7MT451_KEY`) when it is there;
without it the section is rebuilt from the model by FUDGE's rule.

**The target** (E4b) is the ``unorthodox`` particle ``tnsl-<ZSYMAM>`` with
MF1's AWR as its mass, in place of the pseudo-ZA nuclide MF1/451 names, and
each reaction has Q = 0 and a neutron multiplicity of 1 over the evaluation's
domain (1e-5 eV to EMAX): the nodes GNDS requires and FUDGE writes.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from kika._constants import NEUTRON_MASS_AMU
from kika.nuclear_data.model import (
    EVAL_LABEL, TNSL_INTERACTION, TNSL_PROCESSES, Axes, Axis, CoherentElastic,
    ConversionReport, DoubleDifferentialCrossSection, EndfProvenance,
    FreeGasApproximation, Gridded2d, Gridded3d, IncoherentElastic,
    IncoherentInelastic, PhysicalQuantity, Reaction, Regions1d, ScatteringAtom,
    SCTApproximation, SelfScatteringKernel, ThermalNeutronScatteringLaw,
    TargetInfo, TargetInfoElement, TargetInfoNuclide, ThermalNeutronScatteringLaw1d,
    Unorthodox, XYs1d, pidFromZA,
)
from kika._constants import ATOMIC_NUMBER_TO_SYMBOL
from kika.nuclear_data.model import Evaluated, Nuclide
from kika.nuclear_data.model.axes import Grid
from kika.nuclear_data.model.enums import (ENDF_INT_TO_INTERPOLATION,
                                           INTERPOLATION_TO_ENDF_INT, GridStyle)
from kika.nuclear_data.model.reaction_id import ReactionId

__all__ = ["MF7_KEY", "MF7MT451_KEY", "LLN_REFUSAL", "attachThermalScattering",
           "encodeMF7Sections"]

#: Where each TSL reaction keeps its ENDF bookkeeping, on its provenance.
MF7_KEY = "mf7"
#: Where the suite keeps MF7/MT451's text.
MF7MT451_KEY = "mf7mt451"

LLN_REFUSAL = (
    "MF7/MT4 LLN=1 (ln S stored) is not read by kika: no evaluation measured "
    "uses it, and FUDGE's reading of it is marked unverified by FUDGE itself"
)

_LABELS = {CoherentElastic: "coherent-elastic",
           IncoherentElastic: "incoherent-elastic",
           IncoherentInelastic: "incoherent-inelastic"}


# ---------------------------------------------------------------------------
# Small conversions
# ---------------------------------------------------------------------------

def _function1d(xs, ys, interp, axes: Axes):
    """ENDF (x, y, NBT/INT) → ``XYs1d`` for one region, ``Regions1d`` for more."""
    if len(interp) == 1 and int(interp[0][0]) == len(xs):
        return XYs1d(xs=np.asarray(xs, dtype=float), ys=np.asarray(ys, dtype=float),
                     interpolation=ENDF_INT_TO_INTERPOLATION[int(interp[0][1])],
                     axes=axes)
    return Regions1d.fromEndfRegions(xs, ys, [(int(n), int(i)) for n, i in interp], axes=axes)


def _toEndf1d(function) -> Tuple[List[float], List[float], List[Tuple[int, int]]]:
    xs, ys, pairs = function.toEndfRegions()
    return ([float(x) for x in xs], [float(y) for y in ys],
            [(int(n), int(i)) for n, i in pairs])


def _axes1d(xLabel: str, xUnit: str, yLabel: str, yUnit: str) -> Axes:
    return Axes([Axis(1, xLabel, xUnit), Axis(0, yLabel, yUnit)])


def _boundFromFree(free: float, awr: float) -> float:
    return free * ((awr + 1.0) / awr) ** 2


def _freeFromBound(bound: float, awr: float) -> float:
    return bound / ((awr + 1.0) / awr) ** 2


def _law(code: int):
    return ENDF_INT_TO_INTERPOLATION.get(int(code), ENDF_INT_TO_INTERPOLATION[2])


def _code(interpolation) -> int:
    return INTERPOLATION_TO_ENDF_INT[interpolation]


def _href(label: str, form) -> str:
    return (f"/reactionSuite/reactions/reaction[@label='{label}']"
            f"/doubleDifferentialCrossSection/{form.gndsNodeName}[@label='{EVAL_LABEL}']")


# ---------------------------------------------------------------------------
# MF7/MT2
# ---------------------------------------------------------------------------

def _decodeCoherent(coherent, report: ConversionReport):
    table = coherent.table
    if not coherent.is_histogram:
        report.warn(
            f"MF7/MT2 coherent elastic: the Bragg-edge table is not INT=1 "
            f"({table.interp}); S(E) is a staircase and is kept as the file says"
        )
    li = list(table.li)
    temperatureLaw = _law(li[0]) if li and li[0] else _law(2)
    axes = Axes([
        Grid(2, "temperature", "K", style=GridStyle.points, interpolation=temperatureLaw,
             values=np.asarray(table.temperatures, dtype=float)),
        Grid(1, "energy_in", "eV", style=GridStyle.points,
             interpolation=_law(table.interp[0][1]) if table.interp else _law(1),
             values=np.asarray(table.x, dtype=float)),
        Axis(0, "S_cumulative", "eV*b"),
    ])
    form = CoherentElastic(S_table=Gridded2d(values=np.asarray(table.values, dtype=float), axes=axes),
                           label=EVAL_LABEL)
    return form, {"interp": [tuple(p) for p in table.interp], "li": li, "t0": table.t0}


def _decodeIncoherentElastic(incoherent):
    dw = _function1d(incoherent.temperatures, incoherent.w, incoherent.interp,
                     _axes1d("temperature", "K", "DebyeWallerIntegral", "1/eV"))
    form = IncoherentElastic(
        boundAtomCrossSection=PhysicalQuantity(float(incoherent.sb), "b"),
        DebyeWallerIntegral=dw, label=EVAL_LABEL)
    return form, {"interp": [tuple(p) for p in incoherent.interp]}


# ---------------------------------------------------------------------------
# The target and its composition (MF1/451's pseudo-ZA, MF7/MT451)
# ---------------------------------------------------------------------------

def _tslTarget(suite) -> None:
    """Make the suite's target the ``unorthodox`` particle a TSL evaluation is.

    MF1/451 gives a TSL tape a pseudo-ZA (MAT + 100: ``134`` for s-CH4), which
    the header decoder turned into a nuclide ``ZA134`` with no element. The
    scatterer is a molecule or a lattice, not a nuclide, and GNDS -- like
    FUDGE -- states it as ``unorthodox`` with MF1's AWR as its mass. Its id is
    ``tnsl-`` and the evaluation's ZSYMAM, the tape's own name for the
    material. FUDGE takes the name from the *file* name instead; a file name is
    not part of the evaluation and kika does not read meaning into it.
    """
    old = suite.target
    particle = suite.PoPs.particles.pop(old, None) if old in suite.PoPs else None
    name = "_".join((suite.evaluation or "").split()) or old or "unknown"
    pid = f"tnsl-{name}"
    suite.PoPs.add(Unorthodox(id=pid, mass=getattr(particle, "mass", None)))
    suite.target = pid


def _evaluatedStyle(suite) -> Optional[Evaluated]:
    return next((s for s in suite.styles if isinstance(s, Evaluated)), None)


def _tslDomain(suite, report: ConversionReport):
    """The evaluated style's ``projectileEnergyDomain``: 1e-5 eV to MF1's EMAX.

    The schema requires it on ``evaluated`` and the TSL forms' Q and
    multiplicity need it. FUDGE builds it the same way, and also replaces an
    EMAX above 5 eV with 5 eV; kika keeps what the tape says.

    **EMAX = 0 is real**: ENDF/B-VIII.1 tsl-HinH2O and tsl-ortho-H write it. There
    kika does what FUDGE does, 5 eV (``ENDF_ITYPE_2``: EMAX <= EMin gives 5 eV),
    and says so; the tape's own EMAX still goes back out unchanged.
    """
    from kika.nuclear_data.model import RangeQuantity

    style = _evaluatedStyle(suite)
    if style is None:
        return None
    if style.projectileEnergyDomain is not None:
        return style.projectileEnergyDomain
    header = getattr(getattr(suite, "provenance", None), "headerFields", None) or {}
    emax = header.get("emax")
    if emax is None:
        report.lost("MF1/451 is missing, so the TSL evaluation has no "
                    "projectileEnergyDomain and its Q and multiplicity are left unset")
        return None
    if float(emax) <= 1e-5:
        report.approximated(f"MF1/451 EMAX = {float(emax):g} eV is not a domain; the TSL "
                            f"evaluation is given 1e-5 to 5 eV, FUDGE's value for this case")
        emax = 5.0
    style.projectileEnergyDomain = RangeQuantity(min=1e-5, max=float(emax), unit="eV")
    return style.projectileEnergyDomain


def _decodeComposition(suite, composition, report: ConversionReport) -> dict:
    """MF7/MT451 → ``targetInfo``, PoPs nuclides, and bound σ per nuclide.

    FUDGE's mapping (``ENDF_ITYPE_2.readMF7_info``): each isotope becomes a PoPs
    nuclide with mass AWRI x m_n and a ``targetInfo`` nuclide with its atom
    fraction AFI; its free cross section SFI becomes a bound one,
    SFI·((A+1)/A)², returned keyed by pid for the principal scattering atom's
    ``boundAtomCrossSectionByNuclide``. NAS is not modelled (it is
    ``numberPerMolecule`` where it is right, and it is not always right); the
    section itself still travels verbatim in the suite's provenance.
    """
    byNuclide = {}
    elements = []
    for element in composition:
        nuclides = []
        symbol = None
        for isotope in element.isotopes:
            pid = pidFromZA(int(isotope.zai), int(isotope.lis or 0))
            if pid not in suite.PoPs:
                suite.PoPs.add(Nuclide(
                    id=pid, Z=isotope.z, A=isotope.a,
                    mass=PhysicalQuantity(isotope.awr * NEUTRON_MASS_AMU, "amu")
                    if isotope.awr else None))
            if isotope.awr:
                byNuclide[pid] = PhysicalQuantity(
                    _boundFromFree(isotope.sigma_free, isotope.awr), "b")
            nuclides.append(TargetInfoNuclide(pid=pid, atomFraction=float(isotope.atom_fraction)))
            symbol = symbol or ATOMIC_NUMBER_TO_SYMBOL.get(isotope.z)
        elements.append(TargetInfoElement(symbol=symbol or "", nuclides=nuclides))
    style = _evaluatedStyle(suite)
    if style is not None:
        style.targetInfo = TargetInfo(chemicalElements=elements)
    else:
        report.lost("MF7/MT451: the suite has no evaluated style to hang targetInfo on")
    return byNuclide


# ---------------------------------------------------------------------------
# MF7/MT4
# ---------------------------------------------------------------------------

def _pidByMass(awr: float, composition, used: set) -> Optional[str]:
    """The MF7/MT451 nuclide whose AWR is closest to *awr*, within 1 %."""
    best, distance = None, None
    for element in composition:
        for isotope in element.isotopes:
            if not isotope.awr:
                continue
            d = abs(isotope.awr - awr) / isotope.awr
            if d < 0.01 and (distance is None or d < distance):
                best, distance = isotope, d
    if best is None:
        return None
    pid = pidFromZA(int(best.zai))
    return pid if pid not in used else f"{pid}_{len(used)}"


def _secondaryPid(index: int, awr: float, composition, used: set,
                  report: ConversionReport) -> str:
    pid = _pidByMass(awr, composition, used)
    if pid is None:
        pid = f"scatterer{index}"
        report.approximated(
            f"MF7/MT4 scattering atom {index} (AWR={awr:g}) matches no nuclide of "
            f"MF7/MT451 by mass, so it is named {pid!r}; GNDS wants a PoPs id here "
            f"and kika will not guess one from the file name"
        )
    used.add(pid)
    return pid


def _decodeMT4(mt4, composition, principalName: Optional[str],
               report: ConversionReport):
    if (mt4.lln or 0) != 0:
        report.unsupportedNode(f"{LLN_REFUSAL}; the section is absent from the model")
        return None, None
    if not mt4.has_tabulated_s or not mt4.blocks:
        report.unsupportedNode(
            "MF7/MT4 carries no tabulated S(alpha, beta) for its principal "
            "scatterer (B(1) = 0); not decoded")
        return None, None
    first = mt4.blocks[0].table
    for block in mt4.blocks[1:]:
        t = block.table
        if (t.x != first.x or t.interp != first.interp
                or t.temperatures != first.temperatures or t.li != first.li):
            report.unsupportedNode(
                f"MF7/MT4: the alpha grid, its interpolation or the temperatures "
                f"differ between beta = {mt4.blocks[0].beta:g} and beta = "
                f"{block.beta:g}; a gridded3d states one of each, so the section "
                f"is not decoded (FUDGE refuses it too)")
            return None, None
    secondaries = mt4.secondary_scatterers()
    for index, secondary in enumerate(secondaries, start=1):
        if secondary.analytic_flag not in (0.0, 1.0):
            report.unsupportedNode(
                f"MF7/MT4 secondary scatterer {index}: B(6i+1) = "
                f"{secondary.analytic_flag:g} (diffusive motion) has no GNDS kernel; "
                f"the section is not decoded")
            return None, None

    temperatures = np.asarray(first.temperatures, dtype=float)
    betas = np.asarray([b.beta for b in mt4.blocks], dtype=float)
    alphas = np.asarray(first.x, dtype=float)
    values = np.asarray([[block.table.values[t] for block in mt4.blocks]
                         for t in range(len(temperatures))], dtype=float)
    li = list(first.li)
    axes = Axes([
        Grid(3, "temperature", "K", style=GridStyle.points,
             interpolation=_law(li[0]) if li and li[0] else _law(2), values=temperatures),
        Grid(2, "beta", "", style=GridStyle.points,
             interpolation=_law(mt4.beta_interp[0][1]) if mt4.beta_interp else _law(2),
             values=betas),
        Grid(1, "alpha", "", style=GridStyle.points,
             interpolation=_law(first.interp[0][1]) if first.interp else _law(2),
             values=alphas),
        Axis(0, "S_alpha_beta", ""),
    ])
    kernel = Gridded3d(values=values, axes=axes)

    b = list(mt4.b)
    teff = list(mt4.teff)
    teffAxes = _axes1d("temperature", "K", "t_effective", "K")
    used: set = set()

    awr0, m0 = b[2], b[5]
    principalPid = _pidByMass(awr0, composition, used)
    if principalPid is None:
        principalPid = principalName or "principal"
        report.approximated(
            f"MF7/MT4 principal scatterer (AWR={awr0:g}) matches no nuclide of "
            f"MF7/MT451 by mass{' (the tape has no MT451)' if not composition else ''}, "
            f"so it is named {principalPid!r}; kika will not guess a PoPs id from "
            f"the file name")
    used.add(principalPid)
    atoms = [ScatteringAtom(
        pid=principalPid,
        numberPerMolecule=int(round(m0)),
        mass=PhysicalQuantity(awr0 * NEUTRON_MASS_AMU, "amu"),
        e_critical=PhysicalQuantity(b[1], "eV"),
        e_max=PhysicalQuantity(b[3], "eV"),
        boundAtomCrossSection=PhysicalQuantity(_boundFromFree(b[0] / m0, awr0), "b"),
        selfScatteringKernel=SelfScatteringKernel(kernel, symmetric=(mt4.lasym or 0) == 0),
        primaryScatterer=True,
        T_effective=(_function1d(teff[0].temperatures, teff[0].teff, teff[0].interp, teffAxes)
                     if teff else None),
    )]
    nextTeff = 1
    for index, secondary in enumerate(secondaries, start=1):
        base = 6 * index
        pid = _secondaryPid(index, secondary.awr, composition, used, report)
        isSCT = secondary.analytic_flag == 0.0
        tEff = None
        if isSCT and nextTeff < len(teff):
            record = teff[nextTeff]
            tEff = _function1d(record.temperatures, record.teff, record.interp, teffAxes)
            nextTeff += 1
        atoms.append(ScatteringAtom(
            pid=pid,
            numberPerMolecule=int(round(secondary.n_atoms)),
            mass=PhysicalQuantity(secondary.awr * NEUTRON_MASS_AMU, "amu"),
            e_max=PhysicalQuantity(b[base + 3], "eV"),
            boundAtomCrossSection=PhysicalQuantity(
                _boundFromFree(secondary.free_xs / secondary.n_atoms, secondary.awr), "b"),
            selfScatteringKernel=SelfScatteringKernel(
                SCTApproximation() if isSCT else FreeGasApproximation(), symmetric=True),
            T_effective=tEff,
        ))
    form = IncoherentInelastic(scatteringAtoms=atoms, primaryScatterer=principalPid,
                               label=EVAL_LABEL,
                               calculatedAtThermal=(mt4.lat or 0) != 0)
    bookkeeping = {
        "b": b, "lln": mt4.lln or 0, "lasym": mt4.lasym or 0, "lat": mt4.lat or 0,
        "betaInterp": [tuple(p) for p in mt4.beta_interp],
        "alphaInterp": [tuple(p) for p in first.interp],
        "li": li, "t0": first.t0,
        "teffInterp": [[tuple(p) for p in t.interp] for t in teff],
        "scalars": _mt4Scalars(b, len(secondaries)),
    }
    return form, bookkeeping


def _mt4Scalars(b: List[float], ns: int) -> list:
    """What the model holds of the B array, computed exactly as the decoder does."""
    out = [(_boundFromFree(b[0] / b[5], b[2]), b[2] * NEUTRON_MASS_AMU, b[1], b[3],
            int(round(b[5])))]
    for index in range(1, ns + 1):
        base = 6 * index
        out.append((_boundFromFree(b[base + 1] / b[base + 5], b[base + 2]),
                    b[base + 2] * NEUTRON_MASS_AMU, b[base + 3], int(round(b[base + 5])),
                    b[base]))
    return out


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------

def _reaction(form, mt: int, provenance, domain=None) -> Reaction:
    """One TSL reaction, built the way FUDGE builds it (``ENDF_ITYPE_2``).

    Q is 0 and the neutron's multiplicity is 1, both over the evaluation's
    domain: a bound scatterer exchanges energy with the material, it does not
    change any rest mass, and the neutron comes out. These are the two §17
    nodes the schema requires and ENDF states nowhere because they cannot be
    anything else. Without a domain they are left unset and the GNDS writer
    reports the empty nodes.
    """
    label = _LABELS[type(form)]
    reaction = Reaction(id=ReactionId(label=label, ENDF_MT=mt), provenance=provenance)
    reaction.doubleDifferentialCrossSection = DoubleDifferentialCrossSection()
    reaction.doubleDifferentialCrossSection[EVAL_LABEL] = form
    href = _href(label, form)
    reaction.crossSection[EVAL_LABEL] = ThermalNeutronScatteringLaw1d(href=href, label=EVAL_LABEL)
    channel = reaction.outputChannel
    channel.genre = "twoBody"
    channel.process = TNSL_PROCESSES[type(form)]
    product = channel.ensureProduct("n", label="n")
    from kika.nuclear_data.model import Constant1d, Distribution, Multiplicity, Q
    from kika.nuclear_data.model.axes import multiplicityAxes
    if domain is not None:
        low, high = float(domain.min), float(domain.max)
        channel.Q = Q(value=0.0, unit="eV", label=EVAL_LABEL, domainMin=low, domainMax=high)
        product.multiplicity = Multiplicity(form=Constant1d(
            constant=1.0, domainMin_=low, domainMax_=high, axes=multiplicityAxes(),
            label=EVAL_LABEL))
    product.distribution = Distribution()
    product.distribution[EVAL_LABEL] = ThermalNeutronScatteringLaw(href=href, label=EVAL_LABEL)
    return reaction


def _provenance(section, bookkeeping: dict) -> EndfProvenance:
    pad = getattr(section, "pad", None)
    bookkeeping = dict(bookkeeping)
    if pad is not None:
        bookkeeping["pad"] = (pad.pairs, pad.values, pad.interp)
    bookkeeping["lthr"] = getattr(section, "lthr", None)
    return EndfProvenance(mat=getattr(section, "_mat", None), awr=getattr(section, "_awr", None),
                          za=getattr(section, "_za", None), headerFields={MF7_KEY: bookkeeping})


def attachThermalScattering(suite, endf, report: ConversionReport) -> ConversionReport:
    """Decode MF7 into *suite*, which becomes a TSL ``reactionSuite``.

    Called by :func:`~kika.endf.model_adapter.decode.decodeReactionSuite` when
    the tape has an MF7. Reactions, in ENDF order: MT2's coherent and
    incoherent forms (both ``ENDF_MT = 2`` when LTHR = 3, as FUDGE does), then
    MT4.
    """
    mf7 = endf.mf.get(7)
    sections = getattr(mf7, "mt", {}) if mf7 is not None else {}
    suite.interaction = TNSL_INTERACTION

    _tslTarget(suite)
    domain = _tslDomain(suite, report)

    composition = []
    byNuclide = {}
    mt451 = sections.get(451)
    if mt451 is not None:
        composition = list(mt451.elements)
        if suite.provenance is not None:
            # The model states the composition (targetInfo); the text is kept
            # too, because NAS and the digits of SFI are ENDF's alone.
            suite.provenance.headerFields[MF7MT451_KEY] = str(mt451).split("\n")[:-1]
        byNuclide = _decodeComposition(suite, composition, report)

    try:
        from kika.endf.classes.mf7.scatterer import thermal_scatterer
        principalName = thermal_scatterer(endf).principal
    except Exception:  # identity is a convenience here, never a reason to fail
        principalName = None

    mt2 = sections.get(2)
    if mt2 is not None:
        if mt2.coherent is not None:
            form, book = _decodeCoherent(mt2.coherent, report)
            suite.reactions.append(_reaction(form, 2, _provenance(mt2, {"part": "coherent", **book}), domain))
        if mt2.incoherent is not None:
            form, book = _decodeIncoherentElastic(mt2.incoherent)
            suite.reactions.append(_reaction(form, 2, _provenance(mt2, {"part": "incoherent", **book}), domain))

    mt4 = sections.get(4)
    if mt4 is not None:
        form, book = _decodeMT4(mt4, composition, principalName, report)
        if form is not None:
            form.principal.boundAtomCrossSectionByNuclide = dict(byNuclide)
            suite.reactions.append(_reaction(form, 4, _provenance(mt4, book), domain))
    return report


# ---------------------------------------------------------------------------
# The way back
# ---------------------------------------------------------------------------

def _pad(book: dict):
    from kika.endf.utils import PadStyle

    pairs, values, interp = book.get("pad") or (None, None, None)
    kwargs = {k: v for k, v in (("pairs", pairs), ("values", values), ("interp", interp)) if v}
    return PadStyle(**kwargs)


def _temperatureTable(t0, x, rows, temperatures, interp, li):
    from kika.endf.classes.mf7.base import TemperatureTable

    return TemperatureTable(t0=float(t0), lt=len(temperatures) - 1,
                            interp=[tuple(p) for p in interp],
                            x=[float(v) for v in x],
                            values=[[float(v) for v in row] for row in rows],
                            temperatures=[float(t) for t in temperatures],
                            li=list(li))


def _liFor(book: dict, grid, count: int) -> List[int]:
    li = book.get("li")
    if li is not None and len(li) == count:
        return list(li)
    return [_code(grid.interpolation)] * count


def _encodeCoherent(form: CoherentElastic, book: dict):
    from kika.endf.classes.mf7.elastic import CoherentElastic as FlatCoherent

    temperature, energy = form.S_table.grids
    x = np.asarray(energy.values, dtype=float)
    interp = book.get("interp")
    if not interp or int(interp[-1][0]) != x.size:
        interp = [(x.size, _code(energy.interpolation))]
    temperatures = np.asarray(temperature.values, dtype=float)
    li = _liFor(book, temperature, temperatures.size - 1)
    t0 = book.get("t0", temperatures[0])
    if t0 != temperatures[0]:
        t0 = temperatures[0]
    return FlatCoherent(table=_temperatureTable(t0, x, form.S_table.values, temperatures, interp, li))


def _encodeIncoherentElastic(form: IncoherentElastic, book: dict):
    from kika.endf.classes.mf7.elastic import IncoherentElastic as FlatIncoherent

    xs, ys, pairs = _toEndf1d(form.DebyeWallerIntegral)
    return FlatIncoherent(sb=float(form.boundAtomCrossSection.value), interp=pairs,
                          temperatures=xs, w=ys)


def _bArray(form: IncoherentInelastic, book: dict, report: ConversionReport) -> List[float]:
    """The B array: the source's digits where the model still holds what they say."""
    atoms = form.scatteringAtoms
    principal = form.principal
    secondaries = [a for a in atoms if a is not principal]
    current = [(principal.boundAtomCrossSection.value, principal.mass.value,
                principal.e_critical.value if principal.e_critical else 0.0,
                principal.e_max.value, principal.numberPerMolecule)]
    for atom in secondaries:
        flag = 0.0 if isinstance(atom.selfScatteringKernel.kernel, SCTApproximation) else 1.0
        current.append((atom.boundAtomCrossSection.value, atom.mass.value, atom.e_max.value,
                        atom.numberPerMolecule, flag))
    kept = book.get("b")
    if kept is not None and book.get("scalars") == current:
        return list(kept)

    def awr(atom):
        return atom.mass.value / NEUTRON_MASS_AMU

    a0 = awr(principal)
    m0 = float(principal.numberPerMolecule)
    b5 = kept[4] if kept is not None and len(kept) > 4 else 0.0
    b = [_freeFromBound(principal.boundAtomCrossSection.value, a0) * m0,
         principal.e_critical.value if principal.e_critical else 0.0,
         a0, principal.e_max.value, b5, m0]
    for index, (atom, values) in enumerate(zip(secondaries, current[1:]), start=1):
        ai, mi = awr(atom), float(atom.numberPerMolecule)
        b += [values[4], _freeFromBound(atom.boundAtomCrossSection.value, ai) * mi, ai,
              atom.e_max.value,
              kept[6 * index + 4] if kept is not None and len(kept) > 6 * index + 4 else 0.0,
              mi]
    if kept is not None:
        report.approximated(
            "MF7/MT4: the scattering atoms changed in the model, so the B array is "
            "recomputed from bound cross sections and masses (free = bound·(A/(A+1))²)")
    return b


def _encodeMT4(form: IncoherentInelastic, book: dict, provenance, mat, report):
    from kika.endf.classes.mf7.inelastic import BetaBlock, EffectiveTemperature, MF7MT4

    kernel = form.principal.selfScatteringKernel.kernel
    if not isinstance(kernel, Gridded3d):
        raise ValueError(
            f"MF7/MT4 needs the principal atom's S(alpha, beta, T) tabulated; this "
            f"model holds a {type(kernel).__name__}, which ENDF cannot state")
    temperature, beta, alpha = kernel.grids
    temperatures = np.asarray(temperature.values, dtype=float)
    alphas = np.asarray(alpha.values, dtype=float)
    alphaInterp = book.get("alphaInterp")
    if not alphaInterp or int(alphaInterp[-1][0]) != alphas.size:
        alphaInterp = [(alphas.size, _code(alpha.interpolation))]
    betaInterp = book.get("betaInterp")
    betas = np.asarray(beta.values, dtype=float)
    if not betaInterp or int(betaInterp[-1][0]) != betas.size:
        betaInterp = [(betas.size, _code(beta.interpolation))]
    li = _liFor(book, temperature, temperatures.size - 1)
    t0 = temperatures[0]
    blocks = [BetaBlock(beta=float(b),
                        table=_temperatureTable(t0, alphas, kernel.values[:, j, :],
                                                temperatures, alphaInterp, li))
              for j, b in enumerate(betas)]

    teffs = []
    teffInterp = book.get("teffInterp") or []
    for atom in form.scatteringAtoms:
        if atom.T_effective is None:
            continue
        xs, ys, pairs = _toEndf1d(atom.T_effective)
        teffs.append(EffectiveTemperature(interp=pairs, temperatures=xs, teff=ys))
    del teffInterp  # each T_effective carries its own regions

    symmetric = form.principal.selfScatteringKernel.symmetric
    lasym = book.get("lasym", 0) if symmetric is None else (0 if symmetric else 1)
    section = MF7MT4(number=4, _za=provenance.za, _awr=provenance.awr,
                     _mat=mat if mat is not None else provenance.mat, pad=_pad(book),
                     _lat=1 if form.calculatedAtThermal else 0, _lasym=lasym,
                     _lln=book.get("lln", 0), _ns=len(form.scatteringAtoms) - 1,
                     b=_bArray(form, book, report), beta_interp=betaInterp,
                     blocks=blocks, teff=teffs)
    return section


def _encodeMT451(suite, inelastic, anchor, mat, report: ConversionReport):
    """MF7/MT451 from ``targetInfo``, when no source text was kept -- FUDGE's rule.

    (``toENDF6/.../thermalNeutronScatteringLaw.py``.) ZAI from the PoPs nuclide,
    AWRI from its mass, AFI from ``targetInfo``, SFI from the principal atom's
    ``boundAtomCrossSectionByNuclide`` taken back to free, and NAS from the
    ``numberPerMolecule`` of the scattering atom of that element (1 if none).
    """
    from kika.endf.classes.mf7.composition import (MF7MT451, ElementComposition,
                                                   TSLIsotope)
    from kika.nuclear_data.model.pops import zaFromPid

    style = _evaluatedStyle(suite)
    info = getattr(style, "targetInfo", None)
    if info is None or not info.chemicalElements:
        return None
    provenance = anchor[2] if anchor is not None else None
    if provenance is None or provenance.za is None or provenance.awr is None:
        report.lost("MF7/MT451: targetInfo is stated but the MF7 header (ZA, AWR) is "
                    "not known, so the section is not written")
        return None
    principal = inelastic[0].principal if inelastic is not None else None
    bound = principal.boundAtomCrossSectionByNuclide if principal is not None else {}
    perMolecule = {}
    for atom in (inelastic[0].scatteringAtoms if inelastic is not None else []):
        particle = suite.PoPs.particles.get(atom.pid)
        z = getattr(particle, "Z", None)
        if z is not None:
            perMolecule[ATOMIC_NUMBER_TO_SYMBOL.get(z)] = atom.numberPerMolecule
    elements = []
    for element in info.chemicalElements:
        isotopes = []
        for nuclide in element.nuclides:
            particle = suite.PoPs.particles.get(nuclide.pid)
            if particle is None or particle.mass is None:
                report.lost(f"MF7/MT451: {nuclide.pid} has no PoPs mass, so the "
                            f"section is not written")
                return None
            awri = particle.mass.convertedTo("amu").value / NEUTRON_MASS_AMU
            sigma = bound.get(nuclide.pid)
            if sigma is None:
                report.lost(f"MF7/MT451: no bound cross section for {nuclide.pid}, so "
                            f"the section is not written")
                return None
            isotopes.append(TSLIsotope(zai=zaFromPid(nuclide.pid),
                                       lis=int(getattr(particle, "nuclearLevel", 0) or 0),
                                       atom_fraction=nuclide.atomFraction, awr=awri,
                                       sigma_free=_freeFromBound(sigma.value, awri)))
        elements.append(ElementComposition(nas=perMolecule.get(element.symbol, 1),
                                           isotopes=isotopes))
    report.approximated("MF7/MT451 is written from targetInfo (no source text was kept); "
                        "NAS comes from numberPerMolecule")
    return MF7MT451(number=451, _za=provenance.za, _awr=provenance.awr,
                    _mat=mat if mat is not None else provenance.mat, elements=elements)


def encodeMF7Sections(suite, mat: Optional[int], report: ConversionReport):
    """``[(7, MT, section), ...]`` for a TSL suite, and the report.

    MT2 is assembled from the coherent and/or incoherent elastic reactions
    (LTHR = 1, 2 or 3). MT4 comes from the incoherent-inelastic one, and
    MT451 from the text the decoder kept.
    """
    from kika.endf.classes.mf7.elastic import MF7MT2
    from kika.endf.parsers.parse_mf7 import parse_mf7_mt451

    if getattr(suite, "interaction", None) != TNSL_INTERACTION:
        return [], report

    coherent = incoherent = inelastic = None
    for reaction in suite.reactions:
        ddcs = reaction.doubleDifferentialCrossSection
        form = ddcs.get(EVAL_LABEL) if ddcs is not None else None
        if form is None:
            continue
        header = getattr(reaction.provenance, "headerFields", None) or {}
        entry = (form, header.get(MF7_KEY, {}), reaction.provenance)
        if isinstance(form, CoherentElastic):
            coherent = entry
        elif isinstance(form, IncoherentElastic):
            incoherent = entry
        elif isinstance(form, IncoherentInelastic):
            inelastic = entry

    sections = []
    header = getattr(getattr(suite, "provenance", None), "headerFields", None) or {}
    kept451 = header.get(MF7MT451_KEY)
    if kept451 is not None:
        section451 = parse_mf7_mt451(list(kept451))
        if mat is not None:
            section451._mat = int(mat)
        sections.append((7, 451, section451))
    else:
        section451 = _encodeMT451(suite, inelastic, coherent or incoherent or inelastic,
                                  mat, report)
        if section451 is not None:
            sections.append((7, 451, section451))

    if coherent is not None or incoherent is not None:
        anchor = coherent or incoherent
        provenance, book = anchor[2], anchor[1]
        lthr = 3 if coherent and incoherent else (1 if coherent else 2)
        mt2 = MF7MT2(number=2, _za=provenance.za, _awr=provenance.awr,
                     _mat=mat if mat is not None else provenance.mat, pad=_pad(book),
                     _lthr=lthr,
                     coherent=_encodeCoherent(coherent[0], coherent[1]) if coherent else None,
                     incoherent=(_encodeIncoherentElastic(incoherent[0], incoherent[1])
                                 if incoherent else None))
        sections.append((7, 2, mt2))
    if inelastic is not None:
        sections.append((7, 4, _encodeMT4(inelastic[0], inelastic[1], inelastic[2], mat, report)))
    return sections, report
