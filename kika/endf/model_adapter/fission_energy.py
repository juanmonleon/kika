"""ENDF MF1/458 ↔ §18.4 ``fissionEnergyRelease``, and MF1/460 kept verbatim.

**Where it goes.** MF1/458 states how the energy of a fission is shared out:
nine components, each with an uncertainty. GNDS puts them on the fission
channel, ``reaction[MT18]/outputChannel/fissionFragmentData/
fissionEnergyReleases/fissionEnergyRelease``, next to the delayed-neutron
families MF1/455 fills -- the same channel :mod:`.multiplicity` hangs the
nu-bars from, and for the same reason: ENDF writes them in File 1 with no MT,
but they are a property of *fission*. The placement and the nine names are
FUDGE's (``ENDF_ITYPE_0_Misc.getFissionEnergies``), so a kika-written GNDS
file and a FUDGE-written one name the same node the same way.

**The three shapes the file can take, and what each becomes** (ENDF-102 §1.5.1;
census of ENDF/B-VIII.1, JEFF-4.0 and JENDL-5 on 2026-10-08: 106 LFC=0 NPLY=0,
52 LFC=0 NPLY=2, 3 LFC=1 -- B-VIII.1 U-235, U-238 and Pu-239, each with
IFC=1..4 tabulated):

``LFC=0, NPLY=0``  nine constants and their uncertainties → nine degree-0
    ``polynomial1d``, each with its uncertainty as a second ``polynomial1d``.
``LFC=0, NPLY=N``  nine degree-N polynomials, coefficients ``c_n`` in
    ``eV / eV**n``. The uncertainties are per coefficient and uncorrelated as
    far as the file says.
``LFC=1``          the thermal LIST of the NPLY=0 shape, then NFC TAB1s, each
    replacing one component with a table. The tabulated term becomes an
    ``XYs1d`` (a ``regions1d`` if NR > 1, which GNDS cannot hold); the
    polynomial it replaced is not a term any more.

**Where kika departs from FUDGE, and why.** Two of FUDGE's choices lose data
the file states, and neither is part of the GNDS specification:

1. On the way back to ENDF, FUDGE writes a tabulated term's thermal LIST entry
   as the table evaluated at 1e-5 eV *with zero uncertainty*
   (``toENDF6/outputChannelData/fissionEnergyReleased.py``). The file's own
   thermal value and its uncertainty are gone. kika keeps both in the
   provenance and writes them back.
2. FUDGE hard-codes LDRV (``2 if promptProductKE else 1``, its own ``FIXME``).
   LDRV is the evaluator's statement of which components are primary
   evaluations and which are derived from other sections (§1.5.2) -- kika
   keeps the one the file states.

What has no GNDS node at all stays in :attr:`FissionEnergyRelease.provenance`,
and the encoder reads it back. A term edited after decoding is written from the
model; only what the model cannot hold comes from the provenance.

**MF1/460, the delayed photons, has no GNDS node** -- §18.4 has the delayed
gamma *energy* (EGD, a term above) and nothing for the source function, and
FUDGE's reader skips the section. It cannot be modelled before the photon files
it refers to (MF12 and MF15, roadmap E5) are. So it travels verbatim in the
suite's provenance and the tape comes back with it, declared. No tape in the
three libraries on this machine carries one.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from kika.nuclear_data.model import (
    EVAL_LABEL,
    ConversionReport,
    EndfProvenance,
    FissionEnergyRelease,
    FissionFragmentData,
    Polynomial1d,
    Regions1d,
    fissionEnergyReleaseAxes,
)
from kika.nuclear_data.model.functions import XYs1d  # noqa: F401 - also used below
from kika.nuclear_data.model.uncertainties import Uncertainty

from .multiplicity import DEFAULT_EMAX_EV, DEFAULT_EMIN_EV, FISSION_MT

__all__ = [
    "FISSION_ENERGY_MT", "DELAYED_PHOTON_MT", "decodeMF1MT458",
    "attachFissionEnergyRelease", "fissionEnergyReleaseNode",
    "encodeMF1MT458", "encodeMF1MT460",
]

FISSION_ENERGY_MT = 458
DELAYED_PHOTON_MT = 460

#: Where MF1/460's text is kept, in the suite's provenance.
MF1MT460_KEY = "mf1mt460"

#: The energy a tabulated term is evaluated at when no thermal value was kept.
#: ENDF-102 §1.5.1.1: the LIST holds the thermal (or fission-threshold) value.
THERMAL_EV = 0.0253

#: ENDF's per-component record: (value, uncertainty) pairs for nine components.
_N_TERMS = len(FissionEnergyRelease.TERMS)
_PER_ORDER = 2 * _N_TERMS


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _polynomial(coefficients, uncertainties, domain):
    """One term as a ``polynomial1d`` with its coefficients' uncertainties."""
    standard = Polynomial1d(
        coefficients=np.asarray(uncertainties, dtype=float),
        domainMin_=float(domain[0]), domainMax_=float(domain[1]),
        axes=fissionEnergyReleaseAxes(),
    )
    return Polynomial1d(
        coefficients=np.asarray(coefficients, dtype=float),
        domainMin_=float(domain[0]), domainMax_=float(domain[1]),
        axes=fissionEnergyReleaseAxes(),
        uncertainty=Uncertainty(standard=standard),
    )


def decodeMF1MT458(section, report: Optional[ConversionReport] = None,
                   domain=None):
    """One MF1/458 section → ``(FissionEnergyRelease, report)``.

    ``domain`` is ``(EMIN, EMAX)`` in eV for the polynomials, which state no
    range of their own; pass MF1/451's EMAX. The fallback is reported.
    """
    report = report if report is not None else ConversionReport()
    lfc = int(getattr(section, "lfc", 0) or 0)

    if domain is None:
        domain = (DEFAULT_EMIN_EV, DEFAULT_EMAX_EV)
        report.approximated(
            f"MF1/458: the polynomials state no energy range and MF1/451 gave "
            f"no EMAX, so they were given the conventional "
            f"{DEFAULT_EMIN_EV:g}-{DEFAULT_EMAX_EV:g} eV domain"
        )

    if lfc == 0:
        values = [float(v) for v in getattr(section, "_coefficients", [])]
        nply = int(getattr(section, "polynomial_order", 0) or 0)
    elif lfc == 1:
        values = [float(v) for v in getattr(section, "_thermal_values", [])]
        nply = 0
    else:
        report.lost(f"MF1/458: LFC={lfc} is neither 0 nor 1; the section was not decoded")
        return None, report

    orders = len(values) // _PER_ORDER
    if orders == 0 or len(values) % _PER_ORDER or (lfc == 0 and orders != nply + 1):
        report.lost(
            f"MF1/458: the LIST holds {len(values)} values, which is not "
            f"18*(NPLY+1) for NPLY={nply}; the section was not decoded"
        )
        return None, report

    node = FissionEnergyRelease(label=EVAL_LABEL)
    for index, name in enumerate(FissionEnergyRelease.TERMS):
        coefficients = [values[order * _PER_ORDER + 2 * index] for order in range(orders)]
        uncertainties = [values[order * _PER_ORDER + 2 * index + 1] for order in range(orders)]
        setattr(node, name, _polynomial(coefficients, uncertainties, domain))

    headerFields: Dict[str, object] = {"lfc": lfc, "nply": nply}
    if lfc == 1:
        replaced: Dict[str, List[float]] = {}
        ldrv: Dict[str, int] = {}
        regions: Dict[str, List[List[int]]] = {}
        order: List[str] = []
        unplaced: List[dict] = []
        for component in getattr(section, "_tab_components", []):
            ifc = int(component["ifc"])
            pairs = [(int(nbt), int(code)) for nbt, code in component["interpolation"]]
            if not 1 <= ifc <= _N_TERMS:
                # Not a component the format defines. Kept, so the tape still
                # comes back with it, and said.
                unplaced.append({
                    "ldrv": int(component["ldrv"]), "ifc": ifc,
                    "interpolation": [list(p) for p in pairs],
                    "energies": [float(e) for e in component["energies"]],
                    "values": [float(v) for v in component["values"]],
                })
                report.lost(
                    f"MF1/458: a TAB1 declares IFC={ifc}, which names none of "
                    f"the nine components (1-9); it is kept verbatim in the "
                    f"provenance and is in no term"
                )
                continue
            name = FissionEnergyRelease.TERMS[ifc - 1]
            polynomial = getattr(node, name)
            replaced[name] = [float(polynomial.coefficients[0]),
                              float(polynomial.uncertainty.standard.coefficients[0])]
            ldrv[name] = int(component["ldrv"])
            regions[name] = [list(p) for p in pairs]
            order.append(name)
            table = Regions1d.fromEndfRegions(
                component["energies"], component["values"], pairs,
                axes=fissionEnergyReleaseAxes(),
            )
            # One region is an XYs1d: FissionEnergyReleaseSubformType
            # (gnds.xsd:1337) admits polynomial1d or XYs1d and no regions1d,
            # and FUDGE's reader refuses NR > 1 here. All three B-VIII.1 LFC=1
            # tapes are NR=1; the interpolation pairs are kept either way.
            setattr(node, name, table.function1ds[0] if len(table.function1ds) == 1 else table)
        headerFields.update(replaced=replaced, ldrv=ldrv, regions=regions,
                            order=order, unplaced=unplaced)
        if order:
            report.unsupportedNode(
                f"MF1/458 is LFC=1: {', '.join(order)} are tabulated. Each "
                f"table replaces the thermal value the LIST states for it, and "
                f"that value, its uncertainty and the TAB1's LDRV have no GNDS "
                f"node; they are kept in provenance and written back. A table "
                f"carries no uncertainty in the model",
                unaffectedScopes=('cross-sections',),
            )

    node.provenance = EndfProvenance(
        mat=getattr(section, "_mat", None),
        awr=getattr(section, "atomic_weight_ratio", None),
        za=None if getattr(section, "_za", None) is None else int(round(float(section._za))),
        headerFields=headerFields,
    )
    return node, report


def attachFissionEnergyRelease(suite, mf1, report: Optional[ConversionReport] = None):
    """Hang MF1/458 on the fission channel, and keep MF1/460 verbatim.

    Runs after :func:`~kika.endf.model_adapter.multiplicity.attachNubar`, which
    creates the channel's ``fissionFragmentData`` when MF1/455 is present; this
    adds to it rather than replacing it. Does nothing on a tape with neither
    section, which is every non-fissile evaluation.
    """
    report = report if report is not None else ConversionReport()
    sections = getattr(mf1, "mt", {}) if mf1 is not None else {}

    if DELAYED_PHOTON_MT in sections:
        section = sections[DELAYED_PHOTON_MT]
        if suite.provenance is not None:
            suite.provenance.headerFields[MF1MT460_KEY] = str(section).split("\n")
        report.unsupportedNode(
            f"MF1/460 (delayed photons, LO={getattr(section, 'lo', None)}) has "
            f"no GNDS node and refers to MF12/MF15, which kika does not read; "
            f"its records are kept verbatim in the suite's provenance and the "
            f"tape is written back with them"
        )

    if FISSION_ENERGY_MT not in sections:
        return report

    reaction = suite.findReactionByENDF_MT(FISSION_MT)
    if reaction is None:
        report.lost(
            f"MF1 carries MT458 but the evaluation has no MF3/MT{FISSION_MT}, so "
            f"there is no fission channel to hang the energy release on and it "
            f"is absent from this reactionSuite"
        )
        return report

    emax = (suite.provenance.headerFields or {}).get("emax") if suite.provenance else None
    domain = (DEFAULT_EMIN_EV, float(emax)) if emax else None
    node, report = decodeMF1MT458(sections[FISSION_ENERGY_MT], report, domain)
    if node is None:
        return report

    channel = reaction.outputChannel
    if channel.fissionFragmentData is None:
        channel.fissionFragmentData = FissionFragmentData()
    channel.fissionFragmentData.fissionEnergyReleases.append(node)
    return report


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def fissionEnergyReleaseNode(suite, label: str = EVAL_LABEL):
    """The suite's :class:`FissionEnergyRelease` under *label*, or the evaluated one.

    ``None`` when the suite has none. The inverse of the placement
    :func:`attachFissionEnergyRelease` performs.
    """
    reaction = suite.findReactionByENDF_MT(FISSION_MT)
    if reaction is None or reaction.outputChannel.fissionFragmentData is None:
        return None
    releases = list(reaction.outputChannel.fissionFragmentData.fissionEnergyReleases)
    for wanted in (label, EVAL_LABEL):
        for node in releases:
            if node.label == wanted:
                return node
    return None


def _uncertainties(polynomial, length: int, name: str, report) -> List[float]:
    standard = getattr(getattr(polynomial, "uncertainty", None), "standard", None)
    if standard is None:
        report.approximated(
            f"MF1/458: {name} carries no uncertainty, so zero is written for it"
        )
        return [0.0] * length
    out = [float(v) for v in np.asarray(standard.coefficients, dtype=float)]
    return out + [0.0] * (length - len(out))


def _table(function, name: str, kept, report):
    """``(pairs, energies, values)`` of a tabulated term, the file's pairs when still valid."""
    from .encode import usableInterpolationRegions

    if isinstance(function, XYs1d):
        function = Regions1d(function1ds=[function], axes=function.axes)
    xs, ys, pairs = function.toEndfRegions()
    kept = [tuple(p) for p in kept] if kept else None
    usable = usableInterpolationRegions(kept, len(xs))
    if kept and usable is None:
        report.warn(
            f"MF1/458: the interpolation regions kept for {name} no longer "
            f"describe its table, so they are rebuilt from the regions1d"
        )
    return [tuple(p) for p in (usable if usable is not None else pairs)], list(xs), list(ys)


def encodeMF1MT458(suite, mat: Optional[int] = None,
                   report: Optional[ConversionReport] = None, *,
                   label: str = EVAL_LABEL):
    """A ``ReactionSuite`` → its ``MF1MT458``, the section the decoder read.

    "The same section" is measured as: the flat class writes the same text from
    this as from the file -- 161 of 161 MF1/458 in ENDF/B-VIII.1, JEFF-4.0 and
    JENDL-5 (2026-10-08). Against the file's own text, 10 of those differ in
    layout only, all of it the flat writer's and none of it this function's: ZA
    and AWR rewritten in ENDF float notation, and a TAB1's interpolation pairs
    padded with blanks where the file wrote zeros.

    LFC=0 when every term is a polynomial, LFC=1 when any is tabulated -- which
    is what the format allows: a tabulated section's LIST is the NPLY=0 shape,
    so a degree>0 polynomial next to a table cannot be written and is refused.
    """
    from kika.endf.classes.mf1.mf1mt458 import MF1MT458

    report = report if report is not None else ConversionReport()
    node = fissionEnergyReleaseNode(suite, label)
    if node is None:
        raise ValueError(
            "this reactionSuite carries no fissionEnergyRelease, so MF1/458 "
            "cannot be written from it"
        )

    provenance = node.provenance if isinstance(node.provenance, EndfProvenance) else None
    header = dict(getattr(provenance, "headerFields", None) or {})
    fallback = getattr(suite, "provenance", None)

    terms = {name: getattr(node, name) for name in FissionEnergyRelease.TERMS}
    tabulated = [name for name, f in terms.items()
                 if f is not None and not isinstance(f, Polynomial1d)]
    missing = [name for name, f in terms.items() if f is None]
    if missing:
        report.approximated(
            f"MF1/458: {', '.join(missing)} absent from the model; written as zero"
        )

    orders = max([len(f.coefficients) for f in terms.values()
                  if isinstance(f, Polynomial1d)] or [1])
    if tabulated and orders > 1:
        raise ValueError(
            f"MF1/458: {', '.join(tabulated)} are tabulated (LFC=1), whose LIST "
            f"is the constant NPLY=0 shape, and another term is a polynomial of "
            f"degree {orders - 1}; ENDF-6 §1.5.1 has no format for the pair"
        )

    replaced = dict(header.get("replaced") or {})
    columns: Dict[str, List[float]] = {}
    for name, function in terms.items():
        if function is None:
            columns[name] = [0.0] * (2 * orders)
            continue
        if isinstance(function, Polynomial1d):
            coefficients = [float(c) for c in function.coefficients]
            coefficients += [0.0] * (orders - len(coefficients))
            uncertainties = _uncertainties(function, orders, name, report)
        else:
            if name in replaced:
                coefficients, uncertainties = [replaced[name][0]], [replaced[name][1]]
            else:
                coefficients = [float(np.asarray(function.evaluate(THERMAL_EV)))]
                uncertainties = [0.0]
                report.approximated(
                    f"MF1/458: {name} is tabulated and no thermal value was "
                    f"kept for it, so the LIST carries the table at "
                    f"{THERMAL_EV} eV with zero uncertainty"
                )
        columns[name] = [v for pair in zip(coefficients, uncertainties) for v in pair]

    flat: List[float] = []
    for order in range(orders):
        for name in FissionEnergyRelease.TERMS:
            flat += columns[name][2 * order: 2 * order + 2]

    section = MF1MT458()
    za = getattr(provenance, "za", None) or getattr(fallback, "za", None)
    awr = getattr(provenance, "awr", None) or getattr(fallback, "awr", None)
    section._za = float(za) if za is not None else None
    section._awr = float(awr) if awr is not None else None
    section._mat = int(mat) if mat is not None else getattr(provenance, "mat", None)

    if not tabulated:
        section._lfc = 0
        section._nply = orders - 1
        section._coefficients = flat
        section._nfc = 0
        return section, report

    keptOrder = list(header.get("order") or [])
    order = keptOrder if sorted(keptOrder) == sorted(tabulated) else \
        [name for name in FissionEnergyRelease.TERMS if name in tabulated]
    ldrv = dict(header.get("ldrv") or {})
    regions = dict(header.get("regions") or {})

    components = []
    for name in order:
        if name not in ldrv:
            report.approximated(
                f"MF1/458: {name} is tabulated and no LDRV was kept for it; "
                f"written as LDRV=1 (derived), the format's default reading"
            )
        pairs, energies, values = _table(terms[name], name, regions.get(name), report)
        components.append({
            "ldrv": int(ldrv.get(name, 1)),
            "ifc": FissionEnergyRelease.TERMS.index(name) + 1,
            "interpolation": pairs, "energies": energies, "values": values,
        })
    for extra in header.get("unplaced") or []:
        components.append({
            "ldrv": extra["ldrv"], "ifc": extra["ifc"],
            "interpolation": [tuple(p) for p in extra["interpolation"]],
            "energies": list(extra["energies"]), "values": list(extra["values"]),
        })

    section._lfc = 1
    section._nply = 0
    section._thermal_values = flat
    section._tab_components = components
    section._nfc = len(components)
    return section, report


def encodeMF1MT460(suite, mat: Optional[int] = None,
                   report: Optional[ConversionReport] = None):
    """MF1/460 from the records the decoder kept, or ``None`` when there are none."""
    from kika.endf.parsers.parse_mf1 import parse_mt460

    report = report if report is not None else ConversionReport()
    provenance = getattr(suite, "provenance", None)
    lines = (getattr(provenance, "headerFields", None) or {}).get(MF1MT460_KEY)
    if not lines:
        return None, report
    # The SEND record (MT=0) is the reader's, not the section's.
    section = parse_mt460([line for line in lines if line[72:75].strip() == str(DELAYED_PHOTON_MT)])
    if mat is not None:
        section._mat = int(mat)
    return section, report


# ---------------------------------------------------------------------------
# MF5/MT455: the delayed-neutron families' spectra (roadmap E2)
# ---------------------------------------------------------------------------

def _familyMultiplicity(nubarForm, weight, label: str, report):
    """``p_k(E) * nu_d(E)``: the family's multiplicity, as FUDGE builds it.

    Exact when ``p_k`` is a constant -- every ENDF/B-VIII.1 MT455 on this
    machine (LF=5, NK=6, one value per family) -- because then the product is
    the delayed nu-bar's own table scaled. Otherwise (JEFF-4.0's NK=8, ``p_k``
    a histogram/lin-lin mix) the product of two piecewise functions is not
    piecewise of either law, so it is sampled on the union of both grids,
    steps kept, and joined lin-lin: an approximation, said.
    """
    from kika.algebra import interval_laws, sample_on_union, union
    from kika.nuclear_data.model import Multiplicity, multiplicityAxes

    nx, ny, npairs = (nubarForm.toEndfRegions() if isinstance(nubarForm, Regions1d)
                      else Regions1d(function1ds=[nubarForm]).toEndfRegions())
    wform = weight if isinstance(weight, Regions1d) else Regions1d(function1ds=[weight])
    wx, wy, wpairs = wform.toEndfRegions()
    wy = np.asarray(wy, dtype=float)
    if wy.size and np.all(wy == wy[0]):
        table = Regions1d.fromEndfRegions(nx, np.asarray(ny, dtype=float) * float(wy[0]),
                                          npairs, axes=multiplicityAxes())
        function = table.function1ds[0] if len(table.function1ds) == 1 else table
        return Multiplicity(form=function)

    nx, ny = np.asarray(nx, dtype=float), np.asarray(ny, dtype=float)
    wx = np.asarray(wx, dtype=float)
    grid = union([nx, wx[(wx >= nx[0]) & (wx <= nx[-1])]], steps=wx[1:-1])
    values = (sample_on_union(nx, ny, interval_laws(nx.size, npairs), grid)
              * sample_on_union(wx, wy, interval_laws(wx.size, wpairs), grid))
    report.approximated(
        f"delayedNeutron {label!r}: its multiplicity is p_k(E) * nu_d(E) with a "
        f"p_k that is not constant, sampled on the {grid.size}-point union of "
        f"both grids and joined lin-lin -- the product of two tables is not a "
        f"table of either law"
    )
    return Multiplicity(form=XYs1d(xs=grid, ys=values, axes=multiplicityAxes()))


def attachDelayedSpectra(suite, mf5mt, report: Optional[ConversionReport] = None):
    """MF5/MT455 → each §18.4 ``delayedNeutron``'s product: spectrum and multiplicity.

    ENDF writes the NK precursor families' spectra as the NK subsections of
    MF5/455 and their decay constants as the NNF of MF1/455, in the same order.
    GNDS (and FUDGE) put each spectrum on its family's product as an
    ``uncorrelated`` -- isotropic in the lab, ENDF's convention for a spectrum
    stated without an MF4 -- and give the product the multiplicity
    ``p_k(E) * nu_d(E)``. The §21.3 delayed ``multiplicitySum`` then lists the
    families as its summands, which it could not while they held nothing.

    The section's bytes are kept on ``delayedNeutrons.provenance`` and written
    back from there. Returns ``False`` (and reports) when the families cannot be
    placed, so the caller can fall back to the old declared loss.
    """
    from kika.nuclear_data.model import (Add, Distribution, Frame, Isotropic2d,
                                         Product, Uncorrelated)

    from .energy import decodeMF5Families
    from .multiplicity import DELAYED_NUBAR_LABEL, delayedNeutronMultiplicityHref

    report = report if report is not None else ConversionReport()
    reaction = suite.findReactionByENDF_MT(FISSION_MT)
    data = getattr(getattr(reaction, "outputChannel", None), "fissionFragmentData", None)
    if data is None or not len(data.delayedNeutrons):
        return False, report
    total = suite.sums.multiplicitySums.byENDF_MT(455)

    families, provenance, report = decodeMF5Families(mf5mt, report)
    data.delayedNeutrons.provenance = provenance
    if not families:
        return True, report
    if len(families) != len(data.delayedNeutrons):
        report.lost(
            f"MF5/MT455 states {len(families)} family spectra and MF1/455 "
            f"{len(data.delayedNeutrons)} decay constants; which spectrum is "
            f"whose is not in the file, so none is placed. The section is "
            f"written back from its bytes"
        )
        return True, report

    weights = [w for w, _ in families]
    sums = np.zeros(1)
    try:
        grid = np.unique(np.concatenate([
            (w.toEndfRegions()[0] if isinstance(w, Regions1d) else w.xs) for w in weights]))
        sums = np.sum([np.asarray(w.evaluate(grid, outOfRange="hold")) for w in weights], axis=0)
    except Exception:  # pragma: no cover - a weight that cannot be evaluated
        pass
    if sums.size and np.max(np.abs(sums - 1.0)) > 1e-6:
        report.warn(
            f"MF5/MT455: the family weights p_k(E) sum to between "
            f"{sums.min():.7g} and {sums.max():.7g}, not 1. They are kept as "
            f"stated -- FUDGE renormalises them, kika does not change the "
            f"evaluator's numbers"
        )

    nubar = total.multiplicity.form if total is not None and total.multiplicity else None
    links = []
    for family, (weight, form) in zip(data.delayedNeutrons, families):
        product = family.product or Product(pid="n", label="n")
        family.product = product
        if nubar is not None:
            product.multiplicity = _familyMultiplicity(nubar, weight, family.label, report)
            links.append(Add(delayedNeutronMultiplicityHref(family.label)))
        if product.distribution is None:
            product.distribution = Distribution()
        product.distribution[EVAL_LABEL] = Uncorrelated(
            angular=Isotropic2d(), energy=form, productFrame=Frame.lab)
    if total is not None and links:
        total.summands.summands = links
    report.approximated(
        f"MF5/MT455: {len(families)} family spectra placed on their "
        f"delayedNeutron products with an isotropic lab angular half -- ENDF "
        f"states no angle for them, and isotropy is §5's convention"
    )
    return True, report
