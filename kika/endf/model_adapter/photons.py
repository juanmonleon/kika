"""MF12-15 ↔ the photon products of a reaction (roadmap E5b, E5d).

ENDF states a reaction's photons in four files: MF12 (or MF13) lists them with
a multiplicity (or a production cross section) each, MF14 gives their angular
distributions and MF15 the spectrum of the continuum among them. GNDS states
the same thing as **products**: one ``photon`` per line, its multiplicity on the
product and an ``uncorrelated`` distribution whose energy half is a
``discreteGamma``, a ``primaryGamma`` or the tabulated continuum. That is the
mapping here, and it is FUDGE's (``ENDF_ITYPE_0_Misc.py``) and that of the 558
distributed ENDF/B-VIII.1-GNDS files:

======================================  =========================================
ENDF                                    model
======================================  =========================================
MF12 LO=1, EG>0, LP=0/1                 ``Uncorrelated(energy=DiscreteGamma)``
MF12 LO=1, EG>0, LP=2                   ``Uncorrelated(energy=PrimaryGamma)``
MF12 LO=1, EG=0, LF=1 + MF15            ``Uncorrelated(energy=XYs2d)``
MF12/MF13 total (NK>1)                  ``multiplicitySum`` with the photons as summands
MF13 (σ_γ)                              the same products, multiplicity = σ_γ / σ
MF14 LI=1, or an isotropic CONT         ``Uncorrelated(angular=Isotropic2d)``
MF14 LTT=1 / LTT=2                      ``XYs2d`` of ``Legendre`` (a_0 = 1 written) / of ``XYs1d``
MF12 LO=2 (+MF14 LI=1)                  the level's PoPs ``decayData`` + ``branching1d/3d``
======================================  =========================================

**Where a photon hangs.** On the reaction's output channel — except for a
discrete-level reaction (MT51-90, 600-648, …), where the photons are the decay
of the excited residual and go into *its* output channel, as FUDGE puts them
(``residuals.attachResiduals`` builds that channel; the placeholder ``photon``
it puts there, with no multiplicity, is replaced). The photons of a **sum**
(MT3, MT4, MT103-107 when they are ``crossSectionSum``) go to an
``orphanProduct`` whose cross section is a ``reference`` to the sum, as in
NNDC's GNDS of N-14 (decision J4 for MT3). A product is matched to its MF14
entry by ``(EG, ES)``, never by position.

**MF13 is reached by division, and that is an approximation** (``conflicts``
§2.6; decided 2026-10-08, roadmap §3.2, completed by J2 and J5). The
multiplicity is σ_γ/σ on the union of MF13's grid and σ's points inside it,
both made lin-lin by :mod:`kika.algebra` first; a point where σ = 0 gets 0 and
is declared. On the way back the section is the **original bytes** while the
SHA-256 of every multiplicity and of σ still match what was read; once either
changed, σ_γ = y·σ on MF13's own grid, declared once per section.

**MF12 LO=2 is a level's decay, and it goes to PoPs** (roadmap E5c, decision
J1 = A). The level is the residual of the reaction's two-body channel; its PoPs
entry gets ``nucleus/energy`` = ES_NS and one electromagnetic ``decayMode`` per
transition (TP, and GP as ``photonEmissionProbabilities`` when LG=2), ending on
the level whose energy is ES_i -- found exactly, or within the format's digits,
and never guessed. The residual's decay channel carries a ``branching1d``/
``branching3d`` photon that points at it. On the way back the section is its
own bytes while the cascade's SHA-256 matches, and rebuilt from PoPs (with LP
and NS as read, not FUDGE's LP=0) once it changed.

**What this module does not model, and keeps whole.** An LO=2 whose MF14 is
anisotropic or whose ES_i names no level of the evaluation, and the photons of
an MT with no cross section to hang them on (MF13 on MT28 and MT32 of N-14 has
no MF3: FUDGE drops those silently).
Their sections travel as text in the suite's provenance under
:data:`PHOTONS_VERBATIM_KEY` and are written back as they came, declared.

**What is ENDF's alone** — ``ES`` (the level a line comes from), ``LP=0`` vs
``LP=1``, the order of the subsections, the MF15 weight p(E) and every unused
header field — goes to the host's provenance under ``"mf12"``/``"mf13"``,
``"mf14"`` and ``"mf15"``, as primitives, beside MF3's and MF6's. For GNDS,
FUDGE's ``ENDFconversionFlags`` ``MF13,ESk=`` is written on each MF13 photon so
FUDGE can return the section (decision J3).

**Deviations from FUDGE, declared.** The domain of a ``discreteGamma`` is that of
its multiplicity, which is what the section states; FUDGE uses the domain of the
reaction's cross section (J6). ``LP`` is kept rather than rebuilt from ``ES``.
MF13's division grid is MF13 ∪ σ (FUDGE: MF13 only) and a zero of σ is
declared rather than skipped (J5).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from kika.nuclear_data.model import (
    EVAL_LABEL,
    ConversionReport,
    DiscreteGamma,
    Distribution,
    EndfProvenance,
    Frame,
    Isotropic2d,
    Legendre,
    Multiplicity,
    PrimaryGamma,
    Regions1d,
    Uncorrelated,
    XYs1d,
    XYs2d,
    angularAxes,
    energyAxes,
    fromEndfTab2,
    multiplicityAxes,
    toEndfTab2,
)

__all__ = ["attachPhotons", "encodePhotonSections", "PHOTONS_VERBATIM_KEY",
           "PHOTON_FILES"]

#: The photon production files, in tape order.
PHOTON_FILES = (12, 13, 14, 15)

#: Where the sections the model does not carry yet are kept, on the suite's
#: provenance: ``{"MF/MT": [lines]}``.
PHOTONS_VERBATIM_KEY = "photonsVerbatim"

#: How close two ``(EG, ES)`` must be to name the same photon in MF12 and MF14,
#: in eV. FUDGE's tolerance (``ENDF_ITYPE_0_Misc.py`` MF14 matching).
_MATCH_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# Small conversions
# ---------------------------------------------------------------------------

def _function1d(interp, x, y, axes=None):
    """A TAB1 → an :class:`XYs1d`, or a :class:`Regions1d` when NR > 1."""
    pairs = [(int(nbt), int(code)) for nbt, code in interp]
    x = np.asarray(x, dtype=float)
    if not pairs and x.size:
        pairs = [(int(x.size), 2)]
    function = Regions1d.fromEndfRegions(x, np.asarray(y, dtype=float), pairs,
                                         axes=axes)
    if len(function.function1ds) == 1:
        function = function.function1ds[0]
        function.axes = axes
    return function


def _tab1Of(function) -> Tuple[List[Tuple[int, int]], List[float], List[float]]:
    from kika.nuclear_data.model.functions.simple import Constant1d

    if isinstance(function, Constant1d):
        # A constant multiplicity (GNDS writes one for a photon of a suite that
        # never saw ENDF): its two end points, lin-lin -- FUDGE's
        # toPointwise_withLinearXYs, which is how its toENDF6 writes it.
        lo, hi, c = float(function.domainMin), float(function.domainMax), float(function.constant)
        return [(2, 2)], [lo, hi], [c, c]
    x, y, pairs = function.toEndfRegions()
    return ([(int(a), int(b)) for a, b in pairs],
            [float(v) for v in x], [float(v) for v in y])


def _multiplicityFromTable(table) -> Multiplicity:
    function = _function1d(table.interp, table.x, table.y, axes=multiplicityAxes())
    function.label = EVAL_LABEL
    return Multiplicity(form=function)


def _pad(section) -> dict:
    return {"pairs": section.pad.pairs, "values": section.pad.values,
            "interp": section.pad.interp}


def _head(section) -> dict:
    return {"mat": section._mat, "za": section._za, "awr": section._awr}


def _digest(function) -> str:
    """SHA-256 of a table's abscissae and values, to tell "untouched" on the way back."""
    _, x, y = _tab1Of(function)
    digest = hashlib.sha256()
    digest.update(np.asarray(x, dtype=float).tobytes())
    digest.update(np.asarray(y, dtype=float).tobytes())
    return digest.hexdigest()


def _linlin(function):
    """``(x, y)`` of *function* re-expressed lin-lin by :mod:`kika.algebra`."""
    from kika.algebra.laws import interval_laws
    from kika.algebra.refine import to_linlin

    pairs, x, y = _tab1Of(function)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return to_linlin(x, y, interval_laws(x.size, pairs))


def _divide(table, sigma) -> Tuple[np.ndarray, np.ndarray, int]:
    """σ_γ / σ as a lin-lin table on MF13's grid ∪ σ's points inside it (J5).

    Returns ``(x, y, zeros)`` with *zeros* the number of points where σ = 0,
    which get a multiplicity of 0 and are declared by the caller.
    """
    from kika.algebra.evaluate import sample_on_union
    from kika.algebra.grid import discontinuities, union

    gx, gy = _linlin(_function1d(table.interp, table.x, table.y))
    sx, sy = _linlin(sigma)
    lo, hi = float(gx[0]), float(gx[-1])
    inside = (sx > lo) & (sx < hi)
    steps = [float(v) for v in discontinuities(gx)]
    steps += [float(v) for v in discontinuities(sx) if lo < v < hi]
    u = union([gx, sx[inside]], steps=steps)
    numerator = sample_on_union(gx, gy, 2, u)
    denominator = sample_on_union(sx, sy, 2, u)
    zero = denominator == 0.0
    y = np.where(zero, 0.0, numerator / np.where(zero, 1.0, denominator))
    return u, y, int(zero.sum())


def _multiply(function, sigma, x) -> List[float]:
    """y·σ read at MF13's own abscissae *x* — a repeated one is a step."""
    from kika.algebra.evaluate import sample_on_union

    fx, fy = _linlin(function)
    sx, sy = _linlin(sigma)
    x = np.asarray(x, dtype=float)
    return [float(v) for v in sample_on_union(fx, fy, 2, x) * sample_on_union(sx, sy, 2, x)]


# ---------------------------------------------------------------------------
# Hosts: a reaction, or an orphan product of a sum
# ---------------------------------------------------------------------------

class _Host:
    """Where one MT's photons go, and the cross section MF13 divides by."""

    def __init__(self, node, sigmaOwner, path: str, kind: str):
        self.node = node              # a Reaction, or an orphan-product Reaction
        self.sigmaOwner = sigmaOwner  # whose crossSection σ is
        self.path = path              # xPath of node
        self.kind = kind              # "reaction" or "orphan"

    @property
    def sigma(self):
        try:
            return self.sigmaOwner.crossSection[EVAL_LABEL]
        except (KeyError, TypeError, AttributeError):
            return None


def _formTag(form) -> str:
    return {"XYs1d": "XYs1d", "Regions1d": "regions1d"}.get(type(form).__name__,
                                                             type(form).__name__)


def _hostFor(suite, mt) -> Optional[_Host]:
    from kika.nuclear_data.model import OutputChannel, Reaction, ReactionId
    from kika.nuclear_data.model.cross_section_forms import Reference

    reaction = suite.findReactionByENDF_MT(mt)
    if reaction is None:
        return None
    if any(reaction is r for r in suite.reactions):
        return _Host(reaction, reaction,
                     f"/reactionSuite/reactions/reaction[@label='{reaction.label}']",
                     "reaction")
    if not any(reaction is s for s in suite.sums):
        return None
    sigma = _Host(None, reaction, "", "").sigma
    if sigma is None:
        return None
    for orphan in suite.orphanProducts:
        if orphan.ENDF_MT == mt:
            break
    else:
        label = f"Orphan product {len(suite.orphanProducts) + 1}"
        orphan = Reaction(id=ReactionId(label=label, ENDF_MT=int(mt)),
                          outputChannel=OutputChannel(genre="NBody"))
        orphan.crossSection[EVAL_LABEL] = Reference(
            href=(f"/reactionSuite/sums/crossSectionSums/crossSectionSum"
                  f"[@label='{reaction.label}']/crossSection/{_formTag(sigma)}"
                  f"[@label='{EVAL_LABEL}']"),
            label=EVAL_LABEL)
        provenance = getattr(reaction, "provenance", None)
        orphan.provenance = EndfProvenance(
            mat=getattr(provenance, "mat", None), awr=getattr(provenance, "awr", None),
            za=getattr(provenance, "za", None))
        suite.orphanProducts.append(orphan)
    return _Host(orphan, reaction,
                 f"/reactionSuite/orphanProducts/orphanProduct[@label='{orphan.label}']",
                 "orphan")


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def attachPhotons(suite, endf, report: Optional[ConversionReport] = None):
    """Put the photons of every MF12-15 section on the products of their host.

    Runs after :func:`~kika.endf.model_adapter.residuals.attachResiduals`, which
    is what builds the decay channel a discrete level's photons go into.
    """
    report = report if report is not None else ConversionReport()
    files = {mf: getattr(endf.mf.get(mf), "mt", {}) for mf in PHOTON_FILES
             if hasattr(endf, "mf") and mf in endf.mf}
    mts = sorted({mt for sections in files.values() for mt in sections})
    for mt in mts:
        sections = {mf: files[mf][mt] for mf in files if mt in files[mf]}
        if 12 in sections and sections[12].lo == 2:
            why = _attachCascade(suite, sections, files.get(12, {}), mt, report)
            if why is not None:
                _keepVerbatim(suite, sections, mt, why, report)
            continue
        why = _whyNotModelled(sections)
        host = None
        if why is None:
            host = _hostFor(suite, mt)
            if host is None:
                why = ("there is no cross section for these photons to hang from "
                       "(an MT with no MF3, or a sum with no values), so neither a "
                       "reaction nor an orphanProduct can carry them")
        if why is None:
            why = _attach(suite, host, sections, mt, report)
        if why is not None:
            _keepVerbatim(suite, sections, mt, why, report)
    return report


def _whyNotModelled(sections) -> Optional[str]:
    production = sections.get(12) or sections.get(13)
    if 12 in sections and 13 in sections:
        return "both MF12 and MF13 state this MT's photons, which ENDF-6 does not allow"
    if production is None:
        return "MF14/MF15 with no MF12 or MF13 to say which photons they describe"
    if 15 in sections and len(sections[15].spectra) != 1:
        return (f"MF15 states {len(sections[15].spectra)} partial distributions; "
                f"one per section is the only shape in the three libraries and "
                f"the only one modelled")
    if sum(1 for p in production.photons if p.lf == 1) > (1 if 15 in sections else 0):
        return "a continuum photon (LF=1) with no MF15 spectrum to describe it"
    return None


def _keepVerbatim(suite, sections, mt, why, report) -> None:
    provenance = getattr(suite, "provenance", None)
    files = "/".join(str(mf) for mf in sorted(sections))
    if provenance is None:
        report.lost(f"MT{mt}: MF{files} could not be modelled ({why}) and the "
                    f"suite has no provenance to keep it in, so it is lost")
        return
    kept = provenance.headerFields.setdefault(PHOTONS_VERBATIM_KEY, {})
    for mf, section in sorted(sections.items()):
        kept[f"{mf}/{mt}"] = str(section).split("\n")
    report.unsupportedNode(
        f"MT{mt}: the photons of MF{files} are not in this reactionSuite because "
        f"{why}. The sections are kept verbatim in the suite's provenance and an "
        f"ENDF tape written from it carries them unchanged"
    )


def _channel(host: _Host, mt):
    """``(channel, residual label, xPath of the channel)`` the photons go into."""
    from .residuals import levelSeries

    node = host.node
    if host.kind == "reaction" and levelSeries(int(mt)) is not None:
        for product in node.outputChannel.products:
            if product.pid != "photon" and product.outputChannel is not None:
                return (product.outputChannel, product.label,
                        f"{host.path}/outputChannel/products/product"
                        f"[@label='{product.label}']/outputChannel")
    return node.outputChannel, None, f"{host.path}/outputChannel"


def _match(pool: List[Tuple[float, float, object]], eg: float, es: float):
    """The first unused entry of *pool* whose ``(EG, ES)`` is this photon's."""
    for k, (eg14, es14, data) in enumerate(pool):
        if data is None:
            continue
        if abs(eg14 - eg) <= _MATCH_TOLERANCE and abs(es14 - es) <= _MATCH_TOLERANCE:
            pool[k] = (eg14, es14, None)
            return data
    return None


def _angularOf(mf14) -> Optional[List[Tuple[float, float, object]]]:
    """MF14 as ``[(EG, ES, angular form)]``, or ``None`` when every photon is isotropic."""
    if mf14 is None or mf14.li == 1:
        return None
    entries = [(p.eg, p.es, Isotropic2d(productFrame=Frame.lab)) for p in mf14.isotropic]
    for photon in mf14.anisotropic:
        axes = angularAxes()
        functions = []
        for k, node in enumerate(photon.nodes):
            if mf14.ltt == 2:
                function = _function1d(node.interp, node.mu, node.p)
            else:
                coefficients = np.empty(len(node.coefficients) + 1, dtype=float)
                coefficients[0] = 1.0
                coefficients[1:] = node.coefficients
                function = Legendre(coefficients=coefficients)
            function.outerDomainValue = float(node.energy)
            function.index = k
            functions.append(function)
        entries.append((photon.eg, photon.es,
                        fromEndfTab2(functions, photon.tab2_interp, axes=axes)))
    return entries


def _continuum(mf15):
    spectrum = mf15.spectra[0]
    axes = energyAxes()
    functions = []
    for k, table in enumerate(spectrum.distributions):
        function = _function1d(table.interp, table.x, table.y)
        function.outerDomainValue = float(table.c2)
        function.index = k
        functions.append(function)
    return fromEndfTab2(functions, spectrum.tab2_interp, axes=axes)


def _attach(suite, host: _Host, sections, mt, report) -> Optional[str]:
    """MF12 LO=1 or MF13 (+MF14, MF15) → photon products. ``None`` on success."""
    from kika.nuclear_data.model import Product
    from kika.nuclear_data.model.sums import Add, MultiplicitySum

    from .residuals import _uniqueLabel

    mf = 13 if 13 in sections else 12
    production, mf14, mf15 = sections[mf], sections.get(14), sections.get(15)
    node = host.node
    provenance = getattr(node, "provenance", None)
    if provenance is None or getattr(provenance, "sourceFormat", None) != "endf":
        return "its reaction carries no ENDF provenance to keep the bookkeeping in"

    sigma = host.sigma
    if mf == 13 and sigma is None:
        return "MF13 is a cross section to divide by σ, and this MT has no σ form"

    angular = _angularOf(mf14)
    if angular is not None and len(angular) != len(production.photons):
        return (f"MF14 lists {len(angular)} photons and MF{mf} "
                f"{len(production.photons)}, so they cannot be paired")

    multiplicities, zeros = [], 0
    for table in production.photons + ([production.total] if production.total else []):
        if mf == 12:
            multiplicities.append(_multiplicityFromTable(table))
            continue
        x, y, n = _divide(table, sigma)
        zeros += n
        function = XYs1d(xs=x, ys=y, axes=multiplicityAxes(), label=EVAL_LABEL)
        multiplicities.append(Multiplicity(form=function))
    total = multiplicities.pop() if production.total else None

    forms = []
    for index, (photon, multiplicity) in enumerate(zip(production.photons, multiplicities)):
        if angular is None:
            angle = Isotropic2d(productFrame=Frame.lab) if mf14 is not None else None
        else:
            angle = _match(angular, photon.eg, photon.es)
            if angle is None:
                return (f"photon {index} (EG={photon.eg}, ES={photon.es}) has no "
                        f"MF14 entry with the same (EG, ES)")
        if photon.lf == 1:
            energy = _continuum(mf15)
        else:
            xs = multiplicity.form.xs if hasattr(multiplicity.form, "xs") else photon.x
            kind = PrimaryGamma if photon.lp == 2 else DiscreteGamma
            energy = kind(value=float(photon.eg), domainMin=float(xs[0]),
                          domainMax=float(xs[-1]), axes=energyAxes())
        forms.append(Uncorrelated(angular=angle, energy=energy, label=EVAL_LABEL,
                                  productFrame=Frame.lab))
    if mf14 is None:
        report.lost(f"MT{mt}: MF{mf} has no MF14 beside it, so the photons' angular "
                    f"halves are absent and their uncorrelated nodes incomplete")
    if zeros:
        report.approximated(
            f"MT{mt}: MF13 divided by σ (decision J5) meets σ = 0 at {zeros} "
            f"point(s); the multiplicity there is written as 0")
    if mf == 13:
        report.approximated(
            f"MT{mt}: MF13 states production cross sections and GNDS a "
            f"multiplicity, so each photon's is σ_γ/σ, lin-lin on MF13's grid ∪ "
            f"σ's (conflicts §2.6). The ENDF section comes back as its own bytes "
            f"while neither changes")

    channel, hostLabel, channelPath = _channel(host, mt)
    # The residual's decay channel arrives with a placeholder photon: no
    # multiplicity, `unspecified`. MF12 is what that placeholder stood for.
    channel.products.products[:] = [
        p for p in channel.products.products
        if not (p.pid == "photon" and p.multiplicity is None)]
    taken = {p.label for p in channel.products}
    labels = []
    for index, (multiplicity, form) in enumerate(zip(multiplicities, forms)):
        existing = next((p for p in channel.products
                         if p.pid == "photon" and p.label not in labels), None)
        if existing is not None and index == 0:
            # MF6 got here first with a LAW=0 photon (ENDF/B-VIII.1 U-235, U-238
            # and Pu-239 MT18 state both). One photon, not two: its multiplicity
            # and distribution are MF12/14/15's, and MF6's subsection stays in
            # MF6's provenance, which is what writes it back.
            report.warn(
                f"MT{mt}: MF6 already gave this reaction a photon ({existing.label!r}, "
                f"LAW=0) and MF12 describes it; the multiplicity and the "
                f"distribution are MF12's, MF14's and MF15's")
            product = existing
        else:
            label = _uniqueLabel("photon", taken)
            taken.add(label)
            product = Product(pid="photon", label=label)
            channel.products.products.append(product)
        labels.append(product.label)
        product.multiplicity = multiplicity
        product.distribution = Distribution()
        product.distribution[EVAL_LABEL] = form

    if total is not None:
        summ = MultiplicitySum(label=f"{node.label} total gamma multiplicity",
                               multiplicity=total, ENDF_MT=int(mt))
        for label in labels:
            summ.summands.append(Add(
                href=f"{channelPath}/products/product[@label='{label}']/multiplicity"))
        suite.sums.multiplicitySums.append(summ)

    fields = {
        **_head(production), "l1": production._head_l1(), "l2": production._l2,
        "n2": production._n2, "pad": _pad(production), "host": hostLabel,
        "total": (None if production.total is None else
                  [production.total.c1, production.total.c2,
                   production.total.l1, production.total.l2]),
        "photons": [{"label": label, "eg": p.eg, "es": p.es, "lp": p.lp, "lf": p.lf}
                    for label, p in zip(labels, production.photons)],
    }
    if mf == 13:
        products = {p.label: p for p in channel.products}
        fields["lines"] = str(production).split("\n")
        fields["sigma"] = _digest(sigma)
        fields["digests"] = [_digest(products[label].multiplicity.form) for label in labels]
        fields["totalDigest"] = _digest(total.form) if total is not None else None
        fields["grids"] = [list(p.x) for p in production.photons]
        fields["totalGrid"] = list(production.total.x) if production.total else None
        fields["interps"] = [[list(i) for i in p.interp] for p in production.photons]
        fields["totalInterp"] = ([list(i) for i in production.total.interp]
                                 if production.total else None)
    _flagPhotons(suite, channelPath, labels, production, mf)
    provenance.headerFields[f"mf{mf}"] = fields
    if mf14 is not None:
        provenance.headerFields["mf14"] = _mf14Fields(mf14, production, labels)
    if mf15 is not None:
        spectrum = mf15.spectra[0]
        continuum = next(label for label, p in zip(labels, production.photons) if p.lf == 1)
        provenance.headerFields["mf15"] = {
            **_head(mf15), "l1": mf15._l1, "l2": mf15._l2, "n2": mf15._n2,
            "pad": _pad(mf15), "label": continuum,
            "weight": {"c1": spectrum.weight.c1, "c2": spectrum.weight.c2,
                       "l1": spectrum.weight.l1, "l2": spectrum.weight.l2,
                       "interp": [list(p) for p in spectrum.weight.interp],
                       "x": list(spectrum.weight.x), "y": list(spectrum.weight.y)},
            "tab2": list(spectrum.tab2_fields),
            "nodes": [[t.c1, t.l1, t.l2] for t in spectrum.distributions],
        }
    return None


def _flagPhotons(suite, channelPath, labels, production, mf: int) -> None:
    """FUDGE's flags on each photon (decision J3), as its ENDF->GNDS writes them
    (``ENDF_ITYPE_0_Misc.addGammaProduct``): ``MF13`` on a photon of MF13, and
    ``ESk=`` on any photon of MF12 or MF13 whose origin level ES is not 0 -- so
    FUDGE, and kika's own GNDS->ENDF (E5e), can give both back."""
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags

    wanted = []
    for label, photon in zip(labels, production.photons):
        items = (["MF13"] if mf == 13 else []) + (
            [f"ESk={float(photon.es)!r}"] if photon.es else [])
        if items:
            wanted.append((f"{channelPath}/products/product[@label='{label}']", ",".join(items)))
    if not wanted:
        return
    flags = EndfConversionFlags.of(suite)
    if flags is None:
        flags = EndfConversionFlags()
        suite.applicationData.entries.append(flags)
    flags.conversions.extend(wanted)


def _mf14Fields(mf14, production, labels) -> dict:
    """MF14's bookkeeping, each entry naming the product it describes."""
    fields = {**_head(mf14), "li": mf14._li, "ltt": mf14._ltt, "nk": mf14._nk,
              "ni": mf14._ni, "pad": _pad(mf14), "isotropic": [], "anisotropic": []}
    if mf14.li == 1:
        return fields
    pool = [(p.eg, p.es, label) for label, p in zip(labels, production.photons)]
    for photon in mf14.isotropic:
        fields["isotropic"].append({"label": _match(pool, photon.eg, photon.es),
                                    "eg": photon.eg, "es": photon.es,
                                    "rest": list(photon.rest)})
    for photon in mf14.anisotropic:
        fields["anisotropic"].append({
            "label": _match(pool, photon.eg, photon.es),
            "eg": photon.eg, "es": photon.es, "l1": photon.l1, "l2": photon.l2,
            "nodes": [[n.c1, n.l1, n.l2, n.n2] for n in photon.nodes]})
    return fields


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def _padStyle(fields):
    from kika.endf.utils import PadStyle

    pad = fields.get("pad") or {}
    return PadStyle(**pad) if pad else PadStyle()


def _channelOf(node, hostLabel):
    if hostLabel is None:
        return node.outputChannel
    for product in node.outputChannel.products:
        if product.label == hostLabel:
            return product.outputChannel
    raise ValueError(f"{node.label}: the photons were read into the decay of "
                     f"{hostLabel!r}, which this reaction no longer has")


def _productByLabel(channel, label, mt):
    for product in channel.products:
        if product.label == label:
            return product
    raise ValueError(f"MT{mt}: the provenance lists photon {label!r} and the "
                     f"channel has no product with that label")


def _evaluated(product):
    distribution = product.distribution
    return distribution.get(EVAL_LABEL) if distribution is not None else None


def _sigmaOf(suite, node):
    """σ of *node*: its own, or the sum its ``reference`` points at."""
    from kika.nuclear_data.model.cross_section_forms import Reference

    form = node.crossSection.get(EVAL_LABEL) if hasattr(node.crossSection, "get") else None
    if isinstance(form, Reference):
        target = suite.findReactionByENDF_MT(node.ENDF_MT)
        return target.crossSection[EVAL_LABEL] if target is not None else None
    return form


def _encodeProduction(suite, node, fields, mf, mt, mat, report):
    from kika.endf.classes.mf12.base import MF12MT, MF13MT, PhotonTable
    from kika.endf.parsers.parse_photons import parse_mf13_mt

    channel = _channelOf(node, fields.get("host"))
    products = [_productByLabel(channel, r["label"], mt) for r in fields["photons"]]
    summ = suite.sums.multiplicitySums.byENDF_MT(mt) if fields.get("total") else None
    if fields.get("total") is not None and (summ is None or summ.multiplicity is None):
        raise ValueError(f"MT{mt}: MF{mf} states a total and the suite has no "
                         f"multiplicitySum for this MT")

    if mf == 13:
        sigma = _sigmaOf(suite, node)
        same = (sigma is not None and _digest(sigma) == fields["sigma"]
                and [_digest(p.multiplicity.form) for p in products] == fields["digests"]
                and (summ is None or _digest(summ.multiplicity.form) == fields["totalDigest"]))
        if same:
            body = [line for line in fields["lines"] if line[72:75].strip() not in ("", "0")]
            section = parse_mf13_mt(body, mt)
            if mat is not None:
                section._mat = int(mat)
            return section
        report.approximated(
            f"MT{mt}: a photon multiplicity or σ changed, so MF13 is rebuilt as "
            f"σ_γ = y·σ on its own grid (conflicts §2.6, decision J2)")
        section = MF13MT(number=mt, _l1=int(fields["l1"]))
    else:
        section = MF12MT(number=mt, _lo=1)
    section._za, section._awr = fields["za"], fields["awr"]
    section._l2, section._n2 = int(fields["l2"]), int(fields["n2"])
    section._mat = mat if mat is not None else fields["mat"]
    section.pad = _padStyle(fields)

    for k, (record, product) in enumerate(zip(fields["photons"], products)):
        energy = getattr(_evaluated(product), "energy", None)
        eg = (float(energy.value) if isinstance(energy, (DiscreteGamma, PrimaryGamma))
              else record["eg"])
        if mf == 13:
            x = fields["grids"][k]
            interp = [tuple(i) for i in fields["interps"][k]]
            y = _multiply(product.multiplicity.form, sigma, x)
        else:
            interp, x, y = _tab1Of(product.multiplicity.form)
        section.photons.append(PhotonTable(c1=eg, c2=record["es"], l1=int(record["lp"]),
                                           l2=int(record["lf"]), interp=interp, x=x, y=y))
    if fields.get("total") is not None:
        if mf == 13:
            x = fields["totalGrid"]
            interp = [tuple(i) for i in fields["totalInterp"]]
            y = _multiply(summ.multiplicity.form, sigma, x)
        else:
            interp, x, y = _tab1Of(summ.multiplicity.form)
        c1, c2, l1, l2 = fields["total"]
        section.total = PhotonTable(c1=c1, c2=c2, l1=int(l1), l2=int(l2),
                                    interp=interp, x=x, y=y)
    return section


def _encodeMF14(node, fields, mt, mat, host):
    from kika.endf.classes.mf14.base import (AngularNode, AnisotropicPhoton,
                                             IsotropicPhoton, MF14MT)

    channel = _channelOf(node, host)
    section = MF14MT(number=mt, _za=fields["za"], _awr=fields["awr"],
                     _li=int(fields["li"]), _ltt=int(fields["ltt"]),
                     _nk=int(fields["nk"]), _ni=int(fields["ni"]),
                     _mat=mat if mat is not None else fields["mat"],
                     pad=_padStyle(fields))
    for record in fields["isotropic"]:
        section.isotropic.append(IsotropicPhoton(eg=record["eg"], es=record["es"],
                                                 rest=tuple(record["rest"])))
    for record in fields["anisotropic"]:
        product = _productByLabel(channel, record["label"], mt)
        angular = getattr(_evaluated(product), "angular", None)
        functions, pairs = toEndfTab2(angular)
        if len(functions) != len(record["nodes"]):
            raise ValueError(f"MT{mt} photon {record['label']!r}: the model has "
                             f"{len(functions)} incident energies and MF14 kept "
                             f"{len(record['nodes'])}")
        photon = AnisotropicPhoton(eg=record["eg"], es=record["es"],
                                   l1=int(record["l1"]), l2=int(record["l2"]),
                                   tab2_interp=pairs)
        for function, (c1, l1, l2, n2) in zip(functions, record["nodes"]):
            angularNode = AngularNode(energy=float(function.outerDomainValue), c1=c1,
                                      l1=int(l1), l2=int(l2), n2=int(n2))
            if isinstance(function, Legendre):
                angularNode.coefficients = [
                    float(c) for c in np.asarray(function.coefficients, dtype=float)[1:]]
            else:
                angularNode.interp, angularNode.mu, angularNode.p = _tab1Of(function)
            photon.nodes.append(angularNode)
        section.anisotropic.append(photon)
    return section


def _encodeMF15(node, fields, mt, mat, host):
    from kika.endf.classes.mf12.base import PhotonTable
    from kika.endf.classes.mf15.base import MF15MT, PhotonSpectrum

    channel = _channelOf(node, host)
    product = _productByLabel(channel, fields["label"], mt)
    energy = getattr(_evaluated(product), "energy", None)
    functions, pairs = toEndfTab2(energy)
    weight = fields["weight"]
    spectrum = PhotonSpectrum(
        weight=PhotonTable(c1=weight["c1"], c2=weight["c2"], l1=int(weight["l1"]),
                           l2=int(weight["l2"]),
                           interp=[tuple(p) for p in weight["interp"]],
                           x=list(weight["x"]), y=list(weight["y"])),
        tab2_fields=tuple(fields["tab2"]), tab2_interp=pairs)
    if len(functions) != len(fields["nodes"]):
        raise ValueError(f"MT{mt}: the model's continuum has {len(functions)} "
                         f"incident energies and MF15 kept {len(fields['nodes'])}")
    for function, (c1, l1, l2) in zip(functions, fields["nodes"]):
        interp, x, y = _tab1Of(function)
        spectrum.distributions.append(PhotonTable(
            c1=c1, c2=float(function.outerDomainValue), l1=int(l1), l2=int(l2),
            interp=interp, x=x, y=y))
    return MF15MT(number=mt, _za=fields["za"], _awr=fields["awr"],
                  _l1=int(fields["l1"]), _l2=int(fields["l2"]), _n2=int(fields["n2"]),
                  _mat=mat if mat is not None else fields["mat"],
                  spectra=[spectrum], pad=_padStyle(fields))


def _hasPhotons(channel) -> bool:
    for product in getattr(channel, "products", None) or ():
        if product.pid == "photon" and product.multiplicity is not None:
            return True
        if product.outputChannel is not None and _hasPhotons(product.outputChannel):
            return True
    return False


def encodePhotonSections(suite, mat: Optional[int] = None,
                         report: Optional[ConversionReport] = None):
    """The MF12-15 sections of *suite*: ``([(MF, MT, section)], report)``.

    From the model for every reaction or orphan product whose provenance kept
    ``"mf12"`` or ``"mf13"``, and from the kept text for the sections the model
    does not carry.
    """
    from kika.endf.parsers.parse_photons import (parse_mf12_mt, parse_mf13_mt,
                                                 parse_mf14_mt, parse_mf15_mt)

    report = report if report is not None else ConversionReport()
    sections = []
    for node in list(suite.reactions) + list(suite.orphanProducts):
        header = getattr(getattr(node, "provenance", None), "headerFields", None) or {}
        mf = 13 if "mf13" in header else 12 if "mf12" in header else None
        if mf is None:
            if "mf6" not in header and _hasPhotons(node.outputChannel):
                report.lost(
                    f"{node.label}: its photons have no ENDF bookkeeping (the "
                    f"suite was not read from ENDF), and deriving MF12-15 from the "
                    f"model is roadmap E5e; the tape is written without them")
            continue
        mt = int(node.ENDF_MT)
        fields = header[f"mf{mf}"]
        host = fields.get("host")
        if mf == 12 and fields.get("lo") == 2:
            sections.append((12, mt, _encodeCascade(suite, fields, mt, mat, report)))
        else:
            sections.append((mf, mt, _encodeProduction(suite, node, fields, mf, mt,
                                                       mat, report)))
        if "mf14" in header:
            sections.append((14, mt, _encodeMF14(node, header["mf14"], mt, mat, host)))
        if "mf15" in header:
            sections.append((15, mt, _encodeMF15(node, header["mf15"], mt, mat, host)))

    parsers = {12: parse_mf12_mt, 13: parse_mf13_mt, 14: parse_mf14_mt, 15: parse_mf15_mt}
    kept = (getattr(getattr(suite, "provenance", None), "headerFields", None) or {}
            ).get(PHOTONS_VERBATIM_KEY) or {}
    for key, lines in kept.items():
        mf, mt = (int(v) for v in key.split("/"))
        body = [line for line in lines if line[72:75].strip() not in ("", "0")]
        section = parsers[mf](body, mt)
        if mat is not None:
            section._mat = int(mat)
        sections.append((mf, mt, section))
    return sections, report


# ---------------------------------------------------------------------------
# MF12 LO=2: a level's cascade, in PoPs (roadmap E5c)
# ---------------------------------------------------------------------------

#: How close an ``ES_i`` must be to a level's energy to name that level, relative.
#: ENDF writes both from the same evaluated level scheme, so they normally agree
#: to the last digit; this only absorbs the 6-7 significant figures of the format.
_LEVEL_TOLERANCE = 1e-5


def _cascadeDigest(levelEnergy: float, transitions) -> str:
    """SHA-256 of a level's energy and its ``(ES, TP[, GP])`` rows, in order."""
    digest = hashlib.sha256()
    digest.update(np.asarray([levelEnergy], dtype=float).tobytes())
    for row in transitions:
        digest.update(np.asarray([float(v) for v in row if v is not None],
                                 dtype=float).tobytes())
        digest.update(b"|" if len(row) < 3 or row[2] is None else b"g")
    return digest.hexdigest()


def _levelEnergies(suite, sections12, series) -> Dict[int, float]:
    """``{level index: energy}`` for every level of *series* the evaluation names.

    From MF12 LO=2's ``ES_NS`` where a section states it, and otherwise from
    MF3's ``QM - QI`` -- the excitation energy the residual decay carries. The
    ground state is 0.
    """
    from .residuals import levelSeries

    start = series[0]
    energies = {0: 0.0}
    for reaction in suite.reactions:
        mt = reaction.ENDF_MT
        found = levelSeries(int(mt)) if mt is not None else None
        if found is None or found[0] != start:
            continue
        provenance = getattr(reaction, "provenance", None)
        qi = getattr(reaction.outputChannel.Q, "value", None)
        qm = getattr(provenance, "qm", None)
        if qi is not None and qm is not None:
            energies[found[2]] = float(qm) - float(qi)
    for mt, section in sections12.items():
        found = levelSeries(int(mt))
        if found is not None and found[0] == start and section.lo == 2:
            energies[found[2]] = float(section.es_ns)
    return energies


def _levelOf(energies: Dict[int, float], es: float) -> Optional[int]:
    """The level index whose energy is *es*, or ``None`` -- never a nearest guess."""
    if es == 0.0:
        return 0
    for index, energy in energies.items():
        if energy == es:
            return index
    matches = [index for index, energy in energies.items()
               if energy and abs(energy - es) <= _LEVEL_TOLERANCE * abs(es)]
    return matches[0] if len(matches) == 1 else None


def _attachCascade(suite, sections, sections12, mt, report) -> Optional[str]:
    """MF12 LO=2 (+MF14 LI=1) → the level's ``decayData`` in PoPs. ``None`` on success.

    The level the reaction leaves is the residual product of its two-body
    channel (``residuals.residualOf``, built here when MF3 alone would not have
    built it); its PoPs entry gets the level energy and one electromagnetic
    ``decayMode`` per transition, and the residual's decay channel gets the
    ``branching1d``/``branching3d`` photon that points at it -- as FUDGE writes
    it and as NNDC's GNDS reads.
    """
    from kika.nuclear_data.model import (ELECTROMAGNETIC, Branching1d, Branching3d,
                                         Decay, DecayData, DecayMode, DecayModes,
                                         DecayPath, Nuclide, PhotonEmissionProbabilities,
                                         PhysicalQuantity, Product, Shell, pidFromZA)
    from kika.nuclear_data.model.pops import zaFromPid

    from .residuals import levelSeries, residualOf

    section, mf14 = sections[12], sections.get(14)
    if set(sections) - {12, 14}:
        return "MF13 or MF15 beside an MF12 LO=2 section, which ENDF-6 does not admit"
    if mf14 is not None and mf14.li != 1:
        return ("its MF14 is anisotropic, and a level's cascade has no place for a "
                "per-transition angular distribution in GNDS (branching3d)")
    series = levelSeries(int(mt))
    if series is None:
        return "LO=2 on an MT that is not a discrete level, which has no level to decay"
    reaction = suite.findReactionByENDF_MT(mt)
    if reaction is None or not any(reaction is r for r in suite.reactions):
        return "there is no MF3 reaction for this level"
    provenance = getattr(reaction, "provenance", None)
    if provenance is None or getattr(provenance, "sourceFormat", None) != "endf":
        return "its reaction carries no ENDF provenance to keep the bookkeeping in"
    residual = residualOf(suite, reaction, report, force=True)
    if residual is None or residual.outputChannel is None:
        return "the residual's decay channel could not be built (see the report)"

    # The level is the series' own, whatever pid the residual product carries:
    # an MF6 that states the recoil writes it with LIP=0 (ENDF/B-VIII.1 Ni-58
    # MT51 gives `Ni58`), and hanging the cascade on that pid put a decay and
    # an excitation energy on the *target*.
    residualZA = zaFromPid(residual.pid)
    levelPid = pidFromZA(residualZA, series[2])
    if series[2] == 0:
        return "LO=2 on the ground state of a series, which has nothing to decay to"
    energies = _levelEnergies(suite, sections12, series)
    # The level's energy in PoPs is MF3's QM - QI: the Q the reaction states is
    # the one FUDGE's toENDF6 writes back as QI + level (reactions/base.py), so
    # a suite written to GNDS and read back derives the tape's QM exactly.
    # MF12's own ES_NS, when it differs (94 sections of ENDF/B-VIII.1, 1 740
    # of JEFF-4.0), is kept for the section and written back with it.
    esNs = float(section.es_ns)
    qm, qi = getattr(provenance, "qm", None), getattr(reaction.outputChannel.Q, "value", None)
    levelEnergy = esNs
    if qm is not None and qi is not None:
        levelEnergy = float(qm) - float(qi)
        if levelEnergy != esNs:
            report.warn(
                f"MT{mt}: MF12 states the level at ES={esNs!r} eV and MF3's QM-QI "
                f"gives {levelEnergy!r} eV; PoPs carries MF3's, the MF12 section "
                f"keeps its own (FUDGE keeps whichever has more digits)")

    lg = int(section.lg)
    modes = DecayModes()
    finals = []
    for k, row in enumerate(section.transitions):
        es, tp = float(row[0]), float(row[1])
        index = _levelOf(energies, es)
        if index is None:
            return (f"transition {k} ends at ES={es!r} eV, which is no level this "
                    f"evaluation names in the series")
        final = pidFromZA(residualZA, index)
        if final not in suite.PoPs.particles:
            suite.PoPs.add(Nuclide(id=final, Z=residualZA // 1000, A=residualZA % 1000,
                                   nuclearLevel=index,
                                   energy=(PhysicalQuantity(value=energies[index], unit="eV")
                                           if index else None)))
        emission = (PhotonEmissionProbabilities(shells=[Shell(label="total",
                                                              value=float(row[2]))])
                    if lg == 2 else None)
        modes.decayModes.append(DecayMode(
            label=str(k), mode=ELECTROMAGNETIC, probability=tp,
            photonEmissionProbabilities=emission,
            decayPath=DecayPath(decays=[Decay(index=k, products=[
                Product(pid="photon", label="photon"), Product(pid=final, label=final)])])))
        finals.append((final, es))

    level = suite.PoPs.particles.get(levelPid)
    if level is None:
        level = Nuclide(id=levelPid, Z=residualZA // 1000, A=residualZA % 1000,
                        nuclearLevel=series[2])
        suite.PoPs.add(level)
    decayData = DecayData(decayModes=modes)
    if getattr(level, "decayData", None) is not None and level.decayData != decayData:
        return (f"{levelPid} already has a different decay from another section, "
                f"and one level decays one way")
    level.energy = PhysicalQuantity(value=levelEnergy, unit="eV")
    level.decayData = decayData

    channel = residual.outputChannel
    channel.products.products[:] = [
        p for p in channel.products.products
        if not (p.pid == "photon" and p.multiplicity is None)]
    photon = Product(pid="photon", label="photon",
                     multiplicity=Multiplicity(form=Branching1d(label=EVAL_LABEL)))
    photon.distribution = Distribution()
    photon.distribution[EVAL_LABEL] = Branching3d(label=EVAL_LABEL, productFrame=Frame.lab)
    channel.products.products.append(photon)

    provenance.headerFields["mf12"] = {
        **_head(section), "lo": 2, "lg": lg, "ns": section._ns, "n2": section._n2,
        "listC2": section.list_c2, "lp": section.lp, "listL2": section.list_l2,
        "nt": section.nt, "pad": _pad(section), "level": levelPid,
        "esNs": esNs, "levelEnergy": levelEnergy,
        "host": residual.label, "finals": finals,
        "lines": str(section).split("\n"),
        "digest": _cascadeDigest(esNs, section.transitions),
    }
    if mf14 is not None:
        provenance.headerFields["mf14"] = _mf14Fields(mf14, None, [])
    return None


def _cascadeRows(suite, fields, mt):
    """The level's ``(ES, TP[, GP])`` rows as the model holds them now."""
    level = suite.PoPs.particles.get(fields["level"])
    decayData = getattr(level, "decayData", None)
    if level is None or decayData is None:
        raise ValueError(f"MT{mt}: MF12 LO=2 names the level {fields['level']!r} and "
                         f"PoPs has no decay for it")
    kept = {final: es for final, es in fields["finals"]}
    rows = []
    for mode in decayData.decayModes:
        final = mode.finalState()
        if final in kept:
            es = kept[final]
        else:
            particle = suite.PoPs.particles.get(final)
            energy = getattr(particle, "energy", None)
            es = float(energy.value) if energy is not None else 0.0
        gp = (mode.photonEmissionProbabilities.total()
              if mode.photonEmissionProbabilities is not None else None)
        rows.append((es, float(mode.probability), gp) if fields["lg"] == 2
                    else (es, float(mode.probability)))
    if fields.get("sortDescending"):
        # A cascade derived for a GNDS-read suite (E5e): FUDGE's order,
        # decreasing final-level energy.
        rows.sort(key=lambda row: row[0], reverse=True)
    energy = getattr(level, "energy", None)
    value = float(energy.value) if energy is not None else 0.0
    if fields.get("esNs") is not None and value == fields.get("levelEnergy"):
        # The level was not moved: the section's own ES_NS, which may differ
        # from MF3's QM - QI that PoPs carries.
        value = float(fields["esNs"])
    return value, rows


def _encodeCascade(suite, fields, mt, mat, report):
    """MF12 LO=2 from PoPs: the kept bytes while the cascade is unchanged."""
    from kika.endf.classes.mf12.base import MF12MT
    from kika.endf.parsers.parse_photons import parse_mf12_mt

    levelEnergy, rows = _cascadeRows(suite, fields, mt)
    if _cascadeDigest(levelEnergy, rows) == fields["digest"]:
        body = [line for line in fields["lines"] if line[72:75].strip() not in ("", "0")]
        section = parse_mf12_mt(body, mt)
        if mat is not None:
            section._mat = int(mat)
        return section
    report.warn(f"MT{mt}: the cascade of {fields['level']} changed, so MF12 LO=2 is "
                f"rebuilt from PoPs (LP and NS as read)")
    values = [float(v) for row in rows for v in row]
    return MF12MT(number=mt, _za=fields["za"], _awr=fields["awr"], _lo=2,
                  _l2=int(fields["lg"]), _ns=int(fields["ns"]), _n2=int(fields["n2"]),
                  _mat=mat if mat is not None else fields["mat"],
                  es_ns=levelEnergy, list_c2=fields["listC2"], lp=int(fields["lp"]),
                  list_l2=int(fields["listL2"]), transition_values=values,
                  nt=len(rows), pad=_padStyle(fields))
