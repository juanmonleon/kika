"""ENDF photo-atomic (NSUB=3: MF23, MF27) and atomic-relaxation (NSUB=6: MF28) ↔ the model.

**Photo-atomic** (:func:`attachPhotoatomic`, :func:`encodePhotoatomicSections`).
Each MF23 section is a reaction with its σ(E); MT501 (total), 516 (pair
production) and 522 (photoelectric) are ``crossSectionSum`` nodes over their
parts, as :func:`_summationMTs` decides for MF3. MF27/MT502, 505 and 506 become
the coherent reaction's :class:`CoherentPhotonScattering` (form factor, real and
imaginary anomalous factors) and MF27/MT504 the incoherent reaction's
:class:`IncoherentPhotonScattering`. The reaction labels are FUDGE's
(``Fe + photon [coherent]``, ``e- + Fe{1s1/2}``) and the suite's interaction is
``atomic``.

**Atomic relaxation** (:func:`decodeRelaxationSublibrary`,
:func:`encodeRelaxationSections`). MF28/MT533 becomes a standalone PoPs whose
:class:`ChemicalElement` carries one :class:`Configuration` per subshell, its
transitions as ``atomicRelaxation`` decay modes, as FUDGE's ``ENDF_ITYPE_6``
writes them.

**What GNDS cannot hold.** The HEAD and TAB1 fields that are zero by ENDF-6's
definition but written otherwise, the subshell's EPE and EFL of MF23 (kept as
read), MF27's Z in the TAB1, a transition's energy ETR (GNDS derives it from
the binding energies) and the tape's own spelling of numbers (the B-VIII.1
photon sublibraries write ``26000.0000`` and a left-aligned ``0.0``). They are
kept in the provenance; the spelling as the field overrides
:mod:`.decay_sublibrary` uses, so a tape read and written back is the same
bytes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from kika.nuclear_data.model import (EVAL_LABEL, ATOMIC_INTERACTION, Atomic, Axes, ChemicalElement,
                                     CoherentPhotonScattering, Configuration, ConversionReport, Decay,
                                     DecayData, DecayMode, DecayModes, DecayPath, EndfProvenance,
                                     IncoherentPhotonScattering, PhysicalQuantity, PoPs, Product,
                                     Reaction, ReactionId, crossSectionAxes, subshellDesignator,
                                     pidFromZA, subshellLabel)
from kika.nuclear_data.model.thermal_scattering import DoubleDifferentialCrossSection

from .photons import _function1d, _tab1Of

__all__ = ["attachPhotoatomic", "encodePhotoatomicSections", "keepPhotoatomicSpelling",
           "decodeRelaxationSublibrary", "encodeRelaxationSections", "PHOTOATOMIC_KEY",
           "RELAXATION_KEY", "PHOTOATOMIC_NSUB", "ELECTROATOMIC_NSUB", "RELAXATION_NSUB",
           "photoatomicReactionLabel"]

PHOTOATOMIC_NSUB = 3
ELECTROATOMIC_NSUB = 113
RELAXATION_NSUB = 6
#: On a reaction's provenance: ``{"mf23": {...}, "mf27": {MT: {...}}}``, the fields
#: and the spelling of its sections as read.
PHOTOATOMIC_KEY = "photoatomic"
#: On the relaxation PoPs' provenance: the MF28 fields GNDS cannot hold.
RELAXATION_KEY = "mf28_533"

#: The MF27 sections the form factors come from, by the reaction they belong to.
_MF27_OF = {502: (502, 505, 506), 504: (504,)}
_SUMS = {501: None, 516: (515, 517), 522: None}


def photoatomicReactionLabel(mt: int, element: str) -> str:
    """FUDGE's label of a photo-atomic reaction (``ENDF_ITYPE_3``)."""
    if mt == 501:
        return f"total"
    if mt == 502:
        return f"{element} + photon [coherent]"
    if mt == 504:
        return f"{element} + photon [incoherent]"
    if mt == 515:
        return f"e- + e- + e-_anti + {element} [pair production: electron field]"
    if mt == 516:
        return "pair product"
    if mt == 517:
        return f"e- + e-_anti + {element} [pair production: nuclear field]"
    if mt == 522:
        return "photoelectric absorption"
    if 534 <= mt <= 572:
        return f"e- + {element}{{{subshellLabel(mt - 533)}}}"
    return f"MT{mt}"


def _axes(xLabel, xUnit, yLabel, yUnit) -> Axes:
    return Axes.forFunction1d(yLabel, yUnit, xLabel, xUnit)


#: The axes FUDGE gives each MF27 table (its abscissa is called ``energy_in``
#: even where it is x in 1/Ang).
_FF_AXES = {502: ("energy_in", "1/Ang", "coherent form factor", ""),
            504: ("energy_in", "1/Ang", "incoherent scattering function", ""),
            505: ("energy_in", "eV", "imaginary part of anomalous scattering factor", ""),
            506: ("energy_in", "eV", "real part of anomalous scattering factor", "")}


def _fields(section) -> dict:
    """The HEAD and TAB1 fields ENDF-6 defines as zero or as metadata, as read."""
    t = section.table
    return {"za": section._za, "awr": section._awr, "head": [section._l1, section._l2, section._n1, section._n2],
            "c1": t.c1, "c2": t.c2, "l1": t.l1, "l2": t.l2,
            "pad": [section.pad.pairs, section.pad.values, section.pad.interp]}


def _summands(mt: int, present) -> List[int]:
    if mt == 501:
        return [m for m in present if m in (502, 504, 515, 517) or 534 <= m <= 572]
    if mt == 516:
        return [m for m in (515, 517) if m in present]
    if mt == 522:
        return [m for m in present if 534 <= m <= 572]
    return []


def attachPhotoatomic(suite, endf, report: ConversionReport) -> ConversionReport:
    """MF23 (and MF27) of a photo-atomic tape → the suite's reactions, sums and forms.

    An electro-atomic tape (NSUB=113: MF23 and MF26) goes to
    :func:`_attachElectroatomic`.
    """
    from kika.nuclear_data.model import Add, CrossSectionSum, Summands

    nsub = (getattr(suite.provenance, "headerFields", None) or {}).get("nsub")
    if nsub is not None and int(nsub) == ELECTROATOMIC_NSUB:
        return _attachElectroatomic(suite, endf, report)
    mf23 = endf.mf[23]
    mf27 = endf.mf.get(27)
    suite.interaction = ATOMIC_INTERACTION
    element = suite.target
    present = sorted(mf23.mt)
    _atomicPoPs(suite)
    sums = {mt for mt in _SUMS if mt in present and _summands(mt, present)}
    for mt in present:
        section = mf23.mt[mt]
        function = _function1d(section.table.interp, section.table.x, section.table.y, axes=crossSectionAxes())
        function.label = EVAL_LABEL
        provenance = EndfProvenance(mat=section._mat, za=section._za, awr=section._awr,
                                    headerFields={PHOTOATOMIC_KEY: {"mf23": _fields(section)}})
        reactionId = ReactionId(label=photoatomicReactionLabel(mt, element), ENDF_MT=mt)
        if mt in sums:
            links = [Add(f"/reactionSuite/reactions/reaction[@label='{photoatomicReactionLabel(p, element)}']"
                         f"/crossSection") for p in _summands(mt, present) if p not in sums]
            reaction = CrossSectionSum(id=reactionId, provenance=provenance, summands=Summands(links))
            reaction.crossSection[EVAL_LABEL] = function
            suite.sums.append(reaction)
            continue
        reaction = Reaction(id=reactionId, provenance=provenance)
        reaction.crossSection[EVAL_LABEL] = function
        reaction.outputChannel = _outputChannel(mt, element, function, section.table.c1, reactionId.label)
        if mt in _MF27_OF and mf27 is not None:
            tables = {}
            for part in _MF27_OF[mt]:
                kept = mf27.mt.get(part)
                if kept is None:
                    continue
                x, xUnit, y, yUnit = _FF_AXES[part]
                f = _function1d(kept.table.interp, kept.table.x, kept.table.y, axes=_axes(x, xUnit, y, yUnit))
                tables[part] = f
                provenance.headerFields[PHOTOATOMIC_KEY].setdefault("mf27", {})[part] = _fields(kept)
            ddcs = DoubleDifferentialCrossSection()
            if mt == 502 and 502 in tables:
                ddcs[EVAL_LABEL] = CoherentPhotonScattering(
                    formFactor=tables[502], realAnomalousFactor=tables.get(506),
                    imaginaryAnomalousFactor=tables.get(505), label=EVAL_LABEL)
            elif mt == 504 and 504 in tables:
                ddcs[EVAL_LABEL] = IncoherentPhotonScattering(scatteringFactor=tables[504], label=EVAL_LABEL)
            if len(ddcs):
                reaction.doubleDifferentialCrossSection = ddcs
        suite.reactions.append(reaction)
    if mf27 is not None:
        for mt in sorted(mf27.mt):
            if not any(mt in parts for parts in _MF27_OF.values()):
                report.lost(f"MF27/MT{mt} is not a form factor ENDF-6 defines; not decoded")
    return report


#: The photon, as FUDGE's PoPs of a photo-atomic suite states it.
_PHOTON = dict(spin=1.0, parity=1, charge=0)
#: The leptons a photo- or electro-atomic suite names: (pid, parity, charge).
_LEPTONS = (("e-", 1, -1), ("e-_anti", -1, 1))
#: The electron mass FUDGE's PoPs writes, amu.
_ELECTRON_MASS_AMU = 5.48579909e-4
#: Q of pair production in FUDGE's photo-atomic GNDS (2 m_e c^2 rounded), eV.
_PAIR_Q = -1.022e6


def _atomicPoPs(suite, positron: bool = True) -> None:
    """The particles of a photo- or electro-atomic suite as FUDGE states them.

    The target is the element (a ``chemicalElement`` with no isotopes, not a
    nuclide of A = 0); the photon a gauge boson; the electron and positron
    leptons. A photo-atomic tape has no neutron, which MF1's decoder adds.
    """
    from kika._constants import ATOMIC_NUMBER_TO_NAME, SYMBOL_TO_ATOMIC_NUMBER
    from kika.nuclear_data.model import Lepton, Particle

    pops = suite.PoPs
    pops.particles.pop("n", None)
    target = pops.particles.pop(suite.target, None)
    z = getattr(target, "Z", None) or SYMBOL_TO_ATOMIC_NUMBER.get(suite.target)
    if z is not None:
        pops.chemicalElements[suite.target] = ChemicalElement(symbol=suite.target, Z=int(z),
                                                              name=ATOMIC_NUMBER_TO_NAME.get(int(z)))
    photon = pops.particles.get("photon") or Particle(id="photon")
    photon.mass = PhysicalQuantity(value=0.0, unit="amu")
    photon.spin = PhysicalQuantity(value=_PHOTON["spin"], unit="hbar")
    photon.parity, photon.charge, photon.halflife = _PHOTON["parity"], _PHOTON["charge"], "stable"
    pops.particles["photon"] = photon
    for pid, parity, charge in _LEPTONS:
        if pid == "e-_anti" and not positron:
            continue
        pops.particles[pid] = Lepton(id=pid, mass=PhysicalQuantity(value=_ELECTRON_MASS_AMU, unit="amu"),
                                     spin=PhysicalQuantity(value=0.5, unit="hbar"), parity=parity,
                                     charge=charge, halflife="stable")


def _outputChannel(mt: int, element: str, function, epe: float, label: str):
    """FUDGE's output channel of a photo-atomic reaction: genre, process, Q and products.

    Q spans the cross section's domain: 0 for coherent and incoherent
    scattering, ``-1.022e6`` eV for pair production, ``-EPE`` (the subshell's
    binding energy, MF23's C1) for photo-ionisation. Every product has a
    multiplicity of one; the scattered photon's distribution is its reaction's
    double-differential form, every other product's is unspecified.
    """
    from kika.nuclear_data.model import (CoherentPhotonScatteringLink, Distribution,
                                         IncoherentPhotonScatteringLink, Multiplicity, OutputChannel,
                                         Product, Q, Unspecified, multiplicityAxes)
    from kika.nuclear_data.model.functions.simple import Constant1d

    xs, _, _ = function.toEndfRegions()
    lo, hi = float(xs[0]), float(xs[-1])
    if mt in (502, 504):
        genre, process, q = "twoBody", ("coherent" if mt == 502 else "incoherent"), 0.0
        pids = [("photon", "photon"), (element, element)]
    elif mt == 515:
        genre, process, q = "NBody", "pair production: electron field", _PAIR_Q
        pids = [("e-", "e-"), ("e-", "e-__a"), ("e-_anti", "e-_anti"), (element, element)]
    elif mt == 517:
        genre, process, q = "NBody", "pair production: nuclear field", _PAIR_Q
        pids = [("e-", "e-"), ("e-_anti", "e-_anti"), (element, element)]
    elif 534 <= mt <= 572:
        vacancy = f"{element}{{{subshellLabel(mt - 533)}}}"
        genre, process, q = "NBody", None, -float(epe)
        pids = [("e-", "e-"), (vacancy, vacancy)]
    else:
        return OutputChannel()
    products = []
    for pid, plabel in pids:
        distribution = Distribution()
        if pid == "photon":
            kind = CoherentPhotonScatteringLink if mt == 502 else IncoherentPhotonScatteringLink
            node = "coherentPhotonScattering" if mt == 502 else "incoherentPhotonScattering"
            distribution[EVAL_LABEL] = kind(
                href=f"/reactionSuite/reactions/reaction[@label='{label}']/doubleDifferentialCrossSection/"
                     f"{node}[@label='{EVAL_LABEL}']", label=EVAL_LABEL)
        else:
            distribution[EVAL_LABEL] = Unspecified(label=EVAL_LABEL)
        multiplicity = Multiplicity(form=Constant1d(constant=1.0, domainMin_=lo, domainMax_=hi,
                                                    axes=multiplicityAxes(), label=EVAL_LABEL))
        products.append(Product(pid=pid, label=plabel, multiplicity=multiplicity, distribution=distribution))
    channel = OutputChannel(genre=genre, process=process,
                            Q=Q(value=q, unit="eV", label=EVAL_LABEL, domainMin=lo, domainMax=hi))
    for product in products:
        channel.products.products.append(product)
    return channel


# ---------------------------------------------------------------------------
# Electro-atomic: MF23 + MF26
# ---------------------------------------------------------------------------

#: The electro-atomic reactions FUDGE writes, by MT: (label pattern, genre, process).
_ELECTRON_REACTIONS = {
    525: ("e- + {el} [large angle electro-atomic scattering]", "twoBody", "large angle electro-atomic scattering"),
    526: ("e- + {el} [large angle Coulomb scattering]", "twoBody", "large angle Coulomb scattering"),
    527: ("e- + {el} + photon [bremsstrahlung]", "NBody", "bremsstrahlung"),
    528: ("e- + {el} [excitation]", "NBody", "excitation"),
}


def electroatomicReactionLabel(mt: int, element: str) -> str:
    """FUDGE's label of an electro-atomic reaction (``ENDF_ITYPE_3``)."""
    if mt == 501:
        return "total"
    if mt == 522:
        return "photoelectric absorption"
    if mt in _ELECTRON_REACTIONS:
        return _ELECTRON_REACTIONS[mt][0].format(el=element)
    if 534 <= mt <= 572:
        return f"e- + e- + {element}{{{subshellLabel(mt - 533)}}}"
    return f"MT{mt}"


def _electronSummands(mt: int, present) -> List[int]:
    if mt == 501:
        return [m for m in present if m in (526, 527, 528) or 534 <= m <= 572]
    if mt == 522:
        return [m for m in present if 534 <= m <= 572]
    return []


def _attachElectroatomic(suite, endf, report: ConversionReport) -> ConversionReport:
    """MF23 + MF26 of an electro-atomic tape → reactions, sums, the large-angle elastic as an
    incomplete reaction, and the products MF26 states, as FUDGE writes them."""
    from kika.nuclear_data.model import Add, CrossSectionSum, Summands

    mf23 = endf.mf[23]
    mf26 = endf.mf.get(26)
    suite.interaction = ATOMIC_INTERACTION
    suite.projectile = "e-"
    element = suite.target
    present = sorted(mf23.mt)
    _atomicPoPs(suite, positron=False)
    sums = {mt for mt in (501, 522) if mt in present and _electronSummands(mt, present)}
    for mt in present:
        section = mf23.mt[mt]
        function = _function1d(section.table.interp, section.table.x, section.table.y, axes=crossSectionAxes())
        function.label = EVAL_LABEL
        provenance = EndfProvenance(mat=section._mat, za=section._za, awr=section._awr,
                                    headerFields={PHOTOATOMIC_KEY: {"mf23": _fields(section)}})
        label = electroatomicReactionLabel(mt, element)
        reactionId = ReactionId(label=label, ENDF_MT=mt)
        if mt in sums:
            links = [Add(f"/reactionSuite/reactions/reaction[@label='{electroatomicReactionLabel(p, element)}']"
                         f"/crossSection") for p in _electronSummands(mt, present) if p not in sums]
            reaction = CrossSectionSum(id=reactionId, provenance=provenance, summands=Summands(links))
            reaction.crossSection[EVAL_LABEL] = function
            suite.sums.append(reaction)
            continue
        reaction = Reaction(id=reactionId, provenance=provenance)
        reaction.crossSection[EVAL_LABEL] = function
        section26 = mf26.mt.get(mt) if mf26 is not None else None
        reaction.outputChannel = _electronChannel(mt, element, function, section.table.c1, section26,
                                                  provenance.headerFields[PHOTOATOMIC_KEY], report)
        if mt == 525:
            # The large-angle elastic is part of the total elastic MT526: FUDGE
            # files it as an incomplete reaction, outside the sum.
            suite.incompleteReactions.append(reaction)
        else:
            suite.reactions.append(reaction)
    return report


def _constantOne(lo, hi):
    from kika.nuclear_data.model import Multiplicity, multiplicityAxes
    from kika.nuclear_data.model.functions.simple import Constant1d

    return Multiplicity(form=Constant1d(constant=1.0, domainMin_=lo, domainMax_=hi, axes=multiplicityAxes(),
                                        label=EVAL_LABEL))


def _unspecifiedProduct(pid, label, lo, hi):
    from kika.nuclear_data.model import Distribution, Product, Unspecified

    distribution = Distribution()
    distribution[EVAL_LABEL] = Unspecified(label=EVAL_LABEL)
    return Product(pid=pid, label=label, multiplicity=_constantOne(lo, hi), distribution=distribution)


def _electronChannel(mt: int, element: str, function, epe: float, section26, book: dict,
                     report: ConversionReport):
    """The output channel of an electro-atomic reaction, its MF26 products decoded into it.

    MF26's LAW=1 and LAW=2 products go through MF6's decoder (their records are
    MF6's); LAW=8 (energy transfer) becomes the electron's ``averageProductEnergy``,
    ``E - ET``. The fields the model does not hold are kept in ``book['mf26']``.
    """
    from kika.endf.classes.mf6.base import MF6MT
    from kika.nuclear_data.model import (AverageProductEnergy, Distribution, Frame, OutputChannel, Product, Q,
                                         XYs1d)

    from .energy_angle import decodeMF6MT

    xs, _, _ = function.toEndfRegions()
    lo, hi = float(xs[0]), float(xs[-1])
    if mt in _ELECTRON_REACTIONS:
        _, genre, process = _ELECTRON_REACTIONS[mt]
        q = 0.0
    else:
        genre, process, q = "NBody", None, -float(epe)
    channel = OutputChannel(genre=genre, process=process,
                            Q=Q(value=q, unit="eV", label=EVAL_LABEL, domainMin=lo, domainMax=hi))
    decoded = {}            # pid -> (distribution form) from LAW 1/2
    averages = {}           # pid -> XYs1d of E - ET
    if section26 is not None:
        law8 = {}
        kept = MF6MT(number=mt, _za=section26._za, _awr=section26._awr, _jp=section26._jp, _lct=1,
                     _mat=section26._mat)
        kept.pad = section26.pad
        for index, product in enumerate(section26.products):
            if product.law == 8:
                body = product.law_data
                law8[index] = {"zap": product.zap, "awp": product.awp, "lip": product.lip,
                               "y": [list(product.y_interp), list(product.y_energies), list(product.y_values)],
                               "head": [body.c1, body.c2, body.l1, body.l2], "interp": list(body.interp)}
                e = np.asarray(body.energies, dtype=float)
                averages[pidFromZA(product.za)] = XYs1d(
                    xs=e, ys=e - np.asarray(body.transfer, dtype=float), label=EVAL_LABEL,
                    axes=_axes("energy_in", "eV", "averageProductEnergy", "eV"))
            else:
                kept.products.append(product)
        kept._nk = len(kept.products)
        entries, provenance6, report = decodeMF6MT(kept, report)
        for plabel, pid, _, form in entries:
            if form is not None and type(form).__name__ == "AngularTwoBody":
                form.productFrame = Frame.centerOfMass
            decoded[plabel] = form
        book["mf26"] = {"provenance": provenance6, "law8": law8, "nk": section26._nk,
                        "lct": section26._lct, "jp": section26._jp}

    def product(pid, label, form=None):
        if form is None and pid not in averages:
            return _unspecifiedProduct(pid, label, lo, hi)
        distribution = Distribution()
        distribution[EVAL_LABEL] = form if form is not None else _unspecifiedForm()
        p = Product(pid=pid, label=label, multiplicity=_constantOne(lo, hi), distribution=distribution)
        if form is None and pid in averages:
            p.averageProductEnergy = AverageProductEnergy()
            p.averageProductEnergy[EVAL_LABEL] = averages[pid]
        return p

    if mt == 525:
        products = [product("e-", "e-", decoded.get("e-")), _unspecifiedProduct(element, element, lo, hi)]
    elif mt == 526:
        products = [_unspecifiedProduct("e-", "e-", lo, hi), _unspecifiedProduct(element, element, lo, hi)]
    elif mt == 527:
        products = [product("photon", "photon", decoded.get("photon")), product("e-", "e-"),
                    _unspecifiedProduct(element, element, lo, hi)]
    elif mt == 528:
        products = [product("e-", "e-"), _unspecifiedProduct(element, element, lo, hi)]
    elif 534 <= mt <= 572:
        vacancy = f"{element}{{{subshellLabel(mt - 533)}}}"
        products = [product("e-", "e-", decoded.get("e-")), _unspecifiedProduct("e-", "e-__a", lo, hi),
                    _unspecifiedProduct(vacancy, vacancy, lo, hi)]
    else:
        products = []
    for p in products:
        channel.products.products.append(p)
    return channel


def _unspecifiedForm():
    from kika.nuclear_data.model import Unspecified

    return Unspecified(label=EVAL_LABEL)


def _encodeMF26(reaction, book: dict, mt: int, mat: int, report: ConversionReport):
    """MF26 of an electro-atomic reaction: MF6's encoder for LAW 1/2, LAW=8 from averageProductEnergy."""
    from kika.endf.classes.mf26 import MF26LawEnergyTransfer, MF26MT
    from kika.endf.classes.mf6.products import MF6Product

    from .energy_angle import encodeMF6MT

    kept = book["mf26"]
    provenance6 = kept["provenance"]
    labels = [record["label"] for record in provenance6.headerFields["mf6"]["products"]]
    byLabel = {p.label: p for p in reaction.outputChannel.products}
    forms = {}
    for label in labels:
        p = byLabel.get(label)
        if p is not None and p.distribution is not None:
            forms[label] = p.distribution.forms.get(EVAL_LABEL)
    section6, report = encodeMF6MT(forms, provenance6, mt, report)
    products = list(section6.products)
    for index in sorted(kept["law8"], key=int):
        fields = kept["law8"][index]
        owner = next((p for p in reaction.outputChannel.products
                      if p.pid == pidFromZA(int(round(fields["zap"]))) and p.averageProductEnergy is not None), None)
        if owner is None:
            report.lost(f"MF26/MT{mt}: the energy transfer of product {index} has no averageProductEnergy")
            continue
        ape = owner.averageProductEnergy.forms[EVAL_LABEL]
        e = np.asarray(ape.xs, dtype=float)
        body = MF26LawEnergyTransfer(c1=fields["head"][0], c2=fields["head"][1], l1=fields["head"][2],
                                     l2=fields["head"][3], interp=[tuple(x) for x in fields["interp"]],
                                     energies=list(e), transfer=list(e - np.asarray(ape.ys, dtype=float)))
        yi, ye, yv = fields["y"]
        products.insert(int(index), MF6Product(zap=fields["zap"], awp=fields["awp"], lip=fields["lip"],
                                               y_interp=[tuple(x) for x in yi], y_energies=list(ye),
                                               y_values=list(yv), law_data=body))
    section = MF26MT(number=mt, _za=section6._za, _awr=section6._awr, _jp=kept["jp"], _lct=kept["lct"],
                     _nk=kept["nk"], _mat=mat)
    section.pad = section6.pad
    section.products = products
    return section


def _tab1Section(cls, mt: int, function, fields: dict, mat: int, mf: int, defaults: dict):
    from kika.endf.classes.mf12.base import PhotonTable
    from kika.endf.utils import PadStyle

    interp, x, y = _tab1Of(function)
    f = fields or {}
    head = f.get("head") or [0, 0, 0, 0]
    pad = f.get("pad")
    section = cls(number=mt, _za=f.get("za", defaults["za"]), _awr=f.get("awr", defaults["awr"]),
                  _l1=head[0], _l2=head[1], _n1=head[2], _n2=head[3], _mat=mat,
                  table=PhotonTable(c1=f.get("c1", defaults.get("c1", 0.0)), c2=f.get("c2", defaults.get("c2", 0.0)),
                                    l1=f.get("l1", 0), l2=f.get("l2", 0), interp=interp, x=x, y=y))
    if pad is not None:
        section.pad = PadStyle(pairs=pad[0], values=pad[1], interp=pad[2])
    return section


def encodePhotoatomicSections(suite, mat: Optional[int], report: ConversionReport):
    """``[(MF, MT, section), ...]`` (MF23 and MF27) of a photo-atomic suite, and the report."""
    from kika.endf.classes.mf23.base import MF23MT, MF27MT
    from kika.nuclear_data.model import zaFromPid

    from .decay_sublibrary import _TextSection, _applyOverrides

    if getattr(suite, "interaction", None) != ATOMIC_INTERACTION:
        return [], report
    element = getattr(suite.PoPs, "chemicalElements", {}).get(suite.target)
    z = getattr(element, "Z", None) or getattr(suite.PoPs.particles.get(suite.target), "Z", None)
    if z is None:
        from kika._constants import SYMBOL_TO_ATOMIC_NUMBER
        z = SYMBOL_TO_ATOMIC_NUMBER.get(suite.target)
    defaults = {"za": float(1000 * int(z)) if z else 0.0,
                "awr": float(getattr(getattr(suite.provenance, "attributes", None) or {}, "get",
                                     lambda *_: 0.0)("atomic_weight_ratio") or 0.0)}
    sections = []
    for container in (suite.reactions, suite.sums, suite.incompleteReactions):
        for reaction in container:
            mt = reaction.ENDF_MT
            if mt is None:
                continue
            book = ((getattr(reaction.provenance, "headerFields", None) or {}).get(PHOTOATOMIC_KEY) or {})
            form = reaction.crossSection.forms.get(EVAL_LABEL)
            if form is None:
                report.lost(f"MF23/MT{mt}: no evaluated cross section; not written")
                continue
            rowMat = int(mat if mat is not None else getattr(reaction.provenance, "mat", 0) or 0)
            section = _tab1Section(MF23MT, mt, form, book.get("mf23"), rowMat, 23, defaults)
            sections.append((23, mt, _spelled(section, book.get("spelling", {}).get(f"23/{mt}"), rowMat, 23, report)))
            if book.get("mf26") is not None:
                s26 = _encodeMF26(reaction, book, mt, rowMat, report)
                sections.append((26, mt, _spelled(s26, book.get("spelling", {}).get(f"26/{mt}"), rowMat, 26, report)))
            ddcs = reaction.doubleDifferentialCrossSection
            form27 = ddcs.forms.get(EVAL_LABEL) if ddcs is not None else None
            parts = {}
            if isinstance(form27, CoherentPhotonScattering):
                parts = {502: form27.formFactor, 505: form27.imaginaryAnomalousFactor, 506: form27.realAnomalousFactor}
            elif isinstance(form27, IncoherentPhotonScattering):
                parts = {504: form27.scatteringFactor}
            for part, function in parts.items():
                if function is None:
                    continue
                kept = (book.get("mf27") or {}).get(part)
                if kept is None:
                    kept = {"c2": float(z or 0) if part in (502, 504) else 0.0}
                s27 = _tab1Section(MF27MT, part, function, kept, rowMat, 27, defaults)
                sections.append((27, part, _spelled(s27, book.get("spelling", {}).get(f"27/{part}"),
                                                    rowMat, 27, report)))
    return sections, report


def _spelled(section, overrides, mat, mf, report):
    """The section with the tape's own spelling of its fields restored, when it was kept."""
    from .decay_sublibrary import _TextSection, _applyOverrides

    if not overrides:
        return section
    lines = str(section).split("\n")[:-1]
    lines = _applyOverrides(lines, overrides, report, what=f"MF{mf}/MT{section.number}")
    return _TextSection(lines, mat, mf)


def keepPhotoatomicSpelling(suite, sourcePath) -> None:
    """Keep, on each reaction, every field the rebuilt MF23/MF27 sections spell otherwise than the tape.

    Also the tape's own conventions (sequence numbers, control records), on the
    suite's provenance, as the decay sublibrary keeps them.
    """
    from .decay_sublibrary import _overrides, _rawSection

    sections, _ = encodePhotoatomicSections(suite, None, ConversionReport())
    byMt = {}
    for container in (suite.reactions, suite.sums, suite.incompleteReactions):
        for reaction in container:
            if reaction.ENDF_MT is not None:
                byMt[int(reaction.ENDF_MT)] = reaction
    for mf, mt, section in sections:
        owner = byMt.get(mt if mf in (23, 26) else (502 if mt in (502, 505, 506) else 504))
        if owner is None:
            continue
        raw = _rawSection(sourcePath, mf, mt)
        if not raw:
            continue
        rebuilt = str(section).split("\n")[:-1]
        book = owner.provenance.headerFields[PHOTOATOMIC_KEY]
        silent = ConversionReport()
        overrides = _overrides([line[:66] for line in rebuilt], [line[:66] for line in raw], silent,
                               what=f"MF{mf}/MT{mt}")
        if overrides.get("fields") or "verbatim" in overrides:
            book.setdefault("spelling", {})[f"{mf}/{mt}"] = overrides


# ---------------------------------------------------------------------------
# MF28: atomic relaxation ↔ a standalone PoPs
# ---------------------------------------------------------------------------

def _vacancyPid(element: str, *designators: int) -> str:
    return f"{element}{{{','.join(subshellLabel(d) for d in designators)}}}"


def decodeRelaxationSublibrary(endf, report: Optional[ConversionReport] = None,
                               sourcePath=None) -> Tuple[PoPs, ConversionReport]:
    """An atomic-relaxation tape (NSUB=6, parsed by ``read_endf``) → a standalone PoPs."""
    from kika._constants import ATOMIC_NUMBER_TO_NAME, ATOMIC_NUMBER_TO_SYMBOL
    from .decay_sublibrary import _keepTapeConventions, _overrides, _rawSection
    from .decode import TAPE_ID_KEY, decodeMF1MT451

    report = report if report is not None else ConversionReport()
    mt451 = endf.mf[1].mt[451]
    _, style, provenance, report = decodeMF1MT451(mt451, report)
    tapeId = getattr(endf, "tape_id", None)
    if tapeId is not None:
        provenance.headerFields[TAPE_ID_KEY] = tapeId
    section = getattr(endf.mf.get(28), "mt", {}).get(533) if 28 in endf.mf else None
    if section is None:
        raise ValueError("an atomic relaxation tape (NSUB=6) needs MF28/MT533")
    z = int(round(section._za)) // 1000
    symbol = ATOMIC_NUMBER_TO_SYMBOL[z]
    pops = PoPs(name=f"ENDF atomic relaxation for Z = {z}", version="1.0", styles=[style],
                provenance=provenance)
    atomic = Atomic()
    energies = {}
    for shell in section.subshells:
        decayData = None
        if shell.transitions:
            modes = []
            for i, t in enumerate(shell.transitions):
                emitted = "photon" if t.radiative else "e-"
                vacancy = _vacancyPid(symbol, t.subj) if t.radiative else _vacancyPid(symbol, t.subj, t.subk)
                path = DecayPath([Decay(index=0, products=[Product(pid=emitted, label=emitted),
                                                           Product(pid=vacancy, label=vacancy)])])
                modes.append(DecayMode(label=str(i), mode="atomicRelaxation", probability=float(t.probability),
                                       decayPath=path, probabilityLabel="BR"))
            decayData = DecayData(decayModes=DecayModes(modes))
            energies[subshellLabel(shell.subi)] = [float(t.energy) for t in shell.transitions]
        atomic.configurations.append(Configuration(
            subshell=subshellLabel(shell.subi), electronNumber=float(shell.electrons),
            bindingEnergy=PhysicalQuantity(value=float(shell.binding_energy), unit="eV", label=EVAL_LABEL),
            decayData=decayData))
    element = ChemicalElement(symbol=symbol, Z=z, name=ATOMIC_NUMBER_TO_NAME.get(z),
                              atomic=atomic)
    pops.chemicalElements[symbol] = element
    book = {"za": section._za, "awr": section._awr, "head": [section._l1, section._l2, section._n2],
            "mat": section._mat, "transitionEnergies": energies,
            "subshellFields": [[s.c2, s.l1, s.l2, s.ntr_as_written, list(s.values[2:6])] for s in section.subshells],
            "pad": [section.pad.pairs, section.pad.values, section.pad.interp] if hasattr(section, "pad") else None}
    provenance.headerFields[RELAXATION_KEY] = book
    if sourcePath is not None:
        rebuilt = str(_buildMF28(pops, book, int(section._mat))).split("\n")[:-1]
        raw = _rawSection(sourcePath, 28, 533)
        if raw:
            book["overrides"] = _overrides([l[:66] for l in rebuilt], [l[:66] for l in raw], report,
                                           what="MF28/MT533")
    _keepTapeConventions(provenance, mt451, sourcePath)
    pops.report = report
    return pops, report


def _element(pops) -> ChemicalElement:
    elements = [e for e in pops.chemicalElements.values() if getattr(e, "atomic", None) is not None]
    if len(elements) != 1:
        raise ValueError(f"an atomic relaxation PoPs carries one element with atomic data; this one has "
                         f"{len(elements)}")
    return elements[0]


def _buildMF28(pops, book: dict, mat: int):
    """MF28/MT533 from the element's configurations and the fields kept from the tape."""
    from kika.endf.classes.mf28.base import MF28MT, RelaxationSubshell

    element = _element(pops)
    energies = book.get("transitionEnergies") or {}
    extras = book.get("subshellFields") or []
    shells = []
    binding = {c.subshell: float(c.bindingEnergy.value) for c in element.atomic.configurations}
    for k, c in enumerate(element.atomic.configurations):
        modes = list(c.decayData.decayModes.decayModes) if c.decayData is not None else []
        kept = energies.get(c.subshell)
        values = [float(c.bindingEnergy.value), float(c.electronNumber)]
        extra = extras[k] if k < len(extras) else [0.0, 0, 0, len(modes), [0.0, 0.0, 0.0, 0.0]]
        values += list(extra[4])
        for i, mode in enumerate(modes):
            products = mode.decayPath.decays[0].products
            vacancy = next(p.pid for p in products if p.pid not in ("photon", "e-"))
            labels = vacancy[vacancy.index("{") + 1:-1].split(",")
            subj = subshellDesignator(labels[0])
            subk = subshellDesignator(labels[1]) if len(labels) > 1 else 0
            if kept is not None and i < len(kept):
                etr = kept[i]
            else:
                # GNDS has no transition energy: the binding energies give it
                etr = binding[c.subshell] - binding.get(labels[0], 0.0) - (binding.get(labels[1], 0.0) if subk else 0.0)
            values += [float(subj), float(subk), float(etr), float(mode.probability), 0.0, 0.0]
        shells.append(RelaxationSubshell(subi=c.designator, c2=extra[0], l1=extra[1], l2=extra[2],
                                         ntr_as_written=len(modes), values=values))
    head = book.get("head") or [0, 0, 0]
    section = MF28MT(number=533, _za=book.get("za", float(1000 * element.Z)), _awr=book.get("awr", 0.0),
                     _l1=head[0], _l2=head[1], _n2=head[2], _mat=mat, subshells=shells)
    pad = book.get("pad")
    if pad is not None and hasattr(section, "pad"):
        from kika.endf.utils import PadStyle
        section.pad = PadStyle(pairs=pad[0], values=pad[1], interp=pad[2])
    return section


def encodeRelaxationSections(pops, mat: int, report: ConversionReport):
    """``[(28, 533, section)]`` of an atomic-relaxation PoPs."""
    from .decay_sublibrary import _TextSection, _applyOverrides

    book = ((getattr(pops.provenance, "headerFields", None) or {}).get(RELAXATION_KEY) or {})
    section = _buildMF28(pops, book, int(mat))
    overrides = book.get("overrides")
    if overrides:
        lines = _applyOverrides(str(section).split("\n")[:-1], overrides, report, what="MF28/MT533")
        section = _TextSection(lines, int(mat), 28)
    return [(28, 533, section)]


# ---------------------------------------------------------------------------
# MF1/451: the spelling of its HEAD and CONT records
# ---------------------------------------------------------------------------

#: On the provenance: the MF1/451 fields the rebuilt section spells otherwise than the tape.
MF1_SPELLING_KEY = "mf1_spelling"
#: The numeric records of MF1/451 (HEAD and the three CONTs); the text after
#: them is kept verbatim by the MF1 encoder already.
_MF1_RECORDS = 4


def keepMF1Spelling(provenance, styles, sourcePath) -> None:
    """Keep the MF1/451 HEAD/CONT fields the encoder would spell otherwise than the tape."""
    from types import SimpleNamespace

    from .decay_sublibrary import _overrides, _rawSection
    from .encode import encodeMF1MT451

    raw = _rawSection(sourcePath, 1, 451)
    if not raw:
        return
    from .encode import _MF1_REQUIRED_FIELDS
    fields = provenance.headerFields
    for name in _MF1_REQUIRED_FIELDS:
        if fields.get(name) is None:
            # A field the tape leaves blank (EMAX of the B-VIII.0 EADL tapes):
            # 0 in the model, and the blank comes back as the kept spelling.
            fields[name] = 0
    section, _ = encodeMF1MT451(SimpleNamespace(styles=styles, provenance=provenance),
                                mat=provenance.mat, report=ConversionReport())
    rebuilt = str(section).split("\n")[:_MF1_RECORDS]
    overrides = _overrides([l[:66] for l in rebuilt], [l[:66] for l in raw[:_MF1_RECORDS]],
                           ConversionReport(), what="MF1/MT451")
    if overrides.get("fields"):
        provenance.headerFields[MF1_SPELLING_KEY] = overrides


def spelledMF1(section, provenance, mat, report):
    """MF1/451 with the HEAD/CONT spelling kept by :func:`keepMF1Spelling` restored."""
    from .decay_sublibrary import _TextSection, _applyOverrides

    overrides = (getattr(provenance, "headerFields", None) or {}).get(MF1_SPELLING_KEY)
    if not overrides:
        return section
    lines = str(section).split("\n")[:-1]
    head = _applyOverrides(lines[:_MF1_RECORDS], {"fields": overrides["fields"], "records": _MF1_RECORDS},
                           report, what="MF1/MT451")
    return _TextSection(head + lines[_MF1_RECORDS:], int(mat), 1)
