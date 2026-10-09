""":class:`~kika.nuclear_data.model.suite.ReactionSuite` → G4NDL inelastic files.

Phase 10, step 5: the mirror of :mod:`kika.g4ndl.inelastic_decode`, built the
way :mod:`kika.g4ndl.encode` is. The model goes to the records the parser
produces (:mod:`kika.g4ndl.inelastic_records`), the records to text
(:mod:`kika.g4ndl.inelastic_format`), and every text is parsed back with the
strict reader and compared record for record before anything is written.

**Physics from the model, bookkeeping carried.** σ, Q, the angular, energy
and energy-angle forms and the multiplicities come from the model; ``dataType``
and law follow from the *shape* of each form, never from the provenance (an
``angularTwoBody`` replaced by an ``uncorrelated`` is written as MF4 + MF5).
What the model has no slot for — ``infoType``, the section ``dummy``, ``LR``,
``targetMass``, ``T`` and ``tempdep``, the ZAP/AWP/LIP and Q values of an MF6
product, ``ND``, ``targetCode`` — comes from the reaction's
:class:`~kika.nuclear_data.model.provenance.G4NDLInelasticProvenance` while its
shape still matches, and takes the value every real file has otherwise (0 for
flags, ``T`` and ``tempdep``), with a warning. A quantity that is physics and
cannot be defaulted (a ``targetMass`` for an angular section, a product's
AWP) is refused rather than invented.

**Sections the model does not carry** (photon production, laws kika has no
form for) are written back from their verbatim text, unchanged. They belong
to the reaction they were read with; a reaction whose model data changed
keeps its photon sections as they were, and the report says so, because
kika cannot rescale what it does not model.

**Order.** When every section of a file is accounted for by the provenance
(same reactions, same section count), the file's own order is kept; otherwise
the order is canonical: the lumped reaction first, then by MT, each
reaction's sections by ``dataType``.

**The sums are derived** (roadmap Phase 10, D10-1): the lumped σ (MT4,
MT103-107) and the ``Inelastic/CrossSection`` total follow their parts. A sum
that adds up is written as it is; one whose parts were edited is rebuilt
(:func:`kika.algebra.add`); one that was itself edited is refused; one that is
missing is made (:func:`_derivedSums`). :func:`partialSumCheck` measures how far
each sits from its parts. Geant4 never adds them up itself, and it uses all
three: the process σ is ``Inelastic/CrossSection``
(``G4ParticleHPInelasticData``), a channel directory is chosen by its own σ —
for a composite channel the **lumped** one, ``theXsection[50]``
(``G4ParticleHPChannelList`` → ``G4ParticleHPInelasticCompFS::GetXsec``) — and
only then a partial by the partials' σ (``SelectExitChannel``). So a lumped σ
left stale after replacing MT51 keeps (n,n') as likely as before and changes
only how it splits among levels, and a stale total keeps the process σ.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from kika.g4ndl.decode import EVALUATED_LABEL, RECONSTRUCTED_LABEL
from kika.g4ndl.encode import KEEP, _checkUnits, _replaceFile, _tab2 as _elasticTab2, targetKey
from kika.g4ndl.exceptions import G4NDLError, G4NDLFormatError, G4NDLUnsupportedError
from kika.g4ndl.inelastic_decode import (
    EMITTED, INELASTIC_SUM_LABEL, LUMPED_MTS, isLumped, lumpOf,
)
from kika.g4ndl.inelastic_format import formatInelasticFS, inelasticDifferences
from kika.g4ndl.inelastic_parse import parse_inelastic_fs, parseSectionBody, photonSlot
from kika.g4ndl.inelastic_records import (
    CHANNEL_MT, COMPOSITE_CHANNELS, DT_ANGULAR, DT_CROSS_SECTION, DT_ENERGY,
    DT_ENERGY_ANGLE, AngularBody, ContinuumBody, ContinuumEnergy, CrossSectionBody,
    DiscreteTwoBodyBody, DiscreteTwoBodyEnergy, EnergyAngleBody, EnergyBody, EnergyLaw,
    InelasticFSRecord, LabAngleEnergyAngle, LabAngleEnergyBody, LabAngleEnergyEnergy,
    NBodyBody, Pairs, ProductRecord, Section, Tab1,
)
from kika.g4ndl.parse import parse_cross_section
from kika.g4ndl.records import (
    FRAME_CM, FRAME_LAB, REP_ISOTROPIC, REP_LEGENDRE, REP_TABULATED, AngularBlock,
    CrossSectionRecord, Interpolation, LegendreRecord, TabulatedRecord,
)
from kika.g4ndl.tokens import HEADER_TAG, TokenStream
from kika.nuclear_data.model import (
    AngularEnergy, AngularTwoBody, ConversionReport, CrossSectionSum, EnergyAngular, Frame,
    G4NDLInelasticProvenance, Isotropic2d, KalbachMann, Legendre,
    NBodyPhaseSpace, Regions1d, Regions2d, Uncorrelated, Unspecified, XYs1d, XYs2d,
    toEndfTab2, toEndfTab3, zaFromPid,
)
from kika.nuclear_data.model.enums import INTERPOLATION_TO_ENDF_INT

__all__ = ["encodeInelastic", "writeInelastic", "partialSumCheck", "rebuildInelasticSums",
           "channelOf", "stampSums", "SUM_FLOOR", "SUM_RTOL"]

#: Relative tolerance of :func:`partialSumCheck`. The distributed files print
#: σ to 7 significant digits, so a sum of a few dozen partials sits within a
#: few 1e-7 of a total computed from unrounded values.
SUM_RTOL = 1.0e-4

#: Below this fraction of its maximum a sum is not compared relatively.
SUM_FLOOR = 1.0e-3

_FRAME_FLAG = {Frame.lab: FRAME_LAB, Frame.centerOfMass: FRAME_CM}
_HEADER_KEYS = ("position", "infoType", "dataType", "dummy")

#: MT → base channel.
_BASE_CHANNEL = {mt: ch for ch, mt in CHANNEL_MT.items() if ch not in COMPOSITE_CHANNELS}


def channelOf(reaction) -> Optional[str]:
    """The ``Fxx`` a reaction is written to: its provenance's, else from its MT."""
    p = reaction.provenance
    if isinstance(p, G4NDLInelasticProvenance) and p.channel and p.channel != "CrossSection":
        return p.channel
    mt = reaction.id.ENDF_MT
    if mt is None:
        return None
    return lumpOf(mt) or _BASE_CHANNEL.get(mt)


# ------------------------------------------------------------------ primitives

def _tab1(function, what: str, declared=None) -> Tab1:
    """A model table → ``Tab1``. A histogram in the model is refused: Geant4
    would read the code 1 lin-lin.

    ``declared`` is the ``(NBT, INT)`` the decoder read as lin-lin from a code 1
    (``code1`` in the provenance). It is written instead of the model's pairs
    while those are still the declared ones with 1 read as 2 — the same table
    to Geant4, and the same bytes.
    """
    if not isinstance(function, (XYs1d, Regions1d)):
        raise G4NDLUnsupportedError(f"{what}: a {type(function).__name__}; G4NDL holds a "
                                    f"pointwise table here")
    x, y, pairs = function.toEndfRegions()
    codes = [int(c) for _, c in pairs]
    if 1 in codes or not set(codes) <= {2, 3, 4, 5}:
        raise G4NDLUnsupportedError(
            f"{what}: interpolation codes {sorted(set(codes))}; Geant4 reads code 1 "
            f"lin-lin, so only 2-5 are written")
    if declared is not None:
        declared = [(int(b), int(c)) for b, c in declared]
        if [(int(b), int(c)) for b, c in pairs] == [(b, 2 if c == 1 else c)
                                                    for b, c in declared]:
            pairs = declared
            codes = [c for _, c in declared]
    return Tab1(Interpolation(tuple(int(b) for b, _ in pairs), tuple(codes)),
                np.array(x, dtype=np.float64), np.array(y, dtype=np.float64))


def _tab1FromDict(d) -> Tab1:
    return Tab1(Interpolation(tuple(int(b) for b in d["nbt"]), tuple(int(c) for c in d["codes"])),
                np.array(d["x"], dtype=np.float64), np.array(d["y"], dtype=np.float64))


def _declaredPairs(pairs, declared):
    """``declared`` (a code 1 read lin-lin) while ``pairs`` are still it with 1 read as 2."""
    pairs = [(int(b), int(c)) for b, c in pairs]
    if declared:
        declared = [(int(b), int(c)) for b, c in declared]
        if pairs == [(b, 2 if c == 1 else c) for b, c in declared]:
            return declared
    return pairs


def _interp2(form, what: str, declared=None) -> Tuple[List, Interpolation]:
    functions, pairs = toEndfTab2(form)
    pairs = _declaredPairs(pairs, declared)
    return functions, Interpolation(tuple(b for b, _ in pairs), tuple(c for _, c in pairs))


def _interp3(form, what: str, declared=None) -> Tuple[List, Interpolation]:
    nodes, pairs = toEndfTab3(form)
    pairs = _declaredPairs(pairs, declared)
    return nodes, Interpolation(tuple(b for b, _ in pairs), tuple(c for _, c in pairs))


def _code(interpolation) -> int:
    return INTERPOLATION_TO_ENDF_INT[interpolation]


def _carry(entry, key, n, default, what, report):
    values = list(entry.get(key) or []) if entry else []
    if len(values) == n:
        return values
    if values:
        report.warn(f"{what}: the provenance has {key} for {len(values)} entries and the "
                    f"model now has {n}; all are written as {default!r}")
    return [default] * n


# ------------------------------------------------------------------ entry point

def encodeInelastic(suite, *, header=KEEP, targetMass: Optional[float] = None,
                    report: Optional[ConversionReport] = None,
                    ) -> Tuple[Optional[CrossSectionRecord], Dict[str, InelasticFSRecord],
                               ConversionReport]:
    """The inelastic channels of ``suite`` → ``(crossSection, {channel: record}, report)``.

    ``crossSection`` is the ``Inelastic/CrossSection`` record, from the
    ``inelastic`` sum (``None`` when the suite has no inelastic channel).
    ``targetMass`` is the AWR for sections that need one and whose provenance
    has none. The sums are derived from their parts first
    (:func:`_derivedSums`); ``suite`` itself is not changed.
    """
    report = report if report is not None else ConversionReport()
    suite = _derivedSums(suite, report)
    byChannel: "OrderedDict[str, List]" = OrderedDict()
    total = None
    for reaction in list(suite.reactions) + list(suite.sums):
        if reaction.id.label == INELASTIC_SUM_LABEL and reaction.id.ENDF_MT is None:
            total = reaction
            continue
        ch = channelOf(reaction)
        if ch is None:
            continue
        byChannel.setdefault(ch, []).append(reaction)
    files = {}
    for ch in sorted(byChannel):
        files[ch] = _encodeFile(ch, byChannel[ch], suite, header, targetMass, report)
    cs = _encodeTotal(total, header) if total is not None else None
    return cs, files, report


def _headerOf(header, provenance) -> Optional[Tuple[str, str]]:
    if header == KEEP:
        return tuple(provenance.header) if provenance is not None and provenance.header else None
    if header is None:
        return None
    label = str(header)
    if not label or len(label.split()) != 1:
        raise ValueError(f"header label {label!r} must be one non-empty token")
    return (HEADER_TAG, label)


def _encodeTotal(reaction, header) -> CrossSectionRecord:
    form = reaction.crossSection.get(RECONSTRUCTED_LABEL) if reaction.crossSection else None
    if form is None:
        raise G4NDLUnsupportedError("the 'inelastic' sum has no 'recon' cross section")
    x, y, pairs = form.toEndfRegions()
    if {int(c) for _, c in pairs} != {2}:
        raise G4NDLUnsupportedError("the 'inelastic' sum is not lin-lin; Geant4 reads it so")
    p = reaction.provenance if isinstance(reaction.provenance, G4NDLInelasticProvenance) else None
    bk = tuple(p.bookkeeping) if p is not None and p.bookkeeping else (0, 0)
    return CrossSectionRecord(None, _headerOf(header, p), bk, np.array(x, dtype=np.float64),
                              np.array(y, dtype=np.float64))


# ------------------------------------------------------------------ one file

def _encodeFile(channel, reactions, suite, header, targetMass, report) -> InelasticFSRecord:
    composite = channel in COMPOSITE_CHANNELS
    if not composite and len(reactions) != 1:
        raise G4NDLUnsupportedError(
            f"{channel} is a single-reaction channel and the suite has "
            f"{[r.id.label for r in reactions]} for it")
    provs = [r.provenance if isinstance(r.provenance, G4NDLInelasticProvenance) else None
             for r in reactions]
    known = [p for p in provs if p is not None]
    # Sections: (position or None, mt, Section-with-text-or-body).
    pending = []
    for reaction, p in zip(reactions, provs):
        pending.extend(_reactionSections(channel, composite, reaction, p, suite, targetMass,
                                         report))
    positions = [pos for pos, _, _ in pending]
    nSections = {p.nSections for p in known}
    keepOrder = (all(pos is not None for pos in positions) and len(nSections) == 1
                 and sorted(positions) == list(range(nSections.pop())))
    if keepOrder:
        pending.sort(key=lambda t: t[0])
    else:
        if known:
            report.warn(f"{channel}: the sections are not those the file was read with, so "
                        f"they are written in canonical order (lumped first, then by MT)")
        rank = {mt: (0 if isLumped(mt) else 1, mt) for _, mt, _ in pending}
        pending.sort(key=lambda t: (rank[t[1]], t[2][1].dataType if isinstance(t[2], tuple)
                                    else t[2].dataType))
    # Verbatim bodies are parsed now, in file order, because a photon energy
    # section depends on the photon section before it in the same slot.
    photons: Dict[int, Optional[object]] = {}
    sections = []
    for _, mt, item in pending:
        if isinstance(item, tuple):  # (text, Section-without-body)
            text, head = item
            slot = photonSlot(head.sfType) if composite else 50
            body = parseSectionBody(text, head.dataType, composite, photons, slot,
                                    f"{channel} MT{mt} dataType={head.dataType}")
            sections.append(Section(head.infoType, head.dataType, head.sfType, head.dummy, body))
        else:
            sections.append(item)
    if composite and not any(isLumped(r.id.ENDF_MT) and r.crossSection
                             and RECONSTRUCTED_LABEL in r.crossSection for r in reactions):
        raise G4NDLUnsupportedError(
            f"{channel} has no lumped cross section (MT{LUMPED_MTS[channel][0]}) under "
            f"'{RECONSTRUCTED_LABEL}': Geant4 weighs the channel by it "
            f"(G4ParticleHPInelasticCompFS::GetXsec), and a file without it has a null "
            f"pointer there. Build it with rebuildInelasticSums(suite, create=True)")
    first = known[0] if known else None
    qvalue = qdummy = None
    if not composite:
        reaction = reactions[0]
        qvalue = _q(reaction)
        qdummy = int(first.Qdummy) if first is not None and first.Qdummy is not None else 0
    return InelasticFSRecord(None, _headerOf(header, first), channel, composite,
                             tuple(sections), qvalue, qdummy)


def _reactionSections(channel, composite, reaction, p, suite, targetMass, report):
    mt = reaction.id.ENDF_MT
    where = f"{channel} MT{mt}"
    entries = list(p.sections) if p is not None else []
    modelled = _modelledSections(channel, composite, reaction, entries, suite, targetMass,
                                 where, report)
    out = []
    used = set()
    for entry in entries:
        dt = entry["dataType"]
        head = Section(int(entry.get("infoType", 1)), dt, mt if composite else None,
                       int(entry["dummy"]) if composite and entry.get("dummy") is not None
                       else (0 if composite else None), None)
        if entry.get("verbatim") is not None:
            if dt in (DT_ANGULAR, DT_ENERGY, DT_ENERGY_ANGLE, 12, 13, 14, 15) and dt in modelled:
                # The model now holds what the file held verbatim: the model wins.
                continue
            out.append((entry["position"], mt, (entry["verbatim"], head)))
            continue
        if dt in modelled:
            out.append((entry["position"], mt, _withHeader(modelled[dt], head)))
            used.add(dt)
        else:
            report.warn(f"{where}: the file had a dataType={dt} section and the model no "
                        f"longer holds what it said; it is not written")
    for dt in sorted(set(modelled) - used):
        head = Section(1, dt, mt if composite else None, 0 if composite else None, None)
        out.append((None, mt, _withHeader(modelled[dt], head)))
    return out


def _withHeader(body, head: Section) -> Section:
    return Section(head.infoType, head.dataType, head.sfType, head.dummy, body)


def _entry(entries, dt) -> dict:
    for e in entries:
        if e["dataType"] == dt and e.get("verbatim") is None:
            return e
    return {}


def _q(reaction) -> float:
    """The channel's Q in eV, 0 when the model states none."""
    q = reaction.outputChannel.Q if reaction.outputChannel is not None else None
    return float(q.value) if q is not None and q.value is not None else 0.0


def _endfProvenance(reaction):
    p = getattr(reaction, "provenance", None)
    return p if getattr(p, "sourceFormat", None) == "endf" else None


def _fromEndfMF6(reaction) -> dict:
    """An ENDF-read reaction's MF6 bookkeeping, in the shape a G4NDL entry has.

    ZAP, AWP, LIP and LAW per product, ``ND`` of a LAW=1, ``APSX`` of a LAW=6,
    the section's LCT and AWR: what
    :mod:`kika.endf.model_adapter.energy_angle` keeps in the reaction's
    ``headerFields["mf6"]``. Read as plain data; nothing of ``kika.endf`` is
    imported. The ENDF Q values give the product's: QM ground state, QI actual.
    """
    p = _endfProvenance(reaction)
    mf6 = (p.headerFields or {}).get("mf6") if p is not None else None
    if not mf6:
        return {}
    qi = _q(reaction)
    qm = float(p.qm) if p.qm is not None else qi
    products = []
    for r in mf6.get("products") or []:
        if r.get("label") is None:      # a negative LAW: no model product
            continue
        rec = {"label": r["label"], "massCode": float(r["zap"]), "mass": float(r["awp"]),
               "isomerFlag": int(r["lip"]), "distLaw": int(r["law"]),
               "groundStateQ": qm, "actualStateQ": qi}
        fields = r.get("law_fields") or {}
        if "nd" in fields:
            rec["nDiscrete"] = list(fields["nd"])
        if "apsx" in fields:
            rec["totalMass"] = float(fields["apsx"])
        products.append(rec)
    out = {"products": products, "frameFlag": int(mf6.get("lct", 2))}
    if mf6.get("awr") is not None:
        out["targetMass"] = float(mf6["awr"])
    return out


def _modelledSections(channel, composite, reaction, entries, suite, targetMass, where,
                      report) -> Dict[int, object]:
    """The bodies of the dataType 3/4/5/6 sections the model states for this reaction."""
    out: Dict[int, object] = {}
    form = reaction.crossSection.get(RECONSTRUCTED_LABEL) if reaction.crossSection else None
    if form is not None:
        x, y, pairs = form.toEndfRegions()
        if {int(c) for _, c in pairs} != {2}:
            raise G4NDLUnsupportedError(
                f"{where}: the cross section has interpolation codes "
                f"{sorted({int(c) for _, c in pairs})}; G4NDL is read lin-lin. Linearise it")
        _checkUnits(form.axes, {"energy_in": "eV", "crossSection": "b"}, f"{where} sigma")
        e3 = dict(_entry(entries, DT_CROSS_SECTION))
        endf = _endfProvenance(reaction)
        if not e3 and endf is not None and endf.lr is not None:
            e3["LR"] = endf.lr
        q = _q(reaction)
        out[DT_CROSS_SECTION] = CrossSectionBody(
            q if composite else None,
            (int(e3["LR"]) if e3.get("LR") is not None else 0) if composite else None,
            Pairs(np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)))
    products = list(reaction.outputChannel.products) if reaction.outputChannel else []
    # The decaying residual of a level reaction (endf/model_adapter/residuals.py,
    # and what FUDGE's GNDS carries too) states no multiplicity and an
    # `unspecified` distribution: G4NDL has nowhere to put it and Geant4 makes the
    # recoil from kinematics, so it is not a product G4NDL writes.
    mf6Labels = {r.get("label") for r in ((_entry(entries, DT_ENERGY_ANGLE)
                                           or (_fromEndfMF6(reaction) if not entries else {}))
                                          .get("products") or [])}
    products = [p for p in products if not _isBareResidual(p, EMITTED[channel], mf6Labels)]
    # The photons of the MF12-15 sections (D10-2) are written as dataType
    # 12-15 below, not as an MF4/MF5/MF6 product.
    photonFields, photonEntry = _photonFields(reaction, entries)
    if photonFields is not None:
        from kika.g4ndl.photons import photonLabels
        labels = set(photonLabels(photonFields))
        products = [p for p in products if p.label not in labels]
        out.update(_photonSections(reaction, photonEntry, suite, targetMass, where, report))
    if form is None and (products or any(e["dataType"] == DT_CROSS_SECTION for e in entries)):
        raise G4NDLUnsupportedError(
            f"{where} has no '{RECONSTRUCTED_LABEL}' cross section: G4NDL is pointwise "
            f"sigma at 0 K, lin-lin. For an ENDF tape, add it with "
            f"kika.endf.model_adapter.pendf.readReconstructed (NJOY RECONR, 0 K)")
    if not products:
        return out
    emitted = EMITTED[channel]
    e4, e5, e6 = (_entry(entries, DT_ANGULAR), _entry(entries, DT_ENERGY),
                  _entry(entries, DT_ENERGY_ANGLE))
    if not e6 and not entries:
        e6 = _fromEndfMF6(reaction)
    if (e6 and e6.get("products")) or any(pr.multiplicity is not None for pr in products):
        out[DT_ENERGY_ANGLE] = _energyAngle(products, e6, suite, targetMass, where, report)
    elif len(products) == 1:
        # MF4/MF5 name no particle: the channel implies it, so a product of
        # another particle is a suite that says something G4NDL cannot. Geant4
        # would hand the distribution to the channel's particle anyway.
        if products[0].pid != emitted:
            raise G4NDLUnsupportedError(
                f"{where}: the product with the MF4/MF5 distribution is "
                f"{products[0].pid!r}, and {channel} emits {emitted!r}; G4NDL's "
                f"MF4/MF5 carry no particle, so it cannot be written as another")
        form = _evalForm(products[0])
        if form is None:
            return out
        angular, energy, frame = _split(form, where)
        mass = _mass(e4.get("targetMass"), targetMass, suite, where)
        out[DT_ANGULAR] = _angular(angular, frame, mass, e4, where, report)
        if energy is not None:
            out[DT_ENERGY] = _energy(energy, e5, where, report)
    else:
        raise G4NDLUnsupportedError(
            f"{where}: products {[pr.label or pr.pid for pr in products]} cannot be written: "
            f"either one product (the {emitted!r}) with a distribution and no multiplicity "
            f"(MF4/MF5) or products with multiplicities (MF6)")
    return out


def _photonFields(reaction, entries):
    """``(ENDF photon headerFields, the G4NDL photon entry)`` of a reaction, or ``(None, None)``."""
    for entry in entries:
        if entry.get("photons") is not None and entry.get("verbatim") is None:
            return entry["photons"].get("endf"), entry["photons"]
    endf = _endfProvenance(reaction)
    fields = getattr(endf, "headerFields", None) or {}
    if not entries and ({"mf12", "mf13"} & set(fields)):
        return fields, None
    return None, None


def _photonSections(reaction, bag, suite, targetMass, where, report) -> Dict[int, object]:
    from kika.g4ndl.photons import photonBodiesFromModel

    mass = targetMass if targetMass is not None else (bag or {}).get("targetMass")
    mat = getattr(_endfProvenance(reaction), "mat", None) or 0
    bodies = photonBodiesFromModel(suite, reaction, bag, mat, report, mass)
    if bodies is None:
        report.warn(f"{where}: the photons could not be written from the model (a "
                    f"cascade, or no MF12/MF13 bookkeeping); they are not written")
        return {}
    mean, angular, energies = bodies
    out = {13 if mean.__class__.__name__ == "PhotonPartialsBody" else 12: mean}
    if angular is not None:
        out[14] = angular
    if energies is not None and energies.needed:
        out[15] = energies
    return out


def _isBareResidual(product, emitted, mf6Labels=()) -> bool:
    """A product that says nothing but that it exists: only ``unspecified``
    distributions, and no multiplicity -- or the multiplicity 1 and decay channel
    of a level's residual (roadmap E5c, where a cascade hangs) that no MF6
    section states. Never the channel's emitted particle."""
    from kika.nuclear_data.model import Unspecified

    if product.pid in (emitted, "photon"):
        return False
    d = product.distribution
    forms = list(d.forms.values()) if d is not None else []
    if not (bool(forms) and all(isinstance(f, Unspecified) for f in forms)):
        return False
    if product.multiplicity is None:
        return True
    return product.outputChannel is not None and product.label not in set(mf6Labels)


def _evalForm(product):
    d = product.distribution
    if d is None or EVALUATED_LABEL not in d.keys():
        return None
    return d[EVALUATED_LABEL]


def _mass(carried, explicit, suite, where) -> float:
    if carried is not None:
        return float(carried)
    if explicit is not None:
        return float(explicit)
    for p in (getattr(suite, "provenance", None),):
        for name in ("targetMass", "awr"):
            v = getattr(p, name, None)
            if v is not None:
                return float(v)
    raise G4NDLUnsupportedError(f"{where}: no targetMass to write; pass targetMass=")


# ------------------------------------------------------------------ MF4 / MF5

def _split(form, where):
    if isinstance(form, AngularTwoBody):
        if form.recoilHref is not None:
            raise G4NDLUnsupportedError(f"{where}: a recoil reference is MF6 LAW=4, not MF4")
        return form.angular, None, form.productFrame
    if isinstance(form, Uncorrelated):
        if isinstance(form.energy, NBodyPhaseSpace):
            raise G4NDLUnsupportedError(f"{where}: an N-body phase space is MF6 LAW=6")
        return form.angular, form.energy, form.productFrame
    if isinstance(form, Isotropic2d):
        return form, None, form.productFrame
    raise G4NDLUnsupportedError(f"{where}: a {type(form).__name__} cannot be written as MF4/MF5")


def _angular(angular, frame, mass, entry, where, report) -> AngularBody:
    flag = _FRAME_FLAG.get(frame)
    if flag is None:
        raise G4NDLUnsupportedError(f"{where}: frame {frame!r} has no frameFlag")
    if angular is None or isinstance(angular, Isotropic2d):
        return AngularBody(REP_ISOTROPIC, mass, flag)
    kinds = {("Legendre" if isinstance(f, Legendre) else "table")
             for leaf in _leaves(angular) for f in leaf.function1ds}
    if len(kinds) != 1:
        raise G4NDLUnsupportedError(f"{where}: MF4 here holds Legendre or tables, not both")
    kind = kinds.pop()
    functions, interp = _elasticTab2(angular, kind)
    n = len(functions)
    temps = _carry(entry, "T", n, 0.0, f"{where} MF4", report)
    deps = _carry(entry, "tempdep", n, 0, f"{where} MF4", report)
    records = []
    for i, f in enumerate(functions):
        if kind == "Legendre":
            c = np.asarray(f.coefficients, dtype=np.float64)
            if c.size == 0 or abs(c[0] - 1.0) > 1e-12:
                raise G4NDLUnsupportedError(f"{where}: Legendre record {i + 1} has a_0 != 1")
            records.append(LegendreRecord(float(temps[i]), float(f.outerDomainValue),
                                          int(deps[i]), c[1:].copy()))
        else:
            mu, p, pairs = f.toEndfRegions()
            codes = tuple(int(c) for _, c in pairs)
            if 1 in codes:
                raise G4NDLUnsupportedError(f"{where}: table {i + 1} is a histogram in mu")
            records.append(TabulatedRecord(float(temps[i]), float(f.outerDomainValue),
                                           int(deps[i]),
                                           Interpolation(tuple(int(b) for b, _ in pairs), codes),
                                           np.array(mu, dtype=np.float64),
                                           np.array(p, dtype=np.float64)))
    block = AngularBlock(interp, tuple(records))
    if kind == "Legendre":
        return AngularBody(REP_LEGENDRE, mass, flag, legendre=block)
    return AngularBody(REP_TABULATED, mass, flag, tabulated=block)


def _leaves(node):
    if isinstance(node, Regions2d):
        out = []
        for c in node.function2ds:
            out.extend(_leaves(c))
        return out
    if isinstance(node, XYs2d):
        return [node]
    raise G4NDLUnsupportedError(f"an angular form of type {type(node).__name__}")


def _energy(energy, entry, where, report) -> EnergyBody:
    functions, interp = _interp2(energy, where, (entry.get("code1") or {}).get("incident"))
    code1 = entry.get("code1") or {}
    spectra = tuple((float(f.outerDomainValue),
                     _tab1(f, f"{where} MF5 spectrum {k + 1}", code1.get(f"spectrum {k + 1}")))
                    for k, f in enumerate(functions))
    if entry.get("probability") is not None:
        prob = _tab1FromDict(entry["probability"])
    else:
        e = [spectra[0][0], spectra[-1][0]]
        prob = Tab1(Interpolation((2,), (2,)), np.array(e), np.array([1.0, 1.0]))
    nd = entry.get("nDistFunc")
    nd = int(nd) if nd is not None and (int(nd) == len(spectra) or
                                        (int(nd) == 0 and len(spectra) == 1)) else len(spectra)
    return EnergyBody(float(entry.get("dummy5", 0.0)),
                      (EnergyLaw(1, prob, nDistFunc=nd, interpolation=interp, spectra=spectra),))


# ------------------------------------------------------------------ MF6

def _lct(products, e6) -> int:
    if e6.get("frameFlag") is not None:
        return int(e6["frameFlag"])
    frames = {}
    for pr in products:
        f = _evalForm(pr)
        if f is not None and getattr(f, "productFrame", None) is not None:
            frames[pr.pid] = f.productFrame
    vals = set(frames.values())
    if vals <= {Frame.lab}:
        return 1 if vals else 2
    if vals == {Frame.centerOfMass}:
        return 2
    return 3


def _energyAngle(products, e6, suite, targetMass, where, report) -> EnergyAngleBody:
    recs = {r["label"]: r for r in (e6.get("products") or [])}
    order = [r["label"] for r in (e6.get("products") or [])]
    byLabel = {(pr.label or pr.pid): pr for pr in products}
    labels = [lab for lab in order if lab in byLabel] + \
             [lab for lab in byLabel if lab not in order]
    mass = _mass(e6.get("targetMass"), targetMass, suite, where)
    lct = _lct(products, e6)
    out = []
    for i, lab in enumerate(labels):
        pr = byLabel[lab]
        rec = recs.get(lab, {})
        tag = f"{where} MF6 product {lab}"
        if pr.multiplicity is None or pr.multiplicity.form is None:
            raise G4NDLUnsupportedError(f"{tag} has no multiplicity; MF6 states a yield per product")
        yld = _tab1(pr.multiplicity.form, f"{tag} yield", (rec.get("code1") or {}).get("yield"))
        zap = float(rec["massCode"]) if "massCode" in rec else float(zaFromPid(pr.pid))
        if "mass" not in rec:
            raise G4NDLUnsupportedError(f"{tag}: no AWP (product mass ratio) to write")
        law, body = _productBody(_evalForm(pr), rec, tag, report, suite)
        out.append(ProductRecord(zap, float(rec["mass"]), int(rec.get("isomerFlag", 0)), law,
                                 float(rec.get("groundStateQ", 0.0)),
                                 float(rec.get("actualStateQ", 0.0)), yld, body))
    n = e6.get("nProducts")
    n = int(n) if n is not None and (int(n) == len(out) or (int(n) == 0 and len(out) == 1)) \
        else len(out)
    return EnergyAngleBody(mass, lct, n, tuple(out))


def _productBody(form, rec, tag, report, suite):
    if form is None:
        law = rec.get("distLaw")
        if law == 4:
            return 4, None
        raise G4NDLUnsupportedError(f"{tag} has no distribution")
    if isinstance(form, Unspecified):
        return 0, None
    if isinstance(form, AngularTwoBody):
        if form.recoilHref is not None:
            return 4, None
        if isinstance(form.angular, Isotropic2d):
            return 3, None
        return 2, _twoBody(form.angular, rec, tag)
    if isinstance(form, Uncorrelated):
        if not isinstance(form.angular, Isotropic2d):
            raise G4NDLUnsupportedError(f"{tag}: MF6 has no uncorrelated form with an "
                                        f"anisotropic angular half")
        if isinstance(form.energy, NBodyPhaseSpace):
            if "totalMass" not in rec:
                raise G4NDLUnsupportedError(f"{tag}: N-body phase space with no total mass")
            return 6, NBodyBody(float(rec["totalMass"]), int(form.energy.numberOfProducts))
        return 1, _continuumUncorrelated(form.energy, rec, tag, report, suite)
    if isinstance(form, EnergyAngular):
        return 1, _continuumLegendre(form.xys3d, rec, tag, report, suite)
    if isinstance(form, KalbachMann):
        return 1, _continuumKalbach(form, rec, tag, report, suite)
    if isinstance(form, AngularEnergy):
        return 7, _labAngleEnergy(form.xys3d, rec, tag, report)
    raise G4NDLUnsupportedError(f"{tag}: a {type(form).__name__} has no MF6 law")


def _targetCode(rec, suite) -> float:
    if "targetCode" in rec:
        return float(rec["targetCode"])
    key = targetKey(suite)
    return float(key.Z * 1000 + (key.A or 0))


def _continuum(rec, tag, report, suite, lang, lep, interp, energies) -> ContinuumBody:
    nds = _carry(rec, "nDiscrete", len(energies), 0, tag, report)
    rows = []
    for k, (e, table) in enumerate(energies):
        rows.append(ContinuumEnergy(float(e), int(nds[k]), table.shape[1] - 1, table))
    return ContinuumBody(_targetCode(rec, suite), lang, lep, interp, tuple(rows))


def _lepOf(functions, tag) -> int:
    codes = {_code(f.interpolation) for f in functions}
    if len(codes) != 1:
        raise G4NDLUnsupportedError(f"{tag}: the outgoing-energy interpolation varies "
                                    f"between incident energies; LEP is one code")
    return codes.pop()


def _continuumUncorrelated(energy, rec, tag, report, suite) -> ContinuumBody:
    functions, interp = _interp2(energy, tag, (rec.get("code1") or {}).get("incident"))
    if not all(isinstance(f, XYs1d) for f in functions):
        raise G4NDLUnsupportedError(f"{tag}: LAW=1 spectra are one table each, no regions")
    lep = _lepOf(functions, tag)
    energies = [(f.outerDomainValue, np.column_stack([np.asarray(f.xs, dtype=np.float64),
                                                      np.asarray(f.ys, dtype=np.float64)]))
                for f in functions]
    return _continuum(rec, tag, report, suite, 1, lep, interp, energies)


def _continuumLegendre(xys3d, rec, tag, report, suite) -> ContinuumBody:
    nodes, interp = _interp3(xys3d, tag, (rec.get("code1") or {}).get("incident"))
    lep = _lepOf(nodes, tag)
    energies = []
    for node in nodes:
        if not isinstance(node, XYs2d):
            raise G4NDLUnsupportedError(f"{tag}: an energyAngular node is not one XYs2d")
        widths = {len(f.coefficients) for f in node.function1ds}
        if len(widths) != 1:
            raise G4NDLUnsupportedError(f"{tag}: Legendre orders vary within one incident energy")
        table = np.array([[float(f.outerDomainValue), *np.asarray(f.coefficients, dtype=np.float64)]
                          for f in node.function1ds], dtype=np.float64)
        energies.append((node.outerDomainValue, table))
    return _continuum(rec, tag, report, suite, 1, lep, interp, energies)


def _continuumKalbach(form: KalbachMann, rec, tag, report, suite) -> ContinuumBody:
    halves = [("f", form.f), ("r", form.r)] + ([("a", form.a)] if form.a is not None else [])
    flist = {}
    interp = None
    for name, half in halves:
        functions, ip = _interp2(half, tag, (rec.get("code1") or {}).get("incident"))
        if interp is None:
            interp = ip
        elif ip != interp:
            raise G4NDLUnsupportedError(f"{tag}: Kalbach-Mann f, r and a differ in their "
                                        f"incident-energy interpolation")
        flist[name] = functions
    lep = _lepOf(flist["f"], tag)
    energies = []
    for k, f in enumerate(flist["f"]):
        cols = [np.asarray(f.xs, dtype=np.float64), np.asarray(f.ys, dtype=np.float64)]
        for name, _ in halves[1:]:
            g = flist[name][k]
            if not np.array_equal(np.asarray(g.xs), cols[0]) or g.outerDomainValue != f.outerDomainValue:
                raise G4NDLUnsupportedError(f"{tag}: Kalbach-Mann {name} is not on f's grid")
            cols.append(np.asarray(g.ys, dtype=np.float64))
        energies.append((f.outerDomainValue, np.column_stack(cols)))
    return _continuum(rec, tag, report, suite, 2, lep, interp, energies)


def _twoBody(angular, rec, tag) -> DiscreteTwoBodyBody:
    functions, interp = _interp2(angular, tag, (rec.get("code1") or {}).get("incident"))
    out = []
    for f in functions:
        if isinstance(f, Legendre):
            c = np.asarray(f.coefficients, dtype=np.float64)
            if c.size == 0 or abs(c[0] - 1.0) > 1e-12:
                raise G4NDLUnsupportedError(f"{tag}: a two-body Legendre with a_0 != 1")
            out.append(DiscreteTwoBodyEnergy(float(f.outerDomainValue), 0, c.size - 1, c[1:].copy()))
        elif isinstance(f, XYs1d):
            code = _code(f.interpolation)
            if code not in (2, 4):
                raise G4NDLUnsupportedError(f"{tag}: a two-body table interpolated {code}")
            vals = np.empty(2 * len(f.xs), dtype=np.float64)
            vals[0::2] = f.xs
            vals[1::2] = f.ys
            out.append(DiscreteTwoBodyEnergy(float(f.outerDomainValue), 10 + code, len(f.xs), vals))
        else:
            raise G4NDLUnsupportedError(f"{tag}: a two-body node of type {type(f).__name__}")
    n = rec.get("nEnergy")
    n = int(n) if n is not None and (int(n) == len(out) or (int(n) == 0 and len(out) == 1)) \
        else len(out)
    return DiscreteTwoBodyBody(n, interp, tuple(out))


def _labAngleEnergy(xys3d, rec, tag, report) -> LabAngleEnergyBody:
    nodes, interp = _interp3(xys3d, tag, (rec.get("code1") or {}).get("incident"))
    ncos = _carry(rec, "nCosTh", len(nodes), None, tag, report)
    out = []
    for k, node in enumerate(nodes):
        functions, mi = _interp2(node, tag,
                                 (rec.get("code1") or {}).get(f"energy {k + 1} mu"))
        code1 = rec.get("code1") or {}
        angles = tuple(LabAngleEnergyAngle(float(f.outerDomainValue),
                                           _tab1(f, f"{tag} energy {k + 1} mu {m + 1}",
                                                 code1.get(f"energy {k + 1} mu {m + 1}")))
                       for m, f in enumerate(functions))
        n = ncos[k]
        n = int(n) if n is not None and (int(n) == len(angles) or
                                         (int(n) == 0 and len(angles) == 1)) else len(angles)
        out.append(LabAngleEnergyEnergy(float(node.outerDomainValue), n, mi, angles))
    return LabAngleEnergyBody(interp, tuple(out))


# ------------------------------------------------------------------ coherence

def _sigmaOf(reaction):
    form = reaction.crossSection.get(RECONSTRUCTED_LABEL) if reaction.crossSection else None
    return form


def _sumParts(suite) -> "OrderedDict[str, Tuple[object, List]]":
    """Each sum the suite decomposes → ``(the sum, its parts)``, keyed by name.

    A lumped MT (MT4, MT103-107) whose channel has partials in the suite, over
    those partials; and the ``inelastic`` total over every channel reaction,
    with a lumped MT standing in for its channel only when the suite has none
    of its partials.
    """
    out: "OrderedDict[str, Tuple[object, List]]" = OrderedDict()
    reactions = {r.id.ENDF_MT: r for r in suite.reactions if r.id.ENDF_MT is not None}
    total = None
    for r in list(suite.sums) + list(suite.reactions):
        mt = r.id.ENDF_MT
        if r.id.label == INELASTIC_SUM_LABEL and mt is None:
            total = r
        elif mt is not None and isLumped(mt):
            ch = lumpOf(mt)
            parts = [x for m, x in reactions.items() if m != mt and lumpOf(m) == ch]
            if parts:
                out[f"MT{mt}"] = (r, parts)
    if total is not None:
        lumps = {lumpOf(m) for m in reactions if not isLumped(m)}
        parts = [x for m, x in reactions.items() if channelOf(x) is not None
                 and not (isLumped(m) and lumpOf(m) in lumps)]
        if parts:
            out[INELASTIC_SUM_LABEL] = (total, parts)
    return out


def _digest(form) -> str:
    import hashlib
    if form is None:
        return "none"
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(form.xs, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(form.ys, dtype=np.float64).tobytes())
    h.update(str(getattr(form, "interpolation", "")).encode())
    return h.hexdigest()


def _partsDigest(parts) -> str:
    import hashlib
    h = hashlib.sha256()
    for label, d in sorted((p.id.label, _digest(_sigmaOf(p))) for p in parts):
        h.update(f"{label}:{d};".encode())
    return h.hexdigest()


def stampSums(suite) -> None:
    """Record, on each sum's provenance, the digest of its σ and of its parts'.

    Called by the decoder once the isotope is read, so that the encoder can
    tell later which side of a sum that no longer adds up was edited. A sum
    without G4NDL provenance (one from an ENDF tape) is not stamped.
    """
    for total, parts in _sumParts(suite).values():
        p = total.provenance
        if isinstance(p, G4NDLInelasticProvenance):
            p.crossSectionDigest = _digest(_sigmaOf(total))
            p.partsDigest = _partsDigest(parts)


def partialSumCheck(suite, floor: float = SUM_FLOOR) -> List[dict]:
    """How far each sum sits from its parts.

    One row per lumped MT with partials in the suite (MT4, MT103-107) and one
    for the ``inelastic`` total. Compared at the sum's own nodes, inside the
    energy range every part and the sum cover (a part's table ending below
    the sum's is not a disagreement), with either one-sided limit of the parts
    at a node where one of them jumps, and only where the sum is at least
    ``floor`` of its maximum (near a threshold a few 1e-9 b of print rounding
    is an arbitrary relative error). ``maxRel`` is the largest
    ``|Σ parts - sum| / sum`` there, ``at`` its energy, ``maxAbs`` the
    largest absolute difference in barn anywhere in the common range.
    """
    rows = []
    for name, (r, parts) in _sumParts(suite).items():
        total = _sigmaOf(r)
        forms = [f for f in (_sigmaOf(p) for p in parts) if f is not None]
        if not forms:
            continue
        if total is None:
            rows.append(dict(sum=name, parts=[p.id.label for p in parts], maxRel=float("inf"),
                             at=None, maxAbs=float("inf"), upTo=None))
            continue
        x = np.asarray(total.xs, dtype=np.float64)
        y = np.asarray(total.ys, dtype=np.float64)
        top = min(float(x[-1]), max(float(f.xs[-1]) for f in forms))
        inside = x <= top
        # At a part's discontinuity (a repeated energy, e.g. the step to 0 at
        # the end of its table) the sum may follow either side: both one-sided
        # limits are tried and the smaller disagreement counts.
        diff = None
        for side in (x * (1 - 1e-12), x, x * (1 + 1e-12)):
            s = np.zeros_like(x)
            for f in forms:
                s += np.asarray(f.evaluate(side), dtype=np.float64)
            d = np.abs(s - y)
            diff = d if diff is None else np.minimum(diff, d)
        big = inside & (y >= floor * (float(y.max()) if y.size else 0.0)) & (y > 0)
        rel = np.zeros_like(x)
        rel[big] = diff[big] / y[big]
        k = int(np.argmax(rel)) if rel.size else 0
        rows.append(dict(sum=name, parts=[p.id.label for p in parts],
                         maxRel=float(rel[k]) if rel.size else 0.0,
                         at=float(x[k]) if x.size else None,
                         maxAbs=float(diff[inside].max()) if inside.any() else 0.0,
                         upTo=top))
    return rows


def _missingSums(suite) -> List[str]:
    """The sums G4NDL needs and the suite lacks: a composite channel's lumped MT
    when its partials are there, and the ``inelastic`` total when any channel is."""
    have = {r.id.ENDF_MT for r in list(suite.reactions) + list(suite.sums)}
    out = [f"MT{lumped}" for lumped, partials in LUMPED_MTS.values()
           if lumped not in have and any(m in partials for m in have if m is not None)]
    hasTotal = any(r.id.label == INELASTIC_SUM_LABEL and r.id.ENDF_MT is None
                   for r in suite.sums)
    if not hasTotal and any(channelOf(r) is not None for r in suite.reactions):
        out.append(INELASTIC_SUM_LABEL)
    return out


def rebuildInelasticSums(suite, create: bool = False, only=None) -> List[str]:
    """Replace each lumped σ and the ``inelastic`` total by the sum of its parts.

    The sum is :func:`kika.algebra.add` of the parts' tables: lin-lin on the
    union of their grids, with a repeated energy where a part starts or ends at
    a non-zero value, so it is exact everywhere and not only at the nodes.
    With ``create`` a sum the suite lacks is made first: the lumped MT of a
    composite channel whose partials are there (MT4 over MT51-91, ...) and the
    ``inelastic`` total over every channel reaction. ``only`` restricts the
    rebuild to those names (``"MT4"``, ``"inelastic"``). Mutates ``suite``;
    returns what was rebuilt.

    :func:`encodeInelastic`, and so every G4NDL write, applies the rule of
    :func:`_derivedSums` itself on a copy; calling this is only needed to see
    the rebuilt sums in the model.
    """
    from kika.algebra import LINLIN, add
    from kika.nuclear_data.model import XYs1d as _X
    from kika.nuclear_data.model import (Add, CrossSection, Q, ReactionId, Summands,
                                         crossSectionAxes)
    from kika.nuclear_data.model.enums import Interpolation as I
    if create:
        missing = set(_missingSums(suite))
        for lumped, partials in LUMPED_MTS.values():
            if f"MT{lumped}" not in missing:
                continue
            made = CrossSectionSum(id=ReactionId(label=f"MT{lumped}", ENDF_MT=lumped),
                                   crossSection=CrossSection(), summands=Summands([]))
            # Its Q is the channel's threshold: the largest Q of its partials.
            made.outputChannel.Q = Q(value=max(_q(r) for r in suite.reactions
                                               if r.id.ENDF_MT in partials), unit="eV")
            suite.sums.append(made)
        if INELASTIC_SUM_LABEL in missing:
            suite.sums.append(CrossSectionSum(id=ReactionId(label=INELASTIC_SUM_LABEL),
                                              crossSection=CrossSection(), summands=Summands([])))
    done = []
    for name, (r, parts) in _sumParts(suite).items():
        if only is not None and name not in only:
            continue
        tables = []
        for p in parts:
            f = _sigmaOf(p)
            if f is None:
                continue
            x = np.asarray(f.xs, dtype=np.float64)
            tables.append((x, np.asarray(f.ys, dtype=np.float64),
                           np.full(max(x.size - 1, 0), LINLIN)))
        if not tables:
            continue
        x, y = add(tables)
        r.crossSection[RECONSTRUCTED_LABEL] = _X(xs=x, ys=y, interpolation=I.linlin,
                                                 axes=crossSectionAxes(),
                                                 label=RECONSTRUCTED_LABEL)
        if isinstance(r, CrossSectionSum) and not len(r.summands):
            r.summands = Summands([Add(f"/reactionSuite/reactions/reaction[@label='{q.id.label}']"
                                       f"/crossSection") for q in parts])
        done.append(name)
    return done


def _derivedSums(suite, report: ConversionReport):
    """``suite``, or a copy of it with its sums derived from their parts.

    The rule (roadmap G4NDL Fase 10, D10-1): a sum the suite decomposes is
    derived data. Geant4 uses the total, the lumped σ and the partials each
    for a choice of its own, so a sum that no longer adds up is not an error
    it reports but physics it gets wrong.

    * A sum within :data:`SUM_RTOL` of its parts is written as it is, so an
      untouched library comes back byte for byte.
    * One that is off because its parts changed since it was read, or with
      nothing recording what it was read as (a suite from an ENDF tape), is
      rebuilt from them.
    * One that is off because **the sum itself** was edited is refused: its
      parts define it, so the edit belongs on them. That is the ENDF
      perturbation's rule too (a sum moves its partials only when the file
      does not decompose it).
    * One that nobody edited and is off anyway is the library's own state,
      written as it is, with a warning.
    * A sum G4NDL needs and the suite lacks (the total, a lumped MT) is made.

    The caller's suite is never mutated.
    """
    missing = _missingSums(suite)
    sums = _sumParts(suite)
    rebuild = []
    for row in partialSumCheck(suite):
        if row["maxRel"] <= SUM_RTOL:
            continue
        name = row["sum"]
        total, parts = sums[name]
        p = total.provenance
        stamped = isinstance(p, G4NDLInelasticProvenance) and p.crossSectionDigest is not None
        sumEdited = stamped and _digest(_sigmaOf(total)) != p.crossSectionDigest
        partsEdited = not stamped or _partsDigest(parts) != p.partsDigest
        what = (f"{name} differs from the sum of its {len(parts)} parts by up to "
                f"{row['maxRel']:.3g} relative (at {row['at']:.6g} eV)"
                if row["at"] is not None else f"{name} has no cross section")
        if sumEdited:
            raise G4NDLUnsupportedError(
                f"{what}, and it is the sum that was edited. Its parts define it "
                f"(Geant4 uses both), so edit the parts and the sum follows")
        if not partsEdited:
            report.warn(f"{what}, as read from the library; written as it is")
            continue
        rebuild.append(name)
        report.warn(f"{what}; rebuilt from its parts")
    if not rebuild and not missing:
        return suite
    import copy
    derived = copy.deepcopy(suite)
    rebuildInelasticSums(derived, create=bool(missing), only=set(rebuild) | set(missing))
    for name in missing:
        report.warn(f"{name}: not in the suite; made from its parts, as G4NDL needs it")
    return derived


# ------------------------------------------------------------------ on disk

def _verify(text: str, record, channel: str, what: str):
    try:
        back = parse_inelastic_fs(TokenStream(text), channel)
    except G4NDLFormatError as exc:
        raise G4NDLUnsupportedError(f"{what}: the model holds something the G4NDL grammar "
                                    f"refuses, so the file was not written. {exc}") from None
    diffs = inelasticDifferences(record, back)
    if diffs:
        raise G4NDLError(f"{what}: the written text does not read back: {'; '.join(diffs[:5])}")


def writeInelastic(suite, root, *, compressed: bool = False, header=KEEP,
                   targetMass: Optional[float] = None, elementName: Optional[str] = None,
                   removeStale: bool = True) -> ConversionReport:
    """Write the suite's inelastic channels into the library directory ``root``.

    ``root/Inelastic/CrossSection/<name>`` (when the suite has the
    ``inelastic`` sum) and ``root/Inelastic/Fxx/<name>`` per channel, ``.z``
    when ``compressed``. As for the elastic, a twin of the other variant is
    removed. With ``removeStale`` (default) the isotope's file in a channel
    directory the suite has no reaction for is removed too, and the report
    names it: Geant4 would otherwise read a channel the suite does not have.

    The sums (the total, MT4, MT103-107) are derived from their parts as
    :func:`_derivedSums` says: rebuilt when a part changed, refused when the
    sum itself was edited, made when missing.
    """
    from kika.g4ndl.encode import formatCrossSection
    from kika.g4ndl.encode import recordDifferences as csDifferences
    from kika.g4ndl.names import file_name

    cs, files, report = encodeInelastic(suite, header=header, targetMass=targetMass)
    key = targetKey(suite)
    stem = file_name(key)
    if elementName is not None:
        stem = stem.rsplit("_", 1)[0] + "_" + elementName
    payloads = {}
    if cs is not None:
        text = formatCrossSection(cs)
        back = parse_cross_section(TokenStream(text))
        if csDifferences(cs, back):
            raise G4NDLError(f"Inelastic/CrossSection/{stem}: the text does not read back")
        payloads["Inelastic/CrossSection"] = text
    for ch, record in files.items():
        text = formatInelasticFS(record)
        _verify(text, record, ch, f"Inelastic/{ch}/{stem}")
        payloads[f"Inelastic/{ch}"] = text
    import zlib
    root = Path(root)
    for sub, text in payloads.items():
        data = text.encode("ascii")
        payload = zlib.compress(data, 9) if compressed else data
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        target = d / (stem + (".z" if compressed else ""))
        twin = d / (stem if compressed else stem + ".z")
        _replaceFile(target, payload)
        if twin.exists():
            twin.unlink()
            report.warn(f"removed {twin}: Geant4 reads the .z when both exist")
    if removeStale:
        inel = root / "Inelastic"
        for ch in sorted(CHANNEL_MT):
            if f"Inelastic/{ch}" in payloads:
                continue
            for name in (stem, stem + ".z"):
                stale = inel / ch / name
                if stale.exists():
                    stale.unlink()
                    report.warn(f"removed {stale}: the suite has no reaction for {ch}")
    return report
