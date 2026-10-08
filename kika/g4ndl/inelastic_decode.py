"""G4NDL inelastic records → reactions of a :class:`~kika.nuclear_data.model.suite.ReactionSuite`.

Phase 10, step 3. The counterpart of :mod:`kika.g4ndl.decode` for
``Inelastic/``, and the mapping is the ENDF adapter's
(``kika/endf/model_adapter/{angular,energy,energy_angle}.py``) restated over
the G4NDL records, because G4NDL's inelastic sections *are* ENDF files 3-6
and 12-15 with Geant4's own bookkeeping: a G4NDL suite and an ENDF suite of
the same evaluation come out in the same shape. It is written again here, not
imported, so that the two formats do not depend on each other.

**Reactions.** One file of a base channel (``F02``-``F22``, ``F28``-``F36``)
is one reaction, MT from the directory (``F04`` → MT16). A composite file
(``F01``, ``F23``-``F27``) holds one reaction per ``sfType``: ``F01`` the
lumped MT4 and the partials MT51-91, ``F23`` MT103 and MT600-649, ... A lumped
MT that only states σ while its partials are in the same file is a
:class:`~kika.nuclear_data.model.reactions.CrossSectionSum` in ``suite.sums``:
Geant4 weighs the channel directory by it (``G4ParticleHPChannelList`` →
``G4ParticleHPInelasticCompFS::GetXsec``, ``theXsection[50]``), then picks a
partial by the partials' σ, and samples nothing of its own. A lumped MT with a
distribution of its own is a reaction: Geant4 falls back to it when the
partial it picked has none (``G4ParticleHPInelasticCompFS.cc:308-316``). ``Inelastic/CrossSection``,
the process cross section, is a ``CrossSectionSum`` labelled ``inelastic``
over every channel reaction, with no ENDF MT: it is not MT3, which also holds
capture and fission.

**What is modelled**, section by section (``dataType``):

=====  =========================================================================
3      σ(E) under ``recon`` (lin-lin, 0 K, as the elastic); ``QI`` or the base
       file's ``Qvalue`` is the channel's Q
4      the emitted particle's ``angularTwoBody`` (MF4); with a 5, the
       ``uncorrelated`` of the two
5      one LF=1 partial (``energy`` of the ``uncorrelated``); other laws and
       NK > 1 are not modelled, as in the ENDF adapter
6      one product per subsection, with its multiplicity: LAW 0 ``unspecified``,
       1 LANG=1 ``uncorrelated`` (NA=0) or ``energyAngular``, 1 LANG=2
       ``KalbachMann``, 2 and 3 ``angularTwoBody``, 4 a recoil, 6
       ``NBodyPhaseSpace``, 7 ``angularEnergy``
12-15  **not modelled**: kika's model has no photon-production forms yet, for
       ENDF either (``kika/_write.py``)
=====  =========================================================================

**What is not modelled is kept, not dropped.** Such a section goes to the
reaction's :class:`~kika.nuclear_data.model.provenance.G4NDLInelasticProvenance`
as its G4NDL text, and the report says so; the encoder parses it back. The
same happens to a section the model *could* hold but not exactly — a
corresponding-point or unit-base code on a one-dimensional table, several
interpolation regions where GNDS has no ``regions3d``.

**Interpolation code 1 is read as lin-lin** (roadmap Fase 10, D10-3): that is
what Geant4 evaluates (``G4ParticleHPInterpolator.hh``, the histogram call is
commented out), and the model holds what is transported. ENDF would read a
histogram, so the report calls it an approximation, and the declared
``(NBT, INT)`` is kept in the section's provenance (``code1``) and written back
while the table keeps its regions. A one-dimensional table only — the 58
(G4NDL 4.7.1) and 20 (JEFF-4.0) sections outside the photons are MF5 spectra,
MF6 yields and LAW=7 tables — and to the incident-energy interpolation (N-15's
MF5, Be-9's MF6 LAW=7: five sections in each library). A qualified 11 or 21
still goes verbatim. Every such fallback is
counted in the report, so a reader can see how much of a file reached the
model; the fixed point over both libraries is what shows nothing is lost.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kika.g4ndl.decode import (
    EVALUATED_LABEL, RECONSTRUCTED_LABEL, _block as _elasticBlock,
)
from kika.g4ndl.exceptions import G4NDLUnsupportedError
from kika.g4ndl.inelastic_format import formatSectionBody
from kika.g4ndl.inelastic_records import (
    CHANNEL_MT, DT_ANGULAR, DT_CROSS_SECTION, DT_ENERGY, DT_ENERGY_ANGLE,
    AngularBody, ContinuumBody, CrossSectionBody, DiscreteTwoBodyBody, EnergyAngleBody,
    EnergyBody, InelasticFSRecord, LabAngleEnergyBody, NBodyBody, Section, Tab1,
)
from kika.g4ndl.records import (
    FRAME_CM, FRAME_LAB, REP_ISOTROPIC, REP_LEGENDRE, CrossSectionRecord,
)
from kika.nuclear_data.model import (
    Add,
    AngularEnergy,
    AngularTwoBody,
    ConversionReport,
    CrossSection,
    CrossSectionSum,
    Distribution,
    EnergyAngular,
    Frame,
    G4NDLInelasticProvenance,
    Isotropic2d,
    KalbachMann,
    Legendre,
    Multiplicity,
    NBodyPhaseSpace,
    OutputChannel,
    Q,
    Reaction,
    ReactionId,
    Regions1d,
    Summands,
    Uncorrelated,
    Unspecified,
    XYs1d,
    XYs2d,
    angularAxes,
    crossSectionAxes,
    energyAngularAxes,
    energyAxes,
    fromEndfTab2,
    fromEndfTab3,
    kalbachMannAxes,
    multiplicityAxes,
    pidFromZA,
)
from kika.nuclear_data.model.enums import ENDF_INT_TO_INTERPOLATION
from kika.nuclear_data.model.enums import Interpolation as ModelInterpolation

__all__ = ["decodeInelastic", "INELASTIC_SUM_LABEL", "EMITTED", "LUMPED_MTS",
           "isLumped", "lumpOf", "recoilHref", "frameForProduct"]

#: The label of the ``Inelastic/CrossSection`` sum.
INELASTIC_SUM_LABEL = "inelastic"

#: The particle a composite channel's ``dataType`` 4/5 describe, and the one a
#: base channel's 4/5 describe (its first emitted particle).
EMITTED = {
    "F01": "n", "F23": "H1", "F24": "H2", "F25": "H3", "F26": "He3", "F27": "He4",
    **{f"F{k:02d}": "n" for k in range(2, 23)},
    "F28": "He4", "F29": "He4", "F30": "H1", "F31": "H1", "F32": "H2", "F33": "H3",
    "F34": "H1", "F35": "H1", "F36": "H2",
}

#: Composite channel → its lumped MT and the MT range of its partials.
LUMPED_MTS = {"F01": (4, range(50, 92)), "F23": (103, range(600, 650)),
              "F24": (104, range(650, 700)), "F25": (105, range(700, 750)),
              "F26": (106, range(750, 800)), "F27": (107, range(800, 850))}


def isLumped(mt: int) -> bool:
    return mt in {v[0] for v in LUMPED_MTS.values()}


def lumpOf(mt: int) -> Optional[str]:
    """The composite channel an MT belongs to (lumped or partial), else ``None``."""
    for channel, (lumped, partials) in LUMPED_MTS.items():
        if mt == lumped or mt in partials:
            return channel
    return None


def recoilHref(label: str) -> str:
    """Where a LAW=4 recoil's angular distribution lives: the ENDF adapter's spelling."""
    return (f"../../../../product[@label='{label}']"
            f"/distribution/angularTwoBody[@label='{EVALUATED_LABEL}']")


def frameForProduct(lct: int, zap: int) -> Frame:
    """ENDF's LCT for one product. 3 is the centre of mass for ``A <= 4``, lab above."""
    if lct == 1:
        return Frame.lab
    if lct == 2:
        return Frame.centerOfMass
    return Frame.centerOfMass if int(zap) % 1000 <= 4 else Frame.lab


_FRAME = {FRAME_LAB: Frame.lab, FRAME_CM: Frame.centerOfMass}


class _Verbatim(Exception):
    """A section the model cannot hold exactly; its text goes to the provenance."""


# ------------------------------------------------------------------ primitives

def _pairsOf(interp) -> List[Tuple[int, int]]:
    return list(zip(interp.nbt, interp.codes))


def _function1d(tab: Tab1, what: str, axes=None, label=None, *, code1=None, key=None):
    """A ``Tab1`` → ``XYs1d``, or ``Regions1d`` when it has more than one region.

    A code 1 becomes lin-lin, which is how Geant4 evaluates it; its declared
    pairs go to ``code1[key]`` so that the encoder can write them back.
    """
    codes = set(tab.interpolation.codes)
    if not len(tab) or not tab.interpolation.nRegions:
        raise _Verbatim(f"{what}: an empty table")
    if not codes <= {1, 2, 3, 4, 5}:
        raise _Verbatim(f"{what}: interpolation codes {sorted(codes)} on a "
                        f"one-dimensional table")
    declared = _pairsOf(tab.interpolation)
    wanted = [(b, 2 if c == 1 else c) for b, c in declared]
    if 1 in codes:
        if code1 is None:
            raise _Verbatim(f"{what}: interpolation code 1 here has no slot to keep it")
        code1[key] = [[int(b), int(c)] for b, c in declared]
    function = Regions1d.fromEndfRegions(tab.x.copy(), tab.y.copy(), wanted,
                                         axes=axes, label=label)
    # The model's regions1d stores each boundary node twice and folds the pair
    # back on the way out; a table whose boundary is itself a repeated node
    # would not come back as it was, so it is checked rather than assumed.
    x, y, pairs = function.toEndfRegions()
    if (list(pairs) != wanted or not np.array_equal(x, tab.x)
            or not np.array_equal(y, tab.y)):
        raise _Verbatim(f"{what}: its interpolation regions do not survive the "
                        f"model's regions1d exactly")
    if len(function.function1ds) == 1:
        function = function.function1ds[0]
        if axes is not None:
            function.axes = axes
        if label is not None:
            function.label = label
    return function


def _tab2Pairs(interp, what: str, *, threeD: bool = False, code1=None,
               key=None) -> List[Tuple[int, int]]:
    """The incident-energy ``(NBT, INT)`` for the model; a code 1 as lin-lin.

    As in :func:`_function1d`, the declared pairs go to ``code1[key]``. A
    qualified 11 or 21 (unit base, corresponding points) stays verbatim.
    """
    declared = _pairsOf(interp)
    pairs = [(b, 2 if c == 1 else c) for b, c in declared]
    for _, c in pairs:
        if c % 10 == 1 or c not in set(range(2, 6)) | set(range(12, 16)) | set(range(22, 26)):
            raise _Verbatim(f"{what}: incident-energy interpolation code {c}")
    if pairs != declared:
        if code1 is None:
            raise _Verbatim(f"{what}: incident-energy interpolation code 1 here has no "
                            f"slot to keep it")
        code1[key] = [[int(b), int(c)] for b, c in declared]
    if threeD and len(pairs) > 1:
        raise _Verbatim(f"{what}: {len(pairs)} incident-energy regions, and GNDS has no "
                        f"regions3d to hold them")
    return pairs


def _tab1Dict(t: Tab1) -> dict:
    """A ``Tab1`` the model has no slot for, as plain data."""
    return {"nbt": list(t.interpolation.nbt), "codes": list(t.interpolation.codes),
            "x": t.x.tolist(), "y": t.y.tolist()}


def _sha256(path: Optional[Path]) -> Optional[str]:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path is not None else None


# ------------------------------------------------------------------ entry point

def decodeInelastic(crossSection: Optional[CrossSectionRecord],
                    files: Sequence[InelasticFSRecord], suite, *, library=None,
                    report: Optional[ConversionReport] = None) -> ConversionReport:
    """Add one isotope's inelastic channels to ``suite`` (built by the elastic decoder
    or empty), and the ``Inelastic/CrossSection`` total to ``suite.sums``.

    ``files`` are the parsed ``Inelastic/Fxx`` files of the isotope, in
    channel order. Returns the report, which is also ``suite.report``'s.
    """
    report = report if report is not None else ConversionReport()
    libraryRoot = str(library.root) if library is not None else None
    libraryName = library.root.name if library is not None else None
    channelReactions: List[Tuple[str, object]] = []
    counts = {"modelled": 0, "verbatim": 0}
    for record in files:
        for reaction in _decodeFile(record, libraryRoot, libraryName, report, counts):
            if isinstance(reaction, CrossSectionSum):
                suite.sums.append(reaction)
            else:
                suite.reactions.append(reaction)
                channelReactions.append(reaction)
    if crossSection is not None:
        suite.sums.append(_inelasticSum(crossSection, channelReactions, libraryRoot,
                                        libraryName))
    from kika.g4ndl.inelastic_encode import stampSums
    stampSums(suite)
    if counts["verbatim"]:
        report.unsupportedNode(
            f"{counts['verbatim']} of {counts['modelled'] + counts['verbatim']} inelastic "
            f"sections are not in the model (photon production, laws kika has no "
            f"form for, or tables it cannot hold exactly); each is kept as G4NDL text "
            f"in its reaction's provenance and written back unchanged")
    return report


def _inelasticSum(record: CrossSectionRecord, reactions, libraryRoot, libraryName):
    cs = CrossSection()
    cs[RECONSTRUCTED_LABEL] = XYs1d(xs=record.energy.copy(), ys=record.sigma.copy(),
                                    interpolation=ModelInterpolation.linlin,
                                    axes=crossSectionAxes(), label=RECONSTRUCTED_LABEL)
    links = [Add(f"/reactionSuite/reactions/reaction[@label='{r.id.label}']/crossSection")
             for r in reactions]
    provenance = G4NDLInelasticProvenance(
        library=libraryRoot, libraryName=libraryName, channel="CrossSection",
        path=str(record.path) if record.path is not None else None,
        sha256=_sha256(record.path), header=record.header,
        bookkeeping=tuple(int(b) for b in record.bookkeeping))
    return CrossSectionSum(id=ReactionId(label=INELASTIC_SUM_LABEL), crossSection=cs,
                           provenance=provenance, summands=Summands(links))


# ------------------------------------------------------------------ one file

def _groups(record: InelasticFSRecord) -> "OrderedDict[int, List[Tuple[int, Section]]]":
    groups: "OrderedDict[int, List[Tuple[int, Section]]]" = OrderedDict()
    for k, s in enumerate(record.sections):
        mt = s.sfType if record.composite else CHANNEL_MT[record.channel]
        groups.setdefault(mt, []).append((k, s))
    return groups


def _decodeFile(record: InelasticFSRecord, libraryRoot, libraryName, report, counts):
    groups = _groups(record)
    where = record.path.name if record.path is not None else record.channel
    sha = _sha256(record.path)
    partialMTs = set()
    if record.composite:
        lumped, partials = LUMPED_MTS[record.channel]
        partialMTs = {mt for mt in groups if mt in partials}
    out = []
    for mt, sections in groups.items():
        provenance = G4NDLInelasticProvenance(
            library=libraryRoot, libraryName=libraryName, channel=record.channel,
            path=str(record.path) if record.path is not None else None, sha256=sha,
            header=record.header, composite=record.composite,
            Qvalue=record.Qvalue, Qdummy=record.Qdummy,
            sfType=mt if record.composite else None, nSections=len(record.sections))
        reaction = _decodeReaction(record, mt, sections, provenance,
                                   f"{where} MT{mt}", report, counts)
        sumOnly = (record.composite and isLumped(mt) and partialMTs
                   and all(s.dataType == DT_CROSS_SECTION or e.get("verbatim") is not None
                           for (_, s), e in zip(sections, provenance.sections))
                   and not reaction.outputChannel.products)
        if sumOnly:
            links = [Add(f"/reactionSuite/reactions/reaction[@label='MT{p}']/crossSection")
                     for p in sorted(partialMTs)]
            reaction = CrossSectionSum(id=reaction.id, crossSection=reaction.crossSection,
                                       outputChannel=reaction.outputChannel,
                                       provenance=provenance, summands=Summands(links))
        out.append(reaction)
    return out


def _decodeReaction(record, mt, sections, provenance, where, report, counts) -> Reaction:
    q = record.Qvalue if not record.composite else None
    for _, s in sections:
        if s.dataType == DT_CROSS_SECTION and record.composite:
            q = s.body.QI
    reaction = Reaction(
        id=ReactionId(label=f"MT{mt}", ENDF_MT=mt),
        crossSection=CrossSection(),
        outputChannel=OutputChannel(Q=Q(value=float(q) if q is not None else 0.0, unit="eV")),
        provenance=provenance,
    )
    if q is None:
        report.warn(f"{where}: no Q value in the file; the channel's Q is left at 0")
    channel = reaction.outputChannel
    pid = EMITTED[record.channel]
    angularEntry = None
    for position, s in sections:
        entry = {"position": position, "infoType": s.infoType, "dataType": s.dataType,
                 "dummy": s.dummy}
        provenance.sections.append(entry)
        try:
            if s.dataType == DT_CROSS_SECTION:
                _crossSection(s.body, reaction, entry)
            elif s.dataType == DT_ANGULAR:
                _angular(s.body, channel, pid, entry, f"{where} MF4")
                angularEntry = entry
            elif s.dataType == DT_ENERGY:
                if angularEntry is None or angularEntry.get("verbatim") is not None:
                    raise _Verbatim("an energy distribution with no angular one in the model")
                _energy(s.body, channel, pid, entry, f"{where} MF5")
            elif s.dataType == DT_ENERGY_ANGLE:
                _energyAngle(s.body, channel, entry, f"{where} MF6", report)
            else:
                raise _Verbatim("photon production (MF12-15) has no model form yet")
            counts["modelled"] += 1
            n1 = len(entry.get("code1") or ()) + sum(len(r.get("code1") or ())
                                                     for r in entry.get("products") or ())
            if n1:
                report.approximated(
                    f"{where} dataType={s.dataType}: {n1} table(s) with interpolation code 1 "
                    f"read lin-lin, which is how Geant4 evaluates it "
                    f"(G4ParticleHPInterpolator.hh); ENDF would read a histogram. The code "
                    f"is kept and written back")
        except (_Verbatim, G4NDLUnsupportedError) as exc:
            if s.dataType not in (12, 13, 14, 15):
                report.unsupportedNode(f"{where} dataType={s.dataType}: {exc}; kept as "
                                       f"G4NDL text in the provenance")
            for key in list(entry):
                if key not in ("position", "infoType", "dataType", "dummy"):
                    del entry[key]
            entry["verbatim"] = formatSectionBody(s, record.composite)
            counts["verbatim"] += 1
            if s.dataType == DT_ANGULAR:
                angularEntry = entry
                _dropProduct(channel, pid)
    return reaction


def _dropProduct(channel, pid):
    channel.products.products[:] = [p for p in channel.products if p.pid != pid]


# ------------------------------------------------------------------ dataType 3

def _crossSection(body: CrossSectionBody, reaction, entry) -> None:
    reaction.crossSection[RECONSTRUCTED_LABEL] = XYs1d(
        xs=body.points.x.copy(), ys=body.points.y.copy(),
        interpolation=ModelInterpolation.linlin, axes=crossSectionAxes(),
        label=RECONSTRUCTED_LABEL)
    entry["LR"] = body.LR


# ------------------------------------------------------------------ dataType 4, 5

def _angular(body: AngularBody, channel, pid, entry, where) -> None:
    frame = _FRAME[body.frameFlag]
    if body.repFlag == REP_ISOTROPIC:
        form = AngularTwoBody(angular=Isotropic2d(productFrame=frame), productFrame=frame)
    else:
        block = body.legendre if body.repFlag == REP_LEGENDRE else body.tabulated
        tag = "Legendre" if body.repFlag == REP_LEGENDRE else "table"
        angular = _elasticBlock(block, angularAxes(), where, tag)
        form = AngularTwoBody(angular=angular, productFrame=frame)
        entry["T"] = [float(r.temperature) for r in block.records]
        entry["tempdep"] = [int(r.tempdep) for r in block.records]
    entry["targetMass"] = float(body.targetMass)
    product = channel.ensureProduct(pid)
    product.distribution = Distribution()
    product.distribution[EVALUATED_LABEL] = form
    channel.genre = "twoBody"


def _energy(body: EnergyBody, channel, pid, entry, where) -> None:
    if len(body.partials) != 1 or body.partials[0].law != 1:
        raise _Verbatim(f"NK={len(body.partials)}, laws "
                        f"{[p.law for p in body.partials]}: only one LF=1 partial is "
                        f"modelled, as in the ENDF adapter")
    law = body.partials[0]
    axes = energyAxes()
    functions = []
    code1 = {}
    for k, (e, g) in enumerate(law.spectra):
        f = _function1d(g, f"{where} spectrum {k + 1}", code1=code1, key=f"spectrum {k + 1}")
        f.outerDomainValue = float(e)
        f.index = k
        functions.append(f)
    energy = fromEndfTab2(functions, _tab2Pairs(law.interpolation, where, code1=code1, key="incident"), axes=axes)
    product = channel.ensureProduct(pid)
    existing = product.distribution[EVALUATED_LABEL]
    angular = existing.angular
    if not isinstance(angular, (XYs2d, Isotropic2d)):
        raise _Verbatim(f"the angular half is a {type(angular).__name__}; uncorrelated "
                        f"admits only XYs2d or isotropic2d")
    product.distribution[EVALUATED_LABEL] = Uncorrelated(
        angular=angular, energy=energy, productFrame=existing.productFrame)
    channel.genre = "NBody"
    entry["dummy5"] = float(body.dummy)
    entry["probability"] = _tab1Dict(law.probability)
    entry["nDistFunc"] = int(law.nDistFunc)
    if code1:
        entry["code1"] = code1


# ------------------------------------------------------------------ dataType 6

def _energyAngle(body: EnergyAngleBody, channel, entry, where, report) -> None:
    built = []  # (pid, label, multiplicity, form, record dict)
    seen: Dict[str, int] = {}
    ejectile = None
    for i, p in enumerate(body.products):
        tag = f"{where} product {i + 1}"
        if not float(p.massCode).is_integer() or p.massCode < 0:
            raise _Verbatim(f"{tag}: massCode {p.massCode!r} is not a ZA")
        zap = int(p.massCode)
        pid = pidFromZA(zap, p.isomerFlag)
        label = pid if pid not in seen else f"{pid}__{i}"
        seen.setdefault(pid, i)
        frame = frameForProduct(body.frameFlag, zap)
        rec = {"massCode": float(p.massCode), "mass": float(p.mass),
               "isomerFlag": int(p.isomerFlag), "distLaw": int(p.distLaw),
               "groundStateQ": float(p.groundStateQ), "actualStateQ": float(p.actualStateQ),
               "label": label}
        code1 = {}
        multiplicity = Multiplicity(form=_function1d(p.yield_, f"{tag} yield",
                                                     axes=multiplicityAxes(),
                                                     label=EVALUATED_LABEL,
                                                     code1=code1, key="yield"))
        rec["code1"] = code1
        form = _productForm(p, frame, ejectile, rec, tag, report)
        if not rec["code1"]:
            del rec["code1"]
        if p.distLaw in (2, 3) and form is not None:
            ejectile = label
        built.append((pid, label, multiplicity, form, rec))
    for pid, label, multiplicity, form, rec in built:
        product = channel.ensureProduct(pid, label)
        product.multiplicity = multiplicity
        if form is not None:
            product.distribution = Distribution()
            product.distribution[EVALUATED_LABEL] = form
    channel.genre = "twoBody" if any(p.distLaw == 2 for p in body.products) else "NBody"
    entry["targetMass"] = float(body.targetMass)
    entry["frameFlag"] = int(body.frameFlag)
    entry["nProducts"] = int(body.nProducts)
    entry["products"] = [b[4] for b in built]


def _productForm(p, frame, ejectile, rec, tag, report):
    law = p.distLaw
    if law == 0:
        return Unspecified(productFrame=frame)
    if law == 3:
        return AngularTwoBody(angular=Isotropic2d(productFrame=frame), productFrame=frame)
    if law == 4:
        if ejectile is None:
            report.lost(f"{tag} is LAW=4, a two-body recoil, and no LAW=2 or 3 product "
                        f"precedes it: there is nothing for the recoil to point at, so "
                        f"it has a multiplicity and no distribution")
            return None
        return AngularTwoBody(recoilHref=recoilHref(ejectile), productFrame=frame)
    if law == 1:
        return _continuum(p.body, frame, rec, tag, report)
    if law == 2:
        return _twoBody(p.body, frame, rec, tag)
    if law == 6:
        b: NBodyBody = p.body
        rec["totalMass"] = float(b.totalMass)
        return Uncorrelated(angular=Isotropic2d(productFrame=frame),
                            energy=NBodyPhaseSpace(numberOfProducts=int(b.totalCount)),
                            productFrame=frame)
    assert law == 7
    return _labAngleEnergy(p.body, frame, rec, tag)


def _continuum(b: ContinuumBody, frame, rec, tag, report):
    if not b.energies:
        raise _Verbatim(f"{tag}: LAW=1 with no incident energy")
    lep = b.secondaryInterpolation
    if lep not in ENDF_INT_TO_INTERPOLATION or lep == 6:
        raise _Verbatim(f"{tag}: outgoing-energy interpolation LEP={lep}")
    interpolation = ENDF_INT_TO_INTERPOLATION[lep]
    rec.update(targetCode=float(b.targetCode), angularRep=int(b.angularRep),
               nDiscrete=[int(e.nDiscrete) for e in b.energies],
               nAngularParameters=[int(e.nAngularParameters) for e in b.energies])
    if any(e.nDiscrete for e in b.energies):
        report.lost(f"{tag}: LAW=1 with ND>0 at {sum(1 for e in b.energies if e.nDiscrete)} "
                    f"of {len(b.energies)} incident energies: those outgoing points are "
                    f"discrete lines and the model holds them as ordinary points; ND is "
                    f"kept in the provenance")
    na = {e.nAngularParameters for e in b.energies}
    if b.angularRep == 1:
        if na == {1}:
            axes = energyAxes()
            functions = [XYs1d(xs=e.rows[:, 0].copy(), ys=e.rows[:, 1].copy(),
                               interpolation=interpolation, axes=axes,
                               outerDomainValue=float(e.energy), index=k)
                         for k, e in enumerate(b.energies)]
            energy = fromEndfTab2(functions, _tab2Pairs(b.interpolation, tag, code1=rec.setdefault("code1", {}), key="incident"), axes=axes)
            return Uncorrelated(angular=Isotropic2d(productFrame=frame), energy=energy,
                                productFrame=frame)
        if 0 in na:
            raise _Verbatim(f"{tag}: LANG=1 with no f_0 column")
        axes = energyAngularAxes("energy_out")
        nodes = []
        for k, e in enumerate(b.energies):
            nodes.append(XYs2d(
                function1ds=[Legendre(coefficients=e.rows[j, 1:].copy(),
                                      outerDomainValue=float(e.rows[j, 0]), index=j)
                             for j in range(e.rows.shape[0])],
                interpolation=interpolation, outerDomainValue=float(e.energy), index=k))
        return EnergyAngular(xys3d=fromEndfTab3(nodes, _tab2Pairs(b.interpolation, tag, code1=rec.setdefault("code1", {}), key="incident",
                                                                  threeD=True), axes=axes),
                             productFrame=frame)
    if b.angularRep == 2:
        if len(na) != 1 or not na <= {2, 3}:
            raise _Verbatim(f"{tag}: LANG=2 with nAngularParameters {sorted(na)}")
        columns = {"f": 1, "r": 2}
        if na == {3}:
            columns["a"] = 3
        halves = {}
        pairs = _tab2Pairs(b.interpolation, tag, code1=rec.setdefault("code1", {}), key="incident")
        for component, column in columns.items():
            axes = kalbachMannAxes(component)
            functions = [XYs1d(xs=e.rows[:, 0].copy(), ys=e.rows[:, column].copy(),
                               interpolation=interpolation, axes=axes,
                               outerDomainValue=float(e.energy), index=k)
                         for k, e in enumerate(b.energies)]
            halves[component] = fromEndfTab2(functions, pairs, axes=axes)
        return KalbachMann(productFrame=frame, **halves)
    raise _Verbatim(f"{tag}: LAW=1 LANG={b.angularRep}")


def _twoBody(b: DiscreteTwoBodyBody, frame, rec, tag):
    functions = []
    for k, e in enumerate(b.energies):
        if e.representation == 0:
            c = np.empty(e.nCoefficients + 1)
            c[0] = 1.0
            c[1:] = e.values
            functions.append(Legendre(coefficients=c, outerDomainValue=float(e.energy),
                                      index=k))
        elif e.representation in (12, 14):
            if e.nCoefficients < 2:
                raise _Verbatim(f"{tag}: a tabulated two-body node with {e.nCoefficients} point")
            functions.append(XYs1d(xs=e.values[0::2].copy(), ys=e.values[1::2].copy(),
                                   interpolation=ENDF_INT_TO_INTERPOLATION[e.representation - 10],
                                   outerDomainValue=float(e.energy), index=k))
        else:
            raise _Verbatim(f"{tag}: LAW=2 representation {e.representation}")
    rec["nEnergy"] = int(b.nEnergy)
    angular = fromEndfTab2(functions, _tab2Pairs(b.interpolation, tag, code1=rec.setdefault("code1", {}), key="incident"), axes=angularAxes())
    return AngularTwoBody(angular=angular, productFrame=frame)


def _labAngleEnergy(b: LabAngleEnergyBody, frame, rec, tag):
    if not b.energies:
        raise _Verbatim(f"{tag}: LAW=7 with no incident energy")
    nodes = []
    for k, e in enumerate(b.energies):
        functions = []
        for m, a in enumerate(e.angles):
            f = _function1d(a.spectrum, f"{tag} energy {k + 1} mu {m + 1}",
                            code1=rec.setdefault("code1", {}), key=f"energy {k + 1} mu {m + 1}")
            f.outerDomainValue = float(a.mu)
            f.index = m
            functions.append(f)
        node = fromEndfTab2(functions, _tab2Pairs(e.interpolation, f"{tag} mu", code1=rec.setdefault("code1", {}),
                                                key=f"energy {k + 1} mu"))
        node.outerDomainValue = float(e.energy)
        node.index = k
        nodes.append(node)
    rec["nCosTh"] = [int(e.nCosTh) for e in b.energies]
    axes = energyAngularAxes("mu")
    return AngularEnergy(xys3d=fromEndfTab3(nodes, _tab2Pairs(b.interpolation, tag, code1=rec.setdefault("code1", {}), key="incident",
                                                              threeD=True), axes=axes),
                         productFrame=frame)
