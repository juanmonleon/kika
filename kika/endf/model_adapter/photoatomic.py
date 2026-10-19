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
                                     subshellLabel)
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
        return "pair production"
    if mt == 517:
        return f"e- + e-_anti + {element} [pair production: nuclear field]"
    if mt == 522:
        return "photo-electric"
    if 534 <= mt <= 572:
        return f"e- + {element}{{{subshellLabel(mt - 533)}}}"
    return f"MT{mt}"


def _axes(xLabel, xUnit, yLabel, yUnit) -> Axes:
    return Axes.forFunction1d(yLabel, yUnit, xLabel, xUnit)


#: The axes FUDGE gives each MF27 table (its abscissa is called ``energy_in``
#: even where it is x in 1/Ang).
_FF_AXES = {502: ("energy_in", "1/Ang", "coherent form factor", ""),
            504: ("energy_in", "1/Ang", "incoherent scattering function", ""),
            505: ("energy_in", "eV", "real part of anomalous scattering factor", ""),
            506: ("energy_in", "eV", "imaginary part of anomalous scattering factor", "")}


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
    """MF23 (and MF27) of a photo-atomic tape → the suite's reactions, sums and forms."""
    from kika.nuclear_data.model import Add, CrossSectionSum, Summands

    mf23 = endf.mf[23]
    mf27 = endf.mf.get(27)
    suite.interaction = ATOMIC_INTERACTION
    element = suite.target
    present = sorted(mf23.mt)
    # A photo-atomic tape has no neutron; MF1's decoder adds the default projectile.
    suite.PoPs.particles.pop("n", None)
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
                    formFactor=tables[502], realAnomalousFactor=tables.get(505),
                    imaginaryAnomalousFactor=tables.get(506), label=EVAL_LABEL)
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
    target = suite.PoPs.particles.get(suite.target)
    z = getattr(target, "Z", None)
    if z is None:
        from kika._constants import SYMBOL_TO_ATOMIC_NUMBER
        z = SYMBOL_TO_ATOMIC_NUMBER.get(suite.target)
    defaults = {"za": float(1000 * int(z)) if z else 0.0,
                "awr": float(getattr(getattr(suite.provenance, "attributes", None) or {}, "get",
                                     lambda *_: 0.0)("atomic_weight_ratio") or 0.0)}
    sections = []
    for container in (suite.reactions, suite.sums):
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
            ddcs = reaction.doubleDifferentialCrossSection
            form27 = ddcs.forms.get(EVAL_LABEL) if ddcs is not None else None
            parts = {}
            if isinstance(form27, CoherentPhotonScattering):
                parts = {502: form27.formFactor, 505: form27.realAnomalousFactor, 506: form27.imaginaryAnomalousFactor}
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
    for container in (suite.reactions, suite.sums):
        for reaction in container:
            if reaction.ENDF_MT is not None:
                byMt[int(reaction.ENDF_MT)] = reaction
    for mf, mt, section in sections:
        owner = byMt.get(mt if mf == 23 else (502 if mt in (502, 505, 506) else 504))
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
