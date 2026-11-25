"""What kika's GNDS writer keeps of a tape for the way back: :class:`EndfTapeData`.

:func:`collect` builds it from a suite decoded from ENDF;
:func:`~kika.endf.model_adapter.tape_only.tapeOnlyData` names what it does not hold. :func:`apply` puts the node entries into the provenance
derived for a suite read back from GNDS.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from kika.nuclear_data.model.endf_conversion import EndfTapeData
from kika.nuclear_data.model.provenance import EndfProvenance

from kika.gnds.tape_data import pieces, tables

from .derive import DERIVED, provenanceNodes

__all__ = ["collect", "apply", "levelQms"]


def levelQms(suite) -> List[Tuple[str, object, float, float, str]]:
    """``(path, reaction, QM, derived, residual)`` where QI plus the level's PoPs energy, FUDGE's
    rule, does not give MF3's QM."""
    from kika._records import format_endf_number
    from kika.nuclear_data.model.endf_conversion import reactionQI
    from .derive.reactions import _decayingResidual, _levelEnergy

    reactions = {id(reaction) for reaction in suite.reactions}
    out = []
    for path, reaction in provenanceNodes(suite):
        provenance = reaction.provenance
        if (id(reaction) not in reactions or not isinstance(provenance, EndfProvenance)
                or provenance.qm is None or int(getattr(provenance, "lr", 0) or 0) != 0
                or reaction.outputChannel is None):
            continue
        residual = _decayingResidual(reaction)
        level = _levelEnergy(suite, residual) if residual is not None else None
        qi = reactionQI(reaction, suite)
        if level is None or qi is None:
            continue
        derived = float(qi) + level
        if format_endf_number(derived) != format_endf_number(float(provenance.qm)):
            out.append((path, reaction, float(provenance.qm), derived, residual.pid))
    return out


def collect(suite) -> Optional[EndfTapeData]:
    """The tape data of a suite decoded from ENDF; ``None`` for any other suite."""
    if getattr(getattr(suite, "provenance", None), "sourceFormat", None) != "endf":
        return None
    data = EndfTapeData()
    from .tape_only import KEYS, tapeOnlyNodes

    for path, _, qm, _, _ in levelQms(suite):
        data.nodes.setdefault(path, {})["qm"] = qm
    nodes = dict(provenanceNodes(suite))
    for path, awps in _mf6Awps(suite, nodes).items():
        data.nodes.setdefault(path, {})["mf6Awp"] = awps
    for path, kind, _ in tapeOnlyNodes(suite):
        fields = nodes[path].provenance.headerFields
        kept = data.nodes.setdefault(path, {}).setdefault("headerFields", {})
        kept.update({key: fields[key] for key in KEYS[kind] if key in fields})
    data.mf32 = _mf32(suite)
    data.kinematics = _kinematics(suite)
    data.background = _background(suite)
    return data or None


def _repeats(form) -> list:
    """``[x, y, times]`` of each point a table states several times over."""
    import numpy as np

    runs = []
    for piece in pieces(form):
        xs, ys = np.asarray(piece.xs, float), np.asarray(piece.ys, float)
        same = np.flatnonzero((xs[1:] == xs[:-1]) & (ys[1:] == ys[:-1]))
        for start in same[np.r_[True, np.diff(same) > 1]] if same.size else ():
            times = 1
            while (start + times < xs.size and xs[start + times] == xs[start]
                   and ys[start + times] == ys[start]):
                times += 1
            runs.append([float(xs[start]), float(ys[start]), times])
    return runs


def _background(suite) -> Dict[str, object]:
    """Table points GNDS does not keep, by node path (and table, :func:`tables`).

    ``repeats``: a point a table states several times over, ``[x, y, times]``, which a
    GNDS table holds once. ``absent``: the background regions whose upper edge the
    GNDS writer adds a point at, where the tape has none."""
    from kika.nuclear_data.model import resonance_background
    from kika.nuclear_data.model import EVAL_LABEL

    repeats, absent = {}, {}
    how = resonance_background.plan(suite)
    for path, node in provenanceNodes(suite):
        for suffix, form in tables(node):
            runs = _repeats(form)
            if runs:
                repeats[path + suffix] = runs
            if suffix or how is None:
                continue
            wrapped = resonance_background.wrap(how, getattr(node, "ENDF_MT", None), form,
                                                EVAL_LABEL)
            regions = [name for name, flag in (wrapped[1] if wrapped else {}).items()
                       if resonance_background.DELETE in flag]
            if regions:
                absent[path] = regions
    return {key: value for key, value in (("repeats", repeats), ("absent", absent)) if value}


def _kinematics(suite) -> Dict[str, object]:
    """Each LRF=7 particle pair's masses, charges, spins, parities, PNT and SHF, by
    ``"range/reaction label"``: GNDS takes them from PoPs and the reaction."""
    import dataclasses

    out = {}
    for index, region in enumerate(getattr(suite.resonances, "resolved", None) or ()):
        for reaction in getattr(region.formalism, "resonanceReactions", None) or ():
            if reaction.kinematics is not None:
                out[f"{index}/{reaction.label}"] = dataclasses.asdict(reaction.kinematics)
    return out


def _mf6Awps(suite, nodes) -> Dict[str, Dict[int, float]]:
    """MF6's AWP by reaction path and subsection, where the PoPs mass does not give it."""
    from .derive.energy_angle import _awp

    out = {}
    for path, node in nodes.items():
        provenance = node.provenance
        fields = (provenance.headerFields or {}) if isinstance(provenance, EndfProvenance) else {}
        mf6 = fields.get("mf6")
        if not isinstance(mf6, dict):
            continue
        stated = {int(p["index"]): float(p["awp"]) for p in mf6.get("products", ())
                  if p.get("pid") and p.get("awp") is not None
                  and _awp(suite, p["pid"]) != float(p["awp"])}
        if stated:
            out[path] = stated
    return out


def _mf32(suite) -> Dict[str, object]:
    """MF32's head fields GNDS's parameter covariance has no place for, by range kind.

    NAPS of each range, NLS of an LCOMP=1 or 2 head, L2 of an LRF=7 particle-pair LIST,
    the AWRI of each unresolved L block and the HEAD's AWR where it is not MF1's."""
    from kika.endf.classes.mf32.mf32mt151 import (LCOMP1Body, LCOMP2Body, LCOMP2RMLBody,
                                                  UnresolvedBody)
    from kika.endf.parsers.parse_mf32 import parse_mf32_mt151
    from kika.nuclear_data.model.endf_conversion import MF32MT151_KEY

    texts = {}
    for covariance in getattr(suite.covarianceSuite, "parameterCovariances", None) or ():
        fields = getattr(getattr(covariance, "provenance", None), "headerFields", None) or {}
        if fields.get(MF32MT151_KEY) is not None:
            texts[id(fields[MF32MT151_KEY])] = fields[MF32MT151_KEY]
    if len(texts) != 1:
        return {}
    lines = [line for line in next(iter(texts.values())) if line[70:72].strip() == "32"]
    section = parse_mf32_mt151(lines, 151)
    if len(section.isotopes) != 1:
        return {}
    out: Dict[str, object] = {}
    mf1 = getattr(getattr(suite, "provenance", None), "awr", None)
    if section._awr is not None and mf1 is not None and float(section._awr) != float(mf1):
        out["awr"] = float(section._awr)
    ranges = section.isotopes[0].energy_ranges
    resolved = [r for r in ranges if r.lru == 1]
    for kind, found in (("resolved", resolved), ("unresolved", [r for r in ranges if r.lru == 2])):
        if len(found) != 1:
            continue
        energyRange, body = found[0], found[0].body
        entry = out[kind] = {"naps": int(energyRange.naps)}
        if isinstance(body, (LCOMP1Body, LCOMP2Body)) and body.control is not None:
            entry["nls"] = int(body.control.n1)
        if isinstance(body, LCOMP2RMLBody) and body.particle_pairs is not None:
            entry["pairsL2"] = int(body.particle_pairs.l2)
        if isinstance(body, UnresolvedBody):
            entry["awri"] = {int(block.l1): float(block.c1) for block in body.l_blocks}
            if _unresolvedNotInMF2(suite, body):
                entry["body"] = [_fields(record) for record in
                                 (energyRange.control, body.control, *body.l_blocks, body.matrix)]
    return out


def _fields(record) -> list:
    """A record's six fields, then its LIST values."""
    return [float(record.c1 or 0), float(record.c2 or 0), int(record.l1), int(record.l2),
            int(record.n1), int(record.n2), [float(v) for v in record.values]]


def _unresolvedNotInMF2(suite, body) -> bool:
    """Whether the GNDS parameter covariance cannot link MF32's unresolved rows (JEFF-4.0 Al-26)."""
    from kika.gnds.parameter_layout import unresolvedRowsNotInMF2

    widths = getattr(getattr(suite.resonances, "unresolved", None), "tabulatedWidths", None)
    if widths is None:
        return True
    rows = [(block.l1, values[6 * k + 1]) for block in body.l_blocks
            for values in [block.values] for k in range(block.n2)]
    return bool(unresolvedRowsNotInMF2(rows, widths))


def apply(suite, report) -> None:
    """State the node entries of the suite's tape data on its derived provenance. In place."""
    data = EndfTapeData.of(suite)
    if not data or not data.nodes:
        return
    nodes = {path: node for path, node in provenanceNodes(suite)}
    for path, fields in data.nodes.items():
        node = nodes.get(path)
        provenance = getattr(node, "provenance", None)
        if not isinstance(provenance, EndfProvenance) or provenance.sourceFormat != DERIVED:
            report.lost(f"kika's endfTapeData names {path}, which this suite has no derived "
                        f"ENDF bookkeeping for; what it held is not written")
            continue
        for name, value in fields.items():
            if name == "headerFields":
                provenance.headerFields.update(value)
            elif name == "mf6Awp":
                for product in (provenance.headerFields.get("mf6") or {}).get("products", ()):
                    product["awp"] = value.get(int(product["index"]), product["awp"])
            else:
                setattr(provenance, name, value)
