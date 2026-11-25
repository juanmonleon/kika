"""The ``endfTapeData`` child of kika's applicationData institution.

It holds :class:`~kika.nuclear_data.model.endf_conversion.EndfTapeData`: tape values
GNDS has no node for, which only the way back to ENDF (and the 0 K reconstruction)
reads. A GNDS reader that ignores the institution reads the evaluation GNDS states.
Each entry is JSON with tagged containers, so dict keys, tuples and arrays come back
as they went in and every float as its shortest exact repr.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np

from kika.nuclear_data.model.endf_conversion import EndfTapeData

TAG = "endfTapeData"
VERSION = "1"
_GROUPS = ("mf32", "kinematics", "background")


def _encode(value):
    if isinstance(value, dict):
        return {"d": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"t": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"a": [_encode(v) for v in value.tolist()], "dtype": value.dtype.str}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"no {TAG} form for a {type(value).__name__}")


def _decode(value):
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"d"}:
        return {_decode(k): _decode(v) for k, v in value["d"]}
    if set(value) == {"t"}:
        return tuple(_decode(v) for v in value["t"])
    if set(value) == {"a", "dtype"}:
        return np.asarray(_decode(value["a"]), dtype=np.dtype(value["dtype"]))
    raise ValueError(f"unknown {TAG} value {sorted(value)}")


def _text(value) -> str:
    return json.dumps(_encode(value), separators=(",", ":"), allow_nan=True)


def write(institution: ET.Element, data: EndfTapeData) -> None:
    block = ET.SubElement(institution, TAG, version=VERSION)
    for path, fields in data.nodes.items():
        ET.SubElement(block, "node", path=path).text = _text(fields)
    for group in _GROUPS:
        value = getattr(data, group)
        if value:
            ET.SubElement(block, group).text = _text(value)


def read(block: ET.Element) -> EndfTapeData:
    data = EndfTapeData()
    for child in block:
        if child.attrib and (child.tag != "node" or set(child.attrib) != {"path"}):
            raise ValueError(f"unknown {TAG} attribute on {child.tag}")
        value = _decode(json.loads(child.text or "null"))
        if child.tag == "node":
            if not isinstance(value, dict) or child.get("path") in data.nodes:
                raise ValueError(f"invalid or duplicate {TAG} node {child.get('path')!r}")
            data.nodes[child.get("path")] = value
        elif child.tag in _GROUPS and not getattr(data, child.tag) and isinstance(value, dict):
            setattr(data, child.tag, value)
        else:
            raise ValueError(f"unknown or duplicate {TAG} entry {child.tag}")
    return data


def applyToModel(suite, data: EndfTapeData, report) -> None:
    """Restore the entries that belong to the model: the LRF=7 particle pairs, the
    points an MF3 table states several times over and an LRF=3 range's AWRI by l."""
    _restoreRepeats(suite, data.background.get("repeats") or {}, report)
    _restoreAwriByL(suite, data)
    from kika.nuclear_data.model.resonances.r_matrix import ChannelKinematics, ChannelParticle

    regions = getattr(suite.resonances, "resolved", None) or ()
    for key, value in data.kinematics.items():
        index, _, label = key.partition("/")
        formalism = regions[int(index)].formalism if int(index) < len(regions) else None
        reaction = next((r for r in getattr(formalism, "resonanceReactions", None) or ()
                         if r.label == label), None)
        if reaction is None:
            report.lost(f"kika's {TAG} states the particle pair of {key}, which this "
                        f"suite has no resonance reaction for")
            continue
        reaction.kinematics = ChannelKinematics(
            particleA=ChannelParticle(**value["particleA"]),
            particleB=ChannelParticle(**value["particleB"]),
            penetrability=value["penetrability"], shift=value["shift"],
            effective=value["effective"])


def _restoreAwriByL(suite, data) -> None:
    """Each R-matrix spin group's AWRI, where the tape states one per l (Dy-162, Dy-164)."""
    regions = getattr(suite.resonances, "resolved", None) or ()
    for fields in data.nodes.values():
        ranges = ((fields.get("headerFields") or {}).get("mf2Tape") or {}).get("ranges") or {}
        for index, entry in ranges.items():
            awri = entry.get("awri") or {}
            formalism = regions[int(index)].formalism if awri and int(index) < len(regions) else None
            elastic = {r.label for r in getattr(formalism, "resonanceReactions", None) or ()
                       if r.reactionMT == 2}
            for group in getattr(formalism, "spinGroups", None) or ():
                L = next((c.L for c in group.channels if c.resonanceReaction in elastic), None)
                if L in awri:
                    group.atomicWeightRatio = float(awri[L])


def _restoreRepeats(suite, repeats, report) -> None:
    from kika.nuclear_data.model.provenance import provenanceNodes

    if not repeats:
        return
    found = {path + suffix: form for path, node in provenanceNodes(suite)
             for suffix, form in tables(node)}
    for path, runs in repeats.items():
        candidates = pieces(found[path]) if path in found else []
        for x, y, times in runs:
            hits = [(piece, np.flatnonzero((piece.xs == x) & (piece.ys == y))) for piece in candidates]
            present = sum(found.size for _, found in hits)
            first = next(((piece, found[0]) for piece, found in hits if found.size), None)
            if first is None:
                report.lost(f"kika's {TAG} states ({x}, {y}) {times} times in {path}, "
                            f"whose table has no such point; it is not restored")
                continue
            piece, at = first
            if present < times:
                piece.xs = np.insert(piece.xs, at, [x] * (times - present))
                piece.ys = np.insert(piece.ys, at, [y] * (times - present))


def pieces(form) -> list:
    """The XYs1d tables of a cross-section form, background regions included."""
    from kika.nuclear_data.model import Regions1d, ResonancesWithBackground, XYs1d

    if isinstance(form, ResonancesWithBackground):
        background = form.background
        return [piece for region in ("resolvedRegion", "unresolvedRegion", "fastRegion")
                for piece in pieces(getattr(background, region, None))]
    if isinstance(form, Regions1d):
        return [piece for child in form.function1ds for piece in pieces(child)]
    return [form] if isinstance(form, XYs1d) else []


def tables(node):
    """``(suffix, form)`` for the tables of a node whose points the block may keep: its
    cross section, and an incoherent-elastic Debye-Waller integral."""
    from kika.nuclear_data.model import EVAL_LABEL

    form = getattr(getattr(node, "crossSection", None), "get", lambda _: None)(EVAL_LABEL)
    if form is not None:
        yield "", form
    ddcs = getattr(node, "doubleDifferentialCrossSection", None)
    debyeWaller = getattr(ddcs.get(EVAL_LABEL) if ddcs is not None else None,
                          "DebyeWallerIntegral", None)
    if debyeWaller is not None:
        yield ".DebyeWallerIntegral", debyeWaller
