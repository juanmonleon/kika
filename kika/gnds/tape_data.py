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
    """Restore the entries that belong to the model: the LRF=7 particle pairs and the
    points an MF3 table states several times over."""
    _restoreRepeats(suite, data.background.get("repeats") or {}, report)
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


def _restoreRepeats(suite, repeats, report) -> None:
    from kika.endf.model_adapter.derive import provenanceNodes
    from kika.endf.model_adapter.tape_block import _pieces
    from kika.nuclear_data.model import EVAL_LABEL

    if not repeats:
        return
    nodes = dict(provenanceNodes(suite))
    for path, runs in repeats.items():
        form = getattr(getattr(nodes.get(path), "crossSection", None), "get", lambda _: None)(EVAL_LABEL)
        pieces = _pieces(form) if form is not None else []
        for x, y, times in runs:
            hits = [(piece, np.flatnonzero((piece.xs == x) & (piece.ys == y))) for piece in pieces]
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
