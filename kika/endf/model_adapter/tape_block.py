"""What kika's GNDS writer keeps of a tape for the way back: :class:`EndfTapeData`.

:func:`collect` builds it from a suite decoded from ENDF;
:func:`~kika.endf.model_adapter.tape_only.tapeOnlyData` names what it does not hold. :func:`apply` puts the node entries into the provenance
derived for a suite read back from GNDS.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from kika.nuclear_data.model.endf_conversion import EndfTapeData
from kika.nuclear_data.model.provenance import EndfProvenance

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
    for path, _, qm, _, _ in levelQms(suite):
        data.nodes.setdefault(path, {})["qm"] = qm
    data.mf32 = _mf32(suite)
    return data or None


def _mf32(suite) -> Dict[str, object]:
    """MF32's head fields GNDS's parameter covariance has no place for, by range kind.

    NAPS of each range, NLS of an LCOMP=1 or 2 head, L2 of an LRF=7 particle-pair LIST and
    the AWRI of each unresolved L block."""
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
    return out


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
            else:
                setattr(provenance, name, value)
