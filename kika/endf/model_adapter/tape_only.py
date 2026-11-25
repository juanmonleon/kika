"""The ENDF data a suite holds only in its provenance, kept to write the tape back.

The decoder keeps records that have no node in the model (P(nu), LFC=1's thermal
values, sections it cannot model) verbatim in ``EndfProvenance.headerFields``. Any
writer of another format leaves them behind; :func:`tapeOnlyData` names them.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

from kika.nuclear_data.model.provenance import EndfProvenance

from .derive import provenanceNodes
from .fission_energy import MF1MT460_KEY
from .photons import PHOTONS_VERBATIM_KEY
from .productions import (ACTIVATION_MF8_KEY, ACTIVATION_MF8_OPAQUE_KEY,
                          ACTIVATION_VERBATIM_KEY, _readMf8)

__all__ = ["decayTapeOnlyData", "tapeOnlyData"]


def tapeOnlyData(suite, written: str = "the GNDS file", tapeData=None) -> List[str]:
    """One sentence per kind of ENDF data that only the suite's provenance holds.

    Each sentence names the sections it covers. None of these data changes a cross
    section that MF3 or MF2 states, as the decoder declared when it kept them.

    Parameters
    ----------
    suite
        A ``ReactionSuite``, normally one decoded from an ENDF tape. A suite with no
        ENDF provenance gives an empty list.
    written
        What is being written instead of the tape, as the sentences name it.
    tapeData
        The :class:`EndfTapeData` written beside it, whose entries are not lost.

    Returns
    -------
    list of str
        In a fixed order: MF1/455, MF1/458, MF1/460, MF5, MF6, MF8, MF9/MF10 and
        the photon files.
    """
    def held(path, name):
        entry = tapeData.nodes.get(path, {}) if tapeData is not None else {}
        return name in entry or name in entry.get("headerFields", {})

    found: Dict[str, list] = defaultdict(list)
    for path, kind, info in tapeOnlyNodes(suite):
        if not held(path, KEYS[kind][0]):
            found[kind].append(info)

    sentences = []
    if found["ldg"]:
        sentences.append(
            "MF1/455 (LDG=1): the energy dependence of the precursor decay constants")
    if found["lfc"]:
        names = ", ".join(dict.fromkeys(name for names in found["lfc"] for name in names))
        sentences.append(f"MF1/458 (LFC=1): the thermal values, their uncertainties and "
                         f"the LDRV of {names}")
    if found["mf460"]:
        sentences.append("MF1/460: the delayed fission photons")
    if found["mf5"]:
        parts = [f"MT{mt} (LF={', '.join(str(lf) for lf in lfs)})" for mt, lfs in found["mf5"]]
        sentences.append(f"MF5: the subsections of a law kika does not model "
                         f"({', '.join(parts)})")
    sentences.extend(_mf6Sentences(found["mf6"]))
    for opaque in found["mf8opaque"]:
        count = sum(len(states) for states in opaque.values())
        sentences.append(f"MF8: {count} product state(s) that no MF9 or MF10 describes "
                         f"({_mts(opaque)})")
    for decay in found["mf8decay"]:
        sentences.append(f"MF8 (NO=0): the decay data stated with the product states "
                         f"({', '.join(f'MT{mt}' for mt in decay)})")
    for activation in found["activation"]:
        sentences.append(f"MF9/MF10: the sections kika could not model "
                         f"(MF{', MF'.join(activation)})")
    for photons in found["photons"]:
        sentences.append(f"photons: the sections kika could not model "
                         f"(MF{', MF'.join(photons)})")
    if found["mf7"]:
        sentences.append(f"MF7: the B array's digits, LLN, each TAB1 and TAB2's interpolation "
                         f"regions and each temperature's LI "
                         f"({', '.join(f'MT{mt}' for mt in dict.fromkeys(found['mf7']))})")
    if found["mf7mt451"]:
        sentences.append("MF7/MT451: NAS and the digits of SFI as the tape writes them")
    sentences.extend(_levelQmSentences(suite, held))
    return [f"not in {written}, kept only in the suite's ENDF provenance to write "
            f"the tape back: {sentence}" for sentence in sentences]


#: The provenance keys each kind of tape-only data is kept under, the first one marking it.
KEYS = {
    "ldg": ("ldg", "decayInterpolation", "decayEnergies", "decayData"),
    "lfc": ("replaced", "ldrv", "regions", "order", "unplaced"),
    "mf460": (MF1MT460_KEY,),
    "mf5": ("mf5",),
    "mf6": ("mf6",),
    "mf8opaque": (ACTIVATION_MF8_OPAQUE_KEY,),
    "mf8decay": (ACTIVATION_MF8_KEY,),
    "activation": (ACTIVATION_VERBATIM_KEY,),
    "photons": (PHOTONS_VERBATIM_KEY,),
    "mf7": ("mf7",),
    "mf7mt451": ("mf7mt451",),
}


def tapeOnlyNodes(suite) -> Iterator[Tuple[str, str, object]]:
    """``(path, kind, what the sentence names)`` for each kind of :data:`KEYS` a node keeps."""
    for path, node in provenanceNodes(suite):
        provenance = node.provenance
        if not isinstance(provenance, EndfProvenance):
            continue
        fields = provenance.headerFields or {}
        if fields.get("ldg") == 1:
            yield path, "ldg", True
        if fields.get("replaced"):
            yield path, "lfc", list(fields["replaced"])
        mf5 = fields.get("mf5")
        # A product whose MF5 law is not modelled has no distribution, and the GNDS
        # writer already declares the empty <distribution/> it writes for it.
        if isinstance(mf5, dict) and not mf5.get("parametrised") and not _distributionless(node):
            raw = [p["lf"] for p in mf5.get("partials", ()) if "raw_lines" in p]
            if raw:
                yield path, "mf5", (_mt(path), raw)
        if "mf7" in fields:
            yield path, "mf7", getattr(node, "ENDF_MT", None) or _mt(path)
        mf6 = fields.get("mf6")
        if isinstance(mf6, dict):
            # Only the negative LAWs: a positive one kika does not map leaves its
            # product without a distribution, which the GNDS writer declares.
            kept = [p for p in mf6.get("products", ())
                    if not p.get("modelled", True) and p["law"] < 0]
            if kept:
                lawZero = sum(1 for p in mf6.get("products", ()) if p["law"] == 0)
                yield path, "mf6", (_mt(path), int(mf6.get("jp", 0) or 0), kept, lawZero)
        if node is not suite:
            continue
        if MF1MT460_KEY in fields:
            yield path, "mf460", True
        if fields.get(ACTIVATION_MF8_OPAQUE_KEY):
            yield path, "mf8opaque", fields[ACTIVATION_MF8_OPAQUE_KEY]
        decay = _mf8DecayData(fields.get(ACTIVATION_MF8_KEY) or {})
        if decay:
            yield path, "mf8decay", decay
        for kind in ("activation", "photons"):
            if fields.get(KEYS[kind][0]):
                yield path, kind, fields[KEYS[kind][0]]
        if "mf7mt451" in fields:
            yield path, "mf7mt451", True


def decayTapeOnlyData(pops, written: str = "the GNDS file") -> List[str]:
    """:func:`tapeOnlyData` for a decay or fission-yield ``PoPs`` read from ENDF."""
    from .decay_sublibrary import DECAY_KEY, VERBATIM_KEY
    from .fission_yields import FPY_KEY

    provenance = getattr(pops, "provenance", None)
    if not isinstance(provenance, EndfProvenance):
        return []
    fields = provenance.headerFields or {}
    sentences = []
    book = fields.get(DECAY_KEY)
    if isinstance(book, dict):
        if book.get("layout"):
            sentences.append("MF8/MT457: each spectrum's normalisation (FD, FC), its ERAV "
                             "and the order of its lines")
        overrides = book.get("overrides") or {}
        if "verbatim" in overrides:
            sentences.append("MF8/MT457: the section as read, which the model does not rebuild")
        elif overrides.get("fields"):
            sentences.append(f"MF8/MT457: the text of {len(overrides['fields'])} field(s) the "
                             f"model holds as another number or not at all")
    for mt, entry in sorted((fields.get(FPY_KEY) or {}).items()):
        if len(entry.get("interp") or ()) > 1:
            sentences.append(f"MF8/MT{mt}: the interpolation between incident energies (I)")
        fpyOverrides = entry.get("overrides") or {}
        if "verbatim" in fpyOverrides:
            sentences.append(f"MF8/MT{mt}: the section as read, which the model does not rebuild")
        elif fpyOverrides.get("fields"):
            sentences.append(f"MF8/MT{mt}: the text of {len(fpyOverrides['fields'])} field(s) the "
                             f"model holds as another number or not at all")
    kept = fields.get(VERBATIM_KEY) or []
    if kept:
        sentences.append(f"the sections a decay PoPs has no node for "
                         f"({', '.join(f'MF{mf}/MT{mt}' for mf, mt, _ in kept)})")
    return [f"not in {written}, kept only in the PoPs' ENDF provenance to write the tape "
            f"back: {sentence}" for sentence in sentences]


def _levelQmSentences(suite, held) -> List[str]:
    """MF3's QM of a level reaction where QI plus the level's PoPs energy, FUDGE's rule, is not it."""
    from .tape_block import levelQms

    return [f"MF3/MT{reaction.ENDF_MT}: QM {qm:.7g} eV; QI plus {residual}'s energy gives "
            f"{derived:.7g} eV"
            for path, reaction, qm, derived, residual in levelQms(suite) if not held(path, "qm")]


def _mf6Sentences(entries) -> List[str]:
    multiplicity, deferring = [], []
    for mt, jp, products, lawZero in entries:
        byParticle: Dict[str, int] = defaultdict(int)
        for product in products:
            byParticle[_particle(product["zap"])] += 1
        counts = " and ".join(f"{n} {particle}" for particle, n in byParticle.items())
        if jp > 0:
            # A JP>0 fission section's LAW=0 yields are not the model's multiplicities
            # either (those come from MF1 and MF12), so the whole section stays behind.
            extra = f", and the y(E) of its {lawZero} LAW=0 subsections" if lawZero else ""
            multiplicity.append(f"MT{mt}: {counts} subsections{extra}")
        else:
            deferring.append(f"MT{mt}: {counts} subsections")
    sentences = []
    if multiplicity:
        sentences.append(f"MF6: P(nu), the probability that one fission emits nu "
                         f"particles ({'; '.join(multiplicity)})")
    if deferring:
        sentences.append(f"MF6: the yields of the subsections that defer their "
                         f"distribution to another file ({'; '.join(deferring)})")
    return sentences


def _mf8DecayData(kept: dict) -> List[str]:
    """The MTs whose kept MF8 states carry §8.2 decay records, which no model node reads."""
    mts = []
    for mt, lines in kept.items():
        try:
            section = _readMf8(lines)
        except (ValueError, IndexError, TypeError):
            continue
        if section.no == 0 and any(len(state.lines) > 1 for state in section.states):
            mts.append(mt)
    return mts


def _distributionless(node) -> bool:
    if type(node).__name__ != "Product":
        return False
    distribution = getattr(node, "distribution", None)
    return distribution is None or len(distribution) == 0


def _particle(zap) -> str:
    zap = int(round(float(zap)))
    return {0: "photon", 1: "neutron"}.get(zap, f"ZAP={zap}")


def _mt(path: str) -> str:
    """The MT of the reaction a node sits in, from its path (``reactions[MT18]``)."""
    match = re.search(r"\[MT(\d+)\]", path)
    return match.group(1) if match else "?"


def _mts(table: dict) -> str:
    return ", ".join(f"MT{mt}" for mt in table)
