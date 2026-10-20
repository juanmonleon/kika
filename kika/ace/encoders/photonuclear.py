"""Encode a photonuclear ACE table (``.NNu``).

The inverse of :func:`kika.ace.parsers.parse_photonuclear.parse_photonuclear`:
the grid, the reactions and the particle blocks of a
:class:`~kika.ace.classes.photonuclear.PhotonuclearAce` become XSS as NJOY2016's
``phnout`` lays it out (:mod:`kika.ace.classes.photonuclear`): ESZ and TOT, NON
and ELS when there is an elastic, THN, MTR, LQR, LSIG, SIG, the IXS entries, and
per particle PXS, PHN, MTRP, TYRP, LSIGP, SIGP, LANDP, ANDP, LDLWP and DLWP, each
right after the last. Angular records and energy laws go through the neutron
table encoders. Counts, locators, MT numbers, frames and interpolation codes are
integer words (I20), the rest reals (E20.11), as ACER spells them.
"""
from __future__ import annotations

import copy
import datetime
from typing import Optional

import numpy as np

from kika.ace.classes.header import Header
from kika.ace.classes.photonuclear import EV_PER_MEV, IXS_ENTRY_WORDS, PARTICLE_IPT, PhotonuclearAce
from kika.ace.classes.xss import pack_int_words
from kika.ace.encoders._words import Words
from kika.ace.encoders.blocks import _encode_and_record
from kika.ace.encoders.laws import encode_law_chain


def encode_photonuclear(table: PhotonuclearAce, *, zaid: int, extension: str, awr: float,
                        temperature_mev: float = 0.0, comment: str = "", date: Optional[str] = None,
                        matid: Optional[int] = None, z: int = 0, a: int = 0,
                        izaw: Optional[list] = None) -> PhotonuclearAce:
    """A copy of ``table`` with its header and XSS built from its contents.

    ``table`` needs ``za``, ``energies``, ``total``, ``heating``, ``reactions``
    and ``particles``; ``elastic`` and ``nonelastic`` together or neither.
    """
    e = np.asarray(table.energies, dtype=float)
    nes = e.size
    if nes < 1 or np.any(np.diff(e) < 0.0):
        raise ValueError("a photonuclear table needs an increasing energy grid")
    if (table.elastic is None) != (table.nonelastic is None):
        raise ValueError("the elastic and the nonelastic go together")
    for name in ("total", "heating", "elastic", "nonelastic"):
        v = getattr(table, name)
        if v is not None and np.asarray(v).shape != (nes,):
            raise ValueError(f"{name} has {np.asarray(v).size} values for a grid of {nes}")
    for p in table.particles:
        if p.ipt not in PARTICLE_IPT:
            raise ValueError(f"IPT = {p.ipt} is not a particle a photonuclear table describes")

    w = Words()
    jxs = [0] * 33
    jxs[1] = w.next
    w.reals(e / EV_PER_MEV)
    jxs[2] = w.next
    w.reals(table.total)
    if table.elastic is not None:
        jxs[3] = w.next
        w.reals(table.nonelastic)
        jxs[4] = w.next
        w.reals(table.elastic)
    else:
        jxs[3] = jxs[2]
    if table.heating is not None:
        jxs[5] = w.next
        w.reals(np.asarray(table.heating, dtype=float) / EV_PER_MEV)

    ntr = len(table.reactions)
    jxs[6] = w.next
    w.ints([r.mt for r in table.reactions])
    jxs[7] = w.next
    w.reals([r.q_value / EV_PER_MEV for r in table.reactions])
    jxs[8] = w.next
    lsig = w.reserve(ntr)
    jxs[9] = w.next
    for k, r in enumerate(table.reactions):
        w.set(lsig + k, w.next - jxs[9] + 1)
        _indexed(w, r.first_index, r.cross_section, nes, f"MT {r.mt}")

    jxs[10] = w.next
    entries = w.reserve(IXS_ENTRY_WORDS * len(table.particles))
    jxs[11] = w.next
    for k, p in enumerate(table.particles):
        _particle(w, entries + IXS_ENTRY_WORDS * k, p, nes)

    xss = np.concatenate(([0.0], np.asarray(w.values, dtype=float)))
    length = xss.size - 1
    h = Header()
    h.format_version = "legacy"
    h.zaid, h.extension = int(zaid), extension if extension.startswith(".") else "." + extension
    h.atomic_weight_ratio, h.temperature = float(awr), float(temperature_mev)
    h.date = date or datetime.date.today().strftime("%m/%d/%y")
    h.comment, h.matid = comment[:70], matid
    h.izaw_array = list(izaw) if izaw is not None else [(0, 0.0)] * 17
    nxs = [0] * 17
    nxs[1], nxs[2], nxs[3], nxs[4], nxs[5] = length, int(table.za), nes, ntr, len(table.particles)
    nxs[6], nxs[7], nxs[9], nxs[10], nxs[11], nxs[16] = 2, IXS_ENTRY_WORDS, int(table.isomeric_state), z, a, 1
    h.nxs_array, h.jxs_array = nxs, jxs
    out = copy.copy(table)
    out.header, out.xss_data = h, xss
    out.xss_int_words = pack_int_words(np.asarray(w.is_int, dtype=bool))
    return out


def _indexed(w: Words, first_index, values, nes, where) -> None:
    values = np.asarray(values, dtype=float)
    if first_index < 1 or first_index + values.size - 1 > nes:
        raise ValueError(f"{where}: {values.size} values from grid point {first_index} overrun a grid of {nes}")
    w.int(first_index, values.size)
    w.reals(values)


def _particle(w: Words, entry: int, p, nes) -> None:
    name = PARTICLE_IPT[p.ipt]
    ntrp = len(p.reactions)
    loc = {"ipt": p.ipt, "ntrp": ntrp}
    loc["pxs"] = w.next
    _indexed(w, p.production_first_index, p.production, nes, f"PXS of {name}")
    loc["phn"] = w.next
    _indexed(w, p.heating_first_index, np.asarray(p.heating, dtype=float) / EV_PER_MEV, nes, f"PHN of {name}")
    loc["mtrp"] = w.next
    w.ints([r.mt for r in p.reactions])
    loc["tyrp"] = w.next
    w.ints([r.frame for r in p.reactions])

    loc["lsigp"] = w.next
    lsigp = w.reserve(ntrp)
    loc["sigp"] = w.next
    for k, r in enumerate(p.reactions):
        w.set(lsigp + k, w.next - loc["sigp"] + 1)
        _yield(w, r.yield_, nes, f"{name} MT {r.mt}")

    loc["landp"] = w.next
    landp = w.reserve(ntrp)
    loc["andp"] = w.next
    andp = Words()
    for k, r in enumerate(p.reactions):
        if r.angular_locator in (0, -1):
            w.set(landp + k, r.angular_locator)
        elif r.angular is None:
            raise ValueError(f"{name} MT {r.mt}: an angular record is announced but missing")
        else:
            w.set(landp + k, _encode_and_record(r.angular, andp))
    _append(w, andp)

    loc["ldlwp"] = w.next
    ldlwp = w.reserve(ntrp)
    loc["dlwp"] = w.next
    dlwp = Words()
    for k, r in enumerate(p.reactions):
        w.set(ldlwp + k, dlwp.next)
        encode_law_chain(r.laws, dlwp)
    _append(w, dlwp)

    order = ("ipt", "ntrp", "pxs", "phn", "mtrp", "tyrp", "lsigp", "sigp", "landp", "andp", "ldlwp", "dlwp")
    for k, key in enumerate(order):
        w.set(entry + k, loc[key])


def _yield(w: Words, y, nes, where) -> None:
    w.int(y.mftype)
    if y.mftype == 13:
        _indexed(w, y.first_index, y.values, nes, f"SIGP of {where}")
        return
    if y.mftype not in (6, 12, 16):
        raise ValueError(f"SIGP of {where}: MFTYPE = {y.mftype}")
    e = np.asarray(y.energies, dtype=float)
    v = np.asarray(y.values, dtype=float)
    if e.shape != v.shape or len(y.breakpoints) != len(y.interpolation):
        raise ValueError(f"SIGP of {where}: {e.size} energies, {v.size} values, NBT {y.breakpoints} / INT "
                         f"{y.interpolation}")
    w.int(y.mt_mult, len(y.breakpoints))
    w.ints(y.breakpoints)
    w.ints(y.interpolation)
    w.int(e.size)
    w.reals(e / EV_PER_MEV)
    w.reals(v)


def _append(w: Words, block: Words) -> None:
    w.values.extend(block.values)
    w.is_int.extend(block.is_int)
