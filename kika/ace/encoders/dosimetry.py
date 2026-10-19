"""Encode reactions into a dosimetry ACE table (``.NNy``).

The inverse of :func:`kika.ace.parsers.parse_dosimetry.parse_dosimetry`: the
reactions, in the order given, become MTR, LSIG and SIGD as NJOY2016's ``dosout``
lays them out (:mod:`kika.ace.classes.dosimetry`). Counts, locators, reaction
numbers and the NBT/INT tables are integer words (I20), energies and cross
sections reals (E20.11), so :func:`kika.ace.writers.write_ace.write_ace` spells
the table as ACER does.
"""
from __future__ import annotations

import datetime
from typing import Iterable, Optional

import numpy as np

from kika.ace.classes.dosimetry import EV_PER_MEV, DosimetryAce, DosimetryReaction
from kika.ace.classes.header import Header
from kika.ace.classes.xss import pack_int_words


def encode_dosimetry(reactions: Iterable[DosimetryReaction], *, zaid: int, extension: str, za: int,
                     awr: float, temperature_mev: float, comment: str = "", date: Optional[str] = None,
                     matid: Optional[int] = None) -> DosimetryAce:
    """A :class:`DosimetryAce` with its header and XSS built from ``reactions``.

    Every reaction must have increasing energies (eV) and as many cross
    sections; ``breakpoints``/``interpolation`` either both empty (one lin-lin
    region, written NR = 0) or an interpolation table ending at the last point.
    A repeated reaction number is written as given (it is how ACER names
    several MF 10 products of one MT and LFS); whoever builds the reactions
    decides whether to allow it.
    """
    reactions = list(reactions)
    if not reactions:
        raise ValueError("a dosimetry table needs at least one reaction")
    for r in reactions:
        e, s = np.asarray(r.energies, dtype=float), np.asarray(r.cross_section, dtype=float)
        if e.ndim != 1 or e.shape != s.shape or e.size < 1:
            raise ValueError(f"reaction {r.mt}: {e.size} energies and {s.size} cross sections")
        if np.any(np.diff(e) < 0.0):
            raise ValueError(f"reaction {r.mt}: the energies decrease")
        if len(r.breakpoints) != len(r.interpolation) or (r.breakpoints and r.breakpoints[-1] != e.size):
            raise ValueError(f"reaction {r.mt}: NBT {r.breakpoints} / INT {r.interpolation} does not cover "
                             f"its {e.size} points")

    ntr = len(reactions)
    words, ints = [], []

    def put(values, integer):
        values = list(values)
        words.extend(float(v) for v in values)
        ints.extend([integer] * len(values))

    put([r.mt for r in reactions], True)                       # MTR
    lsig_at = len(words)
    put([0] * ntr, True)                                       # LSIG, filled below
    sigd = len(words) + 1
    for k, r in enumerate(reactions):
        words[lsig_at + k] = float(len(words) + 1 - sigd + 1)
        single = not r.interpolation or (len(r.interpolation) == 1 and r.interpolation[0] == 2)
        if single:
            put([0], True)
        else:
            put([len(r.breakpoints)], True)
            put(r.breakpoints, True)
            put(r.interpolation, True)
        e = np.asarray(r.energies, dtype=float)
        put([e.size], True)
        put(e / EV_PER_MEV, False)
        put(np.asarray(r.cross_section, dtype=float), False)

    xss = np.concatenate(([0.0], np.asarray(words, dtype=float)))
    length = xss.size - 1
    h = Header()
    h.format_version = "legacy"
    h.zaid, h.extension = int(zaid), extension if extension.startswith(".") else "." + extension
    h.atomic_weight_ratio, h.temperature = float(awr), float(temperature_mev)
    h.date = date or datetime.date.today().strftime("%m/%d/%y")
    h.comment, h.matid = comment[:70], matid
    h.izaw_array = [(0, 0.0)] * 17
    h.nxs_array = [0] * 17
    h.nxs_array[1], h.nxs_array[2], h.nxs_array[4] = length, int(za), ntr
    h.jxs_array = [0] * 33
    h.jxs_array[1], h.jxs_array[3], h.jxs_array[6], h.jxs_array[7], h.jxs_array[22] = (
        1, 1, 1 + ntr, sigd, length)
    table = DosimetryAce(header=h, xss_data=xss, xss_int_words=pack_int_words(np.asarray(ints, dtype=bool)),
                         za=int(za))
    table.reactions = list(reactions)
    return table
