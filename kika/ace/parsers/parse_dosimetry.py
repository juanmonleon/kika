"""Decode a dosimetry ACE table into a :class:`~kika.ace.classes.dosimetry.DosimetryAce`.

The layout is NJOY2016's ``dosout`` (``acedo.f90``), described in
:mod:`kika.ace.classes.dosimetry`. The decoder refuses instead of guessing: a
location outside XSS, a reaction whose data runs into the next one, an
interpolation law that is not an ENDF law or energies that decrease raise
``ValueError`` naming the reaction.

NJOY2016 writes the NBT and INT of an MF 10 reaction interleaved (``acedo.f90``,
the MF 10 loop) while it writes MF 3 reactions, and reads every reaction, as two
arrays. A table that carries the interleaved form for a reaction with NR > 1 is
refused: its breakpoints and laws cannot be told apart from the numbers alone.
B-VIII.1 has one such section (Ta-180m1, MT 4, LFS 0).
"""
from __future__ import annotations

import numpy as np

from kika.ace.classes.dosimetry import EV_PER_MEV, DosimetryAce, DosimetryReaction

#: The ENDF interpolation laws a TAB1 may state (ENDF-6 §0.5.2).
_LAWS = {1, 2, 3, 4, 5, 6}


def is_dosimetry_extension(extension) -> bool:
    """True for a ZAID suffix such as ``.80y``: MCNP's class letter ``y`` is dosimetry.

    Parameters
    ----------
    extension : str or None
        The suffix of the ZAID, with or without its leading dot.

    Returns
    -------
    bool
        True when ``extension`` is not empty and, stripped of blanks and lower-cased,
        ends in ``y``. The test is on the last character only: the digits before it
        are not checked.
    """
    return bool(extension) and str(extension).strip().lower().endswith("y")


def parse_dosimetry(header, xss: np.ndarray, filename=None, int_words=None) -> DosimetryAce:
    """Decode the XSS of a dosimetry table whose header is already read.

    Parameters
    ----------
    header
        The parsed ACE header; its ``nxs_array`` and ``jxs_array`` are read
        (NXS(1), NXS(2), NXS(4), JXS(3), JXS(6) and JXS(7)).
    xss : numpy.ndarray
        The 1-based XSS array (``xss[0]`` is a placeholder), ``NXS(1) + 1`` long.
        Energies in it are in MeV and cross sections in barns, as ACE writes them.
    filename : str, optional
        Stored on the result as ``filename``.
    int_words : numpy.ndarray, optional
        The integer-spelling mask of the XSS, stored on the result as
        ``xss_int_words``.

    Returns
    -------
    DosimetryAce
        The table, one ``DosimetryReaction`` per MTR entry in file order. Energies
        are converted to eV; cross sections stay in barns.

    Raises
    ------
    ValueError
        Naming the reaction when it can: if NXS or JXS is short, the XSS length is
        not NXS(1), NXS(4) is not positive, MTR, LSIG or SIGD lies outside the
        table, a reaction's data does not fit its location, NR is negative, NE
        does not fill the data, the NBT/INT pairs are not an ENDF interpolation
        table over NE points (an interleaved NBT/INT list is refused), or the
        energies decrease.
    """
    nxs = list(header.nxs_array)
    jxs = list(header.jxs_array)
    if len(nxs) < 17 or len(jxs) < 33:
        raise ValueError("dosimetry ACE: the NXS or JXS array is short")
    length, ntr = int(nxs[1]), int(nxs[4])
    if xss is None or xss.size != length + 1:
        raise ValueError(f"dosimetry ACE: XSS holds {0 if xss is None else xss.size - 1} values, NXS(1) = {length}")
    mtr, lsig, sigd = int(jxs[3]), int(jxs[6]), int(jxs[7])
    if ntr <= 0:
        raise ValueError(f"dosimetry ACE: NXS(4) = {ntr}, no reactions")
    for name, start, n in (("MTR", mtr, ntr), ("LSIG", lsig, ntr)):
        if start < 1 or start + n - 1 > length:
            raise ValueError(f"dosimetry ACE: {name} XSS({start}..{start + n - 1}) runs outside the table")
    if not 1 <= sigd <= length:
        raise ValueError(f"dosimetry ACE: SIGD = {sigd} is outside the table")

    table = DosimetryAce(filename=filename, header=header, xss_data=xss, xss_int_words=int_words,
                         za=int(nxs[2]))
    numbers = np.rint(xss[mtr:mtr + ntr]).astype(int)
    locations = np.rint(xss[lsig:lsig + ntr]).astype(int)
    ends = list(locations[1:]) + [length - sigd + 2]
    for k, (mt, loc, end) in enumerate(zip(numbers, locations, ends)):
        start, stop = sigd + loc - 1, sigd + end - 1
        if loc < 1 or stop - 1 > length or stop <= start:
            raise ValueError(f"dosimetry ACE: reaction {mt}: LSIG = {loc} does not locate its data")
        table.reactions.append(_reaction(int(mt), xss[start:stop]))
    return table


def _reaction(mt: int, words: np.ndarray) -> DosimetryReaction:
    nr = int(round(words[0]))
    if nr < 0:
        raise ValueError(f"dosimetry ACE: reaction {mt}: NR = {nr}")
    nbt = [int(round(v)) for v in words[1:1 + nr]]
    law = [int(round(v)) for v in words[1 + nr:1 + 2 * nr]]
    i = 1 + 2 * nr
    if i >= words.size:
        raise ValueError(f"dosimetry ACE: reaction {mt}: no NE after the interpolation table")
    ne = int(round(words[i]))
    if ne < 1 or i + 1 + 2 * ne != words.size:
        raise ValueError(f"dosimetry ACE: reaction {mt}: NE = {ne} does not fill its "
                         f"{words.size - i - 1} words")
    if nr and (not set(law) <= _LAWS or nbt != sorted(nbt) or nbt[-1] != ne):
        raise ValueError(f"dosimetry ACE: reaction {mt}: NBT {nbt} / INT {law} is not an interpolation "
                         f"table over {ne} points (an interleaved NBT/INT pair list is refused)")
    energies = np.array(words[i + 1:i + 1 + ne], dtype=float) * EV_PER_MEV
    if np.any(np.diff(energies) < 0.0):
        raise ValueError(f"dosimetry ACE: reaction {mt}: the energies decrease")
    sigma = np.array(words[i + 1 + ne:i + 1 + 2 * ne], dtype=float)
    return DosimetryReaction(mt=mt, energies=energies, cross_section=sigma, breakpoints=nbt, interpolation=law)
