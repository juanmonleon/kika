"""Decode a photonuclear ACE table into a :class:`~kika.ace.classes.photonuclear.PhotonuclearAce`.

The layout is NJOY2016's ``phnout`` (``acepn.f90``), described in
:mod:`kika.ace.classes.photonuclear`. Angular records and energy laws are read
by the same readers as a neutron table's ANDH and DLWH, so every law those
know is read here too. The decoder refuses instead of guessing: a location
outside XSS, a count that does not fit its block, a particle type that is not
one of the seven, or a yield record of an unknown MFTYPE raise
:class:`ValueError` naming the place.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from kika.ace.classes.photonuclear import (
    EV_PER_MEV, IXS_ENTRY_WORDS, PARTICLE_IPT, ParticleProduction, PhotonuclearAce, PhotonuclearReaction,
    ProductReaction, ProductYield,
)


def is_photonuclear_extension(extension) -> bool:
    """True for a ZAID suffix such as ``.24u``: MCNP's class letter ``u`` is photonuclear."""
    return bool(extension) and str(extension).strip().lower().endswith("u")


def parse_photonuclear(header, xss: np.ndarray, filename=None, int_words=None) -> PhotonuclearAce:
    """Decode the XSS of a photonuclear table whose header is already read."""
    nxs = [int(v) for v in header.nxs_array]
    jxs = [int(v) for v in header.jxs_array]
    if len(nxs) < 17 or len(jxs) < 33:
        raise ValueError("photonuclear ACE: the NXS or JXS array is short")
    length, za, nes, ntr, ntype = nxs[1], nxs[2], nxs[3], nxs[4], nxs[5]
    if xss is None or xss.size != length + 1:
        raise ValueError(f"photonuclear ACE: XSS holds {0 if xss is None else xss.size - 1} values, NXS(1) = {length}")
    if nes < 1:
        raise ValueError(f"photonuclear ACE: NXS(3) = {nes}, no energy grid")
    if ntype and nxs[7] != IXS_ENTRY_WORDS:
        raise ValueError(f"photonuclear ACE: NXS(7) = {nxs[7]}, an IXS entry is {IXS_ENTRY_WORDS} words")
    esz, tot, non, els, thn, mtr, lqr, lsig, sig, ixsa = jxs[1:11]

    def take(name, start, n):
        if n < 0 or start < 1 or start + n - 1 > length:
            raise ValueError(f"photonuclear ACE: {name} XSS({start}..{start + n - 1}) runs outside the table")
        return np.array(xss[start:start + n], dtype=float)

    def ints(name, start, n):
        return [int(round(v)) for v in take(name, start, n)]

    table = PhotonuclearAce(filename=filename, header=header, xss_data=xss, xss_int_words=int_words, za=za,
                            isomeric_state=nxs[9])
    table.energies = take("ESZ", esz, nes) * EV_PER_MEV
    if np.any(np.diff(table.energies) < 0.0):
        raise ValueError("photonuclear ACE: the energy grid decreases")
    table.total = take("TOT", tot, nes)
    if els:
        table.elastic = take("ELS", els, nes)
        table.nonelastic = take("NON", non, nes)
    if thn:
        table.heating = take("THN", thn, nes) * EV_PER_MEV

    if ntr:
        numbers = ints("MTR", mtr, ntr)
        qs = take("LQR", lqr, ntr) * EV_PER_MEV
        locations = ints("LSIG", lsig, ntr)
        for mt, q, loc in zip(numbers, qs, locations):
            ie, values = _indexed(take, f"SIG of MT {mt}", sig + loc - 1, nes)
            table.reactions.append(PhotonuclearReaction(mt=mt, q_value=float(q), first_index=ie,
                                                        cross_section=values))

    view = SimpleNamespace(xss_data=xss, header=header, read_report=None)
    entries = [ints(f"IXS entry {k + 1}", ixsa + IXS_ENTRY_WORDS * k, IXS_ENTRY_WORDS) for k in range(ntype)]
    # where each particle's blocks end: the next particle's PXS, or the table's end
    block_ends = [e[2] for e in entries[1:]] + [length + 1]
    for entry, block_end in zip(entries, block_ends):
        table.particles.append(_particle(entry, block_end, take, ints, view, nes))
    return table


def _indexed(take, name, at, nes):
    """An IE, NE, values(NE) record; the values sit on the main grid from point IE."""
    ie, ne = (int(round(v)) for v in take(name, at, 2))
    if ie < 1 or ne < 0 or ie + ne - 1 > nes:
        raise ValueError(f"photonuclear ACE: {name}: IE = {ie}, NE = {ne} do not fit a grid of {nes}")
    return ie, take(name, at + 2, ne)


def _particle(entry, block_end, take, ints, view, nes) -> ParticleProduction:
    from kika.ace.parsers.parse_angular_distribution import read_angular_distribution
    from kika.ace.parsers.parse_energy_distributions import read_energy_distribution

    ipt, ntrp, pxs, phn, mtrp, tyrp, lsigp, sigp, landp, andp, ldlwp, dlwp = entry
    if ipt not in PARTICLE_IPT:
        raise ValueError(f"photonuclear ACE: IPT = {ipt} is not a particle a photonuclear table describes")
    name = PARTICLE_IPT[ipt]
    pie, production = _indexed(take, f"PXS of {name}", pxs, nes)
    hie, heating = _indexed(take, f"PHN of {name}", phn, nes)
    block = ParticleProduction(ipt=ipt, production_first_index=pie, production=production,
                               heating_first_index=hie, heating=heating * EV_PER_MEV)
    mts = ints(f"MTRP of {name}", mtrp, ntrp)
    frames = ints(f"TYRP of {name}", tyrp, ntrp)
    sig_at = ints(f"LSIGP of {name}", lsigp, ntrp)
    and_at = ints(f"LANDP of {name}", landp, ntrp)
    dlw_at = ints(f"LDLWP of {name}", ldlwp, ntrp)
    chains = []
    for mt, frame, ls, la, ld in zip(mts, frames, sig_at, and_at, dlw_at):
        where = f"{name} MT {mt}"
        if frame not in (1, -1):
            raise ValueError(f"photonuclear ACE: {where}: TYRP = {frame}, not ±1")
        y = _yield(take, where, sigp + ls - 1, nes)
        if la > 0:
            angular, kind = read_angular_distribution(view, andp + la - 1, mt, andp), 1
        elif la in (0, -1):
            angular, kind = None, la
        else:
            raise ValueError(f"photonuclear ACE: {where}: LANDP = {la}")
        if ld < 1:
            raise ValueError(f"photonuclear ACE: {where}: LDLWP = {ld}, no energy law")
        laws = read_energy_distribution(view, dlwp + ld - 1, dlwp)
        chains.append(laws)
        block.reactions.append(ProductReaction(mt=mt, frame=frame, yield_=y, angular_locator=kind,
                                               angular=angular, laws=laws))
    _padding_after_chains(view.xss_data, chains, dlw_at, dlwp, block_end)
    return block


def _yield(take, where, at, nes) -> ProductYield:
    mftype = int(round(take(f"SIGP of {where}", at, 1)[0]))
    if mftype == 13:
        ie, values = _indexed(take, f"SIGP of {where}", at + 1, nes)
        return ProductYield(mftype=13, first_index=ie, values=values)
    if mftype not in (6, 12, 16):
        raise ValueError(f"photonuclear ACE: {where}: MFTYPE = {mftype}")
    mt_mult, nr = (int(round(v)) for v in take(f"SIGP of {where}", at + 1, 2))
    regions = [int(round(v)) for v in take(f"SIGP of {where}", at + 3, 2 * nr)]
    ne = int(round(take(f"SIGP of {where}", at + 3 + 2 * nr, 1)[0]))
    data = take(f"SIGP of {where}", at + 4 + 2 * nr, 2 * ne)
    return ProductYield(mftype=mftype, mt_mult=mt_mult, breakpoints=regions[:nr], interpolation=regions[nr:],
                        energies=data[:ne] * EV_PER_MEV, values=data[ne:])


def _padding_after_chains(xss, chains, locators, dlwp, block_end) -> None:
    """Zero words between a chain's last table and the next chain (or the block's end).

    Kept on the last law as ``padding_after`` so the block is written back
    word for word, as for neutron DLW blocks.
    """
    starts = sorted({int(v) for v in locators if int(v) > 0} | {block_end - dlwp + 1})
    for chain in chains:
        last = chain[-1]
        end = getattr(last, "data_end", None)
        if end is None:
            continue
        later = [s for s in starts if s >= end]
        if not later or later[0] == end:
            continue
        gap = later[0] - end
        if not np.any(xss[dlwp + end - 1:dlwp + end - 1 + gap]):
            last.padding_after = gap
