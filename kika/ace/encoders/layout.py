"""Place the encoded blocks of a neutron table and fill NXS and JXS.

The physical order is NJOY's (ACER ``acelod``), which is not the JXS order:

    ESZ NU MTR LQR TYR LSIG SIG LAND AND LDLW DLW UNR DNU BDD DNEDL DNED
    GPD MTRP LSIGP SIGP LANDP ANDP LDLWP DLWP YP | PTYPE NTRO IXS
    and per particle HPD MTRH TYRH LSIGH SIGH LANDH ANDH LDLWH DLWH YH

JXS(21) points into SIG at the MT=18 record; JXS(22) is END, the last word of
the neutron part (before PTYPE); IXS holds absolute positions. A block that is
present but empty (LDLW and DLW of a table without secondary neutrons, ANDP
when every photon is isotropic...) gets the position of what follows it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from kika.ace.classes.xss import pack_int_words
from kika.ace.contract import AceContractError, structure_issues
from kika.ace.encoders import blocks as B
from kika.ace.encoders._words import Words


@dataclass
class EncodedTable:
    nxs: list          # 17 entries, [0] unused
    jxs: list          # 33 entries, [0] unused
    xss: np.ndarray    # float64, [0] = 0.0 placeholder
    is_int: np.ndarray  # bool over XSS(1..n)


class _Layout:
    def __init__(self):
        self.out = Words()
        self.jxs = [0] * 33

    @property
    def pos(self) -> int:
        return self.out.next

    def put(self, j: int, block) -> int:
        """Append ``block`` (or nothing if None, leaving JXS(j) = 0); returns its position."""
        if block is None:
            return 0
        at = self.pos
        if j:
            self.jxs[j] = at
        self.out.extend(block)
        return at


def encode_xss(ace) -> EncodedTable:
    """The XSS, NXS and JXS of a neutron ``Ace`` (``.c``), from its block objects.

    Raises :class:`~kika.ace.contract.AceContractError` if the blocks do not
    agree with each other (:func:`~kika.ace.contract.structure_issues`).
    """
    issues = structure_issues(ace)
    if issues:
        raise AceContractError(issues)
    rmt = ace.reaction_mt_data
    L = _Layout()
    n_tr = len(rmt.incident_neutron)
    n_r = len(rmt.secondary_neutron_mt)
    n_gp = len(rmt.photon_production)

    L.put(1, B.encode_esz(ace))
    L.put(2, B.encode_nu(ace))

    ldlw, dlw, new_ty = B.encode_dlw(ace)
    tyr = Words()
    for mt, ty in zip(rmt.incident_neutron, ace.particle_release.incident_neutron):
        ty = int(ty)
        if abs(ty) > 100:
            ty = int(np.sign(ty)) * new_ty[int(mt)]
        tyr.int(ty)

    L.put(3, B._ints_block(rmt.incident_neutron))
    L.put(4, B._reals_block(ace.q_values.q_values))
    L.put(5, tyr)
    lsig, sig, fis_at = B.encode_sig(ace)
    L.put(6, lsig)
    sig_at = L.put(7, sig)
    if fis_at is not None:
        L.jxs[21] = sig_at + fis_at - 1
    land, and_ = B.encode_and(ace)
    L.put(8, land)
    L.put(9, and_)
    L.put(10, ldlw)
    L.put(11, dlw)
    L.put(23, B.encode_unr(ace))
    L.put(24, B.encode_dnu(ace))
    L.put(25, B.encode_bdd(ace))
    if ace.energy_distributions.delayed_neutron:
        dnedl, dned = B.encode_dned(ace)
        L.put(26, dnedl)
        L.put(27, dned)

    if ace.photon_production_data is not None:
        L.put(12, B.encode_gpd(ace))
        L.put(13, B._ints_block(rmt.photon_production))
        lsigp, sigp = B.encode_sigp(ace)
        L.put(14, lsigp)
        L.put(15, sigp)
        landp, andp = B.encode_andp(ace)
        L.put(16, landp)
        L.put(17, andp)
        ldlwp, dlwp = B.encode_dlwp(ace)
        L.put(18, ldlwp)
        L.put(19, dlwp)
        L.put(20, B.encode_yp(ace))
    L.jxs[22] = L.pos - 1

    ptypes = list(getattr(ace.secondary_particle_types, "particle_ids", []) or [])
    if ptypes:
        counts = [len(rmt.particle_production[i]) if i < len(rmt.particle_production) else 0
                  for i in range(len(ptypes))]
        L.put(30, B._ints_block(ptypes))
        L.put(31, B._ints_block(counts))
        ixs_at = L.put(32, B._ints_block([0] * (10 * len(ptypes))))
        for i, n in enumerate(counts):
            if not n:
                continue
            locs = [L.put(0, B.encode_hpd(ace, i)),
                    L.put(0, B._ints_block(rmt.particle_production[i])),
                    L.put(0, B._ints_block(ace.particle_release.particle_production[i]))]
            lsigh, sigh = B.encode_sigh(ace, i)
            locs += [L.put(0, lsigh), L.put(0, sigh)]
            landh, andh = B.encode_andh(ace, i)
            locs += [L.put(0, landh), L.put(0, andh)]
            ldlwh, dlwh = B.encode_dlwh(ace, i)
            locs += [L.put(0, ldlwh), L.put(0, dlwh)]
            locs.append(L.put(0, B.encode_yh(ace, i)))
            for k, at in enumerate(locs):
                L.out.set(ixs_at - 1 + 10 * i + k, at)

    values, is_int = L.out.arrays()
    nxs = list(ace.header.nxs_array[:17]) + [0] * max(0, 17 - len(ace.header.nxs_array))
    nxs[1] = values.size
    nxs[3] = len(ace.esz_block.energies)
    nxs[4] = n_tr
    nxs[5] = n_r
    nxs[6] = n_gp
    nxs[7] = len(ptypes)
    nxs[8] = len(ace.energy_distributions.delayed_neutron)
    xss = np.empty(values.size + 1)
    xss[0] = 0.0
    xss[1:] = values
    return EncodedTable(nxs=nxs, jxs=L.jxs, xss=xss, is_int=is_int)


def rebuild(ace, strict: bool = True):
    """A new ``Ace`` whose XSS is encoded from ``ace``'s objects and whose blocks are read back from it.

    The header is copied with the new NXS and JXS; the integer spelling of
    every word comes from the encoders, so ``write_ace`` writes NJOY's text.
    """
    from kika.ace.classes.ace import Ace
    from kika.ace.classes.read_report import ReadReport
    from kika.ace.parsers.parse_ace import parse_blocks

    enc = encode_xss(ace)
    new = Ace()
    new.header = copy.deepcopy(ace.header)
    new.header.nxs_array = enc.nxs
    new.header.jxs_array = enc.jxs
    new.xss_data = enc.xss
    new.xss_int_words = pack_int_words(enc.is_int)
    new.read_report = ReadReport(strict=strict)
    return parse_blocks(new)
