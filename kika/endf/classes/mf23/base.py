"""MF23 and MF27: photo-atomic cross sections and form factors (ENDF-6 §23, §27).

These two files make up the photo-atomic sublibrary (NSUB=3), the photon
*interaction* data of an element: how a photon scatters off, is absorbed by,
or makes pairs in the electron cloud of an atom. They are evaluated per
element, so the ZA of every HEAD ends in ``000`` (26000 for iron) and the
tapes carry no neutron files at all. ENDF/B-VIII.0 takes them from EPICS2017.

Both files share one layout, a HEAD followed by a single TAB1, and differ only
in what the TAB1 holds:

* **MF23** gives σ(E) in barns against the incident photon energy in eV. The
  TAB1's C1 and C2 are zero for the reaction totals; for a subshell
  photoelectric cross section (MT 534-572) they carry ``EPE``, the binding
  energy of the subshell, and ``EFL``, the fluorescence yield (eV per
  photoionisation) — values a transport code needs and which ENDF-6 keeps in
  this record because there is nowhere else to put them.
* **MF27** gives an atomic form factor or scattering function against
  ``x = sin(θ/2)/λ`` in inverse Ångström for MT 502 (coherent form factor
  F(x)) and MT 504 (incoherent scattering function S(x)), and an anomalous
  scattering factor against the incident photon energy in eV for MT 505
  (imaginary part) and MT 506 (real part). The TAB1's C2 is Z.

**Every field a record header carries is stored**, including the ones ENDF-6
calls unused, so a section re-emits as the bytes it came in (numbers written
non-canonically excepted, as in :mod:`kika.endf.classes.mf12.base`). NR and NP
are recomputed from the data on the way out, so a section edited in memory
cannot write counts that disagree with its body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..mf12.base import PhotonTable, emit_cont
from ..mt import MT
from ...utils import PadStyle, format_endf_send_record

#: MT numbers of the photo-atomic files and what they mean (ENDF-102,
#: Appendix B). The pair-production triplet is easy to get backwards: 515 is
#: production in the field of the atomic *electrons* (triplet, threshold
#: 4·m_e·c² = 2.044 MeV), 517 in the field of the *nucleus* (threshold
#: 2·m_e·c² = 1.022 MeV), and 516 is their sum. The Fe tape of ENDF/B-VIII.0
#: says so by its thresholds.
PHOTOATOMIC_MT_NAMES = {
    501: "total",
    502: "coherent scattering",
    504: "incoherent scattering",
    505: "anomalous scattering factor, imaginary part",
    506: "anomalous scattering factor, real part",
    515: "pair production, electron field",
    516: "pair production, total",
    517: "pair production, nuclear field",
    522: "photoelectric absorption, total",
    523: "photo-excitation",
}

#: ENDF subshell designators, in the order MT 534 onwards uses them. The
#: designator *number* of MF28 (SUBI) is ``MT - 533``, which is why the list
#: starts at 1 for K.
SUBSHELL_NAMES = [
    "K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5",
    "N1", "N2", "N3", "N4", "N5", "N6", "N7",
    "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9",
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11",
    "Q1", "Q2", "Q3",
]
FIRST_SUBSHELL_MT = 534
LAST_SUBSHELL_MT = FIRST_SUBSHELL_MT + len(SUBSHELL_NAMES) - 1   # 572


def subshell_name(designator: int) -> Optional[str]:
    """The name of ENDF subshell *designator* (1 = K), or None outside 1-39."""
    if 1 <= designator <= len(SUBSHELL_NAMES):
        return SUBSHELL_NAMES[designator - 1]
    return None


def photoatomic_mt_name(mt: int) -> str:
    """A readable name for a photo-atomic MT, subshells included."""
    if FIRST_SUBSHELL_MT <= mt <= LAST_SUBSHELL_MT:
        return f"photoelectric absorption, {SUBSHELL_NAMES[mt - FIRST_SUBSHELL_MT]} subshell"
    return PHOTOATOMIC_MT_NAMES.get(mt, f"MT{mt}")


@dataclass
class PhotoatomicTabMT(MT):
    """A photo-atomic section: one HEAD and one TAB1.

    The shared body of :class:`MF23MT` and :class:`MF27MT`. The HEAD's four
    integer fields are unused by ENDF-6 in both files and are kept as read.
    """

    _za: Optional[float] = None
    _awr: Optional[float] = None
    _l1: int = 0
    _l2: int = 0
    _n1: int = 0
    _n2: int = 0
    _mat: Optional[int] = None
    _mf: int = 23
    table: PhotonTable = field(default_factory=PhotonTable)
    pad: PadStyle = field(default_factory=PadStyle)

    @property
    def zaid(self) -> Optional[int]:
        return int(round(self._za)) if self._za is not None else None

    @property
    def z(self) -> Optional[int]:
        """The atomic number, from ZA (photo-atomic ZA is ``1000·Z``)."""
        za = self.zaid
        return za // 1000 if za is not None else None

    @property
    def atomic_weight_ratio(self) -> Optional[float]:
        return self._awr

    @property
    def name(self) -> str:
        return photoatomic_mt_name(self.number)

    @property
    def interpolation(self):
        return list(self.table.interp)

    @property
    def x(self) -> np.ndarray:
        return np.asarray(self.table.x, dtype=float)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self.table.y, dtype=float)

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf, mt = self._mf, self.number
        head, line_num = emit_cont(self._za, self._awr, self._l1, self._l2,
                                   self._n1, self._n2, mat, mf, mt, 1)
        body, line_num = self.table.emit(mat, mf, mt, line_num, self.pad)
        return "\n".join([head] + body + [format_endf_send_record(mat, mf)])

    def report_gaps(self) -> List[str]:
        """Nothing: every record of the section is read and kept."""
        return []

    def __repr__(self) -> str:
        return f"MF{self._mf}MT({self.number}, {self.name!r}, NP={len(self.table.x)})"


@dataclass(repr=False)
class MF23MT(PhotoatomicTabMT):
    """One MF23 section: a photon interaction cross section σ(E).

    Energies in eV, cross sections in barns, as the file writes them.
    """

    _mf: int = 23

    @property
    def energies(self) -> np.ndarray:
        return self.x

    @property
    def cross_sections(self) -> np.ndarray:
        return self.y

    @property
    def is_subshell(self) -> bool:
        return FIRST_SUBSHELL_MT <= self.number <= LAST_SUBSHELL_MT

    @property
    def subshell(self) -> Optional[str]:
        """``K``, ``L1``... for MT 534-572; None for the reaction totals."""
        if not self.is_subshell:
            return None
        return SUBSHELL_NAMES[self.number - FIRST_SUBSHELL_MT]

    @property
    def epe(self) -> float:
        """``EPE``: the subshell binding energy (eV); 0 for the totals."""
        return self.table.c1

    @property
    def efl(self) -> float:
        """``EFL``: the subshell fluorescence yield (eV/photoionisation); 0 for the totals."""
        return self.table.c2


@dataclass(repr=False)
class MF27MT(PhotoatomicTabMT):
    """One MF27 section: a form factor, scattering function or anomalous factor.

    ``x`` is ``sin(θ/2)/λ`` in 1/Å for MT 502 and 504, and the incident photon
    energy in eV for MT 505 and 506. The TAB1's C2 carries Z.
    """

    _mf: int = 27

    @property
    def table_z(self) -> float:
        """Z as the TAB1 header writes it (its C2)."""
        return self.table.c2

    @property
    def x_unit(self) -> str:
        return "eV" if self.number in (505, 506) else "1/angstrom"

    @property
    def values(self) -> np.ndarray:
        return self.y
