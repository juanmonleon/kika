"""MF8/MT457: the radioactive decay data of a nuclide (ENDF-6 §8.4).

The section, record by record, kept so it is written back as read:

- HEAD ``[ZA, AWR, LIS, LISO, NST, NSP]``;
- LIST ``[T1/2, dT1/2, 0, 0, NC, 0]``: the average decay energies, NC/2 pairs
  (3: light particles, electromagnetic, heavy particles; or 17 by radiation);
- LIST ``[SPI, PAR, 0, 0, 6*NDK, NDK]``: the decay modes, ``(RTYP, RFS, Q, dQ,
  BR, dBR)`` each -- for a stable nuclide (NST=1) six zeros and NDK=0;
- NSP spectra, each a LIST ``[0, STYP, LCON, 0, 6, NER]`` of ``(FD, dFD, ERAV,
  dERAV, FC, dFC)``, then (LCON != 1) NER discrete lines, each a LIST ``[ER,
  dER, 0, 0, NT, 0]`` of NT = 4, 6, 8 or 12 values, then (LCON != 0) the
  continuum, a TAB1 ``[RTYP, 0, 0, LCOV, NR, NP]``, and (LCOV != 0) its
  covariance, a LIST ``[0, 0, LB, 0, NT, NP]``.

The values are kept as the file lays them out (``values`` of each record) and
named through properties, so no reading of ENDF-102 is needed to write them
back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..mt import MT
from ..mf12.base import PhotonTable, emit_cont, emit_list
from ...utils import PadStyle, format_endf_send_record

#: RTYP's integer part: the decay mode (ENDF-6 Table 8.2).
DECAY_MODES = {0: "gamma", 1: "beta-", 2: "beta+/EC", 3: "IT", 4: "alpha", 5: "n",
               6: "SF", 7: "p", 8: "e-", 9: "x-ray", 10: "unknown"}

#: STYP: the radiation a spectrum is of (ENDF-6 Table 8.3).
SPECTRUM_TYPES = {0: "gamma", 1: "beta-", 2: "beta+/EC", 4: "alpha", 5: "n", 6: "SF",
                  7: "p", 8: "e-", 9: "x-ray"}


@dataclass
class DecayMode:
    """One decay mode: ``RTYP`` (e.g. 1.5 = beta- then n), ``RFS`` the final
    isomeric state, ``Q`` with its uncertainty and the branching ratio."""

    rtyp: float
    rfs: float
    q: float
    dq: float
    br: float
    dbr: float


@dataclass
class DiscreteLine:
    """One discrete line: its energy and the NT values of its LIST."""

    er: float
    der: float
    values: List[float] = field(default_factory=list)
    l1: int = 0
    l2: int = 0
    n2: int = 0

    @property
    def rtyp(self) -> float:
        return self.values[0] if self.values else 0.0

    @property
    def intensity(self) -> Tuple[float, float]:
        """``(RI, dRI)``: the relative intensity and its uncertainty."""
        return (self.values[2], self.values[3]) if len(self.values) >= 4 else (0.0, 0.0)


@dataclass
class ContinuousSpectrum:
    """The continuum of a spectrum: RP(E) as a TAB1, and its covariance LIST."""

    table: PhotonTable
    covariance: Optional[Tuple[list, List[float]]] = None   # (header fields, values)

    @property
    def rtyp(self) -> float:
        return self.table.c1

    @property
    def lcov(self) -> int:
        return self.table.l2


@dataclass
class DecaySpectrum:
    """One radiation's spectrum: its normalisation LIST, lines, continuum."""

    styp: float
    lcon: int
    head: List[float] = field(default_factory=list)   # the six values of the LIST
    c1: float = 0.0
    l2: int = 0
    n2: int = 0
    lines: List[DiscreteLine] = field(default_factory=list)
    continuum: Optional[ContinuousSpectrum] = None

    @property
    def fd(self) -> Tuple[float, float]:
        """``(FD, dFD)``: the discrete normalisation factor."""
        return self.head[0], self.head[1]

    @property
    def erav(self) -> Tuple[float, float]:
        """``(ERAV, dERAV)``: the average decay energy of this radiation (eV)."""
        return self.head[2], self.head[3]

    @property
    def fc(self) -> Tuple[float, float]:
        """``(FC, dFC)``: the continuum normalisation factor."""
        return self.head[4], self.head[5]

    @property
    def radiation(self) -> str:
        return SPECTRUM_TYPES.get(int(self.styp), f"STYP={self.styp}")


@dataclass
class MF8MT457(MT):
    """MF8/MT457: one nuclide's decay data, written back as read."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    _lis: int = 0
    _liso: int = 0
    _nst: int = 0
    _mat: Optional[int] = None
    _mf: int = 8
    halflife: float = 0.0
    dhalflife: float = 0.0
    energies_l1: int = 0
    energies_l2: int = 0
    energies_n2: int = 0
    average_energies: List[float] = field(default_factory=list)   # E1, dE1, E2, dE2, ...
    spin: float = 0.0
    parity: float = 0.0
    modes_l1: int = 0
    modes_l2: int = 0
    modes_raw: List[float] = field(default_factory=list)          # 6*NDK (or six zeros)
    modes_n2: int = 0
    spectra: List[DecaySpectrum] = field(default_factory=list)
    pad: PadStyle = field(default_factory=PadStyle)

    @property
    def zaid(self) -> Optional[int]:
        return int(round(self._za)) if self._za is not None else None

    @property
    def is_stable(self) -> bool:
        return self._nst == 1

    @property
    def decay_modes(self) -> List[DecayMode]:
        if self.is_stable:
            return []
        v = self.modes_raw
        return [DecayMode(*v[i:i + 6]) for i in range(0, len(v), 6)]

    @property
    def average_energy_pairs(self) -> List[Tuple[float, float]]:
        v = self.average_energies
        return [(v[i], v[i + 1]) for i in range(0, len(v), 2)]

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf, mt, n = self._mf, self.number, 1
        lines: List[str] = []
        head, n = emit_cont(self._za, self._awr, self._lis, self._liso, self._nst,
                            len(self.spectra), mat, mf, mt, n)
        lines.append(head)
        block, n = emit_list(self.halflife, self.dhalflife, self.energies_l1,
                             self.energies_l2, self.energies_n2, self.average_energies,
                             mat, mf, mt, n, self.pad.values)
        lines += block
        block, n = emit_list(self.spin, self.parity, self.modes_l1, self.modes_l2,
                             self.modes_n2, self.modes_raw, mat, mf, mt, n, self.pad.values)
        lines += block
        for spectrum in self.spectra:
            block, n = emit_list(spectrum.c1, spectrum.styp, spectrum.lcon, spectrum.l2,
                                 spectrum.n2, spectrum.head, mat, mf, mt, n, self.pad.values)
            lines += block
            for line in spectrum.lines:
                block, n = emit_list(line.er, line.der, line.l1, line.l2, line.n2,
                                     line.values, mat, mf, mt, n, self.pad.values)
                lines += block
            if spectrum.continuum is not None:
                block, n = spectrum.continuum.table.emit(mat, mf, mt, n, self.pad)
                lines += block
                if spectrum.continuum.covariance is not None:
                    fields, values = spectrum.continuum.covariance
                    block, n = emit_list(fields[0], fields[1], fields[2], fields[3],
                                         fields[5], values, mat, mf, mt, n, self.pad.values)
                    lines += block
        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def report_gaps(self) -> List[str]:
        return []

    def __repr__(self) -> str:
        state = "stable" if self.is_stable else f"T1/2={self.halflife:g} s"
        return (f"MF8MT457(ZA={self.zaid}, LIS={self._lis}, {state}, "
                f"{len(self.decay_modes)} mode(s), {len(self.spectra)} spectrum(a))")
