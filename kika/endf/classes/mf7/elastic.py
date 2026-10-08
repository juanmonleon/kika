"""MF7/MT2: thermal elastic scattering (ENDF-6 §7.2).

``LTHR`` selects which of two representations the section carries, and it is a
genuine three-way choice rather than a flag::

    LTHR = 1   coherent only     Bragg edges S(E), tabulated per temperature
    LTHR = 2   incoherent only   the Debye-Waller integral W(T)
    LTHR = 3   both, in that order

All three occur in ENDF/B-VIII.1: 86 of the 114 TSL evaluations are LTHR=1, 11
are LTHR=3 (``tsl-NinUN*``, ``tsl-ZrinZrH2``, ``tsl-YinYH2``, ``tsl-CinZrC``,
the mixed lithium hydrides), and the rest are LTHR=2. A further 17 have no MT2
at all — ``tsl-HinH2O`` among them — so *absent* is a fourth state and code
that reaches for ``sections[2]`` must expect a ``KeyError``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .base import MF7MT, TemperatureTable
from ..mf5.analytic import tab1_at
from ...utils import PadStyle, format_endf_send_record, format_tab1

#: Energies a caller may pass: one value or anything numpy reads as a vector.
Energies = Union[float, Sequence[float], np.ndarray]

#: ``LTHR`` values that carry a coherent block, and those that carry an
#: incoherent one. Written as sets because LTHR=3 is in both, and every
#: ``lthr == 1`` test in a reader is a latent bug on the 11 LTHR=3 files.
COHERENT_LTHR = frozenset({1, 3})
INCOHERENT_LTHR = frozenset({2, 3})


@dataclass
class CoherentElastic:
    """Bragg-edge structure: S(E) as a stack of temperatures.

    The energy grid is the set of Bragg edges, and the interpolation is **INT=1
    (histogram)** — S(E) is a staircase that steps at each edge, so interpolating
    it linearly invents scattering between edges where there is none. Every
    coherent evaluation measured writes 1 here; the parser records what it read
    rather than assuming, and :meth:`is_histogram` is how a caller checks.
    """

    table: TemperatureTable = field(default_factory=TemperatureTable)

    @property
    def energies(self) -> List[float]:
        """The Bragg edges, in eV."""
        return self.table.x

    @property
    def temperatures(self) -> List[float]:
        return self.table.temperatures

    @property
    def is_histogram(self) -> bool:
        return all(int(code) == 1 for _, code in self.table.interp)

    def s_at(self, index: int) -> List[float]:
        """S(E) at temperature *index*, one value per Bragg edge."""
        return self.table.row(index)

    def cross_section(self, energies: Energies, temperature: float) -> np.ndarray:
        """σ_coh(E, T) = S(E, T) / E, in barns (ENDF-102 eq. 7.3).

        S is cumulative over the edges, so between edge *i* and edge *i+1* it is
        the value written at edge *i*, and below the first edge it is zero: no
        lattice plane can diffract a neutron whose wavelength is longer than
        twice its spacing. The result is a sawtooth, which drops as 1/E between
        edges and jumps at each one.

        *temperature* must be one the table lists. ``li`` tells how ENDF wants
        two temperatures combined, and doing that silently here would put an
        approximation in front of a caller who asked for data. Refused, too,
        when the table is not histogram-interpolated: the formula reads S as a
        staircase, and on any other INT it would be the wrong function.
        """
        if not self.is_histogram:
            raise ValueError(
                f"coherent elastic S(E) is interpolated with {self.table.interp}; "
                "sigma = S/E needs the histogram (INT=1) every evaluation writes"
            )
        s = np.asarray(self.table.at_temperature(temperature), dtype=float)
        edges = np.asarray(self.energies, dtype=float)
        e = np.asarray(energies, dtype=float)
        # The last edge at or below each E. -1 means below the first edge.
        index = np.searchsorted(edges, e, side="right") - 1
        held = np.where(index >= 0, s[np.clip(index, 0, None)], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(e > 0, held / np.where(e > 0, e, 1.0), 0.0)

    def emit(self, mat: int, mf: int, mt: int, line_num: int,
             pad: PadStyle = PadStyle()) -> Tuple[List[str], int]:
        return self.table.emit(0.0, mat, mf, mt, line_num, pad=pad)

    def describe(self) -> str:
        return (f"coherent: {len(self.energies)} Bragg edges, "
                f"{self.table.n_temperatures} temperature(s)")


@dataclass
class IncoherentElastic:
    """The Debye-Waller integral W(T), with the bound cross section SB.

    One TAB1 and no temperature stack — the temperature dependence *is* the
    x grid here, which is why this is not a :class:`TemperatureTable`.
    """

    sb: float = 0.0
    interp: List[Tuple[int, int]] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    w: List[float] = field(default_factory=list)

    def emit(self, mat: int, mf: int, mt: int, line_num: int,
             pad: PadStyle = PadStyle()) -> Tuple[List[str], int]:
        return format_tab1(
            self.sb, 0.0, 0, 0, self.interp, self.temperatures, self.w,
            mat, mf, mt, line_num, pad=pad.pairs,
        )

    def debye_waller(self, temperature: float) -> float:
        """W′(T) in 1/eV, under the TAB1's own interpolation law.

        Interpolated, unlike the coherent stack: here temperature *is* the x
        axis of a TAB1, so a value between two nodes is what the evaluator's INT
        code defines and not something this method chooses. Outside the table it
        is refused, because holding the end value would invent a lattice.
        """
        ts = np.asarray(self.temperatures, dtype=float)
        if ts.size == 0:
            raise ValueError("incoherent elastic block has no W'(T) table")
        if not ts.min() <= temperature <= ts.max():
            raise KeyError(
                f"{temperature} K is outside the W'(T) table "
                f"[{ts.min()}, {ts.max()}] K"
            )
        return tab1_at(self.temperatures, self.w, self.interp, temperature)

    def cross_section(self, energies: Energies, temperature: float) -> np.ndarray:
        """σ_inc(E, T) in barns (ENDF-102 eq. 7.5).

        σ = SB/2 · (1 − e^(−4EW′)) / (2EW′), which tends to SB as E → 0 and
        falls as SB / (4EW′) once 4EW′ ≫ 1. Written with ``expm1`` so the low-E
        limit is not a cancellation of two numbers near one.
        """
        w = self.debye_waller(temperature)
        e = np.asarray(energies, dtype=float)
        x = 2.0 * e * w
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(x > 0, -np.expm1(-2.0 * x) / np.where(x > 0, x, 1.0), 2.0)
        return 0.5 * self.sb * ratio

    def describe(self) -> str:
        return f"incoherent: SB={self.sb}, {len(self.temperatures)} temperature(s)"


@dataclass
class MF7MT2(MF7MT):
    """One MF7/MT2 section: HEAD, then whichever blocks ``LTHR`` announces."""

    _lthr: Optional[int] = None
    coherent: Optional[CoherentElastic] = None
    incoherent: Optional[IncoherentElastic] = None

    @property
    def lthr(self) -> Optional[int]:
        return self._lthr

    @property
    def has_coherent(self) -> bool:
        return self.coherent is not None

    @property
    def has_incoherent(self) -> bool:
        return self.incoherent is not None

    @property
    def temperatures(self) -> List[float]:
        """The temperature grid, from whichever block defines one.

        Coherent first: on an LTHR=3 file the two blocks need not agree, and the
        coherent stack is the one that carries S(E) per temperature.
        """
        if self.coherent is not None:
            return self.coherent.temperatures
        if self.incoherent is not None:
            return self.incoherent.temperatures
        return []

    def cross_sections(self, energies: Energies, temperature: float
                       ) -> Dict[str, np.ndarray]:
        """Each elastic σ the section states at *temperature*, by name.

        Keys are ``'coherent'`` and ``'incoherent'``, present when the block is,
        and ``'total'`` when both are. A block whose temperatures do not include
        *temperature* raises: on an LTHR=3 file the two grids need not agree,
        and a total missing one of its terms would look like a total.
        """
        result: Dict[str, np.ndarray] = {}
        if self.coherent is not None:
            result["coherent"] = self.coherent.cross_section(energies, temperature)
        if self.incoherent is not None:
            result["incoherent"] = self.incoherent.cross_section(energies, temperature)
        if len(result) == 2:
            result["total"] = result["coherent"] + result["incoherent"]
        return result

    def report_gaps(self) -> List[str]:
        """Empty: MT2 is decoded in full, for every LTHR this format defines.

        Present so MF7 answers the same question MF5 does. See
        :meth:`kika.endf.classes.mf5.base.MF5MT.report_gaps` for why a section
        that skips nothing still has to say so.
        """
        return []

    def head_fields(self) -> List[int]:
        return [self._lthr or 0, 0, 0, 0]

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf = self._mf
        mt = self.number

        head, line_num = self.emit_head(1)
        lines = [head]
        for block in (self.coherent, self.incoherent):
            if block is not None:
                block_lines, line_num = block.emit(mat, mf, mt, line_num,
                                                  pad=self.pad)
                lines.extend(block_lines)
        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def __repr__(self) -> str:
        blocks = ", ".join(
            block.describe()
            for block in (self.coherent, self.incoherent) if block is not None
        )
        return f"MF7MT2(LTHR={self._lthr}, {blocks})"
