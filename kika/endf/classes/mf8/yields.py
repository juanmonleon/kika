"""MF8/MT454 and MT459: independent and cumulative fission product yields (ENDF-6 §8.3).

HEAD ``[ZA, AWR, LE+1, 0, 0, 0]``, then LE+1 LISTs ``[E_i, 0, I_i, 0, 4*NFP,
NFP]`` of ``(ZAFP, FPS, Y, DY)`` -- the yields at one incident energy, with
``I_i`` the interpolation from the previous energy (0 on the first).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..mt import MT
from ..mf12.base import emit_cont, emit_list
from ...utils import PadStyle, format_endf_send_record


@dataclass
class YieldEntry:
    """One product: ``ZAFP`` (its ZA), ``FPS`` (its isomeric state), the yield and its σ."""

    zafp: float
    fps: float
    y: float
    dy: float


@dataclass
class FissionYields:
    """The yields at one incident energy."""

    energy: float
    interpolation: int
    values: List[float] = field(default_factory=list)   # 4*NFP, as written
    c2: float = 0.0
    l2: int = 0

    @property
    def entries(self) -> List[YieldEntry]:
        v = self.values
        return [YieldEntry(*v[i:i + 4]) for i in range(0, len(v), 4)]


@dataclass
class MF8FissionYields(MT):
    """MF8/MT454 (independent) or MT459 (cumulative), written back as read."""

    _za: Optional[float] = None
    _awr: Optional[float] = None
    _l2: int = 0
    _n1: int = 0
    _n2: int = 0
    _mat: Optional[int] = None
    _mf: int = 8
    energies: List[FissionYields] = field(default_factory=list)
    pad: PadStyle = field(default_factory=PadStyle)

    @property
    def zaid(self) -> Optional[int]:
        return int(round(self._za)) if self._za is not None else None

    @property
    def kind(self) -> str:
        return {454: "independent", 459: "cumulative"}.get(self.number, f"MT{self.number}")

    def __str__(self) -> str:
        mat = self._mat if self._mat is not None else 0
        mf, mt, n = self._mf, self.number, 1
        lines: List[str] = []
        head, n = emit_cont(self._za, self._awr, len(self.energies), self._l2, self._n1,
                            self._n2, mat, mf, mt, n)
        lines.append(head)
        for block in self.energies:
            body, n = emit_list(block.energy, block.c2, block.interpolation, block.l2,
                                len(block.values) // 4, block.values, mat, mf, mt, n,
                                self.pad.values)
            lines += body
        lines.append(format_endf_send_record(mat, mf))
        return "\n".join(lines)

    def report_gaps(self) -> List[str]:
        return []

    def __repr__(self) -> str:
        return (f"MF8FissionYields(MT{self.number} {self.kind}, ZA={self.zaid}, "
                f"{len(self.energies)} energies)")
