"""ACE thermal tables (``.NNt``): what MCNP reads for S(α,β), one file per temperature.

The layout is ACER's (NJOY 2016, ``acecm``/``thermal`` blocks), as LANL's
ENDF81SaB and the NEA's JEFF-4.0 thermal ACE write it:

=========  =========================================================
NXS(2)     IDPNI, the inelastic mode (3 in every modern file)
NXS(3)     NIL: cosines per outgoing energy (continuous: NIL − 1)
NXS(4)     NIEB: outgoing energies per incident energy (discrete)
NXS(5)     IDPNC: 0 no elastic, 3 incoherent, 4 coherent, 5 mixed
NXS(6)     NCL: elastic cosines − 1 (incoherent; −1 for coherent)
NXS(7)     IFENG: 0 discrete, 1 skewed, 2 continuous outgoing energies
NXS(8)     NCLI: incoherent cosines − 1, in the mixed case
JXS(1-3)   ITIE (σ_inel), ITIX, ITXE (outgoing table)
JXS(4-6)   ITCE (elastic energies and P or σ), ITCX, ITCA (cosines)
JXS(7-9)   the incoherent half of a mixed elastic block
=========  =========================================================

For coherent elastic, P is the cumulative S(E_i) in **MeV·b** at the Bragg
edges E_i in MeV; σ = P/E. The header's temperature is kT in MeV.

Indices follow the format: ``xss`` is read 1-based (``xss[0]`` is padding)
and a continuous table's locators point one before their record.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .processed import (BraggTable, IncoherentElasticTable, OutgoingTable, ProcessedTemperature,
                        ProcessedTSL, merge_temperatures)

__all__ = ["AceThermalHeader", "read_ace_thermal_header", "read_ace_thermal", "is_ace_thermal",
           "looks_like_ace_thermal"]

MEV = 1.0e6
#: Boltzmann constant in MeV/K: ACE's header carries kT, not T.
K_MEV = 8.617333262e-11

#: ``h-h2o.80t``, ``grph.72t``, ``1001.20t``: an ACE thermal file name.
ACE_THERMAL_NAME = re.compile(r"^(?P<stem>.+)\.(?P<ext>\d{2,3}t)$", re.IGNORECASE)


def looks_like_ace_thermal(lines) -> bool:
    """The first header line of an ACE thermal table: ``h-h2o.80t AWR kT DATE``."""
    if not lines or not lines[0].strip():
        return False
    fields = lines[0].split()
    if len(fields) < 3 or not ACE_THERMAL_NAME.match(fields[0]):
        return False
    try:
        float(fields[1])
        float(fields[2])
    except ValueError:
        return False
    return True


class AceThermalHeader:
    """The two header lines and the IZ list of a thermal table."""

    def __init__(self, zaid: str, awr: float, kt: float, date: str, title: str, mat: str,
                 izaw: List[int], nxs: List[int], jxs: List[int], header_lines: int):
        self.zaid = zaid
        self.awr = awr
        self.kt = kt
        self.date = date
        self.title = title
        self.mat = mat
        self.izaw = izaw
        self.nxs = nxs
        self.jxs = jxs
        self.header_lines = header_lines

    @property
    def temperature(self) -> float:
        """The table's temperature in K (from kT)."""
        return round(self.kt / K_MEV, 1)

    @property
    def name(self) -> str:
        """``H in H2O`` from ``H in H2O - ENDF/B-VIII.1 - 293.6 K - …``; the ZAID stem otherwise."""
        head = self.title.split(" - ")[0].strip()
        return head or self.zaid.split(".")[0]

    @property
    def library(self) -> Optional[str]:
        parts = [p.strip() for p in self.title.split(" - ")]
        return parts[1] if len(parts) > 2 else None

    @property
    def elastic_mode(self) -> Optional[str]:
        if not self.jxs[4]:
            return None
        return {4: "coherent", 5: "mixed"}.get(self.nxs[5], "incoherent")

    @property
    def outgoing_kind(self) -> str:
        return {0: "discrete", 1: "skewed", 2: "continuous"}.get(self.nxs[7], "discrete")


def _ints(lines: Sequence[str]) -> List[int]:
    return [0] + [int(v) for v in " ".join(lines).split()]


def read_ace_thermal_header(path: Union[str, Path]) -> AceThermalHeader:
    """The header of an ACE thermal table, from its first lines only."""
    with open(path, "r", errors="replace") as handle:
        first = handle.readline()
        lines = [first]
        tokens = first.split()
        if tokens and re.match(r"^\d+\.\d+\.\d+$", tokens[0]):  # ACE 2.0.x header
            second = handle.readline()
            awr, kt, date, n_comments = second.split()[:4]
            comments = [handle.readline() for _ in range(int(n_comments))]
            zaid = tokens[1]
            title = (comments[0] if comments else "").strip()
            mat = ""
            head = 2 + int(n_comments)
            lines = [handle.readline() for _ in range(2)]  # the legacy two lines follow
            head += 2
            awr, kt = float(awr), float(kt)
        else:
            second = handle.readline()
            zaid = tokens[0]
            awr, kt = float(tokens[1]), float(tokens[2])
            date = tokens[3] if len(tokens) > 3 else ""
            title, mat = second[:70].strip(), second[70:].strip()
            head = 2
        rest = [handle.readline() for _ in range(10)]
    izaw = [int(float(v)) for v in " ".join(rest[0:4]).split()[0::2]]
    return AceThermalHeader(zaid=zaid, awr=awr, kt=kt, date=date, title=title, mat=mat,
                            izaw=[z for z in izaw if z], nxs=_ints(rest[4:6]),
                            jxs=_ints(rest[6:10]), header_lines=head + 10)


def is_ace_thermal(path: Union[str, Path]) -> bool:
    """True when *path* reads as an ACE thermal table (by its ZAID, then its NXS)."""
    try:
        header = read_ace_thermal_header(path)
    except (OSError, ValueError, IndexError):
        return False
    return header.zaid.lower().endswith("t") and len(header.nxs) > 8 and header.nxs[1] > 0


def _xss(path: Union[str, Path], header: AceThermalHeader) -> np.ndarray:
    with open(path, "r", errors="replace") as handle:
        for _ in range(header.header_lines):
            handle.readline()
        values = np.array(handle.read().split(), dtype=float)
    return np.concatenate([[0.0], values])


def _outgoing(xss: np.ndarray, header: AceThermalHeader, incident: np.ndarray) -> OutgoingTable:
    nxs, jxs = header.nxs, header.jxs
    n = incident.size
    kind = header.outgoing_kind
    if kind == "continuous":
        n_mu = nxs[3] - 1
        idx = jxs[3]
        locators = xss[idx:idx + n].astype(int)
        counts = xss[idx + n:idx + 2 * n].astype(int)
        energies, pdf, cdf, cosines = [], [], [], []
        stride = n_mu + 3
        for loc, m in zip(locators, counts):
            block = xss[loc + 1:loc + 1 + m * stride].reshape(m, stride)
            energies.append(block[:, 0] * MEV)
            pdf.append(block[:, 1] / MEV)
            cdf.append(block[:, 2])
            cosines.append(block[:, 3:])
        return OutgoingTable(kind=kind, incident=incident, energies=energies, cosines=cosines,
                             pdf=pdf, cdf=cdf)
    n_mu = nxs[3]
    m = nxs[4]
    stride = n_mu + 2
    block = xss[jxs[3]:jxs[3] + n * m * stride].reshape(n, m, stride)
    energies = [row[:, 0] * MEV for row in block]
    cosines = [row[:, 1:] for row in block]
    weights = None
    if kind == "skewed":
        # NJOY's skewed grid: the first and last two lines carry less weight.
        w = np.ones(m)
        if m >= 4:
            w[[0, 1, -2, -1]] = [0.1, 0.4, 0.4, 0.1]
        weights = [w] * n
    return OutgoingTable(kind=kind, incident=incident, energies=energies, cosines=cosines,
                         weights=weights)


def _read_block(path: Union[str, Path], header: AceThermalHeader) -> ProcessedTemperature:
    xss = _xss(path, header)
    nxs, jxs = header.nxs, header.jxs
    block = ProcessedTemperature(temperature=header.temperature)

    i = jxs[1]
    n = int(xss[i])
    block.inelastic_energies = xss[i + 1:i + 1 + n] * MEV
    block.inelastic_xs = xss[i + 1 + n:i + 1 + 2 * n].copy()
    block.outgoing_kind = header.outgoing_kind
    table = _outgoing(xss, header, block.inelastic_energies)
    block.outgoing_loader = lambda: table

    mode = header.elastic_mode
    if mode is not None:
        i = jxs[4]
        n = int(xss[i])
        energies = xss[i + 1:i + 1 + n] * MEV
        values = xss[i + 1 + n:i + 1 + 2 * n]
        if mode in ("coherent", "mixed"):
            block.coherent = BraggTable(edges=energies, s=values * MEV)
        else:
            n_mu = nxs[6] + 1
            cos = xss[jxs[6]:jxs[6] + n * n_mu].reshape(n, n_mu) if jxs[6] else None
            block.incoherent = IncoherentElasticTable(energies=energies, xs=values.copy(),
                                                      cosine_energies=energies, cosines=cos)
        if mode == "mixed" and jxs[7]:
            i = jxs[7]
            n = int(xss[i])
            energies = xss[i + 1:i + 1 + n] * MEV
            values = xss[i + 1 + n:i + 1 + 2 * n]
            n_mu = nxs[8] + 1
            cos = xss[jxs[9]:jxs[9] + n * n_mu].reshape(n, n_mu) if jxs[9] else None
            block.incoherent = IncoherentElasticTable(energies=energies, xs=values.copy(),
                                                      cosine_energies=energies, cosines=cos)
    return block


def read_ace_thermal(paths: Union[str, Path, Sequence[Union[str, Path]]]) -> ProcessedTSL:
    """One material from one or more ACE thermal tables (one per temperature).

    The tables must be of the same material: the first one's name, AWR and
    nuclides are the material's, and a table that disagrees is refused.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    blocks = []
    first: Optional[AceThermalHeader] = None
    for path in paths:
        header = read_ace_thermal_header(path)
        if first is None:
            first = header
        elif header.name != first.name or header.izaw != first.izaw:
            raise ValueError(f"{Path(path).name} is {header.name!r}, not {first.name!r}: "
                             "one material per call")
        blocks.append(_read_block(path, header))
    assert first is not None
    blocks = merge_temperatures(blocks)
    emax = max((float(b.inelastic_energies[-1]) for b in blocks
                if b.inelastic_energies is not None and b.inelastic_energies.size), default=None)
    return ProcessedTSL(
        format="ace", name=first.name, awr=first.awr, temperatures=blocks,
        nuclides=[_zaid_nuclide(z) for z in first.izaw], title=first.title,
        library=first.library, emax=emax, paths=[str(p) for p in paths],
        details={"zaid": first.zaid.split(".")[0], "mat": first.mat, "date": first.date,
                 "elastic_mode": first.elastic_mode, "outgoing": first.outgoing_kind,
                 "extensions": [Path(p).suffix.lstrip(".") for p in paths]})


def _zaid_nuclide(za: int) -> str:
    from kika._constants import ATOMIC_NUMBER_TO_SYMBOL

    z, a = divmod(int(za), 1000)
    symbol = ATOMIC_NUMBER_TO_SYMBOL.get(z, f"Z{z}")
    return f"{symbol}{a}" if a else f"{symbol}0"
