"""What every data check reads: the tape, its header, its tables, the raw lines.

:class:`Tape` wraps the parsed :class:`~kika.endf.classes.endf.ENDF` the checks
look at. An MF is parsed the first time a check asks for it (``open_endf`` is
lazy), and an MF the parser could not read at all, or a section it dropped,
is a ``parse_error`` finding, reported once, not an exception in a check.

:class:`Out` collects the findings. A check that fires on many points of one
table reports **one** finding for the table -- how many points, and the worst
-- not one per point: a page of four thousand identical rows says less than
one row with its count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..findings import DEFECT, NOTE, WARN, Finding, Location

__all__ = ["Tape", "Out", "Table", "NEUTRON_MASS_EV", "AMU_EV", "fmt_e", "fmt_b"]

#: CODATA 2018, as AME2020 states them.
AMU_EV = 931.49410242e6
NEUTRON_MASS_U = 1.00866491595
NEUTRON_MASS_EV = NEUTRON_MASS_U * AMU_EV
ELECTRON_MASS_U = 5.48579909065e-4

#: MF1/451 NSUB -> the projectile, as (name, ZA, mass in neutron masses).
PROJECTILES = {
    0: ("photon", 0, 0.0), 10: ("n", 1, 1.0), 10010: ("p", 1001, 0.99862347),
    10020: ("d", 1002, 1.99681752), 10030: ("t", 1003, 2.99013997),
    20030: ("He3", 2003, 2.98960312), 20040: ("alpha", 2004, 3.96713199),
    113: ("e-", -1, 5.4386734e-4), 3: ("photon", 0, 0.0), 12: ("n", 1, 1.0),
}


def fmt_e(e: float) -> str:
    """An energy for a reader: ``1.069 MeV``, ``25.3 meV``, ``2.5e-05 eV``."""
    if e is None or not math.isfinite(e):
        return str(e)
    a = abs(e)
    for unit, scale in (("MeV", 1e6), ("keV", 1e3), ("eV", 1.0)):
        if a >= scale:
            return f"{e / scale:.6g} {unit}"
    if a >= 1e-3:
        return f"{e * 1e3:.4g} meV"
    return f"{e:.4g} eV"


def fmt_b(x: float) -> str:
    """A cross section for a reader: ``1.23 b``, ``4.5 mb``, ``2.1e-07 b`` (ASCII, like every summary)."""
    a = abs(x)
    if a >= 1 or a == 0:
        return f"{x:.6g} b"
    if a >= 1e-3:
        return f"{x * 1e3:.4g} mb"
    return f"{x:.4g} b"


@dataclass
class Table:
    """A TAB1 as written: abscissae, values and the (NBT, INT) pairs."""

    x: np.ndarray
    y: np.ndarray
    pairs: List[Tuple[int, int]]

    @classmethod
    def of(cls, x, y, pairs) -> "Table":
        return cls(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                   [(int(a), int(b)) for a, b in (pairs or [])])

    def laws(self) -> np.ndarray:
        """One law per interval, from the pairs (lin-lin when there are none)."""
        n = self.x.size
        laws = np.full(max(n - 1, 0), 2, dtype=np.int64)
        pairs = self.pairs or [(n, 2)]
        start = 0
        for nbt, code in pairs:
            stop = min(int(nbt) - 1, n - 1)
            if stop > start:
                laws[start:stop] = code
            start = max(start, stop)
        if start < n - 1:
            laws[start:] = pairs[-1][1]
        return laws

    def __call__(self, q) -> np.ndarray:
        """The table read at *q* (zero outside it), each law in closed form."""
        from kika.algebra import evaluate

        laws = np.where(np.isin(self.laws(), (1, 2, 3, 4, 5)), self.laws(), 2)
        return evaluate(self.x, self.y, laws, q)


class Out:
    """The findings of one check run; :meth:`add` is all a check calls."""

    def __init__(self, mat: Optional[int]) -> None:
        self.mat = mat
        self.findings: List[Finding] = []

    def add(self, check: str, level: str, summary: str, *, mf: int, mt: Optional[int] = None,
            **where_and_evidence: Any) -> None:
        loc_keys = {"k", "zap", "t", "l", "range_index", "lb", "ls", "nc", "ni", "band", "block"}
        loc = {k: where_and_evidence.pop(k) for k in list(where_and_evidence) if k in loc_keys}
        self.findings.append(Finding(check, level, summary,
                                     Location(mat=self.mat, mf=mf, mt=mt, **loc),
                                     dict(where_and_evidence)))


class Tape:
    """The tape the checks look at; every accessor is lazy and safe."""

    def __init__(self, endf, source: Optional[str] = None) -> None:
        self.endf = endf
        self.source = source or getattr(endf, "source_path", None)
        self.mat = getattr(endf, "mat", None)
        self._raw = None
        self.parse_failures: Dict[int, str] = {}
        #: The files the run was asked for; a check that looks across files
        #: (the HEAD of every section) stays inside them.
        self.wanted: Tuple[int, ...] = tuple(sorted(endf.files))

    # ---- files and sections ------------------------------------------------

    @property
    def present(self) -> Tuple[int, ...]:
        files = self.endf.files
        return tuple(sorted(files.keys())) if hasattr(files, "keys") else tuple(sorted(files))

    def has(self, mf: int) -> bool:
        return mf in self.endf.files

    def mf(self, mf: int):
        """The parsed MF, or ``None`` when absent or unreadable (recorded)."""
        if mf not in self.endf.files or mf in self.parse_failures:
            return None
        try:
            return self.endf.files[mf]
        except Exception as exc:  # noqa: BLE001 - one unreadable MF must not stop the rest
            self.parse_failures[mf] = f"{type(exc).__name__}: {exc}"
            return None

    def sections(self, mf: int) -> Dict[int, Any]:
        obj = self.mf(mf)
        return dict(getattr(obj, "sections", {}) or {}) if obj is not None else {}

    def section(self, mf: int, mt: int):
        return self.sections(mf).get(mt)

    def parse_errors(self, mf: int) -> Dict[int, str]:
        obj = self.mf(mf)
        return dict(getattr(obj, "parse_errors", {}) or {}) if obj is not None else {}

    # ---- the header --------------------------------------------------------

    @property
    def header(self):
        return self.section(1, 451)

    def _h(self, name: str, default=None):
        h = self.header
        value = getattr(h, name, None) if h is not None else None
        return default if value is None else value

    @property
    def za(self) -> Optional[int]:
        za = self._h("_za")
        if za is None:
            return None
        return int(round(float(za)))

    @property
    def awr(self) -> Optional[float]:
        awr = self._h("_awr")
        return None if awr is None else float(awr)

    @property
    def nsub(self) -> Optional[int]:
        return self._h("_nsub")

    @property
    def projectile(self) -> Tuple[str, int, float]:
        """(name, ZA, mass in neutron masses); neutrons when the header says nothing."""
        return PROJECTILES.get(self.nsub, PROJECTILES[10])

    @property
    def awi(self) -> float:
        awi = self._h("_awi")
        return float(awi) if awi is not None else self.projectile[2]

    @property
    def emax(self) -> Optional[float]:
        e = self._h("_emax")
        return None if e is None or e <= 0 else float(e)

    @property
    def liso(self) -> int:
        return int(self._h("_liso", 0) or 0)

    @property
    def elis(self) -> float:
        return float(self._h("_elis", 0.0) or 0.0)

    @property
    def lrp(self) -> Optional[int]:
        return self._h("_lrp")

    @property
    def lfi(self) -> Optional[int]:
        return self._h("_lfi")

    @property
    def neutron_projectile(self) -> bool:
        return self.nsub in (None, 10)

    # ---- tables ------------------------------------------------------------

    def xs(self, mt: int) -> Optional[Table]:
        sec = self.section(3, mt)
        if sec is None:
            return None
        return Table.of(getattr(sec, "_energies", []), getattr(sec, "_cross_sections", []),
                        getattr(sec, "_interpolation", []))

    def q(self, mt: int) -> Tuple[Optional[float], Optional[float]]:
        """(QM, QI) of MF3/MT, eV."""
        sec = self.section(3, mt)
        if sec is None:
            return None, None
        return float(getattr(sec, "_qm", 0.0) or 0.0), float(getattr(sec, "_qi", 0.0) or 0.0)

    def lr(self, mt: int) -> int:
        sec = self.section(3, mt)
        return int(getattr(sec, "_lr", 0) or 0) if sec is not None else 0

    def resolved_upper(self) -> Optional[float]:
        """The top of the resolved resonance range (MF2), where MF3 stops being a background."""
        from ..covariances import _resonance_upper_limit

        try:
            return _resonance_upper_limit(self.endf)
        except Exception:  # noqa: BLE001
            return None

    def resonance_mts(self) -> frozenset:
        """The MF3 sections MF2 adds resonances to: there, inside the ranges, MF3 is a background.

        SLBW/MLBW/Reich-Moore and the unresolved range feed elastic, capture and
        fission (and their sums); an R-matrix limited range feeds the MT of each
        of its particle pairs, and the sums those belong to.
        """
        from kika._constants import MF3_SUM_RULES

        mts = {1, 2, 3, 18, 19, 27, 101, 102}
        sec = self.section(2, 151)
        for iso in getattr(sec, "isotopes", None) or getattr(sec, "_isotopes", []) or []:
            for rng in getattr(iso, "energy_ranges", []) or []:
                for pp in getattr(getattr(rng, "parameters", None), "particle_pairs", []) or []:
                    mt = int(getattr(pp, "mt", 0) or 0)
                    if mt > 0:
                        mts.add(mt)
        # the sums the fed partials belong to, as far up as they go
        grew = True
        while grew:
            grew = False
            for total, (parts, _) in MF3_SUM_RULES.items():
                plain = {int(str(p).lstrip("@")) for p in parts}
                if total not in mts and plain & mts:
                    mts.add(total)
                    grew = True
        return frozenset(mts)

    def resonance_ranges(self) -> List[Tuple[float, float, int, int]]:
        """``[(EL, EH, LRU, LRF)]`` of every range of MF2/151, all isotopes."""
        sec = self.section(2, 151)
        out = []
        for iso in getattr(sec, "isotopes", None) or getattr(sec, "_isotopes", []) or []:
            for rng in getattr(iso, "energy_ranges", []) or []:
                out.append((float(getattr(rng, "el", 0.0)), float(getattr(rng, "eh", 0.0)),
                            int(getattr(rng, "lru", 0)), int(getattr(rng, "lrf", 0))))
        return out

    # ---- raw lines ---------------------------------------------------------

    def raw(self) -> Optional["Raw"]:
        """The tape's lines grouped by section, or ``None`` with no file behind it."""
        if self._raw is None and self.source and Path(self.source).is_file():
            self._raw = Raw.read(Path(self.source))
        return self._raw


@dataclass
class Raw:
    """Line counts and identifiers of every section, read from the columns 67-80."""

    #: (MF, MT) -> number of lines, its SEND excluded (the NC of the directory).
    lines: Dict[Tuple[int, int], int] = field(default_factory=dict)
    #: (MF, MT) in tape order, first appearance.
    order: List[Tuple[int, int]] = field(default_factory=list)
    #: MAT numbers seen on data lines (tape ends, -1 and 0, left out).
    mats: Dict[int, int] = field(default_factory=dict)
    #: (MF, MT) that appear in more than one run (split sections).
    split: List[Tuple[int, int]] = field(default_factory=list)

    @classmethod
    def read(cls, path: Path) -> "Raw":
        """One pass over the columns 67-75 of every line, grouped into runs.

        The runs are formed on the raw bytes (``itertools.groupby``) and only
        each run's key is parsed: a JENDL-5 tape has a quarter of a million
        lines and a few hundred runs.
        """
        from itertools import groupby

        data = path.read_bytes()
        lines: Dict[Tuple[int, int], int] = {}
        runs: Dict[Tuple[int, int], int] = {}
        order: List[Tuple[int, int]] = []
        mats: Dict[int, int] = {}
        for raw_key, group in groupby(line[66:75] for line in data.splitlines()):
            if len(raw_key) < 9:
                continue
            try:
                mat, mf, mt = int(raw_key[0:4]), int(raw_key[4:6]), int(raw_key[6:9])
            except ValueError:
                continue
            n = sum(1 for _ in group)
            if mat > 0 and mf > 0:      # not the tape id, FEND, MEND or TEND
                mats[mat] = mats.get(mat, 0) + n
            if mf > 0 and mt > 0:
                key = (mf, mt)
                if key not in lines:
                    order.append(key)
                lines[key] = lines.get(key, 0) + n
                runs[key] = runs.get(key, 0) + 1
        split = [k for k in order if runs[k] > 1]
        return cls(lines, order, mats, split)
