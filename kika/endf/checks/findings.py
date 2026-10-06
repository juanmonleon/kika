"""What a check of an ENDF covariance section found, and where.

These are *layer 1* findings: about a section as it is written in the file,
before kika assembles it into a model covariance. None of them blocks anything.
``level`` says how bad the finding is, not what to do about it:

* ``note``   -- true and worth knowing, but not a fault: rounding, rows with no
  variance, a grid that stops short of MF3.
* ``warn``   -- probably a fault, or one that only matters in some uses.
* ``defect`` -- the section is not what ENDF-6 says it should be, or it is not a
  covariance: a count that does not match, |rho| > 1, a negative eigenvalue
  well above rounding.

Plan and decisions: kika-workspace ``docs/library/cov_checks_roadmap.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterator, Optional, Tuple

NOTE = "note"
WARN = "warn"
DEFECT = "defect"
LEVELS = (NOTE, WARN, DEFECT)
_RANK = {level: i for i, level in enumerate(LEVELS)}


@dataclass(frozen=True)
class CovarianceLocation:
    """Where in the file a finding is: as many of the ENDF indices as apply.

    ``ni`` and ``nc`` are 0-based positions of the sub-subsection within its
    subsection (MF33) or sub-subsection (MF34), in file order.
    """

    mat: Optional[int] = None
    mf: Optional[int] = None
    mt: Optional[int] = None
    mat1: Optional[int] = None
    mt1: Optional[int] = None
    l: Optional[int] = None
    l1: Optional[int] = None
    ni: Optional[int] = None
    nc: Optional[int] = None
    lb: Optional[int] = None
    ls: Optional[int] = None

    def __str__(self) -> str:
        parts = []
        if self.mf is not None:
            parts.append(f"MF{self.mf}")
        if self.mt is not None:
            block = f"MT{self.mt}"
            if self.mt1 is not None:
                other = f"MT{self.mt1}"
                if self.mat1:
                    other = f"MAT{self.mat1}/{other}"
                block += f"x{other}"
            parts.append(block)
        if self.l is not None or self.l1 is not None:
            parts.append(f"L{self.l}xL{self.l1}")
        if self.nc is not None:
            parts.append(f"NC[{self.nc}]")
        if self.ni is not None:
            parts.append(f"NI[{self.ni}]")
        if self.lb is not None:
            parts.append(f"LB={self.lb}" + (f" LS={self.ls}" if self.ls is not None else ""))
        return " ".join(parts) if parts else "(file)"


@dataclass(frozen=True)
class CovarianceFinding:
    """One thing that is true about a covariance section, with its evidence."""

    check: str
    level: str
    summary: str
    location: CovarianceLocation = field(default_factory=CovarianceLocation)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in _RANK:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")

    def __str__(self) -> str:
        return f"[{self.level}] {self.location}: {self.check} -- {self.summary}"


@dataclass(frozen=True)
class CovarianceCheckReport:
    """Every finding of :func:`check_covariances` on one file.

    Like the pre-flight's report, it defines no truthiness: "is the file ok"
    depends on which levels the caller cares about, so ask :attr:`defects`.
    """

    findings: Tuple[CovarianceFinding, ...] = ()
    source: Optional[str] = None
    mat: Optional[int] = None

    def __iter__(self) -> Iterator[CovarianceFinding]:
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    def at_least(self, level: str) -> Tuple[CovarianceFinding, ...]:
        """Findings at ``level`` or worse."""
        floor = _RANK[level]
        return tuple(f for f in self.findings if _RANK[f.level] >= floor)

    @property
    def defects(self) -> Tuple[CovarianceFinding, ...]:
        return tuple(f for f in self.findings if f.level == DEFECT)

    @property
    def warnings(self) -> Tuple[CovarianceFinding, ...]:
        return tuple(f for f in self.findings if f.level == WARN)

    @property
    def notes(self) -> Tuple[CovarianceFinding, ...]:
        return tuple(f for f in self.findings if f.level == NOTE)

    def by_check(self, check: str) -> Tuple[CovarianceFinding, ...]:
        return tuple(f for f in self.findings if f.check == check)

    def counts(self) -> Dict[Tuple[str, str], int]:
        """``{(level, check): n}``, worst level first."""
        out: Dict[Tuple[str, str], int] = {}
        for f in sorted(self.findings, key=lambda f: (-_RANK[f.level], f.check)):
            out[(f.level, f.check)] = out.get((f.level, f.check), 0) + 1
        return out

    def to_dataframe(self):
        """One row per finding: level, check, the location fields, summary, evidence."""
        import pandas as pd

        loc_names = [f.name for f in fields(CovarianceLocation)]
        rows = []
        for f in self.findings:
            row = {"level": f.level, "check": f.check}
            row.update({name: getattr(f.location, name) for name in loc_names})
            row["summary"] = f.summary
            row["evidence"] = f.evidence
            rows.append(row)
        return pd.DataFrame(rows, columns=["level", "check", *loc_names, "summary", "evidence"])

    def __str__(self) -> str:
        head = "Covariance check"
        if self.source:
            head += f" of {self.source}"
        if self.mat is not None:
            head += f" (MAT {self.mat})"
        if not self.findings:
            return f"{head}: nothing found"
        tally = ", ".join(
            f"{n} {level}" + ("s" if n != 1 else "")
            for level, n in ((lv, sum(1 for f in self.findings if f.level == lv))
                             for lv in reversed(LEVELS)) if n
        )
        lines = [f"{head}: {tally}"]
        for f in sorted(self.findings, key=lambda f: -_RANK[f.level]):
            lines.append(f"  {f}")
        return "\n".join(lines)
