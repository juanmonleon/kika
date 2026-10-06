"""Layer 1 over a whole library: :func:`check_covariances` on every tape of a directory.

One tape at a time, in series: each is read (only the MF the checks need),
checked and released before the next, so a library of hundreds of tapes runs in
the memory of the largest one. Running several libraries side by side is the
caller's choice, and on a shared 12 GB box it is the wrong one.

A tape that cannot be read, or whose check fails, is recorded on its row and
the walk goes on: in a library census the failures are findings too.

Plan: kika-workspace ``docs/library/cov_checks_roadmap.md``.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from .findings import (
    _RANK,
    DEFECT,
    NOTE,
    WARN,
    CovarianceCheckReport,
    CovarianceFinding,
    CovarianceLocation,
)

__all__ = ["check_covariance_library", "CovarianceLibraryReport", "TapeCheck", "TAPE_PATTERNS"]

#: File names a library directory holds its tapes under: ENDF/B (``.endf``),
#: JEFF (``.jeff``, ``.txt``), JENDL (``.dat``). README and CHANGELOG files do
#: not match.
TAPE_PATTERNS = ("*.endf", "*.jeff", "*.txt", "*.dat")

#: What a covariance file needs from the rest of the tape: MF1 (nu-bar for
#: MF31), MF2 (where MF3 stops being a background), MF3 and MF4 (central values).
_SUPPORT_MF = (1, 2, 3, 4)

Progress = Union[bool, None, Callable[[str], None]]


@dataclass(frozen=True)
class TapeCheck:
    """What happened to one tape: its report, or why there is none."""

    path: Path
    report: Optional[CovarianceCheckReport] = None
    error: Optional[str] = None
    has_covariances: bool = True
    read_seconds: float = 0.0
    check_seconds: float = 0.0

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def mat(self) -> Optional[int]:
        return self.report.mat if self.report is not None else None

    @property
    def ok(self) -> bool:
        """Read and checked (with or without findings)."""
        return self.error is None

    def count(self, level: str) -> int:
        return 0 if self.report is None else sum(1 for f in self.report if f.level == level)


@dataclass(frozen=True)
class CovarianceLibraryReport:
    """Every tape of a library and what :func:`check_covariances` found in each.

    Iterating gives ``(tape, finding)`` pairs. :meth:`summary` is the census
    table -- findings and tapes per (MF, level, check) -- and :meth:`write`
    puts the three tables on disk as TSV.
    """

    tapes: Tuple[TapeCheck, ...] = ()
    directory: Optional[Path] = None
    library: Optional[str] = None

    def __iter__(self) -> Iterator[Tuple[TapeCheck, CovarianceFinding]]:
        for tape in self.tapes:
            if tape.report is not None:
                for finding in tape.report:
                    yield tape, finding

    def __len__(self) -> int:
        return len(self.tapes)

    @property
    def checked(self) -> Tuple[TapeCheck, ...]:
        """Tapes that carry covariances and were checked."""
        return tuple(t for t in self.tapes if t.ok and t.has_covariances)

    @property
    def failed(self) -> Tuple[TapeCheck, ...]:
        """Tapes that could not be read or checked."""
        return tuple(t for t in self.tapes if not t.ok)

    def at_least(self, level: str) -> Tuple[Tuple[TapeCheck, CovarianceFinding], ...]:
        floor = _RANK[level]
        return tuple((t, f) for t, f in self if _RANK[f.level] >= floor)

    def by_check(self, check: str, level: Optional[str] = None):
        return tuple((t, f) for t, f in self
                     if f.check == check and (level is None or f.level == level))

    def tapes_with(self, check: str, level: Optional[str] = None,
                   mf: Optional[int] = None) -> Tuple[TapeCheck, ...]:
        """The tapes with at least one such finding, in library order."""
        hit = {id(t) for t, f in self.by_check(check, level)
               if mf is None or f.location.mf == mf}
        return tuple(t for t in self.tapes if id(t) in hit)

    def report(self, name: str) -> CovarianceCheckReport:
        """The report of one tape, by file name."""
        for tape in self.tapes:
            if tape.name == name:
                if tape.report is None:
                    raise ValueError(f"{name} has no report: {tape.error or 'no covariances'}")
                return tape.report
        raise KeyError(name)

    # ---- tables -----------------------------------------------------------

    def summary(self):
        """One row per (MF, level, check): ``findings`` and ``tapes``, worst first."""
        import pandas as pd

        counts: Dict[Tuple[int, str, str], List] = {}
        for tape, f in self:
            key = (f.location.mf, f.level, f.check)
            n, names = counts.setdefault(key, [0, set()])
            counts[key][0] = n + 1
            names.add(tape.name)
        rows = [{"mf": mf, "level": level, "check": check, "findings": n, "tapes": len(names)}
                for (mf, level, check), (n, names) in counts.items()]
        frame = pd.DataFrame(rows, columns=["mf", "level", "check", "findings", "tapes"])
        if frame.empty:
            return frame
        frame["_rank"] = frame["level"].map(lambda lv: -_RANK[lv])
        frame = frame.sort_values(["mf", "_rank", "tapes", "check"],
                                  ascending=[True, True, False, True])
        return frame.drop(columns="_rank").reset_index(drop=True)

    def files_dataframe(self):
        """One row per tape: MAT, counts by level, timings, and the error if any."""
        import pandas as pd

        rows = [{"file": t.name, "mat": t.mat, "has_covariances": t.has_covariances,
                 "n_defect": t.count(DEFECT), "n_warn": t.count(WARN), "n_note": t.count(NOTE),
                 "read_s": round(t.read_seconds, 2), "check_s": round(t.check_seconds, 2),
                 "error": t.error} for t in self.tapes]
        return pd.DataFrame(rows, columns=["file", "mat", "has_covariances", "n_defect",
                                           "n_warn", "n_note", "read_s", "check_s", "error"])

    def to_dataframe(self, level: str = NOTE):
        """One row per finding at *level* or worse, with its file and location."""
        import pandas as pd

        loc_names = [f.name for f in fields(CovarianceLocation)]
        floor = _RANK[level]
        rows = []
        for tape, f in self:
            if _RANK[f.level] < floor:
                continue
            row = {"file": tape.name, "level": f.level, "check": f.check}
            row.update({name: getattr(f.location, name) for name in loc_names})
            row["summary"] = f.summary
            row["evidence"] = f.evidence
            rows.append(row)
        return pd.DataFrame(rows, columns=["file", "level", "check", *loc_names,
                                           "summary", "evidence"])

    def write(self, out_dir, *, level: str = WARN) -> Dict[str, Path]:
        """Write ``summary``, ``files`` and ``findings`` as TSV into *out_dir*.

        *level* is the floor for the findings table. The default leaves the
        notes out: they are most of the rows (inert rows, rounding, coverage),
        the summary counts them anyway, and a re-run brings them back in minutes.
        Evidence goes as JSON. Returns the paths by table name.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = f"_{self.library}" if self.library else ""
        paths = {name: out / f"covariance_{name}{tag}.tsv"
                 for name in ("summary", "files", "findings")}
        self.summary().to_csv(paths["summary"], sep="\t", index=False)
        self.files_dataframe().to_csv(paths["files"], sep="\t", index=False)
        findings = self.to_dataframe(level)
        findings["evidence"] = findings["evidence"].map(lambda e: json.dumps(e, default=str))
        findings.to_csv(paths["findings"], sep="\t", index=False)
        return paths

    def __str__(self) -> str:
        where = self.library or (str(self.directory) if self.directory else "library")
        checked = self.checked
        with_defect = sum(1 for t in checked if t.count(DEFECT))
        lines = [f"Covariance check of {where}: {len(self.tapes)} tapes, "
                 f"{len(checked)} with covariances checked, {with_defect} with defects, "
                 f"{len(self.failed)} failed"]
        summary = self.summary()
        for row in summary[summary["level"] != NOTE].itertuples():
            lines.append(f"  MF{row.mf} {row.level:<6} {row.check:<36} "
                         f"{row.findings:>6} in {row.tapes} tapes")
        n_notes = int(summary.loc[summary["level"] == NOTE, "findings"].sum()) if len(summary) else 0
        if n_notes:
            lines.append(f"  ... and {n_notes} notes (summary() lists them)")
        for tape in self.failed:
            lines.append(f"  FAILED {tape.name}: {tape.error}")
        return "\n".join(lines)


def _tapes(source, patterns: Sequence[str], recursive: bool) -> Tuple[Optional[Path], List[Path]]:
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        root = Path(source)
        glob = root.rglob if recursive else root.glob
        return root, sorted({p for pattern in patterns for p in glob(pattern) if p.is_file()})
    if isinstance(source, (str, Path)):
        raise FileNotFoundError(f"not a directory: {source}")
    return None, [Path(p) for p in source]


def _reporter(progress: Progress, total: int) -> Callable[[int, str, bool], None]:
    """``(i, message, last)``: a callback gets each message; ``True`` keeps one
    line on stderr, rewritten in place, and ends it with the last."""
    if not progress:
        return lambda i, message, last: None
    if callable(progress):
        return lambda i, message, last: progress(message)

    width = [0]

    def show(i: int, message: str, last: bool) -> None:
        pad = " " * max(0, width[0] - len(message))
        width[0] = len(message)
        sys.stderr.write("\r" + message + pad + ("\n" if last else ""))
        sys.stderr.flush()

    return show


def check_covariance_library(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    mf: Sequence[int] = (31, 33, 34),
    patterns: Sequence[str] = TAPE_PATTERNS,
    recursive: bool = False,
    library: Optional[str] = None,
    progress: Progress = True,
) -> CovarianceLibraryReport:
    """Run :func:`check_covariances` on every tape of a library, one at a time.

    Parameters
    ----------
    source : path or iterable of paths
        A directory of ENDF tapes, or the tapes themselves.
    mf : sequence of int
        The covariance files to check (any of 31, 33, 34). Only these and the
        files that give them central values (MF1-4) are read.
    patterns : sequence of str
        Globs that pick the tapes in a directory. The default covers how
        ENDF/B, JEFF and JENDL name theirs.
    recursive : bool
        Look in subdirectories too.
    library : str, optional
        A tag for the report and the names of the files :meth:`~CovarianceLibraryReport.write`
        writes (e.g. ``"jeff40"``).
    progress : bool or callable
        ``True`` (default) keeps one line on stderr with the tape being checked
        and the running tally; a callable receives each of those lines instead
        (``print`` gives one line per tape); ``False`` is silent.

    Returns
    -------
    CovarianceLibraryReport

    No PENDF is attached, so a block mixing absolute and relative components is
    checked on its relative part, and says so (``mixed_needs_cross_sections``).
    """
    import logging
    import warnings

    from kika.endf import read_endf

    from .covariances import check_covariances

    wanted = sorted({m for m in mf if m in (31, 33, 34)})
    if not wanted:
        raise ValueError(f"mf must name at least one of 31, 33, 34, got {list(mf)}")
    read_mf = sorted(set(_SUPPORT_MF) | set(wanted))
    root, paths = _tapes(source, patterns, recursive)
    report = _reporter(progress, len(paths))
    out: List[TapeCheck] = []
    n_defect = n_failed = 0
    started = time.perf_counter()

    # The parser warns about every MF it skips and logs every quirk it repairs;
    # over hundreds of tapes that drowns the progress line. What matters is in
    # the reports, and a failure is on its row.
    kika_log = logging.getLogger("kika")
    level = kika_log.level
    kika_log.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, path in enumerate(paths, 1):
                report(i, f"[{i}/{len(paths)}] {path.name} ...", False)
                t0 = time.perf_counter()
                try:
                    endf = read_endf(str(path), mf_numbers=read_mf)
                except Exception as exc:  # noqa: BLE001 - recorded, the walk goes on
                    out.append(TapeCheck(path, error=f"read: {type(exc).__name__}: {exc}",
                                         read_seconds=time.perf_counter() - t0))
                    n_failed += 1
                    continue
                t1 = time.perf_counter()
                if not endf.files:
                    # read_endf returns an empty tape for a file that is not
                    # ENDF at all; in a census that is a failure, not a tape
                    # without covariances.
                    out.append(TapeCheck(path, error="read: no ENDF section found in the file",
                                         read_seconds=t1 - t0))
                    n_failed += 1
                    continue
                if not any(m in endf.files for m in wanted):
                    out.append(TapeCheck(path, has_covariances=False, read_seconds=t1 - t0))
                    del endf
                    continue
                try:
                    tape_report = check_covariances(endf, mf=wanted)
                except Exception as exc:  # noqa: BLE001
                    out.append(TapeCheck(path, error=f"check: {type(exc).__name__}: {exc}",
                                         read_seconds=t1 - t0,
                                         check_seconds=time.perf_counter() - t1))
                    n_failed += 1
                    del endf
                    continue
                del endf
                tape = TapeCheck(path, report=tape_report, read_seconds=t1 - t0,
                                 check_seconds=time.perf_counter() - t1)
                out.append(tape)
                n_defect += bool(tape.count(DEFECT))
                report(i, f"[{i}/{len(paths)}] {path.name}: {tape.count(DEFECT)} defects, "
                       f"{tape.count(WARN)} warnings  (so far {n_defect} tapes with defects, "
                       f"{n_failed} failed)", False)
    finally:
        kika_log.setLevel(level)
    report(len(paths), f"{len(paths)} tapes in {time.perf_counter() - started:.0f} s: "
           f"{sum(1 for t in out if t.ok and t.has_covariances)} with covariances, "
           f"{n_defect} with defects, {n_failed} failed", True)
    return CovarianceLibraryReport(tuple(out), directory=root, library=library)
