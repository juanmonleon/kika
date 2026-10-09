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

import functools
import json
import os
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional,
                    Sequence, Tuple, Union)

from .covariances import CHECKED_MF
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

#: What each covariance file needs from the rest of the tape: MF1 (nu-bar for
#: MF31), MF2 (where MF3 stops being a background, and the parameters of MF32),
#: MF3 and MF4 (central values), MF5 (the spectra of MF35). MF40's central
#: values are MF10, which kika does not read. Only what the asked files need is
#: read: MF1-4 of a JENDL-5 tape cost ~6 s, for nothing if only MF35 is asked.
#: MF34 reads MF33 too: a_0 stands for the integrated cross section, whose
#: variance is MF33's (ENDF-6 §34.3).
_SUPPORT_MF = {31: (1,), 32: (2,), 33: (2, 3), 34: (4, 33), 35: (5,), 40: ()}

Progress = Union[bool, None, Callable[[str], None]]
#: ``on_tape(done, total, name, ok)``, called when each tape is done.
OnTape = Callable[[int, int, str, bool], None]
#: ``on_result(done, total, tape)``: the same moment, with the whole :class:`TapeCheck`.
OnResult = Callable[[int, int, "TapeCheck"], None]


@dataclass(frozen=True)
class TapeCheck:
    """What happened to one tape: its report, or why there is none."""

    path: Path
    report: Optional[CovarianceCheckReport] = None
    error: Optional[str] = None
    has_covariances: bool = True
    read_seconds: float = 0.0
    check_seconds: float = 0.0
    #: The covariance files asked for on this tape.
    mf: Tuple[int, ...] = ()
    #: Set only when two tapes of a walk share a file name: the path that tells
    #: them apart (relative to the directory walked, or as given).
    label: Optional[str] = None
    #: ZA and isomeric state from the head of MF1/MT451, read even when the
    #: tape fails or has no covariances: what the report sorts and indexes by.
    za: Optional[int] = None
    liso: int = 0

    @property
    def name(self) -> str:
        """The file name, or :attr:`label` where the file name is not unique."""
        return self.label or self.path.name

    @property
    def target(self) -> Optional[str]:
        """The nuclide in the G4NDL spelling (``Fe56``, ``Am242m1``), when the header gave one."""
        if self.za is None:
            return None
        from kika.library_index import target_name

        Z, A = divmod(self.za, 1000)
        return target_name(Z, A or None, self.liso)

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
    #: True when ``should_stop`` ended the walk early: ``tapes`` holds the ones
    #: done, out of ``planned``.
    stopped: bool = False
    planned: int = 0

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
            return frame.astype({"mf": "Int64"})
        frame["_rank"] = frame["level"].map(lambda lv: -_RANK[lv])
        frame = frame.sort_values(["mf", "_rank", "tapes", "check"],
                                  ascending=[True, True, False, True])
        return frame.drop(columns="_rank").reset_index(drop=True).astype({"mf": "Int64"})

    def files_dataframe(self):
        """One row per tape: MAT, counts by level, timings, and the error if any."""
        import pandas as pd

        rows = [{"file": t.name, "mat": t.mat, "has_covariances": t.has_covariances,
                 "n_defect": t.count(DEFECT), "n_warn": t.count(WARN), "n_note": t.count(NOTE),
                 "read_s": round(t.read_seconds, 2), "check_s": round(t.check_seconds, 2),
                 "error": t.error} for t in self.tapes]
        return pd.DataFrame(rows, columns=["file", "mat", "has_covariances", "n_defect",
                                           "n_warn", "n_note", "read_s", "check_s", "error"]
                            ).astype({"mat": "Int64"})

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
        frame = pd.DataFrame(rows, columns=["file", "level", "check", *loc_names,
                                            "summary", "evidence"])
        # Nullable ints: a column with any None would otherwise be float (1025.0).
        return frame.astype({name: "Int64" for name in loc_names})

    def to_dict(self, level: str = NOTE) -> Dict[str, Any]:
        """The report as plain data, safe for JSON and msgpack.

        ``kika_version``, ``library``, ``stopped``, one entry per tape (name,
        path, MAT, MF asked, status, counts by level, timings), the
        ``summary`` by (MF, level, check) over every finding, and the
        ``findings`` at *level* or worse, each with the ``tape`` it is in.
        Over a whole library the notes are tens of thousands of rows; pass
        ``level="warn"`` to leave them out.
        """
        from .export import library_dict

        return library_dict(self, level)

    def to_markdown(self, path=None, *, level: str = WARN) -> str:
        """A self-contained Markdown report; written to *path* too if given.

        The summary, one row per tape, the findings at *level* or worse by
        tape (notes only counted by default), and the method and thresholds.
        """
        from .export import library_markdown, write_text

        return write_text(library_markdown(self, level), path)

    def to_html(self, path=None, *, level: str = WARN) -> str:
        """The same report as :meth:`to_markdown`, as one HTML page with no external resources."""
        from .export import library_html, write_text

        return write_text(library_html(self, level), path)

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
        lines = [f"Covariance check of {where}: {len(self.tapes)} tapes"
                 + (f" (stopped, of {self.planned})" if self.stopped else "") + ", "
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


def _nuclide(path: Path) -> Tuple[Optional[int], int]:
    """``(ZA, LISO)`` from the head of the tape; ``(None, 0)`` when it has none.

    MF1/MT451 gives both. A tape without it (a covariance-only file, a cut)
    still opens with the HEAD of its first section, whose ZA is the nuclide;
    the isomeric state is then unknown and taken as the ground state.
    """
    from kika.library_index import _endf_float, _endf_header, _head

    try:
        text = _head(path)
        header = _endf_header(text)
    except (OSError, ValueError):
        return None, 0
    if header is not None and header[0] > 0:
        return header[0], header[3]
    for line in text.splitlines():
        try:
            if int(line[70:72]) > 0 and int(line[72:75]) > 0:
                za = int(round(_endf_float(line[0:11])))
                return (za, 0) if za > 0 else (None, 0)
        except ValueError:
            continue
    return None, 0


def _labels(paths: Sequence[Path], root: Optional[Path]) -> List[Optional[str]]:
    """A label for each tape whose file name another tape of the walk shares.

    The path under the directory walked; for tapes given one by one, the path
    under the deepest directory the tapes of that name share
    (``endfb7.1/n/92235.endf`` and ``endfb8.1/n/92235.endf``, not two full paths).
    """
    groups: Dict[str, List[Path]] = {}
    for p in paths:
        groups.setdefault(p.name, []).append(p)
    out: List[Optional[str]] = []
    for p in paths:
        same = groups[p.name]
        if len(same) == 1:
            out.append(None)
            continue
        base = root
        if base is None:
            try:
                base = Path(os.path.commonpath([str(q.resolve().parent) for q in same]))
            except ValueError:  # different drives
                base = None
        try:
            out.append(p.resolve().relative_to(base.resolve()).as_posix() if base else str(p))
        except ValueError:
            out.append(str(p))
    return out


def _tapes(source, patterns: Sequence[str], recursive: bool) -> Tuple[Optional[Path], List[Path]]:
    if isinstance(source, Mapping):
        return None, [Path(p) for p in source]
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


def _checked_mf(asked, where: str) -> Tuple[int, ...]:
    wanted = tuple(sorted({int(m) for m in asked if int(m) in CHECKED_MF}))
    if not wanted:
        raise ValueError(f"mf must name at least one of {', '.join(map(str, CHECKED_MF))}, "
                         f"got {list(asked)}{where}")
    return wanted


def check_covariance_library(
    source: Union[str, Path, Iterable[Union[str, Path]], Mapping[Union[str, Path], Sequence[int]]],
    *,
    mf: Sequence[int] = CHECKED_MF,
    patterns: Sequence[str] = TAPE_PATTERNS,
    recursive: bool = False,
    library: Optional[str] = None,
    progress: Progress = True,
    on_tape: Optional[OnTape] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_result: Optional[OnResult] = None,
) -> CovarianceLibraryReport:
    """Run :func:`check_covariances` on every tape of a library, one at a time.

    Parameters
    ----------
    source : path, iterable of paths, or mapping
        A directory of ENDF tapes, the tapes themselves, or ``{tape: mf}`` to
        check each tape for its own covariance files (a ``None`` value takes
        *mf*). Tapes are walked in the order given.
    mf : sequence of int
        The covariance files to check (any of 31, 32, 33, 34, 35, 40; all by
        default), for every tape a mapping *source* does not set. Only these
        that are present and the files that give them central values
        (``_SUPPORT_MF``) are parsed. A single index scan detects tapes without
        requested covariances without parsing any MF.
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
    on_tape : callable, optional
        ``on_tape(done, total, name, ok)``, structured progress next to (not
        instead of) *progress*: called when each tape is done, ``done`` going
        from 1 to ``total``; ``name`` is :attr:`TapeCheck.name` and ``ok`` is
        False for a tape that could not be read or checked. An exception in it
        is not caught.
    on_result : callable, optional
        ``on_result(done, total, tape)``, at the same moments as *on_tape*,
        with the :class:`TapeCheck` itself: its counts by level, MAT and
        nuclide, for a caller that keeps a running tally.
    should_stop : callable, optional
        Asked before each tape; when it returns true the walk stops there and
        the report of the tapes done so far comes back with ``stopped=True``.
        Nothing stops inside a tape, so the wait is at most one tape.

    Returns
    -------
    CovarianceLibraryReport

    No PENDF is attached, so a block mixing absolute and relative components is
    checked on its relative part, and says so (``mixed_needs_cross_sections``).
    """
    import logging
    import warnings

    from kika.endf import open_endf

    from .covariances import check_covariances

    default = _checked_mf(mf, "")
    per_tape: Dict[Path, Tuple[int, ...]] = {}
    if isinstance(source, Mapping):
        # Refused before the walk, not at tape 600.
        for p, asked in source.items():
            if asked is not None:
                per_tape[Path(p)] = _checked_mf(asked, f" for {Path(p).name}")
    root, paths = _tapes(source, patterns, recursive)
    labels = _labels(paths, root)
    report = _reporter(progress, len(paths))
    out: List[TapeCheck] = []
    n_defect = n_failed = 0
    stopped = False
    started = time.perf_counter()

    def _done(done: int) -> None:
        if on_tape is not None:
            on_tape(done, len(paths), out[-1].name, out[-1].ok)
        if on_result is not None:
            on_result(done, len(paths), out[-1])

    # The parser warns about every MF it skips and logs every quirk it repairs;
    # over hundreds of tapes that drowns the progress line. What matters is in
    # the reports, and a failure is on its row.
    kika_log = logging.getLogger("kika")
    level = kika_log.level
    kika_log.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, (path, label) in enumerate(zip(paths, labels), 1):
                if out:
                    _done(i - 1)
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                wanted = per_tape.get(path, default)
                za, liso = _nuclide(path)
                make = functools.partial(TapeCheck, path, mf=wanted, label=label, za=za, liso=liso)
                report(i, f"[{i}/{len(paths)}] {label or path.name} ...", False)
                t0 = time.perf_counter()
                try:
                    endf = open_endf(str(path))
                    present = set(endf.files)
                    covariances = present.intersection(wanted)
                    if covariances:
                        read_mf = sorted(covariances.union(
                            *(_SUPPORT_MF[m] for m in covariances)))
                        # _Context accesses every support MF it can see. Give
                        # it only those needed by the requested covariances,
                        # and finish parsing here so failures and timing stay
                        # in the read phase. Membership never parses an MF.
                        endf.files = {m: endf.files[m] for m in read_mf if m in present}
                except Exception as exc:  # noqa: BLE001 - recorded, the walk goes on
                    out.append(make(
                        error=f"read: {type(exc).__name__}: {exc}",
                        read_seconds=time.perf_counter() - t0))
                    n_failed += 1
                    continue
                t1 = time.perf_counter()
                if not endf.files:
                    # open_endf returns an empty tape for a file that is not
                    # ENDF at all; in a census that is a failure, not a tape
                    # without covariances.
                    out.append(make(
                        error="read: no ENDF section found in the file",
                        read_seconds=t1 - t0))
                    n_failed += 1
                    continue
                if not any(m in endf.files for m in wanted):
                    out.append(make(has_covariances=False, read_seconds=t1 - t0))
                    del endf
                    continue
                try:
                    tape_report = check_covariances(endf, mf=wanted)
                except Exception as exc:  # noqa: BLE001
                    out.append(make(
                        error=f"check: {type(exc).__name__}: {exc}",
                        read_seconds=t1 - t0,
                        check_seconds=time.perf_counter() - t1))
                    n_failed += 1
                    del endf
                    continue
                del endf
                tape = make(report=tape_report, read_seconds=t1 - t0,
                            check_seconds=time.perf_counter() - t1)
                out.append(tape)
                n_defect += bool(tape.count(DEFECT))
                report(i, f"[{i}/{len(paths)}] {tape.name}: {tape.count(DEFECT)} defects, "
                       f"{tape.count(WARN)} warnings  (so far {n_defect} tapes with defects, "
                       f"{n_failed} failed)", False)
        if out and not stopped:
            _done(len(out))
    finally:
        kika_log.setLevel(level)
    head = (f"stopped after {len(out)} of {len(paths)} tapes" if stopped
            else f"{len(paths)} tapes")
    report(len(out), f"{head} in {time.perf_counter() - started:.0f} s: "
           f"{sum(1 for t in out if t.ok and t.has_covariances)} with covariances, "
           f"{n_defect} with defects, {n_failed} failed", True)
    return CovarianceLibraryReport(tuple(out), directory=root, library=library,
                                   stopped=stopped, planned=len(paths))
