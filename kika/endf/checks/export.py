"""The layer-1 reports as plain data, Markdown and HTML.

One renderer for both reports (one tape, a library), so the page a notebook
writes and the one the desktop app exports are the same. Everything here is
self-contained: no figures, no external stylesheet or script.

The method section is built from the thresholds the checks use, not restated
by hand, so it cannot drift from them.
"""
from __future__ import annotations

import html
import math
import numbers
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .descriptions import CHECKS
from .findings import _RANK, DEFECT, LEVELS, NOTE, WARN, CovarianceFinding
from .symbols import to_html_symbols, to_symbols

SCHEMA = 1
_LEVEL_WORD = {DEFECT: "defect", WARN: "warning", NOTE: "note"}


# ---------------------------------------------------------------------------
# Plain data
# ---------------------------------------------------------------------------

def jsonable(value: Any) -> Any:
    """*value* with only None, bool, int, float, str, list and dict in it.

    numpy scalars and arrays become Python numbers and lists, tuples and sets
    lists, Paths strings, mapping keys strings. NaN and +-inf become ``None``:
    JSON has no spelling for them, and in the evidence they only ever mean
    "not available".
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value).__name__ == "bool_":  # numpy.bool_, without importing numpy
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        x = float(value)
        return x if math.isfinite(x) else None
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        return jsonable(value.tolist())  # numpy array or 0-d scalar
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [jsonable(v) for v in value]
        try:
            return sorted(items)
        except TypeError:
            return items
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def _int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def kika_version() -> str:
    try:
        from kika import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001 - a report without a version beats no report
        return "unknown"


def _sort_key(row: Dict[str, Any]) -> Tuple:
    return (row["mf"] if row["mf"] is not None else 10**6, -_RANK[row["level"]],
            -row.get("tapes", 0), row["check"])


def summary_rows(findings: Iterable[CovarianceFinding]) -> List[Dict[str, Any]]:
    """``[{mf, level, check, findings}]``, by MF and worst level first."""
    counts: Dict[Tuple, int] = {}
    for f in findings:
        key = (_int(f.location.mf), f.level, f.check)
        counts[key] = counts.get(key, 0) + 1
    rows = [{"mf": mf, "level": level, "check": check, "findings": n}
            for (mf, level, check), n in counts.items()]
    return sorted(rows, key=_sort_key)


def _level_counts(findings: Iterable[CovarianceFinding]) -> Dict[str, int]:
    out = {level: 0 for level in reversed(LEVELS)}
    for f in findings:
        out[f.level] += 1
    return out


def _tape_identity(report) -> Dict[str, Any]:
    path = report.source
    return {"name": Path(path).name if path else None, "path": str(path) if path else None,
            "mat": _int(report.mat), "mf": [int(m) for m in report.mf]}


def _findings_at(findings: Iterable[CovarianceFinding], level: str) -> List[CovarianceFinding]:
    floor = _RANK[level]
    return sorted((f for f in findings if _RANK[f.level] >= floor),
                  key=lambda f: (-_RANK[f.level], _int(f.location.mf) or 0,
                                 _int(f.location.mt) or 0, f.check))


def _check_names(rows: Iterable[Dict[str, Any]]) -> List[str]:
    """The check names of summary rows, first seen first (worst level first)."""
    seen: Dict[str, None] = {}
    for r in rows:
        seen.setdefault(r["check"], None)
    return list(seen)


def checks_dict(names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """``{check: {title, mf, levels, description}}`` for *names* (see ``descriptions.CHECKS``)."""
    return {n: CHECKS[n].to_dict() for n in names if n in CHECKS}


def report_dict(report, level: str = NOTE) -> Dict[str, Any]:
    _RANK[level]  # KeyError on a bad level, before any work
    return {
        "kind": "kika.covariance_check",
        "schema": SCHEMA,
        "kika_version": kika_version(),
        "tape": _tape_identity(report),
        "totals": _level_counts(report.findings),
        "counts": [{"mf": r["mf"], "level": r["level"], "check": r["check"],
                    "n": r["findings"]} for r in summary_rows(report.findings)],
        "level": level,
        "findings": [f.to_dict() for f in _findings_at(report.findings, level)],
        "checks": checks_dict(_check_names(summary_rows(report.findings))),
    }


def _tape_dict(tape) -> Dict[str, Any]:
    return {
        "name": tape.name,
        "path": str(tape.path),
        "mat": _int(tape.mat),
        "mf": [int(m) for m in tape.mf],
        "ok": tape.ok,
        "error": tape.error,
        "has_covariances": tape.has_covariances,
        "n_defect": tape.count(DEFECT),
        "n_warn": tape.count(WARN),
        "n_note": tape.count(NOTE),
        "read_s": round(float(tape.read_seconds), 3),
        "check_s": round(float(tape.check_seconds), 3),
    }


def library_summary_rows(library) -> List[Dict[str, Any]]:
    """``[{mf, level, check, findings, tapes}]`` over every tape, worst first."""
    counts: Dict[Tuple, List] = {}
    for tape, f in library:
        entry = counts.setdefault((_int(f.location.mf), f.level, f.check), [0, set()])
        entry[0] += 1
        entry[1].add(tape.name)
    rows = [{"mf": mf, "level": level, "check": check, "findings": n, "tapes": len(names)}
            for (mf, level, check), (n, names) in counts.items()]
    return sorted(rows, key=_sort_key)


def _library_counts(library) -> Dict[str, Any]:
    checked = library.checked
    return {
        "tapes": len(library.tapes),
        "planned": library.planned,
        "checked": len(checked),
        "without_covariances": sum(1 for t in library.tapes if t.ok and not t.has_covariances),
        "failed": len(library.failed),
        "with_defects": sum(1 for t in checked if t.count(DEFECT)),
        "findings": _level_counts(f for _, f in library),
    }


def library_dict(library, level: str = NOTE) -> Dict[str, Any]:
    _RANK[level]
    findings = []
    for tape in library.tapes:
        if tape.report is not None:
            findings += [{"tape": tape.name, **f.to_dict()}
                         for f in _findings_at(tape.report.findings, level)]
    summary = library_summary_rows(library)
    return {
        "kind": "kika.covariance_library_check",
        "schema": SCHEMA,
        "kika_version": kika_version(),
        "library": library.library,
        "directory": str(library.directory) if library.directory else None,
        "stopped": library.stopped,
        "totals": _library_counts(library),
        "tapes": [_tape_dict(t) for t in library.tapes],
        "summary": summary,
        "level": level,
        "findings": findings,
        "checks": checks_dict(_check_names(summary)),
    }


# ---------------------------------------------------------------------------
# Method and thresholds, from the constants the checks use
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x * 100:g} %"


def method_sections() -> List[Tuple[str, List[str]]]:
    """``[(heading, [paragraph, ...])]``: what was checked and where each level starts."""
    from . import covariances as c
    from .mf35 import SUM_RULE_DEFECT

    return [
        ("Scope", [
            "Layer 1: each covariance section (MF31, MF32, MF33, MF34, MF35, MF40) is "
            "checked as it is written in the file, read with kika's ENDF parser, before "
            "any processing. Nothing in the file is changed, and no finding blocks "
            "anything. Processed (multigroup, NJOY/ERRORR) matrices are not checked, nor "
            "blocks against another material (MAT1 != 0), which one file cannot evaluate.",
            "Levels: a *note* is true and worth knowing but not a fault (rounding, rows "
            "with no variance, a grid that stops short of MF3); a *warning* is probably a "
            "fault, or one that only matters in some uses; a *defect* means the section "
            "is not what ENDF-6 says it should be, or is not a covariance.",
        ]),
        ("Structure and completeness", [
            "Counts against the ENDF-6 manual (2023, chapters 31-40): NL, NMT1, NI, NC, "
            "NT against NE/LS, LB/LTY/LCT valid for the file, orders within NL/NL1, grids "
            "strictly increasing (a repeated point is a warning), duplicate and missing "
            "blocks, cross blocks without their self blocks, and NC references that do "
            "not resolve. LS=1 (symmetric storage) in a cross block is a defect: kika, "
            "like any processor, mirrors the stored triangle, and the mirror of a cross "
            "block is not its transpose.",
        ]),
        ("Correlations", [
            f"|rho| above 1 + {c.RHO_DEFECT:g} is a defect; between 1 + {c.RHO_ROUNDING:g} "
            f"and 1 + {c.RHO_DEFECT:g} it is a rounding note. Each block is summed on its "
            "union grid as kika sums it; a cross block is compared with the variances of "
            "its two self blocks on a common grid.",
            "Negative variances and covariances on rows without variance are defects; "
            "rows that are exactly zero (max == min == 0) are notes. Self blocks stored "
            f"in full (LS=0) are compared with their transpose: an asymmetry above "
            f"{c.ASYM_DEFECT:g} is a defect, above {c.ASYM_NOTE:g} a note.",
        ]),
        ("Positive semi-definiteness", [
            "On the sum of each self block, without the rows that are exactly zero. "
            f"|lambda_min| / lambda_max below {c.PSD_NOTE:g} is rounding (note); up to "
            f"{c.PSD_DEFECT:g} a warning; above it a defect. Each LB=5 record is also "
            "decomposed alone, to say which one is already indefinite. MF32 is judged on "
            "its correlation matrix, because resonance parameters mix scales.",
            f"A warning becomes a note when the negative eigenvalue fits in the rounding "
            f"of each element to the {c.ENDF_DIGITS} significant figures of an ENDF field, "
            f"or when at least {_pct(c.RHO_QUANTUM_SHARE)} of the correlations sit on a "
            f"lattice of {', '.join(f'{q:g}' for q in c.RHO_QUANTA)} and the eigenvalue "
            "fits in what that rounding can produce.",
            "A remaining warning is then graded by its consequence: if clipping the "
            "negative eigenvalues changes no sigma (of the rows with sigma at least "
            f"{_pct(c.PSD_IMPACT_FLOOR)} of the largest) by {_pct(c.PSD_IMPACT_NOTE)} or "
            f"more, it becomes a note; if it changes one by {_pct(c.PSD_IMPACT_DEFECT)} or "
            "more, a defect. Notes and defects are never regraded.",
        ]),
        ("Magnitude of the uncertainty", [
            "Large variances are judged against the central values, never on the "
            "relative uncertainty alone. Central values: sigma-bar from a PENDF when one "
            "is attached, otherwise MF3 only above the upper limit of the MF2 ranges (MF33); "
            "nu-bar from MF1 (MF31); the Legendre coefficients of MF4 (MF34); the spectra "
            "of MF5 (MF35). MF40's central values are in MF10, which is not read.",
            "MF31/MF33: a cross section has no upper bound, so sigma_rel > 1 is a note "
            "(with the probability of a negative sample if drawn from a normal); "
            f"sigma_rel > {c.RELATIVE_IMPLAUSIBLE:g} where sigma-bar is at least "
            f"{_pct(c.RELATIVE_THRESHOLD_ZONE)} of the reaction's maximum is a warning "
            "(the measured edge in ENDF/B-VIII.1, JEFF-4.0 and JENDL-5).",
            "MF34: |a_l| <= 1 for l >= 1, so sigma(a_l) > 1 is impossible (Popoviciu) and "
            "a defect; a_0 is excluded, and the relative sigma is scaled by the smallest "
            "|a_l| in the bin.",
            f"MF35: the sum rule max_i |sum_j C_ij| / max |C| above {SUM_RULE_DEFECT:g} is "
            "a defect, as is var(P) > P(1 - P) with P from MF5.",
            "A block mixing absolute and relative components (MF33 LB=8 with LB=5) needs "
            "the cross sections to be summed; without a PENDF only its relative part is "
            "checked, and a note says so.",
        ]),
    ]


# ---------------------------------------------------------------------------
# Markdown and HTML
# ---------------------------------------------------------------------------
#
# Both pages read in the same order: what was checked, the verdict and the
# count per level, the summary by check (worst level first), the tapes (worst
# first), the findings grouped by check within each tape, then the method and
# the legend. A check is named by its title; the ``check`` id, which is what
# the data, the TSV and the code use, goes beside it.

#: How each level is written for a reader. The data keeps ``warn``.
LEVEL_LABEL = {DEFECT: "Defect", WARN: "Warning", NOTE: "Note"}
_LEVEL_PLURAL = {DEFECT: "Defects", WARN: "Warnings", NOTE: "Notes"}
#: One line under each count; the method says it at length.
_LEVEL_MEANS = {
    DEFECT: "Not what ENDF-6 says, or not a covariance",
    WARN: "Probably a fault, or one in some uses",
    NOTE: "True and worth knowing, not a fault",
}
#: A mark beside the colour, so the level reads without it (print, colour blindness).
_LEVEL_MARK = {DEFECT: "\u2715", WARN: "!", NOTE: "i"}
_WORST_FIRST = (DEFECT, WARN, NOTE)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_md_cell(v) for v in row) + " |" for row in rows]
    return out


def _mf(mf: Optional[int]) -> str:
    return "-" if mf is None else f"MF{mf}"


def _plural(n: int, level: str) -> str:
    return f"{n} {_LEVEL_WORD[level]}" + ("s" if n != 1 else "")


def _tally(counts: Dict[str, int]) -> str:
    parts = [_plural(n, level) for level, n in counts.items() if n]
    return ", ".join(parts) if parts else "nothing found"


def _hidden_line(counts: Dict[str, int], level: str) -> Optional[str]:
    hidden = {lv: n for lv, n in counts.items() if n and _RANK[lv] < _RANK[level]}
    if not hidden:
        return None
    return (f"{_tally(hidden)} not listed (counted in the summary; "
            f"level='{min(hidden, key=_RANK.get)}' lists them).")


def _title(check: str) -> str:
    """The check's title from the legend, ASCII; the id where it has none."""
    return CHECKS[check].title if check in CHECKS else check


def _worst(counts: Mapping[str, int]) -> Optional[str]:
    return next((lv for lv in _WORST_FIRST if counts.get(lv)), None)


def _display_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Summary rows worst level first, then by MF and how many tapes."""
    return sorted(rows, key=lambda r: (-_RANK[r["level"]],
                                       r["mf"] if r["mf"] is not None else 10**6,
                                       -r.get("tapes", 0), -r["findings"], r["check"]))


def _groups(findings: Sequence[CovarianceFinding]) -> List[Tuple[str, str, List[CovarianceFinding]]]:
    """``[(level, check, findings)]`` in the order of *findings* (worst first)."""
    out: Dict[Tuple[str, str], List[CovarianceFinding]] = {}
    for f in findings:
        out.setdefault((f.level, f.check), []).append(f)
    return [(level, check, fs) for (level, check), fs in out.items()]


def _verdict(counts: Mapping[str, int], library=None) -> Tuple[Optional[str], str, str]:
    """``(level, headline, detail)``: the one sentence the page opens with."""
    worst = _worst(counts)
    headline = {DEFECT: "Defects found", WARN: "No defects, warnings to review",
                NOTE: "No defects or warnings, notes only", None: "Nothing found"}[worst]
    detail = _tally(dict(counts)).capitalize() + "."
    if library is not None:
        checked = library.checked
        hit = sum(1 for t in checked if t.count(DEFECT))
        detail += f" {hit} of {len(checked)} tapes checked have defects."
        failed = len(library.failed)
        if failed:
            detail += f" {failed} could not be read or checked."
            if worst is None or _RANK[worst] < _RANK[DEFECT]:
                worst, headline = DEFECT, "Tapes that could not be checked"
    return worst, headline, detail


def _tape_order(library) -> List:
    """Failed tapes first, then by defects, warnings and notes; ties keep their order."""
    def key(t):
        return (t.ok, -t.count(DEFECT), -t.count(WARN), -t.count(NOTE))
    return sorted(library.tapes, key=key)


def _tape_status(tape) -> str:
    if not tape.ok:
        return f"failed: {tape.error}"
    return "checked" if tape.has_covariances else "no covariances"


def _base(library) -> Optional[Path]:
    """The directory walked, or the deepest directory every tape is under."""
    if library.directory:
        return Path(library.directory)
    parents = [str(Path(t.path).resolve().parent) for t in library.tapes]
    if not parents:
        return None
    try:
        return Path(os.path.commonpath(parents))
    except ValueError:  # different drives
        return None


def _rel(tape, base: Optional[Path]) -> str:
    """The tape's path relative to *base*, or in full when it is not under it."""
    if base is not None:
        try:
            return Path(tape.path).resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            pass
    return str(tape.path)


def _library_title(library) -> str:
    if library.library:
        return library.library
    if library.directory:
        return Path(library.directory).name or str(library.directory)
    return "a library"


def _library_status(library, counts: Mapping[str, Any]) -> str:
    status = (f"{counts['tapes']} tapes, {counts['checked']} with covariances checked, "
              f"{counts['with_defects']} with defects, {counts['failed']} failed")
    if library.stopped:
        status += f"; stopped after {counts['tapes']} of {counts['planned']}"
    return status


# ---- Markdown ---------------------------------------------------------------

def _md_check(check: str) -> str:
    return f"{to_symbols(_title(check))} (`{check}`)"


def _md_verdict(counts: Mapping[str, int], library=None) -> List[str]:
    _, headline, detail = _verdict(counts, library)
    return [f"> **{headline}.** {detail}", ""]


def _md_summary(rows: Sequence[Dict[str, Any]], tapes: bool) -> List[str]:
    out = ["## Summary", ""]
    if not rows:
        return out + ["Nothing found.", ""]
    header = ("Level", "Check", "MF", "Findings") + (("Tapes",) if tapes else ())
    body = [(f"**{LEVEL_LABEL[r['level']]}**", _md_check(r["check"]), _mf(r["mf"]), r["findings"])
            + ((r["tapes"],) if tapes else ()) for r in _display_rows(rows)]
    return out + _md_table(header, body) + [""]


def _md_findings(findings: Sequence[CovarianceFinding], depth: str) -> List[str]:
    out: List[str] = []
    for level, check, fs in _groups(findings):
        n = len(fs)
        out += [f"{depth} {LEVEL_LABEL[level]} · {_md_check(check)} · "
                f"{n} finding{'s' if n != 1 else ''}", ""]
        out += _md_table(("Location", "What was found"),
                         [(str(f.location), to_symbols(f.summary)) for f in fs])
        out.append("")
    return out


def _md_method() -> List[str]:
    out = ["## Method and thresholds", ""]
    for heading, paragraphs in method_sections():
        out += [f"### {heading}", ""]
        for p in paragraphs:
            out += [to_symbols(p), ""]
    return out


_LEGEND_INTRO = ("One entry per check in this report: the files it applies to, when it "
                 "gives each level, and what the finding means.")


def _md_legend(names: Sequence[str]) -> List[str]:
    entries = [(n, CHECKS[n]) for n in names if n in CHECKS]
    if not entries:
        return []
    out = ["## What each finding means", "", _LEGEND_INTRO, ""]
    for name, d in entries:
        out += [f"### {to_symbols(d.title)} (`{name}`)", "",
                "MF " + ", ".join(map(str, d.mf)), ""]
        out += [f"- **{LEVEL_LABEL[lv]}**: {to_symbols(d.levels[lv])}"
                for lv in _WORST_FIRST if lv in d.levels]
        out += ["", to_symbols(d.description), ""]
    return out


def report_markdown(report, level: str = WARN) -> str:
    _RANK[level]
    tape = _tape_identity(report)
    counts = _level_counts(report.findings)
    title = tape["name"] or "ENDF tape"
    out = [f"# Covariance check of {title}", ""]
    out += _md_verdict(counts)
    out += _md_table(("", ""), [
        ("Tape", tape["name"]), ("Path", tape["path"]), ("MAT", tape["mat"]),
        ("MF checked", ", ".join(map(str, tape["mf"]))),
        ("kika", kika_version()), ("Generated", _now())])
    out.append("")
    rows = summary_rows(report.findings)
    out += _md_summary(rows, tapes=False)
    listed = _findings_at(report.findings, level)
    out += ["## Findings", ""]
    out += _md_findings(listed, "###")
    hidden = _hidden_line(counts, level)
    if hidden:
        out += [hidden, ""]
    elif not listed:
        out += ["None.", ""]
    out += _md_method()
    out += _md_legend(_check_names(_display_rows(rows)))
    return "\n".join(out).rstrip() + "\n"


def library_markdown(library, level: str = WARN) -> str:
    _RANK[level]
    counts = _library_counts(library)
    out = [f"# Covariance check of {_library_title(library)}", ""]
    out += _md_verdict(counts["findings"], library)
    base = _base(library)
    out += _md_table(("", ""), [
        ("Library", library.library), ("Directory", base),
        ("Tapes", _library_status(library, counts)),
        ("kika", kika_version()), ("Generated", _now())])
    out.append("")
    rows = library_summary_rows(library)
    out += _md_summary(rows, tapes=True)
    out += ["## Tapes", ""]
    out += _md_table(("File", "MAT", "MF checked", "Defects", "Warnings", "Notes", "Status", "Path"),
                     [(t.name, t.mat, ", ".join(map(str, t.mf)), t.count(DEFECT), t.count(WARN),
                       t.count(NOTE), _tape_status(t), str(Path(t.path).resolve()))
                      for t in _tape_order(library)])
    out += ["", "## Findings", ""]
    any_listed = False
    for t in _tape_order(library):
        if t.report is None:
            continue
        listed = _findings_at(t.report.findings, level)
        if not listed:
            continue
        any_listed = True
        out += [f"### {_rel(t, base)}" + (f" (MAT {t.mat})" if t.mat is not None else ""), ""]
        out += _md_findings(listed, "####")
    hidden = _hidden_line(counts["findings"], level)
    if hidden:
        out += [hidden, ""]
    elif not any_listed:
        out += ["None.", ""]
    out += _md_method()
    out += _md_legend(_check_names(_display_rows(rows)))
    return "\n".join(out).rstrip() + "\n"


# ---- HTML -------------------------------------------------------------------

_CSS = """
:root{--fg:#1b1e24;--muted:#5f6672;--faint:#8a919c;--bg:#fff;--panel:#f6f7f9;--line:#e1e4e8;
--accent:#2f5d8a;
--defect:#b42318;--defect-bg:#fdecea;--defect-line:#f4b9b2;
--warn:#9a5b00;--warn-bg:#fff4e0;--warn-line:#f2cd8d;
--note:#3d5a80;--note-bg:#eaf1f8;--note-line:#bcd0e5;
--ok:#1f7a4d;--ok-bg:#e8f5ee;--ok-line:#b5dcc6}
@media (prefers-color-scheme:dark){:root{--fg:#e7e9ed;--muted:#a3aab5;--faint:#7c8490;
--bg:#14161a;--panel:#1c1f25;--line:#2f343c;--accent:#8fb6dd;
--defect:#ff8a7d;--defect-bg:#3a1d1b;--defect-line:#6e2e28;
--warn:#f2b54f;--warn-bg:#352a14;--warn-line:#6b5222;
--note:#9fbbe0;--note-bg:#1d2836;--note-line:#34495f;
--ok:#6fd19f;--ok-bg:#16301f;--ok-line:#2b5a3f}}
*{box-sizing:border-box}
body{font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--fg);
background:var(--bg);max-width:1180px;margin:0 auto;padding:32px 20px 64px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
code,.loc{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;font-size:13px}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0}
h1{font-size:28px;line-height:1.2;margin:4px 0 2px;word-break:break-word}
.sub{color:var(--muted);margin:0 0 14px;word-break:break-all;font-size:13px}
.meta{display:flex;flex-wrap:wrap;gap:6px 22px;margin:0 0 22px;padding:0;font-size:13px;color:var(--muted)}
.meta div{display:flex;gap:6px}.meta dt{font-weight:600;color:var(--fg)}.meta dd{margin:0}
h2{font-size:20px;margin:40px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 8px}
.verdict{display:flex;gap:14px;align-items:flex-start;border:1px solid var(--line);
border-left-width:6px;border-radius:10px;padding:14px 18px;margin:0 0 16px;background:var(--panel)}
.verdict .big{font-size:18px;font-weight:700;margin:0}.verdict p{margin:2px 0 0}
.verdict.defect{border-color:var(--defect-line);border-left-color:var(--defect);background:var(--defect-bg)}
.verdict.warn{border-color:var(--warn-line);border-left-color:var(--warn);background:var(--warn-bg)}
.verdict.note,.verdict.none{border-color:var(--ok-line);border-left-color:var(--ok);background:var(--ok-bg)}
.verdict.defect .big{color:var(--defect)}.verdict.warn .big{color:var(--warn)}
.verdict.note .big,.verdict.none .big{color:var(--ok)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:0 0 8px}
.card{border:1px solid var(--line);border-top-width:4px;border-radius:10px;padding:12px 16px;background:var(--bg)}
.card .n{font-size:34px;font-weight:700;line-height:1.1;font-variant-numeric:tabular-nums}
.card .lbl{display:flex;align-items:center;gap:8px;font-weight:600;margin-top:2px}
.card .means{color:var(--muted);font-size:13px;margin-top:2px}
.card.defect{border-top-color:var(--defect)}.card.defect .n{color:var(--defect)}
.card.warn{border-top-color:var(--warn)}.card.warn .n{color:var(--warn)}
.card.note{border-top-color:var(--note)}.card.note .n{color:var(--note)}
.card.zero .n{color:var(--faint)}
.toc{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:13px;margin:18px 0 0;padding:10px 0;
border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.mark{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;
border-radius:50%;font-size:11px;font-weight:800;line-height:1;color:#fff;flex:none}
.defect .mark,.mark.defect{background:var(--defect)}.warn .mark,.mark.warn{background:var(--warn)}
.note .mark,.mark.note{background:var(--note)}
@media (prefers-color-scheme:dark){.mark{color:#14161a}}
.badge{display:inline-flex;align-items:center;gap:6px;padding:2px 10px 2px 3px;border-radius:999px;
font-size:13px;font-weight:700;white-space:nowrap;border:1px solid}
.badge.defect{color:var(--defect);background:var(--defect-bg);border-color:var(--defect-line)}
.badge.warn{color:var(--warn);background:var(--warn-bg);border-color:var(--warn-line)}
.badge.note{color:var(--note);background:var(--note-bg);border-color:var(--note-line)}
.pill{display:inline-block;min-width:28px;padding:0 8px;border-radius:999px;text-align:center;
font-weight:700;font-size:13px;font-variant-numeric:tabular-nums;border:1px solid}
.pill.defect{color:var(--defect);background:var(--defect-bg);border-color:var(--defect-line)}
.pill.warn{color:var(--warn);background:var(--warn-bg);border-color:var(--warn-line)}
.pill.note{color:var(--note);background:var(--note-bg);border-color:var(--note-line)}
.pill.zero{color:var(--faint);background:transparent;border-color:transparent;font-weight:400}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;margin:6px 0 12px;font-size:14px}
th,td{padding:7px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
font-weight:600;background:var(--panel)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:hover{background:var(--panel)}
.ck .ttl{font-weight:600;color:var(--fg)}.id{color:var(--faint);font-size:12px;margin-left:6px}
.file{font-weight:600}.path{display:block;color:var(--faint);font-size:12px;word-break:break-all}
.status-failed{color:var(--defect);font-weight:600}.status-none{color:var(--faint)}
details.tape{border:1px solid var(--line);border-radius:10px;margin:0 0 12px;background:var(--bg)}
details.tape>summary{cursor:pointer;list-style:none;display:flex;flex-wrap:wrap;gap:8px 12px;
align-items:center;padding:10px 14px;background:var(--panel);border-radius:10px}
details.tape[open]>summary{border-bottom:1px solid var(--line);border-radius:10px 10px 0 0}
details.tape>summary::-webkit-details-marker{display:none}
details.tape>summary::before{content:"\\25B8";color:var(--muted);transition:transform .15s}
details.tape[open]>summary::before{transform:rotate(90deg)}
details.tape>summary .tname{font-weight:700;font-size:15px}
details.tape>summary .pills{margin-left:auto;display:flex;gap:6px}
.tbody{padding:4px 14px 6px}
.group{margin:12px 0 4px}
.ghead{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin:0 0 4px}
.ghead .ttl{font-weight:700;font-size:15px}.ghead .cnt{color:var(--muted);font-size:13px}
table.f td.loc{width:30%;white-space:nowrap}
details.more>summary{cursor:pointer;color:var(--accent);font-size:13px;margin:0 0 6px}
table.f{margin-top:2px}
p.muted{color:var(--muted)}
.method p{max-width:80ch}
.legend{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
.entry{border:1px solid var(--line);border-radius:10px;padding:12px 16px;scroll-margin-top:16px}
.entry:target{outline:2px solid var(--accent)}
.entry h3{margin:0 0 2px;font-size:15px}.entry .id{margin:0}
.entry ul{list-style:none;padding:0;margin:8px 0}.entry li{margin:4px 0;display:flex;gap:8px;align-items:baseline}
.entry li .badge{flex:none}.entry p{margin:6px 0 0;color:var(--muted);font-size:14px}
@media print{body{max-width:none;padding:0}.toc{display:none}details.tape>summary::before{content:none}
tbody tr:hover{background:none}h2{break-after:avoid}.entry,.card,.group{break-inside:avoid}}
"""


def _e(value: Any) -> str:
    return "" if value is None else html.escape(str(value))


class _Html(str):
    """A cell already rendered as HTML: not escaped again."""


def _badge(level: str) -> _Html:
    return _Html(f'<span class="badge {level}"><span class="mark">{_LEVEL_MARK[level]}</span>'
                 f"{LEVEL_LABEL[level]}</span>")


def _pill(level: str, n: int) -> _Html:
    cls = level if n else "zero"
    return _Html(f'<span class="pill {cls}" title="{_LEVEL_PLURAL[level]}">{n}</span>')


def _check_cell(check: str) -> _Html:
    return _Html(f'<a class="ck" href="#check-{_e(check)}"><span class="ttl">'
                 f"{to_html_symbols(_title(check))}</span></a>"
                 f'<code class="id">{_e(check)}</code>')


def _html_table(header: Sequence[str], rows: Iterable[Sequence[Any]], *,
                numeric: Sequence[int] = (), code_col: Optional[int] = None,
                css: str = "") -> List[str]:
    head = "".join(f'<th{" class=" + chr(34) + "n" + chr(34) if i in numeric else ""}>{_e(h)}</th>'
                   for i, h in enumerate(header))
    out = [f'<div class="wrap"><table{f" class={chr(34)}{css}{chr(34)}" if css else ""}>',
           f"<thead><tr>{head}</tr></thead><tbody>"]
    for row in rows:
        cells = []
        for i, v in enumerate(row):
            cls = "n" if i in numeric else ("loc" if i == code_col else "")
            cell = v if isinstance(v, _Html) else _e(v)
            cells.append(f"<td{f' class={chr(34)}{cls}{chr(34)}' if cls else ''}>{cell}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    return out


def _html_header(eyebrow: str, title: str, sub: Optional[str],
                 meta: Sequence[Tuple[str, Any]]) -> List[str]:
    out = [f'<p class="eyebrow">{_e(eyebrow)}</p>', f"<h1>{_e(title)}</h1>"]
    if sub:
        out.append(f'<p class="sub">{_e(sub)}</p>')
    out.append('<dl class="meta">' + "".join(
        f"<div><dt>{_e(k)}</dt><dd>{_e(v)}</dd></div>" for k, v in meta if v not in (None, ""))
        + "</dl>")
    return out


def _html_verdict(counts: Mapping[str, int], library=None) -> List[str]:
    level, headline, detail = _verdict(counts, library)
    cls = level or "none"
    mark = (f'<span class="mark {level}">{_LEVEL_MARK[level]}</span>' if level
            else '<span class="mark" style="background:var(--ok)">\u2713</span>')
    return [f'<div class="verdict {cls}">{mark}<div><p class="big">{_e(headline)}</p>'
            f"<p>{_e(detail)}</p></div></div>"]


def _html_cards(counts: Mapping[str, int], tapes: Optional[Mapping[str, int]] = None) -> List[str]:
    out = ['<div class="cards">']
    for lv in _WORST_FIRST:
        n = counts.get(lv, 0)
        where = ""
        if tapes is not None and n:
            k = tapes[lv]
            where = f" · in {k} tape{'s' if k != 1 else ''}"
        out.append(f'<div class="card {lv}{" zero" if not n else ""}"><div class="n">{n}</div>'
                   f'<div class="lbl"><span class="mark">{_LEVEL_MARK[lv]}</span>'
                   f"{_LEVEL_PLURAL[lv]}</div>"
                   f'<div class="means">{_e(_LEVEL_MEANS[lv])}{_e(where)}</div></div>')
    out.append("</div>")
    return out


def _html_toc(items: Sequence[Tuple[str, str]]) -> List[str]:
    return ['<nav class="toc">' + "".join(f'<a href="#{a}">{_e(t)}</a>' for a, t in items)
            + "</nav>"]


def _html_summary(rows: Sequence[Dict[str, Any]], tapes: bool) -> List[str]:
    out = ['<h2 id="summary">Summary by check</h2>']
    if not rows:
        return out + ["<p>Nothing found.</p>"]
    header = ("Level", "Check", "MF", "Findings") + (("Tapes",) if tapes else ())
    body = [(_badge(r["level"]), _check_cell(r["check"]), _mf(r["mf"]), r["findings"])
            + ((r["tapes"],) if tapes else ()) for r in _display_rows(rows)]
    return out + _html_table(header, body, numeric=(3, 4) if tapes else (3,))


#: Rows of a group shown before the rest folds away: one check repeated on
#: every band of a section says the same thing twenty times.
_GROUP_ROWS = 6


def _html_groups(findings: Sequence[CovarianceFinding]) -> List[str]:
    out: List[str] = []
    for level, check, fs in _groups(findings):
        n = len(fs)
        out.append(f'<div class="group"><div class="ghead">{_badge(level)}'
                   f'<a class="ck" href="#check-{_e(check)}"><span class="ttl">'
                   f"{to_html_symbols(_title(check))}</span></a>"
                   f'<code class="id">{_e(check)}</code>'
                   f'<span class="cnt">{n} finding{"s" if n != 1 else ""}</span></div>')
        rows = [(str(f.location), _Html(to_html_symbols(f.summary))) for f in fs]
        shown = rows if n <= _GROUP_ROWS + 2 else rows[:_GROUP_ROWS]
        out += _html_table(("Location", "What was found"), shown, code_col=0, css="f")
        if len(shown) < n:
            out.append(f'<details class="more"><summary>Show the other {n - len(shown)}'
                       "</summary>")
            out += _html_table(("Location", "What was found"), rows[len(shown):],
                               code_col=0, css="f")
            out.append("</details>")
        out.append("</div>")
    return out


def _html_method() -> List[str]:
    out = ['<h2 id="method">Method and thresholds</h2>', '<div class="method">']
    for heading, paragraphs in method_sections():
        out.append(f"<h3>{_e(heading)}</h3>")
        out += [f"<p>{to_html_symbols(p)}</p>" for p in paragraphs]
    return out + ["</div>"]


def _html_legend(names: Sequence[str]) -> List[str]:
    entries = [(n, CHECKS[n]) for n in names if n in CHECKS]
    if not entries:
        return []
    out = ['<h2 id="legend">What each finding means</h2>',
           f'<p class="muted">{_e(_LEGEND_INTRO)}</p>', '<div class="legend">']
    for name, d in entries:
        out.append(f'<div class="entry" id="check-{_e(name)}"><h3>{to_html_symbols(d.title)}</h3>'
                   f'<code class="id">{_e(name)}</code> '
                   f'<span class="id">· MF {_e(", ".join(map(str, d.mf)))}</span><ul>')
        out += [f"<li>{_badge(lv)}<span>{to_html_symbols(d.levels[lv])}</span></li>"
                for lv in _WORST_FIRST if lv in d.levels]
        out += ["</ul>", f"<p>{to_html_symbols(d.description)}</p></div>"]
    return out + ["</div>"]


def _html_page(title: str, body: List[str]) -> str:
    return "\n".join([
        "<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(title)}</title>", f"<style>{_CSS}</style>", "</head>", "<body>",
        *body, "</body>", "</html>", ""])


_TOC = (("summary", "Summary"), ("findings", "Findings"), ("method", "Method"),
        ("legend", "What each finding means"))


def report_html(report, level: str = WARN) -> str:
    _RANK[level]
    tape = _tape_identity(report)
    counts = _level_counts(report.findings)
    name = tape["name"] or "ENDF tape"
    body = _html_header("kika · covariance check", name, tape["path"], [
        ("MAT", tape["mat"]), ("MF checked", ", ".join(map(str, tape["mf"]))),
        ("kika", kika_version()), ("Generated", _now())])
    body += _html_verdict(counts)
    body += _html_cards(counts)
    body += _html_toc(_TOC)
    rows = summary_rows(report.findings)
    body += _html_summary(rows, tapes=False)
    body.append('<h2 id="findings">Findings</h2>')
    listed = _findings_at(report.findings, level)
    body += _html_groups(listed)
    hidden = _hidden_line(counts, level)
    if hidden:
        body.append(f'<p class="muted">{_e(hidden)}</p>')
    elif not listed:
        body.append("<p>None.</p>")
    body += _html_method()
    body += _html_legend(_check_names(_display_rows(rows)))
    return _html_page(f"Covariance check of {name}", body)


def _html_tapes(library) -> List[str]:
    rows = []
    base = _base(library)
    for t in _tape_order(library):
        status = _tape_status(t)
        cls = "status-failed" if not t.ok else ("" if t.has_covariances else "status-none")
        # The full path on hover; under the name only where the name alone is not it.
        full = _e(Path(t.path).resolve())
        rel = _rel(t, base)
        rows.append((
            _Html(f'<span class="file" title="{full}">{_e(t.name)}</span>'
                  + (f'<span class="path">{_e(rel)}</span>' if rel != t.name else "")),
            t.mat, ", ".join(map(str, t.mf)),
            _pill(DEFECT, t.count(DEFECT)), _pill(WARN, t.count(WARN)), _pill(NOTE, t.count(NOTE)),
            _Html(f'<span class="{cls}">{_e(status.capitalize())}</span>')))
    return _html_table(("File", "MAT", "MF checked", "Defects", "Warnings", "Notes", "Status"),
                       rows, numeric=(1, 3, 4, 5))


def library_html(library, level: str = WARN) -> str:
    _RANK[level]
    counts = _library_counts(library)
    base = _base(library)
    title = _library_title(library)
    body = _html_header("kika · covariance check of a library", title,
                        str(base) if base else None, [
                            ("Tapes", _library_status(library, counts)),
                            ("kika", kika_version()), ("Generated", _now())])
    body += _html_verdict(counts["findings"], library)
    tapes_with = {lv: sum(1 for t in library.tapes if t.count(lv)) for lv in _WORST_FIRST}
    body += _html_cards(counts["findings"], tapes_with)
    body += _html_toc(_TOC[:1] + (("tapes", "Tapes"),) + _TOC[1:])
    rows = library_summary_rows(library)
    body += _html_summary(rows, tapes=True)
    body.append('<h2 id="tapes">Tapes</h2>')
    body += _html_tapes(library)
    body.append('<h2 id="findings">Findings</h2>')
    any_listed = False
    for t in _tape_order(library):
        if t.report is None:
            continue
        listed = _findings_at(t.report.findings, level)
        if not listed:
            continue
        any_listed = True
        pills = "".join(_pill(lv, t.count(lv)) for lv in _WORST_FIRST)
        mat = f'<span class="muted">MAT {_e(t.mat)}</span>' if t.mat is not None else ""
        body.append(f'<details class="tape" open><summary><span class="tname">'
                    f"{_e(_rel(t, base))}</span>{mat}"
                    f'<span class="pills">{pills}</span></summary><div class="tbody">')
        body += _html_groups(listed)
        body.append("</div></details>")
    hidden = _hidden_line(counts["findings"], level)
    if hidden:
        body.append(f'<p class="muted">{_e(hidden)}</p>')
    elif not any_listed:
        body.append("<p>None.</p>")
    body += _html_method()
    body += _html_legend(_check_names(_display_rows(rows)))
    return _html_page(f"Covariance check of {title}", body)


def write_text(text: str, path) -> str:
    """Write *text* as UTF-8 to *path* when one is given; return *text*."""
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
