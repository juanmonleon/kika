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


def _tally(counts: Dict[str, int]) -> str:
    parts = [f"{n} {_LEVEL_WORD[level]}" + ("s" if n != 1 else "")
             for level, n in counts.items() if n]
    return ", ".join(parts) if parts else "nothing found"


def _hidden_line(counts: Dict[str, int], level: str) -> Optional[str]:
    hidden = {lv: n for lv, n in counts.items() if n and _RANK[lv] < _RANK[level]}
    if not hidden:
        return None
    return (f"{_tally(hidden)} not listed (counted in the summary; "
            f"level='{min(hidden, key=_RANK.get)}' lists them).")


class _Html(str):
    """A cell already rendered as HTML: not escaped again."""


def _finding_rows(findings: Sequence[CovarianceFinding], html_: bool = False) -> List[Tuple]:
    text = (lambda t: _Html(to_html_symbols(t))) if html_ else to_symbols
    return [(f.level, str(f.location), f.check, text(f.summary)) for f in findings]


_FINDING_HEADER = ("Level", "Location", "Check", "Summary")


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
        out += [f"### `{name}`: {to_symbols(d.title)}", "",
                "MF " + ", ".join(map(str, d.mf)), ""]
        out += [f"- **{lv}**: {to_symbols(d.levels[lv])}"
                for lv in (DEFECT, WARN, NOTE) if lv in d.levels]
        out += ["", to_symbols(d.description), ""]
    return out


def _html_legend(names: Sequence[str]) -> List[str]:
    entries = [(n, CHECKS[n]) for n in names if n in CHECKS]
    if not entries:
        return []
    out = ["<h2>What each finding means</h2>", f'<p class="muted">{_e(_LEGEND_INTRO)}</p>']
    for name, d in entries:
        out.append(f'<h3 id="check-{_e(name)}"><code>{_e(name)}</code>: {to_html_symbols(d.title)}</h3>')
        out.append(f'<p class="muted">MF {_e(", ".join(map(str, d.mf)))}</p><ul>')
        out += [f'<li><span class="{lv}">{lv}</span>: {to_html_symbols(d.levels[lv])}</li>'
                for lv in (DEFECT, WARN, NOTE) if lv in d.levels]
        out += ["</ul>", f"<p>{to_html_symbols(d.description)}</p>"]
    return out


def report_markdown(report, level: str = WARN) -> str:
    _RANK[level]
    tape = _tape_identity(report)
    counts = _level_counts(report.findings)
    title = tape["name"] or "ENDF tape"
    out = [f"# Covariance check of {title}", ""]
    out += _md_table(("", ""), [
        ("Tape", tape["name"]), ("Path", tape["path"]), ("MAT", tape["mat"]),
        ("MF checked", ", ".join(map(str, tape["mf"]))),
        ("kika", kika_version()), ("Generated", _now()), ("Result", _tally(counts))])
    out += ["", "## Summary", ""]
    rows = summary_rows(report.findings)
    if rows:
        out += _md_table(("MF", "Level", "Check", "Findings"),
                         [(_mf(r["mf"]), r["level"], r["check"], r["findings"]) for r in rows])
    else:
        out.append("Nothing found.")
    listed = _findings_at(report.findings, level)
    out += ["", "## Findings", ""]
    if listed:
        out += _md_table(_FINDING_HEADER, _finding_rows(listed))
        out.append("")
    hidden = _hidden_line(counts, level)
    if hidden:
        out += [hidden, ""]
    elif not listed:
        out += ["None.", ""]
    out += _md_method()
    out += _md_legend(_check_names(rows))
    return "\n".join(out).rstrip() + "\n"


def library_markdown(library, level: str = WARN) -> str:
    _RANK[level]
    counts = _library_counts(library)
    where = library.library or (str(library.directory) if library.directory else "a library")
    out = [f"# Covariance check of {where}", ""]
    status = (f"{counts['tapes']} tapes, {counts['checked']} with covariances checked, "
              f"{counts['with_defects']} with defects, {counts['failed']} failed")
    if library.stopped:
        status += f"; stopped after {counts['tapes']} of {counts['planned']}"
    out += _md_table(("", ""), [
        ("Library", library.library), ("Directory", _base(library)),
        ("Tapes", status), ("Findings", _tally(counts["findings"])),
        ("kika", kika_version()), ("Generated", _now())])
    out += ["", "## Summary", ""]
    rows = library_summary_rows(library)
    if rows:
        out += _md_table(("MF", "Level", "Check", "Findings", "Tapes"),
                         [(_mf(r["mf"]), r["level"], r["check"], r["findings"], r["tapes"])
                          for r in rows])
    else:
        out.append("Nothing found.")
    out += ["", "## Tapes", ""]
    out += _md_table(_TAPE_HEADER, _tape_rows(library))
    out += ["", "## Findings", ""]
    base = _base(library)
    any_listed = False
    for t in library.tapes:
        if t.report is None:
            continue
        listed = _findings_at(t.report.findings, level)
        if not listed:
            continue
        any_listed = True
        out += [f"### {_rel(t, base)}" + (f" (MAT {t.mat})" if t.mat is not None else ""), ""]
        out += _md_table(_FINDING_HEADER, _finding_rows(listed))
        out.append("")
    hidden = _hidden_line(counts["findings"], level)
    if hidden:
        out += [hidden, ""]
    elif not any_listed:
        out += ["None.", ""]
    out += _md_method()
    out += _md_legend(_check_names(rows))
    return "\n".join(out).rstrip() + "\n"


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


def _tape_rows(library) -> List[Tuple]:
    return [(t.name, t.mat, ", ".join(map(str, t.mf)), t.count(DEFECT), t.count(WARN),
             t.count(NOTE), _tape_status(t), str(Path(t.path).resolve()))
            for t in library.tapes]


_TAPE_HEADER = ("File", "MAT", "MF checked", "Defects", "Warnings", "Notes", "Status", "Path")


def _tape_status(tape) -> str:
    if not tape.ok:
        return f"failed: {tape.error}"
    return "checked" if tape.has_covariances else "no covariances"


_CSS = """
:root{--fg:#1d1f23;--muted:#5b616b;--bg:#fff;--line:#d9dce1;--head:#f3f4f6;
--defect:#b42318;--warn:#a15c00;--note:#475467}
@media (prefers-color-scheme:dark){:root{--fg:#e6e7ea;--muted:#a0a6b0;--bg:#16181c;
--line:#33373e;--head:#1f2228;--defect:#ff7a6e;--warn:#f0b450;--note:#a6b0bf}}
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--fg);
background:var(--bg);max-width:1200px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 12px}h2{font-size:18px;margin:28px 0 8px;
border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:15px;margin:18px 0 6px}
table{border-collapse:collapse;width:100%;margin:6px 0 12px}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
th{background:var(--head)}td.n{text-align:right;font-variant-numeric:tabular-nums}
table.id{width:auto}table.id th{font-weight:600}
.defect{color:var(--defect);font-weight:600}.warn{color:var(--warn);font-weight:600}
.note{color:var(--note)}code,.loc{font-family:ui-monospace,Consolas,monospace;font-size:13px}
p.muted{color:var(--muted)}.wrap{overflow-x:auto}
"""


def _e(value: Any) -> str:
    return "" if value is None else html.escape(str(value))


def _html_table(header: Sequence[str], rows: Iterable[Sequence[Any]], *,
                numeric: Sequence[int] = (), level_col: Optional[int] = None,
                code_col: Optional[int] = None, css: str = "") -> List[str]:
    out = [f'<div class="wrap"><table{f" class={chr(34)}{css}{chr(34)}" if css else ""}>',
           "<thead><tr>" + "".join(f"<th>{_e(h)}</th>" for h in header) + "</tr></thead><tbody>"]
    for row in rows:
        cells = []
        for i, v in enumerate(row):
            attrs = []
            if i in numeric:
                attrs.append('class="n"')
            elif i == level_col:
                attrs.append(f'class="{_e(v)}"')
            elif i == code_col:
                attrs.append('class="loc"')
            cell = v if isinstance(v, _Html) else _e(v)
            cells.append(f"<td{' ' + ' '.join(attrs) if attrs else ''}>{cell}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    return out


def _id_table(rows: Sequence[Tuple[str, Any]]) -> List[str]:
    out = ['<table class="id"><tbody>']
    out += [f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in rows]
    out.append("</tbody></table>")
    return out


def _html_method() -> List[str]:
    out = ["<h2>Method and thresholds</h2>"]
    for heading, paragraphs in method_sections():
        out.append(f"<h3>{_e(heading)}</h3>")
        out += [f"<p>{to_html_symbols(p)}</p>" for p in paragraphs]
    return out


def _html_page(title: str, body: List[str]) -> str:
    return "\n".join([
        "<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(title)}</title>", f"<style>{_CSS}</style>", "</head>", "<body>",
        *body, "</body>", "</html>", ""])


def _html_findings(findings: Sequence[CovarianceFinding]) -> List[str]:
    return _html_table(_FINDING_HEADER, _finding_rows(findings, html_=True),
                       level_col=0, code_col=1)


def report_html(report, level: str = WARN) -> str:
    _RANK[level]
    tape = _tape_identity(report)
    counts = _level_counts(report.findings)
    title = f"Covariance check of {tape['name'] or 'ENDF tape'}"
    body = [f"<h1>{_e(title)}</h1>"]
    body += _id_table([
        ("Tape", tape["name"]), ("Path", tape["path"]), ("MAT", tape["mat"]),
        ("MF checked", ", ".join(map(str, tape["mf"]))),
        ("kika", kika_version()), ("Generated", _now()), ("Result", _tally(counts))])
    body.append("<h2>Summary</h2>")
    rows = summary_rows(report.findings)
    if rows:
        body += _html_table(("MF", "Level", "Check", "Findings"),
                            [(_mf(r["mf"]), r["level"], r["check"], r["findings"])
                             for r in rows], numeric=(3,), level_col=1)
    else:
        body.append("<p>Nothing found.</p>")
    body.append("<h2>Findings</h2>")
    listed = _findings_at(report.findings, level)
    if listed:
        body += _html_findings(listed)
    hidden = _hidden_line(counts, level)
    if hidden:
        body.append(f'<p class="muted">{_e(hidden)}</p>')
    elif not listed:
        body.append("<p>None.</p>")
    body += _html_method()
    body += _html_legend(_check_names(rows))
    return _html_page(title, body)


def library_html(library, level: str = WARN) -> str:
    _RANK[level]
    counts = _library_counts(library)
    where = library.library or (str(library.directory) if library.directory else "a library")
    title = f"Covariance check of {where}"
    status = (f"{counts['tapes']} tapes, {counts['checked']} with covariances checked, "
              f"{counts['with_defects']} with defects, {counts['failed']} failed")
    if library.stopped:
        status += f"; stopped after {counts['tapes']} of {counts['planned']}"
    body = [f"<h1>{_e(title)}</h1>"]
    body += _id_table([
        ("Library", library.library), ("Directory", _base(library)), ("Tapes", status),
        ("Findings", _tally(counts["findings"])), ("kika", kika_version()),
        ("Generated", _now())])
    body.append("<h2>Summary</h2>")
    rows = library_summary_rows(library)
    if rows:
        body += _html_table(("MF", "Level", "Check", "Findings", "Tapes"),
                            [(_mf(r["mf"]), r["level"], r["check"], r["findings"], r["tapes"])
                             for r in rows], numeric=(3, 4), level_col=1)
    else:
        body.append("<p>Nothing found.</p>")
    body.append("<h2>Tapes</h2>")
    body += _html_table(_TAPE_HEADER, _tape_rows(library), numeric=(1, 3, 4, 5), code_col=7)
    body.append("<h2>Findings</h2>")
    base = _base(library)
    any_listed = False
    for t in library.tapes:
        if t.report is None:
            continue
        listed = _findings_at(t.report.findings, level)
        if not listed:
            continue
        any_listed = True
        body.append(f"<h3>{_e(_rel(t, base))}"
                    + (f" (MAT {_e(t.mat)})" if t.mat is not None else "")
                    + "</h3>")
        body += _html_findings(listed)
    hidden = _hidden_line(counts["findings"], level)
    if hidden:
        body.append(f'<p class="muted">{_e(hidden)}</p>')
    elif not any_listed:
        body.append("<p>None.</p>")
    body += _html_method()
    body += _html_legend(_check_names(rows))
    return _html_page(title, body)


def write_text(text: str, path) -> str:
    """Write *text* as UTF-8 to *path* when one is given; return *text*."""
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
