"""The report of a perturbation run (``report.html`` beside the samples).

Built from the run's events alone (:class:`~kika.sampling.run_log.RunLog`), so a
``run.log.jsonl`` read back from disk gives the same report. It answers, in this
order: did it work; if not, where it stopped and how to go on; what was checked;
what the user has to know (warnings and notes, one entry per cause); what was
perturbed and by how much; how it was drawn; the samples and the files; and,
folded at the end, the run's details and every event.

Statuses follow the processing report: **Failed** is a result not to use or a
file not written; **Warning** can change a result and needs the user's
judgement; **Note** is something kika resolved in a fixed, documented way.
"""
from __future__ import annotations

import datetime as _dt
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["PerturbationReport", "SCHEMA"]

SCHEMA = "kika-perturbation-report/1"

#: The run's stages in order: (id, title, the event kinds that complete it).
STAGES = (
    ("read", "Read", ("read",)),
    ("reconstructed", "Resonance region", ("reconstructed",)),
    ("request", "Request", ("request",)),
    ("assembled", "Covariance", ("assembled",)),
    ("inspected", "Pre-flight", ("inspected", "conditioned")),
    ("drawn", "Draw", ("drawn",)),
    ("applied", "Samples", ("applied", "checked")),
    ("files", "Files", ("emitted", "written")),
)
#: An error's stage (an event kind) to the stage that shows it.
_STAGE_OF_KIND = {"read": "read", "reconstructed": "reconstructed", "request": "request",
                  "assembled": "assembled", "inspected": "inspected",
                  "conditioned": "inspected", "drawn": "drawn", "applied": "applied",
                  "checked": "applied", "emitted": "applied", "written": "files"}

QUANTITY = {31: "neutron multiplicity", 33: "cross section", 34: "angular distribution",
            35: "fission spectrum"}
QUANTITIES = {31: "Neutron multiplicities", 33: "Cross sections", 34: "Angular distributions",
              35: "Fission spectra"}

#: Finding codes the pipeline records, to the title a person reads. A note or
#: warning without a code is shown with its own message as the title.
TITLES = {
    "resonance_background_only": "Only the background moved in the resonance region",
    "not_in_evaluation": "Some stated uncertainties were not used",
    "lumped_covariance": "A lumped covariance moves several reactions",
    "remainder_covariance": "A covariance stated as a difference",
    "sum_statement": "How the file states a total's covariance",
    "sum_block": "A total's own covariance",
    "sums_rederived": "Totals were rebuilt from their parts",
    "sum_rule": "A multiplicity was rebuilt from its sum rule",
    "spectrum_coverage": "Part of the fission spectrum is not perturbed",
    "dropped_steps": "Some spectrum factors were not written",
    "point_band": "A fission-spectrum band stated at one energy",
    "tabulated_angular": "Tabulated angular distributions",
    "factor_not_positive": "Some factors are zero or negative",
    "covariance_from_another_file": "The covariance comes from another file",
    "mat_from_table": "The MAT number was taken from kika's table",
}
#: Codes about how the run was executed, not about the samples: details only.
_MECHANICS = {"workers"}

MARK = {"pass": "✓", "note": "i", "warn": "!", "fail": "✕", "skip": "–"}
LABEL = {"pass": "Passed", "note": "Note", "warn": "Warning", "fail": "Failed", "skip": "Not run"}
_TEXT_MARK = {"pass": "ok", "note": "note", "warn": "WARN", "fail": "FAIL", "skip": "--"}

#: The ensemble check needs this many samples to say anything.
ENSEMBLE_MIN_SAMPLES = 30
#: Above this many samples the table lists only the flagged ones.
SAMPLE_ROWS = 50

_COMPONENT = re.compile(r"MF(\d+)/MT(\d+)(?:\s*\(ZA\s*(\d+)(?:,\s*([^)]*))?\))?")


# --------------------------------------------------------------------------- small helpers

def _esc(text) -> str:
    return html.escape("" if text is None else str(text))


def _size(nbytes) -> str:
    if nbytes is None:
        return ""
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def _secs(seconds) -> str:
    if seconds is None:
        return ""
    return f"{seconds:.2f} s" if seconds < 10 else f"{seconds:.1f} s"


def _plural(n: int, word: str, many: Optional[str] = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


def _reaction(mt: int) -> str:
    from kika._constants import MT_TO_REACTION
    from kika.endf.reconstruction_report import reaction_name

    return reaction_name(mt, MT_TO_REACTION) or ""


def _component(subject: Optional[str]) -> Optional[Dict[str, Any]]:
    """``MF33/MT2 (ZA 26054)`` as its parts; None for anything else."""
    match = _COMPONENT.match(subject or "")
    if not match:
        return None
    mf, mt = int(match[1]), int(match[2])
    return {"mf": mf, "mt": mt, "za": int(match[3]) if match[3] else None,
            "detail": (match[4] or "").strip(), "name": _reaction(mt),
            "quantity": QUANTITY.get(mf, f"MF{mf}")}


def _ranges(numbers: Sequence[int]) -> str:
    """``[1, 2, 3, 5]`` -> ``1-3, 5``."""
    values = sorted(set(int(n) for n in numbers))
    out, start = [], None
    for i, n in enumerate(values):
        if start is None:
            start = n
        if i + 1 == len(values) or values[i + 1] != n + 1:
            out.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ", ".join(out)


def _first_sentence(text: str, limit: int = 160) -> str:
    line = (text or "").strip().splitlines()[0] if text else ""
    cut = re.split(r"(?<=[.;])\s", line, maxsplit=1)[0]
    return cut if len(cut) <= limit else cut[:limit - 1].rstrip() + "…"


def _stamp(at: Optional[str]) -> str:
    if not at:
        return ""
    try:
        moment = _dt.datetime.fromisoformat(at).astimezone(_dt.timezone.utc)
    except ValueError:
        return at
    return moment.strftime("%Y-%m-%d %H:%M UTC")


# --------------------------------------------------------------------------- the report

class PerturbationReport:
    """A perturbation run's report. ``to_html()``, ``to_text()``, ``to_dict()``, ``save()``."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @classmethod
    def from_log(cls, log) -> "PerturbationReport":
        return cls(build([event.to_dict() for event in log]))

    @classmethod
    def from_events(cls, events: Sequence[Dict[str, Any]]) -> "PerturbationReport":
        return cls(build(list(events)))

    @classmethod
    def read(cls, path) -> "PerturbationReport":
        """From a ``run.log.jsonl``, or the run directory holding one."""
        path = Path(path)
        if path.is_dir():
            path = path / "run.log.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        return cls.from_events(events)

    @property
    def status(self) -> str:
        return self._data["summary"]["status"]

    @property
    def headline(self) -> str:
        return self._data["summary"]["headline"]

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._data))

    def to_json(self, **kwargs) -> str:
        return json.dumps(self._data, **({"indent": 1} | kwargs))

    def to_text(self) -> str:
        return _render_text(self._data)

    def to_html(self) -> str:
        return _render_html(self._data)

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def __repr__(self) -> str:
        return f"<PerturbationReport: {self.headline}>"

    def _repr_html_(self) -> str:
        return _render_html(self._data)


# --------------------------------------------------------------------------- build

def build(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The report's data from a run's events (``Event.to_dict`` form)."""
    ev = _Events(events)
    started = ev.first("started")
    options = dict(started["payload"]) if started else {}
    dry = bool(options.get("dryRun"))
    n = int(options.get("nSamples") or 0)
    error = ev.first("error")
    cancelled = ev.first("cancelled")
    finished = ev.first("finished")

    header = _header(ev, options)
    failed_stage = None
    if error is not None:
        kind = (error["payload"] or {}).get("stage")
        failed_stage = _STAGE_OF_KIND.get(kind) or ("applied" if error.get("sample") is not None
                                                     else "read")
    stages = _stages(ev, options, failed_stage, cancelled is not None, finished is not None)
    findings = _findings(ev)
    checks = _checks(ev, options, failed_stage)
    perturbed = _perturbed(ev)
    samples = _samples(ev, n)
    files = _files(ev, options, finished is not None or error is not None or cancelled is not None)
    failure = _failure(ev, error, failed_stage) if error is not None else None

    warnings = sum(1 for f in findings if f["status"] == "warn")
    notes = sum(1 for f in findings if f["status"] == "note")
    written = len({e["sample"] for e in ev.of("emitted")})
    if error is not None and finished is not None:
        errors = ev.of("error")
        status = "fail"
        headline = f"Completed with failures: {_plural(len(errors), 'error')}"
        lead = (_success_lead(ev, options, header, warnings)
                + " Some outputs were not written: see What went wrong.")
    elif error is not None:
        status = "fail"
        headline = f"Failed at {_stage_title(failed_stage).lower()}: {failure['short']}"
        lead = ("No samples were written. " if not written else
                f"{_plural(written, 'sample')} of {n} were written before it stopped. ") \
            + failure["lead"]
    elif cancelled is not None:
        status = "warn"
        headline = f"Stopped: {written} of {_plural(n, 'sample')} written"
        lead = (f"The run was stopped on request. The samples already written are kept and "
                f"listed below; the rest were not drawn to disk.")
    elif finished is None:
        status = "fail"
        last = ev.events[-1]["kind"] if ev.events else "nothing"
        headline = "Did not finish"
        lead = (f"The log ends after '{last}' with no error recorded: the process was "
                f"probably killed. The samples listed below are the ones written.")
    else:
        status = "warn" if warnings else "pass"
        what = "Rehearsal completed" if dry else f"Completed: {_plural(n, 'sample')} written"
        headline = what + (f", with {_plural(warnings, 'warning')}" if warnings else "")
        lead = _success_lead(ev, options, header, warnings)

    return {
        "schema": SCHEMA,
        "header": header,
        "summary": {"status": status, "headline": headline, "lead": lead,
                    "warnings": warnings, "notes": notes, "dryRun": dry,
                    "samples": n, "written": written},
        "stages": stages,
        "failure": failure,
        "checks": checks,
        "findings": findings,
        "perturbed": perturbed,
        "drawing": _drawing(ev, options) if ev.has("drawn") else [],
        "samples": samples,
        "steps": _steps(ev, options, failed_stage),
        "files": files,
        "details": _details(ev, options, header),
        "events": events,
    }


class _Events:
    def __init__(self, events):
        self.events = events

    def of(self, kind, *, subject: Optional[bool] = None):
        out = [e for e in self.events if e["kind"] == kind]
        if subject is True:
            out = [e for e in out if e.get("subject")]
        elif subject is False:
            out = [e for e in out if not e.get("subject")]
        return out

    def first(self, kind, **kw):
        found = self.of(kind, **kw)
        return found[0] if found else None

    def has(self, kind) -> bool:
        return any(e["kind"] == kind for e in self.events)


def _stage_title(stage_id: Optional[str]) -> str:
    return next((title for sid, title, _ in STAGES if sid == stage_id), "the start")


def _header(ev, options) -> Dict[str, Any]:
    read = ev.first("read")
    payload = (read or {}).get("payload") or {}
    evaluation = payload.get("evaluation") or {}
    source = options.get("source") or ""
    file = evaluation.get("file") or (read or {}).get("subject") or Path(source).name or "-"
    nuclide = evaluation.get("nuclide") or Path(file).stem
    started = ev.first("started")
    return {
        "nuclide": nuclide, "file": file, "path": evaluation.get("path") or source,
        "za": evaluation.get("za"), "mat": evaluation.get("mat"),
        "library": evaluation.get("library"), "sha256": evaluation.get("sha256"),
        "format": payload.get("format"), "kika": options.get("kikaVersion"),
        "started": _stamp(started["at"]) if started else "",
        "outputDir": options.get("outputDir"), "formats": list(options.get("formats") or []),
        "nSamples": options.get("nSamples"), "dryRun": bool(options.get("dryRun")),
        "covarianceSource": options.get("covarianceSource"),
    }


def _quantities(ev) -> List[int]:
    mfs = set()
    for e in ev.of("checked", subject=True):
        comp = _component(e["subject"])
        if comp:
            mfs.add(comp["mf"])
    if not mfs:
        for e in ev.of("assembled", subject=True):
            for c in e["payload"].get("components") or []:
                comp = _component(c)
                if comp:
                    mfs.add(comp["mf"])
    return sorted(mfs)


def _formats_text(formats: Sequence[str]) -> str:
    names = {"endf-delta": "ENDF tapes (changed sections only)", "endf-tape": "whole ENDF tapes",
             "gnds": "GNDS", "ace": "ACE", "hdf5": "OpenMC HDF5"}
    return ", ".join(names.get(f, f) for f in formats)


def _success_lead(ev, options, header, warnings) -> str:
    mfs = _quantities(ev)
    what = " and ".join(QUANTITIES.get(mf, f"MF{mf}").lower() for mf in mfs) or "the request"
    what = what[0].upper() + what[1:]
    drawn = {e["subject"] for e in ev.of("checked", subject=True)}
    blocks = len(ev.of("drawn", subject=True))
    moved = _also_moved(ev, drawn)
    source = ("the covariance of another file" if options.get("covarianceSource")
              else "the uncertainties the evaluation states")
    text = (f"{what} of {header['nuclide']} perturbed from {source}: "
            f"{_plural(len(drawn), 'component')} drawn in {_plural(blocks, 'block')}")
    text += f", and {len(moved)} more moved through them. " if moved else ". "
    if options.get("dryRun"):
        text += ("Rehearsal: everything but the files of the samples. The factors, the plan "
                 "and this report are written.")
    else:
        sizes = [e["payload"].get("bytes") or 0 for e in ev.of("emitted")]
        n = int(options.get("nSamples") or 0)
        text += f"{_plural(n, 'sample')} written as {_formats_text(options.get('formats') or [])}"
        text += f" ({_size(sum(sizes))} in all)." if sizes else "."
    if warnings:
        text += " Read the warnings before using the samples."
    return text


def _stages(ev, options, failed_stage, cancelled, finished=False) -> List[Dict[str, Any]]:
    dry = bool(options.get("dryRun"))
    resonance = ev.has("reconstructed") or failed_stage == "reconstructed" or any(
        (e["payload"] or {}).get("code") == "resonance_background_only" for e in ev.of("warning"))
    out, after_failure = [], False
    for sid, title, kinds in STAGES:
        if sid == "reconstructed" and not resonance:
            continue
        reached = any(ev.has(kind) for kind in kinds)
        detail = ""
        if after_failure:
            status = "skip"
        elif sid == failed_stage:
            status, after_failure = "fail", not finished
        elif sid == "reconstructed":
            status = "pass" if ev.has("reconstructed") else "warn"
            detail = "rebuilt by NJOY" if ev.has("reconstructed") else "background only"
        elif sid == "files" and dry:
            status, detail = "skip", "rehearsal"
        elif not reached:
            status = "skip"
        else:
            status = "pass"
        if status in ("pass", "warn"):
            detail = detail or _stage_detail(ev, sid, options)
            if sid == "inspected" and _repairs(ev):
                status = "warn"
            if sid == "applied" and any(e.get("sample") is not None for e in ev.events
                                        if e["level"] != "info"):
                status = "warn"
        if sid == failed_stage:
            detail = "errors" if finished else "stopped here"
        out.append({"id": sid, "title": title, "status": status, "detail": detail})
    if cancelled:
        for stage in out:
            if stage["status"] == "skip" and stage["id"] in ("applied", "files"):
                stage.update(status="warn", detail="stopped")
    return out


def _stage_detail(ev, sid, options) -> str:
    if sid == "read":
        read = ev.first("read")
        return _secs(read.get("seconds")) if read else ""
    if sid == "request":
        return _plural(len(_drawn_components(ev)), "component")
    if sid == "assembled":
        blocks = ev.of("assembled", subject=True)
        if not blocks:
            return ""
        largest = max(int(b["payload"].get("dimension") or 0) for b in blocks)
        return f"{_plural(len(blocks), 'block')}, largest {largest}×{largest}"
    if sid == "inspected":
        repairs = _repairs(ev)
        return _plural(len(repairs), "repair") if repairs else "as stated"
    if sid == "drawn":
        head = ev.first("drawn", subject=False)
        return _secs(head.get("seconds")) if head else ""
    if sid == "applied":
        done = len({e["sample"] for e in ev.of("applied") if e.get("sample") is not None})
        return f"{done} of {options.get('nSamples')}"
    if sid == "files":
        total = sum(e["payload"].get("bytes") or 0 for e in ev.of("emitted"))
        return _size(total) if total else ""
    return ""


def _drawn_components(ev) -> List[str]:
    out: List[str] = []
    for e in ev.of("assembled", subject=True):
        for c in e["payload"].get("components") or []:
            if c not in out:
                out.append(c)
    return out


def _repairs(ev) -> List[Dict[str, Any]]:
    return [e for e in ev.of("conditioned", subject=True)
            if (e["payload"] or {}).get("changed")]


# --------------------------------------------------------------------------- checks

def _checks(ev, options, failed_stage) -> List[Dict[str, str]]:
    n = int(options.get("nSamples") or 0)
    dry = bool(options.get("dryRun"))
    out = []

    checked = ev.of("checked", subject=True)
    bad = [e for e in ev.of("warning") if (e["payload"] or {}).get("code") == "factor_not_positive"]
    if not checked:
        out.append(("skip", "Every factor is finite and positive", "No samples were drawn."))
    elif bad:
        samples = sorted({e["sample"] for e in bad if e.get("sample") is not None})
        out.append(("warn", "Every factor is finite and positive",
                    f"{_plural(len(samples), 'sample')} hold a factor at or below zero "
                    f"(samples {_ranges(samples)}). A cross section or a coefficient scaled by it "
                    f"changes sign."))
    else:
        worst = min(checked, key=lambda e: e["payload"].get("min", 1.0))
        comp = _component(worst["subject"]) or {}
        top = max(e["payload"].get("max", 1.0) for e in checked)
        values = sum(int(e["payload"].get("n") or 0) for e in checked)
        out.append(("pass", "Every factor is finite and positive",
                    f"{values:,} factors over {_plural(len({e['sample'] for e in checked}), 'sample')}. "
                    f"Smallest {worst['payload'].get('min', 0):.3f} ({_who(comp)}, sample "
                    f"{worst['sample']}); largest {top:.3f}."))

    repairs = _repairs(ev)
    head = ev.first("inspected", subject=False)
    title = "The covariance can be drawn as stated"
    if failed_stage == "inspected":
        out.append(("fail", title, _esc_plain((head or {}).get("payload", {}).get("summary")
                                             or "The pre-flight found a block it cannot draw.")
                    + ". No repair applies, so nothing was drawn."))
    elif repairs:
        moved = max((r["payload"].get("stated_diagonal_max_relative_change") or 0.0)
                    for r in repairs)
        remedies = sorted({r["payload"].get("remedy") for r in repairs if r["payload"].get("remedy")})
        status = "warn" if moved > 0.01 else "note"
        out.append((status, title,
                    f"{_plural(len(repairs), 'block')} repaired before the draw "
                    f"({', '.join(_REMEDY.get(r, r) for r in remedies)}); the stated variances "
                    f"moved by up to {moved:.1%}."))
    elif head is not None:
        mode = (ev.first("conditioned", subject=False) or {}).get("payload", {}).get("mode")
        out.append(("pass", title, "Every block is drawn exactly as the file states it."
                    + (" Repairs were switched off for this run." if mode == "none" else "")))
    else:
        out.append(("skip", title, "The run stopped before the pre-flight."))

    title = "Every sample was written"
    formats = list(options.get("formats") or [])
    if dry:
        out.append(("skip", title, "Rehearsal: no sample files are written."))
    elif failed_stage == "applied" or (failed_stage == "files"):
        written = len({e["sample"] for e in ev.of("emitted")})
        out.append(("fail", title, f"{written} of {n} samples written before the run stopped."))
    elif not ev.has("emitted"):
        out.append(("skip", title, "No samples were drawn."))
    else:
        per = {}
        for e in ev.of("emitted"):
            per.setdefault(e["sample"], set()).add(e["subject"])
        complete = [s for s, fmts in per.items() if set(formats) <= fmts]
        status = "pass" if len(complete) == n else "fail"
        out.append((status, title, f"{len(complete)} of {n} samples with every output asked for "
                    f"({_formats_text(formats)})."))

    title = "The ensemble reproduces the stated uncertainty"
    if n < ENSEMBLE_MIN_SAMPLES:
        out.append(("skip", title, f"{_plural(n, 'sample')} are too few to judge a spread "
                    f"(needs {ENSEMBLE_MIN_SAMPLES} or more)."))
    else:
        out.append(("skip", title, "Not measured by this version of kika."))
    return [{"status": s, "title": t, "result": r} for s, t, r in out]


_REMEDY = {"clip": "negative eigenvalues clipped to zero",
           "clip_rescale": "negative eigenvalues clipped, variances restored",
           "higham": "nearest positive semidefinite matrix (Higham)",
           "cap": "correlations capped", "mask_inert": "rows with no variance left out",
           "rescale_to_family_median": "rescaled to the median of its family"}


def _esc_plain(text) -> str:
    return "" if text is None else str(text)


def _who(comp: Dict[str, Any]) -> str:
    if not comp:
        return "-"
    name = comp.get("name")
    label = f"MT{comp['mt']}" + (f" {name}" if name else "")
    if comp.get("mf") not in (33, None):
        label = f"{comp['quantity']} {label}"
    if comp.get("detail"):
        label += f", {comp['detail']}"
    return label


# --------------------------------------------------------------------------- findings

def _findings(ev) -> List[Dict[str, Any]]:
    groups: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    for e in ev.events:
        if e["kind"] not in ("warning", "note"):
            continue
        payload = e["payload"] or {}
        code = payload.get("code")
        if code in _MECHANICS:
            continue
        key = code or e["message"]
        if key not in groups:
            order.append(key)
            groups[key] = {"status": "warn" if e["level"] == "warning" else "note",
                           "title": TITLES.get(code) or _first_sentence(e["message"]),
                           "code": code, "messages": [], "samples": [], "subjects": []}
        group = groups[key]
        if e["level"] == "warning":
            group["status"] = "warn"
        if e["message"] not in group["messages"]:
            group["messages"].append(e["message"])
        if e.get("sample") is not None:
            group["samples"].append(e["sample"])
        if e.get("subject") and e["subject"] not in group["subjects"]:
            group["subjects"].append(e["subject"])

    repairs = _repairs(ev)
    if repairs:
        moved = max((r["payload"].get("stated_diagonal_max_relative_change") or 0.0)
                    for r in repairs)
        attribution = [e["message"] for e in ev.of("inspected", subject=True)
                       if (e["payload"] or {}).get("layer1") is not None]
        items = []
        for r in repairs:
            p = r["payload"]
            items.append(f"{r['subject']}: {_REMEDY.get(p.get('remedy'), p.get('remedy'))}; "
                         f"stated variances moved by up to "
                         f"{(p.get('stated_diagonal_max_relative_change') or 0.0):.1%}. "
                         f"Why: {p.get('reason', '')}")
        key = "__repairs__"
        order.insert(0, key)
        groups[key] = {
            "status": "warn" if moved > 0.01 else "note",
            "title": "The covariance was repaired before the draw", "code": "covariance_repaired",
            "messages": [
                "A block that is not positive semidefinite cannot be drawn as stated. kika applied "
                "the smallest repair that makes it drawable, so the samples follow the repaired "
                "covariance. The repair is saved as conditioning_plan.json: edit it and pass it "
                "back to change it."],
            "samples": [], "subjects": [r["subject"] for r in repairs],
            "items": items + attribution}

    out = []
    for key in order:
        g = groups[key]
        samples = sorted(set(g["samples"]))
        scope = ""
        if samples:
            scope = f"sample {samples[0]}" if len(samples) == 1 else \
                f"{len(samples)} samples ({_ranges(samples)})"
        elif g["subjects"]:
            scope = ", ".join(g["subjects"][:3]) + (" …" if len(g["subjects"]) > 3 else "")
        out.append({"status": g["status"], "title": g["title"], "code": g["code"],
                    "scope": scope, "body": g["messages"][0],
                    "items": g.get("items") or g["messages"][1:]})
    out.sort(key=lambda f: 0 if f["status"] == "warn" else 1)
    return out


# --------------------------------------------------------------------------- perturbed

def _perturbed(ev) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Any]] = {}
    for e in ev.of("checked", subject=True):
        p = e["payload"] or {}
        row = rows.setdefault(e["subject"], {"subject": e["subject"], "lo": None, "hi": None,
                                             "means": [], "semantics": p.get("semantics")})
        if "min" in p:
            row["lo"] = p["min"] if row["lo"] is None else min(row["lo"], p["min"])
            row["hi"] = p["max"] if row["hi"] is None else max(row["hi"], p["max"])
        if "mean" in p:
            row["means"].append(p["mean"])
    drawn = []
    for subject, row in rows.items():
        comp = _component(subject) or {}
        drawn.append({"subject": subject, "who": _who(comp), "mf": comp.get("mf"),
                      "mt": comp.get("mt"), "quantity": comp.get("quantity"),
                      "min": row["lo"], "max": row["hi"],
                      "mean": sum(row["means"]) / len(row["means"]) if row["means"] else None,
                      "multiplicative": "multiplicative" in (row["semantics"] or "")})
    drawn.sort(key=lambda r: (r["mf"] or 0, r["mt"] or 0, r["subject"]))
    moved = _also_moved(ev, set(rows))
    by_mf: Dict[int, List[int]] = {}
    for subject in moved:
        comp = _component(subject)
        if comp:
            by_mf.setdefault(comp["mf"], []).append(comp["mt"])
    also = [{"mf": mf, "quantity": QUANTITIES.get(mf, f"MF{mf}"), "mts": _ranges(mts)}
            for mf, mts in sorted(by_mf.items())]
    unused = []
    for e in ev.of("warning"):
        p = e["payload"] or {}
        if p.get("code") == "not_in_evaluation" and p.get("reactions"):
            unused.append({"mts": _ranges(p["reactions"]), "why": e["message"]})
    return {"drawn": drawn, "also": also, "unused": unused}


def _also_moved(ev, drawn) -> List[str]:
    moved: List[str] = []
    for e in ev.of("applied"):
        for c in (e["payload"] or {}).get("components") or []:
            if c not in drawn and c not in moved:
                moved.append(c)
    return moved


# --------------------------------------------------------------------------- drawing, samples, files

def _drawing(ev, options) -> List[str]:
    out = []
    space = options.get("space")
    if isinstance(space, dict):
        parts = ", ".join(f"{_space_key(k)} {v}" for k, v in space.items())
        out.append(f"Draw space by quantity: {parts}. A log-normal draw keeps a factor positive "
                   f"with mean 1; a linear one (1 + y) can reach zero, and every factor is checked.")
    elif space == "linear":
        out.append("Linear draw: a factor is 1 + y. It can reach zero or below; every factor is checked.")
    elif space:
        out.append("Log-normal draw: a factor is exp(y), so it stays positive and its mean is 1.")
    if 35 in _quantities(ev):
        out.append("Fission-spectrum bands are drawn linearly as changes of probability, whatever "
                   "the space: their covariance is absolute and each row sums to zero.")
    cond = ev.first("conditioned", subject=False)
    mode = (cond or {}).get("payload", {}).get("mode")
    if mode == "auto":
        out.append("Repairs to the covariance chosen by the pre-flight: only what makes a block "
                   "drawable, saved as conditioning_plan.json.")
    elif mode == "plan":
        out.append("Repairs to the covariance taken from the plan passed to the run.")
    elif mode in ("none", "off"):
        out.append("No repairs to the covariance: every block drawn as stated.")
    if options.get("crossSectionSums"):
        mode = options.get("sumBlocks")
        text = ("Totals rebuilt from their parts on every sample, so a total always equals the sum "
                "of its reactions (NJOY rebuilds MT1 and MT4 from their parts anyway).")
        if mode == "undecomposed":
            text += (" A total whose parts carry no covariance of their own moves them all by one "
                     "common factor.")
        elif mode == "fill":
            text += " A total's block also moves the parts that have no covariance of their own."
        out.append(text)
    elif options.get("crossSectionSums") is False:
        out.append("Totals not rebuilt: each reaction moves by its own block only.")
    region = options.get("resonanceRegion")
    if ev.has("reconstructed"):
        out.append("Resonance region rebuilt by NJOY RECONR at 0 K before the factors were applied, "
                   "so the cross sections a code sees there are perturbed.")
    elif region == "evaluated" and any((e["payload"] or {}).get("code") == "resonance_background_only"
                                       for e in ev.of("warning")):
        out.append("Resonance region: the tabulated MF3 background only (see the warning).")
    out.append(f"{str(options.get('samplingMethod', '')).capitalize()} sequence, "
               f"{str(options.get('decompositionMethod', '')).upper()} decomposition, seed "
               f"{options.get('seed')}. The same seed gives the same samples.")
    return out


def _space_key(key) -> str:
    try:
        return QUANTITY.get(int(key), f"MF{key}")
    except (TypeError, ValueError):
        return str(key)


def _samples(ev, n) -> Dict[str, Any]:
    numbers = sorted({e["sample"] for e in ev.events
                      if e.get("sample") is not None and e["kind"] in ("applied", "emitted",
                                                                      "error", "checked")})
    problems: Dict[int, str] = {}
    for e in ev.events:
        if e.get("sample") is None or e["level"] == "info":
            continue
        worst = "fail" if e["level"] == "error" else "warn"
        if problems.get(e["sample"]) != "fail":
            problems[e["sample"]] = worst
    rows = []
    for s in numbers:
        checks = [e["payload"] for e in ev.of("checked") if e.get("sample") == s]
        emitted = [e for e in ev.of("emitted") if e.get("sample") == s]
        applied = [e for e in ev.of("applied") if e.get("sample") == s]
        took = sum((e.get("seconds") or 0.0) for e in emitted + applied)
        rows.append({
            "sample": s, "status": problems.get(s, "pass"),
            "files": [{"format": e["subject"], "path": e["payload"].get("path"),
                       "bytes": e["payload"].get("bytes")} for e in emitted],
            "min": min((c["min"] for c in checks if "min" in c), default=None),
            "max": max((c["max"] for c in checks if "max" in c), default=None),
            "seconds": took or None,
            "messages": [e["message"] for e in ev.events
                         if e.get("sample") == s and e["level"] != "info"]})
    listed_all = len(rows) <= SAMPLE_ROWS
    shown = rows if listed_all else [r for r in rows if r["status"] != "pass"]
    return {"total": n, "done": len(rows), "flagged": len(problems), "listedAll": listed_all,
            "rows": shown}


def _files(ev, options, ended) -> List[Dict[str, Any]]:
    out = []
    by_format: Dict[str, List[Dict[str, Any]]] = {}
    for e in ev.of("emitted"):
        by_format.setdefault(e["subject"], []).append(e)
    for fmt, items in by_format.items():
        total = sum(e["payload"].get("bytes") or 0 for e in items)
        example = items[0]["payload"].get("path")
        out.append({"name": f"NNNN/ ({_plural(len(items), 'file')})",
                    "what": f"{_formats_text([fmt])}, one per sample"
                            + (f", e.g. {Path(example).name}" if example else ""),
                    "bytes": total})
    what = {"factors.parquet": "every factor of every sample, one table",
            "conditioning_plan.json": "the repairs applied; edit it and pass it back to change them"}
    for e in ev.of("written"):
        out.append({"name": e["subject"], "what": what.get(e["subject"], e["message"]),
                    "bytes": (e["payload"] or {}).get("bytes")})
    if options.get("outputDir") and ended:
        if ev.has("finished"):
            out.append({"name": "run_metadata.json", "what": "options and provenance, for a program",
                        "bytes": None})
        out.append({"name": "run.log.jsonl", "what": "every event of the run, for a program",
                    "bytes": None})
        out.append({"name": "report.html", "what": "this report", "bytes": None})
    return out


# --------------------------------------------------------------------------- steps, failure, details

def _steps(ev, options, failed_stage) -> List[str]:
    out = []
    read = ev.first("read")
    if read:
        fmt = (read["payload"] or {}).get("format")
        out.append(f"Read {read.get('subject') or 'the evaluation'}"
                   + (f" ({_SOURCE_FORMAT.get(fmt, fmt)})" if fmt else "") + " and its covariances.")
    if ev.has("reconstructed"):
        out.append("Rebuilt the cross sections of the resonance region with NJOY RECONR at 0 K.")
    request = ev.first("request")
    if request:
        out.append(f"Turned the request into what the file states: {request['message']}.")
    blocks = ev.of("assembled", subject=True)
    if blocks:
        out.append(f"Assembled {_plural(len(blocks), 'covariance block')}: "
                   + "; ".join(f"{b['subject']} ({b['payload'].get('dimension')}×"
                               f"{b['payload'].get('dimension')})" for b in blocks) + ".")
    head = ev.first("inspected", subject=False)
    if head:
        repairs = _repairs(ev)
        out.append("Inspected every block before the draw"
                   + (f" and repaired {_plural(len(repairs), 'block')}." if repairs else "."))
    if ev.has("drawn"):
        out.append(f"Drew {_plural(int(options.get('nSamples') or 0), 'sample')}.")
    if ev.has("applied"):
        formats = options.get("formats") or []
        out.append("For each sample: put the factors on the evaluation, checked them"
                   + ("." if options.get("dryRun") else f", and wrote {_formats_text(formats)}."))
    if ev.has("written"):
        out.append("Wrote the run's own files (factors, plan) and this report.")
    if failed_stage:
        out.append(f"Stopped at {_stage_title(failed_stage).lower()} (see What went wrong).")
    return out


_NJOY_FIXES = [
    ("Tell kika where NJOY is",
     "Name the executable in the run's NJOY field, or set NJOY_EXECUTABLE once for every run "
     "(setx NJOY_EXECUTABLE C:\\path\\to\\njoy.exe). On Windows use a statically linked build: a "
     "MinGW build linked against libgfortran.dll misreads numbers when Windows uses a decimal comma."),
    ("Perturb only the tabulated MF3",
     "Resonance region: evaluated. The run then needs no NJOY, but in the resonance region it "
     "moves only the small background under the resonances, so those energies barely change."),
]


_SOURCE_FORMAT = {"endf": "ENDF-6", "gnds": "GNDS"}


def _failure(ev, error, failed_stage) -> Dict[str, Any]:
    payload = error["payload"] or {}
    exception = payload.get("exception") or ""
    message = error["message"]
    blocking = [e for e in ev.of("inspected", subject=True)
                if (e["payload"] or {}).get("severity") == "blocks"]
    fixes: List[Tuple[str, str]] = []
    short = _first_sentence(message, 110)
    lead = _first_sentence(message, 400)
    if exception == "NjoyNotFoundError":
        short = "NJOY is needed and was not found"
        lead = ("This evaluation gives its cross sections through resonance parameters, where "
                "the tabulated MF3 is only a background. To perturb the cross sections a code "
                "sees there, kika first rebuilds them with NJOY (RECONR), and no NJOY was found.")
        fixes = list(_NJOY_FIXES)
    elif failed_stage == "inspected":
        short = "the covariance cannot be drawn"
        lead = ("The pre-flight found a covariance block that cannot be drawn and that no repair "
                "can make drawable. The block and what is wrong with it are below.")
        fixes.append(("Leave those reactions out",
                      "Ask for every reaction except the ones in the block named below. The rest "
                      "is drawn as usual; the reactions left out stay as evaluated, so their "
                      "uncertainty is missing from the results."))
        if any((e["payload"] or {}).get("check") == "finiteness" for e in blocking):
            fixes.append(("Draw that quantity in linear space",
                          "A factor is then 1 + y, which needs no logarithm, so entries that have "
                          "no value in log space are no obstacle. Factors can then reach zero or "
                          "below; the run warns about each one."))
    return {"stage": failed_stage, "stageTitle": _stage_title(failed_stage),
            "exception": exception, "message": message, "short": short, "lead": lead,
            "sample": error.get("sample"),
            "blocking": [{"subject": e["subject"], "message": e["message"]} for e in blocking],
            "fixes": [{"title": t, "text": x} for t, x in fixes]}


_CHECK_LABEL = {"inert_rows": "Rows with no variance", "rank": "Rank",
                "correlation_bound": "Correlations above 1", "definiteness": "Positive semidefinite",
                "finiteness": "Finite", "symmetry": "Symmetric"}


def _details(ev, options, header) -> Dict[str, Any]:
    timing: Dict[str, float] = {}
    for e in ev.events:
        if e.get("seconds") is None or e["kind"] in ("finished", "error"):
            continue
        if e["kind"] == "applied" and e.get("sample") is None:
            continue
        if e["kind"] in ("drawn", "inspected", "conditioned", "assembled") and e.get("subject"):
            continue
        timing[e["kind"]] = timing.get(e["kind"], 0.0) + float(e["seconds"])
    finished = ev.first("finished")
    blocks = []
    drawn = {e["subject"]: e["payload"] for e in ev.of("drawn", subject=True)}
    for e in ev.of("assembled", subject=True):
        p = e["payload"] or {}
        d = drawn.get(e["subject"], {})
        blocks.append({"subject": e["subject"], "dimension": p.get("dimension"),
                       "union": p.get("union"), "rank": d.get("rank"),
                       "null": d.get("n_null"),
                       "error": d.get("realised_covariance_error")})
    preflight = [{"subject": e["subject"],
                  "check": _CHECK_LABEL.get((e["payload"] or {}).get("check"),
                                            (e["payload"] or {}).get("check") or ""),
                  "severity": (e["payload"] or {}).get("severity"), "message": e["message"]}
                 for e in ev.of("inspected", subject=True)
                 if (e["payload"] or {}).get("block")]
    mechanics = [e["message"] for e in ev.of("note")
                 if (e["payload"] or {}).get("code") in _MECHANICS]
    return {"timing": timing, "elapsed": finished.get("seconds") if finished else None,
            "blocks": blocks, "preflight": preflight, "mechanics": mechanics}


# --------------------------------------------------------------------------- HTML

_EXTRA_CSS = """
.kika-report .stages{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 6px}
.kika-report .stage{display:inline-flex;align-items:center;gap:6px;padding:3px 10px 3px 4px;border-radius:999px;
font-size:13px;font-weight:600;border:1px solid var(--line);background:var(--bg)}
.kika-report .stage .t{font-weight:400;color:var(--muted);font-size:12px}
.kika-report .stage.pass{border-color:var(--pass-line)}
.kika-report .stage.warn{border-color:var(--warn-line);background:var(--warn-bg)}
.kika-report .stage.fail{border-color:var(--fail-line);background:var(--fail-bg)}
.kika-report .stage.skip{color:var(--faint);border-style:dashed}
.kika-report .bar{position:relative;display:inline-block;width:180px;height:14px;vertical-align:middle;
background:var(--panel);border:1px solid var(--line);border-radius:4px}
.kika-report .bar .r{position:absolute;top:3px;height:6px;border-radius:3px;background:var(--accent);opacity:.55}
.kika-report .bar .one{position:absolute;top:-2px;width:1px;height:16px;background:var(--muted)}
.kika-report .bar .m{position:absolute;top:2px;width:8px;height:8px;margin-left:-4px;border-radius:50%;background:var(--accent)}
.kika-report .fix{border:1px solid var(--line);border-radius:10px;padding:10px 16px;margin:8px 0}
.kika-report .fix h3{margin:0 0 4px}.kika-report .fix p{margin:2px 0;color:var(--muted);font-size:14px}
.kika-report tr.warnrow td{background:var(--warn-bg)}
.kika-report .cause{border:1px solid var(--fail-line);background:var(--fail-bg);border-radius:10px;padding:12px 16px;margin:0 0 12px}
.kika-report .cause p{margin:4px 0}
.kika-report pre.msg{white-space:pre-wrap;font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;
font-size:12.5px;margin:6px 0 0}
"""


def _badge(status, text=None) -> str:
    return (f'<span class="badge {status}"><span class="mark">{MARK[status]}</span>'
            f'{_esc(text or LABEL[status])}</span>')


def _bar(lo, hi, mean, a=0.2, b=5.0) -> str:
    import math

    def x(v):
        v = min(max(v, a), b)
        return 100 * (math.log(v) - math.log(a)) / (math.log(b) - math.log(a))
    return (f'<span class="bar" title="{lo:.3f} to {hi:.3f}, mean {mean:.3f}">'
            f'<span class="one" style="left:{x(1):.1f}%"></span>'
            f'<span class="r" style="left:{x(lo):.1f}%;width:{max(x(hi) - x(lo), 1):.1f}%"></span>'
            f'<span class="m" style="left:{x(mean):.1f}%"></span></span>')


def _table(head: Sequence[Tuple[str, bool]], rows: Sequence[str], *, tall=False) -> str:
    th = "".join(('<th class="n">' if num else "<th>") + f"{_esc(h)}</th>" for h, num in head)
    return (f'<div class="wrap{" tall" if tall else ""}"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _render_html(d: Dict[str, Any]) -> str:
    from kika.endf.reconstruction_report import _CSS

    h, s = d["header"], d["summary"]
    status = s["status"] if s["status"] in ("pass", "warn", "fail") else "warn"
    title = f"Perturbation run · {h['nuclide']}"
    sub = " · ".join(_esc(x) for x in (h.get("library"), h.get("file"),
                                       f"kika {h['kika']}" if h.get("kika") else None,
                                       h.get("started")) if x)
    n = h.get("nSamples")
    h1 = (f'{_esc(h["nuclide"])} <span class="muted">· {_plural(int(n or 0), "sample")}'
          + (" · rehearsal" if h.get("dryRun") else "") + "</span>")

    stages = "".join(
        f'<span class="stage {st["status"]}"><span class="mark" style="background:var(--{st["status"]})">'
        f'{MARK[st["status"]]}</span>{_esc(st["title"])}'
        + (f' <span class="t">{_esc(st["detail"])}</span>' if st["detail"] else "") + "</span>"
        for st in d["stages"])
    parts: List[str] = []
    toc: List[Tuple[str, str]] = []

    failure = d.get("failure")
    if failure:
        toc += [("kp-what", "What went wrong")]
        cause = (f'<div class="cause"><p><strong>{_esc(failure["stageTitle"])}'
                 + (f' · sample {failure["sample"]}' if failure.get("sample") is not None else "")
                 + (f' · {_esc(failure["exception"])}' if failure.get("exception") else "")
                 + f'</strong></p><pre class="msg">{_esc(failure["message"])}</pre></div>')
        if failure["blocking"]:
            cause += _table([("Block", False), ("What is wrong", False)],
                            [f"<tr><td>{_esc(b['subject'])}</td><td>{_esc(b['message'])}</td></tr>"
                             for b in failure["blocking"]])
        parts.append(f'<h2 id="kp-what">What went wrong</h2>{cause}')
        if failure["fixes"]:
            toc.append(("kp-fix", "How to go on"))
            parts.append('<h2 id="kp-fix">How to go on</h2>' + "".join(
                f'<div class="fix"><h3>{i}. {_esc(f["title"])}</h3><p>{_esc(f["text"])}</p></div>'
                for i, f in enumerate(failure["fixes"], 1)))

    toc.append(("kp-checks", "Checks"))
    rows = "".join(f'<tr><td>{_badge(c["status"])}</td><td><span class="ttl">{_esc(c["title"])}'
                   f'</span></td><td>{_esc(c["result"])}</td></tr>' for c in d["checks"])
    parts.append('<h2 id="kp-checks">Checks</h2><p class="lead">What was verified. Each check is '
                 'explained under <a href="#kp-about">About the checks</a>.</p>'
                 '<div class="wrap"><table class="checks"><thead><tr><th>Status</th><th>Check</th>'
                 f'<th>Result</th></tr></thead><tbody>{rows}</tbody></table></div>')

    if d["findings"]:
        toc.append(("kp-findings", "Warnings and notes"))
        blocks = []
        for f in d["findings"]:
            items = "".join(f"<li>{_esc(i)}</li>" for i in f["items"])
            blocks.append(
                f'<div class="group"><div class="ghead">{_badge(f["status"])}<span class="ttl">'
                f'{_esc(f["title"])}</span><span class="muted">{_esc(f["scope"])}</span></div>'
                f'<p class="muted">{_esc(f["body"])}</p>'
                + (f'<ul class="items">{items}</ul>' if items else "") + "</div>")
        parts.append('<h2 id="kp-findings">Warnings and notes</h2><p class="lead">One entry per '
                     'cause, with what it means for the samples.</p>' + "".join(blocks))

    p = d["perturbed"]
    if p["drawn"]:
        toc.append(("kp-perturbed", "What was perturbed"))
        rows = []
        for r in p["drawn"]:
            bar = (_bar(r["min"], r["max"], r["mean"])
                   if r["multiplicative"] and r["min"] is not None and r["min"] > 0 else "")
            fmt = (lambda v: "" if v is None else f"{v:.3f}")
            rows.append(f'<tr><td>{_esc(r["who"])}</td><td class="n">{fmt(r["min"])}</td>'
                        f'<td class="n">{fmt(r["max"])}</td><td class="n">{fmt(r["mean"])}</td>'
                        f'<td>{bar}</td></tr>')
        for a in p["also"]:
            rows.append(f'<tr><td colspan="5">Also changed on every sample, through the components '
                        f'above: {_esc(a["quantity"].lower())} MT{_esc(a["mts"])}</td></tr>')
        for u in p["unused"]:
            rows.append(f'<tr class="warnrow"><td colspan="5">Not perturbed: MT{_esc(u["mts"])}. '
                        f'{_esc(u["why"])}</td></tr>')
        n_drawn = d["summary"]["samples"]
        parts.append(
            '<h2 id="kp-perturbed">What was perturbed</h2><p class="lead">Each component drawn, '
            f'with the range of its factors over the {_plural(int(n_drawn or 0), "sample")}. A '
            'factor of 1 leaves the value as evaluated. Fission-spectrum bands are changes of '
            'probability, not factors.</p>'
            + _table([("Component", False), ("Smallest", True), ("Largest", True), ("Mean", True),
                      ("Range (0.2 to 5, log; line at 1)", False)], rows))

    if d["drawing"]:
        toc.append(("kp-drawing", "How it was drawn"))
        parts.append('<h2 id="kp-drawing">How it was drawn</h2><ul class="items">'
                     + "".join(f"<li>{_esc(x)}</li>" for x in d["drawing"]) + "</ul>")

    sm = d["samples"]
    if sm["done"]:
        toc.append(("kp-samples", "Samples"))
        rows = []
        for r in sm["rows"]:
            files = "<br>".join(f'<code>{_esc(Path(f["path"]).name if f["path"] else f["format"])}'
                                f'</code> {_esc(_size(f["bytes"]))}' for f in r["files"])
            msgs = "".join(f'<div class="faint">{_esc(m)}</div>' for m in r["messages"])
            fmt = (lambda v: "" if v is None else f"{v:.3f}")
            rows.append(f'<tr><td>{_badge(r["status"])}</td><td class="n">{r["sample"]:04d}</td>'
                        f'<td>{files or "<span class=muted>none</span>"}{msgs}</td>'
                        f'<td class="n">{fmt(r["min"])}</td><td class="n">{fmt(r["max"])}</td>'
                        f'<td class="n">{_esc(_secs(r["seconds"]))}</td></tr>')
        note = ""
        if not sm["listedAll"]:
            others = sm["done"] - len(sm["rows"])
            rows.append(f'<tr><td>{_badge("pass")}</td><td colspan="5">{others} other samples, '
                        f'every check passed (listed in run.log.jsonl).</td></tr>')
            note = (f'<p class="faint">{sm["done"]} samples: only those with a warning or a '
                    f'failure are listed.</p>')
        parts.append('<h2 id="kp-samples">Samples</h2>'
                     + _table([("Status", False), ("Sample", True), ("Files", False),
                               ("Smallest factor", True), ("Largest", True), ("Took", True)], rows)
                     + note)

    toc.append(("kp-done", "What was done"))
    parts.append('<h2 id="kp-done">What was done</h2><ol class="steps">'
                 + "".join(f"<li>{_esc(x)}</li>" for x in d["steps"]) + "</ol>")

    if d["files"]:
        toc.append(("kp-files", "Files"))
        rows = [f'<tr><td><code>{_esc(f["name"])}</code></td><td>{_esc(f["what"])}</td>'
                f'<td class="n">{_esc(_size(f["bytes"]))}</td></tr>' for f in d["files"]]
        folder = (f'<p class="faint">In <code>{_esc(h["outputDir"])}</code></p>'
                  if h.get("outputDir") else "")
        parts.append('<h2 id="kp-files">Files</h2>' + folder
                     + _table([("File", False), ("What it is", False), ("Size", True)], rows))

    toc.append(("kp-run", "Run details"))
    parts.append('<h2 id="kp-run">Run details</h2>' + _details_html(d))
    toc.append(("kp-about", "About the checks"))
    parts.append('<h2 id="kp-about">About the checks</h2>' + _about_html())

    nav = "".join(f'<a href="#{i}">{_esc(t)}</a>' for i, t in toc)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="light dark">'
            f'<title>{_esc(title)}</title></head><body style="margin:0;background:Canvas">'
            f'<style>{_CSS}{_EXTRA_CSS}</style><div class="kika-report">'
            '<p class="eyebrow">kika · perturbation run</p>'
            f'<h1>{h1}</h1><p class="sub">{sub}</p>'
            f'<div class="verdict {status}"><span class="mark">{MARK[status]}</span><div>'
            f'<p class="big">{_esc(s["headline"])}</p><p>{_esc(s["lead"])}</p></div></div>'
            f'<div class="stages">{stages}</div><nav class="toc">{nav}</nav>'
            + "".join(parts) + "</div></body></html>\n")


def _details_html(d) -> str:
    h = d["header"]
    started = next((e for e in d["events"] if e["kind"] == "started"), None)
    o = (started or {}).get("payload") or {}
    det = d["details"]

    def dl(title, pairs):
        body = "".join(f"<dt>{_esc(k)}</dt><dd>{v}</dd>" for k, v in pairs if v not in (None, ""))
        return f"<section><h3>{_esc(title)}</h3><dl>{body}</dl></section>"

    nuclide = h["nuclide"] + (f" (ZA {h['za']}" + (f", MAT {h['mat']}" if h.get("mat") else "")
                              + ")" if h.get("za") else "")
    space = o.get("space")
    space = ", ".join(f"{_space_key(k)} {v}" for k, v in space.items()) \
        if isinstance(space, dict) else space
    timing = det["timing"]
    names = {"read": "reading", "reconstructed": "resonance region", "assembled": "covariance",
             "inspected": "pre-flight", "conditioned": "repairs", "drawn": "draw",
             "applied": "applying", "checked": "checks", "emitted": "writing",
             "written": "run files"}
    split = ", ".join(f"{names.get(k, k)} {_secs(v)}" for k, v in timing.items() if v >= 0.005)
    elapsed = det.get("elapsed")
    sections = [
        dl("Evaluation", [("Nuclide", _esc(nuclide)), ("Library", _esc(h.get("library"))),
                          ("Format", _esc(_SOURCE_FORMAT.get(h.get("format"), h.get("format")))),
                          ("File", f"<code>{_esc(h.get('path'))}</code>"),
                          ("SHA-256", f"<code>{_esc(h['sha256'])}</code>" if h.get("sha256") else None),
                          ("Covariance", f"<code>{_esc(h['covarianceSource'])}</code>"
                           if h.get("covarianceSource") else "from the same file")]),
        dl("Draw", [("Samples", _esc(o.get("nSamples"))), ("Seed", _esc(o.get("seed"))),
                    ("Space", _esc(space)), ("Sequence", _esc(o.get("samplingMethod"))),
                    ("Decomposition", _esc(o.get("decompositionMethod"))),
                    ("Grouping", _esc(o.get("grouping"))), ("PSD method", _esc(o.get("psdMethod"))),
                    ("Totals", _esc(f"rebuilt ({o.get('sumBlocks')})" if o.get("crossSectionSums")
                                    else "not rebuilt")),
                    ("Resonance region", _esc(o.get("resonanceRegion"))),
                    ("Outputs", _esc(_formats_text(o.get("formats") or []))),
                    ("Rehearsal", "yes" if o.get("dryRun") else None),
                    ("Workers", _esc(o.get("nWorkers")))]),
        dl("Run", [("Started", _esc(h.get("started"))), ("kika", _esc(h.get("kika"))),
                   ("Output folder", f"<code>{_esc(h['outputDir'])}</code>" if h.get("outputDir")
                    else "none"),
                   ("Time", _esc((f"{elapsed:.1f} s" if elapsed is not None else "")
                                 + (f" ({split})" if split else ""))),
                   ("Execution", _esc(" ".join(det["mechanics"])) if det["mechanics"] else None)]),
    ]
    out = f'<div class="meta">{"".join(sections)}</div>'
    if det["blocks"]:
        rows = [f'<tr><td>{_esc(b["subject"])}</td><td class="n">{_esc(b["dimension"])}</td>'
                f'<td class="n">{_esc(b["rank"])}</td><td class="n">{_esc(b["null"])}</td>'
                f'<td class="n">{_fmt3g(b["error"])}</td></tr>'
                for b in det["blocks"]]
        out += ('<details><summary>Covariance blocks drawn</summary>'
                + _table([("Block", False), ("Order", True), ("Rank", True), ("Zero-variance "
                          "directions", True), ("Achieved covariance error", True)], rows)
                + '<ul class="notes"><li>The achieved covariance error compares the samples\' '
                  'covariance with the stated one on the drawn directions. It shrinks roughly as '
                  '1/√N, so with few samples it is large whatever the draw.</li></ul></details>')
    if det["preflight"]:
        rows = [f'<tr><td>{_esc(p["subject"])}</td><td>{_esc(p["check"])}</td>'
                f'<td>{_esc(p["message"])}</td></tr>' for p in det["preflight"]]
        out += ('<details><summary>Pre-flight: everything it measured on the covariance</summary>'
                + _table([("Block", False), ("Check", False), ("Result", False)], rows)
                + "</details>")
    rows = []
    for e in d["events"]:
        cls = ' class="warnrow"' if e["level"] != "info" else ""
        rows.append(f'<tr{cls}><td class="nw">{_esc((e.get("at") or "")[11:23])}</td>'
                    f'<td>{_esc(e["kind"])}</td>'
                    f'<td class="n">{"" if e.get("sample") is None else e["sample"]}</td>'
                    f'<td>{_esc(e.get("subject") or "")}</td><td>{_esc(e["message"])}</td>'
                    f'<td class="n">{_esc(_secs(e.get("seconds")))}</td></tr>')
    out += (f'<details><summary>Every event of the run ({len(rows)}; the same data as '
            f'run.log.jsonl)</summary>'
            + _table([("Time", False), ("Kind", False), ("Sample", True), ("About", False),
                      ("Message", False), ("Took", True)], rows, tall=True) + "</details>")
    return out


_ABOUT = (
    ("kp-about-finite", "Every factor is finite and positive",
     "Each sample is a table of factors per component and group, checked before it touches the "
     "evaluation. A factor that is not a number, or not above zero, would make the value it "
     "scales meaningless (a fission-spectrum band is a change, checked for being finite).",
     (("pass", "Every factor of every sample is finite, and above zero where it multiplies."),
      ("warn", "Some factor is zero or negative: the samples concerned are listed."),
      ("skip", "No samples were drawn."))),
    ("kp-about-cov", "The covariance can be drawn as stated",
     "Before the draw, kika inspects every covariance block (the pre-flight). A block must be "
     "symmetric, finite and positive semidefinite to be drawn as the file states it. When it is "
     "not, kika applies the smallest repair that makes it drawable and records how far it moved "
     "the stated variances.",
     (("pass", "Every block is drawn exactly as the file states it."),
      ("note", "A block was repaired and its stated variances moved by 1 % or less."),
      ("warn", "A block was repaired and its stated variances moved by more than 1 %."),
      ("fail", "No repair applies. Nothing is drawn."))),
    ("kp-about-written", "Every sample was written",
     "Each output asked for is written for each sample, and its size is recorded.",
     (("pass", "Every output of every sample is on disk."),
      ("fail", "Some outputs are missing; the samples concerned are listed."),
      ("skip", "A rehearsal, or no samples drawn."))),
    ("kp-about-ensemble", "The ensemble reproduces the stated uncertainty",
     "Across all samples, the spread of each component's factors is compared with the standard "
     "deviation the file states. With few samples the spread itself is uncertain (about "
     "1/√(2N) relative), so the limit grows as N falls.",
     (("pass", "Every component's spread agrees with the stated one within the limit for N."),
      ("warn", "Some component's spread is outside the limit for N."),
      ("skip", f"Fewer than {ENSEMBLE_MIN_SAMPLES} samples, or not measured by this version."))),
)


def _fmt3g(value) -> str:
    return "" if value is None else f"{value:.3g}"


def _about_html() -> str:
    out = []
    for anchor, title, text, rows in _ABOUT:
        li = "".join(f"<li>{_badge(s)}<span>{_esc(x)}</span></li>" for s, x in rows)
        out.append(f'<div class="entry" id="{anchor}"><div><h3>{_esc(title)}</h3>'
                   f'<p>{_esc(text)}</p></div><ul>{li}</ul></div>')
    return "".join(out)


# --------------------------------------------------------------------------- text

def _render_text(d: Dict[str, Any]) -> str:
    h, s = d["header"], d["summary"]
    lines = [f"kika perturbation run: {h['nuclide']} ({h['file']})", "=" * 72,
             s["headline"], s["lead"], "",
             "Stages  " + "  ".join(f"[{_TEXT_MARK[st['status']]}] {st['title']}"
                                    for st in d["stages"]), ""]
    if d.get("failure"):
        f = d["failure"]
        lines += [f"What went wrong ({f['stageTitle']})", f["message"], ""]
        for i, fix in enumerate(f["fixes"], 1):
            lines += [f"  {i}. {fix['title']}: {fix['text']}"]
        if f["fixes"]:
            lines.append("")
    lines.append("Checks")
    for c in d["checks"]:
        lines.append(f"  [{_TEXT_MARK[c['status']]:>4}] {c['title']}: {c['result']}")
    if d["findings"]:
        lines += ["", "Warnings and notes"]
        for f in d["findings"]:
            scope = f" ({f['scope']})" if f["scope"] else ""
            lines.append(f"  [{_TEXT_MARK[f['status']]:>4}] {f['title']}{scope}")
            lines.append(f"         {f['body']}")
    if d["files"]:
        lines += ["", "Files" + (f" in {h['outputDir']}" if h.get("outputDir") else "")]
        for f in d["files"]:
            lines.append(f"  {f['name']}  {_size(f['bytes'])}  {f['what']}")
    return "\n".join(lines) + "\n"
