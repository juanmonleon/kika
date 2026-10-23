"""The report of a processing run (:meth:`kika.endf.EndfProcessing.report_document`).

It answers two questions, in this order, for anyone who opens it:

1. **Did it go well?** One line at the top, the worst status of every stage and
   every file written, and one sentence of what was obtained.
2. **What exactly was done?** The checks of each stage, the findings behind any
   note or warning, the files written and how each was verified, the choices
   kika made for the user, and the steps in order. Tables of numbers sit folded
   under *Details*; the run's provenance and the meaning of every check close
   the page.

The rules it follows:

- A status says what the outcome means for the user, not how far a number is
  from another. **Failed**: a result not to use, or a file not written.
  **Warning**: something that can change a result and needs the user's
  judgement. **Note**: something in the evaluation that kika resolved in a
  fixed, documented way, with no effect on the output.
- Each check gives one number: its worst case against its limit.
- Plain words: no function or attribute names, MTs with their reaction names,
  energies in the unit a reader expects.
- Text, HTML and JSON come from the same data. The JSON keeps ``summary.status``
  and the flat ``checks`` list of the reconstruction report, which is what the
  app reads for its verdict, and ``processing_stages`` as the run recorded them.

The 0 K reconstruction is the reconstruction report's own: its checks, findings,
tables and wording (:mod:`kika.endf.reconstruction_report`) are reused here, under
the stages they belong to.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .reconstruction_report import (
    _CHECK as _RECON_CHECK, _CLASS, _CSS, _LABEL, _MARK, _RANK, _TEXT_MARK, _TOLERANCE_NOTE, FAIL, NOTE,
    PASS, SKIP, WARN, Check, _ascii, _badge, _conversion_scope, _count, _e, _energy, _g, _html_table,
    _jsonable, _library, _nuclide, _pct, _range_rows, _Raw, _reaction_rows, _rel, _s, _segment_rows,
    _SEGMENT_HEAD, _span, _steps, _technical_notes, _tolerance_text, _wrap,
)

SCHEMA = "kika-processing-report/1"

#: The stages a report lists, in order, with the title a reader sees.
STAGE_TITLES = {
    "read": "Reading the evaluation",
    "reconstruct": "Cross sections at 0 K",
    "broaden": "Doppler broadening",
    "urr": "Unresolved-range probability tables",
    "heating": "Heating and damage",
    "write": "Files written",
}
#: Which of the reconstruction report's checks belong to which stage. Its PENDF
#: check is not used: the files written are checked under "Files written".
_RECON_STAGE = {"read": ("conversion", "parameters"),
                "reconstruct": ("tabulation", "source_sums", "negatives", "grid", "integrals")}

_FORMAT_NAMES = {"ace": "ACE", "hdf5": "OpenMC HDF5", "pendf": "PENDF", "gnds": "GNDS"}
_VERIFIED_BY = {
    "ace": "read back in strict mode and compared word for word",
    "pendf": "every material read back and compared with the tables, within half the tolerance",
    "hdf5": "read back and compared before replacing any file",
    "gnds": "read back; its styles and temperatures checked",
}

#: The checks this report adds to the reconstruction's (same layout: what it
#: compares, then what each outcome means).
CHECKS = (
    Check("stage", "The stage ran",
          "A stage the run asked for either completes or is refused with a reason. A refusal keeps "
          "every result of the stages before it.",
          {PASS: "The stage completed.",
           FAIL: "The stage could not run on this evaluation; the reason is given. Its results are "
                 "missing, and so is any file that needs them."}),
    Check("doppler", "Broadened tables meet their share of the tolerance",
          "Each temperature's table is built by an adaptive engine that compares it, at probes it did "
          "not use to build it, with the exact free-gas broadening of the 0 K cross sections. Its share "
          "is a twentieth of the stated tolerance: the 0 K source keeps half and the thinning of the "
          "grid the rest, so the three cannot add up beyond the tolerance the tables are stated to.",
          {PASS: "Every reaction at every temperature is within its share.",
           FAIL: "A table is outside its share; the engine refuses it rather than publish it."}),
    Check("urr_statistics", "Probability tables reach their statistical target",
          "The tables average over random resonance ladders. Ladders are drawn until the statistical "
          "error of every band's share and of every self-shielding factor is below the target "
          "(0.5 %), with at least 64 and at most 4096 ladders per energy. The error is estimated by "
          "leaving one ladder out at a time (jackknife).",
          {PASS: "Every energy reached the target.",
           WARN: "Some energies stopped at the ladder limit before reaching the target. Their tables "
                 "are usable but less precise than the rest, by the amount shown."}),
    Check("heating_sign", "Heating is not negative",
          "KERMA (MT301) is the energy left locally per reaction. A negative value means the "
          "evaluation gives the outgoing particles more energy than the reaction makes available.",
          {PASS: "No negative heating at any temperature.",
           NOTE: "Negative heating only at a few energies and by less than 0.1 % of the incident "
                 "energy: the evaluation's rounding of its own energy balance. Kept as computed.",
           WARN: "Negative heating deeper than 0.1 % of the incident energy, from the evaluation's own "
                 "energy balance: the reaction named gives its outgoing particles and photons more energy "
                 "than it makes available. It is kept as computed, as NJOY keeps it; energy-deposition "
                 "tallies there inherit it."}),
    Check("notes", "What the processing reports",
          "Building a table can decide things a user should know: shells joined or dropped, transitions "
          "with no energy, data the table approximates. Each is listed.",
          {PASS: "Nothing to report.",
           NOTE: "Listed; none stops the table from being used."}),
    Check("evaluation", "The evaluation passes kika's data checks",
          "Before it is processed, the evaluation's data (every file but the covariances) are checked "
          "against the format, mathematics and physics: the static checks of kika.check, which read the "
          "tape and nothing else. kika processes the evaluation either way; a finding says what in it a "
          "result may inherit. kika.check gives the full report.",
          {PASS: "No defect, warning or note.",
           NOTE: "Notes only: things true of the evaluation and worth knowing, none a fault.",
           WARN: "The evaluation has defects or warnings (listed). The output carries them as the "
                 "evaluation states them."}),
    Check("output", "File written and read back",
          "Every file is written to a temporary place, read back and compared with what was meant to "
          "be written, and only then moved over the destination.",
          {PASS: "Written, read back and verified.",
           NOTE: "Written and verified. Some data this format cannot carry was left out; each item is "
                 "listed.",
           FAIL: "Not written; the reason is given. No existing file was replaced."}),
)
_CHECK = {c.id: c for c in CHECKS}

_HEADLINE = {PASS: ("Processed and verified", "pass"),
             NOTE: ("Processed and verified, with notes", "pass"),
             WARN: ("Processed, with warnings to review", "warn"),
             FAIL: ("Verification failed", "fail")}


# --------------------------------------------------------------------------- data

def build(run) -> dict:
    """The report's data, plain types only, from an :class:`~kika.endf.EndfProcessing`."""
    rd = run.reconstruction.report_document().to_dict()
    plan = run.plan
    raw = [asdict(s) for s in run.stages]
    by_stage = {}
    for s in raw:
        by_stage.setdefault(s["stage"], []).append(s)
    recon_checks = {c["id"]: c for c in rd["checks"]}

    stages = []
    for sid, title in STAGE_TITLES.items():
        if sid == "write":
            continue
        record = (by_stage.get(sid) or [None])[0]
        if sid == "read":
            record = dict(record or {}, seconds=sum((r or {}).get("seconds") or 0.0
                                                    for r in by_stage.get("read", []) + by_stage.get("decode", [])))
        stages.append(_stage(sid, title, record, run, rd, recon_checks))

    _add_evaluation_check(stages, rd["evaluation"].get("path"))
    stages, outputs, checks, summary = _finish(stages, by_stage.get("write", []))
    summary["detail"] = _detail(rd, run, outputs, summary["counts"])
    ev = rd["evaluation"]
    data = dict(
        schema=SCHEMA, kind="neutron",
        generated_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        header=dict(kind="Incident neutron", name=_nuclide(rd), mat=rd["material"].get("mat"),
                    library=_library(ev), file=ev.get("file"), path=ev.get("path"), sha256=ev.get("sha256")),
        material=rd["material"], evaluation=ev,
        software=dict(rd["software"], kika_revision=_revision()),
        plan=dict(temperatures=list(plan.temperatures), broaden=plan.broaden, thinning=plan.thinning,
                  thnmax=plan.thnmax, urr=plan.urr, urr_bins=plan.urr_bins, urr_ladders=plan.urr_ladders,
                  heating=plan.heating, workers=plan.workers),
        temperatures=[float(t) for t in run.temperatures], thnmax_eV=run.thnmax,
        summary=summary, stages=stages, checks=checks, outputs=outputs,
        choices=_choices(run, rd, outputs), steps=_all_steps(run, rd, stages, outputs),
        urr=_urr_rows(run), doppler=_doppler_rows(run), heating=_heating_rows(run),
        reconstruction=rd, processing_stages=_jsonable(raw),
    )
    return _jsonable(data)


def _finish(stages, writes):
    """Add the "Files written" stage and gather the checks and the verdict."""
    outputs = [_output(s) for s in writes]
    write_checks = [_check("output", o["status_check"], f"{o['name']}: {o['result']}", o["items"], stage="write")
                    for o in outputs]
    stages.append(dict(id="write", title=STAGE_TITLES["write"],
                       status="done" if outputs else "not requested",
                       state=("done" if all(o["written"] for o in outputs) else "rejected") if outputs
                       else "not requested",
                       seconds=sum(o["seconds"] or 0.0 for o in outputs) or None,
                       summary=_outputs_line(outputs), checks=write_checks))
    checks = [c for st in stages for c in st["checks"]]
    status = max((c["status"] for c in checks), key=_RANK.get, default=PASS)
    status = PASS if status == SKIP else status
    incomplete = any(st["state"] == "rejected" for st in stages)
    headline = "Processing incomplete" if incomplete else _HEADLINE[status][0]
    counts = {lv: sum(1 for c in checks if c["status"] == lv) for lv in (FAIL, WARN, NOTE, PASS)}
    return stages, outputs, checks, dict(status=status, headline=headline, incomplete=incomplete, counts=counts)


#: Stage titles of the routes other than the neutron one.
_GENERIC_TITLES = {"read": "Reading the evaluation", "thermal": "Thermal scattering tables",
                   "tables": "Building the table"}


def build_generic(proc) -> dict:
    """The report's data for a route of :func:`kika.process` other than the neutron one
    (thermal, photo-atomic, photonuclear, dosimetry): its stages, the notes its
    processing made, the files written, the choices and the steps."""
    from kika import __version__
    parts = proc._report_parts()
    raw = [asdict(s) for s in proc.stages]
    stages = []
    for s in raw:
        if s["stage"] == "write":
            continue
        checks = []
        if s["status"] == "rejected":
            checks.append(_check("stage", FAIL, s["summary"] or "Refused.", stage=s["stage"]))
        notes = list((s.get("details") or {}).get("notes") or [])
        if notes:
            checks.append(_check("notes", NOTE, f"{len(notes)} decision(s) the processing reports.", notes,
                                 stage=s["stage"]))
        elif s["status"] == "done":
            checks.append(_check("notes", PASS, "Built as the method describes; nothing to report.",
                                 stage=s["stage"]))
        stages.append(dict(id=s["stage"], title=_GENERIC_TITLES.get(s["stage"], s["stage"].capitalize()),
                           status=s["status"], state="rejected" if s["status"] == "rejected" else s["status"],
                           seconds=s["seconds"], summary=s["summary"] or "", checks=checks))
    _add_evaluation_check(stages, str(proc.source))
    stages, outputs, checks, summary = _finish(stages, [s for s in raw if s["stage"] == "write"])
    what = f"{parts.get('title', proc.kind)} evaluation {parts.get('what', proc.source.name)}"
    done = [o for o in outputs if o["written"]]
    files = (" Files written and read back: " + ", ".join(
        f"{o['name']} ({', '.join(Path(f).name for f in o['files'])})" for o in done) + ".") if done         else " No file written."
    counts = summary["counts"]
    tally = [f"{counts[FAIL]} failed" if counts[FAIL] else "",
             f"{counts[WARN]} warning{'s' if counts[WARN] != 1 else ''}" if counts[WARN] else "no warnings",
             f"{counts[NOTE]} note{'s' if counts[NOTE] != 1 else ''}" if counts[NOTE] else ""]
    summary["detail"] = f"{what}.{files} " + ", ".join(t for t in tally if t).capitalize() + "."
    steps = list(parts.get("steps") or []) + ["Checked every stage (the checks above)."] + [
        f"Wrote {o['name']}: {', '.join(Path(f).name for f in o['files'])}; {o['verification']}." if o["written"]
        else f"Tried to write {o['name']}, which was refused." for o in outputs]
    data = dict(
        schema=SCHEMA, kind=proc.kind,
        generated_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        header=dict(kind=parts.get("title", proc.kind), name=parts.get("what", proc.source.name), mat=None,
                    library=None, file=proc.source.name, path=str(proc.source), sha256=_sha256(proc.source)),
        software=dict(kika_version=__version__, kika_revision=_revision()),
        plan={}, summary=summary, stages=stages, checks=checks, outputs=outputs,
        choices=list(parts.get("choices") or []), steps=steps, urr=[], doppler=[], heating=[],
        reconstruction=None, processing_stages=_jsonable(raw))
    return _jsonable(data)


#: Data checks already run, by (path, size, mtime): a report is often built twice.
_EVALUATION_CHECKS = {}
#: Findings listed under the check; the rest are counted.
_EVALUATION_ITEMS = 12


def _evaluation_check(path):
    """``(status, result, items)`` of the static data checks on *path*, or ``None``."""
    if not path or not os.path.isfile(path):
        return None
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _EVALUATION_CHECKS:
        from .checks.data import check_data
        from .checks.findings import DEFECT, NOTE as C_NOTE, WARN as C_WARN
        from .checks.suites import described

        try:
            report = check_data(path)
        except Exception as error:  # noqa: BLE001 - a checker failure must not cost the processing report
            _EVALUATION_CHECKS[key] = (NOTE, f"The data checks could not run ({type(error).__name__}: {error}).", [])
            return _EVALUATION_CHECKS[key]
        n = {lv: sum(1 for f in report if f.level == lv) for lv in (DEFECT, C_WARN, C_NOTE)}
        status = WARN if n[DEFECT] or n[C_WARN] else NOTE if n[C_NOTE] else PASS
        parts = [f"{n[DEFECT]} defect{'s' if n[DEFECT] != 1 else ''}" if n[DEFECT] else "",
                 f"{n[C_WARN]} warning{'s' if n[C_WARN] != 1 else ''}" if n[C_WARN] else "",
                 f"{n[C_NOTE]} note{'s' if n[C_NOTE] != 1 else ''}" if n[C_NOTE] else ""]
        tally = ", ".join(p for p in parts if p)
        result = (f"The data checks found {tally} in the evaluation." if tally
                  else "The data checks found nothing in the evaluation.")
        worst = [f for f in report if f.level in (DEFECT, C_WARN)]
        worst.sort(key=lambda f: (f.level != DEFECT, f.location.mf or 0, f.location.mt or 0))
        items = []
        for f in worst[:_EVALUATION_ITEMS]:
            d = described(f.check)
            items.append(f"{'Defect' if f.level == DEFECT else 'Warning'}, {d.title if d else f.check} "
                         f"({f.location}): {f.summary}")
        if len(worst) > _EVALUATION_ITEMS:
            items.append(f"... and {len(worst) - _EVALUATION_ITEMS} more; kika.check gives them all.")
        _EVALUATION_CHECKS[key] = (status, result, items)
    return _EVALUATION_CHECKS[key]


def _add_evaluation_check(stages, path):
    """The data checks' verdict, under the stage that read the evaluation."""
    found = _evaluation_check(path)
    if found is None:
        return
    status, result, items = found
    stage = next((st for st in stages if st["id"] == "read"), stages[0] if stages else None)
    if stage is None:
        return
    stage["checks"].append(_check("evaluation", status, result, items, stage=stage["id"]))


def _sha256(path):
    import hashlib
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _check(cid, status, result, items=(), *, stage, title=None):
    c = _CHECK.get(cid) or _RECON_CHECK[cid]
    return dict(id=cid, stage=stage, title=title or c.title, status=status, result=result, items=list(items))


def _stage(sid, title, record, run, rd, recon_checks):
    """One stage's view: its status, a plain sentence and its checks."""
    record = record or {}
    status = record.get("status") or "not requested"
    category = record.get("category")
    if status == "skipped" and category is None:
        status = "not requested"
    checks = [dict(recon_checks[c], stage=sid) for c in _RECON_STAGE.get(sid, ()) if c in recon_checks]
    if status == "rejected":
        checks.insert(0, _check("stage", FAIL, record.get("summary") or "Refused.", stage=sid))
    elif status == "done":
        checks += _stage_checks(sid, run, record)
    view = dict(id=sid, title=title, status=status, seconds=record.get("seconds"),
                summary=_stage_line(sid, record, run, rd), checks=checks,
                state="rejected" if status == "rejected" else status)
    if category and status != "done":
        from .process import PROCESSING_REJECTION_CATEGORIES
        view["reason"] = PROCESSING_REJECTION_CATEGORIES.get(category, category)
    return view


def _stage_checks(sid, run, record):
    if sid == "broaden":
        ratios = record.get("details", {}).get("verification") or {}
        temps = _temps(list(ratios) or run.temperatures)
        worst = max((r for per in ratios.values() for r in per.values()), default=None)
        share = _doppler_share(run.plan)
        if worst is not None and worst > 1:
            return [_check("doppler", FAIL, f"Outside its share at {temps}.", stage=sid)]
        return [_check("doppler", PASS, f"Every reaction at {temps} is within its share "
                                        f"({_share(share)} of the tolerance)." if share else
                                        f"Every reaction at {temps} is within its tolerance.", stage=sid)]
    if sid == "urr":
        rows = _urr_rows(run)
        if not rows:
            return []
        missed = [(r["temperature_K"], e) for r in rows for e in r["missed_eV"]]
        worst = max(r["worst_error"] for r in rows)
        target = rows[0]["target"]
        n = sum(r["energies"] for r in rows)
        result = (f"{n} tables at {len(rows)} temperature{'s' if len(rows) > 1 else ''}; worst statistical "
                  f"error {_share(worst, 2)}" + (f" (target {_share(target, 1)})" if target else "") + ".")
        if missed:
            return [_check("urr_statistics", WARN, result,
                           [f"{T:g} K, {_energy(e)}: stopped at the ladder limit before reaching the target."
                            for T, e in missed], stage=sid)]
        return [_check("urr_statistics", PASS, result, stage=sid)]
    if sid == "heating":
        rows = _heating_rows(run)
        bad = [r for r in rows if r["negative_points"]]
        if not bad:
            return [_check("heating_sign", PASS, f"KERMA and damage at {_temps([r['temperature_K'] for r in rows])}: "
                                                 "no negative value.", stage=sid)]
        regions = record.get("details", {}).get("negative") or {}
        # One temperature tells the story (the evaluation's balance does not
        # depend on it): the first broadened one, else 0 K.
        key = next((k for k in regions if float(k) > 0), next(iter(regions), None))
        worst = min(r["minimum_per_collision_eV"] for r in bad if r.get("minimum_per_collision_eV") is not None)             if any(r.get("minimum_per_collision_eV") is not None for r in bad) else None
        found = regions.get(key, [])
        material = [g for g in found if g.get("material", True)]
        result = (f"Negative heating in {len(found) or 'some'} energy region{'s' if len(found) != 1 else ''}"
                  + (f", down to {_energy(worst)} per collision" if worst is not None else "")
                  + (f"; {len(found) - len(material)} of them below 0.1 % of the incident energy"
                     if found and len(material) < len(found) else "") + ".")
        items = []
        for g in sorted(found, key=lambda g: g["per_collision_eV"])[:6]:
            where = (f"at {_energy(g['from_eV'])}" if g["from_eV"] == g["to_eV"]
                     else f"{_energy(g['from_eV'])} to {_energy(g['to_eV'])}")
            cause = "; ".join(f"{c['what']} ({_energy(c['per_collision_eV'])} per collision)" for c in g["causes"])
            items.append(f"{where} ({float(key):g} K): down to {_energy(g['per_collision_eV'])} per collision"
                         + ("" if g["from_eV"] == g["to_eV"] else f" at {_energy(g['at_eV'])}")
                         + (f". Its reaction balance is negative: {cause}." if cause else ".")
                         + ("" if g.get("material", True) else " Below 0.1 % of the incident energy."))
        if not items:
            items = [f"{r['temperature_K']:g} K: {r['negative_points']:,} energies between "
                     f"{_energy(r['negative_from_eV'])} and {_energy(r['negative_to_eV'])}." for r in bad]
        return [_check("heating_sign", WARN if (material or not found) else NOTE, result, items, stage=sid)]
    return []


def _share(fraction, digits=0):
    """``0.05`` -> ``5 %``, written as the rest of the report writes percentages."""
    return f"{100 * float(fraction):.{digits}f} %"


def _doppler_share(plan):
    if not plan.thinning:
        return None
    from kika.hdf5.processing import heatedTolerances
    return heatedTolerances(plan.thinning)[0] / float(plan.thinning)


def _urr_rows(run):
    rows = []
    for T, table in sorted((run._urr or {}).items()):
        st = getattr(table, "statistics", None)
        if st is None:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            shielding = np.where(np.abs(st.shielding) > 0, st.shieldingSigma / np.abs(st.shielding), 0.0)
        worst = float(np.nanmax([np.nanmax(shielding), np.nanmax(st.contributionSigma)]))
        converged = np.ones(len(table.energies), bool) if st.converged is None else np.asarray(st.converged, bool)
        ladders = np.asarray(st.ladders)
        rows.append(dict(temperature_K=float(T), energies=int(len(table.energies)), bands=int(table.bands),
                         ladders_min=int(ladders.min()), ladders_median=float(np.median(ladders)),
                         ladders_max=int(ladders.max()), reached=int(converged.sum()),
                         missed_eV=[float(e) for e, ok in zip(table.energies, converged) if not ok],
                         worst_error=worst, target=st.errorTarget))
    return rows


def _doppler_rows(run):
    stage = next((s for s in run.stages if s.stage == "broaden" and s.status == "done"), None)
    if stage is None:
        return []
    share = _doppler_share(run.plan)
    points = stage.details.get("n_points", {})
    return [dict(temperature_K=float(T), up_to_eV=run.thnmax,
                 points=int(points.get(T, points.get(str(T), 0)) or 0), share=share)
            for T in run.temperatures]


def _heating_rows(run):
    rows = []
    for T, tables in sorted((run._heating or {}).items()):
        x, h = tables[301]
        neg = h < 0
        i = int(np.argmin(h)) if len(h) else 0
        per_collision = None
        if neg.any():
            # The same energy as a reader of HEATR's output sees it: per collision.
            try:
                xt, st = run.cross_sections(T or None)[1]
                total = float(np.interp(x[i], xt, st))
                per_collision = float(h[i]) / total if total > 0 else None
            except (KeyError, IndexError):
                per_collision = None
        rows.append(dict(temperature_K=float(T), points=int(len(x)), negative_points=int(neg.sum()),
                         minimum=float(h[i]) if neg.any() else None,
                         minimum_at_eV=float(x[i]) if neg.any() else None,
                         negative_from_eV=float(x[neg].min()) if neg.any() else None,
                         negative_to_eV=float(x[neg].max()) if neg.any() else None,
                         minimum_per_collision_eV=per_collision))
    return rows


def _output(record):
    d = record.get("details") or {}
    fmt = d.get("format") or "file"
    files = d.get("files") or ([d["path"]] if d.get("path") else [])
    size = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) or None
    omitted = list(d.get("omitted") or [])
    written = record.get("status") == "done"
    name = _FORMAT_NAMES.get(fmt, fmt)
    if written:
        status, result = (NOTE if omitted else PASS), f"{_files_text(files)} {_VERIFIED_BY.get(fmt, 'verified')}."
    else:
        status, result = FAIL, f"not written: {record.get('summary') or 'refused'}."
    temps = [float(t) for t in d.get("temperatures") or []]
    # An ACE table is one temperature: the tables come in the order of the temperatures.
    per_file = ({str(f): [t] for f, t in zip(files, temps)} if fmt in ("ace", "dosimetry")
                and len(files) == len(temps) else {str(f): temps for f in files})
    return dict(format=fmt, name=name, files=[str(f) for f in files], directory=d.get("path"),
                temperatures=temps, file_temperatures=per_file, size_bytes=size,
                written=written, seconds=record.get("seconds"), verification=_VERIFIED_BY.get(fmt),
                omitted=omitted, error=d.get("error"), status_check=status, result=result,
                items=[f"Left out: {o}" for o in omitted])


def _file_temps(output, f):
    temps = output.get("file_temperatures", {}).get(str(f), output["temperatures"])
    return ", ".join(f"{t:g} K" for t in temps) or "-"


def _files_text(files):
    names = [Path(f).name for f in files]
    if not names:
        return "Written;"
    return (", ".join(names) if len(names) <= 3 else f"{len(names)} files") + ":"


def _outputs_line(outputs):
    if not outputs:
        return "No file was written in this run."
    done = [o for o in outputs if o["written"]]
    text = "; ".join(f"{o['name']} ({', '.join(Path(f).name for f in o['files'])})" for o in done)
    failed = [o["name"] for o in outputs if not o["written"]]
    return (f"Written and read back: {text}." if done else "") + (
        f" Not written: {', '.join(failed)}." if failed else "")


def _temps(values):
    values = [float(v) for v in values]
    return (", ".join(f"{v:g}" for v in values[:-1]) + f" and {values[-1]:g} K") if len(values) > 1 \
        else (f"{values[0]:g} K" if values else "-")


def _stage_line(sid, record, run, rd):
    """One plain sentence of what the stage produced, or why it did not run."""
    d = record.get("details") or {}
    status = record.get("status")
    if status in (None, "skipped") and not record.get("category"):
        return "Not requested."
    if status in ("skipped", "rejected"):
        from .process import PROCESSING_REJECTION_CATEGORIES
        return PROCESSING_REJECTION_CATEGORIES.get(record.get("category"), record.get("summary") or "")
    if sid == "read":
        ev = rd["evaluation"]
        return (f"{_library(ev)} evaluation of {_nuclide(rd)} (MAT {_s(rd['material'].get('mat'))}), "
                f"{len(rd['reactions'])} reactions.")
    if sid == "reconstruct":
        parts = []
        res = [r for r in rd["resonance_ranges"] if r["lru"] == 1]
        if res:
            n = sum(r.get("resonances") or 0 for r in res)
            parts.append(f"{n:,} resolved resonances ({', '.join(sorted({r['formalism'] for r in res}))}) "
                         f"up to {_energy(max(r['eh_eV'] for r in res))}")
        if any(r["lru"] == 2 for r in rd["resonance_ranges"]):
            urr = [r for r in rd["resonance_ranges"] if r["lru"] == 2]
            parts.append(f"unresolved range {_span(min(r['el_eV'] for r in urr), max(r['eh_eV'] for r in urr))}")
        parts.append(f"{_count(rd['points']['stored'])} points in all")
        return "; ".join(parts) + "."
    if sid == "broaden":
        points = d.get("n_points") or {}
        return (f"To {_temps(d.get('temperatures') or run.temperatures)}, up to {_energy(d.get('thnmax_eV'))}; "
                + ", ".join(f"{_count(n)} points at {float(T):g} K" for T, n in points.items()) + ".")
    if sid == "urr":
        rows = _urr_rows(run)
        if rows:
            r = rows[0]
            return (f"{r['bands']} bands at {r['energies']} energies, at {_temps([x['temperature_K'] for x in rows])}; "
                    f"{r['ladders_median']:g} ladders per energy (median).")
    if sid == "heating":
        temps = d.get("temperatures") or []
        dpa = sorted({name for v in (run.displacements or {}).values() for name in v})
        return (f"KERMA (MT301) and damage energy (MT444) at {_temps(temps)}"
                + (f"; displacements per atom ({', '.join(dpa)})" if dpa else "") + ".")
    return record.get("summary") or ""


def _detail(rd, run, outputs, counts):
    temps = run.temperatures
    what = f"{_nuclide(rd)} ({_library(rd['evaluation'])})" + (f" at {_temps(temps)}" if temps else " at 0 K")
    done = [o for o in outputs if o["written"]]
    files = (" Files written and read back: " + ", ".join(
        f"{o['name']} ({len(o['files'])} table{'s' if len(o['files']) > 1 else ''})" if o["format"] == "ace"
        else o["name"] for o in done) + ".") if done else " No file written."
    failed = [o["name"] for o in outputs if not o["written"]]
    if failed:
        files += f" Not written: {', '.join(failed)}."
    tally = [f"{counts[FAIL]} failed" if counts[FAIL] else "",
             f"{counts[WARN]} warning{'s' if counts[WARN] != 1 else ''}" if counts[WARN] else "no warnings",
             f"{counts[NOTE]} note{'s' if counts[NOTE] != 1 else ''}" if counts[NOTE] else ""]
    return f"{what}.{files} " + ", ".join(t for t in tally if t).capitalize() + "."


def _choices(run, rd, outputs):
    """``[{topic, choice, why}]``: what kika decided for the user, for the stages that ran."""
    plan, out = run.plan, []
    ran = {s.stage for s in run.stages if s.status == "done"}
    temps = list(plan.temperatures)
    from .process import DEFAULT_TEMPERATURES
    out.append(dict(topic="Temperatures", choice=_temps(temps) if temps else "0 K only",
                    why="Room temperature, as the official libraries state it (293.6 K), unless others "
                        "were asked for." if tuple(temps) == DEFAULT_TEMPERATURES else "As asked for."))
    tol = _tolerance_text(rd)
    if "broaden" in ran and plan.thinning:
        share = _doppler_share(plan)
        out.append(dict(
            topic="Accuracy",
            choice=f"Every final table within {_share(plan.thinning, 1)} of the cross section, five times "
                   f"tighter below 0.5 eV, from the evaluation to the file. Half of it goes to the 0 K "
                   f"reconstruction, {_share(share)} to the broadening, the rest to thinning the grid.",
            why="NJOY applies its tolerance once in each module, so its errors add up; kika splits one "
                "tolerance between the stages, so the tables are within the one stated."))
    else:
        out.append(dict(topic="Accuracy", choice=f"The 0 K tables within {tol}.",
                        why="Half of it is used to build the grid; the other half is left for rounding the "
                            "values when a file is written."))
    if "broaden" in ran:
        out.append(dict(topic="Broadening range",
                        choice=f"Up to {_energy(run.thnmax)}; above it the 0 K cross sections are used.",
                        why="As NJOY's BROADR: above the resolved resonances the cross sections are smooth, "
                            "and broadening does not change them."))
    if "urr" in ran:
        out.append(dict(topic="Probability tables",
                        choice=f"{plan.urr_bins} bands; at least 64 ladders per energy, more until the "
                               f"statistical error is 0.5 % (at most {plan.urr_ladders}); widths drawn from "
                               "the exact chi-square distribution.",
                        why="A fixed number of ladders leaves some materials far less precise than others, "
                            "and NJOY's 20-value width table biases the self-shielding."))
    if "heating" in ran:
        out.append(dict(topic="Heating",
                        choice="Photons carried away (NJOY's local = 0); damage energy with the displacement "
                               "energy of each element; displacements per atom by NRT"
                               + (" and arc-dpa" if any("arc-dpa" in v for v in (run.displacements or {}).values())
                                  else "") + ".",
                        why="The transport code deposits the photons where they end; the damage model is the "
                            "one HEATR uses, with arc-dpa added where its constants exist (Fe, Ni, Cu, Pd, Pt, W)."))
    if any(o["format"] == "ace" and o["written"] for o in outputs):
        out.append(dict(topic="ACE tables",
                        choice="Named ZAID plus the suffix of the temperature (.02c at 293.6 K), legacy "
                               "header, one xsdir line each.",
                        why="What MCNP and Serpent read, and the names NJOY's libraries use."))
    return out


def _all_steps(run, rd, stages, outputs):
    steps = [s for s in _steps(rd) if not s.startswith(("Checked the tables", "Wrote "))]
    st = {s["id"]: s for s in stages}
    if st["broaden"]["status"] == "done":
        steps.append(f"Broadened every reaction to {_temps(run.temperatures)} with the free-gas model, up to "
                     f"{_energy(run.thnmax)}, and thinned each temperature's grid to the points the "
                     "accuracy needs.")
    if st["urr"]["status"] == "done":
        steps.append(f"Computed the probability tables of the unresolved range at {_temps(run.temperatures)}.")
    if st["heating"]["status"] == "done":
        steps.append("Computed the heating (KERMA) and damage energy from the evaluation's outgoing "
                     "particles and their distributions, at every temperature and at 0 K.")
    steps.append("Checked every stage (the checks above).")
    for o in outputs:
        if o["written"]:
            steps.append(f"Wrote {o['name']}: {', '.join(Path(f).name for f in o['files'])}; {o['verification']}.")
        else:
            steps.append(f"Tried to write {o['name']}, which was refused.")
    return steps


def _revision():
    """The git commit of a kika run from a checkout, else ``None``."""
    import subprocess
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


# --------------------------------------------------------------------------- object

class ProcessingReport:
    """A processing report. Build it with ``EndfProcessing.report_document()``.

    ``to_text()``, ``to_html()``, ``to_dict()``, ``to_json()``, ``save(path)``."""

    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self._data))

    def to_json(self, **kwargs) -> str:
        return json.dumps(self._data, **({"indent": 1} | kwargs))

    def to_text(self, **kwargs) -> str:
        return _render_text(self._data)

    def to_html(self, **kwargs) -> str:
        return _render_html(self._data)

    def save(self, path):
        """Write by extension: ``.html``/``.htm``, ``.json``, anything else as text."""
        path = Path(path)
        suffix = path.suffix.lower()
        body = (self.to_html() if suffix in (".html", ".htm") else
                self.to_json() if suffix == ".json" else self.to_text())
        path.write_text(body, encoding="utf-8")
        return path

    def __repr__(self):
        s = self._data["summary"]
        return f"<ProcessingReport {self._data['header']['name']}: {s['headline'].lower()}>"

    def _repr_html_(self):
        return _render_html(self._data, fragment=True)


# --------------------------------------------------------------------------- shared pieces

_STATE_MARK = {"done": PASS, "rejected": FAIL, "skipped": SKIP, "not requested": SKIP}


def _stage_status(st):
    """The badge status of a stage: its worst check when it ran."""
    if st["state"] == "rejected":
        return FAIL
    if st["status"] != "done":
        return SKIP
    return max((c["status"] for c in st["checks"]), key=_RANK.get, default=PASS)


def _title(d):
    h = d["header"]
    return h["name"] + (f"  MAT {h['mat']}" if h.get("mat") else "")


def _stage_short(st):
    short = {"read": "Read", "reconstruct": "0 K", "broaden": "Doppler", "urr": "URR tables",
             "heating": "Heating", "write": "Files", "thermal": "Thermal tables",
             "tables": "Table"}.get(st["id"], st["title"])
    if st["status"] == "not requested":
        return f"{short}: not requested"
    if st["status"] == "skipped":
        return f"{short}: not applicable"
    return short


def _detail_tables(d):
    """``[(title, head, rows, numeric)]`` for the folded details."""
    rd = d["reconstruction"]
    out = []
    if d["doppler"]:
        out.append(("Doppler broadening", ("Temperature", "Broadened up to", "Points", "Share of the tolerance"),
                    [(f"{r['temperature_K']:g} K", _energy(r["up_to_eV"]), _count(r["points"]),
                      _share(r["share"]) if r["share"] else "-") for r in d["doppler"]], (2, 3)))
    if d["urr"]:
        out.append(("Probability tables", ("Temperature", "Energies", "Bands", "Ladders (min–median–max)",
                                           "Reached the target", "Worst error"),
                    [(f"{r['temperature_K']:g} K", str(r["energies"]), str(r["bands"]),
                      f"{r['ladders_min']}–{r['ladders_median']:g}–{r['ladders_max']}",
                      f"{r['reached']} of {r['energies']}", _share(r["worst_error"], 2)) for r in d["urr"]],
                    (1, 2, 4, 5)))
    if d["heating"]:
        out.append(("Heating", ("Temperature", "Points", "Negative values"),
                    [(f"{r['temperature_K']:g} K", _count(r["points"]), _count(r["negative_points"]))
                     for r in d["heating"]], (1, 2)))
    times = [(st["title"], f"{st['seconds']:.1f} s") for st in d["stages"] if st.get("seconds")]
    if times:
        out.append(("Time per stage", ("Stage", "Time"), times, (1,)))
    if rd and rd["resonance_ranges"]:
        out.append(("Resonance ranges", ("Energy range", "Kind", "Model", "Content", "How it was used"),
                    [r[:5] for r in _range_rows(rd)], ()))
    if rd and rd["sums"]["graph"]:
        out.append(("Totals rebuilt from their parts", ("Total", "Sum of"),
                    [(g["name"], g["parts_text"]) for g in rd["sums"]["graph"]], ()))
    return out


def _reaction_split(rd):
    """The reaction rows worth a look (a quarter of the tolerance or more, or a remark) and the rest."""
    head, rows = _reaction_rows(rd)
    keep, rest = [], []
    for r, row in zip(rd["reactions"], rows):
        notable = (r["tabulation_verification_max"] or 0) >= 0.25 or (r.get("publication_verification_max") or 0) >= 0.25 \
            or bool(r["negatives"])
        (keep if notable else rest).append(row)
    return head, keep, rest


def _run_items(d):
    rd, sw = d["reconstruction"], d["software"]
    total = sum(st["seconds"] or 0.0 for st in d["stages"])
    version = sw["kika_version"] + (f" (commit {sw['kika_revision']})" if sw.get("kika_revision") else "")
    workers = d["plan"].get("workers")
    run = ("Run", [("Generated", d["generated_utc"].replace("T", " ").replace("+00:00", " UTC")),
                   ("kika", version), ("Total time", f"{total:.1f} s")])
    if rd is None:
        h = d["header"]
        return [("Evaluation", [("Kind", h["kind"]), ("File", _s(h.get("path") or h.get("file"))),
                                ("SHA-256", _s(h.get("sha256")))]), run]
    m, ev = rd["material"], rd["evaluation"]
    return [
        ("Material", [("Nuclide", f"{_nuclide(rd)} (ZA {_s(m.get('za'))}, MAT {_s(m.get('mat'))})"),
                      ("Target spin", _g(m.get("target_spin")))]),
        ("Evaluation", [("Library", _library(ev)), ("Laboratory", _s(ev.get("laboratory"))),
                        ("Date", _s(ev.get("evaluation_date"))), ("Authors", _s(ev.get("authors"))),
                        ("File", _s(ev.get("path") or ev.get("file"))), ("SHA-256", _s(ev.get("sha256")))]),
        ("Run", [("Generated", d["generated_utc"].replace("T", " ").replace("+00:00", " UTC")),
                 ("kika", version),
                 ("Compiled kernels", "used" if sw.get("native_kernel") else "not available: pure-Python "
                                                                             "fallback, same results, slower"),
                 ("Workers", "default" if workers is None else str(workers)),
                 ("Total time", f"{total:.1f} s")]),
    ]


def _legend(d):
    """The checks this report used, in order: the reconstruction's, then its own."""
    from .reconstruction_report import CHECKS as RECON
    used = {c["id"] for c in d["checks"]}
    return [c for c in RECON if c.id in used] + [c for c in CHECKS if c.id in used]


# --------------------------------------------------------------------------- text

def _render_text(d, width=100):
    out, s, rd = [], d["summary"], d["reconstruction"]
    bar = "=" * width
    out += [bar, f" kika processing report  -  {_title(d)}", bar, "",
            f" {_TEXT_MARK[s['status']] if not s['incomplete'] else _TEXT_MARK[FAIL]} {s['headline'].upper()}"]
    out += _wrap(s["detail"], width, "   ")
    out.append("")
    out += _wrap("   ".join(f"{_TEXT_MARK[_stage_status(st)]} {_stage_short(st)}" for st in d["stages"]),
                 width, "   ")

    def section(title):
        out.extend(["", f" {title.upper()}", " " + "-" * len(title)])

    def table(head, rows):
        if not rows:
            out.append("   (none)")
            return
        cells = [list(head)] + [[_ascii(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(head))]
        fmt = lambda r: "   " + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
        out.extend([fmt(cells[0]), "   " + "  ".join("-" * w for w in widths)] + [fmt(r) for r in cells[1:]])

    section("Checks")
    for st in d["stages"]:
        if not st["checks"] and st["status"] != "done":
            out.append(f"   {st['title']}: {st['summary']}")
            continue
        out.append(f"   {st['title']}" + (f"  ({st['seconds']:.1f} s)" if st.get("seconds") else ""))
        for c in st["checks"]:
            out += _wrap(c["result"], width, "            ", first=f"     {_TEXT_MARK[c['status']]} {c['title']}: ")

    flagged = [c for c in d["checks"] if c["items"]]
    if flagged:
        section("Findings")
        for c in flagged:
            out.append(f"   {_TEXT_MARK[c['status']]} {c['title']}")
            meaning = (_CHECK.get(c["id"]) or _RECON_CHECK[c["id"]]).levels.get(c["status"])
            if meaning:
                out += _wrap(meaning, width, "          ")
            for item in c["items"]:
                out += _wrap(item, width, "          ", first="        - ")

    section("Files written")
    if d["outputs"]:
        folders = sorted({str(Path(f).parent) for o in d["outputs"] for f in o["files"]})
        table(("Format", "File", "Temperatures", "Size", "Status"),
              [(o["name"], Path(f).name, _file_temps(o, f),
                _size(os.path.getsize(f) if os.path.isfile(f) else None), "verified" if o["written"] else "not written")
               for o in d["outputs"] for f in (o["files"] or [o["directory"] or "-"])])
        for folder in folders:
            out += _wrap(f"In {folder}", width, "     ", first="   * ")
    else:
        out.append("   No file was written in this run.")

    section("Choices made for you")
    for c in d["choices"]:
        out += _wrap(f"{c['topic']}: {c['choice']}", width, "     ", first="   - ")
        out += _wrap(f"Why: {c['why']}", width, "     ", first="     ")

    section("What was done")
    for i, step in enumerate(d["steps"], 1):
        out += _wrap(step, width, "      ", first=f"   {i}. ")

    section("Details")
    for title, head, rows, _ in _detail_tables(d):
        out += ["", f"   {title}"]
        table(head, rows)
    if rd is not None:
        head, keep, rest = _reaction_split(rd)
        out += ["", "   Reactions at 0 K" + (f" ({len(rest)} more within a quarter of the tolerance not listed)"
                                             if rest and keep else "")]
        table(head, keep or rest)
        out += _wrap(_TOLERANCE_NOTE.format(tol=_tolerance_text(rd)), width, "     ", first="   * ")

    section("Run details")
    for heading, items in _run_items(d):
        out.append(f"   {heading}")
        k = max(len(a) for a, _ in items)
        out += [f"     {a.ljust(k)} : {_ascii(b)}" for a, b in items]

    section("About the checks")
    for c in _legend(d):
        out += ["", f"   {c.title}"]
        out += _wrap(c.what, width, "     ")
        for lv in (PASS, NOTE, WARN, FAIL, SKIP):
            if lv in c.levels:
                out += _wrap(c.levels[lv], width, "            ", first=f"     {_TEXT_MARK[lv]} ")
    out += ["", bar]
    return "\n".join(_ascii(line) for line in out) + "\n"


def _size(n):
    if n is None:
        return "-"
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.3g} {unit}"
        n /= 1000.0


# --------------------------------------------------------------------------- html

_EXTRA_CSS = """
.kika-report .strip{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 6px}
.kika-report .strip .badge{font-weight:600}
.kika-report table.checks tr.grp td{background:var(--panel);font-weight:700;font-size:13px;color:var(--muted);
text-transform:uppercase;letter-spacing:.04em}
.kika-report table.checks tr.grp td span{font-weight:400;text-transform:none;letter-spacing:0;margin-left:8px}
.kika-report .choices dt{font-weight:700;margin-top:10px}.kika-report .choices dd{margin:2px 0 0}
.kika-report .choices dd.why{color:var(--muted);font-size:14px}
"""


def _render_html(d, fragment=False):
    s, rd = d["summary"], d["reconstruction"]
    status = FAIL if s["incomplete"] else s["status"]
    css_class = _HEADLINE[status][1]
    h = d["header"]
    nuclide = h["name"]
    out = ['<div class="kika-report">',
           f'<p class="eyebrow">kika · processing report · {_e(h["kind"])}</p>',
           f'<h1>{_e(nuclide)}' + (f' <span class="muted">· MAT {_e(h["mat"])}</span>' if h.get("mat") else "")
           + "</h1>",
           f'<p class="sub">' + (f'{_e(h["library"])} · ' if h.get("library") else "") + f'{_e(h.get("file"))} · '
           f'kika {_e(d["software"]["kika_version"])} · '
           f'{_e(d["generated_utc"].replace("T", " ").replace("+00:00", " UTC"))}</p>',
           f'<div class="verdict {css_class}"><span class="mark">{_MARK[status]}</span>'
           f'<div><p class="big">{_e(s["headline"])}</p><p>{_e(s["detail"])}</p></div></div>',
           '<div class="strip">' + "".join(
               f'<span class="badge {_CLASS[_stage_status(st)]}"><span class="mark">{_MARK[_stage_status(st)]}'
               f'</span>{_e(_stage_short(st))}</span>' for st in d["stages"]) + "</div>"]
    flagged = [c for c in d["checks"] if c["items"]]
    toc = [("checks", "Checks")] + ([("findings", "Findings")] if flagged else []) + [
        ("files", "Files"), ("choices", "Choices made for you"), ("done", "What was done"),
        ("details", "Details"), ("run", "Run details"), ("about", "About the checks")]
    out.append('<nav class="toc">' + "".join(f'<a href="#kp-{a}">{_e(t)}</a>' for a, t in toc) + "</nav>")

    out.append('<h2 id="kp-checks">Checks</h2>')
    out.append('<p class="lead">What was verified, stage by stage. Each check is explained under '
               '<a href="#kp-about">About the checks</a>.</p>')
    rows = []
    for st in d["stages"]:
        when = f"{st['seconds']:.1f} s" if st.get("seconds") else ""
        rows.append(f'<tr class="grp"><td colspan="3">{_e(st["title"])}<span>{_e(when)}</span></td></tr>')
        if not st["checks"]:
            rows.append(f'<tr><td>{_badge(SKIP if st["status"] != "done" else PASS)}</td>'
                        f'<td colspan="2" class="muted">{_e(st["summary"])}</td></tr>')
        for c in st["checks"]:
            rows.append(f'<tr><td>{_badge(c["status"])}</td><td><a class="ttl" href="#kp-check-{c["id"]}">'
                        f'{_e(c["title"])}</a></td><td>{_e(c["result"])}</td></tr>')
    out.append('<div class="wrap"><table class="checks"><thead><tr><th>Status</th><th>Check</th><th>Result</th>'
               "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    if flagged:
        out.append('<h2 id="kp-findings">Findings</h2>')
        out.append('<p class="lead">The cases behind each note, warning or failure, and what they mean.</p>')
        for c in flagged:
            out.append(f'<div class="group"><div class="ghead">{_badge(c["status"])}<span class="ttl">'
                       f'{_e(c["title"])}</span><span class="muted">{len(c["items"])} case(s)</span></div>')
            meaning = (_CHECK.get(c["id"]) or _RECON_CHECK[c["id"]]).levels.get(c["status"])
            if meaning:
                out.append(f'<p class="muted">{_e(meaning)}</p>')
            shown = c["items"][:8]
            out.append('<ul class="items">' + "".join(f"<li>{_e(x)}</li>" for x in shown) + "</ul>")
            if len(c["items"]) > len(shown):
                out.append(f'<details><summary>Show the other {len(c["items"]) - len(shown)}</summary>'
                           '<ul class="items">' + "".join(f"<li>{_e(x)}</li>" for x in c["items"][8:])
                           + "</ul></details>")
            out.append("</div>")

    out.append('<h2 id="kp-files">Files written</h2>')
    if d["outputs"]:
        rows = [(o["name"], _Raw(f"<code>{_e(Path(f).name)}</code><br><span class=\"faint\">{_e(Path(f).parent)}</span>"),
                 _file_temps(o, f),
                 _size(os.path.getsize(f) if os.path.isfile(f) else None),
                 _Raw(_badge(o["status_check"]) + f' <span class="muted">{_e(o["verification"] or "")}</span>'
                      if o["written"] else _badge(FAIL) + f' <span class="muted">{_e(o["result"])}</span>'))
                for o in d["outputs"] for f in (o["files"] or [o["directory"] or "-"])]
        out.append(_html_table(("Format", "File", "Temperatures", "Size", "Verification"), rows,
                               numeric=(3,), nowrap=(0, 2)))
    else:
        out.append('<p class="muted">No file was written in this run.</p>')

    out.append('<h2 id="kp-choices">Choices made for you</h2>')
    out.append('<p class="lead">What kika decided so that nobody has to write an input deck, and why.</p>')
    out.append('<dl class="choices">' + "".join(
        f'<dt>{_e(c["topic"])}</dt><dd>{_e(c["choice"])}</dd><dd class="why">{_e(c["why"])}</dd>'
        for c in d["choices"]) + "</dl>")

    out.append('<h2 id="kp-done">What was done</h2>')
    out.append('<ol class="steps">' + "".join(f"<li>{_e(x)}</li>" for x in d["steps"]) + "</ol>")

    out.append('<h2 id="kp-details">Details</h2>')
    out.append('<p class="lead">The numbers behind the checks, for whoever wants them.</p>')
    for title, head, rows, numeric in _detail_tables(d):
        out.append(f"<details><summary>{_e(title)}</summary>"
                   + _html_table(head, rows, numeric=numeric, nowrap=(0,)) + "</details>")
    if rd is not None:
        head, keep, rest = _reaction_split(rd)
        numeric = {i for i, h in enumerate(head) if h in ("MT", "Points", "Starts at", "Table error", "PENDF error",
                                                          "Integral error")}
        body = (f'<p class="muted">{_e(_TOLERANCE_NOTE.format(tol=_tolerance_text(rd)))}</p>'
                + (_html_table(head, keep, numeric=numeric) if keep else "")
                + (f"<details><summary>{'The other ' if keep else 'All '}{len(rest)} reactions, each within a "
                   f"quarter of the tolerance</summary>{_html_table(head, rest, numeric=numeric, tall=True)}</details>"
                   if rest else ""))
        out.append(f"<details><summary>Reactions at 0 K ({len(rd['reactions'])})</summary>{body}</details>")
        out.append("<details><summary>Resonance engine: how the energy range was divided and refined</summary>"
                   + _html_table(_SEGMENT_HEAD, _segment_rows(rd, False), numeric=(2, 3, 4, 5), nowrap=(0,))
                   + ("".join(['<ul class="notes">'] + [f"<li><code>{_e(n)}</code></li>" for n in _technical_notes(rd)]
                              + ["</ul>"]) if _technical_notes(rd) else "") + "</details>")
        _, other = _conversion_scope(rd)
        if other:
            out.append(f"<details><summary>{len(other)} message(s) about parts of the file the 0 K reconstruction "
                       "does not use</summary><ul class=\"notes\">"
                       + "".join(f"<li>{_e(m)}</li>" for _, m in other) + "</ul></details>")

    out.append('<h2 id="kp-run">Run details</h2>')
    out.append('<div class="meta">' + "".join(
        f"<section><h3>{_e(h)}</h3><dl>" + "".join(
            f"<dt>{_e(a)}</dt><dd>{f'<code>{_e(b)}</code>' if a in ('SHA-256', 'File') else _e(b)}</dd>"
            for a, b in items) + "</dl></section>" for h, items in _run_items(d)) + "</div>")

    out.append('<h2 id="kp-about">About the checks</h2>')
    if rd is not None:
        out.append(f'<div class="tol"><strong>Tolerance.</strong> {_e(_TOLERANCE_NOTE.format(tol=_tolerance_text(rd)))}'
                   "</div>")
    for c in _legend(d):
        out.append(f'<div class="entry" id="kp-check-{c.id}"><div><h3>{_e(c.title)}</h3><p>{_e(c.what)}</p></div><ul>'
                   + "".join(f"<li>{_badge(lv)}<span>{_e(c.levels[lv])}</span></li>"
                             for lv in (PASS, NOTE, WARN, FAIL, SKIP) if lv in c.levels) + "</ul></div>")
    out.append("</div>")
    content = "<style>" + _CSS + _EXTRA_CSS + "</style>" + "\n".join(out)
    if fragment:
        return content
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="light dark">'
            f"<title>Processing of {_e(nuclide)}</title></head>"
            '<body style="margin:0;background:Canvas">' + content + "</body></html>")
