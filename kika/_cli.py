"""The ``kika`` command: ``kika process`` and ``kika check`` on the command line.

::

    kika process n-092_U_235.endf --to ace hdf5 -o out/
    kika process endfb81/neutrons/ --to ace -o lib/ --temperatures 293.6 600
    kika process tsl-HinH2O.endf -o thermal/

    kika check n-026_Fe_056.endf                    # data and covariances
    kika check n-026_Fe_056.endf --what data --mf 3
    kika check endfb81/neutrons/ --library endfb81 -o reports/

``kika process``: each evaluation goes through :func:`kika.process`; every file
asked for is written into the output folder, and the report beside it
(``<name>.report.html``). One line per evaluation says how it went. The exit
status is 0 when every evaluation was processed and verified (notes and
warnings included), 1 when any was incomplete or refused, 2 for a usage error.

``kika check``: each file, or each folder as a library, goes through
:func:`kika.check`; the report is written into the output folder as one HTML
page (``<name>.check.html``), the format every kika report is read in. One line
per evaluation says what was found. The exit status is 1
when any evaluation has a finding at ``--fail-on`` (``defect`` by default) or
could not be checked, 0 otherwise, 2 for a usage error.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kika", description="kika nuclear data tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "process", help="process evaluations into transport files",
        description="Process ENDF-6 or GNDS evaluations (neutron, thermal scattering, photo-atomic, "
                    "photonuclear) into ACE, OpenMC HDF5, PENDF or GNDS, verified, with a report each. "
                    "kika chooses the settings; the report says which and why.")
    p.add_argument("sources", nargs="+", help="evaluation files, or folders whose evaluations to process")
    p.add_argument("--to", nargs="+", default=["ace"], metavar="FORMAT",
                   help="formats to write: ace (default), hdf5, pendf, gnds, dosimetry")
    p.add_argument("-o", "--output", default=".", help="output folder (default: the current one)")
    p.add_argument("--temperatures", nargs="+", type=float, metavar="K",
                   help="temperatures in kelvin (default: 293.6 K; thermal: the evaluation's)")
    p.add_argument("--kind", choices=("neutron", "thermal", "photoatomic", "photonuclear", "dosimetry"),
                   help="the kind of evaluation, when the header should not decide it")
    p.add_argument("--heating", action="store_true", help="neutron: also keep KERMA and damage in the PENDF")
    p.add_argument("--workers", type=int, help="processes or threads (default: the machine's)")
    p.add_argument("--replace", action="store_true", help="replace tables the output xsdir already names")
    p.add_argument("--no-report", action="store_true", help="do not write the report")
    c = commands.add_parser(
        "check", help="check evaluations: data, covariances, or both",
        description="Check ENDF-6 (or GNDS) evaluations against the format, mathematics and physics: "
                    "the data of every file (MF1-MF28) and the covariances (MF31-35, MF40). Nothing "
                    "is changed; each finding is a note, a warning or a defect, with its evidence.")
    c.add_argument("sources", nargs="+", help="evaluation files, or folders checked as libraries")
    c.add_argument("--what", choices=("all", "data", "covariances"), default="all",
                   help="which checks to run (default: all)")
    c.add_argument("--mf", nargs="+", type=int, metavar="MF", help="only these files")
    c.add_argument("--deep", action="store_true",
                   help="also reconstruct the cross sections and check what codes will use (slower)")
    c.add_argument("-o", "--output", default=".", help="output folder (default: the current one)")
    c.add_argument("--level", choices=("note", "warn", "defect"), default="warn",
                   help="lowest level listed in the report (counts always cover every level)")
    c.add_argument("--library", help="a name for a folder's report")
    c.add_argument("--workers", type=int, help="processes checking a folder's tapes at once "
                                               "(default: the machine's, as for processing)")
    c.add_argument("--recursive", action="store_true", help="look in subfolders too")
    c.add_argument("--fail-on", choices=("note", "warn", "defect", "never"), default="defect",
                   help="exit with 1 when a finding at this level or worse is found (default: defect)")
    c.add_argument("--no-report", action="store_true", help="only print the summary lines")
    args = parser.parse_args(argv)
    if args.command == "check":
        return _check(args)
    return _process(args)


def _evaluations(sources):
    for source in sources:
        path = Path(source)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and not child.name.startswith("."):
                    yield child, True
        else:
            yield path, False


def _process(args) -> int:
    import kika
    from kika._process import ProcessingRouteError

    out = Path(args.output)
    failed = 0
    for path, from_folder in _evaluations(args.sources):
        t0 = time.perf_counter()
        try:
            run = kika.process(path, kind=args.kind, temperatures=args.temperatures,
                               heating=True if args.heating else None, workers=args.workers)
        except (ProcessingRouteError, FileNotFoundError) as error:
            if from_folder:
                print(f"  skip  {path.name}: {error}")
                continue
            print(f"  FAIL  {path.name}: {error}")
            failed += 1
            continue
        formats = [f for f in args.to if f in run.formats]
        skipped = [f for f in args.to if f not in run.formats]
        for fmt in formats:
            try:
                run.write(out, fmt, replace=args.replace)
            except Exception:
                pass        # recorded as a refused write; the report says why
        report = run.report_document()
        data = report.to_dict()
        if not args.no_report:
            out.mkdir(parents=True, exist_ok=True)
            report.save(out / f"{path.stem}.report.html")
        summary = data["summary"]
        mark = {"passed": "ok", "note": "ok", "warning": "warn", "failed": "FAIL"}[summary["status"]]
        if summary["incomplete"]:
            mark = "FAIL"
            failed += 1
        extra = f"; not written as {', '.join(skipped)} ({run.kind})" if skipped else ""
        print(f"  {mark:<4}  {path.name}: {summary['headline']} ({time.perf_counter() - t0:.0f} s){extra}")
    return 1 if failed else 0


def _check(args) -> int:
    import kika
    from kika.endf.checks.findings import _RANK

    out = Path(args.output)
    floor = None if args.fail_on == "never" else _RANK[args.fail_on]
    status = 0

    def failing(counts) -> bool:
        return floor is not None and any(n and _RANK[lv] >= floor for lv, n in counts.items())

    for source in args.sources:
        path = Path(source)
        t0 = time.perf_counter()
        try:
            report = kika.check(path, what=args.what, mf=args.mf, deep=args.deep, workers=args.workers,
                                library=args.library,
                                recursive=args.recursive, progress=True)
        except (FileNotFoundError, ValueError) as error:
            print(f"  FAIL  {path.name}: {error}")
            status = 1
            continue
        library = hasattr(report, "tapes")
        stem = (args.library or path.name) if library else path.stem
        if not args.no_report:
            out.mkdir(parents=True, exist_ok=True)
            report.to_html(out / f"{stem}.check.html", level=args.level)
        elapsed = time.perf_counter() - t0
        if library:
            for tape in report.tapes:
                counts = {lv: tape.count(lv) for lv in ("defect", "warn", "note")}
                bad = not tape.ok or failing(counts)
                status |= int(bad)
                print(f"  {_mark(counts, tape.ok)}  {tape.name}: "
                      + (f"could not be checked ({tape.error})" if not tape.ok else _tally(counts)))
            print(f"  {len(report.tapes)} evaluations in {elapsed:.0f} s")
        else:
            counts = {lv: sum(1 for f in report if f.level == lv) for lv in ("defect", "warn", "note")}
            status |= int(failing(counts))
            print(f"  {_mark(counts, True)}  {path.name}: {_tally(counts)} ({elapsed:.1f} s)")
    return status


def _mark(counts, ok) -> str:
    if not ok or counts["defect"]:
        return "DEF " if ok else "FAIL"
    return "warn" if counts["warn"] else "ok  "


def _tally(counts) -> str:
    words = {"defect": "defect", "warn": "warning", "note": "note"}
    parts = [f"{n} {words[lv]}{'s' if n != 1 else ''}" for lv, n in counts.items() if n]
    return ", ".join(parts) if parts else "nothing found"


if __name__ == "__main__":
    sys.exit(main())
