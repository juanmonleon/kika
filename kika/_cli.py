"""The ``kika`` command: ``kika process`` on the command line.

::

    kika process n-092_U_235.endf --to ace hdf5 -o out/
    kika process endfb81/neutrons/ --to ace -o lib/ --temperatures 293.6 600
    kika process tsl-HinH2O.endf -o thermal/

Each evaluation goes through :func:`kika.process`; every file asked for is
written into the output folder, and the report beside it
(``<name>.report.html``). One line per evaluation says how it went. The exit
status is 0 when every evaluation was processed and verified (notes and
warnings included), 1 when any was incomplete or refused, 2 for a usage error.
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
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
