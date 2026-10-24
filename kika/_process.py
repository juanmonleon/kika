"""``kika.process()`` -- one call to process an evaluation, whatever it is.

The user says what to process; kika reads the tape's header, picks the route
and its settings, runs it and verifies it. Every route returns the same kind
of object: its stages, ``write(directory, format)`` for the files and
``report_document()`` for the report (:mod:`kika.endf.processing_report`).

==================  =================================  ==========================
Evaluation          How kika knows                     Files ``write`` makes
==================  =================================  ==========================
Incident neutron    MF1/451 NSUB = 10, or GNDS         ``ace``, ``hdf5``,
                                                       ``pendf``, ``gnds``,
                                                       ``dosimetry`` (``.y``)
Thermal scattering  NSUB = 12                          ``ace`` (``.t``), ``hdf5``
Photo-atomic        NSUB = 3, with its relaxation      ``ace`` (``.p``)
                    (NSUB = 6) and electron
                    (NSUB = 113) tapes found beside it
Photonuclear        NSUB = 0                           ``ace`` (``.u``)
Dosimetry           asked for (``kind="dosimetry"``):  ``ace`` (``.y``)
                    a pointwise tape such as IRDFF-II
==================  =================================  ==========================

The format is read from the content, never the file name, and a tape whose
route kika cannot decide is refused with what it found -- kika never guesses.
The existing entries (:func:`kika.endf.process_endf`,
:func:`kika.hdf5.thermal_processing.processThermal`, ...) stay as they are; this
is the door in front of them.
"""
from __future__ import annotations

import os
import re
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

__all__ = ["process", "Processing", "ProcessingRouteError"]

#: Sublibraries (MF1/451 NSUB) and the route each takes.
NSUB_ROUTES = {10: "neutron", 12: "thermal", 3: "photoatomic", 0: "photonuclear"}
_NSUB_NAMES = {10: "incident neutron", 12: "thermal scattering", 3: "photo-atomic", 0: "photonuclear",
               6: "atomic relaxation", 113: "electro-atomic", 4: "radioactive decay", 5: "spontaneous fission yields",
               11: "neutron-induced fission yields", 19: "standards"}


class ProcessingRouteError(ValueError):
    """The evaluation is not one ``kika.process`` can take, or its route needs
    something only the user can give (said in the message)."""


# --------------------------------------------------------------------------- the call

def process(source, *, kind: Optional[str] = None, temperatures: Optional[Sequence[float]] = None,
            heating: Optional[bool] = None, name: Optional[str] = None, neutron=None,
            relaxation=None, electrons=None, workers: Optional[int] = None, progress=None, plan=None):
    """Process the evaluation at ``source`` and return a :class:`Processing`.

    Nothing has to be chosen: kika reads the tape's header (MF1/451 NSUB, or the
    GNDS root), takes the route of that kind of evaluation and its tested
    defaults, and verifies every stage. What it decided is in the report
    (``report_document()``, section *Choices made for you*).

    Parameters, all optional:

    kind
        ``"neutron"``, ``"thermal"``, ``"photoatomic"``, ``"photonuclear"`` or
        ``"dosimetry"``, to override the route the header says. ``"dosimetry"``
        is never inferred: a dosimetry tape looks like any neutron tape.
    temperatures
        Kelvin. Neutron and dosimetry: default 293.6 K. Thermal: default every
        temperature the evaluation tabulates.
    heating
        Neutron. Not given: the ACE and HDF5 files carry KERMA (MT301) and
        damage (MT444), computed when they are written. ``True``: the processing
        stage computes them too (in the PENDF). ``False``: no heating anywhere,
        and the files are faster to write; an ACE table's heating numbers are
        then 0 (``ProcessingPlan.heating``).
    name
        Thermal: the OpenMC name of the table (``"c_H_in_H2O"``), when kika cannot
        tell it from the tape.
    neutron
        Thermal: the principal nuclide's neutron tape, whose mass ratio ACER
        writes in the table's header (otherwise the scattering law's).
    relaxation, electrons
        Photo-atomic: the element's atomic-relaxation and electro-atomic tapes,
        when they are not beside the photo-atomic one.
    workers
        Processes or threads; default from ``KIKA_WORKERS`` or the machine. No
        result depends on it.
    progress
        ``progress(stage, fraction)``, as :func:`kika.endf.process_endf` calls it.
    plan
        Neutron: a full :class:`kika.endf.ProcessingPlan`, for whoever needs more
        than the options above.

    Run it under ``if __name__ == "__main__":`` in a script on Windows or macOS,
    where worker processes import the script again.
    """
    path = Path(os.fspath(source))
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    route, header = _route(path, kind)
    if route == "neutron":
        return _neutron(path, header, temperatures, heating, workers, progress, plan)
    if plan is not None:
        raise ProcessingRouteError(f"plan= is for neutron evaluations; this is {_NSUB_NAMES.get(header.get('nsub'), route)}")
    if heating is not None:
        raise ProcessingRouteError(f"heating= is for neutron evaluations; this is {route}")
    if route == "thermal":
        return _thermal(path, header, temperatures, name, neutron, workers, progress)
    for option, value in (("name", name), ("neutron", neutron)):
        if value is not None:
            raise ProcessingRouteError(f"{option}= is for thermal evaluations; this is {route}")
    if route == "photoatomic":
        return _photoatomic(path, header, relaxation, electrons, progress)
    if relaxation is not None or electrons is not None:
        raise ProcessingRouteError(f"relaxation= and electrons= are for photo-atomic evaluations; this is {route}")
    if route == "photonuclear":
        if temperatures is not None:
            raise ProcessingRouteError("a photonuclear table is at 0 K; temperatures= does not apply")
        return _photonuclear(path, header, progress)
    return _dosimetry(path, header, temperatures, workers, progress)


def _route(path, kind):
    """``(route, header)``: what the evaluation is, from its content."""
    from ._identify import identify
    from ._read import UnknownFormatError
    try:
        found = identify(path)
    except UnknownFormatError as error:
        raise ProcessingRouteError(f"{path.name} is not an evaluation to process: {error}") from error
    if found == "gnds":
        header = dict(format="gnds")
        if kind not in (None, "neutron"):
            raise ProcessingRouteError(f"{path.name} is a GNDS evaluation; kika processes GNDS neutron "
                                       f"evaluations only (kind={kind!r} was asked for)")
        return "neutron", header
    if found != "endf":
        raise ProcessingRouteError(f"{path.name} is {found!r}, not an evaluation to process "
                                   f"(an ENDF-6 tape or a GNDS file)")
    header = endf_header(path)
    if kind is not None:
        if kind not in ("neutron", "thermal", "photoatomic", "photonuclear", "dosimetry"):
            raise ProcessingRouteError(f"kind must be neutron, thermal, photoatomic, photonuclear or "
                                       f"dosimetry; got {kind!r}")
        return kind, header
    nsub = header.get("nsub")
    route = NSUB_ROUTES.get(nsub)
    if route is None:
        what = _NSUB_NAMES.get(nsub, f"sublibrary NSUB = {nsub}")
        extra = (" Process the element's photo-atomic tape instead; this one is found beside it."
                 if nsub in (6, 113) else "")
        raise ProcessingRouteError(f"{path.name} is a {what} evaluation, which kika does not process into a "
                                   f"transport table.{extra}")
    if route == "neutron" and header.get("materials", 1) > 1:
        raise ProcessingRouteError(
            f"{path.name} holds {header['materials']} materials. A neutron library is processed one material "
            f"per tape; a dosimetry library such as IRDFF-II is processed with kind='dosimetry'.")
    return route, header


def endf_header(path, *, materials: bool = True) -> dict:
    """``{mat, za, awr, lrp, nsub, materials}`` from an ENDF-6 tape's MF1/451, reading only
    its first lines; ``materials`` (the count, one MF1/451 per MAT) reads the rest."""
    out, mats = {}, set()
    with open(path, "r", encoding="ascii", errors="replace") as handle:
        lines = []
        for line in handle:
            if len(line) >= 75 and line[70:72].strip() == "1" and line[72:75].strip() == "451":
                lines.append(line)
                if len(lines) == 3:
                    break
        if len(lines) < 3:
            raise ProcessingRouteError(f"{Path(path).name} has no MF1/451 header to say what it is")
        head, _, third = lines
        out.update(mat=int(head[66:70]), za=_endf_float(head[0:11]), awr=_endf_float(head[11:22]),
                   lrp=int(head[22:33]), nsub=int(third[44:55]))
        mats.add(out["mat"])
        for line in (handle if materials else ()):
            if len(line) >= 75 and line[72:75].strip() == "451" and line[70:72].strip() == "1":
                mats.add(int(line[66:70]))
    out["materials"] = len(mats)
    return out


def _endf_float(text):
    from ._records import parse_number
    return parse_number(text)


# --------------------------------------------------------------------------- results

class Processing:
    """What :func:`kika.process` returns, whatever the route.

    ``kind``
        The route: ``neutron``, ``thermal``, ``photoatomic``, ``photonuclear``,
        ``dosimetry``.
    ``source``
        The evaluation.
    ``stages``
        One :class:`kika.endf.StageReport` per stage, file writes included.
    ``formats``
        What :meth:`write` can make from it.
    """

    kind = ""
    formats: tuple = ()

    def __init__(self, source):
        self.source = Path(source)
        self.stages = []

    def write(self, directory, format=None, *, replace: bool = False) -> list:
        """Write the processed evaluation into ``directory`` and return the paths.

        ``format`` is one of :attr:`formats` or a list of them; the default is
        ``ace``. Every file is read back and verified before it replaces one; a
        refused file raises :class:`kika.endf.ProcessingOutputError` and is listed in
        the report. ``replace`` lets a table replace one the directory's ``xsdir``
        already names.
        """
        formats = [format or "ace"] if isinstance(format, (str, type(None))) else list(format)
        unknown = [f for f in formats if f not in self.formats]
        if unknown:
            raise ValueError(f"{self.kind} evaluations are written as {', '.join(self.formats)}; "
                             f"got {', '.join(map(repr, unknown))}")
        directory = Path(directory)
        written = []
        for fmt in formats:
            out = self._write(directory, fmt, replace)
            written += out if isinstance(out, list) else [out]
        return written

    def _write(self, directory, fmt, replace):
        raise NotImplementedError

    def report_document(self):
        """The report: did it go well, and what exactly was done (``to_html()``,
        ``to_text()``, ``to_json()``, ``save(path)``). Ask for it after writing."""
        from kika.endf.processing_report import ProcessingReport, build_generic
        return ProcessingReport(build_generic(self))

    # the pieces of the generic report a route fills
    def _report_parts(self) -> dict:
        return {}

    def _stage(self, stage, t0, summary="", details=None, error=None, category=None):
        from kika.endf.process import StageReport
        if error is None:
            record = StageReport(stage, "done", time.perf_counter() - t0, None, summary, dict(details or {}))
        else:
            details = dict(details or {}, error="".join(traceback.format_exception_only(type(error), error)).strip())
            record = StageReport(stage, "rejected", time.perf_counter() - t0,
                                 category or getattr(error, "category", None), str(error), details)
        self.stages.append(record)
        return record

    def _written(self, fmt, t0, paths, directory, temperatures=(), omitted=(), error=None):
        details = dict(format=fmt, path=str(directory), files=[str(p) for p in paths],
                       temperatures=[float(t) for t in temperatures], omitted=list(omitted))
        summary = f"{fmt} {', '.join(Path(p).name for p in paths)}" if error is None else ""
        return self._stage("write", t0, summary, details, error)

    def __repr__(self):
        done = [s.stage for s in self.stages if s.status == "done"]
        return f"<kika.Processing {self.kind} {self.source.name}: {', '.join(done) or 'nothing done'}>"


class NeutronProcessing(Processing):
    """A neutron evaluation processed by :func:`kika.endf.process_endf` (or
    ``process_gnds``). Every attribute of :class:`kika.endf.EndfProcessing`
    (``cross_sections``, ``probability_tables``, ``heating``...) is reachable here."""

    kind = "neutron"
    formats = ("ace", "hdf5", "pendf", "gnds", "dosimetry")

    def __init__(self, source, run):
        super().__init__(source)
        self.run = run

    @property
    def stages(self):
        return self.run.stages

    @stages.setter
    def stages(self, value):
        pass

    def __getattr__(self, name):
        if name == "run":
            raise AttributeError(name)
        return getattr(self.run, name)

    def _name(self):
        target = getattr(self.run.reconstruction.suite, "target", None) or self.source.stem
        return re.sub(r"[^A-Za-z0-9_.-]", "_", str(target))

    def _write(self, directory, fmt, replace):
        if fmt == "ace":
            return self.run.write_ace(directory, replace=replace)
        if fmt == "dosimetry":
            return self.run.write_dosimetry_ace(directory, replace=replace)
        if fmt == "hdf5":
            return self.run.write_hdf5(directory / f"{self._name()}.h5", index=True)
        if fmt == "pendf":
            return self.run.write_pendf(directory / f"{self._name()}.pendf")
        return self.run.write_gnds(directory / f"{self._name()}.gnds.xml")

    def report_document(self):
        return self.run.report_document()


def _neutron(path, header, temperatures, heating, workers, progress, plan):
    import dataclasses
    from kika.endf import ProcessingPlan, process_endf
    if plan is None:
        options = {}
        if temperatures is not None:
            options["temperatures"] = tuple(temperatures)
        if heating is not None:
            options["heating"] = bool(heating)
        if workers is not None:
            options["workers"] = workers
        plan = ProcessingPlan(**options)
    elif any(v is not None for v in (temperatures, heating, workers)):
        plan = dataclasses.replace(plan, **{k: v for k, v in dict(
            temperatures=None if temperatures is None else tuple(temperatures), heating=heating,
            workers=workers).items() if v is not None})
    if header.get("format") == "gnds":
        from kika.gnds import process_gnds
        return NeutronProcessing(path, process_gnds(path, plan, progress))
    return NeutronProcessing(path, process_endf(path, plan, progress))


# --------------------------------------------------------------------------- thermal

class ThermalProcessing(Processing):
    """A thermal scattering law processed into OpenMC's and ACE's thermal tables
    (:func:`kika.hdf5.thermal_processing.processThermal`)."""

    kind = "thermal"
    formats = ("ace", "hdf5")

    def __init__(self, source, name):
        super().__init__(source)
        self.name = name
        self.suite = None
        self.temperatures = ()

    def _write(self, directory, fmt, replace):
        t0 = time.perf_counter()
        try:
            if fmt == "ace":
                from kika._utils import ace_temperature_suffix
                from kika.ace.xsdir import XsdirEntryExists, add_xsdir_entry, build_xsdir_line
                from kika.endf.process import ProcessingOutputError
                from kika.tsl.ace import read_ace_thermal_header
                from kika.tsl.ace_writer import write_thermal_suite
                directory.mkdir(parents=True, exist_ok=True)
                paths, extra = [], 40
                for T in self.temperatures:
                    try:
                        extension = ace_temperature_suffix(T) + "t"
                    except ValueError:
                        # A scattering law tabulates temperatures the standard suffixes
                        # do not name; they are numbered from .40t up, in order.
                        extension, extra = f"{extra}t", extra + 1
                    path = write_thermal_suite(self.suite, directory / f"{self.name}_{T:g}K.t", T,
                                               extension=extension)
                    # Its xsdir line, as for the neutron tables: the file relative to
                    # the xsdir, the XSS length NXS(1), kT in MeV.
                    h = read_ace_thermal_header(path)
                    try:
                        add_xsdir_entry(directory / "xsdir", build_xsdir_line(h.zaid, h.awr, path.name, h.nxs[1], h.kt),
                                        replace=replace)
                    except XsdirEntryExists as error:
                        raise ProcessingOutputError("ace-table-exists", str(error)) from error
                    paths.append(path)
            else:
                import kika
                kika.write(self.suite, directory, format="hdf5")
                paths = [directory / f"{self.name}.h5"]
        except Exception as error:
            from kika.endf.process import ProcessingOutputError
            failure = error if getattr(error, "category", None) else ProcessingOutputError(
                f"{fmt}-failed", f"{type(error).__name__}: {error}")
            self._written(fmt, t0, [], directory, self.temperatures, error=failure)
            raise failure from error
        self._written(fmt, t0, paths, directory, self.temperatures)
        return paths

    def _report_parts(self):
        from kika.hdf5.thermal_names import THERMAL_DATA
        entry = THERMAL_DATA.get(self.name)
        return dict(
            title="Thermal scattering", what=f"{self.name}",
            choices=[
                dict(topic="Temperatures", choice=", ".join(f"{T:g} K" for T in self.temperatures),
                     why="Every temperature the evaluation tabulates, unless others were asked for; kika does "
                         "not interpolate a scattering law between temperatures."),
                dict(topic="Inelastic tables",
                     choice="NJOY THERMR's method for the outgoing energies and 16 equiprobable cosines, on an "
                            "incident grid refined until the cross section reads within 0.1 % between nodes; "
                            "the inelastic cross section integrated directly.",
                     why="THERMR's fixed grid of 118 energies leaves the cross section up to 4 % off between "
                         "nodes, which transport sees."),
                dict(topic="Elastic", choice="From the evaluation: Bragg edges (coherent) and the "
                                             "Debye-Waller integral (incoherent).",
                     why="As OpenMC's libraries take it, exact rather than tabulated."),
                dict(topic="ACE names", choice="The registry's short name and the suffix of the temperature "
                                                "(.02t at 293.6 K); temperatures with no standard suffix are "
                                                "numbered .40t, .41t, ... in increasing order.",
                     why="A table names one temperature, and kika does not reuse a suffix that names another."),
            ] + ([dict(topic="Table name", choice=f"{self.name} ({entry.name} in ACE), mixing {entry.nmix} atom "
                                                  f"type{'s' if entry.nmix > 1 else ''}",
                       why="OpenMC's registry of thermal tables, which also names the nuclides the table applies to.")]
                 if entry else []),
            steps=[f"Read the scattering law of {self.name}.",
                   f"Computed the inelastic and elastic tables at {', '.join(f'{T:g} K' for T in self.temperatures)}."])


def thermal_table_name(path) -> str:
    """The OpenMC name (``c_H_in_H2O``) of a thermal scattering tape, from the
    material it names (``tsl-HinH2O.endf``, ZSYMAM). Only an exact match, once
    spaces, case and the ``in`` are set aside, is taken; anything else raises with
    the closest names, to pass as ``name=``."""
    import difflib
    from kika.hdf5.thermal_names import THERMAL_DATA
    from kika.tsl import read_tsl_header
    header = read_tsl_header(str(path))
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower().replace("c_", "", 1).replace("_in_", "in"))
    wanted = norm(header.name or Path(path).stem.replace("tsl-", ""))
    exact = [n for n in THERMAL_DATA if norm(n) == wanted]
    if len(exact) == 1:
        return exact[0]
    close = difflib.get_close_matches(wanted, {norm(n): n for n in THERMAL_DATA}, n=3, cutoff=0.5)
    names = [n for n in THERMAL_DATA if norm(n) in close]
    raise ProcessingRouteError(
        f"kika cannot tell which OpenMC thermal table {Path(path).name} ({header.name!r}) is"
        + (f"; the closest are {', '.join(names)}" if names else "")
        + ". Pass it as name='c_...'.")


def _thermal(path, header, temperatures, name, neutron, workers, progress):
    from kika.hdf5.thermal_processing import processThermal
    name = name or thermal_table_name(path)
    run = ThermalProcessing(path, name)
    t0 = time.perf_counter()
    if progress:
        progress("thermal", None)
    try:
        run.suite = processThermal(path, neutron, name=name, temperatures=temperatures, workers=workers)
    except Exception as error:
        run._stage("thermal", t0, error=error, category="thermal-failed")
        return run
    if temperatures is not None:
        run.temperatures = tuple(sorted(float(t) for t in temperatures))
    else:
        # The heated styles state kT in eV (OpenMC's convention); back to kelvin, to
        # the six figures an ENDF tape states a temperature with.
        from kika.hdf5.names import K_BOLTZMANN
        run.temperatures = tuple(sorted(float(f"{float(st.temperature.value) / K_BOLTZMANN:.6g}")
                                        for st in run.suite.styles
                                        if getattr(st, "temperature", None) is not None))
    report = getattr(run.suite, "report", None)
    run._stage("thermal", t0, f"{name} at {', '.join(f'{T:g} K' for T in run.temperatures)}",
               dict(name=name, temperatures=list(run.temperatures),
                    notes=list(getattr(report, "warnings", []) or [])))
    if progress:
        progress("thermal", 1.0)
    return run


# --------------------------------------------------------------------------- photons

class TableProcessing(Processing):
    """A route whose result is one ACE table, built in the processing stage
    (photo-atomic, photonuclear) or at writing (dosimetry)."""

    formats = ("ace",)

    def __init__(self, source, kind):
        super().__init__(source)
        self.kind = kind
        self.table = None
        self.notes = []

    def _write(self, directory, fmt, replace):
        t0 = time.perf_counter()
        try:
            path = self._write_table(directory, replace)
        except Exception as error:
            from kika.endf.process import ProcessingOutputError
            failure = error if getattr(error, "category", None) else ProcessingOutputError(
                "ace-failed", f"{type(error).__name__}: {error}")
            self._written("ace", t0, [], directory, error=failure)
            raise failure from error
        self._written("ace", t0, [path], directory, omitted=self._omitted())
        return [path]

    def _omitted(self):
        return []


class PhotoatomicProcessing(TableProcessing):
    def __init__(self, source, relaxation, electrons):
        super().__init__(source, "photoatomic")
        self.relaxation, self.electrons = Path(relaxation), Path(electrons)

    def _write_table(self, directory, replace):
        return self.table.write_ace(directory, replace=replace)

    def _report_parts(self):
        return dict(title="Photo-atomic", what=self.source.name, notes=list(self.notes), choices=[
            dict(topic="Inputs", choice=f"{self.source.name}, {self.relaxation.name} and {self.electrons.name}",
                 why="An EPRDATA table needs the photo-atomic, atomic-relaxation and electro-atomic "
                     "evaluations of the same element; kika found them by sublibrary and element."),
            dict(topic="Heating", choice="From the full relaxation cascade.",
                 why="NJOY and eprdata25 let the L shell overwrite the K shell above Z = 30, which overstates "
                     "the heating (lead at 100 keV by 71 %).")],
            steps=[f"Read the photo-atomic, relaxation and electron evaluations of {self.source.stem}.",
                   "Built the EPRDATA blocks: cross sections, form factors, Compton profiles, "
                   "fluorescence and the electron data."])


def _photoatomic(path, header, relaxation, electrons, progress):
    from kika.ace.processing.photoatomic import process_photoatomic
    z = int(header["za"]) // 1000
    relaxation = Path(relaxation) if relaxation is not None else _sibling(path, z, 6, "atomic-relaxation", "relaxation")
    electrons = Path(electrons) if electrons is not None else _sibling(path, z, 113, "electro-atomic", "electrons")
    run = PhotoatomicProcessing(path, relaxation, electrons)
    t0 = time.perf_counter()
    try:
        run.table = process_photoatomic(path, relaxation, electrons)
    except Exception as error:
        run._stage("tables", t0, error=error, category="photoatomic-failed")
        return run
    run.notes = list(run.table.notes)
    run._stage("tables", t0, f"EPRDATA table of Z = {z}", dict(notes=run.notes))
    return run


def _sibling(path, z, nsub, what, option):
    """The tape of sublibrary ``nsub`` for element ``z`` beside ``path``: in its
    folder, or in a sibling folder of the library (NNDC's ``photoat/``,
    ``atomic_relax/``, ``electrons/`` side by side, same library subfolder)."""
    here = path.parent
    folders = [here] + [p for p in here.parent.iterdir() if p.is_dir() and p != here]
    folders += [p / here.name for p in here.parent.parent.iterdir() if p.is_dir() and (p / here.name).is_dir()]
    code = f"{z:03d}"
    for folder in folders:
        # Beside it, every tape; in the library's other folders, the element's (by its
        # three-digit Z, as NNDC names them).
        pattern = "*" if folder == here else f"*{code}*"
        for candidate in sorted(folder.glob(pattern)):
            if candidate == path or not candidate.is_file():
                continue
            try:
                h = endf_header(candidate, materials=False)
            except (ProcessingRouteError, ValueError, OSError):
                continue
            if h.get("nsub") == nsub and int(h["za"]) // 1000 == z:
                return candidate
    raise ProcessingRouteError(
        f"no {what} tape for Z = {z} beside {path.name} (looked in {here} and the folders next to it). "
        f"Pass it as {option}=.")


class PhotonuclearProcessing(TableProcessing):
    def __init__(self, source):
        super().__init__(source, "photonuclear")

    def _write_table(self, directory, replace):
        from kika.endf.process import write_checked_ace
        return write_checked_ace(self.table, directory, replace=replace)

    def _omitted(self):
        report = getattr(self.table, "report", None)
        return list(getattr(report, "unsupported", []) or [])

    def _report_parts(self):
        return dict(title="Photonuclear", what=self.source.name, notes=list(self.notes), choices=[
            dict(topic="Temperature", choice="0 K", why="Photonuclear evaluations are not Doppler broadened."),
            dict(topic="Kinematics and heating",
                 choice="Relativistic two-body kinematics; photon Kalbach systematics for the emitted "
                        "particles; heating from the direct energy balance.",
                 why="NJOY applies neutron systematics and non-relativistic kinematics to photons and gets "
                     "the heating wrong (deuterium five times too high).")],
            steps=[f"Read {self.source.name} and decoded its reactions and emitted particles.",
                   "Built the photonuclear table: reactions, their sum, heating, and per particle its "
                   "production, yields, angles and energies."])


def _photonuclear(path, header, progress):
    from kika.endf.process import photonuclear_table
    run = PhotonuclearProcessing(path)
    t0 = time.perf_counter()
    try:
        run.table = photonuclear_table(path)
    except Exception as error:
        run._stage("tables", t0, error=error, category="photonuclear-failed")
        return run
    report = getattr(run.table, "report", None)
    run.notes = list(getattr(report, "warnings", []) or []) + list(getattr(report, "approximations", []) or [])
    run._stage("tables", t0, f"photonuclear table {run.table.header.zaid}", dict(notes=run.notes))
    return run


class DosimetryProcessing(TableProcessing):
    def __init__(self, source, temperatures):
        super().__init__(source, "dosimetry")
        self.temperatures = temperatures

    def _write_table(self, directory, replace):
        from kika.endf import dosimetry_ace
        return dosimetry_ace(self.source, directory, replace=replace)

    def _report_parts(self):
        return dict(title="Dosimetry", what=self.source.name, choices=[
            dict(topic="Cross sections", choice="As the tape tabulates them, each with its own interpolation, at "
                                                "the tape's temperature.",
                 why="A dosimetry evaluation is stated pointwise; ACER copies it the same way.")],
            steps=[f"Read {self.source.name}: a pointwise evaluation, no resonances to reconstruct."])


def _dosimetry(path, header, temperatures, workers, progress):
    if header.get("lrp") == 1:
        # Resonance parameters: process it as a neutron tape; its .y tables come
        # from write(..., "dosimetry").
        run = _neutron(path, header, temperatures, None, workers, progress, None)
        run.formats = ("dosimetry",) + tuple(f for f in run.formats if f != "dosimetry")
        return run
    run = DosimetryProcessing(path, temperatures)
    run._stage("read", time.perf_counter(), "pointwise dosimetry evaluation", dict(lrp=header.get("lrp")))
    return run
