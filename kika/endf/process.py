"""Process an ENDF-6 neutron tape in stages: reconstruct, broaden, URR tables, heating.

The staged counterpart of :func:`kika.endf.reconstruct_endf`, for an
application that walks a user through what NJOY does to a neutron tape:

    from kika.endf import inspect_endf_processing, process_endf, ProcessingPlan
    inspect_endf_processing("n-079_Au_197.endf")        # MF1/MF2 only; what applies
    run = process_endf("n-079_Au_197.endf", ProcessingPlan(temperatures=(293.6, 600)))
    run.stages                                          # done / skipped / rejected, each
    run.cross_sections(600)                             # {MT: (E, sigma)} at 600 K
    run.probability_tables[600]["elastic"]              # URR bands
    run.report_document().save("Au197.html")

Stages, in order, with the names ``progress(stage, fraction)`` receives:
``read``, ``decode``, ``reconstruct``, ``broaden:<T>`` (one per temperature,
``T`` formatted ``%g``; the temperatures run in parallel, so their calls
interleave), ``urr``, ``heating``; ``write`` from :meth:`EndfProcessing.write_pendf`.
``fraction`` is ``None`` when a stage starts and then a float in (0, 1].

Failure contract. ``read``, ``decode`` and ``reconstruct`` raise exactly as
``reconstruct_endf`` does (``UnsupportedResonanceError`` /
``ReconstructionConvergenceError``, ``category`` in
``kika.processing.REJECTION_CATEGORIES``): without the 0 K cross sections there
is nothing to show. The later stages never raise: a stage the material does
not admit is ``rejected`` with a category of
:data:`PROCESSING_REJECTION_CATEGORIES`, one that does not apply is
``skipped``, and every earlier result is kept.

The pieces are kika's own and are the ones the OpenMC HDF5 writer uses
(:mod:`kika.hdf5.processing`): free-gas broadening up to ``thnmax`` with the
0 K values above it, as NJOY BROADR leaves them, and PURR's probability tables.
Heating is HEATR's (``local = 0``): MT301 and MT444 at every temperature,
when the plan asks for it (``heating=True``).

Outputs, each appending its own ``write`` stage: :meth:`EndfProcessing.write_pendf`
(one material per temperature, MF2/MT153 where there are URR tables, MT301 and
MT444 when the heating stage ran), :meth:`EndfProcessing.write_hdf5` (OpenMC) and
:meth:`EndfProcessing.write_gnds` (eval, recon and one ``heated`` style per
temperature). A failed output raises
:class:`ProcessingOutputError` and leaves any existing file in place.
"""
from __future__ import annotations

import copy
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

Progress = Callable[[str, "float | None"], None]


class ProcessingOutputError(ValueError):
    """An output of :class:`EndfProcessing` could not be written. ``category``
    is a key of :data:`PROCESSING_REJECTION_CATEGORIES` (or, for the PENDF, of
    ``kika.processing.REJECTION_CATEGORIES``)."""

    def __init__(self, category, message):
        super().__init__(message)
        self.category = category

#: Default temperature: room temperature as the official libraries state it.
DEFAULT_TEMPERATURES = (293.6,)

#: Every category a ``rejected`` or ``skipped`` stage of :func:`process_endf`
#: can carry, with a sentence for the user. The reconstruction's own categories
#: (raised, or on ``write``) are ``kika.processing.REJECTION_CATEGORIES``.
#: Stable names: a change here is an API change.
PROCESSING_REJECTION_CATEGORIES = {
    # broaden (rejected)
    "doppler-unsupported-target":
        "The target is bound or a mixture; free-gas Doppler broadening does not apply to it.",
    "doppler-incomplete-coverage":
        "Some reactions cannot be broadened as independent partial cross sections "
        "(their sums are not fully described by the file).",
    "doppler-domain-not-covered":
        "The 0 K cross sections do not span the evaluated energy range, so the broadening "
        "has no complete source.",
    "doppler-budget-exhausted":
        "Doppler broadening ran out of its point or memory budget for this material.",
    "doppler-verification-failed":
        "The broadened cross sections failed kika's accuracy check, so they are not shown.",
    "doppler-unsupported":
        "This material uses data the Doppler broadening does not support.",
    "doppler-failed":
        "Doppler broadening stopped with an internal error.",
    # urr (skipped / rejected)
    "urr-absent": "The file has no unresolved resonance range; there are no probability tables to compute.",
    "urr-no-temperature": "Probability tables are computed at the requested temperatures, and none was requested.",
    "urr-unsupported": "The unresolved resonance parameters use data the probability-table method does not support.",
    "urr-failed": "The probability tables stopped with an internal error.",
    # heating (skipped / rejected)
    "heating-not-implemented": "Heating (KERMA) is not yet available in this processing chain.",
    "heating-unsupported":
        "The evaluation lacks data HEATR's method needs for the heating (a fission without MF1/458).",
    "heating-failed": "The heating (KERMA and damage) stopped with an internal error.",
    # write (ProcessingOutputError)
    "hdf5-needs-temperature":
        "An OpenMC HDF5 file needs at least one broadened temperature; process with one or more "
        "temperatures and broadening on.",
    "hdf5-failed": "The OpenMC HDF5 file could not be written.",
    "ace-needs-temperature":
        "An ACE table is at one broadened temperature; process with one or more temperatures and "
        "broadening on.",
    "ace-temperature-not-processed": "A temperature asked for in the ACE output was not broadened in this run.",
    "ace-table-exists":
        "The xsdir in the output directory already names this table; pass replace=True to overwrite it.",
    "ace-failed": "The ACE table could not be written, or did not read back as it was written.",
    "pendf-temperature-not-processed":
        "A temperature asked for in the PENDF was not broadened in this run.",
    "pendf-failed": "The PENDF could not be written, or did not read back as it was written.",
    "gnds-failed": "The GNDS file could not be written, or did not read back as it was written.",
    "pendf-needs-endf-source":
        "A PENDF can only be written for an evaluation read from ENDF-6 for now: kika cannot yet "
        "write the resonance parameters (MF2) of a GNDS evaluation back to ENDF-6.",
}

#: The Doppler engine's categories (``DopplerInputError.category``) gathered
#: into the user-facing ones above. Anything else is ``doppler-unsupported``
#: (an ``UnsupportedDopplerError``) or ``doppler-failed``.
_DOPPLER_GROUPS = {
    "bound-target": "doppler-unsupported-target",
    "bound-or-mixture-target": "doppler-unsupported-target",
    "unsupported-target": "doppler-unsupported-target",
    "incomplete-coverage": "doppler-incomplete-coverage",
    "aggregate-without-graph": "doppler-incomplete-coverage",
    "incomplete-total-graph": "doppler-incomplete-coverage",
    "duplicate-source-leaf": "doppler-incomplete-coverage",
    "ambiguous-reaction": "doppler-incomplete-coverage",
    "domain-not-covered": "doppler-domain-not-covered",
    "invalid-domain": "doppler-domain-not-covered",
    "resources": "doppler-budget-exhausted",
    "tabulation-budget-exhausted": "doppler-budget-exhausted",
    "unresolvable-grid": "doppler-budget-exhausted",
    "table-verification-failed": "doppler-verification-failed",
    "reference-verification-failed": "doppler-verification-failed",
    "sum-not-closed": "doppler-verification-failed",
    "source-sum-not-closed": "doppler-verification-failed",
}


@dataclass(frozen=True)
class ProcessingPlan:
    """What :func:`process_endf` does. The defaults are what a user who touches
    nothing should get: the reconstruction as ``reconstruct_endf`` runs it, room
    temperature, broadening to where NJOY BROADR stops, URR tables, no heating
    (``heating=True`` adds HEATR's MT301 and MT444 at every temperature, 0 K
    included; it tabulates the product distributions, which is slow).

    ``temperatures`` are kelvin, > 0; 0 K is the reconstruction and is always
    there, so ``()`` stops after it. They are the temperatures of both the
    broadening and the URR tables; ``broaden=False`` skips the broadening and
    keeps them for the URR tables alone (the tables do not depend on the
    broadened MF3). ``thnmax`` is the top of the broadening in
    eV, or ``"njoy"`` for BROADR's default (the top of the resolved range).
    ``thinning`` is the relative tolerance each temperature's union grid is
    thinned to (BROADR's ``errthn``, 1e-3); ``None`` keeps every point.
    ``urr_bins``/``urr_ladders`` are PURR's bands and ladders. ``workers``
    bounds the processes of the broadening and the URR tables (default: up to
    eight).
    """
    temperatures: tuple = DEFAULT_TEMPERATURES
    broaden: bool = True
    reconstruction: object = None
    thnmax: object = "njoy"
    thinning: float | None = 1.0e-3
    urr: bool = True
    urr_bins: int = 20
    urr_ladders: int = 64
    heating: bool = False
    workers: int | None = None

    def __post_init__(self):
        temps = tuple(sorted({float(t) for t in self.temperatures}))
        if any(not np.isfinite(t) or t <= 0 for t in temps):
            raise ValueError("temperatures are kelvin > 0; 0 K is the reconstruction and always there")
        object.__setattr__(self, "temperatures", temps)
        if self.thnmax != "njoy" and not float(self.thnmax) > 0:
            raise ValueError('thnmax is an energy in eV > 0 or "njoy"')


@dataclass
class StageReport:
    """One stage of a :func:`process_endf` run; plain types only.

    ``status`` is ``done``, ``skipped`` or ``rejected``; ``category`` is set on
    the last two (a key of :data:`PROCESSING_REJECTION_CATEGORIES`) and ``None``
    when a stage was simply not requested. ``summary`` is one English line.

    ``details`` by stage (every key optional; energies in eV, times in s):

    - ``read``: ``mat``, ``za``, ``library``, ``temperature_K`` (MF1's).
    - ``decode``: ``n_reactions``, ``context_source`` (where mass and spin came from).
    - ``reconstruct``: ``n_points`` (stored), ``n_resonances``, ``formalisms``
      (list of str), ``resolved_range_eV`` and ``unresolved_range_eV``
      (``[lo, hi]`` or ``None``), ``native_kernel``.
    - ``broaden``: ``temperatures`` (K, broadened), ``thnmax_eV``,
      ``n_points`` (``{T: union-grid points}``), ``engine_category`` and
      ``error`` when rejected.
    - ``urr``: ``temperatures``, ``n_energies``, ``bins``, ``ladders``,
      ``range_eV``, ``error`` when rejected.
    - ``write``: ``path``, ``temperatures`` written (``[0.0]`` for now).
    """
    stage: str
    status: str
    seconds: float | None = None
    category: str | None = None
    summary: str = ""
    details: dict = field(default_factory=dict)


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _engine_category(error):
    from kika.processing.doppler import DopplerConvergenceError, UnsupportedDopplerError
    engine = getattr(error, "category", None)
    if engine in _DOPPLER_GROUPS:
        return _DOPPLER_GROUPS[engine], engine
    if isinstance(error, MemoryError):
        return "doppler-budget-exhausted", "memory"
    if isinstance(error, UnsupportedDopplerError):
        return "doppler-unsupported", engine
    if isinstance(error, DopplerConvergenceError):
        return "doppler-verification-failed", engine
    return "doppler-failed", engine


def _describe(error) -> str:
    """The error for a stage's summary; a dead worker pool says the likely cause.

    Where processes are spawned (Windows, macOS) every worker imports the
    caller's main script again. A script that starts the run at module level,
    without the ``if __name__ == "__main__":`` guard, makes each worker start
    it too, and multiprocessing stops them: the parent only sees a broken pool.
    """
    from concurrent.futures.process import BrokenProcessPool
    if isinstance(error, BrokenProcessPool):
        return (f"{error} A worker process stopped before doing any work. Where processes are "
                f"spawned (Windows, macOS) this is what happens when the calling script starts "
                f"the run without an `if __name__ == \"__main__\":` guard; add the guard, or "
                f"run in one process with ProcessingPlan(workers=1)")
    return str(error)


def _ranges(evaluation):
    """``(resolved, unresolved)`` summaries of the MF2 ranges evaluation_summary lists."""
    rows = evaluation.get("resonance_ranges") or []
    rrr = [r for r in rows if r.get("lru") == 1]
    urr = [r for r in rows if r.get("lru") == 2]
    resolved = None
    if rrr:
        resolved = dict(formalisms=sorted({r["formalism"] for r in rrr}),
                        resonances=int(sum(r.get("resonances") or 0 for r in rrr)),
                        range_eV=[min(r["el_eV"] for r in rrr), max(r["eh_eV"] for r in rrr)])
    unresolved = None
    if urr:
        unresolved = dict(range_eV=[min(r["el_eV"] for r in urr), max(r["eh_eV"] for r in urr)],
                          case=urr[0].get("urr_case"), lssf=urr[0].get("lssf"))
    return resolved, unresolved


def _resonance_line(resolved, unresolved):
    parts = []
    if resolved:
        lo, hi = resolved["range_eV"]
        parts.append(f"{resolved['resonances']} resolved resonances ({', '.join(resolved['formalisms'])}), "
                     f"{lo:g}-{hi:g} eV")
    if unresolved:
        lo, hi = unresolved["range_eV"]
        parts.append(f"unresolved range {lo:g}-{hi:g} eV")
    return "; ".join(parts) or "no resonance parameters"


def inspect_endf_processing(path) -> dict:
    """What :func:`process_endf` can do with a tape, from MF1 and MF2 alone (seconds).

    Returns ``evaluation`` (MF1/451 identification: ``za``, ``mat``, ``zsymam``,
    ``awr``, ``library``, ``temperature_K``, ``lrp``, ``emax_eV``, ...),
    ``resonance_ranges`` (one dict per MF2 range: ``formalism``, ``el_eV``,
    ``eh_eV``, ``lru``, ``lrf``, ``resonances``, ...), ``resolved``
    (``formalisms``, ``resonances``, ``range_eV``) or ``None``, ``unresolved``
    (``range_eV``, ``case``, ``lssf``) or ``None``, ``reactions`` (the MF3 MTs
    of the MF1 directory) and ``stages``: per stage, ``applicable`` (bool),
    ``predicted_rejection`` (a category or ``None``) and ``reason``.

    The prediction is static -- the flags and formalisms the engine is known
    to refuse. A run can still be rejected on its budgets or verification.
    """
    from kika.endf.read_endf import read_endf
    from .reconstruction_report import evaluation_summary
    path = Path(path)
    tape = read_endf(str(path), mf_numbers=[1, 2])
    evaluation = evaluation_summary(tape, path)
    ranges = evaluation.pop("resonance_ranges", [])
    resolved, unresolved = _ranges(dict(resonance_ranges=ranges))
    head = tape.mf[1].mt[451] if 1 in tape.mf and 451 in tape.mf[1].mt else None
    reactions = sorted({int(mt) for mf, mt, *_ in (getattr(head, "_directory", None) or []) if int(mf) == 3})
    mf2 = tape.mf[2].mt[151] if 2 in tape.mf and 151 in tape.mf[2].mt else None
    nis = len(getattr(mf2, "isotopes", []) or []) if mf2 is not None else 0

    reject, reason = None, _resonance_line(resolved, unresolved)
    if evaluation.get("lrp") == 2:
        reject, reason = "already-reconstructed", "LRP=2: MF3 already holds the resonance contribution"
    elif nis > 1:
        reject, reason = "unsupported-physics", f"{nis} isotopes in MF2; isotope mixtures are not implemented"
    elif any(r.get("lru") == 1 and r.get("lrf") in (4, 5, 6) for r in ranges):
        reject, reason = "unsupported-physics", "Adler-Adler / LRF=5,6 resolved parameters are not implemented"
    temperature = evaluation.get("temperature_K") or 0.0
    stages = dict(
        reconstruct=dict(applicable=True, predicted_rejection=reject, reason=reason),
        broaden=dict(applicable=reject is None, predicted_rejection=None,
                     reason=(f"the evaluation is at {temperature:g} K" if reject is None
                             else "needs the 0 K reconstruction")),
        urr=dict(applicable=unresolved is not None and reject is None,
                 predicted_rejection=None if unresolved is not None else "urr-absent",
                 reason=(_resonance_line(None, unresolved) if unresolved
                         else PROCESSING_REJECTION_CATEGORIES["urr-absent"])),
        heating=dict(applicable=reject is None, predicted_rejection=None,
                     reason=("MT301 and MT444 as NJOY HEATR (local = 0) at every temperature"
                             if reject is None else "needs the 0 K reconstruction")),
    )
    return dict(evaluation=evaluation, resonance_ranges=ranges, resolved=resolved,
                unresolved=unresolved, reactions=reactions, stages=stages)


def _contiguous(x):
    return np.ascontiguousarray(x, dtype=np.float64)


class EndfProcessing:
    """The result of :func:`process_endf`. Picklable; every array is float64."""

    def __init__(self, reconstruction, plan, stages, heated, thnmax, urr_tables, *, broadened=None,
                 source_format="endf"):
        #: The :class:`~kika.endf.EndfReconstruction` (0 K), unchanged.
        self.reconstruction = reconstruction
        self.plan = plan
        #: One :class:`StageReport` per stage, in order.
        self.stages = stages
        self._heated = heated            # {T: {MT: (E, sigma)}}
        #: Top of the broadening, eV (0 K values above it), or ``None``.
        self.thnmax = thnmax
        self._urr = urr_tables           # {T: ProbabilityTables}
        self._cold = None
        # The raw broadened forms ({T: (forms by reaction label, ceiling)}), kept
        # for the outputs that rebuild from them (write_hdf5) instead of
        # broadening again.
        self._broadened = broadened or {}
        #: ``"endf"`` or ``"gnds"``: what the evaluation was read from.
        self.source_format = source_format
        self._aside = {}                 # GNDS: containers set aside for the reconstruction
        self._heating = {}               # {T: {301: (E, h), 444: (E, d)}}, 0.0 for 0 K

    def _written(self, fmt, t0, path, error=None, summary="", **details):
        """Append this output's own ``write`` stage (several outputs, several stages)."""
        details = dict(format=fmt, path=str(Path(path)), **details)
        if error is None:
            self.stages.append(StageReport("write", "done", time.perf_counter() - t0, None, summary, details))
        else:
            details["error"] = "".join(traceback.format_exception_only(type(error), error)).strip()
            self.stages.append(StageReport("write", "rejected", time.perf_counter() - t0,
                                           getattr(error, "category", None), str(error), details))

    @property
    def temperatures(self) -> tuple:
        """The temperatures actually broadened, K (0 K not included)."""
        return tuple(sorted(self._heated))

    def stage(self, name) -> StageReport:
        return next(s for s in self.stages if s.stage == name)

    def cross_sections(self, temperature=None) -> dict:
        """``{MT: (E, sigma)}``, float64 arrays, every reaction with an MT
        including the sums (MT1, 4, 103...).

        ``None`` or 0: the 0 K reconstruction (a repeated energy is a real
        jump). A broadened ``temperature``: the broadened values below
        ``thnmax`` and the 0 K ones above, on that temperature's union grid,
        thinned as the plan says; each reaction starts at its threshold.
        ``KeyError`` for a temperature that was not broadened.
        """
        if not temperature:
            if self._cold is None:
                self._cold = {mt: (_contiguous(t.energies), _contiguous(t.values))
                              for mt, t in self.reconstruction.tables_by_mt().items()}
            return self._cold
        for T, tables in self._heated.items():
            if abs(T - float(temperature)) <= 1e-9 * T:
                return tables
        raise KeyError(f"{temperature} K was not broadened; broadened: {list(self.temperatures)}")

    @property
    def probability_tables(self) -> dict | None:
        """``{T: tables}`` when the URR stage ran, else ``None``. Each ``tables``:
        ``energies`` (NE,), and ``cdf``, ``total``, ``elastic``, ``fission``,
        ``capture``, ``heating`` as (NE, NB) arrays; ``factors`` True when the
        band values multiply the smooth (dilute) cross sections rather than
        being barns; ``interpolation`` (2 lin-lin, 5 log-log between energies);
        ``inelastic`` and ``absorption``, the competing MTs (<= 0: none).
        """
        if not self._urr:
            return None
        from kika.nuclear_data.model.processed_forms import PROBABILITY_TABLE_ROWS
        out = {}
        for T, table in self._urr.items():
            entry = dict(energies=_contiguous(table.energies), factors=bool(table.multiplySmooth),
                         interpolation=int(table.interpolation), inelastic=int(table.inelastic),
                         absorption=int(table.absorption))
            for name in PROBABILITY_TABLE_ROWS:
                entry[name] = _contiguous(table.row(name))
            out[T] = entry
        return out

    @property
    def heating(self) -> dict | None:
        """``{T: {301: (E, h), 444: (E, d)}}`` when the heating stage ran, else ``None``.

        ``T`` in kelvin, ``0.0`` for the reconstruction. MT301 is the heating
        (KERMA times cross section) and MT444 the damage energy production,
        both eV·b on the temperature's MT1 grid, as NJOY HEATR writes them with
        ``local = 0``: photons carried away, fission as ``sigma_f (EFR + EGP)``
        from MF1/458 less its prompt photons. Not OpenMC's MT301, whose fission
        part is ``(EFR + EB) sigma_f``; ``write_hdf5`` computes that one itself.
        """
        if not self._heating:
            return None
        return {T: dict(v) for T, v in self._heating.items()}

    def write_pendf(self, path, *, progress: Progress | None = None, temperatures=None,
                    mat=None, workers=None):
        """Write the PENDF: one material per temperature, as NJOY BROADR leaves it.

        ``temperatures`` (kelvin; ``0`` is the reconstruction) chooses the
        materials; ``None`` writes 0 K and every broadened temperature, in
        increasing order. Each material repeats the MAT with its own TEMP in
        MF1/451 and LRP=2, keeps MF2 and carries the MF3 of its temperature;
        a broadened one also carries MF2/MT153, the URR probability tables in
        PURR's layout, when the URR stage produced them at that temperature
        (written as factors, LSSF=1, with a zero heating row, as PURR leaves
        it without HEATR). When the heating stage ran, every material carries
        MF3/MT301 and MT444 as HEATR adds them (:attr:`heating`).

        The 0 K material is ``EndfReconstruction.write_pendf``'s: verified
        against the resonance physics before and after the reload. Each
        broadened one is read back and compared with the table in memory, every
        point within half the 0.1 % tolerance, and its MT153 parsed back. The
        destination is replaced only if every material passes.

        Raises the reconstruction's errors for the 0 K material, else
        :class:`ProcessingOutputError` (``pendf-temperature-not-processed``,
        ``pendf-failed``). Each call appends its own ``write`` stage.
        """
        import tempfile
        t0 = time.perf_counter()
        path = Path(path)
        report = progress or (lambda *_: None)
        report("write", None)
        if mat is None:
            mat = (self.reconstruction.evaluation or {}).get("mat")
        try:
            if self.source_format != "endf":
                # GNDS -> ENDF-6 cannot write MF2/151 without the ENDF bookkeeping
                # a GNDS file does not carry (QX, LRX, LAD, the particle pairs).
                raise ProcessingOutputError(
                    "pendf-needs-endf-source", PROCESSING_REJECTION_CATEGORIES["pendf-needs-endf-source"])
            wanted = [0.0] + list(self.temperatures) if temperatures is None \
                else sorted({float(t) for t in temperatures})
            for T in wanted:
                if T and T not in self._heated:
                    raise ProcessingOutputError(
                        "pendf-temperature-not-processed",
                        f"{T:g} K was not broadened; broadened: {list(self.temperatures)}")
            if not wanted:
                raise ProcessingOutputError("pendf-failed", "no temperature to write")
            share = 1.0 / len(wanted)
            with tempfile.TemporaryDirectory(prefix="kika-pendf-", dir=path.parent if path.parent.exists() else None) as scratch:
                pieces, tables153 = [], []
                for i, T in enumerate(wanted):
                    one = Path(scratch) / f"T{i}.pendf"
                    step = (lambda stage, f, i=i: report("write", (i + (f or 0.0)) * share))
                    if T == 0.0:
                        self.reconstruction.write_pendf(one, progress=step, mat=mat, workers=workers)
                    else:
                        tables153.append(self._write_heated_material(one, T, mat))
                        step("write", 1.0)
                    if self._heating:
                        _insert_heating(one, self._heating[T], T)
                    pieces.append(one.read_text().rstrip("\n").split("\n"))
                lines = [pieces[0][0]]
                for body in pieces:
                    lines.extend(body[1:-1])      # drop each TPID and TEND, keep the MEND
                lines.append(pieces[0][-1])
                provisional = Path(scratch) / "tape.pendf"
                provisional.write_text("\n".join(lines) + "\n", newline="\n")
                path.parent.mkdir(parents=True, exist_ok=True)
                provisional.replace(path)
        except ProcessingOutputError as error:
            self._written("pendf", t0, path, error)
            raise
        except Exception as error:
            if getattr(error, "category", None) is None:
                error = ProcessingOutputError("pendf-failed", f"{type(error).__name__}: {error}")
                self._written("pendf", t0, path, error)
                raise error
            self._written("pendf", t0, path, error)
            raise
        report("write", 1.0)
        with153 = [T for T, has in zip([t for t in wanted if t], tables153) if has]
        summary = (f"PENDF with {len(wanted)} material{'s' if len(wanted) > 1 else ''} at "
                   f"{', '.join(f'{T:g}' for T in wanted)} K"
                   + (f"; URR tables (MT153) at {', '.join(f'{T:g}' for T in with153)} K" if with153 else "")
                   + ("; heating (MT301, MT444) in every material" if self._heating else "")
                   + "; each material verified on reload")
        self._written("pendf", t0, path, summary=summary, temperatures=wanted, mt153=with153,
                      heating=sorted(HEATING_MTS) if self._heating else [])
        return path

    def _write_heated_material(self, path, T, mat=None) -> bool:
        """One broadened material at ``T`` on its own tape, read back and checked.
        Returns whether it carries MF2/MT153."""
        from kika.endf.read_endf import read_endf
        from kika.endf.model_adapter import decodeMF3MT
        from kika.endf.writers.assemble import writeEndfTape
        from kika.endf.writers.update_directory import update_mf1_directory
        from kika.processing.resonances.grid import error_ratio
        from kika.endf.writers.assemble import _mf3Bearing
        rec = self.reconstruction
        suite = _heated_suite(rec, T, self._heated[T])
        conversion = writeEndfTape(suite, path, mat=mat, label=HEATED_LABEL)
        if not conversion.isClean:
            raise ProcessingOutputError("pendf-failed", f"{T:g} K: the conversion lost data: {conversion}")
        from kika.endf.model_adapter.pendf import trimDescription
        trimDescription(path)  # NJOY's HEATR does not survive a long MF1 text
        has153 = T in (self._urr or {})
        if has153:
            ev = rec.evaluation or {}
            _insert_mt153(path, self._urr[T], T, za=ev.get("za"), awr=ev.get("awr"))
            update_mf1_directory(str(path), added_sections={(2, 153)})
        # Read back: TEMP, LRP and every MF3 against the table in memory.
        tape = read_endf(str(path), mf_numbers=[1, 3])
        head = tape.mf[1].mt[451]
        if abs(float(head._temp) - T) > 1e-6 * T or int(head._lrp) != 2:
            raise ProcessingOutputError("pendf-failed", f"{T:g} K: TEMP or LRP changed on reload")
        options = rec.result.options
        written = {r.ENDF_MT for r in _mf3Bearing(suite) if r.ENDF_MT is not None}
        for mt, (x, y) in self._heated[T].items():
            if mt not in tape.mf[3].mt:
                if mt in written:
                    raise ProcessingOutputError("pendf-failed", f"{T:g} K: MT{mt} missing on reload")
                continue
            reaction, _ = decodeMF3MT(tape.mf[3].mt[mt])
            form = reaction.crossSection["eval"]
            xr, yr = _flat(form)
            keep = np.ones(len(x), bool)
            keep[1:] &= np.diff(x) > 1e-11 * x[1:]       # skip the two faces of a step
            keep[:-1] &= np.diff(x) > 1e-11 * x[:-1]
            back = np.interp(x[keep], xr, yr)
            ratio = error_ratio(back, y[keep], options)
            if ratio.size and float(np.max(ratio)) > 0.5:
                i = int(np.argmax(ratio))
                raise ProcessingOutputError(
                    "pendf-failed", f"{T:g} K MT{mt}: the reload differs at {x[keep][i]:.7g} eV "
                                    f"({back[i]:.7g} b vs {y[keep][i]:.7g} b)")
        if has153:
            _check_mt153(path, self._urr[T])
        return has153

    def _full_suite(self, progress=None, category="hdf5-failed"):
        """The whole tape decoded (distributions, photons, fission data), with
        the reconstruction attached as its 0 K forms. Read again here: the
        processing itself reads MF1-MF3 only."""
        from kika.endf.read_endf import read_endf
        from kika.endf.model_adapter import decodeReactionSuite
        if self.source_format == "gnds":
            # Read whole already; the reconstruction's own suite, copied, with
            # the containers it was reconstructed without.
            suite = copy.deepcopy(self.reconstruction.suite)
            for name, container in self._aside.items():
                setattr(suite, name, copy.deepcopy(container))
            for reaction in _entries_of(suite):
                reaction.crossSection.forms.pop(self.reconstruction.result.label, None)
            labels = [st.label for st in suite.styles if st.label == self.reconstruction.result.label]
            if labels:
                from kika.nuclear_data.model import Styles
                suite.styles = Styles([st for st in suite.styles if st.label not in labels])
            try:
                work, _ = _working_suite(self.reconstruction, suite)
            except ValueError as error:
                raise ProcessingOutputError(category, str(error)) from error
            return work
        try:
            suite, conversion = decodeReactionSuite(read_endf(str(self.reconstruction.path)))
        except Exception as error:
            raise ProcessingOutputError(category, f"the tape could not be read whole: {error}") from error
        suite.report = conversion
        if progress:
            progress("write", 0.1)
        try:
            work, _ = _working_suite(self.reconstruction, suite)
        except ValueError as error:
            raise ProcessingOutputError(category, str(error)) from error
        return work

    def write_gnds(self, path, *, progress: Progress | None = None) -> Path:
        """Write the processed evaluation as GNDS, as FUDGE's ``processProtare`` does.

        The whole evaluation (``eval``), the 0 K reconstruction
        (``crossSectionReconstructed``, ``recon``) and one ``heated`` style per
        broadened temperature (``heated_000``, ``heated_001``... in increasing
        temperature, derived from ``recon``), each with its cross sections.
        The URR probability tables are **not** written: GNDS 2.0, the schema
        kika writes and validates, has no element for them (they are 2.1); the
        stage lists them under ``omitted``. Covariances are not written either.
        The file is read back and its styles and temperatures checked before it
        replaces ``path``.

        Raises :class:`ProcessingOutputError` (``gnds-failed``). Each call
        appends its own ``write`` stage.
        """
        import tempfile
        import kika
        t0 = time.perf_counter()
        path = Path(path)
        if progress:
            progress("write", None)
        temps = list(self.temperatures)
        labels = {T: f"heated_{i:03d}" for i, T in enumerate(temps)}
        omitted = []
        if self._urr:
            omitted.append("URR probability tables (GNDS 2.0 has no element for them)")
        try:
            suite = self._full_suite(progress, category="gnds-failed")
            if getattr(suite, "covarianceSuite", None) is not None:
                suite.covarianceSuite = None
                omitted.append("covariances")
            source = self.reconstruction.result.label
            for T in temps:
                _add_heated(suite, labels[T], T, self._heated[T], source)
            if progress:
                progress("write", 0.5)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="kika-gnds-", dir=path.parent) as scratch:
                provisional = Path(scratch) / path.name
                try:
                    kika.write(suite, provisional, format="gnds")
                    back = kika.read(provisional, format="gnds", covariances=False)
                except Exception as error:
                    raise ProcessingOutputError("gnds-failed", f"{type(error).__name__}: {error}") from error
                styles = {st.label: st for st in back.styles}
                for T, label in labels.items():
                    style = styles.get(label)
                    if style is None or abs(float(style.temperature.value) - T) > 1e-9 * T:
                        raise ProcessingOutputError("gnds-failed", f"{label} at {T:g} K did not read back")
                if source not in styles:
                    raise ProcessingOutputError("gnds-failed", "the 0 K reconstruction did not read back")
                provisional.replace(path)
        except ProcessingOutputError as error:
            self._written("gnds", t0, path, error, temperatures=temps)
            raise
        if progress:
            progress("write", 1.0)
        summary = ("GNDS with eval, recon (0 K)"
                   + "".join(f", {labels[T]} ({T:g} K)" for T in temps)
                   + ("; omitted: " + "; ".join(omitted) if omitted else ""))
        self._written("gnds", t0, path, summary=summary, temperatures=[0.0] + temps,
                      styles=[source] + [labels[T] for T in temps], omitted=omitted)
        return path

    def write_hdf5(self, path, *, progress: Progress | None = None, index: bool = False) -> Path:
        """Write an OpenMC HDF5 neutron file (``data_neutron``) at ``path``.

        Every broadened temperature, on its own grid; the URR tables when the
        URR stage produced them; the distributions tabulated for Monte Carlo,
        KERMA (MT301/901), damage (MT444) and gas production, as
        :func:`kika.hdf5.processing.processSuite` builds them. The broadening
        and the URR tables are reused, not recomputed; the distributions and
        heating are computed here, since only this format needs them, and are
        the slow part. The file is read back and compared before it replaces
        ``path``. ``index`` also writes or updates ``cross_sections.xml`` in
        the same directory.

        Raises :class:`ProcessingOutputError`: ``hdf5-needs-temperature``
        when nothing was broadened (OpenMC needs a temperature), else
        ``hdf5-failed``. Each call appends its own ``write`` stage.
        """
        from kika.hdf5.writer import writeSuite
        t0 = time.perf_counter()
        path = Path(path)
        if progress:
            progress("write", None)
        temps = list(self.temperatures)
        omitted = []
        try:
            if not temps or not self._broadened:
                raise ProcessingOutputError(
                    "hdf5-needs-temperature", PROCESSING_REJECTION_CATEGORIES["hdf5-needs-temperature"])
            work = self._full_suite(progress)
            omit = ()
            resonances = work.resonances
            if resonances is not None and resonances.unresolved is not None and not self._urr:
                omit = ("urr",)
                omitted.append("URR probability tables (the URR stage did not produce them)")
            try:
                writeSuite(
                    work, path.parent, temperatures=temps, fileName=path.name, index=index,
                    urrTables=dict(self._urr) or None, broadened={T: self._broadened[T] for T in temps},
                    thnmax=self.thnmax, context=self.reconstruction.context, thinning=self.plan.thinning,
                    workers=self.plan.workers, omit=omit)
            except Exception as error:
                raise ProcessingOutputError("hdf5-failed", f"{type(error).__name__}: {error}") from error
        except ProcessingOutputError as error:
            self._written("hdf5", t0, path, error, temperatures=temps)
            raise
        if progress:
            progress("write", 1.0)
        summary = (f"OpenMC HDF5 at {', '.join(f'{T:g}' for T in temps)} K"
                   + ("; with URR probability tables" if self._urr else "")
                   + ("; omitted: " + "; ".join(omitted) if omitted else ""))
        self._written("hdf5", t0, path, summary=summary, temperatures=temps, omitted=omitted)
        return path

    def write_ace(self, directory, *, temperatures=None, extension: str | None = None,
                  xsdir: bool = True, replace: bool = False,
                  progress: Progress | None = None) -> list:
        """Write one ACE table per broadened temperature into ``directory``.

        The same processed suite :meth:`write_hdf5` writes (the broadening and
        the URR tables of this run, the distributions tabulated for Monte
        Carlo, KERMA, damage and gas production), written by
        :func:`kika.ace.model_adapter.encodeAce` with ACER's conventions:
        ZAID ``1000 Z + A`` (an isomer ``+ 300 + 100 m``, as LANL names
        Am-242m1 95642), the suffix of the temperature (``.02c`` at 293.6 K),
        the legacy header, the KERMA of HEATR's ``local = 0`` as ESZ's heating
        number. Each table is ``<ZAID><suffix>`` (``8016.02c``); it is read
        back in strict mode and compared word for word before it replaces a
        file. ``xsdir`` adds its line to ``directory/xsdir`` (path relative to
        it), refusing a table name the file already has unless ``replace``.

        ``temperatures`` (kelvin) picks some of the broadened ones; ``extension``
        overrides the suffix and needs a single temperature. The
        charged-particle blocks are not written. Returns the paths written.

        Raises :class:`ProcessingOutputError`: ``ace-needs-temperature`` when
        nothing was broadened, ``ace-temperature-not-processed``,
        ``ace-table-exists``, else ``ace-failed``. Each call appends its own
        ``write`` stage.
        """
        from kika.ace.model_adapter import encodeAce
        from kika.ace.writers.write_ace import write_ace_checked
        from kika.ace.xsdir import XsdirEntryExists, add_xsdir_entry, xsdir_line_for
        from kika.hdf5.processing import processSuite
        from kika.nuclear_data.model import ConversionReport

        t0 = time.perf_counter()
        directory = Path(directory)
        if progress:
            progress("write", None)
        temps = list(self.temperatures)
        written, omitted = [], []
        try:
            if not temps or not self._broadened:
                raise ProcessingOutputError(
                    "ace-needs-temperature", PROCESSING_REJECTION_CATEGORIES["ace-needs-temperature"])
            if temperatures is not None:
                wanted = [float(t) for t in np.atleast_1d(temperatures)]
                missing = [t for t in wanted if not any(abs(t - T) <= 1e-9 * T for T in temps)]
                if missing:
                    raise ProcessingOutputError(
                        "ace-temperature-not-processed",
                        f"{missing} K not broadened in this run; broadened: {temps}")
                temps = [T for T in temps if any(abs(t - T) <= 1e-9 * T for t in wanted)]
            if extension is not None and len(temps) != 1:
                raise ProcessingOutputError(
                    "ace-failed", f"extension={extension!r} names one table, and {len(temps)} "
                                  f"temperatures {temps} were asked for")
            work = self._full_suite(progress, category="ace-failed")
            omit = ()
            resonances = work.resonances
            if resonances is not None and resonances.unresolved is not None and not self._urr:
                omit = ("urr",)
                omitted.append("URR probability tables (the URR stage did not produce them)")
            report = ConversionReport()
            try:
                processed = processSuite(
                    work, temperatures=temps, report=report, urrTables=dict(self._urr) or None,
                    broadened={T: self._broadened[T] for T in temps}, thnmax=self.thnmax,
                    context=self.reconstruction.context, thinning=self.plan.thinning,
                    workers=self.plan.workers, omit=omit)
            except Exception as error:
                raise ProcessingOutputError("ace-failed", f"{type(error).__name__}: {error}") from error
            directory.mkdir(parents=True, exist_ok=True)
            for i, T in enumerate(temps):
                try:
                    ace, _ = encodeAce(processed, T, extension=extension, report=report)
                    name = f"{ace.header.zaid}{ace.header.extension}"
                    target = directory / name
                    write_ace_checked(ace, target)
                except ProcessingOutputError:
                    raise
                except Exception as error:
                    raise ProcessingOutputError("ace-failed",
                                                f"{T:g} K: {type(error).__name__}: {error}") from error
                if xsdir:
                    try:
                        add_xsdir_entry(directory / "xsdir", xsdir_line_for(ace, name), replace=replace)
                    except XsdirEntryExists as error:
                        raise ProcessingOutputError("ace-table-exists", str(error)) from error
                written.append(target)
                if progress:
                    progress("write", (i + 1) / len(temps))
            omitted += list(report.unsupported)
        except ProcessingOutputError as error:
            self._written("ace", t0, directory, error, temperatures=temps, files=[str(p) for p in written])
            raise
        summary = (f"ACE {', '.join(p.name for p in written)}"
                   + ("; with URR probability tables" if self._urr else "")
                   + ("; omitted: " + "; ".join(omitted) if omitted else ""))
        self._written("ace", t0, directory, summary=summary, temperatures=temps,
                      files=[str(p) for p in written], omitted=omitted)
        return written

    def report_document(self):
        """The reconstruction report with a section per processing stage
        (``to_text()``, ``to_html()``, ``to_dict()``, ``save(path)``)."""
        return ProcessingReport(self.reconstruction.report_document(), self.stages, self.thnmax)

    def to_dict(self) -> dict:
        """Everything as plain types and float64 arrays (not lists): ``path``,
        ``plan``, ``evaluation``, ``stages``, ``thnmax_eV``, ``temperatures``,
        ``cross_sections`` (``{T: {MT: {"energies", "values"}}}``, 0.0 for 0 K)
        and ``probability_tables`` (as the property, or ``None``)."""
        plan = asdict(self.plan)
        plan["reconstruction"] = None if self.plan.reconstruction is None else repr(self.plan.reconstruction)
        temps = (0.0,) + self.temperatures
        return dict(
            path=str(self.reconstruction.path), plan=plan,
            evaluation=self.reconstruction.evaluation,
            stages=[asdict(s) for s in self.stages], thnmax_eV=self.thnmax, temperatures=list(temps),
            cross_sections={T: {mt: dict(energies=x, values=y)
                                for mt, (x, y) in self.cross_sections(T).items()} for T in temps},
            probability_tables=self.probability_tables)

    def __repr__(self):
        states = ", ".join(f"{s.stage}={s.status}" for s in self.stages)
        return f"<EndfProcessing {self.reconstruction.path.name}: {states}>"


def _heated_tables(work, label, broadened, hi, thinning):
    """``{MT: (E, sigma)}`` at one temperature: the broadened leaves below ``hi``,
    0 K above, on their union grid; the sums rebuilt from their leaves."""
    from kika.hdf5 import processing as hp
    from kika.processing.resonances.suite import _entries, _graph
    entries = {k: r for k, r in _entries(work).items() if label in r.crossSection.forms}
    graph, _ = _graph(entries, label)
    leaves = [k for k in entries if k not in graph]
    cold = {k: entries[k].crossSection[label] for k in leaves}
    # Each form's abscissae with a repeated energy (a step) nudged up as ACER
    # does, so the union keeps both sides of every step instead of a ramp.
    def knots(form):
        x, y = hp._flatten(form)
        return hp._strictly(x, y)[0]
    hot_grid = np.unique(np.concatenate([knots(broadened[k.label]) for k in leaves]))
    cold_grid = np.unique(np.concatenate([knots(f) for f in cold.values()]))
    grid = np.unique(np.concatenate([hot_grid[hot_grid <= hi], cold_grid[cold_grid > hi]]))
    # Each reaction starts where the evaluation's MF3 starts (its threshold),
    # as BROADR keeps it: a union-grid zero below it would put the first
    # energy under the kinematic threshold, which ACER flags. The leaves are
    # cut there before the sums are built, so the sums stay exact.
    def start(key):
        first = hp._reactionRange(entries[key])[0]
        return min(int(np.searchsorted(grid, first * (1 - 1e-12), side="left")), len(grid) - 1)
    values, starts = {}, {}
    for k in leaves:
        values[k] = np.where(grid <= hi, hp._onGrid(broadened[k.label], grid), hp._onGrid(cold[k], grid))
        starts[k] = start(k)
        values[k][:starts[k]] = 0.0

    def total(key):
        if key not in values:
            values[key] = np.sum([total(c) for c in graph[key]], axis=0)
            starts[key] = min([start(key)] + [starts[c] for c in graph[key]])
        return values[key]
    for key in graph:
        total(key)
    mts = {k: entries[k].ENDF_MT for k in values if entries[k].ENDF_MT is not None}
    if thinning:
        thinned_grid, thinned = hp._thin(grid, {k: (values[k], starts[k]) for k in mts}, float(thinning))
        grid = thinned_grid
        values = {k: v for k, (v, _) in thinned.items()}
        starts = {k: s for k, (_, s) in thinned.items()}
    grid = _contiguous(grid)
    return {mts[k]: (grid[starts[k]:], _contiguous(values[k][starts[k]:])) for k in mts}


def _entries_of(suite):
    from kika.processing.resonances.suite import _entries
    return _entries(suite).values()


#: The label of the broadened forms a PENDF material is written from.
HEATED_LABEL = "heated"

#: ``write_pendf`` writes one material per broadened temperature, with MT153.
PENDF_TEMPERATURES_SUPPORTED = True


def _flat(form):
    """``(x, y)`` of a lin-lin form, the faces of a step kept as a repeated x."""
    from kika.hdf5 import processing as hp
    return hp._flatten(form)


def _heated_suite(rec, T, tables):
    """The reconstruction's (MF1-MF3) suite with a ``heated`` style at ``T``,
    each reaction on its own points (``ownPoints``), as BROADR writes a PENDF."""
    suite = copy.deepcopy(rec.suite)
    source = rec.result.label if rec.result.label in suite.styles.labels else "eval"
    return _add_heated(suite, HEATED_LABEL, T, tables, source, ownPoints=True)


#: ``ownPoints``: a point a reaction holds only because another reaction needs
#: it is dropped when the reaction, read lin-lin, reproduces it to this relative
#: tolerance -- far below the seven figures of an ENDF number, so nothing written
#: changes, while the table shrinks to its own grid.
OWN_POINTS_RTOL = 1.0e-9


def _add_heated(suite, label, T, tables, source, ownPoints=False):
    """Add a ``heated`` style ``label`` at ``T`` (derived from ``source``) and every
    reaction's broadened table under it, on the common union grid, steps
    restored as repeated energies (regions) so a writer keeps both faces.

    ``ownPoints`` keeps, for each reaction, only the points its own table needs
    (:data:`OWN_POINTS_RTOL`). On the union grid every reaction carries the
    knots of all the others -- ~9x the lines of BROADR's material for Fe-56 --
    and NJOY's ACER stops on such a tape."""
    from kika.algebra import thin_common_grid
    from kika.hdf5 import processing as hp
    from kika.nuclear_data.model import Heated, PhysicalQuantity
    grid = max((x for x, _ in tables.values()), key=len)
    for mt, (x, _) in tables.items():
        if x[0] < grid[0]:
            grid = x
    xs = grid.copy()
    # A step was nudged up by ~1e-12 for the union grid; ENDF states it as the
    # same energy twice.
    nudged = np.flatnonzero(np.diff(xs) <= 2e-12 * xs[1:])
    xs[nudged + 1] = xs[nudged]
    # The date the style was made, which gnds.xsd requires on every derived
    # style (as `attach_broadening` dates its own).
    suite.styles.add(Heated(label, derivedFrom=source, date=_today(),
                            temperature=PhysicalQuantity(float(T), "K")))
    for reaction in list(suite.reactions) + list(suite.sums) + list(suite.fissionComponents):
        mt = reaction.ENDF_MT
        if mt is None or mt not in tables:
            continue
        x, y = tables[mt]
        full = np.zeros(len(grid))
        full[len(grid) - len(x):] = y
        if ownPoints:
            first = len(grid) - len(x)
            keep = thin_common_grid(xs, full[None, :], rtol=OWN_POINTS_RTOL, atol=0.0,
                                    keep=[first] if first > 0 else None)
            form = hp._pieces(xs[keep], full[keep])
        else:
            form = hp._pieces(xs, full)
        form.label = label
        reaction.crossSection[label] = form
    return suite


def _mt153_values(table, T):
    """PURR's MF2/MT153 numbers for one temperature: the HEAD and LIST fields
    and the body, per energy E, then each band's probability (not cumulative),
    total, elastic, fission, capture factors and a zero heating row."""
    from kika.processing.resonances.probability_tables import sigfig
    cdf = table.row("cdf")
    probability = np.diff(np.concatenate([np.zeros((cdf.shape[0], 1)), cdf], axis=1), axis=1)
    body = []
    for ie, E in enumerate(table.energies):
        body.append(sigfig(float(E), 7))
        for row in (probability[ie], table.row("total")[ie], table.row("elastic")[ie],
                    table.row("fission")[ie], table.row("capture")[ie]):
            body.extend(sigfig(float(v), 7) for v in row)
        body.extend([0.0] * table.bands)
    return body


def _insert_mt153(path, table, T, *, za, awr):
    """Put MF2/MT153 after MF2/MT151's SEND record, as PURR does."""
    from kika._records import format_endf_number
    lines = Path(path).read_text().rstrip("\n").split("\n")
    mat = int(lines[1][66:70])
    nbin, nunr = table.bands, len(table.energies)
    body = _mt153_values(table, T)

    def record(fields, seq):
        text = "".join(f if isinstance(f, str) else format_endf_number(f) for f in fields)
        return f"{text:<66}{mat:>4}{2:>2}{153:>3}{seq:>5}"

    def integer(n):
        return f"{int(n):>11}"
    new = [record([float(za), float(awr), integer(table.inelastic), integer(table.absorption),
                   integer(table.interpolation), integer(nbin)], 1),
           record([float(T), 0.0, integer(1), integer(0), integer(len(body)), integer(nunr)], 2)]
    for i in range(0, len(body), 6):
        new.append(record([float(v) for v in body[i:i + 6]], len(new) + 1))
    new.append(f"{'':<66}{mat:>4}{2:>2}{0:>3}{99999:>5}")
    at = next(i for i, l in enumerate(lines)
              if l[70:72].strip() == "2" and l[72:75].strip() == "0" and l[66:70].strip() == str(mat))
    lines[at + 1:at + 1] = new[:-1] + [new[-1]]
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")


def _check_mt153(path, table):
    """Parse MF2/MT153 back and compare it with the table, to seven digits."""
    from kika._records import parse_number as parse_endf_number
    lines = [l for l in Path(path).read_text().split("\n")
             if l[70:72].strip() == "2" and l[72:75].strip() == "153"]
    numbers = []
    for l in lines[2:]:
        for k in range(6):
            field = l[11 * k:11 * (k + 1)]
            if field.strip():
                numbers.append(parse_endf_number(field))
    expected = _mt153_values(table, 0.0)
    if len(numbers) != len(expected) or not np.allclose(numbers, expected, rtol=1e-6, atol=0.0):
        raise ProcessingOutputError("pendf-failed", "MF2/MT153 does not read back as written")


#: The heating sections HEATR adds to a PENDF with its default request and damage.
HEATING_MTS = (301, 444)


def _heating_section(mt, energies, values, za, awr, mat):
    """MF3/MT``mt`` text for one heating table, as HEATR writes it (QM = QI = 0)."""
    from kika.endf.model_adapter import encodeMF3MT
    from kika.hdf5 import processing as hp
    from kika.nuclear_data.model import CrossSection, EndfProvenance, OutputChannel, Q, Reaction, ReactionId
    xs = np.array(energies, dtype=float)
    # A step was nudged up by ~1e-12 for the union grid; ENDF states it as the
    # same energy twice (as _add_heated does).
    nudged = np.flatnonzero(np.diff(xs) <= 2e-12 * xs[1:])
    xs[nudged + 1] = xs[nudged]
    form = hp._pieces(xs, np.asarray(values, float))
    form.label = HEATED_LABEL
    reaction = Reaction(ReactionId(f"MT{mt}", ENDF_MT=mt), CrossSection({HEATED_LABEL: form}),
                        outputChannel=OutputChannel(Q=Q(value=0.0, unit="eV")),
                        provenance=EndfProvenance(mat=mat, awr=awr, za=za, qm=0.0, lr=0))
    section, _ = encodeMF3MT(reaction, mat, label=HEATED_LABEL, mergeRegions=True)
    return str(section).rstrip("\n").split("\n")


def _insert_heating(path, tables, T):
    """Put MF3/MT301 and MT444 in the one-material tape at ``path``, in MT
    order, rebuild the MF1 directory, and check they read back."""
    from kika._records import parse_number
    from kika.endf.read_endf import read_endf
    from kika.endf.writers.update_directory import update_mf1_directory
    lines = Path(path).read_text().rstrip("\n").split("\n")
    head = lines[1]
    mat = int(head[66:70])
    za, awr = int(round(parse_number(head[0:11]))), parse_number(head[11:22])
    for mt in HEATING_MTS:
        x, y = tables[mt]
        new = _heating_section(mt, x, y, za, awr, mat)
        mf3 = [i for i, l in enumerate(lines) if l[70:72].strip() == "3" and l[66:70].strip() == str(mat)]
        if not mf3:
            raise ProcessingOutputError("pendf-failed", f"{T:g} K: no MF3 to put MT{mt} in")
        # Before the first record of a higher MT, else before MF3's FEND (MT 0 and
        # the last MF3 line).
        at = next((i for i in mf3 if int(lines[i][72:75]) > mt), None)
        if at is None:
            at = mf3[-1]
        lines[at:at] = new
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")
    if not update_mf1_directory(str(path), added_sections={(3, mt) for mt in HEATING_MTS}):
        raise ProcessingOutputError("pendf-failed", f"{T:g} K: the MF1 directory could not be rebuilt")
    tape = read_endf(str(path), mf_numbers=[3])
    for mt in HEATING_MTS:
        x, y = tables[mt]
        section = tape.mf[3].mt.get(mt)
        if section is None:
            raise ProcessingOutputError("pendf-failed", f"{T:g} K: MT{mt} missing on reload")
        back = np.asarray(section._cross_sections, float)
        scale = float(np.max(np.abs(y))) if len(y) else 1.0
        if len(back) != len(y) or not np.allclose(back, y, rtol=1e-6, atol=1e-9 * (scale or 1.0)):
            raise ProcessingOutputError("pendf-failed", f"{T:g} K: MT{mt} does not read back as written")


def _on_grid(x, y, grid):
    """``y(x)`` read lin-lin on ``grid``, both faces of a step kept, zero below ``x[0]``."""
    from kika.hdf5 import processing as hp
    xs, ys = hp._strictly(np.asarray(x, float), np.asarray(y, float))
    return np.interp(grid, xs, ys, left=0.0, right=float(ys[-1]))


def _heating_grid(run, T):
    """``(grid, {MT: sigma on grid})`` at ``T``: the MT1 grid HEATR works on."""
    from kika.hdf5 import processing as hp
    tables = run.cross_sections(T)
    if T:
        # One union grid per temperature; each table is its tail from its threshold.
        grid = max((x for x, _ in tables.values()), key=len)
        sigma = {}
        for mt, (x, y) in tables.items():
            full = np.zeros(len(grid))
            full[len(grid) - len(x):] = y
            sigma[mt] = full
        return grid, sigma
    # 0 K: each reaction on its own compressed grid; read them all on MT1's.
    x1 = tables[1][0] if 1 in tables else np.unique(np.concatenate([x for x, _ in tables.values()]))
    grid, _ = hp._strictly(x1, np.zeros(len(x1)))
    return grid, {mt: _on_grid(x, y, grid) for mt, (x, y) in tables.items()}


def _fission_heating(out, work, grid, sigma, A):
    """HEATR's fission KERMA, MT318 with ``local = 0`` (``heatr.f90`` nheat, gheat):
    ``sigma_f (EFR + EGP)`` from MF1/458, less the energy of the prompt photons the
    evaluation states for fission. ``None`` for a material that does not fission."""
    from kika.hdf5.fission_energy import evaluate
    from kika.processing import heating as h
    from kika.processing.resonances import UnsupportedResonanceError
    from kika.processing.resonances.suite import _entries
    chances = [mt for mt in (19, 20, 21, 38) if mt in sigma]
    if 18 in sigma:
        sigma_f = sigma[18]
    elif chances:
        sigma_f = np.sum([sigma[mt] for mt in chances], axis=0)
    else:
        return None
    release = None
    for reaction in _entries(work).values():
        data = getattr(reaction.outputChannel, "fissionFragmentData", None)
        if reaction.ENDF_MT in h.FISSION_MTS and data is not None and data.fissionEnergyReleases:
            release = data.fissionEnergyReleases[0]
            break
    if release is None:
        raise UnsupportedResonanceError("a fission with no MF1/458: HEATR's fission heating needs "
                                        "its energy release", category="heating-unsupported")
    h0 = evaluate(release.promptProductKE, grid) + evaluate(release.promptGammaEnergy, grid)
    photon = h._Kinematics(A=A, Q=0.0, awp=0.0)
    gammas = np.zeros(len(grid))
    fissions = [r for r in list(out.reactions) + list(out.fissionComponents) + list(out.sums)
                if r.ENDF_MT in h.FISSION_MTS]
    # The prompt photons of fission are stated once, on MT18 (or the first chance).
    for reaction in sorted(fissions, key=lambda r: r.ENDF_MT != 18):
        found = False
        for product in reaction.outputChannel.products:
            mode = (getattr(product.provenance, "attributes", None) or {}).get("emission_mode", "prompt")
            form = h._distribution(product) if product.pid == "photon" and mode == "prompt" else None
            if form is not None:
                gammas = gammas + h._yield(product, grid) * h._meanEnergyOf(form, grid, photon)
                found = True
        if found:
            break
    return np.where(sigma_f > 0, sigma_f * (h0 - gammas), 0.0)


def _heating(run, plan, report):
    """``(StageReport, {T: {301: (E, h), 444: (E, d)}})``; never raises.

    The product distributions come from the OpenMC processing
    (:func:`kika.hdf5.processing.processSuite`, the slow part), reusing this
    run's broadening; the KERMA from :func:`kika.processing.heating.heatingOf`
    on each temperature's MT1 grid, as HEATR reads its PENDF."""
    from kika.hdf5 import processing as hp
    from kika.processing.heating import heatingOf
    from kika.processing.resonances import UnsupportedResonanceError
    from kika.processing.resonances.suite import _entries
    t0 = time.perf_counter()
    report("heating", None)
    temps = [0.0] + list(run.temperatures)
    details = dict(temperatures=temps, mts=list(HEATING_MTS), local=0)
    try:
        work = run._full_suite(category="heating-failed")
        out = hp.processSuite(work, temperatures=list(run.temperatures), thnmax=run.thnmax or "njoy",
                              context=run.reconstruction.context, thinning=plan.thinning,
                              workers=plan.workers, omit=("urr", "kerma", "damage"),
                              broadened={T: run._broadened[T] for T in run.temperatures})
        report("heating", 0.5)
        target = work.PoPs[work.target]
        Z, A = int(target.Z), int(target.A)
        awr = float(out.provenance.attributes["atomic_weight_ratio"])
        qm = {r.ENDF_MT: float(r.provenance.qm) for r in _entries(work).values()
              if r.ENDF_MT is not None and getattr(r.provenance, "qm", None) is not None}
        tables, notes = {}, []
        for i, T in enumerate(temps):
            grid, sigma = _heating_grid(run, T)
            result = heatingOf(out, grid, sigma, awr=awr, Z=Z, qm=qm, evaluation=work,
                               zaTarget=1000 * Z + A)
            kerma = result.kerma
            fission = _fission_heating(out, work, grid, sigma, awr)
            if fission is not None:
                kerma = kerma + fission
            grid = _contiguous(grid)
            tables[T] = {301: (grid, _contiguous(kerma)), 444: (grid, _contiguous(result.damage))}
            notes = result.notes
            report("heating", 0.5 + 0.5 * (i + 1) / len(temps))
    except Exception as error:
        category = getattr(error, "category", None)
        if category not in PROCESSING_REJECTION_CATEGORIES or not str(category).startswith("heating"):
            category = "heating-unsupported" if isinstance(error, UnsupportedResonanceError) else "heating-failed"
        details["error"] = "".join(traceback.format_exception_only(type(error), error)).strip()
        return StageReport("heating", "rejected", time.perf_counter() - t0, category,
                           f"{PROCESSING_REJECTION_CATEGORIES[category]} ({_describe(error)})", details), {}
    details["notes"] = list(notes)
    details["n_points"] = {T: int(len(t[301][0])) for T, t in tables.items()}
    fissile = any(r.ENDF_MT in (18, 19) for r in _entries(work).values())
    summary = (f"MT301 (KERMA) and MT444 (damage) at {', '.join(f'{T:g}' for T in temps)} K, as NJOY "
               f"HEATR with local = 0" + ("; fission from MF1/458 (EFR + EGP) less its prompt photons"
                                          if fissile else ""))
    return StageReport("heating", "done", time.perf_counter() - t0, None, summary, details), tables


def _working_suite(rec, suite=None):
    """A suite with the reconstruction as its 0 K forms, lin-lin, as the
    broadening reads them: a copy of the reconstruction's own (MF1-MF3) suite,
    which stays untouched so ``write_pendf`` still attaches and verifies the
    reconstruction itself, or ``suite``, a fresh decode of the whole tape for
    an output that needs the distributions too. Its orphan products (photons
    stated on a sum) and productions (MF10) carry no 0 K form; any other
    reaction must be one the reconstruction covered."""
    from kika.hdf5 import processing as hp
    from kika.nuclear_data.model import CrossSectionReconstructed
    from kika.processing.resonances.suite import _entries
    work = copy.deepcopy(rec.suite) if suite is None else suite
    label = rec.result.label
    for key, reaction in _entries(work).items():
        if key not in rec.result.forms:
            if key.container in ("orphanProducts", "productions"):
                continue
            raise ValueError(f"{key} is in the tape but not in the reconstruction")
        reaction.crossSection[label] = copy.deepcopy(rec.result.forms[key])
    work.styles.add(CrossSectionReconstructed(label, derivedFrom=rec.result.source_style,
                                              date=_today()))
    notes = hp._m().ConversionReport()
    hp._linearised(work, notes)
    return work, label


def _broaden(rec, plan, report):
    """``(StageReport, {T: tables}, thnmax, work, raw)``; never raises. ``work`` is
    the suite with its 0 K forms and ``raw`` the broadened forms
    (``{T: (forms, ceiling)}``), both ``None`` on a rejection."""
    from kika.hdf5 import processing as hp
    temps = plan.temperatures
    t0 = time.perf_counter()
    for T in temps:
        report(f"broaden:{T:g}", None)
    thnmax = None
    try:
        work, label = _working_suite(rec)
        if hp.RECON_LABEL != label:
            raise RuntimeError(f"the reconstruction label {label!r} is not the broadening's {hp.RECON_LABEL!r}")
        if plan.thnmax == "njoy":
            thresholds = [hp._reactionRange(r)[0] for r in list(work.reactions) + list(work.fissionComponents)
                          if (r.outputChannel.Q.value or 0.0) < 0]
            thnmax = float(hp.njoyThnmax(work, min(thresholds) if thresholds else None))
        else:
            thnmax = float(plan.thnmax)
        broadened = hp._broadenAll(work, list(temps), thnmax, None, plan.workers,
                                   progress=lambda i, f: report(f"broaden:{temps[i]:g}", float(f)))
        heated = {}
        for T in temps:
            forms, hi = broadened[T]
            heated[T] = _heated_tables(work, label, forms, hi, plan.thinning)
        thnmax = float(broadened[temps[0]][1])
    except Exception as error:
        category, engine = _engine_category(error)
        return StageReport("broaden", "rejected", time.perf_counter() - t0, category,
                           f"{PROCESSING_REJECTION_CATEGORIES[category]} ({_describe(error)})",
                           dict(temperatures=list(temps), thnmax_eV=thnmax, engine_category=engine,
                                error="".join(traceback.format_exception_only(type(error), error)).strip())), \
            {}, thnmax, None, None
    points ={T: int(max(len(x) for x, _ in t.values())) for T, t in heated.items()}
    summary = (f"broadened to {', '.join(f'{T:g}' for T in temps)} K below {thnmax:g} eV "
               f"(0 K values above, as NJOY BROADR); "
               + ", ".join(f"{n:,} points at {T:g} K" for T, n in points.items()))
    return StageReport("broaden", "done", time.perf_counter() - t0, None, summary,
                       dict(temperatures=list(temps), thnmax_eV=thnmax, n_points=points)), \
        heated, thnmax, work, broadened


def _urr(rec, plan, unresolved, report):
    """``(StageReport, {T: ProbabilityTables})``; never raises."""
    if not plan.urr:
        return StageReport("urr", "skipped", summary="not requested"), {}
    if unresolved is None:
        return StageReport("urr", "skipped", category="urr-absent",
                           summary=PROCESSING_REJECTION_CATEGORIES["urr-absent"]), {}
    if not plan.temperatures:
        return StageReport("urr", "skipped", category="urr-no-temperature",
                           summary=PROCESSING_REJECTION_CATEGORIES["urr-no-temperature"]), {}
    from kika.hdf5.processing import urrTablesOf
    from kika.processing.resonances import UnsupportedResonanceError
    from kika.processing.resonances.probability_tables import ProbabilityTableOptions
    t0 = time.perf_counter()
    report("urr", None)
    details = dict(temperatures=list(plan.temperatures), bins=plan.urr_bins, ladders=plan.urr_ladders,
                   range_eV=unresolved["range_eV"])
    try:
        options = ProbabilityTableOptions(bins=plan.urr_bins, ladders=plan.urr_ladders)
        tables = urrTablesOf(rec.suite, plan.temperatures, context=rec.context, workers=plan.workers,
                             options=options)
    except Exception as error:
        category = "urr-unsupported" if isinstance(error, UnsupportedResonanceError) else "urr-failed"
        details["error"] = "".join(traceback.format_exception_only(type(error), error)).strip()
        return StageReport("urr", "rejected", time.perf_counter() - t0, category,
                           f"{PROCESSING_REJECTION_CATEGORIES[category]} ({_describe(error)})", details), {}
    report("urr", 1.0)
    ne = len(next(iter(tables.values())).energies)
    details["n_energies"] = ne
    lo, hi = unresolved["range_eV"]
    return StageReport("urr", "done", time.perf_counter() - t0, None,
                       f"{plan.urr_bins} bands x {ne} energies ({lo:g}-{hi:g} eV) at "
                       f"{', '.join(f'{T:g}' for T in plan.temperatures)} K; "
                       f"{plan.urr_ladders} ladders (PURR's method, statistically equal to NJOY's)",
                       details), dict(tables)


def process_endf(path, plan: ProcessingPlan | None = None, progress: Progress | None = None) -> EndfProcessing:
    """Process an ENDF-6 neutron tape stage by stage (see the module doc for
    the stages, the progress calls and the failure contract).

    Called from a daemonic process, which may not start children, the
    broadening and the URR tables run in this process (``workers=1``); the
    results do not depend on the number of workers."""
    from .reconstruction import reconstruct_endf
    plan, report = _start(plan, progress)
    rec = reconstruct_endf(path, options=plan.reconstruction, progress=report)
    return _finish(rec, plan, report, "endf")


def _start(plan, progress):
    """The plan (one worker in a daemonic process, which may not start
    children) and the progress callback."""
    import multiprocessing
    from dataclasses import replace
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        # A spawned worker is importing the caller's main script, and the
        # script starts a run at module level. Running it here would redo the
        # whole processing in every worker; stop at once, so the parent's pool
        # breaks and its stage says why (_describe).
        raise RuntimeError(
            "process_endf/process_gnds was called while a worker process was importing the "
            "main script: start the run under `if __name__ == \"__main__\":`, or use "
            "ProcessingPlan(workers=1)")
    plan = ProcessingPlan() if plan is None else plan
    if multiprocessing.current_process().daemon and plan.workers != 1:
        plan = replace(plan, workers=1)
    report = (lambda stage, fraction: progress(stage, fraction)) if progress else (lambda *_: None)
    return plan, report


def _finish(rec, plan, report, source_format, decode_details=None, aside=None):
    """Every stage after the reconstruction, the same for ENDF and GNDS."""
    ev = rec.evaluation or {}
    resolved, unresolved = _ranges(ev)
    stats = rec.report
    n_points = int(stats.get("stored_points") or 0)
    stages = [
        StageReport("read", "done", rec.timings.get("read"), None,
                    f"{ev.get('zsymam') or rec.suite.target}, MAT {ev.get('mat')}, "
                    f"{ev.get('library') or 'unknown library'}",
                    dict(mat=ev.get("mat"), za=ev.get("za"), library=ev.get("library"),
                         temperature_K=ev.get("temperature_K"))),
        StageReport("decode", "done", rec.timings.get("decode"), None,
                    f"{len(rec.result.forms)} reactions; mass and spin from {rec.context_source}"
                    + ("; the file's own reconstruction is ignored, kika reconstructs from the "
                       "evaluated resonances" if (decode_details or {}).get("source_reconstruction") else ""),
                    dict(n_reactions=len(rec.result.forms), context_source=rec.context_source,
                         **(decode_details or {}))),
        StageReport("reconstruct", "done", rec.timings.get("reconstruct"), None,
                    f"{_resonance_line(resolved, unresolved)}; {n_points:,} points at 0 K",
                    dict(n_points=n_points,
                         n_resonances=resolved["resonances"] if resolved else 0,
                         formalisms=resolved["formalisms"] if resolved else [],
                         resolved_range_eV=resolved["range_eV"] if resolved else None,
                         unresolved_range_eV=unresolved["range_eV"] if unresolved else None,
                         native_kernel=rec.native_kernel)),
    ]
    heated, thnmax, work, raw = {}, None, None, None
    if plan.temperatures and plan.broaden:
        stage, heated, thnmax, work, raw = _broaden(rec, plan, report)
    else:
        stage = StageReport("broaden", "skipped", summary="not requested" + (
            "" if plan.temperatures else " (0 K only)"))
    stages.append(stage)
    stage, urr_tables = _urr(rec, plan, unresolved, report)
    stages.append(stage)
    run = EndfProcessing(rec, plan, stages, heated, thnmax, urr_tables, broadened=raw,
                         source_format=source_format)
    run._aside = aside or {}
    if not plan.heating:
        stages.append(StageReport("heating", "skipped", summary="not requested"))
    elif plan.temperatures and plan.broaden and not heated:
        stages.append(StageReport("heating", "skipped", summary="the broadening was rejected; "
                                  "heating needs every requested temperature"))
    else:
        stage, run._heating = _heating(run, plan, report)
        stages.append(stage)
    return run


def process_gnds(path, plan: ProcessingPlan | None = None, progress: Progress | None = None) -> EndfProcessing:
    """Process a GNDS neutron evaluation stage by stage: the same stages,
    contract and outputs as :func:`process_endf`, read from a
    ``reactionSuite`` instead of an ENDF tape.

    The 0 K cross sections are always kika's, reconstructed from the evaluated
    resonance parameters (``eval``). A ``crossSectionReconstructed`` (FUDGE's
    ``recon``) or any other processed style the file carries is set aside, and
    the ``decode`` stage says so (``details["source_reconstruction"]``): every
    later table then has kika's tolerance and verification, as from ENDF.
    A file with no resonance parameters takes its evaluated pointwise cross
    sections as they are. The MAT of the outputs is the one the evaluation's
    ``endfCompatible`` text states, when it does.
    """
    import kika
    from kika.processing.resonances import ENGINE_VERSION, ReconstructionOptions, native_available, reconstruct_suite
    from .reconstruction import DEFAULT_MAX_POINTS, EndfReconstruction
    plan, report = _start(plan, progress)
    path = Path(path)
    timings = {}
    t0 = time.perf_counter()
    report("read", None)
    suite = kika.read(str(path), format="gnds", covariances=False)
    timings["read"] = time.perf_counter() - t0
    report("read", 1.0)

    t0 = time.perf_counter()
    report("decode", None)
    set_aside = _set_aside_processed(suite)
    # Orphan products (photons stated on a sum, MT4/103-107) and productions
    # (MF10-like, under a reaction's MT) share another reaction's MT and carry
    # no resonance contribution: they sit out of the reconstruction, as an ENDF
    # tape's MF1-MF3 read never has them, and come back for the outputs that
    # carry them (write_hdf5, write_gnds).
    aside = {name: getattr(suite, name) for name in ("orphanProducts", "productions")}
    for name, container in aside.items():
        setattr(suite, name, type(container)())
    context, source = _context_from_model(suite)
    evaluation = _gnds_evaluation(suite, path)
    timings["decode"] = time.perf_counter() - t0
    report("decode", 1.0)

    t0 = time.perf_counter()
    report("reconstruct", None)
    options = ReconstructionOptions(max_points=DEFAULT_MAX_POINTS) if plan.reconstruction is None \
        else plan.reconstruction
    try:
        result = reconstruct_suite(suite, context, options=options,
                                   progress=lambda f: report("reconstruct", f))
    except ValueError as error:
        if "NeutronContext" not in str(error):
            raise
        from kika.processing.resonances import UnsupportedResonanceError
        raise UnsupportedResonanceError(
            "the GNDS file states no target mass or spin in PoPs, which the resonance "
            "reconstruction needs", category="context-missing") from error
    timings["reconstruct"] = time.perf_counter() - t0
    rec = EndfReconstruction(path, suite, result, context, source, timings, ENGINE_VERSION,
                             evaluation=evaluation, native_kernel=native_available())
    details = dict(source_format="gnds")
    if set_aside:
        details["source_reconstruction"] = f"{', '.join(set_aside)} in the file, ignored"
    return _finish(rec, plan, report, "gnds", details, aside)


def _set_aside_processed(suite):
    """Drop every style that is not evaluated (FUDGE's ``recon``, ``heated``...)
    and the cross-section forms under them; return their labels."""
    from kika.nuclear_data.model import Evaluated, Styles
    from kika.processing.resonances.suite import _entries
    kept = [st for st in suite.styles if isinstance(st, Evaluated)]
    dropped = [st.label for st in suite.styles if not isinstance(st, Evaluated)]
    if not dropped:
        return []
    for reaction in _entries(suite).values():
        for label in dropped:
            if label in reaction.crossSection.forms:
                del reaction.crossSection.forms[label]
    suite.styles = Styles(kept)
    return dropped


def _context_from_model(suite):
    """``(NeutronContext or None, source)``: the RML elastic pair when there is
    one, as for ENDF; otherwise ``None``, and the reconstruction takes the
    target's mass and spin from PoPs."""
    from kika.processing.resonances import NeutronContext
    resonances = suite.resonances
    if resonances is None or (not resonances.resolved and resonances.unresolved is None):
        return None, "no resonance physics"
    for region in resonances.resolved or []:
        for pair in getattr(region.formalism, "resonanceReactions", None) or []:
            if getattr(pair, "reactionMT", None) == 2 and getattr(pair, "kinematics", None) is not None:
                b = pair.kinematics.particleB
                return NeutronContext(b.massRatio, b.spin), "RML elastic pair"
    return None, "PoPs mass and spin"


def _gnds_evaluation(suite, path):
    """What ``evaluation_summary`` gives for an ENDF tape, from the model."""
    import re
    from kika.nuclear_data.model import Evaluated
    from .reconstruction_report import KRM_NAMES, file_sha256
    out = dict(file=Path(path).name, path=str(Path(path).resolve()), sha256=file_sha256(path),
               library=getattr(suite, "evaluation", None) or None)
    target = suite.PoPs[suite.target] if suite.target in suite.PoPs else None
    if target is not None and getattr(target, "Z", None) is not None:
        out["za"] = int(target.Z) * 1000 + int(target.A)
        out["zsymam"] = str(suite.target)
        try:
            neutron = suite.PoPs[suite.projectile]
            out["awr"] = target.mass.convertedTo("amu").value / neutron.mass.convertedTo("amu").value
        except Exception:
            pass
    for st in suite.styles:
        if isinstance(st, Evaluated):
            temperature = getattr(st, "temperature", None)
            if temperature is not None and getattr(temperature, "value", None) is not None:
                out["temperature_K"] = float(temperature.value) if temperature.unit == "K" else None
            text = _endf_compatible_text(st)
            match = re.search(r"MATERIAL\s+(\d+)", text or "")
            if match:
                out["mat"] = int(match.group(1))
            break
    rows = []
    resonances = suite.resonances
    for region in (getattr(resonances, "resolved", None) or []) if resonances is not None else []:
        f = region.formalism
        if hasattr(f, "spinGroups"):
            name = {"ReichMoore": "Reich-Moore", "FullRMatrix": f"R-Matrix Limited ({KRM_NAMES[4]})"}.get(
                str(getattr(f, "approximation", "")), "R-Matrix")
            count = sum(len(g.energies) for g in f.spinGroups)
        else:
            approx = str(getattr(getattr(f, "approximation", None), "value", getattr(f, "approximation", "")))
            name = {"singleLevel": "SLBW", "multiLevel": "MLBW"}.get(approx, "Breit-Wigner")
            count = sum(len(g.resonances) for g in f.resonanceParameters.spinGroups)
        rows.append(dict(lru=1, lrf=None, nro=None, naps=None, formalism=name, el_eV=float(region.domainMin),
                         eh_eV=float(region.domainMax), resonances=int(count)))
    if resonances is not None and resonances.unresolved is not None:
        u = resonances.unresolved
        rows.append(dict(lru=2, lrf=None, nro=None, naps=None, formalism="unresolved", el_eV=float(u.domainMin),
                         eh_eV=float(u.domainMax), urr_case=None, lssf=None))
    out["resonance_ranges"] = rows
    return out


def _endf_compatible_text(style):
    documentation = getattr(style, "documentation", None)
    for name in ("endfCompatible", "body"):
        part = getattr(documentation, name, None)
        text = getattr(part, "text", part)
        if isinstance(text, str) and "MATERIAL" in text:
            return text
    return None


class ProcessingReport:
    """The reconstruction report plus a "Processing stages" section.

    Wraps :class:`~kika.endf.reconstruction_report.ReconstructionReport`
    through its public methods only."""

    def __init__(self, reconstruction_report, stages, thnmax):
        self.reconstruction = reconstruction_report
        self._stages = [asdict(s) for s in stages]
        self._thnmax = thnmax

    def _rows(self):
        for s in self._stages:
            seconds = "-" if s["seconds"] is None else f"{s['seconds']:.1f} s"
            yield s["stage"], s["status"], seconds, s["category"] or "", s["summary"]

    def to_dict(self) -> dict:
        out = self.reconstruction.to_dict()
        out["processing_stages"] = self._stages
        out["thnmax_eV"] = self._thnmax
        return out

    def to_json(self, **kwargs) -> str:
        import json
        return json.dumps(self.to_dict(), **({"indent": 1, "default": str} | kwargs))

    def to_text(self, **kwargs) -> str:
        lines = ["", "PROCESSING STAGES", "-" * 17]
        for stage, status, seconds, category, summary in self._rows():
            lines.append(f"  {stage:<12} {status:<9} {seconds:>9}  {category}")
            lines.append(f"      {summary}")
        return self.reconstruction.to_text(**kwargs).rstrip() + "\n" + "\n".join(lines) + "\n"

    def _section(self):
        import html
        e = html.escape
        rows = "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in row) + "</tr>" for row in self._rows())
        return ('<div class="kika-report"><h2 id="kr-stages">Processing stages</h2><div class="wrap"><table>'
                "<thead><tr><th>Stage</th><th>Status</th><th>Time</th><th>Category</th><th>Summary</th></tr>"
                f"</thead><tbody>{rows}</tbody></table></div></div>")

    def to_html(self, **kwargs) -> str:
        page = self.reconstruction.to_html(**kwargs)
        cut = page.rfind("</body>")
        return page + self._section() if cut < 0 else page[:cut] + self._section() + page[cut:]

    def save(self, path):
        """Write by extension: ``.html``/``.htm``, ``.json``, anything else as text."""
        path = Path(path)
        suffix = path.suffix.lower()
        body = (self.to_html() if suffix in (".html", ".htm") else
                self.to_json() if suffix == ".json" else self.to_text())
        path.write_text(body, encoding="utf-8")
        return path
