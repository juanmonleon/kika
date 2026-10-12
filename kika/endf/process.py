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
Heating (HEATR) and writing the heated MF3, MF2/MT153 and MT301 into the PENDF
are not wired here yet: ``heating`` is reported ``skipped`` and
:meth:`EndfProcessing.write_pendf` writes the verified 0 K PENDF.
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
    "urr-no-temperature": "Probability tables are computed at the broadening temperatures, and none was requested.",
    "urr-unsupported": "The unresolved resonance parameters use data the probability-table method does not support.",
    "urr-failed": "The probability tables stopped with an internal error.",
    # heating (skipped)
    "heating-not-implemented": "Heating (KERMA) is not yet available in this processing chain.",
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
    temperature, broadening to where NJOY BROADR stops, URR tables, no heating.

    ``temperatures`` are kelvin, > 0; 0 K is the reconstruction and is always
    there, so ``()`` stops after it. ``thnmax`` is the top of the broadening in
    eV, or ``"njoy"`` for BROADR's default (the top of the resolved range).
    ``thinning`` is the relative tolerance each temperature's union grid is
    thinned to (BROADR's ``errthn``, 1e-3); ``None`` keeps every point.
    ``urr_bins``/``urr_ladders`` are PURR's bands and ladders. ``workers``
    bounds the processes of the broadening and the URR tables (default: up to
    eight).
    """
    temperatures: tuple = DEFAULT_TEMPERATURES
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
        heating=dict(applicable=False, predicted_rejection="heating-not-implemented",
                     reason=PROCESSING_REJECTION_CATEGORIES["heating-not-implemented"]),
    )
    return dict(evaluation=evaluation, resonance_ranges=ranges, resolved=resolved,
                unresolved=unresolved, reactions=reactions, stages=stages)


def _contiguous(x):
    return np.ascontiguousarray(x, dtype=np.float64)


class EndfProcessing:
    """The result of :func:`process_endf`. Picklable; every array is float64."""

    def __init__(self, reconstruction, plan, stages, heated, thnmax, urr_tables):
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
    def heating(self):
        """KERMA by MT and temperature; ``None`` until the heating stage is wired."""
        return None

    def write_pendf(self, path, *, progress: Progress | None = None, mat=None, workers=None):
        """Write the verified PENDF (see ``EndfReconstruction.write_pendf``).

        For now the 0 K MF3 only: the heated MF3, MF2/MT153 and MT301 are not
        written yet, and the ``write`` stage says so. Raises as
        ``EndfReconstruction.write_pendf`` does; the stage is recorded either way.
        """
        t0 = time.perf_counter()
        self.stages = [s for s in self.stages if s.stage != "write"]
        try:
            out = self.reconstruction.write_pendf(path, progress=progress, mat=mat, workers=workers)
        except Exception as error:
            self.stages.append(StageReport("write", "rejected", time.perf_counter() - t0,
                                           getattr(error, "category", None), str(error),
                                           dict(path=str(path))))
            raise
        note = "0 K cross sections written and verified on reload"
        if self.temperatures:
            note += "; the broadened temperatures are not written to the PENDF yet"
        self.stages.append(StageReport("write", "done", time.perf_counter() - t0, None, note,
                                       dict(path=str(Path(path)), temperatures=[0.0])))
        return out

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
    values = {}
    for k in leaves:
        values[k] = np.where(grid <= hi, hp._onGrid(broadened[k.label], grid), hp._onGrid(cold[k], grid))

    def total(key):
        if key not in values:
            values[key] = np.sum([total(c) for c in graph[key]], axis=0)
        return values[key]
    for key in graph:
        total(key)
    mts = {k: entries[k].ENDF_MT for k in values if entries[k].ENDF_MT is not None}
    starts = {}
    for k in mts:
        nonzero = np.flatnonzero(values[k] != 0)
        starts[k] = max(int(nonzero[0]) - 1, 0) if nonzero.size else 0
    if thinning:
        thinned_grid, thinned = hp._thin(grid, {k: (values[k], starts[k]) for k in mts}, float(thinning))
        grid = thinned_grid
        values = {k: v for k, (v, _) in thinned.items()}
        starts = {k: s for k, (_, s) in thinned.items()}
    return {mts[k]: (_contiguous(grid[starts[k]:]), _contiguous(values[k][starts[k]:])) for k in mts}


def _working_suite(rec):
    """A copy of the decoded suite with the reconstruction as its 0 K forms,
    lin-lin, as the broadening reads them. The tape's own suite is untouched,
    so ``write_pendf`` still attaches and verifies the reconstruction itself."""
    from kika.hdf5 import processing as hp
    from kika.nuclear_data.model import CrossSectionReconstructed
    from kika.processing.resonances.suite import _entries
    work = copy.deepcopy(rec.suite)
    label = rec.result.label
    for key, reaction in _entries(work).items():
        reaction.crossSection[label] = copy.deepcopy(rec.result.forms[key])
    work.styles.add(CrossSectionReconstructed(label, derivedFrom=rec.result.source_style))
    notes = hp._m().ConversionReport()
    hp._linearised(work, notes)
    return work, label


def _broaden(rec, plan, report):
    """``(StageReport, {T: tables}, thnmax)``; never raises."""
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
                           f"{PROCESSING_REJECTION_CATEGORIES[category]} ({error})",
                           dict(temperatures=list(temps), thnmax_eV=thnmax, engine_category=engine,
                                error="".join(traceback.format_exception_only(type(error), error)).strip())), {}, thnmax
    points ={T: int(max(len(x) for x, _ in t.values())) for T, t in heated.items()}
    summary = (f"broadened to {', '.join(f'{T:g}' for T in temps)} K below {thnmax:g} eV "
               f"(0 K values above, as NJOY BROADR); "
               + ", ".join(f"{n:,} points at {T:g} K" for T, n in points.items()))
    return StageReport("broaden", "done", time.perf_counter() - t0, None, summary,
                       dict(temperatures=list(temps), thnmax_eV=thnmax, n_points=points)), heated, thnmax


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
                           f"{PROCESSING_REJECTION_CATEGORIES[category]} ({error})", details), {}
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
    the stages, the progress calls and the failure contract)."""
    from .reconstruction import reconstruct_endf
    plan = ProcessingPlan() if plan is None else plan
    report = (lambda stage, fraction: progress(stage, fraction)) if progress else (lambda *_: None)
    rec = reconstruct_endf(path, options=plan.reconstruction, progress=report)

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
                    f"{len(rec.result.forms)} reactions; mass and spin from {rec.context_source}",
                    dict(n_reactions=len(rec.result.forms), context_source=rec.context_source)),
        StageReport("reconstruct", "done", rec.timings.get("reconstruct"), None,
                    f"{_resonance_line(resolved, unresolved)}; {n_points:,} points at 0 K",
                    dict(n_points=n_points,
                         n_resonances=resolved["resonances"] if resolved else 0,
                         formalisms=resolved["formalisms"] if resolved else [],
                         resolved_range_eV=resolved["range_eV"] if resolved else None,
                         unresolved_range_eV=unresolved["range_eV"] if unresolved else None,
                         native_kernel=rec.native_kernel)),
    ]
    heated, thnmax = {}, None
    if plan.temperatures:
        stage, heated, thnmax = _broaden(rec, plan, report)
    else:
        stage = StageReport("broaden", "skipped", summary="not requested (0 K only)")
    stages.append(stage)
    stage, urr_tables = _urr(rec, plan, unresolved, report)
    stages.append(stage)
    if plan.heating:
        stages.append(StageReport("heating", "skipped", category="heating-not-implemented",
                                  summary=PROCESSING_REJECTION_CATEGORIES["heating-not-implemented"]))
    else:
        stages.append(StageReport("heating", "skipped", summary="not requested"))
    return EndfProcessing(rec, plan, stages, heated, thnmax, urr_tables)


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
