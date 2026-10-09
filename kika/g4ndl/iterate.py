r"""The correction loop with Geant4 in it: measurement → corrected evaluation → G4NDL → …

Roadmap G4NDL §0bis, Fase 14.  An experiment corrected with a transport code
(multiple scattering, missed neutrons) depends on the library the code read, so
the measurement and the library are found together, by iteration:

1. the measurement, corrected with the previous library (or the raw one);
2. the **evaluation** corrected by it through smooth ratios
   (:mod:`kika.nuclear_data.ratio_correction`) -- always the evaluation, never
   the previous iteration's suite, so corrections do not compound;
3. a whole G4NDL library with that isotope's elastic replaced
   (:func:`kika.g4ndl.patch_elastic`), read back and compared with the suite;
4. **the hole**: whoever owns the experiment runs Geant4 on that library and
   hands back the measurement corrected with it (``transport``).

Each iteration gets its own directory, ``iter_NN/``, which must not exist:
nothing is ever overwritten.  It holds the library and ``iteration.json``, the
manifest of what went in (evaluation, measurement and its digest, setup,
assumptions the caller declares) and what came out (r(E), δa_l(E), widths,
χ², the files written with their SHA-256, the read-back check, the change from
the previous iteration).

**Convergence** (proposed in the roadmap): the corrected σ, read through the
experiment on the measurement's own bins, moves less than ``threshold`` of the
statistical uncertainty in every bin between two iterations.

Nothing here runs Geant4: kika does not depend on it.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["LoopMeasurement", "IterationRecord", "iterate_once", "run_loop", "sigma_change"]

ITERATION_MANIFEST = "iteration.json"


@dataclass
class LoopMeasurement:
    """One version of the measurement the loop corrects with.

    ``energies_ev``/``sigma``/``sigma_err`` [b]/``bins_ev``: the angle-integrated
    σ.  The angular part is optional: ``dcs`` and ``dcs_err`` are
    ``(n_E, n_mu)`` [b/sr] at ``dcs_energies_ev`` (``dcs_bins_ev``) and
    ``mu_lab``.  ``version`` says what produced it (``"raw"``, or the library
    that corrected it); it goes in the manifest with :meth:`digest`.
    """

    energies_ev: np.ndarray
    sigma: np.ndarray
    sigma_err: np.ndarray
    bins_ev: Optional[Sequence[Tuple[float, float]]] = None
    version: str = "raw"
    mu_lab: Optional[np.ndarray] = None
    dcs_energies_ev: Optional[np.ndarray] = None
    dcs: Optional[np.ndarray] = None
    dcs_err: Optional[np.ndarray] = None
    dcs_bins_ev: Optional[Sequence[Tuple[float, float]]] = None

    @property
    def has_angular(self) -> bool:
        return self.dcs is not None

    def digest(self) -> str:
        """SHA-256 of every number in it, in a fixed order."""
        h = hashlib.sha256()
        for part in (self.energies_ev, self.sigma, self.sigma_err, self.bins_ev, self.mu_lab,
                     self.dcs_energies_ev, self.dcs, self.dcs_err, self.dcs_bins_ev):
            h.update(b"|" if part is None else np.asarray(part, float).tobytes())
        return h.hexdigest()


@dataclass
class IterationRecord:
    """What one iteration did.  ``manifest`` is what ``iteration.json`` holds."""

    index: int
    directory: Path
    library: Path
    target: str
    suite: Any
    sigma_report: Any
    angular_report: Any
    folded_sigma: np.ndarray
    change: Optional[float]
    converged: bool
    manifest: Dict[str, Any] = field(default_factory=dict)


def sigma_change(previous: IterationRecord, current: IterationRecord,
                 measurement: LoopMeasurement) -> float:
    """``max_i |⟨σ⟩_k − ⟨σ⟩_{k−1}| / δσ_i`` on the measurement's bins."""
    return float(np.max(np.abs(current.folded_sigma - previous.folded_sigma)
                        / np.asarray(measurement.sigma_err, float)))


def _setup_dict(setup) -> Dict[str, Any]:
    tof = setup.tof
    return {"flight_path_m": getattr(tof, "flight_path_m", None),
            "delta_t_ns": getattr(tof, "delta_t_ns", None),
            "delta_t_is_fwhm": getattr(tof, "delta_t_is_fwhm", None),
            "temperature_k": setup.temperature_k,
            "angular_half_width_deg": setup.angular_half_width_deg}


def _table(fn, energies) -> List[List[float]]:
    e = np.asarray(energies, float)
    v = np.atleast_2d(np.asarray(fn(e), float).T).T
    return [[float(x)] + [float(y) for y in row] for x, row in zip(e, v)]


def _read_back(library: Path, target: str, suite, energies, mu_native) -> Dict[str, float]:
    """The library as Geant4's reader sees it, against the suite it was written from."""
    from kika.g4ndl.library import open as open_library
    from kika.nuclear_data.forward import ElasticView

    back = open_library(library).read(target, processes=["elastic"])
    vs, vb = ElasticView.from_suite(suite), ElasticView.from_suite(back)
    grid = vs.xs_energies
    sig = float(np.max(np.abs(vb.sigma(grid) / vs.sigma(grid) - 1.0)))
    e = np.asarray(energies, float)
    pdf = max(float(np.max(np.abs(vb.pdf_native(m)(e) - vs.pdf_native(m)(e))))
              for m in mu_native)
    return {"sigma_max_rel": sig, "pdf_max_abs": pdf}


def iterate_once(evaluation, measurement: LoopMeasurement, setup, base_library, workdir,
                 index: int, *, previous: Optional[IterationRecord] = None,
                 angular: bool = True, sigma_options: Optional[Dict[str, Any]] = None,
                 angular_options: Optional[Dict[str, Any]] = None,
                 evaluation_label: str = "", notes: Optional[Dict[str, Any]] = None,
                 share: str = "hardlink", threshold: float = 0.1,
                 read_back_tolerance: float = 1e-6) -> IterationRecord:
    """One turn: correct ``evaluation`` with ``measurement`` and write ``iter_NN``.

    ``sigma_options``/``angular_options`` go to
    :func:`~kika.nuclear_data.ratio_correction.correct_cross_section` and
    :func:`~kika.nuclear_data.ratio_correction.correct_angular`; the angular
    step runs when ``angular`` and the measurement has DCS.  ``notes`` (the
    caller's assumptions, say) is copied into the manifest verbatim.  A
    read-back that differs from the suite by more than ``read_back_tolerance``
    raises: a library that does not say what the suite says must not reach
    Geant4.
    """
    from kika.g4ndl.patch import patch_elastic
    from kika.nuclear_data import ratio_correction as rc
    from kika.nuclear_data.forward import ElasticView, forward_sigma

    directory = Path(workdir) / f"iter_{index:02d}"
    if directory.exists():
        raise FileExistsError(f"{directory} exists: an iteration is never overwritten")
    m = measurement
    corrected, srep = rc.correct_cross_section(
        evaluation, m.energies_ev, m.sigma, m.sigma_err, setup, bins_ev=m.bins_ev,
        **(sigma_options or {}))
    arep = None
    if angular and m.has_angular:
        corrected, arep = rc.correct_angular(
            corrected, m.dcs_energies_ev, m.mu_lab, m.dcs, m.dcs_err, setup,
            bins_ev=m.dcs_bins_ev, **(angular_options or {}))
    view = ElasticView.from_suite(corrected)
    folded = forward_sigma(view, m.energies_ev, setup, m.bins_ev)

    directory.mkdir(parents=True)
    library = directory / "library"
    patched = patch_elastic(base_library, corrected, library, share=share)
    mu_check = np.linspace(-1.0, 1.0, 21)
    back = _read_back(library, patched.target, corrected, m.energies_ev, mu_check)
    if back["sigma_max_rel"] > read_back_tolerance or back["pdf_max_abs"] > read_back_tolerance:
        raise ValueError(f"the library written in {library} reads back differently from the "
                         f"suite: {back}")

    record = IterationRecord(index, directory, library, patched.target, corrected, srep, arep,
                             folded, None, False)
    if previous is not None:
        record.change = sigma_change(previous, record, m)
        record.converged = record.change < threshold

    manifest = json.loads(patched.manifest.read_text(encoding="utf-8")) if patched.manifest else {}
    anchors = srep.iterations[0].anchors() if srep.iterations else np.asarray(m.energies_ev)
    record.manifest = {
        "iteration": index,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "evaluation": evaluation_label,
        "base_library": str(base_library),
        "target": patched.target,
        "measurement": {"version": m.version, "digest": m.digest(),
                        "n_sigma": int(np.size(m.energies_ev)),
                        "n_dcs": None if m.dcs is None else list(np.shape(m.dcs))},
        "setup": _setup_dict(setup),
        "notes": notes or {},
        "sigma": {"label": srep.label, "factor": srep.first.factor, "chi2N": srep.chi2,
                  "max_pull": srep.max_pull, "converged": srep.converged, "grid": srep.grid,
                  "r_of_E": _table(srep.ratio, anchors)},
        "angular": None if arep is None else {
            "level": arep.level, "degree": arep.degree, "emin_eV": arep.emin,
            "excluded": arep.excluded, "chi2N": arep.chi2, "max_pull": arep.max_pull,
            "converged": arep.converged, "records_touched": arep.records_touched,
            "factor": arep.iterations[0].factor if arep.iterations else None,
            "delta_a_of_E": (_table(arep.delta, arep.iterations[0].anchors())
                             if arep.iterations else [])},
        "files": {"replaced": patched.replaced, "removed": patched.removed,
                  "patch_manifest": manifest},
        "read_back": back,
        "change_over_stat": record.change,
        "threshold": threshold,
        "converged": record.converged,
    }
    (directory / ITERATION_MANIFEST).write_text(
        json.dumps(record.manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def run_loop(evaluation, measurement: LoopMeasurement, setup, base_library, workdir,
             transport: Callable[[Path, IterationRecord], LoopMeasurement], *,
             max_iterations: int = 5, start: int = 0, **kwargs) -> List[IterationRecord]:
    """Iterate until the corrected σ stops moving, or ``max_iterations``.

    ``transport(library, record)`` is the hole: it runs Geant4 on ``library``
    and returns the measurement corrected with it.  It is not called after the
    last iteration.  ``start`` numbers the first directory, to resume a loop
    in the same ``workdir``.  ``kwargs`` go to :func:`iterate_once`.
    """
    records: List[IterationRecord] = []
    m = measurement
    for k in range(start, start + max_iterations):
        rec = iterate_once(evaluation, m, setup, base_library, workdir, k,
                           previous=records[-1] if records else None, **kwargs)
        records.append(rec)
        if rec.converged or k == start + max_iterations - 1:
            break
        m = transport(rec.library, rec)
    return records
