"""Reconstruct an ENDF-6 tape end to end: read, decode, reconstruct, publish.

The entry point an application or a script calls without knowing the model:

    from kika.endf import reconstruct_endf
    rec = reconstruct_endf("n-092_U_235.endf")
    tables = rec.tables_by_mt()          # for plotting; seconds to minutes
    rec.write_pendf("U235.pendf")        # on demand; written, reloaded, verified
    rec.report_document().save("U235_report.html")   # or .txt / .json

Reconstruction and publication are separate on purpose. The tables are what a
plot needs; the PENDF costs a write, a full reload and a second verification
(U-238: ~110 s to reconstruct, ~200 s more to publish), so it is paid only when
a file is wanted.

There is no partial result and no fallback to another processor. Every failure
is a ``kika.processing.UnsupportedResonanceError`` or
``ReconstructionConvergenceError`` whose ``category`` is one of
``kika.processing.REJECTION_CATEGORIES``.

This is format code: it reads ENDF and decodes it to the model, then calls the
format-free engine in ``kika.processing.resonances``. The engine is imported
inside the functions, so ``import kika.endf`` neither wakes the model nor
initialises ``kika.processing`` (which itself imports ``kika.endf``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from kika.processing.resonances import DEFAULT_MAX_POINTS

Progress = Callable[[str, "float | None"], None]


def context_from_endf(tape, suite):
    """Target mass ratio and spin as the evaluation declares them.

    RML: the elastic particle pair of the normalized model. Otherwise the AWRI
    of the first MF2 block that declares one, else the MF1 AWR; SPI of the
    first MF2 range.

    Parameters
    ----------
    tape
        The ENDF tape as read (MF1 and MF2 at least).
    suite
        The reaction suite decoded from it; its ``resonances`` say whether there
        is resonance physics.

    Returns
    -------
    tuple
        ``(context, source)``: a ``NeutronContext`` and a string naming where it
        came from (``"RML elastic pair"``, ``"MF2 AWRI (LRU=1)"`` or ``LRU=2``,
        ``"MF1 AWR"``). ``(None, "no resonance physics")`` when the material has
        no resonance parameters and needs none.

    Raises
    ------
    ~kika.processing.resonances.UnsupportedResonanceError
        With ``category`` ``context-missing`` when MF2 declares no target spin,
        or when neither MF2 nor MF1 gives a mass ratio.
    """
    from kika.processing.resonances import NeutronContext, UnsupportedResonanceError
    resonances = suite.resonances
    if resonances is None or (not resonances.resolved and resonances.unresolved is None):
        return None, "no resonance physics"
    try:
        isotope = tape.mf[2].mt[151].isotopes[0]
        spin = isotope.energy_ranges[0].parameters.spi
    except (KeyError, IndexError, AttributeError) as error:
        raise UnsupportedResonanceError("MF2 declares no target spin", category="context-missing") from error
    for region in getattr(resonances, "resolved", None) or []:
        for pair in getattr(region.formalism, "resonanceReactions", None) or []:
            if getattr(pair, "reactionMT", None) == 2 and getattr(pair, "kinematics", None) is not None:
                b = pair.kinematics.particleB
                return NeutronContext(b.massRatio, b.spin), "RML elastic pair"
    for rng in isotope.energy_ranges:
        for block in getattr(rng.parameters, "l_values", None) or []:
            if getattr(block, "awri", None):
                return NeutronContext(block.awri, spin), f"MF2 AWRI (LRU={rng.lru})"
    awr = getattr(tape.mf[1].mt[451], "_awr", None) if 1 in tape.mf else None
    if not awr:
        raise UnsupportedResonanceError("no AWRI in MF2 and no AWR in MF1", category="context-missing")
    return NeutronContext(awr, spin), "MF1 AWR"


@dataclass(frozen=True)
class ReconstructedTable:
    """One reaction as a lin-lin table. A repeated energy is a real jump (a
    range boundary): keep both points, do not deduplicate or sort them away.

    Returned by :meth:`EndfReconstruction.tables_by_mt`, one per MT.

    Attributes
    ----------
    energies : numpy.ndarray
        Energies in eV, increasing; a jump repeats its energy.
    values : numpy.ndarray
        Cross sections in b at *energies*, interpolated linearly.
    qm : float or None
        The ``QM`` of the MF3 section header in eV, ``None`` when the section
        states none.
    qi : float or None
        The ``QI`` of the MF3 section header in eV; when the header gives none,
        the reaction's Q value as the model derives it, which may be ``None``.
    lr : int or None
        The ``LR`` flag (complex or "breakup" reaction) of the MF3 section,
        ``None`` when it is not stated.
    """
    energies: np.ndarray
    values: np.ndarray
    qm: float | None
    qi: float | None
    lr: int | None


@dataclass
class EndfReconstruction:
    """A reconstructed tape: the model, the result and what produced it.

    Returned by :func:`reconstruct_endf`; not built by hand. Reconstruction and
    publication are separate: :meth:`tables_by_mt` gives the tables, and
    :meth:`write_pendf` writes and verifies the PENDF.

    Attributes
    ----------
    path : pathlib.Path
        The tape that was read.
    suite
        The reaction suite decoded from the tape (MF1 to MF3).
    result
        The ``SuiteReconstructionResult``: the tabulated cross sections of every
        reaction (energies in eV, cross sections in b, linear-linear) and the
        report of the run.
    context
        The ``NeutronContext`` (target mass ratio and spin) the reconstruction
        used, or ``None`` when the material has no resonance physics.
    context_source : str
        Where *context* came from, as ``context_from_endf`` returns it.
    timings : dict
        Seconds per step: ``read``, ``decode``, ``reconstruct``, and
        ``write_verify`` once :meth:`write_pendf` has run.
    engine_version : str
        Version of the reconstruction engine.
    evaluation : dict
        MF1/451 identification and the MF2 ranges as the tape states them.
    native_kernel : bool or None
        Whether the compiled kernel was loaded for this run.
    publication : dict or None
        ``None`` until :meth:`write_pendf` runs; then ``path``, ``mat`` and the
        reload verification (``maxima`` and ``worst``).
    workers : int or None
        The ``workers`` the reconstruction was asked for; the default of
        :meth:`write_pendf`.
    """
    path: Path
    suite: object
    result: object
    context: object | None
    context_source: str
    timings: dict = field(default_factory=dict)
    engine_version: str = ""
    #: MF1/451 identification and the MF2 ranges as the tape states them.
    evaluation: dict = field(default_factory=dict, repr=False)
    #: Whether the compiled kernel was loaded for this run.
    native_kernel: bool | None = None
    #: Set by :meth:`write_pendf`: path, MAT and the reload verification.
    publication: dict | None = None
    #: The ``workers`` the reconstruction was asked for; :meth:`write_pendf`'s default.
    workers: int | None = None
    _attached: bool = field(default=False, repr=False)

    @property
    def report(self):
        """The statistics of the run: the read-only mapping ``result.report``.

        Among its entries are ``domain`` (lowest and highest energy, eV),
        ``points`` and ``stored_points``.
        """
        return self.result.report

    def report_document(self):
        """The full report as a ``ReconstructionReport``.

        It offers ``to_dict()``, ``to_text()``, ``to_html()`` and ``save(path)``.
        Works before and after :meth:`write_pendf`; after it, the verification
        includes the reload.
        """
        from .reconstruction_report import ReconstructionReport, build
        return ReconstructionReport(build(self))

    def tables_by_mt(self) -> dict[int, ReconstructedTable]:
        """``{MT: ReconstructedTable}`` for every reaction with an ENDF MT."""
        from kika.nuclear_data.model import Regions1d
        from kika.processing.resonances.suite import _entries
        entries = _entries(self.suite)
        out = {}
        for mt, key in self.result._mt_keys.items():
            form = self.result.forms[key]
            curves = form.function1ds if isinstance(form, Regions1d) else [form]
            x = np.concatenate([np.asarray(c.xs, float) for c in curves])
            y = np.concatenate([np.asarray(c.ys, float) for c in curves])
            # Adjacent regions share their junction node; keep it twice only
            # where the value actually jumps. Regions evaluated separately can
            # differ in the last bits at a continuous junction (~1e-16), far
            # below any real jump (Au-197 RRR/URR: x27).
            joint = np.flatnonzero((np.diff(x) == 0) &
                                   (np.abs(np.diff(y)) <= 1e-9 * np.maximum(np.abs(y[:-1]), np.abs(y[1:]))))
            x, y = np.delete(x, joint), np.delete(y, joint)
            reaction = entries[key]
            provenance = getattr(reaction, "provenance", None)
            from kika.nuclear_data.model.endf_conversion import reactionQI
            qi = (getattr(provenance, "headerFields", None) or {}).get("qi")
            if qi is None:
                qi = reactionQI(reaction, self.suite)
            lr = getattr(provenance, "lr", None)
            out[mt] = ReconstructedTable(x, y, getattr(provenance, "qm", None), qi,
                                         None if lr is None else int(lr))
        return out

    def write_pendf(self, path, *, progress: Progress | None = None, mat=None, workers: int | None = None):
        """Write the PENDF, reload it and verify it; the destination is replaced
        only if all of that passes.

        The first call attaches the reconstruction to the suite, after verifying
        it; later calls reuse that. The folder of *path* is created if missing.
        Afterwards ``publication`` holds the path, the MAT and the reload
        verification.

        Parameters
        ----------
        path
            The PENDF to write.
        progress
            ``progress("write", fraction)``: ``None`` at the start, then a float
            in (0, 1].
        mat
            MAT of the written material; ``None`` takes the evaluation's own.
        workers
            Workers for the verification physics, which runs in parallel only
            when its work (energies times levels) reaches 1e9, on processes or
            threads by solver and size. ``None`` takes the reconstruction's
            own (:func:`reconstruct_endf`); 1 keeps it in this thread. The
            result is the same either way.

        Returns
        -------
        ConversionReport
            The report of the conversion to ENDF-6, clean when the write
            succeeds.

        Raises
        ------
        ~kika.processing.resonances.ReconstructionConvergenceError
            With ``category`` ``verification-failed`` when a table, sum or the
            reload falls outside the budget, or ``publication-failed`` when the
            file cannot be written or reloaded cleanly.
        ~kika.processing.resonances.UnsupportedResonanceError
            If the engine refuses the suite when it is attached.
        """
        from kika.endf.writers.assemble import PUBLICATION_VERIFY_SHARE, writeReconstructedEndfTape
        from kika.processing.resonances import (ReconstructionConvergenceError,
                                                UnsupportedResonanceError, attach_reconstruction)
        t0 = time.perf_counter()
        workers = self.workers if workers is None else workers
        if progress:
            progress("write", None)
        try:
            if not self._attached:
                attach_reconstruction(self.suite, self.result, workers=workers, progress=(
                    lambda f: progress("write", PUBLICATION_VERIFY_SHARE * f)) if progress else None)
                self._attached = True
            report = writeReconstructedEndfTape(self.suite, self.result, path, mat=mat, workers=workers,
                                                progress=(lambda f: progress("write", f)) if progress else None)
        except (ReconstructionConvergenceError, UnsupportedResonanceError):
            raise
        except ValueError as error:
            raise ReconstructionConvergenceError(str(error), category="publication-failed") from error
        self.timings["write_verify"] = time.perf_counter() - t0
        maxima, worst = self.result.last_verification or ({}, {})
        self.publication = dict(path=str(Path(path)), mat=mat if mat is not None else
                                (self.evaluation or {}).get("mat"), maxima=maxima, worst=worst)
        return report


def reconstruct_endf(path, *, options=None, progress: Progress | None = None,
                     workers: int | None = None) -> EndfReconstruction:
    """Read an ENDF-6 neutron tape and reconstruct every cross section at 0 K.

    ``progress(stage, fraction)`` is called with stages ``read``, ``decode``
    and ``reconstruct``; ``fraction`` is ``None`` at the start of a stage and a
    number in (0, 1] as it advances. There is no partial result: any rejection
    raises.

    Parameters
    ----------
    path
        The ENDF-6 tape; only MF1 to MF3 are read.
    options
        A ``ReconstructionOptions``. ``None`` takes its defaults (relative
        tolerance 1e-3, absolute 1e-8 b) with a budget of 4 000 000 stored
        points.
    progress
        ``progress(stage, fraction)``, as above.
    workers
        Threads that share the level sums of large batches, and workers for the
        verification physics, here and in :meth:`EndfReconstruction.write_pendf`.
        The verification runs in parallel only when its work (energies times
        levels) reaches 1e9, on processes or threads by solver and size.
        ``None`` takes ``KIKA_WORKERS`` or ``min(4, cores // 2)`` at
        below-normal priority (``kika.processing.workers``). The tables do not
        depend on it.

    Returns
    -------
    EndfReconstruction
        The 0 K reconstruction of every cross section, in memory; nothing is
        written.

    Raises
    ------
    ~kika.processing.resonances.UnsupportedResonanceError
        If the evaluation uses data the reconstruction does not support.
    ~kika.processing.resonances.ReconstructionConvergenceError
        If a budget runs out or a check fails. Both errors carry a ``category``
        in ``kika.processing.REJECTION_CATEGORIES``.
    """
    from kika.endf.read_endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.processing.resonances import (ENGINE_VERSION, ReconstructionOptions, native_available,
                                            reconstruct_suite)
    from .reconstruction_report import evaluation_summary
    options = ReconstructionOptions(max_points=DEFAULT_MAX_POINTS) if options is None else options
    report = (lambda stage, fraction: progress(stage, fraction)) if progress else (lambda *_: None)
    timings = {}
    path = Path(path)

    t0 = time.perf_counter()
    report("read", None)
    tape = read_endf(str(path), mf_numbers=[1, 2, 3])
    evaluation = evaluation_summary(tape, path)
    timings["read"] = time.perf_counter() - t0
    report("read", 1.0)

    t0 = time.perf_counter()
    report("decode", None)
    suite, conversion = decodeReactionSuite(tape)
    suite.report = conversion
    context, source = context_from_endf(tape, suite)
    timings["decode"] = time.perf_counter() - t0
    report("decode", 1.0)

    t0 = time.perf_counter()
    report("reconstruct", None)
    result = reconstruct_suite(suite, context, options=options, workers=workers,
                               progress=lambda f: report("reconstruct", f))
    timings["reconstruct"] = time.perf_counter() - t0
    return EndfReconstruction(path, suite, result, context, source, timings, ENGINE_VERSION,
                              evaluation=evaluation, native_kernel=native_available(),
                              workers=workers)
