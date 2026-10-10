"""Reconstruct an ENDF-6 tape end to end: read, decode, reconstruct, publish.

The entry point an application or a script calls without knowing the model:

    from kika.processing.resonances import reconstruct_endf
    rec = reconstruct_endf("n-092_U_235.endf")
    tables = rec.tables_by_mt()          # for plotting; seconds to minutes
    rec.write_pendf("U235.pendf")        # on demand; written, reloaded, verified

Reconstruction and publication are separate on purpose. The tables are what a
plot needs; the PENDF costs a write, a full reload and a second verification
(U-238: ~110 s to reconstruct, ~200 s more to publish), so it is paid only when
a file is wanted.

There is no partial result and no fallback to another processor. Every failure
is an :class:`UnsupportedResonanceError` or a
:class:`ReconstructionConvergenceError` whose ``category`` is one of
:data:`REJECTION_CATEGORIES`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .context import NeutronContext
from .grid import ReconstructionConvergenceError, ReconstructionOptions
from .prepare import UnsupportedResonanceError
from .suite import ENGINE_VERSION, reconstruct_suite

#: Point budget of :func:`reconstruct_endf`. ``ReconstructionOptions`` keeps its
#: conservative 200 000 for direct callers; a whole evaluated actinide needs
#: more (ENDF/B-VIII.1 U-238 stores 1.66 M points, ~1.9 GB peak). Reaching the
#: budget rejects the material with category ``budget-exhausted``; it never
#: relaxes the tolerance.
DEFAULT_MAX_POINTS = 4_000_000

#: Every ``category`` a rejection can carry. Stable names: a change here is an
#: API change for applications that translate them into user text.
REJECTION_CATEGORIES = (
    # The tape or its conversion to the model.
    "already-reconstructed",      # LRP=2: MF3 already holds resonance contributions
    "conversion-not-clean",       # the ENDF->model conversion lost cross-section data
    "context-missing",            # no target mass or spin declared for the resonances
    "missing-resonance-regions",  # backgrounds or LRP=1 without usable MF2 regions
    "incomplete-reactions",
    "domain-not-covered",         # modeled cross sections do not span the evaluated domain
    "internal-gap",               # a source function has a hole inside the domain
    "total-without-graph",        # MT1 present but not an explicit sum of partials
    "sum-without-graph",
    "incomplete-total-graph",
    "missing-radius-policy",
    # Physics the engine does not implement.
    "unsupported-physics",
    # Numerical limits and checks; never relaxed.
    "budget-exhausted",           # point or iteration budget
    "memory-budget-exhausted",
    "verification-failed",        # a table, sum or reload falls outside the budget
    "publication-failed",         # the PENDF could not be written cleanly
)

Progress = Callable[[str, "float | None"], None]


def native_available() -> bool:
    """Whether the compiled resonance kernel is loaded (else NumPy, slower)."""
    from . import _rm_acceleration
    return _rm_acceleration._native is not None


def context_from_endf(tape, suite):
    """Target mass ratio and spin as the evaluation declares them.

    RML: the elastic particle pair of the normalized model. Otherwise the AWRI
    of the first MF2 block that declares one, else the MF1 AWR; SPI of the
    first MF2 range. Returns ``(context or None, source)``; ``None`` when the
    material has no resonance physics and needs none.
    """
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
    range boundary): keep both points, do not deduplicate or sort them away."""
    energies: np.ndarray
    values: np.ndarray
    qm: float | None
    qi: float | None
    lr: int | None


@dataclass
class EndfReconstruction:
    """A reconstructed tape: the model, the result and what produced it."""
    path: Path
    suite: object
    result: object
    context: NeutronContext | None
    context_source: str
    timings: dict = field(default_factory=dict)
    engine_version: str = ENGINE_VERSION
    _attached: bool = field(default=False, repr=False)

    @property
    def report(self):
        return self.result.report

    def tables_by_mt(self) -> dict[int, ReconstructedTable]:
        """``{MT: ReconstructedTable}`` for every reaction with an ENDF MT."""
        from kika.nuclear_data.model import Regions1d
        from .suite import _entries
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
            q = getattr(getattr(reaction, "outputChannel", None), "Q", None)
            qi = (getattr(provenance, "headerFields", None) or {}).get("qi", getattr(q, "value", None))
            lr = getattr(provenance, "lr", None)
            out[mt] = ReconstructedTable(x, y, getattr(provenance, "qm", None), qi,
                                         None if lr is None else int(lr))
        return out

    def write_pendf(self, path, *, progress: Progress | None = None, mat=None):
        """Write the PENDF, reload it and verify it; the destination is replaced
        only if all of that passes. Raises ``ReconstructionConvergenceError``
        (``verification-failed`` or ``publication-failed``) otherwise."""
        from kika.endf.writers.assemble import writeReconstructedEndfTape
        from .suite import attach_reconstruction
        t0 = time.perf_counter()
        if progress:
            progress("write", None)
        try:
            if not self._attached:
                attach_reconstruction(self.suite, self.result)
                self._attached = True
            report = writeReconstructedEndfTape(self.suite, self.result, path, mat=mat)
        except (ReconstructionConvergenceError, UnsupportedResonanceError):
            raise
        except ValueError as error:
            raise ReconstructionConvergenceError(str(error), category="publication-failed") from error
        self.timings["write_verify"] = time.perf_counter() - t0
        if progress:
            progress("write", 1.0)
        return report


def reconstruct_endf(path, *, options: ReconstructionOptions | None = None,
                     progress: Progress | None = None) -> EndfReconstruction:
    """Read an ENDF-6 neutron tape and reconstruct every cross section at 0 K.

    ``progress(stage, fraction)`` is called with stages ``read``, ``decode``
    and ``reconstruct``; ``fraction`` is ``None`` at the start of a stage and a
    number in (0, 1] as it advances. Raises on any rejection (see module doc).
    """
    from kika.endf.read_endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite
    options = ReconstructionOptions(max_points=DEFAULT_MAX_POINTS) if options is None else options
    report = (lambda stage, fraction: progress(stage, fraction)) if progress else (lambda *_: None)
    timings = {}
    path = Path(path)

    t0 = time.perf_counter()
    report("read", None)
    tape = read_endf(str(path), mf_numbers=[1, 2, 3])
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
    result = reconstruct_suite(suite, context, options=options,
                               progress=lambda f: report("reconstruct", f))
    timings["reconstruct"] = time.perf_counter() - t0
    return EndfReconstruction(path, suite, result, context, source, timings)
