"""Tier-2 goldens: the numerics, frozen.

The tier-1 golden proves kika gives back the bytes it was given. This one
proves it gives back the same *numbers* for the multigroup collapse.

Each case writes one ``.npz`` under ``data/``. Regenerate after an intentional
change with::

    REGEN_NUMERIC_GOLDENS=1 pytest kika/processing/tests/test_numeric_goldens.py

and commit the diff **in the same commit as the change that caused it**, with
one line saying why it moved. A golden updated in a commit of its own is a
golden nobody reviewed.

**Tolerance.** Array *shapes* are compared exactly. Values are compared at
``RTOL`` plus an absolute floor tied to each array's own scale (``GOLDEN_ULP``):
CI had not run since 2026-08-07 when the first cross-machine run showed that
``rtol=1e-12`` and stored digests were promises this arithmetic cannot keep.

*The resonance-reconstruction goldens that used to live here are gone with the
legacy reconstructor (``kika.processing.reconstruct``, removed 2026-10-08).
They froze regression, not physics -- its kernels carried the defects listed in
the reconstruction roadmap §4 -- so they were retired rather than regenerated
against the new engine, whose physics gates are independent references
(``kika.processing.resonances`` and its tests). See
``kika-workspace/docs/library/resonance_migration_r9.md`` §2.*
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kika.processing import collapse_covariance, compute_rebin_operator

DATA = Path(__file__).resolve().parent / "data"
REGEN = bool(os.environ.get("REGEN_NUMERIC_GOLDENS"))

#: Values match to this relative tolerance; shapes must match exactly.
#:
#: **1e-12 until 2026-08-17, and it was a tolerance this code cannot honour off
#: this machine.** Measured on a GitHub runner: the Fe-56 MT102 reconstruction
#: differs from the workstation's on 4 of 20 459 points, by 1.33e-15 absolute
#: and 3.66e-12 relative. That is libm, not arithmetic anyone wrote -- capture
#: comes out of a cancellation, so it is where the last ULP surfaces first.
#: 1e-9 keeps roughly three orders of headroom over the observed spread while
#: staying far tighter than any real change: a moved formula, grid or Q value
#: shifts these numbers by parts in 10^3, not parts in 10^9.
#:
#: The digests that used to sit beside these arrays are gone for the same
#: reason -- see the module docstring.
RTOL = 1e-9

#: Absolute slack, in units of the last bit of the array's own largest value.
#:
#: **``rtol`` alone is the wrong measure for these arrays, and loosening it
#: further would have been the wrong fix.** Cross-machine arithmetic noise is
#: absolute at the scale the computation runs at -- a sum over resonances, or a
#: 3x3 collision-matrix inverse, carries the same last-bit uncertainty whether
#: the answer it produces is 300 barns or 1e-7 barns. Dividing that fixed noise
#: by a value nine orders below the array's peak turns it into a large relative
#: number, which is why ``rtol=1e-9`` failed on exactly two points of
#: ``reconstruct_rm_fission[mt102]`` -- 8.83e-07 and 9.34e-08 in an array that
#: peaks at 309 barns, the deep minima between resonances where capture is the
#: difference of two nearly equal terms.
#:
#: Measured over every reconstruction golden on the home workstation against
#: goldens generated elsewhere, as a distance in ULP of each array's maximum:
#:
#:     reconstruct_slbw          <= 0.02 ULP
#:     reconstruct_mlbw          <= 0.07 ULP
#:     reconstruct_rm            <= 8.95 ULP  (mt102)
#:     reconstruct_rm_fission    <= 45.0 ULP  (mt18)
#:
#: 1024 leaves ~23x over the worst observed and still says something very
#: strong: nothing moved by more than one part in 10^13 of the array's peak.
#: A real change -- a moved formula, grid or Q value -- shifts these numbers by
#: parts in 10^3.
#:
#: **What this costs, stated rather than left to be found.** Measured on
#: ``reconstruct_rm_fission[mt102]``: at the peak the floor contributes nothing
#: and ``rtol`` still governs, so a change of 1e-9 relative is caught exactly as
#: before. At the deep minimum that used to fail -- 8.83e-07, nine orders under
#: the peak -- the floor dominates and the gate now catches 1e-4 relative but
#: not 1e-5. That is the trade, and it is the right way round: those points
#: carry no physics, the arithmetic that produces them cannot promise more, and
#: 1e-4 is still two orders tighter than any change worth calling a change.
GOLDEN_ULP = 1024


# ---------------------------------------------------------------------------
# Golden I/O
# ---------------------------------------------------------------------------

def check_golden(name: str, produced: dict[str, np.ndarray]) -> None:
    """Compare *produced* against the committed golden, or rewrite it."""
    path = DATA / f"{name}.npz"

    if REGEN:
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **produced)
        return

    if not path.is_file():
        pytest.fail(
            f"golden {path.name} is missing — generate it with "
            f"REGEN_NUMERIC_GOLDENS=1 and commit it"
        )

    with np.load(path) as golden:
        assert sorted(golden.files) == sorted(produced), (
            f"{name}: array set changed: {sorted(golden.files)} -> {sorted(produced)}"
        )
        for key in sorted(produced):
            want, have = golden[key], np.asarray(produced[key])
            assert have.shape == want.shape, (
                f"{name}[{key}]: shape {want.shape} -> {have.shape}"
            )
            if want.dtype.kind in "US":  # digests compare exactly
                assert have == want, f"{name}[{key}] changed: {want} -> {have}"
                continue
            # atol from the array's own scale, not zero: see GOLDEN_ULP.
            scale = float(np.abs(want).max()) if want.size else 0.0
            np.testing.assert_allclose(
                have, want, rtol=RTOL,
                atol=GOLDEN_ULP * float(np.spacing(scale)) if scale else 0.0,
                err_msg=f"{name}[{key}] moved",
            )


# ---------------------------------------------------------------------------
# Multigroup collapse
# ---------------------------------------------------------------------------

def test_multigroup_collapse_golden():
    """Rebin operator and the congruence transform it drives.

    ``collapse_covariance`` is one line (``M @ C @ M.T``); the content is in
    ``compute_rebin_operator``, which integrates the weighting spectrum over
    every coarse/fine bin overlap. Both are frozen together because it is
    their composition that the pipeline uses.
    """
    coarse = np.array([1.0e-5, 1.0e2, 1.0e4, 1.0e6, 2.0e7])
    fine = np.array([1.0e-5, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 5.0e6, 2.0e7])

    n = len(coarse) - 1
    base = np.arange(1, n + 1, dtype=float)
    cov = 0.01 * np.outer(base, base) + np.diag(0.05 * base)
    cov = 0.5 * (cov + cov.T)

    operator = compute_rebin_operator(coarse, fine)
    collapsed = collapse_covariance(cov, operator)

    # Structural invariants, checked as well as frozen: the operator is
    # row-stochastic and the transform preserves symmetry.
    np.testing.assert_allclose(operator.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(collapsed, collapsed.T, rtol=1e-12)

    check_golden(
        "multigroup_collapse",
        {"operator": operator, "collapsed": collapsed, "coarse_cov": cov},
    )
