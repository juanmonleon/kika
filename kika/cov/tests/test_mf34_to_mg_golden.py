"""The sigma weighting of the MF34 multigroup collapse, frozen before it moves.

``MF34_to_MG`` weights every group by sigma(E)*phi(E), and gets sigma(E) by
calling ``endf_object.reconstruct_xs()`` — with no try/except and, until this
file, not one test. That reconstructor is documented as *"not working
correctly and should not be used"*.

So today's collapsed MF34 matrices are weighted by a sigma nobody trusts, and
nothing would notice if it changed. ``MF34_to_MG`` (aliased
``collapse_to_multigroup``) is what the thesis notebooks call — Sandwich_leg,
ASPIS88/sandwich_leg, PHYSOR26/check_mg, the chapter 3 figures — so "changed"
means chapter numbers move.

This golden is deliberately written *before* the sigma source changes, so the
next commit can say by how much rather than hoping it was small. It is a
characterization test: it asserts today's numbers, not correct ones.

**2026-10-08: the weight is now a constant sigma of 1 b, stated in the
fixture.** Until then it was the legacy in-Python reconstructor, removed with
its defects (reconstruction roadmap §4). The weight here is not meant to be the
right sigma -- production passes NJOY or ``kika.processing.resonances`` -- but a
fixed, engine-free one, so a change to the collapse cannot hide behind a change
to some reconstructor. The tape's own MF3 was tried first and rejected: below
850 keV it is Fe-56's background, ~0 b, which zeroes the lowest group and
leaves it untested. The new engine was tried too: on this 320-level slice it
exceeds its default point budget, which is a correct refusal and no fixture.
Both goldens were regenerated in that commit.

Regenerate after an intentional change with::

    REGEN_NUMERIC_GOLDENS=1 pytest kika/cov/tests/test_mf34_to_mg_golden.py

and commit the diff in the same commit as the change that moved it.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from kika.cov.multigroup import MF34_to_MG
from kika.cov.multigroup.collapse import _pendf_grid
from kika.endf import read_endf

DATA = Path(__file__).resolve().parent / "data"
REGEN = bool(os.environ.get("REGEN_NUMERIC_GOLDENS"))

#: Values match to this relative tolerance; shapes must match exactly.
#:
#: **1e-12 until 2026-08-17, and it was a tolerance this code cannot honour off
#: this machine.** Measured on a GitHub runner: the Fe-56 MT102 reconstruction
#: differs from the workstation's on 4 of 20 459 points, by 3.66e-12 relative.
#: That is libm, not arithmetic anyone wrote -- capture comes out of a
#: cancellation, so it is where the last ULP surfaces first. 1e-9 keeps roughly
#: three orders of headroom over the observed spread while staying far tighter
#: than any real change: a moved formula, grid or Q value shifts these numbers
#: by parts in 10^3, not parts in 10^9.
RTOL = 1e-9

#: Coarse enough to stay small, wide enough to span the MF34 range where the
#: Fe-56 elastic cross section actually varies.
GRID_EV = [1.0e5, 5.0e5, 1.0e6, 5.0e6, 2.0e7]


def _check_golden(name: str, produced: dict) -> None:
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
            np.testing.assert_allclose(
                have, want, rtol=RTOL, atol=0.0, err_msg=f"{name}[{key}] moved"
            )


@pytest.fixture(scope="module")
def collapsed(micro_tape):
    """MF34/MT2 collapsed onto GRID_EV, weighted by a sigma stated here.

    ``endf.pendf`` is populated explicitly with a constant 1 b over the tape's
    domain, so the weight is the flux alone. Naming it makes this a test of the
    *collapse* at a fixed sigma, instead of a test of the collapse and an
    unaccountable weight together — so a future change to either one is
    attributable. See the module docstring for why not MF3 or a reconstructor.
    """
    from kika.nuclear_data.cross_section import CrossSection

    endf = read_endf(str(micro_tape))
    endf.pendf = {2: CrossSection(energies=np.array([1.0e-5, 1.5e8]),
                                  values=np.array([1.0, 1.0]),
                                  reaction=2, nuclide_id=26056)}
    result = MF34_to_MG(endf, energy_grid=GRID_EV, mt=2)
    return endf, result


def test_it_refuses_to_guess_a_sigma(micro_tape):
    """No pendf, no collapse. The defect this file was written around.

    MF34_to_MG used to fill pendf in silently, by calling a reconstructor
    documented as producing incorrect cross sections — no try/except, no test.
    Every collapsed matrix in the thesis carried that weight and nothing said
    so. Refusing loudly is the point.
    """
    endf = read_endf(str(micro_tape))
    assert endf.pendf is None

    with pytest.raises(ValueError, match="pendf"):
        MF34_to_MG(endf, energy_grid=GRID_EV, mt=2)


def _arrays(result) -> dict:
    out = {"energy_grid": np.asarray(result.energy_grid, dtype=float)}
    for i, matrix in enumerate(result.relative_matrices):
        out[f"relative_{i}"] = np.asarray(matrix, dtype=float)
    for i, matrix in enumerate(result.absolute_matrices):
        out[f"absolute_{i}"] = np.asarray(matrix, dtype=float)
    return out


def test_collapsed_matrices_golden(collapsed):
    """The matrices the thesis notebooks consume."""
    _, result = collapsed
    _check_golden("mf34_to_mg_mt2", _arrays(result))


def test_the_sigma_weight_golden(collapsed):
    """The weight itself, separately — this is what the next commit moves.

    Pinning the collapse alone would say *that* something changed; pinning
    sigma says what, and lets the two be compared independently.

    Summarised rather than stored whole: a reconstructed grid runs to ~10^5
    points per MT, which is megabytes of fixture for a number that only has to
    be comparable. Six statistics catch any real change.

    **There used to be a sha256 over the raw float bytes next to them, and it
    was removed 2026-08-17 because it did not survive leaving this machine.**
    The first CI run in ten days -- see the lock commit -- failed here and
    nowhere near here: on a GitHub runner the Fe-56 MT102 reconstruction differs
    from this workstation's on 4 of 20 459 points, by 1.3e-15 absolute and
    3.7e-12 relative. A last-ULP libm difference, which the statistics below
    absorb and a hash cannot. MT102 is where it shows because capture comes out
    of a cancellation, and ``sorted()`` put ``mt102_digest`` first of all keys,
    so the digest aborted the test before any statistic was ever compared.

    So the digest was not catching "a change the statistics would miss" -- it
    was asserting bit-exactness of floating-point arithmetic across machines,
    which is not a property this code has or should claim. Regenerating it on a
    runner would only have moved which machine is wrong.
    """
    endf, _ = collapsed
    assert endf.pendf, "nothing populated pendf, so no sigma was weighted with"

    arrays = {}
    for mt in sorted(endf.pendf):
        section = endf.pendf[mt]
        energies, values = _pendf_grid(section)
        arrays[f"mt{mt}_summary"] = np.array([
            energies.size,
            energies[0], energies[-1],
            values.min(), values.max(), values.sum(),
        ], dtype=float)
    _check_golden("mf34_to_mg_sigma", arrays)


def test_the_collapse_is_symmetric(collapsed):
    """Structural, so it survives the sigma source changing under it."""
    _, result = collapsed
    for i, matrix in enumerate(result.relative_matrices):
        matrix = np.asarray(matrix, dtype=float)
        np.testing.assert_allclose(
            matrix, matrix.T, rtol=1e-10, atol=0.0,
            err_msg=f"relative matrix {i} is not symmetric",
        )


def test_the_grid_is_what_was_asked_for(collapsed):
    _, result = collapsed
    np.testing.assert_allclose(
        np.asarray(result.energy_grid, dtype=float), np.asarray(GRID_EV, dtype=float)
    )
