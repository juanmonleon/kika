"""An absolute MF31/MF33 section is drawn, made relative before the draw.

The appliers multiply, so ``perturbFromModel`` draws relative covariances. A
section the file states in absolute units (LB=0/8/9: barns², or ν̄²) is
restated as ``C_abs[i, j] / (σ̄_i σ̄_j)``, with σ̄ the central value averaged
over each bin of the grid the block is assembled on, 1/E weighted.

No committed micro-tape states an absolute MF31/MF33 section, so these tests
state one: the decoded section is made absolute with the matrix
``C_rel ⊙ σ̄σ̄ᵀ`` on its own grid. On a request whose assembly grid is that same
grid, the conversion has to give back ``C_rel``, and the run has to draw what
it draws from the relative original. That is a check against an independent
expectation, not against the function under test.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import kika.sampling.model_perturbation as model_perturbation
from kika._covariance_forms import require_single_matrix
from kika.endf.model_adapter import decodeCovarianceSuite
from kika.endf.read_endf import read_endf
from kika.sampling.mf31_sampling import _nubar_central_values
from kika.sampling.mf33_sampling import (binAveragedCentral, load_mf33_covariance,
                                         relativiseAbsoluteSections,
                                         relativiseOnAssemblyGrids)
from kika.sampling.model_blocks import _endf_mt, _is_endf_mf, _lift_matrix
from kika.sampling.model_perturbation import perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
MF33_TAPE = str(DATA / "micro_fe56_mf33.endf")
NUBAR_TAPE = str(DATA / "micro_u235_nubar.endf")


def _restated(section, central, union, *, absolute):
    """*section* lifted onto *union*, stated absolute or as the expected relative.

    ``absolute=True``: ``lift(C_rel) ⊙ σ̄σ̄ᵀ``, σ̄ binned on *union* -- what a file
    stating the same uncertainty in barns² (or ν̄²) on those bins would hold.
    ``absolute=False``: ``lift(C_rel)`` with the rows and columns whose σ̄ is
    below the inert floor (``max(1e-3 σ̄_max, 1e-9)``) set to zero -- what the
    conversion has to give back, since an absolute covariance where σ̄ is zero
    states nothing a factor can carry.
    """
    form = require_single_matrix(section.form, "")
    grid = np.asarray(form.rowGrid, dtype=float)
    union = np.asarray(union, dtype=float)
    lift = _lift_matrix(grid, union)
    lifted = lift @ np.asarray(form.matrix, dtype=float) @ lift.T
    sigma = binAveragedCentral(central, union)
    if absolute:
        matrix = lifted * np.outer(sigma, sigma)
    else:
        live = sigma > max(1e-3 * sigma.max(), 1e-9)
        matrix = lifted * np.outer(live, live)
    return dataclasses.replace(section, form=dataclasses.replace(
        form, matrix=matrix, isRelative=not absolute,
        rowGrid=union.copy(), columnGrid=union.copy()))


def _patch_source(monkeypatch, mf, mts, central_of, *, union=None, absolute=True,
                  drop_central=()):
    """Restate the self sections of *mts* in what ``_readSource`` returns."""
    original = model_perturbation._readSource

    def patched(*args, **kwargs):
        suite, covariances, endfObj, path, fmt, reports = original(*args, **kwargs)
        sections = []
        for section in covariances.covarianceSections:
            mt = _endf_mt(section.rowData)
            if (_is_endf_mf(section.rowData, mf) and mt in mts
                    and section.columnData is None):
                grid = (union if union is not None else
                        require_single_matrix(section.form, "").rowGrid)
                section = _restated(section, central_of(endfObj, mt), grid,
                                    absolute=absolute)
            sections.append(section)
        covariances = dataclasses.replace(covariances, covarianceSections=sections)
        for mt in drop_central:
            endfObj.get_file(3).sections.pop(mt, None)
        return suite, covariances, endfObj, path, fmt, reports

    monkeypatch.setattr(model_perturbation, "_readSource", patched)


def _assembly_grid(run):
    (meta,) = run.index.values()
    (grid,) = {tuple(g) for g in meta["grids"].values()}
    return np.asarray(grid, dtype=float)


def _mf3_central(endfObj, mt):
    return endfObj.get_file(3).sections[mt]


def _nubar_central(endfObj, mt):
    return _nubar_central_values(endfObj.get_file(1))[mt]


def _factors(run):
    components = run.samples[0]["set"].components()
    return {c: np.array([s["set"].factors[c] for s in run.samples])
            for c in components}


def test_the_conversion_is_the_carriers_on_its_grid():
    """On one union grid, the model route's conversion and the carrier's are the
    same matrix, bit for bit -- including where σ̄ varies inside a source bin."""
    endf = read_endf(MF33_TAPE)
    suite, _ = decodeCovarianceSuite(endf)
    made = [dataclasses.replace(s, form=dataclasses.replace(
                require_single_matrix(s.form, ""), isRelative=False))
            for s in suite.covarianceSections if _is_endf_mf(s.rowData, 33)]
    _cov, mf3, union, _mts = load_mf33_covariance(MF33_TAPE, None, [4, 16])
    union = np.asarray(union, dtype=float)
    # A refined union, so the lift splits source bins and σ̄ is binned finer.
    union = np.unique(np.concatenate([union, np.sqrt(union[1:] * union[:-1])]))
    mf3 = endf.get_file(3).sections
    binCentral = {mt: binAveragedCentral(mf3[mt], union) for mt in (4, 16)}

    reference = relativiseAbsoluteSections(made, binCentral, union, mf=33)
    got, converted, unresolved, _inert = relativiseOnAssemblyGrids(
        made, {33: {mt: mf3[mt] for mt in (4, 16)}}, lambda za, mf, mt: union)

    assert converted == ["MF33 MT4 x MT4", "MF33 MT16 x MT16"] and unresolved == []
    for a, b in zip(reference, got):
        fa, fb = require_single_matrix(a.form, ""), require_single_matrix(b.form, "")
        assert fb.isRelative is True
        np.testing.assert_array_equal(np.asarray(fb.matrix), np.asarray(fa.matrix))
        np.testing.assert_array_equal(np.asarray(fb.rowGrid), union)


def _assembled(monkeypatch, tape, request, **kwargs):
    """The run, and the blocks it assembled -- what the draw is taken from."""
    captured = []
    original = model_perturbation.assembleRequest

    def capture(*args, **kw):
        blocks, index = original(*args, **kw)
        captured.extend(blocks)
        return blocks, index

    monkeypatch.setattr(model_perturbation, "assembleRequest", capture)
    run = perturbFromModel(tape, request, 2, **kwargs)
    return run, {key: np.asarray(matrix) for key, matrix in captured}


def _draws_what_the_expected_relative_draws(monkeypatch, tape, mf, mt, central_of):
    """The absolute section assembles to the block its expected relative form does.

    Both are stated on the grid the block is assembled on (taken from a run of
    the file as it is), so the assembly lifts neither and any difference is the
    conversion's. Compared on the assembled block and not on drawn factors: the
    two matrices agree to rounding, and an eigenbasis of uncorrelated bins of
    equal variance is free to rotate under a rounding-level change, which moves
    those columns of a draw without changing what is drawn from.
    """
    request = {mf: [mt]}
    kwargs = dict(seed=7, dryRun=True, space="linear")
    union = _assembly_grid(perturbFromModel(tape, request, 2, **kwargs))

    _patch_source(monkeypatch, mf, {mt}, central_of, union=union, absolute=False)
    _run, expected = _assembled(monkeypatch, tape, request, **kwargs)
    monkeypatch.undo()
    _patch_source(monkeypatch, mf, {mt}, central_of, union=union, absolute=True)
    absolute, got = _assembled(monkeypatch, tape, request, **kwargs)

    assert got.keys() == expected.keys()
    for key, matrix in expected.items():
        np.testing.assert_allclose(got[key], matrix, rtol=1e-12,
                                   atol=1e-15 * np.abs(matrix).max())
    assert any("made relative" in n and f"MF{mf} MT{mt} x MT{mt}" in n
               for n in absolute.notes)
    assert not any("not drawn" in n for n in absolute.notes)
    return absolute


def test_an_absolute_mf33_section_is_drawn_as_its_relative_covariance(monkeypatch):
    _draws_what_the_expected_relative_draws(monkeypatch, MF33_TAPE, 33, 4, _mf3_central)


def test_variance_where_the_central_value_is_zero_is_counted():
    """Below a threshold σ̄ is zero: an absolute variance stated there has
    nothing to be relative to, so the relative covariance is zero and the
    conversion says how many bins that is."""
    from types import SimpleNamespace

    suite, _ = decodeCovarianceSuite(read_endf(MF33_TAPE))
    section = next(s for s in suite.covarianceSections if _is_endf_mf(s.rowData, 33))
    form = require_single_matrix(section.form, "")
    grid = np.asarray(form.rowGrid, dtype=float)
    absolute = dataclasses.replace(section, form=dataclasses.replace(
        form, matrix=np.eye(grid.size - 1), isRelative=False))
    threshold = grid[10]
    central = SimpleNamespace(energies=np.array([grid[0], threshold, threshold, grid[-1]]),
                              values=np.array([0.0, 0.0, 2.0, 2.0]))

    sections, converted, _unresolved, inert = relativiseOnAssemblyGrids(
        [absolute], {33: {4: central}}, lambda za, mf, mt: grid)

    label = "MF33 MT4 x MT4"
    assert converted == [label] and inert == {label: 10}
    diagonal = np.diag(np.asarray(require_single_matrix(sections[0].form, "").matrix))
    assert np.all(diagonal[:10] == 0.0)
    np.testing.assert_allclose(diagonal[10:], 0.25)


def test_an_absolute_mf31_section_is_drawn_as_its_relative_covariance(monkeypatch):
    _draws_what_the_expected_relative_draws(
        monkeypatch, NUBAR_TAPE, 31, 456, _nubar_central)


def test_a_section_with_no_central_value_raises_or_is_skipped_and_said(monkeypatch):
    _patch_source(monkeypatch, 33, {16}, _mf3_central, drop_central=(16,))
    with pytest.raises(ValueError, match="cannot be made relative"):
        perturbFromModel(MF33_TAPE, {33: [4, 16]}, 4, seed=7, dryRun=True, space="linear")

    run = perturbFromModel(MF33_TAPE, {33: [4, 16]}, 4, seed=7, dryRun=True, space="linear",
                           onMissing="skip")
    assert [c.mt for c in run.samples[0]["set"].components()] == [4]
    assert any("cannot be made relative" in n and "MF33 MT16 x MT16" in n
               for n in run.notes)
