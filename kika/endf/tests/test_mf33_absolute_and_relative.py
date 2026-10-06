"""MF33 blocks that mix absolute (LB=0/8/9) and relative (LB=1-6) components.

ENDF-6 §33.2 gives the F_k of LB=0/8/9 the dimension of barns², and those of
LB=1/2/5/6 are fractions. Until 2026-10-06 ``_process_ni_records_to_matrix``
summed both into one matrix and called it relative, and projected LB=8 onto a
finer grid as if it were LB=0. The census of JEFF-4.0 and ENDF/B-VIII.1 that
found it (kika-workspace ``docs/library/cov_checks_roadmap.md``, C0) counts
15 039 mixed blocks in 500 JEFF-4.0 tapes, and the summed LB=8 dominates in the
tail: x6400 on Eu-154 MT3.

The synthetic cases pin the rules; the Si-28 case runs the whole path through
NJOY on a real LB=5+8 evaluation.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from kika.endf.classes.endf import ENDF
from kika.endf.classes.mf import MF
from kika.endf.classes.mf33 import (MF33MT, MF33NeedsCrossSections,
                                    NISubSubsectionRecord, Subsection,
                                    mixesAbsoluteAndRelative)
from kika.processing import njoy_pendf_cache as cache

GRID = [1.0, 10.0, 100.0]


def _lb5(values, grid=GRID):
    """An LB=5 LS=1 record: ``values`` is the stored upper triangle."""
    n = len(grid) - 1
    assert len(values) == n * (n + 1) // 2
    return NISubSubsectionRecord(lb=5, ls=1, ne=len(grid), nt=len(grid) + len(values),
                                 energies=list(grid), matrix=list(values))


def _table(lb, f_values, grid=GRID):
    """An LB=0/1/8/9 record: one (E, F) table, the last F unused."""
    return NISubSubsectionRecord(lb=lb, np=len(grid), ne=len(grid), nt=2 * len(grid),
                                 e_table_k=list(grid), f_table_k=list(f_values) + [0.0])


def _section(records, mt=1):
    sec = MF33MT(number=mt, _za=14028.0, _awr=27.737, _mat=1425)
    sec.add_subsection(Subsection(mt1=mt, mat1=0, nc=0, ni=len(records),
                                  ni_records=list(records)))
    return sec


def _xs(value, lo=0.5, hi=200.0):
    """A flat σ(E), spelled the way a CrossSection spells it."""
    return SimpleNamespace(energies=np.array([lo, hi]), values=np.array([value, value]))


# --------------------------------------------------------------------------
# One kind only: unchanged
# --------------------------------------------------------------------------

def test_a_relative_only_block_is_the_decoded_matrix_bit_for_bit():
    rec = _lb5([0.01, 0.002, 0.04])
    sec = _section([rec])
    matrix, grid, is_rel = sec._process_ni_records_to_matrix([rec])
    expected, _ = MF33MT._decode_lb5_matrix(rec)
    assert is_rel is True and grid == GRID
    assert np.array_equal(matrix, expected)


def test_an_absolute_only_block_stays_absolute():
    rec = _table(0, [0.3, 0.5])
    matrix, _, is_rel = _section([rec])._process_ni_records_to_matrix([rec])
    assert is_rel is False
    assert np.array_equal(matrix, np.diag([0.3, 0.5]))


# --------------------------------------------------------------------------
# LB=8 / LB=9 on a grid (manual §33.2)
# --------------------------------------------------------------------------

def test_lb8_on_its_own_grid_is_exactly_f():
    rec = _table(8, [0.04, 0.09])
    matrix, _, is_rel = _section([rec])._process_ni_records_to_matrix([rec])
    assert is_rel is False
    assert np.array_equal(matrix, np.diag([0.04, 0.09]))


def test_lb8_on_a_finer_grid_scales_with_the_group_width_and_stays_uncorrelated():
    rec = _table(8, [0.04, 0.09])
    target = [1.0, 5.0, 10.0, 100.0]
    matrix, _, _ = _section([rec])._process_ni_records_to_matrix([rec], target_grid=target)
    # Var_jj = F_k ΔE_k / ΔE_j for a group inside interval k.
    np.testing.assert_allclose(np.diag(matrix), [0.04 * 9 / 4, 0.04 * 9 / 5, 0.09], rtol=1e-14)
    assert np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0


def test_lb8_on_a_coarser_group_averages_the_pieces_as_uncorrelated():
    rec = _table(8, [0.04, 0.09])
    matrix, _, _ = _section([rec])._process_ni_records_to_matrix([rec], target_grid=[1.0, 100.0])
    expected = (9 / 99) ** 2 * 0.04 + (90 / 99) ** 2 * 0.09
    np.testing.assert_allclose(matrix, [[expected]], rtol=1e-14)


def test_lb9_vanishes_on_its_own_grid_and_grows_as_the_group_narrows():
    rec = _table(9, [0.04, 0.09])
    native, _, _ = _section([rec])._process_ni_records_to_matrix([rec])
    assert np.count_nonzero(native) == 0
    finer, _, _ = _section([rec])._process_ni_records_to_matrix(
        [rec], target_grid=[1.0, 5.0, 10.0, 100.0])
    np.testing.assert_allclose(np.diag(finer), [0.04 * (1 - 4 / 9), 0.04 * (1 - 5 / 9), 0.0],
                               rtol=1e-14, atol=0.0)


# --------------------------------------------------------------------------
# Mixed: converted with σ, refused without it
# --------------------------------------------------------------------------

def test_a_mixed_block_divides_the_absolute_part_by_sigma_squared():
    rel = _lb5([0.01, 0.002, 0.04])
    lb8 = _table(8, [0.04, 0.09])
    sec = _section([rel, lb8])
    sigma = _xs(2.0)
    matrix, _, is_rel = sec._process_ni_records_to_matrix([rel, lb8], xs_row=sigma, xs_col=sigma)
    expected, _ = MF33MT._decode_lb5_matrix(rel)
    expected = expected + np.diag([0.04, 0.09]) / 4.0
    assert is_rel is True
    np.testing.assert_allclose(matrix, expected, rtol=1e-14)


def test_lb0_with_lb1_is_mixed_too():
    lb0, lb1 = _table(0, [0.3, 0.5]), _table(1, [0.01, 0.02])
    assert mixesAbsoluteAndRelative([lb0, lb1])
    matrix, _, is_rel = _section([lb0, lb1])._process_ni_records_to_matrix(
        [lb0, lb1], xs_row=_xs(10.0), xs_col=_xs(10.0))
    assert is_rel is True
    np.testing.assert_allclose(np.diag(matrix), [0.01 + 0.003, 0.02 + 0.005], rtol=1e-14)


def test_a_mixed_block_without_sigma_refuses_and_says_how_to_fix_it():
    rel, lb8 = _lb5([0.01, 0.002, 0.04]), _table(8, [0.04, 0.09])
    with pytest.raises(MF33NeedsCrossSections) as err:
        _section([rel, lb8])._process_ni_records_to_matrix([rel, lb8], mt_label="MT1")
    text = str(err.value)
    assert "LB=[8]" in text and "LB=[5]" in text
    assert "attach_pendf" in text and "NJOY" in text


def test_to_xs_covmat_reads_sigma_from_mf3_sections():
    rel, lb8 = _lb5([0.01, 0.002, 0.04]), _table(8, [0.04, 0.09])
    cov = _section([rel, lb8]).to_xs_covmat(mf3_sections={1: _xs(2.0)})
    assert cov.is_relative == [True]
    np.testing.assert_allclose(np.diag(cov.matrices[0]), [0.01 + 0.01, 0.04 + 0.0225], rtol=1e-14)


# --------------------------------------------------------------------------
# The bin average σ̄ is exact on a lin-lin table
# --------------------------------------------------------------------------

def test_the_bin_average_of_a_linear_sigma_is_closed_form():
    # σ = E on [1, 10]: ∫ σ/E dE / ∫ dE/E = 9 / ln 10.
    table = SimpleNamespace(energies=np.array([1.0, 10.0]), cross_sections=np.array([1.0, 10.0]))
    got = MF33MT._bin_average_xs_exact(table, [1.0, 10.0])
    assert got[0] == pytest.approx(9 / math.log(10), rel=1e-14)


def test_the_bin_average_sees_every_point_of_a_resonant_table():
    # 2001 narrow triangles inside one bin: a fixed-node rule lands on them at
    # random; the closed form integrates them all.
    e = np.linspace(100.0, 200.0, 4001)
    s = np.where(np.arange(e.size) % 2 == 1, 1000.0, 1.0)
    table = SimpleNamespace(energies=e, values=s)
    got = MF33MT._bin_average_xs_exact(table, [100.0, 200.0])[0]
    a =s[:-1] - (s[1:] - s[:-1]) / np.diff(e) * e[:-1]
    b = (s[1:] - s[:-1]) / np.diff(e)
    reference = np.sum(a * np.log(e[1:] / e[:-1]) + b * np.diff(e)) / math.log(2.0)
    assert got == pytest.approx(reference, rel=1e-12)
    assert 490.0 < got < 510.0


# --------------------------------------------------------------------------
# Finding NJOY, and what the user is told when it is not there
# --------------------------------------------------------------------------

def test_no_njoy_anywhere_raises_with_the_three_fixes(monkeypatch):
    monkeypatch.delenv(cache.NJOY_ENV_VAR, raising=False)
    monkeypatch.setattr(cache.shutil, "which", lambda name: None)
    with pytest.raises(cache.NjoyNotFoundError) as err:
        cache.find_njoy_executable("no/such/njoy.exe", why="MF33 MT[1]")
    text = str(err.value)
    assert text.startswith("MF33 MT[1] needs σ(E)")
    assert "NJOY_EXECUTABLE" in text and "setx" in text and "export" in text
    assert "njoy_executable=" in text and "endf.pendf" in text


def test_the_environment_variable_is_honoured(monkeypatch, tmp_path):
    exe = tmp_path / "njoy.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv(cache.NJOY_ENV_VAR, str(exe))
    monkeypatch.setattr(cache.shutil, "which", lambda name: None)
    assert cache.find_njoy_executable() == exe


def test_attach_pendf_keeps_a_pendf_the_caller_set():
    endf = ENDF()
    endf.pendf = {1: _xs(2.0)}
    assert cache.attach_pendf(endf) is endf.pendf


def test_attach_pendf_without_a_source_file_says_so():
    with pytest.raises(cache.NjoyNotFoundError, match="does not know which file"):
        cache.attach_pendf(ENDF())


def test_lb8_no_longer_counts_as_relative_for_the_pendf_sentinel():
    mf33 = SimpleNamespace(sections={
        1: _section([_table(1, [0.01, 0.02])], mt=1),
        2: _section([_lb5([0.01, 0.002, 0.04], ), _table(8, [0.04, 0.09])], mt=2),
    })
    assert cache.mf33_needs_pendf(mf33, 1) is False
    assert cache.mf33_needs_pendf(mf33, 2) is True


def test_the_suite_decode_refuses_a_mixed_block_when_it_has_nothing_to_run_njoy_on():
    from kika.endf.model_adapter import decodeCovarianceSuite
    endf = ENDF()
    mf = MF(number=33)
    mf.add_section(_section([_lb5([0.01, 0.002, 0.04]), _table(8, [0.04, 0.09])]))
    endf.add_file(mf)
    with pytest.raises(cache.NjoyNotFoundError, match="does not know which file"):
        decodeCovarianceSuite(endf)


def test_the_suite_decode_uses_a_pendf_already_attached():
    from kika.endf.model_adapter import decodeCovarianceSuite
    endf = ENDF()
    mf = MF(number=33)
    mf.add_section(_section([_lb5([0.01, 0.002, 0.04]), _table(8, [0.04, 0.09])]))
    endf.add_file(mf)
    endf.pendf = {1: _xs(2.0)}
    suite, report = decodeCovarianceSuite(endf, target="Si28")
    (section,) = suite.covarianceSections
    assert any("NJOY" in w for w in report.warnings)
    assert section is not None


# --------------------------------------------------------------------------
# A real evaluation, through NJOY
# --------------------------------------------------------------------------

def test_si28_b81_lb8_reaches_the_model_divided_by_the_reconstructed_sigma(
        si28_b81_tape, njoy_exe, tmp_path):
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeCovarianceSuite

    endf = read_endf(str(si28_b81_tape), mf_numbers=[1, 2, 3, 33])
    sec = endf.files[33].sections[1]
    (self_sub,) = [s for s in sec.subsections if int(s.mt1) == 1]
    assert mixesAbsoluteAndRelative(self_sub.ni_records)

    cache.attach_pendf(endf, njoy_executable=njoy_exe, cache_dir=tmp_path)
    matrix, grid, is_rel = sec._process_ni_records_to_matrix(
        self_sub.ni_records, xs_row=endf.pendf[1], xs_col=endf.pendf[1])
    assert is_rel is True and np.all(np.isfinite(matrix))

    # MT1 here is LB=0, 1, 1, 8: two absolute components (LB=0 and LB=8).
    assert sorted(int(r.lb) for r in self_sub.ni_records) == [0, 1, 1, 8]
    relative = [r for r in self_sub.ni_records if r.lb in (1, 2, 5, 6)]
    absolute = [r for r in self_sub.ni_records if r.lb in (0, 8, 9)]
    rel_only, _, rel_flag = sec._process_ni_records_to_matrix(relative, target_grid=grid)
    abs_only, _, abs_flag = sec._process_ni_records_to_matrix(absolute, target_grid=grid)
    assert rel_flag is True and abs_flag is False
    sigma = MF33MT._bin_average_xs_exact(endf.pendf[1], grid)
    denom = np.outer(sigma, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        added = np.where(denom > 0, abs_only / denom, 0.0)
    np.testing.assert_allclose(matrix - rel_only, added, rtol=1e-12, atol=1e-300)
    # σ_tot of Si-28 is barns, so the converted part is far below its barns².
    assert np.diag(added).max() < np.diag(abs_only).max()

    suite, report = decodeCovarianceSuite(endf, target="Si28")
    blocks = [s for s in suite.covarianceSections if s.label == "MF33-MT1"]
    assert len(blocks) == 1


def test_a_summation_mt_missing_from_the_pendf_gets_the_sum_of_its_partials():
    # Si-28 of ENDF/B-VIII.1 states MF33 MT3 and gives no MF3 MT3, so RECONR
    # writes none: σ_3 has to be summed from what the PENDF does carry.
    from kika.endf.classes.mf33.mf33 import _xs_section
    mixed = [_lb5([0.01, 0.002, 0.04]), _table(8, [0.04, 0.09])]
    pendf = {4: _xs(1.5), 102: _xs(0.25), 2: _xs(9.0)}
    summed = _xs_section(pendf, 3, mixed)
    np.testing.assert_allclose(summed.values, 1.75)
    # A block that does not mix never looks.
    assert _xs_section(pendf, 3, [mixed[0]]) is None
