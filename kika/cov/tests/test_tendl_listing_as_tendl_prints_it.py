"""read_tendl_matrix on listings laid out as TENDL prints them.

TENDL's ``*-njoy-cov-187g.all.300.K.matrix`` listings give, per reaction and per
pair of reactions, a table of groups and the correlations in thousandths, in panels
of 25 groups, printing only the groups whose standard deviation is not zero. TENDL
states no licence, so its files are not versioned; KIKA_TEST_TENDL_DIR points at a
folder that holds n-Fe056 and n-C010 listings of TENDL-2023.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kika.cov import read_tendl_matrix
from kika.cov.parse_covmat import InvalidDataFormatError
from kika.energy_grids.grids import LANL187

UPPER_EV = [e * 1.0e6 for e in sorted(LANL187, reverse=True)[:-1]]
NG = len(UPPER_EV)
MAT = 2631


def _panels(rho_milli, printed, mts):
    lines = [" <<< correlation matrix >>>",
             f" column material mat-mt=({MAT:4d},{mts[0]:3d})  vs  row material "
             f"mat-mt=({MAT:4d},{mts[1]:3d})"]
    rows, cols = printed
    for k in range(0, len(cols), 25):
        chunk = cols[k:k + 25]
        lines.append("    row" + "".join(f"{c + 1:5d}" for c in chunk))
        lines.append(" column" + "-" * (5 * len(chunk)))
        for g in rows:
            cells = ["*****" if v is None else f"{v:5d}" for v in (rho_milli[g][c] for c in chunk)]
            lines.append(f"{g + 1:5d} " + "".join(cells))
    return lines


def _self(mt, xs, rsd, rho_milli):
    lines = [f" material mat-mt=({MAT:4d},{mt:3d})",
             " grp      energy      x-sec.    rel.s.d.    std.dev."]
    lines += [f"{g + 1:4d}{UPPER_EV[g]:12.4E}{xs[g]:12.4E}{rsd[g]:12.4E}{xs[g] * rsd[g]:12.4E}"
              for g in range(NG)]
    on = [g for g in range(NG) if rsd[g] != 0]
    return lines + _panels(rho_milli, (on, on), (mt, mt))


def _cross(mts, xs, rsd, rho_milli):
    a, b = mts
    lines = [f" 1st material mat-mt=({MAT:4d},{a:3d})  vs  2nd material mat-mt=({MAT:4d},{b:3d})",
             " grp      energy  1st x-sec.  2nd x-sec.  1st r.s.d.  2nd r.s.d."]
    lines += [f"{g + 1:4d}{UPPER_EV[g]:12.4E}{xs[a][g]:12.4E}{xs[b][g]:12.4E}"
              f"{rsd[a][g]:12.4E}{rsd[b][g]:12.4E}" for g in range(NG)]
    on_a = [g for g in range(NG) if rsd[a][g] != 0]
    on_b = [g for g in range(NG) if rsd[b][g] != 0]
    return lines + _panels(rho_milli, (on_a, on_b), mts)


def _case():
    """MT2 known on the 30 highest groups, MT102 on the 40 lowest; one pair block."""
    xs = {2: np.linspace(1.0, 4.0, NG), 102: np.linspace(0.01, 2.0, NG)}
    rsd = {2: np.zeros(NG), 102: np.zeros(NG)}
    rsd[2][:30] = np.round(np.linspace(0.02, 0.09, 30), 4)
    rsd[102][-40:] = np.round(np.linspace(0.05, 0.30, 40), 4)
    rng = np.random.default_rng(7)
    rho = {}
    for mt in (2, 102):
        on = rsd[mt] != 0
        m = np.zeros((NG, NG), dtype=int)
        sub = np.round(np.corrcoef(rng.normal(size=(on.sum(), on.sum() + 5))) * 1000).astype(int)
        m[np.ix_(on, on)] = sub
        rho[mt] = m
    cross = np.zeros((NG, NG), dtype=int)
    cross[np.ix_(rsd[2] != 0, rsd[102] != 0)] = rng.integers(-400, 400, size=(30, 40))
    rho[(2, 102)] = cross
    return xs, rsd, rho


def _write(path, xs, rsd, rho, cross_rsd=None):
    rsd_c = cross_rsd or rsd
    lines = (_self(2, xs[2], rsd[2], rho[2].tolist())
             + _cross((2, 102), xs, rsd_c, rho[(2, 102)].tolist())
             + _self(102, xs[102], rsd[102], rho[102].tolist()))
    path.write_text("\n".join(lines) + "\n")
    return path


def _block(cov, mt, mt1):
    for k in range(cov.num_matrices):
        if (cov.reaction_rows[k], cov.reaction_cols[k]) == (mt, mt1):
            return cov.matrices[k]
    raise KeyError((mt, mt1))


def test_each_block_is_the_correlation_times_the_relative_deviations(tmp_path):
    xs, rsd, rho = _case()
    cov = read_tendl_matrix(_write(tmp_path / "t.matrix", xs, rsd, rho))
    assert cov.num_groups == NG and cov.num_matrices == 3
    assert all(cov.is_relative)
    for mt, mt1, r in ((2, 2, rho[2]), (102, 102, rho[102]), (2, 102, rho[(2, 102)])):
        np.testing.assert_allclose(_block(cov, mt, mt1), r / 1000.0 * np.outer(rsd[mt], rsd[mt1]),
                                   rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(cov.cross_sections[(26056, 2)], xs[2],
                               rtol=1e-4)


def test_the_lines_of_a_pair_panel_are_the_groups_of_the_first_reaction(tmp_path):
    xs, rsd, rho = _case()
    block = _block(read_tendl_matrix(_write(tmp_path / "t.matrix", xs, rsd, rho)), 2, 102)
    assert np.any(block[:30, -40:]) and not np.any(block[30:, :])


def test_the_grid_keeps_the_file_order_and_takes_its_lowest_boundary_from_lanl187(tmp_path):
    xs, rsd, rho = _case()
    events = []
    cov = read_tendl_matrix(_write(tmp_path / "t.matrix", xs, rsd, rho), diagnostics=events)
    np.testing.assert_allclose(cov.energy_grid, sorted(np.array(LANL187) * 1.0e6, reverse=True),
                               rtol=1e-4)
    assert cov.energy_grid[-1] == pytest.approx(1.0e-5, rel=1e-12)
    (event,) = [e for e in events if e.kind == "count_defaulted"]
    assert event.evidence["structure"] == "LANL187"


def test_an_overflowed_correlation_is_nan_and_said(tmp_path):
    xs, rsd, rho = _case()
    lines = _write(tmp_path / "t.matrix", xs, rsd, rho).read_text().splitlines()
    rho_list = rho[2].tolist()
    rho_list[3][5] = rho_list[5][3] = None
    first_panel = len(_self(2, xs[2], rsd[2], rho[2].tolist()))
    lines[:first_panel] = _self(2, xs[2], rsd[2], rho_list)
    path = tmp_path / "o.matrix"
    path.write_text("\n".join(lines) + "\n")
    events = []
    block = _block(read_tendl_matrix(path, diagnostics=events), 2, 2)
    assert np.isnan(block[3, 5]) and np.isnan(block[5, 3]) and np.isnan(block).sum() == 2
    (event,) = [e for e in events if e.kind == "dropped_values"]
    assert event.evidence["reason"] == "overflow"
    assert sorted(event.evidence["cells"]) == [(4, 6), (6, 4)]


def test_pair_deviations_that_differ_from_the_reaction_table_are_said(tmp_path):
    xs, rsd, rho = _case()
    other = {2: rsd[2] * 1.5, 102: rsd[102]}
    events = []
    cov = read_tendl_matrix(_write(tmp_path / "t.matrix", xs, rsd, rho, cross_rsd=other),
                            diagnostics=events)
    np.testing.assert_allclose(_block(cov, 2, 102),
                               rho[(2, 102)] / 1000.0 * np.outer(rsd[2], rsd[102]), rtol=1e-12)
    (event,) = [e for e in events if e.kind == "dropped_values"]
    assert event.evidence["reason"] == "stddev_differs" and event.where["mt"] == 2


def test_a_table_without_its_correlation_matrix_is_refused(tmp_path):
    xs, rsd, rho = _case()
    lines = _self(2, xs[2], rsd[2], rho[2].tolist())[:2 + NG]
    path = tmp_path / "cut.matrix"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(InvalidDataFormatError, match="not followed by its correlation matrix"):
        read_tendl_matrix(path)


def test_the_result_does_not_depend_on_diagnostics(tmp_path):
    xs, rsd, rho = _case()
    path = _write(tmp_path / "t.matrix", xs, rsd, rho)
    plain, watched = read_tendl_matrix(path), read_tendl_matrix(path, diagnostics=[])
    assert plain.energy_grid == watched.energy_grid
    for a, b in zip(plain.matrices, watched.matrices):
        np.testing.assert_array_equal(a, b)


TENDL_DIR = Path(os.environ.get("KIKA_TEST_TENDL_DIR", "__missing__"))
REAL = [("n-Fe056-njoy-cov-187g.all.300.K.matrix", 66, 9481, 26056),
        ("n-C010-njoy-cov-187g.all.300.K.matrix", 54, 0, 619)]


@pytest.mark.parametrize("name, blocks, overflowed, isotope", REAL, ids=[r[0] for r in REAL])
def test_the_tendl_2023_listings_are_read_whole(name, blocks, overflowed, isotope):
    path = TENDL_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} not in KIKA_TEST_TENDL_DIR")
    events = []
    cov = read_tendl_matrix(path, diagnostics=events)
    assert cov.num_groups == 187 and cov.num_matrices == blocks
    assert sum(int(np.isnan(m).sum()) for m in cov.matrices) == overflowed
    assert set(cov.isotope_rows) == {isotope}
    kept = [e.evidence["mat_kept"] for e in events if "mat_kept" in e.evidence]
    assert kept == ([] if isotope > 1000 else [isotope])
    assert not [e for e in events if e.kind in ("skipped_line", "count_mismatch", "row_order",
                                                "index_out_of_range", "dropped_block")]
    assert not [e for e in events if e.evidence.get("reason") == "stddev_differs"]


@pytest.mark.parametrize("name", [r[0] for r in REAL])
def test_the_tendl_2023_panels_are_where_the_deviations_say(name):
    """Self blocks: symmetric to one printed unit, 1 on the diagonal, printed exactly where sigma is not zero.
    Pair blocks: lines inside the support of the first reaction, columns of the second."""
    path = TENDL_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} not in KIKA_TEST_TENDL_DIR")
    events = []
    read_tendl_matrix(path, diagnostics=events)
    on = {e.where["mt"]: e.evidence["stddev"] > 0 for e in events if e.kind == "raw_stddev"}
    for e in (e for e in events if e.kind == "raw_correlation"):
        rho, mt, mt1 = e.evidence["correlation"], e.where["mt"], e.where["mt1"]
        filled = np.isnan(rho) | (rho != 0)
        assert not (filled.any(axis=1) & ~on[mt]).any(), (mt, mt1)
        assert not (filled.any(axis=0) & ~on[mt1]).any(), (mt, mt1)
        if mt == mt1:
            np.testing.assert_allclose(np.nan_to_num(rho, nan=9.0), np.nan_to_num(rho, nan=9.0).T,
                                       rtol=0, atol=1.0e-3 + 1e-12)  # one printed unit
            np.testing.assert_array_equal(np.diag(rho)[on[mt]], 1.0)
            np.testing.assert_array_equal(filled.any(axis=0), on[mt])

