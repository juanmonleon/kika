"""Data modes (``dmode_*`` columns): fully correlated data systematics in the chi^2 (16-oct-2026).

Invariants:
1. An all-zero mode column changes nothing, bit for bit (it is dropped per group).
2. V2/V3 (Woodbury) and V4 (dense) equal a direct solve of D + uu' + vv' + MM' [+ Sigma_eval],
   with and without per-energy blocking of u, v (the modes are never blocked).
3. V1's diagonal carries the modes' variance.
4. `_noise_inv_apply` with modes equals the dense inverse; so `coherent_cell_offsets` sees them.
5. `exfor_data_modes.add_mode_columns` builds rel * y_eval + abs on the matching rows only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts import chi2_metrics as CM  # noqa: E402
from scripts import exfor_data_modes as DM  # noqa: E402


def _frame(seed=0, n_e=5, angles=(25.0, 40.0, 92.0, 127.0)):
    rng = np.random.default_rng(seed)
    rows = []
    for exp in ("K", "T"):
        for i in range(n_e):
            for a in angles:
                y_eval = 0.5 + 0.3 * np.cos(np.radians(a))
                rows.append({"library": "L", "experiment_id": exp, "energy_mev": 1.0 + 0.1 * i,
                             "mu": np.cos(np.radians(a)), "angle_deg": a,
                             "y_eval": y_eval, "y_exp": y_eval * (1 + 0.05 * rng.standard_normal()),
                             "sigma_exp_stat": 0.02 * y_eval,
                             "sigma_sys_indep_rel": 0.04, "sigma_sys_dep_rel": 0.03})
    df = pd.DataFrame(rows)
    k = (df.experiment_id == "K").to_numpy()
    df["dmode_eff92"] = np.where(k & (df.angle_deg == 92.0), 0.05 * df.y_eval, 0.0)
    df["dmode_var"] = np.where(k & (df.angle_deg >= 92.0), -0.2 * df.y_eval + 0.01, 0.0)
    return df


def _dense(df, block=None, bids=None):
    D, u, v, r = CM._components(df)
    if bids is not None:
        mask = bids[:, None] == bids[None, :]
        S = np.diag(D) + (np.outer(u, u) + np.outer(v, v)) * mask
    else:
        S = np.diag(D) + np.outer(u, u) + np.outer(v, v)
    M = CM._data_modes(df)
    if M is not None:
        S = S + M @ M.T
    if block is not None:
        S = S + block
    return float(r @ np.linalg.solve(S, r))


def test_zero_mode_column_is_bit_identical():
    df = _frame().drop(columns=["dmode_eff92", "dmode_var"])
    a = CM.chi2_per_experiment_variants(df)
    df["dmode_zero"] = 0.0
    b = CM.chi2_per_experiment_variants(df)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


@pytest.mark.parametrize("blocked", [False, True])
def test_variants_match_dense(blocked):
    df = _frame()
    col = "energy_mev" if blocked else None
    rng = np.random.default_rng(3)
    ev = {}
    for exp, sub in df.groupby("experiment_id"):
        A = 0.01 * rng.standard_normal((len(sub), 3)) * sub.y_eval.to_numpy()[:, None]
        ev[("L", exp)] = A @ A.T
    out = CM.chi2_per_experiment_variants(df, ev, systematic_block_col=col).set_index("experiment_id")
    for exp, sub in df.groupby("experiment_id"):
        sub = sub.reset_index(drop=True)
        bids = CM._resolve_block_codes(sub, col)
        assert out.loc[exp, "chi2_v2"] == pytest.approx(_dense(sub, None, bids), rel=1e-10)
        assert out.loc[exp, "chi2_v4"] == pytest.approx(_dense(sub, ev[("L", exp)], bids), rel=1e-10)
        D, u, v, r = CM._components(sub)
        M = CM._data_modes(sub)
        v1 = np.sum(r ** 2 / (D + u ** 2 + v ** 2 + np.sum(M * M, axis=1)))
        assert out.loc[exp, "chi2_v1"] == pytest.approx(v1, rel=1e-12)


def test_whitened_z_sums_to_chi2_with_modes():
    df = _frame()
    out = CM.chi2_per_experiment_variants(df).set_index("experiment_id")
    z = CM.whitened_residuals_per_experiment_variant(df, variant="V2")
    for exp, idx in df.groupby("experiment_id").groups.items():
        assert float(np.sum(z.loc[idx] ** 2)) == pytest.approx(out.loc[exp, "chi2_v2"], rel=1e-10)


def test_noise_inverse_with_modes():
    rng = np.random.default_rng(1)
    n = 30
    D = 0.5 + rng.random(n); v = rng.standard_normal(n); M = rng.standard_normal((n, 2))
    bids = np.repeat(np.arange(3), 10)
    Y = rng.standard_normal((n, 4))
    S = np.diag(D) + np.outer(v, v) * (bids[:, None] == bids[None, :]) + M @ M.T
    np.testing.assert_allclose(CM._noise_inv_apply(D, v, bids, Y, M), np.linalg.solve(S, Y), rtol=1e-10)
    S0 = np.diag(D) + np.outer(v, v) * (bids[:, None] == bids[None, :])
    np.testing.assert_allclose(CM._noise_inv_apply(D, v, bids, Y), np.linalg.solve(S0, Y), rtol=1e-12)


def test_add_mode_columns(tmp_path):
    df = _frame().drop(columns=["dmode_eff92", "dmode_var"])
    spec = pd.DataFrame([
        {"mode": "e92", "exp_key": "K", "angle_deg": 92.4, "e_lo": 0.0, "e_hi": 99.0, "rel": 0.05, "abs": 0.0},
        {"mode": "cv", "exp_key": "K", "angle_deg": 127.0, "e_lo": 1.0, "e_hi": 1.2, "rel": 0.1, "abs": -0.02},
        {"mode": "cv", "exp_key": "K", "angle_deg": 127.0, "e_lo": 1.2, "e_hi": 9.0, "rel": 0.3, "abs": 0.0},
    ])
    p = tmp_path / "spec.csv"; spec.to_csv(p, index=False)
    DM.add_mode_columns(df, str(p))
    k = (df.experiment_id == "K").to_numpy()
    exp92 = np.where(k & (df.angle_deg == 92.0), 0.05 * df.y_eval, 0.0)
    np.testing.assert_array_equal(df["dmode_e92"].to_numpy(), exp92)
    m127 = k & (df.angle_deg == 127.0)
    lo = m127 & (df.energy_mev < 1.2 - 1e-12)
    exp_cv = np.where(lo, 0.1 * df.y_eval - 0.02, np.where(m127 & ~lo, 0.3 * df.y_eval, 0.0))
    np.testing.assert_allclose(df["dmode_cv"].to_numpy(), exp_cv, rtol=0, atol=1e-15)
    assert not df.loc[~k, ["dmode_e92", "dmode_cv"]].to_numpy().any()


def test_coherent_cell_offsets_see_the_modes():
    """d_c and s_c^2 are the GLS estimate under Sigma_n = D + vv' + MM' with U profiled."""
    rng = np.random.default_rng(5)
    n = 40
    D = (0.02 + 0.01 * rng.random(n)) ** 2; v = 0.02 * rng.standard_normal(n)
    y = 0.5 + 0.3 * rng.random(n); U = (0.04 * y)[:, None]
    M = np.column_stack([np.where(np.arange(n) < 20, 0.05 * y, 0.0), 0.03 * y * np.sign(np.arange(n) - 20)])
    cells = np.repeat(np.arange(4), 10)
    r = 0.03 * rng.standard_normal(n)
    o = CM.coherent_cell_offsets(D, v, U, None, r, y, cells, None, M)
    S = np.diag(D) + np.outer(v, v) + M @ M.T
    Si = np.linalg.inv(S)
    for t in o.itertuples(index=False):
        X = np.column_stack([np.where(cells == t.cell, y, 0.0), U])
        A = X.T @ Si @ X
        beta = np.linalg.solve(A, X.T @ Si @ r)
        assert t.d == pytest.approx(beta[0], rel=1e-8)
        assert t.s2 == pytest.approx(np.linalg.inv(A)[0, 0], rel=1e-8)
