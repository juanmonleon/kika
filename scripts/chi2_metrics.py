"""Mahalanobis chi^2 with full per-experiment covariance.

Per-experiment covariance for one (library, experiment) block:

    Sigma = D + u u^T + v v^T + Sigma_eval,
        D          = diag(sigma_stat^2)                            [EXFOR uncorrelated]
        u          = sigma_indep_rel * y_eval                      [EXFOR rank-1 normalization]
        v          = sigma_dep_rel  * y_eval                       [EXFOR rank-1 shape]
        Sigma_eval = dense N x N from MF34 (and MF33 in the library pipeline),
                     built per (library, experiment) by scripts.eval_covariance.

Both `u` and `v` model correlated systematics that share a single draw per
experiment per MC sample (manifest ``correlated: true``). `u` has constant
per-row amplitude (energy-independent normalization, e.g. Kinney's 8.06%);
`v` has per-row amplitude varying with energy (piecewise_E calibration shape,
e.g. Cierjacks's 7-12% ERR-T). Putting either on the diagonal would treat
correlated systematics as independent noise and depress chi^2/N.

Sigma_eval carries cross-energy MF34 correlations (mat[k_j, k_k] off-diagonal),
cross-order MF34 correlations propagated through shared a_L sensitivities,
and cross-energy MF33 correlations (library pipeline only). For an experiment
whose evaluation has no MF34/MF33, the caller passes a zero block (or omits
the entry) and Sigma collapses back to the rank-2 EXFOR structure.

Chi^2 is solved via Cholesky on the dense Sigma per experiment for V4:

    chi^2 = r^T Sigma^{-1} r          (cho_solve)
    z     = L^{-1} r,    L L^T = Sigma  (whitened residuals, sum z^2 = chi^2)

V2 and V3 (no dense Sigma_eval) use the closed-form Woodbury reduction for
rank-2 updates D + UU^T where U = [u, v] is N x 2 — still O(N), safe for
28k-point experiments.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular

# Pipeline σ_stat floor — mirrors precompute_chi2_*_c0.py's K&S c₀ fit
# (`sigma_stat = max(sigma_stat, 0.01·|y|)`) and chi2_all_variants_audit.py's
# `_refloor_D`. Without this floor, V4 χ² inflates significantly because the
# rank-deficient Σ_eval inversion amplifies tiny-D points. Keep the three
# downstream scripts (precompute, cluster, audit) consistent on this floor.
SIGMA_STAT_FLOOR_REL = 0.01
# Tiny floor on D_ii to avoid 0/0 when sigma_stat = 0 and y_exp = 0.
DIAGONAL_FLOOR_REL = 1e-3
DIAGONAL_FLOOR_ABS = 1e-30

REQUIRED_COLUMNS = (
    "library", "experiment_id",
    "y_exp", "y_eval",
    "sigma_exp_stat",
    "sigma_sys_indep_rel", "sigma_sys_dep_rel",
)


# ── Sidecar I/O re-export (single import point for notebooks) ────────────────

from scripts.eval_covariance import load_eval_cov, save_eval_cov  # noqa: F401


# ── Core ──────────────────────────────────────────────────────────────────────

def _components(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (D, u, v, r) per row.

    D carries σ_stat² only — σ_dep no longer goes on the diagonal because it's
    a correlated shape mode, not independent noise. Both u (normalization) and
    v (shape) are rank-1 contributions handled in `_per_group_sigma`.

    u and v are referred to the evaluation under test (y_eval), not to the
    measured y_exp. Built on y_exp they are Peelle's Pertinent Puzzle
    (Neudecker, Frühwirth & Leeb, NSE 170 (2012) 54): the metric then rewards
    an evaluation that sits below the data over one that sits above it by the
    same amount. Measured 2026-10-09 (kika-workspace
    docs/chi2-mf4/ppp_peelle_assessment_2026-10-09.md): V2 moves +3…+20 %.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"chi2_metrics: input frame missing columns {sorted(missing)}")

    y_exp = df["y_exp"].to_numpy()
    sigma_stat = df["sigma_exp_stat"].to_numpy()
    sigma_stat = np.maximum(sigma_stat, SIGMA_STAT_FLOOR_REL * np.abs(y_exp))
    y_eval = df["y_eval"].to_numpy()
    u = df["sigma_sys_indep_rel"].to_numpy() * y_eval
    v = df["sigma_sys_dep_rel"].to_numpy() * y_eval

    D = sigma_stat ** 2
    floor = (DIAGONAL_FLOOR_REL * np.abs(y_exp)) ** 2 + DIAGONAL_FLOOR_ABS
    D = np.maximum(D, floor)
    r = y_exp - y_eval
    return D, u, v, r


def _select_block(
    eval_cov: Dict[Tuple, np.ndarray], key: Tuple, sub: pd.DataFrame,
) -> Optional[np.ndarray]:
    """Look up a (library, experiment_id) block, slicing by `_eval_pos` if the
    sub-DataFrame is a row-subset of the original parquet."""
    block = eval_cov.get(tuple(str(p) for p in key))
    if block is None or "_eval_pos" not in sub.columns:
        return block
    pos = sub["_eval_pos"].to_numpy(dtype=int)
    if (pos < 0).any():
        return block  # rows without eval_pos predate the sidecar; skip slicing.
    if pos.size == block.shape[0] and np.array_equal(pos, np.arange(pos.size)):
        return block  # 15-oct: the whole block in order -- no 3.3 GB np.ix_ copy for Cierjacks
    return block[np.ix_(pos, pos)]


def _resolve_block_codes(
    df: pd.DataFrame, systematic_block_col: Optional[str],
) -> Optional[np.ndarray]:
    """Integer block code per row for restricting the EXFOR rank-1 systematics.

    When `systematic_block_col` is given (e.g. ``"energy_mev"``), the rank-1
    normalization (u) and shape (v) modes are correlated only *within* rows
    that share a code — i.e. block-diagonal in that column. This matches a c_0
    that was fit independently per block: with c_0 fit per incident energy, the
    single experiment-wide normalization mode penalises energy-to-energy level
    differences the fit already removed, inflating chi^2. Restricting the
    systematic to per-energy correlation makes the covariance consistent with
    the per-energy fit. None ⇒ one global block (the original behaviour).
    """
    if systematic_block_col is None:
        return None
    if systematic_block_col not in df.columns:
        raise ValueError(
            f"chi2_metrics: systematic_block_col={systematic_block_col!r} "
            f"not in frame columns {sorted(df.columns)}"
        )
    codes, _ = pd.factorize(df[systematic_block_col].to_numpy(), sort=False)
    return codes.astype(np.int64)


def _per_group_sigma(
    D: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    eval_cov_block: Optional[np.ndarray],
    block_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Assemble Sigma = diag(D) + u u^T + v v^T + Sigma_eval for one experiment.

    When `block_ids` is given, the EXFOR rank-1 systematic modes (u, v) are
    masked to block-diagonal: they correlate only within rows sharing a block
    id (e.g. per incident energy). Sigma_eval (MF34/MF33) is left fully dense —
    the evaluated covariance is genuinely cross-energy. See `_resolve_block_codes`.
    """
    # 15-oct: ONE N x N float64 buffer, built in row chunks, instead of five or six live N^2
    # temporaries (diag, two outers, the mask product, the float64 copy of the block, sigma + sigma.T):
    # at Cierjacks' N = 28 631 each is 6.6 GB, which is where the 47 GB MaxRSS came from. Same
    # operations on every element in the same order -- rank-1 terms, mask, then D, then the block,
    # then the symmetrisation -- so the matrix is bit-for-bit the one the old assembly returned.
    n = D.size
    if eval_cov_block is not None and eval_cov_block.size and eval_cov_block.shape != (n, n):
        raise ValueError(
            f"eval_cov block has shape {eval_cov_block.shape}, expected {(n, n)}"
        )
    sigma = np.empty((n, n), dtype=float)
    step = max(1, _ASSEMBLY_CHUNK_ELEMENTS // max(n, 1))
    for a in range(0, n, step):
        b = min(n, a + step)
        rows = sigma[a:b]
        np.multiply(u[a:b, None], u[None, :], out=rows)
        rows += v[a:b, None] * v[None, :]
        if block_ids is not None:
            rows *= (block_ids[a:b, None] == block_ids[None, :])
    sigma.flat[::n + 1] += D            # == diag(D) + rank1: x + 0.0 off the diagonal is exact
    if eval_cov_block is not None and eval_cov_block.size:
        for a in range(0, n, step):
            b = min(n, a + step)
            sigma[a:b] += eval_cov_block[a:b]       # float32 -> float64 is exact
    _symmetrize_inplace(sigma, step)
    return sigma


_ASSEMBLY_CHUNK_ELEMENTS = 1 << 24      # 128 MB of float64 per row chunk


def _symmetrize_inplace(S: np.ndarray, step: int) -> None:
    """``S <- 0.5 * (S + S.T)`` without an N^2 temporary, bit for bit.

    IEEE addition is commutative, so 0.5 * (S_ij + S_ji) is the same number whichever triangle
    computes it; each off-diagonal pair is computed once and written to both."""
    n = S.shape[0]
    for a in range(0, n, step):
        b = min(n, a + step)
        d = S[a:b, a:b]
        d[...] = 0.5 * (d + d.T)
        for c in range(b, n, step):
            e = min(n, c + step)
            m = 0.5 * (S[a:b, c:e] + S[c:e, a:b].T)
            S[a:b, c:e] = m
            S[c:e, a:b] = m.T


def _solve(sigma: np.ndarray, r: np.ndarray, label: str = "") -> Tuple[float, np.ndarray]:
    """Return (chi^2, whitened residual z) for one experiment.

    Uses Cholesky; falls back to a small jitter on the diagonal when the block
    is numerically indefinite (warn-only, never silently fixes the matrix).
    """
    try:
        c, low = cho_factor(sigma, lower=True, overwrite_a=False, check_finite=False)
    except np.linalg.LinAlgError:
        trace = float(np.trace(sigma))
        jitter = max(1e-12 * abs(trace) / max(sigma.shape[0], 1), 1e-30)
        warnings.warn(
            f"[{label}] Sigma not PSD under Cholesky; adding jitter={jitter:.3g}",
            RuntimeWarning,
            stacklevel=2,
        )
        c, low = cho_factor(
            sigma + jitter * np.eye(sigma.shape[0]),
            lower=True, overwrite_a=False, check_finite=False,
        )
    chi2 = float(r @ cho_solve((c, low), r, check_finite=False))
    z = solve_triangular(
        c if low else c.T, r, lower=True, check_finite=False,
    )
    return chi2, z


def _solve_built(build, r: np.ndarray, label: str = "", keep=None,
                 keep_args: Tuple = ()) -> Tuple[float, np.ndarray]:
    """`_solve` on the matrix ``build()`` returns, factorised IN PLACE (15-oct).

    `_solve` keeps the caller's matrix (``overwrite_a=False``), i.e. a second N^2 copy for the
    factor. Here the matrix is rebuilt only in the rare case the first Cholesky fails and the
    jitter retry needs it intact. Same LAPACK calls on the same numbers: chi^2 and z are those
    of `_solve`."""
    sigma = build()
    try:
        c, low = cho_factor(sigma, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError:
        sigma = build()
        trace = float(np.trace(sigma))
        jitter = max(1e-12 * abs(trace) / max(sigma.shape[0], 1), 1e-30)
        warnings.warn(
            f"[{label}] Sigma not PSD under Cholesky; adding jitter={jitter:.3g}",
            RuntimeWarning,
            stacklevel=2,
        )
        sigma.flat[::sigma.shape[0] + 1] += jitter      # == sigma + jitter * eye
        c, low = cho_factor(sigma, lower=True, overwrite_a=True, check_finite=False)
    chi2 = float(r @ cho_solve((c, low), r, check_finite=False))
    z = solve_triangular(
        c if low else c.T, r, lower=True, check_finite=False,
    )
    if keep is not None:
        keep(*keep_args, (c, low), r)
    return chi2, z


def _rank2_chi2_woodbury(
    D: np.ndarray, u: np.ndarray, v: np.ndarray, r: np.ndarray,
) -> float:
    """Closed-form chi^2 for Σ = diag(D) + u uᵀ + v vᵀ via Woodbury (O(N)).

    Let U = [u, v] (N × 2). Then
        Σ⁻¹ = D⁻¹ − D⁻¹ U M⁻¹ Uᵀ D⁻¹,    M = I₂ + Uᵀ D⁻¹ U
    so
        rᵀ Σ⁻¹ r = rᵀ D⁻¹ r − (Uᵀ D⁻¹ r)ᵀ M⁻¹ (Uᵀ D⁻¹ r).

    Generalises the rank-1 Sherman-Morrison closed form previously used; for
    v ≡ 0 the result equals the rank-1 formula exactly (s_uv = s_vv = 0 ⇒
    M = diag(1 + s_uu, 1), inverse trivial).
    """
    d = np.maximum(D, 1e-300)
    s_uu = float(np.sum(u * u / d))
    s_vv = float(np.sum(v * v / d))
    s_uv = float(np.sum(u * v / d))
    s_ur = float(np.sum(u * r / d))
    s_vr = float(np.sum(v * r / d))
    s_rr = float(np.sum(r * r / d))
    det = (1.0 + s_uu) * (1.0 + s_vv) - s_uv ** 2
    if det <= 0.0:
        # Should not happen for PSD D + UUᵀ; fall back to direct rank-1 if v=0.
        return s_rr - s_ur ** 2 / max(1.0 + s_uu, 1e-300)
    correction = (
        (1.0 + s_vv) * s_ur ** 2
        - 2.0 * s_uv * s_ur * s_vr
        + (1.0 + s_uu) * s_vr ** 2
    ) / det
    return s_rr - correction


def _rank2_chi2_woodbury_blocked(
    D: np.ndarray, u: np.ndarray, v: np.ndarray, r: np.ndarray,
    block_ids: Optional[np.ndarray] = None,
) -> float:
    """Sum of per-block rank-2 Woodbury chi^2.

    With `block_ids=None` this is exactly `_rank2_chi2_woodbury` over all rows.
    Otherwise Σ = diag(D) + u uᵀ + v vᵀ is block-diagonal in `block_ids` (the
    systematics correlate only within a block), so χ² = Σ_b r_bᵀ Σ_b⁻¹ r_b. Still
    O(N) and avoids materialising any dense matrix.
    """
    if block_ids is None:
        return _rank2_chi2_woodbury(D, u, v, r)
    total = 0.0
    for b in np.unique(block_ids):
        m = block_ids == b
        total += _rank2_chi2_woodbury(D[m], u[m], v[m], r[m])
    return total


def mahalanobis_chi2_per_experiment(
    df: pd.DataFrame,
    eval_cov: Optional[Dict[Tuple, np.ndarray]] = None,
    *,
    group_cols: Tuple[str, ...] = ("library", "experiment_id"),
    systematic_block_col: Optional[str] = None,
) -> pd.DataFrame:
    """One row per group with N, chi^2, chi^2_per_N.

    Parameters
    ----------
    df : per-row chi^2 frame (REQUIRED_COLUMNS).
    eval_cov : dict mapping group key tuple -> dense (N, N) Sigma_eval block.
        Missing keys are treated as zero blocks (no MF34/MF33 contribution).
        Keys are matched as `tuple(str(...) for v in group)` to align with
        the on-disk .npz schema produced by scripts.eval_covariance.
    systematic_block_col : optional column name (e.g. ``"energy_mev"``) that
        restricts the EXFOR rank-1 systematic modes to correlate only within
        each block. See `_resolve_block_codes`. None ⇒ one experiment-wide mode.
    """
    D_all, u_all, v_all, r_all = _components(df)
    eval_cov = eval_cov or {}
    block_codes = _resolve_block_codes(df, systematic_block_col)

    groups = df.groupby(list(group_cols), observed=True, sort=False)
    out_rows = []
    for key, sub in groups:
        if not isinstance(key, tuple):
            key = (key,)
        idx = sub.index
        ord_pos = df.index.get_indexer(idx)
        D = D_all[ord_pos]
        u = u_all[ord_pos]
        v = v_all[ord_pos]
        r = r_all[ord_pos]

        block = _select_block(eval_cov, key, sub)
        bids = None if block_codes is None else block_codes[ord_pos]
        label = "/".join(str(p) for p in key)
        chi2, _z = _solve_built(lambda: _per_group_sigma(D, u, v, block, bids), r, label=label)
        N = len(sub)
        out_rows.append({
            **{col: val for col, val in zip(group_cols, key)},
            "N": N,
            "chi2": chi2,
            "chi2_per_N": chi2 / N if N > 0 else np.nan,
        })
    return pd.DataFrame(out_rows)


def chi2_per_experiment_variants(
    df: pd.DataFrame,
    eval_cov: Optional[Dict[Tuple, np.ndarray]] = None,
    *,
    group_cols: Tuple[str, ...] = ("library", "experiment_id"),
    include_eval_diag: bool = True,
    systematic_block_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute all four chi^2 variants per group in a single pass.

    Variants (Σ definition per experiment):
      V1 textbook diag   : Σ = diag(σ_stat² + (σ_indep·y)² + (σ_dep·y)² + σ_eval_diag²)
      V2 rank-2 only     : Σ = D + u uᵀ + v vᵀ                (D excludes σ_eval)
      V3 rank-2 + diag   : Σ = (D + diag(σ_eval_diag²)) + u uᵀ + v vᵀ
      V4 rank-2 + dense  : Σ = D + u uᵀ + v vᵀ + Σ_eval_block (this is the existing
                                                                `mahalanobis_chi2_per_experiment`)

    For V1 and V3 we read σ_eval_diag from the parquet column of the same name
    (set by precompute_chi2_*_c0.py to √ diag(block)). When `eval_cov` is None
    we still need σ_eval_diag in df for V1/V3 to be meaningful; if the column
    is absent it is treated as zero (degenerates V1→ rank-0 textbook with only
    EXFOR diagonal, V3→V2).

    V4 is computed only if `eval_cov` is provided; otherwise the V4 column is
    NaN.

    Setting `include_eval_diag=False` zeroes σ_eval_diag everywhere, turning V1
    into the "EXFOR-only" variant (σ² = σ_stat² + (σ_indep·y)² + (σ_dep·y)²)
    that removes the per-library evaluator cushion. V3 then collapses to V2
    and V4 is unchanged (the dense block is independent of the diagonal flag).

    `systematic_block_col` (e.g. ``"energy_mev"``) makes the EXFOR rank-1
    systematics block-diagonal per block in V2/V3/V4 — use it when c_0 was fit
    per block so the covariance does not penalise per-block level differences
    the fit already removed. V1 is diagonal and unaffected. None ⇒ one global
    experiment-wide mode (original behaviour).
    """
    return chi2_variants_and_z(
        df, eval_cov, group_cols=group_cols, include_eval_diag=include_eval_diag,
        systematic_block_col=systematic_block_col, z_variant=None,
    )[0]


def chi2_variants_and_z(
    df: pd.DataFrame,
    eval_cov: Optional[Dict[Tuple, np.ndarray]] = None,
    *,
    group_cols: Tuple[str, ...] = ("library", "experiment_id"),
    include_eval_diag: bool = True,
    systematic_block_col: Optional[str] = None,
    z_variant: Optional[str] = "V4",
    on_group=None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """`chi2_per_experiment_variants` AND `whitened_residuals_per_experiment_variant` in one pass.

    15-oct. The analysis asked for the per-experiment chi^2 and then, separately, for the
    whitened residuals, of every subset -- and the subsets and subentries are unions of whole
    experiments, so each large block (Cierjacks N = 28 631) was assembled and factorised 8 times
    per library, ~31 of the 69 minutes of a run. Here every (library, experiment) group is
    factorised once; ``z`` for ``z_variant`` = "V4" comes from the SAME factor (`_solve` always
    returned it). Per group the numbers are those of the two functions, bit for bit, so a subset's
    table is a row filter of this one.

    ``on_group(key, ord_pos, sigma_factor, r, eval_block)``, when given, is called after each V4
    factorisation with ``sigma_factor = (c, low)`` and the group's Sigma_eval block (or None) --
    the hook for per-cell diagnostics that need the factor or the declared covariance.

    Returns ``(per_exp, z)``; ``z`` is None when ``z_variant`` is None.
    """
    if z_variant is not None:
        z_variant = z_variant.upper()
        if z_variant not in {"V1", "V2", "V3", "V4"}:
            raise ValueError(f"variant must be V1/V2/V3/V4, got {z_variant!r}")
    D_all, u_all, v_all, r_all = _components(df)
    eval_cov = eval_cov or {}
    block_codes = _resolve_block_codes(df, systematic_block_col)
    if include_eval_diag:
        sigma_eval_diag = (
            df["sigma_eval_diag"].to_numpy(dtype=float)
            if "sigma_eval_diag" in df.columns
            else np.zeros(len(df), dtype=float)
        )
    else:
        sigma_eval_diag = np.zeros(len(df), dtype=float)

    z_full = None if z_variant is None else np.full(len(df), np.nan, dtype=float)
    groups = df.groupby(list(group_cols), observed=True, sort=False)
    out_rows = []
    for key, sub in groups:
        if not isinstance(key, tuple):
            key = (key,)
        idx = sub.index
        ord_pos = df.index.get_indexer(idx)
        D = D_all[ord_pos]
        u = u_all[ord_pos]
        v = v_all[ord_pos]
        r = r_all[ord_pos]
        sed = sigma_eval_diag[ord_pos]
        N = r.size
        label = "/".join(str(p) for p in key)

        # V1 textbook: diag( D + u² + v² + σ_eval_diag² ). The v² term here
        # encodes a *worst-case* point-by-point inflation that ignores σ_dep's
        # within-experiment correlation; included so V1 vs V2/V3/V4 measures
        # exactly the effect of marginalising the correlated modes.
        v1_denom = np.maximum(D + u ** 2 + v ** 2 + sed ** 2, 1e-300)
        v1 = float(np.sum(r ** 2 / v1_denom))

        bids = None if block_codes is None else block_codes[ord_pos]

        # V2 rank-2 only: D + uuᵀ + vvᵀ — Woodbury closed form (O(N)),
        # block-diagonal in `bids` when per-block systematics are requested.
        v2 = _rank2_chi2_woodbury_blocked(D, u, v, r, bids)

        # V3 rank-2 + diag eval: (D + σ_eval_diag²) + uuᵀ + vvᵀ — same form.
        v3 = _rank2_chi2_woodbury_blocked(D + sed ** 2, u, v, r, bids)

        # V4 rank-2 + dense eval (EXFOR rank-1 masked per block, Σ_eval dense).
        if eval_cov:
            block = _select_block(eval_cov, key, sub)
            build = lambda: _per_group_sigma(D, u, v, block, bids)  # noqa: E731
            if on_group is None:
                v4, z4 = _solve_built(build, r, label=f"{label}/V4")
            else:
                v4, z4 = _solve_built(build, r, label=f"{label}/V4",
                                      keep=lambda k_, o_, f_, r_: on_group(k_, o_, f_, r_, block),
                                      keep_args=(key, ord_pos))
            del block
        else:
            v4 = float("nan")
            # without a sidecar V4's Sigma has no eval block, as the whitened function computes it
            z4 = (_solve_built(lambda: _per_group_sigma(D, u, v, None, bids), r,
                               label=f"{label}/V4")[1] if z_variant == "V4" else None)

        if z_full is not None:
            if z_variant == "V4":
                z_full[ord_pos] = z4 if z4 is not None else np.nan
            elif z_variant == "V1":
                z_full[ord_pos] = r / np.sqrt(np.maximum(D + u ** 2 + v ** 2 + sed ** 2, 1e-300))
            else:
                D_eff = D if z_variant == "V2" else D + sed ** 2
                _c, z_full[ord_pos] = _solve_built(
                    lambda: _per_group_sigma(D_eff, u, v, None, bids), r,
                    label=f"{label}/{z_variant}")

        out_rows.append({
            **{col: val for col, val in zip(group_cols, key)},
            "N": N,
            "chi2_v1": v1,
            "chi2_v2": v2,
            "chi2_v3": v3,
            "chi2_v4": v4,
            "chi2_v1_per_N": v1 / N if N > 0 else np.nan,
            "chi2_v2_per_N": v2 / N if N > 0 else np.nan,
            "chi2_v3_per_N": v3 / N if N > 0 else np.nan,
            "chi2_v4_per_N": v4 / N if N > 0 else np.nan,
        })
    z = None if z_full is None else pd.Series(z_full, index=df.index, name=f"z_{z_variant.lower()}")
    return pd.DataFrame(out_rows), z


def whitened_residuals_per_experiment(
    df: pd.DataFrame,
    eval_cov: Optional[Dict[Tuple, np.ndarray]] = None,
    *,
    group_cols: Tuple[str, ...] = ("library", "experiment_id"),
    systematic_block_col: Optional[str] = None,
) -> pd.Series:
    """Per-row whitened residual z aligned to df.index.

    Satisfies (z**2).groupby(group_cols).sum() == chi2 from
    mahalanobis_chi2_per_experiment, to numerical precision. `systematic_block_col`
    blocks the EXFOR rank-1 systematics per block (see `_resolve_block_codes`).
    """
    D_all, u_all, v_all, r_all = _components(df)
    eval_cov = eval_cov or {}
    block_codes = _resolve_block_codes(df, systematic_block_col)

    z_full = np.full(len(df), np.nan, dtype=float)
    for key, sub in df.groupby(list(group_cols), observed=True, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        idx = sub.index
        ord_pos = df.index.get_indexer(idx)
        D = D_all[ord_pos]
        u = u_all[ord_pos]
        v = v_all[ord_pos]
        r = r_all[ord_pos]

        block = _select_block(eval_cov, key, sub)
        bids = None if block_codes is None else block_codes[ord_pos]
        label = "/".join(str(p) for p in key)
        _chi2, z = _solve_built(lambda: _per_group_sigma(D, u, v, block, bids), r, label=label)
        z_full[ord_pos] = z
    return pd.Series(z_full, index=df.index, name="z")


def whitened_residuals_per_experiment_variant(
    df: pd.DataFrame,
    eval_cov: Optional[Dict[Tuple, np.ndarray]] = None,
    *,
    variant: str = "V4",
    group_cols: Tuple[str, ...] = ("library", "experiment_id"),
    include_eval_diag: bool = True,
    systematic_block_col: Optional[str] = None,
) -> pd.Series:
    """Per-row whitened residual z for one of the four chi^2 variants.

    variant ∈ {"V1","V2","V3","V4"}:
      V1 textbook diag  : z_i = r_i / sqrt(D_i + u_i² + v_i² + σ_eval_diag_i²)
      V2 rank-2 only    : Σ = D + u uᵀ + v vᵀ                (dense Cholesky)
      V3 rank-2 + diag  : Σ = (D + diag σ_eval_diag²) + u uᵀ + v vᵀ
      V4 rank-2 + dense : Σ = D + u uᵀ + v vᵀ + Σ_eval_block (current default)

    sum(z²) per group equals chi²_v{n} from chi2_per_experiment_variants
    to numerical precision.

    Setting `include_eval_diag=False` zeroes σ_eval_diag everywhere (EXFOR-only
    diagonal); affects V1 and V3, leaves V2/V4 unchanged.
    """
    variant = variant.upper()
    if variant not in {"V1", "V2", "V3", "V4"}:
        raise ValueError(f"variant must be V1/V2/V3/V4, got {variant!r}")
    D_all, u_all, v_all, r_all = _components(df)
    eval_cov = eval_cov or {}
    block_codes = _resolve_block_codes(df, systematic_block_col)
    if include_eval_diag:
        sigma_eval_diag = (
            df["sigma_eval_diag"].to_numpy(dtype=float)
            if "sigma_eval_diag" in df.columns
            else np.zeros(len(df), dtype=float)
        )
    else:
        sigma_eval_diag = np.zeros(len(df), dtype=float)

    z_full = np.full(len(df), np.nan, dtype=float)
    for key, sub in df.groupby(list(group_cols), observed=True, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        ord_pos = df.index.get_indexer(sub.index)
        D = D_all[ord_pos]
        u = u_all[ord_pos]
        v = v_all[ord_pos]
        r = r_all[ord_pos]
        sed = sigma_eval_diag[ord_pos]
        label = "/".join(str(p) for p in key) + f"/{variant}"

        if variant == "V1":
            denom = np.sqrt(np.maximum(D + u ** 2 + v ** 2 + sed ** 2, 1e-300))
            z_full[ord_pos] = r / denom
            continue

        bids = None if block_codes is None else block_codes[ord_pos]
        if variant == "V2":
            D_eff, block = D, None
        elif variant == "V3":
            D_eff, block = D + sed ** 2, None
        else:  # V4
            D_eff, block = D, _select_block(eval_cov, key, sub)
        _chi2, z = _solve_built(lambda: _per_group_sigma(D_eff, u, v, block, bids), r, label=label)
        z_full[ord_pos] = z
    return pd.Series(z_full, index=df.index, name=f"z_{variant.lower()}")


# ── Conditional chi^2 per (energy band x angle band) cell (15-oct) ─────────────


def conditional_cell_scores(
    factor: Tuple[np.ndarray, bool],
    r: np.ndarray,
    U: np.ndarray,
    cell_of_row: np.ndarray,
    y_eval: np.ndarray,
) -> pd.DataFrame:
    """Where, inside one experiment, the residual disagrees with its covariance.

    ``factor`` is the Cholesky factor of the experiment's V4 Sigma (`cho_factor` form); it is
    OVERWRITTEN here by Sigma^-1 (LAPACK dpotri), so call this last. ``U`` (N x k) holds the
    normalisation modes profiled out with a flat prior -- the experiment's sigma_sys_indep column,
    split per block when the systematics are per block. Sigma already contains U U', and adding
    t U U' with t -> infinity gives the profiled precision

        Q = Sigma^-1 - Sigma^-1 U (U' Sigma^-1 U)^-1 U' Sigma^-1.

    For a cell c (its rows), the Gaussian conditional of r_c given every other row is
    e_c = Q_cc^-1 (Q r)_c with covariance Q_cc^-1, so

        chi2_c = (Q r)_c' Q_cc^-1 (Q r)_c  ~  chi2(rank Q_cc),

    and the signed, standardised mean of the cell is w_c = 1'e_c / sqrt(1' Q_cc^-1 1) ~ N(0, 1).
    Per point, zhat_i = (Q r)_i / sqrt(Q_ii) is the one-point cell (order-free, unlike the
    Cholesky z). The cell chi^2 LOCATES disagreement; the cells condition on overlapping rests,
    so they do not add up to the experiment's chi^2.

    Returns one row per cell with the sums an aggregation over experiments needs exactly:
    chi2, dof, w_num = 1'e_c, w_den2 = 1'Q_cc^-1 1, and the raw relative residuals.
    """
    from scipy.linalg.lapack import dpotri

    c, low = factor
    n = r.size
    Sr = cho_solve((c, low), r, check_finite=False)
    k = U.shape[1]
    if k:
        SU = cho_solve((c, low), U, check_finite=False)
        M = U.T @ SU
        Mi = np.linalg.pinv(M)
        Qr = Sr - SU @ (Mi @ (U.T @ Sr))
    else:
        Qr = Sr
    inv, info = dpotri(c, lower=1 if low else 0, overwrite_c=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"dpotri info {info}")
    if not low:
        raise ValueError("conditional_cell_scores expects a lower factor")
    # only the lower triangle of `inv` is Sigma^-1; diag(Q)
    q_diag = np.diagonal(inv).copy()
    if k:
        q_diag -= np.einsum("ij,ij->i", SU @ Mi, SU)
    zhat = Qr / np.sqrt(np.maximum(q_diag, 1e-300))
    rel = r / np.where(y_eval != 0, y_eval, np.nan)

    rows = []
    for cell in np.unique(cell_of_row):
        idx = np.flatnonzero(cell_of_row == cell)          # ascending: rows >= cols is the lower triangle
        A = inv[np.ix_(idx, idx)]
        A = np.tril(A) + np.tril(A, -1).T
        if k:
            G = SU[idx]
            A = A - G @ Mi @ G.T
        q = Qr[idx]
        w, V = np.linalg.eigh(0.5 * (A + A.T))
        keep = w > w.max() * 1e-10 if w.size and w.max() > 0 else np.zeros(w.size, bool)
        Ainv = (V[:, keep] / w[keep]) @ V[:, keep].T       # Q_cc^-1 on its range
        e = Ainv @ q
        one = np.ones(idx.size)
        rows.append({
            "cell": cell, "n": int(idx.size),
            "chi2": float(q @ e), "dof": int(keep.sum()),
            "w_num": float(one @ e), "w_den2": float(one @ Ainv @ one),
            "rel_resid_sum": float(np.nansum(rel[idx])),
            "cond_rel_sum": float(np.nansum(e / np.where(y_eval[idx] != 0, y_eval[idx], np.nan))),
            "n_z1": int(np.sum(np.abs(zhat[idx]) < 1.0)), "n_z2": int(np.sum(np.abs(zhat[idx]) < 2.0)),
        })
    return pd.DataFrame(rows)


def cell_codes(energy_mev: np.ndarray, angle_deg: np.ndarray,
               e_edges: np.ndarray, a_edges: np.ndarray) -> np.ndarray:
    """Integer cell id = i_E * n_A + i_A on the given band edges (outermost bands closed)."""
    ie = np.clip(np.searchsorted(e_edges, energy_mev, side="right") - 1, 0, len(e_edges) - 2)
    ia = np.clip(np.searchsorted(a_edges, angle_deg, side="right") - 1, 0, len(a_edges) - 2)
    return ie * (len(a_edges) - 1) + ia


def profiled_mode_matrix(u: np.ndarray, block_ids: Optional[np.ndarray]) -> np.ndarray:
    """The normalisation mode(s) a cell score profiles: u, or u split per systematic block."""
    if block_ids is None:
        return u[:, None] if np.any(u != 0) else np.zeros((u.size, 0))
    cols = [np.where(block_ids == b, u, 0.0) for b in np.unique(block_ids)]
    cols = [c_ for c_ in cols if np.any(c_ != 0)]
    return np.column_stack(cols) if cols else np.zeros((u.size, 0))


def _noise_inv_apply(D: np.ndarray, v: np.ndarray, block_ids: Optional[np.ndarray],
                     Y: np.ndarray) -> np.ndarray:
    """Sigma_n^-1 Y for Sigma_n = diag(D) + sum_b v_b v_b' (v masked to each block b).

    The blocks have disjoint support, so the Woodbury capacitance is diagonal: O(N m)."""
    Y = Y.reshape(Y.shape[0], -1)
    bid = (np.zeros(D.size, dtype=np.int64) if block_ids is None
           else np.unique(block_ids, return_inverse=True)[1].ravel())
    vd = v / D
    s = np.bincount(bid, weights=v * vd)                                  # v_b' D^-1 v_b
    T = np.stack([np.bincount(bid, weights=vd * Y[:, j], minlength=s.size)
                  for j in range(Y.shape[1])], axis=1)                   # v_b' D^-1 Y
    return Y / D[:, None] - vd[:, None] * (T / (1.0 + s)[:, None])[bid]


def coherent_cell_offsets(
    D: np.ndarray, v: np.ndarray, U: np.ndarray, block_ids: Optional[np.ndarray],
    r: np.ndarray, y_eval: np.ndarray, cell_of_row: np.ndarray,
    eval_block: Optional[np.ndarray],
) -> pd.DataFrame:
    """Per cell of ONE experiment: its relative offset from the evaluation, for the covariance.

    15-oct (Juan: the cell map is for finding where the covariance UNDER-declares, not for the
    central value). For each cell c the GLS fit

        r = x_c d_c + U a + eps,   x_c = y_eval on the cell's rows (0 elsewhere),
        eps ~ N(0, Sigma_n),       Sigma_n = diag(D) + v v'   (the experiment's OWN covariance;
                                   u u' is replaced by the free modes U -- the normalisation
                                   is profiled, as in `conditional_cell_scores`)

    gives d_c (relative; the cell against the rest of the experiment, which is all a shape
    covariance like MF34 can be asked about) with s_c^2 = var(d_c | experiment) and weights h_c
    (d_c = h_c' r). The evaluation's declared covariance enters only as g_c^2 = h_c' Sigma_eval h_c:
    the variance the declared band gives that same estimator. Experiments whose d_c agree with
    each other but sit many sqrt(s^2 + g^2) from zero mark a cell where Sigma_eval under-declares;
    the pooling is `cell_meta_analysis`.

    A cell whose x_c lies in the span of U (an experiment with one cell) has no offset: skipped.
    """
    cells = np.unique(cell_of_row)
    k = U.shape[1]
    nc = cells.size
    X = np.zeros((r.size, nc))
    X[np.arange(r.size), np.searchsorted(cells, cell_of_row)] = y_eval
    Z = np.concatenate([X, U, r[:, None]], axis=1)
    P = _noise_inv_apply(D, v, block_ids, Z)                   # Sigma_n^-1 [X, U, r]
    A = Z[:, :-1].T @ P[:, :-1]
    A = 0.5 * (A + A.T)
    b = Z[:, :-1].T @ P[:, -1]
    ui = np.arange(nc, nc + k)
    rows, H = [], []
    for j, cell in enumerate(cells):
        sel = np.concatenate([[j], ui]).astype(int)
        As = A[np.ix_(sel, sel)]
        if k:
            schur = As[0, 0] - As[0, 1:] @ np.linalg.lstsq(As[1:, 1:], As[1:, 0], rcond=None)[0]
            if not schur > 1e-10 * As[0, 0]:
                continue
        Ai = np.linalg.pinv(As)
        rows.append({"cell": cell, "n": int(np.sum(cell_of_row == cell)),
                     "d": float(Ai[0] @ b[sel]), "s2": float(Ai[0, 0])})
        H.append(P[:, sel] @ Ai[:, 0])                         # d = h' r
    out = pd.DataFrame(rows, columns=["cell", "n", "d", "s2"])
    g2 = np.zeros(len(out))
    if eval_block is not None and eval_block.size and H:
        Hm = np.stack(H, axis=1)
        step = max(1, _ASSEMBLY_CHUNK_ELEMENTS // max(r.size, 1))
        for a in range(0, r.size, step):                        # no float64 copy of the block
            e = min(r.size, a + step)
            g2 += np.einsum("ij,ij->j", Hm[a:e], np.asarray(eval_block[a:e], dtype=float) @ Hm)
    out["g2"] = g2
    return out


def cell_meta_analysis(long: pd.DataFrame) -> pd.DataFrame:
    """Random-effects pooling of `coherent_cell_offsets` over the experiments of each cell.

    Per (library, cell), with d_e, s_e^2 and g_e^2 per experiment:

        d_e = delta + b_e + eps_e,   b_e ~ N(0, tau^2) (between experiments),  eps_e ~ N(0, s_e^2)

    DerSimonian-Laird tau^2; delta_hat = sum w d / sum w with w = 1 / (s^2 + tau^2), se^2 = 1/sum w.
    The evaluation's error is COMMON to the experiments of a cell, so it does not average down:
    sigma_eval = sqrt(sum w g^2 / sum w), and

        z = delta_hat / sqrt(se^2 + sigma_eval^2)   ~ N(0, 1) where Sigma_eval is honest,
        f = sqrt(max(delta_hat^2 - se^2, 0)) / sigma_eval   (offset in units of the band).

    In sample (the evaluation was fitted to these data) delta_hat is pulled toward 0, so an
    |z| > 2 here is, if anything, an under-statement.
    """
    rows = []
    for (lib, cell), s in long.groupby(["library", "cell"], observed=True):
        s = s[np.isfinite(s["d"]) & (s["s2"] > 0)]
        m = len(s)
        if m == 0:
            continue
        d, s2, g2 = s["d"].to_numpy(), s["s2"].to_numpy(), s["g2"].to_numpy()
        w0 = 1.0 / s2
        d_fe = np.sum(w0 * d) / np.sum(w0)
        Q = float(np.sum(w0 * (d - d_fe) ** 2))
        den = np.sum(w0) - np.sum(w0 ** 2) / np.sum(w0)
        tau2 = max(0.0, (Q - (m - 1)) / den) if m > 1 and den > 0 else 0.0
        w = 1.0 / (s2 + tau2)
        delta = float(np.sum(w * d) / np.sum(w))
        se2 = float(1.0 / np.sum(w))
        sig_eval = float(np.sqrt(np.sum(w * g2) / np.sum(w)))
        tot = np.sqrt(se2 + sig_eval ** 2)
        rows.append({
            "library": lib, "cell": cell, "n_exp": m, "n": int(s["n"].sum()),
            "delta": delta, "se": np.sqrt(se2), "tau": np.sqrt(tau2), "sigma_eval": sig_eval,
            "z": delta / tot if tot > 0 else np.nan,
            "f": np.sqrt(max(delta ** 2 - se2, 0.0)) / sig_eval if sig_eval > 0 else np.nan,
            "Q": Q, "frac_pos": float(np.mean(d > 0)), "top_exp_share": float(w.max() / w.sum()),
        })
    return pd.DataFrame(rows)


def aggregate_chi2_by_library(
    per_exp: pd.DataFrame,
    lib_labels: Optional[dict] = None,
    *,
    library_col: str = "library",
) -> pd.DataFrame:
    """Sum chi^2 and N across experiments within each library.

    Returns columns: Library (display label), library_key, N, chi2, chi2_per_N.
    """
    grp = per_exp.groupby(library_col, observed=True).agg(
        N=("N", "sum"), chi2=("chi2", "sum")
    ).reset_index()
    grp["chi2_per_N"] = np.where(grp["N"] > 0, grp["chi2"] / grp["N"], np.nan)
    grp = grp.rename(columns={library_col: "library_key"})
    grp["Library"] = (
        grp["library_key"].map(lib_labels) if lib_labels is not None else grp["library_key"]
    )
    return grp[["Library", "library_key", "N", "chi2", "chi2_per_N"]]


def warn_if_not_psd(
    D: np.ndarray, u: np.ndarray, eval_cov_block: Optional[np.ndarray] = None,
    label: str = "",
) -> None:
    """Warn-only PSD diagnostic for one experiment block. Never auto-repairs."""
    bad_D = (~np.isfinite(D)) | (D <= 0)
    bad_u = ~np.isfinite(u)
    n_bad_D = int(bad_D.sum())
    n_bad_u = int(bad_u.sum())
    bad_eval = False
    if eval_cov_block is not None and eval_cov_block.size:
        eig = np.linalg.eigvalsh(0.5 * (eval_cov_block + eval_cov_block.T))
        bad_eval = bool(eig.min() < -1e-10 * max(abs(eig.max()), 1.0))
    if n_bad_D or n_bad_u or bad_eval:
        prefix = f"[{label}] " if label else ""
        warnings.warn(
            f"{prefix}covariance not PSD: "
            f"{n_bad_D} non-positive D, {n_bad_u} non-finite u, "
            f"{'eval_cov has negative eigenvalue' if bad_eval else ''}. "
            "No auto-repair.",
            RuntimeWarning,
            stacklevel=2,
        )
