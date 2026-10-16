"""Data modes: the doubt about a correction applied to EXFOR points, as covariance (16-oct-2026).

The corrected chi2 arms (KIKA_EXFOR_CORRECTIONS, `precompute_chi2_predictive.py`) divide Kinney and
Cierjacks by per-detector efficiencies as if exact. An evaluation whose covariance declares the doubt
about those corrections is then scored against data that deny it, and the doubt looks superfluous by
construction (kika-workspace docs/chi2-mf4/splines/revision_metrica_2026-10-16.md §3). This module
puts the doubt back on the DATA side: each mode is a fully correlated rank-1 term of the experiment's
covariance, a column ``dmode_<name>`` that `chi2_metrics._data_modes` picks up.

Spec (CSV; kika-workspace myworkspace/chi2/bspline_window/w18c_data_modes.py writes it), one line per
support piece of a mode:

    mode, exp_key, angle_deg, e_lo, e_hi, rel, abs

A row of experiment ``exp_key`` whose CM angle (from ``mu``) is within ANGLE_TOL of ``angle_deg`` and
whose nominal ``energy_mev`` is in [e_lo, e_hi) gets ``rel * y_eval + abs`` in column ``dmode_<mode>``.
``rel`` multiplies y_eval, as the metric's own u and v do (Peelle-safe); ``abs`` is b/sr. Lines of the
same mode add up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANGLE_TOL = 1.5          # deg, = CORR_ANGLE_TOL of precompute_chi2_predictive.py
PREFIX = "dmode_"


def add_mode_columns(df: pd.DataFrame, spec_path: str) -> pd.DataFrame:
    """Return ``df`` with one ``dmode_<mode>`` column per mode of the spec (in place, and returned)."""
    spec = pd.read_csv(spec_path, dtype={"exp_key": str})
    need = {"mode", "exp_key", "angle_deg", "e_lo", "e_hi", "rel", "abs"}
    if not need <= set(spec.columns):
        raise SystemExit(f"{spec_path}: expected columns {sorted(need)}, found {list(spec.columns)}")
    exp = df["experiment_id"].astype(str).to_numpy()
    ang = np.degrees(np.arccos(np.clip(df["mu"].to_numpy(dtype=float), -1.0, 1.0)))
    e = df["energy_mev"].to_numpy(dtype=float)
    y_eval = df["y_eval"].to_numpy(dtype=float)
    for mode, lines in spec.groupby("mode", sort=True):
        col = np.zeros(len(df))
        hit = np.zeros(len(df), dtype=bool)
        for t in lines.itertuples(index=False):
            m = (exp == str(t.exp_key).strip()) & (np.abs(ang - float(t.angle_deg)) <= ANGLE_TOL) \
                & (e >= float(t.e_lo)) & (e < float(t.e_hi))
            if np.any(hit & m):
                raise SystemExit(f"{spec_path}: mode {mode!r} has overlapping lines")
            hit |= m
            col[m] = float(t.rel) * y_eval[m] + float(t.abs)
        df[PREFIX + str(mode)] = col
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per mode: rows touched and the rms of the mode relative to y_eval on them."""
    out = []
    y = df["y_eval"].to_numpy(dtype=float)
    for c in [c for c in df.columns if str(c).startswith(PREFIX)]:
        m = df[c].to_numpy(dtype=float)
        nz = m != 0
        rel = m[nz] / np.where(y[nz] != 0, y[nz], np.nan)
        out.append({"mode": c[len(PREFIX):], "rows": int(nz.sum()),
                    "rms_rel": float(np.sqrt(np.nanmean(rel ** 2))) if nz.any() else 0.0})
    return pd.DataFrame(out)
