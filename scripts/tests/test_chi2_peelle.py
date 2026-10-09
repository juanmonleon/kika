"""Peelle's Pertinent Puzzle in the chi2 scoring (2026-10-09).

The textbook case of Neudecker, Frühwirth & Leeb, NSE 170 (2012) 54: two
measurements q = 2.40 ± 0.12 and 2.00 ± 0.10 sharing a normalization
N = 1.00 ± 0.15. The posterior mean is the stat-only weighted mean, 2.1639;
the GLS with the normalization mode built on the data gives 1.8857.
"""
import numpy as np
import pandas as pd

from scripts.chi2_metrics import _components
from scripts.precompute_chi2_exfor_c0 import fit_c0_from_ks

Q = np.array([2.40, 2.00])
S = np.array([0.12, 0.10])
SN = 0.15


def _ks_frame():
    # a_l = 0 below, so b = 1 and the model is y = c0: one quantity, two points.
    return pd.DataFrame({
        "mu": [0.3, -0.3],
        "value": Q,
        "sigma_stat": S,
        "sigma_sys": SN * Q,
        "sigma_sys_indep_rel": [SN, SN],
        "sigma_sys_dep_rel": [0.0, 0.0],
    })


def test_c0_fit_is_free_of_peelle():
    c0, _ = fit_c0_from_ks(_ks_frame(), np.zeros(3))
    np.testing.assert_allclose(c0, 2.1639, atol=5e-5)


def test_data_referenced_mode_reproduces_the_puzzle():
    # Guard on the reference value itself: the old construction gives 1.8857.
    C = np.diag(S ** 2) + SN ** 2 * np.outer(Q, Q)
    ci = np.linalg.inv(C)
    np.testing.assert_allclose(ci.sum(1) @ Q / ci.sum(), 1.8857, atol=5e-5)


def test_chi2_modes_follow_the_evaluation():
    df = pd.DataFrame({
        "library": "L", "experiment_id": "E",
        "y_exp": Q, "y_eval": [2.2, 2.2],
        "sigma_exp_stat": S,
        "sigma_sys_indep_rel": [SN, SN], "sigma_sys_dep_rel": [0.02, 0.02],
    })
    _, u, v, _ = _components(df)
    np.testing.assert_allclose(u, SN * 2.2)
    np.testing.assert_allclose(v, 0.02 * 2.2)
