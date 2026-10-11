"""fastTMC, the normality tests and the sandwich module's logging.

Defects found while documenting kika.UQ (11-oct-2026): a summary table that
failed when a tally had mean 0, relative uncertainties with the sign of the
mean, an Anderson-Darling "p-value" that was a critical value, and a
``logging.basicConfig`` run at import.
"""
import logging
import subprocess
import sys

import numpy as np
import pytest

from kika.UQ.fastTMC import create_summary_table, fastTMC
from kika.UQ.normality import anderson_darling_normal_p_value, normality_tests


def _samples(mean, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return mean + 0.05 * abs(mean) * rng.standard_normal(n), np.full(n, 0.01)


def test_a_negative_response_has_a_positive_relative_uncertainty():
    y, sig = _samples(-2.0)
    result = fastTMC(y_val=y, y_sigma=sig, n_bootstrap=50, random_seed=1, verbose=False)
    assert result["percent_unc_observed"] > 0 and result["percent_unc_nuclear_data"] > 0
    assert result["is_reliable"]
    lo, hi = result["ci"]["percent_unc_observed"]
    assert 0 < lo < hi
    positive = fastTMC(y_val=-y, y_sigma=sig, n_bootstrap=50, random_seed=1, verbose=False)
    assert result["percent_unc_observed"] == pytest.approx(positive["percent_unc_observed"])


def test_the_summary_table_skips_a_zero_mean_without_failing():
    good = fastTMC(*_samples(1.0), n_bootstrap=20, random_seed=1, verbose=False)
    zero = dict(good, mean_tally=0.0)
    table = create_summary_table({"keff": good, "empty": zero})
    assert list(table.index) == ["keff"]


@pytest.mark.parametrize("critical,level", [(0.631, 0.10), (0.752, 0.05), (1.035, 0.01)])
def test_the_anderson_darling_p_value_meets_the_tabulated_critical_values(critical, level):
    assert anderson_darling_normal_p_value(critical, 10 ** 6) == pytest.approx(level, abs=2e-3)


def test_the_anderson_darling_decision_follows_alpha():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(400)
    ad = {a: next(r for r in normality_tests(y, alpha=a) if r["test_name"] == "Anderson-Darling")
          for a in (0.01, 0.99)}
    p = ad[0.01]["p_value"]
    assert 0.0 <= p <= 1.0 and p == ad[0.99]["p_value"]
    assert ad[0.01]["is_normal"] == (p > 0.01) and ad[0.99]["is_normal"] == (p > 0.99)


def test_importing_the_sandwich_leaves_the_root_logger_alone():
    code = ("import logging; import kika.UQ.sandwich; "
            "r = logging.getLogger(); print(len(r.handlers), r.level)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["0", str(logging.WARNING)]
