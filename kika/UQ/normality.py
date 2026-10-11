"""
Normality testing and diagnostic data for Monte Carlo sample distributions.

Provides statistical tests and data for histogram / Q-Q plot generation.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Optional


def normality_tests(y_val: np.ndarray, alpha: float = 0.05) -> List[Dict]:
    """
    Run normality tests on sample data.

    Parameters
    ----------
    y_val : array-like
        Sample values.
    alpha : float
        Significance level for the is_normal decision (default 0.05).

    Returns
    -------
    list of dict
        Each dict has: test_name, statistic, p_value, is_normal.
        Tests included:
        - Shapiro-Wilk (N <= 5000, subsampled otherwise)
        - Anderson-Darling
        - D'Agostino-Pearson K² (N >= 20)
    """
    y_val = np.asarray(y_val, dtype=float)
    # Remove NaN/inf
    y_val = y_val[np.isfinite(y_val)]
    n = len(y_val)

    if n < 8:
        return []

    results = []

    # Shapiro-Wilk (limit to 5000 samples)
    if n <= 5000:
        sw_stat, sw_p = stats.shapiro(y_val)
    else:
        rng = np.random.default_rng(42)
        subsample = rng.choice(y_val, size=5000, replace=False)
        sw_stat, sw_p = stats.shapiro(subsample)
    results.append({
        "test_name": "Shapiro-Wilk",
        "statistic": float(sw_stat),
        "p_value": float(sw_p),
        "is_normal": bool(sw_p > alpha),
    })

    # Anderson-Darling. scipy gives the statistic and critical values only;
    # the p-value is D'Agostino & Stephens' approximation for a normal with
    # estimated mean and variance, and the decision uses alpha like the others.
    ad_stat = float(stats.anderson(y_val, dist="norm").statistic)
    ad_p = anderson_darling_normal_p_value(ad_stat, n)
    results.append({
        "test_name": "Anderson-Darling",
        "statistic": ad_stat,
        "p_value": ad_p,
        "is_normal": bool(ad_p > alpha),
    })

    # D'Agostino-Pearson K² (requires N >= 20)
    if n >= 20:
        try:
            k2_stat, k2_p = stats.normaltest(y_val)
            results.append({
                "test_name": "D'Agostino-Pearson",
                "statistic": float(k2_stat),
                "p_value": float(k2_p),
                "is_normal": bool(k2_p > alpha),
            })
        except Exception:
            pass

    return results


def anderson_darling_normal_p_value(statistic: float, n: int) -> float:
    """p-value of the Anderson-Darling test of normality, mean and variance estimated.

    R. B. D'Agostino and M. A. Stephens, *Goodness-of-Fit Techniques* (1986),
    Table 4.9: the statistic is corrected to ``A*² = A²(1 + 0.75/n + 2.25/n²)``
    and the p-value read from four exponential fits in ``A*²``.
    """
    a = float(statistic) * (1.0 + 0.75 / n + 2.25 / (n * n))
    if a >= 0.6:
        p = np.exp(1.2937 - 5.709 * a + 0.0186 * a * a)
    elif a >= 0.34:
        p = np.exp(0.9177 - 4.279 * a - 1.38 * a * a)
    elif a >= 0.2:
        p = 1.0 - np.exp(-8.318 + 42.796 * a - 59.938 * a * a)
    else:
        p = 1.0 - np.exp(-13.436 + 101.14 * a - 223.73 * a * a)
    return float(min(max(p, 0.0), 1.0))


def histogram_data(y_val: np.ndarray, n_bins: int = 30) -> Dict:
    """
    Compute histogram bins/counts and a normal PDF overlay.

    Parameters
    ----------
    y_val : array-like
        Sample values.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict with keys:
        bin_edges: list of float (n_bins + 1 values)
        bin_centers: list of float (n_bins values)
        counts: list of float (normalized density, n_bins values)
        normal_pdf: list of float (normal PDF at bin centers, n_bins values)
    """
    y_val = np.asarray(y_val, dtype=float)
    y_val = y_val[np.isfinite(y_val)]

    counts, bin_edges = np.histogram(y_val, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mu = np.mean(y_val)
    sigma = np.std(y_val)
    if sigma > 0:
        normal_pdf = stats.norm.pdf(bin_centers, mu, sigma)
    else:
        normal_pdf = np.zeros_like(bin_centers)

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "counts": counts.tolist(),
        "normal_pdf": normal_pdf.tolist(),
    }


def qq_plot_data(y_val: np.ndarray) -> Dict:
    """
    Compute Q-Q plot data (theoretical vs sample quantiles).

    Parameters
    ----------
    y_val : array-like
        Sample values.

    Returns
    -------
    dict with keys:
        theoretical: list of float (theoretical normal quantiles)
        sample: list of float (sorted sample quantiles)
    """
    y_val = np.asarray(y_val, dtype=float)
    y_val = y_val[np.isfinite(y_val)]
    y_sorted = np.sort(y_val)
    n = len(y_sorted)

    # Theoretical quantiles from standard normal
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(probabilities)

    # Scale sample to standard normal for comparison
    mu = np.mean(y_sorted)
    sigma = np.std(y_sorted)
    if sigma > 0:
        sample_standardized = (y_sorted - mu) / sigma
    else:
        sample_standardized = np.zeros(n)

    return {
        "theoretical": theoretical.tolist(),
        "sample": sample_standardized.tolist(),
    }
