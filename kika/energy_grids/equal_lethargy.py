"""Equal-lethargy (log-spaced) energy-grid helper."""

from __future__ import annotations

import numpy as np


def equal_lethargy_grid(e_low: float, e_high: float, n_bins: int) -> np.ndarray:
    """Return ``n_bins + 1`` edges log-spaced between ``e_low`` and ``e_high``.

    A lethargy grid is uniform in ``u = ln(E)``; equivalently, adjacent edges
    have a constant ratio. This is the natural grid for averaging resonance-
    region cross sections with a 1/E flux.

    Parameters
    ----------
    e_low, e_high : float
        Positive bounds with ``e_high > e_low``. The function does not convert
        units: the edges are in the unit of the bounds.
    n_bins : int
        Number of bins (produces ``n_bins + 1`` edges).

    Returns
    -------
    np.ndarray
        Monotonically increasing edges from *e_low* to *e_high*, length
        ``n_bins + 1``.

    Raises
    ------
    ValueError
        If a bound is not positive, ``e_high <= e_low``, or ``n_bins < 1``.
    """
    if e_low <= 0 or e_high <= 0:
        raise ValueError("energy bounds must be positive")
    if e_high <= e_low:
        raise ValueError("e_high must be > e_low")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    return np.logspace(np.log10(e_low), np.log10(e_high), n_bins + 1)
