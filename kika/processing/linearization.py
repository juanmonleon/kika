"""
Adaptive energy grid generation (broken-stick linearization).

Generates an energy mesh where cross sections can be linearly interpolated
within a specified tolerance.
"""

import numpy as np
from typing import Callable, Optional


def linearize(sigma_func: Callable[[np.ndarray], np.ndarray],
              E_lo: float,
              E_hi: float,
              tol: float = 1e-3,
              initial_points: Optional[np.ndarray] = None,
              max_points: int = 500_000,
              min_spacing: float = 1e-6,
              errint: Optional[float] = None) -> np.ndarray:
    """Adaptive broken-stick linearization of sigma(E).

    Uses an iterative pass-based approach: on each pass, evaluate midpoints
    of all current segments in a single vectorized call, then insert those
    that exceed tolerance.  This is much faster than point-by-point evaluation.

    Parameters
    ----------
    sigma_func : callable
        Function E_array -> sigma_array (barns). Must accept 1-D numpy arrays.
    E_lo, E_hi : float
        Energy range (eV).
    tol : float
        Relative tolerance for linearization (default 0.1%).
    initial_points : array, optional
        Initial grid points to seed the algorithm (e.g. resonance energies).
    max_points : int
        Safety limit on total grid size.
    min_spacing : float
        Minimum spacing between adjacent points (eV).
    errint : float, optional
        Resonance integral tolerance.  After the main linearization loop,
        a Gauss-Legendre quadrature check catches narrow resonances that
        fall entirely between grid points.  Default: ``tol / 20000``.
        Set to 0 to disable.

    Returns
    -------
    ndarray
        Sorted energy grid where sigma is adequately linearized.
    """
    # Build initial grid
    pts = [E_lo, E_hi]
    if initial_points is not None:
        for p in initial_points:
            if E_lo < p < E_hi:
                pts.append(float(p))

    grid = np.array(sorted(set(pts)))
    sigma_vals = sigma_func(grid)

    # Iterative refinement: each pass checks all midpoints at once
    max_passes = 50
    for _ in range(max_passes):
        if len(grid) >= max_points:
            break

        # Compute all midpoints
        midpoints = 0.5 * (grid[:-1] + grid[1:])
        spacings = grid[1:] - grid[:-1]

        # Skip segments that are too narrow
        wide_enough = spacings >= min_spacing
        if not np.any(wide_enough):
            break

        # Evaluate sigma at all midpoints (single vectorized call)
        sigma_mid_exact = sigma_func(midpoints)

        # Linear interpolation estimates
        sigma_mid_linear = 0.5 * (sigma_vals[:-1] + sigma_vals[1:])

        # Relative error
        ref = np.maximum(np.abs(sigma_mid_exact), np.abs(sigma_mid_linear))
        ref = np.maximum(ref, 1e-30)
        rel_err = np.abs(sigma_mid_exact - sigma_mid_linear) / ref

        # Find segments that need refinement
        needs_refine = (rel_err > tol) & wide_enough

        if not np.any(needs_refine):
            break

        # Insert midpoints where needed — merge into existing grid
        insert_E = midpoints[needs_refine]
        insert_sigma = sigma_mid_exact[needs_refine]

        # Merge via sorted insertion (preserves cached sigma values)
        idx = np.searchsorted(grid, insert_E)
        grid = np.insert(grid, idx, insert_E)
        sigma_vals = np.insert(sigma_vals, idx, insert_sigma)

    # --- Resonance integral verification pass ---
    # Catches narrow resonances that fall entirely between grid points.
    # Compare trapezoidal integral vs 3-point Gauss-Legendre per segment.
    if errint is None:
        errint = tol / 20000.0  # NJOY default

    if errint > 0 and len(grid) > 1:
        gl_nodes = np.array([0.5 - np.sqrt(3.0 / 5.0) / 2.0,
                             0.5,
                             0.5 + np.sqrt(3.0 / 5.0) / 2.0])
        gl_weights = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])

        dE = grid[1:] - grid[:-1]
        trap_integral = 0.5 * (sigma_vals[:-1] + sigma_vals[1:]) * dE

        # Evaluate sigma at GL nodes for each segment
        gl_E = grid[:-1, None] + gl_nodes[None, :] * dE[:, None]  # (n_seg, 3)
        gl_sigma = np.array([sigma_func(gl_E[:, j]) for j in range(3)]).T
        gl_integral = np.sum(gl_sigma * gl_weights[None, :], axis=1) * dE

        int_err = np.abs(gl_integral - trap_integral)
        needs_split = int_err > errint * dE

        if np.any(needs_split):
            split_mids = 0.5 * (grid[:-1][needs_split] + grid[1:][needs_split])
            split_sigma = sigma_func(split_mids)
            idx = np.searchsorted(grid, split_mids)
            grid = np.insert(grid, idx, split_mids)
            sigma_vals = np.insert(sigma_vals, idx, split_sigma)

    return grid


#: Relative tolerance to which :func:`linearize_table` reproduces a non-linear
#: law with lin-lin panels. 1e-4 sits between FUDGE's two settings -- 1e-3 as
#: the ``XYs1d`` default, 1e-5 for its own processing (``heat``,
#: ``processMultiGroup``) -- and an order of magnitude above the round-off of the
#: six-significant-digit ENDF float, so a table that is written back out is not
#: refined below what the format can carry.
TABLE_LINEARIZATION_TOLERANCE = 1e-4

#: Fractions of a panel, in the law's own variable, at which the lin-lin chord
#: is tested. One midpoint is not enough: on a steep log-log panel the largest
#: deviation sits well off the middle, and a single test lets it through.
_PANEL_TEST_FRACTIONS = np.array([0.25, 0.5, 0.75])


def linearize_table(xs, ys, nbt_int_pairs, tol: float = TABLE_LINEARIZATION_TOLERANCE,
                    max_passes: int = 40, snap=None):
    """Re-express a table stated on any ENDF interpolation laws as lin-lin.

    Returns ``(xs, ys)`` such that lin-lin interpolation on them reproduces the
    original table's own laws to a relative *tol* everywhere. The original
    points are all kept, with their values untouched; points are only added.

    * **lin-lin** (INT=2) panels, and INT=6 (which
      :func:`~kika.processing.interpolation.interpolate_1d` also evaluates
      lin-lin), are copied as they are.
    * **histogram** (INT=1) panels get the left limit of their right end as a
      repeated abscissa, ``(x_{i+1}, y_i)``, which is the exact lin-lin form of a
      step -- ENDF's own way of writing a discontinuity.
    * **lin-log, log-lin, log-log** (INT=3, 4, 5) panels are bisected, pass by
      pass and all panels at once, until the chord agrees with the law at a
      quarter, a half and three quarters of the panel. Log-x laws split in
      ln x, log-y laws in x. A panel whose law
      :func:`~kika.processing.interpolation.interpolate_1d` itself evaluates
      lin-lin (a zero or negative y on a log-y law, a non-positive x on a log-x
      law) agrees with its chord and is left alone -- the table means what the
      interpolator says it means.

    A panel stops splitting at a width of ten ulps, or after *max_passes*.

    *snap*, when given, maps an array of abscissae to the nearest ones the
    output can actually hold -- the eleven-column ENDF float, say, which at
    1 MeV resolves 1 eV. Every candidate point is snapped *before* the law is
    evaluated at it, so the value stored is the law's value at the energy that
    will be written. Without it, a point added mid-resonance and then rounded
    on output carries a value from up to half a resolution step away: on JENDL-5
    Fe-56 MT2 that alone left a 3e-4 error against a 1e-4 tolerance. A panel
    whose candidates all snap onto its own ends cannot be split further.
    """
    from .interpolation import _base_int_code, _interp_pair_vec, interval_codes

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2:
        return x.copy(), y.copy()
    codes = np.array([_base_int_code(int(c)) for c in interval_codes(x.size, nbt_int_pairs)],
                     dtype=int)
    width = np.diff(x)

    # Each added point carries the interval it falls in, so the merge below
    # is a sort on (interval, x) and never has to decide how a new point
    # ranks against a repeated original abscissa.
    added_x, added_y, added_at = [], [], []

    step = np.flatnonzero((codes == 1) & (width > 0) & (y[:-1] != y[1:]))
    added_x.append(x[step + 1])
    added_y.append(y[step])
    added_at.append(step)

    active = np.flatnonzero(np.isin(codes, (3, 4, 5)) & (width > 0))
    x1, y1, x2, y2 = x[active], y[active], x[active + 1], y[active + 1]
    law, owner = codes[active], active
    for _ in range(max_passes):
        if owner.size == 0:
            break
        logx = (law != 4) & (x1 > 0) & (x2 > 0)
        frac = _PANEL_TEST_FRACTIONS[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            test = np.where(logx[:, None],
                            x1[:, None] * (x2 / np.where(logx, x1, 1.0))[:, None] ** frac,
                            x1[:, None] + frac * (x2 - x1)[:, None])
        if snap is not None:
            test = np.asarray(snap(test.ravel()), dtype=float).reshape(test.shape)
        usable = (test > x1[:, None]) & (test < x2[:, None])
        exact = np.empty_like(test)
        for code in np.unique(law):
            rows = law == code
            for j in range(test.shape[1]):
                exact[rows, j] = _interp_pair_vec(test[rows, j], x1[rows], y1[rows],
                                                  x2[rows], y2[rows], int(code))
        chord = y1[:, None] + (test - x1[:, None]) / (x2 - x1)[:, None] * (y2 - y1)[:, None]
        scale = np.maximum(np.maximum(np.abs(exact), np.abs(chord)), 1e-300)
        error = np.where(usable, np.abs(exact - chord) / scale, 0.0)
        worst = np.argmax(error, axis=1)
        rows = np.arange(owner.size)
        split = ((error[rows, worst] > tol)
                 & (x2 - x1 > 10 * np.finfo(float).eps * (np.abs(x1) + np.abs(x2))))
        if not split.any():
            break
        xm, ym = test[rows, worst][split], exact[rows, worst][split]
        added_x.append(xm)
        added_y.append(ym)
        added_at.append(owner[split])
        keep = split
        x1, y1, x2, y2 = (np.concatenate([x1[keep], xm]), np.concatenate([y1[keep], ym]),
                          np.concatenate([xm, x2[keep]]), np.concatenate([ym, y2[keep]]))
        law = np.concatenate([law[keep], law[keep]])
        owner = np.concatenate([owner[keep], owner[keep]])

    new_x = np.concatenate(added_x)
    if new_x.size == 0:
        return x.copy(), y.copy()
    # Original point j ranks 2j; anything added inside interval i ranks 2i+1.
    all_x = np.concatenate([x, new_x])
    all_y = np.concatenate([y, np.concatenate(added_y)])
    rank = np.concatenate([2 * np.arange(x.size), 2 * np.concatenate(added_at) + 1])
    order = np.lexsort((all_x, rank))
    return all_x[order], all_y[order]
