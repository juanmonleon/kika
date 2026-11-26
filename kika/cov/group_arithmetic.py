"""Arithmetic that turns the components of one covariance block into one matrix.

Format-free: ENDF's NI sub-subsections and GNDS's ``mixed`` components both reach
here as arrays on energy grids, so a block gives the same matrix whichever format
stated it.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "collapse_form",
    "form_states_sum",
    "CovarianceSumNotResolved",
    "table_bin_average",
    "bin_average_xs",
    "project_piecewise_constant",
    "short_range_variance",
    "sum_block_components",
    "MixedKindsNeedCrossSections",
]


class MixedKindsNeedCrossSections(ValueError):
    """A block sums absolute and relative components and no σ(E) was given."""


def bin_average_xs(energies, values, edges, laws=2) -> np.ndarray:
    """Lethargy-flux (1/E) group average of a table over each bin, exactly.

    The group cross section MF33 is a covariance of: ``int σ/E dE / int dE/E``
    over the whole bin, with σ zero outside its own table. A bin whose lower
    edge is not positive has no 1/E weight and gets the flat average; a bin of
    zero width gets zero. *laws* is one ENDF code or one per interval.
    """
    from kika.algebra import group_averages

    e = np.asarray(energies, dtype=float)
    v = np.asarray(values, dtype=float)
    g = np.asarray(edges, dtype=float)
    out = np.zeros(max(g.size - 1, 0))
    if e.size < 2 or g.size < 2:
        return out
    positive = g[:-1] > 0
    k = int(np.argmax(positive)) if positive.any() else g.size - 1
    if k > 0:
        out[:k] = group_averages(e, v, laws, g[:k + 1])
    if positive.any():
        out[k:] = group_averages(e, v, laws, g[k:], "1/x")
    return np.nan_to_num(out, nan=0.0)


def table_bin_average(xs_source, energy_grid) -> np.ndarray:
    """:func:`bin_average_xs` of an ``MF3MT``, a ``CrossSection``, an ``(E, σ)`` pair or an ``energies``/``values`` table.

    The source's own interpolation laws are used when it states them; a bare
    table or pair is read lin-lin.
    """
    from kika.algebra import interval_laws

    if isinstance(xs_source, tuple):
        energies, values = xs_source
        return bin_average_xs(energies, values, energy_grid, 2)
    energies = np.asarray(xs_source.energies, dtype=float)
    values = getattr(xs_source, 'values', None)
    if values is None:
        values = getattr(xs_source, 'cross_sections')
    if hasattr(xs_source, 'energy_interpolation'):
        laws = interval_laws(energies.size, list(xs_source.energy_interpolation))
    elif hasattr(xs_source, 'interval_laws'):
        laws = xs_source.interval_laws()
    else:
        laws = 2
    return bin_average_xs(energies, values, energy_grid, laws)


def project_piecewise_constant(
    component_matrix: np.ndarray,
    native_row_grid: Sequence[float],
    union_grid: Sequence[float],
    native_col_grid: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Project an interval matrix onto a union grid using piecewise constant assumption."""
    union_arr = np.asarray(union_grid, dtype=float)
    row_arr = np.asarray(native_row_grid, dtype=float)
    col_arr = np.asarray(native_col_grid, dtype=float) if native_col_grid is not None else row_arr

    u_lo = union_arr[:-1, None]
    u_hi = union_arr[1:, None]
    delta_u = u_hi - u_lo

    r_lo = row_arr[:-1][None, :]
    r_hi = row_arr[1:][None, :]
    overlap = np.maximum(0.0, np.minimum(u_hi, r_hi) - np.maximum(u_lo, r_lo))
    with np.errstate(divide='ignore', invalid='ignore'):
        T_row = np.where(delta_u > 0, overlap / delta_u, 0.0)

    c_lo = col_arr[:-1][None, :]
    c_hi = col_arr[1:][None, :]
    overlap_c = np.maximum(0.0, np.minimum(u_hi, c_hi) - np.maximum(u_lo, c_lo))
    with np.errstate(divide='ignore', invalid='ignore'):
        T_col = np.where(delta_u > 0, overlap_c / delta_u, 0.0)

    return T_row @ component_matrix @ T_col.T


def short_range_variance(energies: Sequence[float], f_values: Sequence[float],
                         output_grid: Sequence[float], lb: int) -> np.ndarray:
    """Variance of a short-range self-scaling term (ENDF LB=8/9) on ``output_grid``, absolute (b²).

    ENDF-6 §33.2: each F_k is an *uncorrelated* contribution, so the result
    is diagonal, and its size depends on the width of the group it lands in.
    For a group j inside the interval k:

    * LB=8: Var_jj = F_k · ΔE_k / ΔE_j  (grows as the group narrows);
    * LB=9: Var_jj = F_k · (1 − ΔE_j / ΔE_k)  (zero on the native grid).

    A group j that overlaps several intervals k averages them as
    uncorrelated pieces: with o_jk the overlap,
    LB=8 → Σ_k o_jk·F_k·ΔE_k / ΔE_j², LB=9 → Σ_k (o_jk/ΔE_j)²·F_k·(1 − o_jk/ΔE_k).
    Both reduce to the manual's formulas when j ⊂ k, and LB=8 gives F_k
    when j = k. Projecting LB=8 the way LB=0 is projected (the same value in
    every sub-bin, fully correlated) is right for LB=0 and wrong here.
    """
    energies = np.asarray(energies, dtype=float)
    if energies.size < 2:
        raise ValueError(f"LB={lb} requires NE >= 2, got {energies.size}")
    f_values = np.asarray(f_values[:energies.size - 1], dtype=float)
    grid = np.asarray(output_grid, dtype=float)

    if lb == 8 and grid.size == energies.size and np.array_equal(grid, energies):
        return f_values.copy()

    d_k = np.diff(energies)
    d_j = np.diff(grid)
    overlap = np.maximum(
        0.0,
        np.minimum(grid[1:, None], energies[None, 1:])
        - np.maximum(grid[:-1, None], energies[None, :-1]),
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        if lb == 8:
            var = (overlap * (f_values * d_k)[None, :]).sum(axis=1) / d_j ** 2
        else:
            frac_j = overlap / d_j[:, None]
            frac_k = np.where(d_k[None, :] > 0, overlap / d_k[None, :], 0.0)
            var = (frac_j ** 2 * f_values[None, :] * (1.0 - frac_k)).sum(axis=1)
    return np.where(d_j > 0, var, 0.0)


#: One matrix component: ``(matrix, row_grid, col_grid or None)``, grids as lists.
MatrixComponent = Tuple[np.ndarray, List[float], Optional[List[float]]]
#: One short-range term: ``(energies, F_k, lb)`` with lb 8 or 9.
ShortRangeComponent = Tuple[List[float], Sequence[float], int]


def sum_block_components(
    relative: List[MatrixComponent],
    absolute: List[MatrixComponent],
    short_range: List[ShortRangeComponent],
    output_grid: List[float],
    xs_row=None,
    xs_col=None,
    label: str = '',
    logger=None,
) -> Tuple[np.ndarray, List[float], bool]:
    """Sum one block's components on ``output_grid``: ``(matrix, grid, is_relative)``.

    Relative and absolute components are summed separately. A block that holds
    only one kind comes back in that kind. A block that holds both comes back
    relative, with the absolute part divided by σ̄_i·σ̄_j, the 1/E bin averages
    of ``xs_row`` / ``xs_col`` (anything :func:`table_bin_average` reads); without
    them it raises :class:`MixedKindsNeedCrossSections`. Short-range terms go
    on the diagonal through :func:`short_range_variance`.
    """
    m = len(output_grid) - 1

    def _summed(components):
        total = np.zeros((m, m))
        for comp_mat, row_grid, col_grid in components:
            if col_grid is None and row_grid == output_grid:
                projected = comp_mat
            elif col_grid is not None and row_grid == output_grid and col_grid == output_grid:
                projected = comp_mat
            else:
                projected = project_piecewise_constant(
                    comp_mat, row_grid, output_grid, native_col_grid=col_grid
                )
            total += projected
        return total

    rel_total = _summed(relative)
    if not (absolute or short_range):
        return rel_total, output_grid, True

    abs_total = _summed(absolute)
    for energies, f_values, lb in short_range:
        abs_total[np.diag_indices(m)] += short_range_variance(energies, f_values, output_grid, lb)
    if not relative:
        return abs_total, output_grid, False

    if xs_row is None or xs_col is None:
        raise MixedKindsNeedCrossSections(
            f"covariance block {label}: it sums absolute (barns²) and relative "
            f"components, which can only be added after dividing the absolute part "
            f"by σ_i·σ_j, and below the resolved-resonance upper limit σ(E) has to "
            f"be reconstructed. Pass σ(E) of the row and column reactions."
        )
    s_row = table_bin_average(xs_row, output_grid)
    s_col = s_row if xs_col is xs_row else table_bin_average(xs_col, output_grid)
    denom = np.outer(s_row, s_col)
    lost = (denom <= 0) & (abs_total != 0)
    if lost.any() and logger is not None:
        logger.warning(
            f"covariance block {label}: {int(lost.sum())} absolute covariance entries "
            f"sit where σ̄ = 0 and cannot be made relative; dropped."
        )
    with np.errstate(divide='ignore', invalid='ignore'):
        converted = np.where(denom > 0, abs_total / np.where(denom > 0, denom, 1.0), 0.0)
    return rel_total + converted, output_grid, True


class CovarianceSumNotResolved(ValueError):
    """A covariance form states a sum of other sections (ENDF NC LTY=0), which this collapse does not resolve."""


def _is_sum(form) -> bool:
    return getattr(form, "summands", None) is not None


def form_states_sum(form) -> bool:
    """Whether a §25 form is, or holds, a ``sum`` of other sections."""
    if _is_sum(form):
        return True
    return any(_is_sum(c) for c in (getattr(form, "components", None) or ()))


def collapse_form(form, xs_row=None, xs_col=None, label: str = '', logger=None
                  ) -> Tuple[np.ndarray, List[float], bool]:
    """A §25 covariance form as one matrix: ``(matrix, grid, is_relative)``.

    A ``covarianceMatrix`` comes back as it is; a ``mixed`` of matrices and
    short-range self-scaling variances is summed on the union of its components'
    grids by :func:`sum_block_components`, so a GNDS ``mixed`` gives the matrix
    ENDF's NI sub-subsections give. *xs_row* / *xs_col* (σ(E) of the row and
    column reactions) are read only when absolute and relative components meet.
    A form holding a ``sum`` raises :class:`CovarianceSumNotResolved`.
    """
    if form_states_sum(form):
        raise CovarianceSumNotResolved(
            f"covariance block {label} is stated as a sum of other sections "
            f"(ENDF NC LTY=0); it is not resolved into a matrix here")
    components = getattr(form, "components", None)
    components = list(components) if components is not None else [form]

    all_energies = set()
    relative, absolute, short_range = [], [], []
    for component in components:
        width = getattr(component, "dependenceOnProcessedGroupWidth", None)
        inner = getattr(component, "matrix", None)
        if hasattr(component, "dependenceOnProcessedGroupWidth"):
            if inner is None or inner.rowGrid is None:
                raise ValueError(f"covariance block {label}: a short-range variance with no matrix or grid")
            grid = [float(v) for v in inner.rowGrid]
            values = list(np.diagonal(np.asarray(inner.matrix, dtype=float))) + [0.0]
            all_energies.update(grid)
            short_range.append((grid, values, 9 if (width or "") == "direct" else 8))
            continue
        if inner is None or getattr(component, "rowGrid", None) is None:
            raise ValueError(
                f"covariance block {label}: a component of type "
                f"{type(component).__name__} states no matrix on a grid")
        matrix = np.asarray(inner, dtype=float)
        rows = [float(v) for v in component.rowGrid]
        cols = ([float(v) for v in component.columnGrid]
                if getattr(component, "columnGrid", None) is not None else None)
        if cols == rows:
            cols = None
        all_energies.update(rows)
        if cols is not None:
            all_energies.update(cols)
        (relative if component.isRelative else absolute).append((matrix, rows, cols))

    output_grid = sorted(all_energies)
    if len(output_grid) < 2:
        raise ValueError(f"covariance block {label}: fewer than two grid boundaries")
    return sum_block_components(relative, absolute, short_range, output_grid,
                                xs_row=xs_row, xs_col=xs_col, label=label, logger=logger)
