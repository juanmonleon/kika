"""Covariance of a reaction stated as a sum of others (ENDF NC LTY=0, GNDS ``sum``), format-free.

The tape path (``MF33MT.resolve_nc_lty0``) and the model path
(``kika.sampling.mf33_sampling.per_mt_covariances_from_suite``) both call
:func:`sum_rule_covariance`, handing it the contributors' blocks through two
callables, so a sum gives the same matrix whichever format stated it.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .group_arithmetic import MixedKindsNeedCrossSections, project_piecewise_constant, table_bin_average

__all__ = ["sum_rule_covariance", "sum_rule_cross_covariance", "combine_sum_and_explicit"]

#: ``(matrix, grid, is_relative)``, the shape every block travels in here.
Bundle = Tuple[np.ndarray, List[float], bool]


def sum_rule_covariance(
    derived_mt: int,
    terms: Sequence[Tuple[float, int]],
    self_block: Callable[[int, Set[int]], Optional[Bundle]],
    cross_blocks: Callable[[Set[int]], Mapping[Tuple[int, int], Bundle]],
    xs: Optional[Mapping[int, object]] = None,
    resolving: Optional[Set[int]] = None,
    logger=None,
) -> Optional[Bundle]:
    """Self-covariance of ``σ_d = Σ_i c_i σ_i`` by the bilinear rule, relative, on the contributors' union grid.

    ``Cov_abs(d, d) = Σ_i Σ_j c_i c_j Cov_abs(i, j)``, the contraction NJOY
    ERRORR's ``akxy`` does, on the data's own grids. *terms* are ``(c_i, MT_i)``.
    *self_block(mt, resolving)* gives a contributor's self-covariance (``None``
    when it has none), *cross_blocks(mts)* every stored block between two of
    them keyed ``(row MT, column MT)``; a missing cross block counts as zero.
    *xs* maps MT to σ(E) for the relative↔absolute conversions; without it the
    relative matrices are summed directly, with a warning. Returns ``None``
    when no contributor has a covariance.
    """
    def _log(level, message):
        if logger is not None:
            getattr(logger, level)(message)

    mti_list = [int(mt) for _, mt in terms]
    ci_list = [float(c) for c, _ in terms]
    nci = len(mti_list)
    if nci == 0:
        _log("warning", f"MT{derived_mt}: NC LTY=0 has no contributing reactions")
        return None
    _log("info", f"MT{derived_mt}: resolving NC LTY=0 from {nci} reactions: "
                 f"{list(zip(ci_list, mti_list))}")

    resolving = (resolving or set()) | {derived_mt}

    cov_by_mt: Dict[int, Bundle] = {}
    for mt_i in mti_list:
        if mt_i in resolving:
            _log("warning", f"MT{derived_mt}: skipping MT{mt_i} (circular NC reference)")
            continue
        try:
            bundle = self_block(mt_i, resolving)
        except MixedKindsNeedCrossSections:
            raise
        except Exception as e:
            _log("warning", f"MT{derived_mt}: failed to resolve MT{mt_i}: {e}")
            continue
        if bundle is None:
            continue
        mat, grid, is_rel = bundle
        mat = np.asarray(mat, dtype=float)
        if (grid is not None and mat.ndim == 2
                and mat.shape[0] == mat.shape[1]
                and mat.shape[0] == len(grid) - 1):
            cov_by_mt[mt_i] = (mat, list(grid), bool(is_rel))

    if not cov_by_mt:
        _log("warning", f"MT{derived_mt}: no contributing MT covariances found")
        return None

    all_energies: Set[float] = set()
    for _, grid, _ in cov_by_mt.values():
        all_energies.update(grid)
    union_grid = sorted(all_energies)
    if len(union_grid) < 2:
        return None
    m = len(union_grid) - 1

    proj_cov: Dict[int, np.ndarray] = {}
    is_rel_by_mt: Dict[int, bool] = {}
    for mt_i, (mat, grid, is_rel) in cov_by_mt.items():
        is_rel_by_mt[mt_i] = is_rel
        proj_cov[mt_i] = mat if grid == union_grid else project_piecewise_constant(mat, grid, union_grid)

    have_xs = xs is not None
    any_relative = any(is_rel_by_mt.get(mt, True) for mt in mti_list if mt in proj_cov)
    if any_relative and not have_xs:
        _log("warning",
             f"MT{derived_mt}: NC LTY=0 sum rule requires cross-section data "
             f"(mf3_sections) to convert relative covariances to absolute. "
             f"Falling back to direct summation of relative covariances — "
             f"result will be approximate.")

    xs_bin: Dict[int, np.ndarray] = {}
    xs_d: Optional[np.ndarray] = None
    if have_xs:
        for mt_i in mti_list:
            if mt_i in xs and mt_i in proj_cov:
                try:
                    xs_bin[mt_i] = table_bin_average(xs[mt_i], union_grid)
                except Exception as e:
                    _log("warning", f"MT{derived_mt}: bin-average σ failed for MT{mt_i}: {e}")
        if derived_mt in xs:
            try:
                xs_d = table_bin_average(xs[derived_mt], union_grid)
            except Exception as e:
                _log("warning", f"MT{derived_mt}: bin-average σ_derived failed: {e}")
                xs_d = None
        if xs_d is None:
            xs_d = np.zeros(m)
            for ii, mt_i in enumerate(mti_list):
                if mt_i in xs_bin:
                    xs_d = xs_d + ci_list[ii] * xs_bin[mt_i]

    cross_index = cross_blocks(set(mti_list))
    cross_cov: Dict[Tuple[int, int], Tuple[np.ndarray, bool]] = {}
    for ii in range(nci):
        for jj in range(ii + 1, nci):
            mt_i, mt_j = mti_list[ii], mti_list[jj]
            r = cross_index.get((mt_i, mt_j))
            if r is None:
                r = cross_index.get((mt_j, mt_i))
            if r is None:
                continue
            cmat, cgrid, cis_rel = r
            proj_cross = cmat if cgrid == union_grid else project_piecewise_constant(cmat, cgrid, union_grid)
            cross_cov[(mt_i, mt_j)] = (proj_cross, bool(cis_rel))

    cov_abs = np.zeros((m, m))
    for ii, mt_i in enumerate(mti_list):
        c_i = ci_list[ii]
        for jj, mt_j in enumerate(mti_list):
            c_j = ci_list[jj]
            if ii == jj:
                pair = proj_cov.get(mt_i)
                if pair is None:
                    continue
                if is_rel_by_mt.get(mt_i, True) and have_xs and mt_i in xs_bin:
                    pair_abs = pair * np.outer(xs_bin[mt_i], xs_bin[mt_i])
                else:
                    pair_abs = pair
            else:
                entry = cross_cov.get((mt_i, mt_j))
                transposed = False
                if entry is None:
                    entry = cross_cov.get((mt_j, mt_i))
                    transposed = entry is not None
                if entry is None:
                    continue
                pair, pair_is_rel = entry
                if transposed:
                    pair = pair.T
                if pair_is_rel and have_xs and mt_i in xs_bin and mt_j in xs_bin:
                    pair_abs = pair * np.outer(xs_bin[mt_i], xs_bin[mt_j])
                else:
                    pair_abs = pair
            cov_abs = cov_abs + c_i * c_j * pair_abs

    if have_xs and xs_d is not None:
        # A floor on σ̄_d: the union grid inherits foreign threshold edges, where
        # σ̄_d collapses to ~1e-7 σ̄_max and 1/σ̄ blows the placeholder variance up.
        xs_max = float(np.max(xs_d)) if xs_d.size else 0.0
        sigma_floor = max(1.0e-3 * xs_max, 1.0e-9)
        xs_safe = np.where(xs_d > sigma_floor, xs_d, 0.0)
        denom = np.outer(xs_safe, xs_safe)
        with np.errstate(divide='ignore', invalid='ignore'):
            final = np.where(denom > 0, cov_abs / np.where(denom > 0, denom, 1.0), 0.0)
        diag = np.diag(final).copy()
        np.fill_diagonal(final, np.maximum(diag, 0.0))
    else:
        final = cov_abs

    final = 0.5 * (final + final.T)

    diag = np.diag(final)
    _log("info", f"MT{derived_mt}: NC LTY=0 resolved — {m} bins, sqrt(diag) range "
                 f"[{np.sqrt(float(max(diag.min(), 0.0))):.4g}, {np.sqrt(float(diag.max())):.4g}]")
    return final, list(union_grid), True


def sum_rule_cross_covariance(
    derived_mt: int,
    terms: Sequence[Tuple[float, int]],
    other_mt: int,
    block: Callable[[int, int], Optional[Bundle]],
    xs: Optional[Mapping[int, object]] = None,
    logger=None,
) -> Optional[Bundle]:
    """Cross block ``Cov(d, MT1)`` of ``σ_d = Σ_i c_i σ_i`` with another reaction, relative, on the blocks' union grid.

    ``Cov_abs(d, MT1) = Σ_i c_i Cov_abs(MT_i, MT1)`` follows from the linear
    relation of ENDF-6 §33.2.2.1. §33.3.2 (a.3) allows an NC LTY=0 only in a
    self block, so a cross block stated that way is off-format and read with
    this rule. *block(a, b)* gives ``Cov(a, b)``
    (rows a, columns b) on one grid, ``None`` when the file states none, which
    counts as zero. σ(E), the σ̄ floor and the fallback without *xs* are those
    of :func:`sum_rule_covariance`.
    """
    def _log(level, message):
        if logger is not None:
            getattr(logger, level)(message)

    pieces = []
    for coefficient, mt_i in terms:
        bundle = block(int(mt_i), int(other_mt))
        if bundle is None:
            continue
        mat, grid, is_rel = bundle
        mat = np.asarray(mat, dtype=float)
        if grid is None or mat.ndim != 2 or mat.shape != (len(grid) - 1, len(grid) - 1):
            continue
        pieces.append((float(coefficient), int(mt_i), mat, list(grid), bool(is_rel)))
    if not pieces:
        _log("warning", f"MT{derived_mt}→MT{other_mt}: no contributor has a block with MT{other_mt}")
        return None

    union_grid = sorted({e for *_, grid, _ in pieces for e in grid})
    if len(union_grid) < 2:
        return None
    m = len(union_grid) - 1

    have_xs = xs is not None
    bins: Dict[int, np.ndarray] = {}
    if have_xs:
        for mt in {p[1] for p in pieces} | {int(other_mt)}:
            if mt in xs:
                bins[mt] = table_bin_average(xs[mt], union_grid)
        if derived_mt in xs:
            xs_d = table_bin_average(xs[derived_mt], union_grid)
        else:
            xs_d = np.zeros(m)
            for c, mt_i, *_ in pieces:
                if mt_i in bins:
                    xs_d = xs_d + c * bins[mt_i]
    elif any(p[4] for p in pieces):
        _log("warning", f"MT{derived_mt}→MT{other_mt}: NC LTY=0 cross block summed from relative "
                        f"covariances without σ(E); the result is approximate.")

    cov_abs = np.zeros((m, m))
    for c, mt_i, mat, grid, is_rel in pieces:
        proj = mat if grid == union_grid else project_piecewise_constant(mat, grid, union_grid)
        if is_rel and have_xs and mt_i in bins and int(other_mt) in bins:
            proj = proj * np.outer(bins[mt_i], bins[int(other_mt)])
        cov_abs = cov_abs + c * proj

    if have_xs and int(other_mt) in bins:
        def safe(s):
            floor = max(1.0e-3 * (float(np.max(s)) if s.size else 0.0), 1.0e-9)
            return np.where(s > floor, s, 0.0)
        denom = np.outer(safe(xs_d), safe(bins[int(other_mt)]))
        with np.errstate(divide='ignore', invalid='ignore'):
            final = np.where(denom > 0, cov_abs / np.where(denom > 0, denom, 1.0), 0.0)
    else:
        final = cov_abs
    return final, list(union_grid), True


def combine_sum_and_explicit(sum_bundle: Optional[Bundle], explicit: Optional[Bundle],
                             label: str = '', logger=None) -> Optional[Bundle]:
    """A block's resolved sum plus its explicit components (ENDF-6 §33.3.3 item 3), on their union grid.

    The result takes the sum's relative-ness; a non-zero explicit part of the
    other kind is added anyway, with a warning, as the tape path always has.
    """
    if sum_bundle is not None and explicit is not None:
        nc_matrix, nc_grid, nc_is_rel = sum_bundle
        ni_mat, ni_grid, ni_is_rel = explicit
        ni_is_zero = not np.any(np.abs(ni_mat) > 0)
        if nc_is_rel != ni_is_rel and not ni_is_zero and logger is not None:
            logger.warning(
                f"{label}: NC result is {'relative' if nc_is_rel else 'absolute'} but NI "
                f"term is {'relative' if ni_is_rel else 'absolute'}; adding in NC's space "
                f"— NI noise term may be mis-scaled.")
        combined_grid = sorted(set(nc_grid or []) | set(ni_grid))
        if len(combined_grid) < 2:
            return None
        nc_proj = (nc_matrix if list(nc_grid) == combined_grid
                   else project_piecewise_constant(nc_matrix, nc_grid, combined_grid))
        ni_proj = (ni_mat if list(ni_grid) == combined_grid
                   else project_piecewise_constant(ni_mat, ni_grid, combined_grid))
        return nc_proj + ni_proj, combined_grid, bool(nc_is_rel)
    if sum_bundle is not None:
        nc_matrix, nc_grid, nc_is_rel = sum_bundle
        return nc_matrix, list(nc_grid) if nc_grid else [], bool(nc_is_rel)
    return explicit
