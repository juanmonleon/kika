"""Layer 1 for MF35: the covariance of an energy distribution, band by band.

An MF35 band (ENDF-6 §35, always ``LS=1, LB=7``) is the absolute covariance of
the **group probabilities** ``P_i = ∫_{E'_i}^{E'_{i+1}} χ(E→E') dE'`` of the
spectrum of MF5, for the incident energies ``[E1, E2)`` of the band
(:mod:`kika.endf.classes.mf35.mf35` measures why it is P and not χ). Two
consequences make MF35 checkable beyond what MF33 is:

* **The sum rule.** ``Σ_i P_i = 1`` for every spectrum, so ``C·1 = 0``. The
  evaluations that impose it leave ``max_i |Σ_j C_ij| / max|C|`` at 3.1e-3 at
  most (ENDF/B-VIII.1, JEFF-4.0 and JENDL-5, 6-oct-2026); 25 JEFF-4.0 tapes,
  and band 0 of ENDF/B-VIII.1 Pu-238 and Pu-239, sit between 0.07 and 2.8 --
  not the covariance of a normalised spectrum under either reading (the
  dE-weighted sum does not vanish either). Nothing lies between, so 1e-2 is a
  clean edge and above it is a defect.
* **A bound that holds.** ``P_i`` is a probability, so ``0 <= P_i <= 1`` and
  ``var(P_i) <= P_i (1 - P_i)`` (Bhatia-Davis), which is never above 1/4
  (Popoviciu). With MF5 the bound uses the largest ``P_i (1 - P_i)`` over the
  incident nodes of the band, the most lenient choice; without MF5, sigma > 1/2
  is still impossible. Measured, only JEFF-4.0 U-239 breaks it (sigma up to 2919).

The rest is MF33's self-block check (negative variances, |rho|, PSD in three
levels) on each band, and the bands themselves: counted against NK, non-empty,
contiguous, and covering the incident energies of MF5.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .findings import DEFECT, NOTE, WARN, CovarianceFinding, CovarianceLocation

#: max_i |sum_j C_ij| / max|C| above this is not a normalised spectrum's covariance.
SUM_RULE_DEFECT = 1e-2
#: Relative tolerance on band edges and incident ranges (ENDF energies, 6-7 figures).
_EDGE_RTOL = 1e-6


def check_mf35(ctx, mf_obj, out: List[CovarianceFinding]) -> None:
    from .covariances import _bins, _check_self_block, _count, _grid_findings

    for mt, sec in sorted(mf_obj.mt.items()):
        mat = sec._mat
        head = CovarianceLocation(mat=mat, mf=35, mt=mt)
        _count(sec._nk, len(sec.subsections), "NK", head, out)
        tabulated = _spectrum(ctx, mt, head, out)

        prev_e2 = None
        for b, band in enumerate(sec.subsections):
            loc = CovarianceLocation(mat=mat, mf=35, mt=mt, band=b, lb=band.lb, ls=band.ls)
            e1, e2 = float(band.e1), float(band.e2)
            if not e1 < e2:
                level = DEFECT if e1 > e2 else WARN
                out.append(CovarianceFinding(
                    "band_empty", level,
                    f"the band covers no incident energy (E1={e1:.6g}, E2={e2:.6g} eV)", loc,
                    {"e1": e1, "e2": e2}))
            if prev_e2 is not None and not np.isclose(e1, prev_e2, rtol=_EDGE_RTOL, atol=0.0):
                gap = e1 > prev_e2
                out.append(CovarianceFinding(
                    "band_gap" if gap else "band_overlap", WARN if gap else DEFECT,
                    (f"no band covers {prev_e2:.6g}-{e1:.6g} eV" if gap else
                     f"overlaps the previous band, which ends at {prev_e2:.6g} eV: two "
                     "covariances for the same incident energies"), loc,
                    {"previous_e2": prev_e2, "e1": e1}))
            prev_e2 = max(e2, prev_e2) if prev_e2 is not None else e2

            grid = band.energy_grid()
            if not _grid_findings(grid, loc, out):
                continue
            matrix = band.matrix()
            _check_self_block(matrix, list(grid), loc, out, (), None)
            if not np.any(matrix):
                continue
            _sum_rule(band, matrix, loc, out)
            _probability_bound(matrix, grid, band, tabulated, loc, out, _bins)

        if tabulated is not None and sec.subsections:
            _incident_coverage(sec, tabulated, head, out)


def _spectrum(ctx, mt: int, head, out):
    """The single tabulated MF5 partial of this MT, or None (with the reason as a finding)."""
    if 5 not in ctx.endf.files:
        ctx.unavailable.setdefault((35, "MF5"), []).append(mt)
        _unavailable(ctx, head, out, "MF5 not read")
        return None
    sec5 = ctx.endf.files[5].mt.get(mt)
    if sec5 is None:
        out.append(CovarianceFinding(
            "missing_distribution", DEFECT,
            f"MF35/MT{mt} has no MF5/MT{mt}: a covariance of a spectrum the file does not give",
            head))
        return None
    parts = list(getattr(sec5, "partials", None) or getattr(sec5, "subsections", None) or [])
    from ..classes.mf5.partials import MF5PartialTabulated
    if len(parts) != 1 or not isinstance(parts[0], MF5PartialTabulated):
        ctx.unavailable.setdefault((35, "MF5"), []).append(mt)
        _unavailable(ctx, head, out,
                     "MF5 is not a single tabulated (LF=1) spectrum, so only the "
                     "bound sigma <= 1/2 is checked")
        return None
    return parts[0]


def _unavailable(ctx, loc, out, why: str) -> None:
    mts = ctx.unavailable[(35, "MF5")]
    for i, f in enumerate(out):
        if f.check == "central_values_unavailable" and f.location.mf == 35:
            out[i] = CovarianceFinding(f.check, f.level, f.summary, f.location, {"mts": list(mts)})
            return
    out.append(CovarianceFinding(
        "central_values_unavailable", NOTE, f"group probabilities not evaluable: {why}",
        CovarianceLocation(mat=loc.mat, mf=35), {"mts": list(mts)}))


def _sum_rule(band, matrix: np.ndarray, loc, out) -> None:
    residual = band.row_sum_residual()
    if residual <= SUM_RULE_DEFECT:
        return
    scale = float(np.max(np.abs(matrix)))
    out.append(CovarianceFinding(
        "sum_rule_violated", DEFECT,
        f"rows do not sum to zero: max|sum_j C_ij| / max|C| = {residual:.3g}; the group "
        "probabilities of a normalised spectrum sum to 1, so their covariance has C.1 = 0 "
        f"(the evaluations that impose it stay below {SUM_RULE_DEFECT:g})", loc,
        {"row_sum_residual": residual, "total_sum_over_max": float(matrix.sum()) / scale}))


def _probability_bound(matrix, grid, band, tabulated, loc, out, _bins) -> None:
    sigma = np.sqrt(np.clip(np.diag(matrix), 0.0, None))
    bound = np.full(sigma.size, 0.5)
    source = "P in [0, 1] (Popoviciu)"
    p_max = None
    if tabulated is not None:
        nodes = [k for k, e in enumerate(tabulated.incident_energies)
                 if band.e1 * (1 - _EDGE_RTOL) <= e <= band.e2 * (1 + _EDGE_RTOL)]
        if nodes:
            probs = np.array([np.asarray(tabulated.group_integrals(k, grid), dtype=float)
                              for k in nodes])
            p_max = probs.max(axis=0)
            bound = np.sqrt(np.max(np.clip(probs * (1.0 - probs), 0.0, None), axis=0))
            source = "var(P) <= P(1-P) (Bhatia-Davis), with MF5's largest P(1-P) in the band"
            zero = (p_max <= 0.0) & (sigma > 0.0)
            if zero.any():
                idx = np.flatnonzero(zero)
                out.append(CovarianceFinding(
                    "variance_where_distribution_is_zero", WARN,
                    f"{idx.size} groups carry a variance where MF5 gives no probability at any "
                    f"incident node of the band (largest sigma {sigma[idx].max():.3g})", loc,
                    {"n": int(idx.size), "bins": _bins(grid, idx[:10])}))
    over = (sigma > bound) & (bound > 0)
    if not over.any():
        return
    idx = np.flatnonzero(over)
    k = int(idx[np.argmax(sigma[idx] / bound[idx])])
    out.append(CovarianceFinding(
        "variance_exceeds_physical_bound", DEFECT,
        f"sigma(P) above what a probability allows in {idx.size} groups (worst {sigma[k]:.3g} "
        f"at {grid[k]:.4g}-{grid[k + 1]:.4g} eV against at most {bound[k]:.3g}; {source})", loc,
        {"n": int(idx.size), "worst_sigma": float(sigma[k]), "worst_bound": float(bound[k]),
         "worst_bin": _bins(grid, [k])[0],
         "worst_p": None if p_max is None else float(p_max[k])}))


def _incident_coverage(sec, tabulated, head, out) -> None:
    inc = np.asarray(tabulated.incident_energies, dtype=float)
    lo = min(float(b.e1) for b in sec.subsections)
    hi = max(float(b.e2) for b in sec.subsections)
    gaps = []
    if lo > inc[0] * (1 + _EDGE_RTOL) and lo > 1e-5 * (1 + _EDGE_RTOL):
        gaps.append(f"starts at {lo:.4g} eV, MF5 from {inc[0]:.4g} eV")
    if hi < inc[-1] * (1 - _EDGE_RTOL):
        gaps.append(f"stops at {hi:.4g} eV, MF5 goes to {inc[-1]:.4g} eV")
    if gaps:
        out.append(CovarianceFinding(
            "band_coverage", NOTE,
            "the bands " + "; ".join(gaps) + ": the spectrum has no covariance there", head,
            {"bands": [lo, hi], "mf5_incident": [float(inc[0]), float(inc[-1])]}))
