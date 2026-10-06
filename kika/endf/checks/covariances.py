"""Layer 1 of the covariance checks: a covariance section as written in the file.

MF31, MF33 and MF34 are checked here; MF32, MF35 and MF40 in :mod:`.mf32`,
:mod:`.mf35` and :mod:`.mf40`, which reuse these pieces and say what differs.

:func:`check_covariances` reads what kika's parser kept -- the records with
their LB/LS, counts and grids, and the triangle before it is mirrored -- and
reports what is wrong with it, without assembling a model covariance. Three
families of checks, in the order of the plan (kika-workspace
``docs/library/cov_checks_roadmap.md``, phases C2-C4):

**Structure (C2)** -- what ENDF-6 §31-34 pins down without looking at a value:
counts (NL/NMT1, NI, NC, the number of (L, L1) sub-subsections) against what was
read, NT against NE/LS/NER, LB valid for the file, LTY, grids strictly
increasing, (L, L1) inside the range NL/NL1/LTT allow, and **LB=5 LS=1 in a
cross block** -- a symmetric triangle where the block is not symmetric, which
kika mirrors and so turns into |rho| > 1 (the "defect A" of JEFF-4.0 MF34).
An MT the parser could not read at all is a ``parse_error``.

**Values (C3)** -- on each block summed on its own union grid the way kika sums
it: |rho| > 1 (a cross block needs the variances of both self blocks, on a
common grid), negative variances, rows with no variance (``max == min`` exactly,
never a threshold on sigma), covariance where the variance is zero, and the size
of the uncertainty against the central values -- MF3 or a PENDF for MF33, nu-bar
of MF1 for MF31, the a_l of MF4 for MF34. Only bounds that hold are used: sigma(a_l)
> 1 is impossible (|a_l| <= 1), while a cross section has no upper bound, so its
sigma_rel > 1 is a note and only sigma_rel > 10 outside a threshold region -- past
anything the three major libraries state -- a warning. Without central values a
block is reported "not evaluable", not guessed.

**Positive semi-definiteness (C4)** -- the eigenvalues of every self block, in
three levels by |lambda_min| / lambda_max: below 1e-6 a note, up to 1e-3 a
warning, above that a defect. Two explanations downgrade a negative eigenvalue
to a note, and both are measured, not assumed: it is within the rounding of the
values to the six significant figures an ENDF field holds, or the block's
correlations are quantised (JENDL-5 writes sigma_i sigma_j rho_ij with rho
rounded to 0.001, which leaves eigenvalues down to -2e-4 lambda_max in a
near-low-rank matrix) and lambda_min of the correlation matrix is inside what
that rounding can do. A defect also says which LB=5 record is already
indefinite on its own.

Policy (decided with the maintainer): every finding only informs. Nothing here
raises on a bad section, changes it or proposes a remedy.

A block that mixes absolute (LB=0/8/9) and relative components needs sigma(E)
to be summed (``MF33NeedsCrossSections``). Pass ``xs_sections`` or attach a
PENDF to the tape (``kika.processing.attach_pendf``); otherwise only its
relative part is checked, and a note says so.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..classes.mf33.mf33 import ABSOLUTE_LB, RELATIVE_LB, MF33MT, mixesAbsoluteAndRelative
from ...utils import get_endf_logger
from .findings import (
    DEFECT,
    NOTE,
    WARN,
    CovarianceCheckReport,
    CovarianceFinding,
    CovarianceLocation,
)
# At module scope, not in the dispatch: check_covariances is reached from the
# desktop app (the pre-flight), and a function-scope import is invisible to
# PyInstaller. These modules take what they reuse from here lazily, so there is
# no import cycle.
from .mf32 import check_mf32
from .mf35 import check_mf35
from .mf40 import check_mf40

logger = get_endf_logger(__name__)

#: |rho| up to 1 + this is rounding of the values (an ENDF field keeps 6-7
#: significant figures); above it, and up to RHO_DEFECT, still a note.
RHO_ROUNDING = 1e-6
RHO_DEFECT = 1e-3

#: The three PSD levels on |lambda_min| / lambda_max (roadmap, decisions).
PSD_NOTE = 1e-6
PSD_DEFECT = 1e-3

#: Rows whose sigma is below this fraction of the block's largest are left out
#: of the clipping impact (``_clipping_impact``).
PSD_IMPACT_FLOOR = 0.01
#: A light PSD warning becomes a note below this change of sigma, and a defect at
#: or above the second (``_grade_by_impact``, where the measurements are).
PSD_IMPACT_NOTE = 0.02
PSD_IMPACT_DEFECT = 0.25

#: Relative asymmetry of a self block (LS=0 stores both triangles).
ASYM_NOTE = 1e-6
ASYM_DEFECT = 1e-3

#: Significant figures an 11-column ENDF float keeps at worst (d.ddddd+nn).
ENDF_DIGITS = 6

#: Correlation quanta tried, coarsest first, and the share of off-diagonal
#: correlations that must sit on the lattice for it to count as quantised.
RHO_QUANTA = (1e-2, 1e-3, 1e-4)
RHO_QUANTUM_SHARE = 0.98
#: Slack on the spectral norm of a symmetric rounding error, ~2 s sqrt(n).
RHO_QUANTUM_SLACK = 1.5

#: MF33/MF31: sigma_bar below this fraction of the reaction's maximum is a
#: threshold region, where any relative uncertainty is legitimate.
RELATIVE_THRESHOLD_ZONE = 0.01
#: The measured edge of sigma_rel outside threshold regions in ENDF/B-VIII.1,
#: JEFF-4.0 and JENDL-5 (6-oct-2026): nothing reaches it.
RELATIVE_IMPLAUSIBLE = 10.0

VALID_LB = {
    31: frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9}),
    33: frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9}),
    34: frozenset({0, 1, 2, 5, 6}),
    40: frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9}),
}
#: Every covariance file layer 1 checks. MF30 is absent from the three major
#: libraries (6-oct-2026) and has no parser.
CHECKED_MF = (31, 32, 33, 34, 35, 40)
VALID_LTY = frozenset({0, 1, 2, 3})
VALID_LCT = frozenset({0, 1, 2})
#: MTs ENDF-6 §33.2.3 reserves for lumped reactions.
LUMPED_MT = range(851, 871)

#: How many items an evidence list keeps.
_EVIDENCE_ITEMS = 10


def check_covariances(
    endf,
    *,
    mf: Sequence[int] = CHECKED_MF,
    xs_sections: Optional[Dict[int, object]] = None,
) -> CovarianceCheckReport:
    """Check the covariance files of one ENDF tape as they are written.

    Parameters
    ----------
    endf : ENDF
        A tape from :func:`kika.endf.read_endf`. Read MF2/MF3 (MF33, and MF2
        for MF32) , MF4 (MF34) and MF5 (MF35) along with the covariances, or
        the checks that need central values say they could not run.
    mf : sequence of int
        Which covariance files to check; any of 31, 32, 33, 34, 35, 40
        (:data:`CHECKED_MF`, the default). Others are ignored.
    xs_sections : dict, optional
        sigma(E) by MT, e.g. a PENDF from ``kika.processing.read_pendf_mf3_sections``.
        Defaults to ``endf.pendf``. Used to sum mixed absolute/relative MF33
        blocks and as the MF33 central values; without it MF3 is used only
        above the upper limit of the resonance ranges, where it is the cross
        section and not a background.

    Returns
    -------
    CovarianceCheckReport
    """
    xs = xs_sections if xs_sections is not None else getattr(endf, "pendf", None)
    ctx = _Context(endf, xs or None)
    out: List[CovarianceFinding] = []
    mat = None
    for mf_number in mf:
        mf_obj = endf.files.get(mf_number)
        if mf_obj is None or mf_number not in CHECKED_MF:
            continue
        for mt, why in sorted(getattr(mf_obj, "parse_errors", {}).items()):
            out.append(CovarianceFinding(
                "parse_error", DEFECT,
                f"the parser could not read this section and dropped it ({why})",
                CovarianceLocation(mf=mf_number, mt=mt), {"error": why}))
        if mf_number == 34:
            _check_mf34(ctx, mf_obj, out)
        elif mf_number == 32:
            check_mf32(ctx, mf_obj, out)
        elif mf_number == 35:
            check_mf35(ctx, mf_obj, out)
        elif mf_number == 40:
            check_mf40(ctx, mf_obj, out)
        else:
            _check_mf33(ctx, mf_number, mf_obj, out)
        for sec in mf_obj.mt.values():
            mat = mat or getattr(sec, "_mat", None)
    return CovarianceCheckReport(tuple(out), source=getattr(endf, "source_path", None), mat=mat)


# ---------------------------------------------------------------------------
# Context: the central values a tape can give
# ---------------------------------------------------------------------------


class _Context:
    def __init__(self, endf, xs: Optional[Dict[int, object]]) -> None:
        self.endf = endf
        self.xs = xs
        self.mf3 = dict(endf.files[3].mt) if 3 in endf.files else {}
        self.mf4 = dict(endf.files[4].mt) if 4 in endf.files else {}
        self.mf1 = dict(endf.files[1].mt) if 1 in endf.files else {}
        self.eh = _resonance_upper_limit(endf)
        self.unavailable: Dict[Tuple[int, str], List[int]] = {}

    def cross_section(self, mt: int):
        """(sigma source, lowest energy where it is the cross section) or None."""
        if self.xs:
            src = self.xs.get(mt)
            if src is None:
                src = _summed_partials(self.xs, mt)
            return (src, 0.0) if src is not None else None
        if self.eh is None:
            return None
        sec = self.mf3.get(mt)
        if sec is None and self.mf3:
            # A summation MT the tape leaves out of MF3 (MT3, MT4, ...): the sum of
            # its partials, which above the resonance ranges is the cross section.
            sec = _summed_partials(self.mf3, mt)
        return (sec, self.eh) if sec is not None else None


def _resonance_upper_limit(endf) -> Optional[float]:
    """Highest EH of a resolved or unresolved range in MF2, 0 if none, None if MF2 unread."""
    mf2 = endf.files.get(2)
    if mf2 is None:
        return None
    sec = mf2.mt.get(151)
    if sec is None:
        return None
    eh = 0.0
    for iso in getattr(sec, "isotopes", []) or []:
        for er in getattr(iso, "energy_ranges", []) or []:
            if getattr(er, "lru", 0) in (1, 2):
                eh = max(eh, float(er.eh))
    return eh


def _summed_partials(xs: Dict[int, object], mt: int):
    from types import SimpleNamespace
    from ..writers.redundant import resolve_sum_components

    parts = resolve_sum_components(mt, xs)
    if not parts:
        return None
    tables = []
    for part in parts:
        sec = xs[part]
        values = getattr(sec, "values", None)
        if values is None:
            values = getattr(sec, "cross_sections")
        tables.append((np.asarray(sec.energies, dtype=float), np.asarray(values, dtype=float)))
    grid = np.unique(np.concatenate([e for e, _ in tables]))
    total = np.zeros(grid.size)
    for e, v in tables:
        total += np.interp(grid, e, v, left=0.0, right=0.0)
    return SimpleNamespace(energies=grid, values=total)


# ---------------------------------------------------------------------------
# Record-level structure, shared by MF33 and MF34
# ---------------------------------------------------------------------------


def _expected_nt(rec) -> Optional[int]:
    lb = rec.lb
    if lb in (0, 1, 2, 3, 4, 8, 9):
        return 2 * int(rec.np or 0)
    if lb == 5:
        ne = int(rec.ne or 0)
        m = ne - 1
        return ne + (m * (m + 1) // 2 if rec.ls == 1 else m * m)
    if lb == 6:
        return 1 + len(rec.row_energies) * len(rec.col_energies)
    return None


def _grids(rec) -> List[Sequence[float]]:
    if rec.lb in (0, 1, 2, 8, 9):
        return [rec.e_table_k] if rec.e_table_k else []
    if rec.lb in (3, 4):
        return [rec.e_table_k, rec.e_table_l]
    if rec.lb == 5:
        return [rec.energies]
    if rec.lb == 6:
        return [rec.row_energies, rec.col_energies]
    return []


def _decoded(sec, rec):
    """Decode one record with the section's own decoder; raises ValueError."""
    lb = rec.lb
    if lb in (0, 1, 2):
        return sec._decode_lb012_matrix(rec)
    if lb in (3, 4):
        return sec._decode_lb34_matrix(rec)
    if lb == 5:
        return sec._decode_lb5_matrix(rec)
    if lb == 6:
        return sec._decode_lb6_matrix(rec)
    return None


def _grid_findings(g: Sequence[float], loc, out: List[CovarianceFinding]) -> bool:
    """An energy grid: at least two points, never decreasing. True if usable."""
    arr = np.asarray(g, dtype=float)
    if arr.size < 2:
        out.append(CovarianceFinding(
            "grid_too_short", DEFECT, f"an energy grid with {arr.size} point(s) defines no bin",
            loc, {"n": int(arr.size)}))
        return False
    steps = np.diff(arr)
    if np.any(steps < 0):
        k = int(np.argmax(steps < 0))
        out.append(CovarianceFinding(
            "grid_not_increasing", DEFECT,
            f"energies decrease at index {k + 1} ({arr[k]:.6g} -> {arr[k + 1]:.6g} eV)",
            loc, {"index": k + 1, "energies": [float(arr[k]), float(arr[k + 1])]}))
        return False
    elif np.any(steps == 0):
        idx = np.flatnonzero(steps == 0)
        out.append(CovarianceFinding(
            "grid_repeated_point", WARN,
            f"{idx.size} energ{'y is' if idx.size == 1 else 'ies are'} repeated, "
            f"leaving zero-width bins (first {arr[idx[0]]:.6g} eV)",
            loc, {"energies": [float(arr[i]) for i in idx[:_EVIDENCE_ITEMS]]}))
    return True


def _check_record(mf_number: int, sec, rec, loc: CovarianceLocation,
                  out: List[CovarianceFinding]) -> bool:
    """Structure of one LIST record. True if its values can be used."""
    usable = True
    if rec.lb not in VALID_LB[mf_number]:
        out.append(CovarianceFinding(
            "lb_invalid", DEFECT, f"LB={rec.lb} is not defined for MF{mf_number}", loc,
            {"lb": rec.lb, "valid": sorted(VALID_LB[mf_number])}))
        return False
    if mf_number == 34 and rec.lb in (0, 1, 2) and (rec.lt or 0) > 0:
        out.append(CovarianceFinding(
            "lt_not_decoded", NOTE,
            f"LB={rec.lb} with LT={rec.lt} (two tables) is kept verbatim but kika does "
            "not decode it, so its values are not checked", loc, {"lt": rec.lt}))
        return False
    expected = _expected_nt(rec)
    if expected is not None and rec.nt is not None and int(rec.nt) != expected:
        out.append(CovarianceFinding(
            "nt_mismatch", DEFECT,
            f"NT={rec.nt} but the counts in the record imply {expected}", loc,
            {"nt": rec.nt, "expected": expected}))
    for g in _grids(rec):
        usable = _grid_findings(g, loc, out) and usable
    if rec.lt is not None and rec.lb in (3, 4) and int(rec.lt or 0) > int(rec.np or 0):
        out.append(CovarianceFinding(
            "count_mismatch", DEFECT, f"LT={rec.lt} pairs in the second table but NP={rec.np}",
            loc, {"lt": rec.lt, "np": rec.np}))
    if usable:
        try:
            _decoded(sec, rec)
        except (ValueError, IndexError) as exc:
            out.append(CovarianceFinding(
                "decode_error", DEFECT, f"kika cannot decode the record: {exc}", loc,
                {"error": str(exc)}))
            usable = False
    return usable


def _count(declared, found: int, what: str, loc, out) -> None:
    declared = int(declared or 0)
    if declared != found:
        out.append(CovarianceFinding(
            "count_mismatch", DEFECT,
            f"{what} = {declared} declared, {found} read", loc,
            {"field": what, "declared": declared, "read": found}))


# ---------------------------------------------------------------------------
# Matrix-level checks, shared
# ---------------------------------------------------------------------------


def _bins(grid: Sequence[float], idx: Iterable[int]) -> List[List[float]]:
    g = list(grid)
    return [[float(g[i]), float(g[i + 1])] for i in idx]


def _half_ulp_norm(matrix: np.ndarray) -> float:
    """Frobenius norm of the rounding of every element to ENDF_DIGITS figures."""
    a = np.abs(matrix[matrix != 0])
    if a.size == 0:
        return 0.0
    ulp = 0.5 * 10.0 ** (np.floor(np.log10(a)) - (ENDF_DIGITS - 1))
    return float(np.sqrt(np.sum(ulp ** 2)))


def _rho_quantum(corr: np.ndarray) -> Optional[float]:
    """The coarsest lattice the off-diagonal correlations sit on, if any."""
    n = corr.shape[0]
    if n < 3:
        return None
    off = corr[np.triu_indices(n, 1)]
    off = off[off != 0]
    if off.size < 3:
        return None
    for q in RHO_QUANTA:
        # rho recomputed from the stored values misses its lattice by up to ~3e-5
        # (Na-23 MT79 of JENDL-5); q/20 keeps a random rho from landing on it by
        # chance more than 10 % of the time, against the 98 % asked for.
        on = np.abs(off / q - np.round(off / q)) * q < min(5e-5, q / 20)
        if np.mean(on) >= RHO_QUANTUM_SHARE:
            return q
    return None


def _check_self_block(matrix: np.ndarray, grid: Sequence[float], loc, out,
                      records: Sequence[Tuple[int, object]], sec) -> None:
    """C3 and C4 on one self block (MT, MT) or (L, L) on its union grid."""
    m = matrix.shape[0]
    d = np.diag(matrix).copy()

    neg = np.flatnonzero(d < 0)
    if neg.size:
        k = int(neg[np.argmin(d[neg])])
        out.append(CovarianceFinding(
            "negative_variance", DEFECT,
            f"{neg.size} of {m} variances are negative (worst {d[k]:.3e} in "
            f"{grid[k]:.4g}-{grid[k + 1]:.4g} eV)", loc,
            {"n": int(neg.size), "worst": float(d[k]), "bins": _bins(grid, neg[:_EVIDENCE_ITEMS])}))

    zero_rows = np.array([np.max(r) == np.min(r) == 0 for r in matrix]) if m else np.array([])
    if zero_rows.any():
        idx = np.flatnonzero(zero_rows)
        out.append(CovarianceFinding(
            "inert_rows", NOTE, f"{idx.size} of {m} rows are exactly zero (no stated uncertainty)",
            loc, {"n": int(idx.size), "of": m, "bins": _bins(grid, idx[:_EVIDENCE_ITEMS])}))
    lonely = np.flatnonzero((d == 0) & ~zero_rows)
    if lonely.size:
        out.append(CovarianceFinding(
            "covariance_without_variance", DEFECT,
            f"{lonely.size} rows have zero variance but non-zero covariances (|rho| is infinite)",
            loc, {"n": int(lonely.size), "bins": _bins(grid, lonely[:_EVIDENCE_ITEMS])}))

    scale = float(np.max(np.abs(matrix))) if m else 0.0
    if scale > 0:
        asym = float(np.max(np.abs(matrix - matrix.T))) / scale
        if asym > ASYM_NOTE:
            level = DEFECT if asym > ASYM_DEFECT else NOTE
            out.append(CovarianceFinding(
                "asymmetric_self_block", level,
                f"the block is not symmetric: max|C - C^T| / max|C| = {asym:.2e}", loc,
                {"relative_asymmetry": asym}))

    live = d > 0
    if live.sum() >= 2:
        sub = matrix[np.ix_(live, live)]
        sym = 0.5 * (sub + sub.T)
        s = np.sqrt(np.diag(sym))
        corr = sym / np.outer(s, s)
        _rho_finding(corr, np.flatnonzero(live), np.flatnonzero(live), grid, loc, out,
                     self_block=True)
    # Eigenvalues over every row that is not exactly zero: a row with no variance
    # but with covariances (B-10 MT102 of JEFF-4.0) is what makes the block
    # indefinite, and leaving it out would hide that.
    active = ~zero_rows
    if active.sum() >= 2:
        sub = matrix[np.ix_(active, active)]
        _psd_finding(0.5 * (sub + sub.T), loc, out, records, sec, grid)


def _rho_finding(corr: np.ndarray, rows: np.ndarray, cols: np.ndarray, grid, loc, out,
                 self_block: bool, extra: Optional[dict] = None) -> None:
    a = np.abs(corr)
    if self_block:
        a = a.copy()
        np.fill_diagonal(a, 0.0)
    worst = float(a.max()) if a.size else 0.0
    if worst <= 1 + RHO_ROUNDING:
        return
    i, j = np.unravel_index(int(np.argmax(a)), a.shape)
    n_bad = int(np.sum(a > 1 + RHO_DEFECT))
    level = DEFECT if worst > 1 + RHO_DEFECT else NOTE
    evidence = {
        "max_abs_rho": worst,
        "n_above": n_bad,
        "at": _bins(grid, [int(rows[i]), int(cols[j])]),
    }
    if extra:
        evidence.update(extra)
    summary = (f"|rho| up to {worst:.4g} ({n_bad} entries above 1+{RHO_DEFECT:g})"
               if level == DEFECT else
               f"|rho| up to {worst:.7g}, within rounding of the stored values")
    if extra and extra.get("stored_triangle_max_abs_rho") is not None:
        tri = extra["stored_triangle_max_abs_rho"]
        summary += (f"; the stored LS=1 triangle alone reaches {tri:.4g}, so the mirror "
                    "kika applies is what " + ("creates" if tri <= 1 + RHO_DEFECT else "worsens")
                    + " it")
    out.append(CovarianceFinding("correlation_out_of_bounds", level, summary, loc, evidence))


def _quantised_allowance(records: Sequence[Tuple[int, object]], sec, grid):
    """How negative rounded correlations can make the summed block, or None.

    Some evaluations (JENDL-5) write each LB=5 record as sigma_i sigma_j rho_ij with
    rho rounded to a lattice q. The rounding is a symmetric error E with entries
    roughly uniform in +-q/2, whose spectral norm is about 2 (q/sqrt(12)) sqrt(n)
    (the edge of Wigner's semicircle), so lambda_min(R) >= -that, and the record
    D R D >= -that x max(d^2). Projecting onto the union grid scales a negative
    eigenvalue by at most ||T||^2, the most union bins one native bin feeds, and
    by Weyl the sum is no more negative than the sum of its parts. LB=0/1/2/8/9
    are positive semi-definite by construction and add nothing. Every LB=5 record
    must be quantised for the bound to say anything; LB=3/4/6 do not occur in a
    self block of a well-formed file and void it.
    """
    g_union = np.asarray(grid, dtype=float)
    total, quanta = 0.0, set()
    for _, rec in records:
        if rec.lb in (0, 1, 2, 8, 9):
            continue
        if rec.lb != 5:
            return None
        try:
            mat, g = sec._decode_lb5_matrix(rec)
        except (ValueError, IndexError):
            return None
        mat = 0.5 * (mat + mat.T)
        d = np.diag(mat)
        live = d > 0
        if live.sum() < 3:
            continue
        sub = mat[np.ix_(live, live)]
        s = np.sqrt(np.diag(sub))
        q = _rho_quantum(sub / np.outer(s, s))
        if q is None:
            return None
        quanta.add(q)
        corr_bound = RHO_QUANTUM_SLACK * 2.0 * (q / math.sqrt(12.0)) * math.sqrt(int(live.sum()))
        native = np.asarray(g, dtype=float)
        overlap = np.maximum(0.0, np.minimum(g_union[1:, None], native[None, 1:])
                             - np.maximum(g_union[:-1, None], native[None, :-1]))
        width = (g_union[1:] - g_union[:-1])[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(width > 0, overlap / width, 0.0)
        gain = float(np.max(np.sum(t ** 2, axis=0))) if t.size else 1.0
        total += corr_bound * float(d.max()) * gain
    if not quanta:
        return None
    return total, quanta


def _clipping_impact(ev: np.ndarray, vecs: np.ndarray, sym: np.ndarray) -> Optional[float]:
    """How much forcing the block to PSD would move its uncertainties.

    Setting the negative eigenvalues to zero (the nearest PSD matrix in the
    Frobenius norm) adds sum_k |lambda_k| v_ik^2 to each variance. Returned is
    the largest relative change of sigma over the rows whose sigma is at least
    PSD_IMPACT_FLOOR of the block's largest: a tail row with sigma ~1e-12 would
    otherwise turn any absolute correction into an enormous relative one.
    """
    d = np.diag(sym)
    live = d > 0
    if not live.any():
        return None
    added = (vecs ** 2) @ (-np.minimum(ev, 0.0))
    sd = np.sqrt(np.where(live, d, 0.0))
    rows = live & (sd >= PSD_IMPACT_FLOOR * sd.max())
    return float(np.max(np.sqrt(d[rows] + added[rows]) / sd[rows] - 1.0))


def _grade_by_impact(level: str, reason: str, impact: Optional[float]) -> Tuple[str, str]:
    """Regrade a light PSD warning by what forcing the block to PSD does to sigma.

    lambda_min / lambda_max places a block between rounding and a defect, but in
    that light band it says nothing about consequence. Measured on ENDF/B-VIII.1,
    JEFF-4.0 and JENDL-5 (6-oct-2026), the light warnings of MF31/33/34 change
    sigma by under 2 % in 141 of 159 cases, while 4 change it by 33-155 % (Pt-190
    MT54, Ac-227 MT107, Eu-154 MT3, Eu-156 MF34 L=3): no PSD matrix near them keeps
    the stated variances. The rest sit at 13 % or below, which is where the
    defect edge goes. Every MF35 band that keeps the sum rule stays under 0.8 %.

    Only a warning moves. A note explained by rounding stays a note, and a defect
    stays a defect: 149 defects change sigma by under 2 % because clipping keeps
    the variances and breaks the correlations, which the ratio does see.
    """
    if level != WARN or impact is None:
        return level, reason
    if impact < PSD_IMPACT_NOTE:
        return NOTE, (f"forcing the block to PSD changes no sigma by more than {impact:.2%} "
                      f"(below {PSD_IMPACT_NOTE:.0%})")
    if impact >= PSD_IMPACT_DEFECT:
        return DEFECT, (f"forcing the block to PSD changes a sigma by {impact:.0%}: no "
                        "positive semi-definite matrix near it keeps the stated variances")
    return level, f"forcing the block to PSD changes a sigma by {impact:.1%}"


def _psd_finding(sym: np.ndarray, loc, out,
                 records: Sequence[Tuple[int, object]], sec, grid) -> None:
    ev = np.linalg.eigvalsh(sym)
    lam_min, lam_max = float(ev[0]), float(ev[-1])
    if lam_min >= 0 or lam_max <= 0:
        if lam_max <= 0 < sym.shape[0]:
            out.append(CovarianceFinding(
                "not_positive_semidefinite", DEFECT, "no positive eigenvalue at all", loc,
                {"lambda_min": lam_min, "lambda_max": lam_max}))
        return
    ratio = -lam_min / lam_max
    evidence = {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "ratio": ratio,
        "n_negative": int(np.sum(ev < 0)),
        "sigma_change_if_clipped": _clipping_impact(*np.linalg.eigh(sym), sym),
    }
    level = NOTE if ratio < PSD_NOTE else (WARN if ratio <= PSD_DEFECT else DEFECT)
    reason = ""
    ulp = _half_ulp_norm(sym)
    evidence["rounding_bound"] = ulp
    if -lam_min <= ulp:
        level, reason = NOTE, f"within the rounding of the values to {ENDF_DIGITS} figures"
    elif level == WARN:
        allowance = _quantised_allowance(records, sec, grid)
        if allowance is not None:
            bound, quanta = allowance
            evidence.update({"rho_quantum": sorted(quanta), "quantised_rounding_bound": bound})
            if -lam_min <= bound:
                level = NOTE
                reason = ("the correlations are rounded to "
                          + "/".join(f"{q:g}" for q in sorted(quanta))
                          + f" and that rounding can reach lambda_min (bound {bound:.2e})")
    level, reason = _grade_by_impact(level, reason, evidence["sigma_change_if_clipped"])
    # Which record is already indefinite on its own.
    alone = []
    for k, rec in records:
        if rec.lb != 5:
            continue
        try:
            mat, _ = sec._decode_lb5_matrix(rec)
        except (ValueError, IndexError):
            continue
        w = np.linalg.eigvalsh(0.5 * (mat + mat.T))
        if w[-1] > 0 and -w[0] / w[-1] > PSD_DEFECT:
            alone.append({"ni": k, "ratio": float(-w[0] / w[-1])})
    if alone:
        evidence["records_indefinite_alone"] = alone
    summary = f"lambda_min/lambda_max = -{ratio:.2e}"
    if reason:
        summary += f"; {reason}"
    if alone:
        summary += "; already indefinite in " + ", ".join(f"NI[{a['ni']}]" for a in alone) + " alone"
    out.append(CovarianceFinding("not_positive_semidefinite", level, summary, loc, evidence))


def _relative_uncertainty(rel: np.ndarray, central: np.ndarray, valid: np.ndarray, grid,
                          loc, out) -> None:
    """sigma_rel of a cross section or nu-bar, judged against what is measured, not assumed.

    A cross section is bounded below by 0 and not above, so no variance is
    impossible: a lognormal carries any sigma_rel. Two findings, and neither
    guesses a scale:

    * ``relative_uncertainty_above_one`` (note) -- sigma_rel > 1. Not a fault of
      the file; it says that a *normal* draw goes negative with probability
      Phi(-1/sigma_rel) (16 % at 100 %), which is the sampler's choice of space
      to make. Over the three libraries thousands of bins sit between 1 and 10,
      outside threshold regions too: there is no edge at 1 to call a defect.
    * ``implausible_relative_uncertainty`` (warn) -- sigma_rel > 10 where sigma_bar
      is at least 1 % of the reaction's maximum (outside a threshold region,
      where sigma_bar -> 0 makes any ratio legitimate). That is past the measured
      edge of the distribution: in ENDF/B-VIII.1, JEFF-4.0 and JENDL-5 no bin
      outside a threshold region reaches it, so what does is an outlier of the
      kind a unit error makes (LB=8 summed as relative gave 53 on Eu-154).
    """
    smax = float(central[valid].max())
    above = valid & (rel > 1.0)
    if not above.any():
        return
    from math import erf, sqrt

    def p_negative(r):
        return 0.5 * (1.0 + erf(-1.0 / r / sqrt(2.0)))

    idx = np.flatnonzero(above)
    k = int(idx[np.argmax(rel[idx])])
    bulk = central >= RELATIVE_THRESHOLD_ZONE * smax
    implausible = above & bulk & (rel > RELATIVE_IMPLAUSIBLE)
    if implausible.any():
        j = np.flatnonzero(implausible)
        w = int(j[np.argmax(rel[j])])
        out.append(CovarianceFinding(
            "implausible_relative_uncertainty", WARN,
            f"sigma_rel up to {rel[w]:.3g} in {j.size} bins where the cross section is not "
            f"near a threshold (sigma_bar {central[w]:.4g}, {central[w] / smax:.0%} of its maximum), "
            f"past the largest value any of the three major libraries states there "
            f"({RELATIVE_IMPLAUSIBLE:g})", loc,
            {"n": int(j.size), "worst_rel": float(rel[w]), "worst_bin": _bins(grid, [w])[0],
             "central_over_max": float(central[w] / smax)}))
    out.append(CovarianceFinding(
        "relative_uncertainty_above_one", NOTE,
        f"sigma_rel above 100 % in {idx.size} bins (worst {rel[k]:.3g} at "
        f"{grid[k]:.4g}-{grid[k + 1]:.4g} eV, sigma_bar {central[k] / smax:.1%} of the maximum); "
        f"a normal draw there is negative with probability {p_negative(rel[k]):.0%}", loc,
        {"n": int(idx.size), "n_outside_threshold_zone": int((above & bulk).sum()),
         "worst_rel": float(rel[k]), "worst_bin": _bins(grid, [k])[0],
         "p_negative_if_normal": p_negative(float(rel[k]))}))


def _legendre_bound(sigma_abs: np.ndarray, central: np.ndarray, valid: np.ndarray, grid,
                    loc, out) -> None:
    """sigma(a_l) > 1 is impossible, not just large.

    f(mu) >= 0 bounds every normalised Legendre coefficient, |a_l| <= 1, and a
    quantity confined to an interval of width 2 has a standard deviation of at
    most 1 whatever its distribution (Popoviciu's inequality). A stated sigma
    above 1 cannot be the uncertainty of a physical angular distribution.
    """
    over = valid & (sigma_abs > 1.0)
    if not over.any():
        return
    idx = np.flatnonzero(over)
    k = int(idx[np.argmax(sigma_abs[idx])])
    out.append(CovarianceFinding(
        "variance_exceeds_physical_bound", DEFECT,
        f"sigma(a_l) above 1 in {idx.size} bins even with the smallest |a_l| in the bin "
        f"(worst {sigma_abs[k]:.4g} at {grid[k]:.4g}-{grid[k + 1]:.4g} eV, |a_l| >= "
        f"{central[k]:.3g}); |a_l| <= 1 allows at most 1", loc,
        {"n": int(idx.size), "worst_sigma_abs": float(sigma_abs[k]),
         "worst_central": float(central[k]), "worst_bin": _bins(grid, [k])[0]}))


# ---------------------------------------------------------------------------
# MF31 / MF33
# ---------------------------------------------------------------------------


def _assemble33(sec: MF33MT, recs, mt: int, mt1: int, xs, grid=None):
    """(matrix, grid, relative, partial) or None, summed the way kika sums it."""
    partial = False
    x_row = x_col = None
    if mixesAbsoluteAndRelative(recs):
        from ..classes.mf33.mf33 import _xs_section
        x_row = _xs_section(xs, mt, recs)
        x_col = _xs_section(xs, mt1, recs)
        if x_row is None or x_col is None:
            recs = [r for r in recs if int(r.lb) in RELATIVE_LB]
            partial = True
            x_row = x_col = None
    out = sec._process_ni_records_to_matrix(list(recs), f"MT{mt}x{mt1}", target_grid=grid,
                                            xs_row=x_row, xs_col=x_col)
    if out is None:
        return None
    matrix, g, relative = out
    return matrix, list(g), relative, partial


def _check_mf33(ctx: _Context, mf_number: int, mf_obj, out: List[CovarianceFinding]) -> None:
    sections: Dict[int, MF33MT] = dict(mf_obj.mt)
    usable: Dict[Tuple[int, int, int], List[Tuple[int, object]]] = {}

    # Structure.
    for mt, sec in sorted(sections.items()):
        mat = sec._mat
        head = CovarianceLocation(mat=mat, mf=mf_number, mt=mt)
        if sec._mtl:
            if sec.subsections:
                out.append(CovarianceFinding(
                    "count_mismatch", DEFECT,
                    f"a lumped component (MTL={sec._mtl}) should carry no subsections, it has "
                    f"{len(sec.subsections)}", head, {"mtl": sec._mtl}))
            continue
        _count(sec._nl, len(sec.subsections), "NL", head, out)
        seen = set()
        for sub in sec.subsections:
            mat1, mt1 = _mat1(sub, mat), int(sub.mt1 or 0)
            loc = CovarianceLocation(mat=mat, mf=mf_number, mt=mt, mat1=mat1, mt1=mt1)
            if (mat1, mt1) in seen:
                out.append(CovarianceFinding(
                    "duplicate_block", DEFECT, "this (MAT1, MT1) subsection appears twice", loc))
            seen.add((mat1, mt1))
            _count(sub.nc, len(sub.nc_records), "NC", loc, out)
            _count(sub.ni, len(sub.ni_records), "NI", loc, out)
            for k, nc in enumerate(sub.nc_records):
                nloc = CovarianceLocation(mat=mat, mf=mf_number, mt=mt, mat1=mat1, mt1=mt1, nc=k)
                if nc.lty not in VALID_LTY:
                    out.append(CovarianceFinding(
                        "lty_invalid", DEFECT, f"LTY={nc.lty} is not defined", nloc, {"lty": nc.lty}))
                elif nc.lty == 0 and int(nc.nci or 0) != len(nc.ci):
                    _count(nc.nci, len(nc.ci), "NCI", nloc, out)
                if nc.e1 is not None and nc.e2 is not None and not nc.e1 < nc.e2:
                    out.append(CovarianceFinding(
                        "grid_not_increasing", DEFECT,
                        f"the NC range is empty or reversed (E1={nc.e1:.6g}, E2={nc.e2:.6g})", nloc))
            good = []
            for k, rec in enumerate(sub.ni_records):
                rloc = CovarianceLocation(mat=mat, mf=mf_number, mt=mt, mat1=mat1, mt1=mt1, ni=k,
                                          lb=rec.lb, ls=rec.ls if rec.lb == 5 else None)
                if _check_record(mf_number, sec, rec, rloc, out):
                    good.append((k, rec))
            if good:
                usable[(mt, mat1, mt1)] = good

    _completeness33(ctx, mf_number, sections, usable, out)

    # Self blocks.
    self_grids: Dict[int, List[float]] = {}
    for (mt, mat1, mt1), good in sorted(usable.items()):
        if mat1 != 0 or mt1 != mt:
            continue
        sec = sections[mt]
        loc = CovarianceLocation(mat=sec._mat, mf=mf_number, mt=mt, mat1=0, mt1=mt)
        recs = [r for _, r in good]
        res = _assemble33(sec, recs, mt, mt, ctx.xs)
        if res is None:
            continue
        matrix, grid, relative, partial = res
        self_grids[mt] = grid
        if partial:
            _note_partial(loc, recs, out)
        _check_self_block(matrix, grid, loc, out, good, sec)
        _mf33_magnitude(ctx, mt, matrix, grid, relative, loc, out)
        _mf33_coverage(ctx, mt, grid, loc, out)

    # Cross blocks.
    for (mt, mat1, mt1), good in sorted(usable.items()):
        if mat1 == 0 and mt1 == mt:
            continue
        sec = sections[mt]
        loc = CovarianceLocation(mat=sec._mat, mf=mf_number, mt=mt, mat1=mat1, mt1=mt1)
        ls1 = [(k, r) for k, r in good if r.lb == 5 and r.ls == 1]
        if mat1 != 0:
            for k, r in ls1:
                _ls1_finding(loc, k, r, None, out)
            continue
        row_self = usable.get((mt, 0, mt))
        col_self = usable.get((mt1, 0, mt1))
        if row_self is None or col_self is None or mt1 not in sections:
            # A missing partner is a C5 finding (`missing_partner`, `missing_self_block`).
            for k, r in ls1:
                _ls1_finding(loc, k, r, None, out)
            continue
        recs = [r for _, r in good]
        grid = sorted({e for r in recs for g in _grids(r) for e in g}
                      | set(self_grids.get(mt, [])) | set(self_grids.get(mt1, [])))
        res = _assemble33(sec, recs, mt, mt1, ctx.xs, grid)
        rr = _assemble33(sec, [r for _, r in row_self], mt, mt, ctx.xs, grid)
        cc = _assemble33(sections[mt1], [r for _, r in col_self], mt1, mt1, ctx.xs, grid)
        if res is None or rr is None or cc is None:
            continue
        if res[3]:
            _note_partial(loc, recs, out)
        tri = {}
        for k, r in ls1:
            t = _triangle_rho(r, lambda g: _diag33(sec, row_self, mt, ctx.xs, g),
                              lambda g: _diag33(sections[mt1], col_self, mt1, ctx.xs, g))
            tri[k] = t
            _ls1_finding(loc, k, r, t, out)
        extra = {"stored_triangle_max_abs_rho": max(tri.values())} if tri and all(
            v is not None for v in tri.values()) else None
        _check_cross_block(res[0], np.diag(rr[0]), np.diag(cc[0]), grid, loc, out, extra)


def _diag33(sec, self_good, mt, xs, grid):
    res = _assemble33(sec, [r for _, r in self_good], mt, mt, xs, list(grid))
    return None if res is None else np.diag(res[0])


def _note_partial(loc, recs, out) -> None:
    lbs = sorted({int(r.lb) for r in recs})
    out.append(CovarianceFinding(
        "mixed_needs_cross_sections", NOTE,
        "the block mixes absolute (LB 0/8/9) and relative components and no sigma(E) was "
        "given, so only the relative part was checked (attach a PENDF to check the sum)",
        loc, {"lb": lbs}))


def _bin_average_nubar(sec, grid) -> np.ndarray:
    """nu-bar averaged over each bin, from five log-spaced points (it is smooth)."""
    g = np.asarray(grid, dtype=float)
    out = np.zeros(g.size - 1)
    for i in range(g.size - 1):
        lo, hi = g[i], g[i + 1]
        if hi <= lo:
            continue
        pts = np.geomspace(lo, hi, 5) if lo > 0 else np.linspace(lo, hi, 5)
        out[i] = float(np.mean(np.asarray(sec.get_nubar(pts, out_of_range="zero"), dtype=float)))
    return out


def _mf33_magnitude(ctx, mt, matrix, grid, relative, loc, out) -> None:
    g = np.asarray(grid, dtype=float)
    if loc.mf == 31:
        # MF31 is the covariance of nu-bar, whose central values are MF1 MT452/455/456.
        sec1 = ctx.mf1.get(mt)
        if sec1 is None or not hasattr(sec1, "get_nubar"):
            ctx.unavailable.setdefault((31, "MF1"), []).append(mt)
            _unavailable_note(ctx, loc, out, why="MF1 not read or without this nu-bar")
            return
        central, floor = _bin_average_nubar(sec1, grid), 0.0
    else:
        got = ctx.cross_section(mt)
        if got is None:
            ctx.unavailable.setdefault((loc.mf, "MF3" if not ctx.xs else "PENDF"), []).append(mt)
            _unavailable_note(ctx, loc, out)
            return
        src, floor = got
        central = MF33MT._bin_average_xs_exact(src, grid)
    valid = (g[:-1] >= floor) & (central > 0) & (g[1:] > g[:-1])
    if not valid.any():
        return
    var = np.clip(np.diag(matrix), 0.0, None)
    if relative:
        rel = np.sqrt(var)
        sigma_abs = rel * central
    else:
        sigma_abs = np.sqrt(var)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(central > 0, sigma_abs / central, np.inf)
    _relative_uncertainty(rel, central, valid, grid, loc, out)


def _unavailable_note(ctx, loc, out, why: Optional[str] = None) -> None:
    """One note per file, updated in place, listing the MTs with no central values."""
    key = (31, "MF1") if loc.mf == 31 else (loc.mf, "MF3" if not ctx.xs else "PENDF")
    mts = ctx.unavailable[key]
    if len(mts) > 1:
        for i, f in enumerate(out):
            if f.check == "central_values_unavailable" and f.location.mf == loc.mf:
                out[i] = CovarianceFinding(f.check, f.level, f.summary,
                                           f.location, {"mts": list(mts)})
                return
    if why is None:
        why = ("no PENDF given and MF2 not read, so MF3 cannot be told apart from a background"
               if ctx.eh is None and not ctx.xs else "no sigma(E) for these MTs")
    out.append(CovarianceFinding(
        "central_values_unavailable", NOTE,
        f"uncertainty magnitudes not evaluable: {why}", CovarianceLocation(mat=loc.mat, mf=loc.mf),
        {"mts": list(mts)}))


def _mf33_coverage(ctx, mt, grid, loc, out) -> None:
    sec3 = ctx.mf3.get(mt)
    if sec3 is None:
        return
    e = np.asarray(sec3.energies, dtype=float)
    x = np.asarray(sec3.cross_sections, dtype=float)
    if e.size < 2:
        return
    nz = np.flatnonzero(x > 0)
    if nz.size == 0:
        return
    lo, hi = float(e[nz[0]]), float(e[-1])
    gaps = []
    if grid[0] > lo * (1 + 1e-6) and grid[0] > 1e-5 * (1 + 1e-6):
        gaps.append(f"starts at {grid[0]:.4g} eV, MF3 is non-zero from {lo:.4g} eV")
    if grid[-1] < hi * (1 - 1e-6):
        gaps.append(f"stops at {grid[-1]:.4g} eV, MF3 goes to {hi:.4g} eV")
    if gaps:
        out.append(CovarianceFinding(
            "grid_coverage", NOTE, "the covariance " + "; ".join(gaps), loc,
            {"cov_range": [float(grid[0]), float(grid[-1])], "mf3_range": [lo, hi]}))


# ---------------------------------------------------------------------------
# Completeness (C5): the blocks and references a section implies
# ---------------------------------------------------------------------------


def _mat1(sub, own) -> int:
    """MAT1 as kika reads it: the file's own MAT is the same material as 0.

    ENDF-6 §33.3.1 b.1 asks for MAT1=0; some evaluations write their own MAT
    (ENDF/B-VIII.1 Np-237, JEFF-4.0 H-1), and ``MF33MT.to_xs_covmat`` already
    treats that as intra-material. A ``mat1_is_own_mat`` note says it happened.
    """
    mat1 = int(sub.mat1 or 0)
    return 0 if own and mat1 == int(own) else mat1


def _overlap(a1, a2, b1, b2) -> bool:
    return None not in (a1, a2, b1, b2) and a1 < b2 and b1 < a2


def _has_self(sec, mt: int) -> bool:
    return any(_mat1(s, sec._mat) == 0 and int(s.mt1 or 0) == mt for s in sec.subsections)


def _lty0_records(sec, mt: int):
    return [nc for s in sec.subsections if _mat1(s, sec._mat) == 0 and int(s.mt1 or 0) == mt
            for nc in s.nc_records if nc.lty == 0]


def _material_notes(sec, mf_number: int, mt: int, out) -> None:
    """MAT1 written as the file's own MAT, and blocks with another material."""
    mat = sec._mat
    own = sorted(int(s.mt1 or 0) for s in sec.subsections if mat and int(s.mat1 or 0) == int(mat))
    head = CovarianceLocation(mat=mat, mf=mf_number, mt=mt)
    if own:
        out.append(CovarianceFinding(
            "mat1_is_own_mat", NOTE,
            f"{len(own)} subsection(s) write MAT1={mat}, the file's own MAT, where ENDF-6 "
            "§33.3.1 asks for MAT1=0; read as the same material, as kika does", head,
            {"mt1": own}))
    external = sorted({(int(s.mat1), int(s.mt1 or 0)) for s in sec.subsections
                       if _mat1(s, mat) != 0})
    if external:
        out.append(CovarianceFinding(
            "external_material", NOTE,
            f"{len(external)} block(s) with another material (MAT1/MT1 "
            + ", ".join(f"{a}/{b}" for a, b in external[:_EVIDENCE_ITEMS])
            + "): not evaluable from one file, and their partners must be in MAT1's own "
            "file (ENDF-6 §33.3.2 b)", head, {"mat1_mt1": [list(p) for p in external]}))


def _completeness33(ctx, mf_number: int, sections: Dict[int, MF33MT], usable, out) -> None:
    """C5 for MF31/MF33 (ENDF-6 §33.3.2): partners, lumped reactions, NC references."""
    lumped_into: Dict[int, List[int]] = {}
    for mt, sec in sections.items():
        if sec._mtl:
            lumped_into.setdefault(int(sec._mtl), []).append(mt)

    for mt, sec in sorted(sections.items()):
        mat = sec._mat
        head = CovarianceLocation(mat=mat, mf=mf_number, mt=mt)
        if sec._mtl:
            mtl = int(sec._mtl)
            if mtl not in LUMPED_MT:
                out.append(CovarianceFinding(
                    "unresolved_reference", DEFECT,
                    f"MTL={mtl} names a lumped reaction outside 851-870, the range ENDF-6 "
                    "§33.2.3 reserves for them", head, {"mtl": mtl}))
            elif mtl not in sections:
                out.append(CovarianceFinding(
                    "unresolved_reference", DEFECT,
                    f"a component of the lumped reaction MT{mtl}, which has no section: the "
                    "component carries no covariance and the lump that should carry it is "
                    "missing", head, {"mtl": mtl}))
            continue
        if not sec.subsections:
            continue
        if mt in LUMPED_MT and mt not in lumped_into:
            out.append(CovarianceFinding(
                "lumped_without_components", WARN,
                f"the lumped reaction MT{mt} is not named as MTL by any section, so the cross "
                "section its covariance is relative to cannot be summed", head))
        _material_notes(sec, mf_number, mt, out)
        if not _has_self(sec, mt):
            out.append(CovarianceFinding(
                "missing_self_block", DEFECT,
                f"no subsection (MT{mt}, MT{mt}): the section states covariances of MT{mt} "
                "with other reactions but not its own variance (ENDF-6 §33.3.2 a.1)", head,
                {"mt1": sorted(int(s.mt1 or 0) for s in sec.subsections)}))
        for sub in sec.subsections:
            mat1, mt1 = _mat1(sub, mat), int(sub.mt1 or 0)
            loc = CovarianceLocation(mat=mat, mf=mf_number, mt=mt, mat1=mat1, mt1=mt1)
            if mat1 == 0 and mt1 != mt:
                _partner33(ctx, sections, usable, sec, loc, out)
            if sub.nc_records:
                _nc_references(sections, sec, sub, loc, out)


def _partner33(ctx, sections, usable, sec, loc, out) -> None:
    mt, mt1 = loc.mt, loc.mt1
    partner = sections.get(mt1)
    if partner is None or partner._mtl:
        why = (f"MT{mt1} has no section in this file" if partner is None else
               f"MT{mt1} is a component of the lumped MT{partner._mtl} and has no covariance")
        out.append(CovarianceFinding(
            "missing_partner", DEFECT,
            f"the covariance with MT{mt1} is stated, but {why}: its variance, and |rho| of "
            "this block, are undefined (ENDF-6 §33.3.2 a.1)", loc))
        return
    if mt1 > mt:
        return
    mirror = [s for s in partner.subsections
              if _mat1(s, partner._mat) == 0 and int(s.mt1 or 0) == mt]
    if not mirror:
        out.append(CovarianceFinding(
            "cross_block_below_diagonal", NOTE,
            f"stored in MT{mt} with MT1={mt1} < MT; ENDF-6 §33.3.1 b.4 gives a cross block "
            f"once, in the section of the lower MT. Nothing is lost: kika places it by "
            "transposition", loc))
        return
    a, b = usable.get((mt, 0, mt1)), usable.get((mt1, 0, mt))
    if a is None or b is None:
        return
    recs_a, recs_b = [r for _, r in a], [r for _, r in b]
    grid = sorted({e for r in recs_a + recs_b for g in _grids(r) for e in g})
    ra = _assemble33(sec, recs_a, mt, mt1, ctx.xs, grid)
    rb = _assemble33(partner, recs_b, mt1, mt, ctx.xs, grid)
    if ra is None or rb is None:
        return
    _mirror_finding(ra[0], rb[0], ra[2], rb[2], loc, f"MT{mt1}xMT{mt}", out)


def _mirror_finding(a: np.ndarray, b: np.ndarray, rel_a: bool, rel_b: bool, loc,
                    other: str, out) -> None:
    """A cross block stated twice, as (X, Y) and as (Y, X): do they agree?"""
    scale = max(float(np.max(np.abs(a))) if a.size else 0.0,
                float(np.max(np.abs(b))) if b.size else 0.0)
    diff = float(np.max(np.abs(a - b.T))) / scale if scale > 0 and a.shape == b.T.shape else 0.0
    if rel_a == rel_b and a.shape == b.T.shape and diff <= ASYM_DEFECT:
        out.append(CovarianceFinding(
            "symmetric_block_repeated", NOTE,
            f"also stated as {other}, and the two agree (max difference {diff:.1e} of the "
            "largest element): redundant, not contradictory", loc, {"relative_difference": diff}))
        return
    out.append(CovarianceFinding(
        "symmetric_block_conflict", DEFECT,
        f"also stated as {other}, and the two disagree ("
        + (f"max difference {diff:.2e} of the largest element"
           if rel_a == rel_b else "one is relative, the other absolute")
        + "); kika places both in the same joint block, and the one placed last wins "
        "without a word", loc,
        {"relative_difference": diff, "relative": [rel_a, rel_b]}))


def _derives(sections, start: int, target: int) -> bool:
    """True if ``start``'s LTY=0 chain reaches ``target``."""
    stack, seen = [start], set()
    while stack:
        mt = stack.pop()
        if mt in seen:
            continue
        seen.add(mt)
        sec = sections.get(mt)
        if sec is None:
            continue
        for nc in _lty0_records(sec, mt):
            for x in nc.xmti:
                m = int(round(x))
                if m == target:
                    return True
                stack.append(m)
    return False


def _nc_references(sections, sec, sub, loc, out) -> None:
    """NC sub-subsections: where they may stand, and whether what they name exists."""
    mt = loc.mt
    is_self = loc.mat1 == 0 and loc.mt1 == mt
    lty0 = [(k, nc) for k, nc in enumerate(sub.nc_records) if nc.lty == 0]
    for k, nc in enumerate(sub.nc_records):
        nloc = CovarianceLocation(mat=loc.mat, mf=loc.mf, mt=mt, mat1=loc.mat1, mt1=loc.mt1, nc=k)
        if nc.lty not in VALID_LTY:
            continue  # lty_invalid (C2)
        if nc.lty in (0, 1) and not is_self:
            out.append(CovarianceFinding(
                "nc_misplaced", DEFECT,
                f"an NC sub-subsection with LTY={nc.lty} in a cross block; ENDF-6 §33.3.2 "
                "allows it only in the self block (MT, MT)"
                + ("; kika would put the derived self covariance of this MT here"
                   if nc.lty == 0 else ""), nloc, {"lty": nc.lty}))
        if nc.lty == 0:
            for x in nc.xmti:
                mti = int(round(x))
                target = sections.get(mti)
                why = None
                if mti == mt:
                    why = "is the derived reaction itself"
                elif target is None:
                    why = "has no section in this file"
                elif target._mtl:
                    why = (f"is a component of the lumped MT{target._mtl}, which carries no "
                           "covariance of its own")
                elif not _has_self(target, mti):
                    why = f"has no self covariance (MT{mti}, MT{mti})"
                if why is not None:
                    out.append(CovarianceFinding(
                        "unresolved_reference", DEFECT,
                        f"NC LTY=0 derives this covariance from MT{mti}, which {why}; kika "
                        "leaves that term out, so the derived covariance comes out smaller "
                        "than the file states", nloc, {"xmti": mti}))
                    continue
                chained = [n for n in _lty0_records(target, mti)
                           if _overlap(nc.e1, nc.e2, n.e1, n.e2)]
                if chained:
                    out.append(CovarianceFinding(
                        "nc_chained", DEFECT,
                        f"NC LTY=0 derives this covariance from MT{mti}, which is itself "
                        f"derived (LTY=0) over {chained[0].e1:.6g}-{chained[0].e2:.6g} eV, "
                        f"overlapping {nc.e1:.6g}-{nc.e2:.6g} eV; ENDF-6 §33.3.1 does not "
                        "allow a constituent derived in the same range", nloc,
                        {"xmti": mti, "range": [nc.e1, nc.e2],
                         "constituent_range": [chained[0].e1, chained[0].e2]}))
                elif _derives(sections, mti, mt):
                    out.append(CovarianceFinding(
                        "nc_chained", NOTE,
                        f"MT{mti} is derived (LTY=0), in another energy range, from a chain "
                        f"that leads back to MT{mt}. The format allows it; kika's resolver "
                        "ignores E1-E2 and cuts the cycle, so what it assembles for this MT "
                        "may differ from what the file states", nloc, {"xmti": mti}))
        elif nc.lty == 1:
            own = int(loc.mat or 0)
            mats, mts = int(nc.mats or 0), int(nc.mts or 0)
            if mats in (0, own):
                out.append(CovarianceFinding(
                    "unresolved_reference", DEFECT,
                    f"LTY=1 names MATS={mats}, this material; ENDF-6 §33.3.3 reserves LTY=1 "
                    "for ratios to a standard of another material", nloc, {"mats": mats}))
            elif not any(int(s.mat1 or 0) == mats and int(s.mt1 or 0) == mts
                         for s in sec.subsections):
                out.append(CovarianceFinding(
                    "unresolved_reference", DEFECT,
                    f"LTY=1 names the standard MAT{mats}/MT{mts}, but this section has no "
                    f"subsection (MT{mt}; MAT{mats}, MT{mts}) for the covariance with it "
                    "(ENDF-6 §33.3.2 a.4)", nloc, {"mats": mats, "mts": mts}))
        if nc.lty in (1, 2, 3):
            out.append(CovarianceFinding(
                "ratio_to_standard", NOTE,
                f"LTY={nc.lty} ties this covariance to the standard MAT{nc.mats}/MT{nc.mts} "
                "of another file; kika does not resolve it, so that component is missing "
                "from what it assembles", nloc, {"lty": nc.lty, "mats": nc.mats, "mts": nc.mts}))

    for (k, a), (_, b) in zip(lty0, lty0[1:]):
        if _overlap(a.e1, a.e2, b.e1, b.e2):
            out.append(CovarianceFinding(
                "nc_ranges_overlap", DEFECT,
                f"two NC LTY=0 sub-subsections overlap ({a.e1:.6g}-{a.e2:.6g} and "
                f"{b.e1:.6g}-{b.e2:.6g} eV); ENDF-6 §33.3.1 requires disjoint ranges",
                CovarianceLocation(mat=loc.mat, mf=loc.mf, mt=mt, mat1=loc.mat1, mt1=loc.mt1,
                                   nc=k + 1)))
    rules = {tuple(zip(np.round(nc.ci, 9), [int(round(x)) for x in nc.xmti])) for _, nc in lty0}
    if len(rules) > 1:
        out.append(CovarianceFinding(
            "nc_only_first_resolved", NOTE,
            f"{len(lty0)} NC LTY=0 sub-subsections with different sums of reactions; kika "
            "resolves only the first and applies it over the whole range", loc,
            {"rules": [[[float(c), m] for c, m in r] for r in sorted(rules)]}))


def _completeness34(sections, usable, out) -> None:
    """C5 for MF34 (ENDF-6 §34.2): partners of every block, and orders with no variance."""
    selfs = {(mt, l) for (mt, l, mt1, l1) in usable if mt1 == mt and l1 == l}
    for mt, sec in sorted(sections.items()):
        mat = sec._mat
        if not sec.subsections:
            continue
        head = CovarianceLocation(mat=mat, mf=34, mt=mt)
        _material_notes(sec, 34, mt, out)
        if not _has_self(sec, mt):
            out.append(CovarianceFinding(
                "missing_self_block", DEFECT,
                f"no subsection MT1={mt}: none of the Legendre orders of MT{mt} has a "
                "variance, yet covariances with other reactions are stated", head))
        lmin = _l_min(sec._ltt)
        for sub in sec.subsections:
            mat1, mt1 = _mat1(sub, mat), int(sub.mt1 or 0)
            if mat1 != 0:
                continue
            loc = CovarianceLocation(mat=mat, mf=34, mt=mt, mat1=0, mt1=mt1)
            if mt1 == mt:
                declared = set(range(lmin, lmin + int(sub.nl or 0)))
                silent = sorted(declared - {l for (m, l) in selfs if m == mt})
                if silent and len(silent) < len(declared):
                    out.append(CovarianceFinding(
                        "order_without_variance", NOTE,
                        f"NL declares a_{silent[0]}"
                        + (f" and {len(silent) - 1} more" if len(silent) > 1 else "")
                        + " with no variance block; legal (ENDF-6 §34.2: not all L need be "
                        "given), and those orders are left unperturbed", loc, {"l": silent}))
            else:
                partner = sections.get(mt1)
                if partner is None:
                    out.append(CovarianceFinding(
                        "missing_partner", DEFECT,
                        f"covariances with MT{mt1} are stated, but MF34 has no section MT{mt1}",
                        loc))
                    continue
                if mt1 < mt:
                    _mirror34(sections, usable, sec, partner, sub, loc, out)
            for ss in sub.sub_subsections:
                l, l1 = int(ss.l or 0), int(ss.l1 or 0)
                if (mt1, l1) == (mt, l) or (mt, l, mt1, l1) not in usable:
                    continue
                absent = [f"a_{x} of MT{m}" for m, x in ((mt, l), (mt1, l1))
                          if (m, x) not in selfs]
                if not absent:
                    continue
                zero = all(not np.any(_decoded_values(r)) for _, r in usable[(mt, l, mt1, l1)][1])
                out.append(CovarianceFinding(
                    "missing_partner", NOTE if zero else DEFECT,
                    f"a covariance between a_{l} of MT{mt} and a_{l1} of MT{mt1} is stated, but "
                    + " and ".join(absent) + " ha" + ("ve" if len(absent) > 1 else "s")
                    + " no variance block"
                    + (", and the block is all zeros, so nothing follows" if zero else
                       ": |rho| is undefined, and in the joint it is a row with covariances "
                       "and no variance"),
                    CovarianceLocation(mat=mat, mf=34, mt=mt, mat1=0, mt1=mt1, l=l, l1=l1),
                    {"absent": absent}))


def _decoded_values(rec) -> np.ndarray:
    for name in ("matrix", "rect_matrix", "f_table_k"):
        vals = getattr(rec, name, None)
        if vals:
            return np.asarray(vals, dtype=float)
    return np.zeros(0)


def _mirror34(sections, usable, sec, partner, sub, loc, out) -> None:
    mt, mt1 = loc.mt, loc.mt1
    mirror = [s for s in partner.subsections
              if _mat1(s, partner._mat) == 0 and int(s.mt1 or 0) == mt]
    if not mirror:
        out.append(CovarianceFinding(
            "cross_block_below_diagonal", NOTE,
            f"stored in MT{mt} with MT1={mt1} < MT; ENDF-6 §34.2 gives the subsections for "
            "MT1 >= MT. Nothing is lost: kika places it by transposition", loc))
        return
    for ss in sub.sub_subsections:
        l, l1 = int(ss.l or 0), int(ss.l1 or 0)
        a, b = usable.get((mt, l, mt1, l1)), usable.get((mt1, l1, mt, l))
        if a is None or b is None:
            continue
        recs_a, recs_b = [r for _, r in a[1]], [r for _, r in b[1]]
        grid = sorted({e for r in recs_a + recs_b for g in _grids(r) for e in g})
        ra, rb = _assemble34(sec, recs_a, grid), _assemble34(partner, recs_b, grid)
        if ra is None or rb is None:
            continue
        sloc = CovarianceLocation(mat=loc.mat, mf=34, mt=mt, mat1=0, mt1=mt1, l=l, l1=l1)
        _mirror_finding(ra[0], rb[0], ra[2], rb[2], sloc, f"MT{mt1} L{l1} x MT{mt} L{l}", out)


# ---------------------------------------------------------------------------
# Cross blocks and LS=1, shared
# ---------------------------------------------------------------------------


def _check_cross_block(block: np.ndarray, d_row: np.ndarray, d_col: np.ndarray, grid, loc, out,
                       extra: Optional[dict]) -> None:
    live_r, live_c = d_row > 0, d_col > 0
    dead = (~live_r[:, None] | ~live_c[None, :]) & (block != 0)
    if dead.any():
        rows = np.flatnonzero(dead.any(axis=1))
        out.append(CovarianceFinding(
            "covariance_without_variance", DEFECT,
            f"{int(dead.sum())} covariances sit where one of the two variances is zero or "
            "negative (|rho| is undefined)", loc,
            {"n": int(dead.sum()), "row_bins": _bins(grid, rows[:_EVIDENCE_ITEMS])}))
    if not live_r.any() or not live_c.any():
        return
    sub = block[np.ix_(live_r, live_c)]
    corr = sub / np.sqrt(np.outer(d_row[live_r], d_col[live_c]))
    _rho_finding(corr, np.flatnonzero(live_r), np.flatnonzero(live_c), grid, loc, out,
                 self_block=False, extra=extra)


def _triangle_rho(rec, diag_row, diag_col) -> Optional[float]:
    """max |rho| of the stored LS=1 triangle alone, on the record's own grid."""
    g = list(rec.energies)
    dr, dc = diag_row(g), diag_col(g)
    if dr is None or dc is None:
        return None
    m = len(g) - 1
    vals = np.asarray(rec.matrix, dtype=float)
    if vals.size != m * (m + 1) // 2 or dr.size != m or dc.size != m:
        return None
    i, j = np.triu_indices(m)
    denom = dr[i] * dc[j]
    ok = denom > 0
    if not ok.any():
        return None
    return float(np.max(np.abs(vals[ok]) / np.sqrt(denom[ok])))


def _ls1_finding(loc: CovarianceLocation, k: int, rec, tri: Optional[float], out) -> None:
    rloc = dataclasses.replace(loc, ni=k, nc=None, lb=5, ls=1)
    summary = ("LB=5 LS=1 (a symmetric triangle) in a cross block, which is not symmetric; "
               "kika mirrors the triangle into the lower half")
    evidence = {}
    if not np.any(np.asarray(rec.matrix, dtype=float)):
        out.append(CovarianceFinding(
            "ls1_in_cross_block", NOTE, summary + ", but the triangle is all zeros, so the "
            "mirror changes nothing", rloc, evidence))
        return
    if tri is not None:
        evidence["stored_triangle_max_abs_rho"] = tri
        summary += f" (the stored triangle alone has |rho| up to {tri:.4g})"
    out.append(CovarianceFinding("ls1_in_cross_block", DEFECT, summary, rloc, evidence))


# ---------------------------------------------------------------------------
# MF34
# ---------------------------------------------------------------------------


def _assemble34(sec, recs, grid=None):
    """(matrix, grid, relative, partial) or None for one (L, L1) sub-subsection."""
    comps = []
    for rec in recs:
        lb = int(rec.lb)
        if lb in (0, 1, 2):
            if (rec.lt or 0) > 0:
                continue
            m, g = sec._decode_lb012_matrix(rec)
            comps.append((lb, m, list(g), None))
        elif lb == 5:
            m, g = sec._decode_lb5_matrix(rec)
            comps.append((lb, m, list(g), None))
        elif lb == 6:
            m, rg, cg = sec._decode_lb6_matrix(rec)
            comps.append((lb, m, list(rg), list(cg)))
    if not comps:
        return None
    kinds = {lb == 0 for lb, *_ in comps}
    partial = len(kinds) == 2
    if partial:
        comps = [c for c in comps if c[0] != 0]
    relative = comps[0][0] != 0
    if grid is None:
        grid = sorted({e for _, _, rg, cg in comps for e in rg + (cg or [])})
    grid = list(grid)
    if len(grid) < 2:
        return None
    total = np.zeros((len(grid) - 1, len(grid) - 1))
    for _, m, rg, cg in comps:
        total += sec._project_matrix_piecewise_constant(
            m, rg, grid, is_lb6=cg is not None, native_col_point_grid=cg)
    return total, grid, relative, partial


def _l_min(ltt) -> int:
    return 0 if int(ltt or 1) in (2, 3) else 1


def _check_mf34(ctx: _Context, mf_obj, out: List[CovarianceFinding]) -> None:
    from ..parsers.parse_mf34 import _expected_subsubsection_count

    sections = dict(mf_obj.mt)
    usable: Dict[Tuple[int, int, int, int], Tuple[object, List[Tuple[int, object]]]] = {}

    for mt, sec in sorted(sections.items()):
        mat = sec._mat
        head = CovarianceLocation(mat=mat, mf=34, mt=mt)
        _count(sec._nmt1, len(sec.subsections), "NMT1", head, out)
        lmin = _l_min(sec._ltt)
        for sub in sec.subsections:
            mt1, mat1 = int(sub.mt1 or 0), _mat1(sub, mat)
            loc = CovarianceLocation(mat=mat, mf=34, mt=mt, mat1=mat1, mt1=mt1)
            nss = _expected_subsubsection_count(sub.nl, sub.nl1, mt, mt1, sec._ltt)
            _count(nss, len(sub.sub_subsections), "NSS (from NL, NL1)", loc, out)
            nl, nl1 = int(sub.nl or 0), int(sub.nl1 or 0)
            seen = set()
            for ss in sub.sub_subsections:
                l, l1 = int(ss.l or 0), int(ss.l1 or 0)
                sloc = CovarianceLocation(mat=mat, mf=34, mt=mt, mat1=mat1, mt1=mt1, l=l, l1=l1)
                if not (lmin <= l < lmin + nl and lmin <= l1 < lmin + nl1):
                    out.append(CovarianceFinding(
                        "l_out_of_range", DEFECT,
                        f"(L, L1) = ({l}, {l1}) is outside what NL={nl}, NL1={nl1} and "
                        f"LTT={sec._ltt} allow (L from {lmin})", sloc,
                        {"nl": nl, "nl1": nl1, "ltt": sec._ltt}))
                if mt1 == mt and l > l1:
                    out.append(CovarianceFinding(
                        "l_out_of_range", DEFECT,
                        f"MT1 = MT stores only L <= L1, found ({l}, {l1})", sloc))
                if (l, l1) in seen:
                    out.append(CovarianceFinding(
                        "duplicate_block", DEFECT, "this (L, L1) sub-subsection appears twice", sloc))
                seen.add((l, l1))
                if ss.lct is not None and int(ss.lct) not in VALID_LCT:
                    out.append(CovarianceFinding(
                        "lct_invalid", WARN, f"LCT={ss.lct} is not 0, 1 or 2", sloc, {"lct": ss.lct}))
                _count(ss.ni, len(ss.records), "NI", sloc, out)
                good = []
                for k, rec in enumerate(ss.records):
                    rloc = CovarianceLocation(mat=mat, mf=34, mt=mt, mat1=mat1, mt1=mt1, l=l, l1=l1,
                                              ni=k, lb=rec.lb, ls=rec.ls if rec.lb == 5 else None)
                    if _check_record(34, sec, rec, rloc, out):
                        good.append((k, rec))
                if good and mat1 == 0:
                    usable[(mt, l, mt1, l1)] = (sec, good)
                elif mat1 != 0:
                    for k, rec in good:
                        if rec.lb == 5 and rec.ls == 1:
                            _ls1_finding(sloc, k, rec, None, out)

    _completeness34(sections, usable, out)

    # Self-order blocks (MT, L) x (MT, L).
    self_grids: Dict[Tuple[int, int], List[float]] = {}
    for (mt, l, mt1, l1), (sec, good) in sorted(usable.items()):
        if mt1 != mt or l1 != l:
            continue
        loc = CovarianceLocation(mat=sec._mat, mf=34, mt=mt, mat1=0, mt1=mt, l=l, l1=l)
        res = _assemble34(sec, [r for _, r in good])
        if res is None:
            continue
        matrix, grid, relative, partial = res
        self_grids[(mt, l)] = grid
        if partial:
            out.append(CovarianceFinding(
                "mixed_absolute_relative", WARN,
                "the block mixes absolute (LB=0) and relative components; kika sums them as "
                "relative, so only the relative part was checked", loc))
        _check_self_block(matrix, grid, loc, out, good, sec)
        _mf34_magnitude(ctx, mt, l, matrix, grid, relative, loc, out)

    # Cross blocks: L != L1 or MT != MT1.
    for (mt, l, mt1, l1), (sec, good) in sorted(usable.items()):
        if mt1 == mt and l1 == l:
            continue
        loc = CovarianceLocation(mat=sec._mat, mf=34, mt=mt, mat1=0, mt1=mt1, l=l, l1=l1)
        ls1 = [(k, r) for k, r in good if r.lb == 5 and r.ls == 1]
        row_self = usable.get((mt, l, mt, l))
        col_self = usable.get((mt1, l1, mt1, l1))
        if row_self is None or col_self is None:
            for k, r in ls1:
                _ls1_finding(loc, k, r, None, out)
            continue
        recs = [r for _, r in good]
        grid = sorted({e for r in recs for g in _grids(r) for e in g}
                      | set(self_grids.get((mt, l), [])) | set(self_grids.get((mt1, l1), [])))
        res = _assemble34(sec, recs, grid)
        rr = _assemble34(row_self[0], [r for _, r in row_self[1]], grid)
        cc = _assemble34(col_self[0], [r for _, r in col_self[1]], grid)
        if res is None or rr is None or cc is None:
            continue
        tri = {}
        for k, r in ls1:
            t = _triangle_rho(
                r,
                lambda g: _diag34(row_self, g),
                lambda g: _diag34(col_self, g))
            tri[k] = t
            _ls1_finding(loc, k, r, t, out)
        extra = {"stored_triangle_max_abs_rho": max(tri.values())} if tri and all(
            v is not None for v in tri.values()) else None
        _check_cross_block(res[0], np.diag(rr[0]), np.diag(cc[0]), grid, loc, out, extra)


def _diag34(self_entry, grid):
    sec, good = self_entry
    res = _assemble34(sec, [r for _, r in good], list(grid))
    return None if res is None else np.diag(res[0])


def _mf34_magnitude(ctx, mt, l, matrix, grid, relative, loc, out) -> None:
    if l < 1:
        # a_0 = 1 by normalisation: a stated a_0 covariance (LTT=3) is not the
        # uncertainty of a coefficient bounded by |a_l| <= 1, so the bound says nothing.
        return
    sec4 = ctx.mf4.get(mt)
    if sec4 is None or not hasattr(sec4, "extract_legendre_coefficients"):
        ctx.unavailable.setdefault((34, "MF4"), []).append(mt)
        mts = ctx.unavailable[(34, "MF4")]
        for i, f in enumerate(out):
            if f.check == "central_values_unavailable" and f.location.mf == 34:
                out[i] = CovarianceFinding(f.check, f.level, f.summary, f.location,
                                           {"mts": sorted(set(mts))})
                return
        out.append(CovarianceFinding(
            "central_values_unavailable", NOTE,
            "uncertainty magnitudes not evaluable: MF4 not read or without Legendre coefficients",
            CovarianceLocation(mat=loc.mat, mf=34), {"mts": [mt]}))
        return
    g = np.asarray(grid, dtype=float)
    n_sub = 9
    sub_e = np.column_stack([np.linspace(g[c], g[c + 1], n_sub) for c in range(g.size - 1)]).T
    try:
        coeffs = sec4.extract_legendre_coefficients(sub_e.ravel(), max_legendre_order=max(l, 1),
                                                    out_of_range="zero")
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        out.append(CovarianceFinding(
            "central_values_unavailable", NOTE, f"MF4 a_{l} could not be evaluated: {exc}", loc))
        return
    vals = np.asarray(coeffs.get(l, np.zeros(sub_e.size)), dtype=float).reshape(sub_e.shape)
    width = g[1:] - g[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        central = np.where(width > 0, np.trapezoid(vals, sub_e, axis=1) / width, 0.0)
    # A relative sigma becomes absolute through the a_l it is relative to, and ENDF
    # does not pin which one inside a bin where a_l moves (O-16 at 6.5 MeV crosses
    # resonances). The smallest |a_l| the bin holds -- its average or any sampled
    # point -- gives the smallest sigma_abs the file can mean, so a breach of the
    # bound with it is certain, not an artefact of the averaging.
    reference = np.minimum(np.abs(central), np.min(np.abs(vals), axis=1))
    var = np.clip(np.diag(matrix), 0.0, None)
    sigma_abs = np.sqrt(var) * reference if relative else np.sqrt(var)
    valid = width > 0
    _legendre_bound(sigma_abs, reference, valid, grid, loc, out)
