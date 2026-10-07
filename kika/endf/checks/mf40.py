"""Layer 1 for MF40: covariances of the production cross sections of MF10.

An MF40 block is an MF33 block named by two (MT, LFS) pairs instead of two MTs
(ENDF-6 §40), so every MF33 check applies record by record and block by block,
with the final state carried in the location: structure (counts, LB, NT, grids,
LB=5 LS=1 in a cross block), the self block of each (MT, LFS) (variances, |rho|,
PSD in three levels) and |rho| of the cross blocks against the two self blocks.

What MF40 cannot be checked against is its central values: kika does not read
MF10, so magnitudes, and the absolute part of a block that mixes LB 0/8/9 with
relative records, are reported as not evaluable rather than guessed.

Measured on ENDF/B-VIII.1 (29 tapes, 152 sections) and JEFF-4.0 (440 tapes,
3 223 sections), 6-oct-2026: every record is LB=5 LS=1 save one LB=6, every
XMF1 is 10, and three subsections are cross blocks.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .findings import DEFECT, NOTE, CovarianceFinding, CovarianceLocation


def check_mf40(ctx, mf_obj, out: List[CovarianceFinding]) -> None:
    from .covariances import (RELATIVE_LB, VALID_LTY, _assemble33, _check_cross_block,
                              _check_record, _check_self_block, _count, _grids, _ls1_finding,
                              _triangle_rho)

    views: Dict[Tuple[int, int], object] = {}
    usable: Dict[Tuple[int, int, int, int, int], List[Tuple[int, object]]] = {}
    mat = None

    for mt, sec in sorted(mf_obj.mt.items()):
        mat = mat or sec._mat
        head = CovarianceLocation(mat=sec._mat, mf=40, mt=mt)
        _count(sec._ns, len(sec.states), "NS", head, out)
        seen_lfs = set()
        for state in sec.states:
            lfs = int(state.lfs)
            sloc = CovarianceLocation(mat=sec._mat, mf=40, mt=mt, lfs=lfs)
            if lfs in seen_lfs:
                out.append(CovarianceFinding(
                    "duplicate_block", DEFECT, f"final state LFS={lfs} appears twice", sloc))
            seen_lfs.add(lfs)
            _count(state.nl, len(state.subsections), "NL", sloc, out)
            view = state.as_mf33(mt, sec._mat, sec._za, sec._awr)
            views[(mt, lfs)] = view
            seen = set()
            has_self = False
            for sub in state.subsections:
                mat1 = int(sub.mat1 or 0)
                mat1 = 0 if sec._mat and mat1 == int(sec._mat) else mat1
                mt1, lfs1 = int(sub.mt1 or 0), int(sub.xlfs1 or 0)
                xmf1 = int(sub.xmf1 or 0)
                loc = CovarianceLocation(mat=sec._mat, mf=40, mt=mt, lfs=lfs, mat1=mat1,
                                         mt1=mt1, lfs1=lfs1)
                if (mat1, mt1, lfs1) in seen:
                    out.append(CovarianceFinding(
                        "duplicate_block", DEFECT, "this (MAT1, MT1, LFS1) subsection appears twice",
                        loc))
                seen.add((mat1, mt1, lfs1))
                if xmf1 not in (0, 10):
                    out.append(CovarianceFinding(
                        "external_material", NOTE,
                        f"a block with MF{xmf1} data (XMF1={xmf1}): not evaluable from MF40 alone",
                        loc, {"xmf1": xmf1}))
                    continue
                if mat1 == 0 and mt1 == mt and lfs1 == lfs:
                    has_self = True
                _count(sub.nc, len(sub.nc_records), "NC", loc, out)
                _count(sub.ni, len(sub.ni_records), "NI", loc, out)
                for k, nc in enumerate(sub.nc_records):
                    if nc.lty not in VALID_LTY:
                        out.append(CovarianceFinding(
                            "lty_invalid", DEFECT, f"LTY={nc.lty} is not defined",
                            CovarianceLocation(mat=sec._mat, mf=40, mt=mt, lfs=lfs, mat1=mat1,
                                               mt1=mt1, lfs1=lfs1, nc=k), {"lty": nc.lty}))
                good = []
                for k, rec in enumerate(sub.ni_records):
                    rloc = CovarianceLocation(mat=sec._mat, mf=40, mt=mt, lfs=lfs, mat1=mat1,
                                              mt1=mt1, lfs1=lfs1, ni=k, lb=rec.lb,
                                              ls=rec.ls if rec.lb == 5 else None)
                    if _check_record(40, view, rec, rloc, out):
                        good.append((k, rec))
                if good:
                    usable[(mt, lfs, mat1, mt1, lfs1)] = good
            if state.subsections and not has_self:
                out.append(CovarianceFinding(
                    "missing_self_block", DEFECT,
                    f"no subsection (MT{mt}/LFS{lfs}, MT{mt}/LFS{lfs}): the state's covariances "
                    "with others are given but not its own variance", sloc))

    # Self blocks.
    diag: Dict[Tuple[int, int], Tuple[np.ndarray, List[float]]] = {}
    partial_mts = []
    for (mt, lfs, mat1, mt1, lfs1), good in sorted(usable.items()):
        if (mat1, mt1, lfs1) != (0, mt, lfs):
            continue
        view = views[(mt, lfs)]
        loc = CovarianceLocation(mat=view._mat, mf=40, mt=mt, lfs=lfs, mat1=0, mt1=mt, lfs1=lfs)
        recs = [r for _, r in good]
        if any(int(r.lb) not in RELATIVE_LB for r in recs):
            partial_mts.append(mt)
        res = _assemble33(view, recs, mt, mt, None)
        if res is None:
            continue
        matrix, grid, _relative, _partial = res
        diag[(mt, lfs)] = (np.diag(matrix), grid)
        _check_self_block(matrix, grid, loc, out, good, view)

    # Cross blocks.
    for (mt, lfs, mat1, mt1, lfs1), good in sorted(usable.items()):
        if (mat1, mt1, lfs1) == (0, mt, lfs):
            continue
        view = views[(mt, lfs)]
        loc = CovarianceLocation(mat=view._mat, mf=40, mt=mt, lfs=lfs, mat1=mat1, mt1=mt1,
                                 lfs1=lfs1)
        ls1 = [(k, r) for k, r in good if r.lb == 5 and r.ls == 1]
        partner = views.get((mt1, lfs1)) if mat1 == 0 else None
        if mat1 == 0 and partner is None:
            out.append(CovarianceFinding(
                "missing_partner", DEFECT,
                f"the covariance with MT{mt1}/LFS{lfs1} is stated, but that state has no "
                "covariance in this file: its variance, and |rho| of this block, are undefined",
                loc))
        if partner is None or (mt, lfs) not in diag or (mt1, lfs1) not in diag:
            for k, r in ls1:
                _ls1_finding(loc, k, r, None, out)
            continue
        recs = [r for _, r in good]
        grid = sorted({e for r in recs for g in _grids(r) for e in g}
                      | set(diag[(mt, lfs)][1]) | set(diag[(mt1, lfs1)][1]))
        res = _assemble33(view, recs, mt, mt1, None, grid)
        row = _assemble33(view, [r for _, r in usable[(mt, lfs, 0, mt, lfs)]], mt, mt, None, grid)
        col = _assemble33(partner, [r for _, r in usable[(mt1, lfs1, 0, mt1, lfs1)]],
                          mt1, mt1, None, grid)
        if res is None or row is None or col is None:
            continue
        tri = {}
        for k, r in ls1:
            def d_of(v, m, key):
                return lambda g: (lambda a: None if a is None else np.diag(a[0]))(
                    _assemble33(v, [x for _, x in usable[key]], m, m, None, list(g)))
            t = _triangle_rho(r, d_of(view, mt, (mt, lfs, 0, mt, lfs)),
                              d_of(partner, mt1, (mt1, lfs1, 0, mt1, lfs1)))
            tri[k] = t
            _ls1_finding(loc, k, r, t, out)
        extra = {"stored_triangle_max_abs_rho": max(tri.values())} if tri and all(
            v is not None for v in tri.values()) else None
        _check_cross_block(res[0], np.diag(row[0]), np.diag(col[0]), grid, loc, out, extra)

    if mf_obj.mt:
        why = "kika does not read MF10, the production cross sections MF40 is about"
        if partial_mts:
            why += ("; blocks with absolute (LB 0/8/9) records were checked on their relative "
                    "part only")
        out.append(CovarianceFinding(
            "central_values_unavailable", NOTE, f"uncertainty magnitudes not evaluable: {why}",
            CovarianceLocation(mat=mat, mf=40),
            {"mts": sorted(mf_obj.mt), "mixed_mts": sorted(set(partial_mts))}))
