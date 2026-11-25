"""ENDF MF32 → the resolved-range parameter covariance FUDGE writes to GNDS.

FUDGE (``ENDF_ITYPE_0_Misc.readMF32``) links each resonance table once (the
Breit-Wigner table, or each R-Matrix spin group's) and states the matrix over
every cell of those tables, zeros where nothing is varied: Na-23 is 23 x 6 =
138 rows. kika's ENDF decoder keeps MF32's own rows, one link per resonance,
which a GNDS reader cannot place. This module rebuilds FUDGE's layout from the
MF32 text the decoder kept, with FUDGE's arithmetic, against the tables kika's
GNDS writer writes, and the ``ENDFconversionFlags`` FUDGE leaves for the way
back (``LRF3``, ``LCOMP=``, ``NDIGIT=``, ``sortByL``).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from kika.nuclear_data.model import (DataLink, ParameterCovariance,
                                     ParameterCovarianceMatrix, ParameterLink)

__all__ = ["fudgeParameterCovariances", "PARAMETER_COVARIANCE_HREF"]

#: The label FUDGE gives the resolved range's covariance, and the href its flags hang on.
LABEL = "resolved resonances"
PARAMETER_COVARIANCE_HREF = (
    f"$covariances#/covarianceSuite/parameterCovariances/parameterCovariance[@label='{LABEL}']"
    f"/parameterCovarianceMatrix[@label='eval']")


class _Lines:
    """FUDGE's ``funkyF``/``funkyFI`` walk over an MF32 section's records."""

    def __init__(self, lines: Sequence[str]):
        self.lines = [line for line in lines if line[70:72].strip() == "32"]
        self.index = 0

    def raw(self) -> str:
        line = self.lines[self.index]
        self.index += 1
        return line

    def floats(self) -> List[float]:
        from kika._records import parse_number

        line = self.raw()
        out = []
        for k in range(6):
            value = parse_number(line[11 * k:11 * (k + 1)])
            out.append(0.0 if value is None else float(value))
        return out

    def ints(self) -> List[float]:
        values = self.floats()
        return values[:2] + [int(v) for v in values[2:]]


def _intg(lines: _Lines, nnn: int, nm: int, ndigit: int) -> np.ndarray:
    """``read_LCOMP2_correlation``: the INTG records as a correlation matrix, FUDGE's arithmetic."""
    from kika.endf.classes.mf32.mf32mt151 import IntgMatrix

    raw = [lines.raw()[:66] for _ in range(nm)]
    entries = IntgMatrix(ndigit=ndigit, nnn=nnn, nm=nm, lines=raw).entries
    matrix = np.zeros((nnn, nnn))
    for row, col, vals in entries:
        vals = list(vals)[:row - col]
        row -= 1
        col -= 1
        if row >= nnn or col >= row:
            raise ValueError(f"an INTG record names ({row + 1}, {col + 1}) outside NNN={nnn}")
        matrix[row, col:col + len(vals)] = vals
    for idx in range(nnn):
        matrix[idx, idx:] = matrix[idx:, idx]
    matrix[matrix > 0] += 0.5
    matrix[matrix < 0] -= 0.5
    matrix += np.eye(nnn, dtype="float") * 10 ** ndigit
    matrix /= float(10 ** ndigit)
    return matrix


def _lcomp1(lines: _Lines, dim: int, size: int) -> np.ndarray:
    data: List[float] = []
    full, rest = divmod(size, 6)
    for _ in range(full):
        data += lines.floats()
    if rest:
        data += lines.floats()[:rest]
    matrix = np.zeros((dim, dim))
    start, length = 0, dim
    for i in range(dim):
        matrix[i, i:] = matrix[i:, i] = data[start:start + length]
        start += length
        length -= 1
    return matrix


def _swapRows(matrix, i1, i2, nrows):
    rows = matrix[i1:i1 + nrows].copy()
    matrix[i1:i1 + nrows] = matrix[i2:i2 + nrows]
    matrix[i2:i2 + nrows] = rows
    cols = matrix[:, i1:i1 + nrows].copy()
    matrix[:, i1:i1 + nrows] = matrix[:, i2:i2 + nrows]
    matrix[:, i2:i2 + nrows] = cols


def _rearrangeRows(matrix, i1, i2, nrows):
    rows = matrix[i1:i2].copy()
    matrix[i1:i1 + nrows] = matrix[i2:i2 + nrows]
    matrix[i1 + nrows:i2 + nrows] = rows
    cols = matrix[:, i1:i2].copy()
    matrix[:, i1:i1 + nrows] = matrix[:, i2:i2 + nrows]
    matrix[:, i1 + nrows:i2 + nrows] = cols


def _consistent(mf2, mf32, matrix, mpar, lcompZero):
    """``check_MF32_consistency``: MF32's resonances onto the table rows, zeros for those it omits."""
    if not set(mf32).issubset(mf2):
        raise ValueError("MF32's resonance parameters are not File 2's")
    if len(mf2) != len(mf32):
        add = 0
        for resonance in mf2:
            if resonance not in mf32:
                mf32.append(resonance)
                add += mpar if mpar else len(resonance)
        dim = len(matrix) + add
        grown = np.zeros((dim, dim))
        grown[:len(matrix), :len(matrix)] = matrix
        matrix = grown
    if mf32 != mf2 or lcompZero:
        extended = mf32 + [v for v in mf2 if v not in mf32]
        mf2Starts, mf32Starts, total = [], [], 0
        for sub in mf2:
            mf2Starts.append(total)
            total += len(sub)
        total = 0
        for sub in extended:
            mf32Starts.append(total)
            total += len(sub)
        for i1 in range(len(mf2)):
            i2 = extended.index(mf2[i1])
            if i2 != i1:
                if mpar:
                    _swapRows(matrix, mpar * i1, mpar * i2, mpar)
                    extended[i1], extended[i2] = extended[i2], extended[i1]
                else:
                    _rearrangeRows(matrix, mf2Starts[i1], mf32Starts[i2], len(mf2[i1]))
                    extended.insert(i1, extended.pop(i2))
    return matrix


def _toModel(matrix, tables, rowHref, dap):
    """*matrix* over *tables*, one link per non-empty table, absolute as in every NNDC file;
    a scattering-radius uncertainty first, as its own row (FUDGE's ``scatteringRadius`` link)."""
    links, values, start = [], [], 0
    if dap:
        links.append(ParameterLink(label="scatteringRadius",
                                   href="/reactionSuite/resonances/scatteringRadius",
                                   nParameters=1, matrixStartIndex=0))
        grown = np.zeros((len(matrix) + 1, len(matrix) + 1))
        grown[0, 0] = dap[0] ** 2
        grown[1:, 1:] = matrix
        matrix, start = grown, 1
    for label, href, rows in tables:
        n = sum(len(row) for row in rows)
        for row in rows:
            values += list(row)
        if n == 0:
            continue
        links.append(ParameterLink(label=label, href=href, nParameters=n, matrixStartIndex=start))
        start += n
    if len(values) + bool(dap) != len(matrix):
        raise ValueError(f"{len(matrix)} covariance rows for {len(values)} table cells")
    # FUDGE's current code makes the matrix relative where it can; all 130 NNDC
    # files are absolute, so the version that wrote them did not.
    form = ParameterCovarianceMatrix(matrix=matrix, parameters=links, isRelative=False,
                                     label="eval")
    return ParameterCovariance(label=LABEL, rowData=DataLink(href=rowHref), form=form)


def _breitWigner(lines, lrf, tables):
    (label, href, rows), = tables
    mf2 = [(row[0], row[4], row[5]) for row in rows]
    flags = []
    _, _, _, lcomp, nls, isr = lines.ints()
    dap = [10 * lines.floats()[1]] if isr else []
    if lcomp == 0:
        flags.append("LCOMP=0")
        resonances, covars = [], []
        for _ in range(nls):
            _, _, _, _, _, nrs = lines.ints()
            for _ in range(nrs):
                resonances.append(lines.floats())
                covars.append(lines.floats() + lines.floats())
        terms = list(zip(*covars))
        dEsq, dNsq, dNdG, dGsq, dNdF, dGdF, dFsq, _, _, _, dJsq, _ = terms
        mpar = 4 if any(dFsq) else 3
        if any(dJsq):
            raise ValueError("MF32 states an uncertainty on J")
        nrs = len(resonances)
        matrix = np.zeros((mpar * nrs, mpar * nrs))
        for r in range(nrs):
            matrix[r * mpar, r * mpar] = dEsq[r]
            matrix[r * mpar + 1, r * mpar + 1] = dNsq[r]
            matrix[r * mpar + 2, r * mpar + 1:r * mpar + 3] = [dNdG[r], dGsq[r]]
            if mpar == 4:
                matrix[r * mpar + 3, r * mpar + 1:r * mpar + 4] = [dNdF[r], dGdF[r], dFsq[r]]
        for r in range(mpar * nrs):
            matrix[r, r:] = matrix[r:, r]
        mf32 = [(lis[0], lis[3], lis[4]) for lis in resonances]
    elif lcomp == 1:
        _, _, _, _, nsrs, nlrs = lines.ints()
        if nsrs != 1 or nlrs:
            raise ValueError(f"LCOMP=1 with NSRS={nsrs}, NLRS={nlrs} is not converted")
        _, _, mpar, _, length, nrb = lines.ints()
        dim = nrb * mpar
        size = dim * (dim + 1) // 2
        if size + 6 * nrb != length:
            raise ValueError("an LCOMP=1 block whose length is not its matrix's")
        resonances = [lines.floats() for _ in range(nrb)]
        mf32 = [(lis[0], lis[3], lis[4]) for lis in resonances]
        matrix = _lcomp1(lines, dim, size)
    elif lcomp == 2:
        flags.append("LCOMP=2")
        _, _, _, _, _, nrsa = lines.ints()
        resonances = [lines.floats() for _ in range(2 * nrsa)]
        mf32 = [(lis[0], lis[3], lis[4]) for lis in resonances[::2]]
        _, _, ndigit, nnn, nm, _ = lines.ints()
        mpar = nnn // nrsa
        diagonal = []
        for i in range(nrsa):
            dE, _, _, dN, dG, dF = resonances[2 * i + 1]
            diagonal += [dE, dN, dG] + ([dF] if mpar == 4 else [])
        if len(diagonal) != nnn:
            raise ValueError(f"an LCOMP=2 matrix of NNN={nnn} for {len(diagonal)} uncertainties")
        flags.append(f"NDIGIT={ndigit}")
        matrix = _intg(lines, nnn, nm, ndigit) * np.outer(diagonal, diagonal)
    else:
        raise ValueError(f"LCOMP={lcomp}")
    # L, J and GT take no covariance: zero rows where the table has those columns.
    width = len(rows[0]) if rows else mpar + 3
    if width != mpar + 3:
        raise ValueError(f"MF32 varies {mpar} parameters and the table has {width} columns")
    index = []
    for r in range(len(mf32)):
        index += [r * width + 1, r * width + 2, r * width + 3]
    dim = len(mf32) * width
    keep = np.array([k for k in range(dim) if k not in set(index)])
    grown = np.zeros((dim, dim))
    grown[keep.reshape(-1, 1), keep] = matrix
    unsorted = mf32 != sorted(mf32, key=lambda res: res[0])
    if unsorted or lcomp == 0:
        flags.append("sortByL")
    matrix = _consistent(mf2, mf32, grown, width, lcomp == 0)
    return matrix, flags, dap


def _reichMoore(lines, tables):
    """LRF=3, whose GNDS tables are FUDGE's (L, J, channel spin) groups, capture before elastic."""
    mf2 = [(row[0], row[2], row[1]) for _, _, rows in tables for row in rows]
    flags = ["LRF3"]
    _, _, _, lcomp, nls, isr = lines.ints()
    dap = []
    if isr:
        _, _, _, _, mls, _ = lines.ints()
        for _ in range(-(-mls // 6)):
            dap += lines.floats()
        dap = [10 * d for d in dap[:mls]]
        if len(dap) > 1:
            raise ValueError("an L-dependent scattering-radius uncertainty (MLS>1) is not converted")
    if lcomp == 1:
        _, _, _, _, nsrs, nlrs = lines.ints()
        if nsrs != 1 or nlrs:
            raise ValueError(f"LCOMP=1 with NSRS={nsrs}, NLRS={nlrs} is not converted")
        _, _, mpar, _, length, nrb = lines.ints()
        dim = nrb * mpar
        size = dim * (dim + 1) // 2
        if size + 6 * nrb != length:
            raise ValueError("an LCOMP=1 block whose length is not its matrix's")
        resonances = [lines.floats() for _ in range(nrb)]
        mf32 = [(lis[0], lis[2], lis[3]) for lis in resonances]
        matrix = _lcomp1(lines, dim, size)
    elif lcomp == 2:
        flags.append("LCOMP=2")
        _, _, _, _, _, nrsa = lines.ints()
        resonances = [lines.floats() for _ in range(2 * nrsa)]
        mf32 = [(lis[0], lis[2], lis[3]) for lis in resonances[::2]]
        _, _, ndigit, nnn, nm, _ = lines.ints()
        mpar = nnn // nrsa
        diagonal = []
        for i in range(nrsa):
            dE, _, dN, dG, dF1, dF2 = resonances[2 * i + 1]
            diagonal += [dE, dN, dG] + ([dF1] if mpar == 4 else [dF1, dF2] if mpar == 5 else [])
        if len(diagonal) != nnn:
            raise ValueError(f"an LCOMP=2 matrix of NNN={nnn} for {len(diagonal)} uncertainties")
        flags.append(f"NDIGIT={ndigit}")
        matrix = _intg(lines, nnn, nm, ndigit) * np.outer(diagonal, diagonal)
    else:
        raise ValueError(f"LRF=3 with LCOMP={lcomp} is not converted")
    widths = {len(row) for _, _, rows in tables for row in rows}
    if widths and widths != {mpar}:
        raise ValueError(f"MF32 varies {mpar} parameters and the tables have {sorted(widths)} columns")
    unsorted = mf32 != sorted(mf32, key=lambda res: res[0])
    if unsorted or lcomp == 0:
        flags.append("sortByL")
    matrix = _consistent(mf2, mf32, matrix, mpar, lcomp == 0)
    for i in range(len(mf2)):
        _swapRows(matrix, mpar * i + 1, mpar * i + 2, 1)
    return matrix, flags, dap


def _rMatrixLimited(lines, tables):
    mf2 = [tuple(row) for _, _, rows in tables for row in rows]
    _, _, ifg, lcomp, njs, isr = lines.ints()
    if isr:
        raise ValueError("a scattering-radius uncertainty (ISR=1) is not converted")
    mf32 = []
    if lcomp == 1:
        _, _, _, _, nsrs, nlrs = lines.ints()
        _, _, njsx, _, _, _ = lines.ints()
        for _ in range(njsx):
            _, _, nch, nrb, _, _ = lines.ints()
            count = -(-(nch + 1) // 6)
            for i in range(max(1, nrb)):
                vals: List[float] = []
                for _ in range(count):
                    vals += lines.floats()
                if nrb > 0:
                    mf32.append(tuple(vals[:nch + 1]))
        _, _, _, _, n, nparb = lines.ints()
        matrix = _lcomp1(lines, nparb, n)
        flags = ["LCOMP=1"]
    elif lcomp == 2:
        _, _, npp, njsx, _, _ = lines.ints()
        for _ in range(2 * npp):
            lines.raw()
        uncertainties: List[float] = []
        for _ in range(njs):
            _, _, _, _, _, nch = lines.ints()
            for _ in range(nch):
                lines.raw()
            _, _, _, nrsa, _, _ = lines.ints()
            if nrsa == 0:
                lines.raw()
            count = -(-(nch + 1) // 6)
            for _ in range(nrsa):
                vals, uncerts = [], []
                for _ in range(count):
                    vals += lines.floats()
                for _ in range(count):
                    uncerts += lines.floats()
                mf32.append(tuple(vals[:nch + 1]))
                uncertainties += uncerts[:nch + 1]
        _, _, ndigit, nnn, nm, _ = lines.ints()
        rsd = np.array(uncertainties)
        matrix = _intg(lines, nnn, nm, ndigit) * np.outer(rsd, rsd)
        flags = ["LCOMP=2", f"NDIGIT={ndigit}"]
    else:
        raise ValueError(f"LRF=7 with LCOMP={lcomp} is not converted")
    matrix = _consistent(mf2, mf32, matrix, None, False)
    return matrix, flags, []


def fudgeParameterCovariances(suite, tables, report) -> Tuple[Optional[list], List[Tuple[str, str]]]:
    """*suite*'s parameter covariances in FUDGE's layout, and the flags to write with them.

    *tables* is ``(rowHref, [(label, href, rows), ...])`` for the one resolved
    range, as kika's GNDS writer writes its resonance tables. Returns
    ``(None, [])`` when there is nothing to convert (no MF32 text: a suite read
    from GNDS already has FUDGE's layout).
    """
    from kika.endf.model_adapter.parameter_covariances import MF32MT151_KEY

    covariances = list(getattr(suite.covarianceSuite, "parameterCovariances", None) or ())
    texts = {}
    for covariance in covariances:
        header = getattr(getattr(covariance, "provenance", None), "headerFields", None) or {}
        if header.get(MF32MT151_KEY) is not None:
            texts[id(header[MF32MT151_KEY])] = header[MF32MT151_KEY]
    if len(texts) != 1:
        return None, []
    lines = _Lines(next(iter(texts.values())))
    try:
        lines.ints()
        _, _, _, _, ner, _ = lines.ints()
        _, _, lru, lrf, nro, _ = lines.ints()
        if ner != 1 or lru != 1 or nro or tables is None:
            raise ValueError(f"MF32 with NER={ner}, LRU={lru}, NRO={nro} is not converted")
        rowHref, groups = tables
        if lrf in (1, 2):
            matrix, flags, dap = _breitWigner(lines, lrf, groups)
        elif lrf == 3:
            matrix, flags, dap = _reichMoore(lines, groups)
        elif lrf == 7:
            matrix, flags, dap = _rMatrixLimited(lines, groups)
        else:
            raise ValueError(f"LRF={lrf} is not converted")
        converted = _toModel(matrix, groups, rowHref, dap)
    except (ValueError, IndexError) as why:
        report.lost(f"MF32 is written in kika's own layout, one link per resonance, not "
                    f"FUDGE's: {why}")
        return None, []
    others = [c for c in covariances if type(c).__name__ != "ParameterCovariance"]
    return [converted] + others, [(PARAMETER_COVARIANCE_HREF, ",".join(flags))]
