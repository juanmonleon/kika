"""MF32/MT151 for a suite whose parameter covariance came from GNDS.

The covariance is FUDGE's layout (one link per resonance table, the matrix over
every cell, see :mod:`kika.gnds.parameter_layout`) and the tape is written as
FUDGE's ``toENDF6`` writes it (``brownies/legacy/toENDF6/covariances/
modelParameters.py``): LRF=1/2 with LCOMP=0, 1 or 2, LRF=3 with LCOMP=0, 1 or 2
and LRF=7 with LCOMP=1 or 2, the form chosen by the ``LCOMP=``, ``NDIGIT=``,
``sortByL`` and ``LRF3`` flags FUDGE leaves on the matrix. Resonances are put
back in MF32's order (by L when ``sortByL``, else by energy) and their rows of
zeros dropped. The unresolved range's averageParameterCovariances become one LRU=2
body, its averages read back from the flags FUDGE leaves on each matrix. An
L-dependent scattering-radius uncertainty and several resolved ranges are not
written, and say so.
"""
from __future__ import annotations

from typing import List, Optional

import re

import numpy as np

from kika.nuclear_data.model import ConversionReport

__all__ = ["encodeMF32FromGnds"]


class _Records:
    """ENDF records of one section, as FUDGE's endfFormats builds them."""

    def __init__(self):
        self.lines: List[str] = []

    def head(self, c1, c2, l1, l2, n1, n2):
        from kika._records import format_endf_number

        self.lines.append(format_endf_number(c1) + format_endf_number(c2)
                          + "".join(f"{int(v):11d}" for v in (l1, l2, n1, n2)))

    def data(self, values):
        from kika._records import format_endf_number

        self.lines.append("".join(format_endf_number(v) for v in values).ljust(66))

    def dataList(self, values):
        values = list(values)
        for k in range(0, len(values), 6):
            self.data(values[k:k + 6])

    def intg(self, entries, ndigit):
        from kika.endf.utils import format_intg

        lines, _ = format_intg(entries, ndigit, 0, 0, 0, 1)
        self.lines += [line[:66] for line in lines]


def _swapRows(matrix, i1, i2, nrows):
    rows = matrix[i1:i1 + nrows].copy()
    matrix[i1:i1 + nrows] = matrix[i2:i2 + nrows]
    matrix[i2:i2 + nrows] = rows
    cols = matrix[:, i1:i1 + nrows].copy()
    matrix[:, i1:i1 + nrows] = matrix[:, i2:i2 + nrows]
    matrix[:, i2:i2 + nrows] = cols


def _lcomp2(records, matrix, ndigit, nnn):
    """FUDGE's ``writeLCOMP2``: the correlations packed to NDIGIT digits."""
    nints = 13 if ndigit == 3 else 56 // (ndigit + 1)
    rsd = np.sqrt(matrix.diagonal())
    rsd[rsd == 0] = 1
    corr = matrix / np.outer(rsd, rsd)
    corr *= 10 ** ndigit
    corr[corr > 0] -= 0.5
    corr[corr < 0] += 0.5
    corr = np.rint(corr)
    entries = []
    for i in range(len(corr)):
        vals = corr[i, :i]
        j = 0
        while j < i:
            if vals[j] != 0:
                entries.append((i + 1, j + 1, [int(v) for v in vals[j:j + nints]]))
                j += nints
            else:
                j += 1
    records.head(0, 0, ndigit, nnn, len(entries), 0)
    records.intg(entries, ndigit)


def _reorder(matrix, mpar, wanted, have):
    """Swap MPAR-row blocks so the resonances run in *wanted* order (FUDGE's loop)."""
    have = list(have)
    for i in range(len(wanted)):
        i2 = have.index(wanted[i])
        if i2 != i:
            _swapRows(matrix, mpar * i, mpar * i2, mpar)
            have[i], have[i2] = have[i2], have[i]


def _upper(matrix):
    out = []
    for i in range(len(matrix)):
        out += list(matrix[i][i:])
    return out


def _breitWigner(records, resonances, region, matrix, lcomp, sortByL, ndigit, awt, spi, dap):
    from kika.gnds.encode_resonances import _breitWignerRows
    from .resonances import resolvedRangeRadius
    from kika.nuclear_data.model.resonances import radiusToEndf

    formalism = region.formalism
    rows = _breitWignerRows(formalism)
    table = [(L, r.energy, r.spin, r.totalWidth or 0.0, r.neutronWidth, r.captureWidth,
              r.fissionWidth or 0.0) for L, r in rows]
    gnds = [(t[1], t[4], t[5]) for t in table]
    ls = [t[0] for t in table]
    nls = 0 if lcomp in (1, 2) or not sortByL else len(set(ls))
    radius = resolvedRangeRadius(resonances, region)
    if radius is None:
        raise ValueError("the Breit-Wigner range states no constant radius for AP")
    records.head(spi, radiusToEndf(radius), 0, lcomp, nls, int(dap is not None))
    if dap is not None:
        records.head(0, dap, 0, 0, 0, 0)
    width = len(matrix) // len(table)
    drop = {k for i in range(len(table)) for k in (i * width + 1, i * width + 2, i * width + 3)}
    keep = np.array(sorted(set(range(len(matrix))) - drop))
    matrix = matrix[keep.reshape(-1, 1), keep]
    mpar = len(matrix) // len(table)
    if sortByL:
        table.sort()
        _reorder(matrix, mpar, [(t[1], t[4], t[5]) for t in table], gnds)
    if lcomp == 0:
        index = 0
        for L in sorted(set(ls)):
            nrs = ls.count(L)
            records.head(awt, 0, L, 0, 18 * nrs, nrs)
            for i in range(index, len(table)):
                if table[i][0] != L:
                    break
                records.data(table[i][1:7])
                block = matrix[mpar * i:mpar * (i + 1), mpar * i:mpar * (i + 1)]
                lis = [block[0, 0], block[1, 1], block[2, 1], block[2, 2]]
                lis += ([block[3, 1], block[3, 2], block[3, 3], 0, 0, 0, 0, 0] if mpar == 4
                        else [0] * 8)
                records.dataList(lis)
            index += nrs
    elif lcomp == 1:
        records.head(awt, 0, 0, 0, 1, 0)
        nrb = len(table)
        nvs = (nrb * mpar) * (nrb * mpar + 1) // 2
        records.head(0, 0, mpar, 0, nvs + 6 * nrb, nrb)
        for row in table:
            records.data(row[1:7])
        records.dataList(_upper(matrix))
    else:
        dat = np.sqrt(matrix.diagonal())
        body, omit, nres = _Records(), [], 0
        for i in range(len(table)):
            uncerts = [dat[mpar * i], 0, 0, dat[mpar * i + 1], dat[mpar * i + 2], 0]
            if mpar == 4:
                uncerts[-1] = dat[mpar * i + 3]
            if not any(uncerts):
                omit += [True] * mpar
                continue
            omit += [False] * mpar
            body.dataList(table[i][1:7])
            body.dataList(uncerts)
            nres += 1
        records.head(awt, 0, 0, 0, 12 * nres, nres)
        records.lines += body.lines
        if any(omit):
            mask = ~np.array(omit)
            matrix = matrix[mask][:, mask]
        _lcomp2(records, matrix, ndigit, nres * mpar)


def _reichMoore(records, resonances, region, groups, matrix, lcomp, sortByL, ndigit, awt, spi,
                ignoreChannelSpin, dap):
    from .resonances import resolvedRangeRadius
    from kika.gnds.encode_resonances import _rmRole
    from kika.nuclear_data.model.resonances import radiusToEndf

    formalism = region.formalism
    roles = {r.label: _rmRole(r) for r in formalism.resonanceReactions}
    table = {k: [] for k in ("Ls", "energies", "Js", "elastic", "capture", "f1", "f2")}
    for group in groups:
        position = {roles.get(c.resonanceReaction): k for k, c in enumerate(group.channels)
                    if roles.get(c.resonanceReaction) != "fission"}
        fission = [k for k, c in enumerate(group.channels)
                   if roles.get(c.resonanceReaction) == "fission"]
        entrance = group.channels[position["neutron"]]
        J = float(group.spin)
        if not ignoreChannelSpin and J != 0 and entrance.channelSpin < spi:
            J *= -1
        for energy, widths in zip(group.energies, group.widths):
            table["Ls"].append(entrance.L)
            table["energies"].append(energy)
            table["Js"].append(J)
            table["elastic"].append(widths[position["neutron"]])
            table["capture"].append(widths[position["capture"]])
            table["f1"].append(widths[fission[0]] if len(fission) > 0 else 0)
            table["f2"].append(widths[fission[1]] if len(fission) > 1 else 0)
    nres = len(table["energies"])
    ls = table["Ls"]
    radius = resolvedRangeRadius(resonances, region)
    if radius is None:
        raise ValueError("the Reich-Moore range states no constant radius for AP")
    lad = int(lcomp == 2 and bool(getattr(formalism, "supportsAngularReconstruction", False)))
    # FUDGE's code keeps NLS for LCOMP=1 here; the tapes (Mn-55) write it only for LCOMP=0.
    nls = 0 if lcomp in (1, 2) or not sortByL else len(set(ls))
    records.head(spi, radiusToEndf(radius), lad, lcomp, nls, int(dap is not None))
    if dap is not None:
        records.head(0, 0, 0, 0, 1, 1)
        records.data([dap])
    mpar = len(matrix) // nres
    sortedTable = sorted(zip(ls, table["energies"], table["Js"], table["elastic"],
                             table["capture"], table["f1"], table["f2"]))
    if not sortByL:
        sortedTable.sort(key=lambda row: row[1])
    _reorder(matrix, mpar, [(r[1], r[3], r[4]) for r in sortedTable],
             list(zip(table["energies"], table["elastic"], table["capture"])))
    for i in range(len(sortedTable)):
        _swapRows(matrix, mpar * i + 1, mpar * i + 2, 1)
    omitRow = np.sum(matrix, axis=1) == 0
    if any(omitRow):
        omitResonance = []
        for i in range(len(sortedTable)):
            if np.all(omitRow[mpar * i:mpar * (i + 1)]):
                omitResonance.append(i)
            else:
                omitRow[mpar * i:mpar * (i + 1)] = False
        for i in omitResonance[::-1]:
            sortedTable.pop(i)
        nres = len(sortedTable)
        matrix = matrix[~omitRow][:, ~omitRow]
    if lcomp == 0:
        index = 0
        for L in sorted(set(ls)):
            nrs = ls.count(L)
            records.head(awt, 0, L, 0, 18 * nrs, nrs)
            for i in range(index, len(sortedTable)):
                if sortedTable[i][0] != L:
                    break
                records.data(sortedTable[i][1:7])
                block = matrix[mpar * i:mpar * (i + 1), mpar * i:mpar * (i + 1)]
                lis = [block[0, 0], block[1, 1], block[2, 1], block[2, 2]]
                lis += ([block[3, 1], block[3, 2], block[3, 3], 0, 0, 0, 0, 0] if mpar == 4
                        else [0] * 8)
                records.dataList(lis)
            index += nrs
    elif lcomp == 1:
        records.head(awt, 0, 0, 0, 1, 0)
        nvs = (nres * mpar) * (nres * mpar + 1) // 2
        records.head(0, 0, mpar, 0, nvs + 6 * nres, nres)
        for row in sortedTable:
            records.data(row[1:])
        records.dataList(_upper(matrix))
    else:
        dat = np.sqrt(matrix.diagonal())
        body, count = _Records(), 0
        for i in range(len(sortedTable)):
            uncerts = [dat[mpar * i], 0, dat[mpar * i + 1], dat[mpar * i + 2], 0, 0]
            if mpar == 5:
                uncerts[-2:] = [dat[mpar * i + 3], dat[mpar * i + 4]]
            if not any(uncerts):
                continue
            body.dataList(sortedTable[i][1:])
            body.dataList(uncerts)
            count += 1
        records.head(awt, 0, 0, 0, 12 * count, count)
        records.lines += body.lines
        _lcomp2(records, matrix, ndigit, count * mpar)


def _rMatrixLimited(records, suite, region, groups, matrix, links, lcomp, ndigit, mat):
    from kika.nuclear_data.model import EndfProvenance

    from .derive import DerivationContext
    from .derive.resonances import _rMatrixLimitedRange
    from .resonances import _encodeRMatrixLimited

    context = DerivationContext(suite=suite, mat=mat)
    fields = _rMatrixLimitedRange(suite.resonances, region, context, ConversionReport(),
                                  "the resolved range")
    rml = _encodeRMatrixLimited(region.formalism, fields)
    ifg = int(region.formalism.reducedWidthAmplitudes)
    rows = [tuple(row) for group in groups for row in group[2]]
    if lcomp == 2:
        uncertainties = list(np.sqrt(matrix.diagonal()))
        records.head(0, 0, ifg, lcomp, len(links), 0)
        pairs = rml.particle_pairs
        records.head(0, 0, len(pairs), 0, 12 * len(pairs), 2 * len(pairs))
        for p in pairs:
            records.data([p.ma, p.mb, p.za, p.zb, p.ia, p.ib])
            records.data([p.q, p.pnt, p.shf, p.mt, p.pa, p.pb])
        uidx, nnn = 0, 0
        for (label, href, table), sg in zip(groups, rml.spin_groups):
            nrsa = len(table)
            if nrsa == 0:
                continue
            nch = len(sg.channels)
            nnn += nrsa * (nch + 1)
            records.head(sg.aj, sg.pj, sg.kbk, 0, 6 * nch, nch)
            for c in sg.channels:
                records.data([c.ipp, c.l, c.sch, c.bnd, c.ape, c.apt])
            nx = (nch // 6 + 1) * nrsa
            records.head(0, 0, 0, nrsa, 12 * nx, nx)
            for res in table:
                for j in range(nch // 6 + 1):
                    line = list(res[j * 6:j * 6 + 6])
                    records.data(line + [0] * (6 - len(line)))
                these = uncertainties[uidx:uidx + len(res)]
                for j in range(nch // 6 + 1):
                    line = these[j * 6:max(j * 6 + 6, nch)]
                    records.data(line + [0] * (6 - len(line)))
                uidx += len(res)
        _lcomp2(records, matrix, ndigit, nnn)
    elif lcomp == 1:
        records.head(0, 0, 0, lcomp, 0, 0)
        records.head(0, 0, 0, 0, 1, 0)
        records.head(0, 0, len([g for g in groups if g[2]]), 0, 0, 0)
        keep = np.zeros(len(matrix), dtype=bool)
        start, nparb = 0, 0
        for (label, href, table), sg in zip(groups, rml.spin_groups):
            n = sum(len(row) for row in table)
            if n == 0:
                continue
            ncol = len(table[0])
            nonzero = np.any(matrix[start:start + n], axis=1).reshape(len(table), ncol).any(axis=1)
            for idx, include in enumerate(nonzero):
                keep[start + idx * ncol:start + (idx + 1) * ncol] = include
            nch = len(sg.channels)
            nrb = int(nonzero.sum())
            nx = (nch // 6 + 1) * nrb
            records.head(0, 0, nch, nrb, 6 * nx, nx)
            for res, include in zip(table, nonzero):
                if not include:
                    continue
                for j in range(nch // 6 + 1):
                    line = list(res[j * 6:j * 6 + 6])
                    records.data(line + [0] * (6 - len(line)))
            if nrb == 0:
                records.data([0] * 6)
            nparb += nrb * (nch + 1)
            start += n
        matrix = matrix[keep][:, keep]
        records.head(0, 0, 0, 0, nparb * (nparb + 1) // 2, nparb)
        records.dataList(_upper(matrix))
    else:
        raise ValueError(f"LRF=7 with LCOMP={lcomp}")


_URR_ROW = re.compile(r"/Ls/L\[@label='(\d+)'\]/Js/J\[@label='(\d+)'\]/"
                      r"(levelSpacing|widths/width\[@label='(\d+)'\])$")


def _urrRow(link):
    match = _URR_ROW.search(link.href if link is not None else "")
    if match is None:
        raise ValueError(f"an average parameter covariance on {getattr(link, 'href', None)!r}")
    lpos, jpos, kind, width = match.groups()
    return int(lpos), int(jpos), -1 if kind == "levelSpacing" else int(width)


def _urrRadius(resonances, widths):
    from kika.nuclear_data.model.resonances import MODEL_RADIUS_UNIT, radiusToEndf

    radius = widths.scatteringRadius
    if radius is not None and widths.radiusUnit not in (None, MODEL_RADIUS_UNIT):
        raise ValueError(f"the unresolved range states its radius in {widths.radiusUnit!r}")
    stated = resonances.scatteringRadius
    if radius is None and stated is not None and not stated.isEnergyDependent:
        radius = stated.constant
    if radius is None:
        raise ValueError("the unresolved range states no constant radius for AP")
    return radiusToEndf(float(radius))


def _unresolved(records, suite, averages, conversions, awr, neutron, spi):
    """The sections become one LRU=2 body, laid out as FUDGE's ``averageParametersToENDF6``.

    AWRI is the unresolved range's own mass, as the resolved body's is (FUDGE writes
    the target's, which 15 of B-VIII.1's 44 such tapes do not state)."""
    from kika.gnds.parameter_layout import _urrLayout
    from .resonances import formalismAtomicWeightRatio

    region = suite.resonances.unresolved
    awri = formalismAtomicWeightRatio(region.tabulatedWidths, neutron) or awr
    layout = _urrLayout(region.tabulatedWidths)
    diagonal = {}
    for section in averages:
        if not section.crossTerm:
            form = section.form
            matrix = np.asarray(form.matrix, dtype=float)
            if matrix.shape != (1, 1) or not form.isRelative:
                raise ValueError(f"{section.label!r} is not one relative value over the range")
            href = (f"$covariances#/covarianceSuite/parameterCovariances/averageParameter"
                    f"Covariance[@label='{section.label}']/covarianceMatrix[@label='{form.label}']")
            flags = conversions.flagsFor(href) if conversions is not None else {}
            diagonal[_urrRow(section.rowData)] = (matrix[0, 0], flags)
    byL = {}
    for (L, J), (lpos, jpos, group) in layout.items():
        byL.setdefault((lpos, L), []).append((jpos, J, group))
    records.head(region.domainMin, region.domainMax, 2, 1, 0, 0)
    records.head(spi, _urrRadius(suite.resonances, region.tabulatedWidths), 0, 0, len(byL), 0)
    rows, values, mpars = [], [], set()
    for (lpos, L), groups in byL.items():
        stated = groups[0][2].atomicWeightRatio
        records.head(awri if stated is None else stated, 0, L, 0, 6 * len(groups), len(groups))
        for jpos, J, group in groups:
            average = {"D": 0.0, "GNO": 0.0, "GG": 0.0, "GF": 0.0, "GX": 0.0}
            mpar = 0
            for k in range(-1, len(group.channels)):
                found = diagonal.get((lpos, jpos, k))
                if found is None:
                    break
                value, flags = found
                if not flags:
                    raise ValueError(f"no MF32 averages are stated for L={L} J={J}")
                average.update({key: float(v) for key, v in flags.items()})
                rows.append((lpos, jpos, k))
                values.append(value)
                mpar += 1
            mpars.add(mpar)
            records.data([average["D"], J, average["GNO"], average["GG"], average["GF"],
                          average["GX"]])
    if len(mpars) != 1 or len(rows) != len(diagonal):
        raise ValueError("the unresolved averages do not vary the same parameters for every L, J")
    npar = len(rows)
    matrix = np.diag(values)
    for section in averages:
        if section.crossTerm:
            i, j = rows.index(_urrRow(section.rowData)), rows.index(_urrRow(section.columnData))
            matrix[i, j] = matrix[j, i] = float(np.asarray(section.form.matrix, dtype=float)[0, 0])
    records.head(0, 0, mpars.pop(), 0, npar * (npar + 1) // 2, npar)
    records.dataList(v for i, row in enumerate(matrix) for v in row[i:])


def _urrFission(widths):
    return int(any(channel.label == "fission"
                   and (np.any(channel.widths) if channel.widths is not None
                        else bool(channel.constantWidth))
                   for group in widths.spinGroups for channel in group.channels))


def encodeMF32FromGnds(suite, mat: Optional[int], report: ConversionReport):
    """``(MF32MT151 or None, report)`` for *suite*'s resonance parameter covariances."""
    from kika.endf.parsers.parse_mf32 import parse_mf32_mt151
    from kika.nuclear_data.model import AverageParameterCovariance, ParameterCovariance
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags
    from .derive import DerivationContext
    from .derive.resonances import _fileNeutronMass

    covariances = list(suite.covarianceSuite.parameterCovariances)
    resolvedOnes = [c for c in covariances if isinstance(c, ParameterCovariance)]
    averages = [c for c in covariances if isinstance(c, AverageParameterCovariance)]
    if not covariances or len(resolvedOnes) > 1             or len(resolvedOnes) + len(averages) != len(covariances):
        report.lost(f"MF32: {len(covariances)} parameter covariances, {len(resolvedOnes)} of "
                    f"them resolved; kika writes one resolved range's and the unresolved "
                    f"range's; MF32 is not written")
        return None, report
    resonances = suite.resonances
    conversions = EndfConversionFlags.of(suite)
    try:
        target = suite.PoPs.get(suite.target)
        neutron = _fileNeutronMass(DerivationContext(suite=suite))
        awr = float(target.mass.convertedTo("amu").value) / neutron
        za = int(target.ZA)
        spi = float(getattr(target.spin, "value", target.spin) or 0.0)
        records = _Records()
        lfw = None
        if resolvedOnes:
            lfw = _resolved(records, suite, resolvedOnes[0], conversions, awr, neutron, spi,
                            mat)
        if averages:
            if resonances.unresolved is None:
                raise ValueError("unresolved averages are varied but there is no unresolved range")
            _unresolved(records, suite, averages, conversions, awr, neutron, spi)
            if lfw is None:
                lfw = _urrFission(resonances.unresolved.tabulatedWidths)
        header = _Records()
        header.head(za, awr, 0, 0, 1, 0)
        header.head(za, 1.0, 0, lfw, int(bool(resolvedOnes)) + int(bool(averages)), 0)
    except (ValueError, KeyError, IndexError, AttributeError) as why:
        report.lost(f"MF32 from the GNDS parameter covariance is not written: {why}")
        return None, report
    mat = int(mat) if mat is not None else 0
    lines = [f"{line:66s}{mat:4d}{32:2d}{151:3d}{k + 1:5d}"
             for k, line in enumerate(header.lines + records.lines)]
    section = parse_mf32_mt151(lines, 151)
    section._mat = mat
    report.approximated("MF32/MT151 is written from the GNDS parameter covariance as FUDGE's "
                        "toENDF6 writes it")
    return section, report


def _resolved(records, suite, covariance, conversions, awr, neutron, spi, mat):
    """The resolved range's body, as FUDGE's ``toENDF6`` writes it; returns LFW."""
    from kika.gnds.encode_resonances import resonanceTables
    from kika.nuclear_data.model.resonances import BreitWigner, RMatrix
    from .resonances import formalismAtomicWeightRatio

    form = covariance.form
    resonances = suite.resonances
    regions = getattr(resonances, "resolved", None) or []
    if len(regions) != 1:
        raise ValueError(f"{len(regions)} resolved ranges")
    region = regions[0]
    tabled = resonanceTables(resonances, ConversionReport())
    if tabled is None:
        raise ValueError("no resonance tables")
    _, groups = tabled
    links = list(form.parameters)
    matrix = np.array(form.matrix, dtype=float)
    dap = None
    if links and links[0].href.endswith("/scatteringRadius") and links[0].nParameters == 1:
        links.pop(0)
        variance = matrix[0, 0]
        if form.isRelative:
            variance *= float(resonances.scatteringRadius.constant) ** 2
        dap = float(np.sqrt(variance)) / 10.0
        matrix = matrix[1:, 1:]
    sizes = [sum(len(row) for row in rows) for _, _, rows in groups]
    if [s for s in sizes if s] != [link.nParameters for link in links]:
        raise ValueError("the matrix's parameter links are not the resonance tables "
                         "(an L-dependent scattering-radius uncertainty, or another layout)")
    if form.isRelative:
        values = np.array([v for _, _, rows in groups for row in rows for v in row], dtype=float)
        matrix *= np.outer(values, values)
        matrix[matrix == -0.0] = 0
    href = (f"$covariances#/covarianceSuite/parameterCovariances/parameterCovariance"
            f"[@label='{covariance.label}']/parameterCovarianceMatrix[@label='{form.label}']")
    flags = conversions.flagsFor(href) if conversions is not None else {}
    lcomp = int(flags["LCOMP"]) if "LCOMP" in flags else 1
    ndigit = int(flags["NDIGIT"]) if "NDIGIT" in flags else 2
    sortByL = "sortByL" in flags
    awt = formalismAtomicWeightRatio(region.formalism, neutron) or awr
    formalism = region.formalism
    if isinstance(formalism, BreitWigner):
        from kika.nuclear_data.model.resonances import BreitWignerApproximation
        lrf = {BreitWignerApproximation.singleLevel: 1, BreitWignerApproximation.multiLevel: 2}[
            BreitWignerApproximation(formalism.approximation)]
        lfw = int(any(r.fissionWidth for g in formalism.resonanceParameters.spinGroups
                      for r in g.resonances))
    elif isinstance(formalism, RMatrix):
        from kika.gnds.endf_conversion import rMatrixHref
        rmFlags = conversions.flagsFor(rMatrixHref(suite, 0)) if conversions else {}
        lrf = 3 if "LRF3" in flags or "LRF3" in rmFlags else 7
        lfw = int(any(r.reactionMT in (18, 19) for r in formalism.resonanceReactions))
    else:
        raise ValueError(f"a {type(formalism).__name__} range")
    calculate = (formalism.radiusPolicy.channelMode == "mass" if formalism.radiusPolicy is not None
                 else bool(getattr(formalism, "calculateChannelRadius", True)))
    naps = 0 if lrf == 7 else int(not calculate)
    records.head(region.domainMin, region.domainMax, 1, lrf, 0, naps)
    if lrf in (1, 2):
        _breitWigner(records, resonances, region, matrix, lcomp, sortByL, ndigit, awt, spi, dap)
    elif lrf == 3:
        from kika.gnds.encode_resonances import _rmGroupsForGNDS, _withoutEmptyFission
        models = _rmGroupsForGNDS(formalism, ConversionReport())
        _, models = _withoutEmptyFission(formalism, models, ConversionReport())
        ignore = "ignoreChannelSpin" in (conversions.flagsFor(rMatrixHref(suite, 0))
                                         if conversions else {})
        _reichMoore(records, resonances, region, models, matrix, lcomp, sortByL, ndigit,
                    awt, spi, ignore, dap)
    else:
        if dap is not None:
            raise ValueError("a scattering-radius uncertainty on an LRF=7 range")
        _rMatrixLimited(records, suite, region, groups, matrix, links, lcomp, ndigit, mat)
    return lfw
