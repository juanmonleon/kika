"""An electro-atomic evaluation → the electron blocks of an EPRDATA table (JXS 19-27).

The method of LANL's eprdata25 (NJOY3), so the two agree number for number:

- **ESZE** (JXS 19): the union of every MF23 grid, merged pairwise in order of size
  (a value repeated as often as the grid repeating it most; a grid that opens or
  closes on a non-zero value inside the other gets that point repeated), each
  cross section lin-lin on it (a step at a table's end is dropped first),
  cut below 10 eV (a step at 10 eV keeps only its upper copy). Columns: energy
  (MeV), total = ((σ_el + σ_brem) + σ_exc) + Σσ_ion, elastic (MT525, the large-angle
  one), bremsstrahlung, excitation, Σσ_ion, then each subshell's σ_ion.
- **EXCIT** (JXS 20) and **BREML** (JXS 26): the excitation's and the
  bremsstrahlung's energy loss, ``E − Ē`` on their own grids (Ē the electron's
  ``averageProductEnergy``).
- **ELASI/ELAS** (JXS 21-22): the large-angle elastic's tabulated P(μ|E), as μ and
  its cumulative trapezoid, normalised (the last value set to 1).
- **EION** (JXS 23): per subshell, the knock-on electron's spectra as E′ (MeV) and
  cdf; a header of the counts and of where each block starts.
- **BREMI/BREME** (JXS 24-25): the bremsstrahlung photon's spectra, likewise.
- **SELAS** (JXS 27): the transport-corrected elastic, μ̄′(E)·σ_525(E) with μ̄′ the
  mean of 1 − μ of each angular table (exact for lin-lin) interpolated log-log in E,
  then the total elastic MT526.

**Locators.** The EION header names where each subshell's block starts as an XSS
index. Here those indices are counted **from the first electron word** (as if
JXS(19) were 1); :func:`kika.ace.encoders.photoatomic.encode_photoatomic` adds
where the electron blocks actually start, so the photon blocks before them can
change length.
"""
from __future__ import annotations

import bisect
import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

__all__ = ["electronBlocks", "ELECTRON_CUTOFF"]

#: The electron cross sections start here (eV), as eprdata25 cuts them: the
#: elastic is not defined below it, and a total without it would be incomplete.
ELECTRON_CUTOFF = 10.0
_MICRO = 1e-6


def _setUnion(a, b):
    """``std::set_union`` of two sorted grids: each value as often as the grid repeating it most."""
    ca, cb = Counter(a), Counter(b)
    out: List[float] = []
    for k in sorted(set(ca) | set(cb)):
        out += [k] * max(ca.get(k, 0), cb.get(k, 0))
    return out


def _firstJump(grid, x1, x2, zero1, zero2):
    """The grid that starts later, if it opens on a non-zero value, gets its start repeated."""
    if x1 == x2:
        return zero1 and zero2
    later = x1 < x2
    zero, result = (zero2, zero1) if later else (zero1, zero2)
    if not zero:
        x = x2 if later else x1
        i = bisect.bisect_left(grid, x)
        if grid[i + 1] != x:
            grid.insert(i, x)
    return result


def _lastJump(grid, x1, x2, zero1, zero2):
    """The grid that stops earlier, if it closes on a non-zero value, gets its end repeated."""
    if x1 == x2:
        return zero1 and zero2
    earlier = x1 < x2
    zero, result = (zero1, zero2) if earlier else (zero2, zero1)
    if not zero:
        x = x1 if earlier else x2
        i = bisect.bisect_left(grid, x)
        if grid[i + 1] != x:
            grid.insert(i, x)
    return result


def _union(grids):
    """The union grid as NJOY3's unioniser builds it: the grids merged pairwise in order of
    size (smallest first, the earlier one first among equals), each merge repeating the start
    or end of a grid that opens or closes inside the other on a non-zero value."""
    ordered: List = []
    for g in grids:
        position = bisect.bisect_left([len(o[0]) for o in ordered], len(g[0]))
        ordered.insert(position, g)
    (x0, y0), (x1, y1) = ordered[0], ordered[1]
    grid = _setUnion(x0, x1)
    firstZero = _firstJump(grid, x0[0], x1[0], y0[0] == 0, y1[0] == 0)
    lastZero = _lastJump(grid, x0[-1], x1[-1], y0[-1] == 0, y1[-1] == 0)
    for x, y in ordered[2:]:
        lo, hi = grid[0], grid[-1]
        grid = _setUnion(x, grid)
        firstZero = _firstJump(grid, x[0], lo, y[0] == 0, firstZero)
        lastZero = _lastJump(grid, x[-1], hi, y[-1] == 0, lastZero)
    return grid


def _trimmed(x, y):
    """A table with a step at either end loses the outer point of it."""
    if len(x) > 1 and x[0] == x[1]:
        x, y = x[1:], y[1:]
    if len(x) > 1 and x[-1] == x[-2]:
        x, y = x[:-1], y[:-1]
    return x, y


def _evaluate(grid, x, y):
    """One lin-lin table on the union grid: 0 outside it, a threshold's first copy 0."""
    x, y = _trimmed(x, y)
    out = [0.0] * len(grid)
    j = bisect.bisect_left(grid, x[0])
    if j + 1 < len(grid) and grid[j + 1] == x[0]:
        j += 1
    t = 0
    while j < len(grid):
        g = grid[j]
        if g < x[t]:
            x0, x1, y0, y1 = x[t - 1], x[t], y[t - 1], y[t]
            out[j] = y0 + (y1 - y0) / (x1 - x0) * (g - x0)
        else:
            out[j] = y[t]
            t += 1
            if t == len(x):
                break
        j += 1
    return out


def _prune(grid, columns, cutoff=ELECTRON_CUTOFF):
    i = bisect.bisect_left(grid, cutoff)
    if i >= len(grid) or grid[i] != cutoff:
        # 10 eV is not a node: insert it with each table's interpolated value
        x0, x1 = grid[i - 1], grid[i]
        grid = grid[:i] + [cutoff] + grid[i:]
        for k, v in columns.items():
            y0, y1 = v[i - 1], v[i]
            columns[k] = v[:i] + [y0 + (y1 - y0) / (x1 - x0) * (cutoff - x0)] + v[i:]
    if i + 1 < len(grid) and grid[i + 1] == cutoff:
        i += 1
    return grid[i:], {k: v[i:] for k, v in columns.items()}


def _cumulative(x, y):
    c = [0.0]
    for k in range(1, len(x)):
        c.append(c[-1] + 0.5 * (x[k] - x[k - 1]) * (y[k - 1] + y[k]))
    return c


def _normalised(x, p):
    integral = _cumulative(x, p)[-1]
    pn = [v / integral for v in p]
    c = _cumulative(x, pn)
    c[-1] = 1.0
    return pn, c


def _meanLinLin(x, y):
    num = 0.0
    for k in range(1, len(x)):
        xl, xr, yl, yr = x[k - 1], x[k], y[k - 1], y[k]
        if xr == xl:
            continue
        delta = xr - xl
        slope = (yr - yl) / delta / 3.0
        const = 0.5 * (xr * yl - xl * yr) / delta
        num += xr * xr * (slope * xr + const) - xl * xl * (slope * xl + const)
    return num / _cumulative(x, y)[-1]


def _table(reaction):
    x, y, regions = reaction.crossSection["eval"].toEndfRegions()
    if any(int(law) != 2 for _, law in regions):
        raise ValueError(f"electro-atomic MT{reaction.ENDF_MT} is not lin-lin")
    return [float(v) for v in x], [float(v) for v in y]


def _spectra(form2d) -> List[Tuple[float, List[float], List[float]]]:
    return [(float(f.outerDomainValue), [float(v) for v in f.xs], [float(v) for v in f.ys])
            for f in form2d.function1ds]


def _product(reaction, label):
    return next(p for p in reaction.outputChannel.products if p.label == label)


def _energyLoss(reaction) -> Tuple[List[float], List[float]]:
    ape = _product(reaction, "e-").averageProductEnergy["eval"]
    e = [float(v) for v in ape.xs]
    return e, [(ei - float(a)) * _MICRO for ei, a in zip(e, ape.ys)]


def electronBlocks(electrons) -> Tuple[List[int], np.ndarray, List]:
    """``(NXS(8..12), words, JXS(19..32) relative to the first word)`` of ``electrons``.

    ``electrons`` is an electro-atomic ``reactionSuite`` from the model (projectile
    ``e-``). The words are laid out ESZE, EXCIT, ELASI, ELAS, EION, BREMI, BREME,
    BREML, SELAS; the EION header's block indices count from the first word
    (see the module docstring).
    """
    reactions = {r.ENDF_MT: r for r in list(electrons.reactions) + list(electrons.incompleteReactions)
                 + list(electrons.sums) if r.ENDF_MT is not None}
    mts = sorted(reactions)
    ionisation = [mt for mt in mts if mt >= 534]
    tables = {mt: _table(reactions[mt]) for mt in mts}
    grid = _union([tables[mt] for mt in mts])
    columns = {mt: _evaluate(grid, *tables[mt]) for mt in mts}
    grid, columns = _prune(grid, columns)
    ne = len(grid)
    nssh = len(ionisation)

    words: List[float] = []
    rel: Dict[int, int] = {}

    def block(k, values):
        rel[k] = len(words)
        words.extend(float(v) for v in values)

    elastic = columns[525] if 525 in columns else columns[526]
    ion = [columns[mt] for mt in ionisation]
    iontot = list(ion[0])
    for c in ion[1:]:
        iontot = [u + v for u, v in zip(iontot, c)]
    total = [e + b for e, b in zip(elastic, columns[527])]
    total = [t + x for t, x in zip(total, columns[528])]
    total = [t + i for t, i in zip(total, iontot)]
    block(19, [e * _MICRO for e in grid] + total + elastic + columns[527] + columns[528] + iontot
          + [v for c in ion for v in c])

    e, loss = _energyLoss(reactions[528])
    nxl = len(e)
    block(20, [v * _MICRO for v in e] + loss)

    elasticForm = _product(reactions[525], "e-").distribution["eval"].angular
    angular = _spectra(elasticForm)
    na = len(angular)
    lengths, locators, data, at = [], [], [], 0
    for _, mu, p in angular:
        _, c = _normalised(mu, p)
        lengths.append(len(mu))
        locators.append(at)
        at += 2 * len(mu)
        data += mu + c
    block(21, [a[0] * _MICRO for a in angular] + lengths + locators)
    block(22, data)

    # EION: the header, then one block per subshell
    rel[23] = len(words)
    header = len(words)
    words.extend([0.0] * (3 * nssh))
    for k, mt in enumerate(ionisation):
        spectra = _spectra(_product(reactions[mt], "e-").distribution["eval"].energy)
        start = len(words) + 1                       # 1-based, counted from the first electron word
        n = len(spectra)
        words[header + k] = float(n)
        words[header + nssh + k] = float(start)
        words[header + 2 * nssh + k] = float(start + 3 * n)
        lengths, locators, data, at = [], [], [], 0
        for _, ep, p in spectra:
            _, c = _normalised(ep, p)
            lengths.append(len(ep))
            locators.append(at)
            at += 2 * len(ep)
            data += [v * _MICRO for v in ep] + c
        words.extend([s[0] * _MICRO for s in spectra] + lengths + locators + data)

    photon = _spectra(_product(reactions[527], "photon").distribution["eval"].energy)
    nb = len(photon)
    lengths, locators, data, at = [], [], [], 0
    for _, ep, p in photon:
        _, c = _normalised(ep, p)
        lengths.append(len(ep))
        locators.append(at)
        at += 2 * len(ep)
        data += [v * _MICRO for v in ep] + c
    block(24, [s[0] * _MICRO for s in photon] + lengths + locators)
    block(25, data)

    e, loss = _energyLoss(reactions[527])
    nbl = len(e)
    block(26, [v * _MICRO for v in e] + loss)

    energies = [a[0] for a in angular]
    means = []
    for _, mu, p in angular:
        pn, _ = _normalised(mu, p)
        means.append(_meanLinLin([1.0 - m for m in mu][::-1], pn[::-1]))

    def mubar(x):
        k = bisect.bisect_right(energies, x) - 1
        if k >= len(energies) - 1:
            return means[-1]
        x0, x1, y0, y1 = energies[k], energies[k + 1], means[k], means[k + 1]
        if x == x0:
            return y0
        return y0 * math.exp(math.log(x / x0) * math.log(y1 / y0) / math.log(x1 / x0))

    block(27, [mubar(x) * s for x, s in zip(grid, columns[525])] + columns[526])

    nxs = [ne, nxl, na, nb, nbl]
    jxs = [rel.get(k) for k in range(19, 33)]
    return nxs, np.asarray(words, dtype=float), jxs
