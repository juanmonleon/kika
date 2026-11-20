"""A written OpenMC file against the official one of the same evaluation.

The two files are not expected to be identical -- the grids differ, the
tabulations differ, the official one came through NJOY -- so the comparison is
physical and every difference is meant to be attributed:

- **structure**: reactions present in one file only, and per reaction the
  ``redundant`` and ``center_of_mass`` flags, the number of products, the product
  particles and the distribution types;
- **cross sections**: for every reaction and temperature in both, both read
  lin-lin (what the C++ does) on the union of the two grids, the largest
  relative difference and where it is. The difference is divided by the larger
  of the two values, never by less than ``floor`` times the largest value of
  either cross section, so nothing is ignored;
- **integrals that do not depend on the grid**: the resonance integral
  (0.5 eV to 1 MeV, 1/E weight) and the Maxwellian average at the file's kT,
  for capture, elastic and fission;
- **size**: the bytes of the datasets (number of elements times item size, for
  the temperatures both files hold), and the ratio of the two.

:func:`compareNeutron` returns a :class:`Comparison`; :meth:`Comparison.summary`
renders it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import records as R

__all__ = ["Comparison", "compareNeutron", "resonanceIntegral", "maxwellianAverage"]


@dataclass
class Comparison:
    """The differences between two neutron files, as :func:`compareNeutron` finds them.

    A report and not a verdict: nothing in it is a pass or a fail, and no tolerance
    is applied. Reactions are named as in the files (``"reaction_102"``) and
    temperatures as their group names (``"294K"``).

    Attributes
    ----------
    first, second : str
        The two file paths, as given.
    sizeRatio : float
        Bytes of the datasets of the first file over those of the second, counting
        only the temperatures both files hold (and ``0K``). NaN until computed.
    onlyFirst, onlySecond : list of str
        Reactions present in one file only, sorted.
    structural : list of str
        One line per difference in the ``redundant`` or ``center_of_mass``
        attribute, the number of products, or the product particle or
        distribution types of a reaction in both files.
    crossSections : dict
        ``(reaction, temperature) -> (difference, energy)``: the largest relative
        difference of the two cross sections (dimensionless) and the energy in eV
        where it occurs.
    thresholds : dict
        ``reaction -> (first, second)``: the first energy in eV of the reaction's
        cross section in each file, for the reactions where the two differ. A
        reaction compared at several temperatures keeps the last one.
    integrals : dict
        ``(mt, temperature, quantity) -> (first, second, difference)`` for MT 2,
        18 and 102, with ``quantity`` ``"RI"`` or ``"maxwellian"`` (both in barn);
        ``difference`` is ``(first - second) / second`` (signed, a fraction), or 0
        when ``second`` is 0.
    """

    first: str
    second: str
    sizeRatio: float = float("nan")
    onlyFirst: List[str] = field(default_factory=list)
    onlySecond: List[str] = field(default_factory=list)
    structural: List[str] = field(default_factory=list)
    #: ``(reaction, T) -> (max relative difference, energy)``
    crossSections: Dict[Tuple[str, str], Tuple[float, float]] = field(default_factory=dict)
    #: ``reaction -> (first threshold, second threshold)`` where they differ
    thresholds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: ``(mt, T, quantity) -> (first, second, relative difference)``
    integrals: Dict[Tuple[int, str, str], Tuple[float, float, float]] = field(default_factory=dict)

    def worst(self, n=10):
        """The ``n`` entries of ``crossSections`` with the largest difference.

        Parameters
        ----------
        n
            How many entries to return.

        Returns
        -------
        list
            ``((reaction, temperature), (difference, energy))`` in descending order of
            ``difference``, at most ``n`` of them.
        """
        return sorted(self.crossSections.items(), key=lambda kv: -kv[1][0])[:n]

    def summary(self) -> str:
        """The comparison as text, for printing.

        The first line gives the two file names (without directory) and the size
        ratio. Then come the reactions in one file only, the first 20 structural
        lines, one line per integral (MT, temperature, quantity, the two values and
        the signed difference in per cent) and the ten largest cross-section
        differences with their energy in eV.
        """
        lines = [f"{Path(self.first).name} vs {Path(self.second).name}: size ratio {self.sizeRatio:.3f}"]
        if self.onlyFirst or self.onlySecond:
            lines.append(f"  reactions only in first {self.onlyFirst}, only in second {self.onlySecond}")
        lines += [f"  {s}" for s in self.structural[:20]]
        for (mt, T, q), (a, b, d) in sorted(self.integrals.items()):
            lines.append(f"  MT{mt:<4d} {T:>6} {q:<11} {a:12.6g} {b:12.6g}  {100 * d:+.3f} %")
        for (rx, T), (d, e) in self.worst(10):
            lines.append(f"  {rx} {T}: max |rel diff| {d:.3e} at {e:.6e} eV")
        return "\n".join(lines)


def _xs(tree, name, rx, T):
    energy = tree[f"{name}/energy/{T}"].data
    xs = tree[f"{name}/reactions/{rx}/{T}/xs"]
    threshold = int(R.attr(xs, "threshold_idx"))
    return energy[threshold:], xs.data


def resonanceIntegral(E, sigma, lo=0.5, hi=1.0e6):
    """∫ σ(E) dE/E over [lo, hi], lin-lin σ, exact per panel.

    Parameters
    ----------
    E, sigma
        The table: energies in eV, ascending, and cross sections in barn. σ is
        taken as 0 outside ``E[0]``..``E[-1]``.
    lo, hi
        Limits of the integral in eV (defaults 0.5 eV and 1 MeV).

    Returns
    -------
    float
        The integral in barn (``dE/E`` is dimensionless).
    """
    E, sigma = np.asarray(E, float), np.asarray(sigma, float)
    grid = np.unique(np.concatenate([[lo, hi], E[(E > lo) & (E < hi)]]))
    s = np.interp(grid, E, sigma, left=0.0, right=0.0)
    x0, x1, y0, y1 = grid[:-1], grid[1:], s[:-1], s[1:]
    slope = (y1 - y0) / (x1 - x0)
    return float(np.sum((y0 - slope * x0) * np.log(x1 / x0) + slope * (x1 - x0)))


def maxwellianAverage(E, sigma, kT):
    """⟨σ⟩ over a Maxwellian flux E e^{-E/kT}/kT², up to 50 kT, lin-lin σ.

    The average is the integral of σ times the flux over the integral of the
    flux, both by the trapezoid rule on a grid of 4000 points spaced
    geometrically from ``max(E[0], 1e-5 eV)`` to 50 kT, plus every point of ``E``
    below 50 kT. σ is 0 outside ``E[0]``..``E[-1]``.

    Parameters
    ----------
    E, sigma
        The table: energies in eV, ascending, and cross sections in barn.
    kT
        Temperature in eV (the value stored under ``kTs/<T>`` in the file).

    Returns
    -------
    float
        The flux-weighted average cross section in barn.
    """
    E, sigma = np.asarray(E, float), np.asarray(sigma, float)
    hi = 50.0 * kT
    grid = np.unique(np.concatenate([np.geomspace(max(E[0], 1e-5), hi, 4000), E[E < hi]]))
    s = np.interp(grid, E, sigma, left=0.0, right=0.0)
    w = grid * np.exp(-grid / kT) / kT ** 2
    return float(np.trapezoid(s * w, grid) / np.trapezoid(w, grid))


def _datasetBytes(tree, temperatures) -> int:
    """Bytes of the data a file holds for these temperatures (and everything else).

    The size ratio is per temperature set: the official files carry six, a
    file written for a benchmark one or two.
    """
    total = 0
    for path, node in R.walk(tree):
        if not isinstance(node, R.Dataset):
            continue
        parts = path.strip("/").split("/")
        temps = [p for p in parts if p.endswith("K") and p[:-1].isdigit()]
        if any(t not in temperatures and t != "0K" for t in temps):
            continue
        total += int(np.prod(node.shape or ())) * np.dtype(node.dtype).itemsize
    return total


def compareNeutron(first, second, *, floor: float = 1.0e-6, reactions=None) -> Comparison:
    """Compare two neutron files of the same evaluation; nothing is passed or failed.

    The temperatures compared are the ``kTs`` groups both files have, and the
    reactions those both have. For each reaction and temperature the two cross
    sections (read lin-lin, from each file's threshold) are put on the union of
    their grids, from the larger of the two second energies up to the smaller of
    the two last ones. The first interval above a threshold is left out because
    RECONR moves a threshold, so a zero meets a non-zero there; the thresholds
    that differ are reported in ``thresholds``. The difference at each point is
    ``|a - b| / max(|a|, |b|, floor * peak)``, with ``peak`` the largest absolute
    value of either cross section on that grid; the largest is kept with its
    energy. The integrals are on each file's own grid, with the kT of the first
    file. Structure, integrals and size are as described in :class:`Comparison`.

    Parameters
    ----------
    first, second
        Paths of the two ``.h5`` files; each holds one nuclide.
    floor
        Relative to ``peak``: the denominator of the difference is never smaller
        than ``floor * peak``, so a point far below the reaction's own scale does
        not dominate the maximum.
    reactions
        Collection of MT numbers to compare in the cross sections and the
        structure; ``None`` is all the reactions both files have. The lists of
        reactions present in one file only are not filtered by it.

    Returns
    -------
    Comparison
    """
    out = Comparison(str(first), str(second))
    with R.openFile(first) as a, R.openFile(second) as b:
        na, nb = next(iter(a.children)), next(iter(b.children))
        common = set(a[f"{na}/kTs"].children) & set(b[f"{nb}/kTs"].children)
        out.sizeRatio = _datasetBytes(a, common) / _datasetBytes(b, common)
        ra, rb = a[f"{na}/reactions"].children, b[f"{nb}/reactions"].children
        out.onlyFirst = sorted(set(ra) - set(rb))
        out.onlySecond = sorted(set(rb) - set(ra))
        common = sorted(set(ra) & set(rb))
        if reactions is not None:
            common = [rx for rx in common if int(rx.split("_")[1]) in reactions]
        for rx in common:
            for attr in ("redundant", "center_of_mass"):
                va, vb = R.attr(ra[rx], attr), R.attr(rb[rx], attr)
                if va is not None and vb is not None and int(va) != int(vb):
                    out.structural.append(f"{rx} @{attr}: {int(va)} vs {int(vb)}")
            pa = [k for k in ra[rx].children if k.startswith("product_")]
            pb = [k for k in rb[rx].children if k.startswith("product_")]
            if len(pa) != len(pb):
                out.structural.append(f"{rx}: {len(pa)} vs {len(pb)} products")
            else:
                for p in pa:
                    da = [R.text(R.attr(ra[rx][p][d], "type")) for d in ra[rx][p].children
                          if d.startswith("distribution_")]
                    db = [R.text(R.attr(rb[rx][p][d], "type")) for d in rb[rx][p].children
                          if d.startswith("distribution_")]
                    if da != db or R.text(R.attr(ra[rx][p], "particle")) != R.text(R.attr(rb[rx][p], "particle")):
                        out.structural.append(f"{rx}/{p}: {da} vs {db}")
        temps = sorted(set(a[f"{na}/kTs"].children) & set(b[f"{nb}/kTs"].children))
        for T in temps:
            kT = float(a[f"{na}/kTs/{T}"].data)
            for rx in common:
                if T not in ra[rx].children or T not in rb[rx].children:
                    continue
                Ea, sa = _xs(a, na, rx, T)
                Eb, sb = _xs(b, nb, rx, T)
                # The first interval above a threshold is left out: RECONR moves a
                # threshold to the energy it recomputes from Q and the mass ratio,
                # rounded up (ENDF/B-VIII.1 Fe-56 MT52: 2 122 705 eV in the file,
                # 2 122 706 in the official library), so a zero meets a non-zero
                # there by construction. It is reported apart, as `thresholds`.
                lo = max(Ea[min(1, len(Ea) - 1)], Eb[min(1, len(Eb) - 1)])
                hi = min(Ea[-1], Eb[-1])
                if Ea[0] != Eb[0]:
                    out.thresholds[rx] = (float(Ea[0]), float(Eb[0]))
                grid = np.union1d(Ea, Eb)
                grid = grid[(grid >= lo) & (grid <= hi)]
                if grid.size == 0:
                    continue
                ya, yb = np.interp(grid, Ea, sa), np.interp(grid, Eb, sb)
                peak = max(float(np.max(np.abs(ya))), float(np.max(np.abs(yb))), 1e-300)
                scale = np.maximum(np.maximum(np.abs(ya), np.abs(yb)), floor * peak)
                rel = np.abs(ya - yb) / scale
                i = int(np.argmax(rel))
                out.crossSections[(rx, T)] = (float(rel[i]), float(grid[i]))
                mt = int(rx.split("_")[1])
                if mt in (2, 18, 102):
                    for name, fn in (("RI", lambda E, s: resonanceIntegral(E, s)),
                                     ("maxwellian", lambda E, s: maxwellianAverage(E, s, kT))):
                        va, vb = fn(Ea, sa), fn(Eb, sb)
                        out.integrals[(mt, T, name)] = (va, vb, (va - vb) / vb if vb else 0.0)
    return out
