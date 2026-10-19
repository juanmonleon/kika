"""A photo-atomic evaluation and its atomic relaxation → the blocks of an EPRDATA table.

The photon and relaxation blocks of a :class:`~kika.ace.classes.photoatomic.PhotoatomicAce`
(Hughes, LA-UR-13-27377; the NXS(6) = 3 layout of eprdata14/25), computed from
the model: the photo-atomic ``reactionSuite`` (MF23/MF27) and the relaxation PoPs
(MF28). Every block follows the method LANL's eprdata25 (NJOY3) uses, so the two
agree number for number, except where eprdata25 carries a defect kika does not
reproduce (the Compton shells, :mod:`.compton`; the Eu-63 profile value).

**Energy grid.** The multiset union of every MF23 grid: a value repeated in any
table (an absorption edge) is repeated in the union, and a table that starts or
stops at a non-zero value later or earlier than the union gets its end point
repeated, so every edge is a step. At a repeated energy the first copy holds the
value below, the second the value above. Cross sections are evaluated lin-lin
(every MF23 table of ENDF/B-VIII.1 is lin-lin). The photo-electric column is the
sum of the subshells and the pair column MT515 + MT517, not the evaluated
MT522 and MT516.

**Relaxation.** A subshell's transition probabilities are normalised to one;
the radiative ones are listed first, by the subshell the vacancy moves to, then
the non-radiative ones by the two subshells. A transition's energy is the
difference of binding energies (radiative E_i − E_j; non-radiative
E_i − E_j − E_k), as GNDS defines it: ENDF's ETR is not used.

**Heating numbers** (LHNM), per collision:
``H = [σ_inc (E − Ē_inc) + σ_pair (E − 2mc²) + σ_pe (E − F)] / σ_tot``, with Ē_inc
the mean energy of the Compton-scattered photon (Klein-Nishina times the
incoherent scattering function) and F the fluorescence energy NJOY's ``alax``
model removes above the K or L edge (:func:`_fluorescence`).
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from kika.ace.classes.photoatomic import ComptonProfile, PhotoatomicAce, RelaxationTransitionAce

from .compton import comptonShells

__all__ = ["photoatomicTable", "PhotoatomicNotes", "ELECTRON_REST_ENERGY", "unionGrid", "refinedGrid",
           "LOGLOG_RTOL"]

#: m_e c² in eV: CODATA 2022 electron mass, as NJOY3 evaluates it (5.485799090441e-4 amu).
ELECTRON_REST_ENERGY = 510998.95069304795
#: (1e-10 e / h / c): x = sin(θ/2)/λ in 1/Å per eV of photon energy.
_X_PER_EV = 1.0e-10 * 1.602176634e-19 / 6.62607015e-34 / 299792458.0
_FIRST_SUBSHELL_MT = 534


@dataclass
class PhotoatomicNotes:
    """What the table builder did that a user should know."""

    compton: List[str] = field(default_factory=list)
    nonPositiveTransitions: int = 0


# ---------------------------------------------------------------------------
# Tables on the union grid
# ---------------------------------------------------------------------------

def _trimmed(x, y):
    """A table with a step at either end loses the outer point of it."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size > 1 and x[0] == x[1]:
        x, y = x[1:], y[1:]
    if x.size > 1 and x[-1] == x[-2]:
        x, y = x[:-1], y[:-1]
    return x, y


def unionGrid(tables) -> np.ndarray:
    """The union of ``[(x, y), ...]``: a value as often as the table repeating it most,
    and the start (end) of a table that opens (closes) on a non-zero value inside the
    union repeated, so the opening is a step."""
    counts: Dict[float, int] = {}
    for x, _ in tables:
        values, n = np.unique(x, return_counts=True)
        for v, c in zip(values.tolist(), n.tolist()):
            counts[v] = max(counts.get(v, 0), min(c, 2))
    lo = min(float(x[0]) for x, _ in tables)
    hi = max(float(x[-1]) for x, _ in tables)
    for x, y in tables:
        if x[0] > lo and y[0] != 0.0:
            counts[float(x[0])] = 2
        if x[-1] < hi and y[-1] != 0.0:
            counts[float(x[-1])] = 2
    grid = np.array(sorted(counts), dtype=float)
    return np.repeat(grid, [counts[v] for v in grid.tolist()])


def _onGrid(x, y, grid) -> np.ndarray:
    """A lin-lin table on the union grid: 0 outside it, below-then-above at a repeated energy."""
    out = np.zeros(grid.size)
    if x.size == 0:
        return out
    # which copy of a repeated energy each grid point is
    copy = np.zeros(grid.size, dtype=int)
    copy[1:] = (grid[1:] == grid[:-1]).astype(int)
    first = np.searchsorted(x, grid, side="left")
    last = np.searchsorted(x, grid, side="right")
    inside = (grid >= x[0]) & (grid <= x[-1])
    for i in np.flatnonzero(inside):
        g = grid[i]
        if last[i] > first[i]:                 # a node of the table
            n = last[i] - first[i]
            repeated = (i + 1 < grid.size and grid[i + 1] == g) or copy[i]
            if g == x[0] and n == 1 and repeated:
                out[i] = 0.0 if copy[i] == 0 else y[0]          # the table opens here
            elif g == x[-1] and n == 1 and repeated:
                out[i] = y[-1] if copy[i] == 0 else 0.0         # the table closes here
            else:
                out[i] = y[first[i] + min(copy[i], n - 1)]
        else:
            k = first[i]
            out[i] = y[k - 1] + (y[k] - y[k - 1]) / (x[k] - x[k - 1]) * (g - x[k - 1])
    return out


#: How far log-log (what MCNP and OpenMC read between nodes) may stray from the
#: lin-lin the evaluation states: the tolerance of the rest of the processing.
LOGLOG_RTOL = 1.0e-3


def _loglogMisses(grid, columns, rtol=LOGLOG_RTOL):
    """Intervals where a column read log-log misses its lin-lin by more than ``rtol``.

    Tested at a quarter, the half and three quarters of every interval between two
    distinct nodes where the column is positive at both ends. Returns a bool per
    interval.
    """
    a, b = grid[:-1], grid[1:]
    wide = b > a
    bad = np.zeros(a.size, dtype=bool)
    for y in columns:
        ya, yb = y[:-1], y[1:]
        live = wide & (ya > 0.0) & (yb > 0.0)
        if not live.any():
            continue
        i = np.flatnonzero(live)
        la, lb = np.log(a[i]), np.log(b[i])
        lya, lyb = np.log(ya[i]), np.log(yb[i])
        for t in (0.25, 0.5, 0.75):
            x = a[i] + t * (b[i] - a[i])
            lin = ya[i] + t * (yb[i] - ya[i])
            log = np.exp(lya + (lyb - lya) * (np.log(x) - la) / (lb - la))
            bad[i] |= np.abs(log - lin) > rtol * np.abs(lin)
    return bad


def _openings(grid, columns, rtol=LOGLOG_RTOL):
    """A node at a + √rtol·(b − a) in every interval where a column opens from 0 (no step).

    Log-log cannot read a zero; the transport puts an infinitesimal there, so the
    first sub-interval loses its lin-lin area, which this bounds to rtol of the interval's.
    """
    a, b = grid[:-1], grid[1:]
    new = []
    for y in columns:
        opens = (b > a) & (y[:-1] == 0.0) & (y[1:] > 0.0)
        new.append(a[opens] + np.sqrt(rtol) * (b[opens] - a[opens]))
    return np.concatenate(new) if new else np.empty(0)


def refinedGrid(tables, columnsOf, rounds: int = 60) -> np.ndarray:
    """The union grid with nodes added until every column reads log-log within :data:`LOGLOG_RTOL`.

    ``tables`` are the ``(x, y)`` the union is made of; ``columnsOf(grid)`` gives
    the columns the transport interpolates, on a grid. Every node of the union is
    kept; added nodes split an interval at its midpoint.
    """
    grid = unionGrid(tables)
    opening = _openings(grid, columnsOf(grid))
    if opening.size:
        grid = np.sort(np.concatenate((grid, opening)))
    for _ in range(rounds):
        bad = _loglogMisses(grid, columnsOf(grid))
        if not bad.any():
            return grid
        mid = 0.5 * (grid[:-1][bad] + grid[1:][bad])
        grid = np.sort(np.concatenate((grid, mid)))
    raise ValueError(f"the photon grid does not read log-log within {LOGLOG_RTOL} after {rounds} rounds")


def _table(reaction):
    x, y, regions = reaction.crossSection["eval"].toEndfRegions()
    if any(int(law) != 2 for _, law in regions):
        raise ValueError(f"MT{reaction.ENDF_MT} is not lin-lin; photo-atomic tables are written lin-lin")
    return _trimmed(x, y)


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

@dataclass
class _Transition:
    subj: int
    subk: int
    energy: float
    probability: float


@dataclass
class _Subshell:
    designator: int
    label: str
    electrons: float
    binding: float
    transitions: List[_Transition]

    def radiative(self):
        return [t for t in self.transitions if t.subk == 0]

    def totalRadiative(self):
        return sum(t.probability for t in self.radiative())

    def averageRadiative(self, lo=None, hi=None):
        rad = [t for t in self.radiative() if (lo is None or t.subj >= lo) and (hi is None or t.subj <= hi)]
        p = sum(t.probability for t in rad)
        return p, (sum(t.energy * t.probability for t in rad) / p if p else 0.0)


def _relaxation(pops, notes: PhotoatomicNotes) -> Tuple[int, List[_Subshell]]:
    from kika.nuclear_data.model import subshellDesignator

    elements = [e for e in pops.chemicalElements.values() if getattr(e, "atomic", None) is not None]
    if len(elements) != 1:
        raise ValueError("the relaxation PoPs must carry one element with atomic data")
    element = elements[0]
    configurations = element.atomic.configurations
    binding = {c.subshell: float(c.bindingEnergy.value) for c in configurations}
    out = []
    for c in sorted(configurations, key=lambda c: c.designator):
        rad, nonrad = [], []
        modes = c.decayData.decayModes.decayModes if c.decayData is not None else []
        for mode in modes:
            products = mode.decayPath.decays[0].products
            vacancy = next(p.pid for p in products if "{" in p.pid)
            labels = vacancy[vacancy.index("{") + 1:-1].split(",")
            j = subshellDesignator(labels[0])
            if len(labels) == 1:
                rad.append(_Transition(j, 0, binding[c.subshell] - binding[labels[0]], float(mode.probability)))
            else:
                k = subshellDesignator(labels[1])
                e = binding[c.subshell] - binding[labels[0]] - binding[labels[1]]
                nonrad.append(_Transition(j, k, e, float(mode.probability)))
        rad.sort(key=lambda t: t.subj)
        nonrad.sort(key=lambda t: (t.subj, t.subk))
        total = sum(t.probability for t in rad) + sum(t.probability for t in nonrad)
        for t in rad + nonrad:
            t.probability = t.probability / total
            if t.energy <= 0.0:
                notes.nonPositiveTransitions += 1
        out.append(_Subshell(c.designator, c.subshell, float(c.electronNumber), binding[c.subshell], rad + nonrad))
    return element.Z, out


# ---------------------------------------------------------------------------
# Incoherent mean outgoing energy
# ---------------------------------------------------------------------------

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
#: Extra breaks in x, ratio 1.2 from 1e-8 to 1e14 1/Å: the S table's nodes alone can
#: leave a piece where E'(x) spans orders of magnitude (a table that stops short).
_GEOMETRIC = np.geomspace(1.0e-8, 1.0e14, int(np.ceil(np.log(1.0e22) / np.log(1.2))) + 1)


def _breaks(sx):
    return np.unique(np.concatenate((sx, _GEOMETRIC)))


def _incoherentMean(energies: np.ndarray, sx: np.ndarray, sy: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Ē' of Compton scattering weighted by S(x), at each incident energy (eV).

    ∫E' f dE' / ∫ f dE' over [E/(1+2k), E], f = S(x) (1 + μ² + k k' (1−μ)²) (the
    Klein-Nishina dσ/dE' up to a constant), integrated in x, where S is lin-lin:
    16-point Gauss-Legendre on every interval of the S table, each also cut at a
    geometric ladder in x (ratio 1.2), which is exact to rounding for the smooth factors.
    """
    mc2 = ELECTRON_REST_ENERGY
    out = np.zeros(energies.size)
    xmax_tab = sx[-1]
    for start in range(0, energies.size, chunk):
        e = energies[start:start + chunk]
        k = e / mc2
        # x at the Compton edge (μ = −1): 1 − μ = 2
        xtop = _X_PER_EV * e * np.sqrt(0.5 * 2.0)
        num = np.zeros(e.size)
        den = np.zeros(e.size)
        bounds = np.concatenate((_breaks(sx), [np.inf]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            lo = np.minimum(a, xtop)
            hi = np.minimum(b, xtop)
            live = hi > lo
            if not live.any():
                continue
            h = 0.5 * (hi - lo)
            m = 0.5 * (hi + lo)
            x = m[:, None] + h[:, None] * _GL_X[None, :]
            # E'(x): 1/E' = 1/E + 2 x² / (c E)² / mc²
            ce = (_X_PER_EV * e)[:, None]
            inv = 1.0 / e[:, None] + 2.0 * (x / ce) ** 2 / mc2
            ep = 1.0 / inv
            kp = ep / mc2
            mu = 1.0 + 1.0 / k[:, None] - 1.0 / kp
            s = np.interp(np.minimum(x, xmax_tab), sx, sy)
            f = s * (1.0 + mu * mu + k[:, None] * kp * (1.0 - mu) ** 2)
            jac = ep * ep * 4.0 * x / (ce * ce * mc2)          # |dE'/dx|
            w = (h[:, None] * _GL_W[None, :]) * jac * live[:, None]
            num += (w * f * ep).sum(axis=1)
            den += (w * f).sum(axis=1)
        out[start:start + chunk] = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), e)
    return out


# ---------------------------------------------------------------------------
# Fluorescence (NJOY's alax model, as eprdata25 writes it)
# ---------------------------------------------------------------------------

def _isClose(a, b):
    eps = np.finfo(float).eps
    return a == b or abs(a - b) < max(100 * eps * (abs(a) + abs(b)), 100 * eps)


def _edgeRatio(binding: Optional[float], grid, pe) -> float:
    if binding is None:
        return 0.0
    for i in np.flatnonzero(grid[1:] == grid[:-1]):
        if _isClose(binding, grid[i]):
            return pe[i] / pe[i + 1]
    return 0.0


def _fluorescence(z: int, shells: List[_Subshell], grid, pe):
    """``(edges, ejection probabilities, yields, energies, F(E))`` of the alax model, in eV."""
    by = {s.designator: s for s in shells}
    K, L1, L2, L3 = (by[d].binding if d in by else 0.0 for d in (1, 2, 3, 4))
    Lbar = (L1 + L2 + L3) / 3.0
    rK = _edgeRatio(by[1].binding if 1 in by else None, grid, pe)
    r1 = _edgeRatio(by[2].binding if 2 in by else None, grid, pe)
    r2 = _edgeRatio(by[3].binding if 3 in by else None, grid, pe)
    r3 = _edgeRatio(by[4].binding if 4 in by else None, grid, pe)
    rL = r1 * r2 * r3
    k = by.get(1)
    PK = k.totalRadiative() if k else 0.0
    _, EK = k.averageRadiative() if k else (0.0, 0.0)

    def single(subj):
        if k is None:
            return 0.0, 0.0
        for t in k.radiative():
            if t.subj == subj:
                return t.probability, t.energy
        return 0.0, 0.0

    p2, e2 = single(3)
    p3, e3 = single(4)
    pM = eM = pO = eO = 0.0
    if k is not None and z >= 37:
        pM, eM = k.averageRadiative(5, 9)
        pO, eO = k.averageRadiative(10, 39)
    elif k is not None and 20 <= z <= 36:
        pO, eO = k.averageRadiative(5, 39)
    PL = EL = 0.0
    if z > 30:
        w1 = (r2 * r3 - rL) / (1 - rL)
        w2 = (r3 - r2 * r3) / (1 - rL)
        w3 = (1 - r3) / (1 - rL)
        P = [by[d].totalRadiative() if d in by else 0.0 for d in (2, 3, 4)]
        E = [by[d].averageRadiative()[1] if d in by else 0.0 for d in (2, 3, 4)]
        PL = P[0] * w1 + P[1] * w2 + P[2] * w3
        EL = (E[0] * P[0] * w1 + E[1] * P[1] * w2 + E[2] * P[2] * w3) / PL if PL else 0.0
    if z <= 11:
        edges = probs = yields = lines = []
    elif z <= 19:
        edges, probs, yields, lines = [K, K], [rK, 1.0], [0.0, (1 - rK) * PK], [0.0, EK]
    elif z <= 30:
        f = (1 - rK) / (p3 + p2 + pO)
        edges = [K] * 4
        probs = [rK, rK + f * p3, rK + f * (p3 + p2), 1.0]
        yields = [0.0, (1 - rK) * p3, (1 - rK) * (p3 + p2), (1 - rK) * (p3 + p2 + pO)]
        lines = [0.0, e3, e2, eO]
    elif z <= 36:
        t = rK * (p2 + p3 + pO) / (1 - rK)
        b, g = (1 - rL) * PL, (1 - rK) / rK
        edges = [Lbar, Lbar, K, K, K]
        probs = [rL, 1.0, 1 + p3 / t, 1 + (p3 + p2) / t, 1 / rK]
        yields = [0.0, b, b + g * p3, b + g * (p3 + p2), b + g * (p3 + p2 + pO)]
        lines = [0.0, EL, e3, e2, eO]
    else:
        t = rK * (p2 + p3 + pM + pO) / (1 - rK)
        b, g = (1 - rL) * PL, (1 - rK) / rK
        edges = [Lbar, Lbar, K, K, K, K]
        probs = [rL, 1.0, 1 + p3 / t, 1 + (p3 + p2) / t, 1 + (p3 + p2 + pM) / t, 1 / rK]
        yields = [0.0, b, b + g * p3, b + g * (p3 + p2), b + g * (p3 + p2 + pM), b + g * (p3 + p2 + pM + pO)]
        lines = [0.0, EL, e3, e2, eM, eO]
    F = np.zeros(grid.size)
    aboveK = np.searchsorted(grid, K, side="right")
    if 12 <= z <= 19:
        F[aboveK:] = PK * EK
    else:
        F[aboveK:] = p2 * e2 + p3 * e3 + pM * eM + pO * eO
        if z > 30:
            F[np.searchsorted(grid, Lbar, side="right"):] = PL * EL
    return (np.asarray(edges, float), np.asarray(probs, float), np.asarray(yields, float),
            np.asarray(lines, float), F)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def photoatomicTable(photoat, relaxation, *, extension: str = ".00p", awr: Optional[float] = None,
                     comment: Optional[str] = None, date: Optional[str] = None
                     ) -> Tuple[PhotoatomicAce, PhotoatomicNotes]:
    """The photon and relaxation blocks of the EPRDATA table of ``photoat`` and ``relaxation``.

    ``photoat`` is a photo-atomic ``reactionSuite`` (interaction ``atomic``) and
    ``relaxation`` the element's atomic-relaxation PoPs, both from the model. The
    electron blocks are not part of this table; ``electron_words`` is empty.

    The energy grid is the union of the evaluation's grids (eprdata25's) with
    nodes added where a table read log-log, as the transport reads it, would
    miss the lin-lin data by more than :data:`LOGLOG_RTOL` (:func:`refinedGrid`).
    """
    from kika.ace.classes.header import Header
    from kika.nuclear_data.model import ATOMIC_INTERACTION

    if getattr(photoat, "interaction", None) != ATOMIC_INTERACTION or photoat.projectile != "photon":
        raise ValueError("photoat must be a photo-atomic reactionSuite (projectile photon, interaction 'atomic')")
    notes = PhotoatomicNotes()
    reactions = {r.ENDF_MT: r for r in list(photoat.reactions) + list(photoat.sums) if r.ENDF_MT is not None}
    tables = {mt: _table(r) for mt, r in sorted(reactions.items())}
    subshellMts = sorted(mt for mt in tables if mt >= _FIRST_SUBSHELL_MT)

    def columns(g):
        """What the transport interpolates log-log: the four ESZG columns and every subshell."""
        on = {mt: _onGrid(*tables[mt], g) for mt in tables if mt in (502, 504, 515, 517) or mt in subshellMts}
        zero = np.zeros(g.size)
        pe = sum((on[mt] for mt in subshellMts), zero)
        return [on[502], on[504], on.get(515, zero) + on.get(517, zero), pe] + [on[mt] for mt in subshellMts]

    grid = refinedGrid([tables[mt] for mt in sorted(tables)], columns)
    on = {mt: _onGrid(*tables[mt], grid) for mt in tables}
    pe = np.zeros(grid.size)
    for mt in subshellMts:
        pe = pe + on[mt]
    pair = on.get(515, np.zeros(grid.size)) + on.get(517, np.zeros(grid.size))
    coh, inc = on[502], on[504]

    z, shells = _relaxation(relaxation, notes)
    if [s.designator for s in shells] != [mt - 533 for mt in subshellMts]:
        raise ValueError("the relaxation subshells are not the photo-ionisation subshells of the photo-atomic tape")

    t = PhotoatomicAce()
    t.z, t.layout, t.format_flag = int(z), "eprdata", 3
    t.energies, t.incoherent, t.coherent, t.photoelectric, t.pair_production = grid, inc, coh, pe, pair

    form = reactions[504].doubleDifferentialCrossSection["eval"].scatteringFactor
    sx, sy, _ = form.toEndfRegions()
    t.incoherent_x, t.incoherent_scattering_function = np.asarray(sx, float), np.asarray(sy, float)
    coherent = reactions[502].doubleDifferentialCrossSection["eval"].formFactor
    cx, cf, _ = coherent.toEndfRegions()
    cx, cf = np.asarray(cx, float), np.asarray(cf, float)
    g = cf * cf / z / z
    steps = 0.5 * (cx[1:] * cx[1:] - cx[:-1] * cx[:-1]) * (g[:-1] + g[1:])
    t.coherent_x, t.coherent_form_factor = cx, cf
    t.coherent_integrated_form_factor = np.concatenate(([0.0], np.cumsum(steps)))

    edges, probs, yields, lines, F = _fluorescence(z, shells, grid, pe)
    t.fluorescence_edge_energies, t.fluorescence_ejection_probabilities = edges, probs
    t.fluorescence_yields, t.fluorescence_energies = yields, lines

    mean = _incoherentMean(grid, t.incoherent_x, t.incoherent_scattering_function)
    total = ((coh + inc) + pair) + pe
    heat = ((grid - mean) * inc + (grid - 2.0 * ELECTRON_REST_ENERGY) * pair + (grid - F) * pe)
    t.heating = np.where(total > 0.0, heat / np.where(total > 0.0, total, 1.0), 0.0)

    compton, notes.compton = comptonShells(z, [(s.label, s.electrons, s.binding) for s in shells])
    t.shell_electrons = np.array([c.electrons for c in compton])
    t.shell_binding_energies = np.array([c.bindingEnergy for c in compton])
    t.shell_probabilities = np.cumsum(t.shell_electrons) / t.shell_electrons.sum()
    t.compton_profiles = [ComptonProfile(interpolation=2, pz=c.pz, pdf=c.pdf, cdf=c.cdf) for c in compton]

    electrons = np.array([s.electrons for s in shells])
    t.subshell_designators = np.array([s.designator for s in shells], dtype=int)
    t.subshell_electrons = electrons
    t.subshell_binding_energies = np.array([s.binding for s in shells])
    t.subshell_vacancy_probabilities = np.cumsum(electrons) / electrons.sum()
    t.subshell_num_transitions = np.array([len(s.transitions) for s in shells], dtype=int)
    t.subshell_photoelectric = {mt - 533: on[mt] for mt in subshellMts}
    for s in shells:
        cumulative, rows = 0.0, []
        for i, tr in enumerate(s.transitions):
            cumulative += tr.probability
            p = 1.0 if i == len(s.transitions) - 1 else cumulative
            rows.append(RelaxationTransitionAce(tr.subj, tr.subk, tr.energy, p))
        t.relaxation[s.designator] = rows

    h = Header()
    h.format_version = "legacy"
    h.zaid, h.extension = 1000 * int(z), extension
    awr = awr if awr is not None else float(getattr(photoat.provenance, "awr", None) or 0.0)
    h.atomic_weight_ratio, h.temperature = float(awr), 0.0
    h.date = date or datetime.date.today().strftime("%d/%m/%y")
    h.comment = (comment or f"{photoat.target} photo-atomic, kika")[:70]
    h.matid, h.mat_packed = 100 * int(z), True
    h.izaw_array, h.izaw_decimals = [(0, 0.0)] * 17, 7
    t.header = h
    return t, notes
