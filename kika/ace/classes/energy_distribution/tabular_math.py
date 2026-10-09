"""Evaluation of the tabulated ACE secondary distributions (LAW=1, 4, 44, 61).

One place for the rules the ACE format leaves to the transport code, written as
MCNP and OpenMC apply them, so that every class evaluates its tables the same
way:

* Between two incident energies with lin-lin interpolation (INT=2, the NJOY
  default when NR=0) a table is chosen at random, the lower one with probability
  ``1 - f``. Its continuous part is then rescaled onto the interpolated outgoing
  range (unit-base, or "scaled", interpolation), and its discrete lines are kept
  where they are. INT=1 always uses the lower table. Any other INT is treated as
  lin-lin, as MCNP does.
* A table holds ``ND`` discrete lines first (``INTT' = 10*ND + INTT``); the
  probability of line ``k`` is ``CDF(k) - CDF(k-1)``. The remaining points are a
  continuous density, histogram (INTT=1) or lin-lin (INTT=2).
* LAW=44 angles follow Kalbach-87 with ``r`` and ``a`` taken at the lower point
  of the outgoing-energy bin for a histogram table and interpolated in E' for a
  lin-lin one. LAW=61 uses the angular table of the lower point for a histogram
  table, and for a lin-lin table the table of whichever bin edge the sampled CDF
  value is closer to, which is each edge half of the time.

The pdf returned for an intermediate incident energy is the exact density of
that sampling scheme for the continuous part, on the union grid of the two
tables with a node doubled wherever the density steps (so a mix of a histogram
and a lin-lin table is exact read lin-lin); the CDF returned is its integral
on top of the discrete lines. The angular pdf is the marginal over the
outgoing energy.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: 3-point Gauss-Legendre nodes and weights on [0, 1]
_GL_T = np.array([0.5 - 0.5 * np.sqrt(0.6), 0.5, 0.5 + 0.5 * np.sqrt(0.6)])
_GL_W = np.array([5.0, 8.0, 5.0]) / 18.0


def interval_scheme(nbt: Sequence[int], interp: Sequence[int], i: int) -> int:
    """ENDF interpolation scheme of the interval between points ``i`` and ``i+1`` (0-based)."""
    if len(nbt) == 0:
        return 2
    for boundary, scheme in zip(nbt, interp):
        if i + 2 <= int(boundary):
            return int(scheme)
    return int(interp[-1])


def bracket(energies: Sequence[float], energy: float) -> Tuple[int, float]:
    """Lower table index and interpolation fraction; outside the grid the end table, f=0."""
    grid = np.asarray(energies, dtype=float)
    if energy <= grid[0]:
        return 0, 0.0
    if energy >= grid[-1]:
        return len(grid) - 1, 0.0
    i = int(np.searchsorted(grid, energy, side="right")) - 1
    return i, float((energy - grid[i]) / (grid[i + 1] - grid[i]))


def split_table(table: Dict) -> Dict:
    """Separate a tabulated distribution into its discrete lines and its continuous part."""
    nd = int(table.get("n_discrete", 0))
    x = np.asarray(table["e_out"], dtype=float)
    p = np.asarray(table["pdf"], dtype=float)
    c = np.asarray(table["cdf"], dtype=float)
    out = {
        "intt": int(table.get("intt", 2)),
        "discrete_energies": x[:nd],
        "discrete_probabilities": np.diff(np.concatenate(([0.0], c[:nd]))),
        "e_out": x[nd:],
        "pdf": p[nd:],
        "cdf": c[nd:],
    }
    for key in ("r", "a", "lc"):
        if key in table:
            out[key] = np.asarray(table[key], dtype=float)
    return out


def pdf_on_grid(x: np.ndarray, p: np.ndarray, intt: int, grid: np.ndarray) -> np.ndarray:
    """Density of a continuous table at ``grid``; zero outside its range."""
    grid = np.asarray(grid, dtype=float)
    out = np.zeros_like(grid)
    if len(x) < 2:
        return out
    inside = (grid >= x[0]) & (grid <= x[-1])
    if intt == 1:
        k = np.clip(np.searchsorted(x, grid[inside], side="right") - 1, 0, len(x) - 2)
        out[inside] = p[k]
    else:
        out[inside] = np.interp(grid[inside], x, p)
    return out


def pdf_limits(x: np.ndarray, p: np.ndarray, intt: int, grid: np.ndarray):
    """Left and right limits of a continuous table's density at ``grid``.

    A repeated node is a step (NJOY closes ranges that way), so the two limits
    differ there; outside the table both are zero.
    """
    grid = np.asarray(grid, dtype=float)
    left, right = np.zeros_like(grid), np.zeros_like(grid)
    if len(x) < 2:
        return left, right
    last = len(x) - 1

    def value(k, g):
        if intt == 1:
            return p[k]
        width = x[k + 1] - x[k]
        t = np.divide(g - x[k], width, out=np.zeros_like(g), where=width > 0)
        return p[k] + (p[k + 1] - p[k]) * t

    kr = np.searchsorted(x, grid, side="right") - 1
    ok = (kr >= 0) & (kr < last)
    right[ok] = value(kr[ok], grid[ok])
    kl = np.searchsorted(x, grid, side="left") - 1
    ok = (kl >= 0) & (kl < last)
    left[ok] = value(kl[ok], grid[ok])
    return left, right


def cdf_on_grid(x: np.ndarray, p: np.ndarray, c: np.ndarray, intt: int, grid: np.ndarray) -> np.ndarray:
    """Sampling CDF of a continuous table at ``grid``, integrating the density inside each bin."""
    grid = np.asarray(grid, dtype=float)
    if len(x) < 2:
        return np.full_like(grid, c[-1] if len(c) else 0.0)
    k = np.clip(np.searchsorted(x, grid, side="right") - 1, 0, len(x) - 2)
    t = np.clip(grid, x[0], x[-1]) - x[k]
    if intt == 1:
        return c[k] + p[k] * t
    width = x[k + 1] - x[k]
    slope = np.divide(p[k + 1] - p[k], width, out=np.zeros_like(width), where=width > 0)
    return c[k] + p[k] * t + 0.5 * slope * t * t


def _scaled(part: Dict, lo: float, hi: float) -> Dict:
    """Map a continuous part linearly onto [lo, hi], keeping its probability."""
    x = part["e_out"]
    if len(x) < 2 or x[-1] == x[0]:
        return part
    s = (hi - lo) / (x[-1] - x[0])
    out = dict(part)
    out["e_out"] = lo + (x - x[0]) * s
    out["pdf"] = part["pdf"] / s
    return out


def interpolate_tables(low: Dict, high: Dict, frac: float, scheme: int) -> Dict:
    """Distribution at an incident energy between two tables (see the module notes).

    Returns the continuous part as ``e_out``/``pdf``/``cdf`` and the discrete
    lines as ``discrete_energies``/``discrete_probabilities``.
    """
    lo, hi = split_table(low), split_table(high)
    if scheme == 1 or frac <= 0.0:
        return _as_result(lo)
    if frac >= 1.0:
        return _as_result(hi)

    x_lo, x_hi = lo["e_out"], hi["e_out"]
    if len(x_lo) >= 2 and len(x_hi) >= 2:
        e1 = (1.0 - frac) * x_lo[0] + frac * x_hi[0]
        ek = (1.0 - frac) * x_lo[-1] + frac * x_hi[-1]
        lo_s, hi_s = _scaled(lo, e1, ek), _scaled(hi, e1, ek)
    else:
        lo_s, hi_s = lo, hi
    nodes = np.union1d(lo_s["e_out"], hi_s["e_out"])
    left = np.zeros_like(nodes)
    right = np.zeros_like(nodes)
    for part, scaled, w in ((lo, lo_s, 1.0 - frac), (hi, hi_s, frac)):
        l_t, r_t = pdf_limits(scaled["e_out"], scaled["pdf"], part["intt"], nodes)
        left += w * l_t
        right += w * r_t
    intt = 1 if lo["intt"] == 1 and hi["intt"] == 1 else 2
    if intt == 1:
        # histogram: the value of the bin that starts at each node
        grid, pdf = nodes, right
        cont = np.concatenate(([0.0], np.cumsum(pdf[:-1] * np.diff(grid))))
    else:
        # lin-lin on the union grid, a node doubled where the density jumps;
        # exact for any mix of histogram and lin-lin tables
        jump = ~np.isclose(left, right, rtol=1e-12, atol=0.0)
        grid = np.repeat(nodes, np.where(jump, 2, 1))
        pdf = np.empty_like(grid)
        pos = np.cumsum(np.where(jump, 2, 1)) - 1
        pdf[pos] = right
        pdf[pos[jump] - 1] = left[jump]
        cont = np.concatenate(([0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))))
    discrete_p = np.concatenate(((1.0 - frac) * lo["discrete_probabilities"],
                                 frac * hi["discrete_probabilities"]))
    cdf = discrete_p.sum() + cont
    return {
        "intt": intt,
        "n_discrete": len(lo["discrete_energies"]) + len(hi["discrete_energies"]),
        "discrete_energies": np.concatenate((lo["discrete_energies"], hi["discrete_energies"])),
        "discrete_probabilities": discrete_p,
        "n_points": len(grid),
        "e_out": grid,
        "pdf": pdf,
        "cdf": cdf,
    }


def _as_result(part: Dict) -> Dict:
    return {
        "intt": part["intt"],
        "n_discrete": len(part["discrete_energies"]),
        "discrete_energies": np.array(part["discrete_energies"]),
        "discrete_probabilities": np.array(part["discrete_probabilities"]),
        "n_points": len(part["e_out"]),
        "e_out": np.array(part["e_out"]),
        "pdf": np.array(part["pdf"]),
        "cdf": np.array(part["cdf"]),
    }


def distribution_at(incident_energies, tables, nbt, interp, energy: float) -> Optional[Dict]:
    """Interpolated distribution of a law with one table per incident energy."""
    if len(incident_energies) == 0 or len(tables) == 0:
        return None
    i, frac = bracket(incident_energies, energy)
    if frac == 0.0 or i + 1 >= len(tables):
        return _as_result(split_table(tables[i])) if tables[i] is not None else None
    if tables[i] is None or tables[i + 1] is None:
        return None
    return interpolate_tables(tables[i], tables[i + 1], frac, interval_scheme(nbt, interp, i))


def bin_weights(part: Dict) -> np.ndarray:
    """Probability of each continuous bin (between consecutive points)."""
    return np.clip(np.diff(part["cdf"]), 0.0, None)


def kalbach_pdf(mu: np.ndarray, a, r) -> np.ndarray:
    """Kalbach-87 angular density ``a/(2 sinh a) [cosh(a mu) + r sinh(a mu)]``.

    ``a`` and ``r`` broadcast against ``mu``; written in exponentials of
    non-positive arguments so it does not overflow for large ``a``.
    """
    mu = np.asarray(mu, dtype=float)
    a = np.asarray(a, dtype=float)
    r = np.asarray(r, dtype=float)
    small = np.abs(a) < 1e-8
    a_safe = np.where(small, 1.0, a)
    # a/(2 sinh a) * (cosh(a mu) + r sinh(a mu)), multiplied through by e^-a
    pref = a_safe / (-np.expm1(-2.0 * a_safe))
    val = 0.5 * pref * ((1.0 + r) * np.exp(a_safe * (mu - 1.0))
                        + (1.0 - r) * np.exp(-a_safe * (mu + 1.0)))
    return np.where(small, 0.5, val)


def kalbach_angular_table(table: Dict, mu: np.ndarray) -> np.ndarray:
    """Angular density of one LAW=44 table, marginal over the outgoing energy."""
    part = split_table(table)
    mu = np.asarray(mu, dtype=float)
    out = np.zeros_like(mu)
    nd = len(part["discrete_energies"])
    r_all = np.asarray(table["r"], dtype=float)
    a_all = np.asarray(table["a"], dtype=float)
    for k in range(nd):
        out += part["discrete_probabilities"][k] * kalbach_pdf(mu, a_all[k], r_all[k])
    r, a, p = r_all[nd:], a_all[nd:], part["pdf"]
    if len(r) < 2:
        return out
    w = bin_weights(part)
    if part["intt"] == 1:
        out += np.sum(w[:, None] * kalbach_pdf(mu[None, :], a[:-1, None], r[:-1, None]), axis=0)
        return out
    # lin-lin: r, a and the density are linear across the bin
    t = _GL_T[None, :]
    rt = r[:-1, None] + (r[1:] - r[:-1])[:, None] * t
    at = a[:-1, None] + (a[1:] - a[:-1])[:, None] * t
    dens = p[:-1, None] * (1.0 - t) + p[1:, None] * t
    gw = _GL_W[None, :] * dens
    norm = gw.sum(axis=1, keepdims=True)
    gw = np.divide(gw, norm, out=np.full_like(gw, 1.0 / 3.0), where=norm > 0)
    g = kalbach_pdf(mu[None, None, :], at[:, :, None], rt[:, :, None])
    out += np.einsum("b,bj,bjm->m", w, gw, g)
    return out


def tabular_angular_table(table: Dict, angular_tables: Dict, mu: np.ndarray) -> np.ndarray:
    """Angular density of one LAW=61 table, marginal over the outgoing energy.

    ``angular_tables`` maps each LC locator to its table; LC=0 is isotropic.
    """
    part = split_table(table)
    mu = np.asarray(mu, dtype=float)
    lcs = np.asarray(table["lc"], dtype=int)

    def ang(lc):
        if lc == 0:
            return np.full_like(mu, 0.5)
        t = angular_tables[int(lc)]
        cos = np.asarray(t["cosines"], dtype=float)
        return pdf_on_grid(cos, np.asarray(t["pdf"], dtype=float), int(t["jj"]), mu)

    out = np.zeros_like(mu)
    nd = len(part["discrete_energies"])
    for k in range(nd):
        out += part["discrete_probabilities"][k] * ang(lcs[k])
    cont = lcs[nd:]
    if len(cont) < 2:
        return out
    w = bin_weights(part)
    if part["intt"] == 1:
        for k in range(len(cont) - 1):
            out += w[k] * ang(cont[k])
    else:
        for k in range(len(cont) - 1):
            out += w[k] * 0.5 * (ang(cont[k]) + ang(cont[k + 1]))
    return out


def angular_at(incident_energies, tables, nbt, interp, energy: float, table_pdf) -> np.ndarray:
    """Angular marginal at ``energy``: the two bracketing tables mixed as they are sampled."""
    i, frac = bracket(incident_energies, energy)
    if frac == 0.0 or i + 1 >= len(tables) or interval_scheme(nbt, interp, i) == 1:
        return table_pdf(tables[i])
    return (1.0 - frac) * table_pdf(tables[i]) + frac * table_pdf(tables[i + 1])


def tab1(x, y, nbt, interp, q: float) -> float:
    """An ENDF TAB1 read with its own (NBT, INT) regions, held at the ends.

    The parameter tables of the ACE laws (temperatures, Watt a and b, yields,
    nu, probabilities) carry NBT/INT like any ENDF TAB1; NR=0 means lin-lin.
    Outside the table the end value holds, as MCNP reads them.
    """
    from kika.algebra.evaluate import evaluate
    from kika.algebra.laws import interval_laws

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return 0.0
    if len(x) == 1:
        return float(y[0])
    pairs = [(int(b), int(s)) for b, s in zip(nbt, interp)]
    return float(evaluate(x, y, interval_laws(len(x), pairs), q, outside="hold"))
