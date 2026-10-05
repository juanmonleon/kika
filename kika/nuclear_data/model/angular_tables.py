"""Perturb a tabulated angular distribution f(mu) by Legendre-order factors.

MF34 is a covariance of the Legendre coefficients ``a_l(E)``. A tabulated MF4
(LTT=2, and the upper half of LTT=3) stores ``f(mu, E)`` as a table and has no
``a_l`` to multiply. This module is the arithmetic that bridges the two, for
one incident energy; the energy dimension -- bin edges, repeated energies --
is :func:`kika.nuclear_data.model.perturbation.applyTabulatedFactors`.

**The method** (``docs/library/mf4_tabulated_perturbation_roadmap.md``, D-B).
Any table decomposes exactly as

    f(mu) = sum_{l in orders} (2l+1)/2 a_l P_l(mu) + r(mu)

where ``r`` is everything the orders MF34 covers do not hold -- higher orders,
the forward peak. Scaling ``a_l -> c_l a_l`` and writing the result back as a
table on the **same mu nodes** is

    f'(mu_i) = f(mu_i) + sum_l (2l+1)/2 (c_l - 1) a_l P_l(mu_i)

which never builds ``r``: the correction is added to the evaluator's table. The
naive alternative -- resum ``sum c_l a_l P_l`` and tabulate that -- throws ``r``
away, so the sample stops looking like the evaluation even with every factor
at 1.

**The projection is exact, not a quadrature over the whole interval.** A lin-lin
table is linear *between* nodes and kinked *at* each one; a single Gauss rule
over [-1, 1] converges badly on that and the error lands on ``a_l``, which is
what gets multiplied. Here each panel ``[mu_i, mu_i+1]`` is integrated on its
own: ``f`` is linear there, ``f P_l`` a polynomial of degree ``l+1``, and
Gauss-Legendre with ``ceil((l+2)/2)`` points integrates it with no error at
all. A histogram panel (INT=1) is a constant times ``P_l``, also exact. Other
laws are refused: a log law makes the integrand transcendental, and no MF4
table measured carries one (U-238 and Al-27 of JEFF-4.0, U-238 of ENDF/B-VIII.1
are lin-lin throughout).

What it guarantees by construction:

* every factor 1 gives back ``f`` **bit for bit** -- the correction is not
  computed at all;
* the orders not named are not touched, which is the tabulated analogue of the
  Legendre applier leaving the orders above NL alone;
* ``integral f' = integral f`` in the continuum, since ``int P_l = 0`` for
  ``l >= 1``. On the table, interpolated lin-lin between nodes, it is not exact:
  the lin-lin interpolant of a polynomial does not integrate to zero. ``info``
  carries the size of that, and refining mu is decision D3.

Positivity is **reported, not enforced**, which is what the Legendre applier
does too: whether a sample may carry a negative probability is the sampler's
call. With lin-lin in mu, ``f' >= 0`` at every node is ``f' >= 0`` everywhere,
so the node count is exact.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial import legendre as _legendre

__all__ = ["legendreMoments", "perturbTabulatedAngular", "tableIntegral",
           "EXACT_ANGULAR_LAWS"]

#: ENDF INT codes whose panels the projection integrates exactly: histogram
#: and lin-lin.
EXACT_ANGULAR_LAWS = (1, 2)

#: How far a table's mu range may sit from [-1, 1] and still be the full range.
MU_RANGE_ATOL = 1.0e-9


def _panelLaws(mu: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """The ENDF INT of each panel ``[mu_i, mu_i+1]`` under cumulative NBT."""
    laws = np.empty(max(mu.size - 1, 0), dtype=int)
    start = 0
    for nbt, code in pairs:
        end = int(nbt) - 1                     # last point index of this region
        laws[start:end] = int(code)
        start = end
    if start < laws.size:
        laws[start:] = int(pairs[-1][1]) if pairs else 2
    return laws


def _checkTable(mu: np.ndarray, p: np.ndarray, laws: np.ndarray) -> None:
    if mu.ndim != 1 or mu.size < 2 or p.shape != mu.shape:
        raise ValueError(f"a table of f(mu) needs two or more (mu, f) pairs, got "
                         f"{mu.shape} and {p.shape}")
    if np.any(np.diff(mu) < 0.0):
        raise ValueError("mu must be ascending")
    if abs(mu[0] + 1.0) > MU_RANGE_ATOL or abs(mu[-1] - 1.0) > MU_RANGE_ATOL:
        raise ValueError(
            f"the table spans [{mu[0]}, {mu[-1]}], not [-1, 1]: a_l = int f P_l "
            f"over the whole cosine range, and a table that stops short has no "
            f"stated value for the rest of it")
    bad = sorted(set(laws.tolist()) - set(EXACT_ANGULAR_LAWS))
    if bad:
        raise NotImplementedError(
            f"angular interpolation law(s) {bad} make the projection integrand "
            f"transcendental; only histogram (1) and lin-lin (2) are integrated "
            f"exactly, and no measured MF4 table uses another")


def legendreMoments(mu, p, orders: Sequence[int],
                    pairs: Optional[Sequence[Tuple[int, int]]] = None
                    ) -> Dict[int, float]:
    """``{l: a_l}`` with ``a_l = int_{-1}^{1} f(mu) P_l(mu) dmu``, exactly.

    *pairs* is the table's ``(NBT, INT)`` list; ``None`` means one lin-lin
    region. No normalisation is applied: ``a_0`` is the table's own integral,
    and a relative factor on ``a_l`` means the same thing either way.
    """
    mu = np.asarray(mu, dtype=float)
    p = np.asarray(p, dtype=float)
    pairs = list(pairs) if pairs else [(mu.size, 2)]
    laws = _panelLaws(mu, pairs)
    _checkTable(mu, p, laws)
    orders = [int(order) for order in orders]
    if not orders:
        return {}

    lo, hi = mu[:-1], mu[1:]
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    nodes, weights = _legendre.leggauss(math.ceil((max(orders) + 2) / 2))
    x = mid[:, None] + half[:, None] * nodes[None, :]          # (panels, q)
    slope = np.where(hi > lo, (p[1:] - p[:-1]) / np.where(hi > lo, hi - lo, 1.0), 0.0)
    linear = p[:-1, None] + slope[:, None] * (x - lo[:, None])
    flat = np.broadcast_to(p[:-1, None], x.shape)
    f = np.where((laws == 1)[:, None], flat, linear)
    scale = half[:, None] * weights[None, :]

    return {order: float(np.sum(f * _legendre.legval(x, _unit(order)) * scale))
            for order in orders}


def _unit(order: int) -> np.ndarray:
    coefficients = np.zeros(order + 1)
    coefficients[order] = 1.0
    return coefficients


def tableIntegral(mu, p, pairs=None) -> float:
    """``int f dmu`` under the table's own law, panel by panel."""
    mu = np.asarray(mu, dtype=float)
    p = np.asarray(p, dtype=float)
    laws = _panelLaws(mu, list(pairs) if pairs else [(mu.size, 2)])
    width = np.diff(mu)
    return float(np.sum(np.where(laws == 1, p[:-1] * width,
                                 0.5 * (p[:-1] + p[1:]) * width)))


def perturbTabulatedAngular(mu, p, orderFactors: Mapping[int, float], *,
                            pairs: Optional[Sequence[Tuple[int, int]]] = None
                            ) -> Tuple[np.ndarray, dict]:
    """``(pPrime, info)`` -- one table, perturbed by one factor per order.

    Parameters
    ----------
    mu, p
        The table: ascending cosines over [-1, 1] and ``f`` at each.
    orderFactors
        ``{l: c_l}`` for this incident energy, ``l >= 1``. Order 0 is the cross
        section's magnitude and is refused, as it is by the Legendre applier.
    pairs
        The table's ``(NBT, INT)``; ``None`` is one lin-lin region.

    Returns
    -------
    pPrime
        ``f'`` on the same nodes, same length, same law.
    info
        ``moments`` (``a_l``), ``deltas`` (``(c_l - 1) a_l``), ``min_p``,
        ``n_negative`` (nodes with ``f' < 0``), ``integral_before`` and
        ``integral_after`` under the table's law.
    """
    mu = np.asarray(mu, dtype=float)
    p = np.asarray(p, dtype=float)
    factors = {int(order): float(c) for order, c in orderFactors.items()}
    if 0 in factors:
        raise ValueError(
            "Legendre order 0 is the cross-section magnitude, not a shape "
            "coefficient; its factor belongs on the cross section")

    moments = legendreMoments(mu, p, sorted(factors), pairs)
    deltas = {order: (factors[order] - 1.0) * moments[order] for order in factors}
    active = {order: delta for order, delta in deltas.items() if delta != 0.0}

    if active:
        series = np.zeros(max(active) + 1)
        for order, delta in active.items():
            series[order] = 0.5 * (2 * order + 1) * delta
        correction = _legendre.legval(mu, series)
        pPrime = p + correction
    else:
        pPrime = p.copy()

    info = {
        "moments": moments,
        "deltas": deltas,
        "min_p": float(pPrime.min()),
        "n_negative": int(np.sum(pPrime < 0.0)),
        "integral_before": tableIntegral(mu, p, pairs),
        "integral_after": tableIntegral(mu, pPrime, pairs),
    }
    return pPrime, info
