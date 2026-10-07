"""Closed-form integrals of a table under each of its five laws.

Every law integrates in closed form over a panel ``[x1, x2]``, with
``L = ln(x2/x1)`` and ``m = ln(y2/y1)``:

====  ==================================================================
law   ``int y dx``
====  ==================================================================
1     ``y1 dx``
2     ``(y1 + y2) dx / 2``
3     ``dx [y1 + (y2 - y1) h(L)]``, ``h(L) = 1/(1 - e^-L) - 1/L``
4     ``y1 dx expm1(m)/m``
5     ``y1 x1 L expm1(s)/s``, ``s = m + L``
====  ==================================================================

``expm1(s)/s`` and ``h`` are the forms that stay accurate as their argument
goes to zero; ``h`` switches to its series below ``|L| = 1e-3``. The log-log
row is where FUDGE goes wrong (``ptwXY_integration.c``: its series branch is
taken for every decreasing ``y`` steeper than ``1/x`` and has the wrong sign on
its third term, +6 % for ``y ~ x^-2`` on ``[1, 2]``). Here there is no branch:
``expm1(s)/s`` is exact for any ``s`` and is one at zero.

The weight ``1/x`` (a lethargy or ``1/E`` flux) is also closed form on every
law but log-lin, whose ``int e^{kx}/x dx`` is an exponential integral; a
log-lin table has to be :func:`~kika.algebra.refine.to_linlin`-ed first, and is
refused here rather than approximated.

Group integrals cut the table at the group edges, give each cut the value its
own law gives there -- a sub-panel of a law is the same law -- and sum whole
panels. Nothing is sampled, so a group integral *is* the table's integral.

Legendre moments ``int y P_l dx`` (:func:`legendre_moments`) are exact on a
lin-lin or histogram table: on a panel the integrand is a polynomial of degree
``l + 1``, which Gauss-Legendre with ``l // 2 + 2`` nodes per panel integrates
exactly. A table under another law is made lin-lin to a stated tolerance first.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .evaluate import evaluate
from .laws import HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG, validate
from .refine import LINEARIZATION_TOLERANCE, to_linlin

__all__ = ["panel_integrals", "cumulative_integral", "integral", "group_integrals",
           "group_averages", "legendre_moments", "legendre_coefficients", "WEIGHTS"]

#: The weights :func:`panel_integrals` knows: none, and ``1/x``.
WEIGHTS = (None, "1/x")


def _expm1_over(s: np.ndarray) -> np.ndarray:
    """``expm1(s)/s``, one at zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.expm1(s) / s
    return np.where(s == 0, 1.0, out)


def _h(L: np.ndarray) -> np.ndarray:
    """``1/(1 - e^-L) - 1/L``: the weight of ``y2`` in a lin-log panel's mean."""
    small = np.abs(L) < 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = -1.0 / np.expm1(-L) - 1.0 / L
    series = 0.5 + L / 12.0 - L ** 3 / 720.0
    return np.where(small, series, direct)


def _q(d: np.ndarray) -> np.ndarray:
    """``1 - ln(1 + d)/d``, the lin-lin panel's ``1/x`` correction, ``d = dx/x1``."""
    small = np.abs(d) < 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = 1.0 - np.log1p(d) / d
    series = d / 2.0 - d ** 2 / 3.0 + d ** 3 / 4.0 - d ** 4 / 5.0
    return np.where(small, series, direct)


def panel_integrals(x, y, laws, weight: Optional[str] = None) -> np.ndarray:
    """``int_{x_i}^{x_{i+1}} y w dx`` for every interval, under its own law.

    *weight* is ``None`` (``w = 1``) or ``"1/x"``. A zero-width interval
    contributes nothing.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    x, y, laws = validate(x, y, laws)
    if x.size < 2:
        return np.zeros(0)
    x1, x2, y1, y2 = x[:-1], x[1:], y[:-1], y[1:]
    dx = x2 - x1
    out = np.zeros(dx.size)
    wide = dx > 0
    if weight == "1/x" and np.any(wide & (x1 <= 0)):
        raise ValueError("a 1/x weight needs positive abscissae")
    with np.errstate(divide="ignore", invalid="ignore"):
        for code in np.unique(laws[wide]):
            m = wide & (laws == code)
            a1, a2, b1, b2, d = x1[m], x2[m], y1[m], y2[m], dx[m]
            if weight is None:
                if code == HISTOGRAM:
                    v = b1 * d
                elif code == LINLIN:
                    v = 0.5 * (b1 + b2) * d
                elif code == LINLOG:
                    v = d * (b1 + (b2 - b1) * _h(np.log(a2 / a1)))
                elif code == LOGLIN:
                    v = b1 * d * _expm1_over(np.log(b2 / b1))
                else:
                    L = np.log(a2 / a1)
                    v = b1 * a1 * L * _expm1_over(np.log(b2 / b1) + L)
            else:
                L = np.log(a2 / a1)
                if code == HISTOGRAM:
                    v = b1 * L
                elif code == LINLIN:
                    v = b1 * L + (b2 - b1) * _q(d / a1)
                elif code == LINLOG:
                    v = 0.5 * (b1 + b2) * L
                elif code == LOGLOG:
                    v = b1 * L * _expm1_over(np.log(b2 / b1))
                else:
                    raise ValueError(
                        "a log-lin panel has no closed-form 1/x integral (it is "
                        "an exponential integral); linearize the table first "
                        "with kika.algebra.to_linlin")
            out[m] = v
    return out


def cumulative_integral(x, y, laws, weight: Optional[str] = None) -> np.ndarray:
    """``[0, int_{x0}^{x1}, int_{x0}^{x2}, ...]`` -- exact on the stated laws."""
    p = panel_integrals(x, y, laws, weight)
    return np.concatenate(([0.0], np.cumsum(p)))


def group_integrals(x, y, laws, edges, weight: Optional[str] = None) -> np.ndarray:
    """``int_{e_j}^{e_{j+1}} y w dx`` for every group of *edges*.

    *edges* is non-decreasing and may reach past the table on either side --
    the table is zero there, so that part contributes nothing. Each edge that
    falls inside a panel cuts it into two panels of the same law, valued by
    that law at the edge, so the result is the table's own integral and not a
    quadrature of it.
    """
    x, y, laws = validate(x, y, laws)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or np.any(np.diff(edges) < 0):
        raise ValueError("edges must be a non-decreasing 1-d array")
    n_groups = max(edges.size - 1, 0)
    if x.size < 2 or n_groups == 0:
        return np.zeros(n_groups)
    # Edges strictly inside the domain that are not already nodes become cuts.
    inner = edges[(edges > x[0]) & (edges < x[-1])]
    cuts = np.setdiff1d(inner, x)
    if cuts.size:
        k = np.searchsorted(x, cuts, side="right") - 1
        # The value at a cut is the law's own value there. Panels with a
        # repeated abscissa never contain a cut (cuts are not nodes).
        vals = evaluate(x, y, laws, cuts)
        at = np.searchsorted(x, cuts, side="right")
        xr = np.insert(x, at, cuts)
        yr = np.insert(y, at, vals)
        lr = np.insert(laws, k + 1, laws[k])  # both halves keep the law
        x, y, laws = xr, yr, lr
    cumulative = cumulative_integral(x, y, laws, weight)
    # The integral up to an edge: everything left of it. At an edge equal to a
    # repeated abscissa the zero-width interval adds nothing either way.
    c = np.clip(edges, x[0], x[-1])
    idx = np.searchsorted(x, c, side="left")
    totals = cumulative[idx]
    return np.diff(totals)


def integral(x, y, laws, lo: Optional[float] = None, hi: Optional[float] = None,
             weight: Optional[str] = None) -> float:
    """``int_lo^hi y w dx``; the limits default to the table's own domain.

    Limits outside the table add nothing; ``hi <= lo`` gives zero.
    """
    x, y, laws = validate(x, y, laws)
    if x.size < 2:
        return 0.0
    a = float(x[0]) if lo is None else float(lo)
    b = float(x[-1]) if hi is None else float(hi)
    if b <= a:
        return 0.0
    return float(group_integrals(x, y, laws, [a, b], weight)[0])


def group_averages(x, y, laws, edges, weight: Optional[str] = None) -> np.ndarray:
    """``int y w / int w`` over every group of *edges*, exactly.

    The table is zero outside its own domain, so a group that reaches past it
    averages that zero in: this is the mean of the function over the group, not
    over the part of the group the table happens to cover. A caller that wants
    the latter clips *edges* to the domain first. A group of zero weight (zero
    width) is ``nan``. With ``weight="1/x"`` every edge must be positive.
    """
    edges = np.asarray(edges, dtype=float)
    num = group_integrals(x, y, laws, edges, weight)
    lo, hi = edges[:-1], edges[1:]
    if weight is None:
        den = hi - lo
    else:
        if np.any(edges <= 0):
            raise ValueError("a 1/x weight needs positive group edges")
        den = np.log(hi / lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def legendre_moments(x, y, laws, max_order: int,
                     tol: float = LINEARIZATION_TOLERANCE) -> np.ndarray:
    """``[int y P_0 dx, ..., int y P_L dx]`` over the table's own domain.

    The Legendre polynomials are those of ``[-1, 1]``; the table is zero
    outside its domain, so a cosine table that does not reach both ends of
    ``[-1, 1]`` has to be extended by the caller if it means something else
    there. Exact for lin-lin and histogram panels (see the module docstring);
    log laws are re-expressed lin-lin to *tol* first
    (:func:`~kika.algebra.refine.to_linlin`), the one approximation, and a
    stated one.
    """
    x, y, laws = validate(x, y, laws)
    out = np.zeros(int(max_order) + 1)
    if x.size < 2:
        return out
    x, y = to_linlin(x, y, laws, tol)
    wide = np.diff(x) > 0
    a, b = x[:-1][wide], x[1:][wide]
    ya, yb = y[:-1][wide], y[1:][wide]
    nodes, weights = np.polynomial.legendre.leggauss(int(max_order) // 2 + 2)
    half = 0.5 * (b - a)
    t = (0.5 * (a + b))[:, None] + half[:, None] * nodes
    f = ya[:, None] + (yb - ya)[:, None] * (0.5 * (nodes + 1.0))
    p = np.polynomial.legendre.legvander(t.ravel(), int(max_order))
    return (half[:, None] * weights * f).ravel() @ p


def legendre_coefficients(mu, f, laws, max_order: int,
                          tol: float = LINEARIZATION_TOLERANCE) -> np.ndarray:
    """``a_0..a_L`` of an angular distribution tabulated in the cosine, ``a_0 = 1``.

    ``a_l = int f P_l dmu / int f dmu`` over ``[-1, 1]`` -- the convention
    ``f = sum (2l+1)/2 a_l P_l`` of ENDF MF4 -- with a table that stops short
    of either end held at its end value out to it: a cosine outside a table's
    range is the table's end point, not an absence of data. The integrals are
    :func:`legendre_moments`.
    """
    mu, f, laws = validate(mu, f, laws)
    if mu.size == 0:
        return np.zeros(int(max_order) + 1)
    if mu[0] > -1.0:
        mu, f, laws = np.r_[-1.0, mu], np.r_[f[0], f], np.r_[LINLIN, laws]
    if mu[-1] < 1.0:
        mu, f, laws = np.r_[mu, 1.0], np.r_[f, f[-1]], np.r_[laws, LINLIN]
    moments = legendre_moments(mu, f, laws, max_order, tol)
    if abs(moments[0]) > 1e-15:
        moments = moments / moments[0]
    return moments
