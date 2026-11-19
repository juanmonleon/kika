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

The weight ``1/x`` (an average that is uniform in ``ln x``) is also closed form on every
law but log-lin, whose ``int e^{kx}/x dx`` is an exponential integral; a
log-lin table has to be :func:`~kika.algebra.to_linlin`-ed first, and is
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

from .evaluate import _read_checked
from .laws import HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG, validate, vanishing_panels
from .refine import LINEARIZATION_TOLERANCE, to_linlin

__all__ = ["panel_integrals", "cumulative_integral", "integral", "group_integrals",
           "group_averages", "interval_integrals", "interval_averages",
           "log_window_averages", "legendre_moments", "legendre_coefficients", "WEIGHTS"]

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
        out = 1.0 - np.log1p(d) / d
    if small.any():
        e = d[small]
        out[small] = e / 2.0 - e ** 2 / 3.0 + e ** 3 / 4.0 - e ** 4 / 5.0
    return out


def panel_integrals(x, y, laws, weight: Optional[str] = None) -> np.ndarray:
    """``int_{x_i}^{x_{i+1}} y w dx`` for every interval, under its own law.

    *weight* is ``None`` (``w = 1``) or ``"1/x"``. A zero-width interval
    contributes nothing.

    Parameters
    ----------
    x, y, laws
        The table: abscissae (non-decreasing), ordinates, and one interpolation
        law per interval (codes 1-5). A single code applies to every interval.
    weight : {None, "1/x"}
        The weight under the integral. None is a weight of 1.

    Returns
    -------
    numpy.ndarray
        One integral per interval, ``len(x) - 1`` values (none for fewer than
        two points).

    Raises
    ------
    ValueError
        If the table is not valid (:func:`~kika.algebra.validate`); if
        *weight* is not one of the two above; or if it is ``"1/x"`` and an
        interval of positive width starts at ``x <= 0``, or has the log-lin law
        (4) and is not zero throughout (no closed form: see
        :func:`~kika.algebra.to_linlin`).
    """
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    x, y, laws = validate(x, y, laws)
    if x.size < 2:
        return np.zeros(0)
    return _integrate_panels(x[:-1], x[1:], y[:-1], y[1:], laws, weight)


def _integrate_panels(x1, x2, y1, y2, laws, weight) -> np.ndarray:
    """The integrals of the panels ``(x1, y1)-(x2, y2)`` under *laws*, checked arrays."""
    dx = x2 - x1
    out = np.zeros(dx.size)
    wide = dx > 0
    if weight == "1/x" and np.any(wide & (x1 <= 0)):
        raise ValueError("a 1/x weight needs positive abscissae")
    codes = np.unique(laws[wide])
    with np.errstate(divide="ignore", invalid="ignore"):
        for code in codes:
            # One law over panels that all have width is the common case, and
            # needs no copies.
            m = (slice(None) if codes.size == 1 and wide.all()
                 else wide & (laws == code))
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
                elif np.all(vanishing_panels(b1, b2)):
                    v = np.zeros(np.shape(b1))
                else:
                    raise ValueError(
                        "a log-lin panel has no closed-form 1/x integral (it is "
                        "an exponential integral); linearize the table first "
                        "with kika.algebra.to_linlin")
            if code in (LOGLIN, LOGLOG):
                v = np.where(vanishing_panels(b1, b2), 0.0, v)
            out[m] = v
    return out


def cumulative_integral(x, y, laws, weight: Optional[str] = None) -> np.ndarray:
    """``[0, int_{x0}^{x1}, int_{x0}^{x2}, ...]`` -- exact on the stated laws.

    Parameters
    ----------
    x, y, laws
        The table: abscissae (non-decreasing), ordinates, and one interpolation
        law per interval (codes 1-5). A single code applies to every interval.
    weight : {None, "1/x"}
        The weight under the integral. None is a weight of 1.

    Returns
    -------
    numpy.ndarray
        The running integral at every abscissa, starting at 0: ``len(x)``
        values for a table of two points or more.

    Raises
    ------
    ValueError
        As :func:`panel_integrals`.
    """
    p = panel_integrals(x, y, laws, weight)
    return np.concatenate(([0.0], np.cumsum(p)))


def group_integrals(x, y, laws, edges, weight: Optional[str] = None) -> np.ndarray:
    """``int_{e_j}^{e_{j+1}} y w dx`` for every group of *edges*.

    *edges* is non-decreasing and may reach past the table on either side --
    the table is zero there, so that part contributes nothing. Each edge that
    falls inside a panel cuts it into two panels of the same law, valued by
    that law at the edge, so the result is the table's own integral and not a
    quadrature of it.

    Parameters
    ----------
    x, y, laws
        The table: abscissae (non-decreasing), ordinates, and one interpolation
        law per interval (codes 1-5). A single code applies to every interval.
    edges : array_like
        Group boundaries, non-decreasing. They may reach past the table on
        either side, where the table is zero.
    weight : {None, "1/x"}
        The weight under the integral. None is a weight of 1.

    Returns
    -------
    numpy.ndarray
        One integral per group, ``len(edges) - 1`` values.

    Raises
    ------
    ValueError
        If *edges* is not a non-decreasing one-dimensional array, or on any of
        the conditions of :func:`panel_integrals`.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    x, y, laws = validate(x, y, laws)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or np.any(np.diff(edges) < 0):
        raise ValueError("edges must be a non-decreasing 1-d array")
    n_groups = max(edges.size - 1, 0)
    if x.size < 2 or n_groups == 0:
        return np.zeros(n_groups)
    cumulative = np.concatenate(([0.0], np.cumsum(
        _integrate_panels(x[:-1], x[1:], y[:-1], y[1:], laws, weight))))
    # The integral up to an edge: every whole panel left of it, plus the piece
    # of the panel it falls in, from that panel's start to the edge -- the same
    # law, valued at the edge by that law. At an edge on a node the piece is
    # empty, and at a repeated abscissa the zero-width interval adds nothing.
    c = np.clip(edges, x[0], x[-1])
    k = np.minimum(np.searchsorted(x, c, side="right") - 1, x.size - 2)
    totals = cumulative[k]
    inside = c > x[k]
    if inside.any():
        j, e = k[inside], c[inside]
        totals[inside] += _integrate_panels(
            x[j], e, y[j], _read_checked(x, y, laws, e, "point", "zero"), laws[j], weight)
    return np.diff(totals)


def integral(x, y, laws, lo: Optional[float] = None, hi: Optional[float] = None,
             weight: Optional[str] = None) -> float:
    """``int_lo^hi y w dx``; the limits default to the table's own domain.

    Limits outside the table add nothing; ``hi <= lo`` gives zero.

    Parameters
    ----------
    x, y, laws
        The table: abscissae (non-decreasing), ordinates, and one interpolation
        law per interval (codes 1-5). A single code applies to every interval.
    lo, hi : float, optional
        The limits of integration. None is the first and the last abscissa.
    weight : {None, "1/x"}
        The weight under the integral. None is a weight of 1.

    Returns
    -------
    float
        The integral; zero for a table of fewer than two points.

    Raises
    ------
    ValueError
        If the table is not valid (:func:`~kika.algebra.validate`), and,
        when the limits enclose a range, on the conditions of
        :func:`panel_integrals`.
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

    Parameters
    ----------
    x, y, laws
        The table: abscissae (non-decreasing), ordinates, and one interpolation
        law per interval (codes 1-5). A single code applies to every interval.
    edges : array_like
        Group boundaries, non-decreasing. They may reach past the table on
        either side, where the table is zero.
    weight : {None, "1/x"}
        The weight of the mean: the numerator is ``int y w`` and the
        denominator ``int w``. None is a weight of 1.

    Returns
    -------
    numpy.ndarray
        One mean per group, ``len(edges) - 1`` values; ``nan`` where the group
        has zero width.

    Raises
    ------
    ValueError
        If *weight* is ``"1/x"`` and an edge is not positive, or on any of the
        conditions of :func:`group_integrals`.
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


def interval_integrals(x, y, laws, lo, hi, weight: Optional[str] = None) -> np.ndarray:
    """``int_{lo_i}^{hi_i} y w dx`` for intervals that may overlap.

    :func:`group_integrals` needs edges that tile the line; here each interval
    stands on its own -- a window sliding along the table, say. The intervals'
    ends are sorted into one grid, the table is integrated exactly on every
    piece of it, and each interval sums only its own pieces. Differencing a
    running total instead would cancel: a narrow interval far from a region
    where ``y w`` is large can hold 1e-6 of the total below it, and the
    difference of two totals would lose those six digits.
    The cost is the number of piece ends inside each interval, summed, so
    overlapping intervals are cheap when there are thousands of them and
    quadratic when every one spans most of the others. ``hi <= lo`` gives zero.

    Parameters
    ----------
    x, y, laws
        The table: abscissae (non-decreasing), ordinates, and one interpolation
        law per interval (codes 1-5). A single code applies to every interval.
    lo, hi : array_like
        The ends of the intervals: one-dimensional, finite, of one shape. The
        intervals need not be sorted or disjoint.
    weight : {None, "1/x"}
        The weight under the integral. None is a weight of 1.

    Returns
    -------
    numpy.ndarray
        One integral per interval, in the order of *lo*; zero where
        ``hi <= lo``.

    Raises
    ------
    ValueError
        If *lo* and *hi* are not one-dimensional arrays of one shape, or hold a
        non-finite value; or on any of the conditions of
        :func:`group_integrals` (apart from the one on *edges*).
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.shape != hi.shape or lo.ndim != 1:
        raise ValueError("lo and hi must be 1-d arrays of one shape")
    if lo.size == 0:
        return np.zeros(0)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("interval ends must be finite")
    ends = np.unique(np.concatenate((lo, hi)))
    pieces = np.append(group_integrals(x, y, laws, ends, weight), 0.0)
    a = np.searchsorted(ends, lo)
    b = np.searchsorted(ends, np.maximum(hi, lo))
    # reduceat over (a_0, b_0, a_1, b_1, ...) sums pieces[a_i:b_i] at even
    # positions; an empty interval (a == b) returns pieces[a] and is zeroed.
    sums = np.add.reduceat(pieces, np.column_stack((a, b)).ravel())[::2]
    return np.where(b > a, sums, 0.0)


def interval_averages(x, y, laws, lo, hi, weight: Optional[str] = None) -> np.ndarray:
    """``int y w / int w`` over each interval ``[lo_i, hi_i]``, which may overlap.

    The same mean as :func:`group_averages` -- zero outside the table, ``nan``
    for an interval of zero weight -- for intervals that need not tile the
    line (:func:`interval_integrals`).

    Parameters
    ----------
    x, y, laws
        The table: abscissae (non-decreasing), ordinates, and one interpolation
        law per interval (codes 1-5). A single code applies to every interval.
    lo, hi : array_like
        The ends of the intervals: one-dimensional, finite, of one shape.
    weight : {None, "1/x"}
        The weight of the mean: the numerator is ``int y w`` and the
        denominator ``int w``. None is a weight of 1.

    Returns
    -------
    numpy.ndarray
        One mean per interval, in the order of *lo*; ``nan`` where
        ``hi <= lo``.

    Raises
    ------
    ValueError
        If *weight* is ``"1/x"`` and an end is not positive, or on any of the
        conditions of :func:`interval_integrals`.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    num = interval_integrals(x, y, laws, lo, hi, weight)
    if weight is None:
        den = hi - lo
    else:
        if np.any(lo <= 0) or np.any(hi <= 0):
            raise ValueError("a 1/x weight needs positive interval ends")
        den = np.log(hi / lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def log_window_averages(x, y, laws, centres, width: float,
                        weight: Optional[str] = None, bounds=None) -> np.ndarray:
    """The mean of the table over a window of fixed width in ``ln x``, at each centre.

    At a centre ``c`` the window is ``[c e^(-width/2), c e^(width/2)]``, clipped
    to the table's domain and to *bounds* ``(lo, hi)`` if given, and the mean is
    :func:`interval_averages`' exact one. Clipping makes the window one-sided
    near either end: the zero outside the table is not averaged in.
    Sliding the window gives a smooth curve with no group edges to place: at
    the geometric centre of a group of an equal-``ln x`` grid of that width it
    *is* that group's :func:`group_averages`. ``nan`` where the clipped window
    is empty.

    Parameters
    ----------
    x, y, laws
        The table: abscissae (non-decreasing), ordinates, and one interpolation
        law per interval (codes 1-5). A single code applies to every interval.
        At least two abscissae.
    centres : array_like
        Window centres: one-dimensional and positive.
    width : float
        Width of the window in ``ln x``: finite and positive.
    weight : {None, "1/x"}
        The weight of the mean, as in :func:`interval_averages`.
    bounds : (float, float), optional
        ``(lo, hi)``: a further clip of the windows, inside the table's domain.

    Returns
    -------
    numpy.ndarray
        One mean per centre, in its order.

    Raises
    ------
    ValueError
        If *x* has fewer than two abscissae, *centres* is not a
        one-dimensional array of positive values, or *width* is not finite and
        positive; or on any of the conditions of :func:`interval_averages`.
    """
    x = np.asarray(x, dtype=float)
    centres = np.asarray(centres, dtype=float)
    if x.size < 2:
        raise ValueError("need at least 2 abscissae")
    if centres.ndim != 1 or np.any(centres <= 0):
        raise ValueError("centres must be a 1-d array of positive abscissae")
    if not (np.isfinite(width) and width > 0):
        raise ValueError("width must be a positive interval of ln x")
    floor, ceiling = float(x[0]), float(x[-1])
    if bounds is not None:
        floor, ceiling = max(floor, float(bounds[0])), min(ceiling, float(bounds[1]))
    half = np.exp(0.5 * float(width))
    lo = np.clip(centres / half, floor, ceiling)
    hi = np.clip(centres * half, floor, ceiling)
    return interval_averages(x, y, laws, lo, hi, weight)


def legendre_moments(x, y, laws, max_order: int,
                     tol: float = LINEARIZATION_TOLERANCE) -> np.ndarray:
    """``[int y P_0 dx, ..., int y P_L dx]`` over the table's own domain.

    The Legendre polynomials are those of ``[-1, 1]``; the table is zero
    outside its domain, so a cosine table that does not reach both ends of
    ``[-1, 1]`` has to be extended by the caller if it means something else
    there. Exact for lin-lin and histogram panels (see the module docstring);
    log laws are re-expressed lin-lin to *tol* first
    (:func:`~kika.algebra.to_linlin`), the one approximation, and a
    stated one.

    Parameters
    ----------
    x, y, laws
        The table: abscissae (non-decreasing), ordinates, and one interpolation
        law per interval (codes 1-5). A single code applies to every interval.
    max_order : int
        The highest order ``L`` of the moments.
    tol : float
        Relative tolerance of the lin-lin re-expression of the log laws
        (:func:`~kika.algebra.to_linlin`); unused for lin-lin and
        histogram panels.

    Returns
    -------
    numpy.ndarray
        The moments of orders 0 to *max_order*, ``max_order + 1`` values; zeros
        for a table of fewer than two points.

    Raises
    ------
    ValueError
        If the table is not valid (:func:`~kika.algebra.validate`).
    RefinementError
        If the re-expression of a log law does not converge
        (:func:`~kika.algebra.to_linlin`).
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

    Parameters
    ----------
    mu, f, laws
        The table in the cosine: abscissae (non-decreasing), ordinates, and
        one interpolation law per interval (codes 1-5). A single code applies
        to every interval.
    max_order : int
        The highest order ``L`` of the coefficients.
    tol : float
        As in :func:`legendre_moments`.

    Returns
    -------
    numpy.ndarray
        ``a_0..a_L``, ``max_order + 1`` values, normalised to ``a_0 = 1``
        (left as they are when the integral of *f* is 1e-15 or less in
        magnitude); zeros for an empty table.

    Raises
    ------
    ValueError
        If the table is not valid (:func:`~kika.algebra.validate`).
    RefinementError
        As :func:`legendre_moments`.
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
