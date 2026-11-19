"""Error bars as one path per direction: ``Axes.errorbar`` makes one path per point,
and on a log axis transforms each in Python (3 x 20 000 points with x and y errors
take seconds). Here the segments of a direction share one ``Line2D``, split by NaN."""

from typing import Optional

import numpy as np


def _below_above(err, n: int):
    arr = np.asarray(err, dtype=float)
    try:
        arr = np.broadcast_to(arr, (2, n)) if arr.ndim == 2 else np.broadcast_to(arr, (n,))
    except ValueError:
        raise ValueError(f"errors must be a scalar, ({n},) or (2, {n}); got shape {np.shape(err)}")
    if arr.ndim == 2:
        return np.abs(arr[0]), np.abs(arr[1])
    return np.abs(arr), np.abs(arr)


def _polyline(a0, a1, b):
    gap = np.full_like(a0, np.nan)
    return np.column_stack([a0, a1, gap]).ravel(), np.column_stack([b, b, gap]).ravel()


def draw_errorbars(ax, x, y, yerr=None, xerr=None, *, color=None, linewidth: float = 1.5,
                   capsize: float = 0.0, capthick: Optional[float] = None,
                   alpha: Optional[float] = None, zorder: float = 2):
    """``ax.errorbar(x, y, yerr, xerr, fmt='none', ...)``, except that overlapping bars under ``alpha`` do not darken each other.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes drawn on.
    x, y : array_like
        Points, of one length. The errors are in the units of the axis they
        belong to.
    yerr, xerr : array_like, optional
        Error of each point: a scalar, ``(n,)`` for a symmetric error, or
        ``(2, n)`` for ``(below, above)``. Signs are ignored. None draws no
        bars in that direction.
    color : optional
        Colour of bars and caps. None takes the colour of the axes' cycle for
        each line drawn.
    linewidth : float
        Width of the bars, in points.
    capsize : float
        Length of the caps, in points. 0 draws no caps.
    capthick : float, optional
        Width of the caps, in points. None leaves matplotlib's default for the
        marker edge.
    alpha : float, optional
        Opacity of bars and caps.
    zorder : float
        Drawing order.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The lines drawn: one per direction for the bars, then one per direction
        for the caps. Empty when ``x`` has no points.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` differ in length, or an error array cannot be
        broadcast to ``(n,)`` or ``(2, n)`` (a scalar, or a ``(1,)`` or
        ``(1, n)`` array, is accepted and broadcast).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if y.size != n:
        raise ValueError(f"x and y must have the same length, got {n} and {y.size}")
    if n == 0:
        return []

    common = {'color': color, 'alpha': alpha, 'zorder': zorder}
    cap = {**common, 'linestyle': 'none', 'markersize': 2.0 * capsize, 'markeredgecolor': color}
    if capthick is not None:
        cap['markeredgewidth'] = capthick

    spans = []
    for err, dep, indep, horizontal in ((xerr, x, y, True), (yerr, y, x, False)):
        if err is not None:
            below, above = _below_above(err, n)
            spans.append((dep - below, dep + above, indep, horizontal))

    # Bars before caps, as Axes.errorbar adds them: the caps sit on top.
    artists = []
    for lo, hi, indep, horizontal in spans:
        a, b = _polyline(lo, hi, indep)
        xs, ys = (a, b) if horizontal else (b, a)
        artists += ax.plot(xs, ys, linewidth=linewidth, solid_capstyle='butt', **common)
    if capsize > 0:
        for lo, hi, indep, horizontal in spans:
            ends, at = np.r_[lo, hi], np.r_[indep, indep]
            xs, ys = (ends, at) if horizontal else (at, ends)
            artists += ax.plot(xs, ys, marker='|' if horizontal else '_', **cap)
    return artists
