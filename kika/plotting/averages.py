"""The averaged layers a series can carry, drawn alike by every builder.

A series is averaged by attaching ``PlotData.metadata['group_average_overlay']``,
a dict with ``bounds_used`` (the averaged span) and ``weighting``, and one or
both of two layers:

* ``centres``, ``values`` -- a sliding-window average
  (:func:`kika.processing.resonance_window_average`), drawn as a solid curve;
* ``edges``, ``xs`` -- group averages
  (:func:`kika.processing.resonance_group_average`), drawn as dashed steps.

With both, ``primary`` (``'window'``, the default, or ``'steps'``) names the one
that stands for the series: it carries the series label and the pointwise trace
meets it at the edges of the span.

Each series is averaged or not on its own; how the averaged ones are drawn is
one setting for the figure, *display*:

* ``'pointwise'`` -- only the pointwise curve, no layer;
* ``'average'`` -- the pointwise curve masked inside ``bounds_used``, the
  layers drawn there, the pointwise curve going on outside;
* ``'both'`` -- the pointwise curve faded over its whole range, the layers on
  top inside ``bounds_used``.

:class:`~kika.plotting.PlotBuilder` draws them on any plot
(:meth:`~kika.plotting.PlotBuilder.set_group_average`), and so on the main
panel of a comparison, which is a ``PlotBuilder``; the comparison adds the
difference panel. The averages themselves are computed elsewhere: this module
only draws.
"""
from __future__ import annotations

import copy
from typing import Iterable, List, Optional, Tuple

import numpy as np

__all__ = ["DISPLAYS", "overlay_of", "overlay_layers", "plot_layer",
           "main_panel_copy", "draw_layers", "average_bounds", "shade",
           "fit_y_to_layers", "layer_values"]

#: The values *display* takes.
DISPLAYS = ('pointwise', 'average', 'both')

#: Legend suffix of each averaged layer when it is not the series' only trace.
LAYER_NAMES = {'window': 'window avg', 'steps': 'group avg'}

#: Opacity of a pointwise trace drawn under its averages, so they read on top.
FADED_POINTWISE_ALPHA = 0.35

Layer = Tuple[str, np.ndarray, np.ndarray]


def _check_display(display: str) -> None:
    if display not in DISPLAYS:
        raise ValueError(f"display must be one of {DISPLAYS}, got {display!r}")


def overlay_layers(overlay: dict) -> List[Layer]:
    """The averaged layers an overlay carries, as ``(kind, x, y)``.

    For ``'window'`` x are the centres and y the averages at them; for
    ``'steps'`` x are the group edges and y the one value per group. The
    overlay's ``primary`` layer comes first (the window when unstated).
    """
    layers: List[Layer] = []
    if overlay.get('centres') is not None and overlay.get('values') is not None:
        x = np.asarray(overlay['centres'], dtype=float)
        y = np.asarray(overlay['values'], dtype=float)
        if x.size >= 2 and y.size == x.size:
            layers.append(('window', x, y))
    if overlay.get('edges') is not None and overlay.get('xs') is not None:
        x = np.asarray(overlay['edges'], dtype=float)
        y = np.asarray(overlay['xs'], dtype=float)
        if x.size >= 2 and y.size == x.size - 1:
            layers.append(('steps', x, y))
    if overlay.get('primary') == 'steps':
        layers.sort(key=lambda layer: layer[0] != 'steps')
    return layers


def overlay_of(data) -> Optional[dict]:
    """The ``group_average_overlay`` of a series, or None when it has no usable layer."""
    metadata = getattr(data, 'metadata', None)
    overlay = metadata.get('group_average_overlay') if metadata else None
    if not overlay or overlay.get('bounds_used') is None:
        return None
    if not overlay_layers(overlay):
        return None
    return overlay


def plot_layer(ax, kind: str, x: np.ndarray, y: np.ndarray, *,
               dashed: bool, **kwargs) -> None:
    """Draw one layer: steps when there is one y per interval, a curve otherwise.

    A difference of group averages against a pointwise curve is a ``'steps'``
    layer with one y per abscissa, so it is drawn as the curve it is.
    """
    linestyle = '--' if dashed else '-'
    if y.size == x.size - 1:
        # steps-post needs one extra y to match the edges; repeat the last.
        ax.plot(x, np.concatenate([y, y[-1:]]), drawstyle='steps-post',
                linestyle=linestyle, **kwargs)
    else:
        ax.plot(x, y, linestyle=linestyle, **kwargs)


def main_panel_copy(data, display: str):
    """The copy of an averaged series the plot draws as its pointwise trace, or None.

    None means draw *data* itself: it carries no overlay, or *display* is
    ``'pointwise'``. In ``'both'`` the copy is the whole curve, faded. In
    ``'average'`` it is masked inside ``bounds_used``, its first and last
    masked values bridged to the first layer's ends so the line meets the
    average instead of stopping short, and it has no label: the first layer
    stands for the series in the legend.
    """
    _check_display(display)
    if display == 'pointwise':
        return None
    overlay = overlay_of(data)
    if overlay is None:
        return None
    out = copy.copy(data)
    out.metadata = dict(data.metadata)
    out.metadata.pop('group_average_overlay', None)
    if display == 'both':
        out.alpha = FADED_POINTWISE_ALPHA
        return out
    out.label = None
    lo, hi = float(overlay['bounds_used'][0]), float(overlay['bounds_used'][1])
    first_layer = overlay_layers(overlay)[0][2]
    x = np.asarray(data.x, dtype=float)
    y = np.asarray(data.y, dtype=float)
    out.x = x
    idx = np.flatnonzero((x >= lo) & (x <= hi))
    if idx.size == 0:
        out.y = y
        return out
    first_in, last_in = int(idx[0]), int(idx[-1])
    new_y = y.copy()
    new_y[first_in:last_in + 1] = np.nan
    if np.isfinite(first_layer[0]):
        new_y[first_in] = float(first_layer[0])
    if last_in > first_in and np.isfinite(first_layer[-1]):
        new_y[last_in] = float(first_layer[-1])
    out.y = new_y
    return out


def draw_layers(ax, data, color, linewidth: Optional[float], display: str) -> bool:
    """Draw the averaged layers of one series in its color. True if anything was drawn.

    The window is solid, the steps dashed. In ``'average'`` the first layer
    carries the series' own label (its pointwise trace has none); otherwise
    each layer says which average it is.
    """
    _check_display(display)
    if display == 'pointwise':
        return False
    overlay = overlay_of(data)
    if overlay is None:
        return False
    series_label = data.label or ''
    for k, (kind, x, y) in enumerate(overlay_layers(overlay)):
        if display == 'average' and k == 0:
            label = series_label or None
        else:
            label = f'{series_label} ({LAYER_NAMES[kind]})' if series_label else None
        plot_layer(ax, kind, x, y, dashed=(kind == 'steps'),
                   linewidth=(linewidth or 1.5), color=color, label=label, alpha=0.95)
    return True


def layer_values(data, display: str) -> np.ndarray:
    """The finite y of the layers :func:`draw_layers` would draw, for fitting an axis to them."""
    if display == 'pointwise':
        return np.zeros(0)
    overlay = overlay_of(data)
    if overlay is None:
        return np.zeros(0)
    ys = [y[np.isfinite(y)] for _, _, y in overlay_layers(overlay)]
    return np.concatenate(ys) if ys else np.zeros(0)


def average_bounds(series: Iterable) -> Optional[Tuple[float, float]]:
    """The span covered by the averages of any of *series*, or None if none is averaged."""
    spans = [overlay_of(d)['bounds_used'] for d in series if overlay_of(d) is not None]
    if not spans:
        return None
    return (float(min(s[0] for s in spans)), float(max(s[1] for s in spans)))


def shade(axes, bounds: Optional[Tuple[float, float]]) -> None:
    """Tint the averaged span on each of *axes*."""
    if bounds is None:
        return
    for ax in axes:
        ax.axvspan(bounds[0], bounds[1], color='grey', alpha=0.08, linewidth=0, zorder=0)


def fit_y_to_layers(ax, y_lim) -> None:
    """Rescale the y axis to everything drawn, layers included, unless set.

    For a panel whose limits were fixed before its layers were drawn: with the
    pointwise trace masked inside the span, they would fit only what is left
    of it outside.
    """
    if y_lim is not None and any(v is not None for v in y_lim):
        return
    ax.relim(visible_only=True)
    ax.set_autoscaley_on(True)
    ax.autoscale_view(scalex=False, scaley=True)
