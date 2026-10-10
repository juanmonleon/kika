"""
Comparison utilities for plotting cross sections and other nuclear data.

This module provides:
- Grid interpolation between datasets with different x-grids
- Difference/ratio computation (absolute and relative)
- ComparisonBuilder for dual-subplot comparison figures

Examples
--------
>>> # Quick difference check using utility functions
>>> from kika.plotting import compute_difference, PlotBuilder
>>> result = compute_difference(ref_data, cmp_data, mode='relative')
>>> fig = PlotBuilder().add_data(result.difference).build()

>>> # Full dual-panel comparison figure
>>> from kika.plotting import ComparisonBuilder
>>> fig = (ComparisonBuilder()
...     .set_reference(ref_data)
...     .add_comparison(cmp_data)
...     .set_difference_panel(mode='relative')
...     .set_scales(log_x=True, log_y=True)
...     .build())
"""

from typing import Optional, Tuple, List, Literal, Union
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from kika.algebra import (difference, interpolate_to_grid, method_laws,
                          read_pair, steps_difference, window_difference)

from . import averages as _averages
from .plot_data import PlotData, DifferencePlotData, PlotItem
from .plot_builder import PlotBuilder, _NOT_SET
from .styles import (
    Style,
    get_style,
    _get_color_palette,
    format_energy_axis_ticks,
)
from ._backend_utils import (
    _is_notebook,
    _detect_interactive_backend,
    _configure_figure_interactivity,
)

# ``interpolate_to_grid`` and ``method_laws`` moved to :mod:`kika.algebra` in
# October 2026 with the rest of the arithmetic of a comparison; they stay
# importable from here.
__all__ = ["ComparisonBuilder", "ComparisonResult", "compute_difference",
           "interpolate_to_grid", "method_laws"]


def _unwrap(data):
    """PlotItem -> its PlotData; anything else unchanged."""
    return data.data if isinstance(data, PlotItem) else data


# ---------------------------------------------------------------------------
# Auto-interpolation defaults by PlotData subclass
# ---------------------------------------------------------------------------

#: Where the 'ref: <label>' annotation goes, per corner: axes x, y, ha, va.
_REFERENCE_LABEL_ANCHORS = {
    'upper left': (0.02, 0.97, 'left', 'top'),
    'upper right': (0.98, 0.97, 'right', 'top'),
    'lower left': (0.02, 0.03, 'left', 'bottom'),
    'lower right': (0.98, 0.03, 'right', 'bottom'),
}

_INTERPOLATION_DEFAULTS = {
    'CrossSectionPlotData': 'log-log',
    'AngularDistributionPlotData': 'lin-lin',
    'LegendreCoeffPlotData': 'log-log',
    'MultigroupCrossSectionPlotData': 'lin-lin',
    'MultigroupUncertaintyPlotData': 'lin-lin',
}


# ---------------------------------------------------------------------------
# Difference computation
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """
    Result of comparing two datasets.

    Attributes
    ----------
    reference : PlotData
        Reference data on the common grid.
    comparison : PlotData
        Comparison data on the common grid.
    difference : DifferencePlotData
        Computed difference (relative or absolute).
    common_x : np.ndarray
        Common x-grid used for the comparison.
    valid_mask : np.ndarray
        Boolean mask – ``True`` where both datasets have finite values.
    """
    reference: PlotData
    comparison: PlotData
    difference: DifferencePlotData
    common_x: np.ndarray
    valid_mask: np.ndarray


def compute_difference(
    reference: PlotData,
    comparison: PlotData,
    mode: Literal['relative', 'absolute'] = 'relative',
    interpolation: Optional[str] = 'log-log',
    grid: Literal['reference', 'comparison', 'union'] = 'union',
    relative_in_percent: bool = True,
) -> ComparisonResult:
    """
    Compute the difference between two :class:`PlotData` datasets.

    Handles different x-grids by interpolating onto a common grid.

    Parameters
    ----------
    reference : PlotData
        Baseline dataset.
    comparison : PlotData
        Dataset to compare against the baseline.
    mode : {'relative', 'absolute'}
        * ``'relative'``: ``(comparison - reference) / reference``
        * ``'absolute'``: ``comparison - reference``
    interpolation : str
        Interpolation method passed to :func:`interpolate_to_grid`.
    grid : {'reference', 'comparison', 'union'}
        Which grid to use as common grid.

        * ``'union'`` (default): every abscissa of both series
          (:func:`kika.algebra.on_common_grid`). Neither loses a node, so the
          extremes of the difference of two lin-lin tables are all in it.
        * ``'reference'``: the reference's own grid; the comparison is only
          sampled there. Whatever it has between those nodes is not seen: on
          six ENDF/B-VIII.1 materials, KIKA against NJOY's grid showed a
          maximum difference of 5e-4 where the union shows 1.0-2.0e-3.
        * ``'comparison'``: the comparison's own grid, likewise.
    relative_in_percent : bool
        If ``True`` and *mode* is ``'relative'``, multiply by 100.

    Returns
    -------
    ComparisonResult

    Raises
    ------
    ValueError
        If the datasets have no overlapping x-range.
    """
    x_ref = np.asarray(reference.x, dtype=float)
    y_ref = np.asarray(reference.y, dtype=float)
    x_cmp = np.asarray(comparison.x, dtype=float)
    y_cmp = np.asarray(comparison.y, dtype=float)

    # Overlapping range
    x_lo = max(x_ref[0], x_cmp[0])
    x_hi = min(x_ref[-1], x_cmp[-1])

    if x_lo >= x_hi:
        raise ValueError(
            f"No overlapping x-range. "
            f"Reference: [{x_ref[0]:.6e}, {x_ref[-1]:.6e}], "
            f"Comparison: [{x_cmp[0]:.6e}, {x_cmp[-1]:.6e}]"
        )

    if grid not in ('reference', 'comparison', 'union'):
        raise ValueError(f"Unknown grid option: {grid!r}")
    if mode not in ('relative', 'absolute'):
        raise ValueError(f"Unknown mode: {mode!r}")
    common_x, (y_ref_interp, y_cmp_interp) = read_pair(
        (x_ref, y_ref), (x_cmp, y_cmp), interpolation or 'lin-lin',
        {'reference': 0, 'comparison': 1, 'union': 'union'}[grid])

    valid = np.isfinite(y_ref_interp) & np.isfinite(y_cmp_interp)
    diff = difference(y_ref_interp, y_cmp_interp, mode, relative_in_percent)

    # Build output PlotData objects preserving original styling
    ref_out = PlotData(
        x=common_x, y=y_ref_interp,
        label=reference.label,
        color=reference.color,
        linestyle=reference.linestyle,
        linewidth=reference.linewidth,
        plot_type=reference.plot_type,
        metadata={**reference.metadata, 'interpolated': grid != 'reference'},
    )
    cmp_out = PlotData(
        x=common_x, y=y_cmp_interp,
        label=comparison.label,
        color=comparison.color,
        linestyle=comparison.linestyle,
        linewidth=comparison.linewidth,
        plot_type=comparison.plot_type,
        metadata={**comparison.metadata, 'interpolated': grid != 'comparison'},
    )

    diff_out = DifferencePlotData(
        x=common_x,
        y=diff,
        difference_type=mode,
        reference_label=reference.label,
        comparison_label=comparison.label,
    )

    return ComparisonResult(
        reference=ref_out,
        comparison=cmp_out,
        difference=diff_out,
        common_x=common_x,
        valid_mask=valid,
    )


# ---------------------------------------------------------------------------
# ComparisonBuilder
# ---------------------------------------------------------------------------

class ComparisonBuilder:
    """
    Builder for comparison plots with an optional difference panel.

    Creates figures in two layouts:

    1. **Single-panel** – overlay of reference + comparisons.
    2. **Dual-panel** – main overlay on top, difference panel below
       (enabled via :meth:`set_difference_panel`).

    Examples
    --------
    >>> fig = (ComparisonBuilder()
    ...     .set_reference(ref_data)
    ...     .add_comparison(cmp_data)
    ...     .set_difference_panel(mode='relative')
    ...     .set_labels(title='Elastic XS', x_label='Energy (MeV)',
    ...                 y_label='Cross Section (b)')
    ...     .set_scales(log_x=True, log_y=True)
    ...     .build())
    """

    def __init__(
        self,
        style: Union[str, Style] = 'light',
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 100,
        font_family: Optional[str] = None,
        notebook_mode: Optional[bool] = None,
        interactive: Optional[bool] = None,
        interpolation: Optional[str] = None,
        grid_strategy: Literal['reference', 'comparison', 'union'] = 'union',
    ):
        get_style(style)  # fail early on an unknown style name
        self._style = style
        self._figsize = figsize
        self._dpi = dpi
        self._font_family = font_family
        self._notebook_mode = notebook_mode
        self._interactive = interactive
        self._interpolation = interpolation
        self._grid_strategy = grid_strategy

        # Data
        self._reference: Optional[PlotData] = None
        self._reference_styling: dict = {}
        self._comparisons: List[Tuple[PlotData, dict]] = []
        self._overlays: List[Tuple[PlotData, dict]] = []
        self._scatter_overlays: List[Tuple[PlotData, dict]] = []

        # Difference-panel config
        self._show_diff_panel: bool = False
        self._diff_only: bool = False
        self._diff_mode: str = 'relative'
        self._diff_y_label: Optional[str] = None
        self._diff_y_lim: Optional[Tuple[float, float]] = None
        self._relative_in_percent: bool = True
        self._height_ratios: Tuple[float, float] = (3.0, 1.0)
        self._zero_line: bool = True
        self._diff_log_y: bool = False

        # Shared settings forwarded to PlotBuilder instances
        self._title = _NOT_SET
        self._x_label: Optional[str] = None
        self._y_label: Optional[str] = None
        self._use_log_x: bool = False
        self._use_log_y: bool = False
        self._x_lim: Optional[Tuple[float, float]] = None
        self._y_lim: Optional[Tuple[float, float]] = None
        self._legend_loc: str = 'best'
        self._legend_ncol: Optional[int] = None
        self._grid: Optional[bool] = None  # None: the style decides
        self._grid_alpha: Optional[float] = None  # None: the style's grid.alpha
        self._show_minor_grid: Optional[bool] = None
        self._minor_grid_alpha: float = 0.15
        self._show_minor_grid_x: Optional[bool] = None
        self._show_minor_grid_y: Optional[bool] = None

        # Averages. A series is averaged on its own, by carrying a
        # 'group_average_overlay' (kika.plotting.averages); these say how
        # every averaged series is drawn (see set_group_average).
        self._main_display: Literal['pointwise', 'average', 'both'] = 'both'
        self._shade_average_range: bool = False
        # "ref: <label>" annotation on the diff and diff-only panels.
        self._show_reference_label: bool = True
        self._reference_label_fontsize: float = 11
        self._reference_label_loc: str = 'upper left'

    # ---- fluent API -------------------------------------------------------

    def set_reference(self, data: PlotData, **styling) -> 'ComparisonBuilder':
        """Set the reference (baseline) dataset. A PlotItem is accepted too."""
        self._reference = _unwrap(data)
        self._reference_styling = styling
        return self

    def add_comparison(self, data: PlotData, **styling) -> 'ComparisonBuilder':
        """Add a comparison dataset. A PlotItem is accepted too."""
        self._comparisons.append((_unwrap(data), styling))
        return self

    def add_overlay(self, data: PlotData, **styling) -> 'ComparisonBuilder':
        """Add overlay data to the main panel only.

        Overlays are rendered on the main panel but are NOT included in
        difference computations.
        """
        self._overlays.append((_unwrap(data), styling))
        return self

    def add_scatter_overlay(self, data: PlotData, **styling) -> 'ComparisonBuilder':
        """Add scatter overlay data (e.g., EXFOR) to both main and diff panels.

        Scatter overlays are rendered on the main panel AND their difference
        against the reference is shown as scatter points in the diff panel.
        """
        self._scatter_overlays.append((_unwrap(data), styling))
        return self

    def set_difference_panel(
        self,
        mode: Literal['relative', 'absolute'] = _NOT_SET,
        y_label: Optional[str] = _NOT_SET,
        y_lim: Optional[Tuple[Optional[float], Optional[float]]] = _NOT_SET,
        height_ratios: Tuple[float, float] = _NOT_SET,
        relative_in_percent: bool = _NOT_SET,
        zero_line: bool = _NOT_SET,
        only: bool = _NOT_SET,
        log_y: bool = _NOT_SET,
    ) -> 'ComparisonBuilder':
        """
        Enable and configure the difference sub-panel.

        Can be called multiple times — only the parameters you pass are
        updated; everything else keeps its previous value.

        Parameters
        ----------
        mode : {'relative', 'absolute'}
            Difference type (default ``'relative'``).
        y_label : str, optional
            Custom y-axis label for the diff panel.
        y_lim : tuple, optional
            ``(min, max)`` y-axis limits for the diff panel.
        height_ratios : tuple
            ``(main, diff)`` height ratio (default ``(3.0, 1.0)``).
        relative_in_percent : bool
            Multiply relative differences by 100 (default ``True``).
        zero_line : bool
            Draw a dashed zero reference line (default ``True``).
        only : bool
            If ``True``, show *only* the difference panel (no main overlay).
        log_y : bool
            Use logarithmic y-axis on the difference panel (default ``False``).
        """
        self._show_diff_panel = True
        if mode is not _NOT_SET:
            self._diff_mode = mode
        if y_label is not _NOT_SET:
            self._diff_y_label = y_label
        if y_lim is not _NOT_SET:
            self._diff_y_lim = y_lim
        if height_ratios is not _NOT_SET:
            self._height_ratios = height_ratios
        if relative_in_percent is not _NOT_SET:
            self._relative_in_percent = relative_in_percent
        if zero_line is not _NOT_SET:
            self._zero_line = zero_line
        if only is not _NOT_SET:
            self._diff_only = only
        if log_y is not _NOT_SET:
            self._diff_log_y = log_y
        return self

    def set_labels(
        self,
        title=_NOT_SET,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> 'ComparisonBuilder':
        """Set plot labels."""
        if title is not _NOT_SET:
            self._title = title
        if x_label is not None:
            self._x_label = x_label
        if y_label is not None:
            self._y_label = y_label
        return self

    def set_scales(
        self, log_x: bool = False, log_y: bool = False,
    ) -> 'ComparisonBuilder':
        """Set axis scales."""
        self._use_log_x = log_x
        self._use_log_y = log_y
        return self

    def set_limits(
        self,
        x_lim: Optional[Tuple[Optional[float], Optional[float]]] = None,
        y_lim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> 'ComparisonBuilder':
        """Set axis limits for the main panel."""
        self._x_lim = x_lim
        self._y_lim = y_lim
        return self

    def set_legend(
        self, loc: str = 'best', ncol: Optional[int] = None,
    ) -> 'ComparisonBuilder':
        """Set legend placement."""
        self._legend_loc = loc
        self._legend_ncol = ncol
        return self

    def set_grid(
        self,
        grid: Optional[bool] = True,
        alpha: Optional[float] = 0.3,
        show_minor: Optional[bool] = None,
        minor_alpha: float = 0.15,
        show_minor_x: Optional[bool] = None,
        show_minor_y: Optional[bool] = None,
    ) -> 'ComparisonBuilder':
        """Configure grid display settings for reference and comparison panels.

        `show_minor_x` / `show_minor_y` override `show_minor` for one axis; see
        `PlotBuilder.set_grid`. Both panels get the same grid, since they share
        the abscissa and are read as one figure.
        """
        self._grid = grid
        self._grid_alpha = alpha
        self._show_minor_grid = show_minor
        self._minor_grid_alpha = minor_alpha
        self._show_minor_grid_x = show_minor if show_minor_x is None else show_minor_x
        self._show_minor_grid_y = show_minor if show_minor_y is None else show_minor_y
        return self

    def set_group_average(
        self,
        main_display: Literal['pointwise', 'average', 'both'] = 'both',
        shade_range: bool = False,
    ) -> 'ComparisonBuilder':
        """How averaged series are drawn, on the main panel and the diff.

        Averaging is not a mode of the comparison: each series is averaged or
        not on its own, by carrying ``PlotData.metadata['group_average_overlay']``
        (the payload, and what ``main_display`` does on the main panel, are in
        :mod:`kika.plotting.averages`; the main panel is a
        :class:`PlotBuilder` and draws them with
        :meth:`PlotBuilder.set_group_average`). The comparison compares
        whatever each series draws, so a series can be compared with its own
        average.

        On the diff panel, for each comparison:

        * both averaged -- the diff of every layer they both carry on the
          same abscissae, value by value;
        * one averaged -- each of its layers against the other's pointwise
          curve: a window at its centres, the steps against the pointwise
          curve on its own grid, group by group (:mod:`kika.algebra.compare`);
        * neither -- the pointwise diff alone.

        Inside the averaged span the pointwise diff gives way to the averaged
        one whatever ``main_display`` is: through resonances a pointwise diff
        is a solid band that hides it. A relative diff divides by the
        reference, averaged or not. Unless the y limits are set, each panel's
        y axis covers the averaged layers too. ``shade_range`` tints the
        averaged span on every panel, so a span narrower than the plot is
        visible.
        """
        if main_display not in _averages.DISPLAYS:
            raise ValueError(f"main_display must be one of {_averages.DISPLAYS}, "
                             f"got {main_display!r}")
        self._main_display = main_display
        self._shade_average_range = shade_range
        return self

    def set_reference_label(
        self, show: bool = True, fontsize: Optional[float] = None,
        loc: Optional[str] = None,
    ) -> 'ComparisonBuilder':
        """Toggle the 'ref: <label>' annotation on the diff panel.

        ``fontsize`` scales the annotation; pass the legend fontsize so the
        label tracks it. ``loc`` is the corner it sits in, one of
        ``'upper left'`` (the default), ``'upper right'``, ``'lower left'``
        and ``'lower right'``, so it can be moved off a curve that runs
        under it. ``None`` keeps the current value of either.
        """
        if loc is not None and loc not in _REFERENCE_LABEL_ANCHORS:
            raise ValueError(f"loc must be one of {sorted(_REFERENCE_LABEL_ANCHORS)}, "
                             f"got {loc!r}")
        self._show_reference_label = show
        if fontsize is not None:
            self._reference_label_fontsize = fontsize
        if loc is not None:
            self._reference_label_loc = loc
        return self

    def _draw_reference_label(self, ax) -> None:
        """The 'ref: <label>' annotation in its corner of *ax*, when shown."""
        if not self._show_reference_label:
            return
        x, y, ha, va = _REFERENCE_LABEL_ANCHORS[self._reference_label_loc]
        ax.text(
            x, y, f"ref: {self._reference.label or 'reference'}",
            transform=ax.transAxes, fontsize=self._reference_label_fontsize,
            va=va, ha=ha, fontstyle='italic', alpha=0.7,
        )

    # ---- interpolation inference ------------------------------------------

    def _infer_interpolation(self, data: PlotData) -> str:
        """The data's own interpolation law when it states one, else a guess from its class."""
        if getattr(data, 'interpolation', None) in ('log-log', 'lin-lin', 'log-lin', 'lin-log'):
            return data.interpolation
        class_name = type(data).__name__
        return _INTERPOLATION_DEFAULTS.get(class_name, 'log-log')

    # ---- averaged layers on the difference panel ---------------------------

    def _layers_diff(
        self, ref: PlotData, cmp: PlotData, interpolation: str,
    ) -> Optional[Tuple[List[Tuple[str, np.ndarray, np.ndarray]], Tuple[float, float]]]:
        """Return ``(layers, bounds)`` of averaged diffs for one comparison, or None.

        Either side may be averaged, both, or neither (None):

        * both -- each layer both carry on the same abscissae is diffed value
          by value; the app averages every series over one range with one
          width, so they match. A layer they do not share is dropped, and
          None (no layer left) leaves the pointwise diff standing.
        * one -- each of its layers against the other's pointwise curve, read
          under *interpolation*: a window at its centres
          (:func:`kika.algebra.window_difference`), the steps on the pointwise
          grid group by group (:func:`kika.algebra.steps_difference`).

        A relative diff divides by the reference, averaged or not.
        """
        ref_overlay, cmp_overlay = _averages.overlay_of(ref), _averages.overlay_of(cmp)
        if ref_overlay is None and cmp_overlay is None:
            return None
        mode, percent = self._diff_mode, self._relative_in_percent
        layers: List[Tuple[str, np.ndarray, np.ndarray]] = []
        if ref_overlay is not None and cmp_overlay is not None:
            cmp_layers = {kind: (x, y) for kind, x, y in _averages.overlay_layers(cmp_overlay)}
            for kind, ref_x, ref_y in _averages.overlay_layers(ref_overlay):
                if kind not in cmp_layers:
                    continue
                cmp_x, cmp_y = cmp_layers[kind]
                if ref_x.shape != cmp_x.shape or not np.allclose(ref_x, cmp_x):
                    continue
                layers.append((kind, ref_x, difference(ref_y, cmp_y, mode, percent)))
            spans = (ref_overlay['bounds_used'], cmp_overlay['bounds_used'])
        else:
            if ref_overlay is not None:
                overlay, pointwise, side = ref_overlay, cmp, 'reference'
            else:
                overlay, pointwise, side = cmp_overlay, ref, 'comparison'
            px = np.asarray(pointwise.x, dtype=float)
            py = np.asarray(pointwise.y, dtype=float)
            for kind, x, y in _averages.overlay_layers(overlay):
                if kind == 'window':
                    layers.append((kind, x, window_difference(
                        x, y, px, py, interpolation, average_is=side,
                        mode=mode, percent=percent)))
                else:
                    u, diff = steps_difference(
                        x, y, px, py, interpolation, average_is=side,
                        mode=mode, percent=percent)
                    if u.size:
                        layers.append((kind, u, diff))
            spans = (overlay['bounds_used'],)
        if not layers:
            return None
        lo = float(min(span[0] for span in spans))
        hi = float(max(span[1] for span in spans))
        return layers, (lo, hi)

    @staticmethod
    def _mask_pointwise_in_range(
        diff_data: DifferencePlotData, lo: float, hi: float,
        bridge_values: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Replace diff values inside [lo, hi] with NaN so the averaged
        diff can occupy that region without visual overlap.

        ``bridge_values=(left, right)`` overrides the first/last masked
        y so the pointwise line has real (non-NaN) endpoints at the
        range boundaries. Without bridges, matplotlib drops the last
        segment before NaN and the first segment after NaN, leaving a
        visible gap between the pointwise and the averaged trace. The
        bridge values should be the averaged diff's leading and
        trailing values so the pointwise line meets it at the boundary.
        """
        x = np.asarray(diff_data.x, dtype=float)
        y = np.asarray(diff_data.y, dtype=float)
        in_range = (x >= lo) & (x <= hi)
        if not np.any(in_range):
            return
        idx = np.where(in_range)[0]
        first_in, last_in = int(idx[0]), int(idx[-1])
        y = y.copy()
        y[first_in:last_in + 1] = np.nan
        if bridge_values is not None:
            left, right = bridge_values
            if np.isfinite(left):
                y[first_in] = float(left)
            if last_in > first_in and np.isfinite(right):
                y[last_in] = float(right)
        diff_data.y = y

    def _apply_overlay_diff(
        self, diff_data: DifferencePlotData, cmp_data: PlotData, interpolation: str,
    ) -> Optional[List[Tuple[str, np.ndarray, np.ndarray]]]:
        """Diff the averaged layers of one comparison and mask its pointwise diff.

        Returns the averaged diff layers to draw, or None. The pointwise diff
        is masked inside the range and bridged to the first layer, in every
        display mode (see :meth:`set_group_average`).
        """
        overlay_diff = self._layers_diff(self._reference, cmp_data, interpolation)
        if overlay_diff is None:
            return None
        layers, (lo, hi) = overlay_diff
        first = layers[0][2]
        bridge = (float(first[0]), float(first[-1])) if first.size else (np.nan, np.nan)
        self._mask_pointwise_in_range(diff_data, lo, hi, bridge_values=bridge)
        return layers

    def _draw_overlay_diff(
        self, ax, layers: List[Tuple[str, np.ndarray, np.ndarray]],
        color: Optional[str], linewidth: float,
    ) -> None:
        """Render the averaged diff layers on a diff panel.

        A lone layer is solid, so the diff reads as one continuous curve
        stitched into the pointwise diff; with both, the steps are dashed
        as on the main panel.
        """
        for kind, x, y in layers:
            _averages.plot_layer(
                ax, kind, x, y, dashed=(kind == 'steps' and len(layers) > 1),
                linewidth=linewidth or 1.5, color=color, label=None, alpha=1.0,
            )

    def _shade_range(self, *axes) -> None:
        """Tint the averaged span on each axis, when asked to."""
        if not self._shade_average_range:
            return
        series = [self._reference] + [data for data, _ in self._comparisons]
        _averages.shade(axes, _averages.average_bounds(series))


    # ---- build ------------------------------------------------------------

    def build(self, show: bool = False) -> plt.Figure:
        """
        Build and return the comparison figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._reference is None:
            raise ValueError(
                "No reference data set. Call set_reference() first."
            )
        if not self._comparisons and not self._scatter_overlays:
            raise ValueError(
                "No comparison or scatter overlay data. "
                "Call add_comparison() or add_scatter_overlay() at least once."
            )

        # Everything is compared in the reference's units (a no-op for data
        # without unit metadata)
        self._to_reference_units()

        # Resolve interpolation: explicit value wins, otherwise infer
        interpolation = self._interpolation
        if interpolation is None:
            interpolation = self._infer_interpolation(self._reference)

        # Pre-compute differences if the panel is requested
        results: List[ComparisonResult] = []
        if self._show_diff_panel:
            for cmp_data, _ in self._comparisons:
                result = compute_difference(
                    reference=self._reference,
                    comparison=cmp_data,
                    mode=self._diff_mode,
                    interpolation=interpolation,
                    grid=self._grid_strategy,
                    relative_in_percent=self._relative_in_percent,
                )
                results.append(result)

        if self._show_diff_panel and self._diff_only:
            return self._build_diff_only_panel(results, show, interpolation)
        elif self._show_diff_panel:
            return self._build_dual_panel(results, show, interpolation)
        else:
            return self._build_single_panel(show)

    # ---- internal ---------------------------------------------------------

    def _to_reference_units(self) -> None:
        """Express comparisons and overlays in the reference's units.

        Differences computed between an eV curve and an MeV one are meaningless
        and used to be computed without complaint.
        """
        import warnings
        from .units import MixedQuantityWarning, UnitError, data_to_units

        ref = self._reference
        x_unit, y_unit = getattr(ref, 'x_unit', None), getattr(ref, 'y_unit', None)
        if not x_unit and not y_unit:
            return

        def convert(entries):
            out = []
            for data, styling in entries:
                try:
                    data = data_to_units(data, x_unit, y_unit)
                except UnitError as exc:
                    warnings.warn(f"{data.label or 'A curve'} left unconverted: {exc}",
                                  MixedQuantityWarning, stacklevel=4)
                out.append((data, styling))
            return out

        self._comparisons = convert(self._comparisons)
        self._overlays = convert(self._overlays)
        self._scatter_overlays = convert(self._scatter_overlays)

    def _resolve_diff_y_label(self) -> str:
        """Build the default y-axis label for difference panels."""
        if self._diff_y_label is not None:
            return self._diff_y_label
        if self._diff_mode == 'relative':
            return (
                'Relative Diff (%)'
                if self._relative_in_percent
                else 'Relative Diff'
            )
        return 'Absolute Diff'

    def _build_diff_only_panel(
        self, results: List[ComparisonResult], show: bool,
        interpolation: str = 'log-log',
    ) -> plt.Figure:
        """Single-panel figure showing only difference curves."""
        builder = PlotBuilder(
            style=self._style,
            figsize=self._figsize,
            dpi=self._dpi,
            font_family=self._font_family,
            notebook_mode=self._notebook_mode,
            interactive=self._interactive,
        )

        colors = _get_color_palette(self._style)
        overlay_diff_draws: List[Tuple[List[Tuple[str, np.ndarray, np.ndarray]], str, float]] = []
        for i, (result, (cmp_data, _)) in enumerate(
            zip(results, self._comparisons)
        ):
            diff_data = result.difference
            # Use per-series diff_label from metadata if provided:
            #   not present → use comparison label (default)
            #   empty string → suppress label (None)
            #   non-empty → use as-is
            raw_diff_label = cmp_data.metadata.get('diff_label')
            if raw_diff_label is None:
                diff_data.label = cmp_data.label
            else:
                diff_data.label = raw_diff_label or None
            color_idx = (i + 1) % len(colors)
            diff_color = cmp_data.color if cmp_data.color else colors[color_idx]

            layers = self._apply_overlay_diff(diff_data, cmp_data, interpolation)
            if layers is not None:
                overlay_diff_draws.append((layers, diff_color, cmp_data.linewidth or 1.5))

            builder.add_data(
                diff_data, color=diff_color, linewidth=cmp_data.linewidth or 1.5
            )

        # Add scatter overlays to diff-only panel
        for ovl_data, ovl_styling in self._scatter_overlays:
            try:
                ovl_result = compute_difference(
                    reference=self._reference,
                    comparison=ovl_data,
                    mode=self._diff_mode,
                    interpolation=interpolation,
                    grid='comparison',
                    relative_in_percent=self._relative_in_percent,
                )
                ovl_diff = ovl_result.difference
                ovl_diff.label = ovl_data.label
                ovl_diff_styling = {
                    'color': ovl_data.color or ovl_styling.get('color'),
                    'marker': ovl_data.marker or ovl_styling.get('marker', 'o'),
                    'markersize': ovl_data.markersize or ovl_styling.get('markersize', 5),
                    'linestyle': 'none',
                }
                builder.add_data(ovl_diff, **ovl_diff_styling)
            except Exception:
                pass

        builder.set_labels(
            title=self._title,
            x_label=self._x_label,
            y_label=self._resolve_diff_y_label(),
        )
        builder.set_scales(log_x=self._use_log_x, log_y=self._diff_log_y)
        # In diff-only view the diff curve is the only plot, so its Y axis is
        # driven by the main figure-settings Y limits (self._y_lim). Fall back
        # to the comparison panel's diff Y limits only when those are unset.
        diff_only_y_lim = self._y_lim if self._y_lim is not None else self._diff_y_lim
        builder.set_limits(x_lim=self._x_lim, y_lim=diff_only_y_lim)
        builder.set_legend(loc=self._legend_loc, ncol=self._legend_ncol)
        builder.set_grid(
            grid=self._grid,
            alpha=self._grid_alpha,
            show_minor=self._show_minor_grid,
            minor_alpha=self._minor_grid_alpha,
            show_minor_x=self._show_minor_grid_x,
            show_minor_y=self._show_minor_grid_y,
        )

        fig = builder.build(show=False)
        ax = fig.axes[0]

        # Group-average bin-diff traces (step-post) on top of the
        # masked pointwise diff.
        for layers, color, lw in overlay_diff_draws:
            self._draw_overlay_diff(ax, layers, color, lw)
        if overlay_diff_draws:
            _averages.fit_y_to_layers(ax, diff_only_y_lim)
        self._shade_range(ax)

        # Zero reference line
        if self._zero_line:
            ax.axhline(
                y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7,
            )

        self._draw_reference_label(ax)

        if show:
            plt.show()

        return fig

    def _build_single_panel(self, show: bool) -> plt.Figure:
        """Single-panel overlay plot."""
        builder = PlotBuilder(
            style=self._style,
            figsize=self._figsize,
            dpi=self._dpi,
            font_family=self._font_family,
            notebook_mode=self._notebook_mode,
            interactive=self._interactive,
        )

        builder.add_data(self._reference, **self._reference_styling)
        for cmp_data, styling in self._comparisons:
            builder.add_data(cmp_data, **styling)
        for ovl_data, ovl_styling in self._overlays:
            builder.add_data(ovl_data, **ovl_styling)
        for ovl_data, ovl_styling in self._scatter_overlays:
            builder.add_data(ovl_data, **ovl_styling)

        builder.set_labels(
            title=self._title, x_label=self._x_label, y_label=self._y_label,
        )
        builder.set_scales(log_x=self._use_log_x, log_y=self._use_log_y)
        builder.set_limits(x_lim=self._x_lim, y_lim=self._y_lim)
        builder.set_legend(loc=self._legend_loc, ncol=self._legend_ncol)
        builder.set_grid(
            grid=self._grid,
            alpha=self._grid_alpha,
            show_minor=self._show_minor_grid,
            minor_alpha=self._minor_grid_alpha,
            show_minor_x=self._show_minor_grid_x,
            show_minor_y=self._show_minor_grid_y,
        )

        # The averaged layers are drawn by the builder itself.
        builder.set_group_average(self._main_display, self._shade_average_range)
        fig = builder.build(show=False)

        if show:
            plt.show()
        return fig

    def _dual_panel_grid(self) -> bool:
        """Grid on/off for the dual panel, whose builders draw on axes they do not own.

        Called inside the style context, so ``rcParams`` is the style's.
        """
        return self._grid if self._grid is not None else bool(mpl.rcParams['axes.grid'])

    def _build_dual_panel(self, *args, **kwargs) -> plt.Figure:
        """Open the style's rc context around :meth:`_draw_dual_panel`."""
        notebook = (
            self._notebook_mode if self._notebook_mode is not None
            else _is_notebook()
        )
        with get_style(self._style).context(
            notebook_mode=notebook,
            figsize=self._figsize,
            dpi=self._dpi,
            font_family=self._font_family,
        ):
            return self._draw_dual_panel(*args, **kwargs)

    def _draw_dual_panel(
        self, results: List[ComparisonResult], show: bool,
        interpolation: str = 'log-log',
    ) -> plt.Figure:
        """Dual-panel figure: main overlay + difference panel."""
        # Resolve notebook / interactive settings
        notebook = (
            self._notebook_mode if self._notebook_mode is not None
            else _is_notebook()
        )
        interactive = self._interactive
        if interactive is None and notebook:
            interactive = _detect_interactive_backend()

        # The style's rc context is opened by _build_dual_panel around this call
        fig, (ax_main, ax_diff) = plt.subplots(
            nrows=2, ncols=1,
            figsize=self._figsize,
            dpi=self._dpi,
            gridspec_kw={
                'height_ratios': list(self._height_ratios),
                'hspace': 0.05,
            },
            sharex=True,
        )

        if notebook and interactive:
            _configure_figure_interactivity(fig, interactive)

        # --- Main panel via PlotBuilder on existing axes ---
        main_builder = PlotBuilder(
            style=self._style, ax=ax_main,
            font_family=self._font_family,
            notebook_mode=notebook,
        )
        # The builder draws the averaged layers of every averaged series, with
        # its pointwise trace masked or faded (kika.plotting.averages).
        main_builder.add_data(self._reference, **self._reference_styling)
        for cmp_data, styling in self._comparisons:
            main_builder.add_data(cmp_data, **styling)
        main_builder.set_group_average(self._main_display, self._shade_average_range)
        for ovl_data, ovl_styling in self._overlays:
            main_builder.add_data(ovl_data, **ovl_styling)
        for ovl_data, ovl_styling in self._scatter_overlays:
            main_builder.add_data(ovl_data, **ovl_styling)
        main_builder.set_labels(title=self._title, y_label=self._y_label)
        main_builder.set_scales(log_x=self._use_log_x, log_y=self._use_log_y)
        main_builder.set_limits(x_lim=self._x_lim, y_lim=self._y_lim)
        main_builder.set_legend(loc=self._legend_loc, ncol=self._legend_ncol)
        main_builder.set_grid(
            grid=self._dual_panel_grid(),
            alpha=self._grid_alpha,
            show_minor=self._show_minor_grid,
            minor_alpha=self._minor_grid_alpha,
            show_minor_x=self._show_minor_grid_x,
            show_minor_y=self._show_minor_grid_y,
        )
        main_builder.build()

        # Hide x-axis labels on main panel (shared with diff panel)
        ax_main.set_xlabel('')
        ax_main.tick_params(axis='x', labelbottom=False)

        # --- Difference panel ---
        diff_builder = PlotBuilder(
            style=self._style, ax=ax_diff,
            font_family=self._font_family,
            notebook_mode=notebook,
        )

        colors = _get_color_palette(self._style)
        # Track overlay diffs to draw after the builder renders the
        # pointwise traces — this way the step trace sits on top of
        # the masked pointwise and shares the same axis limits.
        overlay_diff_draws: List[Tuple[List[Tuple[str, np.ndarray, np.ndarray]], str, float]] = []
        for i, (result, (cmp_data, _)) in enumerate(
            zip(results, self._comparisons)
        ):
            # Use the comparison series' own color if specified; fall back to palette
            color_idx = (i + 1) % len(colors)
            diff_color = cmp_data.color if cmp_data.color else colors[color_idx]
            diff_styling = {'color': diff_color, 'linewidth': cmp_data.linewidth or 1.5}
            diff_data = result.difference
            # In dual-panel mode: labels are suppressed by default (colors match main panel)
            # but a per-series diff_label in metadata overrides this
            raw_diff_label = cmp_data.metadata.get('diff_label')
            if raw_diff_label is None:
                diff_data.label = None  # Default: suppress in dual-panel
            else:
                diff_data.label = raw_diff_label or None

            # If both reference and this comparison carry averages, diff
            # them, fade or mask the pointwise diff, and queue the averaged
            # diff to draw on top.
            layers = self._apply_overlay_diff(diff_data, cmp_data, interpolation)
            if layers is not None:
                overlay_diff_draws.append((layers, diff_color, cmp_data.linewidth or 1.5))

            diff_builder.add_data(diff_data, **diff_styling)

        # Add scatter overlays to diff panel (e.g., EXFOR experimental data)
        for ovl_data, ovl_styling in self._scatter_overlays:
            try:
                ovl_result = compute_difference(
                    reference=self._reference,
                    comparison=ovl_data,
                    mode=self._diff_mode,
                    interpolation=interpolation,
                    grid='comparison',  # Keep scatter x-points
                    relative_in_percent=self._relative_in_percent,
                )
                ovl_diff = ovl_result.difference
                ovl_diff.label = None  # Suppress legend in diff panel
                # Preserve scatter marker style
                ovl_diff_styling = {
                    'color': ovl_data.color or ovl_styling.get('color'),
                    'marker': ovl_data.marker or ovl_styling.get('marker', 'o'),
                    'markersize': ovl_data.markersize or ovl_styling.get('markersize', 5),
                    'linestyle': 'none',
                }
                diff_builder.add_data(ovl_diff, **ovl_diff_styling)
            except Exception:
                pass  # Skip if interpolation fails for this overlay

        diff_builder.set_labels(
            y_label=self._resolve_diff_y_label(), x_label=self._x_label,
        )
        diff_builder.set_scales(log_x=self._use_log_x, log_y=self._diff_log_y)
        diff_builder.set_limits(x_lim=self._x_lim, y_lim=self._diff_y_lim)
        diff_builder.set_grid(
            grid=self._dual_panel_grid(),
            alpha=self._grid_alpha,
            show_minor=self._show_minor_grid,
            minor_alpha=self._minor_grid_alpha,
            show_minor_x=self._show_minor_grid_x,
            show_minor_y=self._show_minor_grid_y,
        )
        diff_builder.build()

        # Group-average bin-diff traces (step-post) rendered after the
        # pointwise diff so they sit on top of the NaN-masked gaps.
        for layers, color, lw in overlay_diff_draws:
            self._draw_overlay_diff(ax_diff, layers, color, lw)
        if overlay_diff_draws:
            _averages.fit_y_to_layers(ax_diff, self._diff_y_lim)
        # The main panel's builder shaded its own.
        self._shade_range(ax_diff)

        # Remove legend from diff panel — colors match the main panel
        _diff_legend = ax_diff.get_legend()
        if _diff_legend:
            _diff_legend.remove()

        # Zero reference line
        if self._zero_line:
            ax_diff.axhline(
                y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7,
            )

        # Reference annotation — colors match the main panel legend
        self._draw_reference_label(ax_diff)

        if show:
            plt.show()

        return fig
