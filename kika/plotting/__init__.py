"""
Plotting infrastructure for KIKA.

This module provides a flexible, object-oriented approach to creating plots
by separating data representation from visual styling and plot composition.

- :func:`plottable` turns any source (ENDF, PENDF, ACE, EXFOR, the model, a path)
  into a :class:`PlotItem` of a physical quantity, in canonical units.
- :class:`PlotBuilder` draws them in a :class:`Style` (see :func:`list_styles`),
  converting to the display units set with ``set_units``.
"""

from .plot_data import (
    PlotData,
    LegendreCoeffPlotData,
    LegendreUncertaintyPlotData,
    AngularDistributionPlotData,
    CrossSectionPlotData,
    DifferencePlotData,
    MultigroupCrossSectionPlotData,
    MultigroupXSPlotData,
    MultigroupUncertaintyPlotData,
    UncertaintyBand,
    HeatmapPlotData,
    CovarianceHeatmapData,
    LegendreHeatmapData,
    MF34HeatmapData,
    PlotItem,
    Provenance,
)
from .units import (
    QUANTITIES,
    Quantity,
    get_quantity,
    UnitError,
    MixedQuantityWarning,
)
from .quantities import (
    plottable,
    register_adapter,
    supported_quantities,
    NotPlottable,
    ReconstructionRequired,
)
from .styles import (
    Style,
    get_style,
    list_styles,
    register_style,
    style_names,
)
from .plot_builder import PlotBuilder
from .heatmap_builder import HeatmapBuilder
from .comparison import (
    ComparisonBuilder,
    ComparisonResult,
    compute_difference,
    interpolate_to_grid,
)

__all__ = [
    'PlotData',
    'LegendreCoeffPlotData',
    'LegendreUncertaintyPlotData',
    'AngularDistributionPlotData',
    'CrossSectionPlotData',
    'DifferencePlotData',
    'MultigroupCrossSectionPlotData',
    'MultigroupXSPlotData',
    'MultigroupUncertaintyPlotData',
    'UncertaintyBand',
    'HeatmapPlotData',
    'CovarianceHeatmapData',
    'LegendreHeatmapData',
    'MF34HeatmapData',
    'PlotItem',
    'Provenance',
    'QUANTITIES',
    'Quantity',
    'get_quantity',
    'UnitError',
    'MixedQuantityWarning',
    'plottable',
    'register_adapter',
    'supported_quantities',
    'NotPlottable',
    'ReconstructionRequired',
    'Style',
    'get_style',
    'list_styles',
    'register_style',
    'style_names',
    'PlotBuilder',
    'HeatmapBuilder',
    'ComparisonBuilder',
    'ComparisonResult',
    'compute_difference',
    'interpolate_to_grid',
]
