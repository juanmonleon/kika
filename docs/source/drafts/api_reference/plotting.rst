kika.plotting
=============

Plots of nuclear data from any source kika reads. A source (an ENDF or ACE file, an
EXFOR dataset, the model, a path) becomes an item of one physical quantity in canonical
units, and a builder draws the items in a style and in the units the reader asks for.

.. currentmodule:: kika.plotting

From a source to something that can be drawn
--------------------------------------------

.. autofunction:: plottable

.. autofunction:: supported_quantities

.. autofunction:: register_adapter

.. autoclass:: PlotItem

.. autoclass:: Provenance

.. py:data:: QUANTITIES

   The quantities a plot can show, by name: a dictionary of :class:`Quantity`.

.. autoexception:: NotPlottable

.. autoexception:: ReconstructionRequired

Quantities and units
--------------------

.. autoclass:: Quantity

.. autofunction:: get_quantity

.. autoexception:: UnitError

.. autoclass:: MixedQuantityWarning

Drawing
-------

.. autoclass:: PlotBuilder

.. autofunction:: draw_errorbars

.. autoclass:: HeatmapBuilder

.. autofunction:: heatmap_grid

.. autofunction:: grid_columns

.. autofunction:: shared_norm

Comparing curves
----------------

.. autoclass:: ComparisonBuilder

.. autoclass:: ComparisonResult

.. autofunction:: compute_difference

``kika.plotting.interpolate_to_grid`` is :func:`kika.algebra.interpolate_to_grid`.

Styles
------

.. autoclass:: Style

.. autofunction:: get_style

.. autofunction:: list_styles

.. autofunction:: style_names

.. autofunction:: style_parts

.. autofunction:: compose_style

.. autofunction:: register_style

The data a plot holds
---------------------

.. autoclass:: PlotData

.. autoclass:: CrossSectionPlotData

.. autoclass:: MultigroupCrossSectionPlotData

``MultigroupXSPlotData`` is an alias of :class:`MultigroupCrossSectionPlotData`.

.. autoclass:: MultigroupUncertaintyPlotData

.. autoclass:: UncertaintyBand

.. autoclass:: DifferencePlotData

.. autoclass:: AngularDistributionPlotData

.. autoclass:: LegendreCoeffPlotData

.. autoclass:: LegendreUncertaintyPlotData

.. autoclass:: HeatmapPlotData

.. autoclass:: CovarianceHeatmapData

.. autoclass:: LegendreHeatmapData

``MF34HeatmapData`` is an alias of :class:`LegendreHeatmapData`.
