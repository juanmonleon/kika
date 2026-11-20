kika.njoy
=========

Running NJOY with a deck kika writes, and reading NJOY decks. How the chain is
built and checked is in the methods chapter.

.. automodule:: kika.njoy.run_njoy
   :members: run_njoy, run_njoy_with_pendf

.. autoclass:: kika.njoy.InputDeck
   :members:

.. automodule:: kika.njoy.reader
   :members: read_deck, read_deck_file, ParsedDeck, ParsedModule

.. autoexception:: kika.njoy.locale_guard.NjoyDecimalLocaleError

.. currentmodule:: kika.njoy

Isotopes and module definitions
-------------------------------

.. autofunction:: load_isotopes

.. autofunction:: load_module_definitions

.. autofunction:: get_mat_number

PLOTR input
-----------

.. autoclass:: PlotrJob

.. autoclass:: PlotrPlot

.. autoclass:: PlotrCurve

.. autoclass:: EndfSource

VIEWR input
-----------

.. autoclass:: ViewrPlot

.. autoclass:: PageSetup

.. autoclass:: Plot

.. autoclass:: Curve

.. autoclass:: DataPoint2D

.. autoclass:: CurveStyle

.. autoclass:: AxisRange

.. autoclass:: View3D

.. autoclass:: Orientation

.. autoclass:: FontStyle

.. autoclass:: PlotType

.. autoclass:: GridStyle

.. autoclass:: LegendStyle

.. autoclass:: CurveColor

.. autoclass:: PageColor

.. autoclass:: SymbolType

.. autoclass:: DashType
