kika.sinbad
===========

.. automodule:: kika.sinbad
   :no-members:

.. currentmodule:: kika.sinbad

Opening a benchmark
-------------------

.. autofunction:: read

``open_benchmark`` is an alias of :func:`read`.

.. autoclass:: SinbadBenchmark

.. autoclass:: Calculations

Data objects
------------

.. autoclass:: DataObject

.. autoclass:: DataObjectCollection

What a data object holds
------------------------

.. autoclass:: Content

.. autoclass:: Table

.. autoclass:: Column

.. autoclass:: Grid

.. autoclass:: GridContent

.. autoclass:: Gridded

.. autoclass:: Axes

.. autoclass:: Axis

.. autoclass:: XYs1d
   :no-index:

.. autoclass:: Polynomial1d

.. autoclass:: Materials

.. autoclass:: Material
   :no-index:

.. autoclass:: Geometry

.. autoclass:: FactorChain

.. autoclass:: Double

.. autoclass:: UnknownContent

The entry around the data
-------------------------

.. autoclass:: Identification

.. autoclass:: Documentation

.. autoclass:: Status

.. autoclass:: Issue

.. autoclass:: Absence

.. autoclass:: ExternalFile

.. autoclass:: Normalisation

.. autoclass:: ValueConvention

.. autoclass:: CoordinateFrame

.. autoclass:: Position

.. autoclass:: Detector

.. autoclass:: RadiationSource

.. autoclass:: Calculation

.. autoclass:: Comparison
   :no-index:

Corrections and uncertainties
-----------------------------

.. autoclass:: Correction

.. autoclass:: UncertaintyBudget

.. autoclass:: UncertaintyComponent

.. autofunction:: covariance

.. autofunction:: correlation

Plots
-----

.. autofunction:: plot_profile

.. autofunction:: plot_ce

.. autofunction:: plot_correlation

Vocabularies
------------

.. autofunction:: vocabularies

.. autodata:: QUANTITIES
   :no-value:

.. autodata:: NATURES
   :no-value:

.. autodata:: CORRELATION_SCOPES
   :no-value:

.. autodata:: ABSENCE_KINDS
   :no-value:

Errors
------

.. autoexception:: SinbadError

.. autoexception:: SinbadFormatError

.. autoexception:: LabelNotFoundError

.. autoexception:: AmbiguousLabelError

.. autoexception:: ContentTypeError

.. autoexception:: BenchmarkMismatchError

.. autoexception:: IncompleteBudgetError
