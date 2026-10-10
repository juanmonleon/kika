Reading, writing and the model
==============================

The doors in and out of the canonical model, and its main containers. How the model is
organised, and why, is in the architecture section.

The doors
---------

.. autofunction:: kika.read

.. autofunction:: kika.write

.. autofunction:: kika.identify

The model
---------

.. py:module:: kika.nuclear_data.model.suite

.. autoclass:: kika.nuclear_data.model.suite.ReactionSuite
   :members: cross_section, summary

.. py:module:: kika.nuclear_data.model.covariances

.. autoclass:: kika.nuclear_data.model.covariances.CovarianceSuite
   :no-members:

.. py:module:: kika.nuclear_data.model.conversion

.. autoclass:: kika.nuclear_data.model.conversion.ConversionReport
   :members: isClean, isCleanFor, summary

.. py:module:: kika.nuclear_data.model.functions.xys1d

.. autoclass:: kika.nuclear_data.model.functions.xys1d.XYs1d
   :no-members:

.. py:module:: kika.nuclear_data.model.functions.regions1d

.. autoclass:: kika.nuclear_data.model.functions.regions1d.Regions1d
   :no-members:
