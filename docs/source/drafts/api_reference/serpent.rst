kika.serpent
============

Serpent input and output files: the input deck, and the sensitivity, detector, results,
history and depletion outputs, each read into its own object.

.. currentmodule:: kika.serpent

Input
-----

.. autofunction:: read_serpent_input

.. autoclass:: SerpentInput

.. autoclass:: SensitivityConfig

.. autoclass:: SensPerturbation

.. autoclass:: SensResponse

.. autoclass:: DetectorDefinition

.. autoclass:: EnergyGrid

.. autoclass:: TimeGrid

Sensitivity output
------------------

.. autofunction:: read_sensitivity_file

.. autofunction:: parse_sensitivity_text

.. autoclass:: SensitivityFile

.. autoclass:: SensitivitySet

.. autoclass:: Response

.. autoclass:: Perturbation

.. autoclass:: PertCategory

.. autoclass:: Material
   :no-index:

.. autoclass:: Nuclide
   :no-index:

Detector output
---------------

.. autofunction:: read_detector_file

.. autofunction:: parse_detector_text

.. autoclass:: DetectorFile

.. autoclass:: DetectorResult

Results output
--------------

.. autofunction:: read_results_file

.. autofunction:: parse_results_text

.. autoclass:: ResultsFile

History output
--------------

.. autofunction:: read_history_file

.. autofunction:: parse_history_text

.. autoclass:: HistoryFile

.. autoclass:: HistoryVariable

Depletion output
----------------

.. autofunction:: read_depletion_file

.. autofunction:: parse_depletion_text

.. autoclass:: DepletionFile

.. autoclass:: DepletionMaterial
