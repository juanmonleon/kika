Sampling and perturbation
=========================

Drawing realisations from a covariance and applying them to an evaluation. The
methods behind these functions are in the methods pages; the entries here are
their contracts.

Perturbing from the model
-------------------------

.. py:module:: kika.sampling.model_perturbation

.. autofunction:: kika.sampling.model_perturbation.perturbFromModel

.. autoclass:: kika.sampling.model_perturbation.RunResult
   :members:

.. autoclass:: kika.sampling.model_perturbation.AceOptions
   :members:

.. autoclass:: kika.sampling.model_perturbation.HDF5Options
   :members:

Applying factors to model nodes
-------------------------------

.. py:module:: kika.nuclear_data.model.perturbation

.. autofunction:: kika.nuclear_data.model.perturbation.applyFactors

.. autofunction:: kika.nuclear_data.model.perturbation.applyLegendreFactors

.. autofunction:: kika.nuclear_data.model.perturbation.applyTabulatedFactors

.. py:module:: kika.nuclear_data.model.angular_tables

.. autofunction:: kika.nuclear_data.model.angular_tables.legendreMoments

.. autofunction:: kika.nuclear_data.model.angular_tables.perturbTabulatedAngular

.. autofunction:: kika.nuclear_data.model.angular_tables.tableIntegral

Positivity of angular distributions
-----------------------------------

.. py:module:: kika.sampling.mf4_positivity

.. autofunction:: kika.sampling.mf4_positivity.repair_tabulated_positivity

File-level ENDF sampler
-----------------------

.. py:module:: kika.sampling.endf_perturbation

The sampler that reads and writes ENDF files directly, without the model.

.. autofunction:: kika.sampling.endf_perturbation.perturb_ENDF_files

Legacy file-level samplers
--------------------------

The samplers that work on PENDF, ACE and ENDF files by quantity, without the model.
The user guide's page on the legacy samplers says when each one is used.

.. autofunction:: kika.sampling.pendf_perturbation.perturb_PENDF_files

.. autofunction:: kika.sampling.nubar_perturbation.perturb_nubar_files

.. autofunction:: kika.sampling.pfns_perturbation.perturb_pfns_files

.. autofunction:: kika.sampling.combined_perturbation.perturb_ENDF_PENDF_combined

.. autofunction:: kika.sampling.joint_perturbation.perturb_joint_mf33_mf34

.. autofunction:: kika.sampling.ace_perturbation.perturb_ACE_files

.. autofunction:: kika.sampling.reprocess_endf_to_ace.reprocess_endf_to_ace
