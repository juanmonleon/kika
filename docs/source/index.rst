KIKA Documentation
===================

**A Python library for nuclear data**

KIKA |version| is a Python library for nuclear data. It

- reads and writes ENDF-6, GNDS, ACE, G4NDL and OpenMC HDF5 through one in-memory
  model, shaped after GNDS;
- processes an evaluation for transport without NJOY: resonance reconstruction,
  Doppler broadening, unresolved-resonance probability tables, heating; PENDF,
  OpenMC HDF5 or GNDS out;
- checks the covariances an evaluation states, and draws perturbed evaluations from
  them for uncertainty quantification;
- reads MCNP and Serpent inputs and outputs, PERT cards, sensitivities and SDF files,
  for the sandwich rule and benchmark similarity (:math:`c_k`);
- plots what it reads.

Quick Links
-----------

* :doc:`about/overview`
* :doc:`user_guide/quickstart` 
* :doc:`user_guide/reading_files`
* :doc:`user_guide/tutorials/index`
* :doc:`api_reference/index`
* :doc:`license`

.. toctree::
   :maxdepth: 1
   :caption: About
   :hidden:

   about/overview
   license

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/quickstart
   user_guide/the-model
   user_guide/reading_files
   user_guide/working-with-endf
   user_guide/checking-evaluations
   user_guide/processing-an-evaluation
   user_guide/running-a-perturbation
   user_guide/legacy-samplers
   user_guide/tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api_reference/index