API Reference
=============

This section provides detailed documentation of the KIKA API.

Package Overview
----------------

KIKA is structured into several subpackages:

* **kika.mcnp**: For MCNP input files (materials, PERT cards) and tally output: MCTAL files and FMESH mesh tallies (MESHTAL, HDF5 runtape)
* **kika.sensitivities**: For sensitivity analysis and SDF file generation
* **kika.ace**: For parsing ACE format nuclear data files
* **kika.cov**: For covariance matrix handling and visualization
* **kika.UQ**: For strict sensitivity/covariance alignment, sandwich propagation, and c-k
* **kika.benchmarks**: For DICE/ICSBEP SQLite profiles and benchmark ranking
* **kika.endf**: For reading Evaluated Nuclear Data Files (ENDF)
* **kika.energy_grids**: Predefined energy group structures
* **kika.sampling** and **kika.nuclear_data.model**: Drawing realisations from a covariance and applying them to an evaluation

Module Reference
----------------

.. toctree::
   :maxdepth: 2
   
   input
   mctal
   sensitivities
   ace
   cov
   uq
   benchmarks
   model
   endf
   sampling
