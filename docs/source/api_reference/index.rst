API Reference
=============

The documentation of every public function and class of kika, generated from the code
and grouped by what it is for.

The model and its algebra
-------------------------

The canonical model that every format is read into, the front door to read and write any format, and the mathematics of tabulated functions.

.. toctree::
   :maxdepth: 2

   model
   nuclear-data
   algebra

Evaluated-data formats
----------------------

Reading, checking and writing each format.

.. toctree::
   :maxdepth: 2

   endf
   endf-checks-and-dcs
   gnds
   ace
   hdf5
   g4ndl
   tsl

Processing
----------

Reconstruction, broadening and the files a transport code reads, by kika or through NJOY.

.. toctree::
   :maxdepth: 2

   processing
   processing-resonances
   processing-doppler
   njoy

Covariances, sampling and uncertainty
-------------------------------------

Covariance matrices, drawing realisations from them, sensitivities and the propagation of uncertainty.

.. toctree::
   :maxdepth: 2

   cov
   cov-assembly
   sampling
   sampling-core
   sensitivities
   uq
   uq-tmc

Transport codes
---------------

Inputs and outputs of MCNP and Serpent, material compositions and weight windows.

.. toctree::
   :maxdepth: 2

   input
   mctal
   serpent
   materials
   wwinp

Experiments and benchmarks
--------------------------

Measured data from EXFOR, and the criticality and shielding benchmark collections.

.. toctree::
   :maxdepth: 2

   exfor
   benchmarks
   sinbad

Energy grids and plotting
-------------------------

Group structures, folding on an energy grid, and the plotting helpers.

.. toctree::
   :maxdepth: 2

   energy-grids
   utils-energy-folding
   plotting
