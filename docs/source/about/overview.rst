Overview
========

KIKA is a Python library for nuclear data: it reads the formats evaluated and processed
data come in, processes an evaluation for transport, checks and samples its covariances,
and reads the inputs and outputs of the transport codes that use the data. Arrays are
NumPy, tables pandas or xarray, plots matplotlib.

Contributions are welcome on GitHub at
`https://github.com/juanmonleon/KIKA.git <https://github.com/juanmonleon/KIKA.git>`_.

Current Features
----------------

The Model and the Formats
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``kika.read`` and ``kika.write`` for ENDF-6, GNDS, ACE, G4NDL and OpenMC HDF5, through
  one in-memory model shaped and named after GNDS
- A conversion report with each read and write: what was lost, approximated or refused
- Thermal-scattering libraries indexed and read in any of these formats

Processing an Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

- Resonance reconstruction (Breit-Wigner, Reich-Moore, R-matrix limited, unresolved
  range) and Doppler broadening, without NJOY
- Unresolved-resonance probability tables, heating (KERMA) and damage
- PENDF, OpenMC HDF5 or GNDS output, each read back and checked before it is kept

Covariances and Sampling
~~~~~~~~~~~~~~~~~~~~~~~~

- Checks of the covariance files of a tape or a library (MF 31-35, 40)
- Conditioning before a draw, and perturbed evaluations drawn from the covariances
  (cross-sections, angular distributions, multiplicities, fission spectra), written as
  ENDF, ACE, GNDS or OpenMC HDF5

MCNP Input Processing
~~~~~~~~~~~~~~~~~~~~~

- Parsing MCNP input files for materials and PERT cards
- Conversion between atomic and weight fractions for materials
- Generation and printing of materials in MCNP input format
- PERT card generation for sensitivity analysis

MCTAL Analysis
~~~~~~~~~~~~~~

- Parsing of MCNP tally output (MCTAL) files, every tally axis kept
- Mesh tallies (FMESH) from MESHTAL text files and MCNP 6.3 HDF5 runtapes
- Tally results and their uncertainties as labelled xarray or pandas tables
- Access to Tally Fluctuation Chart (TFC) data
- Conversion of tally results to pandas DataFrames
- Visualization of tally convergence

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

- Preparation of MCNP input files for sensitivity calculations using PERT cards
- Processing and analysis of MCNP perturbation calculations
- Generation and visualization of sensitivity profiles
- Creation of Sensitivity Data Files (SDF) compatible with SCALE
- Analysis of linearity and visualization of sensitivity coefficients

ACE File Processing
~~~~~~~~~~~~~~~~~~~

- Parsing of ACE (A Compact ENDF) format nuclear data files
- Reading and processing xsdir files
- Cross-section data extraction and analysis

Covariance Matrix Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Reading covariance matrices from SCALE and NJOY outputs
- Covariance matrix decomposition and analysis
- Heatmap visualization for correlation and covariance data
- Support for MF34 angular distribution covariances

ENDF Data Processing
~~~~~~~~~~~~~~~~~~~~

- Reading and parsing Evaluated Nuclear Data Files (ENDF)
- Support for various ENDF file sections (MF/MT combinations)
- Data extraction and conversion utilities

Energy Group Structures
~~~~~~~~~~~~~~~~~~~~~~~

- Predefined energy group structures commonly used in nuclear analysis
- Support for SCALE, VITAMIN, and other standard group structures

Additional Features
~~~~~~~~~~~~~~~~~~~

- **Serpent**: Support for Serpent Monte Carlo code outputs
- **NJOY**: Integration with NJOY nuclear data processing results
- **Sampling**: Tools for uncertainty quantification through sampling
- **Plotting**: Consistent visualization utilities for nuclear data

