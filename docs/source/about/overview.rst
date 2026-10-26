Overview
========

KIKA is a Python library for nuclear data: it reads the formats evaluated and processed
data come in, processes an evaluation for transport, checks and samples its covariances,
and reads the inputs and outputs of the transport codes that use the data. Arrays are
NumPy, tables pandas or xarray, plots matplotlib.

Contributions are welcome on GitHub at
`https://github.com/juanmonleon/kika <https://github.com/juanmonleon/kika>`_.

Current Features
----------------

The Model and the Formats
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``kika.read`` and ``kika.write`` for ENDF-6, GNDS, ACE, G4NDL and OpenMC HDF5, through
  one in-memory model shaped and named after GNDS
- A conversion report with each read and write: what was lost, approximated or refused
- ENDF tapes read section by section on demand, edited and written back
- Thermal-scattering libraries indexed and read in any of these formats

Processing an Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

- Resonance reconstruction (Breit-Wigner, Reich-Moore, R-matrix limited, unresolved
  range) and Doppler broadening, without NJOY
- Unresolved-resonance probability tables, heating (KERMA), damage and gas production
- PENDF, ACE, OpenMC HDF5 or GNDS output, each read back and checked before it is kept,
  with an HTML report of the run; ``kika.process`` and the ``kika process`` command
- ACE tables for thermal scattering, dosimetry, photo-atomic and photonuclear data

Checking an Evaluation
~~~~~~~~~~~~~~~~~~~~~~

- Physics and consistency checks of a tape or a whole library, with a report;
  ``kika.check`` and the ``kika check`` command
- Checks of the covariance files (MF 31-35, 40)

Covariances and Sampling
~~~~~~~~~~~~~~~~~~~~~~~~

- Covariance matrices from ENDF, SCALE and NJOY, their decomposition and plots
- Conditioning before a draw, and perturbed evaluations drawn from the covariances
  (cross-sections, angular distributions, multiplicities, fission spectra), written as
  ENDF, ACE, GNDS or OpenMC HDF5

Sensitivities and Uncertainty Propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- PERT cards for MCNP sensitivity calculations, and sensitivity coefficients from their
  results, with second-order checks of linearity
- Sensitivity Data Files (SDF) read and written in the SCALE format, from MCNP or Serpent
- The sandwich rule, the similarity index :math:`c_k`, and Total Monte Carlo statistics
- ICSBEP/DICE benchmarks ranked by similarity, and SINBAD experimental covariances

Transport Codes
~~~~~~~~~~~~~~~

- MCNP input files: materials and PERT cards read, materials created, converted and
  written back
- MCNP tally output (MCTAL), every tally axis kept, as labelled xarray or pandas tables,
  with the Tally Fluctuation Chart
- Mesh tallies (FMESH) from MESHTAL text files and MCNP 6.3 HDF5 runtapes
- Serpent materials and sensitivity output, and MCNP weight-window (WWINP) files

Experimental Data
~~~~~~~~~~~~~~~~~

- Angular distributions from EXFOR, with frame conversion and the energy resolution of
  each experiment

Energy Group Structures
~~~~~~~~~~~~~~~~~~~~~~~

- Predefined group structures (SCALE, VITAMIN-J, CASMO, WIMS, ECCO, XMAS and others)
