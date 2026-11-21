# Formats

This part describes, one chapter per format, how kika reads a nuclear-data file into its
canonical model and writes the model back. Each chapter says which parts of the format
are read, which model object each lands in, what the reading assumes, what it refuses,
and what is lost on the way back out.

The model itself, and the rules every adapter follows (no silent conversion, refuse
instead of guessing, the conversion report), are in Part I: {doc}`../../architecture/index`.
What kika computes once the data is in the model (reconstruction, broadening, group
averages, perturbation) is in Part III. Each chapter stands on its own and can be read
alone, after the notation appendix.

- {doc}`endf`: ENDF-6 tapes, section by section.
- {doc}`endf-special-data`: thermal scattering laws, photon production, decay and
  fission-yield sublibraries, photo-atomic data.
- {doc}`gnds`: GNDS 2.0 and 2.1.
- {doc}`ace`: ACE tables of continuous-energy Monte Carlo codes.
- {doc}`g4ndl`: the neutron data library of Geant4.
- {doc}`hdf5`: OpenMC's HDF5 nuclear data.
- {doc}`covariance-files`: COVERX, BOXER and COVFIL multigroup covariances.

```{toctree}
:hidden:

endf
endf-special-data
gnds
ace
g4ndl
hdf5
covariance-files
```
