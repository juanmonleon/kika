# What kika does: a map of the manual

Every capability of kika, grouped by what it is for, with the page that documents it.
How kika is built is in {doc}`../architecture/index`; how to use it, in the user guide.

## Reading and writing data

The formats kika reads into its model and writes back, with what each reading assumes,
refuses and loses:

- {doc}`formats/endf`: ENDF-6 tapes, section by section.
- {doc}`formats/endf-special-data`: thermal scattering laws, photon production, decay and
  fission-yield sublibraries, photo-atomic data.
- {doc}`formats/gnds`: GNDS 2.0 and 2.1.
- {doc}`formats/ace`: ACE tables of Monte Carlo codes.
- {doc}`formats/g4ndl`: Geant4's neutron data library.
- {doc}`formats/covariance-files`: COVERX, BOXER and COVFIL multigroup covariances.

## Mathematics of tabulated data

- {doc}`algebra/tabulated-functions`: the interpolation laws, exact integrals and group
  averages, linearisation, sums and the Gaussian fold every other method rests on.

## Processing

- {doc}`processing/resonance-reconstruction`: cross-sections from resolved resonance
  parameters (SLBW, MLBW, Reich-Moore, R-Matrix Limited).
- {doc}`processing/unresolved-resonances`: infinitely dilute averages in the unresolved
  range.
- {doc}`processing/doppler-broadening`: cross-sections at a higher temperature, by
  free-gas convolution of a colder table.
- {doc}`processing/group-averaging`: flux-weighted group and window averages.
- {doc}`processing/differential-cross-sections`: $d\sigma/d\Omega$, frame transforms and
  time-of-flight resolution.
- {doc}`processing/njoy`: running NJOY with a fixed, checked chain.

## Covariances

- {doc}`covariance/covariance-checks`: what is checked in a covariance section as written.
- {doc}`covariance/conditioning-and-psd`: the pre-flight and the repairs before a draw.
- {doc}`covariance/multigroup-collapse`: collapsing MF 34 and relative covariances onto
  group structures.

## Sampling and perturbation

- {doc}`sampling/perturbation-pipeline`: the whole scheme, from a covariance to written
  realisations.
- {doc}`perturbation/cross-section-perturbation`: factors on cross-sections, sums, the
  resonance region.
- {doc}`perturbation/legendre-angular-perturbation`: factors on Legendre coefficients,
  frames, the magnitude order.
- {doc}`perturbation/tabulated-angular-perturbation`: factors on tabulated angular
  distributions.
- {doc}`perturbation/multiplicity-perturbation`: $\bar\nu$ with its sum rule.
- {doc}`perturbation/fission-spectrum-perturbation`: the prompt fission neutron spectrum
  from MF 35.

## Sensitivity and uncertainty quantification

- {doc}`uq/sensitivities`: first-order sensitivities from MCNP PERT and Serpent, SDF files.
- {doc}`uq/sandwich`: propagation by the sandwich rule.
- {doc}`uq/similarity-and-benchmarks`: $c_k$ similarity and benchmark ranking.
- {doc}`uq/total-monte-carlo`: Total Monte Carlo and the separation of nuclear-data and
  statistical variance.

## Not documented yet

- **The EXFOR-to-ENDF evaluation pipeline** (`scripts/` of the repository: energy grid,
  Legendre fits with model averaging, Monte Carlo covariances, adaptive multigroup collapse,
  $\chi^2$ against EXFOR). It is not part of the library package yet; its page follows when
  it is.
- **The shared notation page.**
