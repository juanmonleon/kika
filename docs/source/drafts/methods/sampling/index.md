# Sampling and perturbation

kika draws realisations of an evaluation from the covariances it states and writes each
one as a complete evaluation, ready to be processed and run. This chapter describes that
scheme: the draw of correlated factors, and how each kind of data takes its factors.

Read {doc}`perturbation-pipeline` first: it covers the whole scheme, from a covariance to
the written realisations, and which factor goes to which page. The pages after it are one
per kind of data perturbed, and each can be read alone. Preparing the covariances for the
draw is in the previous chapter; using the realisations (Total Monte Carlo) is in the
next.

- {doc}`perturbation-pipeline`: the whole scheme, from a covariance to written
  realisations.
- {doc}`../perturbation/cross-section-perturbation`: factors on cross-sections, sums,
  the resonance region.
- {doc}`../perturbation/legendre-angular-perturbation`: factors on Legendre
  coefficients, frames, the magnitude order.
- {doc}`../perturbation/tabulated-angular-perturbation`: factors on tabulated angular
  distributions.
- {doc}`../perturbation/multiplicity-perturbation`: $\bar\nu$ with its sum rule.
- {doc}`../perturbation/fission-spectrum-perturbation`: the prompt fission neutron
  spectrum from MF 35.

```{toctree}
:hidden:

perturbation-pipeline
../perturbation/cross-section-perturbation
../perturbation/legendre-angular-perturbation
../perturbation/tabulated-angular-perturbation
../perturbation/multiplicity-perturbation
../perturbation/fission-spectrum-perturbation
```
