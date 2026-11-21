# Sensitivity and uncertainty quantification

This chapter describes how kika turns nuclear-data covariances into an uncertainty on a
calculated response, by two routes. The first is deterministic: sensitivity coefficients
from a Monte Carlo transport code, combined with a covariance by the sandwich rule, and
compared between an application and benchmarks by their similarity. The second is
stochastic: Total Monte Carlo, from transport calculations run on sampled evaluations.

The pages of the first route build on one another and are in that order. The second
route uses the realisations of the previous chapter and can be read on its own.

- {doc}`sensitivities`: first-order sensitivities from MCNP PERT and Serpent, SDF files.
- {doc}`sandwich`: propagation by the sandwich rule.
- {doc}`similarity-and-benchmarks`: $c_k$ similarity and benchmark ranking.
- {doc}`total-monte-carlo`: Total Monte Carlo and the separation of nuclear-data and
  statistical variance.

```{toctree}
:hidden:

sensitivities
sandwich
similarity-and-benchmarks
total-monte-carlo
```
