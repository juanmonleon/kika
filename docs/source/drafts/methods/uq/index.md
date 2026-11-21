# Sensitivity and uncertainty quantification

This chapter describes how kika turns nuclear-data covariances into an uncertainty on a
calculated response, by two routes. The first is deterministic: sensitivity coefficients
from a Monte Carlo transport code, combined with a covariance by the sandwich rule. The
second is stochastic: Total Monte Carlo, from transport calculations run on sampled
evaluations.

The sandwich rule takes the sensitivities of the first page, so the two are in that
order. The second
route uses the realisations of the previous chapter and can be read on its own.

- {doc}`sensitivities`: first-order sensitivities from MCNP PERT and Serpent, SDF files.
- {doc}`sandwich`: propagation by the sandwich rule.
- {doc}`total-monte-carlo`: Total Monte Carlo and the separation of nuclear-data and
  statistical variance.

```{toctree}
:hidden:

sensitivities
sandwich
total-monte-carlo
```
