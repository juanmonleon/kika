# Checking evaluated data

Before an evaluation is processed or sampled, kika can check what it states against the
format, against mathematics (a probability that integrates to one, a sum that equals its
parts) and against physics. Each check returns findings; none of them changes the data.

This chapter covers every file of an ENDF tape except the covariances. The checks of the
covariance files have their own page, in the covariance chapter
({doc}`../covariance/covariance-checks`), because what to do about their findings is
decided there, by the pre-flight before a draw. Both pages report findings in the same
vocabulary, defined on the covariance page.

- {doc}`data-checks`: what is checked in MF 1-28, against the format, mathematics and
  physics.

```{toctree}
:hidden:

data-checks
```
