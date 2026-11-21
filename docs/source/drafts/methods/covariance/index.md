# Covariances

The covariances of an evaluation are what sampling and uncertainty propagation start from.
This chapter describes what kika does with them before either: checks a covariance section
as the file writes it, prepares a set of blocks for a draw (the pre-flight and the repairs
that make a matrix positive semi-definite), and collapses a covariance onto a group
structure.

Reading and writing covariances is in Part II (the ENDF and GNDS chapters, and
{doc}`../formats/covariance-files` for the multigroup formats). Drawing from them is the
next chapter. The pages are in the order a covariance goes through them.

- {doc}`covariance-checks`: what is checked in a covariance section as written.
- {doc}`conditioning-and-psd`: the pre-flight and the repairs before a draw.
- {doc}`multigroup-collapse`: collapsing MF 34 and relative covariances onto group
  structures.

```{toctree}
:hidden:

covariance-checks
conditioning-and-psd
multigroup-collapse
```
