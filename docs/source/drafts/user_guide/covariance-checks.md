# The covariance checks, one by one

This page lists every check {py:func}`kika.check` runs on the covariance files of an
ENDF-6 tape (MF 31, 32, 33, 34, 35 and 40): what it looks at, what a finding means for
whoever uses the file, and when it gives each level. How to run the checks and read
their reports is in {doc}`/user_guide/checking-evaluations`.

The texts below are the ones the HTML reports print in their legend and the desktop app
shows beside a finding: the page is built from the same table in the code
(`kika.endf.checks.CHECKS`), with the thresholds the checks use.

## The three levels

Every finding has one level. In the code and on the command line they are written
`defect`, `warn` and `note`.

- **Defect**: the data are not what the format or physics allows, and using them gives a
  wrong result. A negative variance is one: no variance can be negative.
- **Warning**: it can change a result and needs judgement. An MF 34 block in the LAB
  frame for a reaction whose MF 4 is in the CM frame is one: ENDF-6 allows it, but kika's
  samplers convert only two-body scattering and refuse the others.
- **Note**: true and worth knowing, not a fault. Rows of a matrix that are all zero are
  one: those energies are given no uncertainty, which is legal and common at the edges of
  a grid.

A check gives one or more of the levels, and each entry below says when. "Always" means
that the check gives that level whenever it finds something.

Each entry opens with the check's name, as it appears in a finding's `check` field, and
the files it runs on.

## Structure

Whether the section can be read the way ENDF-6 says: counts that match the content,
flags the format defines, energy grids that increase, blocks written once and stored the
way their form allows.

```{kika-covariance-checks} structure
```

## Values

Whether each matrix is a covariance (correlations within ±1, no negative variance,
symmetric, positive semi-definite on its own and joined with the blocks that covary with
it), and whether its uncertainties are plausible against the central values they belong
to.

```{kika-covariance-checks} values
```

## Completeness

Whether what a covariance refers to is there: the self block of each reaction, the
central values in MF 1, MF 3 or MF 4, the partner of each cross block, the reactions
named by the NC derivations of MF 31 and MF 33, and a grid that covers the cross section.

```{kika-covariance-checks} completeness
```

## MF 32 against MF 2

Whether the covariance of the resonance parameters is about the parameters MF 2 uses:
the same formalism, the same ranges, the same resonances and widths.

```{kika-covariance-checks} mf32_vs_mf2
```

## MF 35 against MF 5

Whether the covariance of the outgoing spectra fits the spectra of MF 5: a sum rule
that keeps every sample normalised, and incident-energy bands that neither overlap nor
leave gaps.

```{kika-covariance-checks} mf35_vs_mf5
```
