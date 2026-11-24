# The data checks, one by one

This page lists every check {py:func}`kika.check` runs on the data of an ENDF-6
(Evaluated Nuclear Data File) tape, MF 1 to MF 28: what it looks at, what a finding means
for whoever uses the file, and when it gives each level. The covariance files have their
own page, {doc}`/user_guide/covariance-checks`; the levels are the same, and so is how to
run the checks and read their reports ({doc}`/user_guide/checking-evaluations`).

The texts below are the ones the HTML reports print in their legend and the desktop app
shows beside a finding: the page is built from the same table in the code
(`kika.endf.checks.data.DATA_CHECKS`), with the thresholds the checks use.

The checks come in eight families, one per group of files. `mf=` runs every family that
reads one of the files it names (`kika.check(path, mf=[4])` runs the distributions); a
family reads the other files it needs on its own, without checking them. No family looks
at the contents of MF 9, 10 or 26.

Each entry opens with the check's name, as it appears in a finding's `check` field, and
the files it runs on. "Always" means that the check gives that level whenever it finds
something.

## General (MF 1)

The MT 451 header and directory against the tape, the ZA and AWR every section repeats,
AWR against the AME2020 masses, ν̄ and the delayed neutrons, and the fission energy
release of MT 458.

Two kinds of check that are not MF 1's alone are listed here too. The reading checks (a
file or a section the parser could not read) apply to every file asked for. The table
checks (`table_*`) apply to every tabulated section, and the family that reads the table
runs them.

```{kika-data-checks} general
```

## Resonances (MF 2)

The resonance parameters: energy ranges and isotope abundances, the spins a channel can
reach, widths and the scattering radius, and the unresolved parameters with their degrees
of freedom.

```{kika-data-checks} resonances
```

## Cross-sections (MF 3)

Sign, thresholds, Q values against AME2020, the order and Q of level reactions, coverage
up to EMAX, and every total against the sum of its parts. On a tape with resonance
parameters (`LRP = 1`), MF 3 inside the resolved range is a background, and the checks
treat it as one.

The checks on MF 2 and 3 together are the deep checks: they run only with `deep=True`, on
kika's 0 K reconstruction of the tape, and say whether the cross-sections a code will use
are positive and add up.

```{kika-data-checks} xs
```

## Distributions (MF 4 to 6)

Normalisation, positive Legendre expansions, cosines in [−1, 1], outgoing energies within
what the reaction makes available, Kalbach fractions, multiplicities, charge and mass
balance, and reactions with no distribution or a distribution with no reaction. Some of
these checks also run on the photon distributions of MF 14 and 15, when the photon family
reads them.

```{kika-data-checks} distributions
```

## Thermal scattering (MF 7)

The shape and sign of S(α, β), increasing temperatures, the Bragg edges of coherent
elastic scattering, and the effective temperatures.

```{kika-data-checks} thermal
```

## Decay and fission yields (MF 8)

Half-lives, branching ratios, decay Q against AME2020, average energies within Q and the
decay spectra, and the fission yields: their sums, charge, the ν̄ they imply, independent
against cumulative.

```{kika-data-checks} decay
```

## Photons (MF 12 to 15)

Level schemes against the levels of MF 3, photon energies within their level, transition
probabilities, yields against their sum, and MF 14 and 15 against the photons of MF 12
and 13.

```{kika-data-checks} photons
```

## Photo-atomic and atomic relaxation (MF 23 to 28)

The cross-sections of MF 23 and their totals, the form factors of MF 27 at their limits,
and the subshell occupancies, binding energies and transition probabilities of MF 28.

```{kika-data-checks} photoatomic
```
