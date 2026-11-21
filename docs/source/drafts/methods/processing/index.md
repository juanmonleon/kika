# Processing

An evaluation is not what a transport code reads: it states resonance parameters where a
Monte Carlo code needs pointwise cross-sections, at 0 K where the calculation needs the
temperature of the material, and it leaves heating, self-shielding in the unresolved range
and the sampling tables of secondary particles to the processing. This chapter describes
how kika does that processing, one page per stage, and how the stages are chained.

Read {doc}`processing-chain` first: it gives the order of the stages, what each passes to
the next, how the tolerance is shared between them and which files come out. The pages
after it follow the chain (resonances, broadening, the unresolved range, heating and gas
production, the tables a Monte Carlo code samples), then the tables written for other
sublibraries, the route through NJOY itself, and two calculations used outside the chain
(group averages and differential cross-sections). The formats the results are written in
are in Part II.

- {doc}`processing-chain`: the stages of processing an evaluation, their order and the
  tolerances they share.
- {doc}`resonance-reconstruction`: cross-sections from resolved resonance parameters
  (SLBW, MLBW, Reich-Moore, R-Matrix Limited).
- {doc}`doppler-broadening`: cross-sections at a higher temperature, by free-gas
  convolution of a colder table.
- {doc}`unresolved-resonances`: infinitely dilute averages in the unresolved range.
- {doc}`probability-tables`: unresolved-resonance probability tables.
- {doc}`heating`: heating (KERMA) and damage cross-sections.
- {doc}`gas-production`: gas production (MT 203-207).
- {doc}`tabulation`: evaluated distributions as the tables Monte Carlo codes sample.
- {doc}`thermal-scattering-tables`: thermal inelastic scattering as cross-sections and
  outgoing tables.
- {doc}`dosimetry-tables`: dosimetry ACE tables.
- {doc}`photoatomic-tables`: photo-atomic ACE tables.
- {doc}`photonuclear-tables`: photonuclear ACE tables.
- {doc}`njoy`: running NJOY with a fixed, checked chain.
- {doc}`group-averaging`: flux-weighted group and window averages.
- {doc}`differential-cross-sections`: $d\sigma/d\Omega$, frame transforms and
  time-of-flight resolution.

```{toctree}
:hidden:

processing-chain
resonance-reconstruction
doppler-broadening
unresolved-resonances
probability-tables
heating
gas-production
tabulation
thermal-scattering-tables
dosimetry-tables
photoatomic-tables
photonuclear-tables
njoy
group-averaging
differential-cross-sections
```
