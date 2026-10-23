# Checking the data of an evaluation

```{admonition} At a glance
:class: note
**Computes:** a list of findings about everything an ENDF tape states except its
covariances (MF 1 to MF 28), each with a level (note, warning, defect), a location in the
file and its evidence, against the format, mathematics and physics, without changing
anything.

**Applies to:** ENDF-6 tapes of every sublibrary (incident neutrons, photonuclear, thermal
scattering, radioactive decay, fission yields, photo-atomic and atomic relaxation), read
with kika's ENDF parser; GNDS files through their ENDF-6 translation.

**Entry points:** {py:func}`kika.check`, {py:func}`kika.endf.checks.data.check_data`,
{py:func}`kika.endf.checks.check_evaluation`, {py:func}`kika.endf.checks.check_library`;
the reports are {py:class}`kika.endf.checks.CheckReport` and
{py:class}`kika.endf.checks.LibraryReport`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

An ENDF (Evaluated Nuclear Data File) tape is a sequence of sections (MF, MT), each a
sequence of records whose counts, interpolation laws and values the format pins down
(ENDF-6 §0.5). Most of what a processing code computes rests on a handful of quantities
read from them: the cross-sections $\sigma(E)$ of MF 3, the reaction Q values QM (the
ground-state mass difference) and QI (the Q of the level or continuum the MT names), the
mass ratio AWR of the target to the neutron, the distributions of MF 4 to MF 6, the photon
multiplicities of MF 12 and MF 13, and the resonance parameters of MF 2 (ENDF-6 Ch. 1-15).

A tape can be well formed and still state something the physics does not allow: a
cross-section non-zero below its threshold, a resonance spin no neutron channel reaches, a
spectrum that gives a product more energy than the reaction releases, a level photon with
more energy than its level. Codes read such data as written, and their results inherit it.

A reaction with QI < 0 can only occur above the laboratory threshold at which the final
state can be formed at rest in the centre of mass. With the masses of target, projectile
and final state as energies, $M_T$, $m$ and $M_T + m - Q$,

$$
E_\mathrm{th} = \frac{(M_T + m - Q)^2 - (M_T + m)^2}{2 M_T}
\approx -Q\,\frac{A + a}{A},
$$ (eq-data-checks-threshold)

with $A$ = AWR and $a$ the projectile's mass in neutron masses. The total kinetic energy of
the products of a reaction is $E + Q$ in the laboratory and $E A/(A + a) + Q$ in the
centre of mass, so no single product can carry more.

A Legendre series $f(\mu) = \sum_\ell \frac{2\ell+1}{2} a_\ell P_\ell(\mu)$ with $a_0 = 1$
integrates to 1 over $[-1, 1]$; it is a probability density only if it is nowhere negative,
and $|a_\ell| \le 1$ is necessary for that, because $|P_\ell(\mu)| \le 1$ on $[-1, 1]$.

## What it computes

Input: a tape, the families of checks to run (data, covariances or both) and optionally
the files to check. Output: a `CheckReport`, a tuple of findings in the vocabulary of the
covariance checks ({doc}`../covariance/covariance-checks`). Each finding is

- a **check**, the name of the rule (`xs_below_threshold`, `resonance_spin`, …);
- a **level**: `note` (true and worth knowing, not a fault), `warn` (can change a result
  and needs judgement) or `defect` (the data are not what ENDF-6 or physics allows, and
  using them gives a wrong result);
- a **location**: MAT, MF, MT and, where they apply, the subsection $k$ (the partial
  distribution of MF 5, the product of MF 6, the photon of MF 12), the product's ZAP, the
  resonance range and the temperature of MF 7;
- a one-line **summary** and the **evidence**, the numbers the rule measured.

A check that fires on many points of one table, or on many incident energies of one
distribution, gives **one** finding, with how many points or energies share it and the
worst of them. Nothing is changed, nothing is refused, and no finding blocks processing.

## Method

The checks come in families, one per group of files. A family runs when the files asked
for include any of its own, and reads the others it needs on its own (the cross-section
family reads MF 2 to know where MF 3 is a background). Every check is *static* (it reads
the tape and nothing else) except those of the deep family, which reconstruct the
cross-sections.

### Tables

Every TAB1 of every file is checked before its values are used (ENDF-6 §0.5.2): the
interpolation regions (NBT strictly increasing, the last equal to NP, every INT a defined
law, 1-5 and 6 in MF 3), the abscissae (non-decreasing; a value repeated twice is a step,
three times with different values has no reading), and the logarithmic laws (a
non-positive abscissa under INT = 3 or 5, or a negative value under INT = 4 or 5, has no
logarithm). Pairs read as $(0, 0)$ after a positive abscissa are **blank pairs**: NP counts
fields the record left empty. They are reported as such, wherever in the table they are,
and the table is read without them by the checks that follow.

### Header, $\bar\nu$ and fission energy (MF 1)

- Every section's HEAD record repeats the ZA and AWR of MF 1/451.
- The flags agree with the tape: LRP with the presence of MF 2, LFI with fission reactions,
  LIS with ELIS, AWI = 1 on a neutron sublibrary.
- The directory of MF 1/451 lists exactly the sections on the tape. Its line counts are
  compared with the tape, read from columns 67-75 of every line. A section whose lines
  appear in two places is split.
- AWR is compared with the AME2020 atomic mass of the target over the neutron mass
  (Wang et al. 2021).
- EMAX is reached by the cross-sections.
- $\bar\nu$ is positive and below 20 (delayed: below 1). $\bar\nu_\mathrm{t} = \bar\nu_\mathrm{d} + \bar\nu_\mathrm{p}$
  holds wherever MT 452, 455 and 456 are given. The $\bar\nu$ tables reach the top of the
  fission cross-section.
- A delayed $\bar\nu$ has its spectra (MF 5/455), with one spectrum per precursor family of
  MF 1/455. Decay constants are positive. Energy-dependent abundances add up to 1.
- The components of MT 458 are not negative, and they add up: ET is the sum of the first
  seven components and ER = ET − ENU (ENDF-6 §1.5). The LFC = 1 tables are not negative
  either.

### Resonance parameters (MF 2)

- Each isotope's ranges have EL < EH and follow each other with no gap or overlap, the
  unresolved range above the resolved one. The abundances of an element add up to 1.
- **Spins.** A neutron on a target of spin $I$ forms the channel spins
  $s \in \{|I - \tfrac12|, I + \tfrac12\}$, and with orbital momentum $\ell$ a resonance's
  $J$ is reachable when $|s - \ell| \le J \le s + \ell$ in integer steps, for some $s$
  (Blatt & Weisskopf 1952). Every resonance, every unresolved $(\ell, J)$ state and every
  R-matrix channel (whose own particle spins form its channel spin) is tested. An allowed
  $(\ell, J)$ with no resonance is reported, and so are ranges that state different target
  spins.
- **Widths.** SLBW and MLBW widths are non-negative, and $\Gamma = \Gamma_n + \Gamma_\gamma + \Gamma_f$
  unless a competitive width is declared (LRX = 1). Reich-Moore $\Gamma_n$ and $\Gamma_\gamma$
  are non-negative; the fission widths are signed amplitudes. Unresolved level spacings are
  positive, average widths non-negative, the degrees of freedom of the neutron width 1 or 2,
  and energy-dependent parameters span their range.
- **Scattering radius.** AP is compared with the systematics $0.123\,A^{1/3} + 0.08$
  (in $10^{-12}$ cm). A non-positive AP is a defect only where it is used: a Reich-Moore
  range that gives every $\ell$ its own APL does not use it.

### Cross-sections (MF 3)

- A negative value is a defect, except inside the resonance ranges of a tape whose MF 2 adds
  resonances (LRP = 1), for a reaction MF 2 contributes to (elastic, capture, fission,
  every R-matrix channel, and the sums they belong to). There MF 3 is a background.
- **Threshold.** For QI < 0, the first non-zero value is compared with the lower of the
  relativistic threshold {eq}`eq-data-checks-threshold` and its first-order form, so that an
  evaluator who rounds either way is not faulted. Production sections (MT 201-207), lumped,
  summed and fission reactions are left out. A threshold table that starts with a non-zero
  value starts with a step.
- **Q against masses.** QM is the ground-state Q of the products the MT names (ENDF-6
  Appendix B), computed from AME2020 atomic masses (atomic masses give the nuclear Q
  because the electrons balance). For an isomeric target, the reading closer to the file is
  taken: the masses alone, or the masses plus ELIS. The levels and continuum of one family
  share one finding. Reactions whose residual breaks up (LR > 0), redundant totals, lumped
  and fission reactions and elemental targets are left out.
- **Levels.** Within a family (MT 51-91, 600-649, …), the excitation of each level is
  $Q_\mathrm{ref} - \mathrm{QI}$, with $Q_\mathrm{ref}$ the QM common to the family's levels
  whose residual does not break up. It is non-negative, grows with the MT and has no gaps,
  and the continuum starts at or above the last level.
- **Sums.** A stated total (MT 1, 3, 4, 16, 18, 27, 101, 103-107) is compared with the sum of
  its partials, by the sum rules of ENDF-6 Appendix B (MT 1 = 2 + 3, MT 3 = the non-elastic reactions, MT 4 = 51-91, …), as kika's reconstruction applies them. The comparison is
  made **at the energies the total is stated at**. At a node where a partial steps, the side
  of the step closer to the stated value is taken. With LRP = 1, only the energies above the
  resonance ranges are compared.
- A reaction that ends below EMAX at a non-zero value drops to zero there.

### Distributions (MF 4, 5, 6)

- **Densities.** A tabulated angular distribution covers $\mu \in [-1, 1]$, is
  non-negative and integrates to 1. A Legendre one has $|a_\ell| \le 1$, and its
  **negative area**, $\int_{-1}^{1} \max(-f, 0)\,d\mu$, is evaluated on the Chebyshev-Lobatto
  cosines of degree $\max(128, 8L)$. That is the probability a sampler would have to clip.
  MF 5 weights $p_k(E)$ lie in $[0, 1]$ and add up to 1 at every energy of their common
  range. Tabulated spectra are non-negative and normalised. Analytic laws have positive
  parameters. MF 6 LAW = 1 lines and continuum add up to 1, Kalbach's $r$ lies in $[0, 1]$,
  and LAW = 2 and 7 are densities in $\mu$.
- **Kinematics.** At each incident energy, the highest outgoing energy with probability is
  compared with $E + Q$ in the laboratory or $E A/(A + a) + Q$ in the centre of mass, with
  $Q = \max(\mathrm{QM}, \mathrm{QI})$. The frame is LCT's; with LCT = 3 the heavy residuals
  are in the laboratory, and photons always are. A negative photon energy on an MF 6 line is
  a primary photon, $|E'| + E A/(A + 1)$. Lines with no probability are not counted.
  Where the spectrum is known, the fraction of its probability beyond the limit sets the
  level.
- **Products.** In MF 6, a constant neutron multiplicity matches the MT's (two for MT 16,
  one for MT 22, …), except for residuals that break up. Charge and mass number are
  conserved: $\sum_k y_k Z_k$ and $\sum_k y_k A_k$ over the massive products equal those of
  target plus projectile at every energy of their common range. A section that lists only
  the particles a code transports, without the residual, is not faulted.
- **Completeness.** Every reaction that emits neutrons has an angular distribution and,
  unless it is two-body, an energy distribution. Fission chances (MT 19-21, 38) use MT 18's.
  No distribution is given for a reaction MF 3 lacks, and no reaction's neutrons are in both
  MF 6 and MF 4/MF 5. A distribution does not stop before its cross-section does.

### Photons (MF 12-15)

- A discrete photon carries at most its level's excitation, the ES it states or the capture
  Q. With transition probabilities (LO = 2), the level is at its MT's excitation, every lower
  level is a level of MF 3, the ground state or (for the inelastic levels of an isomeric
  target) the target's own state, and the probabilities from one level add up to 1.
- A continuum photon (LF = 1) has its spectrum in MF 15, and an MF 15 section has a
  continuum photon to belong to. MF 14 gives as many photons as MF 12/13 when the photons
  are listed one by one (LO = 1) and are not all isotropic. MF 12 multiplicities have their
  MF 3 cross-section. A stated total yield is the sum of its photons, compared as MF 3 sums
  are. MF 14 Legendre series and MF 15 spectra are densities.
- **Capture energy.** At the lowest incident energy, the capture photons carry the
  excitation of the compound nucleus:

$$
\sum_k y_k\, E_{\gamma,k} = Q_{102} + E\,\frac{A}{A+1},
$$ (eq-data-checks-capture)

  with the continuum's mean energy from MF 15 at the nearest incident energy.

### Thermal scattering (MF 7)

Every temperature of a table holds as many values as its grid, temperatures and the
$\beta$ grid increase, and each $\alpha$ table is checked as a TAB1. That catches a
logarithmic law over a stored $\ln S$ (LLN = 1) that is not positive.
$S(\alpha, \beta)$ stored as itself is non-negative. The coherent structure factor is
cumulative over the Bragg edges and never decreases. The incoherent elastic bound
cross-section and Debye-Waller integral are positive, and the latter grows with
temperature. The effective temperature of the short-collision-time approximation is not
below the physical one.

### Decay data and fission yields (MF 8)

- The half-life is non-negative, and zero exactly when NST says the nuclide is stable.
  Branching ratios lie in $[0, 1]$ and add up to 1, within their stated uncertainties.
- **Decay Q against AME2020.** For the modes whose daughter RTYP fixes (β−, EC/β+, α, n, p),
  Q is the parent mass (plus ELIS) minus the daughter's and the emitted particle's. An
  isomeric transition releases ELIS. Modes to an excited daughter, chained modes and
  isomers whose MF 1 gives no excitation are left out.
- The average energies of the light, electromagnetic and heavy radiations add up to at most
  $\sum \mathrm{BR}\cdot Q$ (neutrinos take the rest). Lines and continua lie below the
  largest Q and have non-negative intensities.
- **Yields.** Yields are non-negative. At each incident energy, the independent yields
  (MT 454) add up to 2 and carry the charge of the fissioning nucleus, $\sum y_i Z_i = Z_c$.
  Their mass implies a prompt $\bar\nu = A_c - \sum y_i A_i$ between 0 and 10
  (Jaffke et al. 2018). No independent yield exceeds the cumulative one of the same product
  by more than three standard deviations.

### Photo-atomic data (MF 23, 27, 28)

- MT 501 is the sum of MT 502, 504, 516 and 522, MT 516 of 515 and 517, and MT 522 of the
  subshells 534-599, compared at the energies of the total. Cross-sections are
  non-negative.
- The coherent form factor tends to $Z$ at zero momentum transfer. The incoherent
  scattering function tends to $Z$ at large momentum transfer and to 0 at zero.
- The subshells of MF 28 hold $Z$ electrons, binding energies are positive, the transition
  probabilities of a vacancy add up to 1, and no transition releases more than the
  vacancy's binding energy.

### Reconstructed cross-sections (deep)

Asked for with `deep=True`: the cross-sections are reconstructed at 0 K as the processing
reconstructs them ({doc}`../processing/resonance-reconstruction`). A negative reconstructed
cross-section is a defect, and a stated total is compared with the sum of its reconstructed
parts. Capture is tested for $1/v$ from $10^{-5}$ to $0.1$ eV: $\sigma\sqrt{E}$ is
constant far from resonances.

## Conventions and choices

- **The file as written, not the model.** The checks read the parsed ENDF sections, with
  their counts, interpolation records and the fields GNDS has no node for. A defect of the
  file (blank pairs, a wrong count, a split section) is visible there and normalised away by
  any conversion. A GNDS file is checked as the ENDF-6 tape kika writes from it, and the
  report says so with a note (`gnds_translated`).
- **A level says what the finding means for a result, not how far a number is from
  another.** A large kinematic excess carrying a millionth of the probability changes
  nothing and is a note. A small one carrying a percent is a defect.
- **No false positives over the major libraries.** Every threshold was set by sweeping
  ENDF/B-VIII.1, JEFF-4.0 and JENDL-5 (and the ENDF/B-VIII.1 sublibraries) and reading every
  warning and defect each level gave. A rule that fired on a convention and not on a fault
  was changed. The conventions it led to are part of the method above: fission chances use
  MT 18's distributions, a redundant total has no Q of its own, a Reich-Moore range with
  every APL given does not use AP, MF 14 counts the whole cascade under LO = 2, MF 3 is a
  background inside the resonance ranges.
- **Relativistic threshold, the lower of two readings.** FIZCON's non-relativistic
  threshold with a 0.1 % tolerance is known to fault correct files (PREPRO removed its own
  threshold correction for that reason). kika compares with the lower of the relativistic
  and first-order thresholds.
- **Sums at the total's nodes.** Between its own nodes, a stated total interpolates through
  the thresholds of its partials. A union-grid comparison then reports one point per
  threshold that the file never states. Comparing at the stated nodes asks what the
  evaluator wrote.
- **Kinematic limits of 1 % and 5 % of the available energy.** At 100-200 MeV the gap
  between relativistic and non-relativistic kinematics, and between the laboratory and the
  CM for an incident photon, is about 1 %. Below that a spectrum cannot be faulted. A last
  interval that starts within the limit is the resolution of the outgoing grid (coarse
  bins near a threshold) and is a note.
- **Decay Q is a warning only for nuclides that live a minute or more.** Short-lived exotic
  nuclides have masses AME2020 extrapolates and Q values from older tables. Their
  differences are notes. A Q off by a factor of 100 or more is a unit error and a defect
  whatever the half-life.
- **What FUDGE, FIZCON and PSYCHE do and kika does not copy.** kika does not copy FUDGE's
  `WeightsDontSumToOne`, which fires only when the weight sum is never 1 (kika tests every
  energy). It does not call a constant multiplicity below 1 negative. It does not use
  PSYCHE's semi-empirical mass formula with a 5 % band (kika uses AME2020 with keV
  tolerances), nor FIZCON's "NL should be even". Checks no other code performs: MT 458
  LFC = 1 tables, IFY ≤ CFY and the charge and mass sums of fission yields, decay Q against a
  mass table, the form-factor limits, the energy of capture photons and a $1/v$ test.
  TODO(ref): the FIZCON, PSYCHE and FUDGE manuals and sources these statements rest on.

## Limits and failure modes

- Nothing is refused. A tape that cannot be read, or a file the parser drops, is a finding
  (`file_unreadable`, `section_unreadable`) and the walk of a library goes on.
- Not checked: detailed balance and the sum rules of $S(\alpha, \beta)$ (tables truncate
  $\beta$, and no threshold could be set without false positives), the energy balance of
  every reaction including recoils (the processing's heating computes it and reports a
  negative KERMA, {doc}`../processing/heating`), and Wick's limit.
- The kinematic and Q checks use the file's own Q. When QM is wrong, the limit is wrong too,
  and both checks report it.
- Only the first material of a tape is checked (a note says how many it holds).

## Validation

Each rule has a witness: a verbatim cut of a published evaluation (its MF 1/451 and the
sections named) that triggers it, in `kika/endf/tests/data/data_checks/`.

| Witness | Finding | Test |
|---|---|---|
| ENDF/B-VIII.1 Pu-243 | delayed $\bar\nu$ without MF 5/455; an unresolved $J = I$ at $\ell = 0$ | `test_data_checks.py::test_pu243_delayed_neutrons_without_spectra_and_an_unreachable_urr_spin` |
| ENDF/B-VIII.1 U-241 | $J = \tfrac12$ on $I = \tfrac12$ at $\ell = 0$ | `test_data_checks.py::test_u241_spin_one_half_at_l0` |
| JENDL-5 Pr-141 | unresolved parameters stop at 100 keV of a 160 keV range | `test_data_checks.py::test_pr141_unresolved_parameters_short_of_their_range` |
| ENDF/B-VIII.1 N-14 | (n,2n) neutrons beyond $E A/(A+1) + Q$ in the CM | `test_data_checks.py::test_n14_two_neutrons_beyond_what_n2n_makes_available` |
| ENDF/B-VIII.1 Co-58 | QM of MT 32 and 41 MeV off AME2020 | `test_data_checks.py::test_co58_q_values_copied_across_reactions` |
| ENDF/B-VIII.1 Cm-240 | a decay constant of 0 | `test_data_checks.py::test_cm240_zero_decay_constant` |
| ENDF/B-VIII.1 Th-231 | $\bar\nu_\mathrm{t} \ne \bar\nu_\mathrm{d} + \bar\nu_\mathrm{p}$ | `test_data_checks.py::test_th231_total_nubar_is_the_prompt_alone` |
| JEFF-4.0 Mo-100, Cs-134 | blank pairs at the end and in the middle of MF 3 | `test_data_checks.py::test_jeff40_blank_pairs_wherever_they_are` |
| ENDF/B-VIII.1 H-1 | nothing at MF 3 and MF 4 | `test_data_checks.py::test_a_clean_cut_has_nothing_at_mf3_and_mf4` |

The physics the rules rest on is tested on its own: the Q of Fe-56(n,2n) from AME2020 and
the relativistic threshold against its first-order form, the spin algebra, and the negative
area of $f = \tfrac12 + \tfrac32 \mu$ (exactly $1/3$)
(`test_data_checks.py::test_threshold_and_q_from_masses`, `::test_spin_algebra`,
`::test_legendre_negative_area`). `test_data_checks.py::test_every_data_check_is_described_and_ids_are_unique`
keeps every check id described, and the ids of the data and covariance checks apart.

## Diagnostics

The findings are the diagnostics. Each check id has one text, the same in the report's
legend, the desktop app and this site: what it tests, when it gives each level and why it
matters. The ids that should worry a user most are those that change what a code computes:

- `delayed_spectra_missing`: delayed neutrons with no energy;
- `distribution_missing`: neutrons with no energy distribution;
- `xs_below_threshold`, `kinematic_limit`, `photon_energy`, `capture_energy`: energy
  created or lost;
- `za_balance`: charge or mass not conserved;
- `table_blank_pairs`, `table_grid_order`: tables codes read differently;
- `resonance_spin`: wrong statistical weights.

A report of kind `data` gives its method section (how each level is drawn) and its legend
(what each finding means) after the findings.

## Implementation

| Step | Module |
|---|---|
| Families, the run over the asked files | `kika.endf.checks.data` |
| Tape access, raw section scan, findings collector | `kika.endf.checks.data.context` |
| TAB1 checks | `kika.endf.checks.data.tables` |
| MF 1 | `kika.endf.checks.data.general` |
| MF 2 | `kika.endf.checks.data.resonances` |
| MF 3, reactions and masses | `kika.endf.checks.data.xs`, `kika.endf.checks.data.reactions` |
| MF 4-6 and the Legendre density test | `kika.endf.checks.data.distributions`, `kika.endf.checks.data.legendre` |
| MF 12-15 | `kika.endf.checks.data.photons` |
| MF 7 | `kika.endf.checks.data.thermal` |
| MF 8 | `kika.endf.checks.data.decay` |
| MF 23, 27, 28 | `kika.endf.checks.data.photoatomic` |
| Reconstructed cross-sections | `kika.endf.checks.data.deep` |
| Texts and method section | `kika.endf.checks.data.descriptions`, `kika.endf.checks.data.method` |
| Data and covariances on one tape, libraries, workers | `kika.endf.checks.evaluation`, `kika.endf.checks.library` |
| Reports (shared with the covariance checks) | `kika.endf.checks.findings`, `kika.endf.checks.suites`, `kika.endf.checks.export` |

## References

- Trkov, A., Herman, M., Brown, D. A. (eds.), *ENDF-6 Formats Manual*, BNL-203218-2018-INRE,
  revision of 2023.
- Wang, M., Huang, W. J., Kondev, F. G., Audi, G., Naimi, S., "The AME 2020 atomic mass
  evaluation (II)", *Chinese Physics C* 45, 030003 (2021).
- Blatt, J. M., Weisskopf, V. F., *Theoretical Nuclear Physics*, Wiley (1952).
- Jaffke, P., et al., *Nuclear Science and Engineering* 190, 258 (2018), arXiv:1709.01183.
  TODO(ref): full author list and title.
