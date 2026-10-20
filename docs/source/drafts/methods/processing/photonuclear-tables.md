# Writing photonuclear tables

```{admonition} At a glance
:class: note
**Computes:** the photonuclear ACE (A Compact ENDF) table of a nuclide: reaction cross sections, their sum, a heating number and, for each particle the reactions emit, its production, yields, angles and energies.
**Applies to:** ENDF-6 (Evaluated Nuclear Data File) photonuclear evaluations (sublibrary `g`, NSUB = 0): MF 3, MF 4, MF 5, MF 6, MF 1 MT 452/455/456/458, and the level de-excitation MF 12/MF 14 states on the residuals.
**Entry points:** {py:func}`kika.endf.photonuclear_ace`, {py:func}`kika.processing.photonuclear.photonuclearTable`, {py:class}`kika.ace.classes.photonuclear.PhotonuclearAce`.
**Status:** experimental.
```

## Background

A photonuclear table (class letter `u`, `26056.01u`) is what MCNP reads to transport
photons through nuclear reactions: the giant resonance, quasi-deuteron absorption,
photofission. The projectile is a photon of energy $E$ (eV) and momentum $E/c$, with no
rest mass, which changes the kinematics every neutron table assumes.

The layout is the one NJOY2016's ACER writes with `iopt = 5` (`acepn.f90`, `phnout`):

- `NXS(1)` is the length of `XSS`, `NXS(2)` the target's ZA, `NXS(3)` the number of grid
  energies NES, `NXS(4)` the reactions NTR, `NXS(5)` the particle kinds NTYPE,
  `NXS(6)` = 2, `NXS(7)` = 12 (the words of one particle entry), `NXS(9)` the isomeric
  state, `NXS(10)` and `NXS(11)` Z and A, `NXS(16)` = 1.
- `JXS(1)` points at the grid (MeV), followed by the total cross section (`JXS(2)`);
  `JXS(3)` at the nonelastic and `JXS(4)` at the elastic when the evaluation has one
  (`JXS(3)` = `JXS(2)` and `JXS(4)` = 0 otherwise); `JXS(5)` at the heating number
  (MeV per collision); `JXS(6)` to `JXS(9)` at MTR, the Q values (MeV), the locators and
  the cross sections, each `IE, NE` and NE values from grid point IE; `JXS(10)` at the
  particle entries.
- Each particle entry has 12 words: IPT (1 n, 2 γ, 9 p, 31 d, 32 t, 33 ³He, 34 α), the
  number of reactions, and the locations of its production cross section (PXS), its
  heating number (PHN), its reactions (MTRP), their frames (TYRP, +1 laboratory, −1 centre
  of mass), and the yields, angles and energy laws of each (SIGP, ANDP, DLWP, with their
  locator blocks). The angular records and the energy laws are those of neutron tables
  ({doc}`../formats/ace`).

Kalbach-Mann systematics for an incident photon (ENDF-6 §6.2.3.2) differ from a
neutron's in slope and in shape:

$$
a_\gamma(E, E_b) = a_n(E, E_b)\,\sqrt{\frac{E}{2 m_n}}\;\min\!\left(4, \max\!\left(1, \frac{9.3}{\sqrt{E_b}}\right)\right),
\qquad
f(\mu, E, E_b) = \frac{f_0}{2}\left[(1 - r) + r\,\frac{a}{\sinh a}\,e^{a\mu}\right],
$$ (eq-photonuclear-tables-kalbach)

with $E$, $E_b$ (the emitted particle's centre-of-mass energy) and the neutron mass
$m_n$ in MeV, and $a_n$ the neutron's slope with $E$ in place of the neutron's energy
(compound nucleus = target + neutron). $a_\gamma = 4\,a_n\sqrt{E/2m_n}$ at $E_b = 0$.

## What it computes

From the evaluation of a photon on a nucleus of rest energy $M_A$:

1. The reaction cross sections $\sigma_r(E)$, lin-lin, on the union grid of the
   reactions, each from its first evaluated energy; and their sum
   $\sigma_t = \sum_r \sigma_r$ over the reactions written.
2. For each particle kind $k$, the reactions that emit it with their yields $y_{r,k}(E)$,
   their frames, angular records and energy laws; the production cross section
   $\sum_r y_{r,k}\,\sigma_r$ and the heating number
   $\mathrm{PHN}_k = \sum_r y_{r,k}\,\sigma_r \langle E'_{r,k}\rangle / \sigma_t$, with
   $\langle E'\rangle$ the mean laboratory energy.
3. The heating number, the energy that stays where the collision happens:

   $$
   \mathrm{THN}(E) = \frac{1}{\sigma_t}\sum_r \sigma_r(E)\Big[\sum_{k \in \{p,d,t,{}^3\mathrm{He},\alpha\}} y_{r,k}\langle E'_{r,k}\rangle + \sum_{R} y_{r,R}\langle E_{r,R}\rangle + F_r(E)\Big],
   $$ (eq-photonuclear-tables-thn)

   the light charged particles, the residual nuclei $R$, and for fission $F_r$, the
   fragments' share. Neutrons and photons are transported and stay out of it.

## Method

1. **Reactions.** Every reaction of the evaluation is written, except the sums: MT 201-207
   (particle production), MT 1 and MT 3 (when anything else is stated), MT 4 beside levels
   50-91 and MT 103-107 beside their levels. A cross section in a law other than lin-lin
   is linearised to a relative $10^{-4}$, the original points kept. The grid is the union
   of the reactions' grids, a repeated energy kept as a step.
2. **Two-body kinematics.** For a particle $b$ of a two-body reaction (MF 6 LAW 2), and
   for a light recoil that comes with it (LAW 4), the centre-of-mass energy is exact and
   relativistic. With $s = M_A^2 + 2EM_A$, $M_B = M_A - m_b - Q$:

   $$
   p^{*2} = \frac{\big(s - (m_b + M_B)^2\big)\big(s - (m_b - M_B)^2\big)}{4s},\qquad
   T_b^* = \sqrt{p^{*2} + m_b^2} - m_b ,
   $$ (eq-photonuclear-tables-twobody)

   above the threshold $E_{th} = \big((m_b + M_B)^2 - M_A^2\big)/2M_A$. The rest energies are
   nuclear masses of the 2020 Atomic Mass Evaluation (atomic less $Z$ electron masses). $Q$
   is the reaction's, except in MT 5, which lumps channels and states $Q = 0$: there it is
   the channel's, $M_A - m_b - M_B^{(0)} - E_{lev}$, with the residual's ground-state mass
   and its level energy from the evaluation. $T_b^*(E)$ is written as a chain of ACE
   law-33 pieces $C(E - E_c)$, each the chord of $T_b^*$ between its ends, halved until
   $|T_b^* - \text{chord}| \le 10^{-4}\,T_b^*$ at 10 %, 25 %, 50 %, 75 % and 90 % of every
   piece; each piece applies with probability 1 on its own interval. A recoil's angle is
   its partner's with $\mu \to -\mu$.
3. **Kalbach-Mann.** The slope and the shape are {eq}`eq-photonuclear-tables-kalbach`
   (the evaluator's slope when the evaluation gives one). The shape is not ACE law 44's,
   so the distribution is written as law 61: the outgoing spectrum and $r$ as for a neutron
   ({doc}`tabulation`), and at each outgoing energy the cosine density of
   {eq}`eq-photonuclear-tables-kalbach` refined on $[-1, 1]$ to the tabulation's tolerance.
4. **Other distributions** (Legendre and tabulated MF 6, MF 4/MF 5) are tabulated as for a
   neutron ({doc}`tabulation`).
5. **De-excitation photons.** When a reaction states no photon of its own and leaves a
   residual level whose de-excitation the evaluation gives (MF 12/MF 14 with LO = 2,
   carried by the residual), the cascade's lines are written on that reaction: one entry
   whose yield is the sum of the lines' yields and whose energy law is a table of discrete
   lines, each with probability $y_i / \sum y_i$, isotropic in the laboratory.
6. **Fission neutrons.** MT 18's prompt neutrons (multiplicity $\bar\nu_p$) and its
   delayed families $i$ ($\bar\nu_i$, each with its spectrum) are one entry: the yield is
   $\bar\nu_p + \sum_i \bar\nu_i$ and the energy law a chain, the prompt law with
   probability $\bar\nu_p/\bar\nu$ and each family's spectrum with $\bar\nu_i/\bar\nu$. When
   the prompt law carries its angle (law 61), each family is written as law 61 with
   isotropic cosine tables.
7. **Laboratory energies.** A centre-of-mass mean energy goes to the laboratory as

   $$
   \langle E'_{lab}\rangle = \langle E'_{cm}\rangle + E_c + 2\sqrt{E_c}\,\langle\sqrt{E'_{cm}}\,\mu\rangle,
   \qquad E_c = \tfrac{1}{2}\,m_b\,\beta^2,\quad \beta = \frac{E}{E + M_A},
   $$ (eq-photonuclear-tables-boost)

   $\beta$ the velocity of the centre of mass the photon's momentum gives it (with
   $c = 1$). Between two incident energies a spectrum is read by unit-base interpolation,
   as ACE laws 4, 44 and 61 are sampled.
8. **Residuals.** A residual with a distribution contributes its mean laboratory energy;
   a two-body recoil its exact energy, $T_B^*$ by {eq}`eq-photonuclear-tables-twobody`
   with $b$ and $B$ exchanged, taken to the laboratory with its partner's angle reversed. A
   residual without one (MF 6 LAW 0, the residuals of MT 5 in most evaluations) recoils
   against what was emitted. In the centre of mass its momentum balances theirs, so, with
   the emissions taken as independent,

   $$
   \langle E_R\rangle = y_R\,E_c(M_R) + \frac{1}{M_R}\sum_i y_i\,m_i\,\langle E'_{i,cm}\rangle,
   $$ (eq-photonuclear-tables-recoil)

   over the products $i$ of that reaction, a photon counting $\langle E'\rangle^2/2$ for
   $m\langle E'\rangle$, $y_R$ the residuals' total yield and $M_R$ their mean mass.
9. **Fission.** $F_r = \mathrm{EFR}(E) + \mathrm{EGP}(E)$ from MF 1 MT 458 when the
   evaluation has it, as for neutrons ({doc}`heating`). Without it,
   $F_r = Q_r - \sum y\langle E'_n\rangle$ over the fission neutrons, which keeps the delayed
   betas' and gammas' energy local too.
10. **Production and heating blocks.** PXS starts one grid point below the lowest energy
    where any reaction produces the particle (the reaction's first energy or the first
    positive yield, whichever is higher). PHN and THN are
    {eq}`eq-photonuclear-tables-thn` on the grid. MTRP lists MF 4/MF 5 reactions first,
    then MF 6 ones, ascending. Yields are MFTYPE 6 (12 for MF 4/MF 5 neutrons and for
    de-excitation photons), lin-lin.
11. **Encoding and writing.** The blocks are laid out in `phnout`'s order. Counts,
    locators, MT numbers, frames and interpolation codes are integer words (I20), the rest
    reals (E20.11). The header has ZAID $1000Z + A$ (an isomer $+\,300 + 100m$), the
    suffix `.01u` (0 K) unless another is asked for, temperature 0. The table is written
    through a temporary file, read back in strict mode and compared word for word before
    it replaces an existing one, and its line is added to the `xsdir`.

## Conventions and choices

- **Kalbach-Mann is the photon's.** ENDF-6 defines Eq. {eq}`eq-photonuclear-tables-kalbach`
  for an incident photon, and FUDGE implements it. ACER computes a neutron's slope and
  writes law 44, a neutron's shape. Law 44 cannot hold the photon's shape, so kika writes
  law 61, a larger table.
- **Two-body energies are exact.** ACER writes one law 33 with
  $E' = (A - A_b)/A\,(E + Q)$, which neglects the photon's momentum and puts the threshold
  at $-Q$. One law 33 cannot follow $T_b^*$: the chord over the whole range misses it by
  more than $10^{-4}$ in 753 of the 781 two-body products of ENDF/B-VIII.1, and ACER's form
  by up to 2.6 % (Be-9). The chain of pieces holds it to $10^{-4}$ with at most 13 pieces.
- **MT 5's two-body channels take their own Q.** MT 5 states $Q = 0$. With it, He-3's
  deuteron of $\gamma + {}^3\mathrm{He} \to p + d$ would carry the 5.49 MeV binding as
  kinetic energy; ACER's law 33 does.
- **Masses from AME2020.** The evaluation's AWR and AWP are rounded: from them He-3's
  $p + d$ Q comes out −6.52 MeV, where AME2020 gives −5.49 MeV. The table is packaged with
  kika (`data/ame2020_masses.npz`), regenerated from the evaluation's `mass_1.mas20` by
  `data/make_ame_table.py`.
- **The centre of mass moves with the photon's momentum.** A neutron of energy $E$ gives
  it $\beta = \sqrt{2E/m_n}/(A + 1)$, 2.9 to 9.7 times the photon's; ACER takes its
  laboratory energies with that. The difference in Be-9's charged particles' heating is up
  to 3 %.
- **Distributions stated in the centre of mass are written there** (TYRP = −1), as the
  evaluation states them and as ACER writes them. How a code reading the table takes them
  to the laboratory with a photon projectile is the reader's, and is not documented for
  the `u` class. kika's own laboratory energies (PHN, THN) use
  {eq}`eq-photonuclear-tables-boost`.
- **Sums are not reactions.** ACER adds MT 201-207 into the total: Zr-90's total comes out
  10.9 times MT 5 at the giant resonance, and its PHN and THN are per that total. kika
  leaves them out, and the other sums (H-2 states MT 3 beside its only partial, MT 50).
- **De-excitation photons are written.** ACER does not read MF 12 to MF 15. Eight
  evaluations of ENDF/B-VIII.1 state photons there (719 sections, seven of them on the
  residuals' levels), and ACER's tables have none of them.
- **Delayed neutrons are written.** ACER writes the prompt $\bar\nu$ only (MT 456 over
  MT 452 when both are given). The table has no delayed block, and a chain in MT 18's entry
  keeps the number of neutrons and their spectra. An evaluation that gives a family no
  spectrum (Th-232 and Pu-238 of ENDF/B-VIII.1) has those neutrons left out, and the report
  says so.
- **The heating is what stays.** {eq}`eq-photonuclear-tables-thn` sums the energies of the
  particles and nuclei that deposit locally. ACER builds its THN from $E + Q$ terms: with
  $Q = 0$ in MT 5 that keeps every emitted nucleon's binding energy as heat (5.1 MeV per
  collision in Bi-209 at 8 MeV, where the recoil is a fraction of a keV, and five times the
  exact balance of H-2's photodisintegration), counts capture twice and adds the law-2
  particles' energy. The balance $E + Q - \sum_{n,\gamma} y\langle E'\rangle$ is not used
  instead: MT 5 states no $Q$, and its yields do not conserve A and Z in a quarter of the
  points measured.
- **Unit-base interpolation of spectra.** It is how ACE laws 4, 44 and 61 are sampled
  between incident energies, so the heating matches what a transport code deposits. ACER
  averages the spectra at fixed outgoing energy instead: U-235's MT 5 photons between its
  tables at 5.26 and 6.75 MeV have a mean energy of 99 keV in ACER's PHN and 187 keV in
  kika's at 5.66 MeV.
- **Each reaction starts at its first evaluated energy**, as in ACER. A neutron table
  starts one point below the first non-zero value; here the evaluation's threshold point is
  kept so that the reactions keep their stated ranges.

## Limits and failure modes

- {py:func}`~kika.endf.photonuclear_ace` raises {py:class}`~kika.endf.ProcessingOutputError`:
  - `ace-failed` for a tape that is not photonuclear, for one with resonance parameters,
    and when the table cannot be built or does not read back as written;
  - `ace-table-exists` when the `xsdir` already names the table, unless `replace=True`.
- A two-body particle whose emitted or residual nucleus is not in AME2020 raises
  `KeyError` (none in ENDF/B-VIII.1).
- A Monte Carlo form an ACE law cannot hold (an N-body phase space, a prompt chain with
  delayed families) raises `NotImplementedError`; none in ENDF/B-VIII.1.
- The recoil of {eq}`eq-photonuclear-tables-recoil` neglects the correlations between the
  emitted particles' momenta. It is exact for one emitted particle.
- The decoder ({py:func}`kika.ace.read_ace`) refuses a table whose locators fall outside
  `XSS`, whose counts do not fit their blocks, with a particle kind not in the seven, a
  frame other than ±1 or a yield record of an unknown MFTYPE.

## Validation

- Format: the six tables NJOY2016 writes for its photonuclear tests (U-235 and Bi-209 of
  ENDF/B-VIII.0, Co-59, He-3 and Pu-239 of the IAEA library, Ra-226 of TENDL-2019) and an
  LA150u table (C-12) are read, encoded from what was read and written: words, integer
  spellings and bytes come back unchanged
  (`kika/ace/tests/test_photonuclear_ace.py::test_a_table_reads_and_writes_back_byte_for_byte`,
  run where those tables are on disk).
- Kinematics and systematics without data: Eq. {eq}`eq-photonuclear-tables-kalbach`
  normalised and with $\langle\mu\rangle = r(\coth a - 1/a)$ to $10^{-10}$; the photon slope
  is the neutron's times its factor; AME2020 gives He-3's break-up Q; the two-body energies
  conserve energy and momentum; the centre-of-mass energy is $\tfrac12 m_b\beta^2$
  (`kika/processing/tests/test_photonuclear_tables.py`).
- Content, on ENDF/B-VIII.1 where it is on disk (same file):
  - H-2's heating is the exact balance $E + Q - \langle E_n\rangle$ of its
    photodisintegration to $10^{-3}$
    (`::test_deuteron_photodisintegration_heats_by_the_exact_balance`);
  - He-3's deuteron takes what the $p + d$ channel's Q leaves
    (`::test_lumped_two_body_takes_the_channel_q`);
  - U-235's fission neutrons are $\bar\nu_p + \bar\nu_d$, with one law per delayed family
    (`::test_fission_neutrons_include_the_delayed_ones`);
  - {py:func}`kika.endf.photonuclear_ace` writes He-3's table, which reads back
    (`::test_photonuclear_ace_writes_a_table_that_reads_back`).
- Against NJOY2016.78 ACER `iopt = 5`, after RECONR at 0.001 as its tests run it, on
  sixteen ENDF/B-VIII.1 evaluations (H-2, He-3, Be-9, C-12, O-16, Ca-40, Fe-56, Co-59,
  Zr-90, Nb-93, Pb-208, Bi-209, Th-232, U-235, U-238, Pu-239), on the energies both grids
  have (measured 2026-10-20):
  - the same reactions (but Zr-90's MT 201-207) and particle kinds, and the same reactions
    for each particle;
  - cross sections within 4·10⁻⁷ (NJOY's seven digits), except U-235 and U-238, which
    state a law other than lin-lin, within 7·10⁻⁴ (NJOY linearises to 10⁻³, kika to 10⁻⁴);
  - production cross sections within 8·10⁻⁷, except where kika writes what NJOY drops:
    Co-59's de-excitation photons, and the delayed neutrons of U-235, U-238 and Pu-239
    (0.1 to 1.2 %);
  - particle heating numbers equal where the laws are equal and away from thresholds
    (Pb-208 within 2·10⁻⁵, Ca-40 within 6·10⁻⁵); the other differences are the choices
    above, two-body kinematics, the centre of mass's velocity and the interpolation of
    spectra (Be-9's charged particles, up to 3 %);
  - THN equal for Pb-208 and Ca-40 (2·10⁻⁶ and 3·10⁻⁵), where every reaction states its
    products' energies; elsewhere the differences of the heating bullet above.
- TODO(next): the whole ENDF/B-VIII.1 library through the entry point and NJOY's reader.

## Diagnostics

The report of {py:func}`~kika.processing.photonuclear.photonuclearTable` (`table.report`)
says:

- which MTs were left out as sums;
- which delayed families had no spectrum;
- that residuals without a distribution were given their momentum-balance recoil, and in
  which MTs;
- that fission without MF 1 MT 458 keeps the delayed betas' and gammas' energy local;
- each product left out, and why (no distribution, no multiplicity, no Monte Carlo form).

None of these is an error: the first four are the method; the last one should be read.

## Implementation

- {py:mod}`kika.processing.photonuclear.tables`: reactions, grid, products, two-body
  kinematics, de-excitation photons, delayed neutrons, heating
  ({py:func}`~kika.processing.photonuclear.photonuclearTable`). The masses are
  `data/ame2020_masses.npz`, regenerated by `data/make_ame_table.py`.
- {py:mod}`kika.processing.tabulation.kalbach` and {py:mod}`kika.processing.tabulation.correlated`:
  the photon's slope and shape.
- {py:mod}`kika.processing.heating`: the mean laboratory energies, with the photon's
  centre-of-mass velocity.
- {py:mod}`kika.ace.encoders.photonuclear`: the layout.
- {py:mod}`kika.ace.parsers.parse_photonuclear` and {py:mod}`kika.ace.classes.photonuclear`:
  decoding and the decoded table.
- {py:mod}`kika.endf.process`: the entry point, header and `xsdir`.

## References

- A. C. Kahler (ed.), *The NJOY Nuclear Data Processing System, Version 2016*,
  LA-UR-17-20093, Los Alamos National Laboratory (2016): ACER, `iopt = 5`.
- ENDF-6 Formats Manual, CSEWG Document ENDF-102, BNL-203218-2018-INRE (2018): §1.2,
  §6.2.3.2 (Kalbach-Mann for incident photons), §12, §14.
- M. B. Chadwick, P. G. Young and S. Chiba, *Photonuclear angular distribution systematics
  in the quasideuteron regime*, J. Nucl. Sci. Technol. 32, 1154-1158 (1995).
- W. J. Huang, M. Wang, F. G. Kondev, G. Audi and S. Naimi, *The AME 2020 atomic mass
  evaluation (I)*, Chinese Physics C 45, 030002 (2021).
- C. Kalbach, *Systematics of continuum angular distributions: extensions to higher
  energies*, Phys. Rev. C 37, 2350 (1988).
