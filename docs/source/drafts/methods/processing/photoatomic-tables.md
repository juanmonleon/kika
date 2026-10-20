# Writing photo-atomic tables

```{admonition} At a glance
:class: note
**Computes:** the photo-atomic ACE (A Compact ENDF) table of an element in the EPRDATA layout (photon interactions, atomic relaxation, Compton profiles and electron interactions) from its ENDF-6 (Evaluated Nuclear Data File) evaluations.
**Applies to:** the photo-atomic (MF 23, MF 27), atomic-relaxation (MF 28) and electro-atomic (MF 23, MF 26) sublibraries of one element.
**Entry points:** {py:func}`kika.endf.process_photoatomic`, {py:meth}`kika.endf.PhotoatomicProcessing.write_ace`, {py:func}`kika.ace.processing.photoatomic.photoatomicTable`.
**Status:** experimental.
```

## Background

A photon transport code samples five interactions with atoms:

- coherent (Rayleigh) scattering;
- incoherent (Compton) scattering;
- photo-electric absorption, by subshell;
- pair production in the nuclear field and in the electron field.

Each interaction has a cross-section $\sigma(E)$ in MF 23 (ENDF-6 §23). The angular
distributions of the two scatterings come from the form factor $F(x)$ and the incoherent
scattering function $S(x)$ of MF 27 (§27). Here $x = \sin(\theta/2)/\lambda$, in Å⁻¹. A
photo-ionisation leaves a vacancy, and the atom relaxes by emitting X-rays and Auger
electrons, with the probabilities of MF 28 (§28). The electron sublibrary gives the
electrons' cross-sections and their secondary distributions (MF 23, MF 26). Compton
scattering on a bound electron is broadened by the electron's momentum, its Compton
profile $J(p_z)$, which ENDF does not carry. Every library takes the profiles from Biggs,
Mendelsohn and Mann (1975).

The EPRDATA layout (Hughes 2013, Hughes 2016; NXS(6) = 3, the layout of LANL's eprdata14
and eprdata25) holds all of this in one table. MCNP reads every block. OpenMC reads the
photon, Compton and relaxation blocks.

## What it computes

From the three evaluations of one element, one table with these blocks:

- **ESZG:** the energy grid and $\ln\sigma$ of incoherent, coherent, photo-electric and pair
  production.
- **JINC, JCOH:** $S(x)$; and $F(x)$ with its integral
  $I(x) = \int_0^{x^2} F(\sqrt u)^2/Z^2\,du$, from which the coherent angle is sampled.
- **JFLO:** the K- and L-shell fluorescence data of the older MCPLIB treatment.
- **LHNM:** the heating number, the mean energy deposited per collision.
- **LNEPS, LBEPS, LPIPS, LSWD, SWD:** per Compton shell, its electron number, binding
  energy, cumulative probability and profile.
- **SUBSH, SPHEL, RELO, XPROB:** the subshells, their photo-ionisation cross-sections and
  their relaxation transitions.
- **ESZE to SELAS:** the electron cross-sections, excitation and bremsstrahlung energy
  losses, elastic angular tables, ionisation and bremsstrahlung spectra, and the
  transport-corrected elastic cross-section.

The heating number at photon energy $E$ is

$$
H(E) = \frac{\sigma_{\mathrm{inc}}\,(E - \bar E_{\mathrm{inc}} - X_C) + \sigma_{\mathrm{pair}}\,(E - 2m_ec^2)
+ \sum_s \sigma_s\,(E - X_s)}{\sigma_{\mathrm{coh}} + \sigma_{\mathrm{inc}} + \sigma_{\mathrm{pair}} + \sigma_{\mathrm{pe}}},
$$ (eq-photoatomic-tables-heating)

with $\bar E_{\mathrm{inc}}$ the mean energy of the Compton-scattered photon, $\sigma_s$ the
photo-ionisation cross-section of subshell $s$, $X_s$ the mean x-ray energy the relaxation
cascade radiates from a vacancy in $s$, and $X_C(E)$ its mean after an incoherent
collision (see *Method*). Here $m_ec^2 = 510\,998.950\,69$ eV.

## Method

### Energy grid and cross-sections

1. Each MF 23 table loses a step at either end: a repeated first or last energy keeps only
   its inner copy.
2. The grid is the multiset union of every table's energies. A value repeated in any table
   (an absorption edge) is repeated in the grid. A table that starts or stops on a non-zero
   value inside the grid gets that energy repeated, so the opening is a step.
3. Each cross-section is evaluated lin-lin on the grid, the law every MF 23 table of
   ENDF/B-VIII.1 states:
   - it is zero outside the table;
   - at a repeated energy, the first copy holds the value below and the second the value
     above.
4. The photo-electric column is the sum of the subshell cross-sections and the pair column
   MT 515 + MT 517, not the evaluated MT 522 and MT 516.
5. **Log-log refinement.** MCNP and OpenMC interpolate every column log-log, while the
   evaluation is lin-lin. Every interval where a column read log-log misses its lin-lin
   value by more than $10^{-3}$, at a quarter, half or three quarters of the interval, is
   split at its midpoint until none does. The columns checked are incoherent, coherent,
   pair, photo-electric and each subshell. An interval a column opens from zero (the pair
   threshold) first gets a node at

   $$
   E = a + \sqrt{10^{-3}}\,(b - a),
   $$ (eq-photoatomic-tables-opening)

   since log-log cannot read a zero, and the transport's infinitesimal loses at most
   $10^{-3}$ of the interval's lin-lin area. Every node of step 2 is kept.

### Form factors

$S(x)$ and $F(x)$ are written as the evaluation tabulates them. $I$ is the trapezoid in
$u = x^2$:

$$
I_k = I_{k-1} + \tfrac12\,(x_k^2 - x_{k-1}^2)\,\frac{F_{k-1}^2 + F_k^2}{Z^2},\qquad I_1 = 0.
$$ (eq-photoatomic-tables-coherent)

### Relaxation

- **Normalisation.** A subshell's transition probabilities are normalised to one.
- **Order.** The radiative transitions are listed first, ordered by the subshell the
  vacancy moves to. The non-radiative ones follow, ordered by their two subshells.
- **Energy.** A transition's energy is the difference of binding energies:

  $$
  E_{i\to j} = E_i - E_j \;\;\text{(radiative)},\qquad E_{i\to jk} = E_i - E_j - E_k \;\;\text{(Auger)}.
  $$ (eq-photoatomic-tables-transition)

- **Cumulative probability.** XPROB holds the running sum of the probabilities, the last
  set to one.
- **Subshell arrays.** SUBSH holds each subshell's designator, electrons, binding energy,
  the cumulative electron fraction and its number of transitions.

### Heating numbers

- **Compton mean.** $\bar E_{\mathrm{inc}}$ is

  $$
  \bar E_{\mathrm{inc}} = \frac{\int E'\,f\,dE'}{\int f\,dE'},\qquad
  f(E') = S(x)\left(1 + \mu^2 + k\,k'(1-\mu)^2\right),
  $$ (eq-photoatomic-tables-incoherent)

  over $E' \in [E/(1+2k), E]$, with $k = E/m_ec^2$, $k' = E'/m_ec^2$,
  $\mu = 1 + 1/k - 1/k'$ and $x = c\,E\sqrt{(1-\mu)/2}$, $c = 10^{-10}e/(hc)$ in Å⁻¹/eV.
  $f$ is the Klein-Nishina $d\sigma/dE'$ up to a constant. The integral is taken in $x$
  over every interval of the $S$ table, each also cut at a geometric ladder of ratio 1.2,
  with 16-point Gauss-Legendre: $S$ is linear on each piece and the other factors are
  smooth.
- **Cascade x-rays.** With $p_t$ and $E_t$ the probability and energy of each
  transition that fills a vacancy in $s$,

  $$
  X_s = \sum_{t\ \mathrm{radiative}} p_t\,(E_t + X_j) + \sum_{t\ \mathrm{non\text{-}radiative}} p_t\,(X_j + X_k),
  $$ (eq-photoatomic-tables-cascade)

  where $j$ (and $k$) are the subshells the transition leaves a vacancy in.
  - A subshell with no transitions is filled without radiation: $X_s = 0$.
  - A radiative line of non-positive energy carries nothing.
  - $X_C(E)$ is the mean of $X_s$ over the electrons bound below $E$, each equally likely
    to be struck.
- **Fluorescence block.** JFLO follows the MCPLIB fluorescence model of NJOY's `acepho`:
  K-shell lines above the K edge and, for Z > 30, L-shell lines above the mean L edge,
  weighted by the edge jumps of the photo-electric cross-section, with NFLO = 0, 2, 4, 5
  or 6 by Z. It is what a code without the relaxation data samples; the heating number
  does not use it.

### Compton shells

The profiles are BMM's, normalised: $p = J/\int J\,dp_z$, and the cumulative trapezoid
with its last value set to one. They are matched to the evaluation's subshells as follows.

- A BMM shell without $j$ (Z < 36) stands for every subshell of the same $n$ and $l$.
- A BMM shell with $j$ stands for the subshell of the same $n$, $l$ and $j$. It also
  stands for a subshell of the same $n$ and $l$ whose $j$ BMM does not list (Pb's 6p3/2
  joins 6p1/2).
- A BMM shell with no subshell in the evaluation is dropped (Ir 6s1/2, Ce 5d3/2).

A shell's electron number is the sum over its subshells, and its binding energy their
population-weighted mean.

### Electron blocks

- **Grid and cross-sections.** The electron cross-sections go on their own union grid, the
  same way as the photon grid (steps 1-4), and are cut below 10 eV. Elastic scattering
  starts there, and a total below it would be incomplete.
- **Distributions.** Each tabulated distribution is normalised, with its cumulative last
  value set to one.
- **Energy losses.** The excitation and bremsstrahlung losses are $E - \bar E$ of the
  outgoing electron, on their own grids.
- **Transport-corrected elastic.** SELAS is $\overline{1-\mu}\,\sigma_{\mathrm{el}}$,
  where $\overline{1-\mu}$ is the exact mean of each lin-lin angular table, interpolated
  log-log in energy. It is followed by the total elastic MT 526.

## Conventions and choices

- **EPRDATA, not MCPLIB.**
  - NJOY2016's `acepho` writes the older MCPLIB layout: fixed form-factor grids, no
    subshells, no Compton profiles.
  - OpenMC refuses a table without subshells, and MCNP needs EPRDATA for detailed
    relaxation and electrons.
  - kika writes the layout of LANL's eprdata25 (NJOY3), whose method it follows block by
    block.
- **All three evaluations are required.** An EPRDATA table is photons, relaxation and
  electrons together. A table with the electron blocks missing is not produced: MCNP would
  read NXS(6) = 3 and expect them.
- **The grid reads log-log within the tolerance.** eprdata25 writes the evaluation's
  lin-lin nodes alone. Between them, the log-log reading of the transport codes misses the
  evaluation by up to 9 % (Fe pair threshold) and 4.6 % (H coherent at 1 eV). The
  refinement costs 160 to 260 points per element (about 2 %).
- **Transition energies from the binding energies.**
  - MF 28's ETR do not conserve energy with the binding energies of the same file (Fe
    K → L2: 6349.85 eV against 7117 − 726 = 6391 eV).
  - GNDS has no node for them.
  - eprdata25 recomputes them, and kika does the same.
  - Where the binding energies make an Auger transition's energy zero or negative (117
    transitions in 41 elements of ENDF/B-VIII.1), it is written as it comes out, and the
    run notes the count.
- **The Compton mean is integrated exactly.** eprdata25 uses an adaptive Gauss-Lobatto
  quadrature at $10^{-8}$. Where the heating number is $E - \bar E$ of nearly equal terms,
  that leaves up to $2.4\cdot10^{-4}$ relative error in $H$, and one negative heating
  number (Ce, 5.7 eV). kika's piecewise Gauss-Legendre agrees with an adaptive reference
  at $10^{-13}$ to $10^{-10}$.
- **Compton shells follow BMM's list.**
  - BMM omits one $j$-component of an open shell in 25 elements (two in Pa, U, Np and Bk).
  - eprdata25 writes the shell arrays from the evaluation's subshells and the profiles
    from BMM's list, so in those elements LNEPS, LBEPS and LPIPS are out of step with the
    profiles and with NXS(5). For example, Pb reads LBEPS(1) = 1.33 MeV and its electrons
    sum to 80.67.
  - kika writes one shell per profile.
- **The Eu-63 profile is the paper's.** NJOY3's transcription of BMM reads the 2p1/2 value
  at $p_z = 30$ a.u. as `4.61-03`, which C++ evaluates as 1.61. The paper's value is
  $4.61\cdot10^{-3}$.
- **The heating number follows the relaxation cascade, not NJOY's fluorescence model.**
  - NJOY's model subtracts one shell's fluorescence from the whole photo-electric
    cross-section. For Z > 30, above the K edge it subtracts the L lines only, as the
    L value overwrites the K one. eprdata25 does the same.
  - The transport samples the whole cascade from the subshell that was ionised, and
    follows the x-rays it emits. A heating number that also counts their energy as
    deposited counts it twice, wherever a collision tally uses it.
  - In lead at 100 keV (ENDF/B-VIII.1, eprdata25, 2026-10-20), NJOY's number is 91 keV per
    collision against 36 keV with the cascade.
  - Vacancies the evaluation gives no transitions for are filled locally, with no
    radiation. A code may emit a photon there instead (OpenMC does), and that is the
    residual left in *Validation*.
- **The coherent integral is the trapezoid in $x^2$.**
  - The transport samples within an interval as if $F^2$ were linear in $x^2$, so the
    trapezoid is the integral consistent with how the table is read.
  - The exact integral of $F$ lin-lin in $x$ differs by at most $7.3\cdot10^{-4}$ of an
    interval's mass, below the processing tolerance.

## Limits and failure modes

- {py:func}`~kika.endf.process_photoatomic` raises `ValueError`:
  - when an evaluation is missing;
  - when one is not the sublibrary it should be (photo-atomic NSUB 3, relaxation NSUB 6,
    electro-atomic NSUB 113);
  - when the three are not of one element.
- The table builder refuses:
  - a photo-atomic table whose law is not lin-lin;
  - relaxation subshells that are not the photo-ionisation subshells;
  - a transition leaving a vacancy in a subshell with no binding energy.
- {py:meth}`~kika.endf.PhotoatomicProcessing.write_ace` raises
  {py:class}`~kika.endf.ProcessingOutputError` with `ace-table-exists` when the `xsdir`
  already names the table (unless `replace=True`), and with `ace-failed` otherwise.
- Only ENDF-6 input is processed. A GNDS evaluation is read into the same model, but the
  entry point takes the ENDF tapes.

## Validation

- **Union grid and steps.** A repeated edge, a table opening on a step, and lin-lin
  values between nodes
  (`kika/ace/tests/test_photoatomic_tables.py::test_union_repeats_edges_and_opens_tables_on_steps`).
- **Log-log refinement.** The refined grid reads log-log within $10^{-3}$ of the lin-lin
  data, keeping every node (`::test_the_grid_reads_log_log_within_the_tolerance`).
- **Compton mean.** The incoherent mean equals an adaptive quadrature to $10^{-9}$
  relative in $E - \bar E$ (`::test_incoherent_mean_agrees_with_an_adaptive_quadrature`).
- **Compton shells.** The matching rules
  (`::test_compton_shells_light_element_groups_by_n_and_l`,
  `::test_compton_shells_missing_j_partner_joins_its_sibling`,
  `::test_compton_shells_drop_a_shell_the_evaluation_lacks`,
  `::test_europium_profile_has_the_paper_value`).
- **Entry point.** On ENDF/B-VIII.1 hydrogen: the table is written with its `xsdir` line
  and reads back with every block, electrons included; a repeated name is refused; all
  three evaluations are required (`kika/processing/tests/test_process_photoatomic.py`).
- **Against LANL's eprdata25.** Same ENDF/B-VIII.1 input, all 100 elements, on
  eprdata25's nodes (measured 2026-10-20):
  - grid, ESZG, SPHEL: $\le 5\cdot10^{-12}$ (the 13 printed digits);
  - JINC, JCOH, JFLO, SUBSH, XPROB: $\le 5\cdot10^{-13}$;
  - the Compton blocks of the 75 elements where eprdata25's arrays are consistent:
    $\le 5\cdot10^{-13}$;
  - electron blocks: 2 736 460 words, 340 differing in the 13th digit, plus Ni and Ag,
    where a subshell opens on a non-zero value inside the grid: kika keeps the step,
    eprdata25 ramps from the 10 eV cut;
  - heating numbers, below the first subshell that radiates (where both are
    $E - \bar E_{\mathrm{inc}}$ weighted): 99.9 % of points within $10^{-6}$. Where they
    differ (up to $2.4\cdot10^{-4}$), kika agrees with an adaptive reference to
    $10^{-10}$ and eprdata25 does not. Above it they differ by the cascade (see
    *Conventions and choices*).

  The tests enforce this on hydrogen
  (`kika/processing/tests/test_process_photoatomic.py::test_hydrogen_agrees_with_eprdata25`),
  iron and lead (`test_photoatomic_tables.py::test_blocks_agree_with_eprdata25`), and the
  electron blocks of iron and lead
  (`kika/ace/tests/test_photoatomic_electrons.py`), when eprdata25 is present.
- **Heating against OpenMC's analog heating.** OpenMC 0.16.0, LED electrons, a point source
  in a sphere of three mean free paths of water, iron or lead, at 5 keV (iron) and 20 keV,
  100 keV, 1 MeV and 10 MeV, $2\cdot10^5$ photons ($5\cdot10^4$ at 10 MeV); the track-length
  sum of $\phi\,\Sigma_t H$ against the analog energy deposition (2026-10-20):
  - kika's number: within 2.8 % in every case and within $2.5\sigma$ in 8 of 13; NJOY's
    model: +71 % (lead, 100 keV) and +5.9 % (lead, 1 MeV);
  - what is left (iron 5 keV +1.3 %, water 20-100 keV +0.4 %, lead 20 keV +2.8 % and
    100 keV +0.9 %) is
    OpenMC's photon at the binding energy for a vacancy with no transitions: a number that
    removes it too agrees within $2.1\sigma$ there;
  - at 10 MeV, run again with 2·10⁵ (water) and 4·10⁵ (iron, lead) photons and the
    heating split by reaction and by incident energy: water −0.34 %, iron +0.13 %, lead
    +0.12 %, every split within 1.8σ, the first collisions' reaction rates equal to
    OpenMC's to 10⁻⁷ (measured 2026-10-20). The −0.9 % of the shorter runs, shared by the
    three materials, came from their one random seed: their first collisions drew the
    same numbers.
- **kika's tables against eprdata25 in OpenMC.** Same runs, leaving-photon spectrum:
  $\chi^2/\nu$ between 0.25 and 1.63 (p ≥ 0.19) in all 13 cases. With thick-target
  bremsstrahlung (TTB) electrons instead, the same 13 cases (2026-10-20): water and iron
  give the same histories as eprdata25, bit for bit; lead gives $\chi^2/\nu$ between 0.14 and
  0.76 (p ≥ 0.97) and analog heating within $0.3\sigma$, and its flux differs (−8.1σ at
  1 MeV) because kika's Compton shells follow BMM's list (see *Conventions and choices*), as
  with LED.
  Under TTB the sum of $\phi\,\Sigma_t H$ exceeds the analog heating at 10 MeV (+1.7 %
  water, +7.5 % iron, +19 % lead) for every heating number, OpenMC's own included: TTB
  emits the bremsstrahlung as photons that can leave, where $H$ deposits it locally.
- **ENDF evaluations through the model.** The ENDF/B-VIII.1 photoat, atomic_relax and
  electrons tapes (100 each) are read into the model and written back byte for byte
  (`kika/endf/model_adapter/tests/test_photoatomic_model.py`,
  `test_electroatomic_model.py`).
- **Cost.** The 100 elements take 62 s on 10 processes, 3 to 9 s each (measured
  2026-10-20).

## Diagnostics

{py:attr}`kika.endf.PhotoatomicProcessing.notes` lists:

- each subshell joined to a Compton shell of another $j$ ("subshell 6p3/2 has no BMM
  profile; it joins 6p1/2");
- each BMM shell dropped;
- the number of relaxation transitions whose energy from the binding energies is not
  positive.

None of these changes the cross-sections. The Compton entries say which profile stands
for which electrons.

## Implementation

- {py:mod}`kika.ace.processing.photoatomic.tables`: grid, cross-sections, form factors,
  relaxation, fluorescence and heating numbers, Compton shells, and the header
  ({py:func}`~kika.ace.processing.photoatomic.photoatomicTable`).
- {py:mod}`kika.ace.processing.photoatomic.compton`: the BMM table and its matching to the
  subshells. The data file is `data/bmm_compton_profiles.npz`, regenerated from NJOY3's
  transcription by `data/make_bmm_table.py`.
- {py:mod}`kika.ace.processing.photoatomic.electrons`: the electron blocks.
- {py:mod}`kika.ace.processing.photoatomic.process`: the entry point and writing.
- `kika/ace/encoders/photoatomic.py`: the layout ({doc}`../formats/ace`).
- `kika/endf/model_adapter/photoatomic.py` and `kika/gnds/photoatomic.py`: the three
  sublibraries in the model.

## References

- F. Biggs, L. B. Mendelsohn and J. B. Mann, "Hartree-Fock Compton profiles for the
  elements", *Atomic Data and Nuclear Data Tables* 16, 201-309 (1975).
- H. G. Hughes, *An Electron/Photon/Relaxation Data Library for MCNP6*, LA-UR-13-27377,
  Los Alamos National Laboratory (2013).
- H. G. Hughes, *Improvements in Electron-Photon-Relaxation Data for MCNP6*, LA-UR-16-20840,
  Los Alamos National Laboratory (2016).
- W. Haeck and M. Lively, *Release of eprdata25: ENDF/B-VIII.1-Based ACE Data Files for
  Photo- and Electro-atomic Transport*, LA-UR-26-27338, Los Alamos National Laboratory (2026).
- ENDF-6 Formats Manual, CSEWG Document ENDF-102, BNL-203218-2018-INRE (2018): §23, §26,
  §27, §28.
