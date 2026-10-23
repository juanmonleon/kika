# Computing gas-production cross-sections

```{admonition} At a glance
:class: note
**Computes:** the production cross-sections of protons, deuterons, tritons, He-3 and
alphas (MT 203 to 207), in b, at each processed temperature, on the energy grid of that
temperature.

**Applies to:** an incident-neutron evaluation processed for Monte Carlo; written to the
OpenMC HDF5 file (redundant), to the ACE table and to the dosimetry ACE table. Not
written to the PENDF.

**Entry points:** {py:func}`kika.hdf5.processing.processSuite`, with
{py:func}`kika.hdf5.processing.gasMultiplicities` for the count of one reaction; through
{py:func}`kika.process` and {py:func}`kika.endf.process_endf`
({py:meth}`kika.endf.EndfProcessing.write_hdf5`, {py:meth}`kika.endf.EndfProcessing.write_ace`,
{py:meth}`kika.endf.EndfProcessing.write_dosimetry_ace`).

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

ENDF-6 reserves MT 203 to 207 for the total production of a light particle: MT 203 for
protons, 204 for deuterons, 205 for tritons, 206 for He-3 and 207 for alphas (ENDF-6
Appendix B; Brown, 2023). Each is the number of those particles a collision produces, summed over every
reaction and weighted by its cross-section. It is a response, not a reaction: no particle
is transported through it, and a transport code uses it only to count, for instance the
helium and hydrogen made in the steel of a vessel or a first wall, or the tritium bred in
lithium.

An evaluation may state MT 203-207 in MF 3, and most do not. NJOY computes them from the
reactions, in its module GASPR, and writes them to the PENDF tape, from which ACER copies
them into the ACE table (MacFarlane et al., 2017). The official OpenMC libraries are made
from those ACE tables, so they carry GASPR's sums.

A reaction tells which particles it emits in two ways. Its products, with their
multiplicities, are stated in MF 4 or MF 6. Its MT also defines them: MT 22 is
(n,n$\alpha$), MT 103 is (n,p), and MT 600 to 649 are (n,p) to the levels of the residual
(ENDF-6 Appendix B). For an inelastic level (MT 51 to 91), the LR flag of its MF 3 says
what the residual level breaks up into, for instance LR = 33 for a triton (ENDF-6 §3).

## What it computes

For each light particle $g \in \{\mathrm{p}, \mathrm{d}, \mathrm{t}, {}^3\mathrm{He}, \alpha\}$
and each processed temperature,

$$
\sigma_g(E) = \sum_r y_{g,r}(E)\,\sigma_r(E)
$$ (eq-gas-production-sum)

where $r$ runs over the partial reactions of the evaluation, $\sigma_r(E)$ is the
cross-section of $r$ at that temperature (on the temperature's common grid), and
$y_{g,r}(E)$ is how many particles $g$ reaction $r$ produces per collision. The result is
written as MT 203 ($g = \mathrm{p}$), 204 (d), 205 (t), 206 (${}^3$He) and 207 ($\alpha$).
A particle that no reaction produces gives no MT.

## Method

1. **The reactions summed.** Every reaction of the processed evaluation and every fission
   chance. The redundant sums (MT 1, 3, 4, 27, 101, and MT 103-107 or MT 16 when their
   levels are given) are not partials and are left out, as is the evaluation's own MT
   203-207 (see *Conventions and choices*).

2. **The particles a reaction states.** When a reaction has among its products at least
   one proton, deuteron, triton, He-3 or alpha (from MF 4 or MF 6, a residual included
   when the file lists it), $y_{g,r}(E)$ is the multiplicity the file gives each product
   of kind $g$, summed when several are. A two-body product that states no multiplicity
   counts as one.

3. **The particles its MT defines.** A reaction that states no light particle (MF 3
   alone, or products that are only neutrons and heavy nuclei) produces what its MT
   defines:
   - MT 11, 16, 17, 22-25, 28-30, 32-37, 41, 42, 44, 45, 102-109, 111-117 and 152-200: the
     particles of their definition (ENDF-6 Appendix B); MT 102, radiative capture, emits
     none;
   - MT 51 to 91: one neutron and the particles LR names, as in the LR table of ENDF-6 §3
     (LR = 22: an alpha; 23: three alphas; 28: a proton; 29: two alphas; 32: a deuteron;
     33: a triton; 34: a He-3; 35: a deuteron and two alphas; 36: a triton and two
     alphas; 24, 25 and 30 add neutrons to an alpha or two);
   - MT 600-649, 650-699, 700-749, 750-799 and 800-849: one proton, deuteron, triton, He-3
     or alpha respectively.

   To these it adds the residual, when it is light itself. With the target $(Z, A)$ and
   the emitted particles $(z_i, a_i)$, $n_i$ of each,

   $$
   Z_\mathrm{R} = Z - \sum_i n_i z_i, \qquad A_\mathrm{R} = A + 1 - \sum_i n_i a_i
   $$ (eq-gas-production-residual)

   A residual that is a proton, deuteron, triton, He-3 or alpha counts as one more of its
   kind; Be-8 counts as two alphas, the two it breaks up into. A reaction whose MT has no
   fixed list of particles (MT 2, 5, 18-21, 38 and the sums) contributes nothing by this
   rule; MT 5 contributes through the products it states (step 2).

4. **The sum.** Each count multiplies the reaction's cross-section at the temperature, on
   that temperature's common grid, and {eq}`eq-gas-production-sum` adds them. A
   tabulated multiplicity is evaluated on the grid with its own interpolation law. Each
   MT starts at the first grid point of the reactions that contribute to it.

## Conventions and choices

- **A stated product counts before the MT's definition.** The file's multiplicity can
  depend on energy, and it is the only source for MT 5, whose particles have no fixed
  list. GASPR reads the MF 6 yields of MT 5 and counts every other MT by its definition;
  where an evaluation states the particles its MT defines, as it should, the two are the
  same.
- **The residual counts when it is light.** A light residual is a gas atom made by the
  collision like any emitted particle: Li-7's levels with LR = 33 break up into a triton
  and an alpha, and the alpha is the residual; Li-7 (n,2n$\alpha$) leaves a deuteron;
  Be-9 (n,2n) leaves Be-8, two alphas; capture on H-1 makes a deuteron. GASPR counts the
  residual the same way, Be-8 included.
- **The evaluation's own MT 203-207 is replaced by the sum.** It is a total over the
  reactions, not one of them: summed with them it would count each particle twice, and
  treated as a reaction it would add its particles to the heating. kika writes
  {eq}`eq-gas-production-sum` in its place, as GASPR does, because the sum is consistent
  with the reactions the table transports. Neither is the better one in every file.
  ENDF/B-VIII.1 Li-7 states, in its MT 207, the two alphas of each capture (Li-8 decays
  to Be-8, two alphas): 4.6 b at $10^{-5}$ eV, which no reaction of the file emits and
  the sum, kika's as NJOY's, leaves out. JENDL-5 F-19 states MT 203-207 only up to
  20 MeV, while its MT 5 goes on emitting protons up to 200 MeV (0.44 b at 190 MeV), which
  the sum keeps. The report says, for each replaced MT, how far the two are apart
  (*Diagnostics*).
- **The sum is taken after the broadening.** The partials are broadened first and the gas
  production is their weighted sum at each temperature, as GASPR runs after BROADR. No
  broadening is applied to the sum itself.
- **Not in the PENDF.** NJOY's GASPR writes MT 203-207 to the PENDF tape, in place of
  any the evaluation states. kika's PENDF carries the evaluation's sections reconstructed
  and broadened, with the heating ({doc}`processing-chain`): an MT 203-207 the
  evaluation states stays in it as stated, broadened like the others, and the sum goes
  to the HDF5 and ACE files, which are where a transport code reads it.
- **The dosimetry table keeps a stated MT 203-207.** A dosimetry tape usually carries a
  few reactions, not all of them, so the sum over its reactions would be incomplete; the
  dosimetry ACE table writes an MF 3 MT 203-207 as the tape states it, and the sum only
  for the MTs the tape does not state ({doc}`dosimetry-tables`).

## Limits and failure modes

None raised: a reaction whose particles cannot be counted contributes nothing, and an
evaluation that produces no light particle gives no MT 203-207.

- **Particles from the decay of the residual are not counted.** A radioactive residual
  that later emits an alpha (Li-8, through Be-8) or a proton is not followed; only what
  the collision produces, and a residual that is itself light, count. GASPR does the
  same.
- **MT 5 without stated products counts nothing.** Its particles have no fixed list.
- **A stated MT 203-207 is not checked against its reactions as data.** The replacement
  measures the difference and reports it; whether the evaluation's total is right is for
  {doc}`../checks/data-checks`.

## Validation

- **The count of one reaction**, `kika/hdf5/tests/test_gas_from_the_mt.py`: a level
  series emits its particle, an MT its definition, an LR level its break-up, a stated
  product as stated; a light residual counts (Li-7 levels with LR = 33, Li-7 MT 24 and 25,
  H-1 capture, H-2 and Be-9 (n,2n)), a heavy one does not, and the MTs beyond 117 emit
  their definition.
- **A stated MT 203-207 replaced**,
  `kika/hdf5/tests/test_gas_from_the_mt.py::test_a_stated_gas_production_is_replaced_by_the_sum`
  (with the tapes, `--deep`): on ENDF/B-VIII.1 Li-7 and JENDL-5 F-19 the evaluation's MT
  203-207 is no reaction of the output, the sum is written redundant, the report has one
  entry per replaced MT, and between 1 MeV and 20 MeV the sum differs from the stated
  total by less than $10^{-3}$ of its peak.
- **The report's check**,
  `kika/hdf5/tests/test_gas_from_the_mt.py::test_the_report_checks_a_replaced_gas_production`:
  no check without a stated MT 203-207, and passed, note or warning by the share of the
  stated peak.
- **Against NJOY 2016.78 GASPR**, run as RECONR (tolerance 0.001), BROADR (293.6 K,
  0.001) and GASPR on ENDF/B-VIII.1 H-2, Li-6, Li-7, Be-9 and B-11, against kika's sum at
  293.6 K with its default tolerances: the largest relative difference over the NJOY
  points between 1 MeV and 20 MeV is $7.2\cdot10^{-4}$ (H-2 MT 205 at 1 MeV), and
  $4.4\cdot10^{-4}$ or less for every MT of the other four (measured 2026-10-23). The
  difference is the two codes' grids within their tolerances. ENDF/B-VIII.1 B-10 is not
  compared: NJOY's RECONR stops on it ("ill-behaved threshold").
- **Against NJOY's ACE tables** of 29 ENDF/B-VIII.1 evaluations (N-14, Fe-54, 57, 58,
  Cr-50 to 54, Mn-55, Ni-58 to 64, Ga-69, 71, Pu-238 to 242 among them): MT 203-207
  within $6\cdot10^{-7}$ relative between 2 MeV and 20 MeV (measured 2026-10-15). N-14
  states its MT 600-849 in MF 3 alone, so its MT 203, 204, 205 and 207 come from the MT
  definitions of step 3.

## Diagnostics

The processing report of {py:func}`kika.process` (and of
{py:meth}`kika.endf.EndfProcessing.report_document`) has a check, *Gas production as the
evaluation states it*, under the files written, when the evaluation states MT 203-207 and
an HDF5 or ACE table was written. For each MT the evaluation states, it gives the largest
difference between the evaluation's total and the sum that replaces it, where it is, and
its share of the stated peak (a point where an energy is given twice, a step, is not
compared):

- **passed**: every share is $10^{-3}$ or less, the tolerance of the processing;
- **note**: the largest is between $10^{-3}$ and $10^{-2}$, the evaluation's rounding;
- **warning**: one is above $10^{-2}$. The evaluation's total holds particles its
  reactions do not emit, or misses some they do. Read where: below the thresholds of
  every reaction it is particles the file states only in the total (ENDF/B-VIII.1 Li-7's
  capture alphas, 4.6 b at $10^{-5}$ eV); above the range of the total, reactions the total
  does not cover (JENDL-5 F-19 above 20 MeV); in between, the total and its partials
  interpolated by different laws (Li-7's MT 204 against its MT 104, 1 mb near 9.4 MeV).
  The files carry the sum, as NJOY's do, and a gas-production tally made with them
  differs from the evaluation's total there.

An evaluation that does not state MT 203-207 has no such check. The same comparison is in
the {py:class}`kika.nuclear_data.model.conversion.ConversionReport` of
{py:func}`kika.hdf5.processing.processSuite` (`suite.report.warnings`, one entry per MT),
and as numbers in `suite.gasReplaced`.

## Implementation

- `kika.hdf5.processing`: `EMITTED_BY_MT` and `GAS_SERIES`, the particles of each MT;
  `_emitted` and `_impliedGas`, the count by the MT with the residual;
  {py:func}`kika.hdf5.processing.gasMultiplicities`, the count of one reaction (stated
  products, or the MT); `_gas`, the sum at each temperature; `_replacedGas`, the
  comparison with a stated MT 203-207 (`suite.gasReplaced` and the report entry).
- `kika.endf.processing_report`: the *Gas production as the evaluation states it* check,
  from the comparison the HDF5 and ACE write stages carry.
- `kika.endf.model_adapter.derive.reactions.LR_PRODUCTS`: the break-up of each LR.
- `kika.ace.model_adapter.dosimetry`: the same count for the dosimetry ACE table.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: §3 (MF 3 and its LR flag), Appendix B (the MT definitions,
  MT 203-207).
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: GASPR.
