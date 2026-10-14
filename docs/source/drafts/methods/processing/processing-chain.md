# Processing an evaluation for transport

```{admonition} At a glance
:class: note
**Computes:** from one neutron evaluation, its cross-sections at 0 K and at each requested
temperature, its unresolved-resonance probability tables, its heating and damage
cross-sections, and the files a transport code reads (PENDF, OpenMC HDF5, GNDS with
heated styles).

**Applies to:** incident-neutron evaluations read from ENDF-6 (one material, MF 1-3 for
the stages, the whole tape for the outputs that need distributions) or from a GNDS
`reactionSuite`.

**Entry points:** {py:func}`kika.endf.process_endf`, {py:func}`kika.gnds.process_gnds`,
{py:func}`kika.endf.inspect_endf_processing`, {py:class}`kika.endf.ProcessingPlan`,
{py:class}`kika.endf.EndfProcessing`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

An evaluation is not what a transport code reads. It states resonance parameters where a
Monte Carlo code needs pointwise cross-sections, it states them at 0 K (or at the
evaluator's temperature) where the calculation needs the temperature of the material,
and it leaves self-shielding in the unresolved range, heating and the sampling tables of
secondary particles to the processing. NJOY (MacFarlane et al. 2017) does this as a chain
of modules, each reading the tape the previous one wrote: RECONR reconstructs the
resonances and linearises every MF 3 section into a PENDF (pointwise ENDF), BROADR
Doppler-broadens it, HEATR adds the heating (MT 301) and damage (MT 444)
cross-sections, PURR adds probability tables for the unresolved range (MF 2/MT 153),
GASPR the gas production, THERMR the thermal scattering, and ACER writes the ACE file.
OpenMC builds its HDF5 library by running that chain and converting the ACE files.

kika has its own implementation of the physics of RECONR, BROADR, PURR and HEATR, and
this page is about how they are chained: in what order, on which data, what each stage
passes to the next, and which files come out. Each stage has its own methods page; the
route through NJOY itself is {doc}`njoy`.

## What it computes

Input: an evaluation and a plan (temperatures $T_1 < \dots < T_n$ in K, the top of the
broadening $E_\text{max}$, the thinning tolerance, the URR and heating switches).
Output: a result holding, for each temperature, one table per reaction, and, when
requested, the probability tables and the heating tables, from which the files are
written on demand.

At a temperature $T$, every reaction $r$ whose cross-section the evaluation states as a
partial (a leaf of its sum graph) is

$$
\sigma_r(E, T) =
\begin{cases}
\sigma^{\mathrm{D}}_r(E, T), & E \le E_\text{max},\\
\sigma_r(E, 0), & E > E_\text{max},
\end{cases}
$$ (eq-processing-chain-heated)

with $\sigma_r(E, 0)$ the 0 K reconstruction ({doc}`resonance-reconstruction`) and
$\sigma^\mathrm{D}_r$ its free-gas Doppler broadening ({doc}`doppler-broadening`). Every
sum (MT 1, 4, 103-107 and the others the evaluation states as sums) is rebuilt from its
leaves on the same grid,

$$
\sigma_s(E, T) = \sum_{r \in s} \sigma_r(E, T),
$$ (eq-processing-chain-sums)

so a sum is exact at every grid point of every temperature. The probability tables and
the heating are computed at the same temperatures, and the heating also at 0 K.

## Method

The stages run in this order; each records a {py:class}`~kika.endf.process.StageReport`
under the name the progress callback receives.

**1. `read` and `decode`.** From ENDF-6, MF 1 to MF 3 are read into the canonical model;
from GNDS, the whole `reactionSuite`, with every processed style it carries
(`crossSectionReconstructed`, `heated`, …) and their forms set aside, so that every later
table is kika's own, with kika's tolerance and verification. The target's mass ratio and
spin come from the R-matrix elastic pair when there is one, otherwise from MF 2 (AWRI) or
MF 1 (AWR) of an ENDF-6 tape, and from PoPs for GNDS.

**2. `reconstruct`** (the work of RECONR). The resonance contribution is reconstructed and
every section linearised at 0 K, as {doc}`resonance-reconstruction` describes, with its
relative tolerance of $10^{-3}$. This stage raises on failure: without the 0 K
cross-sections nothing later has a source.

**3. `broaden:<T>`** (the work of BROADR), one per temperature, run in parallel processes.
Each leaf is broadened from 0 K to $T$ below $E_\text{max}$ ({doc}`doppler-broadening`)
and assembled by {eq}`eq-processing-chain-heated`:

- **The grid.** The temperature has one grid: the union of the broadened grids at or below
  $E_\text{max}$ and of the 0 K grids above it. A step (a repeated energy) is moved up by
  about $10^{-12}$ relative for the union, so both of its sides survive, and is written
  back as a repeated energy.
- **Thresholds.** Each reaction starts at the first energy of its 0 K table. The leaves
  are set to zero below that point before the sums are built.
- **Thinning.** The grid is then thinned once for all reactions together (BROADR's
  `errthn`): a point is dropped only if every reaction, read lin-lin between the points
  kept, still reproduces its value at that point,

  $$
  \lvert \tilde y_r(E_j) - y_r(E_j) \rvert \le \varepsilon \lvert y_r(E_j) \rvert + a_r,
  \qquad a_r = 10^{-10} \max_j \lvert y_r(E_j) \rvert ,
  $$ (eq-processing-chain-thin)

  with $\varepsilon = 10^{-3}$ by default (`thinning=None` keeps every point). The values
  at the points kept are not changed, the first point of every reaction is kept, and the
  sweep is greedy from the low end ({py:func}`kika.algebra.thin_common_grid`). The
  floor $a_r$ keeps a reaction that is zero over a stretch from forcing every point of it.
- **$E_\text{max}$.** By default (`thnmax="njoy"`) it is where NJOY BROADR stops when its
  `thnmax` is left at 0, which is what OpenMC's library generation leaves it at: the top
  of the resolved resonance range; with only an unresolved range, its lower end; with no
  resonance parameters, the first reaction threshold; never above 6.5 MeV
  ({py:func}`kika.hdf5.processing.njoyThnmax`). A number in eV replaces it.

**4. `urr`** (the work of PURR). When the evaluation has an unresolved range, its
probability tables are computed at each temperature of the plan, with 20 bands and 64
ladders by default ({doc}`probability-tables`). They are computed from the unresolved
parameters, not from the broadened MF 3, so `broaden=False` with temperatures still gives
them.

**5. `heating`** (the work of HEATR, only when `heating=True`). MT 301 (heating, eV·b)
and MT 444 (damage energy production, eV·b) at 0 K and at every broadened temperature,
on that temperature's MT 1 grid, as HEATR computes them with `local = 0`
({doc}`heating`). The method needs the secondary distributions, which are tabulated
for this stage from the whole evaluation ({doc}`tabulation`); that makes it the slowest
stage, and the reason it is off by default.

**6. Outputs**, each written on request and each adding its own `write` stage:

- **PENDF** ({py:meth}`kika.endf.EndfProcessing.write_pendf`): one material per
  temperature, in increasing temperature, 0 K first, as NJOY BROADR leaves a tape. Each
  material repeats the MAT, states its temperature in MF 1/MT 451 (TEMP) with LRP = 2,
  keeps MF 2 and carries the MF 3 of its temperature; where the URR stage produced
  tables at that temperature it carries them in MF 2/MT 153, and when the heating stage
  ran, MF 3/MT 301 and MT 444.
- **OpenMC HDF5** ({py:meth}`kika.endf.EndfProcessing.write_hdf5`): every broadened
  temperature, with the URR tables, the tabulated distributions, KERMA (MT 301, 901),
  damage (MT 444) and gas production (MT 203-207), as
  {py:func}`kika.hdf5.processing.processSuite` builds them ({doc}`../formats/hdf5`). The
  broadening and the URR tables of the run are reused, not recomputed.
- **GNDS** ({py:meth}`kika.endf.EndfProcessing.write_gnds`): the evaluation (`eval`), the
  0 K reconstruction (`recon`) and one `heated` style per temperature (`heated_000`,
  `heated_001`, … in increasing temperature), as FUDGE's processing writes a processed
  evaluation.

Every output is written to a temporary file, read back and checked, and only then moved
over the destination; an existing file is never left half-written.

## Conventions and choices

- **Stages after the reconstruction never raise.** A stage the material does not admit is
  `rejected`, one that does not apply is `skipped`, each with a category and a sentence,
  and every earlier result is kept. A failed heating stage does not cost the user the
  broadened cross-sections.
- **The default $E_\text{max}$ is BROADR's default.** The official OpenMC libraries were
  broadened to it, so a file kika writes with the default is broadened over the same range
  as the file a user compares it with. Above $E_\text{max}$ the 0 K values are used, as
  BROADR leaves them: the resonances there are unresolved or absent, and the broadening
  would change the smooth cross-sections by less than the tolerance.
- **Thinning as BROADR's `errthn`, measured instead of estimated.** The criterion
  {eq}`eq-processing-chain-thin` is tested at every dropped point of every reaction, so
  the thinned table reproduces the unthinned one to $\varepsilon$ by construction. It is
  a different route to the same kind of table, not a claim of a smaller one.
- **Each reaction starts at its threshold.** A zero on the union grid below a reaction's
  first energy would put its first point under the kinematic threshold, which ACER flags;
  BROADR keeps the threshold the same way.
- **Sums from their leaves, never broadened on their own.** Broadening a sum and its
  partials separately does not keep the sum rule at a finite tolerance;
  {eq}`eq-processing-chain-sums` keeps it exactly.
- **URR tables from the parameters, at the plan's temperatures.** PURR also reads only
  the unresolved parameters and the smooth cross-sections; the tables of different
  temperatures share their ladders, so they are correlated as PURR's are
  ({doc}`probability-tables`).
- **MT 301 differs between the PENDF and the HDF5, on purpose.** The PENDF carries HEATR's
  definition (fission as $\sigma_f\,(E_\text{FR} + E_\text{GP})$ from MF 1/MT 458 less its
  prompt photons), because the readers of a PENDF expect HEATR's numbers; the HDF5
  carries OpenMC's ($\sigma_f\,(E_\text{FR} + E_\text{B})$), because OpenMC reads it
  ({doc}`heating`).
- **MT 153 is written as PURR writes it without HEATR**: band values as factors on the
  smooth cross-sections (LSSF = 1) and a zero heating row.
- **A GNDS file's own processed styles are ignored.** kika reconstructs from the evaluated
  resonance parameters, so a GNDS and an ENDF-6 copy of the same evaluation go through the
  same arithmetic. A file with no resonance parameters keeps its evaluated pointwise
  cross-sections.
- **The GNDS output leaves out the URR tables and the covariances.** GNDS 2.0, the
  version kika writes and validates, has no element for probability tables; the `write`
  stage lists both under `omitted`.
- **The number of worker processes changes nothing in the result.** Called from a daemon
  process, which may not start children, the run uses one worker.

What NJOY does in its chain and this one does not: an ACE file (ACER: the HDF5 is the
Monte Carlo output here; an ACE file is written through {doc}`njoy`), multigroup
cross-sections and covariances (GROUPR, ERRORR), gas production in the PENDF (GASPR: it
is in the HDF5 only), thermal scattering in the PENDF (THERMR: thermal tables are built
separately, {doc}`thermal-scattering-tables`), and UNRESR's self-shielded cross-sections.

## Limits and failure modes

Raised by the reconstruction, before any later stage (`UnsupportedResonanceError`,
`ReconstructionConvergenceError`, with a category of `kika.processing.REJECTION_CATEGORIES`):
an evaluation whose MF 3 already holds the resonance contribution (LRP = 2), an MF 2 with
more than one isotope, resolved parameters in Adler-Adler or LRF = 5, 6, and a
reconstruction that does not converge within its budget. {py:func}`kika.endf.inspect_endf_processing`
predicts the first three from MF 1 and MF 2 alone, in seconds; a run can still be rejected
on its budgets or verification.

Rejected or skipped stages (category, then what it means):

- `broaden`: `doppler-unsupported-target` (bound target or mixture),
  `doppler-incomplete-coverage` (sums not fully described by their partials),
  `doppler-domain-not-covered` (the 0 K tables do not span the evaluated range),
  `doppler-budget-exhausted`, `doppler-verification-failed`, `doppler-unsupported`,
  `doppler-failed`.
- `urr`: `urr-absent` (no unresolved range; skipped), `urr-no-temperature` (none
  requested; skipped), `urr-unsupported`, `urr-failed`.
- `heating`: `heating-unsupported` (a fission without MF 1/MT 458), `heating-failed`; and
  skipped when the broadening was rejected, since heating needs every requested
  temperature.

Outputs raise {py:class}`~kika.endf.process.ProcessingOutputError` with a category:
`hdf5-needs-temperature` (nothing was broadened), `hdf5-failed`,
`pendf-temperature-not-processed` (a temperature asked for in the PENDF was not
broadened), `pendf-failed` (including a material that does not read back),
`pendf-needs-endf-source` (a PENDF from a GNDS evaluation: the MF 2 of a GNDS
evaluation cannot be written back to ENDF-6 without bookkeeping GNDS does not carry),
`gnds-failed`.

Not done: an ACE file, multigroup data, and incident particles other than the neutron.

## Validation

The tests below run on a Fe-56 micro-tape (MF 1-3, 320 Reich-Moore resonances up to
850 keV; `kika/endf/tests/data/micro_fe56_structural.endf`) at 293.6 K and 900 K unless
marked, in `kika/endf/tests/test_process_endf.py`:

- every stage is reported in order (`::test_every_stage_is_reported_in_order`), with one
  progress name per temperature (`::test_progress_names_each_temperature`);
- every broadened table is strictly increasing in energy and MT 1 equals MT 2 + MT 102 to
  $10^{-12}$ relative at every point (`::test_heated_tables_keep_the_reactions_and_the_sum_rule`);
- above $E_\text{max}$ the values are the 0 K ones to $2\cdot10^{-3}$ relative, and below
  it the resonance peaks come down (`::test_broadening_stops_at_thnmax`);
- a Doppler rejection keeps the 0 K result (`::test_a_doppler_rejection_keeps_the_0K_result`);
  0 K only (`::test_zero_kelvin_only`); a non-positive temperature is refused
  (`::test_plan_rejects_non_positive_temperatures`); the inspection reads MF 1 and MF 2 only
  (`::test_inspection_reads_mf1_mf2_only`); every category has its sentence
  (`::test_every_category_has_a_sentence`);
- URR tables on a whole tape, Cf-252 of ENDF/B-VIII.1 (`::test_urr_tables_on_a_whole_tape`),
  and without broadening (`::test_urr_tables_without_broadening`);
- the PENDF has one material per temperature (`::test_pendf_has_one_material_per_temperature`),
  refuses a temperature that was not processed
  (`::test_pendf_refuses_an_unprocessed_temperature`) and carries MT 153 at each
  temperature (`::test_pendf_carries_mt153_at_each_temperature`);
- the HDF5 written from a run equals, record for record, the one the HDF5 writer produces
  from the tape alone (`::test_hdf5_equals_the_hdf5_writer_from_scratch`), and needs a
  broadened temperature (`::test_hdf5_needs_a_broadened_temperature`);
- the GNDS output has one `heated` style per temperature
  (`::test_gnds_has_a_heated_style_per_temperature`); a GNDS evaluation runs every stage
  (`::test_process_gnds_runs_every_stage`), writes its outputs
  (`::test_process_gnds_outputs`) and kika reprocesses its own GNDS
  (`::test_process_gnds_reprocesses_kikas_own_gnds`);
- the heating stage gives MT 301 and MT 444 on each MT 1 grid
  (`::test_the_heating_stage_gives_mt301_and_mt444_on_each_mt1_grid`) and the PENDF
  carries them in every material (`::test_the_pendf_carries_the_heating_in_every_material`);
- through both doors, Au-197 of ENDF/B-VIII.1 read as ENDF-6 and as GNDS gives, at
  293.6 K, MT 1, 2 and 102 within $10^{-3}$ relative at the 99.9th percentile of the
  points (`::test_gnds_and_endf_routes_agree_once_broadened`; it needs both files on disk
  and is skipped otherwise).

The thinning is checked on its own: every row is reproduced at every dropped point, the
points named mandatory stay and no value is changed, and a kink is kept
(`kika/algebra/tests/test_thin.py`). The comparisons of each stage with NJOY and FUDGE are
on the stage's page.

All passed on 2026-10-14, except those that need a whole tape or a library on disk (Cf-252
and Au-197 of ENDF/B-VIII.1), which were skipped.

## Diagnostics

`run.stages` is the list of {py:class}`~kika.endf.process.StageReport`, one per stage and
one per output written: `stage`, `status` (`done`, `skipped`, `rejected`), `seconds`,
`category` (set when skipped or rejected for a reason; `None` when simply not requested),
`summary` (one line) and `details`:

- `read`: `mat`, `za`, `library`, `temperature_K`;
- `decode`: `n_reactions`, `context_source` (where mass and spin came from) and, for
  GNDS, `source_reconstruction` when the file's own processed styles were set aside;
- `reconstruct`: `n_points`, `n_resonances`, `formalisms`, `resolved_range_eV`,
  `unresolved_range_eV`, `native_kernel`;
- `broaden`: `temperatures`, `thnmax_eV`, `n_points` per temperature, and on a rejection
  `engine_category` (the Doppler engine's own category) and `error`;
- `urr`: `temperatures`, `n_energies`, `bins`, `ladders`, `range_eV`;
- `heating`: `temperatures`, `mts`, `local`, `n_points`, `notes`;
- `write`: `format`, `path`, `temperatures`, and `omitted` (what the format could not
  carry) or `mt153` and `heating` for a PENDF.

**A `rejected` broadening** means the temperatures are missing from every output; its
`engine_category` says why. **`omitted`** on a `write` stage lists data the file does not
carry. `report_document()` gives the reconstruction report with a "Processing stages"
section, as text, HTML or a dictionary.

## Implementation

- `kika.endf.process`: the plan, the stages, the result and its outputs;
  `process_gnds` and the GNDS reading in the same module.
- `kika.endf.reconstruction`: the reconstruction stage and its report.
- `kika.processing.doppler`: the broadening; `kika.hdf5.processing`: the parallel
  broadening of every reaction, the default $E_\text{max}$, the URR tables of a suite and
  the processing for the HDF5 and the heating stage.
- `kika.algebra.thin`: the thinning.
- `kika.processing.resonances.probability_tables`, `kika.processing.heating`: the URR and
  heating stages.
- `kika.endf.writers` and `kika.endf.model_adapter.pendf`: the PENDF materials;
  `kika.hdf5.writer`: the HDF5; `kika.gnds`: the GNDS.

## References

- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: RECONR, BROADR, HEATR, PURR, GASPR, THERMR, ACER.
- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: MF 1/MT 451 (TEMP, LRP), MF 2/MT 153, MF 3.
