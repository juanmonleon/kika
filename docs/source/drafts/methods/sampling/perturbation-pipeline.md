# Sampling an evaluation from its covariances

```{admonition} At a glance
:class: note
**Computes:** $N$ realisations of an evaluation drawn from the covariances it states
(cross-sections, angular distributions, multiplicities and the prompt fission neutron
spectrum, alone or together), each written as an ENDF tape, a GNDS file or an ACE file.

**Applies to:** ENDF tapes and GNDS files with MF 31, 33, 34 or 35 covariances (GNDS
covariance sections of the same quantities).

**Entry points:** {py:func}`kika.sampling.model_perturbation.perturbFromModel`; the
draw alone is `kika.sampling.core.draw_samples`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

An evaluation states its uncertainties as covariance matrices of the quantities it
tabulates (ENDF-6 Ch. 30-35): MF 31 for the neutron multiplicities $\bar\nu$, MF 33
for cross-sections, MF 34 for the Legendre coefficients of angular distributions, MF 35
for the prompt fission neutron spectrum. MF 31, 33 and 34 are mostly **relative**: an
element $\Sigma_{ij}$ is the covariance of $\delta x_i / x_i$ and $\delta x_j / x_j$,
on energy bins that need not be the evaluation's grid. MF 35 is **absolute**: the
covariance of the probabilities of energy groups of the outgoing spectrum, whose rows sum
to zero because the spectrum stays normalised.

Sampling an evaluation means drawing vectors consistent with those matrices and applying
each to the evaluated data. Repeated $N$ times and propagated through a transport
calculation, the spread of the results is the nuclear-data uncertainty of the response
(the Total Monte Carlo approach).

## What it computes

Input: a source evaluation, a **request** naming what to perturb (by covariance file,
`{33: None, 34: [2]}`, or by quantity, `{"crossSection": None}`), the number of samples
$N$ and a seed. Output: for each sample $s = 1, \dots, N$, a set of factors (or, for
MF 35, of absolute deltas) per component and covariance bin,

$$
f^{(s)}_{k} \;\text{with}\; \mathbb{E}\big[f^{(s)}_k\big] = 1,\qquad
\operatorname{Cov}\!\left(f_k, f_l\right) = \Sigma_{kl},
$$ (eq-perturbation-pipeline-target)

applied to the model nodes the components belong to, and written in the formats asked
for. Here $k$ runs over every row of the requested covariance: a (reaction, energy bin)
for a cross-section, a (reaction, Legendre order, energy bin) for an angular
distribution.

## Method

**1. Read.** The source is read once, into the model and its covariance suite (see
{doc}`../../architecture/canonical-model`). The covariance may come from a second tape
(`covarianceSource`), with a warning, and never from a different nuclide. On an ENDF
tape whose resonance region is stated as parameters (LRP=1), the MF 3 there is only a
background; by default (`resonanceRegion="reconstructed"`) NJOY RECONR reconstructs the
region once at 0 K, and the cross-sections are perturbed as reconstructed (LRP=2), so
that a factor in the resonance region acts on the resonance cross-section and not on the
background.

**2. Components.** The request is resolved into components, each keyed as
(ZA, MF, MT, index), where the index is the Legendre order for MF 34, the band for MF 35
and 0 otherwise. A request for a sum the file decomposes (MT 4 with MT 51-91 present)
is a request for the sections that decompose it. A request for a quantity the evaluation
does not state is refused (`onMissing="raise"`), or dropped with a warning
(`onMissing="skip"`) for requests written against many files.

**3. Blocks.** The components are partitioned into matrices that are drawn
independently. With the default `grouping="mf"`, all requested components of one
covariance file form one block, and two files are merged into one block only where the
evaluation states a covariance between them (an MF 34 L=0 section correlated with
MF 33, for example). `grouping="stated"` splits further, into connected components of
the blocks the file states.

**4. Pre-flight.** A relative block drawn in log space is first transformed element by
element into the covariance of $\ln f$ ({eq}`eq-perturbation-pipeline-logcov`); the
pre-flight, the conditioning plan and the decomposition all act on that matrix, the one
the draw samples. A block drawn linearly, and MF 35, are used as stated. Every block is
inspected before drawing
(`kika.cov.conditioning.inspect_blocks`): non-finite entries, negative variances,
correlations outside $[-1, 1]$, negative eigenvalues, sum rules. The recommended
**conditioning plan repairs definiteness and nothing else**: a block with a negative
eigenvalue is projected onto the positive semi-definite cone by clipping the eigenvalues
when $|\lambda_\text{min}|/\lambda_\text{max} < 10^{-2}$, and by Higham's nearest-PSD
projection (keeping the diagonal) above; a block that carries a sum rule is always
clipped, because a congruence destroys $\Sigma\mathbf{1} = 0$. The plan is written to the
run directory. A caller may hand in an edited plan, or `conditioningPlan=False` to draw
every block exactly as stated.

**5. Draw.** Each block $M$ ($n \times n$: $\Sigma^{\ln}$ of
{eq}`eq-perturbation-pipeline-logcov` in log space, $\Sigma$ otherwise) is decomposed by
SVD (or an eigendecomposition; Cholesky is refused, since these matrices are
rank-deficient by construction), $M = V \Lambda V^\top$. A direction with a negative
eigenvalue, which a block drawn without conditioning can carry, gets zero variance: a
negative variance cannot be realised, and drawing it with $|\lambda_i|$ would add variance
in a direction the evaluation does not state. The directions kept are those with
$\lambda_i > \tau\,\lambda_\text{max}$; with the default $\tau$ for relative blocks every
direction is kept, and for MF 35 $\tau = 10^{-10}$. With $V_r, \Lambda_r$ the kept part,
$F = V_r \Lambda_r^{1/2}$, and $z^{(s)}$ a vector of independent standard normals,

$$
y^{(s)} = F\, z^{(s)}.
$$ (eq-perturbation-pipeline-correlated)

The normals come from a scrambled Sobol sequence (the default), a Latin hypercube or a
pseudo-random generator, mapped through the inverse normal CDF; block $i$ of a run is
seeded with $\text{seed} + 1009\, i$, so that two blocks never draw the same $z$. Then:

$$
f^{(s)}_k =
\begin{cases}
\exp\!\big(y^{(s)}_k - \tfrac12 \Sigma^{\ln}_{kk}\big) & \text{log space (default)},\\[2pt]
1 + y^{(s)}_k & \text{linear space},
\end{cases}
\qquad
\delta^{(s)}_k = y^{(s)}_k \;\text{for MF 35}.
$$ (eq-perturbation-pipeline-factors)

The space can be set for the whole run or per quantity (`space={33: "log", 34:
"linear"}`). MF 35 is always drawn linearly, as deltas.

**6. Apply.** Each sample becomes a `PerturbationSet`, and its factors are put on the
model nodes they belong to, as new forms labelled `realization-NNNN` beside the evaluated
`eval` form:

- **cross-sections**: each bin's factor multiplies the cross-section over that bin; a bin
  edge becomes a repeated energy (a step), the lower copy carrying the factor of the bin
  that ends there;
- **stated sums** (`crossSectionSums=True`, the default): a sum is never scaled by its own
  block; it is rebuilt as $S' = S + \sum_p (p' - p)$ over its partials $p$, so the tape
  stays consistent and NJOY, which rebuilds MT 1 and MT 4 from their partials, sees the
  same perturbation. What a sum's own block does is set by `sumBlocks`;
- **angular distributions**: on Legendre coefficients, or on tables of $f(\mu)$ as in
  {doc}`../perturbation/tabulated-angular-perturbation`, in the covariance's frame when it
  is not the distribution's;
- **multiplicities**: on $\bar\nu$, keeping $\bar\nu_\text{total} = \bar\nu_p +
  \bar\nu_d$;
- **the fission spectrum**: the group deltas of MF 35 on the outgoing spectrum.

**7. Write.** Each sample is written by the emitters asked for, then its label is removed
from the suite, so memory does not grow with $N$:

- `endf-delta`: re-encode only the sections the sample changed and splice them into a
  copy of the source tape; everything else is the source, byte for byte;
- `endf-tape`: the whole tape written from the model;
- `gnds`: a GNDS file carrying the evaluated form and the realisation;
- `ace`: NJOY run on the sample's ENDF tape, one ACE file per temperature.

With `nWorkers > 1` the read, the pre-flight and the draw still happen once, in the
calling process, and the samples are applied and written by worker processes; the
factors are the same whatever the number of workers. `dryRun=True` does everything but
write tapes.

### Derivation

*Why the log-space factors reproduce the stated covariance.* If $y \sim
\mathcal{N}(0, \Sigma^{\ln})$, then $\mathbb{E}[e^{y_k}] = e^{\Sigma^{\ln}_{kk}/2}$, so
$f_k = e^{y_k - \Sigma^{\ln}_{kk}/2}$ has $\mathbb{E}[f_k] = 1$, and the covariance of two
log-normal factors with unit mean is

$$
\operatorname{Cov}(f_k, f_l) = e^{\Sigma^{\ln}_{kl}} - 1 .
$$ (eq-perturbation-pipeline-lognormal)

Choosing

$$
\Sigma^{\ln}_{kl} = \ln\!\left(1 + \Sigma_{kl}\right)
$$ (eq-perturbation-pipeline-logcov)

makes it equal to the stated relative covariance $\Sigma_{kl}$ exactly, at any size of the
uncertainty. (Taking $\Sigma$ itself as the covariance of $\ln f$ would give
$e^{\Sigma_{kl}} - 1$, a variance 1.14 times the stated one at a 50 % relative standard
deviation.)

## Conventions and choices

- **The log-space draw reproduces the stated relative covariance exactly**: $\ln f$ is
  drawn from $\ln(1+\Sigma)$ ({eq}`eq-perturbation-pipeline-logcov`), the transformation the
  file-level samplers of PENDF, $\bar\nu$ and ACE files (through
  `kika.sampling.multigroup_draw`) also apply (see
  {doc}`../../user_guide/legacy-samplers`).
- **The pre-flight conditions the matrix that is sampled.** In log space that is
  $\ln(1+\Sigma)$, so a definiteness repair, a threshold or an edited plan acts on it, and
  the covariance of the factors is $e^{M} - 1$ of the repaired $M$.
- **A negative eigenvalue is drawn with zero variance.** With conditioning (the default)
  only numerical residue is left negative; with `conditioningPlan=False` a block stated as
  indefinite is drawn in its non-negative part, and its negative directions count as null
  in the diagnostics.
- **Log space by default** for relative covariances: a cross-section is a positive
  magnitude and a log-normal factor cannot be negative. **Linear per quantity when the
  quantity is signed**: a Legendre coefficient near zero has an uncertainty that
  straddles zero, which a log-normal cannot represent.
- **MF 35 is drawn as absolute deltas, linearly**, whatever `space` says: its covariance
  is absolute and its rows sum to zero, and a linear draw preserves that sum, so every
  perturbed spectrum stays normalised.
- **Blocks follow the file.** Components are drawn together exactly when the evaluation
  states a covariance between them, so the ensemble has the file's correlations, cross
  terms between files included, and nothing more.
- **Repairs are decided before the draw and recorded**, never made inside it. The only
  default repair is the one a draw cannot avoid (definiteness); capping variances,
  rescaling outlier bins or masking rows would change what the evaluation states, and are
  only applied when a plan says so.
- **The rank is not filled in.** Null directions are counted and reported; inventing
  variance in them would claim an uncertainty the evaluator did not state. For MF 35 they
  are dropped from the draw, because decomposition noise in a null direction, divided by
  a group probability of $10^{-17}$, would become a perturbation of order $10^7$.
- **Quasi-random by default.** A scrambled Sobol sequence covers the sample space more
  evenly than pseudo-random draws for the same $N$.
- **Sums are rebuilt from their partials**, never scaled by their own block. NJOY RECONR
  discards a tape's MT 1 and MT 4 and rebuilds them from the partials, so a perturbation
  put on MT 1 alone would never reach the ACE file.
- **The resonance region is perturbed as reconstructed** by default, because a factor on
  the MF 3 background of an LRP=1 tape perturbs almost nothing.
- **The source tape is patched, not rewritten** (`endf-delta`, the default), so that two
  samples differ only where they were perturbed and the sections the model does not
  decode survive.

## Limits and failure modes

Refused:

- an unknown emitter, before anything is drawn; `ace` without its `AceOptions`;
  `endf-delta` without the path of the tape it patches; `nWorkers > 1` without a path
  (`ValueError`);
- Cholesky as the decomposition (`ValueError`);
- a request for a quantity the evaluation does not state, unless `onMissing="skip"`;
  a request where nothing matches, always;
- with `covarianceSource`, a cross-section block for a reaction the evaluation does not
  state at all (no section and no partials, so a factor has nothing to move): a
  `ValueError` under `onMissing="raise"`, a note and the block left out under
  `onMissing="skip"`;
- a covariance taken from a tape of another nuclide.

Reported instead of done: null directions; the repairs the plan made; sums left stale when
`crossSectionSums=False`; selections dropped under `onMissing="skip"`; what an emitter
could not carry; absolute MF 31, 33 or 34 sections of a relative request, which are not
drawn (the appliers multiply, and this route does not convert an absolute section to a
relative one), named in a note; a perturbed angular distribution with $f(\mu) < 0$, a
Legendre series checked at 101 cosines or a table at its nodes, counted and left as
computed unless `angularPositivity="repair"` moves the perturbed orders to the nearest
non-negative distribution (orders above the highest perturbed one and $a_0$ are kept; see
{doc}`../perturbation/legendre-angular-perturbation` and
{doc}`../perturbation/tabulated-angular-perturbation`).

Limits of the method: the realised covariance converges to the stated one (to the
conditioned one where the plan repaired it) as $N \to \infty$; with a few hundred
samples, correlations near zero are poorly resolved. A covariance stated on energy bins
coarser than the evaluation's grid perturbs the cross-section by a step function.

## Validation

The tests below passed on 2026-10-11 unless marked.

- **The draw.** The realised covariance of linear draws reproduces the input
  (`kika/sampling/tests/test_core_sampling.py::test_the_realised_covariance_reproduces_the_input`);
  a component with no stated variance gets no delta
  (`::test_a_component_with_no_stated_variance_gets_no_delta`); a zero-sum covariance
  draws zero-sum deltas (`::test_a_zero_sum_covariance_draws_zero_sum_deltas`); rank
  deficiency is recorded and not repaired (`::test_rank_deficiency_is_recorded_and_not_repaired`);
  block seeds differ by a fixed stride (`::test_seeds_differ_per_block_by_a_fixed_stride`);
  log-space factors reproduce a relative covariance with 50 % standard deviations
  (`::test_log_space_factors_reproduce_the_relative_covariance`), and a matrix already in
  log space is not transformed twice (`::test_a_prepared_log_matrix_is_not_transformed_twice`);
  a direction stated with negative variance is drawn with none
  (`kika/sampling/tests/test_a_negative_direction_is_not_folded.py`).
- **Against the shipped MF 33 sampler.** With the same plan, the draw is bit-identical to
  `generate_samples`
  (`kika/sampling/tests/test_mf33_migration_equivalence.py::test_the_draw_is_bit_identical`),
  and the plan is its repair sequence (`::test_the_plan_is_generate_samples_repair_sequence`).
- **Blocks.** A request for MF 33 (MF 34) assembles the matrix the MF 33 (MF 34) pipeline
  ships, and asking for both changes neither
  (`kika/sampling/tests/test_a_mixed_request_groups_what_the_file_correlates.py::test_an_mf33_request_assembles_the_matrix_the_mf33_pipeline_ships`,
  `::test_an_mf34_request_assembles_the_matrix_the_mf34_pipeline_ships`,
  `::test_asking_for_both_at_once_changes_neither`); a stated cross-file block puts both
  quantities in one draw (`::test_a_stated_cross_file_block_puts_both_quantities_in_one_draw`).
- **End to end.** One realisation covers both quantities, two samples differ only where
  they were perturbed, a reaction the request did not name is untouched, and the GNDS
  file carries the evaluation and the realisation
  (`kika/sampling/tests/test_perturbing_from_the_model_end_to_end.py`, 32 tests). Sums
  hold on every realisation
  (`kika/sampling/tests/test_the_cross_section_sums_hold_on_a_realisation.py`); the
  delta emitter leaves the rest of the tape alone
  (`kika/sampling/tests/test_the_delta_emitter_leaves_the_rest_of_the_tape_alone.py`);
  worker processes write the same samples
  (`kika/sampling/tests/test_samples_are_written_by_worker_processes.py`); an absolute
  section left out of a relative request is named
  (`kika/sampling/tests/test_a_request_in_the_models_vocabulary.py::test_an_absolute_section_left_out_of_a_relative_request_is_named`);
  a reaction only the covariance source states is refused or noted
  (`kika/sampling/tests/test_covariance_can_come_from_another_file.py::test_a_reaction_only_the_covariance_states_is_refused_or_noted`);
  a negative Legendre series is counted, and repaired within the perturbed band on request
  (`kika/sampling/tests/test_a_perturbed_legendre_series_is_checked_for_positivity.py`).
- **Pre-flight.** Conditioning first and drawing gives what repairing inside the draw
  gives, for every PSD method
  (`kika/sampling/tests/test_conditioning_matches_the_sampler.py::test_conditioning_first_draws_what_repairing_inside_the_draw_draws`).
- **Resonance region.** The resonance cross-section moves by its factor and the ACE file
  carries the perturbation in the resolved range
  (`kika/sampling/tests/test_the_resonance_region_is_perturbed_as_reconstructed.py`).
  `TODO(check)`: needs NJOY and `--deep`; not re-run for this page.

## Diagnostics

{py:func}`~kika.sampling.model_perturbation.perturbFromModel` returns a `RunResult`:

- `samples`: per sample, its label, its `PerturbationSet`, the files written and each
  applier's diagnostics (`applied`; for an angular distribution, `legendre` and `tables`
  count the negative series and table nodes and what was repaired).
- `diagnostics`: per block, from the draw: `n`, `rank`, `n_null`, `null_fraction`,
  `min_over_max_eigenvalue`, `realised_covariance_error`, `space`, `seed`. **A large
  `realised_covariance_error` or a negative `min_over_max_eigenvalue`** means the matrix
  was not drawn as stated; read `conditioning`.
- `conditioningPlan`, `conditioning`, `conditioningMode`, `report`: the plan, the steps it
  ran, how it came to be (`auto`, `explicit`, `none`), and the pre-flight findings.
- `sectionFindings`, `attribution`: the structural checks of the covariance sections as
  written, and each pre-flight finding traced to the sections that explain it.
- `notes`: things true of every sample that no file states (a stale sum, a dropped
  selection, an absolute section left out, an assumption `sumBlocks="fill"` made). **Read
  them.**
- `log` and `files`: the run log and the run-level files.

In the run directory: `factors.parquet` (every realisation's factors, one table per run,
with its index in the file's metadata; `readFactorsIndex` reads it), `conditioning_plan.json`, `run.log` and `run.log.jsonl`,
`run_metadata.json`, and one directory per sample with its files.

## Implementation

- `kika.sampling.model_perturbation`: `perturbFromModel`, the stages, the emitters.
- `kika.sampling.joint_blocks`, `kika.sampling.model_blocks`: components, blocks,
  grouping.
- `kika.cov.conditioning`: the pre-flight and the conditioning plan;
  `kika.cov.decomposition`: SVD, eigendecomposition, clip and Higham projections.
- `kika.sampling.core`: `draw_samples` (from the stated relative covariance) and
  `draw_prepared_samples` (from a block already in the space of the draw);
  `kika.sampling.generators`: the Sobol, Latin
  hypercube and pseudo-random normals.
- `kika.sampling.perturbation_set`: a sample's factors, applied to the model, and the
  factors table.
- `kika.nuclear_data.model.perturbation`: the appliers on model nodes;
  `kika.sampling.cross_section_sums`: the sum rules; `kika.sampling.resonance_region`: the
  reconstructed resonance region.
- `kika.sampling.run_log`: the run log.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: Ch. 30-35 (covariances).
- N. J. Higham, "Computing a nearest symmetric positive semidefinite matrix", *Linear
  Algebra and its Applications* 103, 103-118 (1988).
- `TODO(ref)`: a reference for Total Monte Carlo (Koning and Rochman) and for Sobol
  sequences.
