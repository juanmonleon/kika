# Perturbing a cross-section

```{admonition} At a glance
:class: note
**Computes:** a sample of the MF 3 cross-sections of an evaluation, given one factor per
reaction and covariance energy bin drawn from MF 33 (or from the order 0 of MF 34), with
the stated sums of MF 3 kept equal to their partials and the resonance region perturbed as
reconstructed.

**Applies to:** cross-section nodes of the model (`crossSection` of a reaction or a sum),
read from ENDF or GNDS; any MT of MF 3.

**Entry points:** {py:func}`kika.sampling.model_perturbation.perturbFromModel`,
{py:func}`kika.nuclear_data.model.perturbation.applyFactors`,
`kika.nuclear_data.model.perturbation.refineAtBinEdges`,
`kika.sampling.cross_section_sums.planCrossSectionSums`,
`kika.sampling.cross_section_sums.rederiveSum`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

MF 3 of an ENDF (Evaluated Nuclear Data File) evaluation tabulates each reaction's
cross-section $\sigma(E)$ as a TAB1 record: energies $x_1 \le \dots \le x_n$, values, and
interpolation regions, each with one of the ENDF interpolation laws (histogram, lin-lin,
lin-log, log-lin, log-log) (ENDF-6 Ch. 3). A discontinuity is written as a repeated energy,
the first copy carrying the limit from below and the second the limit from above
(ENDF-6 §0.5, `TODO(check)`: section number).

Some MTs are **sums** of others, and the tape states both: MT 1 is MT 2 plus MT 3, MT 3 the
non-elastic channels, MT 4 the levels MT 51-91, MT 103 the levels MT 600-649, and so on
(ENDF-6 Ch. 3, `TODO(check)`: section of the summation table). A tape is consistent only
if each stated sum equals its stated partials.

MF 33 states the covariance of the relative cross-sections, $\delta\sigma_i/\sigma_i$, on
energy bins of its own, coarser than MF 3 as a rule (ENDF-6 §33.2). A section may also say
that a reaction's covariance is a lump of several (a non-zero MTL, MT 851-870) or a
combination of others (an NC-type subsection with LTY=0) (ENDF-6 §33.2). A sample of the
covariance is one factor $c_{rk}$ per reaction $r$ and bin $k = 1, \dots, K$ of edges
$E_0 < E_1 < \dots < E_K$ (how the factors are drawn is in
{doc}`../sampling/perturbation-pipeline`). A section stated as absolute (in b²) is drawn
as the relative covariance it implies, divided by the evaluation's cross-section averaged
over each bin, so it too becomes one factor per bin
({eq}`eq-perturbation-pipeline-relativise`). This page is how kika puts those factors on
the tabulated cross-sections.

On a tape with LRP=1 in MF 1/451, MF 2 gives resonance parameters for the resolved (LRU=1)
and unresolved (LRU=2) ranges, and MF 3 there is only a background added to the
cross-section the parameters reconstruct (ENDF-6 §1.1, Ch. 2). With LRP=2, MF 3 is the
complete cross-section and MF 2 is for information.

## What it computes

Input: the evaluated cross-section of each reaction, one block of factors per perturbed
reaction on its bin edges, the sum structure of the tape, and the sum mode. Output: a new
form beside the evaluated one, labelled with the sample, on every cross-section node the
sample moves. For a reaction perturbed by its own block, at every energy $E$ of the refined
table and between them,

$$
\sigma'(E) = c(E)\,\sigma(E), \qquad
c(E) = \begin{cases} c_{rk} & E_{k-1} \le E < E_k,\\ 1 & \text{outside } [E_0, E_K), \end{cases}
$$ (eq-cross-section-perturbation-result)

where at a bin edge the cross-section steps from the factor of the bin below to the factor
of the bin above, written as a repeated energy. A stated sum $S$ is never scaled by a block
of its own. It becomes

$$
S'(E) = S(E) + \sum_{p \,\in\, \mathcal{P}(S)} \big(p'(E) - p(E)\big),
$$ (eq-cross-section-perturbation-sum)

with $\mathcal{P}(S)$ the leaf partials under $S$ that moved, so that the sum moves by
exactly what its parts moved and stays the evaluation's value wherever they did not.

## Method

**1. Refine at the bin edges.** The factor is piecewise constant, so the table needs a
repeated energy at every bin edge $E_k$ strictly inside its span $(x_1, x_n)$ (to an
absolute $10^{-10}$ eV). Per such edge, counting the energies the table already has there:

- **two or more**: nothing is inserted; the evaluation already steps there (a threshold, most
  often);
- **one**: a second copy with the same value is inserted beside it;
- **none**: two copies are inserted, both with the value the interpolation region containing
  $E_k$ gives under its own law.

An edge on a boundary between two interpolation regions belongs to the region that ends
there, so that the pair stays adjacent in the flattened table. The insertions are made
inside the region that owns the edge, so the $(\mathrm{NBT}, \mathrm{INT})$ pairs follow from
the region lengths and are never recomputed by hand. An edge at or beyond the ends of the
table inserts nothing.

**2. The factor of each tabulated energy.** On the flattened table $x_1, \dots, x_m$,

$$
c_i = \begin{cases}
c_{rk},\; E_{k-1} < x_i \le E_k & \text{if } x_i = x_{i+1} \text{ (first of a pair)},\\
c_{rk},\; E_{k-1} \le x_i < E_k & \text{otherwise},
\end{cases}
\qquad c_i = 1 \text{ when no such } k \text{ exists},
$$ (eq-cross-section-perturbation-bins)

that is, the first copy of a repeated energy takes the factor of the bin that ends there and
every other point the factor of the bin that starts there. Consequences: at the lower edge
$E_0$ of the coverage the table steps from 1 to $c_{r1}$; at the upper edge $E_K$ it steps
from $c_{rK}$ back to 1; a point exactly at $E_K$ that is not the first of a pair (the last
energy of a table that ends there) takes 1; a point exactly at $E_0$ that starts the table
(a threshold on a bin edge) takes $c_{r1}$. An evaluated step at a threshold is kept: both its
copies take the factor their side of the step falls in.

**3. Scale.** Each tabulated value is multiplied by its $c_i$, and every region keeps its
law. Inside one bin the factor is a constant, and each ENDF interpolation law is invariant
under multiplying $y$ by a constant (a line, a line in $\ln E$, a line of $\ln y$ in $E$ or in
$\ln E$, a constant), so {eq}`eq-cross-section-perturbation-result` holds between the nodes as
well as at them. A cross-section that is zero (below a threshold, or a background) stays
zero.

**4. The sum rules.** With `crossSectionSums=True` (the default), what each MT is moved by
is planned before anything is applied (`planCrossSectionSums`). The sums are read off the
tape, not off MF 33: the leaf partials of a sum are the MTs of its rule that MF 3 states,
where a rule entry that is itself a sum is used when the tape states it and replaced by its
own partials when not. The rules form a tree (no MT has two parents; a tape where one would is
refused). Then:

1. A partial with a block of its own is perturbed by it. Partials govern.
2. Every sum with a moved partial under it is re-derived by {eq}`eq-cross-section-perturbation-sum`,
   deepest first (MT 4 before MT 3 before MT 1).
3. A sum the file **decomposes** (some section under it, partial or smaller sum, carries a
   block of its own) is perturbed only through those sections. Its own block is not drawn,
   and the partials under it with no block stay as evaluated.
4. A sum the file does **not decompose** (nothing under it carries a block) moves every leaf
   partial under it by its own factor, all alike, and is then re-derived.

`sumBlocks` selects what a sum's own block may do:

| `sumBlocks` | Decomposed sum | Undecomposed sum |
|---|---|---|
| `"undecomposed"` (default) | block not drawn; uncovered partials as evaluated | block moves every partial under it |
| `"fill"` | block drawn and carried to the partials that have no block of their own | block moves every partial under it |
| `"never"` | block not drawn | refused if the request names the sum; otherwise not drawn, with a warning |

A partial without a block takes the block of the nearest perturbed sum above it, when the
mode allows that sum's block to reach it. The blocks a mode does not use are removed before
the draw, so the draw, the factors table and `run_metadata.json` hold only what reaches the
tape.

Three further statements of MF 33 are honoured in every mode:

- **Lumps.** The members of a lumped reaction (non-zero MTL) take the lump's block as if it
  were theirs, unless they carry one of their own.
- **Remainders.** A partial $q$ whose MF 33 is stated as one total $T$ minus other sections
  $X$ (an NC subsection, LTY=0, coefficients $+1$ on $T$ and $-1$ on the others, over
  $[e_1, e_2]$) takes the difference, {eq}`eq-cross-section-perturbation-remainder` below.
- **Virtual sums.** A covariance for a sum the tape gives only the partials of (MT 103 with
  MF 3 for MT 600-649 alone) reaches those partials. There is no section of its own to
  rebuild.

The remainder is taken with $T$'s own block drawn (the total is *anchored*, even when
the file decomposes it), over the intersection of $[e_1, e_2]$ with the blocks' spans:

$$
q' = q + (T' - T) - \sum_X (X' - X),
$$ (eq-cross-section-perturbation-remainder)

where $T'$ is $T$ under its own factors. Where $q$ is exactly zero it stays zero. The sums
above are then rebuilt from their partials, which returns $T'$ for the total. A combination
of any other shape, or one whose partial is itself a sum in MF 3, is not applied, and the
run says so.

The order is: leaves, then the sums that contain no remainder, then the remainders, then the
sums above them.

**5. Re-derive a sum.** {eq}`eq-cross-section-perturbation-sum` is evaluated, for every
energy of a union grid, as a left and a right limit, because a step has two values. The
union grid holds the sum's own energies and, within the span $[E_0, E_K]$ of each block
that moved a partial, the partial's evaluated and perturbed energies. Off its block a move
is zero by construction, not by rounding: the left limit at $E_0$ and the right limit at $E_K$
carry no move. Outside every span, therefore, each of the sum's values is the evaluation's,
bit for bit. Where the left and right moves differ the sum gets a repeated energy; an
evaluated step of the sum keeps all its copies. Outside its tables a cross-section is zero.

The sum keeps its own interpolation regions and laws, with one exception: a panel that moved
and whose own law is not lin-lin has no law that represents a sum of differently-interpolated
tables. It is made lin-lin, on enough added energies that every form in the sum is lin-lin
there to the linearisation tolerance of $10^{-4}$ (`kika.algebra.to_linlin`). A lin-lin tape
gets no added energies beyond the partials' own.

**6. The resonance region.** When the request moves a cross-section (MF 33, or order 0 of
MF 34), the source is an ENDF tape with LRP=1, and MF 2 states a resolved or unresolved
range, then with `resonanceRegion="reconstructed"` (the default):

1. NJOY (the nuclear data processing code) RECONR reconstructs the tape once, at 0 K, with
   the fractional tolerance `reconstructionTolerance` (0.001 by default). The result is
   cached by the bytes of the tape and the tolerance.
2. A **base tape** is made from the source: its MF 3 replaced by RECONR's, the MF 1/451
   directory rebuilt, LRP set to 2. MF 2 is kept, for information and for the unresolved
   self-shielding.
3. The model is decoded from the base tape, and steps 1 to 5 act on the reconstructed
   cross-sections. Every ENDF a run writes therefore carries the reconstructed MF 3 with
   LRP=2.
4. For ACE, NJOY is given two tapes and RECONR is skipped: RECONR's PENDF with the sample's
   MF 3 spliced in, and the sample's ENDF with the source's own MF 3 and LRP put back, so that
   the unresolved-range processing (PURR) reads the background and the parameters it expects.
   The perturbation is applied at 0 K and Doppler-broadened afterwards.

`resonanceRegion="evaluated"` perturbs MF 3 as the tape states it, background included.

## Conventions and choices

- **A bin edge is a repeated energy, never a ramp.** The factor of a bin holds up to its
  edge, as the covariance states it. Joining two factors by interpolating between the last
  energy below an edge and the first above it would apply, over that panel, a factor no bin
  carries.
- **The inserted value at an edge comes from the region's own law**, not from a straight
  line. On a log-log panel the straight line is a different cross-section. On a lin-lin
  table (every PENDF) the two agree exactly.
- **The covariance's coverage is exact**: outside $[E_0, E_K)$ the factor is 1, and the table
  steps at both ends of the coverage. A covariance says nothing beyond its grid, so the
  evaluation is kept there.
- **The point at $E_K$ belongs to no bin** under {eq}`eq-cross-section-perturbation-bins`, so
  a table that ends exactly at the top of the grid keeps its last value. For a cross-section
  with thousands of points this changes one point; the multiplicity applier, where it does not,
  treats it differently (see {doc}`./multiplicity-perturbation`).
- **Partials govern; a sum follows.** A sum stated beside its partials is redundant, and
  NJOY RECONR discards a tape's MT 1 and MT 4 and rebuilds them from the partials. A factor
  left on the sum alone would never reach an ACE file, and the tape would state a total that
  is not the sum of anything.
- **A sum is re-derived as a delta and not resummed.** Resumming would also remove whatever
  residual the evaluation's own sum has against its partials, everywhere, including where
  nothing was perturbed; on a tape cut down to a few sections it would replace the total by
  the sum of the survivors. The delta moves the sum by what the parts moved and nowhere else.
- **A decomposed sum's block is discarded by default.** Once the file states some partials
  apart, it no longer says how the rest share the sum's uncertainty. `"fill"` assumes they
  share it fully, and every realisation's notes say so.
- **An undecomposed sum's block moves its partials alike.** It is the only reading of that
  covariance that reaches a processed file. `"never"` refuses it instead of guessing.
- **A requested partial with no covariance of its own is refused**, and no other block (a
  sum's included) stands in for it: a covariance perturbs the quantity it was stated for.
  Naming a sum the file decomposes is a request for the sections that decompose it; they are
  added, and the run says so.
- **The resonance region is perturbed as reconstructed.** A factor on an LRP=1 background
  perturbs almost nothing below the resonance boundary. Perturbing the cross-section RECONR
  reconstructs is what perturbing a PENDF does.
- **MF 33 and the order 0 of MF 34 are not combined.** Both state the uncertainty of the same
  $\sigma(E)$; applying both would scale it twice. Which to use is part of the request (see
  {doc}`./legendre-angular-perturbation`).

## Limits and failure modes

Refused:

- a factor block whose length is not the number of bins, or bin edges that are not a
  strictly ascending one-dimensional array of at least two energies (`ValueError`);
- a request with both an MF 33 block and an MF 34 order-0 block for one reaction
  (`ValueError`);
- an additive block on a cross-section under the sum rules (`ValueError`);
- an unknown `sumBlocks`, or a sum tree in which one MT is a partial of two sums
  (`ValueError`);
- under `"never"`, a request naming an undecomposed sum with `onMissing="raise"`, or a request
  every block of which is a sum's this mode does not use (`ValueError`);
- a requested MT that MF 33 gives no covariance for, unless `onMissing="skip"`;
- in a run, an absolute MF 33 section whose reaction has no cross-section to divide by
  (neither in MF 3 nor as a sum of stated partials, or a GNDS source): a `ValueError`, or
  left out with a note under `onMissing="skip"`;
- a perturbed MT the tape does not state, nor any of its partials: in a run, a block for
  such a reaction (one a `covarianceSource` states and the evaluation does not) is refused
  before the draw (`ValueError`), or left out with a note under `onMissing="skip"`; a
  `PerturbationSet` that reaches one when applied directly raises `KeyError`;
- reconstructing the resonance region from a parsed object instead of a path (`ValueError`),
  or without an NJOY executable.

Reported instead of done: the sums re-derived, the partials moved by another MT's block
(and under `"fill"`, that this is an assumption), the blocks discarded or dropped, the lumps
and remainders taken, the MF 33 combinations not applied, a re-derived sum or remainder that
went negative; with `crossSectionSums=False`, every sum left stale. With
`resonanceRegion="evaluated"` on a tape with a resonance region, a warning that those energies
are barely perturbed.

Limits of the method:

- A covariance on bins coarser than the evaluation's grid perturbs $\sigma(E)$ by a step
  function, with a discontinuity at every bin edge.
- A re-derived sum can go negative only when the evaluation states it below its partials.
  Where a partial is itself negative (a background inside the resolved range) a negative sum
  is not counted.
- In the unresolved range, PURR builds the probability tables from the unperturbed MF 2
  parameters, so the self-shielded cross-sections an ACE file carries there do not carry the
  factor. `TODO(check)`: whether tables stated as factors on the smooth cross-section (LSSF=1)
  let it through has not been verified.

## Validation

The tests below passed on 2026-10-11 unless marked.

**Against the file-level applier.** On `micro_fe56_structural.endf` (MT 1, 2 and 102, a
logarithmic grid of 11 bins whose edges fall between tabulated energies, factors
$1 + 0.25\,z$), `applyFactors` gives the same energies and values as
`kika.sampling.mf33_sampling.perturb_pointwise_xs`, exactly, and the encoded MF 3 section is
byte-identical
(`kika/nuclear_data/model/tests/test_applying_a_perturbation_to_a_node.py::test_the_model_applier_reproduces_the_format_applier`,
`::test_the_encoded_section_is_byte_identical`); the NBT of the last region is the number of
points (`::test_the_pairs_come_from_the_regions_and_not_from_arithmetic`).

**The rules of steps 1-3**, value by value in the same file: an edge the table repeats is
left alone, an edge it has once gains a twin
(`::test_an_edge_the_table_already_repeats_is_left_alone`,
`::test_an_edge_the_table_has_once_gains_its_twin`); on a log-log panel from (1, 1) to
(16, 256) the inserted point at 4 is 16, where a straight line gives 52
(`::test_an_edge_the_table_does_not_have_is_interpolated_under_its_own_law`); an edge on a
region boundary keeps the pair adjacent
(`::test_a_region_boundary_edge_keeps_the_pair_adjacent_in_the_flat_table`); factors 2 and 3
on edges 1, 5, 9 give 2, 2, 3, 1 on 1, 5, 5, 9 (`::test_the_step_takes_the_lower_bin_on_its_first_half`);
points outside the grid keep their value (`::test_points_outside_the_grid_keep_their_value`);
a mismatched block is refused (`::test_a_factor_block_that_does_not_match_its_grid_is_refused`),
and so is a descending grid (`::test_a_descending_grid_is_refused`); the first copy of the
pair at the last edge counts as covered (`::test_the_first_copy_at_the_last_edge_counts_as_covered`).

**The sum rules.** In
`kika/sampling/tests/test_the_cross_section_sums_hold_on_a_realisation.py`:

- {eq}`eq-cross-section-perturbation-sum` holds to $10^{-14}$ relative at every point, steps
  included (`::test_the_sum_moves_by_exactly_what_its_partial_moved`); outside the block every
  value of a sum 3 % off its partial comes back as the same float
  (`::test_outside_the_block_the_sum_is_the_evaluations_value_for_value`); a two-region sum
  keeps its regions, laws and own step, and a log-log panel that moved is lin-lin and within
  $10^{-4}$ of the exact sum between nodes (`::test_the_sums_regions_and_laws_survive`).
- The plan: decomposed, undecomposed, `"fill"`, `"never"`, lumps, remainders, virtual sums
  (`::test_a_decomposed_sums_block_reaches_none_of_its_partials`,
  `::test_an_undecomposed_sum_moves_all_its_partials_alike`,
  `::test_fill_rides_the_nearest_sum_as_before`,
  `::test_never_refuses_a_sum_block_with_nowhere_to_go`,
  `::test_a_lump_moves_its_members_as_their_own_block`,
  `::test_a_remainder_anchors_its_total`, `::test_a_sum_stated_only_through_its_partials`).
- A remainder: with MT 2 stated as MT 1 minus MT 102, the rebuilt MT 1 carries MT 1's own
  factors and MT 2 satisfies {eq}`eq-cross-section-perturbation-remainder`, both to
  $10^{-12}$ relative
  (`::test_a_remainder_takes_the_difference_and_the_total_keeps_its_own_move`).
- Through the written tape: on `micro_fe56_xs_and_angular.endf`, the move of MT 1 equals the
  move of MT 2 to $2\cdot10^{-6}$ of MT 1 over 4 000 energies, what seven significant digits
  allow (`::test_the_written_total_moved_by_what_its_partials_moved`); the request-level rules
  in `::test_naming_a_decomposed_sum_names_what_decomposes_it`,
  `::test_naming_an_undecomposed_sum_moves_its_partials`,
  `::test_a_named_partial_without_a_covariance_is_refused`,
  `::test_the_switch_restores_the_old_behaviour`.
- On a whole evaluation (ENDF/B-VIII.1 Fe-56, every MF 33 block, resonance region
  reconstructed), every re-derived sum moved by what its leaves moved to $10^{-5}$ of the
  sum, MT 4 and MT 103 moved MT 51-91 and MT 600-649, and MT 1's block was discarded
  (`::test_every_sum_holds_on_a_whole_evaluation`; slow, needs the tape and NJOY; run with
  `-m slow`). The JENDL-5 Fe-56 remainder case
  (`::test_jendl5_fe56_elastic_is_the_remainder_it_is_stated_as`): `TODO(check)`: not re-run,
  the tape was not reachable.

**The resonance region.** In
`kika/sampling/tests/test_the_resonance_region_is_perturbed_as_reconstructed.py`, on
ENDF/B-VIII.1 Fe-56 with NJOY (run with `-m slow`): the evaluated MT 2 is below
$10^{-2}$ b at 1 keV, 30 keV and 400 keV and the base tape's above 1 b, with LRP=2
(`::test_the_tape_states_a_background_and_the_base_the_cross_section`); inside the
resolved range the written MT 2 over the reconstructed one equals the factor to
$2\cdot10^{-6}$ (`::test_the_resonance_cross_section_moves_by_its_factor`); the sample
differs from the base only in MF 3 MT 1, MT 2 and MF 1/451
(`::test_the_delta_differs_from_its_base_only_where_it_should`); NJOY gets the perturbed
PENDF and the original ENDF (`::test_njoy_gets_the_perturbed_pendf_and_the_original_endf`).
The ACE comparison (`::test_the_ace_carries_the_perturbation_in_the_resolved_range`, which
requires a median deviation of the ACE ratio from the factor below $10^{-5}$ and a 99th
percentile below $10^{-3}$ over more than 1 000 energies below 850 keV): `TODO(check)`: not
re-run.

**Absolute sections.** An absolute MF 33 section is made relative as the file-level sampler
makes it, bit for bit on its grid, and assembles to its relative covariance; bins below
the inert floor are counted, and a section with no cross-section to divide by is refused or
skipped with a note
(`kika/sampling/tests/test_an_absolute_section_is_made_relative.py`, passed 2026-10-11).

## Diagnostics

{py:func}`~kika.nuclear_data.model.perturbation.applyFactors` returns, with the perturbed
node: `min_factor`, `max_factor` (over the tabulated energies), `frac_out_of_coverage` (the
fraction of the refined table's energies no bin covers, which keep factor 1; the first copy
of the pair at $E_K$ carries the last bin's factor and is covered) and `n_inserted`.
**`frac_out_of_coverage` near 1** means the block hardly reaches the reaction.

In a run of {py:func}`~kika.sampling.model_perturbation.perturbFromModel`, each sample's
`applied` entry for a cross-section carries these, and under the sum rules:

- `factor_from`: the MT whose block moved this partial (with `lumped: True` for a lump);
- `remainder_of`: the total this partial took the difference of;
- `rederived_from`: for a rebuilt sum, the leaves it was rebuilt from, with
  `n_inserted`, `max_rel_change`, `max_abs_change` and `n_negative`. `max_rel_change` is
  large wherever the sum is near zero (a background); read `max_abs_change` there.
  **`n_negative` above zero** means the evaluation states that sum below its partials, and
  the realisation is not usable as written;
- `own_block`: for a sum the request named, the leaves its own block reached,
  `"discarded"`, or `"remainder"` (with `anchors`); `virtual: True` for a sum with no MF 3
  section.

The run's `notes` say the same in words (the sums re-derived, the riders, the discarded
blocks, the negative sums). `run_metadata.json` records `crossSectionSums`, `sumBlocks` (its
`mode` and the `discarded`, `undecomposed`, `dropped`, `anchored`, `lumped` and
`remainders` it found) and `resonanceRegion` (`mode`, `reconstructed`, `ranges`, and the
`pendf`, `base` and `tolerance` of a reconstruction). How to read a run is in
{doc}`../../user_guide/running-a-perturbation`.

## Implementation

- `kika.nuclear_data.model.perturbation`: the refinement at the bin edges
  (`refineAtBinEdges`), the bin rule on the flattened table, the scaling and its diagnostics
  ({py:func}`~kika.nuclear_data.model.perturbation.applyFactors`).
- `kika.sampling.cross_section_sums`: the sum tree (`sumTree`, from the rules in
  `kika._constants.MF3_SUM_RULES` resolved against the tape by
  `kika.endf.writers.redundant.resolve_sum_components`), the MF 33 lump and remainder
  statements (`readSumStatements`), the screen of decomposed and undecomposed sums
  (`screenSumClaims`), the plan (`planCrossSectionSums`) and the re-derivation
  (`rederiveSum`).
- `kika.sampling.perturbation_set`: `PerturbationSet.applyToSuite`, which resolves which
  block claims each cross-section and applies the plan.
- `kika.sampling.model_perturbation`: the request expansion for named sums, the removal of
  unused sum blocks before the draw, the resonance-region reconstruction and the two NJOY
  inputs of a reconstructed sample.
- `kika.sampling.resonance_region`: when a tape needs reconstruction
  (`needsReconstruction`), the base tape (`reconstructedBase`), the MF 3 splice (`spliceMF3`)
  and the LRP flag (`setLRP`).

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual: Data Formats and Procedures for the Evaluated
  Nuclear Data Files ENDF/B-VI, ENDF/B-VII and ENDF/B-VIII*, Brookhaven National
  Laboratory (2023), doi:10.2172/2007538. Cited as ENDF-6: §0.5 (interpolation), §1.1
  (MF 1/451, LRP), Ch. 2 (resonance parameters), Ch. 3 (MF 3), §33.2 (MF 33).
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: RECONR and PURR. `TODO(ref)`: chapter numbers, and a citation for
  RECONR's rebuilding of MT 1 and MT 4 from the partials.
