# Perturbing the fission neutron multiplicities

```{admonition} At a glance
:class: note
**Computes:** a sample of the average numbers of neutrons per fission $\bar\nu(E)$
(total, delayed and prompt), given one factor per MF 31 energy bin and member, such that
$\bar\nu_\text{total} = \bar\nu_d + \bar\nu_p$ holds exactly on every realisation.

**Applies to:** MF 1 MT 452, 455 and 456, tabulated as $\bar\nu(E)$ (LNU=2) or stated as a
polynomial (LNU=1, tabulated first), with an MF 31 covariance; the model nodes the ENDF adapter maps them to (the fission product's
`multiplicity` for MT 456, the `multiplicitySum` nodes for MT 452 and MT 455).

**Entry points:** {py:func}`kika.sampling.model_perturbation.perturbFromModel`;
`kika.sampling.mf31_sampling.perturbNubarFamilyOnModel` for the family rule;
`kika.nuclear_data.model.perturbation.applyNubarFactors` for one member.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

MF 1 of an ENDF (Evaluated Nuclear Data File) evaluation of a fissile nuclide gives the
average number of neutrons emitted per fission as a function of the incident energy $E$:
the total $\bar\nu(E)$ in MT 452, the delayed part $\bar\nu_d(E)$ in MT 455 and the
prompt part $\bar\nu_p(E)$ in MT 456 (ENDF-6 Ch. 1). Each is either a polynomial in $E$
(LNU=1) or a table with interpolation laws (LNU=2). When MT 455 is present the three are
not independent: the total is the sum of its parts,

$$
\bar\nu_{452}(E) = \bar\nu_{455}(E) + \bar\nu_{456}(E).
$$ (eq-multiplicity-perturbation-sum)

MF 31 states the covariance of these multiplicities in exactly the records MF 33 uses for
cross-sections (ENDF-6 Ch. 31): mostly relative, on energy bins that need not be the MF 1
grid, with cross-blocks between the members where the evaluator correlated them. How the
blocks are assembled and drawn, and how a draw becomes one factor $f_k$ per bin $k$ with
$\mathbb{E}[f_k] = 1$, is {doc}`../sampling/perturbation-pipeline`. This page starts from
the factors.

## What it computes

Input: the evaluated tables $\bar\nu_m(E)$ for the members $m \in \{452, 455, 456\}$ the
evaluation carries, the MF 31 bin edges $g_0 < g_1 < \dots < g_K$ (one grid shared by the
whole family), and a factor vector $f^{(m)}_k$, $k = 0, \dots, K-1$, for each member that
has an MF 31 block. Output: one perturbed table per member that changed, each a single
lin-lin region, where every non-derived member is

$$
\bar\nu'_m(E_i) = c_m(E_i)\, \bar\nu_m(E_i)
$$ (eq-multiplicity-perturbation-result)

at the nodes $E_i$ of its refined table (step 1 below), with $c_m(E)$ the factor of the
bin containing $E$, and the derived member (MT 452) is rebuilt as the sum of the perturbed
components, {eq}`eq-multiplicity-perturbation-sum`. Between nodes the perturbed tables
are interpolated lin-lin.

Which factor vector each member takes depends on what MF 31 covers:

| MF 31 blocks for | MT 455 and MT 456 take | MT 452 |
|---|---|---|
| MT 455 and/or MT 456 | their own block; a component without one keeps factor 1 | derived from the components |
| MT 452 only | the block of MT 452 | derived, so all three scale together |
| all three | their own blocks | derived; its own block is discarded |

## Method

**0. Tabulate a polynomial.** A member stated as a polynomial in $E$ (LNU=1) cannot take a
factor per bin on its coefficients, so it is first evaluated on a table: each MF 31 bin
divided into eight geometrically equal intervals, clipped to the polynomial's own range,
whose two ends are kept. The member is perturbed as that table and written as one
(LNU=2).

**1. Refine the table where the factor steps.** The factor is piecewise constant in $E$,
but a member of the family cannot carry a repeated energy (see *Conventions and choices*),
so the step is approximated. For every interior edge $g_k$, $0 < k < K$, with
$E_\text{first} < g_k \le E_\text{last}$ (the table's own range), two energies are
candidates for insertion:

$$
g_k \;(\text{if } g_k < E_\text{last}), \qquad g_k\,(1 - s), \quad s = 10^{-3}
\;(\text{if } g_k(1-s) > E_\text{first}).
$$ (eq-multiplicity-perturbation-nodes)

The outer edges are steps too where the table runs past them, since outside
$[g_0, g_K]$ the factor is 1: if $E_\text{first} < g_0 < E_\text{last}$, the candidates are
$g_0$ and the shoulder $g_0(1-s)$ below it; if $E_\text{first} < g_K < E_\text{last}$, they
are $g_K$ and the shoulder $g_K(1+s)$ above it (the node at $g_K$ takes the last bin's
factor, step 2). A table that ends on an outer edge has no step there and gains nothing.
A candidate within a relative
$10^{-6}$ of a node already in the table is dropped: ENDF's eleven-character field would
write the two as one energy. Every inserted node is valued by lin-lin interpolation on the
**original** table, so the refined table is the same function as the evaluated one.

**2. The factor of each node.** A node takes the factor of the bin that contains it,
$c(E) = f_k$ for $g_k \le E < g_{k+1}$, so a node at an interior edge takes the factor of
the bin above it and the shoulder $g_k(1-s)$ that of the bin below. A node at exactly
$g_K$ takes $f_{K-1}$ (the last bin is closed at the top). A node outside $[g_0, g_K]$
takes 1. The ramp between two neighbouring factors is therefore confined to
$[g_k(1-s), g_k]$, inside the lower bin, and at the outer edges to $[g_0(1-s), g_0]$ and
$[g_K, g_K(1+s)]$, outside the grid.

**3. Scale.** Each node value is multiplied by its factor,
{eq}`eq-multiplicity-perturbation-result`, and the result is written as one lin-lin
region, whatever regions and laws the evaluated table had.

**4. Derive the total.** With the components perturbed, MT 452 is rebuilt on the union
$U$ of the components' refined grids:

$$
\bar\nu'_{452}(E) = \tilde\nu'_{455}(E) + \tilde\nu'_{456}(E), \qquad E \in U,
$$ (eq-multiplicity-perturbation-derived)

where $\tilde\nu'_m$ is the lin-lin interpolant of member $m$, held constant at its first
(last) value below (above) its own table. The derived table replaces the evaluated MT 452
grid with $U$. Because no member carries a repeated energy, a sum of lin-lin interpolants
on the union of their nodes is itself lin-lin on that union, so
{eq}`eq-multiplicity-perturbation-sum` holds at every energy, not only at the nodes.

**5. Incomplete family.** The total is derived only when all three members are present.
Otherwise (MT 452 alone, or a family missing MT 455 or MT 456) each member that has a
block is perturbed on its own by steps 1-3 and no sum is enforced.

### Derivation

*How far the realised bin factor departs from the drawn one.* Inside the sliver
$[g_k(1-s), g_k]$ the perturbed table runs lin-lin from $f_{k-1}\bar\nu$ to $f_k\bar\nu$,
so the local factor there is off $f_{k-1}$ by at most $|f_k - f_{k-1}|$ and on average by
about half of it. Averaged over bin $k-1$ with the $1/E$ weight MF 31 bins are defined
with, the sliver holds a fraction $-\ln(1-s)/L_{k-1} \approx s/L_{k-1}$ of the bin, where
$L_{k-1} = \ln(g_k/g_{k-1})$ is the bin's width in lethargy. The realised bin factor is
then off by

$$
|\bar c_{k-1} - f_{k-1}| \approx \frac{s}{2 L_{k-1}}\, |f_k - f_{k-1}|.
$$ (eq-multiplicity-perturbation-ramp)

Inserting the edges alone ($s = 0$, no shoulder) would leave the ramp spanning everything
between the last node of a bin and the edge, which on a sparse table is the whole bin.

## Conventions and choices

- **A step is a shoulder node, not a repeated energy.** A cross-section states a step as a
  repeated energy ({doc}`./cross-section-perturbation`). A $\bar\nu$ table may not, for
  two reasons: {eq}`eq-multiplicity-perturbation-sum` is exact as a piecewise-linear
  identity only if no member has a repeated energy, and NJOY (the processing code) reads
  MF 1 into fixed-size buffers, where a shoulder node per edge costs less than subdividing
  every bin.
- **The shoulder is a fraction of the energy**, $g_k(1-s)$. MF 31 bins span decades, so
  $sE$ is always a small part of a bin. (The fission-spectrum applier measures its
  shoulder against the group width instead, for a reason that does not apply here; see
  {doc}`./fission-spectrum-perturbation`.) $s = 10^{-3}$ stays three orders of magnitude
  clear of the $10^{-6}$ relative resolution of an ENDF float, so the shoulder is never
  written as a repeat of the edge.
- **The sum rule is enforced, not checked.** The total is rebuilt from its components and
  its own block discarded. Perturbing all three independently would write a tape whose
  total is not the sum of its parts.
- **The total is the derived member.** `perturbNubarFamilyOnModel` accepts another member
  as the derived one (`derived_mt`); {py:func}`~kika.sampling.model_perturbation.perturbFromModel`
  always derives MT 452.
- **A component with no block of its own rides the total's.** When only MT 452 carries
  covariance, scaling only the total and then rebuilding it from unchanged components would
  undo the perturbation; scaling the components by the total's factor keeps every member
  consistent.
- **Rebuilding the total repairs the evaluation's own residual.** Evaluations satisfy
  {eq}`eq-multiplicity-perturbation-sum` only approximately (a coarse delayed table, for
  instance, does not reproduce the $\bar\nu_d$ the total was built from). The rebuilt total
  is the sum of the components, so it moves by that residual even where every factor is 1.
  The size is measured on the evaluated family and reported (`baseline_residual`, see
  *Diagnostics*); it is not corrected away, since there is no other consistent choice.
- **The perturbed tables are lin-lin, in one region.** The sum rule is stated on lin-lin
  interpolants and the MF 1 writer takes a single region. A table that had several regions
  or another law is flattened, and `regions_collapsed` says so.
- **Constant extrapolation in the derived sum.** Where one component stops short of the
  other (a delayed table to 20 MeV beside a prompt one to 30 MeV), the shorter one is held
  at its last value over the rest of the union grid. The total then stays continuous at
  the end of the shorter table. `TODO(check)`: whether a delayed $\bar\nu$ held constant
  above its table is what the evaluator intends; the file states nothing there.

## Limits and failure modes

Refused (`ValueError`):

- a request for a member the suite does not carry, or a factor block for a member with no
  $\bar\nu$ node;
- a family with no $\bar\nu$ node at all;
- blocks of one family stated on different grids (the family is perturbed on one grid,
  which MF 31 guarantees when it is assembled on the global union, as the pipeline does);
- a factor vector whose length is not the number of bins, fewer than two bin edges, or
  bin edges that are not strictly ascending (a bin lookup on a descending grid would put
  every factor on the wrong bin);
- a derived member whose contributor carries no table (a polynomial component with no
  block of its own): skipping it would drop that component from the total;
- a multiplicity perturbation without the resolver that maps an MT to its model node
  (only when `PerturbationSet.applyToSuite` is called directly; `perturbFromModel` always
  passes one).

Not supported:

- **absolute MF 31 blocks (LB=0).** The model decode keeps them absolute and the default
  request selects relative sections only, so on the model path an absolute block is not
  drawn; the run names it in its notes (see {doc}`../sampling/perturbation-pipeline`). The
  file-level sampler converts it with bin-averaged $\bar\nu$.

Limits of the method:

- the factor is a step only to within the shoulder ramp,
  {eq}`eq-multiplicity-perturbation-ramp`;
- the step at a coverage edge is a shoulder ramp of the same width, outside the grid;
- a polynomial member is perturbed on its eight-point-per-bin table, so its perturbed form
  is the lin-lin interpolant of the polynomial there, not the polynomial itself;
- MT 452 is written on the union of the components' grids, not on its own;
- a realisation of a covariance stated on coarse bins perturbs $\bar\nu$ by a step function.

## Validation

The tests below passed on 2026-10-11 unless marked.

**Against the file-level applier.** On the committed `micro_u235_nubar.endf` (the MF 1 and
MF 31 of a real U-235 evaluation, 96, 5 and 95 points for MT 452, 455 and 456), the model
decode carries exactly the section's arrays
(`kika/nuclear_data/model/tests/test_applying_a_perturbation_to_a_multiplicity.py::test_the_model_carries_the_same_table_the_section_does`),
and `applyNubarFactors` reproduces `apply_factors_to_mf1_nubar` bit for bit, energies,
values and diagnostics, for each member
(`::test_the_two_appliers_produce_the_same_table`); the shoulder is the same constant on
both sides (`::test_the_shoulder_is_the_same_number_on_both_sides`). The family rule on
the model reproduces the one on MF 1 sections point for point
(`kika/sampling/tests/test_a_mixed_request_groups_what_the_file_correlates.py::test_the_family_rule_on_the_model_reproduces_the_one_on_the_sections`).

**The refinement.** No repeated or descending energy is inserted
(`kika/nuclear_data/model/tests/test_applying_a_perturbation_to_a_multiplicity.py::test_the_refinement_inserts_no_duplicate_energies`);
a node and a shoulder go in at every interior edge, on the original interpolant
(`::test_a_node_and_a_shoulder_go_in_at_every_interior_edge`); the last point of a table
on the last edge takes the last bin's factor
(`::test_the_last_point_of_a_table_takes_the_last_bins_factor`); a table that runs past
the grid steps back to the evaluation at the outer edge, and the file-level applier inserts
the same nodes (`::test_a_table_past_the_grid_steps_back_to_baseline_at_the_last_edge`,
`::test_the_two_appliers_agree_past_the_grid`); a polynomial member is tabulated and scaled
(`::test_a_polynomial_nubar_is_tabulated_and_scaled`); the perturbed node stays
writable as one lin-lin region (`::test_a_perturbed_multiplicity_is_still_writable`). On
a six-point table across 15 bins with factors alternating $\pm 5\,\%$, the realised
bin-averaged factor is within 0.01 of the drawn one
(`kika/sampling/tests/test_mf31_sampling.py::test_sparse_table_realises_the_block_it_was_given`).
The same bound on three real tapes is
`::test_realised_factors_track_the_block_on_real_tapes`; `TODO(check)`: needs external
tapes, not re-run.

**The family rule** (synthetic consistent family, in `kika/sampling/tests/test_mf31_sampling.py`).
Sum rule to $10^{-12}$ in every coverage pattern:
`::test_family_prompt_only_derives_total_and_keeps_delayed`,
`::test_family_total_only_scales_all_three_together` (all three scale by 1.04 in the
bin),
`::test_family_all_three_independent_components_derive_total` (the total's own block is
ignored), `::test_family_derived_component_via_override`. On the model, a component with
no block rides the total's, and the sum holds to $10^{-14}$
(`kika/sampling/tests/test_a_mixed_request_groups_what_the_file_correlates.py::test_a_component_with_no_block_of_its_own_rides_the_total`).
Refusals: `::test_incomplete_family_does_not_overwrite_the_total`,
`::test_unperturbable_contributor_raises_instead_of_vanishing`,
`::test_no_nubar_sections_raises`, `::test_factor_block_without_a_section_raises`. The
residual of a consistent family is below $10^{-12}$ and bins one member does not span are
not scored (`::test_sum_rule_residual_is_zero_for_a_consistent_family`,
`::test_sum_rule_residual_ignores_bins_the_family_does_not_span`).

**End to end** (`kika/sampling/tests/test_perturbing_from_the_model_end_to_end.py`, on
`micro_u235_nubar.endf`, request `{31: [455, 456]}`). The written delta tape satisfies
{eq}`eq-multiplicity-perturbation-sum` to $5\cdot10^{-6}$ relative at 25 energies, what
ENDF's digits leave (`::test_the_written_nubar_tape_satisfies_the_sum_rule`); MT 452 is
rewritten although the request did not name it
(`::test_the_derived_total_is_rewritten_although_it_was_not_requested`); MF 3 and MF 31
are left byte for byte (`::test_a_nubar_run_leaves_the_rest_of_the_tape_alone`); two
samples differ only in MF 1 (`::test_two_nubar_samples_differ_only_in_the_multiplicity`);
the evaluation is back on the suite after each sample
(`::test_the_evaluation_is_back_on_the_suite_after_each_sample`); the GNDS and whole-tape
emitters carry the realisation
(`::test_a_gnds_nubar_realisation_sits_beside_its_evaluation`,
`::test_the_whole_tape_emitter_writes_the_perturbed_nubar`,
`::test_a_gnds_nubar_realisation_comes_back_when_the_file_is_read`).

## Diagnostics

In each sample's `applied` entry of a
{py:class}`~kika.sampling.model_perturbation.RunResult`, per component of MF 31:

- for a member perturbed by factors: `min_factor`, `max_factor` (over the nodes of the
  refined table), `frac_out_of_coverage` (the fraction of its nodes outside
  $[g_0, g_K]$, which kept factor 1), `n_inserted` (edges and shoulders added) and
  `regions_collapsed`. **A `frac_out_of_coverage` well above zero** means a good part of
  the table lies where MF 31 states nothing.
- for the derived member (keyed MT 452 even though the request did not name it):
  `derived_from_sum_rule`, `n_points`, `contributors`, and `baseline_residual`, the
  evaluation's own departure from {eq}`eq-multiplicity-perturbation-sum` before anything
  was perturbed: `max_bin_rel` (the largest $|\bar\nu_{452} - \bar\nu_{455} -
  \bar\nu_{456}|/\bar\nu_{452}$ of the $1/E$-weighted bin averages over the bins all three
  span), `argmax_bin`, `n_bins_scored`, `n_bins_uncovered`. **Compare `max_bin_rel` with
  the MF 31 relative standard deviation**: it is the central-value shift every sample
  carries.

The run's `notes` say once that MT 452 was rebuilt from the sum rule, that its own block
was discarded, and the worst residual repaired.

## Implementation

- `kika.nuclear_data.model.perturbation`: one member. `refineForNubar` (edges and
  shoulders, `NUBAR_STEP_SHOULDER`, `NUBAR_ENERGY_RTOL`), `applyNubarFactors` (the bin
  rule, the top-edge clamp, the single lin-lin region and its diagnostics).
- `kika.sampling.mf31_sampling`: the family. `perturbNubarFamilyOnModel` (which block each
  member takes, the derived sum) and `sumRuleResidualOnModel`; beside them, their MF 1
  twins `perturb_nubar_family`, `apply_factors_to_mf1_nubar` and `sum_rule_residual`,
  which the file-level sampler runs and the model path is gated against.
- `kika.sampling.perturbation_set`: dispatch (`PerturbationSet.applyToSuite`), the check
  that the family shares one grid, and the realisation placed beside the evaluated form.
- `kika.endf.model_adapter.multiplicity`: which model node each MT names (`nubarNode`),
  and the MF 1 encoder the realisation is written through.
- `kika.sampling.model_perturbation`: the run note on the rebuilt total.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual: Data Formats and Procedures for the Evaluated
  Nuclear Data Files ENDF/B-VI, ENDF/B-VII and ENDF/B-VIII*, Brookhaven National
  Laboratory (2023), doi:10.2172/2007538. Cited as ENDF-6: Ch. 1 (MF 1, MT 452, 455, 456),
  Ch. 31 (MF 31).
