# Perturbing a prompt fission neutron spectrum

```{admonition} At a glance
:class: note
**Computes:** a sample of a tabulated fission spectrum $\chi(E' \mid E)$, given one
absolute delta per outgoing-energy group and incident-energy band drawn from an MF 35
covariance; every sample stays non-negative and keeps the evaluation's normalisation.

**Applies to:** MF 5 energy distributions tabulated as $\chi(E' \mid E)$ (LF=1), with an
MF 35 covariance (LB=7) for the same MT, usually MT 18; the `uncorrelated` distribution
node of the product that carries the table.

**Entry points:** {py:func}`kika.sampling.model_perturbation.perturbFromModel`;
`kika.nuclear_data.model.perturbation.applySpectrumFactors` for the table;
`kika.sampling.mf35_sampling.pfns_ratio_rule` for what a sample may be.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

MF 5 gives the energy distribution of the neutrons a reaction emits as
$\chi(E' \mid E)$, the probability density in outgoing energy $E'$ at incident energy $E$,
normalised so that $\int \chi(E' \mid E)\,dE' = 1$ at every $E$ (ENDF-6 Ch. 5). With LF=1
it is a table: a list of incident energies $E_n$, each with its own outgoing grid, its own
interpolation laws in $E'$, and one law for the incident axis.

MF 35 does not state the covariance of $\chi$ itself. For each **band** of incident energy
$[E_1, E_2]$ it states the absolute covariance $C$ of the group-integrated probabilities

$$
P_j(E) = \int_{g_j}^{g_{j+1}} \chi(E' \mid E)\, dE', \qquad j = 0, \dots, J-1,
$$ (eq-fission-spectrum-perturbation-groups)

on an outgoing group structure $g_0 < \dots < g_J$ of its own (ENDF-6 Ch. 35). Because the
probabilities sum to one, the rows of $C$ sum to zero, $C\mathbf{1} = 0$ up to the
precision of the file. Bands are disjoint in incident energy, and the format has no record
for a covariance between two of them.

A draw of a band is a vector of absolute deltas $\delta_j$ on the $P_j$ (how it is drawn is
{doc}`../sampling/perturbation-pipeline`, with the MF 35 particulars under *Conventions and
choices* below). The node stores $\chi$, not $P_j$, so applying a delta means integrating
the table, turning the delta into a ratio, scaling the table so that its group integrals
come out right, and checking that they did.

## What it computes

Input: the tabulated $\chi(E' \mid E)$; for each requested band $b$ its incident range,
its group boundaries $g^{(b)}$ and one delta vector $\delta^{(b)}$. Output: a distribution
of the same kind, at the evaluation's incident energies plus the ones a band edge needs,
where at every incident node $E$ inside band $b$, with $P^0_j$ the evaluated probabilities
{eq}`eq-fission-spectrum-perturbation-groups` at that node,

$$
\int_{g_j}^{g_{j+1}} \chi'(E' \mid E)\, dE' = \kappa(E)\, x_j(E)\, P^0_j(E),
\qquad
\int \chi'(E' \mid E)\, dE' = \int \chi(E' \mid E)\, dE',
$$ (eq-fission-spectrum-perturbation-result)

where $x_j \ge 0$ are the ratios of the delta, made feasible (step 3), and $\kappa$ the
scalar that puts the integral back where it was (step 6). Within a group, away from the
group's edges, $\chi'$ has the shape of $\chi$: $\chi' = \kappa\, x_j\, \chi$. Incident
nodes outside every requested band, and outgoing energies outside $[g_0, g_J]$, are not
scaled.

The same delta vector applies to every incident node of the band; the ratios differ from
node to node because each node has its own $P^0_j$.

## Method

**1. Steps in incident energy.** The perturbation is piecewise constant in $E$, so it steps
at the edges of every band. A node *at* an edge is not enough; the node on the other side
of the step has to be close to it, or the change ramps across the whole incident interval
to the next node. With $s_o = 10^{-3}$, for each band:

- below its lower edge, a node at $E_1(1 - s_o)$;
- at its upper edge, a node at $E_2(1 + s_o)$ when the node at $E_2$ belongs to the band
  (the last band, closed at the top, or a band stated at a point), and at $E_2(1 - s_o)$
  when it does not (a half-open band).

A point is inserted only strictly inside the tabulated incident range and not within a
relative $10^{-6}$ of an existing node, and a point band with no node at its energy gets
none. Contiguous bands need no new node at a shared edge:
the upper point of one band is the lower point of the next, and is inserted once. The
table at an inserted node is the evaluated interpolant there: under a lin-lin incident law,
the linear blend of the two neighbouring tables on the union of their outgoing grids,
which is exact for two lin-lin tables; under a histogram law, a copy of the lower table.

Band membership: a band is $[E_1, E_2)$, the last one $[E_1, E_2]$. A band stated at a
point ($E_1 = E_2$) contains exactly the node at $E_1$, and is looked up first, so it takes
that node from a range band that starts at the same energy.

**2. The group probabilities.** At each node $E$ in band $b$: $P^0_j$ by
{eq}`eq-fission-spectrum-perturbation-groups`, each panel integrated in closed form under
its own law; the whole integral $T = \int \chi\, dE'$; the covered mass $S = \sum_j P^0_j$.
$T$ and $S$ differ when the table reaches outside $[g_0, g_J]$ (a table from $E' = 0$
against groups from $10^{-5}$ eV, for instance).

**3. The ratios** (`pfns_ratio_rule`). A group with $P^0_j = 0$ has no ratio: its delta is
set to 0 and its ratio to 1. Every other group gets

$$
r_j = 1 + \frac{\delta_j}{P^0_j}.
$$ (eq-fission-spectrum-perturbation-ratio)

The ratios are then projected onto $\{x \ge 0,\ \sum_j P^0_j x_j = S\}$, the
$P^0$-weighted Euclidean projection

$$
x = \arg\min_{x \ge 0,\ \sum_j P^0_j x_j = S}\ \sum_j P^0_j\, (x_j - r_j)^2
\quad\Longrightarrow\quad
x_j = \max(r_j + t, 0),
$$ (eq-fission-spectrum-perturbation-projection)

with one scalar $t$ (groups with $P^0_j = 0$ are left at their ratio). $t$ is found in
closed form: sort the $r_j$ in decreasing order, and take the longest prefix $k$ for which
$t_k = (S - \sum_{i \le k} P^0_i r_i) / \sum_{i \le k} P^0_i$ keeps $r_k + t_k \ge 0$. When
the ratios are already feasible and their weighted sum is exactly $S$, $t = 0$ and nothing
changes. Otherwise $t$ absorbs the drift $\sum_j \delta_j$ that a draw of a matrix with
$C\mathbf{1} \approx 0$ leaves, and clips the groups a draw drove negative.

**4. Steps in outgoing energy.** For every boundary $g_{j+1}$ at which the ratio changes
($x_{j+1} \neq x_j$) and which lies strictly inside the table, two outgoing energies are
inserted, the boundary and a shoulder just below it:

$$
g_{j+1}, \qquad g_{j+1} - s\,(g_{j+1} - g_j), \quad s = 10^{-3}.
$$ (eq-fission-spectrum-perturbation-shoulder)

The outer edges of the grid step too, where the table reaches past them and the ratio there
is not 1: $g_0$ with a shoulder at $g_0 - s(g_1 - g_0)$, and $g_J$ with one at
$g_J + s(g_J - g_{J-1})$, both outside the grid. Each inserted point is valued under the
law of the panel it falls in and splits that panel into two panels of the same law, so the
refined table is the same function. A point within a relative $10^{-6}$ of an existing node
is not inserted.

**5. Scale.** Each outgoing node is multiplied by the ratio of its group, a node on a
boundary taking the group above it and a shoulder the group below; nodes outside
$[g_0, g_J]$ keep factor 1. Scaling both ends of a panel by one number keeps the panel's
law, so away from the shoulders $\chi' = x_j\,\chi$ exactly, under any of the five ENDF
laws.

**6. Renormalise.** The whole table is multiplied by $\kappa = T / \int \chi'\, dE'$, so
the node's integral is what the evaluation had (not 1: an evaluation's own departure from
$\int\chi = 1$ is kept, not repaired).

**7. Check.** The group integrals of the result are measured against
$\kappa\, x_j\, P^0_j$ and the worst departure is reported per node (see *Diagnostics*).
Steps 2-7 run on every node of every requested band, and every band of one reaction is
applied in one call, because they act on one table.

### Derivation

*Why the result keeps the normalisation without step 6.* After the projection
$\sum_j P^0_j x_j = S$, so the covered mass is unchanged by step 5, and the mass outside
the grid is not scaled. The integral of $\chi'$ therefore equals $T$ except for the
shoulder slivers, where the table runs lin-lin from one group's factor to the next. In the
sliver below $g_{j+1}$, of width $s(g_{j+1} - g_j)$, the factor departs from $x_j$ by at
most $|x_{j+1} - x_j|$; with $\bar\chi$ the spectrum there, the group integral moves by
about

$$
\tfrac12\, s\,(g_{j+1} - g_j)\, \bar\chi\, |x_{j+1} - x_j|,
$$ (eq-fission-spectrum-perturbation-ramp)

which is a fraction $\approx \tfrac12 s |x_{j+1} - x_j|$ of the group for a spectrum that
varies slowly across it. $\kappa$ then removes the net change of the integral, and step 7
measures what is left per group.

*Why the shoulder is a fraction of the group width.* A shoulder at $g(1-s)$, a fraction of
the energy, spans $s\,g$; on a linear top-end grid with groups much narrower than $g$ (the
top groups of a Cf-252 evaluation are $2\cdot10^5$ eV wide at $2\cdot10^7$ eV), that is a
tenth of the group, and {eq}`eq-fission-spectrum-perturbation-ramp` becomes the size of
the perturbation. A fraction of the group width bounds it whatever the grid geometry.

*What the stated covariance says about the mean energy.* Write the mean outgoing energy
of one node as $q = \sum_g e_g P_g / \sum_g P_g$, with $e_g$ the mean energy within group
$g$. The applier keeps the in-group shape, so $e_g$ does not move, and to first order

$$
\delta q = \sum_g \tilde w_g\, \delta P_g, \qquad \tilde w_g = \frac{e_g - q}{\sum_h P_h},
\qquad
\sigma_q = \sqrt{\tilde w^\top C\, \tilde w}.
$$ (eq-fission-spectrum-perturbation-moment)

The same holds for the fraction of the spectrum above a threshold $t$,
$F(>t) = \sum_g a_g(t) P_g / \sum_g P_g$, with $a_g(t)$ the fraction of group $g$'s
probability above $t$ in place of $e_g$. Centring the weights on $q$ only removes the drift
of a matrix whose rows do not sum exactly to zero.

## Conventions and choices

- **Absolute deltas, drawn linearly.** MF 35 is absolute and $C\mathbf{1} \approx 0$, so a
  linear draw has $\sum_j \delta_j \approx 0$ and keeps the normalisation to within the
  drift the file carries; a log-normal draw would not, and has no meaning for an absolute
  covariance. The deltas are kept in double precision: stored as $1 + \delta$ in single
  precision they vanish for all but the largest groups.
- **What the draw does that is particular to MF 35** (the draw itself is the pipeline's):
  each band is its own block and is never merged with another band or another file; the
  directions with eigenvalue at or below $10^{-10}$ of the largest are dropped from the draw
  (decomposition noise on a group holding $10^{-17}$ of the spectrum would become an
  enormous ratio); each drawn delta is clipped at 5 of its group's **stated** standard
  deviations, because the eigenvalue clipping that repairs a slightly indefinite band adds
  a little variance on the diagonal, which divided by a tiny $P^0_j$ becomes a spike; and a
  conditioning plan may repair a band only by eigenvalue clipping (`clip`) or not at all,
  since every other repair breaks $C\mathbf{1} \approx 0$. A plan that does otherwise is
  refused.
- **Drawn in the same realisation as the rest of the run.** The MF 35 bands are
  decomposed apart (they are absolute and drawn linearly) but realised in one call with
  the relative blocks of the run (MF 31, 33, 34), so under Sobol they take their own
  columns of the run's one sequence ({doc}`../sampling/perturbation-pipeline`). Separate
  Sobol sequences paired by replica would correlate $\bar\nu$ and $\chi$ of one run, where
  the file states no covariance between them. A run whose blocks are all MF 35 draws
  exactly what the file-level path draws.
- **Ratios, not deltas, on the table.** The node stores $\chi$; scaling a group by
  $x_j$ moves its probability by $x_j P^0_j - P^0_j = \delta_j$ (before projection), which
  is the delta that was drawn, and keeps the in-group shape the evaluator stated.
- **The projection is weighted by $P^0$.** The unweighted projection onto the simplex,
  $P_j = \max(y_j - \theta, 0)$, shifts every group by the same absolute amount; on a
  spectrum whose group probabilities span $10^{-17}$ to $10^{-1}$, absorbing a drift of
  $3\cdot10^{-5}$ that way zeroes the small groups (see *Validation*). The weighted one shifts every **ratio**
  by the same $t$, so a group moves in proportion to its probability.
- **The normalisation target is what the node had, not 1.** The covered sum $S$ and the
  whole integral $T$ are measured on the evaluated node. Forcing the covered part to 1
  would move mass outside the groups that no covariance speaks for, and forcing $T$ to 1
  would correct the evaluation, which is not the sampler's job.
- **Mass outside the groups is carried, never perturbed.** It is scaled only by $\kappa$,
  so it moves with the spectrum. How much of each node lies outside is reported, because
  a run whose bands cover all of the spectrum and one whose bands cover most of it write
  tapes that look alike.
- **A step is a shoulder node, not a repeated energy.** As for the multiplicities
  ({doc}`./multiplicity-perturbation`), and because NJOY (the processing code) reads MF 5
  into fixed-size buffers.
- **Every inserted node is an exact refinement**, in both axes. A zero perturbation then
  reproduces the evaluation at every node, inserted ones included; copying the edge's own
  table instead of interpolating would shift the central value by a few parts in a
  thousand and look like physics.
- **A band at a point perturbs one node.** A band with $E_1 = E_2$ is a covariance at one
  incident energy, and the file states none around it, so only the node at exactly that
  energy moves, and the run says so by name.
- **The perturbed tables keep their laws.** A log-interpolated panel stays
  log-interpolated, and a table keeps its regions.
- **The covariance is written unchanged.** Each output carries the evaluation's MF 35 next
  to a spectrum that has moved, and the run metadata says so (`mf35_unchanged`).
- **The stated uncertainty of the mean energy is reported, not imposed.** The ensemble
  carries the library's MF 35 as stated: no constraint, no rejection of samples. What MF 35
  implies for $\langle E' \rangle$ and for $F(>1\ \text{MeV})$ and $F(>3\ \text{MeV})$,
  {eq}`eq-fission-spectrum-perturbation-moment`, is computed on the first evaluated node of
  each band and recorded with the run, beside the spread of the drawn deltas (before the
  projection). Where kika holds an independent value (only U-235 at 0.0253 eV, the IAEA
  standards value $\langle E' \rangle = 2.00 \pm 0.01$ MeV, Carlson et al. 2018, p. 27), the
  ratio of
  the two uncertainties goes into the run's notes.

## Limits and failure modes

Refused:

- a reaction whose energy distribution is a parametrised spectrum (evaporation, general
  evaporation, simple Maxwellian fission, Watt, Madland-Nix, weighted functionals): MF 35 covers group integrals of a table, and
  the file carries no covariance of the formula's parameters (`ValueError`);
- a reaction with no product carrying a tabulated $\chi$, or with more than one
  (photons are never candidates), since the file does not say which one MF 35 is about
  (`ValueError`);
- an incident node to insert between two tables that are not lin-lin in $E'$ (the blend
  of two log-interpolated tables is not a table any law states), or into a container of
  several incident-energy regions (`NotImplementedError`);
- an incident axis interpolated by any law other than histogram or lin-lin, unit-base
  interpolation (INT=22) included (`NotImplementedError`);
- a delta vector whose length is not the band's number of groups, an MF 35 block without
  its band, and a conditioning plan that repairs a band with anything but `clip`
  (`ValueError`).

Reported instead of done: steps dropped under `maxOutgoingPoints` (the smallest first, two
inserted points per step); clipped deltas and clipped ratios; frozen groups; a point band
with no node at its energy, which perturbs nothing; the mass outside the bands' groups.

Limits of the method: the group integrals hold to within the shoulder ramp,
{eq}`eq-fission-spectrum-perturbation-ramp`; a ratio driven near zero makes the relative
error of that group large while its absolute error stays small (gate on the mass error,
see *Diagnostics*); a perturbed table grows (on the committed Cf-252 tape by a factor
between 1.2 and 2.5), which is what `maxOutgoingPoints` caps; the clamp at 5 standard
deviations and the projection change the realised distribution of the deltas slightly
where groups carry almost no probability.

## Validation

The tests below passed on 2026-10-11 unless marked. The committed fixtures are
`micro_pfns_cov.endf` (eight groups, two bands) and `micro_cf252_pfns.endf` (four real
LB=7 bands over a real Cf-252 MF 5 table), under `kika/endf/tests/data/`.

**Against the file-level applier.** On both fixtures, with real drawn deltas,
`applySpectrumFactors` reproduces `kika.sampling.mf35_sampling.perturb_pfns_partial` bit for
bit, tables and diagnostics
(`kika/sampling/tests/test_perturbing_a_fission_spectrum_on_the_model.py::test_the_model_applier_reproduces_the_format_applier`);
the shoulder constants are the same on both sides
(`::test_the_two_appliers_write_a_step_the_same_way`); the model integrates exactly what
the ENDF class integrates (`::test_the_model_node_integrates_exactly_what_the_endf_class_does`,
`::test_refining_the_incident_axis_is_the_same_exact_blend`). The model's draw is the
file-level draw bit for bit
(`kika/sampling/tests/test_the_model_draws_a_fission_spectrum_as_the_legacy_does.py::test_the_model_draw_is_the_legacy_draw`).
The two parts on purpose for a request of some bands only, where the model inserts the step
below the lowest band and the file-level applier does not
(`kika/sampling/tests/test_the_model_writes_a_fission_spectrum_as_the_legacy_does.py::test_a_band_subset_is_where_the_two_appliers_part_on_purpose`),
and for log-interpolated tables, which the file-level applier rewrites as lin-lin
(`kika/sampling/tests/test_a_log_interpolated_spectrum_is_perturbed_exactly.py::test_the_legacy_applier_relabels_a_log_lin_spectrum_as_lin_lin`).
`TODO(check)`: the whole-tape gates of the two writers on real evaluations
(`test_the_model_writes_a_fission_spectrum_as_the_legacy_does.py`) need external tapes and
were not re-run.

**Exact cases.** A zero perturbation reproduces $\chi$ to $10^{-14}$ of its maximum at every
node, inserted ones included
(`kika/sampling/tests/test_perturbing_a_fission_spectrum_on_the_model.py::test_a_zero_perturbation_reproduces_chi_pointwise`;
on the file-level applier, to $5\cdot10^{-14}$ relative at 10 000 random $(E, E')$,
`kika/sampling/tests/test_mf35_sampling.py::test_r_identically_one_reproduces_chi_pointwise`);
one ratio in every group gives $P_j = P^0_j$ to $10^{-12}$ after projection and
renormalisation (`::test_a_uniform_factor_gives_exactly_r_times_p0`). On a log-lin
spectrum, away from the shoulders $\chi' = \kappa\, x_j\, \chi$ to $10^{-12}$ and the regions
reach the written section
(`kika/sampling/tests/test_a_log_interpolated_spectrum_is_perturbed_exactly.py::test_the_perturbed_spectrum_is_the_factor_times_chi_between_steps`,
`::test_the_regions_survive_and_reach_the_endf_section`,
`::test_a_zero_perturbation_gives_the_log_lin_spectrum_back`); an incident node between two
log tables is refused (`::test_an_incident_node_between_log_tables_is_refused_by_name`).

**Budgets on Cf-252** (`kika/sampling/tests/test_mf35_sampling.py`, file-level applier, which
the model reproduces bit for bit on this tape). Over four samples, worst group mass error
below $10^{-5}$ of the covered spectrum, relative group error below 0.1 and renormalisation
error below $10^{-4}$ (`::test_shoulder_ramp_error_stays_inside_its_budget`); 16 samples
normalised to $10^{-6}$ and non-negative everywhere
(`::test_every_sample_is_normalised_and_positive`); table growth between 1.2 and 2.5
(`::test_the_table_grows_but_stays_inside_the_measured_bound`); the point cap drops steps
and says so (`::test_max_outgoing_points_drops_the_smallest_steps_and_says_so`); a group with
$P^0 = 0$ is frozen and the sum still closes to $10^{-12}$
(`::test_empty_group_freezes_its_delta_and_the_sum_still_closes`).

**The draw** (same file). Deltas in double precision and linear
(`::test_generate_pfns_samples_returns_float64_and_absolute_deltas`); every sample's delta
sum below $10^{-3}$ in absolute value (`::test_every_drawn_sample_has_a_vanishing_delta_sum`);
the 5-standard-deviation clamp leaves the realised standard deviation of every group holding
more than $10^{-4}$ of the spectrum within 5 % of the stated one, and the groups it clamps
often hold less than $10^{-10}$ of it
(`::test_the_sigma_clamp_touches_only_groups_that_carry_no_mass`); on 512 samples the
median over retained modes of $|\mathrm{Var}(v_k^\top\delta)/\lambda_k - 1|$ is inside
$3\sqrt{2/N}$ and less than $10^{-3}$ of the draw leaks into the null space
(`::test_realised_covariance_matches_mf35_on_the_non_null_subspace`). No group without stated
variance is perturbed and no delta exceeds 5 stated standard deviations on the model path
(`kika/sampling/tests/test_the_model_draws_a_fission_spectrum_as_the_legacy_does.py::test_no_group_is_drawn_beyond_what_the_file_states`);
a plan repairing a band with `clip_rescale` or `higham` is refused
(`::test_a_plan_that_would_break_the_sum_rule_is_refused`); the projection leaves the sum
within $10^{-12}$ on every sample (`::test_every_sample_reports_its_projection_budgets`).

**The projection** (`kika/sampling/tests/test_pfns_positivity.py`). Identity when already
feasible (`::test_projection_is_identity_when_already_feasible`); sum restored to $10^{-12}$
and non-negative on 200 random cases (`::test_projection_restores_the_sum_to_machine_precision`);
a drift of $3\cdot10^{-5}$ on groups from $10^{-17}$ to 0.5 moves every ratio by less than
$10^{-4}$ and zeroes none, where the unweighted projection zeroes at least three
(`::test_projection_does_not_annihilate_small_groups`); only negative ratios are clipped
(`::test_projection_clips_only_genuinely_negative_ratios`).

**Where the factor stops** (`kika/sampling/tests/test_the_spectrum_factor_stops_where_the_covariance_does.py`,
synthetic spectrum). A point band moves its node and nothing at the shoulders either side
(`::test_a_point_band_perturbs_the_node_at_its_point_and_nothing_else`), takes its node from
a range band that starts there (`::test_a_point_band_takes_its_node_from_a_band_that_starts_there`),
and moves nothing when the grid has no node at its energy
(`::test_a_point_band_with_no_node_at_its_point_moves_nothing_and_says_so`); above the last
band and inside a gap between bands the group probabilities move by less than $10^{-12}$
(`::test_the_factor_stops_at_the_top_of_the_last_band`,
`::test_the_factor_stops_below_a_gap_between_two_bands`); contiguous bands get one new
node only (`::test_contiguous_bands_get_no_new_outer_node`). A partial band request moves
nothing below the requested band
(`kika/sampling/tests/test_perturbing_a_fission_spectrum_on_the_model.py::test_a_partial_band_request_steps_where_the_perturbation_starts`).
`TODO(check)`: the tape witnesses in the same file (point band of a U-238 evaluation, the
partial band and grid edge of a U-233 evaluation) need external tapes and were not re-run.

**End to end** (`kika/sampling/tests/test_perturbing_a_fission_spectrum_on_the_model.py`).
The delta tape moves MF 5 and one directory record, and leaves MF 3 and MF 35 byte for byte
(`::test_a_pfns_run_writes_a_delta_tape_that_only_moved_mf5`); the written tape normalises
to $10^{-5}$ (`::test_the_written_tape_still_normalises`); one block per band, recorded
(`::test_the_run_records_the_band_of_every_block`); the bands are drawn as deltas
(`::test_the_spectrum_blocks_are_drawn_as_deltas_and_not_as_factors`).

**The mean energy** (`kika/sampling/tests/test_the_run_reports_what_mf35_states_for_the_mean_energy.py`).
$\langle E' \rangle$ and $F(>1\ \text{MeV})$ of a triangle exact to $10^{-14}$
(`::test_the_functionals_are_exact_on_a_lin_lin_table`); the drawn spread of
{eq}`eq-fission-spectrum-perturbation-moment` follows the stated one within its 3σ
sampling band over 20 000 draws (`::test_the_stated_sigma_is_the_sandwich_and_the_drawn_one_follows_it`);
every run writes the records (`::test_every_run_writes_c10_beside_its_samples`).
`TODO(check)`: the U-235 comparison with the standards value
(`::test_jeff40_u235_reports_its_thermal_sigma_against_the_standard`) needs an external tape
and was not re-run.

## Diagnostics

In each sample's `applied` entry of a
{py:class}`~kika.sampling.model_perturbation.RunResult`, per MF 35 band (the worst over the
band's incident nodes):

- `n_nodes`, `n_outer_inserted` (incident nodes added for the steps, per reaction),
  `total_outgoing_inserted`;
- `max_group_mass_error`: $\max_j |P_j - \kappa x_j P^0_j| / S$. **The one to watch**: it
  says whether the table carries the perturbation that was computed. `max_group_ratio_error`
  is the same as a relative error per group, and is large wherever a ratio was driven near
  zero, without meaning anything is wrong;
- `max_renormalisation_error`, $|\kappa - 1|$: what the shoulder ramps cost the integral;
  read it against `input_normalisation_max_abs`, the evaluation's own $|\int\chi - 1|$ in
  the band;
- `max_frac_mass_outside_bands`: the fraction of a node's integral outside the groups,
  carried and not perturbed;
- from the ratio rule: `max_sum_error_before_projection`, `max_sum_error_after_projection`,
  `max_projection_shift` ($|t|$), `total_clipped` and `max_clipped_mass_fraction` (the
  probability of the groups clipped to zero, as a fraction of $S$), `total_groups_frozen`.
  **A clipped mass fraction that is not tiny** means the sample was moved to stay a
  spectrum;
- `total_steps_dropped`: steps not written under `maxOutgoingPoints`. **Not zero** means the
  tape is not the sample in those groups;
- for a band at a point: `band_kind`, `energy`, `node_at_point` and a `note`.

Per band, from the draw, in `RunResult.diagnostics`: `null_tol`, `sigma_clamp`,
`n_clamped`, `clamped_fraction`, `row_sum_residual` ($\max_i |\sum_j C_{ij}| / \max|C|$),
`normalisation_drift` ($\sqrt{|\mathbf{1}^\top C \mathbf{1}|}$), `spectral_fidelity_median`,
`gate_tolerance`, `passes_spectral_gate`, `null_leakage`, `n_modes_retained`,
`max_abs_delta_sum`. **`passes_spectral_gate` false** on a large ensemble means the deltas do
not reproduce the band.

`RunResult.spectrumMoments` (and `mf35StatedMoments` in `run_metadata.json`) holds, per MT
and band, the probe energy, `meanEnergy`, the tail fractions, their stated and drawn
relative standard deviations, the constraint applied (none), and the reference where one
exists. The run's `notes` say that MF 35 is written unchanged, how much of the spectrum lies
outside the bands, any dropped steps, any point band, and the mean-energy comparison where a
reference exists.

## Implementation

- `kika.nuclear_data.model.perturbation`: the table. `applySpectrumFactors` (per node:
  integrate, ask for the ratios, refine, scale, renormalise, check), the band rule and the
  incident shoulders, `SPECTRUM_STEP_SHOULDER`, `SPECTRUM_OUTER_SHOULDER`,
  `SPECTRUM_ENERGY_RTOL`, `summariseSpectrumNodes`.
- `kika.nuclear_data.model.functions.higher`: the two-dimensional table (`groupIntegrals`,
  `normalisation`, `replaceTable`, `evaluateAtOuter`, `insertOuterNode`); `kika.algebra`:
  closed-form group integrals and evaluation under each ENDF law, and the laws of a refined
  table.
- `kika.sampling.mf35_sampling`: the sample. `pfns_ratio_rule` (frozen groups, ratios,
  projection), the clamp (`clamp_to_stated_sigma`, `SIGMA_CLAMP`), `NULL_TOL`,
  `SAMPLING_SPACE`, the per-band draw report; beside them, the file-level applier
  `perturb_pfns_partial` the model path is gated against.
- `kika.sampling.pfns_positivity`: `check_ratios` and `project_ratios_to_simplex`.
- `kika.sampling.pfns_moments`: `spectrumFunctionals`, `statedSpectrumMoments` and the note.
- `kika.sampling.perturbation_set`: dispatch, the choice of the product, and the realisation
  placed beside the evaluated distribution.
- `kika.sampling.model_perturbation`: the delta draw, the refusal of sum-breaking repairs,
  the notes and the run metadata.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual: Data Formats and Procedures for the Evaluated
  Nuclear Data Files ENDF/B-VI, ENDF/B-VII and ENDF/B-VIII*, Brookhaven National
  Laboratory (2023), doi:10.2172/2007538. Cited as ENDF-6: Ch. 5 (MF 5), Ch. 35 (MF 35).
- A. D. Carlson, V. G. Pronyaev et al., "Evaluation of the neutron data standards",
  *Nuclear Data Sheets* 148, 143-188 (2018), doi:10.1016/j.nds.2018.02.002: the U-235
  thermal prompt fission neutron spectrum, mean energy $2.00 \pm 0.01$ MeV (p. 27; the
  uncertainty raised from 5 keV to 10 keV for unrecognised shape uncertainties).
