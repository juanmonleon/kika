# Perturbing a Legendre angular distribution

```{admonition} At a glance
:class: note
**Computes:** a sample of an angular distribution stated as Legendre coefficients
$a_\ell(E)$, given one factor per Legendre order and incident-energy bin drawn from an
MF 34 covariance, in the covariance's frame when it is not the distribution's; the
factors of order 0 go to the cross-section.

**Applies to:** MF 4 with LTT=1, and the Legendre half of LTT=3; any reaction in the
distribution's own frame, elastic and discrete inelastic scattering (MT 2, MT 51-90)
when MF 34 is in the other frame.

**Entry points:** {py:func}`kika.sampling.model_perturbation.perturbFromModel`,
{py:func}`kika.nuclear_data.model.perturbation.applyLegendreFactors`;
`kika._legendre_frames.scaleInOtherFrame` for one coefficient vector.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

MF 4 of an ENDF (Evaluated Nuclear Data File) evaluation with LTT=1 gives the angular
distribution of the emitted neutron at a list of incident energies $E_j$, each as a
vector of Legendre coefficients (ENDF-6 Ch. 4),

$$
f(\mu, E_j) = \sum_{\ell=0}^{N_j} \frac{2\ell+1}{2}\, a_\ell(E_j)\, P_\ell(\mu),
\qquad a_0 = 1,
$$ (eq-legendre-angular-perturbation-series)

with the length $N_j$ free to change from energy to energy, and an interpolation law on
the incident-energy axis. The file writes $a_1, \dots, a_{N_j}$; $a_0 = 1$ is implicit,
because $f$ is normalised and the size of the reaction is the cross-section of MF 3. LTT=3
uses this representation below a transition energy and tables of $f(\mu)$ above it.

MF 34 states the relative covariance of the $a_\ell$ on incident-energy bins that may
differ from order to order (ENDF-6 §34.1). It may also carry sections with $\ell = 0$:
since $a_0$ is identically 1, an L=0 section is a statement about the cross-section, not
about the shape. L0×L0 is the relative variance of $\sigma(E)$ of that reaction on
MF 34's grid (a second estimate beside MF 33), and L0×Lℓ the correlation between
$\sigma$ and $a_\ell$. ENDF-6 §34.1-34.3 describe these sections. The flag LCT says
whether the coefficients the covariance describes are in the laboratory (LAB) or
centre-of-mass (CM) frame, and that need not be the frame of MF 4.

A sample of that covariance is, for every order $\ell$ and bin $k$ the covariance names,
a factor $c_{\ell k}$; how they are drawn is in
{doc}`../sampling/perturbation-pipeline`. This page is how kika applies them to a
Legendre distribution. Tables of $f(\mu)$ are the subject of
{doc}`./tabulated-angular-perturbation`.

## What it computes

Input: the Legendre distribution, the set $\mathcal{L}$ of perturbed orders (all
$\ell \ge 1$), and per order its bin edges and factors. Output: a distribution of the
same kind, at the same incident energies plus the repeated energies a bin edge needs,
where at every incident energy $E$, in the distribution's own frame,

$$
a'_\ell(E) =
\begin{cases}
c_\ell(E)\, a_\ell(E) & \ell \in \mathcal{L},\\
a_\ell(E) & \text{otherwise},
\end{cases}
$$ (eq-legendre-angular-perturbation-result)

with $c_\ell(E)$ the factor of the bin of order $\ell$ that contains $E$, and 1 outside
the bins of that order. When the covariance is in the other frame, the same statement
holds for the coefficients $b_\ell$ of that frame, and only the change comes back
({eq}`eq-legendre-angular-perturbation-solve`). Factors of order 0 are not applied to
the distribution: they multiply the cross-section of the same reaction, on MF 34's bins,
exactly as an MF 33 factor does (see {doc}`./cross-section-perturbation`).

## Method

**1. Incident-energy steps.** The factors are piecewise constant in $E$, and ENDF states a
step as a repeated incident energy. The edges of every perturbed order's bins are pooled.
Each edge strictly inside the span of the Legendre region's energies (to an absolute
$10^{-10}$ eV) is made a repeated energy:

- if the region already tabulates that energy twice or more, nothing is inserted;
- if it tabulates it once, that coefficient vector is duplicated;
- if it falls between $E_j$ and $E_{j+1}$, the vector at the edge is built and inserted
  twice. Under a lin-lin energy law it is
  $a_\ell(E) = (1-t)\,a_\ell(E_j) + t\,a_\ell(E_{j+1})$ with
  $t = (E - E_j)/(E_{j+1} - E_j)$, the shorter vector padded with zeros (an order the
  evaluation does not write is zero); under a flat law it is a copy of the vector at
  $E_j$.

**2. The factor of each energy.** On the refined list, each energy takes, per order, the
factor of the bin that contains it ($E_\text{lo} \le E < E_\text{hi}$), with one
exception: the lower copy of a repeated energy takes the factor of the bin that ends
there. Outside an order's bins the factor is 1. This is the same bin rule the
cross-section applier uses, {doc}`./cross-section-perturbation`.

**3a. Same frame.** Each coefficient is multiplied by its factor,
{eq}`eq-legendre-angular-perturbation-result`. An order beyond the length of the vector
at some energy is left absent there: the factor of a coefficient the evaluation does not
write is not applied to a zero it would have to invent.

**3b. The covariance in the other frame.** The factors describe the coefficients of the
covariance's frame, so they are applied there and only the change is brought back. With
$m(\mu)$ the cosine, in the covariance's frame, of the direction whose cosine is $\mu$ in
the distribution's frame (the two-body map and $\gamma$ of
{doc}`./tabulated-angular-perturbation`, Method step 4), the coefficients in the other
frame are $b = T a$, with

$$
T_{\ell k}(\gamma) = \frac{2k+1}{2} \int_{-1}^{1} P_k(\mu)\, P_\ell\big(m(\mu)\big)\, d\mu ,
$$ (eq-legendre-angular-perturbation-transfer)

evaluated by Gauss-Legendre quadrature with $\max(64,\, n_\text{in} + n_\text{out} + 16)$
points. Row 0 of $T$ is $(1, 0, 0, \dots)$, because a change of frame keeps the
normalisation. A finite series in one frame is an infinite series in the other, so the
change is solved on a truncated set of orders $0, \dots, N$:

$$
\Delta b_\ell = \begin{cases} (c_\ell - 1)\, b_\ell & \ell \in \mathcal{L}\\ 0 & \ell \le N,\ \ell \notin \mathcal{L}\end{cases},
\qquad
\Delta a = \big(T_{[0..N],[0..N]}\big)^{-1} \Delta b ,
\qquad a' = a + \Delta a .
$$ (eq-legendre-angular-perturbation-solve)

The square solve puts exactly $c_\ell b_\ell$ in each perturbed order of the other frame
and leaves every other order up to $N$ where it was, $\ell = 0$ included. $N$ starts two
above the highest perturbed order and grows by two until the two highest orders of
$\Delta a$ are below $10^{-10}$ times its largest entry (`PAD_TOLERANCE`), or until $N$ is
40 above the highest perturbed order (`MAX_PAD`). Orders the evaluation did not carry
are kept only where $|a'_\ell|$ exceeds the same $10^{-10}$ times the largest correction,
so the vector grows only by what the correction put there. A node where every factor is
1 is left untouched. A node with $\gamma \ge 1$ (at or just above an inelastic threshold,
where the cosine map is not one-to-one) has no vector with the requested coefficients in
the other frame, and is left as the evaluation has it.

**4. Order 0.** A request for MF 34 that names no orders takes the orders from 1, and
the run records a note saying what the L=0 sections it left out state (whether their
L0×Lℓ blocks carry anything, with $10^{-12}$ as the threshold of a null block). A request
that names order 0 explicitly is a request to perturb $\sigma$ from MF 34: its factors are
applied to the reaction's cross-section by the cross-section applier, on MF 34's L=0
bins, and the sum rules of MF 3 then rebuild the sums above it. Since, under the default
`grouping="mf"`, all requested components of MF 34 form one block, those factors are drawn together with the shape,
L0×Lℓ correlation included.

**5. Mixed distributions.** In an LTT=3 distribution only the Legendre region is
perturbed by this method; the tabulated region gets the same factors through
{doc}`./tabulated-angular-perturbation`, so a bin that spans the transition energy acts
on both sides of it.

### Derivation

*Why the square solve gives the drawn factors in the other frame.* Let $a'$ have orders
$0..N$ changed by $\Delta a$. Its coefficients in the other frame are
$b' = T(a + \Delta a) = b + T\Delta a$. Restricted to orders $0..N$, $T\Delta a$ equals
$\Delta b$ by construction, so $b'_\ell = c_\ell b_\ell$ for $\ell \in \mathcal{L}$ and
$b'_\ell = b_\ell$ for every other $\ell \le N$. Above $N$, $T\Delta a$ is not
controlled: that is the part a finite CM change spills into higher LAB orders (or the
reverse), and it shrinks as $N$ grows, because the correction converges. The stopping
rule bounds what is left there at about $10^{-10}$ of the largest correction. The limit
of this solve as $N \to \infty$ is the closed form that the tabulated case evaluates
directly ({doc}`./tabulated-angular-perturbation`, Derivation).

## Conventions and choices

- **Only the perturbed orders move.** Order 0, and every order the covariance does not
  name, keep the evaluation's value in the frame the covariance is in. There is no
  truncation order to choose.
- **Order 0 is the cross-section, never the shape.** In MF 4 $a_0 = 1$ by construction, so
  scaling it would put the magnitude into the normalisation of $f$, and no file would
  read it back that way. Applying the L=0 factor to MF 3 is what the covariance says.
- **L=0 is left out unless named.** A request for "the angular distribution" should not
  rewrite MF 3. Asking for MF 33 and MF 34 together then draws $\sigma$ from MF 33 and the
  shape from MF 34's L≥1, and the run's note says that the L0×Lℓ correlation, if the file
  states one, is not in the draw.
- **The factors act in the covariance's frame.** A LAB covariance describes LAB
  coefficients. Applying its factors to the CM coefficients perturbs a different
  quantity: for a heavy target the LAB $a_1$ is mostly kinematics ($\approx 2/(3A)$), and a
  10 % factor on the CM $a_1$ moves it by less than 1 % (see *Validation*).
- **Only the change is converted back,** never the whole distribution. With every factor
  at 1, $\Delta b = 0$ and the vector is the evaluation's, bit for bit; a conversion and
  its inverse would lose that to the truncation.
- **The pad is grown until it converges, not fixed.** For elastic scattering $\gamma = 1/A$
  is small and a few extra orders suffice; near $\gamma \to 1$ (light targets, or an
  inelastic level just above threshold) many more are needed.
- **Where an order's bins end inside the region, the default is a step**
  (`coverageEdges="step"`): the outermost edges are repeated energies like any other, so
  the factor holds up to the edge and the evaluation resumes beyond it. `"ramp"` inserts
  no energy at an order's first and last edge, so the change runs lin-lin from the last
  perturbed energy to the first unperturbed one; it reproduces the file-level sampler
  {py:func}`kika.sampling.endf_perturbation.perturb_ENDF_files`, and exists to compare
  against it. `perturbFromModel` always uses `"step"`.
- **Log space by default.** `perturbFromModel` draws every relative block in log space
  unless told otherwise; for a coefficient whose uncertainty straddles zero,
  `space={34: "linear"}` is the choice that can represent it (see
  {doc}`../sampling/perturbation-pipeline`).
- **No renormalisation; positivity is the caller's decision.** $a_0$ is kept (in the
  other frame to rounding, since row 0 of $T$ is $(1, 0, \dots)$), so $\int f'\,d\mu = 1$
  holds. A large enough factor can make
  {eq}`eq-legendre-angular-perturbation-series` negative at some $\mu$; the applier itself
  leaves it as computed, because whether a sample may carry a negative probability is a
  decision about the sample. `perturbFromModel` checks every perturbed vector and, by
  default (`angularPositivity="report"`), counts the negative ones and leaves them;
  `angularPositivity="repair"` projects them as below. The file-level sampler projects
  them under `enforce_positivity=True`.

### Positivity

`perturbFromModel` and `perturb_ENDF_files(enforce_positivity=True)` apply the same rule.
Every perturbed vector is checked on $n = 101$ equally spaced cosines
(`positivity_check_points`), using `kika.sampling.mf4_positivity.check_mf4_positivity`. A
vector negative at any of them is, under a repair, replaced by
`kika.sampling.mf4_positivity.project_mf4_to_positive`,

$$
a^\star = \arg\min_{x} \tfrac12 \|x - a'\|_2^2
\quad\text{subject to}\quad
\sum_\ell (2\ell+1)\, x_\ell P_\ell(\mu_j) \ge 0 \;\; (j = 1..n),\qquad
x_0 = 1,\quad x_\ell = a'_\ell \;\; (\ell > \ell_{\max}),
$$ (eq-legendre-angular-perturbation-projection)

with $\ell_{\max}$ the highest perturbed order, solved by SLSQP (tolerance $10^{-12}$,
500 iterations at most). The constraint holds at the check points only, so a projected
vector can still dip slightly below zero between them; orders at or below $\ell_{\max}$
that the covariance does not name can move. The file-level sampler logs each projection
as a positivity event (MT, energy, the minimum before, the largest change); a run of
`perturbFromModel` reports the counts under `legendre` (see *Diagnostics*).

## Limits and failure modes

Refused (`ValueError` unless stated):

- a factor block for order 0 handed to the Legendre applier: it is the magnitude;
- orders with factors and no bin edges, or edges and no factors, or a number of factors
  that is not the number of bins, or bin edges that are not strictly ascending;
- a distribution with no Legendre region (a pure table goes to
  {doc}`./tabulated-angular-perturbation`);
- `coverageEdges` other than `"step"` or `"ramp"`;
- a bin edge between two tabulated energies when the energy law is neither lin-lin nor
  flat (`NotImplementedError`): the vector there would be one no evaluation stated;
- a frame change for a reaction other than MT 2 and MT 51-90, which are not two-body
  neutron scattering; a frame change without the target's AWR, or for an inelastic level
  without its Q; orders of one reaction stated in different frames;
- MF 33 and MF 34's order 0 for the same reaction in one sample: both would multiply
  $\sigma$, and which estimate a run wants is the caller's decision.

Reported instead of done: an order the distribution does not carry at any energy
(`orders_absent`); bins that reach into the tabulated half of an LTT=3 distribution
(`tabulated_uncovered`, which the tabulated applier then covers); nodes with
$\gamma \ge 1$ left unperturbed (`n_degenerate`); a pad that reached its cap
(`pad_capped`); L=0 sections left out of a request (a note on the run).

Limits of the method: a relative covariance cannot perturb a coefficient that is zero;
the step in $E$ is exact only at the inserted energies, and lin-lin between them, as the
file states; and the truncated solve leaves up to about $10^{-10}$ of the correction in
orders of the other frame the covariance does not name.

## Validation

The tests below passed on 2026-10-11. The fixture of the first two groups is
`kika/endf/tests/data/micro_fe56_structural.endf`: Fe-56 elastic scattering from
JEFF-4.0, LTT=3 in CM, a Legendre region of 3 960 incident energies with vectors of 4 to
32 orders, and an MF 34 stating orders 1 to 6 on grids of 42, 42, 42, 23, 12 and 13 bins.
The factors are constructed, $1 + 0.05\cos k$ over the flat index, not drawn.

**Against the file-level applier.** With `coverageEdges="ramp"`, the model applier
refines to the same incident energies, bit for bit, and gives the same coefficients, bit
for bit, at every energy except 44 (22 bin edges, each written twice). There the
file-level applier writes the evaluated value for every order whose own grid has no edge
at that energy, a notch the covariance does not state; the model applier writes the
evaluated value times that order's factor. The test asserts that this, and nothing
else, is the difference, and that $a_0 = 1$ everywhere
(`kika/nuclear_data/model/tests/test_applying_a_perturbation_to_an_angular_distribution.py::test_the_two_appliers_refine_the_same_energies`,
`::test_the_two_appliers_produce_the_same_coefficients`). The default `"step"` adds
exactly one energy, a repeated 20 MeV at the upper end of every order's bins
(`::test_the_default_states_the_step_the_block_implies`). The covariance stops at
20 MeV and the tables start at 45 MeV, so nothing reaches the tabulated half
(`::test_the_fe56_covariance_stops_short_of_the_tabulated_half`); a block that does is
reported per order (`::test_bins_that_reach_the_tabulated_half_are_reported_per_order`).
The draw itself, 252 rows with 87 null directions, is bit-identical to the file-level
sampler's (`kika/sampling/tests/test_mf34_migration_equivalence.py::test_the_draw_itself_is_bit_for_bit`).

**The frame change.** In `kika/sampling/tests/test_mf34_frame_conversion.py`:

| Check | Agreement | Test |
|---|---|---|
| $T$ against an independent CM→LAB transform (256 points), A = 1.9, 11.9, 55.45, 236 | 10⁻¹³ absolute | `::test_the_transfer_matrix_agrees_with_the_collapse_transform` |
| LAB←CM after CM→LAB is the identity, $\gamma$ = 1/236, 1/12, 0.4 | 10⁻¹⁰ absolute | `::test_the_two_directions_are_inverse` |
| LAB orders 1-6 take exactly their factors, A = 11.9, 55.45, 236 | 10⁻¹² relative; orders ≥ 7 move < 10⁻¹¹; condition number < 2 | `::test_the_lab_coefficients_take_exactly_the_drawn_factors` |
| LAB distribution, CM covariance, $\gamma$ = 1/30 | 10⁻¹¹ relative | `::test_the_other_direction_lab_distribution_cm_covariance` |
| every factor 1 | bit for bit | `::test_unit_factors_return_the_evaluation_bit_for_bit` |
| $\gamma \ge 1$ | node left alone | `::test_a_node_at_threshold_is_left_alone` |

Applying a 10 % LAB factor to the CM $a_1$ of a U-238-like vector (A = 236,
$a_1 = 2\cdot10^{-4}$) moves the LAB $a_1$ by less than 1 %; applied in LAB it moves by
10 % to 10⁻¹⁰ (`::test_applying_in_cm_is_not_the_same_thing`). On the fixture with
every MF 34 LCT rewritten to LAB, the LAB coefficients of more than 1 000 perturbed
nodes move by their factors to 10⁻¹⁰ relative, no node is degenerate and the largest
condition number is below 1.1
(`::test_the_model_applier_moves_the_lab_coefficients_by_the_drawn_factors`,
`::test_the_model_applier_reports_the_conversion`); off the bin edges the file-level
applier with the same conversion agrees to 10⁻¹⁵
(`::test_the_legacy_applier_agrees_with_the_model_off_the_bin_edges`).

**Order 0.** On a fixture tape with L=0 sections
(`kika/sampling/tests/test_the_magnitude_order_of_mf34.py`): `{34: None}` perturbs
orders 1 and 2 and writes only MF 4, with a note
(`::test_an_angular_request_leaves_the_cross_section_alone`); naming order 0 moves MF 3
of the reaction and its total (`::test_naming_order_zero_still_perturbs_sigma_from_mf34`);
`{33: None, 34: None}` draws MF 33 and L≥1, and the note says the cross term is not in
the draw (`::test_cross_section_and_angular_together_are_drawn`). The L=0 factor reaches
the cross-section node, and MF 33 with L=0 on one reaction is refused
(`kika/sampling/tests/test_perturbation_set.py::test_the_magnitude_order_lands_on_the_cross_section`,
`::test_two_claims_on_one_cross_section_are_refused`); the Legendre applier refuses it
(`kika/nuclear_data/model/tests/test_applying_a_perturbation_to_an_angular_distribution.py::test_the_magnitude_order_is_refused_rather_than_applied`).
The file-level applier skips L=0 factors without touching the coefficients
(`kika/sampling/tests/test_mf4_l0_magnitude.py`, 4 tests).

**Positivity.** A vector that is already non-negative comes back
unchanged; $a_1 = 0.9$, negative at $\mu = -1$, is projected with $a_0$ held at 1 to
10⁻¹⁰ and is non-negative on the 101 check points to 10⁻⁷ and on 1 001 points to
10⁻³ of its maximum; frozen orders stay at their values
(`kika/sampling/tests/test_mf4_positivity.py::test_already_positive_returns_unchanged`,
`::test_forward_peaked_violates_positivity_at_backscatter`,
`::test_frozen_indices_are_pinned`). On the model route a negative series is counted and
left under `"report"`, and under `"repair"` projected with $a_0$ and the orders above the
perturbed one unchanged, a series that was non-negative left alone
(`kika/sampling/tests/test_a_perturbed_legendre_series_is_checked_for_positivity.py`).

## Diagnostics

{py:func}`~kika.nuclear_data.model.perturbation.applyLegendreFactors` returns, with the
perturbed distribution:

- `per_order`: per order, `min_factor`, `max_factor` and `n_scaled`. In the same frame,
  `n_scaled` counts the coefficients multiplied, factor-1 ones included; in the other
  frame, the nodes converted.
- `n_inserted`: incident energies added for bin edges.
- `orders_absent`: requested orders with `n_scaled` of 0, which in the same frame means
  the evaluation writes that order at no energy. **Worth a look**: the covariance
  describes a coefficient the evaluation does not carry, and nothing was perturbed.
- `tabulated_from` and `tabulated_uncovered`: the first energy of a tabulated region (or
  `None`), and per order the bins that reach above it (`n_bins`, `fraction`,
  `energy_max`).
- `frame`, when there is a frame change: `direction`, `awr`, `q`, `n_converted`,
  `n_degenerate`, `max_gamma`, `max_condition`, `max_solved` (the largest $N+1$) and
  `pad_capped`. **`n_degenerate` above zero** means nodes at an inelastic threshold were
  left unperturbed; **`pad_capped` above zero** means the solve stopped at `MAX_PAD`
  before converging, so orders the covariance does not name were moved by more than
  the tolerance.

In a run of `perturbFromModel`, each sample's `applied` entry for an MF 34 component
carries `n_inserted`, its order's `per_order` values, `frame`, and `legendre`
(`n_negative_series`, `n_repaired`, `min_sum`: the negative vectors found, those repaired,
and the lowest value of $\sum_\ell (2\ell+1)\, a_\ell P_\ell(\mu) = 2f(\mu)$ on the check points); an L=0 component
carries the cross-section applier's diagnostics. The notes of the run
(`RunResult.notes`) say which L=0 sections were left out and whether their correlation
with the shape is lost.

## Implementation

- `kika.nuclear_data.model.perturbation`: the Legendre applier (`applyLegendreFactors`):
  finding the Legendre regions, inserting the repeated energies, interpolating a vector
  at an edge, the bin rule, the diagnostics; `MAGNITUDE_ORDER` and `COVERAGE_EDGES`.
- `kika._legendre_frames`: the kinematics (`twoBodyGamma`), the transfer matrix
  (`transferMatrix`), the padded solve (`scaleInOtherFrame`, `PAD_TOLERANCE`,
  `MAX_PAD`), and `FrameConversion`, which holds the frames, AWR and Q of one reaction.
- `kika.sampling.perturbation_set`: dispatch. An MF 34 component of order 0 goes with the
  cross-sections (and the refusal next to MF 33), the others to the angular appliers,
  with the frame conversion built from the reaction's AWR and Q.
- `kika.sampling.joint_blocks`: leaving order 0 out of a request that names no orders,
  and the note (`resolveMagnitudeOrder`).
- `kika.sampling.mf34_cross`: reading the L=0 cross blocks of MF 34 on their own grids,
  with their consistency checks, for the joint MF 33/MF 34 file-level sampler.
- `kika.sampling.mf4_positivity`: the check and the projection of a Legendre vector
  (`check_mf4_positivity`, `project_mf4_to_positive`), used by the file-level sampler.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual: Data Formats and Procedures for the Evaluated
  Nuclear Data Files ENDF/B-VI, ENDF/B-VII and ENDF/B-VIII*, Brookhaven National
  Laboratory (2023), doi:10.2172/2007538. Cited as ENDF-6: Ch. 4 (MF 4 Legendre
  coefficients), §34.1-34.3 (MF 34, its frame flag and its L=0 sections).
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: §8.12 (two-body kinematics), as used on
  {doc}`./tabulated-angular-perturbation`.
- `TODO(ref)`: a reference for the SLSQP method (D. Kraft) used in the projection.
