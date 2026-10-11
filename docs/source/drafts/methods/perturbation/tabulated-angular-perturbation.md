# Perturbing a tabulated angular distribution

```{admonition} At a glance
:class: note
**Computes:** a sample of a tabulated angular distribution $f(\mu, E)$, given one factor
per Legendre order and incident-energy bin drawn from an MF 34 covariance.

**Applies to:** MF 4 with LTT=2, and the tabulated half of LTT=3; elastic and discrete
inelastic scattering (MT 2, MT 51-90) when MF 34 is in the other frame.

**Entry points:** {py:func}`kika.sampling.model_perturbation.perturbFromModel`,
{py:func}`kika.nuclear_data.model.perturbation.applyTabulatedFactors`,
{py:func}`kika.nuclear_data.model.angular_tables.perturbTabulatedAngular`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

MF 4 of an ENDF (Evaluated Nuclear Data File) evaluation gives the angular distribution
of the emitted neutron, $f(\mu, E)$ with $\int_{-1}^{1} f\,d\mu = 1$, in one of three
representations (ENDF-6 Ch. 4): Legendre coefficients (LTT=1), tables of $f$ against
$\mu$ (LTT=2), or Legendre below a transition energy and tables above it (LTT=3). A table
is stated at a list of incident energies $E_j$, each on its own cosine nodes
$\mu_1 = -1 < \dots < \mu_n = 1$, with an interpolation law (INT) for each panel
$[\mu_i, \mu_{i+1}]$ and one for the incident-energy axis.

MF 34 states the covariance of the Legendre coefficients

$$
a_\ell(E) = \int_{-1}^{1} f(\mu, E)\, P_\ell(\mu)\, d\mu ,
\qquad
f(\mu) = \sum_\ell \frac{2\ell+1}{2}\, a_\ell\, P_\ell(\mu),
$$ (eq-tabulated-angular-perturbation-moments)

for orders $\ell \ge 1$, relative, on incident-energy bins that may differ from order to
order (ENDF-6 §34.1). Its flag LCT says whether those coefficients are in the laboratory
(LAB) or centre-of-mass (CM) frame, and it need not be the frame MF 4 uses.

A sample of that covariance is, for every order $\ell$ the covariance names and every
bin $k$ of that order, a factor $c_{\ell k}$ (drawing the factors is not part of this
page). A table has no $a_\ell$ to multiply. This page is how kika applies the factors to
it anyway.

## What it computes

Input: the tabulated distribution, the set $\mathcal{L}$ of perturbed orders, and per
order the bin edges and the factors. Output: a distribution of the same kind, tabulated
at the same incident energies (plus the repeated energies a bin edge needs), each table
on its own nodes and under its own law, where at every incident energy

$$
f'(\mu_i) = f(\mu_i) + \sum_{\ell \in \mathcal{L}} \frac{2\ell+1}{2}\,(c_\ell - 1)\,
a_\ell\, P_\ell(\mu_i), \qquad i = 1, \dots, n,
$$ (eq-tabulated-angular-perturbation-result)

with $a_\ell$ the moments of the table itself, {eq}`eq-tabulated-angular-perturbation-moments`,
and $c_\ell$ the factor of the bin that contains $E$. This is the table of the
distribution whose orders in $\mathcal{L}$ are scaled by $c_\ell$ and whose every other
order is untouched: split $f = \sum_{\ell\in\mathcal{L}} \frac{2\ell+1}{2} a_\ell P_\ell
+ r$, scale the sum, keep $r$. The remainder $r$ (the higher orders, the forward peak) is
never built, because the correction is added to the evaluator's table.

## Method

**1. The moments, exactly.** Each panel is integrated on its own. On a lin-lin panel $f$
is linear, so $f P_\ell$ is a polynomial of degree $\ell + 1$; on a histogram panel (INT=1)
$f$ is the constant $f(\mu_i)$ and $f P_\ell$ has degree $\ell$. Gauss-Legendre with

$$
q = \left\lceil \frac{\ell_{\max} + 2}{2} \right\rceil
$$ (eq-tabulated-angular-perturbation-points)

points per panel integrates both with no truncation error, so $a_\ell$ is exact to
rounding. No normalisation is applied: $a_0$ is the table's own integral.

**2. The correction.** With $\Delta_\ell = (c_\ell - 1)\,a_\ell$, the Legendre series
$\sum_{\ell\in\mathcal L} \frac{2\ell+1}{2}\Delta_\ell P_\ell$ is evaluated at the table's
nodes and added, {eq}`eq-tabulated-angular-perturbation-result`. The output keeps the
nodes, the panel laws and the $(\mathrm{NBT}, \mathrm{INT})$ pairs of the input. Where
every factor is 1 the table is passed through unchanged, so the result equals the
evaluation bit for bit.

**3. Incident energy.** The factors are piecewise constant in $E$, and an ENDF table states
a step as a repeated incident energy. The union of every order's bin edges is collected.
An edge strictly inside the span of tabulated energies becomes a repeated energy:

- if the evaluation already tabulates that energy twice, nothing is inserted;
- if it tabulates it once, that table is duplicated;
- if it falls between $E_j$ and $E_{j+1}$, the table at the edge is built and inserted
  twice. Under a lin-lin energy law it is

  $$
  f(\mu, E) = (1-t)\, f_j(\mu) + t\, f_{j+1}(\mu), \qquad t = \frac{E - E_j}{E_{j+1} - E_j},
  $$ (eq-tabulated-angular-perturbation-energy)

  on the union of the two cosine grids, which is exact for two lin-lin tables (a sum of
  piecewise-linear functions is piecewise linear on the union of their breakpoints).
  Under a flat energy law it is a copy of $f_j$.

Each incident energy then takes, per order, the factor of the bin that contains it, with
one exception: the lower copy of a repeated energy takes the factor of the bin that ends
there. Outside an order's bins the factor is 1. The table at each energy is then
perturbed by steps 1 and 2.

**4. A covariance in the other frame.** When MF 34 and MF 4 disagree on the frame, the
factors scale the coefficients of the covariance's frame,

$$
b_\ell = \int_{-1}^{1} f(\mu)\, P_\ell\big(m(\mu)\big)\, d\mu ,
$$ (eq-tabulated-angular-perturbation-other-moments)

where $m$ maps the table's cosine to the cosine of the same direction in the covariance's
frame. The shift comes back to the table's nodes in closed form,

$$
f'(\mu_i) = f(\mu_i) + \sum_{\ell \in \mathcal{L}} \frac{2\ell+1}{2}\,(c_\ell - 1)\, b_\ell\,
P_\ell\big(m(\mu_i)\big)\, m'(\mu_i).
$$ (eq-tabulated-angular-perturbation-other-result)

For two-body neutron scattering $A(n,n')A^*$, with $\gamma$ the ratio of the
centre-of-mass speed to the emitted neutron's speed in CM,

$$
\gamma(E) = \frac{1}{A} \sqrt{\frac{E}{E + Q\,(A+1)/A}}, \qquad
\mu_L = \frac{\mu_C + \gamma}{\sqrt{1 + 2\gamma\mu_C + \gamma^2}}, \qquad
\frac{d\mu_L}{d\mu_C} = \frac{1 + \gamma\mu_C}{\left(1 + 2\gamma\mu_C + \gamma^2\right)^{3/2}},
$$ (eq-tabulated-angular-perturbation-kinematics)

with $A$ the target's mass in neutron masses (AWR) and $Q$ the reaction's QI, zero for
elastic (NJOY 2016 manual §8.12, Eqs. 309 and 311, where $\gamma = 1/R$; for elastic
scattering, ENDF-6 App. D, Eqs. D.71-D.72). For a CM table and a LAB covariance,
$m(\mu) = \mu_L(\mu)$ and $m' = d\mu_L/d\mu_C$. The reverse case uses the inverse map,
$\mu_C = -\gamma(1-\mu_L^2) + \mu_L\sqrt{1 - \gamma^2(1-\mu_L^2)}$, and
$m' = 1/(d\mu_L/d\mu_C)$. The integrand of {eq}`eq-tabulated-angular-perturbation-other-moments`
is smooth inside a panel but not a polynomial, so it takes $\max(q, 16)$ Gauss points per
panel instead of $q$. The map is one-to-one only for $\gamma < 1$; a node with $\gamma \ge 1$
(at an inelastic threshold) is left as the evaluation has it.

### Derivation

*Why {eq}`eq-tabulated-angular-perturbation-other-result` is exact.* Let $\nu = m(\mu)$
be the cosine in the covariance's frame and $g(\nu)$ the distribution there, so that
$g(\nu)\,d\nu = f(\mu)\,d\mu$. Then $b_\ell = \int g(\nu) P_\ell(\nu)\,d\nu$, which is
{eq}`eq-tabulated-angular-perturbation-other-moments` after the change of variable. The
perturbed distribution in that frame is $g + \Delta g$ with
$\Delta g(\nu) = \sum_{\ell\in\mathcal L} \frac{2\ell+1}{2}(c_\ell - 1) b_\ell P_\ell(\nu)$:
by orthogonality its moments are $c_\ell b_\ell$ for $\ell \in \mathcal{L}$ and $b_\ell$
for every other order. Carried back as a density,
$\Delta f(\mu) = \Delta g(m(\mu))\, m'(\mu)$, which is
{eq}`eq-tabulated-angular-perturbation-other-result`. A finite series in one frame is an
infinite series in the other, which is why the Legendre representation has to solve for a
truncated image; a table is only ever evaluated, so it needs none, and the result is the
limit that solve converges to.

*Why the integral is kept in the continuum and not on the table.* $\int_{-1}^{1} P_\ell\,
d\mu = 0$ for $\ell \ge 1$, and in the other frame $\int \Delta f\,d\mu = \int \Delta g\,d\nu
= 0$ for the same reason. But the written table interpolates the correction lin-lin (or
holds it constant on a histogram panel) between nodes, and that interpolant of a
polynomial does not integrate to zero. The difference shrinks with the node spacing.

## Conventions and choices

- **The sample is written as a table on the evaluation's own nodes.** The sample and the
  evaluation then differ only by the perturbation. The evaluator tabulated because a
  Legendre series of the distribution needs many orders, and NJOY (the processing code)
  tabulates in $\mu$ again anyway.
- **The correction is added; the scaled orders are not resummed.** Tabulating
  $\sum_{\ell\in\mathcal L} \frac{2\ell+1}{2} c_\ell a_\ell P_\ell$ would throw away $r(\mu)$,
  the forward peak included, and the sample would not reproduce the evaluation even with
  every factor at 1.
- **The perturbed orders are the ones the covariance names.** There is no truncation order
  to choose: an order MF 34 says nothing about is not changed.
- **The projection is exact panel by panel, and not one quadrature over $[-1, 1]$.** A
  lin-lin table has a kink at every node, a single Gauss rule over the whole range converges
  slowly on it, and the error would land on $a_\ell$, the number that gets multiplied.
- **The covariance is relative,** so $\Delta_\ell = (c_\ell - 1)\,a_\ell$ is proportional to
  $a_\ell$ and vanishes where $a_\ell \approx 0$. The $a_\ell$ is kika's projection of the
  table. If the evaluator built MF 34 against a different set of coefficients, the absolute
  shift kika applies differs from theirs.
- **In the other frame, the factors act where the covariance states them, and only the shift
  comes back.** Applying LAB factors to CM coefficients would perturb a different quantity
  from the one the covariance describes. Converting the whole distribution and back would
  lose the bit-for-bit identity at factors of 1.
- **No renormalisation.** The change of $\int f'\,d\mu$ is reported (see *Diagnostics*),
  never corrected: rescaling the table would move every order, including those the
  covariance does not name.
- **Positivity is reported, not enforced, by default,** as on the Legendre path. With lin-lin or
  histogram panels, $f' \ge 0$ at every node is $f' \ge 0$ everywhere, so the count of
  negative nodes is exact. `perturbFromModel(angularPositivity="repair")` instead replaces
  the shifts $\Delta_\ell$ by the nearest ones (least squares on the vector of shifts) that
  make every node non-negative. Only the perturbed orders move, in the covariance's frame when it is the
  other one. `"report"` is the default because whether a sample may carry a negative
  probability is a decision about the sample, not about the arithmetic.
- **In LTT=3, both halves get the same factors.** The Legendre half is perturbed by the
  Legendre path and the tabulated half by this one, so a bin that spans the transition
  energy is applied on both sides of it.
- **Where an order's bins end inside the table, the default is a step** (`coverageEdges="step"`):
  the outermost edge is a repeated energy like any other, so the factor holds up to the
  edge and the evaluation resumes beyond it. `"ramp"` inserts no energy at an order's first
  and last edge, and the change runs lin-lin from the last perturbed energy to the first
  unperturbed one. It exists to compare against the file-level ENDF sampler
  ({py:mod}`kika.sampling.endf_perturbation`), which does that. `perturbFromModel` always
  uses `"step"`.

## Limits and failure modes

Refused:

- a factor for order 0, which is the cross-section magnitude and not a shape coefficient
  (`ValueError`);
- a table whose $\mu$ panels use a law other than histogram or lin-lin
  (`NotImplementedError`): a logarithmic law makes the integrand of
  {eq}`eq-tabulated-angular-perturbation-moments` transcendental;
- a table that does not span $[-1, 1]$ (to $10^{-9}$), is not ascending in $\mu$, or has
  fewer than two nodes (`ValueError`);
- a bin edge between two tabulated energies when the energy law is neither lin-lin nor
  flat, or when either neighbouring table is not lin-lin in $\mu$ (`NotImplementedError`):
  the table there would be a central value no evaluation stated;
- a frame change for a reaction other than MT 2 and MT 51-90, which are not two-body neutron
  scattering, or without the target's AWR (`ValueError`);
- orders of one reaction stated in different frames, factors without a grid or a grid
  without factors, a grid that is not strictly ascending, or a distribution with no
  tabulated region (`ValueError`).

Reported instead of done: a node with $\gamma \ge 1$ is left unperturbed and counted; a
negative $f'$ is left as computed and counted; the change of the integral is measured and
left.

Limits of the method: the correction is exact at the nodes and interpolated between them,
so a coarse cosine grid leaves a larger error on the moments and on the integral (see
*Validation*); a relative covariance cannot perturb an order whose $a_\ell$ is zero; and
the absolute shift depends on the projection, as stated under *Conventions and choices*.

## Validation

The tests are in
`kika/nuclear_data/model/tests/test_perturbing_a_tabulated_angular_distribution.py` unless
named otherwise. All values were measured on 2026-10-10.

**Against the Legendre path.** The Legendre coefficients of Fe-56 elastic scattering (the
test's JEFF-4.0 fixture, orders 1-6 perturbed on that file's MF 34 grids, factors
$1 + 0.05\cos(k + \ell)$) are tabulated on $n$ nodes $\mu_i = \cos\theta_i$, uniform in
$\theta$, and both representations are perturbed with the same factors. They can differ only
because the projection of a lin-lin table is not the coefficients it was tabulated from. With
$\delta$ the largest such difference in any $a_\ell$ (the *floor*), the test requires
$\max_i |f'_{\text{table}} - f'_{\text{Legendre}}| \le \max|c-1| \cdot \delta \cdot
\sum_{\ell=1}^{6} \frac{2\ell+1}{2}$, times $(1+\gamma)/(1-\gamma)^2$ (the largest Jacobian)
in the other frame.

| Case | Worst difference in $f'$ | Bound | Floor $\delta$ | Integral change | Test |
|---|---|---|---|---|---|
| 401 nodes, same frame | 3.1·10⁻⁵ | 4.2·10⁻⁴ | 3.5·10⁻⁴ | 4.5·10⁻⁶ | `::test_the_two_appliers_agree_to_the_projection_floor` |
| 21 nodes, same frame | 1.2·10⁻² | 0.16 | 0.13 | 1.7·10⁻³ | `::test_a_coarse_mu_grid_raises_the_projection_floor` |
| 401 nodes, LAB covariance on CM, A = 55.5 | 3.0·10⁻⁵ | 4.4·10⁻⁴ | 3.5·10⁻⁴ | | `::test_the_two_appliers_agree_across_frames_to_the_projection_floor` |
| the same, A = 11.9 | 4.0·10⁻⁵ | 5.4·10⁻⁴ | 3.5·10⁻⁴ | | same test |

Applying the LAB factors to the CM table instead misses by 1.7·10⁻² (A = 55.5) and
9.0·10⁻² (A = 11.9); the test requires the miss to exceed ten times the bound.

**The closed form against the padded solve.** For A = 11.9, 55.5 and 236, in both
directions, {eq}`eq-tabulated-angular-perturbation-other-result` agrees with the Legendre
path's truncated solve to 2·10⁻¹³ to 8·10⁻¹³ of the largest correction (tolerance 10⁻¹¹),
`::test_the_closed_form_is_what_the_padded_solve_converges_to`.

**A real evaluation, end to end.** The tables of U-238 elastic scattering from JEFF-4.0
(39 incident energies, up to 91 cosines, CM) with the MF 34 of U-238 elastic scattering
from ENDF/B-VIII.1 (orders 1 and 2, LAB): eight samples written to ENDF, read back and
projected onto the LAB moments reproduce $(c_\ell - 1)\,b_\ell$ to 1.4·10⁻⁵ on the
perturbed orders and move orders 3-6 by at most 2·10⁻⁷ (tolerances 10⁻⁴ and 10⁻⁵), what
ENDF's seven significant digits leave.
`kika/sampling/tests/test_perturbing_a_tabulated_distribution_from_the_model.py::test_a_real_covariance_from_another_library_comes_back_out_of_the_tables`.

**Exact cases.** Factors of 1 give the table back bit for bit, in either frame
(`::test_every_factor_one_gives_the_table_back_bit_for_bit`,
`::test_every_factor_one_in_another_frame_gives_the_table_back_bit_for_bit`); the moments
of a two-node lin-lin table are exact to 10⁻¹⁵
(`::test_the_projection_of_a_linlin_table_is_exact`); $f = \frac12(1 + 3a_1\mu)$ with
$c_1 = 1.1$ comes out as $\frac12(1 + 3.3a_1\mu)$ to 10⁻¹⁴
(`::test_an_analytic_case_comes_out_exact`); a node at an inelastic threshold is left
alone and counted (`::test_a_node_at_an_inelastic_threshold_is_left_alone_and_counted`).
The energy rules of step 3 are checked value by value in
`::test_an_edge_between_two_tables_gets_the_interpolated_table_twice`,
`::test_an_edge_on_a_tabulated_energy_duplicates_that_table` and
`::test_coverage_that_ends_inside_the_table_steps_or_ramps`.

## Diagnostics

{py:func}`~kika.nuclear_data.model.perturbation.applyTabulatedFactors` returns, with the
perturbed distribution:

- `per_order`: per order, the smallest and largest factor applied and how many tables it
  touched (`min_factor`, `max_factor`, `n_scaled`). `n_scaled` of 0 means the order's bins
  do not reach the tables.
- `n_inserted`: incident energies added for bin edges.
- `orders_absent`: always empty, since a table carries every order.
- `n_negative_nodes` and `min_p`: incident energies whose perturbed table is negative
  somewhere, and the most negative value. **Worth a look whenever it is not zero**: under
  `"report"` the negative values are written as computed. Strongly forward-peaked tables,
  where $f(-1)$ is small, go negative under modest factors on $a_1$.
- `max_integral_change`: the worst $|\int f'\,d\mu / \int f\,d\mu - 1|$ over the tables,
  under each table's own law. In the Fe-56 comparison of *Validation* it is 4.5·10⁻⁶ on
  401 nodes and 1.7·10⁻³ on 21; a larger value points to a coarse cosine grid.
- `positivity_events`: one entry per repaired table under `"repair"` (`energy`,
  `min_before`, `min_after`, `max_delta_change`, the last being the largest change made to
  a shift $\Delta_\ell$).
- `frame`, when there is a frame change: `direction`, `awr`, `q`, `n_converted`,
  `max_gamma` and `n_degenerate`. **`n_degenerate` above zero** means nodes at an
  inelastic threshold were left unperturbed.

In a run of `perturbFromModel`, each sample's `applied` entry for an MF 34 component
carries the same quantities under `tables` (`min_p`, `n_negative_nodes`,
`max_integral_change`, `n_repaired`, and `frame`).

## Implementation

- `kika.nuclear_data.model.angular_tables`: one table at one incident energy. The panel
  projection ({py:func}`~kika.nuclear_data.model.angular_tables.legendreMoments`), the
  correction ({py:func}`~kika.nuclear_data.model.angular_tables.perturbTabulatedAngular`),
  the integral under the table's law
  ({py:func}`~kika.nuclear_data.model.angular_tables.tableIntegral`).
- `kika.nuclear_data.model.perturbation`: the incident-energy dimension. Finding the
  tabulated regions, inserting the repeated energies, interpolating a table at an edge,
  the bin rule, and the diagnostics (`applyTabulatedFactors`).
- `kika._legendre_frames`: the kinematics, the cosine maps and their Jacobians, and the
  column $\frac{2\ell+1}{2} P_\ell(m(\mu)) m'(\mu)$ of a unit shift (`correctionBasis`);
  `FrameConversion` holds the frames, AWR and Q of one reaction.
- `kika.sampling.perturbation_set`: dispatch. A Legendre region goes to the Legendre path,
  a tabulated region to this one, and an LTT=3 distribution to both.
- `kika.sampling.mf4_positivity`: the optional repair
  (`repair_tabulated_positivity`).

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual: Data Formats and Procedures for the Evaluated
  Nuclear Data Files ENDF/B-VI, ENDF/B-VII and ENDF/B-VIII*, Brookhaven National
  Laboratory (2023), doi:10.2172/2007538. Cited as ENDF-6: Ch. 4 (MF 4), §34.1 (MF 34),
  App. D (two-body kinematics).
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791. Cited as NJOY 2016 manual: §8.12.
