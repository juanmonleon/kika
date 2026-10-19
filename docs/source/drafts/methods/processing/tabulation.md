# Tabulating secondary distributions for Monte Carlo

```{admonition} At a glance
:class: note
**Computes:** from the evaluated distribution of one product, the tables a Monte Carlo
code samples: normalised densities on lin-lin (or histogram) grids with their cumulative,
in $\mu$ and in $E'$, for angular distributions, energy spectra, Kalbach-Mann and
correlated energy-angle distributions.

**Applies to:** the model's distribution forms (`angularTwoBody`, `uncorrelated`,
`KalbachMann`, `energyAngular`, `angularEnergy`) as decoded from ENDF-6 MF 4, MF 5, MF 6
and MF 15 or from GNDS; neutron and photon products in the OpenMC HDF5 writer, and the
light charged particles as well for an ACE table; charged particles and recoils in the
heating.

**Entry points:** {py:func}`kika.processing.tabulation.tabulateDistribution`,
{py:class}`kika.processing.tabulation.TabulationContext`,
{py:class}`kika.processing.tabulation.TabulationOptions`,
{py:func}`kika.processing.tabulation.tabulateAngular`,
{py:func}`kika.processing.tabulation.tabulateEnergy`,
{py:func}`kika.processing.tabulation.tabulateKalbachMann`,
{py:func}`kika.processing.tabulation.tabulateEnergyAngular`,
{py:func}`kika.processing.tabulation.tabulateAngularEnergy`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

An evaluation states the distribution of a secondary particle in the form the evaluator
chose: Legendre coefficients $a_\ell(E)$ for an angular distribution, a parametrised law
(evaporation, Madland-Nix) or a table for an energy spectrum, Kalbach-Mann parameters, or
a joint density in $(\mu, E')$ (ENDF-6 §4, §5, §6). A continuous-energy Monte Carlo code
samples tables: at each incident energy $E$, a density in $\mu$ or $E'$ that is linear (or
constant) between nodes, with its cumulative, so that a random number is inverted on the
cumulative directly. NJOY's ACER makes those tables for an ACE file, and OpenMC's HDF5
files carry the same ones (MacFarlane et al. 2017, ACER chapter). kika makes them on the
model, so the format writer only serialises them.

Throughout, $\mu$ is the cosine of the emission angle, $E'$ the outgoing energy (eV),
and an angular distribution in Legendre form is

$$
f(\mu) = \sum_{\ell=0}^{N} \frac{2\ell+1}{2}\, a_\ell\, P_\ell(\mu), \qquad a_0 = 1.
$$ (eq-tabulation-legendre)

## What it computes

For each incident energy of the evaluated form, one table $\{x_j, p_j, c_j\}$ in $x = \mu$
or $x = E'$: nodes $x_j$, a density $p_j \ge 0$ normalised to 1, and its cumulative
$c_j$, with $c_1 = 0$ and $c_n = 1$ (the model's `xs_pdf_cdf1d`). Between nodes the
density is linear (lin-lin) or constant (histogram). For a lin-lin table

$$
c_j = \frac{1}{Z}\sum_{l<j} \frac{(p_l + p_{l+1})(x_{l+1} - x_l)}{2}, \qquad
Z = \sum_{l<n} \frac{(p_l + p_{l+1})(x_{l+1} - x_l)}{2},
$$ (eq-tabulation-cdf)

and $p_j$ is the stated (or evaluated) density divided by $Z$; for a histogram the
interval area is $p_l(x_{l+1}-x_l)$. When the first $n_d$ points of a spectrum are
discrete lines (ENDF's ND), their weights come first in the cumulative and the continuum
follows: $c_j = \sum_{l\le j} w_l/Z$ over the lines, then the lines' total plus
{eq}`eq-tabulation-cdf` over the continuum, with $Z$ the sum of both.

The result is a new form under the `MonteCarlo` label, of the kind the evaluated one
maps to:

| evaluated | Monte Carlo |
|---|---|
| `angularTwoBody`, elastic (MT 2) | `angularTwoBody` of $\mu$ tables |
| `angularTwoBody`, a discrete level (MT 51-90, 600-648, 650-698, 700-748, 750-798, 800-848, 875-890) with a Q value | `uncorrelated`: $\mu$ tables and a level energy law |
| `angularTwoBody`, a photon with a Q value (two-body capture) | `uncorrelated`: $\mu$ tables and a primary photon line |
| `uncorrelated` | `uncorrelated` of $\mu$ and $E'$ tables, or the law kept; a sum of weighted spectra becomes a mixture |
| `KalbachMann` | `KalbachMann` with $f$ as tables and $r$, $a$ on $f$'s grid |
| `energyAngular`, `angularEnergy` | `energyAngularMC`: $E'$ tables, and a $\mu$ table at each $E'$ |

## Method

### Refinement to a tolerance

A function $f$ that can be evaluated anywhere (a Legendre series, a parametrised
spectrum) is tabulated by bisection from a set of seed points. In each pass every open
interval $[x_l, x_{l+1}]$ is tested at $t = 1/4$, $1/2$ and $3/4$ of its width:

$$
\big| f(x_t) - \big[f(x_l) + t\,(f(x_{l+1}) - f(x_l))\big] \big|
\le r_\text{tol}\,|f(x_t)| + a_\text{tol}\,\max_j |f(x_j)|,
$$ (eq-tabulation-criterion)

where the maximum runs over every value computed so far. An interval that fails at any of
the three points is split at its midpoint, and its two halves stay open; the others are
closed. An interval narrower than four units in the last place of its ends is not split.
Refinement stops when no interval is open, and raises when the next pass would exceed
`maxPoints` (2000 by default) nodes in one table. The defaults are
$r_\text{tol} = a_\text{tol} = 10^{-3}$.

### Angular distributions

- **Legendre: refinement.** The series {eq}`eq-tabulation-legendre` is evaluated exactly
  and refined by {eq}`eq-tabulation-criterion` at a quarter of the tolerances
  ($r_\text{tol}/4$, $a_\text{tol}/4$), from $\max(3, s, 4N+1)$ equally spaced seeds on
  $[-1, 1]$, where $s$ is `angularSeeds` (3 by default) and $N$ the order of the series.
- **Legendre: moment control.** The first two moments of the table, integrated exactly on
  its lin-lin density,
  
  $$
  \langle \mu^k \rangle_\text{table} = \sum_l \int_{\mu_l}^{\mu_{l+1}} \mu^k\, p(\mu)\, d\mu,
  \qquad k = 1, 2,
  $$ (eq-tabulation-moments)

  must match those of the series within `angularMomentAtol` ($10^{-4}$ absolute). The
  series' moments are exact: $\langle\mu\rangle = a_1$ and
  $\langle\mu^2\rangle = 1/3 + 2a_2/3$ when $a_0$ exceeds $\sum_{\ell\ge1}(2\ell+1)|a_\ell|$
  (the density is then positive, since $|P_\ell| \le 1$); otherwise the polynomial is
  integrated piecewise between its real roots in $[-1, 1]$, over the intervals where it
  is positive (with `negative="clip"`). A table that misses gets the midpoints of the
  intervals with the largest integrated discrepancy between the density and its chord at
  $t = 1/4, 1/2, 3/4$, the fewest that carry half of the total, and is checked again, for
  at most 30 rounds.
- **Legendre: thinning.** The table is then thinned
  ({py:func}`kika.algebra.thin_common_grid`) to three quarters of $r_\text{tol}$,
  relative to the density alone, with an absolute floor of $10^{-7}$ on the normalised
  density, keeping $\mu = -1$, 0 and 1:

  $$
  \lvert \tilde p(\mu_j) - p(\mu_j) \rvert \le \tfrac{3}{4}\, r_\text{tol}\, p(\mu_j) + 10^{-7}.
  $$ (eq-tabulation-angular-thin)

  The thinned table is kept only if its moments still pass the moment control; otherwise
  the refined one is. Refinement at a quarter and thinning at three quarters keep the
  table within $r_\text{tol}$ of the series. An isotropic series stays at three points; a
  forward-peaked one keeps tens of points.
- **Tabulated.** The stated points are kept. A table under another interpolation law is
  converted to lin-lin first ({doc}`../algebra/tabulated-functions`, tolerance
  $10^{-4}$); a histogram stays a histogram.
- **Regions.** A distribution in two regions (ENDF LTT=3: Legendre below, tabulated
  above) is flattened into one list of incident energies. The boundary energy, which both
  regions state, keeps both functions: the region above at the boundary energy, the
  region below at the largest floating-point number under it.
- **Isotropic.** An isotropic distribution stays isotropic and no table is written, since
  OpenMC samples $\mu$ uniformly where the angle is absent. Two exceptions carry an
  explicit table: the elastic (MT 2) and the two-body photon get a two-point table
  $p = 1/2$ on $[-1, 1]$ at $10^{-5}$ eV and 20 MeV.

### Energy spectra

- **Tables** (MF 5 LF=1, MF 6 LAW=1 NA=0, MF 15): the stated points, normalised with
  {eq}`eq-tabulation-cdf`, histogram or lin-lin as stated, with their discrete lines.
- **Lines with a lin-lin continuum.** A table with discrete lines and a lin-lin
  continuum is written as lines followed by a histogram continuum. Each lin-lin panel
  $[a, b]$ with end densities $p_a$, $p_b$ is cut into
  $n = \lceil \sqrt{\lvert p_b - p_a\rvert (b - a) / (24\,\delta)}\,\rceil$ equal
  subpanels, $\delta$ = `mixedCdfAtol` ($10^{-6}$), and each subpanel into two constant
  pieces, split at its midpoint, with the densities of the linear density at one sixth
  and five sixths of the subpanel. The histogram then keeps each subpanel's mass and its
  first two moments exactly, and its cumulative is within $\delta$ of the lin-lin one.
  The size of this projection is set by $\delta$, and `maxPoints` does not cap it. A
  histogram continuum of a single panel is cut the same way, so that it has at least two.
- **Maxwellian fission, evaporation and Watt** (LF=7, 9, 11): kept as laws; OpenMC samples
  them analytically.
- **General evaporation** (LF=5) and **Madland-Nix** (LF=12), which OpenMC does not sample:
  tabulated at the incident energies their parameters are given at. The seeds of the
  general evaporation $P(E'|E) = g(E'/\theta(E))/\theta(E)$ are $\theta(E)\,x_i$ over the
  abscissae $x_i$ of $g$, up to $E - U$; when $g$ is a histogram, the table is that
  histogram, $g$ evaluated at each interval's midpoint, because a lin-lin refinement of a
  step never converges. The seeds of Madland-Nix are $E' = 0$ and 63 logarithmically
  spaced points over nine decades below the law's upper display energy (`energySeeds`,
  64). Both are then refined by {eq}`eq-tabulation-criterion`. At an incident energy below
  the law's support the spectrum is a histogram concentrated at $E' = 0$, on
  $[0, \max(10^{-9}E, 10^{-11}\text{ eV})]$.
- **Photon lines**: a discrete photon of energy $E_\gamma$ (ACE law 2, LP=0) and a primary
  photon of binding energy $E_\gamma$ (LP=2, which needs the target's mass ratio $A$; the
  emitted energy is then $E_\gamma + E\,A/(A+1)$).
- **N-body phase space** and level energies are kept as laws.
- **A sum of weighted spectra** (ENDF NK > 1) becomes a mixture: one `uncorrelated` term
  per spectrum, each with the same angular tables and its own weight $p_k(E)$.

### Discrete-level and two-body kinematics

A neutron emitted from a discrete level of a two-body reaction gets no $E'$ table: its
energy follows from the Q value, as OpenMC's level-inelastic law states it, with

$$
E_\text{th} = \frac{A+1}{A}\,|Q|, \qquad m = \left(\frac{A}{A+1}\right)^2, \qquad
E' = m\,(E - E_\text{th}) \ \text{(centre of mass)}.
$$ (eq-tabulation-level)

The two-body photon of radiative capture (MF 6 LAW=2 for a photon, H-1 capture) is a
primary photon with binding energy $E_\gamma = Q - Q^2/(2(A+1)m_nc^2)$, $m_nc^2$ the
neutron rest energy.

### Kalbach-Mann

The spectrum $f(E'|E)$ (ENDF MF 6 LAW=1 LANG=2) becomes normalised tables with their
discrete lines. The pre-compound fraction $r$ is interpolated lin-lin onto the outgoing
energies of $f$, and so is the slope $a$ when the evaluator gives it. When only $r$ is
given (NA=1, as in every Kalbach-Mann record of ENDF/B-VIII.1), the slope comes from
Kalbach's systematics (Kalbach 1988; ENDF-6 §6.2.3.2), at the table's incident energy
and at each outgoing energy taken as the emitted energy $\epsilon$:

$$
\begin{aligned}
e_a &= \frac{E\,A_t}{A_t + A_p} + S_a, \qquad
e_b = \frac{\epsilon\,(A_r + A_b)}{A_r} + S_b,\\
X_1 &= \frac{\min(e_a, 130)\,e_b}{e_a}, \qquad
X_3 = \frac{\min(e_a, 41)\,e_b}{e_a},\\
a &= 0.04\,X_1 + 1.8\times10^{-6}\,X_1^3 + 6.7\times10^{-7}\,M\,m\,X_3^4,
\end{aligned}
$$ (eq-tabulation-kalbach)

energies in MeV, with $A_t$, $A_p$, $A_r$, $A_b$ the mass numbers of target, projectile,
residual and emitted particle, $M = 1$ for an incident neutron, and $m = 1/2$ for an
emitted neutron, 1 for a proton, deuteron, triton or helium-3 and 2 for an alpha. The
separation energies $S_a$ (projectile from the compound) and $S_b$ (emitted particle
from the compound) are Kalbach's mass formula (her eq. 4) minus the break-up energy of
the light particle (ENDF-6 Appendix H). As in OpenMC's implementation, mass numbers
are used for the masses and natural carbon is taken as C-12.

### Correlated energy-angle distributions

- **`energyAngular`** (MF 6 LAW=1 LANG=1, Legendre or tabulated in $\mu$ at each $E'$):
  at each $(E, E')$ the magnitude $f_0$ is $a_0$ of the Legendre node, or the integral of a
  tabulated node. The $E'$ table is $f_0$ on the outgoing energies, normalised by
  {eq}`eq-tabulation-cdf`. The $\mu$ table at each $E'$ is the node divided by $f_0$,
  tabulated as an angular distribution above; where $f_0 = 0$ it is isotropic.
- **Joint refinement of `energyAngular`.** A transport code interpolates the spectrum and
  the normalised angular law separately in $E'$, which is not the lin-lin interpolation
  of the joint coefficients $f_\ell(E')$ the evaluation states. When the nodes are
  Legendre, lin-lin in $E'$ and carry no discrete lines, outgoing energies are added at
  midpoints until, on each interval $[E'_a, E'_b]$ with joint midpoint
  $\bar f_\ell = (f_{\ell,a} + f_{\ell,b})/2$,

  $$
  \sum_\ell \tfrac{2\ell+1}{2}\left\lvert \bar f_\ell - \bar f_0\,
  \tfrac{1}{2}\Big(\tfrac{f_{\ell,a}}{f_{0,a}} + \tfrac{f_{\ell,b}}{f_{0,b}}\Big)\right\rvert
  \le r_\text{tol} \sum_\ell \tfrac{2\ell+1}{2}\lvert \bar f_\ell\rvert + a_\text{tol}\, P,
  $$ (eq-tabulation-joint)

  with $P$ the largest $\sum_\ell \tfrac{2\ell+1}{2}\lvert f_\ell\rvert$ of the surface;
  since $\lvert P_\ell(\mu)\rvert \le 1$, the left side bounds the error of the density at
  every $\mu$. An end where $f_0 = 0$ takes the normalised law of the other end. The
  tolerances are the ordinary ones, as ACER's.
- **`angularEnergy`** (MF 6 LAW=7, $\mu$ outermost, laboratory frame): at each $E$ the
  outgoing grid is the union of the $E'$ grids of all the cosines $\mu_k$, and the joint
  density $f(\mu_k, E')$ is each cosine's table read lin-lin on it (zero outside its own
  range). The spectrum is $\int f(\mu, E')\,d\mu$ by trapezoids over the $\mu_k$, and the
  $\mu$ table at each $E'$ is the column $f(\mu_k, E')$, normalised over the $\mu_k$
  (isotropic where the spectrum is zero or there is one cosine). The result has $E'$
  outermost, the order OpenMC's correlated distribution requires.

### Negative densities

A truncated Legendre series can turn negative. With `negative="clip"` (the default), the
nodes where the density is negative are set to zero and the table is renormalised; with
`negative="raise"`, the table is refused. The refinement runs on the series as stated, so
the clipped table is lin-lin between the refined nodes of the signed function.

## Conventions and choices

- **The refinement tolerance is relative to the value and to the table's peak.** The
  absolute term of {eq}`eq-tabulation-criterion` is relative to the largest value of the
  table: a density a thousand times below its peak is not refined to 0.1 % of itself,
  which would put most points where almost no particle is emitted. The bound
  {eq}`eq-tabulation-criterion` holds at every test point.
- **The angular thinning is relative to the density alone.** An absolute term on the
  peak would let the backward tail of a forward-peaked law go, and move its moments; the
  floor of $10^{-7}$ on a normalised density is NJOY's. NJOY's ACER refines a Legendre
  law to $2\cdot10^{-4}$ and thins it to $2\cdot10^{-3}$ relative (MacFarlane et al. 2017,
  ACER); kika refines to a quarter of $r_\text{tol}$ and thins to three quarters, so the
  table it writes stays within $r_\text{tol} = 10^{-3}$ of the series.
- **Seeds against cancellation, thinning against size.** The $4N+1$ seeds keep an
  extremum of a series of order $N$ (at most $N-1$ of them) from sitting alone between
  two seeds, but they are far more points than the shape needs: an order-60 law starts
  at 241 cosines that a few tens reproduce. The thinning removes them, and the moment
  control decides whether the thinned table is accepted.
- **Moments to $10^{-4}$.** A density reproduced to $r_\text{tol} = 10^{-3}$ can move
  $\langle\mu\rangle$ by up to $10^{-3}$; a control at $10^{-4}$ keeps the moments ten times
  inside that, and a tighter one would refuse to thin nearly isotropic tables, whose
  moments are close to zero.
- **Three test points per interval, not one.** A midpoint can sit on the chord by
  coincidence while the function bulges on both sides of it. With a midpoint-only test
  and three seeds, Fe-56's elastic at 1.438 MeV (ENDF/B-VIII.1) would keep $[-1, 0]$ as
  one segment, and the table's mean cosine would be 0.025 instead of $a_1 = -0.073$;
  testing at a quarter, a half and three quarters closes that case (see *Validation*).
- **No Legendre form is written.** Neither ACE nor OpenMC's HDF5 has a Legendre form for
  an angular distribution: ACE has isotropic, 32 equiprobable bins and tabulated
  ($\mu$, pdf, cdf), and OpenMC reads tables. A table is also what a transport code
  samples directly, by inverting its cumulative, where a series that can turn negative
  would need rejection sampling. After the thinning, a table holds about as many numbers
  as the series it comes from.
- **Correlated laws at the ordinary tolerances.** The joint refinement of `energyAngular`
  uses the same $r_\text{tol}$ and $a_\text{tol}$ as every other table, as ACER does.
- **Negative densities are clipped and renormalised.** This is ACER's rule
  (TODO(check): the ACER routine that applies it), and it changes the distribution only
  where the evaluation states an unphysical density. The alternative, refusing, is
  available as `negative="raise"`.
- **Stated tables keep their points.** A tabulated distribution is not re-gridded: the
  evaluator's points are what the evaluation asserts, and lin-lin interpolation between
  them is what the evaluation defines (after conversion of the law when it is not
  lin-lin).
- **Analytic spectra stay analytic where OpenMC samples them.** Tabulating a Maxwellian,
  evaporation or Watt spectrum would add points and an interpolation error to what the
  transport code computes exactly.
- **Kalbach's slope as OpenMC computes it.** The two approximations of OpenMC's
  implementation (mass numbers for masses, C-12 for natural carbon) are kept, so the
  slopes are those of OpenMC's own conversion of the same evaluation.
- **Both sides of a boundary between angular regions are kept.** ENDF states the
  boundary energy in both regions, and their densities may differ there. OpenMC finds the
  interval of an incident energy with a lower-bound search, so a repeated energy would
  select the law below at the exact boundary; moving the law below to the largest
  floating-point number under the boundary keeps both, and OpenMC and kika's own
  interpolation then both use the law above at and beyond the boundary. Only the
  processed table's coordinate moves; the evaluated form is not changed.
- **Lines with a lin-lin continuum become a histogram.** OpenMC 0.16 interpolates the
  discrete entries of such a table on its lin-lin branch; a histogram continuum avoids
  that, and the two-piece projection keeps the mass and the first two moments of every
  panel exactly, so the mean energy and the heating it feeds do not move.

## Limits and failure modes

- `ValueError` when a table would need more than `maxPoints` nodes (the moment control
  and the joint refinement of `energyAngular` included), when the moment control does
  not converge in 30 rounds or the joint refinement in 40, when a density integrates to
  zero, with `negative="raise"` and a negative density (a negative interval between the
  roots of a series included, even where the refined nodes are all positive), when a
  primary photon has no mass ratio, and for an unknown `negative` policy or a
  non-finite or negative `angularMomentAtol`, or a `mixedCdfAtol` that is not positive
  and finite.
- `TypeError` for a form with no tabulation (an angular distribution that is not
  `XYs2d`, `regions2d` or isotropic; an energy form outside the list above; a
  distribution kind outside the table above).
- `NotImplementedError` for Kalbach's systematics with a projectile other than a neutron.
- `None` (no Monte Carlo form) for a recoil product, an `unspecified` or branching
  distribution, and a two-body product that is neither elastic, nor a discrete level with
  a Q value, nor a photon with a Q value. The OpenMC writer then leaves the product out
  and says so in the report.
- Negative densities are counted and reported only for `angularTwoBody` and
  `uncorrelated` forms. In Kalbach-Mann and correlated forms they are clipped (or refused)
  by the same rule, without a report entry.
- The tables are built per incident energy of the evaluation; nothing is added between
  incident energies, which the transport code interpolates.

## Validation

In `kika/processing/tabulation/tests/test_tabulation.py`:

- `test_refine_reaches_the_tolerance`: $e^x$ on $[0, 5]$ refined at
  $r_\text{tol} = 10^{-4}$, $a_\text{tol} = 0$ is reproduced lin-lin within
  $2\times10^{-4}$ (relative) at 10 001 points.
- `test_legendre_isotropic_is_three_points`: an isotropic series gives three points with
  $p = 1/2$.
- `test_forward_peaked_legendre_is_refined_and_reproduced`: a positive series of order 3
  is reproduced within $2\times10^{-3}$ of its peak at 4 001 points, with the default
  tolerances.
- `test_legendre_table_keeps_the_mean_cosine_where_a_midpoint_sits_on_the_chord`: the
  ENDF/B-VIII.1 Fe-56 elastic coefficients at 1.438 MeV give a table whose mean cosine is
  within $2\times10^{-3}$ of $a_1$.
- `test_negative_legendre_is_clipped_and_renormalised`: a series that turns negative is
  clipped to a valid table, and refused with `negative="raise"`.
- `test_histogram_cdf_and_discrete_lines`: two lines and a histogram continuum, with the
  weights and the cumulative of {eq}`eq-tabulation-cdf`.
- `test_angular_regions_retain_both_sides_of_a_boundary`: two regions sharing an energy
  keep both laws, the lower one just under the boundary, and the mean cosine survives
  the HDF5 encoding; `test_angular_boundary_coordinates_match_openmc_lower_bound_without_source_edits`:
  OpenMC's lower-bound search selects the same law as kika's interpolation at and around
  the boundary, to $10^{-14}$, and the evaluated form is unchanged.
- `test_angular_moment_control_matches_clipped_polynomial_integrals`,
  `test_angular_moments_positive_proof_and_clipping_keep_exact_integrals`: the exact
  moments of a clipped series, and the table's within $10^{-6}$ of them;
  `test_angular_moment_control_detects_hidden_negative_interval`: a series negative only
  between nodes is refused with `negative="raise"`;
  `test_angular_moments_catch_midpoint_cancellation_within_capacity`: an order-6 series
  whose midpoint test cancels gets its moments to $10^{-5}$ in under 500 points.
- `test_mixed_histogram_retains_lines_mass_and_two_continuum_moments`: the lines' weight
  and the first two moments of the continuum to $10^{-12}$, the cumulative within
  `mixedCdfAtol`; `test_mixed_histogram_is_a_projection_maxpoints_does_not_drop`: the
  projection is not capped by `maxPoints`;
  `test_mixed_single_histogram_panel_gets_a_transport_safe_second_panel` and
  `test_pure_lines_do_not_enter_the_continuum_interpolation_branch`.
- `test_evaporation_table_from_a_law`: every table of a general evaporation is
  normalised, starts at 0, ends at 1 and is monotone.
- `test_level_reaction_gets_openmcs_kinematics`: the threshold and mass ratio of
  {eq}`eq-tabulation-level` for Fe-56 MT 51.
- `test_kalbach_slope_is_of_order_one_and_grows_with_the_emitted_energy`: the systematic
  slope is finite and increases with the emitted energy (no comparison with OpenMC).
- `test_kalbach_mann_gets_a_slope_on_f_grid`: $a$ is computed on $f$'s outgoing grid.
- `test_energy_angular_legendre_becomes_mc`: a Legendre `energyAngular` becomes valid
  $E'$ and $\mu$ tables.

The angular thinning is exercised by every Legendre test above (the isotropic series
keeps exactly $\mu = -1$, 0 and 1), but no test yet measures a thinned table of a high
order against its series. TODO(check): add one.

Every table in these tests is checked to start at 0, end at 1, be monotone and have a
non-negative density. The OpenMC writer's contract checker rejects a file whose
cumulative does not end at 1
(`kika/hdf5/tests/test_contract.py::test_a_bad_cdf_is_caught`).

## Diagnostics

The conversion report passed in the {py:class}`kika.processing.tabulation.TabulationContext`
receives, per reaction and product, an `approximated` entry
"MT*n* *particle*: *k* tables had negative densities set to zero before normalising"
for `angularTwoBody` and `uncorrelated` forms. It is worth reading: a large *k* means
the evaluation's Legendre series are negative over a wide range, and the tables differ
from the series there. In the OpenMC writer
({py:func}`kika.hdf5.processing.processSuite`, step 4), a product that could not be
tabulated, has no distribution, or has a multiplicity that is not a function is listed as
`lost`, with the reason, and is not written.

## Implementation

- `kika.processing.tabulation.tables`: the options, the refinement, normalisation and
  cumulative, the negative-density rule.
- `kika.processing.tabulation.angular`: Legendre and tabulated $P(\mu|E)$, the moment
  control and the angular thinning, regions; the Legendre density is evaluated by a
  native kernel when it is built (`kika.algebra._polynomial_native`), by NumPy otherwise.
- `kika.algebra.thin`: the thinning.
- `kika.processing.tabulation.energy`: $P(E'|E)$ tables, parametrised laws, photon lines,
  the histogram projection of lines with a continuum.
- `kika.processing.tabulation.kalbach`: Kalbach's slope systematics.
- `kika.processing.tabulation.correlated`: Kalbach-Mann, `energyAngular` and its joint
  refinement, `angularEnergy`.
- `kika.processing.tabulation.distribution`: the dispatch on the evaluated form, the
  level and two-body photon kinematics.
- `kika.hdf5.processing` (step 4 of {py:func}`kika.hdf5.processing.processSuite`) calls the
  dispatch for every neutron and photon product, with the number of discrete lines from
  the MF 6 record; a missing distribution of MT 2, or of a neutron from MT 51-90, is
  taken as isotropic in the centre of mass; an isotropic photon gets an explicit
  two-point histogram over the reaction's energy range. `kika.processing.heating` uses the
  same dispatch for charged particles and recoils. The chain is {doc}`processing-chain`.

## References

- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: ACER.
- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: §4 (MF 4), §5 (MF 5), §6 (MF 6; §6.2.3.2 Kalbach-Mann), Appendix H.
- C. Kalbach, "Systematics of continuum angular distributions: Extensions to higher
  energies", *Physical Review C* 37, 2350-2370 (1988), doi:10.1103/PhysRevC.37.2350.
- P. K. Romano, N. E. Horelik, B. R. Herman, A. G. Nelson, B. Forget and K. Smith,
  "OpenMC: A state-of-the-art Monte Carlo code for research and development", *Annals of
  Nuclear Energy* 82, 90-97 (2015), doi:10.1016/j.anucene.2014.07.048.
