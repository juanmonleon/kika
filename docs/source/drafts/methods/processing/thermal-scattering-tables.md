# Computing thermal inelastic scattering tables

```{admonition} At a glance
:class: note
**Computes:** at one temperature, the incoherent inelastic cross-section $\sigma_\text{in}(E)$
on THERMR's incident grid and, at each incident energy, the outgoing-energy table (pdf and
cdf) with 16 equiprobable cosines at each outgoing energy: the "continuous" representation
of OpenMC's thermal files (ACER `iwt=2`). From a tabulated $S(\alpha,\beta)$, or from the
closed-form free gas of one nuclide.

**Applies to:** the incoherent inelastic law of a thermal scattering evaluation (ENDF-6
MF 7/MT 4, NSUB=12, or the GNDS TNSL form decoded into the model); for the free gas, the
elastic cross-section of an incident-neutron evaluation broadened to the temperature.

**Entry points:** {py:func}`kika.processing.thermal.inelasticTables`,
{py:func}`kika.processing.thermal.freeGasTables`,
{py:class}`kika.processing.thermal.ThermalInelasticOptions`,
{py:func}`kika.hdf5.thermal_processing.processThermal` (the whole OpenMC thermal file).

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

Below a few eV a neutron scatters off a bound atom, and the energy and angle it leaves
with depend on the motion of the lattice or molecule. An evaluation states this through
the scattering law $S(\alpha,\beta,T)$ of the principal atom (ENDF-6 §7.4), a function of
the dimensionless momentum and energy transfers

$$
\alpha = \frac{E + E' - 2\mu\sqrt{E E'}}{A\,kT}, \qquad \beta = \frac{E' - E}{kT},
$$ (eq-thermal-scattering-tables-alpha-beta)

where $E$ and $E'$ are the incident and outgoing neutron energies (eV, laboratory frame),
$\mu$ the cosine of the scattering angle, $A$ the ratio of the principal atom's mass to
the neutron's, $T$ the material temperature and $k$ Boltzmann's constant. For a
symmetric law the double-differential cross-section per principal atom is

$$
\sigma(E \to E', \mu) = \frac{\sigma_b}{2kT}\sqrt{\frac{E'}{E}}\;
S(\alpha, |\beta|)\, e^{-\beta/2},
$$ (eq-thermal-scattering-tables-ddxs)

with $\sigma_b = \sigma_\text{free}\,((A+1)/A)^2$ the bound-atom cross-section. Where
$S$ is not tabulated, ENDF-6 §7.4 replaces it by the short-collision-time (SCT)
approximation, a Gaussian in $\alpha - |\beta|$ with an effective temperature $T_\text{eff}$.
The free gas is the same law for an unbound nucleus, in closed form:

$$
S_\text{fg}(\alpha,\beta) = \frac{1}{\sqrt{4\pi\alpha}}\,
\exp\!\left[-\frac{(\alpha+\beta)^2}{4\alpha}\right].
$$ (eq-thermal-scattering-tables-free-gas)

A Monte Carlo code does not evaluate {eq}`eq-thermal-scattering-tables-ddxs`: it samples
tables. NJOY's THERMR integrates the law into $\sigma(E\to E')$ and equiprobable cosines on
an adaptive grid, and ACER turns that into a normalised outgoing-energy table per incident
energy (MacFarlane et al. 2017, THERMR and ACER chapters). OpenMC's thermal files are
those tables. kika computes them itself, without NJOY. The elastic parts (coherent and
incoherent) are not computed here: they come from the evaluation in closed form, as
{doc}`../formats/endf-special-data` describes.

## What it computes

**From $S(\alpha,\beta)$.** Input: the model's incoherent inelastic law and one temperature
the law tabulates. Output, at each incident energy $E_i$ of THERMR's grid up to the law's
upper energy $E_\text{max}$:

- the cross-section $\sigma_\text{in}(E_i)/n_\text{mix}$, where
  $\sigma_\text{in}(E) = \int\!\!\int \sigma(E\to E',\mu)\,d\mu\,dE'$ is computed per
  principal atom and $n_\text{mix}$ is the number of atom types the OpenMC table mixes;
- the outgoing-energy table $\{E'_j, p_j, c_j\}$, a density $p$ normalised to 1 and
  linear between nodes, with its cumulative $c$;
- at each $E'_j$, 16 cosines $\bar\mu_k$, each the mean of $\mu$ over one sixteenth of the
  angular distribution at $(E_i, E'_j)$:

$$
\bar\mu_k = \frac{16}{\sigma(E\to E')}\int_{\mu_{k-1}}^{\mu_k} \mu\,
\sigma(E \to E', \mu)\, d\mu,
\qquad \int_{\mu_{k-1}}^{\mu_k} \sigma(E\to E',\mu)\,d\mu = \frac{\sigma(E\to E')}{16},
$$ (eq-thermal-scattering-tables-cosines)

with $\mu_0 = -1$, $\mu_{16} = 1$ and $\sigma(E\to E') = \int_{-1}^{1}\sigma(E\to E',\mu)\,d\mu$.

**For the free gas.** Input: the nuclide's mass ratio $A$, its elastic cross-section at
the temperature, and an upper energy (10 eV by default). The distributions are those of
{eq}`eq-thermal-scattering-tables-free-gas` for a free cross-section of 1 b
($\sigma_b = ((A+1)/A)^2$). The cross-section carried is not their integral: it is the
nuclide's elastic cross-section at the temperature, read lin-lin at each incident energy
and divided by $n_\text{mix}$, as THERMR writes it for the free gas.

The result is returned as model forms: the cross-section as an `XYs1d` on the incident
energies, the distribution as an `EnergyAngularMC` (outgoing-energy tables, and per
outgoing energy a set of equiprobable discrete cosines), and the highest incident energy.

## Method

The S(α,β) path reproduces THERMR (`calcem`, `sigl`, `sig`, `terpq`) and ACER
(`acesix` with `iwt=2`) step by step, with their roundings; the reason is under
*Conventions and choices*. The free-gas path keeps THERMR's incident grid and outgoing
panels and replaces the angular integration and the final table by exact ones.

### Inputs from the scattering law

At the requested temperature $T$ the law must hold an $S$ table within $T/500$; there is no
interpolation in temperature. The table is stored as $\ell = \ln S$, with $\ell = -225$
wherever $S \le e^{-225}$. The other inputs are taken from the principal atom: $A$ (by
default its mass over the neutron mass; {py:func}`kika.hdf5.thermal_processing.processThermal`
passes B(3) of MF 7/MT 4 exactly as the tape states it), $\sigma_b$,
$kT_\text{eff} = k\,T_\text{eff}(T)$ (or $kT$ when the law states none), $E_\text{max}$,
whether $S$ is symmetric (`lasym`) and whether it was computed at $T_0 = 0.0253/k$
(`lat`, the model's `calculatedAtThermal`). Here $k = 8.617333262\times10^{-5}$ eV/K.

A secondary atom enters only when it follows the SCT approximation: its mass ratio $A_2$,
its effective temperature, and the weight $\sigma_{b2}/n_\text{p}$, its bound cross-section
divided by the number $n_\text{p}$ of principal atoms per molecule. A secondary
free-gas atom is not used. If $\ell(\alpha_1,\beta_1) > \ell(\alpha_2,\beta_1)$ (a liquid,
where $S$ falls with $\alpha$ at $\beta=0$), the diffusion constant is

$$
c_\text{liq} = \frac{\big[\ell(\alpha_1,\beta_1) - \ell(\alpha_1,\beta_2)\big]\,\alpha_1}{\beta_2^2},
$$ (eq-thermal-scattering-tables-cliq)

and $c_\text{liq} = 0$ otherwise.

### The double-differential cross-section

For given $(E, E', \mu)$, $\alpha$ and $\beta$ follow {eq}`eq-thermal-scattering-tables-alpha-beta`
with $kT$ the temperature of the table, and $\alpha$ is floored at $10^{-6}$. With `lat=1`,
the values used to look up the table are $\tilde\alpha = \alpha\,kT/kT_0$ and
$\tilde\beta = |\beta|\,kT/kT_0$; the factor $e^{-\beta/2}$ always takes the unscaled
$\beta$. With `lasym=1` the lookup uses the signed $\beta$.

- **SCT** when $\tilde\alpha > \alpha_\text{max}$, when $\tilde\beta$ is outside the
  $\beta$ grid, or when $A\tilde\alpha \ge 30$ or $\tilde\beta \ge 30$ and any corner of
  the cell holds $\ell = -225$:

$$
\sigma = \frac{\sigma_b}{2kT}\sqrt{\frac{E'}{E}}\,
\frac{\exp\!\left[-\dfrac{(\alpha-|\beta|)^2\,kT}{4\alpha\,kT_\text{eff}} - \dfrac{|\beta|+\beta}{2}\right]}
{\sqrt{4\pi\alpha\,kT_\text{eff}/kT}},
$$ (eq-thermal-scattering-tables-sct)

  plus the same term with $\sigma_{b2}/n_\text{p}$, $\alpha A/A_2$ and the secondary's
  $T_\text{eff}$ when there is an SCT secondary.
- **Diffusion** when $c_\text{liq}\ne 0$, $\tilde\alpha < \alpha_1$, `lasym=0` and
  $\tilde\beta \le 0.2$: $\ell = \ell(\alpha_1,\beta_1) + \tfrac12\ln(\alpha_1/\tilde\alpha)
  - c_\text{liq}\tilde\beta^2/\tilde\alpha$, floored at $-225$.
- **Table** otherwise. With $i$, $j$ the cells holding $\tilde\alpha$ and $\tilde\beta$
  (moved one node down when they are the last cell), $\ell$ is interpolated by three-point
  quadratics: first in $\alpha$ through nodes $i, i+1, i+2$ on rows $j$, $j+1$, $j+2$,
  then in $\beta$ through the three results. Each quadratic falls back as THERMR's
  `terpq` does: below the first node, constant if the function decreases, else $\ell$
  linear in $\ln x$; above the last node, constant if the function increases, else
  linear; lin-lin by halves when two neighbouring values differ by more than 2; and the
  result is floored at $-225$.

Then $\sigma = \frac{\sigma_b}{2kT}\sqrt{E'/E}\,e^{\ell - \beta/2}$, zero when
$\ell - \beta/2 \le -225$, and any value below $10^{-10}$ is set to zero.

### Angular integration and cosines

At each $(E, E')$ the cosine axis is linearised twice, on a stack of at most 20 points
starting from $\{-1, \mu_0, +1\}$, where

$$
\mu_0 = \frac{E + E' - \big(\sqrt{1+b^2}-1\big)A\,kT}{2\sqrt{EE'}}, \qquad b = |\beta|
\ (\times kT/kT_0 \text{ with } \texttt{lat=1}),
$$ (eq-thermal-scattering-tables-mu0)

replaced by 0.99 when $|\mu_0| > 1 - 10^{-3}$ and by 0 when $E' = 0$. A midpoint (rounded
to 8 significant figures) is accepted when
$|\sigma(\mu_m) - \bar\sigma| \le \tau\,|\sigma(\mu_m)| + \tau\,\sigma_\text{max}/50$, the two
end values differ by at most $\bar\sigma + \sigma_\text{max}/100$ and the interval is
shorter than 0.5, or when the interval is shorter than $10^{-5}$. Here $\tau$ is half the
tolerance `tol`, $\bar\sigma$ the chord value and $\sigma_\text{max} \ge 10^{-3}$ the
largest of the three starting values. The first pass gives
$\sigma(E\to E')$ by trapezoids; below $10^{-32}$ everything at that $E'$ is zero. The
second pass splits the same piecewise-linear $\sigma(\mu)$ into 16 bins of equal area,
each edge found by solving the panel's quadratic area equation (its linear form when the
relative slope is below $10^{-3}$), and stores the mean cosine of each bin
{eq}`eq-thermal-scattering-tables-cosines`, integrated exactly on the linear panels.

### Outgoing-energy grid and cross-section

The incident energies are THERMR's fixed grid of 118 values from $10^{-5}$ to 10 eV
({py:data}`kika.processing.thermal.EGRID`), each rounded to 8 significant figures,
stopping at the first one above $E_\text{max}$ or within $10^{-5}$ eV of it. At each $E$:

1. The first outgoing point is $E' = 0$, where $\sigma(E\to E') = 0$.
2. Panels end at $E' = E \mp \beta_j kT$ for every tabulated $\beta_j$ (with `lat=1`,
   $kT_0$ instead of $kT$; with `lasym=1`, only $E + \beta_j kT$), skipping nodes not above
   the previous one; each node is rounded to 8 figures, and the node at $E' = E$ is moved
   one unit in the 8th figure.
3. Each panel is bisected (stack of 20, midpoints to 8 figures). A midpoint is accepted
   when $\sigma(E\to E')$ is within `tol` of the chord relatively, each cosine within
   `tol` absolutely, and the sum of the 16 cosines within
   $2\,\texttt{tol}\,|\sum_k\bar\mu_k| + 10^{-5}$. A panel whose trapezoid area is below
   $5\times10^{-7}$ is not subdivided, nor the first panel after crossing $E' = E$.
4. $\sigma_\text{in}(E)$ is the trapezoid sum of $\sigma(E\to E')$ over the accepted
   points, rounded to 9 figures. Stored values carry THERMR's rounding (9 figures, 8
   below $10^{-9}$) and cosines are clipped to $[-1, 1]$. The last point, at
   $E + \beta_\text{max}kT$, is set to zero, and the table ends one point after the last
   non-zero $\sigma(E\to E')$.

The incident energies are finally stored as an ACE file states them (MeV to 12
significant figures, read back in eV).

### From THERMR's points to the Monte Carlo table

On the S(α,β) path the points then go through what THERMR leaves on its tape and what
ACER does with it:

1. **The tape.** The density is divided by $\sigma_\text{in}(E)$ and rounded to 7
   significant figures (6 below $10^{-9}$); every number is then rounded as NJOY's
   11-column ENDF writer prints it (9 figures in the fixed form it uses for positive
   values between 0.1 and $10^7$, else 7, 6 or 5 by exponent width). Two outgoing
   energies that differ beyond that precision become equal, and the second is skipped.
2. **ACER `iwt=2`.** The cosines of the first and last points are copied from their
   neighbours (where $\sigma = 0$ they carry no information). Walking up in $E'$, the
   trapezoid area since the last written point is accumulated, and a point is written as
   soon as it exceeds $10^{-6}$ of the (unit) total; intervals below that are merged into
   the next. Each written point carries $E'$, its density, its interval area and THERMR's
   cosines at that $E'$. Then $c_j = \sum_{l\le j} a_l / \sum_l a_l$ and
   $p_j = y_j/\sum_l a_l$. The point $E' = 0$ is never written, so $c_1 > 0$.
3. **OpenMC's form.** The cosines are sorted. When $c_1 > 0$, a point $E' = 0$ with
   $p = c = 0$ and isotropic cosines $-1 + (2k-1)/16$ is put in front, as OpenMC's
   `from_ace` does.

### The free gas

The free-gas path differs from the above in four places.

- **Kernel.** $\sigma = \frac{\sigma_b}{2kT}\sqrt{E'/E}\,S_\text{fg}$ with
  {eq}`eq-thermal-scattering-tables-free-gas`, $\sigma_b = ((A+1)/A)^2$, no floor on
  $\alpha$, and the $\beta$ panels on THERMR's fixed free-gas grid of 45 values from 0 to
  3500 ({py:data}`kika.processing.thermal.FREE_GAS_BETA`).
- **Angular integration, in closed form.** With $u = \sqrt\alpha$, $d\alpha = 2u\,du$
  removes the $1/\sqrt\alpha$ singularity at $E' = E$, $\mu = 1$, and

$$
\sigma(E \to E') = \frac{\sigma_b A}{4E\sqrt\pi}\, M(u_-, u_+), \qquad
M(a,b) = \int_a^b e^{-p(u)^2}\,du, \quad p(u) = \frac{u + \beta/u}{2},
$$ (eq-thermal-scattering-tables-fg-sigma)

  over $u_\pm = (\sqrt{E'} \pm \sqrt{E})/\sqrt{AkT}$ (absolute value for $u_-$), with

$$
M(a,b) = \frac{\sqrt\pi}{2}\Big\{\operatorname{erf}p(b) - \operatorname{erf}p(a)
+ e^{-\beta}\big[\operatorname{erf}q(b) - \operatorname{erf}q(a)\big]\Big\}, \qquad
q(u) = \frac{u - \beta/u}{2}.
$$ (eq-thermal-scattering-tables-fg-mass)

  The differences are taken with `erfc` on tails of one sign, and for $\beta < 0$ the
  second term through the scaled complementary error function, which avoids overflow; an
  interval narrower than $10^{-5}\max(1,b)$, or one where the two terms cancel to below
  $10^{-8}$ of their size, uses an 8-point Gauss-Legendre rule instead. The cosine is
  $\mu(u) = 1 - (u^2 - u_-^2)\,AkT/(2\sqrt{EE'})$. The 16 bin edges are found by
  bisection (48 steps) on $M$, and each bin's mean cosine by adaptive 8-point
  Gauss-Legendre quadrature to $10^{-11}$ of the bin's mass. At $E' = 0$ the cosines
  are the isotropic midpoints.
- **Outgoing grid.** No area threshold stops the bisection (THERMR's $5\times10^{-7}$ is
  0 here); instead, with the scale
  $\sigma_\text{pk} = \frac{\sigma_b A}{4E}\operatorname{erf}\sqrt{E/(AkT)}$, the cosine
  tests are skipped where $\sigma(E\to E') < 10^{-12}\sigma_\text{pk}$ and
  $\texttt{tol}\cdot10^{-8}\sigma_\text{pk}$ is added to the cross-section test.
- **Table.** No tape rounding and no ACER merge: $p$ and $c$ are built from the same
  trapezoid areas on the accepted grid, $c_j = \sum_{l<j} a_l/\sum_l a_l$,
  $p_j = y_j/\sum_l a_l$ (with $a_l$ the area between points $l$ and $l+1$), so that the
  density and its cumulative describe the same mass. The cosines are clipped to
  $[-1, 1]$ and those of the end points are kept as computed. A negative density, or a
  zero total, raises.

The incident energies are each computed independently, in threads (`workers`, by default
up to 8), and the tables do not depend on their number.

## Conventions and choices

- **The S(α,β) path reproduces THERMR + ACER, roundings included.** The official OpenMC
  files are THERMR + ACER output, and a Monte Carlo result changes with the tables it
  samples; reproducing the algorithm (including `sigfig`, the 11-column tape rounding and
  the $10^{-6}$ merge) puts kika's tables on the same outgoing points as the official
  ones, so a difference in a transport result is not a processing artefact. The method's
  own approximations are reproduced with it: the quadratic interpolation of $\ln S$
  ignores the interpolation laws the tape states, the 20-point stacks limit the
  resolution of long panels, and the incident grid is THERMR's fixed one (OpenMC samples
  at the nearest incident energy, so the grid is part of the data).
- **The cross-section is the integral at each incident energy.** THERMR writes
  $\sigma_\text{in}$ Lagrange-interpolated (order 5) onto the principal nuclide's PENDF
  grid, and ACER reads it back lin-lin at the incident energies. kika keeps the direct
  integral, so the thermal table does not depend on the neutron evaluation of the
  principal nuclide. The difference to the official files is at most 0.03 %, and 0.69 %
  at the phonon kink of H in ZrH (see *Validation*). Which of the two is closer to the
  exact integral there has not been measured.
- **Secondary atoms as THERMR treats them.** An SCT secondary enters with
  $\sigma_{b2}/n_\text{p}$, without its own number of atoms per molecule; a free-gas
  secondary is ignored. This keeps the tables equal to NJOY's; whether the missing
  factor is a defect of THERMR has not been settled against an independent reference.
- **No temperature interpolation.** $S(\alpha,\beta)$ is used only at the temperatures it
  is tabulated at, as THERMR requires: interpolating a scattering law in temperature is a
  physics choice that the evaluation does not make.
- **The free gas does not reproduce THERMR.** THERMR integrates the free-gas kernel over
  $\mu$ with the same piecewise-linear panels and the $\alpha \ge 10^{-6}$ floor as for a
  tabulated law, which clips the integrable singularity at $E' = E$, $\mu = 1$; its
  $5\times10^{-7}$ area threshold leaves the low-$E'$ tails of the higher incident
  energies unrefined; and ACER's merge can join a non-zero density to a distant zero, so
  that the table's cdf no longer integrates its pdf. Against an analytic reference, the
  pointwise density of NJOY's free-gas table is off by up to +27 % (H-1, 9.85 eV), and for
  U-238 below $2.5\times10^{-4}$ eV $\sigma$ and $\langle E'\rangle$ are low by 0.45 % and
  0.46 %, independently of `tol` (NJOY version TODO(check), measured TODO(check) date).
  kika integrates the angle in closed form ({eq}`eq-thermal-scattering-tables-fg-mass`),
  refines the tails against the peak scale, and builds pdf and cdf from the same areas.
  The free-gas law, the incident grid, the $\beta$ panels and the cross-section (the
  broadened elastic) are THERMR's.
- **The free-gas cross-section is the broadened elastic, not the kernel's integral.** This
  is what THERMR writes for `iinc=1`: the kernel gives the shape, the evaluation the
  magnitude. The integral is computed and used only as the normalisation of each table.
- **Per molecule, as OpenMC stores it.** THERMR's cross-section is per principal atom;
  dividing by $n_\text{mix}$ is ACER's step, and $n_\text{mix}$ comes from OpenMC's
  registry of table names.

## Limits and failure modes

- `ValueError`: the principal atom's kernel is not a tabulated $S(\alpha,\beta,T)$; the
  temperature is not tabulated within $T/500$; more than one SCT secondary atom;
  {py:func}`kika.processing.thermal.inelasticTables` called with `iinc≠2` or with `emax`
  set (the S(α,β) path takes the evaluation's $E_\text{max}$);
  {py:func}`kika.processing.thermal.freeGasTables` called with `iinc≠1`, a mass ratio
  $\le 0$, `emax` $\le 0$, a temperature outside $(0, 3000]$ K (above 3000 K THERMR
  rescales its incident grid, which kika does not do), an elastic table that is not two
  equal-length 1-d arrays with ascending energies, or one that does not reach an incident
  energy; a free-gas density that is negative or integrates to zero.
- `MemoryError` when one incident energy needs more outgoing points than
  `ThermalInelasticOptions.capacity` (50 000 by default).
- `ImportError` when the compiled kernel `kika.processing._thermal_native` is missing:
  there is no Python fallback, because a pure-Python THERMR takes hours per temperature.
- Not covered by a comparison with the official files: `lasym=1`, `lat=0` (ortho- and
  para-hydrogen and deuterium, methane) and SCT secondaries (benzene, SiO₂). The code
  follows THERMR's branches for them; the tables have not been compared.
- The free-gas checks are on sampled energies, temperatures and mass ratios (see
  *Validation*), not a bound over every $A$, $T$ and $E$.

## Validation

The kernel's pieces are tested in `kika/processing/tests/test_thermal.py`:

- `test_sig_at_a_table_node_is_the_tabulated_s`: at a node $(\alpha_i, \beta_j)$ of a
  synthetic table, the double-differential cross-section equals
  {eq}`eq-thermal-scattering-tables-ddxs` with the tabulated $S$, to $10^{-12}$.
- `test_sigfig_is_njoys`, `test_endf11_reads_back_as_njoy_writes`: NJOY's rounding to
  significant figures and the 11-column ENDF form, on fixed values.
- `test_acer_merges_intervals_below_one_millionth_of_the_area`: intervals carrying less
  than $10^{-6}$ of the area are merged into the next point, and the cdf ends at 1.
- `test_sig_free_gas_is_the_closed_form_kernel`: the free-gas kernel equals
  {eq}`eq-thermal-scattering-tables-free-gas` to $10^{-13}$.
- `test_calcem_free_gas_integral_within_the_declared_budget`: the integral of the
  free-gas tables at $T = 293.6$ K, for $A = 0.99917$, 15.858 and 236.006 and
  $E = 10^{-5}$, 0.0253 and 1.55 eV, is within $2\times10^{-3}$ (relative) of the closed
  form $\sigma/\sigma_\text{free} = (1 + 1/(2y^2))\operatorname{erf}y + e^{-y^2}/(y\sqrt\pi)$,
  $y = \sqrt{AE/kT}$.
- `test_free_gas_tables_carry_the_elastic_and_normalised_spectra`: the free-gas
  cross-section is the elastic read lin-lin at the incident energies, and every cdf ends
  at 1 to $10^{-12}$.
- `test_corrected_free_gas_pdf_and_cdf_describe_the_same_tail_mass`: the cdf increments
  equal the trapezoid areas of the pdf to $10^{-9}$, the areas sum to 1 and the pdf is
  non-negative.
- `test_free_gas_options_are_checked`: the refusals of `iinc`, temperature and elastic
  range.

### Comparison with the official OpenMC files

These comparisons are not tests of this repository: the official files and OpenMC are
neither dependencies of kika nor the truth.

- **Tables.** Against OpenMC's official ENDF/B-VIII.1 thermal files (processed with NJOY
  THERMR + ACER, `tol` $= 10^{-3}$, 16 cosines; NJOY version TODO(check)), for H in H₂O,
  graphite, H in ZrH and Zr in ZrH₂, each at two temperatures (293.6 K or 296 K, and
  600 K), kika's tables, built with the same `tol`, have the same outgoing
  points at every incident energy, $E'$ equal to the last bit, cdf within
  $5\times10^{-7}$, densities within $10^{-6}$ and cosines within $10^{-8}$; the elastic
  parts are identical (graphite's Bragg data within $7\times10^{-12}$). The inelastic
  cross-section differs by at most 0.03 %, and by 0.69 % at the phonon kink of H in ZrH,
  for the reason given under *Conventions and choices* (measured 2026-10-11).
- **Transport.** OpenMC 0.16.0 run with the official thermal files and with kika's, the
  neutron data the same, $10^8$ active histories per side and independent seeds:
  $|\Delta k| \le 3\sigma$ and the flux in each of 32 groups from $10^{-5}$ eV to 20 MeV
  within $3\sigma$, in five test models (a UO₂ cell in light water, a graphite lattice, a
  TRIGA-type ZrH system, a heavy-water cell and a reflected homogeneous UO₂-water medium).
  The largest $|z|$ of $k$ is 1.1, and of a flux group 2.9 (measured 2026-10-13). These are
  regression models, not experimental benchmarks. Evaluation: ENDF/B-VIII.1 TODO(check).

## Diagnostics

When a report is passed, {py:func}`kika.processing.thermal.inelasticTables` and
{py:func}`kika.processing.thermal.freeGasTables` add one warning per temperature naming
the method (THERMR `calcem` and ACER `iwt=2`, or the free-gas integration) and the number
of incident energies. {py:func}`kika.hdf5.thermal_processing.processThermal` adds one
warning per file saying the inelastic part is kika's and where the elastic part came
from. None of these signals a problem; they say what produced the tables. The returned
`energy_max` is the highest incident energy, which is what OpenMC uses as the top of the
thermal treatment.

## Implementation

- `kika.processing.thermal`: the inputs from the model law (`thermalInputs`,
  `freeGasInputs`), the incident grid, the tape rounding and ACER's merge, the free-gas
  table, the threads over incident energies, and the model forms.
- `kika.processing._thermal_native` (C, strict IEEE double precision): THERMR's `sig`,
  `terpq`, `sigl` and `calcem` for one incident energy, the closed-form free-gas angular
  integration, `sigfig` and the 11-column rounding.
- `kika.hdf5.thermal_processing`: the whole OpenMC thermal file. For a scattering law, the
  inelastic part from {py:func}`kika.processing.thermal.inelasticTables` at each
  temperature the law tabulates (or those asked for), the table name looked up in OpenMC's
  registry for the atoms and $n_\text{mix}$, and the elastic part from the evaluation: the
  incoherent one as $\sigma_b$ and the Debye-Waller integral $W'(T)$, the coherent one as
  the Bragg edges and cumulative structure factors at the temperature (the edges ACER
  keeps: walking up the edges to $E_\text{max}$, an edge is kept when the cumulative
  $E\sigma$ grows by more than $(c_\text{last}+1)\times10^{-6}$ eV·b, each value rounded
  as the tape stores it and divided by $n_\text{mix}$), both for a mixed one. The file's
  atomic weight ratio is that of the principal nuclide's neutron tape when one is given,
  else B(3). With `ThermalInelasticOptions(iinc=1)` the input is instead a neutron tape:
  it is reconstructed and broadened to each temperature by kika
  ({doc}`processing-chain`), and {py:func}`kika.processing.thermal.freeGasTables` makes
  the table from its elastic; there is no elastic part, the temperatures must be given,
  and the table name is the caller's.

## References

- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: THERMR (`calcem`, `sigl`, `sig`, `terpq`) and ACER (`acesix`).
- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: §7.2 (elastic), §7.4 (incoherent inelastic, SCT approximation).
- P. K. Romano, N. E. Horelik, B. R. Herman, A. G. Nelson, B. Forget and K. Smith,
  "OpenMC: A state-of-the-art Monte Carlo code for research and development", *Annals of
  Nuclear Energy* 82, 90-97 (2015), doi:10.1016/j.anucene.2014.07.048: thermal data
  (`from_ace`, the `incoherent_inelastic` continuous distribution).
