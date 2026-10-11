# Broadening scalar cross-sections in a free gas

```{admonition} At a glance
:class: note
**Computes:** pointwise and linear-table neutron cross-sections at a higher target temperature.
**Applies to:** complete scalar neutron/laboratory-frame source tables for one nuclide.
**Entry points:** {py:func}`kika.processing.doppler.evaluate_broadened`, {py:func}`kika.processing.doppler.broaden_curve`, {py:func}`kika.processing.doppler.broaden_suite`.
**Status:** experimental.
```

## Background

A Maxwellian free-gas target changes the reaction rate through the relative
projectile/target velocity. The scalar operator follows Beck, *Correcting a cross
section for a thermal target*, sections Theory and Heating a heated cross section.
It uses a non-relativistic neutron laboratory-frame energy. Bound-atom thermal
scattering and secondary distributions require separate models.

## What it computes

Let $A=M/m_n$, $\Delta T=T_1-T_0$, $h=\sqrt{k_B\Delta T/A}$,
$x=\sqrt E$, and $z=x/h$. The constant is
$k_B=8.617333262145\times10^{-5}$ eV/K. For $\Delta T>0$ the complete supported
input table $\sigma_0$ determines the result:

$$
\sigma_{T_1}(E)=\frac{1}{x^2h\sqrt\pi}
\int_0^\infty y^2\sigma_0(y^2)
\left[e^{-(y-x)^2/h^2}-e^{-(y+x)^2/h^2}\right]dy.
$$ (eq-doppler-operator)

This includes the velocity weighting in the rate; it is not a Gaussian
convolution of $\sigma(E)$ in energy. Equal temperatures return the input
function. A hotter convolution-compatible source uses the temperature increment
(Beck, Heating a heated cross section). The operator makes no claim about unknown
source reconstruction error.

## Method

### Analytic panels

The input is linear in energy on every panel. Set $y=x+h s$ and split the supported
thermal range into local intervals $s=s_0+d u$, $0\le u\le1$, with $d\le1/4$.
The weighted cross-section on an interval is a polynomial
$P(u)=(1+s/z)^2\sigma_0(E(1+s/z)^2)$ of degree at most four.
The two Gaussian Taylor sequences start at
$g_0^+=e^{-s_0^2}$ and $g_0^-=e^{-(s_0+2z)^2}$ and satisfy

$$
n g_n^\pm=-2b_\pm d\,g_{n-1}^\pm-2d^2g_{n-2}^\pm,
\qquad b_+=s_0,\quad b_-=s_0+2z,
$$ (eq-doppler-recurrence)

with coefficients below index zero interpreted as zero. At $z<1/8$, evolve
$D_n=g_n^+-g_n^-$ and $H_n=g_n^++g_n^-$ jointly. Their initial values use
`expm1` for $D_0$; replace the first coefficient in the recurrence by
$-2(z+s_0)d$ and add $2zdH_{n-1}$ to the difference equation and
$2zdD_{n-1}$ to the sum equation. Both retain $-2d^2$ times the preceding
second coefficient. This avoids subtracting nearly equal Gaussian sequences.

For $P(u)=\sum_{j=0}^4p_j u^j$, the default $N=80$ terms give

$$
M_j=\sum_{n=0}^{N-1}\frac{D_n}{n+j+1},\qquad
\sigma_{panel}=\frac{d}{\sqrt\pi}\sum_{j=0}^4p_jM_j.
$$ (eq-doppler-moments)

For 80 terms, an immutable matrix $W_{jn}=1/(n+j+1)$ computes the five reductions
with NumPy; other requested term counts use scalar compensated sums. Recurrence
and panel totals remain unchanged. Let $\mathcal N=\sum_n|D_n|$ and
$R=4\sum_{n=N-4}^{N-1}|D_n|$. The moment diagnostic uses
$R+256\epsilon\mathcal N$ and charges another $83\epsilon\mathcal N$ for the
default matrix reduction. The panel diagnostic multiplies it by
$d\sum_j|p_j|/\sqrt\pi$; accumulated panel totals add
$8\epsilon\sum|\sigma_{panel}|$. These are floating-point diagnostics, not a
rigorous truncation certificate. Very narrow input panels use their normalised
local coordinate to preserve the input width. Identically zero source panels
contribute exactly zero and skip moment evaluation.

### Support and tails

Use the input envelope $|\sigma_0(y^2)|\le M+K/y$, including an explicit lower
$1/v$ continuation. With $I_q(R)=\int_R^\infty s^q e^{-s^2}ds$,
$I_0=\sqrt\pi\operatorname{erfc}(R)/2$, $I_1=e^{-R^2}/2$, and
$I_2=Re^{-R^2}/2+I_0/2$, the tail diagnostic is

$$
\eta_{tail}=\frac{2}{\sqrt\pi}\left[
M\left(I_0+\frac{2I_1}{z}+\frac{I_2}{z^2}\right)
+\frac{K}{hz}\left(I_0+\frac{I_1}{z}\right)\right].
$$ (eq-doppler-tail)

It is rounded conservatively in float64. Increase the radius from at most eight
up to the configured maximum (26 by default) until the tail allocation is met.
No output-energy by input-panel matrix is formed: sorted prepared edges select
relevant panels, and output evaluation proceeds in blocks.

### Tables and sums

Structural seeds retain source knots and thermal flanks of peaks/valleys. Three
refinement fractions and four check fractions guide insertion. A separate family
of four fractions must pass before acceptance; a failing final check inserts a
point and repeats refinement. For candidate table value $c$, reference $r$ and
reference uncertainty estimate $\eta$, acceptance uses

$$
\frac{|c-r|+\eta}{f\,[\mathrm{atol}+\mathrm{rtol}\max(|r|-\eta,0)]}\le1.
$$ (eq-doppler-budget)

Default allocations are kernel 0.15, tails 0.10, tabulation 0.45 and serialisation
0.20, leaving 0.10 unallocated. The table allocation includes evaluator uncertainty.
Suites heat exclusive leaves on a common mesh and derive all declared aggregates
from their graph. Each sum has its own acceptance scale; cancellation cannot
enlarge its relative allowance. The source sum balance is checked beforehand.

## Conventions and choices

- Internal arrays use eV and b, temperatures K; the mass ratio is target/neutron.
- Source support is explicit: zero, constant or $1/v$ below the table; zero or
  constant above it. Changing continuation changes the mathematical input.
- Contiguous input regions preserve jumps. A boundary belongs to the right;
  a preceding region stores its left limit independently.
- `zero_below` applies a declared output convention at a threshold and records
  the removed contribution. It does not describe ordinary convolution alone.
- Source signs are rejected by default; explicit signed processing never clips.
- Output windows restrict tabulation, while integration uses the full source
  support. Structural nodes are retained and the mesh is not thinned.
- A suite source temperature follows a validated style or declaration. PoPs
  masses and explicit masses must agree. Metadata never establish an unknown
  heating history as convolution-compatible merely by copying a style.

## Limits and failure modes

The processor refuses cooling, unrepresentable thermal scales, inconsistent
temperatures/masses, unsupported interpolation/units, incomplete or overlapping
reaction coverage, source sum imbalance and exhausted accuracy/resources.
Natural/mixture targets, non-neutron/non-laboratory projectiles, finite-dilution
unresolved-resonance processing and physical-input error claims with no source
envelope are unsupported. Unresolved resonances require a complete explicitly
declared infinite-dilution mean; it is not added a second time.

Mutable output/source conflicts fail before attachment. Histories with unknown
temperature model or prior threshold truncation cannot be reheated without a
valid model. Estimated scratch limits exclude retained arrays and total process
resident memory. Empirical probes do not prove a whole-domain bound. Secondary
distributions/covariances remain at their original treatment.

## Validation

`kika/processing/doppler/tests/test_core.py::test_constant_gaussian_pair_and_limits`
checks the analytic constant-source expression,
$\sigma_0[(1+1/(2z^2))\operatorname{erf}(z)+e^{-z^2}/(\sqrt\pi z)]$.
The same file tests inverse-velocity behaviour, identity, narrow increments,
sign/linearity and block invariance. Independent numerical quadrature checks
local moments in
`kika/processing/doppler/tests/test_moments.py::test_weighted_moments_against_independent_quadrature`.
Table/suite tests cover thresholds, graphs, freshness, temperature history,
rejection and atomic attachment in `test_table_suite.py`.

## Diagnostics

Pointwise reports distinguish kernel estimates and tail bounds. Table reports
give points, regions, evaluations/iterations and acceptance ratios. Suite reports
add source hashes, mass/temperature origins, graph, source balance and unresolved
resonance semantics. `global_error_bound=False` always distinguishes these checks
from certification. Fractions of the full accuracy budget and the table-specific
ratio have different denominators and must not be conflated.

## Implementation

`kika.processing.doppler.core` prepares immutable curves and integrates panels.
`table` refines and verifies interpolated forms; `suite` prepares semantic input,
derives sums and attaches an independent Heated copy atomically. Temperature
styles do not automatically heat every other object in a reaction suite.
See {doc}`doppler-broadening` for a worked API workflow and {doc}`doppler`
for generated contracts.

## References

Bret R. Beck, *Correcting a cross section for a thermal target*, Lawrence
Livermore National Laboratory, FUDGE documentation, Theory (`rrEq`,
`continuousEq`) and Heating a heated cross section (`continuousEq2`).
[Upstream source](https://github.com/LLNL/fudge/blob/master/crossSectionAdjustForHeatedTarget/Doc/crossSectionAdjustForHeatedTarget.tex).
