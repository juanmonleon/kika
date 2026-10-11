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

The Python reference uses an immutable matrix $W_{jn}=1/(n+j+1)$ for the
default 80 terms and scalar compensated sums for other requested counts.
The compiled C kernel uses strict float64 arithmetic without fast-math. For
the default term limit it first tries a complete source panel of thermal width
at least one when $z\ge0.125$. Define the closed Gaussian moments

$$
G_j(a,d)=\int_0^1u^j e^{-(a+du)^2}du.
$$ (eq-doppler-closed-definition)

With $g_a=e^{-a^2}$ and $g_b=e^{-(a+d)^2}$, integration by parts gives

$$
\begin{aligned}
G_0&=\frac{\sqrt\pi}{2d}[\operatorname{erf}(a+d)-\operatorname{erf}(a)],\\
G_1&=\frac{g_a-g_b}{2d^2}-\frac{a}{d}G_0,\\
G_j&=\frac{j-1}{2d^2}G_{j-2}-\frac{a}{d}G_{j-1}-\frac{g_b}{2d^2},\quad j\ge2.
\end{aligned}
$$ (eq-doppler-closed-recurrence)

Same-sign tail differences use `erfc`; crossing intervals use `erf`.
The required moments are $G_j(s_0,d)-G_j(s_0+2z,d)$. Absolute roundoff
provisions propagate through each recurrence operation and the compensated
polynomial contraction. The closed result is accepted only if this provision
is at most $3\times10^{-13}$ times the absolute contracted integral. Otherwise
the panel is subdivided and evaluated with the local series above. Non-default
term limits retain the series path. A Gaussian starting beyond 40 is represented
by zero with a `DBL_MIN` absolute moment provision, which exceeds its neglected
value; that provision is included in the contraction.

The closed path uses the sum of the maxima of the two Gaussians as its
$\mathcal N$ for the geometry provision below. If the total native estimate
exceeds the point budget, the point is retried using local panels; acceptance
still requires the same budget. Resource preflight reserves the possible local
fallback subdivisions. `analytic_subpanels` records the panels actually
integrated, so native and Python counts can differ.

For the series path, the C kernel contracts the polynomial into the coefficient
weights before accumulating the panel integral.

For the default maximum of 80 terms, the compiled recurrence may stop early.
Suppose the absolute coefficients obey
$|c_k|\le A|c_{k-1}|/k+B|c_{k-2}|/k$. Summing this inequality beyond coefficient
$n$, for $n+1>A+B$, bounds the omitted coefficient norm by

$$
T_n\le\frac{(A+B)|c_n|+B|c_{n-1}|}{n+1-A-B}.
$$ (eq-doppler-coefficient-tail)

For each Gaussian sequence, $A=2|b_\pm|d$ and $B=2d^2$; add their bounds.
For the coupled low-$z$ recurrence, use $|D_n|+|H_n|$ and
$A=2(|z+s_0|+z)d$. The omitted moment is at most $T_n/(n+2)$.
The diagnostic pads the coefficient bound by $1+128\epsilon$ and permits
stopping after at least three coefficients in production, or nine in the
private moment diagnostic, only when that moment bound is
at most $4\epsilon$ times the accumulated difference norm. Other maximum
term counts retain the requested number of coefficients.

Let $\mathcal N$ be the accumulated absolute difference norm, including the
omitted coefficient bound. For an early stop, $R$ is the omitted moment bound;
for a full series, $R=4\sum_{n=N-4}^{N-1}|D_n|$. The moment diagnostic uses
$R+256\epsilon\mathcal N$, with another $83\epsilon\mathcal N$ for default
reciprocal-weight reductions. The panel diagnostic multiplies it by
$d\sum_j|p_j|/\sqrt\pi$; accumulated panel totals add
$8\epsilon\sum|\sigma_{panel}|$. The floating-point implementation and
full-series stopping diagnostics are not interval-arithmetic certificates.
Very narrow source panels use their normalised local coordinate to preserve
the input width. Identically zero source panels skip moment evaluation.

### Geometry sensitivity

Rationalised source widths avoid subtracting neighbouring square roots, but
partially clipped narrow panels amplify location error by the reciprocal source
width. Let $d_s$ be the source width in the thermal coordinate, $s_l$ its left
location, and $b_l,b_r$ the clipping limits. A limit is treated as touching a
source edge when it crosses that edge or lies within
$32\epsilon(|s_l|+|b|+d_s)$ of it. For touching narrow panels, set

$$
\delta_s=32\epsilon(|s_l|+B+d_s),\qquad
\beta=64\epsilon(1+|s_0|+d)^2+\frac{\delta_s}{d},
$$ (eq-doppler-geometry-location)

where $B$ sums the absolute values of touching clipping limits. Otherwise
$\delta_s=0$ and the last term in $\beta$ is absent. For a linear panel,
$\Delta v=v_1-v_0$, the narrow-panel height provision is
$S_\sigma=64\epsilon(|v_0|+|\Delta v|)+|\Delta v|\delta_s/d_s$.
For a wider linear panel add
$64\epsilon|\Delta v|[|E-E_l|+|hs_0|(2\sqrt E+|hs_0|)]/(E_r-E_l)$
instead of the clipping term. A constant panel uses $64\epsilon|v_0|$;
the inverse-velocity form uses only the relative provision. The additional
panel estimate is

$$
\eta_{geometry}=\frac d{\sqrt\pi}\mathcal N
\left[\beta\sum_j|p_j|+
S_\sigma\left(\left|1+\frac{s_0}{z}\right|+\frac d z\right)^2\right].
$$ (eq-doppler-geometry-estimate)

This float64 sensitivity provision is tested against high-precision Gaussian
integrals; it remains a diagnostic rather than a directed-rounding certificate.
It is charged to the existing kernel allocation, which is never enlarged.


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
point and repeats refinement. Already accepted, unchanged intervals reuse their
acceptance ratios in later refinement iterations. That cache retains only
ratios and uses a quarter of the scratch allowance. A second, construction-only
cache reuses identical source values and error estimates, including reserved
probes needed again for the final table check. It distinguishes semantic left
limits, processes temporary data in bounded blocks and allows one eighth of
`max_work_bytes` with conservative entry accounting. Every verification location
and interpolation/error check is retained; these fractions remain distinct from
the refinement/check families. Both caches are discarded before returning.
Subsequent `verify_table()` calls compute fresh source values and still detect
mutated or reread outputs.
For candidate table value $c$, reference $r$ and
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
  the removed contribution. Published exact zeros have zero kernel
  and tail uncertainty; maximum uncertainties of the discarded convolution are
  retained separately in the point report. This convention is additional to
  ordinary convolution.
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
`kika/processing/doppler/tests/test_native.py` checks compiled moments against
independent quadrature, compiled/Python pointwise agreement, narrow-peak tables,
resource failures, unavailable-backend behaviour and accuracy after interval reuse.
`test_geometry.py` exercises high-precision clipping anchors and imposed-zero
uncertainty; `test_hybrid.py` checks conditioned closed moments against
independent quadrature and verifies fallback/resource behaviour.
`test_prepared_samples.py` covers native storage ownership, serialisation,
bounded semantic sample reuse and fresh verification after construction.
Table/suite tests cover thresholds, graphs, freshness, temperature history,
rejection and atomic attachment in `test_table_suite.py`.

`kika/processing/doppler/tests/test_component_runs.py` compares grouped
component diagnostics with the ordered contributions of individual panels,
including signed data, repeated and interleaved labels, Unicode labels and
subclass dictionary semantics. It also checks shared region-label ownership.

## Diagnostics

Pointwise reports distinguish kernel estimates and tail bounds. Pointwise, table
and suite reports identify the resolved `backend` and core engine version. Table reports
give points, regions, evaluations/iterations and acceptance ratios. Suite reports
add source hashes, mass/temperature origins, graph, source balance and unresolved
resonance semantics. `global_error_bound=False` always distinguishes these checks
from certification. Fractions of the full accuracy budget and the table-specific
ratio have different denominators and must not be conflated.

## Implementation

`kika.processing.doppler.core` prepares immutable curves and dispatches analytic
panel integration to `_doppler_native` or the Python reference. The package build
compiles the C kernel without the NumPy C ABI. Immutable numeric panels own
their source tuples and labels and prepare roots, slopes and panel kinds once.
Each region shares one immutable owner label across its panels. Component
diagnostics accumulate consecutive panel contributions in a native scalar,
preserving the original order of floating-point additions. A Python float is
materialised only when the label changes or integration ends. Revisited labels
resume the prior stored contribution; Unicode subclasses retain per-panel
dictionary updates. This avoids allocating an object per source panel without
changing values, error estimates, budgets or verification samples.
Their byte count is reported separately from result arrays. Pickling and deep
copying rebuild the process-local native storage from the canonical source.
The native backend executes each bounded block in one call. It selects thermal
radii, locates prepared panels by binary search, evaluates each contribution,
checks the point budget, retries the local series when required, and preserves
the failing point's energy/category without returning a partial block. Python
then applies the publication threshold and accumulates diagnostics in point
order. Integer-radius tail terms are prepared once with the reference special
function and the same operation order; this preserves the scalar tail bounds.
The scratch estimate reserves `512 + 128*n_owners` bytes per block point for
buffer copies, tuples, scalars and component dictionaries. The effective block
size shrinks to fit the remaining allowance. Each block's rows are released
before the next block is allocated. The owner count follows from
the region/support metadata without building a set over all source panels.
The source fingerprint is cached on the immutable prepared curve and hashes
its contiguous buffers without temporary copies. Contiguous native float64 buffers
include unaligned inputs, read with `memcpy`; strided inputs are copied within
the bounded block. No NumPy C ABI is required. The scalar native controller and
Python reference remain available as controls and compatibility fallbacks.
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


The native block controller is checked against the scalar controller, including
identical accepted values/errors/tails/components and the same rejected point
and budget, in `kika/processing/doppler/tests/test_blocks.py`. Additional checks
cover strided/reversed/unaligned queries, constrained scratch, failed buffer
exports, late failures without partial results, and ordered prepared edges.
