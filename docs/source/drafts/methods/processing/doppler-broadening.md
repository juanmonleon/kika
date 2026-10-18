# Broadening scalar cross-sections in a free gas

```{admonition} At a glance
:class: note
**Computes:** pointwise and linear-table neutron cross-sections at a higher target temperature.
**Applies to:** complete scalar neutron/laboratory-frame source tables for one nuclide.
**Entry points:** {py:func}`kika.processing.doppler.evaluate_broadened`, {py:func}`kika.processing.doppler.broaden_curve`, {py:func}`kika.processing.doppler.broaden_suite`.
**Status:** experimental — not yet in a release (after 0.4.0).
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
stopping after at least three coefficients when the moments are contracted with
a panel's polynomial, or nine when the moments themselves are returned, only
when that moment bound is
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

### Moments of dense source blocks

Dense, nonnegative linear-in-energy sources can reuse moments of contiguous
source panels at one thermal scale $h$. Each block has centre $c$ and half-width
$d$ in $y=\sqrt E$, with $d/h\le1/2$. For $t=(y-c)/d$, prepare

$$
\mu_j=\int_{c-d}^{c+d} y^2\sigma_0(y^2)t^j\,dy,
\qquad j=0,\ldots,16.
$$ (eq-doppler-source-block-moments)

On a linear source panel, the integrand is a polynomial of degree at most
$j+4$. An eleven-point Gauss-Legendre rule therefore evaluates these moments
exactly in exact arithmetic, without replacing or compressing the source.
Panel widths use rationalised square-root differences; interpolation uses the
normalised panel coordinate. Positive source mass gives $|\mu_j|\le\mu_0$.
For $s_\pm=(c\mp\sqrt E)/h$ and $\delta=d/h$, expand each Gaussian as

$$
e^{-(s+\delta t)^2}=e^{-s^2}\sum_{j=0}^{16}a_jt^j+\text{remainder},
\quad a_0=1,\quad
a_j=\frac{-2s\delta a_{j-1}-2\delta^2a_{j-2}}j,
$$ (eq-doppler-block-expansion)

where $a_{-1}=0$. Contract the coefficients with $\mu_j$ and divide the
Gaussian difference by $\sqrt\pi hE$. With $A=2|s|\delta$ and $B=2\delta^2$,
the coefficient-tail bound in {eq}`eq-doppler-coefficient-tail` bounds the
omitted integral by $e^{-s^2}\mu_0T_{16}/(\sqrt\pi hE)$; it has no additional
moment denominator because the source weight is arbitrary inside a block.

Moment preparation uses a float64 diagnostic

$$
\eta_\mu=\mu_0[1024+8n]\epsilon(1+|c|/d),
$$ (eq-doppler-block-roundoff)

where $n$ is the source-panel count in the block. It covers the tested coordinate,
quadrature, polynomial-power and summation sensitivity. With
$N_a=\sum_{j=0}^{16}|a_j|$, each Gaussian charges
$e^{-s^2}[(\mu_0+\eta_\mu)T_{16}+\eta_\mu N_a+
512\epsilon\mu_0N_a]/(\sqrt\pi hE)$ to the kernel estimate. This float provision
is empirical, not directed interval arithmetic. When the reflected
Gaussian has $s_+-\delta>8$, it may be omitted with the bound
$(\mu_0+\eta_\mu)e^{-(s_+-\delta)^2}(1+64\epsilon)/(\sqrt\pi hE)$.

Whole blocks intersecting the six-width thermal window cover that window and
may also include part of its tails. The original support-tail envelope still
bounds what is omitted and is never reduced to claim this overlap. Sparse gaps
use the existing analytic panel evaluator. Poor conditioning, excessive error,
negative results, larger required radii or exhausted point resources return to
the original point evaluator. Low scaled energies $\sqrt E/h<1$ use that
fallback to retain its cancellation treatment. Signed sources use the original
path throughout.

Preparation is a performance choice: the native path requires the default
80-term setting, radius allowance at least six, at least 64 MiB of work and
a sufficiently dense source/domain. Small direct requests do not initiate
preparation. The prepared curve retains at most one thermal-scale cache;
its native numeric storage is capped at one sixteenth of `max_work_bytes`.
Changing temperature/work allowance replaces the cache, while pickling and
deep copying discard it. A fresh verification reevaluates the kernel and its
diagnostics; it may reuse these immutable source moments.

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
integrals; it remains a diagnostic, not a directed-rounding certificate.
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

It is rounded conservatively in float64. Start at six thermal widths, or the
configured maximum if smaller. An absolute tail bound below its allocation of
`atol` passes immediately. Otherwise evaluate the convolution and require the
tail to fit its allocation of $\mathrm{atol}+\mathrm{rtol}\max(|\sigma|-\eta_{tail},0)$;
increase the radius up to the configured maximum (26 by default) if it fails.
No output-energy by input-panel matrix is formed: sorted prepared edges select
relevant panels, and output evaluation proceeds in blocks.

### Tables and sums

Heated tables start from a coarse lethargy mesh and thermal flanks of every
source peak, valley, plateau transition and region boundary. Source extrema are
detected in bounded chunks with both neighbouring slopes, so an artificial
chunk edge cannot hide a feature. Cold knots are retained for an identity
temperature increment; a heated table does not inherit their full density.

Refinement predicts chord error from second differences of the node values.
For interval width $\Delta E$ and the larger neighbouring curvature estimate
$\kappa$, the prediction is

$$
e_{pred}=\frac{\Delta E^2}{8}\kappa.
$$ (eq-doppler-chord-prediction)

Intervals whose predicted error exceeds their table allocation receive
$\lceil1.1\sqrt{e_{pred}/b}\rceil$ equal subdivisions, capped at 64 per
iteration, where $b$ is the available table budget. This is an insertion
heuristic; it does not certify an interval. A separate family of four reserved
fractions must pass against the immutable source before acceptance. A failing
reserved check inserts its worst probe and repeats refinement.

Accepted intervals retain numeric endpoint pairs in a bounded certificate
cache using one eighth of `max_work_bytes`, at 16 bytes per pair. A certificate
is reused only while both endpoints remain neighbours; insertion invalidates
the old chord. Off-grid values are not retained. Before returning, node
uncertainties must also fit the table allocation and published arrays must
match the checked nodes exactly. These checks reuse construction evidence;
they are not a second independent source evaluation. The cache is discarded
before returning. Subsequent `verify_table()` and attachment checks compute
fresh source values at all nodes and reserved fractions, detecting mutable or
reread output changes.

For candidate table value $c$, reference $r$ and
reference uncertainty estimate $\eta$, acceptance uses

$$
\frac{|c-r|+\eta}{f\,[\mathrm{atol}+\mathrm{rtol}\max(|r|-\eta,0)]}\le1.
$$ (eq-doppler-budget)

Default allocations are kernel 0.15, tails 0.10, tabulation 0.45 and serialisation
0.20, leaving 0.10 unallocated. The table allocation includes evaluator uncertainty.
Suites heat exclusive leaves on a common mesh and derive all declared aggregates
from their graph. Each sum has its own acceptance scale; cancellation cannot
enlarge its relative allowance. The source sum balance is checked beforehand on source knots, midpoints and
left limits. Region-wise vector interpolation preserves right-owned boundaries;
aggregate addition retains compensated `math.fsum` scaling.

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
- **The kernel is the same integral as BROADR's and FUDGE's, with its error
  accounted.** NJOY BROADR integrates each linear panel with the SIGMA1 analytic
  moments of order 0 to 4, switching to a series where they lose precision; FUDGE
  uses closed moments with a centred expansion for narrow panels. kika uses closed
  moments where its error diagnostic accepts them and a local series otherwise,
  and charges each error term to an explicit share of the budget
  ({eq}`eq-doppler-budget`). This is a different route to the same operator
  {eq}`eq-doppler-operator`: no gain in accuracy of the kernel over BROADR's is
  claimed, only that its error is stated.
- **The mesh is not thinned.** BROADR thins the heated table to `errthn` (and
  `thnmax`), and FUDGE to half its accuracy, by removing nodes that the chord of
  their neighbours reproduces at the nodes. The tolerance then holds at the nodes
  and not between them: on Fe-56 of ENDF/B-VIII.1 at 293.6 K (MT 2 and 102, up to
  850 keV), sampling the BROADR table at 100 000 random energies against kika's
  pointwise kernel gives a maximum relative error of 1.0 % at the default
  `errthn` $=10^{-3}$, with 8.4 % of the MT 102 points above $10^{-3}$; it stays
  below $10^{-3}$ everywhere only with `errthn` $\le 10^{-4}$. kika's table, built
  at `rtol` $=10^{-3}$ and checked at interior points, gives $5\cdot10^{-4}$
  there, with no point above $10^{-3}$ (NJOY 2016.78, measured 2026-10-11).
- **The output window is the one asked for.** The upper limit given to BROADR is
  not always the one it heats to: on ENDF/B-VIII.1 Zr-90 a run asked for a higher
  limit heated only up to 200 keV, and H-1 up to 6.5 MeV, leaving the rest at the
  cold values (NJOY 2016.78, measured 2026-10-11). kika has no automatic cut-off;
  the window is an argument, and every node inside it is heated and checked.
- **The source below its first energy is part of the input.** The integral
  {eq}`eq-doppler-operator` reaches below the first energy of the cold table, so
  kika asks for the continuation (zero, constant or $1/v$) and integrates it. In
  FUDGE, `lowerlimit='oneOverV'` shapes only the output below the first energy and
  does not add that contribution to the integral; to reproduce FUDGE, kika is run
  with `zero` support. Comparisons of the two codes that use $1/v$ in kika and
  `oneOverV` in FUDGE therefore differ by that term, and not by an error of
  either.

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
bounded semantic sample evaluation and fresh verification after construction.
`test_construction_certificates.py` covers exact publication, node uncertainty,
certificate invalidation and extrema across chunk boundaries.
`test_source_moment_blocks.py` checks moment estimates against the panel path,
component closure, signed/low-temperature fallback, capsule ownership, cache
replacement, scaled-energy rejection and unaligned buffers.
Table/suite tests cover thresholds, graphs, freshness, temperature history,
rejection and atomic attachment in `test_table_suite.py`.

`kika/processing/doppler/tests/test_component_runs.py` compares grouped
component diagnostics with the ordered contributions of individual panels,
including signed data, repeated and interleaved labels, Unicode labels and
subclass dictionary semantics. It also checks shared region-label ownership.
The native block controller is checked against the scalar controller, including
identical accepted values/errors/tails/components and the same rejected point
and budget, in `kika/processing/doppler/tests/test_blocks.py`. Additional checks
cover strided/reversed/unaligned queries, constrained scratch, failed buffer
exports, late failures without partial results, and ordered prepared edges.

### Comparison with FUDGE and NJOY

These comparisons are not tests of this repository: FUDGE and NJOY are neither
dependencies of kika nor the truth. FUDGE 6.14.0 and NJOY 2016.78 (Linux build),
on evaluations of ENDF/B-VIII.1.

- **FUDGE, on the same cold input.** Both codes heated the identical cold tables of
  Fe-56, Zr-90, Au-197, U-235, U-238 and Pu-239 (MT 102), with the same support
  convention (`zero` or `constant`), at 1, 293.6, 600 and 1200 K and relative
  tolerances $10^{-3}$, $10^{-4}$ and $10^{-5}$ ($10^{-8}$ b absolute). Over
  8 055 probes, the largest difference is 0.31 of the sum of the two codes'
  declared tolerances (measured 2026-10-12). The two codes compute the same
  heated cross-section.
- **NJOY, on its own cold table.** BROADR was compared with kika's kernel applied
  to NJOY's own RECONR table, so that only the broadening differs. At the probes,
  BROADR's residual is at most 0.93 of its budget, and kika's pointwise kernel
  agrees with an independent numerical quadrature on the same input to
  $1.1\cdot10^{-10}$ of the budget (Fe-56, Zr-90, Au-197, U-235, U-238 at
  293.6 K and above; measured 2026-10-11). Between the nodes of a thinned table the
  error is larger, as stated under *Conventions and choices*. Pu-239 was not
  compared with NJOY: the SAMMY routines of its RECONR take a wrong channel spin for the effective fission channel of
  this evaluation (PNT=-1, IA=IB=0), so the cold table differs before broadening.
- **BROADR's own node values.** Over every BROADR node of Fe-56 below 850 keV (default
  input, $10^{-3}$), 97-99 % agree with the exact kernel to the seven written figures.
  The worst MT 2 nodes are the seven-figure rounding of their energies. The worst MT 102
  nodes, up to $2.6\cdot10^{-3}$ at 121 020.4 eV, are a defect of BROADR, shown with a
  traced build and a line-by-line replay. There `bsigma` is started from a panel above
  the node, which it can only move up, and its Taylor series `hnabb` mishandles the sign
  of the negative arguments that start produces. Either alone is harmless (measured
  2026-10-18).
- **Time.** On a thermal window (Fe-56 and Au-197 at 600 K, $10^{-7}$ to 20 eV,
  MT 2 and 102, `rtol` $=10^{-3}$, one core, median of three runs, measured
  2026-10-12) kika takes 0.31 s and 0.30 s, FUDGE 0.15 s and 0.19 s, NJOY 0.13 s:
  on small tables both are about twice as fast. On the whole domain of Fe-56 MT 102
  at 600 K ($10^{-5}$ eV to 150 MeV, identical cold input) kika takes 12 s and
  FUDGE 18 s (median of three alternated runs).

## Diagnostics

Pointwise reports distinguish kernel estimates and tail bounds. Pointwise, table
and suite reports identify the resolved `backend` and core engine version. Table reports
give points, regions, evaluations/iterations and acceptance ratios. Suite reports
add source hashes, mass/temperature origins, graph, source balance and unresolved
resonance semantics. `global_error_bound=False` always distinguishes these checks
from certification. Fractions of the full accuracy budget and the table-specific
ratio have different denominators and must not be conflated. The point report's
`moment_blocks` distinguishes computed/accepted blocks, legacy fallbacks,
query count and retained native cache bytes; cache preparation may have an
upfront cost. `analytic_subpanels` includes accepted moment blocks as numerical
integration units, so it is not the count of original source panels.

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
`_blocks` prepares temperature-local source moments and applies the same point
budgets with the original evaluator as fallback. The native moment capsule owns
its source panels and labels. Cache arrays are copied and validated once;
subsequent queries cannot alter them. Preparation and working blocks are bounded
separately from retained source/result arrays.
`table` refines and verifies interpolated forms; `suite` prepares semantic input,
derives sums and attaches an independent Heated copy atomically. Temperature
styles do not automatically heat every other object in a reaction suite.
See {doc}`../../user_guide/doppler-broadening` for a worked API workflow and {doc}`../../api_reference/processing-doppler`
for generated contracts.

## References

Bret R. Beck, *Correcting a cross section for a thermal target*, Lawrence
Livermore National Laboratory, FUDGE documentation, Theory (`rrEq`,
`continuousEq`) and Heating a heated cross section (`continuousEq2`).
[Upstream source](https://github.com/LLNL/fudge/blob/master/crossSectionAdjustForHeatedTarget/Doc/crossSectionAdjustForHeatedTarget.tex).

