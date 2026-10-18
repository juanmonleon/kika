# Averaging cross-sections over the unresolved resonance range

```{admonition} At a glance
:class: note
**Computes:** the infinitely dilute average elastic, capture and fission cross-sections
of the unresolved resonance range at 0 K, from the average resonance parameters of
MF 2 (LRU=2), with the width-fluctuation averages integrated exactly over
$\chi^2$ width distributions of real degrees of freedom.

**Applies to:** an unresolved range of an incident-neutron model suite (ENDF-6 cases A,
B and C, LSSF=0 or 1, or the equivalent GNDS data), as part of the reconstruction of
{doc}`resonance-reconstruction`.

**Entry points:** through {py:func}`kika.endf.reconstruct_endf` and
{py:func}`kika.processing.resonances.reconstruct_suite`; the width-fluctuation averages alone are
{py:func}`kika.processing.resonances.fluctuations.width_products`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

Above some energy the resonances of a nucleus overlap or are too close to be resolved
by experiment, and an evaluation states only their statistics (ENDF-6 Ch. 2, LRU=2):
for each spin sector $(\ell, J)$ a mean level spacing $D_{\ell J}(E)$, average partial
widths $\bar\Gamma_{x}(E)$ for the neutron, capture, fission and competitive channels,
and the number of degrees of freedom $\nu_x$ of the $\chi^2$ distribution each width is
assumed to follow (AMUN, AMUG, AMUF, AMUX). The neutron width is given reduced, as
$\bar\Gamma_n^0$. Case A states constant parameters and no fission, case B constant
parameters with fission widths tabulated on an energy grid, case C every parameter
tabulated on an energy grid with its interpolation law INT.

The cross-section averaged over many resonances, for a flux that the resonances do not
depress (infinite dilution), follows from averaging the single-level formulas over the
distributions of the widths. Because a width appears in the numerator and in the total
width of the denominator, the average of $\Gamma_n\Gamma_x/\Gamma$ is not the ratio of
average widths: the width-fluctuation correction is this difference. MF 3 holds a
background in the range; with LSSF=1 it already holds the dilute average itself, and the
MF 2 parameters are there only for self-shielding.

## What it computes

Input: the unresolved range of the model (its spin groups, parameter functions, radius
policy and interpolation conventions) and the target mass ratio $A$ and spin $I$. Output,
at an energy $E$ of the range, the resonance contributions $\sigma^\text{res}_x(E)$ for
MT 2, 18 and 102 and their sum, which are added to the MF 3 background as in
{doc}`resonance-reconstruction` and tabulated to the same tolerance. With $k$, $\beta =
\pi/k^2$ and $g_J$ as in {eq}`eq-resonance-reconstruction-kinematics`,

$$
\begin{aligned}
\sigma_x(E) &= 2\pi\beta \sum_{\ell, J} \frac{g_J}{D_{\ell J}}\, Q_{nx}^{\ell J},
\qquad x = \gamma, f,\\
\sigma_n(E) &= 4\beta \sum_\ell (2\ell+1)\sin^2\varphi_\ell
+ 2\pi\beta \sum_{\ell, J} \frac{g_J}{D_{\ell J}}
\Big(Q_{nn}^{\ell J} - 2\bar\Gamma_n^{\ell J}\sin^2\varphi_\ell\Big),
\end{aligned}
$$ (eq-unresolved-resonances-sigma)

with $\sigma_t = \sigma_n + \sigma_\gamma + \sigma_f$ and the width-fluctuation averages

$$
Q_{nc}^{\ell J} = \left\langle \frac{\Gamma_n \Gamma_c}{\Gamma_n + \Gamma_\gamma + \Gamma_f + \Gamma_x} \right\rangle ,
$$ (eq-unresolved-resonances-q)

the average over independent widths $\Gamma_y = \bar\Gamma_y\,\chi^2_{\nu_y}/\nu_y$ of
the sector. The competitive width enters the denominator only.

## Method

**1. Parameters at $E$.** $D_{\ell J}(E)$ and every $\bar\Gamma_y(E)$ are evaluated from
their own functions (a constant, or a table with its interpolation law; a step in a table
is kept with both one-sided values). The physical average neutron width is

$$
\bar\Gamma_n^{\ell J}(E) = \nu_n\,\bar\Gamma_n^{0,\ell J}(E)\,\sqrt{E}\;\frac{P_\ell(k a_c)}{k a_c},
$$ (eq-unresolved-resonances-neutron-width)

with $E$ in eV, $P_\ell$ from {eq}`eq-resonance-reconstruction-recurrence` and $a_c$ the
channel radius ($P_\ell/\rho$ is the ENDF $V_\ell$: 1, $\rho^2/(1+\rho^2)$, ...). A width
the model marks as physical is used as it is. The factor $\nu_n$ is NJOY's: its
`unfac` returns $V_\ell$ multiplied by AMUN (`reconr.f90:4473-4492`), so UNRESR's width is
GN0 $\sqrt{E}\,\nu_n V_\ell$, the same product.

**2. Radii.** The phase radius $a_\varphi$ is AP($E$) when the range has NRO=1, else AP;
the channel radius $a_c$ follows NAPS: 0, the mass formula
{eq}`eq-resonance-reconstruction-mass-radius` with the AWRI of the sector's block; 1,
$a_\varphi$; 2, the constant AP (with a warning, as ENDF-6 defines NAPS=2 for resolved
ranges). The radius of a resolved range is never used here.

**3. Width-fluctuation averages.** With $1/\Gamma = \int_0^\infty e^{-t\Gamma}\,dt$ and
the Laplace transforms of the $\chi^2$ distributions, every average reduces to one
integral. Let $\mathcal{F}$ be the channels with $\nu_y > 0$ (fluctuating),
$\theta_y = 2\bar\Gamma_y/\nu_y$, and $\Gamma_\text{fix}$ the sum of the widths with
$\nu_y = 0$ (fixed). Then

$$
\begin{aligned}
Q_{nc} &= \bar\Gamma_n\bar\Gamma_c \int_0^\infty
\frac{e^{-t\Gamma_\text{fix}}\prod_{y\in\mathcal{F}}(1+\theta_y t)^{-\nu_y/2}}
{(1+\theta_n t)^{[n\in\mathcal{F}]}\,(1+\theta_c t)^{[c\in\mathcal{F}]}}\,dt ,
\qquad c \ne n,\\
Q_{nn} &= \bar\Gamma_n^2\Big(1+\frac{2}{\nu_n}\Big) \int_0^\infty
\frac{e^{-t\Gamma_\text{fix}}\prod_{y\in\mathcal{F}}(1+\theta_y t)^{-\nu_y/2}}
{(1+\theta_n t)^{2}}\,dt ,
\end{aligned}
$$ (eq-unresolved-resonances-laplace)

where $[\cdot]$ is 1 when the condition holds and 0 otherwise (for $\nu_n = 0$ the factor
$1 + 2/\nu_n$ is replaced by 1). The integrals are computed in $s = \ln t$ over the whole
real line by adaptive quadrature (relative tolerance $2\cdot10^{-12}$), after scaling all
widths by the largest. Each must report an error below $10^{-10}$ of its value, and the
averages must conserve the entrance width,

$$
\Big\lvert \sum_c Q_{nc} \big/ \bar\Gamma_n - 1 \Big\rvert \le 2\cdot10^{-10},
$$ (eq-unresolved-resonances-balance)

the sum running over all channels, competitive included; otherwise the evaluation
raises. Two limits are exact: a sector whose only non-zero width is the neutron one has
$Q_{nn} = \bar\Gamma_n$, and a sector with every $\nu_y = 0$ has $Q_{nc} = \bar\Gamma_n
\bar\Gamma_c/\sum_y\bar\Gamma_y$.

**4. Energy dependence.** How $\sigma$ varies between parameter energies depends on what
the data declare:

- **No interpolation law** (case A, or a model group without one): $\sigma$ is computed
  at every energy the tabulation asks for, from the interpolated parameters.
- **A law for the cross-sections** (case B, INT=2; case C, the INT of its J-states):
  $\sigma$ is computed at the nodes of the declared energy grid and interpolated between
  them with that law. A panel whose ends
  differ by more than a factor 3 first gets $\lceil 10\log_{10}(b/a)\rceil + 1$ nodes
  equally spaced in $\ln E$ (at least ten per decade), at which $\sigma$ is computed
  from parameters interpolated by their own laws. Every step of a parameter or radius
  table becomes a node with both one-sided values, so interpolation never bridges a
  jump. All sectors must share one law and one grid.
- **Per-sector conventions** (a model with an explicit aggregate policy, which GNDS and
  kika's GNDS extensions can state and ENDF cannot): each sector's terms are tabulated on
  its own grid with its own law and summed, and the potential term
  $4\beta\sum_\ell(2\ell+1)\sin^2\varphi_\ell$ is either computed continuously or
  tabulated with its own law.

**5. LSSF=1.** The resonance contribution is zero at every energy of the range: MF 3
already holds the dilute cross-section. The parameters are not evaluated, and nothing is
added.

**6. Tabulation.** The range is one or more segments of the material's grid (cut at its
boundaries, at histogram steps and at parameter jumps), seeded with the nodes above, and
refined and verified exactly as in {doc}`resonance-reconstruction`. At the boundary with
the resolved range the right-hand region owns the point and the jump is kept.

## Conventions and choices

- **Real degrees of freedom.** $\nu_y$ may be any non-negative real number; it is not
  rounded to an integer, and the integral is computed for it directly, with no
  fixed-point quadrature table. $\nu_y = 0$ means a width that does not fluctuate.
- **Capture in cases A and B is fixed** ($\nu_\gamma = 0$): these cases state no AMUG.
- **Cross-sections are interpolated, not parameters**, wherever a law is declared, and
  the wide-panel fill makes that interpolation fine enough where the grid is coarse.
- **Competition only widens the denominator.** Its cross-section belongs to MF 3 (of its
  own MT), so no competitive resonance contribution is produced.
- **One potential term per $\ell$**, with the phase radius, whatever the number of $J$
  sectors of that $\ell$.
- **Negative results are kept and reported**; a log interpolation law (INT=4, 5) on
  values that are not all positive (or all zero) is refused.

## Limits and failure modes

What this page computes is the infinitely dilute average at 0 K, which is what a transport
code needs where self-shielding in the range does not matter. Self-shielding is carried by
probability tables and Bondarenko factors (what NJOY PURR produces, MacFarlane et al.
2017), which kika computes from the same parameters at the temperatures asked for:
{doc}`probability-tables`. What is computed nowhere: a competitive cross-section from
MF 2; interference between the unresolved and resolved ranges; Bondarenko factors by NJOY
UNRESR's quadrature (the probability tables give them by sampling, as PURR does).

Refused (`UnsupportedResonanceError` or `ValueError`), before any evaluation:

- a range without a radius policy (`missing-radius-policy`); an unknown NAPS;
- a $J$ that $\ell$ and $I$ cannot reach (`invalid-J`), a duplicate $(\ell, J)$, sectors
  of one $\ell$ with different masses;
- a sector without a neutron width, a non-zero reduced $\bar\Gamma_n^0$ with $\nu_n = 0$,
  a channel that cannot be identified as neutron, capture, fission or competitive, a
  physical-width convention on a channel other than the neutron one, a negative $\nu$;
- parameter tables that do not cover the range, overlap or leave gaps; non-positive
  spacings or negative widths; an interpolation law outside INT=1-5;
- sectors with different cross-section laws or grids and no aggregate policy;
- more than one unresolved range in MF 2 (the conversion then reports unsupported data);
- a neutron penetrability that underflows, or a quadrature that fails its error or
  balance check ({eq}`eq-unresolved-resonances-balance`).

## Validation

The tests below passed on 2026-10-11, with the compiled integrand loaded.

- **Against a direct average.** The cross-sections of a sector with fixed capture,
  fission and competitive widths, for $\nu_n = 0.75, 1, 1.5, 2.4$, against an
  independent quadrature over the $\chi^2$ density of the neutron width (not a Laplace
  transform), relative $3\cdot10^{-10}$
  (`kika/processing/tests/test_unresolved.py::test_cross_sections_against_direct_gamma_average`).
- **Width-fluctuation averages.** 28 reference values from independent direct and
  Laplace calculations, relative $2\cdot10^{-12}$, with and without the compiled
  integrand (`kika/processing/tests/test_unresolved_fluctuations.py::test_independent_width_products`);
  invariance under scaling of all widths by $10^{-240}$ to $10^{240}$ and under
  permutation, $3\cdot10^{-12}$ (`::test_scaling_and_permutation`); the fixed-width limit,
  $10^{-15}$ (`::test_fixed_width_limit`); $\nu = 10^{12}$ tends to it, $3\cdot10^{-11}$
  (`::test_large_real_nu_converges_to_fixed_widths`); the compiled integrand against
  Python, $3\cdot10^{-13}$
  (`::test_native_quadrature_callback_matches_python_and_unavailable_binary`).
- **Interpolation.** Cross-sections, not parameters, are interpolated, $10^{-14}$
  (`kika/processing/tests/test_unresolved.py::test_interpolate_cross_sections_not_parameters`);
  a panel from 100 eV to 10 keV gets 21 nodes, at most $10^{0.1}$ apart
  (`::test_wide_panel_is_filled_before_sigma_interpolation`); every scalar law follows
  the shared algebra
  (`kika/processing/tests/test_unresolved_completion.py::test_all_scalar_sigma_laws_follow_declared_algebra`);
  the ENDF case B reader declares INT=2 and the kernel uses it
  (`kika/processing/tests/test_unresolved.py::test_case_b_reader_declares_sigma_law_and_kernel_uses_it`).
- **Ownership and conventions.** LSSF=1 adds nothing
  (`::test_self_shielding_only_never_reconstructs_again`); MF 3 competition and LSSF in
  a full suite (`::test_suite_assembly_retains_mf3_competition_and_lssf`); the resolved
  radius is never a fallback (`::test_global_radius_is_never_a_urr_fallback`); mixed
  laws are refused (`::test_mixed_sigma_conventions_are_not_resolved_by_the_last_group`);
  per-sector policies sum their terms and one potential
  (`kika/processing/tests/test_unresolved_completion.py::test_mixed_group_policies_sum_resonance_terms_and_one_potential`).
- **Boundaries and publication.** The resolved/unresolved boundary keeps its jump, and the
  result is published and reloaded
  (`kika/processing/tests/test_unresolved.py::test_rrr_urr_boundaries_and_native_publication`).

No test in this repository compares these averages with NJOY UNRESR at infinite
dilution; the comparison was made once on the corpus (2026-10-08, NJOY 2016.78, Linux
build, tolerance $10^{-6}$; 1161 unresolved ranges accepted). The largest relative
differences in capture are 0.26 % (Ba-140), 0.67 % (Kr-83) and 4.5 % (Tb-160), all
ENDF/B-VIII.1, and persist at $10^{-5}$. At the evaluation's own nodes Tb-160 differs by
at most 0.0044 %: between nodes kika interpolates the cross section by the stated law
where UNRESR evaluates the parameters, and an independent evaluation confirms kika at the
energies of largest difference.

## Diagnostics

The reconstruction report ({doc}`resonance-reconstruction`) describes each unresolved
range in its resonance-range rows (case, LSSF, number of sectors and energies, and the
treatment: the INT used, evaluation at every grid energy, or the LSSF=1 note that MF 3
already holds the dilute cross-section). `preparation_notes` records the law and number of
nodes, and the conventions applied ("real nu; fixed nu=0; competition only in total
width"). A caller of `PreparedResonances.evaluate` with a `diagnostics` dictionary also
receives `urr_max_quadrature_relative_error` (the largest relative error a quadrature
reported), `urr_max_width_product_balance_error` (the left side of
{eq}`eq-unresolved-resonances-balance`) and `urr_laplace_integrals` (how many integrals
were computed).

## Implementation

- `kika.processing.resonances.unresolved`: preparation (radii, parameter functions,
  conventions, grids), the cross-sections of {eq}`eq-unresolved-resonances-sigma`, and
  their tabulation on the declared grids.
- {py:mod}`kika.processing.resonances.fluctuations`: `width_products`, the integrals of
  {eq}`eq-unresolved-resonances-laplace`; its integrand has a compiled form in
  `kika.processing.resonances._rm_native`, used when loaded.
- `kika.endf.model_adapter.resonances`: the reading of cases A, B and C and of NRO, NAPS
  and LSSF into the model.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: Ch. 2 (unresolved parameters, LRU=2), App. D (unresolved
  cross-sections).
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: UNRESR, PURR.
- C. E. Porter and R. G. Thomas, "Fluctuations of nuclear reaction widths", *Physical
  Review* 104, 483-491 (1956): the $\chi^2$ width distributions. The width-fluctuation
  integrals are those of NJOY's UNRESR (MacFarlane et al. 2017, above).
