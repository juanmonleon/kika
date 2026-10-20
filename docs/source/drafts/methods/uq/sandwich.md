# Propagating covariances to a response with the sandwich rule

```{admonition} At a glance
:class: note
**Computes:** the first-order nuclear-data uncertainty of a response,
$\sigma_R/R = \sqrt{S^\top C S}$, from a multigroup sensitivity profile and relative
covariance matrices, with its breakdown by nuclide and reaction, the part due to
cross-reaction correlations, and a confidence interval for the statistical uncertainty of
the sensitivities.

**Applies to:** sensitivity profiles from SDF files, MCNP, Serpent or the benchmark
database; multigroup cross-section covariances (MF 33 type, and $\bar\nu$ blocks under
MT 452, 455 and 456) and multigroup Legendre-coefficient covariances (MF 34 type).

**Entry points:** {py:func}`kika.UQ.sandwich.sandwich_uncertainty_propagation`;
the alignment alone, {py:func}`kika.UQ.alignment.align_sensitivity_covariance` and
{py:func}`kika.UQ.alignment.prepare_covariance`.

**Status:** stable — since 0.1.1 (strict alignment since 0.3.0).
```

## Background

To first order in the parameters $x_i$ of the nuclear data, a change $\delta x$ moves the
response by $\delta R / R = \sum_i S_i\,\delta x_i / x_i$, with $S_i$ the sensitivity
coefficients of {doc}`sensitivities`. If the relative changes have covariance $C_{ij} =
\operatorname{Cov}(\delta x_i/x_i, \delta x_j/x_j)$, the relative variance of the response is

$$
\left(\frac{\sigma_R}{R}\right)^2 = \sum_{i,j} S_i\, C_{ij}\, S_j = S^\top C\, S ,
$$ (eq-sandwich-rule)

the first-order (sandwich) rule. Its terms are relative because both $S$ and $C$ are:
ENDF (Evaluated Nuclear Data File) evaluations state their covariances mostly relative
(MF 33 for cross-sections, MF 34 for Legendre coefficients, ENDF-6 Ch. 33-34), processed
into multigroup form, and a sensitivity is a relative derivative.
Here $i$ runs over parameters (nuclide, reaction or Legendre order, energy group).
(First-order propagation: Cacuci 2003.)

## What it computes

Input: one sensitivity profile with its response value $R$, and covariance blocks
$C_{kl}$ (one $G \times G$ matrix per pair of parameter keys $k$, $l$ on the profile's
$G$-group grid). Output:

$$
V = S^\top C\, S, \qquad u = \sqrt{|V|}, \qquad \sigma_R = u\,|R| ,
$$ (eq-sandwich-result)

with $u$ the relative uncertainty, and with $S_k$ the $G$ sensitivities of key $k$,

$$
V_k = S_k^\top (C S)_k, \qquad
A_k = S_k^\top C_{kk}\, S_k, \qquad
V_\text{corr} = \sum_k (V_k - A_k) = \sum_{k \ne l} S_k^\top C_{kl}\, S_l ,
$$ (eq-sandwich-contributions)

where $\sum_k V_k = V$. $V_k$ is the contribution of key $k$ with its correlations to every
other key, $A_k$ its contribution alone (correlations between its own groups included),
and $V_\text{corr}$ what cross-reaction correlations add. A bootstrap interval for $u$
accounts for the statistical uncertainty of $S$.

## Method

**1. The profile.** An SDF object is turned into a format-neutral profile (ascending grid in
MeV, region-integrated profiles only; see {doc}`sensitivities`). Then:

- MT 1 is always removed: it is the sum of the reactions the profile also lists, and a
  covariance of the total would count the same uncertainty twice;
- MT 0 (a pseudo-total some files carry) is removed unless the reaction filter names it;
- the reaction filter keeps, per ZAID, the listed MTs (an empty list keeps all of that
  ZAID, and an absent ZAID is dropped), or with the key `"ALL_NUCLIDES"` the listed MTs for
  every nuclide;
- $\bar\nu$ is kept once per nuclide. With the total (MT 452) and one or both components
  (prompt MT 456, delayed MT 455) present, `nubar_mode="total"` (the default) keeps the
  total and drops the components; `"components"` keeps the components and drops the total,
  but only when both are present, and keeps the total otherwise. The mode may be given per
  ZAID.

Nothing left after this is an error.

**2. Parameter keys.** A profile entry with MT below 4000 is the cross-section key
(xs, ZAID, MT); MT $4000 + \ell$ is the Legendre key (legendre, ZAID, 2, $\ell$), the
order-$\ell$ coefficient of elastic scattering. Covariance blocks carry the same keys:
cross-section blocks by (ZAID, MT) of row and column, Legendre blocks by (ZAID, MT,
$\ell$).

**3. The grids.** The profile's grid, converted to MeV, is the reference. Every
covariance's grid is converted from its stated unit to MeV and must have the same number
of boundaries and agree boundary by boundary within a relative tolerance (default
$10^{-6}$ here, with no absolute tolerance). No interpolation or condensation is done;
condensing is a separate, explicit step ({doc}`sensitivities`).

**4. The blocks.** Each block is checked to be square, finite and of the grid's size.
A block flagged absolute is made relative with the cross-sections the covariance carries,

$$
C^\text{rel}_{kl,gh} = \frac{C^\text{abs}_{kl,gh}}{\sigma_{k,g}\,\sigma_{l,h}} ,
$$ (eq-sandwich-relative)

and a covariance with no flags at all is taken as relative, which the report records. A
diagonal block ($k = l$) that is not symmetric is symmetrised as $(C + C^\top)/2$ when
the asymmetry is within $5\cdot10^{-7}$ relative and $2\cdot10^{-8}$ absolute (the
single-precision rounding of binary COVERX libraries), and refused otherwise. Each block is
stored with its transpose under $(l, k)$; a block given twice must agree to $10^{-12}$.

**5. Matching keys.** A key of the profile is covered when the covariance has its
diagonal block $C_{kk}$. By default (`alias_policy="exact"`) only the same key covers it.
With `alias_policy="tsurfer"`, an uncovered key may be covered by an alias: the bound or
natural ZAIDs 4309 and 4509 to 4009, 1801 and 1901 to 1001, 6312 to 6000; MT 101 and 102
for each other; and the MCNP identifier MT −2 to MT 101. An alias with two covering
candidates is refused. When two keys of one profile map to the same covariance key, the
one needing fewer changes wins (the exact key, then a ZAID alias, then an MT alias) and
the other is excluded and recorded; a tie is refused. A key nothing covers is an error
(`MissingCovarianceError`), unless `missing="drop"`, which drops it and records the loss.
Any other negative MT is refused. The alias table is TSURFER's (SCALE 6.3.1 manual).

**6. Assembly.** The covered keys are sorted (by kind, ZAID, MT, order) and laid out
$G$ groups each, giving $S$ (zeros for a key a profile does not list) and the absolute
standard deviations $\sigma_S$ of its entries. $C$ is assembled block by block; a pair of
keys with no block gets zeros, and is recorded. The assembled matrix must be symmetric to
$10^{-12}$.

**7. The rule.** {eq}`eq-sandwich-result` and {eq}`eq-sandwich-contributions` are
evaluated with one product $C S$. The absolute uncertainty needs $R$: a profile with no
response value, or a zero one, is refused.

**8. The bootstrap** (on by default). With $\varepsilon^{(b)}$ a vector of independent
standard normals, $b = 1, \dots, B$ (default $B = 1000$, optional seed),

$$
S^{(b)} = S + \sigma_S \odot \varepsilon^{(b)}, \qquad
u^{(b)} = \sqrt{\big|S^{(b)\top} C\, S^{(b)}\big|} ,
$$ (eq-sandwich-bootstrap)

and the interval at level $q$ (default 0.95) is the pair of percentiles
$100\,(1-q)/2$ and $100\,(1+q)/2$ of the $u^{(b)}$; their mean and standard
deviation (with $B - 1$) are reported too. It needs $\sigma_S$ for every entry.

## Conventions and choices

- **Relative throughout.** The rule is evaluated on relative quantities, and $u$ is the
  relative uncertainty whatever the response's unit; the absolute one is $u\,|R|$.
- **Strict alignment.** Grids must agree after unit conversion and keys must be covered,
  or the call fails. Partial overlaps, interpolated grids and silently dropped reactions
  give a number that looks like the answer and is not; dropping is available, explicit and
  recorded in the report with how much of the profile it removed.
- **A missing block is a zero correlation.** Two covered keys with no stated covariance
  between them are uncorrelated, which is what an evaluation that states no cross block
  asserts.
- **Totals are not counted with their parts.** MT 1 is removed, and $\bar\nu$ is counted
  once, because a profile and a covariance library usually state both the total and the
  partials.
- **Aliases are opt-in.** The TSURFER table maps keys that name the same physics in
  different libraries (a bound-atom ZAID, capture as MT 101 or 102); applying it by default
  would hide a mismatch between the profile and the library.
- **$\sqrt{|V|}$.** A covariance that is not positive semi-definite can give $V < 0$; the
  uncertainty is taken from $|V|$, and $V$ is reported with its sign so that the case is
  visible.
- **The bootstrap is parametric and covers $S$ only.** The groups of $S$ are drawn
  independently from normals with the stated absolute standard deviations. It does not
  account for the uncertainty of the covariance itself, nor for the Monte Carlo
  uncertainty of $R$, which is reported (`response_error`) and not combined.

## Limits and failure modes

Refused (`ValueError`, or its subclasses `AlignmentError` and `MissingCovarianceError`):

- no covariance; a confidence level outside $(0, 1)$; fewer than two bootstrap samples;
- no reaction left after the filters; no key left after alignment;
- a covariance grid that does not match the profile's, a covariance without a multigroup
  grid, an unknown energy unit;
- a block that is not square, not finite, of the wrong size, a diagonal block that is
  materially asymmetric, a block given twice with different values, an assembled matrix
  that is not symmetric;
- an absolute block without the row and column cross-sections, or with a zero one;
- an uncovered key under `missing="error"`; an ambiguous alias; an alias collision of equal
  standing; a negative MT other than −2;
- a bootstrap over a profile with unknown sensitivity uncertainties (repeat with
  `bootstrap=False`); a profile without a non-zero response value.

Reported instead of done: everything the alignment decided, in `alignment_report` (see
*Diagnostics*); excluded $\bar\nu$ entries, as a warning.

Limits of the method: the rule is first-order, so it describes the response only where
the response is linear in the data over the spread the covariance implies (the
second-order coefficients of {doc}`sensitivities` measure this); the result is only as
fine as the common group structure, and only covers the keys both sides state.

## Validation

The tests below passed on 2026-10-11 unless marked.

- **Hand calculations.** One group, one reaction, $S = 0.1$, $C = 0.04$: $V = 0.004$;
  two groups and two reactions with a block-diagonal $C$: $V = 1.80625\cdot10^{-3}$ and
  each $V_k$ equal to its hand value; one group, two reactions with correlation 0.5:
  $V = 5.2\cdot10^{-3}$ and $V_\text{corr} = 1.2\cdot10^{-3}$. All to $10^{-10}$
  (`kika/UQ/tests/test_sandwich.py::TestSandwichFormulaBasic::test_hand_calculation_verification_simple`,
  `::TestSandwichFormulaBasic::test_hand_calculation_2x2_matrix`,
  `::TestSandwichFormulaBasic::test_hand_calculation_with_correlation`).
- **Contributions.** $\sum_k V_k$ and $V - V_\text{corr}$ agree to $10^{-10}$
  (`::TestContributionCalculations::test_contributions_sum_to_total`); Legendre-only and
  mixed cross-section and Legendre profiles propagate with both covariance types
  (`::TestSandwichFormulaLegendre::test_legendre_only_basic_propagation`,
  `::TestSandwichFormulaMixed::test_mixed_data_propagation`).
- **The bootstrap** equals the percentiles, mean and standard deviation of
  {eq}`eq-sandwich-bootstrap` drawn independently with the same seed
  (`::TestSandwichFormulaBasic::test_bootstrap_uses_additive_absolute_sensitivity_sigma`).
- **Selections.** MT 0 is excluded unless named
  (`::test_mt0_pseudo_total_is_excluded_from_uq`); with $\bar\nu$ total and components
  present, $V = 0.3^2 \cdot 0.0025$ in `"total"` mode and the sum of the component terms
  in `"components"` mode, to $10^{-9}$
  (`::TestNubarDoubleCounting::test_total_nubar_kept_components_dropped`,
  `::TestNubarDoubleCounting::test_components_mode_keeps_prompt_delayed`).
- **Alignment** (`kika/UQ/tests/test_alignment.py`, 19 tests): union alignment with
  inserted zeros and zero blocks recorded; eV covariance grids converted, a grid off by
  10 % at one boundary refused; an absolute block made relative,
  {eq}`eq-sandwich-relative`; each TSURFER alias refused under `"exact"` and applied under
  `"tsurfer"`; coverage after `missing="drop"` (0.5 of the keys, 0.25 of $\sum|S|$ in the
  test); collisions, duplicate blocks, single-precision asymmetry and material asymmetry.
- **Against CALINS on a real case.** For the HEU-COMP-INTER-003-001 profile of the DICE
  database condensed to SCALE 44 groups, with the SCALE 6.2.3 44-group covariance library,
  $u$ is required to match a frozen CALINS value to $5\cdot10^{-10}$ absolute
  (`kika/benchmarks/tests/test_real_dice_scale44.py::test_real_dice_scale44_sandwich_and_ck_match_frozen_calins_reference`).
  It needs a DICE database in kika's schema and the SCALE 44-group COVERX file, and
  is skipped without them (passed 2026-10-20).

## Diagnostics

{py:func}`~kika.UQ.sandwich.sandwich_uncertainty_propagation` returns an
`UncertaintyResult`:

- `total_variance` ($V$) and `total_uncertainty` ($u$); `relative_uncertainty` is the same
  number as `total_uncertainty`, both relative; `response_value` ($R$) and
  `response_error` (the Monte Carlo uncertainty of $R$, not combined);
- `contributions`: per key, `variance_contribution` ($V_k$), `relative_contribution`
  ($V_k/V$), `uncertainty_contribution` ($\sqrt{|V_k|}$) and `auto_variance_contribution`
  ($A_k$); a Legendre key is shown as MT $4000 + \ell$. **A negative $V_k$** means
  correlations with other keys reduce the variance;
- `correlation_effects` ($V_\text{corr}$); `n_reactions`, `n_energy_groups`;
- `bootstrap_ci_low`, `bootstrap_ci_high`, `bootstrap_mean`, `bootstrap_std`,
  `bootstrap_n_samples`, `ci_level`. **An interval that is wide compared with $u$** means
  the statistical uncertainty of the sensitivities matters for the answer;
- `alignment_report`: `aliases`, `dropped` and `missing_covariance`, `policy_exclusions`
  (keys removed by the filters, MT 0, MT 1, $\bar\nu$ and alias preference),
  `zeros_inserted`, `zero_covariance_blocks`, `assumptions` (unit conversions, blocks
  treated as relative, conversions from absolute, symmetrisations), and
  `parameter_coverage` and `sensitivity_coverage`, the fractions of the profile's keys and
  of its $\sum |S|$ that were kept. **Coverage below 1** means part of the response's
  sensitivity has no covariance and is missing from $u$.

## Implementation

- `kika.UQ.sandwich`: the profile selections, the rule, the contributions, the bootstrap.
- `kika.UQ.alignment`: parameter keys, grid checks, block normalisation, aliases,
  assembly and the report; `prepare_covariance` normalises the blocks once for several
  profiles on one grid.
- `kika.sensitivities.profile`: the format-neutral profile.
- `kika.cov.cross_section_covariance`, `kika.cov.multigroup.mg_legendre_covariance`: the
  covariance containers the blocks come from.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual: Data Formats and Procedures for the Evaluated
  Nuclear Data Files ENDF/B-VI, ENDF/B-VII and ENDF/B-VIII*, Brookhaven National
  Laboratory (2023), doi:10.2172/2007538. Cited as ENDF-6: Ch. 33-34 (covariances of
  cross-sections and of angular distributions).
- D. G. Cacuci, *Sensitivity and Uncertainty Analysis, Volume I: Theory*, Chapman &
  Hall/CRC (2003): first-order (sandwich) uncertainty propagation.
- W. Wieselquist and R. A. Lefebvre (eds.), *SCALE 6.3.1 User Manual*, ORNL/TM-SCALE-6.3.1,
  Oak Ridge National Laboratory (2023), doi:10.2172/1959594: TSURFER (the alias table).
- B. Petillon, P. Arthur, R. Vuiart, V. Jaiswal and M. Brovchenko, "CALINS: a new ASNR
  software for sensitivity and uncertainty analysis in nuclear criticality safety
  evaluations", NCSD 2025, American Nuclear Society, 282-289 (2025),
  doi:10.13182/NCSD25-47777.
