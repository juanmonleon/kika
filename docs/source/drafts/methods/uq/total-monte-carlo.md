# Separating nuclear-data and statistical variance in Total Monte Carlo

```{admonition} At a glance
:class: note
**Computes:** from the results of $N$ transport calculations, one per sampled evaluation,
each with its Monte Carlo statistical uncertainty: the observed spread, the mean
statistical uncertainty, the nuclear-data uncertainty that remains when the second is
subtracted from the first, bootstrap confidence intervals for them, their convergence with
the number of samples, and tests of the normality of the results.

**Applies to:** any set of per-sample results with relative statistical uncertainties
(MCNP tallies, $k_\text{eff}$), one response or a table of responses.

**Entry points:** `kika.UQ.fastTMC.fastTMC`, `kika.UQ.fastTMC.create_summary_table`,
`kika.UQ.convergence.convergence_analysis`, `kika.UQ.normality.normality_tests`,
`kika.UQ.normality.histogram_data`, `kika.UQ.normality.qq_plot_data`.

**Status:** experimental — since 0.1.1 (convergence and normality since 0.2.0).
```

## Background

In Total Monte Carlo (TMC), the evaluation is sampled $N$ times from its uncertainties
(for example with {doc}`../sampling/perturbation-pipeline`), the transport calculation is
run once per sample, and the spread of the $N$ results is the uncertainty of the response.
Each result $y_i$ also carries the Monte Carlo statistical uncertainty of its own run,
$\sigma_{\text{stat},i}$, so the observed spread mixes both. When the runs are statistically
independent of the sampling, the variances add,

$$
\sigma_\text{obs}^2 = \sigma_\text{ND}^2 + \overline{\sigma_\text{stat}^2} ,
$$ (eq-total-monte-carlo-decomposition)

and the nuclear-data part is what is left when the mean statistical variance is subtracted
from the observed one. Fast TMC uses this to run each sample with fewer histories
(a larger $\sigma_{\text{stat},i}$) and still recover $\sigma_\text{ND}$, provided the
statistical part does not dominate (Total Monte Carlo: Koning and Rochman 2008; Fast TMC:
Rochman et al. 2014).

## What it computes

Input: the results $y_1, \dots, y_N$ and their relative statistical uncertainties
$s_1, \dots, s_N$ (fractions). Output, with $\bar y$ the mean,

$$
\sigma_\text{obs}^2 = \frac{1}{N-1}\sum_{i=1}^N (y_i - \bar y)^2, \qquad
\overline{\sigma_\text{stat}^2} = \frac{1}{N}\sum_{i=1}^N (s_i\, y_i)^2, \qquad
\sigma_\text{ND}^2 = \max\!\left(\sigma_\text{obs}^2 - \overline{\sigma_\text{stat}^2},\, 0\right),
$$ (eq-total-monte-carlo-estimators)

each reported as a percentage of $|\bar y|$ ($100\,\sigma/|\bar y|$), with percentile bootstrap
intervals, and a flag that says whether the decomposition can be used,

$$
\text{reliable} \iff \frac{\sqrt{\overline{\sigma_\text{stat}^2}}}{\sigma_\text{obs}} < t ,
$$ (eq-total-monte-carlo-reliability)

with the threshold $t$ (default 0.5).

## Method

**1. The decomposition.** `fastTMC` evaluates {eq}`eq-total-monte-carlo-estimators` and
{eq}`eq-total-monte-carlo-reliability`. A result is unreliable when $\sigma_\text{obs} = 0$.
Given a table of responses (one column per tally, one row per sample, and a matching table
of relative uncertainties), it does this per column, on the first `n_samples` rows when
asked, and skips columns whose mean is zero. Given the results alone (no uncertainties), it
reports $\sigma_\text{obs}$ and its interval only, with the statistical and nuclear-data
parts set to zero and the flag set to true, since there is no decomposition to judge.

**2. The bootstrap.** For each of $B$ replicates (default 1000, optional seed), $N$ indices
are drawn with replacement, the pairs $(y_i, s_i)$ at those indices are taken together, and
{eq}`eq-total-monte-carlo-estimators` is evaluated on them. The interval at level $\gamma$
(default 0.95) is the pair of percentiles $100\,(1-\gamma)/2$ and $100\,(1+\gamma)/2$ of the
replicates, for $\bar y$, $\sigma_\text{obs}$ and $\sigma_\text{ND}$. The replicates are
drawn in blocks to bound memory; the stream of random numbers, and so the result, is the
same as drawing them one at a time.

**3. Convergence.** `convergence_analysis` evaluates the same quantities on the first $n$
samples, for a sequence of $n$, in the order the samples are given:

- `"random"`: every $n$ from 2 to $N$ when $N \le 200$, otherwise about 200 values of $n$
  spaced geometrically from 2 to $N$, both ends included;
- `"sobol"`: the powers of two, $n = 2, 4, 8, \dots \le N$;
- `"lhs"`: no analysis (the function returns `None`).

At each $n$ it reports the running mean, standard deviation and relative uncertainty
$100\,\sigma_\text{obs}/|\bar y|$ with bootstrap intervals ($B$ replicates of size $n$), and,
when the statistical uncertainties are given, the statistical and nuclear-data parts and an
interval for the latter. One generator serves every $n$, so a seed fixes the whole sequence.

**4. Normality.** `normality_tests` drops non-finite values and, with at least 8 left,
runs:

- Shapiro-Wilk, on all values up to 5000 and on a random subsample of 5000 (fixed seed 42)
  above; normal when $p > \alpha$ (default 0.05);
- Anderson-Darling, with the p-value of D'Agostino and Stephens (1986, Table 4.9) for a
  normal distribution whose mean and variance are estimated: the statistic $A^2$ is
  corrected to $A^{*2} = A^2\,(1 + 0.75/N + 2.25/N^2)$ and

  $$
  p =
  \begin{cases}
  \exp(1.2937 - 5.709\,A^{*2} + 0.0186\,A^{*4}) & A^{*2} \ge 0.6,\\
  \exp(0.9177 - 4.279\,A^{*2} - 1.38\,A^{*4}) & 0.34 \le A^{*2} < 0.6,\\
  1 - \exp(-8.318 + 42.796\,A^{*2} - 59.938\,A^{*4}) & 0.2 \le A^{*2} < 0.34,\\
  1 - \exp(-13.436 + 101.14\,A^{*2} - 223.73\,A^{*4}) & A^{*2} < 0.2,
  \end{cases}
  $$ (eq-total-monte-carlo-anderson-darling)

  clipped to $[0, 1]$; normal when $p > \alpha$;
- D'Agostino-Pearson $K^2$, with at least 20 values; normal when $p > \alpha$.

`histogram_data` gives a density histogram (default 30 bins) and the normal density with
the sample's mean and standard deviation (with $N$ in the denominator) at the bin centres;
`qq_plot_data` gives the standard normal quantiles at the plotting positions
$(i - \tfrac12)/N$ against the sorted values standardised with the same mean and standard
deviation.

## Conventions and choices

- **Unbiased observed variance, mean statistical variance.** $\sigma_\text{obs}^2$ uses
  $N - 1$; the statistical part is the mean of the per-sample variances $(s_i y_i)^2$, the
  expected statistical variance of one draw.
- **A negative difference is zero.** When the statistical part exceeds the observed spread,
  $\sigma_\text{ND}$ is set to 0, not to an imaginary value; with any threshold below 1 the
  reliability flag is then false.
- **The reliability rule is a ratio of standard deviations.** With $t = 0.5$ the
  statistical part must be below half the observed spread, so that the subtraction does not
  rest on the difference of two similar numbers. When it fails, the observed uncertainty is
  the one to report, and the printed summary says so.
- **Pairs are resampled together.** Each bootstrap replicate keeps $y_i$ with its own
  $s_i$, so the interval for $\sigma_\text{ND}$ includes the variability of both terms.
- **Percentages of the magnitude of the mean.** `fastTMC` and `convergence_analysis` divide
  by $|\bar y|$, in the estimates and in every bootstrap replicate, so a negative response
  (a reactivity difference, say) has the same positive relative uncertainties as its
  opposite.
- **Prefixes, not resamples, for convergence.** The running statistics follow the samples
  in the order they were drawn, which is what a run that stopped at $n$ would have given.
  A Sobol' sequence is balanced only at powers of two, so only those points are shown; a
  prefix of a Latin hypercube design is not itself a Latin hypercube, so no curve is drawn.

## Limits and failure modes

Refused (`ValueError`):

- for one response: results and uncertainties of different lengths; uncertainties whose
  mean exceeds 1, taken as percentages instead of fractions;
- both a single response and tables given, or neither; tables without their uncertainty
  table, columns missing from it, or an unknown column name or index.

Reported instead of done: an unreliable decomposition is flagged (`is_reliable`, and a
printed message); in the table mode, a column with zero mean is skipped and one whose
analysis fails is skipped with the error printed. `convergence_analysis` returns an empty
list for fewer than 2 samples, `normality_tests` for fewer than 8.

Limits of the method: {eq}`eq-total-monte-carlo-decomposition` assumes that the statistical
error of each run is independent of the sample and that its stated uncertainty is right;
the decomposition needs $N$ large enough for $\sigma_\text{obs}$ itself to be known, which
the bootstrap intervals and the convergence curves show.

## Validation

In `kika/UQ/tests/test_fast_tmc_and_normality.py`:

- a negative response gets positive percentages and intervals, equal to those of its
  opposite (`::test_a_negative_response_has_a_positive_relative_uncertainty`);
- the summary table leaves out a response with zero mean
  (`::test_the_summary_table_skips_a_zero_mean_without_failing`);
- {eq}`eq-total-monte-carlo-anderson-darling` gives 0.10, 0.05 and 0.01 to $2\cdot10^{-3}$
  at the tabulated critical values 0.631, 0.752 and 1.035 for large $N$
  (`::test_the_anderson_darling_p_value_meets_the_tabulated_critical_values`), and the
  decision follows $\alpha$ (`::test_the_anderson_darling_decision_follows_alpha`).

No test exercises `kika.UQ.convergence`, the bootstrap intervals' coverage, or the
Shapiro-Wilk and D'Agostino-Pearson tests.

## Diagnostics

- `fastTMC` for one response returns a dictionary: `mean_tally`, `percent_unc_observed`,
  `percent_unc_statistical`, `percent_unc_nuclear_data`, `is_reliable`,
  `stat_to_obs_ratio` (the left side of {eq}`eq-total-monte-carlo-reliability`),
  `bootstrap` (the replicates) and `ci` (the intervals of the mean, of the observed and of
  the nuclear-data uncertainty). **`is_reliable` false** means the nuclear-data value is not
  usable: report `percent_unc_observed`, or rerun with more histories.
- For tables it returns `create_summary_table`'s data frame, one row per response:
  `Mean_Tally`, `Total_Unc_%`, `Statistical_%`, `Nuclear_Data_%`, `Is_Reliable` and the
  interval bounds.
- `convergence_analysis`: one dictionary per $n$ (`n`, `running_mean`, `running_std`,
  `running_unc_pct`, their intervals, and with uncertainties `running_stat_unc_pct`,
  `running_nd_unc_pct` and the interval of the latter). **An interval that is still
  shrinking at $n = N$** means more samples would change the answer.
- `normality_tests`: per test `test_name`, `statistic`, `p_value` and `is_normal`.

## Implementation

- `kika.UQ.fastTMC`: the decomposition, the reliability flag, the bootstrap and the summary
  table.
- `kika.UQ.convergence`: the running statistics and their intervals.
- `kika.UQ.normality`: the normality tests, the histogram and the Q-Q data.

## References

- A. J. Koning and D. Rochman, "Towards sustainable nuclear energy: putting nuclear physics
  to work", *Annals of Nuclear Energy* 35, 2024-2030 (2008), doi:10.1016/j.anucene.2008.06.004.
- D. Rochman, W. Zwermann, S. C. van der Marck, A. J. Koning, H. Sjöstrand, P. Helgesson and
  B. Krzykacz-Hausmann, "Efficient use of Monte Carlo: uncertainty propagation", *Nuclear
  Science and Engineering* 177, 337-349 (2014), doi:10.13182/NSE13-32.
- R. B. D'Agostino and M. A. Stephens (eds.), *Goodness-of-Fit Techniques*, Marcel Dekker
  (1986): Table 4.9, the Anderson-Darling p-value with estimated parameters.
- `TODO(ref)`: the Shapiro-Wilk and D'Agostino-Pearson tests.
