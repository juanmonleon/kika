# Comparing an application with benchmarks by similarity

```{admonition} At a glance
:class: note
**Computes:** the covariance-weighted similarity $c_k$ of two sensitivity profiles, the
sandwich uncertainty of a benchmark stored in the benchmark database, a ranking of the
database's benchmarks by $c_k$ against an application, and a screening of the benchmarks
by their integral sensitivity to one nuclide and reaction in an energy region.

**Applies to:** sensitivity profiles from SDF files, MCNP, Serpent or the benchmark
database (built from the user's own DICE sensitivity files); the multigroup covariances
of {doc}`sandwich`.

**Entry points:** {py:func}`kika.UQ.similarity.similarity_ck`,
{py:func}`kika.benchmarks.uq.benchmark_uncertainty`,
{py:func}`kika.benchmarks.uq.rank_benchmarks_by_ck`,
{py:func}`kika.benchmarks.find_sensitive_benchmarks`,
{py:func}`kika.benchmarks.rank_benchmarks`.

**Status:** stable — since 0.3.0.
```

## Background

Two systems respond to the same nuclear data through their sensitivity profiles $S_a$
(the application) and $S_e$ (a benchmark experiment). With $C$ the relative covariance of
the data, the first-order uncertainties of the two responses are
$\sigma_a^2 = S_a^\top C S_a$ and $\sigma_e^2 = S_e^\top C S_e$ ({doc}`sandwich`), and the
covariance between them is $S_a^\top C S_e$. Their correlation coefficient,

$$
c_k = \frac{S_a^\top C\, S_e}{\sqrt{\left(S_a^\top C\, S_a\right)\left(S_e^\top C\, S_e\right)}},
$$ (eq-similarity-and-benchmarks-ck)

is the fraction of the two nuclear-data uncertainties they share: $c_k = 1$ when the
profiles are proportional with a positive factor, $-1$ with a negative one, near 0 when
the data that drive one hardly affect the other. A benchmark with $c_k$ close to 1 tests
the same data, with the same weights, as the application ($c_k$ as TSUNAMI-IP defines it,
SCALE 6.3.1 manual).

The ICSBEP benchmarks distributed with DICE come with sensitivity profiles in SDF form,
several per benchmark when it was modelled with more than one code, library or group
structure. kika ships no benchmark data: the database is built from the user's own DICE
files (Briggs 2005; Nouri et al. 2003).

## What it computes

{py:func}`~kika.UQ.similarity.similarity_ck`: input, two profiles and the covariance (as
for {doc}`sandwich`); output, $c_k$ by {eq}`eq-similarity-and-benchmarks-ck`, the three
quadratic forms, and per parameter key $k$ a local value from its diagonal block alone,

$$
c_{k}^{(k)} = \frac{S_{a,k}^\top C_{kk}\, S_{e,k}}
{\sqrt{\left(S_{a,k}^\top C_{kk}\, S_{a,k}\right)\left(S_{e,k}^\top C_{kk}\, S_{e,k}\right)}} ,
$$ (eq-similarity-and-benchmarks-local)

which is a diagnostic: the local values do not add up to $c_k$.

{py:func}`~kika.benchmarks.uq.rank_benchmarks_by_ck`: $c_k$ of the application against
every candidate profile in the database, sorted. {py:func}`~kika.benchmarks.uq.benchmark_uncertainty`:
{eq}`eq-sandwich-result` for one database profile, with its stored $k_\text{eff}$ as the
response. {py:func}`~kika.benchmarks.find_sensitive_benchmarks`: the benchmark profiles
whose integral sensitivity to a (nuclide, reaction) in an energy region passes a
threshold, sorted by its magnitude.

## Method

**1. Selections.** Each profile is filtered before alignment: MT 1 is removed unless
`include_mt1=True`; MT 0 is removed unless the `include` list names it; `include` keeps
only the listed (ZAID, MT) pairs, `exclude` removes the listed ones; and $\bar\nu$ is
counted once, as in {doc}`sandwich` (`nubar_mode`, default `"total"`). Nothing left is an
error.

**2. Alignment.** Both profiles are aligned together with the covariance
({py:func}`~kika.UQ.alignment.align_sensitivity_covariance`, described in
{doc}`sandwich`): the second grid must match the first, the keys are the union of both
profiles' keys, and a key one profile lacks is a zero in its vector. The default grid
tolerance is a relative $10^{-6}$.

**3. The value.** With $V_a = S_a^\top C S_a$, $V_e = S_e^\top C S_e$ and
$V_{ae} = S_a^\top C S_e$, $c_k = V_{ae}/\sqrt{V_a V_e}$. It is undefined, and refused, when
$V_a$ or $V_e$ is zero, negative or not finite. A value with $|c_k| > 1 + 10^{-10}$ is
refused, since by the Cauchy-Schwarz inequality it cannot occur for a symmetric positive
semi-definite $C$; within that margin it is clipped to $[-1, 1]$. The local values,
{eq}`eq-similarity-and-benchmarks-local`, are computed where both local variances are
positive and are `None` otherwise.

**4. Ranking.** The covariance is normalised once for the application's grid
({py:func}`~kika.UQ.alignment.prepare_covariance`) and reused for every candidate. The
candidates are the database profiles, one per benchmark (its preferred variant) by
default, optionally restricted to a list of benchmark IDs. Each gets
`similarity_ck` with the same options; a candidate that fails to align stops the ranking
with an `AlignmentError` that names it. The rows are sorted by $c_k$ descending (or by
$|c_k|$ with `rank_by_absolute=True`), then by benchmark ID and profile ID, and cut at
`limit` (default 100; `None` for all).

**5. Benchmark uncertainty.** The profile is read from the database as a format-neutral
profile, with the profile's computed $k_\text{eff}$ and its uncertainty as the response,
and passed to {py:func}`~kika.UQ.sandwich.sandwich_uncertainty_propagation` with the
caller's options.

**6. Screening.** When the database is built, each stored profile of a benchmark gets its
integral sensitivity per region,

$$
S_\text{region} = \sum_{g\,:\,E_{g+1} \in \text{region}} S_g ,
$$ (eq-similarity-and-benchmarks-region)

with each group assigned by its upper edge $E_{g+1}$: thermal if $E_{g+1} \le 0.625$ eV,
epithermal if $0.625\text{ eV} < E_{g+1} \le 100$ keV, fast above, and total for every
group; also $\sum_g |S_g|$. {py:func}`~kika.benchmarks.find_sensitive_benchmarks` resolves the
nuclide (ZAID, `Fe56` or `Fe-56`) and the reaction (MT, an alias such as `elastic` or
`capture`, or kika's reaction label), and returns the region-integrated profiles (unit 0,
region 0) of that nuclide and reaction with $|S_\text{region}|$ strictly above
`sensitivity_threshold` (default 0) and, when given, $|S_\text{region}| / \sum_g |S_g|$
strictly above `fraction_threshold`, sorted by $|S_\text{region}|$ descending, at most
`limit` rows. With no reaction, every MT except 0 matches. By default only the preferred
variant of each benchmark is searched. {py:func}`~kika.benchmarks.rank_benchmarks` is the
same search with no thresholds.

**7. Preferred variants.** Of the profiles of one benchmark, the preferred one is the
first in the order: code KENO, then MCNP, then others; library ENDF/B-VIII.1, VIII.0,
VII.1, VII.0, ABBN-93, then others; group structure 238 groups, 299 groups, continuous
energy, then others; then the duplicate number of the file name (the `.<n>` suffix,
0 when absent), lowest first.

## Conventions and choices

- **Signed ranking by default.** A benchmark with $c_k$ near $-1$ shares the application's
  uncertainty but with the opposite sign of the effect; it is not a substitute for a
  similar benchmark, so the ranking puts it last unless `rank_by_absolute=True`.
- **Strict ranking.** A candidate that cannot be aligned fails the whole ranking. Leaving it
  out would make the ranking look complete when it is not; `missing="drop"` is the
  explicit way to rank on the covered keys, and each row carries the coverage of both
  profiles.
- **Deterministic ties.** Equal values are ordered by benchmark ID and profile ID, so that
  the same database gives the same ranking.
- **Local values are not a decomposition.** $c_k$ is a ratio of quadratic forms and does
  not split into additive terms; the per-key values say which keys are similar on their
  own, nothing more.
- **Regions by the upper edge.** On the DICE group structures 0.625 eV and 100 keV are
  group boundaries, so the assignment is exact there; on a grid where a group straddles a
  limit, the whole group goes to the region of its upper edge.
- **System totals only.** Screening and the UQ adapters use only region-integrated
  profiles; per-mixture or per-region breakdowns (when the database stores them) would be
  counted twice.
- **Single precision in the database.** Profiles and grids are stored as 32-bit floats; the
  default grid tolerance of $10^{-6}$ absorbs that rounding when a database profile is
  aligned with a covariance in double precision. The region integrals are computed from the
  parsed values before they are stored.

## Limits and failure modes

Refused:

- `similarity_ck`: everything the alignment refuses ({doc}`sandwich`); no reaction left
  after the filters; $V_a$ or $V_e$ zero, negative or not finite
  (`ZeroSimilarityVarianceError`); $|c_k| > 1 + 10^{-10}$ (`ValueError`);
- `rank_benchmarks_by_ck`: a negative `limit` (`ValueError`); an application that is not a
  profile or SDF object (`TypeError`); a candidate that cannot be aligned (`AlignmentError`
  naming the benchmark and profile);
- `find_sensitive_benchmarks`: an unknown region or reaction name (`ValueError`); an
  isotope or reaction of the wrong type (`TypeError`);
- every database call: no database configured or found (`DatabaseNotConfiguredError`), an
  unknown benchmark or profile (`BenchmarkNotFoundError`).

Reported instead of done: the alignment report of every pair, with its coverage.

Limits of the method: $c_k$ inherits the first-order assumption and the covariance of
{doc}`sandwich`, and is only as good as the covariance: two systems sensitive to data
with no stated uncertainty look dissimilar. It measures the similarity of the
nuclear-data uncertainty, not of the systems.

## Validation

The tests below passed on 2026-10-11 unless marked.

- **The definition.** A profile against itself gives $c_k = 1$ and against its negative
  $-1$ (`kika/UQ/tests/test_similarity.py::test_ck_positive_and_negative_unit_limits`).
  A two-reaction, two-group case gives $V_a = 0.0613$, $V_e = 0.06982$,
  $V_{ae} = 0.04117$ and $c_k = 0.63$, equal to a frozen CALINS result to a relative
  $10^{-6}$ (`::test_ck_matches_frozen_calins_reference`).
- **Refusals and selections.** A zero profile is refused
  (`::test_ck_zero_covariance_norm_is_an_error`); a missing covariance fails unless
  `missing="drop"` (`::test_similarity_missing_policy_is_forwarded`); MT 0 and MT 1 are
  excluded and recorded (`::test_similarity_excludes_mt0_pseudo_total_by_default`,
  `::test_similarity_records_policy_exclusions`); a covariance grid off by
  $5\cdot10^{-7}$ relative at one boundary still aligns
  (`::test_similarity_default_grid_tolerance_covers_float32_boundaries`).
- **The database adapters.** On a one-benchmark database built from a 4-group SDF,
  `benchmark_uncertainty` gives the same variance and bootstrap interval as the direct call,
  `similarity_ck` of the profile with itself is 1, and the ranking normalises the
  covariance once (`kika/benchmarks/tests/test_benchmarks.py::test_neutral_profile_and_uq_adapters`);
  equal $c_k$ are ordered by benchmark ID (`::test_ck_ranking_tie_breaks_by_benchmark_id`).
- **Screening.** For a U-235 fission profile of $(0.05, 0.05, 0.15, 0.25)$ on groups with
  upper edges $10^{-5}$ eV, 0.625 eV, 100 keV and 20 MeV, the totals are 0.10 thermal,
  0.25 fast and 0.5 in all, to $10^{-6}$ (`::test_screening_and_integrals`); thresholds
  are strict (`::test_threshold_filters`); MT 0 is screened only when asked for
  (`::test_mt_zero_is_only_screened_when_explicit`); isotope and reaction names resolve
  (`::test_isotope_and_reaction_aliases`).
- **Against CALINS on a real case.** HEU-COMP-INTER-003-001 against HEU-COMP-INTER-003-002
  from the DICE database, condensed to SCALE 44 groups, with the SCALE 6.2.3 44-group
  covariance library: $c_k$ is required to match a frozen CALINS value to
  $5\cdot10^{-10}$ absolute
  (`kika/benchmarks/tests/test_real_dice_scale44.py::test_real_dice_scale44_sandwich_and_ck_match_frozen_calins_reference`).
  It needs a DICE database in kika's schema and the SCALE 44-group COVERX file, and
  is skipped without them (passed 2026-10-20).

## Diagnostics

- `SimilarityResult`: `value` (also `ck`), `variance_a`, `variance_b`, `cross_covariance`;
  `reaction_similarity`, the local values with their three quadratic forms;
  `parameter_keys` and `index` (what each element of the aligned vectors is);
  `alignment_report` ({doc}`sandwich`). **Coverage below 1 on either profile** means
  $c_k$ ignores part of that profile's sensitivity.
- `BenchmarkSimilarity` rows of a ranking: `benchmark_id`, `profile_id`, `ck`,
  `n_parameters`, the parameter and sensitivity coverage of the application and of the
  benchmark, `code`, `library`, `group_structure`, and the full `result`.
- Screening rows: benchmark ID, category, spectrum and type, profile ID, code, library,
  group structure, computed $k_\text{eff}$, the nuclide and reaction, `s_region`, `s_total`,
  `s_abs_total` and `fraction` ($|S_\text{region}| / \sum_g |S_g|$).

## Implementation

- `kika.UQ.similarity`: the selections, $c_k$ and the local values.
- `kika.UQ.alignment`: alignment of the two profiles, shared normalised covariance.
- `kika.benchmarks.uq`: the database adapters and the ranking.
- `kika.benchmarks.screening`, `kika.benchmarks.database`: name resolution, the screening
  query, candidates and profiles.
- `kika.benchmarks.ingest`: the database build, the region integrals and the preferred
  variants; `kika.benchmarks._constants`: the region limits, the priority lists and the
  reaction aliases.

## References

- W. Wieselquist and R. A. Lefebvre (eds.), *SCALE 6.3.1 User Manual*, ORNL/TM-SCALE-6.3.1,
  Oak Ridge National Laboratory (2023), doi:10.2172/1959594: TSUNAMI-IP, the definition of $c_k$.
- J. B. Briggs, "ICSBEP benchmarks for nuclear data applications", *AIP Conference
  Proceedings* 769, 1717-1722 (2005), doi:10.1063/1.1945340.
- A. Nouri, P. Nagel, J. B. Briggs and T. Ivanova, "DICE: database for the International
  Criticality Safety Benchmark Evaluation Program Handbook", *Nuclear Science and
  Engineering* 145, 11-19 (2003), doi:10.13182/NSE03-15.
- B. Petillon, P. Arthur, R. Vuiart, V. Jaiswal and M. Brovchenko, "CALINS: a new ASNR
  software for sensitivity and uncertainty analysis in nuclear criticality safety
  evaluations", NCSD 2025, American Nuclear Society, 282-289 (2025),
  doi:10.13182/NCSD25-47777.
