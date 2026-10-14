# Computing unresolved-resonance probability tables

```{admonition} At a glance
:class: note
**Computes:** at each energy of the unresolved resonance range and each temperature,
probability tables of the total, elastic, fission and capture cross-sections: bands of
the total cross-section with their probabilities and conditional partial cross-sections,
stated as factors on the smooth (dilute) cross-sections, with a statistical error for
every number of the table.

**Applies to:** the unresolved range (MF 2, LRU=2, LSSF=0 or 1) of an incident-neutron
model suite read from ENDF-6 or GNDS; written to a PENDF as MF 2/MT 153 and to an
OpenMC HDF5 file as its `urr` tables.

**Entry points:** {py:func}`kika.processing.resonances.probability_tables.probabilityTables`
with {py:class}`kika.processing.resonances.probability_tables.ProbabilityTableOptions`;
through {py:func}`kika.endf.process_endf` (the `urr` stage of a
{py:class}`kika.endf.ProcessingPlan`) and {py:func}`kika.hdf5.processing.processSuite`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

In the unresolved resonance range an evaluation states the statistics of the resonances
and not the resonances themselves (ENDF-6 Ch. 2, LRU=2): for each spin sector
$(\ell, J)$ a mean level spacing $D$, average widths $\bar\Gamma_y$ of the neutron,
capture, fission and competitive channels, and the number of degrees of freedom $\nu_y$ of
the $\chi^2$ distribution each width follows. The infinitely dilute average cross-section
follows from these parameters by averaging over the width distributions
({doc}`unresolved-resonances`). It is the right cross-section only where the flux is not
depressed by the resonances. In a dense absorber the flux dips at every resonance, and the
reaction rate depends on how the cross-section is distributed around its average, not on
the average alone.

The probability-table method (Levitt 1972) carries that distribution. At an energy $E$ the
total cross-section, seen at random energies near $E$, is a random variable; its range is
cut into $M$ bands, and the table gives each band's probability $p_b$ and the mean total,
elastic, fission and capture cross-sections conditional on the total being in that band.
A Monte Carlo code samples a band at each collision and uses its cross-sections. The
distribution is obtained by drawing resonance ladders: sequences of resonances whose
spacings follow Wigner's distribution and whose widths follow the $\chi^2$ distributions
of the evaluation (Porter and Thomas 1956), evaluated with the single-level Breit-Wigner
formula and Doppler broadened. This is what NJOY PURR does (MacFarlane et al. 2017);
kika follows PURR's procedure and keeps its choices where they are choices.

The same bands give the Bondarenko self-shielding factor (Bondarenko 1964): with a
background cross-section $\sigma_0$ standing for the other nuclides of a mixture, the
flux-weighted cross-section of reaction $x$ relative to its dilute average.

## What it computes

Input: the unresolved region of the model with its target mass ratio $A$ and spin $I$,
the energies $E_i$ of the table, the temperatures $T$ (K), the smooth background
cross-sections at each $E_i$ (from MF 3), and the reactions that compete in the range.
Output, for each temperature, a `ProbabilityTables` of the model
(`kika.nuclear_data.model.processed_forms`): an array $(N, 6, M)$ over the $N$ energies
and $M$ bands, holding for each band the cumulative probability and the total, elastic,
fission, capture and heating factors, with the interpolation law between energies (2,
lin-lin), the two competition flags, `multiplySmooth` true, and the statistics of the
table (`ProbabilityTableStatistics`).

At one energy, let $\sigma_x(\varepsilon)$, $x \in \{t, n, f, \gamma\}$, be the total,
elastic, fission and capture cross-sections of a ladder at the sample energy
$\varepsilon$, and let the $L$ ladders give $LN$ samples. With $\mathcal{B}_b$ the samples
whose total falls in band $b$, of number $N_b$,

$$
p_b = \frac{N_b}{LN},\qquad
\bar\sigma_{x,b} = \frac{1}{N_b}\sum_{s\in\mathcal{B}_b}\sigma_x(\varepsilon_s),\qquad
\bar\sigma_x = \frac{1}{LN}\sum_{s}\sigma_x(\varepsilon_s),\qquad
\phi_{x,b} = \frac{\bar\sigma_{x,b}}{\bar\sigma_x} .
$$ (eq-probability-tables-factors)

The table stores the cumulative probabilities $\sum_{b'\le b} p_{b'}$ and the factors
$\phi_{x,b}$. A transport code multiplies each factor by the smooth cross-section of its
reaction at the collision energy, so that, since $\sum_b p_b\,\phi_{x,b} = 1$ by
construction, the table reproduces the smooth cross-section on average and carries only
its distribution. The smooth cross-section is the MF 3 background plus the dilute average
of {doc}`unresolved-resonances` (LSSF=0), or MF 3 alone (LSSF=1).

## Method

### Energies of the table

The table energies are PURR's: the two ends of the range rounded to seven significant
figures and moved one unit of the seventh figure inward, the interior energies of the
declared parameter grid (the level-spacing energies of the first spin sector, else the
range's energy grid), and, in every gap between consecutive nodes wider than a factor
1.26, the energies of NJOY's standard unresolved grid (10 eV to 8.5 MeV, about ten per
decade) that lie inside the gap and more than 1 % above the previous energy. A range whose
parameters are constant (no declared grid) is filled the same way over its whole span.

### Frozen parameters

At each table energy $E$ every sector's parameters are evaluated from their own functions
and then held fixed for the ladders: the spacing $D$, the average widths
$\bar\Gamma_n, \bar\Gamma_\gamma, \bar\Gamma_f, \bar\Gamma_x$ (the physical neutron width
from the reduced one as in {eq}`eq-unresolved-resonances-neutron-width`), the degrees of
freedom, the hard-sphere phase shift $\varphi_\ell$ at the phase radius, and
$g_J = (2J+1)/[2(2I+1)]$. With $k^2 = 2 m_n E\,[A/(A+1)]^2/\hbar^2$, the potential
cross-section is

$$
\sigma_\text{pot} = \frac{4\pi}{k^2}\sum_\ell (2\ell+1)\sin^2\varphi_\ell ,
$$ (eq-probability-tables-potential)

one term per $\ell$, taken from the first sector of that $\ell$. The radii and the
refusals are those of the dilute average ({doc}`unresolved-resonances`), whose
preparation is reused; for LSSF=1 the parameters are prepared all the same, since the
tables need them even where MF 3 already holds the average.

### Ladders

A ladder is abstract: it starts at $E_\text{low} = 10$ eV and spans
$0.9\,n_\text{max}\,D_\text{min}$, with $n_\text{max} = 1000$ and $D_\text{min}$ the
smallest spacing of the sectors at $E$. Every sector draws its own sequence over the whole
span. The first resonance of a sector is at $E_\text{low} + D\sqrt{4/\pi}\,u$ and each
next one a Wigner spacing further,

$$
E_{r+1} - E_r = D\sqrt{-\tfrac{4}{\pi}\ln u},\qquad
p(s) = \frac{\pi s}{2D^2}\,e^{-\pi s^2/(4D^2)} ,
$$ (eq-probability-tables-wigner)

with $u$ uniform on $(0, 1)$, until the end of the span plus one $D\sqrt{4/\pi}$. Each
resonance draws its widths independently, channel by channel:

- $\bar\Gamma_y \le 0$: the width is zero;
- $\nu_y = 0$: the width is $\bar\Gamma_y$ (it does not fluctuate);
- $\nu_y \in \{1, 2, 3, 4\}$: $\Gamma_y = (\bar\Gamma_y/\nu_y)\,c_{\nu_y, j}$, with $j$
  uniform on the 20 entries of PURR's table $c_{\nu, j}$ of equiprobable $\chi^2_\nu$
  quantiles;
- any other real $\nu_y$: $\Gamma_y = \bar\Gamma_y\,\chi^2_{\nu_y}/\nu_y$ from the
  continuous distribution.

The total width is $\Gamma = \Gamma_n + \Gamma_\gamma + \Gamma_f + \Gamma_x$; the
competitive width enters $\Gamma$ only. A resonance with $\Gamma = 0$ is dropped.

### Cross-sections at the sample energies

$N = 10\,000$ sample energies are drawn uniformly over the span less $300/\rho$ at each
end, with $\rho = \sum 1/D$ the total level density, so that no sample sees the edge of a
ladder. With the Doppler width $\Delta = \sqrt{4 k_B T E/A}$ (at the table energy $E$),
each resonance $r$ of a sector adds, at every sample within
$31.83\,\Gamma_r + \max(31.83\,\Gamma_r, 20\,\Delta)$ of $E_r$,

$$
\begin{aligned}
\sigma_n &\mathrel{+}= \sigma_r^0\Big[\big(\cos 2\varphi_\ell - 1 + \Gamma_{n,r}/\Gamma_r\big)\,\psi(x_r, \xi_r)
 + \sin 2\varphi_\ell\;\chi(x_r, \xi_r)\Big],\\
\sigma_\gamma &\mathrel{+}= \sigma_r^0\,\frac{\Gamma_{\gamma,r}}{\Gamma_r}\,\psi(x_r, \xi_r),\qquad
\sigma_f \mathrel{+}= \sigma_r^0\,\frac{\Gamma_{f,r}}{\Gamma_r}\,\psi(x_r, \xi_r),\\
\sigma_r^0 &= \frac{4\pi}{k^2}\,g_J\,\frac{\Gamma_{n,r}}{\Gamma_r},\qquad
x_r = \frac{2(\varepsilon - E_r)}{\Gamma_r},\qquad \xi_r = \frac{\Gamma_r}{\Delta} ,
\end{aligned}
$$ (eq-probability-tables-slbw)

the single-level Breit-Wigner terms with the free-gas Doppler line shapes (ENDF-6
App. D),

$$
\psi(x,\xi) = \frac{\sqrt\pi\,\xi}{2}\,\operatorname{Re} w\!\Big(\frac{\xi(x+i)}{2}\Big),\qquad
\chi(x,\xi) = \frac{\sqrt\pi\,\xi}{2}\,\operatorname{Im} w\!\Big(\frac{\xi(x+i)}{2}\Big),
$$ (eq-probability-tables-psi-chi)

where $w(z) = e^{-z^2}\operatorname{erfc}(-iz)$ is the Faddeeva function; at $T = 0$,
$\psi = 1/(1+x^2)$ and $\chi = x/(1+x^2)$. The sample's cross-sections are then

$$
\sigma_n = \max\big(\sigma_\text{pot} + \textstyle\sum_r(\ldots) + \sigma^\text{bg}_{2},\;10^{-6}\ \text{b}\big),\quad
\sigma_\gamma = \textstyle\sum_r(\ldots) + \sigma^\text{bg}_{102},\quad
\sigma_f = \textstyle\sum_r(\ldots) + \sigma^\text{bg}_{18},\quad
\sigma_t = \sigma_n + \sigma_\gamma + \sigma_f + \sigma^\text{bg}_c ,
$$ (eq-probability-tables-sample)

with $\sigma^\text{bg}_{\text{MT}}$ the smooth backgrounds below and $\sigma^\text{bg}_c$
the competition. All temperatures use the same ladders and the same sample energies, so
the tables of different temperatures are correlated as PURR's are.

### Bands

The band boundaries come from the totals of the first ladder, sorted, at each
temperature. With $n_e = \lfloor N/(M - 8.24)\rfloor$, boundary $i = 1, \ldots, M-1$ is
the sorted total of rank $r_i$, where $r_1 = \max(\lfloor n_e/200\rfloor, 1)$ and
$r_{i+1} = r_i + \lfloor w_i n_e\rfloor$ with

$$
(w_1, \ldots, w_{M-2}) = \big(\tfrac{1}{40}, \tfrac{1}{10}, \tfrac{1}{4}, \tfrac{1}{2},
\underbrace{1, \ldots, 1}_{M-10}, \tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{10}, \tfrac{1}{40}\big) ;
$$ (eq-probability-tables-bands)

a boundary not above the previous one is set to the previous one times 1.05, and the last
band is open above. The nominal probabilities are therefore
$(\tfrac{1}{200}, \tfrac{1}{40}, \tfrac{1}{10}, \tfrac{1}{4}, \tfrac{1}{2}, 1, \ldots, 1,
\tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{10}, \tfrac{1}{40}, \tfrac{1}{200})/(M - 8.24)$:
finer in both tails, where self-shielding lives. A sample of total $\sigma_t$ belongs to
the band $b$ with $t_{b-1} < \sigma_t \le t_b$. The samples of every ladder are then
accumulated in these fixed bands and {eq}`eq-probability-tables-factors` gives the table.
The cumulative probabilities are rounded to seven significant figures and the last is
set to 1. The heating row holds $10^6$ in every band, which is how the official OpenMC
files store PURR's unit factor; no transport code reads it (OpenMC uses the cumulative
probabilities, the elastic, fission and capture rows and the reaction the inelastic flag
names).

### Backgrounds and competition

The smooth backgrounds are PURR's (its reading of MF 3), at each table energy (the two
ends nudged inward by a relative $10^{-6}$):

- **LSSF=0.** $\sigma^\text{bg}_2$, $\sigma^\text{bg}_{18}$, $\sigma^\text{bg}_{102}$ are
  the MF 3 values of MT 2, 18 and 102, and
  $\sigma^\text{bg}_c = \sigma_1 - \sigma_2 - \sigma_{18} - \sigma_{102}$ from MF 3 (MT 1,
  or the sum of the exclusive reactions when the file has no MT 1).
- **LSSF=1.** MF 3 already is the dilute average, so the partial backgrounds are zero.
  Only the competition is kept: the excess of the total over $\sigma_2 + \sigma_{18} +
  \sigma_{102}$ where it exceeds $10^{-6}$ of the total, zero below the first energy with
  an excess, and zero everywhere when no reaction competes.

A reaction competes when the total exceeds elastic, fission and capture somewhere in the
range, and it is one of the MF 3 sections PURR reads, other than MT 1-4, 18, 19 and 102,
with its threshold below the top of the range. The competition flags follow PURR: the
inelastic flag is the MT of the only competing level reaction (MT 51-91), 4 when there
are several, and $-1$ when there is none; the absorption flag is the MT of the only other
competing reaction, 0 when there are several, and $-1$ when there is none.

### Statistical error

The ladder is the statistical unit: the samples of one ladder share its resonances and
are not independent, while different ladders are independent and identically distributed
given the bands. For every estimate $\theta$ of the table (each $p_b$, each
$\phi_{x,b}$, each band's share $p_b\phi_{x,b}$ of the dilute average, each Bondarenko
factor), let $\theta_{(i)}$ be the same estimate from the sums of all ladders but $i$,
with the bands held at their boundaries. The delete-one-ladder jackknife error is

$$
s_\theta = \sqrt{\frac{L-1}{L}\sum_{i=1}^{L}\big(\theta_{(i)} - \bar\theta_{(\cdot)}\big)^2},
\qquad \bar\theta_{(\cdot)} = \frac{1}{L}\sum_i \theta_{(i)} ,
$$ (eq-probability-tables-jackknife)

and infinite when $L < 2$. The Bondarenko factors are computed from the bands at the
backgrounds $\sigma_0$ of the options (1, 10, 100, 1 000 and 10 000 b by default):

$$
\sigma_x(\sigma_0) = \frac{\sum_b p_b\,\sigma_{x,b}/(\sigma_{t,b} + \sigma_0)}
{\sum_b p_b/(\sigma_{t,b} + \sigma_0)},\qquad
f_x(\sigma_0) = \frac{\sigma_x(\sigma_0)}{\sum_b p_b\,\sigma_{x,b}} ,
$$ (eq-probability-tables-bondarenko)

$f_x = 1$ for a reaction with no average. Two bases are available for the band
cross-sections $\sigma_{x,b}$. By default they are the band means
$\bar\sigma_{x,b}$ of the ladder cross-sections themselves (`ladder_cross_sections`).
With the smooth cross-sections of a processed file (`openmc_transport`, which
{py:func}`kika.hdf5.processing.processSuite` uses), they are what OpenMC's reader forms:
$\phi_{n,b}$ times the smooth elastic, $\phi_{f,b}$ times the smooth fission, the capture
factor times the smooth non-fission absorption, and a total rebuilt as their sum plus the
smooth inelastic competition. The basis changes only the statistics; the table is the
same.

With an error target $\tau$ (`errorTarget`), the ladders are checked every `checkEvery`
ladders (10 by default) below the cap `ladders`: drawing stops at an energy once, at
every temperature and for every reaction whose factors are not all exactly 1, every
band's $s_{p\phi}$ is at most $\tau$ and every $f_x(\sigma_0)$ has a relative error at
most $\tau$. The random stream is the same with or without the target, so a table that
stopped after $n$ ladders is identical to one drawn with $n$ ladders.

## Conventions and choices

- **PURR's procedure, kika's random numbers.** Every choice listed in *Method* (ladder
  span, sample count, margins, reach of a resonance, $\chi^2$ table, band spacing,
  competition rule, energy grid) is PURR's. The random numbers are NumPy's, seeded per
  energy by `(seed, energy index)`, so the tables agree with PURR's statistically, band
  by band, and not bit for bit; they do not depend on how the energies are split among
  processes.
- **64 ladders and 20 bands by default.** This is what OpenMC's `make_ace` asks PURR
  for (`20 64`). The ladders are shared by all temperatures.
- **The tables are factors (`multiplySmooth`).** Each band's conditional mean is divided
  by the mean of all samples, so the table carries no magnitude. The magnitude comes from
  the smooth cross-section the transport code already has, and a table computed with
  slightly different parameters cannot shift the dilute average.
- **The table as a whole is normalised, not each ladder.** PURR can rescale every ladder
  to the dilute average (its `nmode = 1`); it does not by default, and kika does not at
  all: the rescaling divides by a ladder's mean resonance elastic, which interference
  can put near zero.
- **PURR's discrete $\chi^2$ table for 1-4 degrees of freedom.** The 20 equiprobable
  quantiles have no extreme tail, and that shapes the outer bands; the continuous
  distribution gives longer tails there. A real $\nu$ outside 1-4 (a non-integer AMUX,
  for example) uses the continuous distribution, which PURR cannot state.
- **LSSF=1 backgrounds are zero, not MF 3 minus the dilute average.** That is PURR's
  reading. MF 3 minus the average of the parameters would add a deterministic bias
  wherever the evaluation's MF 3 is not that average.
- **Approximate line shapes by default.** The compiled kernel evaluates the Faddeeva
  function with Humlíček's W4 approximation (Humlíček 1982), relative error of order
  $10^{-4}$, the precision class of PURR's own rational approximations;
  `native=False` evaluates it exactly in NumPy, the reference the kernel is tested
  against. The error is far below the statistical error of the table.
- **The statistics follow the ladder.** A per-sample error would treat the
  $N$ correlated samples of a ladder as independent and understate the error. The
  jackknife holds the band boundaries fixed, as FRENDY's statistical check does
  (`TODO(check)`: FRENDY's method, and a reference for it).

## Limits and failure modes

Refused before any ladder is drawn (`UnsupportedResonanceError` or `ValueError`): every
refusal of the unresolved preparation ({doc}`unresolved-resonances`: no radius policy, a
$J$ that $\ell$ and $I$ cannot reach, a sector without a neutron width, parameter tables
that do not cover the range, and the others listed there). Also refused (`ValueError`):
an `errorTarget` that is not positive, a `checkEvery` below 2, and smooth cross-sections
for the OpenMC basis that are not finite, non-negative arrays of one row per energy. A
`ProbabilityTables` whose array is not $(N, 6, M)$ over its $N$ energies is refused when
it is built.

Known limits of the method:

- The band boundaries come from one ladder, so they are themselves random; the jackknife
  does not include that variability.
- The band spacing {eq}`eq-probability-tables-bands` has $M - 10$ central bands and is
  meant for $M \ge 11$; the options do not check $M$.
- The ladders use the single-level Breit-Wigner formula with no interference between
  sectors, and the competitive width is never a cross-section of its own.
- Interference with the resolved range is not represented.
- The statistics are attached to the tables in memory; no file format kika writes
  carries them.

In {py:func}`kika.endf.process_endf` the stage never raises: a material without an
unresolved range is `skipped` (`urr-absent`), a plan without temperatures is `skipped`
(`urr-no-temperature`), parameters the method does not support give `rejected`
(`urr-unsupported`), and any other error `rejected` (`urr-failed`); every earlier result
is kept. {py:func}`kika.hdf5.processing.processSuite` raises `HDF5FormatError` when the
tables name an inelastic competitor of which the evaluation has no level, and
`ValueError` when tables passed in miss a requested temperature. The GNDS output of
{py:class}`kika.endf.EndfProcessing` omits the tables (GNDS 2.0, the schema kika writes,
has no element for them) and lists them under `omitted`.

## Validation

The tests below passed on 2026-10-14, with the compiled kernel loaded; the four on a whole
evaluation need the ENDF/B-VIII.1 Cf-252 tape and the `--deep` option, and were not run
on that date.

- **Line shapes.** The compiled ladder kernel against the exact NumPy reference
  ({eq}`eq-probability-tables-slbw` with `scipy.special.wofz`) at $\Delta = 0$, 0.05 and
  1, relative to the largest value, below $2\cdot10^{-4}$
  (`kika/processing/tests/test_probability_tables.py::test_native_ladder_matches_the_exact_reference`);
  the kernel's own $\sin$, $\cos$ within $2.3\cdot10^{-16}$ and $\exp$ within a relative
  $4.5\cdot10^{-16}$, and its W4 within a relative $10^{-4}$ of `wofz` over
  $-20 \le x \le 20$, $10^{-6} \le y \le 20$ (`::test_ladder_special_functions`).
- **PURR's conventions.** The competition flags (`::test_competition_flags_follow_purr`);
  band boundaries increasing and open above (`::test_band_bounds_are_increasing_and_open_at_the_top`);
  the $\chi^2$ quantile table averages $\nu$ within 2 %
  (`::test_chi_square_quantiles_have_the_right_means`); NJOY's seven-figure rounding
  (`::test_sigfig_is_njoys`); the grid's shaded ends and filled gaps
  (`::test_purr_grid_shades_the_ends_and_fills_wide_gaps`).
- **Statistical error.** With equal band counts the probabilities have no error, below
  $10^{-12}$ (`::test_jackknife_of_a_mean_is_the_clt_error`); out of sample, over 12
  independent replicas of 16 ladders, the spread of the elastic $f(1\ \text{b})$ over its
  mean jackknife error is between 0.5 and 1.6
  (`::test_statistics_match_the_spread_of_independent_replicas`); without a target every
  ladder is drawn (`::test_default_draws_every_ladder_and_reports_no_target`); a reached
  target gives the same table as that many ladders, and an unreachable one runs to the cap
  and says so (`::test_an_error_target_stops_early_with_the_same_table_as_that_many_ladders`);
  invalid targets are refused (`::test_error_target_must_be_positive`).
- **OpenMC basis.** The shielding uses the partial factors times the smooth
  cross-sections and ignores the total row (`::test_transport_shielding_uses_partial_factors_and_ignores_the_total_row`);
  the basis leaves the table unchanged (`::test_transport_diagnostics_do_not_change_fixed_count_tables`);
  redundant sums are not counted in the smooth absorption
  (`::test_processed_transport_absorption_excludes_redundant_sums`); the report line
  names only reactions with an average (`::test_report_line_names_only_reactions_with_an_average`).
- **On a whole evaluation** (Cf-252, ENDF/B-VIII.1, 16 ladders, skipped when the tape is
  not on the machine). The stage produces 20 bands at every energy with cumulative
  probabilities non-decreasing and ending at 1
  (`kika/endf/tests/test_process_endf.py::test_urr_tables_on_a_whole_tape`), without
  broadening (`::test_urr_tables_without_broadening`); every PENDF material carries
  MF 2/MT 153 (`::test_pendf_carries_mt153_at_each_temperature`); the HDF5 written from
  the tables of the stage equals the one the HDF5 writer computes from scratch
  (`::test_hdf5_equals_the_hdf5_writer_from_scratch`).

No test in this repository compares the tables with an independent calculation of the
same ladders' statistics on an evaluation; the comparison with PURR below is external.

### Comparison with NJOY PURR

This comparison is not a test of this repository: NJOY is neither a dependency of kika
nor the truth. The bands cannot be compared one by one, since their boundaries come from a
random ladder and no two tables share them; the comparison that does not depend on the
bands is the Bondarenko factor {eq}`eq-probability-tables-bondarenko`. kika's tables (20
bands, 64 ladders, at 294 K) were compared with those of the OpenMC library of
ENDF/B-VIII.1, processed by NJOY PURR (`TODO(check)`: the NJOY version that produced it),
at every energy of the grid (measured 2026-10-11). The grid and the competition flags are
identical to the official file's: U-235 (19 energies, flags (4, 0)), U-238 (31, (4, 0)),
Pu-239 (51, (51, −1)). The largest $\lvert f_\text{kika}/f_\text{PURR} - 1\rvert$ over the
energies, at $\sigma_0$ = 1, 10 and 100 b:

| Nuclide | MT | smallest $f$ of the official file at 1 b | 1 b | 10 b | 100 b |
|---|---|---|---|---|---|
| U-235 | 2 | 0.99 | 0.02 % | 0.01 % | 0.00 % |
| U-235 | 18 | 0.90 | 0.21 % | 0.14 % | 0.03 % |
| U-235 | 102 | 0.90 | 0.25 % | 0.17 % | 0.04 % |
| U-238 | 2 | 0.85 | 0.48 % | 0.27 % | 0.07 % |
| U-238 | 102 | 0.87 | 0.35 % | 0.20 % | 0.04 % |

The self-shielding itself is 10-15 % at 1 b, and the differences are of the order of the
statistical spread of 64 ladders (`TODO(check)`: the statistical error of both tables at
these energies was not stated with the comparison, nor how the factors of each table
were turned into barns for {eq}`eq-probability-tables-bondarenko`). The time for U-235 at
one temperature, 20 bands and 64 ladders, is 99 s for kika in one process and 24 s in
eight, against 99 s for PURR on the same machine (`TODO(check)`: NJOY version and date of
this timing).

## Diagnostics

- **Stage report** of {py:func}`kika.endf.process_endf` (`stage("urr")`): `status`
  (`done`, `skipped`, `rejected`), `category`, a one-line `summary` (bands, energies,
  range, temperatures, ladders), and `details`: `temperatures`, `n_energies`, `bins`,
  `ladders`, `range_eV`, and `error` when rejected.
- **`EndfProcessing.probability_tables`**: per temperature, `energies`, the six rows
  (`cdf`, `total`, `elastic`, `fission`, `capture`, `heating`) as $(N, M)$ arrays,
  `factors`, `interpolation`, `inelastic` and `absorption`.
- **Statistics** (`ProbabilityTables.statistics`): `ladders` drawn per energy, the
  jackknife errors `probabilitySigma`, `factorSigma` and `contributionSigma` (the band
  share $p_b\phi_{x,b}$, the statistic to watch: it is the band's error in units of the
  dilute average), `shielding` and `shieldingSigma` (the factors
  {eq}`eq-probability-tables-bondarenko` and their errors at each $\sigma_0$),
  `shieldingBasis`, and, with a target, `errorTarget` and `converged` per energy.
  `worst()` gives, per reaction, the largest band share error and the largest relative
  error of $f$ at each $\sigma_0$. A `converged` false means the target was not reached
  within the cap at that energy.
- **Conversion report** of {py:func}`kika.hdf5.processing.processSuite`: one warning
  stating that the tables are kika's (ladders, samples, bands, energies) and statistically
  equal to PURR's, and one line per temperature with the worst relative error of
  $f(\sigma_0)$ at the smallest $\sigma_0$ and the worst band share per reaction, the
  ladders drawn, the basis, and whether the target was reached at every energy. Tables
  passed in from another suite (a perturbation run sharing the unperturbed tables) are
  reported as not perturbed.

## Implementation

- `kika.processing.resonances.probability_tables`: the grid (`purrGrid`), the
  backgrounds and competition (`purrInputs`, {py:func}`kika.processing.resonances.probability_tables.competitionFlags`),
  the band boundaries ({py:func}`kika.processing.resonances.probability_tables.bandBounds`),
  the ladders, the sample cross-sections, the factors and the jackknife
  ({py:func}`kika.processing.resonances.probability_tables.probabilityTables`), energies in
  parallel processes when `workers` > 1.
- `kika.processing.resonances._rm_native`: the compiled ladder kernel (`urr_ladder`),
  with its own exponential, sine and cosine and Humlíček's W4; used when built and
  `native` is true.
- `kika.processing.resonances.unresolved`: the preparation of the sectors, shared with the
  dilute average.
- `kika.nuclear_data.model.processed_forms`: `ProbabilityTables` and
  `ProbabilityTableStatistics`.
- `kika.hdf5.processing`: `urrTablesOf` (grid, backgrounds and tables of a suite), the
  OpenMC smooth cross-sections of the statistics, and the attachment of the tables to
  MT 2, 18 and 102 under the `urr` style of each temperature.
- `kika.endf.process`: the `urr` stage, and MF 2/MT 153 of the PENDF. MT 153 is placed
  after MF 2/MT 151 in each broadened material, in PURR's layout: a HEAD record with ZA,
  AWR, the two competition flags, the interpolation law and $M$; a LIST record with the
  temperature, LSSF=1 and the number of energies; then, per energy, $E$, the band
  probabilities (not cumulative), the total, elastic, fission and capture factors, and a
  heating row of zeros (as PURR leaves it without HEATR), all to seven significant
  figures. The section is parsed back and compared with the table (relative $10^{-6}$)
  before the file is replaced, and the MF 1 directory is rebuilt.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: Ch. 2 (unresolved parameters, LRU=2), App. D (single-level
  Breit-Wigner, Doppler line shapes $\psi$ and $\chi$).
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: PURR.
- L. B. Levitt, The probability table method for treating unresolved neutron resonances in
  Monte Carlo calculations, *Nucl. Sci. Eng.* 49 (1972) 450-457.
- C. E. Porter and R. G. Thomas, Fluctuations of nuclear reaction widths, *Phys. Rev.* 104
  (1956) 483-491.
- I. I. Bondarenko (ed.), *Group Constants for Nuclear Reactor Calculations*, Consultants
  Bureau, New York (1964).
- J. Humlíček, Optimized computation of the Voigt and complex probability functions,
  *J. Quant. Spectrosc. Radiat. Transfer* 27 (1982) 437-444.
- K. Tada, A. Yamamoto, S. Kunieda, C. Konno, R. Kondo, T. Endo, G. Chiba, M. Ono and
  M. Tojo, "Development of nuclear data processing code FRENDY version 2", *Journal of
  Nuclear Science and Technology* 61, 830-839 (2024), doi:10.1080/00223131.2023.2278600.
  `TODO(ref)`: whether it describes the statistical check of probability tables.
- `TODO(ref)`: Wigner's spacing distribution.
