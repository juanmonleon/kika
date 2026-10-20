# Computing first-order sensitivity coefficients

```{admonition} At a glance
:class: note
**Computes:** multigroup sensitivity coefficients $S_{k,g}$ of a Monte Carlo response to
the cross-sections of one nuclide, with their statistical uncertainty, from MCNP PERT
tallies or from Serpent sensitivity output; writes and reads them as SCALE Sensitivity
Data Files (SDF) and condenses them onto coarser nested group structures.

**Applies to:** MCNP inputs and MCTAL files with PERT cards (METHOD=2 and 3), Serpent
`.sens` output, SDF files in the verbose and the compact dialect.

**Entry points:** {py:func}`kika.mcnp.pert_generator.perturb_material`,
{py:func}`kika.mcnp.pert_generator.generate_pert_cards`,
{py:func}`kika.sensitivities.sensitivity_processing.compute_sensitivity`,
{py:func}`kika.sensitivities.sensitivity_processing.compute_total_sensitivity`,
{py:func}`kika.sensitivities.sensitivity_processing.create_sdf_data`,
{py:func}`kika.sensitivities.sensitivity_processing.create_sdf_from_serpent`,
`kika.sensitivities.sdf_parser.read_sdf`,
{py:func}`kika.sensitivities.condensation.condense_sensitivity_profile`.

**Status:** stable — since 0.1.1 (condensation since 0.3.0).
```

## Background

The sensitivity coefficient of a response $R$ (a tally, $k_\text{eff}$) to a parameter
$x$ is the relative change of the response per relative change of the parameter,

$$
S = \frac{\partial R / R}{\partial x / x} = \frac{x}{R}\,\frac{\partial R}{\partial x}.
$$ (eq-sensitivities-definition)

In multigroup form the parameter is the cross-section $\sigma_k(E)$ of one reaction $k$ of
one nuclide, scaled by the same factor over one energy group $g = [E_g, E_{g+1}]$, and
$S_{k,g}$ is the sensitivity to that group. The coefficients are dimensionless; the
sensitivity to a uniform change of $\sigma_k$ at all energies is $\sum_g S_{k,g}$. Plots
show the sensitivity per unit lethargy, $S_{k,g}/\Delta u_g$ with
$\Delta u_g = \ln(E_{g+1}/E_g)$.

MCNP estimates the change of a tally under a change of the material of a set of cells
with its PERT card: the card names the cells (CELL), the material that replaces theirs
(MAT) at a density (RHO), the reactions whose cross-sections change (RXN) and the
incident-energy range of the change (ERG). With METHOD=2 the tally of that card is the
first-order term $c_1$ of the Taylor expansion of the tally change, with METHOD=3 the
second-order term $c_2$ alone, so that for a perturbation scaled by $p$

$$
R(p) \approx R_0 + c_1\,p + c_2\,p^2 .
$$ (eq-sensitivities-taylor)

The PERT card and the differential-operator method are in the MCNP 6.3.0 manual.

Serpent computes the sensitivities itself and writes them, with their relative
statistical errors, to a `.sens` file (Aufiero et al. 2015). kika reads them and does not
recompute them.

The SCALE Sensitivity Data File (SDF) is the text format TSUNAMI writes and reads:
a header with the number of groups and profiles, the response and its uncertainty, the
group boundaries in eV from the highest down, then one block per (nuclide, reaction,
unit, region) profile with the groupwise sensitivities and their standard deviations.
(SCALE 6.3.1 manual, TSUNAMI.)

## What it computes

From MCNP: input, the input file with its PERT cards, the MCTAL file, the tally number and
the nuclide (ZAID). Output, a `SensitivityData` object: for every tally energy bin and for
the tally's energy integral, per reaction $k$ and perturbation group $g$,

$$
S_{k,g} = \frac{c_{1,k,g}}{R_0}, \qquad
\varepsilon_{S,k,g} = \sqrt{\varepsilon_0^2 + \varepsilon_{1,k,g}^2},
$$ (eq-sensitivities-coefficient)

where $R_0$ is the unperturbed tally, $c_{1,k,g}$ the METHOD=2 tally of the card for
reaction $k$ and group $g$, and $\varepsilon_0$, $\varepsilon_1$ their MCNP relative errors;
$\varepsilon_S$ is relative. When METHOD=3 cards are present, the Taylor coefficients
$c_1$, $c_2$, their relative errors and the ratio $c_2/c_1$ are kept as well
({py:class}`kika.sensitivities.sensitivity.TaylorCoefficients`).

From Serpent: the sensitivities as Serpent states them. From either: an SDF object
({py:class}`kika.sensitivities.sdf.SDFData`) whose groupwise uncertainties are absolute
standard deviations, $\sigma_{S,k,g} = |S_{k,g}|\,\varepsilon_{S,k,g}$, and which writes
and reads SDF files.

## Method

**1. The perturbed material.** {py:func}`~kika.mcnp.pert_generator.perturb_material`
builds the material MAT names. The composition is converted to atom fractions if it is
given by weight, normalised to $\sum_i f_i = 1$, and the fraction of the perturbed nuclide
$n$ is doubled. With $T = 1 + f_n$ the new total, the fractions are renormalised,
$f'_i = f_i/T$ for $i \ne n$ and $f'_n = 2f_n/T$, and the atom density becomes

$$
N' = N\,T, \qquad\text{so that}\qquad N' f'_i = N f_i \;(i \ne n), \qquad N' f'_n = 2\,N f_n .
$$ (eq-sensitivities-perturbed-material)

Every other nuclide keeps its number density, and the perturbed nuclide's is doubled: a
+100 % change of its macroscopic cross-sections, $p = 1$ in
{eq}`eq-sensitivities-taylor`. When the original density is given, the mass densities
follow from the mean atomic mass, $\rho = N \bar A / N_A \cdot 10^{24}$ with $N$ in
atoms/(b·cm), $\bar A = \sum_i f_i A_i$ before and $\sum_i f'_i A_i$ after; $A_i$ is
kika's atomic mass of the nuclide, or its mass number when the table has none. The
perturbed material gets the mass density $\rho'$ (in g/cm³) and the ID
`material_id * 100 + 1` unless one is given, and is written back in the original fraction
type unless another is asked for.
{py:func}`~kika.mcnp.pert_generator.perturb_materials` does the same for several
materials that contain the nuclide.

**2. The PERT cards.** {py:func}`~kika.mcnp.pert_generator.generate_pert_cards` writes,
for each material block, each reaction and each pair of consecutive group boundaries
$(E_g, E_{g+1})$, one card

```text
PERTn:n CELL=<cells> MAT=<pert_material> RHO=<pert_density> METHOD=2 RXN=<mt> ERG=<E_g> <E_g+1>
```

followed, with `order=2` (the default), by the same card with METHOD=3. Card numbers
continue after the highest PERT number already in the input. With no boundaries the ERG
keyword is left out and the cards are energy-integrated. The boundaries are written as
given, in MeV, the unit of MCNP's ERG keyword. When the nuclide is named, each material
block is preceded by the comment `c kika:pert_zaid=<ZAID> pert_mat=<material>`, which is
how the reader later knows which nuclide and which original material each card perturbs.
Lines longer than 80 columns are continued with `&`.

**3. The coefficients.**
{py:func}`~kika.sensitivities.sensitivity_processing.compute_sensitivity` reads the PERT
cards and the MCTAL file. The cards of METHOD=2 that belong to the nuclide (and to the
material, when one is selected) are grouped by reaction, in card order, and the $j$-th
card of a reaction is taken as group $j$ of the perturbation grid, which is the sorted
union of every card's ERG boundaries. For a tally over several cells, one cell is
selected by its identifier in the MCTAL file. For each tally energy bin $i$, labelled
$E_{i-1}$ to $E_i$ (the first from 0), and for the tally's energy integral when MCTAL
carries one, {eq}`eq-sensitivities-coefficient` gives $S_{k,g}$ and its relative error.
The METHOD=3 card $j$ of a reaction is paired with the METHOD=2 card $j$ of the same
reaction.

**4. Second order.** The nonlinearity at a fractional perturbation $p$ is the ratio of
the second- to the first-order term,

$$
\eta_{k,g}(p) = \frac{c_{2,k,g}\,p^2}{c_{1,k,g}\,p} = \frac{c_{2,k,g}}{c_{1,k,g}}\,p ,
$$ (eq-sensitivities-nonlinearity)

set to NaN where $c_1 = 0$, and {eq}`eq-sensitivities-taylor` is the second-order estimate
of the perturbed response. A small $|\eta|$ over the perturbation range a covariance
implies says that the first-order coefficient describes the response there.

**5. A nuclide in several materials.**
{py:func}`~kika.sensitivities.sensitivity_processing.compute_total_sensitivity` computes
the coefficients material by material (all on the same perturbation grid, which it
checks) and adds them,

$$
S_{k,g} = \sum_m S_{m,k,g}, \qquad
\varepsilon_{S,k,g} = \frac{\sqrt{\sum_m \big(S_{m,k,g}\,\varepsilon_{m,k,g}\big)^2}}{|S_{k,g}|},
$$ (eq-sensitivities-total)

with $\varepsilon = 0$ where the sum is zero; $c_1$ and $c_2$ are added the same way.

**6. Serpent.** `kika.serpent.parse_sens.read_sensitivity_file` reads the header counts
(materials, nuclides, perturbations, energy groups) and checks every list and every data
block against them. ZAI values ending in 0 (the ground state) lose that digit, so
`260560` becomes ZAID 26056. Perturbation labels are mapped to MT numbers: `total xs` to
1, `mt N xs` to $N$, an inelastic-scattering label to 4, `nubar total`, `prompt`,
`delayed` to 452, 456, 455, `chi total`, `prompt`, `delayed` to 1018, 1019, 1020, and a
Legendre moment `<channel> leg mom L` to $4000 + L$, which is also what the `leg=`
selector of `kika.serpent.sens.SensitivityFile` looks up for an order $L$. Responses
written per bin (`<name>_BIN_<k>`) are grouped under their base name.
{py:func}`~kika.sensitivities.sensitivity_processing.create_sdf_from_serpent` takes, for
every material, nuclide and MT, the energy-dependent values and relative errors of one
response, converts the errors to absolute ones as above and skips profiles whose every
value is below $10^{-15}$ in magnitude. When one MT carries two labels (Serpent writes
`mt 4 xs` and an inelastic-scattering label for the same reaction), its profile is their
average: the arithmetic mean of the values and the root mean square of the relative
errors. A material or nuclide filter selects the entries the file has; an entry the file
lacks is reported with a warning and selects nothing. Several files can be combined if their energy grids
agree. The grid is taken as Serpent writes it, in MeV: on the SCALE 44-group structure
Serpent's `SENS_E` runs from $10^{-11}$ to 20, with an edge at $2.53\cdot10^{-8}$, the
thermal 0.0253 eV. Serpent states no response value, so $(R_0, \sigma_{R_0})$ default to
$(1.0, 0.01)$ unless given.

**7. SDF objects and files.**
{py:func}`~kika.sensitivities.sensitivity_processing.create_sdf_data` takes the
coefficients of one tally energy key from one or more `SensitivityData` (all on the same
perturbation grid, with the same $R_0$ and $\varepsilon_0$ unless a common
`response_values` is given) and builds one profile per nuclide and reaction, skipping
profiles that are zero in every group. The response uncertainty becomes absolute,
$\sigma_{R_0} = |R_0|\,\varepsilon_0$; a `response_values` pair is taken as already
absolute.

{py:meth}`~kika.sensitivities.sdf.SDFData.write_file` writes the standard dialect: the
title, the group and profile counts, `R0 +/- sigma_R0`, the boundaries in eV from the
highest down, and per profile (sorted by ZAID and MT) a header line, a `unit region`
line, a fixed line, five scalars and then the sensitivities and the absolute standard
deviations from the highest group down, five per line in `ES14.6`. The five scalars are

$$
S = \sum_g S_g,\quad \sigma_S = \Big(\sum_g \sigma_g^2\Big)^{1/2},\quad
\sum_g |S_g|,\quad
S_\text{osc} = \sum_{g:\,\operatorname{sign} S_g = -\operatorname{sign} S} S_g,\quad
\sigma_\text{osc} = \Big(\sum_{g:\,\operatorname{sign} S_g = -\operatorname{sign} S} \sigma_g^2\Big)^{1/2},
$$ (eq-sensitivities-scalars)

where zero groups belong to neither side, and a profile with $S = 0$ takes the negative
side when it has one.

`kika.sensitivities.sdf_parser.read_sdf` reads both dialects found in SCALE output,
recognised per profile by its header line:

- **verbose**: `<nuclide> <reaction> <ZAID> <MT>` (the reaction name may contain spaces),
  a `unit region` line (trailing text is ignored), a second metadata line, the five
  scalars, the sensitivities and the standard deviations;
- **compact**: six fields, `<nuclide> <reaction> <ZAID> <MT> <unit> <value>`, one summary
  line and the sensitivities only. The unit is the header's and the region is set to 0.
  The dialect states no standard deviations, so they are unknown: the reaction is marked
  `error_known=False`, its `error` field holds zeros only because the format needs
  numbers, and a profile built from it carries `uncertainty=None`.

The response line may carry `+/-` and an uncertainty or not (then $\sigma_{R_0} = 0$).
Boundaries are converted from eV to MeV and both the grid and the groups are put in
ascending order. Profiles after the declared count (a verification footer) are not read.
For each verbose profile the five scalars are recomputed from the groups,
{eq}`eq-sensitivities-scalars`, with the stored integral's sign deciding the oscillating
side, and compared to the stored ones within $5\cdot10^{-7}$ relative and $5\cdot10^{-7}$
times $\sum_g |S_g|$ (or $\sum_g \sigma_g$) absolute; a mismatch is logged and the groups
are kept. An SDF written by older versions of kika (title line ending
`MCNP to SCALE sdf <N>gr`) stores relative uncertainties and MeV boundaries; it is read
only with `uncertainty_convention="relative"`, and its uncertainties are converted to
absolute on reading.

{py:meth}`~kika.sensitivities.sdf.SDFData.group_inelastic_reactions` replaces, per
nuclide, MT 51-91 by their sum under MT 4, with the standard deviations added in
quadrature. {py:meth}`~kika.sensitivities.sdf.SDFData.to_sensitivity_profile` hands the
data to the uncertainty calculations as a format-neutral profile on an ascending MeV grid,
keeping only the region-integrated profiles (unit 0, region 0).

**8. Condensation.**
{py:func}`~kika.sensitivities.condensation.condense_sensitivity_profile` maps a profile
onto a coarser grid whose boundaries are all boundaries of the source grid. Each target
boundary is matched to a source boundary within a relative tolerance (default $10^{-5}$,
which absorbs the six-digit rounding of DICE files), after converting both grids to MeV;
the first and last boundaries must match. With $P_{Gg} = 1$ when source group $g$ lies in
target group $G$ and 0 otherwise,

$$
S_G = \sum_g P_{Gg}\, S_g, \qquad \sigma_G = \Big(\sum_g P_{Gg}\, \sigma_g^2\Big)^{1/2}.
$$ (eq-sensitivities-condensation)

The response, its uncertainty and the label are kept; the condensation is appended to
the profile's metadata (`condensation_history`).

## Conventions and choices

- **A +100 % perturbation of the nuclide alone.** Doubling the nuclide's number density
  and keeping every other one makes the perturbation of the selected reactions exactly
  $p = 1$, so $c_1/R_0$ is the sensitivity with no further scaling. The first-order term
  is linear in $p$, so the size of the perturbation does not change $S$ as long as it is
  the one the division assumes; $c_2$ is the term at $p = 1$ and scales as $p^2$.
- **The reader assumes $p = 1$.** `compute_sensitivity` divides $c_1$ by $R_0$ only. Cards
  written for another perturbation size give coefficients scaled by that size.
- **Cards are matched to groups by order.** The $j$-th METHOD=2 card of a reaction is
  group $j$; this holds for cards written by `generate_pert_cards`, which writes them in
  ascending energy over one grid shared by every reaction.
- **Statistical errors are combined as independent.** $\varepsilon_S$ in
  {eq}`eq-sensitivities-coefficient` neglects the correlation between $R_0$ and $c_1$,
  which MCNP estimates from the same histories; the sum over materials,
  {eq}`eq-sensitivities-total`, and the condensation, {eq}`eq-sensitivities-condensation`,
  treat their terms as independent too. The condensation report states that assumption.
- **Absolute uncertainties inside kika, relative ones at the boundary.** MCNP and Serpent
  state relative errors and SCALE files absolute standard deviations; `SDFData` and the
  profiles hold absolute ones, so that a zero sensitivity can carry a non-zero
  uncertainty and nothing is converted twice.
- **MeV inside kika, eV in SDF files.** Grids are stored ascending in MeV; the writer
  converts to eV and to SCALE's descending order, the reader converts back.
- **Groups are the truth, the scalars are checked.** The five scalars of an SDF profile are
  derived quantities; the reader recomputes them, reports a disagreement and keeps the
  groups.
- **Condensation is exact or refused.** Summing sensitivities over nested groups is exact;
  a non-nested target would need a projection with its own assumptions about the shape
  within a group, which kika does not make silently.
- **Unknown uncertainties stay unknown.** A profile without uncertainties is condensed
  without them, not with zeros.

## Limits and failure modes

Refused (`ValueError` unless named otherwise):

- `compute_sensitivity`: no ZAID given and none in the metadata or the input comments;
  metadata with several ZAIDs, or several materials and no `material`; a nuclide perturbed
  in several materials and no `material` (use `compute_total_sensitivity`); a cell not in
  the tally;
- `compute_total_sensitivity`: no material found; perturbation grids that differ between
  materials;
- `generate_pert_cards`: boundaries not strictly ascending; an `energy_range` that leaves
  fewer than two; list arguments whose length does not match the materials, or lists of
  lists for reactions or energies; a material that is not in the input or has no density;
- `perturb_material`: a material or nuclide not found, or a perturbed-material ID already
  in use;
- `SDFData` and its profiles: a grid that is not strictly monotonic or has negative or
  non-finite values, a profile whose length is not the number of groups, negative or
  non-finite uncertainties; `create_sdf_data`: inputs with different perturbation grids
  or different $R_0$, $\varepsilon_0$ without `response_values`;
- `read_sdf`: a malformed header, count line or reaction block, counts that do not match
  the declared ones, a file of the older kika dialect read as absolute; `merge`: different
  grids, energy labels, $R_0$ or $\sigma_{R_0}$, or a (ZAID, MT) present twice;
  `group_inelastic_reactions`: an MT 4 already present without `replace=True`;
- `read_sensitivity_file`: a missing header count or list, or a block whose size does not
  match the counts (`ParseError`);
- `condense_sensitivity_profile`: a target grid that is not strictly increasing, does not
  cover the same range, has a boundary missing from the source grid, or an unknown unit
  (`CondensationError`).

Reported instead of done: profiles that are zero in every group are skipped with a
printed message; SDF scalar mismatches are logged as warnings; in
`create_sdf_from_serpent`, a material or nuclide named by a filter and absent from the file
is reported with a warning (and selects nothing), and a profile that cannot be extracted is
skipped with a printed warning.

Limits of the method: the coefficients are first-order; their statistical errors neglect
the correlations described above; `generate_pert_cards` writes cross-section cards only,
so sensitivities to angular distributions, $\bar\nu$ or the fission spectrum come from
Serpent or SCALE files and not from MCNP; when `pert_material` and `pert_density` are not
given, the cards name the original material at its original density, and the
perturbation is not the one {eq}`eq-sensitivities-perturbed-material` defines. A
compact-dialect profile has unknown uncertainties, so the bootstrap of {doc}`sandwich`
refuses it.

## Validation

The tests below passed on 2026-10-11.

- **MCNP to SDF, end to end.** Coefficients of Fe-56, Fe-54, H-1 and O-16 from four MCNP
  PERT runs (tally 4, 44 perturbation groups) are combined, MT 51-91 grouped into MT 4,
  and the SDF written is identical line by line to a stored reference file; the response
  uncertainty is $|R_0|\,\varepsilon_0$
  (`kika/sensitivities/tests/test_sdf.py::test_sdf`). The reference is a stored kika
  output, so this test guards against change and is not an independent check.
- **Uncertainty conventions.** Relative errors become $|S|\varepsilon$ and a zero
  sensitivity with a non-zero uncertainty has an infinite relative error
  (`kika/sensitivities/tests/test_sdf_absolute_uncertainties.py::test_reaction_absolute_contract_and_relative_constructor`);
  the older kika dialect is refused as absolute and converted when read as relative
  (`::test_historical_kika_requires_explicit_relative_mode`); the writer uses eV and
  absolute uncertainties and reads back what it wrote
  (`::test_standard_writer_uses_ev_and_absolute_uncertainties`); grouping MT 51 and 52
  adds the values and the standard deviations in quadrature
  (`::test_plot_band_and_inelastic_grouping_use_absolute_sigma`); Serpent relative errors
  are converted once (`::test_serpent_native_relative_errors_convert_at_sdf_boundary`).
- **SCALE files.** A trimmed 238-group SCALE file is read with its grid converted to MeV
  and recognised as SCALE 238
  (`kika/sensitivities/tests/test_sdf.py::test_read_scale_dialect_sdf`); reaction names
  with spaces (`::test_read_verbose_reaction_name_with_spaces`); unit and region indices,
  a trailing footer and a response line without an uncertainty
  (`kika/sensitivities/tests/test_sdf_scale_dialect.py`, 3 tests); a compact-dialect file
  gives profiles with unknown uncertainties
  (`::test_compact_dialect_uncertainties_are_unknown_not_zero`).
- **Serpent.** The `leg=` selector finds the Legendre moments the label parser names
  (`kika/serpent/tests/test_serpent_perturbation_label.py::test_the_leg_selector_finds_what_the_parser_labels`);
  two labels of one MT are averaged into one profile, and a filter that matches nothing
  selects nothing
  (`kika/sensitivities/tests/test_serpent_to_sdf_filters.py::test_two_labels_for_one_mt_are_averaged_into_one_profile`,
  `::test_a_filter_that_matches_nothing_selects_nothing`).
- **Condensation.** A hand case gives $S = (-1, 7)$ and $\sigma = (\sqrt{0.1^2+0.2^2},
  \sqrt{0.3^2+0.4^2})$
  (`kika/sensitivities/tests/test_condensation.py::test_exact_condensation_sums_sensitivity_and_propagates_absolute_sigma`);
  SCALE 238 to 44 and 252 to 56 groups conserve the integral to better than $10^{-12}$
  (`::test_standard_scale_nested_condensations`); 238 to 56 is refused as non-nested
  (`::test_scale_238_to_56_is_rejected_as_non_nested`); a boundary rounded to six digits
  is accepted (`::test_dice_six_digit_boundary_rounding_is_accepted`).
- **PERT cards.** Single-line and continued PERT cards are parsed into their fields
  (`kika/mcnp/tests/test_pert_parsing.py`, 5 tests). `perturb_material` is exercised
  only to check that the perturbed material exists
  (`kika/mcnp/tests/test_materials.py::test_mat_perturbation`); no test checks the
  fractions or densities of {eq}`eq-sensitivities-perturbed-material`, the cards
  `generate_pert_cards` writes, `compute_total_sensitivity` or the second-order
  coefficients.
- **Serpent labels.** Inelastic-scattering labels map to MT 4
  (`kika/serpent/tests/test_serpent_perturbation_label.py`, 5 tests). No test reads a
  Serpent `.sens` file.

## Diagnostics

- `SensitivityData.data[energy][mt]` is a
  {py:class}`~kika.sensitivities.sensitivity.Coefficients` with `values` ($S$), `errors`
  (relative), `r0` and `e0` (the unperturbed tally and its relative error);
  `values_per_lethargy` divides by $\Delta u_g$. The `energy` keys are `"lower_upper"`
  strings and `"integral"`.
- `SensitivityData.coefficients[energy][mt]` holds `c1`, `c2`, their relative errors and
  `ratio` ($c_2/c_1$); `calculate_nonlinearity_by_bin(p)` gives
  {eq}`eq-sensitivities-nonlinearity` with $p$ in percent. **A ratio comparable to 1 at
  the perturbation sizes of interest** means the first-order coefficient does not
  describe the response there.
- `SDFData.relative_response_error` is $\sigma_{R_0}/|R_0|$; each profile's
  `relative_error` is $\sigma_g/|S_g|$.
- Reading an SDF logs `SDF scalar mismatch` (logger `kika.sensitivities.sdf_parser`)
  when the stored scalars disagree with the groups: the file was edited, or written by a
  tool that rounds differently.
- `CondensationReport`: `boundary_indices` (where each target boundary sits on the source
  grid), `uncertainty_method`, `assumptions` and `max_integral_drift`, the largest change
  of $\sum_g S_g$ over the profiles, which is rounding only.

## Implementation

- `kika.mcnp.pert_generator`: the perturbed material and its density, the PERT cards.
- `kika.mcnp.parse_input`, `kika.mcnp.perturbations`: reading PERT cards and their
  `kika:` comments; grouping cards by method, reaction, material and nuclide.
- `kika.mcnp.parse_mctal`: the tally and perturbation results.
- `kika.sensitivities.sensitivity_processing`: the coefficients from MCNP, the sum over
  materials, the SDF objects from MCNP and Serpent results.
- `kika.sensitivities.sensitivity`: `SensitivityData`, `Coefficients`,
  `TaylorCoefficients`.
- `kika.serpent.parse_sens`, `kika.serpent.sens`, `kika.serpent.utils`: the `.sens`
  reader, its data model and the perturbation labels.
- `kika.sensitivities.sdf`, `kika.sensitivities.sdf_parser`: the SDF model, writer and
  reader.
- `kika.sensitivities.profile`, `kika.sensitivities.condensation`: the format-neutral
  profile and its exact condensation.

## References

- *MCNP Code Version 6.3.0 Theory & User Manual*, Los Alamos National Laboratory (2022): the PERT card and the differential-operator perturbation method.
- M. Aufiero, A. Bidaud, M. Hursin, J. Leppänen, G. Palmiotti, S. Pelloni and P. Rubiolo,
  "A collision history-based approach to sensitivity/perturbation calculations in the
  continuous energy Monte Carlo code SERPENT", *Annals of Nuclear Energy* 85, 245-258
  (2015), doi:10.1016/j.anucene.2015.05.008.
- V. Valtavirta, M. Aufiero and J. Leppänen, "Collision-history based
  sensitivity/perturbation calculation capabilities in Serpent 2.1.30", BEPU 2018, Lucca
  (2018): the sensitivity output of Serpent 2.
- W. Wieselquist and R. A. Lefebvre (eds.), *SCALE 6.3.1 User Manual*, ORNL/TM-SCALE-6.3.1,
  Oak Ridge National Laboratory (2023), doi:10.2172/1959594: TSUNAMI and the Sensitivity Data File format.
