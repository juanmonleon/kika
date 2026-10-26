# Sensitivity coefficients from MCNP perturbations

How to turn the PERT-card results of an MCNP run into sensitivity coefficients, look at
them, check how linear the response is, and write them as an SDF (Sensitivity Data
File) for SCALE or for an uncertainty calculation. The examples run on
`examples/data/inputfile_example_1.i` and its MCTAL file `mctalfile_example_1.m`, a
Fe-56 shell with 1760 PERT cards (8 reactions, 44 groups, all methods); the outputs shown
are what they printed. Setting up the PERT cards is {doc}`pert_cards`.

## Computing the coefficients

{py:func}`kika.sensitivities.sensitivity_processing.compute_sensitivity` reads the PERT
cards from the input file and their results from the MCTAL file, for one tally and one
nuclide:

```python
import kika

sens = kika.compute_sensitivity(
    inputfile="examples/data/inputfile_example_1.i",
    mctalfile="examples/data/mctalfile_example_1.m",
    tally=4,
    zaid=26056,
    label="Fe-56",
)
print(sens)
```

```text
Label:                           Fe-56
Tally ID:                        4
Tally Name:                      shell Detector
Nuclide (ZAID):                  Fe-56 (26056)
Number of detector energy bins:  3
Number of perturbation bins:     44
Energy structure:                SCALE44
Reactions available:             1, 2, 4, 51, 52, 102, 103, 107
Second Order data available:     True

Energy group ranges:
  - 0.00e+00_1.00e-01
  - 1.00e-01_1.00e+00
  - 1.00e+00_3.00e+00
  - integral
```

The result is a {py:class}`~kika.sensitivities.sensitivity.SensitivityData`. Its
coefficients are the relative change of the tally per relative change of the nuclide's
cross-section, per perturbation group: MCNP's first-order estimate of the tally change
(`METHOD=2`) divided by the unperturbed tally. The perturbed material doubles the
nuclide, a relative change of exactly 1, so no further division is needed. They are given
for each energy bin of the tally (the *detector* energies, named `"lower_upper"` in MeV)
and for the tally summed over them (`"integral"`).

When the nuclide sits in several perturbed materials,
{py:func}`~kika.sensitivities.sensitivity_processing.compute_total_sensitivity` adds the
contributions of each. When the input file holds the cards of several nuclides,
`pert_metadata` says which cards belong to which (see
{py:func}`~kika.mcnp.parse_input.read_mcnp`).

## Reading the coefficients

`sens.data[detector_energy][mt]` is a {py:class}`~kika.sensitivities.sensitivity.Coefficients`:

```python
sens.energies     # ['0.00e+00_1.00e-01', '1.00e-01_1.00e+00', '1.00e+00_3.00e+00', 'integral']
sens.reactions    # [1, 2, 4, 51, 52, 102, 103, 107]

c = sens.data["integral"][2]
c.r0, c.e0        # (1.13584e-08, 0.0057): the unperturbed tally and its relative error
sum(c.values)     # -1.3414...: the elastic sensitivity summed over the 44 groups
c.to_dataframe().tail(3)
```

```text
      energy  reaction  e_lower  e_upper  sensitivity   error
41  integral         2   4.8000   6.4340          0.0  0.0057
42  integral         2   6.4340   8.1873          0.0  0.0057
43  integral         2   8.1873  20.0000          0.0  0.0057
```

`values` are the coefficients per group, `pert_energies` the group boundaries in MeV,
and `errors` the **relative** errors of the coefficients. They combine the error of the
tally change with that of the unperturbed tally, so a group the tally does not respond
to has a zero coefficient with the tally's own relative error (`e0`), as in the groups
above 4.8 MeV here. `values_per_lethargy` divides by the lethargy width of each group.
`sens.to_dataframe()` gives every detector energy and reaction
in one table.

## Plotting

```python
sens.plot_sensitivity(energy="integral", reaction=[1, 2, 102])
sens.plot_sensitivity(energy="integral", reaction=2, energy_range=(1e-3, 20), xlog=True)
```

{py:func}`~kika.sensitivities.sensitivity_processing.plot_sens_comparison` draws several
`SensitivityData` together (several nuclides, or the same one from two runs):

```python
kika.plot_sens_comparison([sens_fe56, sens_cr52], energy="integral", reactions=[1, 2],
                          energy_range=(0, 3))
```

## Checking linearity with the second-order terms

With second-order cards (`METHOD=3`) in the run, `sens.coefficients` holds the Taylor
coefficients of the response, {py:class}`~kika.sensitivities.sensitivity.TaylorCoefficients`,
per detector energy and reaction. The ratio of the second-order to the first-order
term says how far the linear estimate holds:

```python
sens.plot_ratio(energy="integral", reaction=[2], top_n=5)
sens.plot_perturbed_response(energy="integral", reaction=[2], p_range=(-20, 20), top_n=3)
sens.plot_second_order_contribution(energy="integral", reaction=[2], p_range=(-10, 10), top_n=3)
```

`plot_ratio` shows the groups where the ratio is largest; `plot_perturbed_response`
compares the first-order and the second-order response over a range of perturbations in
percent; `plot_second_order_contribution` shows what the second-order term adds. A
sensitivity used for a small perturbation is safe where the second-order term is small
over that range.

## Writing an SDF file

{py:func}`~kika.sensitivities.sensitivity_processing.create_sdf_data` collects the
coefficients of one detector energy into an {py:class}`~kika.sensitivities.sdf.SDFData`,
for one or several nuclides. A tuple `(sens, [mts])` keeps only some reactions:

```python
sdf = kika.create_sdf_data([sens], energy="integral", title="Fe-56 example")
len(sdf.data)                        # 8: one SDFReactionData per reaction
sdf.data[0].nuclide, sdf.data[0].reaction_name, sdf.data[0].mt    # ('Fe-56', '(n,total)', 1)

only = kika.create_sdf_data([(sens, [1, 2])], energy="integral", title="total and elastic")
```

`group_inelastic_reactions` sums the discrete-level reactions (MT 51-91) of each nuclide
into MT 4 and removes them; `replace=True` overwrites an MT 4 that is already there,
which otherwise raises `ValueError`:

```python
sdf.group_inelastic_reactions(replace=True)
[d.mt for d in sdf.data]             # [1, 2, 4, 102, 103, 107]
sdf.write_file(output_dir="sdf")     # writes sdf/Fe-56_example_integral.sdf
```

{py:func}`~kika.sensitivities.sdf_parser.read_sdf` reads an SDF file back, whether kika
or SCALE wrote it, and `SDFData.merge` joins several into one. `to_sensitivity_profile`
turns an `SDFData` into a {py:class}`~kika.sensitivities.profile.SensitivityProfile`,
the format-neutral input of the uncertainty calculations (the sandwich rule and the
similarity index $c_k$, see {doc}`../../api_reference/uq`), which also accept an
`SDFData` directly:

```python
sdf = kika.read_sdf("sdf/Fe-56_example_integral.sdf")
profile = sdf.to_sensitivity_profile()
```
