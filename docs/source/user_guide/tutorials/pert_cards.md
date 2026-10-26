# Setting up a sensitivity calculation with PERT cards

How to prepare an MCNP input file for a sensitivity calculation: make a perturbed copy of
a material, write it into the input file, and generate the PERT cards that perturb it
reaction by reaction and energy group by energy group. The examples run on
`examples/data/inputfile_example_0.i`, copied to the working directory as `model.i`; the
outputs shown are what they printed. Reading the results back is
{doc}`sensitivity_analysis`.

## The approach

A PERT card tells MCNP to estimate how a tally would change if, in the given cells, the
material were replaced by another material at another density, for one reaction and one
energy range. To get the sensitivity to one nuclide, the replacement material is the
original with that nuclide's amount doubled (a +100 % change of its atom density), and
the cards are written once per reaction and per energy group. MCNP's first-order
(`METHOD=2`) and second-order (`METHOD=3`) estimates of the change then give the
sensitivity coefficients and their Taylor terms.

## 1. Perturb the material

{py:func}`kika.mcnp.pert_generator.perturb_material` works on the material collection
read from the file. It adds a material with the nuclide's fraction raised by 100 %,
numbered `material_id * 100 + 1` unless `pert_mat_id` says otherwise, and returns it:

```python
import kika

inp = kika.read_mcnp("model.i")
pert_mat = kika.perturb_material(inp.materials, 300000, "Fe56", density=-7.85)
```

```text
Density change: 7.8500e+00 -> 1.4828e+01 g/cm3
Perturbed material 30000001 added to collection (100% increase in Fe56)
```

`density` is the density of the original material, with MCNP's sign convention
(negative in g/cm³, positive in atoms/b-cm). It is needed when the material card does
not carry one, as here, where the density is on the cell cards. From it kika computes
the density of the perturbed material, which is what the PERT cards need. The nuclide
can be given by symbol (`"Fe56"`) or ZAID (`26056`). `fraction_type` chooses atomic or
weight fractions for the new material (by default those of the original), and
`in_place=False` returns a new collection and leaves `inp.materials` as it was.
{py:func}`~kika.mcnp.pert_generator.perturb_materials` does the same for several
materials holding the same nuclide.

## 2. Write it into the input file

`perturb_material` changes only the collection in memory. Write the new material into
the file with `write_to_mcnp`, which adds the materials that are new and leaves the
others as they were:

```python
inp.materials.write_to_mcnp("model.i")    # [30000001]
```

## 3. Generate the PERT cards

{py:func}`kika.mcnp.pert_generator.generate_pert_cards` writes the cards into the input
file. It reads the density for `RHO=` from the material card it is given, so the
perturbed material has to be in the file first (step 2):

```python
kika.generate_pert_cards(
    "model.i",
    cell=[3, 5, 7, 9],                       # the cells holding the material
    reactions=[1, 2, 4, 51, 102],            # MT numbers
    material=30000001,                       # the perturbed material
    energies=kika.energy_grids.SCALE44,      # group boundaries, in MeV
    order=2,                                 # METHOD=2 and METHOD=3
)
```

```text
PERT1:n CELL=3 5 7 9 MAT=30000001 RHO=-1.482784e+01 METHOD=2 RXN=1 &
     ERG=1.000000e-11 3.000000e-09
PERT2:n CELL=3 5 7 9 MAT=30000001 RHO=-1.482784e+01 METHOD=3 RXN=1 &
     ERG=1.000000e-11 3.000000e-09
...
```

That is 5 reactions × 44 groups × 2 methods = 440 cards. The arguments:

- `energies`: the group boundaries in MeV, MCNP's unit, written as given.
  `kika.energy_grids` holds the usual structures (`SCALE44`, `SCALE56`, `SCALE238`,
  `SCALE252`, `VITAMINJ175`, ...). `energy_range=(low, high)` keeps only the boundaries
  inside a range, and `energies=None` writes energy-integrated cards.
- `order`: `1` writes only first-order cards (`METHOD=2`); `2`, the default, also the
  second-order ones (`METHOD=3`) that {doc}`sensitivity_analysis` uses to check how
  linear the response is.
- `errors=True` adds the exact-error cards (`METHOD=-2`, `-3` and `1`). They are rarely
  needed and cost run time.
- `density` overrides the density read from the file.
- `in_place=False` writes to a new file with the suffix `_pert_cards` instead of the
  input file.
- With lists for `cell`, `material`, `pert_material` and `pert_density`, one call writes
  the cards for several materials; see the function's documentation.

`generate_PERTcards` is the same function under its older name.

## 4. Check what was written

Reading the file back shows the cards:

```python
p = kika.read_mcnp("model.i").perturbation
len(p.pert), p.reactions, p.materials    # (440, [1, 2, 4, 51, 102], [30000001])
```

The input file is then ready for MCNP. The MCTAL file of the run, with the input file,
is what {doc}`sensitivity_analysis` starts from.
