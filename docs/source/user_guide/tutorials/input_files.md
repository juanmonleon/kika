# Working with MCNP input files

How to read the materials and PERT cards of an MCNP input file, change or create
materials, and write them back. The examples run on the input files under
`examples/data/` in the kika repository; the outputs shown are what they printed.

kika reads two parts of an MCNP input file: the material cards and the PERT cards.
Cells, surfaces and the rest of the file are not parsed, and they are left untouched
when materials are written back.

## Reading an input file

{py:func}`kika.mcnp.parse_input.read_mcnp` returns an
{py:class}`~kika.mcnp.input.Input` with two attributes, `materials` and `perturbation`.
Printing it gives a summary of both:

```python
import kika

inp = kika.read_mcnp("examples/data/inputfile_example_1.i")
print(inp)
```

```text
============================================================
                      MCNP Input Data
============================================================

------------------------------------------------------------
                         MATERIALS
------------------------------------------------------------
Number of materials:      4
Material IDs:             100000, 200100, 300000, 300001
Atomic fraction mats:     4

Use .materials to access material data.

------------------------------------------------------------
                       PERTURBATIONS
------------------------------------------------------------
Number of perturbations:  1760
Perturbation IDs:         1-1760
Reactions available:      1, 2, 4, 51, 52, 102, 103, 107
Methods available:        -3, -2, 1, 2, 3
Perturbed materials:      300001
Energy range:             1.00e-11 - 2.00e+01 MeV
Energy bins:              44
Energy structure:         SCALE44
```

## Materials

`inp.materials` is a {py:class}`~kika.mcnp.material.MaterialCollection`. Materials are
looked up by number through `by_id`:

```python
materials = inp.materials
list(materials.by_id)             # [100000, 200100, 300000, 300001]

mat = materials.by_id[300000]
mat.fraction_type, mat.libs       # ('ao', {'nlib': '06c'})
```

Each {py:class}`~kika.mcnp.material.Material` keeps its nuclides in `nuclide`, keyed by
symbol and reachable by ZAID too. Fractions are stored as positive numbers, and
`fraction_type` says whether they are atomic (`'ao'`) or weight (`'wo'`) fractions; the
sign MCNP uses on the card is applied only when the material is written.

```python
for symbol, nuclide in mat.nuclide.items():
    print(symbol, nuclide.zaid, f"{nuclide.fraction:.6e}", nuclide.libs)
```

```text
C12 6012 9.126127e-03 {}
C13 6013 1.022318e-04 {}
...
Fe56 26056 8.810613e-01 {}
...
```

```python
mat.nuclide["Fe56"] is mat.nuclide[26056]    # the same nuclide
mat.get_effective_library("Fe56")            # '06c', the material's nlib
```

A nuclide's own `libs` override the material's; `get_effective_library` gives the one
MCNP would use.

## Atomic and weight fractions

`to_weight_fraction` and `to_atomic_fraction` convert a material in place. `copy` gives
a new material with another number, so the original is kept:

```python
w = mat.copy(new_id=300100)
w.to_weight_fraction()
w.fraction_type, w.nuclide["Fe56"].fraction    # ('wo', 0.8888965...)
w.to_atomic_fraction()
w.fraction_type, w.nuclide["Fe56"].fraction    # ('ao', 0.8810613...)
```

## Creating a material

A material is built nuclide by nuclide, or element by element for natural elements.
`add_nuclide` and `add_element` take the fraction type of the value they are given:

```python
from kika.materials import Material, MaterialCollection

steel = Material(id=500, libs={"nlib": "80c"})
steel.add_element("Fe", 0.7, "wo")
steel.add_element("Cr", 0.2, "wo")
steel.add_nuclide("Ni58", 0.1, "wo")
print(steel.to_mcnp())
```

```text
m500 nlib=80c
     24000         -2.000000e-01   $ Cr
     26000         -7.000000e-01   $ Fe
     28058         -1.000000e-01   $ Ni58
```

`expand_natural_elements` replaces natural elements with their isotopes, by natural
abundance. Given element symbols it expands only those:

```python
steel.expand_natural_elements("Cr")
list(steel.nuclide)    # ['Fe', 'Ni58', 'Cr50', 'Cr52', 'Cr53', 'Cr54']
steel.expand_natural_elements()
list(steel.nuclide)    # ['Ni58', 'Cr50', 'Cr52', 'Cr53', 'Cr54', 'Fe54', 'Fe56', 'Fe57', 'Fe58']
```

A density is set with its unit, and is written as a comment on the material card, with
the atom density kika derives from it:

```python
steel.set_density(7.9, "g/cc")
print(steel.to_mcnp())
```

```text
m500 nlib=80c  $ KIKA_DENSITY: 7.900000e+00 g/cc, 8.614467e-02 atoms/b-cm
     24050         -8.347373e-03   $ Cr50
     ...
     28058         -1.000000e-01   $ Ni58
```

`to_serpent` writes the same material as a Serpent `mat` card.

## Writing materials back to an input file

`MaterialCollection.write_to_mcnp` updates the material cards of an input file. It
rewrites only the materials that changed or are new, so the others keep their formatting
and comments, and returns the numbers it wrote. Without `output_filepath`, the input file
is updated in place.

```python
inp = kika.read_mcnp("model.i")
inp.materials.add_material(steel)
inp.materials.write_to_mcnp("model.i", output_filepath="model_new.i")    # [500]
sorted(kika.read_mcnp("model_new.i").materials.by_id)    # [500, 100000, 200100, 300000]
```

`MaterialCollection.from_mcnp` and `from_serpent` read only the materials of a file;
`to_file` and `write_to_serpent` write them.

## PERT cards

`inp.perturbation` holds the PERT cards of the file. Each card is in `pert`, keyed by its
number:

```python
pert = inp.perturbation
len(pert.pert)       # 1760
pert.reactions       # [1, 2, 4, 51, 52, 102, 103, 107]
pert.materials       # [300001]
print(pert.pert[1])
```

```text
Perturbation ID: 1
Particle type:   n
Cells:           3, 5, 7, 9
Material:        300001
Density:         1.603930e-01
Method:          2
Reaction:        1
Energy range:    1.000000e-11 - 3.000000e-09 MeV
```

`pert_energies` gives the energy boundaries of the cards (45 here, the SCALE 44-group
structure), and `to_dataframe` one row per card:

```python
pert.to_dataframe().head(3)
```

```text
   particle        cell  material       rho  method  reaction         e_min         e_max  zaid original_material
id
1         n  3, 5, 7, 9    300001  0.160393       2         1  1.000000e-11  3.000000e-09  None              None
2         n  3, 5, 7, 9    300001  0.160393       3         1  1.000000e-11  3.000000e-09  None              None
3         n  3, 5, 7, 9    300001  0.160393      -2         1  1.000000e-11  3.000000e-09  None              None
```

The `zaid` and `original_material` columns are filled when the cards are read with
`pert_metadata`, which tells `read_mcnp` which nuclide and material each range of cards
belongs to. With it, `pert.zaids`, `reactions_for_zaid` and `materials_for_zaid` answer
per nuclide.

Writing PERT cards for a sensitivity study is the subject of {doc}`pert_cards`.
