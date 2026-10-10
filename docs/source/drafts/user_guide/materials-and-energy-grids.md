# Materials and energy-group structures

## Materials

A `kika.materials.Material` is a composition: nuclides or natural elements with atomic
(`ao`) or weight (`wo`) fractions, a density and a temperature. It converts between the
two kinds of fraction, expands natural elements into their isotopes with natural
abundances, and writes MCNP and Serpent material cards.

```python
from kika.materials import Material

steel = Material(id=10, name="steel", fraction_type="wo", density=7.9, density_unit="g/cc")
steel.add_element("Fe", 0.70, fraction_type="wo")
steel.add_element("Cr", 0.18, fraction_type="wo")
steel.add_element("Ni", 0.12, fraction_type="wo")
steel.expand_natural_elements()     # Fe -> Fe54, Fe56, Fe57, Fe58, ...
steel.to_atomic_fraction()          # or to_weight_fraction()
print(steel.to_mcnp())              # an MCNP m card, with the density in a comment
print(steel.to_serpent())           # a Serpent mat card
```

Other operations: `add_nuclide` (by symbol `"Fe56"` or ZAID `26056`),
`update_nuclide_fraction`, `remove_nuclide`, `normalize`, `to_integer_fractions`,
`set_density`/`convert_density` (g/cc and atoms/b-cm), `set_temperature`, and
`set_library` for the ACE library suffix of each nuclide. A nuclide is reached as
`steel.nuclide["Fe56"]` or `steel.nuclide[26056]`.

A `MaterialCollection` holds several materials. Materials are read from MCNP input
(`kika.read_mcnp(...)`, see {doc}`/user_guide/tutorials/input_files`) and from Serpent
input (`kika.materials.parse_serpent.read_serpent_materials(path)`).

## Energy-group structures

`kika.energy_grids` holds the standard group structures as arrays of group boundaries,
**in MeV**, ascending: `VITAMINJ174`, `VITAMINJ175`, `SCALE44`, `SCALE56`, `SCALE238`,
`SCALE252`, `WIMS69`, `ECCO33`, `XMAS172`, `LANL30`, `CASMO12`, `CASMO40`, `CASMO70`,
`ONEGROUP20`.

```python
from kika import energy_grids

energy_grids.SCALE44            # 45 boundaries, 1e-11 to 20 MeV
energy_grids.equal_lethargy_grid(1e-5, 2e7, 100)   # 101 boundaries, equal in ln E
```

`equal_lethargy_grid` takes its bounds in the unit you give it. The structures are used by
the multigroup collapse ({doc}`../methods/covariance/multigroup-collapse`), the group
averages ({doc}`../methods/processing/group-averaging`) and the sensitivity condensation.
