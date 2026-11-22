# Changelog

## Unreleased

### Added

- `kika.processing.heating.dpa`: displacements per atom from a displacement cross section
  (`EndfProcessing.displacements`, NRT or arc-dpa) folded with a neutron flux, pointwise or
  in groups. A flux gives dpa/s; a fluence gives dpa.
- `kika.materials`: OpenMC `materials.xml` written (`Material.to_openmc`,
  `MaterialCollection.to_openmc`) and read (`read_openmc_materials`), without OpenMC installed.
- `MaterialCollection.emission_report(code, library=...)`: per material, what the MCNP, Serpent
  or OpenMC cards leave out and which tables an xsdir, xsdata or `cross_sections.xml` lacks, as
  an HTML page. `Material.emission_losses` and `Material.library_tables` give the parts.
- `Material.to_dict` / `from_dict`, `Material.keywords` (MCNP m-card keywords) and
  `Material.thermal_scattering` (the `mt` card, read by `read_mcnp`); `Nuclide.z`, `.a`,
  `.isomer`; `MaterialCollection[id or name]`.

### Changed results

- Atomic masses come from AME2020. U-234, U-235 and U-238 had no mass (it fell back to the mass
  number: 235.0 for 235.0439), so weight/atom conversions with uranium, the EXFOR target mass and
  the mass fallbacks of MF1/451 and MF6 move by up to about 2e-4.
- Natural Rh, Sn and La expand into Rh-103, Sn-112.. and La-138/139 (they gave Rh-99, Sn-102 and
  La-140).
- `perturb_material` took a natural element's mass as 1.0; densities of materials with natural
  elements change.
- `N_AVOGADRO` is the SI value 6.02214076e23 (it was 6.02214e23).
- `expand_natural_elements` on a weight-fraction material splits each element by its isotopes'
  mass shares and leaves the other nuclides as they were (it renormalised the material), and adds
  to an isotope already present (it overwrote it).

### Changed

- An MF33 block that sums absolute (LB=0/8/9) and relative components no longer needs NJOY.
  `decodeCovarianceSuite` (and `kika.read`) takes σ(E) from the tape's own MF3 when it has no
  resonance ranges, and from kika's reconstruction otherwise; a PENDF set on `endf.pendf` first
  (NJOY's, through `attach_pendf`) is still used as it is. New:
  `kika.processing.pointwise_cross_sections`. Against NJOY RECONR the matrices agree to 2e-7.
- MCNP and Serpent cards are written with every number in its shortest exact form and read back
  before they are returned; a card that does not read back raises.
- The material readers refuse what they used to skip: a nuclide listed twice for the same data
  class, a token that is neither a keyword nor a ZAID and its fraction, a Serpent material with no
  nuclides, an `mt` card for a material the deck lacks. `add_nuclide` refuses a fraction type
  other than the material's, and a library suffix of an unknown data class.
- `to_serpent` refuses a material with no density (it wrote -1.0 g/cc) and writes only the neutron
  library of each nuclide.
- `write_to_mcnp` replaces each card where it stands (`force_rewrite` deleted the cards between
  the first and last material), sees a change of material library, keyword, name or `mt` card,
  and keeps the file's line endings and bytes. Serpent reads and writes keep them too.
- Isomer ZAIDs in the LANL (95642) and NEA (95342) forms are read as the isomer they name.
- Materials, nuclides and collections print as one line.

### Removed

- `kika.mcnp.material` and `kika.mcnp.parse_materials`: import from `kika.materials`.
- `Nuclide.convert_natural_element`, which printed: use `Material.expand_natural_elements`.
