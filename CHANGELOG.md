# Changelog

## Unreleased

### Added

- `kika.endf.writers.assemble.energyCovarianceTape(suite, covarianceSuite)`: the MF35 sections
  and the MF5 sections they are relative to, built in memory as `writeEndfTape` writes them, so
  `MF35MT` (bands, `relative_uncertainty`) works on a suite read from GNDS without a tape.
  Bit-identical to the written and re-read tape; ENDF/B-VIII.1 U-235 takes 0.2 s instead of 8 s.
- `kika check --format html json md`: the check report also as JSON (`to_dict`) and Markdown,
  beside or instead of the HTML page, for a pipeline that reads the findings.
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

- MF32 LCOMP=2 (LRF=1-3): the matrix covers ER, GN, GG and, on a fissile nuclide, the fission
  widths, as ENDF-6 §32.2.3.1-2 states, zero uncertainties included. kika counted only the widths
  with a non-zero uncertainty, so sections that follow the format (JEFF-4.0 Tc-98, Re-186m, Pt-191,
  Ac-227; JENDL-5 Pu-240) were refused with a `decode_error` defect. They now decode and pass.
- An MF33 block that sums absolute (LB=0/8/9) and relative components no longer needs NJOY.
  `decodeCovarianceSuite` (and `kika.read`) takes σ(E) from the tape's own MF3 when it has no
  resonance ranges, and from kika's reconstruction at 1e-3 otherwise; a PENDF set on
  `endf.pendf` first (NJOY's, through `attach_pendf`) is still used as it is. New:
  `kika.processing.pointwise_cross_sections`. The matrices move by at most 2e-7 (relative) from
  the NJOY RECONR ones on the IRDFF-II and ENDF/B-VIII.1 tapes compared.
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
