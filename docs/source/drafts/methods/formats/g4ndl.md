# Reading and writing G4NDL

```{admonition} At a glance
:class: note
**Reads:** G4NDL libraries, the neutron data of Geant4's high-precision (ParticleHP)
models: elastic, inelastic, capture and fission, per isotope.

**Writes:** the same files from the model, and a copy of a whole library with one isotope
replaced.

**Into:** a {py:class}`~kika.nuclear_data.model.suite.ReactionSuite` per isotope (see
{doc}`../../architecture/canonical-model`).

**Entry points:** {py:func}`kika.read` (`format="g4ndl"`, `target=`), {py:func}`kika.write`,
`kika.g4ndl.open`, `kika.g4ndl.patch_isotope`, `kika.g4ndl.patch_elastic`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

A G4NDL library is a directory tree, one directory per process (`Elastic`, `Inelastic`,
`Capture`, `Fission`), each with a `CrossSection` subdirectory and the final-state data
(`FS`, and for inelastic the channel directories `F01` to `F36` and the residual level
schemes `Gammas`). Each file holds one isotope as a stream of whitespace-separated tokens,
plain or compressed as a raw zlib stream with the suffix `.z`. The format has no
specification document of its own: what a file means is what Geant4's reader does with
it. kika follows the reader of Geant4 11.4.3 (`source/processes/hadronic/models/particle_hp`);
Geant4 is not a dependency of kika.

## Coverage

| Part | What kika does |
|---|---|
| `Elastic/CrossSection`, `Elastic/FS` (MT 2) | read and written |
| `Inelastic/CrossSection`, `F01`-`F36`, `Gammas` | read and written; `kika.g4ndl.inelastic_decode` states what reaches the model |
| `Capture/CrossSection`, `Capture/FS` or `FSMF6` (MT 102) | read and written |
| `Fission/CrossSection`, `FS`, chances `FC` to `LC`, fragment yields `FF` (MT 18, 19-21, 38) | read and written |
| `ThermalScattering` (`Coherent`, `Incoherent`, `Inelastic`) | read one material at a time, as a processed thermal-scattering suite (σ(E) and the Bragg table enter the model; the outgoing-neutron sampling tables stay beside it); not written. A neutron read of the same root leaves the folder out, and its report says so |

## Reading

**1. The stream.** A file is opened as text, or decompressed as zlib when it ends in `.z`
(never gzip or zip); a decompressed size above 512 MB is refused.

**2. Records.** The tokens are parsed into records with the same grammar as Geant4's
reader: integers where Geant4 reads integers, floats where it reads floats.

**3. Into the model.** Without an intermediate format-shaped object:

- **The cross-section** becomes a lin-lin `XYs1d` under the `recon` style, a reconstructed
  cross-section derived from `eval`: Geant4 reads it as pointwise data at 0 K, which it
  broadens itself, and the file states no interpolation, so lin-lin is what it means.
- **The angular distribution** goes under `eval`. The `repFlag` values 1, 2 and 3 are
  ENDF's LTT 1, 2 and 3 with the same records, so the mapping is the one of ENDF MF 4:
  $a_0 = 1$ is prepended to the stored $a_1 \dots a_{NL}$, a tabulated record with one
  region is an `XYs1d` and with several a `Regions1d`, and `repFlag=3` is a `Regions2d` of
  the Legendre block and the table. `frameFlag` is the product frame.
- **Tokens with no model slot** (the bookkeeping integers, `T` and `tempdep` per energy,
  the optional source header, a declared histogram code) are kept in a G4NDL provenance
  record, so that the file can be written back.

## Assumptions and defaults

- **Interpolation code 1 in $\mu$ is read as lin-lin**, because Geant4 evaluates it so (its
  histogram branch is disabled); the report calls it an approximation, and the declared
  code is kept and written back.
- **`tempdep` other than 0 is accepted** (Geant4 reads and discards it) and reported as a
  warning.
- The evaluation a library came from is not stated by its files; the report says so.

## Ambiguous and inconsistent input

- **A `repFlag=0` record carries the frame twice**; Geant4 keeps the second, and so does
  kika.
- **Legendre records of different order** at neighbouring energies are interpolated as if
  the shorter had zero coefficients above its order, which is what Geant4 does.

## Refused

On reading (`G4NDLFormatError`, `G4NDLUnsupportedError`): a stream that does not parse
with Geant4's grammar; interpolation code 1 in incident energy; a decompressed file above
the size limit. A missing isotope raises `IsotopeNotFoundError`.

On writing (`G4NDLUnsupportedError`), each because Geant4 would read something other than
the model says:

- a histogram (code 1) in the model, in $\mu$ or in incident energy, and any code outside
  2-5, including unit-base and corresponding-point qualifiers;
- a cross-section that is not one pointwise lin-lin table. By default the `recon` form is
  written; `crossSectionLabel="eval"` says the evaluated MF 3 is pointwise, and is refused
  even then for a suite with a resonance region (an ENDF tape gets its `recon` form from
  NJOY RECONR, {doc}`../processing/njoy`);
- a Legendre expansion whose $a_0$ is not 1, and units other than eV and b;
- a suite with no target mass: the AWR sets the centre-of-mass kinematics, so it is never
  defaulted.

## Writing

The model is converted to the same records the parser produces, and the records to text.
Every file is **parsed back with the strict reader before it is handed out**, and the
records must equal those that went in, array for array. `repFlag` is set from the shape
of the angular form, never copied from the provenance, so a replaced distribution is
written with the right flag. Floats are written as the shortest string that reads back
as the same double; integers as integers (a `2.0` where Geant4 reads an integer would
misalign the rest of the stream without an error).

`patch_isotope` (or `patch_elastic`) copies a whole library and replaces one isotope's
files with those written from a suite, leaving every other file as it was, and returns a
`PatchResult` naming what was written.

## What the round trip keeps and loses

Read and write gives the same records, array for array; tokens with no model slot come
back from the provenance. Thermal scattering data is read only as a suite of its own,
and is not written.
The distributed libraries print 7 significant figures, kika writes up to 17, so a file
written by kika is not byte-identical to the one it was read from.

## Report

Entries a user meets: an evaluation the files do not name (a loss); a code 1 in $\mu$ read
as lin-lin (an approximation); `tempdep` not zero (a warning); a `ThermalScattering`
directory left out of a neutron read (unsupported; the thermal material is read with its
name as `target`).

## Validation

- `kika/g4ndl/tests/`: parsing, decoding, encoding with read-back, capture, fission,
  inelastic, the library and patching, the physics checks, the tables and the correction
  loop. 429 passed on 2026-10-11.
- Whole distributed libraries, every isotope read and written back
  (`kika/g4ndl/tests/test_full_libraries.py`, `test_capture_full_libraries.py`,
  `test_fission_full_libraries.py`, `test_inelastic_full_libraries.py`). `TODO(check)`:
  they need the libraries on disk; not re-run for this page.
- The `ThermalScattering` folder: a folder of materials is indexed, its Bragg table and
  outgoing tables are read per temperature, and `kika.read` returns its suite
  (`kika/tsl/tests/test_tsl_formats.py::test_a_g4ndl_thermal_folder_is_a_library_of_materials`).
- Agreement with Geant4 itself is checked outside this repository, against a compiled
  Geant4.

## Implementation

- `kika/g4ndl/stream.py`: opening a file; `parse.py`, `records.py`, `tokens.py`: the
  grammar; `inelastic_*.py`, `capture.py`, `fission.py`, `fission_model.py`: the other
  processes.
- `kika/g4ndl/decode.py`, `encode.py`: model in and out; `library.py`: `open` and
  `G4NDLLibrary`; `patch.py`: `patch_isotope`, `patch_elastic`.
- `kika/g4ndl/physics.py`: `checkElastic` (normalisation, positivity found from the roots
  of a Legendre sum's derivative, moments) and the evaluation of $p(\mu \mid E)$ and
  $d\sigma/d\Omega$; `tables.py`: summary tables.
- `kika/tsl/g4ndl.py`: the `ThermalScattering` folder; `kika/tsl/processed.py`: the
  processed thermal suite it is read into.
- `kika/g4ndl/iterate.py`: the correction loop that alternates a measurement corrected
  with Geant4 and a library rewritten from the corrected evaluation; Geant4 runs outside
  kika.

## References

- `TODO(ref)`: Geant4, and its high-precision neutron (ParticleHP) models.
