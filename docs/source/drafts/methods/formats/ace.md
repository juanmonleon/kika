# Reading and writing ACE

```{admonition} At a glance
:class: note
**Reads:** continuous-energy neutron ACE tables (type 1, legacy and 2.0.1 headers) block
by block, and photo-atomic tables.

**Writes:** an ACE table from its parsed object, and its `xsdir` line; no ACE is written
from the model.

**Into:** the parse layer's `Ace` object; the model receives the cross-sections, the
Q-values, the photon production and a reference to the unresolved-range probability
tables (see {doc}`../../architecture/format-adapters`).

**Entry points:** `kika.read_ace`, {py:func}`kika.read`,
`kika.ace.model_adapter.decodeAce`, `kika.ace.writers.write_ace.write_ace`,
`kika.ace.xsdir.create_xsdir_files_for_ace`.

**Status:** stable (parse layer); experimental (model path).
```

## Background

ACE (A Compact ENDF) is the format of continuous-energy Monte Carlo codes such as MCNP and
Serpent. A table is processed data, written by NJOY's ACER: the resonance region is
already reconstructed, every cross-section is on one unionised energy grid, linearly
interpolated, at one temperature (`TODO(ref)`: the ACE format specification). After a header, a table is
three arrays: NXS (sizes), JXS (pointers) and XSS, one long array of reals holding every
block. Energies are in MeV, cross-sections in barns, and the header temperature is $kT$
in MeV.

## Coverage

What kika does with each block of a neutron table:

| Block | Parse layer (`read_ace`) | Model (`decodeAce`) |
|---|---|---|
| Header (ZAID, AWR, temperature, comment, NXS, JXS) | read; legacy and 2.0.1 headers | provenance; temperature as a `heated` style |
| ESZ (grid, total, absorption, elastic, heating) | read | elastic and the other cross-sections, as `gridded` |
| MTR, LSIG, SIG (reaction cross-sections) | read | one reaction per MT, `gridded`; sums (MT 4, 18, 101, ...) summed on demand |
| LQR (Q-values) | read | the reaction's Q, in eV |
| TYR (multiplicity and frame) | read | not decoded |
| LAND, AND (angular distributions) | read: isotropic, equiprobable bins, tabulated, and Kalbach-Mann (the angular part of law 44) | not decoded (reported) |
| LDLW, DLW, DNED (energy distributions, delayed-neutron spectra) | read: laws 1, 2, 3, 4, 5, 7, 9, 11, 22, 24, 44 (Kalbach-Mann), 61, 66, 67 | not decoded (reported) |
| NU, DNU, BDD (multiplicities, delayed-neutron precursors) | read | not decoded (reported) |
| GPD, MTRP, SIGP, LANDP, ANDP, DLWP, YP (photon production) | read | photon products, as the ENDF decoder makes them from MF 12-15 |
| UNR (probability tables) | read | declared by reference; contents not converted (reported) |
| FIS, secondary-particle blocks (PTYPE, NTRO, IXS, HPD) | read | not decoded |

A photo-atomic table (ZAID suffix ending in `p`) is parsed on its own, since its NXS and
JXS mean other things, and returned as a `PhotoatomicAce`.

## Reading

**1. Header.** The first line tells a 2.0.1 header from a legacy one; both are read into
`Header`, with NXS and JXS.

**2. XSS.** The array is read straight from the file in chunks, as reals, with a
placeholder at index 0 so that JXS pointers (1-based, as in the format) index it directly.

**3. Blocks.** Every block is parsed when the file is read, from its JXS pointer: none is
loaded on demand. Each block object is a view of the XSS array, not a copy, so editing a
value through a block edits the table.

**4. Into the model** (`kika.read(path)` or `decodeAce`): energies are converted from MeV
to eV; each reaction's cross-section becomes one lin-lin `XYs1d` (ACE has no
interpolation regions) labelled `gridded`; the Q-value of each reaction in MTR comes from
LQR, in eV, and elastic gets Q = 0. The style chain is `evaluated` → `heated` (at the
header temperature, converted from $kT$ to K) → `griddedCrossSection`, with
`URR_probabilityTables` when the table has them.

## Assumptions and defaults

- **The header temperature is $kT$ in MeV** and is converted to kelvin with Boltzmann's
  constant.
- **LQR is aligned with MTR by position**; when their lengths differ, the shorter one is
  used and a warning says so.
- **MT 2 has Q = 0** by definition.
- **No reconstruction step is claimed.** A continuous-energy ACE has almost certainly been
  through RECONR, but the file does not say so, so no `crossSectionReconstructed` style is
  added.

## Ambiguous and inconsistent input

- A table with no LQR block: no reaction gets a Q-value, reported as a loss.
- A summed reaction (MT 4, 18, 101, 27, 3) has no Q of its own; its missing Q is reported.
- A table with no header temperature: the `heated` style has none, with a warning.

## Refused

A table whose XSS cannot be read raises from the parser. Writing refuses an `Ace` object
with no header or no XSS (`ValueError`), and an existing output file unless
`overwrite=True` (`FileExistsError`).

## Writing

`write_ace(ace, filepath, overwrite=False)` writes the header and the XSS array from the
parsed object, without modifying it. When every XSS value is an integer the array is
written in integer format; otherwise in floating-point notation. `update_cross_sections`
changes cross-sections in place in the XSS (the operation the ACE-level perturbation
uses), and `create_xsdir_files_for_ace` writes the `xsdir` lines for a set of tables.
`kika.write` does not write ACE: an ACE file is produced from an ENDF tape by NJOY (see
{doc}`../processing/njoy`).

## What the round trip keeps and loses

- **Parse layer, read and write:** every XSS value comes back bitwise identical; the text
  does not, because a whole-number value is written in floating-point notation.
- **Model:** the evaluation behind the table is not in the file and is reported as lost;
  QM and LR have no ACE counterpart (reported); ACE has no covariances (reported). The
  angular and energy distributions, multiplicities and delayed-neutron data are not
  decoded (reported, as not affecting cross-sections).

## Report

Entries a user meets after `kika.read` of an ACE file: the absent evaluation (a loss, on
every table); no LQR block (a loss); a reaction with no Q (a loss, normal for sums); QM,
LR and covariances (unsupported, on every table); the probability tables declared by
reference (unsupported); photon production entries for laws without a model node; one
unsupported entry per block of secondary-particle data the table carries and the model
does not receive: the neutron angular distributions (AND, with their MTs), the neutron
energy distributions (DLW, with their MTs), the delayed-neutron spectra (DNED), the
secondary-particle angular and energy distributions, $\bar\nu$ (NU/DNU, naming prompt,
total, delayed) and the delayed-neutron precursors (BDD, with the number of groups). These
entries state that cross-sections are unaffected, so they do not make the conversion
unclean for a cross-section use; a block absent from the table produces no entry.

## Validation

- Parsers follow the format manual block by block
  (`kika/ace/tests/test_parsers_follow_the_manual.py`); blocks are views of the XSS
  (`kika/ace/tests/test_blocks_are_views_of_the_xss.py`); writing does not modify the
  object (`kika/ace/tests/test_write_ace_does_not_mutate.py`); a direct MT 18 update
  (`kika/ace/tests/test_update_cross_sections_direct_mt18.py`); photo-atomic tables
  (`kika/ace/tests/test_photoatomic_ace.py`); the model decoder and its photons
  (`kika/ace/model_adapter/tests/test_ace_decode.py`, `test_ace_photons.py`). 90 tests
  passed on 2026-10-11. Every unconverted block is named and the conversion stays clean for
  cross-sections; an absent block says nothing
  (`kika/ace/model_adapter/tests/test_ace_decode_reports_what_it_drops.py`).
- Reading a real Fe-56 table and writing it back returns all XSS entries bitwise identical
  (`kika/ace/tests/test_ace_roundtrip.py`). `TODO(check)`: needs a full ACE file and
  `--deep`; not re-run for this page.
- `kika/ace/xsdir.py` and `kika/ace/comparison` have no tests.

## Implementation

- `kika/ace/parsers/`: `parse_ace.py` (`read_ace`) and one parser per block.
- `kika/ace/classes/`: `Ace`, `Header`, and one class per block.
- `kika/ace/writers/write_ace.py`: writing; `kika/ace/xsdir.py`: `xsdir` lines.
- `kika/ace/model_adapter/decode.py`: `decodeAce`, `aceStyles`, `qValuesByMT`;
  `photons.py`: `attachAcePhotons`.
- `kika/ace/comparison/compare_ace.py`: block-by-block comparison of two tables.

## References

- `TODO(ref)`: the ACE format specification, full citation.
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: ACER.
