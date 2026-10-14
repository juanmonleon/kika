# Reading and writing ACE

```{admonition} At a glance
:class: note
**Reads:** continuous-energy neutron ACE tables (legacy and 2.0.1 headers) block by
block, and photo-atomic tables.

**Writes:** an ACE table from its parsed object, its `xsdir` line, and a neutron table
from a processed model at one temperature.

**Into:** the parse layer's `Ace` object, and from it the canonical model: cross-sections,
Q-values, the secondary neutrons with their angular and energy distributions and
multiplicities, fission $\bar\nu$ and the delayed-neutron groups, photon production and
the unresolved-range probability tables (see {doc}`../../architecture/format-adapters`).

**Entry points:** {py:func}`kika.ace.parsers.parse_ace.read_ace`, {py:func}`kika.read`,
{py:func}`kika.ace.model_adapter.decodeAce`, {py:func}`kika.ace.model_adapter.encodeAce`,
`kika.ace.writers.write_ace.write_ace`, `kika.ace.xsdir.create_xsdir_files_for_ace`.

**Status:** stable (parse layer); experimental (model path).
```

## Background

ACE (A Compact ENDF) is the format of continuous-energy Monte Carlo codes such as MCNP and
Serpent. A table is processed data, written by NJOY's ACER: the resonance region is
already reconstructed, every cross-section is on one unionised energy grid, linearly
interpolated, at one temperature (`TODO(ref)`: the ACE format specification). After a
header, a table is three arrays: NXS (sizes), JXS (pointers to blocks) and XSS, one long
array holding every block. Each XSS word is written either as an integer (I20) or as a
real (E20.11). Energies are in MeV, cross-sections in barns, angular densities per unit
cosine, energy densities per MeV, and the header temperature is $kT$ in MeV. Locators
inside a block are relative to the start of the block; inside an energy-distribution
block, to the start of that block (JED).

## Coverage

What kika does with each block of a neutron table:

| Block | Parse layer (`read_ace`) | Model (`decodeAce`) |
|---|---|---|
| Header (ZAID, AWR, temperature, comment, NXS, JXS) | read; legacy and 2.0.1 headers | provenance; the temperature names the `heated_<T>` style |
| ESZ (grid, total, absorption, elastic, heating) | read | elastic and the sums MT 1 and MT 101 as cross-sections; the heating number becomes the KERMA, MT 301 |
| MTR, LQR, LSIG, SIG | read | one reaction per MT with its Q in eV; redundant sums with their summands |
| TYR (multiplicity and frame) | read | the neutron product's multiplicity and frame |
| LAND, AND | read: isotropic, equiprobable bins, tabulated, and the marker of the correlated laws | the angular half of each neutron product; elastic as two-body scattering |
| LDLW, DLW | read: laws 1, 2, 3, 4, 5, 7, 9, 11, 22, 24, 33, 44, 61, 66, 67 | the energy (or energy-angle) distribution of each neutron product; laws 22, 24 and 67 reported, not converted |
| NU, DNU, BDD, DNEDL, DNED | read | the fission neutron's $\bar\nu$ (prompt or total), the total $\bar\nu$ (MT 452), the delayed $\bar\nu$ (MT 455), one delayed group per precursor family with its decay constant, multiplicity and spectrum |
| GPD, MTRP, SIGP, LANDP, ANDP, DLWP, YP | read | photon products, as the ENDF decoder makes them from MF 12-15, plus their sampled form |
| UNR | read | probability tables on elastic, fission and capture |
| PTYPE, NTRO, IXS and the per-particle blocks | read | not converted (reported) |

A photo-atomic table (ZAID suffix ending in `p`) is parsed on its own, since its NXS and
JXS mean other things, and returned as a `PhotoatomicAce`. It has no model path.

## Reading

**1. Header.** The first line tells a 2.0.1 header from a legacy one; both are read into
`Header`, with NXS and JXS.

**2. XSS.** The array is read straight from the file in chunks, as reals, with a
placeholder at index 0 so that JXS pointers (1-based, as in the format) index it directly.
Which words were spelled as integers is kept, one bit per word, so the table can be
written back with the same text.

**3. Blocks.** Every block is parsed when the file is read, from its JXS pointer: none is
loaded on demand. Each block object is a view of the XSS array, not a copy, so editing a
value through a block edits the table. A block or reaction that cannot be read raises
(`strict=True`, the default); with `strict=False` the read goes on and every gap is listed
in `ace.read_report`.

**4. Into the model** (`kika.read(path)` or `decodeAce`). Energies are converted from MeV
to eV and densities per MeV to densities per eV. Each reaction's cross-section becomes one
lin-lin `XYs1d` (ACE has no interpolation regions) on its slice of the union grid. The
style chain is `evaluated` → `heated_<T>` (with $kT$ in eV), `URR_<T>` derived from it
when the table has probability tables, and `MonteCarlo` for the sampled distributions.
`<T>` is $kT/k_B$ rounded to the kelvin, as OpenMC names its temperatures, so 293.6 K is
`294K`. These are the labels kika's own processing and the OpenMC HDF5 reader give a
processed suite, so a table read from ACE goes to HDF5, or back to ACE, as it is.

Secondary neutrons become the neutron product of their reaction:

- the multiplicity is the constant $|TY|$, or the tabulated yield when $|TY| > 100$; the
  frame is the centre-of-mass frame when $TY < 0$;
- a law chain becomes a mixture of laws, each with its probability $P(E)$, even when the
  chain has a single law;
- law 2 is a discrete photon line; laws 3 and 33 a level; laws 1 and 4 a table of outgoing
  energies, discrete lines first; laws 5, 7, 9 and 11 the general evaporation, Maxwellian,
  evaporation and Watt spectra; law 44 Kalbach-Mann; law 61 a tabulated energy-angle
  distribution; law 66 N-body phase space;
- an uncorrelated law takes its angle from AND: an isotropic reaction is `Isotropic2d`,
  and 32 equiprobable cosine bins are a histogram.

For fission, the product's multiplicity is the prompt $\bar\nu$ when NU gives prompt and
total, and the total otherwise; the total and the delayed $\bar\nu$ are the multiplicity
sums MT 452 and MT 455. Each delayed group's multiplicity is $\bar\nu_d(E)\,P_g(E)$, with
the precursor probabilities as the file states them.

The cross-sections a sum rule makes redundant (MT 1, 3, 4, 27, 101, MT 103 beside MT
600-649, MT 18 beside its chances) go to the suite's sums with their summands, and the
response functions (MT 203-207, 301, 444) to its productions, so that anything summing the
reactions counts each partial once.

## Assumptions and defaults

- **The header temperature is $kT$ in MeV.** The heated style states it in eV, as
  `eV/k`.
- **LQR is aligned with MTR by position**; when their lengths differ, the shorter one is
  used and a warning says so.
- **MT 2 has Q = 0** by definition.
- **No reconstruction step is claimed.** A continuous-energy ACE has almost certainly been
  through RECONR, but the file does not say so, so no `crossSectionReconstructed` style is
  added.
- **A delayed group keeps the file's precursor probabilities**, even when they do not sum
  to one.

## Ambiguous and inconsistent input

- A table with no LQR block: no reaction gets a Q-value, reported as a loss.
- A summed reaction (MT 4, 18, 101, 27, 3) has no Q of its own; its missing Q is reported.
- A table with no header temperature: its style is named `heated_0K` and states none, with
  a warning.
- **A photon production given as a cross-section (MFTYPE 13)** becomes a multiplicity, the
  photon cross-section divided by its reaction's. Where the reaction's cross-section is
  zero the multiplicity is zero, and the photon cross-section there is kept beside the
  product, so that writing the table back returns it.
- **MFTYPE 13 photons on an MT the table has no cross-section for** hang on one orphan
  product per MT, whose cross-section is the sum of their photon cross-sections; each
  photon's multiplicity is its share of that sum (1 when it is alone).

## Refused

A table whose XSS or one of whose blocks cannot be read raises from the parser in strict
mode. Writing refuses an `Ace` object with no header or no XSS (`ValueError`), and an
existing output file unless `overwrite=True` (`FileExistsError`). Writing a model refuses
a suite that is not in the processed shape (`AceEncodeError`): more than one temperature
without one named, a cross-section that is not an `XYs1d` on a slice of the union grid
(ACE states each cross-section by the grid index it starts at), a distribution with no
sampled form, delayed groups without the delayed $\bar\nu$, a non-integer constant
multiplicity, no ZAID or AWR, or a temperature with no ZAID suffix and none given.

## Writing

**From the parsed object.** `write_ace(ace, filepath, overwrite=False)` writes the header
and the XSS array without modifying the object. Each word is spelled as the file it was
read from spelled it (or as the encoder of its block spells it, as NJOY writes it):
integers as I20 and reals as E20.11, with LF line ends on every platform.
`update_cross_sections` changes cross-sections in place in the XSS (the operation the
ACE-level perturbation uses), and `create_xsdir_files_for_ace` writes the `xsdir` lines.

**From the model** (`encodeAce`). The model is turned into the parsed objects, one block
at a time, and the block encoders lay out the XSS in NJOY's physical order (ESZ, NU, MTR,
LQR, TYR, LSIG, SIG, LAND, AND, LDLW, DLW, UNR, DNU, BDD, DNEDL, DNED, then the photon
blocks GPD, MTRP, LSIGP, SIGP, LANDP, ANDP, LDLWP, DLWP, YP), and fill NXS and JXS. Every
count and locator is recomputed from what is written. The result is read back with the
parser in strict mode, so the object returned is the one a file read from disk gives.
What is derived because the model does not carry it in ACE's form:

- ESZ: the union grid is the longest heated cross-section's energies; the total and
  absorption are MT 1 and MT 101 (summed from the partials when the suite has neither);
  the heating number is the KERMA divided by the total, zero where the total is zero;
- TY: the constant multiplicity, or a value above 100 with the yield written in DLW;
  negative in the centre-of-mass frame; 19 for the fission neutron;
- LOCB = −1 for a reaction whose law carries its own angle (laws 44, 61, 66, 67), and
  LOCB = 0 for an isotropic one;
- a cosine table that is a histogram with equal probabilities becomes equiprobable bins,
  the two-point isotropic histogram an isotropic entry, any other a tabulated entry;
- one interpolation region that is lin-lin is written with NR = 0, as NJOY writes it;
- an MFTYPE 13 photon cross-section is its multiplicity times its reaction's
  cross-section, on the union grid.

What ACE states and the model has no slot for is kept beside the decoded nodes and used
when the same table is written back: the order of MTR and YP (MT 4 goes last when a photon
yield needs it), which law of two that share a model form a chain used (3 or 33, 1 or 4),
the sign of law 61's cosine-table locators, the interpolation code of an outgoing-energy
table that is only lines, law 66's tabulated part, each precursor family's probability
table, each photon's MTRP number and MFTYPE, the probability of a one-law chain, and NXS(2)
and NXS(9-16). A suite that does not come from ACE follows NJOY's rules instead: reactions
with secondary neutrons first and the rest by MT; of the sums, only MT 18 and 103-107 and
those a photon yield is stated on; photons numbered $1000\,\mathrm{MT} + i$ as MFTYPE 16
yields; a probability of 1 over the law's domain; Z and A in NXS(10) and NXS(11).

**The ZAID suffix.** A table kika writes from a suite that does not come from ACE takes
the suffix of its temperature, the one kika's NJOY runs give (`.02c` at 293.6 K, `.06c`
at 600 K; the temperature is matched within 1 K). A table from kika's processing then
takes the place and the name of the one NJOY made, so inputs and `xsdir` files that name
it do not change; the library is named by the file and the `xsdir` line, not by the ZAID.
A temperature with no suffix raises, and an explicit `extension` is taken as given. The
same rule names the tables kika's NJOY runs write when no extension is given. The header
is the legacy one, which MCNP5, MCNP6 and Serpent read.

`kika.write` does not write ACE; an ACE table from an evaluation is produced by NJOY (see
{doc}`../processing/njoy`).

## What the round trip keeps and loses

- **Parse layer, read and write:** the file comes back byte for byte.
- **ACE → model → ACE:** every integer word of the neutron and photon blocks comes back
  equal and every real word within a few units in the last place (energies and
  densities pass through eV and back). Not written: the charged-particle production
  blocks (PTYPE, NTRO, IXS), and NJOY's unused room after some law tables, so the XSS is
  shorter by that room.
- **Model:** the evaluation behind the table is not in the file and is reported as lost;
  QM and LR have no ACE counterpart (reported); ACE has no covariances (reported).

## Report

Entries a user meets after `kika.read` of an ACE file: the absent evaluation (a loss, on
every table); no LQR block (a loss); a reaction with no Q (a loss, normal for sums); QM,
LR and covariances (unsupported, on every table); energy laws without a model form (laws
22, 24, 67); photon entries without a model form; MFTYPE 13 photons stated as
multiplicities, and orphan photons (approximations); the charged-particle blocks
(unsupported, stating that cross-sections are unaffected). A block absent from the table
produces no entry. `encodeAce` adds the loss of a Q-value it writes as 0, the sums it
derives because the suite has none, and the charged-particle blocks it does not write.

## Validation

- Parsers follow the format manual block by block
  (`kika/ace/tests/test_parsers_follow_the_manual.py`); blocks are views of the XSS
  (`kika/ace/tests/test_blocks_are_views_of_the_xss.py`); writing does not modify the
  object (`kika/ace/tests/test_write_ace_does_not_mutate.py`); a direct MT 18 update
  (`kika/ace/tests/test_update_cross_sections_direct_mt18.py`); photo-atomic tables
  (`kika/ace/tests/test_photoatomic_ace.py`).
- Reading a real Fe-56 table and writing it back returns the file byte for byte
  (`kika/ace/tests/test_ace_roundtrip.py::test_ace_roundtrip_is_byte_identical`; needs
  the table and `--deep`).
- The decoder: styles, labels and cross-sections array for array
  (`kika/ace/model_adapter/tests/test_ace_decode.py`), secondary neutrons, fission and
  delayed neutrons (`test_ace_secondaries.py`), photons (`test_ace_photons.py`); a decoded
  table goes to OpenMC's HDF5 without relabelling
  (`test_ace_decode.py::test_an_ace_suite_goes_to_hdf5_without_relabelling`).
- The encoder: ACE → model → ACE on a real Fe-56 table, every integer word equal and every
  real within $10^{-14}$ relative, once the charged-particle blocks and NJOY's unused room
  are taken out of the original
  (`kika/ace/model_adapter/tests/test_ace_encode.py::test_ace_model_ace_is_the_original_word_for_word`);
  the written file reads back in strict mode with the same words
  (`::test_the_written_table_is_a_file_kika_reads_back_strictly`); without the ACE
  bookkeeping, MTR follows NJOY's order with the same cross-sections
  (`::test_without_ace_provenance_mtr_follows_njoys_rule`). These need the table and
  `--deep`. `TODO(check)`: the same check over whole libraries is not in a public test.
- `kika/ace/xsdir.py` and `kika/ace/comparison` have no tests.

## Implementation

- `kika/ace/parsers/`: `parse_ace.py` (`read_ace`) and one parser per block.
- `kika/ace/classes/`: `Ace`, `Header`, the read report, and one class per block.
- `kika/ace/encoders/`: one encoder per block, the energy laws, and the layout that places
  the blocks and fills NXS and JXS; `kika/ace/contract.py`: what a table must satisfy to
  be laid out.
- `kika/ace/writers/write_ace.py`: writing; `kika/ace/xsdir.py`: `xsdir` lines.
- `kika/ace/model_adapter/decode.py`: `decodeAce`, the styles and labels, Q-values, the
  probability tables and where each reaction goes; `secondaries.py`: neutron products,
  fission and delayed neutrons; `photons.py`: photon products; `encode.py`: `encodeAce`.
- `kika/ace/comparison/compare_ace.py`: block-by-block comparison of two tables.

## References

- `TODO(ref)`: the ACE format specification, full citation.
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: ACER.
