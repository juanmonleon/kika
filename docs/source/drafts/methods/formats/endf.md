# Reading and writing ENDF

```{admonition} At a glance
:class: note
**Reads:** ENDF-6 tapes: the incident-particle sublibraries (MF 1 to 7, MF 12 to 15, the
covariances MF 31 to 35), and the decay and fission-yield sublibraries.

**Writes:** a whole ENDF-6 tape from the model (MF 1 to 7, 12 to 15, 31 to 35), or one
section re-encoded and spliced into an existing tape.

**Into:** a {py:class}`~kika.nuclear_data.model.suite.ReactionSuite` with its
`CovarianceSuite` (see {doc}`../../architecture/canonical-model`).

**Entry points:** {py:func}`kika.read`, {py:func}`kika.write`, `kika.read_endf`,
`kika.endf.model_adapter.decodeReactionSuite`,
`kika.endf.model_adapter.decodeCovarianceSuite`,
`kika.endf.writers.assemble.writeEndfTape`.

**Status:** experimental — the model path is not yet in a release (after 0.4.0).
```

## Background

An ENDF-6 tape (Brown, 2023) is a text file of 80-column lines. Each line carries six
data fields of 11 columns (columns 1-66), then the material number MAT (67-70), the file
number MF (71-72), the section number MT (73-75) and a line sequence number NS (76-80).
A *file* MF holds one kind of data (MF 3 cross-sections, MF 4 angular distributions,
MF 33 cross-section covariances); a *section* MT within it is one reaction. Sections are
built from a few record types: CONT (six numbers), LIST (a count and a list of numbers),
TAB1 (a table of $y(x)$ with interpolation regions), TAB2 (interpolation regions for a
table of tables) and TEXT. Energies are in eV and cross-sections in barns.

A TAB1 states its interpolation as $(\mathrm{NBT}_k, \mathrm{INT}_k)$ pairs: the points
up to $\mathrm{NBT}_k$ (counted from 1, cumulatively) are joined by the law
$\mathrm{INT}_k$ (ENDF-6 §0.5.2). A floating-point field is usually written without the
letter E: `1.234567+5` means $1.234567 \times 10^5$.

## Coverage

What kika does with each part of an incident-particle tape when it reads it into the
model (`kika.read`, or `read_endf` followed by `decodeReactionSuite`):

| Part | What kika does |
|---|---|
| MF 1 / MT 451 | Decoded: header flags (LRP, LFI, NLIB, NMOD, ELIS, STA, LIS, LISO, NFOR, AWI, EMAX, LREL, NSUB, NVER, LDRV, TEMP) and the descriptive text go to the provenance; the target becomes a particle in `PoPs`. |
| MF 1 / MT 452, 455, 456 | Decoded: total, delayed and prompt neutron multiplicities. |
| MF 1 / MT 458 | Decoded: fission energy release. |
| MF 1 / MT 460 | Kept verbatim in the provenance (GNDS has no node for delayed fission photons) and written back as it was. |
| MF 2 / MT 151 | Decoded: resonance parameters, by formalism. Adler-Adler (LRF=4) is refused. |
| MF 3 | Decoded: cross-sections, under the label `eval`. |
| MF 4 | Decoded: angular distributions (Legendre, tabulated, or both). |
| MF 5 | Decoded: energy distributions, every law. |
| MF 6 | Decoded: energy-angle distributions, every law except LAW=5, which is kept verbatim. |
| MF 7 | Decoded, on a thermal-scattering tape. |
| MF 12 to 15 | Decoded: photon production, attached to the reactions' products. |
| MF 31 to 35 | Decoded into the separate `CovarianceSuite` (MF 31, 33, 34, 35 as covariance sections; MF 32 as resonance-parameter covariances). |
| MF 8 (incident-particle tapes), MF 23, 27, 28, 40 | Not decoded; reported as unsupported. |

A decay sublibrary (NSUB 4 or 5) and a fission-yield sublibrary (NSUB 11) are decoded
into a `PoPs` tree instead of a reaction suite.

A section the model does not decode is **not kept in the suite**: it is named in the
conversion report. The parse layer (`read_endf`) still holds it, and a sampling run that
patches a tape (see *Writing*) keeps it, because it copies the tape.

## Reading

Reading has two steps. `read_endf` parses the tape into format-shaped objects (one per
MF, one per MT), and the decoders turn those into model nodes.

**1. Lines and sections.** The tape's first line, when its MF and MT are 0 or blank, is
the tape identification record and is kept whole. The material number is taken from the
first line with MF > 0 and MAT > 0. Lines are grouped by (MF, MT); the end records (SEND,
FEND, MEND, TEND) are not data. A tape may be 75 or 80 columns wide (NS is optional);
the width is measured, and kept for writing back. Sequence numbers wrap from 99 999 to 1.

**2. Numbers.** A field with a decimal point and no `E`, written
`mantissa ± exponent`, is reassembled into one decimal string (`1.234567e+5`) and
converted **once** to the nearest double. It is never computed as
$\text{mantissa} \times 10^{\text{exponent}}$, which rounds twice and can land one unit in
the last place away: `2.427894+7` reads as exactly 24 278 940. Any other field goes to
Python's `float()`. A whole number comes back as an integer. A blank field is "no value".

**3. Records.** CONT fields are read positionally. A TAB1 reads NR interpolation pairs,
three to a line, then NP $(x, y)$ pairs. A LIST reads six values to a line, and **a blank
field inside a LIST body is read as zero** (two JEFF-4.0 evaluations leave R-matrix
channel columns blank, and skipping them would shift every later value). The covariance
readers for MF 33 and 34 skip blank fields instead. How a short last line was padded
(blanks or zeros) is recorded, so that it can be reproduced.

**4. Into the model.**

- **Cross-sections (MF 3)** become a `Regions1d` labelled `eval`, one `XYs1d` per
  interpolation region, even when there is only one region. QI becomes the reaction's
  Q-value; QM and LR go to the provenance. The section's (NBT, INT) pairs are kept in the
  provenance too.
- **Angular distributions (MF 4).** LTT=0 is isotropic. LTT=1 (Legendre) and LTT=2
  (tabulated) become an `XYs2d` (a `Regions2d` with more than one incident-energy region).
  LTT=3 becomes a `Regions2d` of exactly two parts, Legendre below the transition energy
  and tabulated above. ENDF leaves $a_0 = 1$ implicit; kika writes it out, and copies the
  other coefficients as given, trailing zeros included (they are the order the evaluator
  declared). Nothing is renormalised. LCT=1 is the laboratory frame, LCT=2 the centre of
  mass. A tabulated $f(\mu)$ with one region is an `XYs1d`, with several a `Regions1d`.
- **MF 4 of MT 600-849** is put on the emitted proton, deuteron, triton, helion or alpha
  (ENDF-6 §4 implies the particle from the MT), and not on a neutron.
- **Multiplicities.** MT 456 (prompt) is the primitive on the product; MT 452 (total)
  and MT 455 (delayed) become multiplicity sums with the values the file states. When
  only MT 452 is present, it is the primitive.
- **Resonances (MF 2)** become nodes by formalism. Radii are converted from ENDF's
  $10^{-12}$ cm to fm. The resonance region is **not reconstructed on read**: the
  `eval` cross-section there is only the background the file states.
- **Units.** Cross-sections in b against eV; multiplicities dimensionless; $\mu$ and
  $f(\mu)$ without a unit; Q-values in eV; masses in amu, from AWR times the neutron
  mass, 1.00866491595 amu.
- **Covariances (MF 31-35)** are decoded by `decodeCovarianceSuite` into a separate
  `CovarianceSuite`; an MF 34 Legendre order is a slice of its section.

## Assumptions and defaults

- **A TAB1 with no interpolation pairs** is one lin-lin region over all its points; the
  report records it as an approximation.
- **ZA is rounded, not truncated.** A ZA written as a float can read back a hair below
  the integer, and truncating it would name the neighbouring nuclide.
- **A blank QI or QM in MF 3 is 0**, which the parser supplies. A section with no Q-value
  at all is reported as lost.
- **An MF 4 LCT other than 1 or 2** is taken as centre of mass, with a warning.
- **An MF 4 with LTT=0 and LI≠1** is read as isotropic, with a warning.
- **An LNU=1 multiplicity** (a polynomial) is given the energy domain of the evaluation
  (EMAX), or $10^{-5}$ eV to 20 MeV when EMAX is absent; the report records the
  approximation.
- **The evaluated style's temperature** is the TEMP of MF 1/451, in K.

## Ambiguous and inconsistent input

- **Is MT 103 a reaction or a sum?** ENDF does not say. kika decides from the tape: a
  stated MT is a **sum** when the file also gives its partials (MT 103 with MT 600-649,
  MT 4 with MT 51-91, MT 1 with MT 2 and MT 3, and the other rules of
  `kika._constants.MF3_SUM_RULES`), and an ordinary reaction otherwise. MT 1 is always a
  sum, because MT 2 is always present. A sum keeps the values the file states; it is not
  recomputed on read, since the file's numbers are the evaluation.
- **Several isotopes in MF 2** are merged into one set of regions, and their abundances
  are not represented: reported as unsupported.
- **More than one unresolved range** in MF 2: reported as unsupported. A range whose LRU
  is not 0, 1 or 2 is dropped.
- **MT 452 and MT 455 without MT 456**: the multiplicities are decoded and a warning says
  the prompt part is missing.
- **MF 33 blocks mixing absolute and relative components**: NJOY RECONR is run (and
  cached) to express them relative to the cross-section, with a warning.
- **An MF 4 section with no MF 3 for its reaction** is reported as lost. **An MF 34 with
  no MF 4** gives a warning.
- **An MF 1/455 with LDG=1** (energy-dependent delayed groups): reported as unsupported,
  its records kept in the provenance.
- **A TAB1 point whose x or y does not parse** is dropped, so fewer than NP points can
  come back, and a non-blank LIST field that does not parse is skipped. Neither is
  reported. `TODO(check)`: whether these two silent cases should be reported.

## Refused

- **MF 2 LRF=4 (Adler-Adler)**, by name: GNDS has no node for it, and no evaluation of
  ENDF/B-VIII.1, JEFF-4.0 or JENDL-5 uses it. The decoder reports it and the ranges after
  it are not read.
- **An MF 2 competitive width that disagrees with the file's QX/LRX**, and an R-matrix
  section whose channel counts disagree (`ValueError`).
- **An INTG record (MF 32) with NDIGIT outside 2-6** (`ValueError`): ENDF-6 §32.2.3 allows
  only those widths.
- **Non-numeric MAT, MF or MT columns** (`ValueError`).
- **A PENDF attached as the `recon` form** that is not 0 K, not lin-lin (so not the
  output of RECONR), or of another material (`ValueError`).

A failure to parse one MF 3 section is logged and that section is dropped, without an
exception.

## Writing

**A whole tape** (`kika.write(suite, path, format="endf")`, or `writeEndfTape`):

1. One encoder per section writes MF 1, 2, 3 to 7, 12 to 15 and 31 to 35, from what the
   model holds and, for the parts the decoder kept verbatim (MF 1/460, MF 6 LAW=5), from
   the provenance. Which sections are written follows the provenance: an MF 4 only
   where the product records an LTT, an MF 6 only where the reaction came from one.
2. Sections are sorted by (MF, MT); SEND, FEND, MEND and TEND are added; the tape
   identification record is the one the tape was read with, or a default, which the
   report records as a loss.
3. The MF 1/451 directory is rebuilt from the tape as written.
4. `label=` chooses which form to write (`eval` by default); a reaction that has no form
   under that label falls back to `eval`, and the report says which.

A tape holding several materials is written by `writeEndfTapes`; a repeated MAT is
refused.

**Interpolation pairs.** MF 3 is written with the file's own (NBT, INT) pairs when they
still describe the table; otherwise they are rebuilt from the model's regions, with a
warning. Two neighbouring regions share their boundary point: it is written once when
the values on both sides agree, and twice (a repeated energy, a step) when they differ.
Regions that are not adjacent cannot be written as one TAB1 and raise.

**Numbers** are written in 11 columns without `E`, correctly rounded once:
6 decimals in the mantissa for an exponent of one digit (7 significant figures), 5 for
two digits, 4 for three. Zero is ` 0.000000+0`; a non-finite value raises. A value the
source wrote in fixed notation with more digits (`0.12640975`) therefore comes back to
7 significant figures. An opt-in precise mode writes fixed notation where it is closer,
and the reconstructed (`recon`) form is written with it. NR and NP are recomputed from
the data; a descending TAB1 is refused (`NonMonotonicTable`), a repeated abscissa is
allowed (it is a step).

**Derived bookkeeping.** A suite that did not come from ENDF (from GNDS, or built in
memory) lacks ENDF's own fields. The writer derives them on a copy: each field comes from
the model, from FUDGE's ENDF conversion flags carried in the GNDS file, or from a named
default, and the report records the derivation. The material number comes from the
caller, then from FUDGE's flag, then from the ENDF material table, and is refused
otherwise. A suite read from ENDF is never derived.

**Splicing into an existing tape.** The sampling pipeline's `endf-delta` output copies
the source tape, re-encodes only the sections a realisation changed (MF 3, MF 4, MF 5,
MF 1/452, 455, 456), replaces each in place, and rebuilds the directory once. The rest of
the tape, including what the model does not decode, is untouched, and the splice keeps
the tape's line endings and width.

## What the round trip keeps and loses

- **Kept exactly:** the arrays the model carries (read, write and read again gives the
  same doubles; the test compares scalar values to 8 decimal places), the tape
  identification record, and, for a section encoded from the
  file it came from, the section's text byte for byte.
- **Kept numerically:** a value the source wrote with more than 7 significant figures,
  which comes back rounded to 7; covariance values, to a relative $10^{-6}$.
- **Not kept in a whole-tape write:** what the model does not decode (MF 8 on an
  incident-particle tape, MF 23, 27, 28, 40), each named in the report; the
  directory, which is rebuilt; and, on the micro tapes of the tests, an MF 34 section
  that the covariance decoder collapses, which the report names.
- **Not checked by the fixed point:** anything the model does not carry is absent from
  both sides of the comparison, so the round trip cannot see it; the report is what says
  it is missing.

## Report

`suite.report` after a read, and the value returned by `kika.write`, is a
`ConversionReport` (see {doc}`../../architecture/canonical-model`). For ENDF, the
entries a user meets most:

- **unsupported**: an MF the decoder does not handle (MF 8, 23, 27, 28, 40), Adler-Adler
  resonances, several isotopes in MF 2, several unresolved ranges. **These mean part of
  the evaluation is not in the model**: a computation that needs it will not see it.
- **losses**: an MF 4 with no MF 3, a section with no Q-value, a default tape
  identification on writing. Worth reading before trusting a written tape.
- **approximations**: a TAB1 with no interpolation pairs taken as lin-lin, an LNU=1
  multiplicity given a default domain, a derived ENDF field. **The kind that looks like
  data**: read them.
- **warnings**: unusual flags (LCT, LI), covariance blocks made relative through RECONR,
  an MF 34 with no MF 4. Informative; nothing was lost.

Structural and value checks of the covariances themselves (`kika.endf.check_covariances`,
run by `decodeCovarianceSuite` when asked) are a separate report, with levels *note*,
*warn* and *defect*; they are not in the conversion report.

## Validation

All tests below passed on 2026-10-11 unless marked otherwise.

- **Number grammar.** `2.427894+7` reads as exactly 24 278 940
  (`kika/endf/tests/test_fast_record_reading.py::test_implicit_exponents_are_one_correctly_rounded_decimal_conversion`);
  the two spellings of one value read as the same double
  (`kika/endf/tests/test_format_constants.py::test_the_two_spellings_of_one_value_read_as_the_same_double`);
  blank, malformed and non-finite fields behave as stated under *Reading*
  (`kika/endf/tests/test_fast_record_reading.py::test_standard_syntax_and_legacy_fallback`,
  `::test_positional_fields_keep_padding_truncation_and_zero_policies`).
- **The model is a fixed point of a whole-tape write**, on the committed micro-tape of
  Fe-56 (JEFF-4.0: MF 1, 2, 3, 4, 34): reactions, resonances and particles
  (`kika/endf/model_adapter/tests/test_endf_tape_fixed_point.py::test_the_reaction_suite_is_a_fixed_point`),
  the header (`::test_the_header_is_a_fixed_point`), the covariances to a relative
  $10^{-6}$ (`::test_the_covariances_are_a_fixed_point_in_their_numbers`), and the report
  names what did not survive (`::test_the_report_names_every_file_that_did_not_survive`).
  The comparison rounds scalar floats to 8 decimal places and compares arrays exactly.
- **Byte identity of re-encoded sections**: every MF 3 section of the micro-tape
  (`kika/endf/model_adapter/tests/test_endf_round_trip.py::test_every_mf3_section_encodes_byte_identically_to_the_file`),
  the MF 1/451 header (`kika/endf/model_adapter/tests/test_header_round_trip.py::test_the_header_encodes_byte_identically_to_the_file`),
  MF 5 of Cf-252 (`test_endf_tape_fixed_point.py::test_the_mf5_section_a_tape_gets_back_is_byte_identical`),
  and MF 6 on Be-9, C-12, Li-6, Ti-50 and U-235 (`::test_a_tape_with_mf6_comes_back_with_all_of_it`).
  The same MF 3 gate on whole evaluations (Fe-56 from JEFF-4.0 and JENDL-5, U-235,
  Th-232) runs with `--deep` (`test_endf_round_trip.py::test_the_same_holds_on_real_tapes`);
  `TODO(check)`: not re-run for this page, the tapes are not on this machine.
- **Splicing an unchanged section** leaves the rest of the tape byte-identical
  (`kika/endf/tests/test_a_splice_keeps_the_tapes_format.py::test_a_splice_of_an_unchanged_section_leaves_the_rest_byte_identical`).
- **Refusals**: Adler-Adler is named and stops the parser and the decoder
  (`kika/endf/tests/test_mf2_adler_adler_is_refused_by_name.py::test_the_parser_names_adler_adler_and_stops`,
  `::test_the_model_decoder_names_it_too`); a repeated abscissa is written as a step
  (`kika/endf/tests/test_a_descending_table_is_never_written.py::test_a_repeated_abscissa_is_a_step_and_is_allowed`).
- **Against FUDGE.** FUDGE's ENDF-to-GNDS translation and kika's read agree on every
  cross-section, Legendre coefficient and resonance energy of the Fe-56 micro-tapes to a
  relative $10^{-14}$ (`kika/endf/model_adapter/tests/test_fudge_in_the_loop.py`,
  `::test_every_cross_section_reads_the_same`,
  `::test_the_legendre_coefficients_read_the_same`,
  `::test_the_resonance_energies_read_the_same`), and an ENDF tape kika writes from the
  NNDC GNDS files of Be-9, Li-6, Fe-56 and U-235 matches FUDGE's in every number of
  MF 1/452-458, 3, 4, 5, 6 and 12-15
  (`kika/endf/model_adapter/tests/test_derive_vs_fudge.py::test_kika_writes_from_nndcs_gnds_what_fudge_writes`,
  measured 2026-10-09 according to the test). `TODO(check)`: both need a Python with
  FUDGE (`KIKA_FUDGE_PYTHON`) and were not re-run for this page.

## Implementation

- `kika/_records.py`: the 11-column field grammar, number parsing and formatting.
- `kika/endf/read_endf.py`, `kika/endf/parsers/`, `kika/endf/classes/`: the parse layer.
- `kika/endf/utils.py`: TAB1, TAB2, LIST and INTG readers and writers.
- `kika/endf/model_adapter/decode.py`: `decodeReactionSuite`, MF 1 and MF 3; `angular.py`
  (MF 4), `energy.py` (MF 5), `energy_angle.py` (MF 6), `thermal_scattering.py` (MF 7),
  `photons.py` (MF 12-15), `resonances.py` (MF 2), `multiplicity.py` (MF 1/452-456),
  `covariances.py` (MF 31-35); `encode.py` and the same modules for the section encoders;
  `derive/` for bookkeeping a non-ENDF suite lacks; `pendf.py` to attach a PENDF as
  `recon`.
- `kika/endf/writers/assemble.py`: the whole-tape writer; `endf_writer.py`: splicing a
  section into an existing tape; `redundant.py`: recomputing stated sums after an edit.
- `kika/_constants.py`: `MF3_SUM_RULES`, the neutron mass.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual: Data Formats and Procedures for the Evaluated
  Nuclear Data Files ENDF/B-VI, ENDF/B-VII and ENDF/B-VIII*, Brookhaven National
  Laboratory (2023), doi:10.2172/2007538. Cited as ENDF-6: §0.5.2 (interpolation), §4
  (MF 4), §32.2.3 (INTG records).
- C. Mattoon, B. Beck and G. Gert, "Managing and Processing Nuclear Data Libraries with
  FUDGE", *EPJ Web of Conferences* 284, 14010 (2023), doi:10.1051/epjconf/202328414010.
