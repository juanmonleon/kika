# Reading and writing ENDF

```{admonition} At a glance
:class: note
**Reads:** ENDF-6 tapes: the incident-particle sublibraries (MF 1 to 10, MF 12 to 15, the
covariances MF 31 to 35), and the decay and fission-yield sublibraries.

**Writes:** a whole ENDF-6 tape from the model (MF 1 to 10, 12 to 15, 31 to 35), or one
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
| MF 8 (the MTs of reactions), MF 9, MF 10 | Decoded: the production of radioactive nuclides, one `production` per final state in `suite.productions`. MF 8 is also kept verbatim in the provenance, for the records the model has no node for. |
| MF 12 to 15 | Decoded: photon production, attached to the reactions' products. |
| MF 31 to 35 | Decoded into the separate `CovarianceSuite` (MF 31, 33, 34, 35 as covariance sections; MF 32 as resonance-parameter covariances). |
| MF 8 / MT 454, 457, 459 on an incident-particle tape, MF 8 states with LMF=6, MF 23, 27, 28, 39, 40 | Not decoded; reported as unsupported (MF 8 is kept verbatim, so it is written back). |

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
- **Radionuclide production (MF 8, 9, 10).** MF 9 and MF 10 state, per final state,
  the product's ZA (IZAP), its level index (LFS), the Q-value to the ground state (QM)
  and to the level (QI), and a TAB1: a multiplicity $y(E)$ in MF 9, a production
  cross-section $\sigma_p(E)$ in MF 10. MF 8 lists the same states and says, through
  LMF, which file holds their yield (ENDF-6 §8.2). Each state becomes one `production`
  reaction, labelled `MT<n> <product>` (`MT102 Am242_e2`), with an output channel of
  genre `production` holding the one product made:

  - **MF 9**: the cross-section is a `reference` to the reaction's own `eval`
    cross-section, and the product's multiplicity is the MF 9 table;
  - **MF 8 with LMF=3** (no MF 9 or MF 10 for the state): the same reference, with a
    multiplicity of 1 over the reaction's domain;
  - **MF 10**: the cross-section is the MF 10 table, and the multiplicity is 1 over the
    table's domain.

  In every case the production cross-section is

  $$
  \sigma_p(E) = \sigma(E)\, y(E)
  $$ (eq-endf-production)

  with $\sigma$ the referenced cross-section for MF 9 and LMF=3, and $y = 1$ for MF 10
  and LMF=3. The channel's Q is QI, over the table's domain. The level energy
  $\mathrm{QM} - \mathrm{QI}$ goes to the product's nuclide in `PoPs` (`nuclearLevel` =
  LFS); QM itself is kept in the provenance. An excited product is aliased as a
  metastable state (`Am242_m1` names `Am242_e2`), numbered by level within each nuclide.
  MF 9 and MF 10 are read on their own: their records name the product and the level, so
  MF 8 is not needed to decode them. FUDGE reaches them only through MF 8, labels a
  production by the form of its multiplicity (`m(E)*Am242_e2 [...]`), and gives the Q and
  a constant multiplicity the whole projectile domain; kika's label does not change when
  the data is edited, and its domains are the ones the file states.
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
- **An MF 8 state whose LMF points at an MF 9 or MF 10 that does not state it** is
  reported lost; MF 8 is kept and written back as read. **An MF 9 on a reaction with no
  MF 3**, or an MF 9 or MF 10 state whose IZAP is not a nuclide (ZAP=-1 for fission), has
  no cross-section or product to name: the section is kept verbatim, reported as
  unsupported, and written back unchanged.
- **A blank field in a LIST or TAB1 body** is read as zero, which is what ENDF-6 means by
  one; NP points and the declared number of LIST values always come back.

## Refused

- **MF 2 LRF=4 (Adler-Adler)**, by name: GNDS has no node for it, and no evaluation of
  ENDF/B-VIII.1, JEFF-4.0 or JENDL-5 uses it. The decoder reports it and the ranges after
  it are not read.
- **An MF 2 competitive width that disagrees with the file's QX/LRX**, and an R-matrix
  section whose channel counts disagree (`ValueError`).
- **An INTG record (MF 32) with NDIGIT outside 2-6** (`ValueError`): ENDF-6 §32.2.3 allows
  only those widths.
- **Non-numeric MAT, MF or MT columns** (`ValueError`).
- **A LIST or TAB1 body field that is neither blank nor a number** (`ValueError`, naming the
  field and the line): skipping it would shift every later value by one slot, or return
  fewer points than NP, and the section would read as valid.
- **A PENDF attached as the `recon` form** that is not 0 K, not lin-lin (so not the
  output of RECONR), or of another material (`ValueError`).

A failure to parse one MF 3 section is logged and that section is dropped, without an
exception.

## Writing

**A whole tape** (`kika.write(suite, path, format="endf")`, or `writeEndfTape`):

1. One encoder per section writes MF 1, 2, 3 to 10, 12 to 15 and 31 to 35, from what the
   model holds and, for the parts the decoder kept verbatim (MF 1/460, MF 6 LAW=5, MF 8),
   from the provenance. Which sections are written follows the provenance: an MF 4 only
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

**A PENDF** (the reconstructed or heated tables, {doc}`../processing/processing-chain`) is
written so that NJOY reads it. A reconstructed or heated MF 3 has one model region per
discontinuity, and NJOY holds at most 20 interpolation regions in a TAB1 (MODER stops on
more), so neighbouring regions of one law are merged as RECONR writes them: the jump as a
repeated energy inside one region, the same function. The same holds for the heating
sections (MT 301, 444). The MF 1/451 descriptive text is cut to ENDF-6's five-line
identification block, as RECONR leaves it: with the 560 lines of an evaluation's
documentation NJOY 2016.78 HEATR stopped, and with 300 it ran (Fe-56 of ENDF/B-VIII.1,
measured 2026-10-13). A heated material carries, for each reaction, only the points its
own table needs, and not every point of the temperature's common grid (which made the
293.6 K material of Fe-56 nine times the size of BROADR's).

On kika's PENDF of Fe-56 of ENDF/B-VIII.1, NJOY 2016.78 (Windows build) runs MODER to ACER
(measured 2026-10-14). At 0 K the ACE file agrees with the RECONR chain's ($10^{-3}$) to
$4.7\cdot10^{-4}$ in MT 102 and $3.6\cdot10^{-4}$ in MT 1 at the 99th percentile of NJOY's
points. At 293.6 K the two ACE files differ more in MT 102 ($3.1\cdot10^{-3}$ at the 99th
percentile, all of it between 1 keV and 850 keV), and the difference is NJOY's. Against a
pointwise broadening of each code's own 0 K MT 102 to $10^{-6}$, at 3 000 random energies
in that range read lin-lin from each ACE file, kika's table is within $1.0\cdot10^{-3}$ at
the 99th percentile (largest $1.3\cdot10^{-3}$: its broadening and thinning tolerances
together) and BROADR's within $7.8\cdot10^{-3}$ (largest $9.9\cdot10^{-3}$). RECONR and
BROADR allow up to ten times their tolerance (`errmax`, default $10\cdot$`err`) where
their integral criterion permits it, and kika holds its tolerance at every point.

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
otherwise. AWR is the target's mass in `PoPs` over the neutron's; the MF 1/451 header
comes from the evaluated style (temperature, projectile energy domain, library) and its
descriptive text from the style's ENDF-compatible documentation; a section's QM and LR
come from the reaction's Q value and from what its residual decays into (LR from the
light particles of the breakup, by ENDF-6's LR table). A suite read from ENDF is never
derived.

**Productions (MF 8, 9, 10).** A production whose cross-section is a reference and whose
multiplicity is tabulated is written to MF 9; one whose cross-section is a table, to
MF 10 (a constant multiplicity $c \ne 1$ is folded in as $c\,\sigma_p$, and reported); a
reference with a multiplicity of 1 is an MF 8 state with LMF=3. IZAP and LFS come from the
product's nuclide, QI from the channel's Q, QM from the provenance or, derived, as QI plus
the level energy in `PoPs`. The TAB1's interpolation pairs are the file's own while they
fit the table; otherwise the model's regions are written with neighbouring regions of the
same law merged (a GNDS file splits a step into two regions, ENDF keeps it in one). MF 8
is written as the provenance kept it while its states are still the productions'; when
they change, or the suite did not come from ENDF, it is rebuilt from the productions with
NO=1 (the decay data of each state is the decay sublibrary's, ENDF-6 §8.2), keeping the
records of any state the model does not carry.

MF 2/151 is derived when the resonances hold a scattering radius and nothing else: one
isotope (abundance 1) with one LRU=0 range, LRP=0. EL and EH are the radius' own energy
domain, or the evaluation's projectile energy domain when the radius states none (the
report says so); SPI is the target's spin in `PoPs`; AP is the constant radius, converted
from fm to $10^{-12}$ cm. A radius that depends on energy, or a target whose spin `PoPs`
does not state, leaves MF 2 unwritten and reported lost; SPI is not assumed. Resolved and
unresolved parameters are not derived: they need fields the model has no node for (QX,
LRX, LAD and the particle-pair columns), so MF 2 is reported lost and the tape's header
can state LRP=1 with no MF 2 after it.

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
- **Kept as the file's text:** MF 8 of a reaction, including the decay lists of its
  states (NO=0) and the LMF=6 states the model does not carry, while the productions are
  unchanged. Through GNDS those records are not carried, and MF 8 comes back rebuilt with
  NO=1.
- **Not kept in a whole-tape write:** what the model does not decode (MF 8 decay and
  yield data on an incident-particle tape, MF 23, 27, 28, 39, 40), each named in the
  report; the
  directory, which is rebuilt; and, on the micro tapes of the tests, an MF 34 section
  that the covariance decoder collapses, which the report names.
- **Not checked by the fixed point:** anything the model does not carry is absent from
  both sides of the comparison, so the round trip cannot see it; the report is what says
  it is missing.

## Report

`suite.report` after a read, and the value returned by `kika.write`, is a
`ConversionReport` (see {doc}`../../architecture/canonical-model`). For ENDF, the
entries a user meets most:

- **unsupported**: an MF the decoder does not handle (MF 23, 27, 28, 39, 40; MF 8 decay
  data on an incident-particle tape; MF 8 LMF=6 states), Adler-Adler
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
  blank, malformed and non-finite fields behave as stated under *Reading* and *Refused* (a
  blank LIST or TAB1 body field reads as zero, an unreadable one raises)
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
  Th-232) runs with `--deep` (`test_endf_round_trip.py::test_the_same_holds_on_real_tapes`,
  4 tests, passed 2026-10-20).
- **A suite read from GNDS** is written as an ENDF tape with MAT 128 for H-2, every
  reaction and an MF 2 holding the scattering radius alone
  (`kika/gnds/tests/test_encode.py::test_a_gnds_sourced_suite_writes_an_endf_tape`); that
  MF 2 states what the radius and `PoPs` state, and a target with no spin gets no MF 2
  (`kika/endf/model_adapter/tests/test_a_gnds_radius_reaches_mf2.py::test_the_radius_alone_is_written_as_the_tape_states_it`,
  `::test_a_target_without_a_spin_is_not_given_one`).
- **Radionuclide production (MF 8-10)**, on two committed cuts: JENDL-5 Co-59 (LMF=3, 9
  and 10, two isomers) and ENDF/B-VIII.1 Am-241 (MF 9, MF 8 with decay lists). Every MF 9
  and MF 10 section re-emits byte for byte
  (`kika/endf/tests/test_mf9_mf10.py::test_every_fixture_section_roundtrips`); a
  whole-tape write gives back MF 8, 9 and 10 as read and the productions as a fixed point
  (`kika/endf/model_adapter/tests/test_productions.py::test_the_tape_comes_back_with_mf8_mf9_mf10_as_read`);
  through GNDS, MF 9 and MF 10 come back as read and MF 8 is rebuilt with the same states
  (`::test_through_gnds_mf9_mf10_return_and_mf8_is_derived`); the ground-state and isomer
  production cross-sections of Am-241 capture add up to the capture cross-section to a
  relative $10^{-12}$ (`::test_the_production_cross_section_is_sigma_times_multiplicity`);
  stripping the provenance and deriving it gives back every field the decoder wrote
  (`kika/endf/model_adapter/tests/test_derive_oracle.py`). On the 34 MF 9 and MF 10
  sections of 25 tapes from ENDF/B-VII.1 and VIII.1, JEFF-3.3 and 4.0 and JENDL-5, every
  section re-emits byte for byte; from the GNDS file FUDGE 6.14 writes for ENDF/B-VII.1
  Am-241, kika writes the tape's MF 9 byte for byte (measured 2026-10-14).
- **Splicing an unchanged section** leaves the rest of the tape byte-identical
  (`kika/endf/tests/test_a_splice_keeps_the_tapes_format.py::test_a_splice_of_an_unchanged_section_leaves_the_rest_byte_identical`).
- **Refusals**: Adler-Adler is named and stops the parser and the decoder
  (`kika/endf/tests/test_mf2_adler_adler_is_refused_by_name.py::test_the_parser_names_adler_adler_and_stops`,
  `::test_the_model_decoder_names_it_too`); a repeated abscissa is written as a step
  (`kika/endf/tests/test_a_descending_table_is_never_written.py::test_a_repeated_abscissa_is_a_step_and_is_allowed`).
- **Against FUDGE** (6.14). FUDGE's ENDF-to-GNDS translation and kika's read agree on every
  cross-section, Legendre coefficient and resonance energy of the Fe-56 micro-tapes to a
  relative $10^{-14}$ (`kika/endf/model_adapter/tests/test_fudge_in_the_loop.py`,
  `::test_every_cross_section_reads_the_same`,
  `::test_the_legendre_coefficients_read_the_same`,
  `::test_the_resonance_energies_read_the_same`), and an ENDF tape kika writes from the
  NNDC GNDS files of Be-9, Li-6, Fe-56 and U-235 matches FUDGE's in every number of
  MF 1/452-458, 3, 4, 5, 6 and 12-15, and N-14 in MF 12-15
  (`kika/endf/model_adapter/tests/test_derive_vs_fudge.py::test_kika_writes_from_nndcs_gnds_what_fudge_writes`).
  "Every number" is field by field: equal as text, or, where kika writes a closer field
  than FUDGE's seven significant digits, kika's value rounded to seven digits is
  FUDGE's. Both run only with a Python that has FUDGE (`KIKA_FUDGE_PYTHON`), and passed
  2026-10-20 with FUDGE 6.14.

## Implementation

- `kika/_records.py`: the 11-column field grammar, number parsing and formatting.
- `kika/endf/read_endf.py`, `kika/endf/parsers/`, `kika/endf/classes/`: the parse layer
  (`parse_mf9_10.py` and `classes/mf9` for MF 9 and MF 10).
- `kika/endf/utils.py`: TAB1, TAB2, LIST and INTG readers and writers.
- `kika/endf/model_adapter/decode.py`: `decodeReactionSuite`, MF 1 and MF 3; `angular.py`
  (MF 4), `energy.py` (MF 5), `energy_angle.py` (MF 6), `thermal_scattering.py` (MF 7),
  `photons.py` (MF 12-15), `productions.py` (MF 8-10), `resonances.py` (MF 2), `multiplicity.py` (MF 1/452-456),
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
  (MF 4), §8.2 (MF 8 of a reaction), §9 and §10 (radionuclide production), §32.2.3 (INTG
  records).
- C. Mattoon, B. Beck and G. Gert, "Managing and Processing Nuclear Data Libraries with
  FUDGE", *EPJ Web of Conferences* 284, 14010 (2023), doi:10.1051/epjconf/202328414010.
