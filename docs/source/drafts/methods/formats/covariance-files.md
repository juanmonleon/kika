# Reading and writing multigroup covariance files

```{admonition} At a glance
:class: note
**Reads:** multigroup cross-section covariances from SCALE COVERX files (binary or text),
NJOY BOXER card images and NJOY COVFIL tapes (the ERRORR output, MF 33 or MF 34).

**Writes:** the same three formats, from one multigroup covariance object.

**Into:** {py:class}`kika.cov.CrossSectionCovariance` (cross-section covariances), or
{py:class}`kika.cov.LegendreCovariance` for an MF 34 COVFIL. These are kika's multigroup
covariance containers; they sit beside the canonical model (see
{doc}`../../architecture/canonical-model`), not inside it.

**Entry points:** {py:func}`kika.cov.parse_covmat.read_coverx`,
{py:func}`kika.cov.parse_covmat.write_coverx`, {py:func}`kika.cov.parse_covmat.read_boxer`,
{py:func}`kika.cov.parse_covmat.write_boxer`, {py:func}`kika.cov.parse_covmat.read_covfil`,
{py:func}`kika.cov.parse_covmat.write_covfil`; the class methods `from_coverx`,
`from_boxer`, `from_covfil`, `from_gendf`, `to_coverx`, `to_boxer`, `to_covfil`.

**Status:** stable — readers and writers since 0.2.0.
```

## Background

An evaluation states covariances on energy bins of its own choosing (ENDF-6 Ch. 33 and 34).
A processing code collapses them to a fixed group structure of $G$ groups with boundaries
$E_1, \dots, E_{G+1}$, and the result is a set of $G \times G$ blocks, one per pair of
parameters $(\text{material}, \text{MT})$. Three file formats carry such a set:

- **COVERX**, the covariance library format of the SCALE system: Fortran unformatted
  sequential records (binary), or a text transcription of the same content. Energies are
  in eV and in descending order, groups are numbered from high energy to low. Blocks are
  stored in banded form (SCALE 6.3.1 manual, COVERX format).
- **BOXER**, the card-image format written by NJOY's COVR module: 80-column header cards,
  each followed by run-length compressed values and control integers. A file holds the
  group boundaries, cross-sections, standard deviations and covariance or correlation
  matrices (NJOY 2016 manual, COVR).
- **COVFIL**, the ENDF-formatted tape written by NJOY's ERRORR module: an MF 1 / MT 451
  section holding the group boundaries, optional MF 3 group cross-sections, and MF 33 (or
  MF 34) sections whose sub-sections hold the matrix row by row (NJOY 2016 manual,
  ERRORR).

In every format a block is the covariance between the vertical (row) parameter and the
horizontal (column) parameter, and a block is normally stated once per unordered pair:
the transposed block is implied. None of the three formats carries a flag saying whether
a block is relative or absolute; what kika assumes is under *Assumptions and defaults*.

## Coverage

| Part | What kika does |
|---|---|
| COVERX binary, records 1 and 3 (identification, description) | Read and discarded. Written from `title`. |
| COVERX binary, record 2 (file control) | NGROUP, NNGRUP, NMMP, NMTRIX decoded; NGGRUP, NTYPE, NHOLL read and not kept. |
| COVERX, group boundaries | Decoded into `energy_grid`. |
| COVERX binary, records of (MATID, MTID, MWGT) | MATID and MTID decoded to identify the cross-section records; MWGT discarded. |
| COVERX binary, cross-section records | The first $G$ values decoded into `cross_sections` when not all zero; the error vector that follows is not read. |
| COVERX, covariance blocks | Decoded into a dense $G \times G$ block, Legendre order 0 only. Higher-order Legendre blocks (NBLOCK > 1) are skipped. |
| COVERX, blocks for an MT kika does not know | Skipped silently. |
| BOXER, ITYPE 0 (group boundaries) | Decoded into `energy_grid`. |
| BOXER, ITYPE 1 (cross-sections) | Decoded into `cross_sections`. |
| BOXER, ITYPE 2 (standard deviations) | Read and used to turn a correlation matrix into a covariance; not kept. |
| BOXER, ITYPE 3 (covariance matrix) | Decoded into a block. |
| BOXER, ITYPE 4 (correlation matrix) | Decoded and converted into a covariance block, {eq}`eq-covariance-files-correlation`; refused without the ITYPE 2 deviations it needs. |
| BOXER, HLIB and HDESCR fields | Read and discarded. |
| COVFIL, tape header line | Skipped. |
| COVFIL, MF 1 / MT 451 | ZA, AWR, temperature, NGRP and the group boundaries decoded; AWR and temperature go to `metadata`. |
| COVFIL, MF 3 | Decoded into `cross_sections`; for an MF 34 tape, MT 251 and above become grouped Legendre coefficients $a_\ell$, with $\ell = \text{MT} - 250$. |
| COVFIL, MF 33 | Decoded into a {py:class}`~kika.cov.CrossSectionCovariance`, every block relative. |
| COVFIL, MF 34 | Decoded into a {py:class}`~kika.cov.LegendreCovariance`, every block relative, frame unknown. |
| COVFIL, MF 31, MF 35 (ERRORR output for multiplicities and spectra) | Refused by the identifier (`kika.identify`), see *Refused*. |
| COVFIL, a second material on the tape | Not read. |

## Reading

### COVERX

The reader tells binary from text by the first four bytes: if all four are printable
ASCII (or tab, line feed, carriage return), the file is read as text. A binary file's byte
order is the one under which the first record marker (the length of record 1) lies between
4 and 1000; both big- and little-endian files are read.

A binary file is a sequence of Fortran records, each framed by its length before and
after; a frame whose two lengths disagree is refused. The records are:

```text
record 1   identification (title, version)
record 2   NGROUP NNGRUP NGGRUP NTYPE NMMP NMTRIX NHOLL        7 integers
record 3   NHOLL six-byte description words
record 4   NNGRUP+1 group boundaries, float32, eV
record 5   NMMP triplets (MATID, MTID, MWGT)
records    one per (MATID, MTID): NGROUP float32 cross-sections, then the errors
then, NMTRIX times:
  control  MAT1 MT1 MAT2 MT2 NBLOCK                            5 integers
  band     NGROUP pairs (JBAND, IJJ), then NBLOCK integers
  data     NBLOCK records of packed values, float32
```

Row $j$ of a block (1-based) holds $\text{JBAND}_j$ values starting at column
$j - \text{IJJ}_j + 1$; a row with $\text{JBAND}_j = 0$ is zero. The first data record is
unpacked into a dense block and the remaining $\text{NBLOCK} - 1$ are read and dropped.
**MAT1 and MT1 are the column parameter, MAT2 and MT2 the row parameter**, so the block
lands under row $(\text{MAT2}, \text{MT2})$ and column $(\text{MAT1}, \text{MT1})$.

A text file is read line by line:

```text
line 1     title
line 2     NGROUP ...                                          integers; only the first is used
lines      NGROUP+1 group boundaries, eV (D or E exponents)
per block  MAT_col MT_col MAT_row MT_row N                     5 integers
           NGROUP*NGROUP values, row-major, dense
```

A block header is any line of exactly five tokens whose second and fourth are integers kika
knows as reaction numbers. The text format carries no cross-sections.

**Energy order.** With `ascending=True` (the default) and a file whose first boundary is
larger than its last, the grid is reversed, every block is reflected in both axes,

$$
\Sigma'_{ij} = \Sigma_{G+1-i,\,G+1-j},
$$ (eq-covariance-files-flip)

and every cross-section vector is reversed, so group 1 is the lowest-energy group. With
`ascending=False` the file's order is kept. With `energy_unit="MeV"` the boundaries are
multiplied by $10^{-6}$; the values are not touched, since a relative covariance is
dimensionless and an absolute one is in units of cross-section.

**Identifiers.** COVERX material identifiers are kept as the file has them; no MAT
translation is applied.

### BOXER

Each card group starts with an 80-column header,

```text
(I1, A3, 8A4, 2(I5,I4), 2(I4,I3), 3I4)
ITYPE HLIB HDESCR MAT MT MAT1 MT1 NVAL NVF NCON NCF NROWM NROWH NCOLH
```

followed by NVAL values in the fixed format chosen by NVF and NCON control integers in the
format chosen by NCF:

| NVF | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|
| Fortran format | 11F7.4 | 10F8.5 | 1P8E9.2 | 1P8E10.3 | 1P7E11.4 | 1P6E12.5 | 1P6E13.6 | 1P5E14.7 |

NCF from 1 to 6 means integers of width NCF, $\lfloor 80/\text{NCF} \rfloor$ per card. A
blank field reads as zero.

**Decompression.** The matrix has NROWH rows and NCOLH columns; NCOLH = 0 means a
symmetric $\text{NROWH} \times \text{NROWH}$ matrix of which only the upper triangle
($j \ge i$) is stored, row by row. The elements are visited in that order, and each
control integer $c$ fills the next $|c|$ elements: for $c < 0$ with the next value from the
value list, for $c > 0$ with the element directly above in the previous row (zero on the
first row). In a symmetric matrix each element is also written to its mirror position.

**Continuation.** A large matrix may span several card groups with the same
(MAT, MT, MAT1, MT1, ITYPE). NROWM is the number of rows still to come after this group;
the next group continues from row $\text{NROWH} - \text{NROWM}$, and the matrix is
complete when NROWM = 0.

**Correlation matrices.** An ITYPE 4 matrix is converted with the ITYPE 2 standard
deviations $s$ of its row and column parameters,

$$
\Sigma_{ij} = \rho_{ij}\, s^{\text{row}}_i\, s^{\text{col}}_j .
$$ (eq-covariance-files-correlation)

**Identifiers and grid.** MAT numbers are translated to ZAID through kika's ENDF MAT table
(a MAT not in the table is kept as the number it is). The group boundaries are kept in the
file's order; they are in eV, and `energy_unit="MeV"` multiplies them by $10^{-6}$.

### COVFIL

Numbers are read field by field in the ENDF grammar of {doc}`endf` (11-column fields, six
per line, the exponent letter optional; a blank field is zero). Line 1 is the tape header
and is skipped. The records kika reads are:

```text
MF1/MT451  HEAD  ZA AWR . . . .
           CONT  TEMP 0 NGRP 0 NGRP+1 0
           NGRP+1 group boundaries, eV
MF3/MT     HEAD  ZA 0 0 0 N 0,   then N group values           (optional)
MF33/MT    HEAD  ZA AWR 0 0 0 NMT1
           per sub-section:  CONT  0 0 MAT1 MT1 0 NG
                             per stored row:  0 0 NW ISTART NW IROW,  then NW values
MF34/MT    HEAD  ZA AWR LTT NMT1 NL NL1
           per sub-section:  CONT  0 0 MT1 L L1 NG
                             rows as for MF33
```

Row IROW of the block receives its NW values from column ISTART on; rows not stated are
zero. A sub-section ends at its row NG, or at the end of the section. MAT1 = 0 means the
column parameter is the tape's own material. The section that follows MF 1 and MF 3
decides the result: MF 33 gives a {py:class}`~kika.cov.CrossSectionCovariance`, MF 34 a
{py:class}`~kika.cov.LegendreCovariance` whose blocks carry Legendre orders $(L, L_1)$ and
the MF 1 grid. MAT numbers are translated to ZAID as for BOXER. The grid is in eV, and
`energy_unit="MeV"` multiplies it by $10^{-6}$.

`from_gendf` and `from_covfil` are the same reader; both raise `TypeError` when the tape is
MF 34, which `LegendreCovariance.from_covfil` reads.

## Assumptions and defaults

- **COVERX and BOXER blocks carry no relative/absolute flag.** The two formats do not
  state it, and kika leaves `is_relative` empty. Every consumer in `kika.cov`, including
  `CrossSectionCovariance.project_to_grid`, reads an unstated class as relative, the form
  in which multigroup covariance libraries are normally distributed.
- **COVFIL blocks are relative.** ERRORR's `irelco` option (card 2) chooses absolute (0)
  or relative (1) covariances, and relative is its default (NJOY 2016.78, `errorr.f90`);
  the MF 33 records it writes do not say which (an MF 34 section writes `irelco` as L2 of
  its HEAD, which kika does not read). A COVFIL written with `irelco` = 0 would be read
  with the wrong flag. MF 34 blocks from COVFIL are in an unknown reference frame
  (LAB or CM), because the records kika reads do not state one.
- **Energies are in eV** in all three formats; `energy_unit="MeV"` on reading converts the
  boundaries to MeV, and any other unit is refused.
- **Group 1 is the lowest group** after a COVERX read with `ascending=True`. BOXER and
  COVFIL grids are kept in the file's order.
- **One material per COVFIL.** The MAT of the first data line is the tape's material; a
  second material is not read, and the writer refuses to write more than one.
- **An all-zero block is not stored.** Every reader skips a block whose entries are all
  exactly zero, which states nothing an absent block does not. A block whose entries are
  small, or whose anticorrelations cancel its variances so that it sums to about zero, is
  kept.
- **An unstated block is not zero.** Only the blocks a file states are stored; the
  assembled matrix (`CrossSectionCovariance.covariance_matrix`) mirrors each off-diagonal
  block into its transpose position and leaves a block never stated as NaN.

## Ambiguous and inconsistent input

- **Which pair is the row.** COVERX writes the column (horizontal) parameter first; kika
  follows that order and stores the block under its row parameter. A reader that took the
  first pair as the row would transpose every cross-reaction block; the SCALE 44-group
  library checks this (see *Validation*).
- **A COVFIL sub-section that does not end at row NG** has no terminator the reader can
  see, so its next CONT would be read as a row. kika's writer always writes row NG (a
  single zero when the row is empty).
- **An unknown MT in COVERX** (one not in kika's reaction table, such as a lumped MT 851)
  is skipped without a report; BOXER and COVFIL keep every MT.

## Refused

- A COVERX binary record whose leading and trailing lengths differ, or a first record
  whose length gives no plausible byte order (`InvalidDataFormatError`): the file is
  corrupt or not COVERX, and every later offset would be wrong.
- A COVERX text file with fewer or more group boundaries than NGROUP + 1, or a boundary
  that is not a number (`InvalidDataFormatError`).
- A BOXER correlation matrix (ITYPE 4) without the ITYPE 2 standard deviations of its row
  or its column parameter (`InvalidDataFormatError`): a correlation cannot be turned into
  a covariance without them, and storing it as one would be wrong by the square of the
  deviations.
- A file from which no block was decoded, and a COVFIL whose first covariance section is
  neither MF 33 nor MF 34 (`EmptyParsingError`).
- An `energy_unit` other than `"eV"` or `"MeV"` (`ValueError`).
- `CrossSectionCovariance.from_covfil` (or `from_gendf`) on an MF 34 tape, and
  `LegendreCovariance.from_covfil` on an MF 33 tape (`TypeError`).
- Writing a {py:class}`~kika.cov.LegendreCovariance` to COVERX or BOXER (`TypeError`):
  neither format has Legendre orders.
- Writing a COVFIL from a covariance with rows for more than one isotope, or an MF 34
  COVFIL with a block whose column isotope is another material (`ValueError`): a COVFIL
  holds one material, and an MF 34 sub-section has no MAT1 field to name another.
- Writing a BOXER file with a MAT above 99999, or a COVFIL with a MAT above 9999
  (`ValueError`): the number does not fit its field, and the file would be corrupt.
- `kika.identify` refuses, by name, an NJOY group tape with no MF 33 or MF 34 (a groupr
  GENDF), and ERRORR output carrying only MF 31 or MF 35: kika has no reader for them, and
  `read_covfil` reads MF 33 and MF 34 only.

## Writing

All three writers take a {py:class}`~kika.cov.CrossSectionCovariance` on one shared group
structure (`num_groups`, `energy_grid`); COVFIL also takes a
{py:class}`~kika.cov.LegendreCovariance`. A grid in MeV is converted to eV first (with a
warning for BOXER and COVFIL).

**COVERX** (`fmt="binary"` by default, big-endian by default, or `fmt="text"`). An
ascending grid is reversed, every block reflected by {eq}`eq-covariance-files-flip` and
every cross-section reversed, so the file is in COVERX's descending order. The writer
derives:

- record 2 as (G, G, 0, 2, NMMP, NMTRIX, NHOLL), with NHOLL the number of six-byte words
  the title needs (at least one);
- the parameter list, the sorted union of every block's row and column parameters and of
  the cross-sections, each with MWGT = 4;
- one cross-section record per parameter, the cross-sections followed by an error vector
  of zeros (the object has no error vector), and zeros where no cross-section is stored;
- per block: the control record (column parameter, then row parameter, NBLOCK = 1), the
  band of each row from its first to its last non-zero column (interior zeros included),
  and the values.

Binary values are float32, about seven significant figures, and the writer warns. The text
writer writes the header line as (G, 2, NMMP, NMTRIX) in I12 fields, the grid and each
dense block in E15.7, five per line, and the block header as five I12 fields ending in 1;
it warns that cross-sections are lost.

**BOXER** writes one card group per item: the grid (ITYPE 0) as a column vector, then for
every block the cross-sections of its two parameters (ITYPE 1, once each) and, for a
diagonal block, the standard deviations (ITYPE 2)

$$
s_i = \sqrt{\max(\Sigma_{ii}, 0)},
$$ (eq-covariance-files-sd)

then every block as a covariance (ITYPE 3) in a single card group (NROWM = 0). A diagonal
block (same material and MT) is written symmetric (NCOLH = 0, upper triangle); any other
block in full. The compression is the inverse of the reading rule: runs of elements equal
to the one above become positive controls, runs of a repeated new value a negative
control and one value. NCF is the narrowest width that fits every control integer. The
value format is `nvf` (default 10, 1P8E10.3); the writer warns for `nvf` ≤ 10 and suggests
14. MAT numbers come from kika's ZAID-to-MAT table; one above 99999 does not fit the I5
field, and the writer refuses it.

**COVFIL** writes a tape header, MF 1 / MT 451, MF 3 when cross-sections are stored (MF 33
only), and the MF 33 or MF 34 sections, then FEND, MEND and TEND. The writer derives:

- the material (MAT and ZA) from the row isotope, which is the same for every block (see
  *Refused*); MAT1 = 0 for a column parameter of that material, its MAT otherwise; only
  that material's cross-sections go into MF 3;
- the MF 1 HEAD as (ZA, AWR, 6, 0, −11, 0) and the CONT as (TEMP, 0, G, 0, G+1, 0), as
  ERRORR writes them (NJOY 2016.78 `covout`, and an ERRORR tape of Fe-56): L1 is the ENDF
  format version and N1 = −11 the flag of an MF 33 or MF 34 covariance tape, which NJOY's
  COVR checks before it reads one
  (`kika/cov/tests/test_covariance_formats.py::test_a_written_covfil_carries_errorrs_tape_flag`);
  kika's reader does not use them;
- AWR and the temperature from `metadata` unless passed explicitly, else zero;
- one MF 33 (MF 34) section per row MT, NMT1 the number of blocks with that row MT; for
  MF 34, NL and NL1 the number of distinct $L$ and $L_1$ among them;
- per row, the span from its first to its last non-zero column; an all-zero row is not
  written, except row G, which is always written so the reader can see the end;
- sequence numbers restarting at 1 in each section.

A MAT above 9999 does not fit the four-column field, and the writer refuses it.

## What the round trip keeps and loses

| Path | Kept | Lost or changed |
|---|---|---|
| COVFIL → model → COVFIL | Blocks, grid, cross-sections, AWR, temperature, relative flag, numerically equal (ENDF floats). | The tape label, any second material. |
| COVERX binary | Blocks, grid and cross-sections to float32 (relative difference within $2 \cdot 10^{-7}$ in the test). | Title, MWGT, error vectors, Legendre blocks above order 0, unknown MTs, the relative flag. |
| COVERX text | Blocks and grid to E15.7. | Cross-sections, title, unknown MTs, the relative flag. |
| BOXER (`nvf=14`) | Blocks, grid and cross-sections, equal to $10^{-12}$ in the test. | HLIB and HDESCR, the relative flag; the lower triangle of a diagonal block that is not exactly symmetric (the upper one is kept and mirrored). With the default `nvf=10`, values keep four significant figures. |

An all-zero block does not come back, since no reader stores one. COVERX keeps the model's
isotope identifiers while BOXER and COVFIL translate them to MAT and back, which is the
identity only for nuclides in kika's MAT table.

## Report

None: the readers return the covariance object and keep no conversion report. What a
writer cannot carry is said as a Python `UserWarning` at write time (float32 precision,
lost cross-sections, MeV converted to eV, a BOXER value format of ten columns or fewer).
What it cannot write correctly at all (a MAT number too wide for its field, a second
material in a COVFIL) is refused, under *Refused*. Everything listed as silent under
*Ambiguous and inconsistent input* has no warning.

## Validation

The tests below passed on 2026-10-11 unless marked. They are in
`kika/cov/tests/test_covariance_formats.py` unless named otherwise, on a 4-group Fe-56
covariance with two diagonal blocks, one cross-reaction block and two cross-section
vectors.

- **COVERX round trip**, text, big-endian binary and little-endian binary: blocks, grid
  and (binary) cross-sections come back to relative $2 \cdot 10^{-7}$ and absolute
  $2 \cdot 10^{-8}$; the text format returns no cross-sections; `energy_unit="MeV"` scales
  the grid by $10^{-6}$; `ascending=False` returns the descending grid with every block
  reflected (`::test_coverx_reader_roundtrip_and_unit_conversion`, 3 cases).
- **COVERX record layout**: NHOLL = 2 for a seven-character title, a description record of
  NHOLL six-byte words, and a cross-section record of $2G$ float32 values
  (`::test_coverx_binary_uses_standard_hollerith_and_xs_record_lengths`).
- **BOXER round trip** at `nvf=14`: equal to $10^{-12}$
  (`::test_boxer_reader_roundtrip`).
- **COVFIL round trip**: equal to $10^{-12}$, AWR and temperature kept, every block
  relative (`::test_covfil_gendf_reader_roundtrip_and_metadata`).
- **Chain COVFIL → COVERX binary → BOXER**: the end equals the start to the float32
  tolerance (`::test_covfil_coverx_boxer_cross_format_chain`).
- **Edge cases**: a block that sums to about zero without being zero comes back from
  BOXER, COVFIL and binary COVERX (`::test_a_block_that_sums_to_nearly_zero_is_kept`);
  `energy_unit="MeV"` scales the BOXER and COVFIL grids
  (`::test_mev_converts_the_values_not_just_the_label`); a COVFIL with two materials and a
  MAT too wide for its field are refused (`::test_covfil_refuses_two_materials`,
  `::test_a_mat_too_wide_for_its_field_is_refused`); `to_covfil` keeps the temperature it
  read (`::test_to_covfil_keeps_the_temperature_it_read`); a BOXER correlation without
  deviations is refused (`::test_a_boxer_correlation_without_deviations_is_refused`);
  `project_to_grid` reads an unstated class as relative
  (`::test_project_to_grid_reads_an_unstated_class_as_relative`).
- **Identification**: files written by the three writers are recognised as `covfil`,
  `coverx` (binary and text) and `boxer`; a groupr tape and ERRORR MF 35 output are refused
  by name (`kika/tests/test_identify.py::test_each_kind_is_recognised_as_itself`,
  `::test_a_groupr_tape_is_named_rather_than_called_covfil`,
  `::test_errorr_mf35_output_is_refused_with_the_reason`,
  `::test_a_binary_file_that_is_not_coverx_is_refused`).
- **Against real files**, value by value against the numbers printed in the files: the
  orientation of a cross-reaction block of the SCALE 44-group library
  (`::test_real_scale44_coverx_cross_block_orientation`), and NJOY ERRORR tapes of Fe-56 for
  MF 33 (56 groups, 152 blocks) and MF 34
  (`::test_real_fe56_mf33_gendf_against_raw_reference_values`,
  `::test_real_fe56_mf34_gendf_against_raw_reference_values`). They read the files named by
  `KIKA_TEST_SCALE44_COVERX`, `KIKA_TEST_FE56_MF33_GENDF` and `KIKA_TEST_FE56_MF34_GENDF`
  and are skipped without them; with them, they passed on 2026-10-20.

There is no oracle check against SCALE or NJOY reading a file kika wrote.

## Implementation

- `kika.cov.parse_covmat`: every reader and writer. COVERX: Fortran record framing,
  binary/text and byte-order detection, banded packing, the energy flip. BOXER: header
  cards, value and control formats, compression and decompression, continuation blocks.
  COVFIL: the MF 1, MF 3, MF 33 and MF 34 parsers and writers, MAT/ZAID translation.
- `kika.cov.cross_section_covariance`: {py:class}`~kika.cov.CrossSectionCovariance` and its
  `from_*` / `to_*` methods, which call the functions above; `covariance_matrix` assembles
  the blocks.
- `kika.cov.legendre_covariance`: `LegendreCovariance.from_covfil` and `to_covfil`.
- `kika._records`: the ENDF field grammar used by the COVFIL reader and writer.
- `kika._identify`: telling COVFIL, COVERX and BOXER apart by content (`kika.identify`).

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538. Cited as ENDF-6: Ch. 33 (MF 33), Ch. 34 (MF 34).
- W. Wieselquist and R. A. Lefebvre (eds.), *SCALE 6.3.1 User Manual*, ORNL/TM-SCALE-6.3.1,
  Oak Ridge National Laboratory (2023), doi:10.2172/1959594: the COVERX format.
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: ERRORR and COVR, the COVFIL and BOXER formats.
