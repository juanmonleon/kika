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
  sequential records (binary), defined in appendix A of Drischler (1980). Energies are in
  eV and in descending order, groups are numbered from high energy to low, and blocks are
  stored in banded form. SCALE also distributes its libraries as text: the same blocks,
  dense, with the group structure given as the upper bound of each group, and no
  cross-sections or standard deviations.
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
the transposed block is implied. Whether a block is relative or absolute is stated in two
places only: the NTYPE of a binary COVERX file, for the whole file, and the `irelco` that
ERRORR writes in the HEAD of an MF 34 section. A COVFIL MF 33 section, a BOXER file and a
COVERX text file do not say; what kika assumes is under *Assumptions and defaults*.

## Coverage

| Part | What kika does |
|---|---|
| COVERX binary, records 1 and 3 (identification, description) | Read and discarded. Written from `title`. |
| COVERX binary, record 2 (file control) | NGROUP, NNGRUP, NTYPE, NMMP, NMTRIX decoded; NTYPE sets `is_relative` and says whether the blocks are correlations. NGGRUP and NHOLL are reported and not kept. |
| COVERX, group boundaries | Decoded into `energy_grid`. A text file that gives only the $G$ upper bounds takes its lowest boundary from the matching structure of `kika.energy_grids`. |
| COVERX binary, records of (MATID, MTID, MWGT) | MATID and MTID decoded to identify the cross-section records; MWGT reported, not kept. |
| COVERX binary, cross-section records | The first $G$ values decoded into `cross_sections` when not all zero; the $G$ standard deviations that follow convert a correlation file (NTYPE 3) and are reported, not kept. |
| COVERX, covariance blocks | Decoded into a dense $G \times G$ block. A binary block split into NBLOK records of consecutive groups is put back together. |
| COVERX, blocks for an MT kika does not know | Skipped and reported. |
| BOXER, ITYPE 0 (group boundaries) | Decoded into `energy_grid`. |
| BOXER, ITYPE 1 (cross-sections) | Decoded into `cross_sections`. |
| BOXER, ITYPE 2 (standard deviations) | Read and used to turn a correlation matrix into a covariance; reported, not kept. |
| BOXER, ITYPE 3 (covariance matrix) | Decoded into a block. |
| BOXER, ITYPE 4 (correlation matrix) | Decoded and converted into a covariance block, {eq}`eq-covariance-files-correlation`, with ITYPE 2 deviations found anywhere in the file; refused when the file has none for one of its parameters. |
| BOXER, HLIB and HDESCR fields | Read and discarded. |
| COVFIL, tape header line | Skipped. |
| COVFIL, MF 1 / MT 451 | ZA, AWR, temperature, NGRP and the group boundaries decoded; AWR and temperature go to `metadata`, ZA names the material when kika's MAT table does not. |
| COVFIL, MF 3 | Decoded into `cross_sections`; for an MF 34 tape, MT 251 and above become grouped Legendre coefficients $a_\ell$, with $\ell = \text{MT} - 250$. |
| COVFIL, MF 33 | Decoded into a {py:class}`~kika.cov.CrossSectionCovariance`, every block flagged relative. ERRORR writes its MF 31 output (MT 452, 455, 456) as MF 33, and it is read as such. |
| COVFIL, MF 34 | Decoded into a {py:class}`~kika.cov.LegendreCovariance`, frame unknown; an ERRORR section (MT 251) takes its flag from `irelco`. |
| COVFIL, MF 35 (ERRORR output for spectra) | Refused by the identifier (`kika.identify`), see *Refused*. |
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
records    one per (MATID, MTID): NGROUP float32 cross-sections, then NGROUP deviations
then, NMTRIX times:
  control  MAT1 MT1 MAT2 MT2 NBLOK                             5 integers
  band     NGROUP pairs (JBAND, IJJ), then LGRP(1..NBLOK)
  data     NBLOK records of packed values, float32
```

NTYPE says what the blocks and the deviations are (Drischler 1980, appendix A): 1 a
covariance matrix and standard deviations, 2 a relative covariance matrix and relative
standard deviations, 3 a correlation matrix and standard deviations. Row $j$ of a block
(1-based) holds $\text{JBAND}_j$ values starting at column $j - \text{IJJ}_j + 1$; a row
with $\text{JBAND}_j = 0$ is zero. The matrix is split into NBLOK blocks of consecutive
groups, block $n$ holding $\text{LGRP}_n$ rows; data record $n$ holds the packed values of
those rows, and the records are put back together in order. **MAT1 and MT1 are the column
parameter, MAT2 and MT2 the row parameter**, so the block lands under row
$(\text{MAT2}, \text{MT2})$ and column $(\text{MAT1}, \text{MT1})$. A correlation block
(NTYPE 3) becomes a covariance with the deviations $s$ of its two parameters,
{eq}`eq-covariance-files-correlation`.

A text file is read line by line:

```text
line 1     title
line 2     NGROUP ...                                          integers; only the first is used
lines      NGROUP upper bounds, or NGROUP+1 boundaries, eV (D or E exponents)
per block  MAT_col MT_col MAT_row MT_row N                     5 integers
           NGROUP*NGROUP values, row-major, dense
```

SCALE's text libraries give the upper bound of each of the NGROUP groups and no lowest
boundary; the grid ends there when the next line is a block header. Its lowest boundary
is then the one of the `kika.energy_grids` structure whose upper bounds these are, to a
relative $10^{-6}$ (SCALE's 44 and 56 groups, $10^{-5}$ eV). In those libraries the third
integer of line 2 is NMTRIX. A block header is any line of exactly five integers; its
block is read when its second and fourth are reaction numbers kika knows. The text format
carries no cross-sections.

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
deviations $s$ of its row and column parameters, once the whole file is read: COVR writes
a cross block before the deviations of its second parameter, which come with that
parameter's own block,

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
MF34/251   HEAD  ZA AWR 0 irelco L L1                          (ERRORR)
           per sub-section:  CONT  0 0 251 L L1 NG,  rows as for MF33
MF34/MT    HEAD  ZA AWR LTT NMT1 NL NL1                        (any other MT)
           per sub-section:  CONT  0 0 MT1 L L1 NG,  rows as for MF33
```

Row IROW of the block receives its NW values from column ISTART on; rows not stated are
zero. A sub-section ends at its row NG, or at the end of the section. MAT1 = 0 means the
column parameter is the tape's own material. The section that follows MF 1 and MF 3
decides the result: MF 33 gives a {py:class}`~kika.cov.CrossSectionCovariance`, MF 34 a
{py:class}`~kika.cov.LegendreCovariance` whose blocks carry Legendre orders $(L, L_1)$ and
the MF 1 grid. ERRORR writes MF 34 under MT 251 and puts `irelco`, not NMT1, in the HEAD
(NJOY 2016.78, `errorr.f90`, `covout`): such a section is read up to its end, and
`irelco` sets the flag of its blocks. The tape's own material is its ZAID in kika's MAT
table, or the ZA of the MF 1 HEAD when the table does not have the MAT; another material
(MAT1) is translated as for BOXER. The grid is in eV, and `energy_unit="MeV"` multiplies
it by $10^{-6}$.

`from_gendf` and `from_covfil` are the same reader; both raise `TypeError` when the tape is
MF 34, which `LegendreCovariance.from_covfil` reads.

## Assumptions and defaults

- **A binary COVERX file is what its NTYPE says.** NTYPE 2 blocks are flagged relative and
  NTYPE 1 blocks absolute. An NTYPE 3 block is a correlation; it is multiplied by the
  deviations of the file and flagged absolute, because the format calls those deviations
  relative only for NTYPE 2. Any other NTYPE leaves the flag unset.
- **BOXER and COVERX text blocks carry no relative/absolute flag.** The files do not state
  it, and kika leaves `is_relative` empty. Every consumer in `kika.cov`, including
  `CrossSectionCovariance.project_to_grid`, reads an unstated class as relative, the form
  in which multigroup covariance libraries are normally distributed. A BOXER file is in
  the unit of the covariances COVR read, which COVR's own `irelco` declares.
- **COVFIL MF 33 blocks are relative.** ERRORR's `irelco` option (card 2) chooses absolute
  (0) or relative (1) covariances, and relative is its default (NJOY 2016.78,
  `errorr.f90`); the MF 33 records it writes do not say which, so a COVFIL written with
  `irelco` = 0 is read with the wrong flag. An ERRORR MF 34 section states `irelco`, and
  its blocks take it.
- **An ERRORR MF 34 section keeps MT 251 as its reaction.** ERRORR writes the section and
  every sub-section under MT 251 and does not record which reaction's Legendre
  coefficients they are. MF 34 blocks from COVFIL are in an unknown reference frame (LAB
  or CM), because the records kika reads do not state one.
- **The lowest boundary of a SCALE text grid comes from kika's group structures**, the one
  whose upper bounds the file gives, because the file does not carry it. A grid that
  matches none is refused (*Refused*).
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
- **A count that does not match the file** (NMT1 against the sub-sections of a COVFIL
  section, the two NW of a row, NVAL against the values a BOXER card group loads) is
  trusted as written and reported.
- **A row number outside the block.** A COVFIL row numbered 0 is placed on row 1, as the
  field's blank default; any other row or band outside the block makes the reader fail.
  Both are reported first.
- **The same pair of parameters stated twice**, in either orientation: both blocks are
  kept, `covariance_matrix` places the last one, and the repetition is reported.
- **An unknown MT in COVERX** (one not in kika's reaction table, such as a lumped MT 851)
  is skipped and reported; BOXER and COVFIL keep every MT.

## Refused

- A COVERX binary record whose leading and trailing lengths differ, or a first record
  whose length gives no plausible byte order (`InvalidDataFormatError`): the file is
  corrupt or not COVERX, and every later offset would be wrong.
- A COVERX binary block whose NBLOK blocks do not add up to NGROUP rows
  (`InvalidDataFormatError`): the rows of each data record cannot be placed.
- A COVERX text file with a grid of neither NGROUP + 1 boundaries nor NGROUP upper bounds,
  NGROUP upper bounds that match no structure of `kika.energy_grids`, or a boundary that
  is not a number (`InvalidDataFormatError`): a lowest boundary kika would have to invent
  changes the width of the last group.
- A correlation matrix (BOXER ITYPE 4, COVERX NTYPE 3) without the standard deviations of
  its row or its column parameter anywhere in the file (`InvalidDataFormatError`): a
  correlation cannot be turned into a covariance without them, and storing it as one
  would be wrong by the square of the deviations.
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
  GENDF), and ERRORR output carrying MF 35: kika has no reader for them, and
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

Binary values are float32, about seven significant figures, and the writer warns. Every
block is written relative (NTYPE 2): an absolute block is divided by the outer product of
the cross-sections of its two parameters, and refused when one is missing or zero. The
text writer writes the header line as (G, 2, NMMP, NMTRIX) in I12 fields, the $G + 1$
boundaries and each dense block in E15.7, five per line, and the block header as five I12
fields ending in 1; it warns that cross-sections are lost. SCALE's own text libraries give
the $G$ upper bounds and put NMTRIX third, so a file kika writes is laid out differently
from them; kika reads both.

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
| COVERX binary | Blocks, grid and cross-sections to float32 (relative difference within $2 \cdot 10^{-7}$ in the test); every block comes back flagged relative. | Title, MWGT, deviations, unknown MTs; an absolute block comes back relative. |
| COVERX text | Blocks and grid to E15.7. | Cross-sections, title, unknown MTs, the relative flag. |
| BOXER (`nvf=14`) | Blocks, grid and cross-sections, equal to $10^{-12}$ in the test. | HLIB and HDESCR, the relative flag; the lower triangle of a diagonal block that is not exactly symmetric (the upper one is kept and mirrored). With the default `nvf=10`, values keep four significant figures. |

An all-zero block does not come back, since no reader stores one. COVERX keeps the model's
isotope identifiers while BOXER and COVFIL translate them to MAT and back, which is the
identity only for nuclides in kika's MAT table. COVERX identifiers are the file's own: the
SCALE libraries name a nuclide by a SCALE identifier (8001001 for O-16), which kika keeps
as it is.

## Report

The readers return the covariance object with no report attached. Each takes an optional
`diagnostics` list, to which it appends one
{py:class}`~kika.cov.parse_covmat.ReadEvent` per thing it saw and did not keep as it was;
the object returned is the same with or without the list. An event has a `kind`, a
one-line `message`, a `where` with the indices the file uses (MAT, MT, MAT1, MT1, $L$,
$L_1$, a 1-based row) and an `evidence` with the values behind it:

| Kind | Meaning | Worry? |
|---|---|---|
| `declared` | A header as the file writes it (NTYPE, MWGT, the ERRORR HEAD, the BOXER library name). | No: it is the record of what the file says. |
| `unit_assumed` | The file does not say relative or absolute; the flag kika set. | When the magnitude of the uncertainties matters, check the producer's settings. |
| `reaction_not_stated` | An ERRORR MF 34 section, whose reaction is not in the file. | Know which reaction ERRORR was run on. |
| `dropped_block` | A block left out: all zero, an MT kika does not know, or a BOXER matrix that ends before its last row. | Yes for an unknown MT or an unfinished matrix. |
| `dropped_values` | Values read and not kept (all-zero cross-sections). | No. |
| `count_mismatch` | A declared count differs from what the file holds. | Yes: the file is not what its header says. |
| `count_defaulted` | A blank count, or the lowest boundary of a SCALE text grid, supplied by kika. | Read its `evidence` to see what was supplied. |
| `record_longer` | A binary record longer than its counts need. | Yes: the counts and the data disagree. |
| `index_out_of_range` | A row, column or band outside the block. | Yes. |
| `row_order` | Rows of a COVFIL block out of order or given twice. | Yes: a repeated row overwrites the first. |
| `repeated_block` | The same pair of parameters stated twice. | Yes: only the last one is assembled. |
| `skipped_line` | A COVERX text block header whose block could not be read. | Yes: the block is lost. |
| `raw_stddev`, `raw_correlation` | Deviations and correlations as written, before kika multiplies them. | No: they are evidence for checks that compare them with the covariance. |

A file written by ERRORR or COVR, or built record by record from the COVERX format, gives
none of `count_mismatch`, `count_defaulted`, `record_longer`, `index_out_of_range`,
`row_order`, `repeated_block` or `skipped_line`. What a writer cannot carry is said as a
Python `UserWarning` at write time (float32 precision, lost cross-sections, MeV converted to
eV, a BOXER value format of ten columns or fewer). What it cannot write correctly at all
(a MAT number too wide for its field, a second material in a COVFIL) is refused, under
*Refused*.

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
- **Against files NJOY wrote**, kept under `kika/cov/tests/data/njoy2016`: two ERRORR
  outputs of the NJOY 2016.78 test suite (test 04, MF 33 of U-235 from ENDF/B-V; test 15,
  MF 34 of U-238) and two BOXER files that COVR (NJOY 2016.78) wrote from the ERRORR tape
  of test 15, MT 2 and MT 64 of U-238 in 30 groups, once as covariances and once as
  correlations. The ERRORR MF 34 section is read with the flag of its `irelco`, also when
  it is 0, and the U-235 tape takes ZA 92235 for a MAT kika's table does not have
  (`kika/cov/tests/test_covfil_reads_errorr_as_written.py`). The two BOXER files give the
  same covariances within the rounding of the two files, $5 \cdot 10^{-5}\, s_i s_j$ for
  $\rho$ written to four decimals plus $1.5 \cdot 10^{-3}$ of the value for the four
  figures of the covariance and of each deviation
  (`kika/cov/tests/test_boxer_as_covr_writes_it.py::test_correlations_give_the_covariances_within_the_rounding_of_the_two_files`).
- **COVERX binary built from the format**, record by record from Drischler (1980), not by
  kika's writer: NTYPE 1, 2 and 3 give absolute, relative and converted correlation
  blocks; a matrix split in two NBLOK blocks comes back whole; blocks that do not add up
  to NGROUP are refused (`kika/cov/tests/test_coverx_binary_by_the_format.py`).
- **SCALE's text libraries**: a grid of SCALE's 44 upper bounds takes its lowest boundary
  from SCALE44, and upper bounds of no structure are refused
  (`kika/cov/tests/test_coverx_text_as_scale_writes_it.py`). The libraries themselves
  (`scale.rev05.44groupcov`, `scale.rev08.56groupcov7.1`, text) are read whole, 2 558 and
  4 563 blocks with none dropped
  (`::test_the_scale_text_libraries_are_read_whole`, skipped unless
  `KIKA_TEST_SCALE_TEXT_DIR` names their folder; passed on 2026-11-25).
- **Read events**: reading with and without `diagnostics` gives the same object, each kind
  of event has a test on an edited copy of a real file or a file built from the format,
  and the files NJOY wrote give no structure event
  (`kika/cov/tests/test_readers_say_what_they_dropped.py`).

There is no oracle check against SCALE or NJOY reading a file kika wrote, and no binary
COVERX library from SCALE has been read.

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
- J. D. Drischler, *The COVERX Service Module of the FORSS System*, ORNL/TM-7181
  (ENDF-291), Oak Ridge National Laboratory (1980): appendix A, the COVERX format.
  TODO(check): the author's initials (the scanned cover is not legible).
- W. Wieselquist and R. A. Lefebvre (eds.), *SCALE 6.3.1 User Manual*, ORNL/TM-SCALE-6.3.1,
  Oak Ridge National Laboratory (2023), doi:10.2172/1959594: the SCALE covariance libraries.
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: ERRORR and COVR, the COVFIL and BOXER formats.
