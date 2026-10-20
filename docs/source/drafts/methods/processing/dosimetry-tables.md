# Writing dosimetry tables

```{admonition} At a glance
:class: note
**Computes:** the reaction cross sections of an evaluation as a dosimetry ACE (A Compact ENDF) table, for reaction-rate tallies.
**Applies to:** ENDF-6 (Evaluated Nuclear Data File) neutron evaluations: MF 3, MF 9 and MF 10, and the KERMA, damage and gas production kika derives.
**Entry points:** {py:meth}`kika.endf.EndfProcessing.write_dosimetry_ace`, {py:func}`kika.endf.dosimetry_ace`, {py:class}`kika.ace.classes.dosimetry.DosimetryAce`.
**Status:** experimental.
```

## Background

A dosimetry table (class letter `y`, `26056.02y`) is the ACE format's container for
reaction cross sections that a transport code folds with a flux and never transports:
activation foils, damage and gas production. It has no shared energy grid and no
secondary distributions. Each reaction carries its own energies and its own
interpolation table, so a cross section can be written exactly as the evaluation
tabulates it.

The layout is the one NJOY2016's ACER writes with `iopt = 3` (`acedo.f90`):

- `NXS(1)` is the length of `XSS`, `NXS(2)` the target's ZA and `NXS(4)` the number of
  reactions.
- `JXS(1)` and `JXS(3)` point at the reaction numbers (MTR), `JXS(6)` at their locators
  (LSIG), `JXS(7)` at the data (SIGD) and `JXS(22)` at the end of the table.
- Each reaction's data is a TAB1 without its CONT record: NR, the breakpoints NBT(1..NR)
  and the laws INT(1..NR) as two arrays, NE, the energies in MeV and the cross sections.
  A single lin-lin region is written as NR = 0.

A production cross section of MF 9 or MF 10 (ENDF-6 §9, §10), to the final state LFS of
a product of charge and mass number $ZA = 1000Z + A$, is numbered as the IRDFF-II ACE
library numbers it:

$$
N = \begin{cases}
(50 + \mathrm{LFS})\cdot 10^6 + ZA & \text{MT} = 5,\\
\mathrm{MT} + 1000\,(10 + \mathrm{LFS}) & \text{otherwise.}
\end{cases}
$$ (eq-dosimetry-tables-number)

MT 4 to the first isomer is 11004, as ACER and FRENDY number it too. MT 5 is the one
reaction with many products, so its products are told apart by what they are: Mg(n,x)Na-24
is 50011024.

## What it computes

From a processed evaluation at temperature $T$, one table with the reactions below, in
this order:

1. Every cross section with an MT that the processing tabulated at $T$, sums included
   (MT 1, 3, 4, 103-107 and the others): reconstructed, Doppler-broadened and thinned,
   lin-lin, one region.
2. A level sum the evaluation does not state, added up from its levels:

   $$
   \sigma_{\Sigma}(E) = \sum_{l \in L(\Sigma)} \sigma_l(E),
   $$ (eq-dosimetry-tables-levels)

   with $L(103) = \{600,\dots,649\}$, $L(104) = \{650,\dots,699\}$ up to
   $L(107) = \{800,\dots,849\}$, $L(16) = \{875,\dots,891\}$ and $L(4) = \{51,\dots,91\}$,
   on the union of the levels' grids. Fe-56 (n,p), the dosimetry reaction MT 103, is one
   of these: ENDF/B-VIII.1 states only its levels.
3. MT 301 and MT 444 (KERMA and damage energy, in eV·b) when the heating stage ran, and
   the gas production MT 203-207, computed as for the continuous-energy table (see
   {doc}`heating` and {doc}`processing-chain`).
4. The production cross sections, numbered by {eq}`eq-dosimetry-tables-number`:
   - MF 10 as the evaluation tabulates it, with its own interpolation laws;
   - MF 9 as the multiplicity times the reaction's processed cross section,
     $\sigma_{\mathrm{MT}\to\mathrm{LFS}}(E) = y(E)\,\sigma_{\mathrm{MT}}(E)$, tabulated
     lin-lin as described under *Method*.

From an evaluation stated pointwise (no resonance parameters to reconstruct), the table
holds every MF 3 cross section and every MF 9 and MF 10 production as the evaluation
tabulates them, at the evaluation's own temperature, each with its own interpolation
table. Nothing is reconstructed, linearised or broadened.

## Method

The table is assembled from the processed suite at one temperature.

1. **Cross sections.** Each processed table is copied as it is. A cross section that is
   zero everywhere is not written.
2. **Level sums.** For each sum MT of {eq}`eq-dosimetry-tables-levels` that the suite
   lacks and whose levels it has, the levels are interpolated lin-lin onto the union of
   their grids, zero outside each level's own range, and added. The table starts one node
   below the first non-zero value.
3. **Heating and gas.** MT 301 and MT 444 are the heating stage's tables at $T$. The gas
   production is $\sum_r k_r(E)\,\sigma_r(E)$ over the reactions on the temperature's
   union grid, where $k_r$ is the number of light particles of the gas's kind that
   reaction $r$ emits (its products' multiplicities, or the number its MT fixes when it
   states none). This is the computation of {doc}`processing-chain`.
4. **MF 9 products.** On the union of the cross section's nodes and the multiplicity's
   nodes, the product is tabulated and every interval is tested at its midpoint:

   $$
   \left|\tfrac{1}{2}\big(p(E_i) + p(E_{i+1})\big) - y(\bar E)\,\sigma(\bar E)\right|
   \le 10^{-3}\,|y(\bar E)\,\sigma(\bar E)|,\qquad \bar E = \tfrac{1}{2}(E_i + E_{i+1}).
   $$ (eq-dosimetry-tables-product)

   An interval that fails is split at its midpoint, up to 40 times. A step of either
   factor stays a step: the product is tabulated piece by piece between the steps (a node
   written twice, or an edge of the multiplicity's domain), each piece evaluated from
   inside it, and the pieces meet at a repeated node.
5. **Numbers that collide.** A number of {eq}`eq-dosimetry-tables-number` that more than
   one product shares (two products of one MT other than 5 and one LFS, which no
   evaluation known states) is not written, and each such product is named in the omitted
   list.
6. **Encoding.** The reactions are laid out in the order above. Counts, locators,
   reaction numbers and interpolation tables are integer words (I20), energies (in MeV)
   and cross sections are reals (E20.11). The header follows {doc}`../formats/ace`: ZAID
   $1000Z + A$ (an isomer $+\,300 + 100m$), the suffix of the temperature with the letter
   `y` (`.02y` at 293.6 K, `.01y` at 0 K), the temperature as $kT$ in MeV.
7. **Writing.** The table is written through a temporary file, read back in strict mode
   and compared word for word before it replaces an existing one. Its line is added to the
   `xsdir` of the directory.

## Conventions and choices

- **MT 1 is written.** ACER and FRENDY skip it. A total-collision tally is as legitimate
  as any other, and the table costs nothing more.
- **Level sums are written when the evaluation does not state them.** RECONR adds them to
  the PENDF that ACER copies, so NJOY's table has them. A table without MT 103 for Fe-56
  would leave out the dosimetry reaction itself.
- **MF 9 productions are written.** ACER reads only MF 10. An MF 9 production is the
  same physical quantity under another ENDF spelling (a yield instead of a cross section),
  and Al-27 of ENDF/B-VIII.1 states its activation products that way.
- **MF 10 keeps its own laws.** The format carries them, and copying them is exact, which
  is what ACER does. FRENDY refuses any law but lin-lin. NJOY's PENDF holds MF 10
  linearised on the reconstruction grid; both tables describe the same function.
- **MT 5 products are numbered by their product.** NJOY2016 writes every MT 5 product of
  one LFS under one number (Cr-52 of ENDF/B-VIII.1 has 24 under 10005), and FRENDY
  declines to process the evaluation. A code that looks a number up finds one of them. The
  IRDFF-II ACE library numbers them by product instead, and kika follows it: Ti, Fe, Ni,
  Cu and Zn of IRDFF-II state nothing but MT 5 products.
- **Colliding numbers are not written.** Should two products still share a number, a
  table that carries both is ambiguous, and kika writes neither.
- **A pointwise evaluation is not broadened.** Asking for a temperature other than the
  tape's would mean broadening data that is a reference as it stands. No processor does
  that for dosimetry. The pointwise entry point writes the tape's own temperature.

## Limits and failure modes

- {py:meth}`~kika.endf.EndfProcessing.write_dosimetry_ace` raises
  {py:class}`~kika.endf.ProcessingOutputError` with the categories of the
  continuous-energy writer:
  - `ace-needs-temperature` when nothing was broadened;
  - `ace-temperature-not-processed` for a temperature the run did not broaden;
  - `ace-table-exists` when the `xsdir` already names the table, unless `replace=True`;
  - `ace-failed` when the table could not be written or did not read back as written.
- {py:func}`~kika.endf.dosimetry_ace` raises `ace-failed`:
  - for a tape with resolved or unresolved resonance parameters, which needs the
    processing first;
  - for a temperature that has no suffix, unless `extension` is given.
- The decoder ({py:func}`kika.ace.read_ace`) refuses a table whose locators do not place
  each reaction's data, whose interpolation table does not cover its points, or whose
  energies decrease. A table written by NJOY2016 with an MF 10 reaction of more than one
  region is refused: ACER writes those breakpoints and laws interleaved, and they cannot
  be told apart. In ENDF/B-VIII.1 this is Ta-180m1, MT 4, LFS 0.

## Validation

- Format: tables built from known reactions come back word for word through the writer
  and the reader, with NJOY's layout
  (`kika/ace/tests/test_dosimetry_ace.py::test_layout_is_dosout`,
  `::test_round_trip_through_the_file`); the interleaved spelling is refused
  (`::test_interleaved_nbt_int_is_refused`); tables written by NJOY2016 ACER come back
  byte for byte, read and written and re-encoded from their reactions
  (`::test_njoy_tables_are_written_back_byte_for_byte`, run when the tables are present).
- Content: every processed cross section comes out as processed, and a missing level sum
  is the sum of its levels
  (`kika/endf/tests/test_dosimetry_tables.py::test_processed_cross_sections_come_out_as_processed`,
  `::test_a_level_sum_the_tape_leaves_out_is_added`); a pointwise tape is written as
  stated, and its MF 9 products meet {eq}`eq-dosimetry-tables-product`
  (`::test_a_pointwise_tape_is_written_as_stated`).
- Against NJOY2016.78 ACER `iopt = 3`, RECONR and BROADR with the official cards
  (`err` 0.001, `errmax` 0.01) at 293.6 K, on eleven ENDF/B-VIII.1 evaluations (Al-27,
  Si-29, Cr-52, Fe-56, Fe-58, Co-59, Ni-58, Nb-93, Rh-103, Au-197, U-235), each reaction of
  one table read on the other's grid (measured 2026-10-20):
  - The median relative difference per reaction is 3·10⁻⁷ to 5·10⁻⁶.
  - The largest differences are attributed to NJOY, in four places:
    - capture valleys at up to 1.4 %, where NJOY's `errmax` relaxes its tolerance tenfold;
    - capture resonances above 1 MeV (Cr-52, Si-29), where the official BROADR cards stop
      broadening at 1 MeV;
    - Al-27 elastic between 59 and 150 MeV, where NJOY's PENDF drops the evaluation's
      points;
    - a spurious zero in U-235 MT 4 at 662 eV.
- Reaction rates against a reference: NJOY2016.78 with RECONR and BROADR at 10⁻⁵ and no
  `errmax` relaxation, folded with a U-235 Watt spectrum ($a$ = 0.988 MeV,
  $b$ = 2.249 MeV⁻¹) and with 1/E over 10⁻⁵ eV to 20 MeV, on Nb-93, Co-59, Au-197 and
  Fe-56 (538 rates, measured 2026-10-20):
  - kika is within 1.0·10⁻⁴ of the reference everywhere;
  - NJOY at 0.001 is up to 5.8·10⁻⁴ off (Fe-56).
  - In one rate NJOY is closer than kika: Au-197 MT 16 with 1/E, where kika is 1.0·10⁻⁴
    off and NJOY matches.
- On NJOY's own PENDF (pointwise, LRP = 2), the pointwise entry point gives NJOY's table
  reaction for reaction, MT 1 added (Nb-93: 39 of 39; Rh-103: 89 of 89).
- OpenMC 0.16 reads the tables as `TableType.DOSIMETRY` with the stated number of
  reactions.
- Against the IAEA's ACE library of IRDFF-II (December 2020 update, 70 materials,
  pointwise ENDF-6 at 293.6 K, measured 2026-10-20): every table is written, and all 161
  reactions of the IAEA's tables come out with the same numbers, the same grid and
  laws, and cross sections within 5·10⁻⁷ (the IAEA prints seven digits). kika's tables
  add MT 1 where the tape states it. The test runs Mg, Al-27 and Ti when IRDFF-II is
  present
  (`kika/endf/tests/test_dosimetry_tables.py::test_irdff2_agrees_with_the_iaea_ace_library`).

## Diagnostics

The `write` stage of the run (the last of {py:attr}`kika.endf.EndfProcessing.stages`) lists:

- `files`: the tables written;
- `omitted`: each product not written, with the reason, prefixed by its temperature. A
  shared number means the table cannot tell those products apart; read the products from
  the evaluation instead.

A table read back exposes {py:attr}`kika.ace.classes.dosimetry.DosimetryAce.repeated_numbers`.
It is not empty for an NJOY table that writes several products under one number, and
{py:meth}`~kika.ace.classes.dosimetry.DosimetryAce.reaction` refuses those numbers.

## Implementation

- {py:mod}`kika.ace.model_adapter.dosimetry`: which reactions, the level sums, gas
  production, MF 9 products and the colliding numbers.
- {py:mod}`kika.ace.encoders.dosimetry`: the layout.
- {py:mod}`kika.ace.parsers.parse_dosimetry` and {py:mod}`kika.ace.classes.dosimetry`:
  decoding and the decoded table.
- {py:mod}`kika.endf.process`: the two entry points, header and `xsdir`.

## References

- A. C. Kahler (ed.), *The NJOY Nuclear Data Processing System, Version 2016*,
  LA-UR-17-20093, Los Alamos National Laboratory (2016): ACER, `iopt = 3`.
- ENDF-6 Formats Manual, CSEWG Document ENDF-102, BNL-203218-2018-INRE (2018): §3, §9, §10.
- A. Trkov et al., *IRDFF-II: A New Neutron Metrology Library*, Nuclear Data Sheets 163,
  1-108 (2020).
