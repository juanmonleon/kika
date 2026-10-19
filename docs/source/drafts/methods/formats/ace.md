# Reading and writing ACE

```{admonition} At a glance
:class: note
**Reads:** continuous-energy neutron ACE tables (legacy and 2.0.1 headers) block by
block, and photo-atomic tables.

**Writes:** an ACE table from its parsed object and its `xsdir` line; a neutron table
from a processed model at one temperature, so from kika's own processing of an ENDF or
GNDS evaluation, from an OpenMC file and from each sample of a perturbation run, without
NJOY; and a thermal scattering table (`.t`).

**Into:** the parse layer's `Ace` object, and from it the canonical model: cross-sections,
Q-values, the secondary neutrons with their angular and energy distributions and
multiplicities, fission $\bar\nu$ and the delayed-neutron groups, photon production, the
light charged particles and the unresolved-range probability tables (see
{doc}`../../architecture/format-adapters`).

**Entry points:** {py:func}`kika.ace.parsers.parse_ace.read_ace`, {py:func}`kika.read`,
{py:func}`kika.write` (`format="ace"`), {py:func}`kika.ace.model_adapter.decodeAce`,
{py:func}`kika.ace.model_adapter.encodeAce`, {py:meth}`kika.endf.EndfProcessing.write_ace`
({py:func}`kika.endf.process_endf`, {py:func}`kika.gnds.process_gnds`),
`AceOptions(engine="kika")` of {py:func}`kika.sampling.model_perturbation.perturbFromModel`,
{py:func}`kika.tsl.ace_writer.write_ace_thermal`, {py:func}`kika.ace.writers.write_ace.write_ace`,
{py:func}`kika.ace.xsdir.add_xsdir_entry`.

**Status:** stable (parse layer); experimental (model path).
```

## Background

ACE (A Compact ENDF) is the format of continuous-energy Monte Carlo codes such as MCNP and
Serpent. A table is processed data, written by NJOY's ACER: the resonance region is
already reconstructed, every cross-section is on one unionised energy grid, linearly
interpolated, at one temperature (Conlin and Romano 2019). After a
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
| PTYPE, NTRO, IXS and the per-particle blocks (MTRH, TYRH, SIGH, ANDH, DLWH, HPD, YH) | read; an MT listed twice in MTRH keeps both entries, each with its own records | one product per particle and MTRH entry on its reaction (protons, deuterons, tritons, He-3, alphas): yield, frame and sampled distribution; HPD kept beside the suite; another particle type reported, not converted |

A photo-atomic table (ZAID suffix ending in `p`) is parsed on its own, since its NXS and
JXS mean other things, and returned as a `PhotoatomicAce`. It has no model path.

A thermal-scattering table (`.NNt`, the ITIE, ITCE, ITXE and ITCA blocks THERMR and ACER
write) is not a neutron table and is not read by `read_ace`. `kika.read` recognises it by
its name field (`h-h2o.80t`, `grph.72t`) and reads it, as OpenMC reads it, into a
processed thermal-scattering suite: the inelastic cross-section and, for the continuous
representation, its outgoing-energy tables with equiprobable cosines; the coherent elastic
Bragg edges (MeV·b converted to eV·b) or the incoherent elastic cross-section and cosines.
See {doc}`hdf5` for the processed forms and
{doc}`../../user_guide/thermal-scattering-libraries` for reading thermal libraries.

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

The light charged particles become products of their reactions, one per entry of the
particle's MTRH list: the yield from SIGH (MFTYPE 12 on its own reaction, the only form
ACER writes there), the frame from TYRH (centre of mass when negative), and the sampled
distribution from DLWH and ANDH, read as the neutron laws are (law 33, the charged-particle
level law, is a level as law 3 is). An MT can appear twice in MTRH (He-3's MT 650,
$n + {}^3\text{He} \to d + d$): each entry keeps its own records, and the second product
is labelled `H2__a`, as FUDGE names a second product of the same particle. HPD (each
particle's production cross-section and heating number) is not a form of any product: it
is kept as the table states it, for the encoder to give back while the cross-sections and
yields it is made of are unchanged.

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
  any other interpolation is written as the form states it, so the incident-energy
  interpolation of an energy law keeps the evaluation's unit-base code (INT = 22) where
  ACER writes NR = 0. It is metadata for the transport code: OpenMC 0.16 ignores the
  incident interpolation of a law and always samples between the two bracketing incident
  tables with unit-base scaling. TODO(check): how MCNP and Serpent use a stated INT;
- an angular distribution is written in one of the three forms AND holds (isotropic, 32
  equiprobable bins, tabulated): ACE has no Legendre form, so a Legendre distribution
  reaches the encoder already tabulated ({doc}`../processing/tabulation`);
- an MFTYPE 13 photon cross-section is its multiplicity times its reaction's
  cross-section, on the union grid;
- a photon's cosines are isotropic or 32 equiprobable bins, the only forms ANDP holds:
  any other cosine table is cut into 32 bins of equal probability by inverting its own
  CDF exactly (linear in a histogram panel, quadratic in a lin-lin one), as ACER does
  with MF14.

What ACE states and the model has no slot for is kept beside the decoded nodes and used
when the same table is written back: the order of MTR and YP (MT 4 goes last when a photon
yield needs it), which law of two that share a model form a chain used (3 or 33, 1 or 4),
the sign of law 61's cosine-table locators, the interpolation code of an outgoing-energy
table that is only lines, law 66's tabulated part, each precursor family's probability
table, each photon's MTRP number and MFTYPE, the probability of a one-law chain, and NXS(2)
and NXS(9-16). A suite that does not come from ACE follows NJOY's rules instead (ACER's choice of what
survives its union grid):

- reactions with secondary neutrons first and the rest by MT;
- of the sums, MT 103-107 and MT 18, or the fission chances instead of MT 18 when MT 19
  has distributions; MT 4 (and MT 3) only when a photon yield is stated on it or the
  unresolved-range tables compete with it (ILF = 4), written last;
- each reaction starts one point before its first nonzero value, so a gas production is
  not written from MT 5's threshold;
- photons numbered $1000\,\mathrm{MT} + i$, with the MFTYPE of the file they were stated
  in: 12 for MF12, 13 for MF13 (written as a production cross-section), 16 for MF6. A
  suite from ENDF says which file through its bookkeeping and one from GNDS through
  FUDGE's conversion flags; a suite that says neither (an OpenMC file, a GNDS file with
  no flags) writes yields, MFTYPE 16, on their reaction, which MCNP reads the same way;
- the delayed $\bar\nu$ is MF1/455's; a suite that states only each delayed family's
  yield (an OpenMC file) writes their sum, and says so;
- ESZ's heating number is NJOY HEATR's KERMA with photons deposited elsewhere
  (`local = 0`), which is what MCNP's tables carry. A processed suite's MT 301 is OpenMC's,
  whose fission part adds the delayed betas' energy to the fragments'; the encoder takes
  $E_B\,\sigma_f$ out again. MT 901, the KERMA with local photons, has no ACE place and
  is reported lost;
- a probability of 1 over the law's domain; Z and A in NXS(10) and NXS(11); an isomer's
  ZAID is $1000\,Z + A + 300 + 100\,m$ (Am-242m1 is 95642), as LANL writes it for MCNP;
- the header carries the date, a title naming the target and the evaluation, and the
  tape's MAT;
- the light charged particles, as NJOY 2016's ACER writes them, when the suite carries
  them as products with a sampled form: PTYPE 9, 31, 32, 33, 34 (proton, deuteron,
  triton, He-3, alpha) in that order; per particle, MTRH lists the producing reactions in
  ascending MT, one entry per product (two deuterons of MT 650 are two entries), TYRH is
  $\pm 1$ by frame ($-1$ in the centre of mass), SIGH each yield as MFTYPE 12 on its own
  reaction, ANDH and DLWH the sampled form (a level's particle as law 33), YH the
  reactions with a yield, and HPD the production cross-section
  $\sum_r y_r\,\sigma_r$ on the union grid, from one point before its first nonzero value,
  with the heating number $\sum_r y_r\,\sigma_r\,\bar E_r / \sigma_t$ in MeV, $\bar E_r$
  the particle's mean emitted energy. A suite read from ACE gets its own HPD back while
  the cross-sections and yields are unchanged, and a rebuilt one, with a report entry,
  once they change. A suite with no light charged products (one read from OpenMC) writes
  no particle blocks, and the report says so.

**The ZAID suffix.** A table kika writes from a suite that does not come from ACE takes
the suffix of its temperature, the one kika's NJOY runs give (`.02c` at 293.6 K, `.06c`
at 600 K; the temperature is matched within 1 K). A table from kika's processing then
takes the place and the name of the one NJOY made, so inputs and `xsdir` files that name
it do not change; the library is named by the file and the `xsdir` line, not by the ZAID.
A temperature with no suffix raises, and an explicit `extension` is taken as given. The
same rule names the tables kika's NJOY runs write when no extension is given. The header
is the legacy one, which MCNP5, MCNP6 and Serpent read.

**From an evaluation.** `process_endf(path, plan).write_ace(directory)` (and the same
from `process_gnds`) writes one table per broadened temperature: the processed suite
`write_hdf5` writes (this run's broadening and probability tables, the distributions
tabulated for Monte Carlo, KERMA, damage and gas production) through `encodeAce`. Each
table is `<ZAID><suffix>` (`8016.02c`), written to a temporary file, read back in strict
mode and compared with what was written (integer words equal, reals to the 12 digits of
E20.11) before it replaces a file, and its line is added to `directory/xsdir`, which
refuses a table name it already has unless `replace=True` (MCNP would read only one of
the two). A refusal is a {py:class}`~kika.endf.ProcessingOutputError` with a category:
`ace-needs-temperature`, `ace-temperature-not-processed`, `ace-table-exists`,
`ace-failed`. NJOY is not run.

**Through `kika.write`.** `kika.write(suite, path, format="ace", temperatures=T)` writes
a processed suite: one read from an ACE or OpenMC file, or
`kika.hdf5.processing.processSuite`'s. An evaluation is refused (`ace-needs-processed-suite`):
an ACE table is one temperature of a processing run. A thermal scattering suite is
written as a `.t` table (below).

**For each sample of a perturbation run.** `AceOptions(engine="kika")` processes each
realisation from its 0 K forms (multiplied at 0 K, then broadened) and writes it with
`encodeAce`, with no tape and no NJOY; the probability tables are computed once, on the
unperturbed evaluation, and shared by every sample. An unperturbed sample is the table
`write_ace` writes.

**Thermal tables.** `kika.tsl.ace_writer.write_ace_thermal` writes one temperature of a
processed thermal law in ACER's layout: ITIE (incident energies, the inelastic σ),
ITXE (continuous outgoing tables: NE locators, each one word before its block, NE counts,
then $E'$, pdf, cdf and the equiprobable cosines per outgoing energy), and the elastic
part at ITCE: the Bragg edges and cumulative structure factors of a coherent one, σ(E)
and its equiprobable cosines for an incoherent one, both for a mixed one (the incoherent
half at JXS(7-9)). The σ's and factors are per molecule (divided by `nmix`). From a
suite of kika's thermal processing (`kika.hdf5.thermal_processing.processThermal`) or
an OpenMC thermal file, the incoherent elastic, which those state as $\sigma_b$ and $W'$,
is written as THERMR computes it: $\sigma(E) = \tfrac{\sigma_b}{2}\,(1 - e^{-2c})/c$
with $c = 2EW'$, on the inelastic incident grid, with the mean cosine of each of 16
equiprobable bins of $p(\mu) \propto e^{-c(1-\mu)}$. NJOY's own table interpolates
that σ, written to 7 digits on the free-atom PENDF grid, back to its energies; kika
evaluates it where it is written.

**Many variants of one table.** A perturbed sample differs from its source table in a
small part of XSS. Since
every XSS value has a fixed place (four 20-column fields per line), `AcePatchWriter` writes
the unperturbed table once and each sample as that text with only the changed lines
re-formatted, with the format `write_ace` uses. The output is byte-identical to
`write_ace`, and any case in which `write_ace` would lay the file out differently (another
header, another XSS length, the integer format switching on or off) falls back to
`write_ace` itself.

## What the round trip keeps and loses

- **Parse layer, read and write:** the file comes back byte for byte.
- **ACE → model → ACE:** every integer word of the neutron and photon blocks comes back
  equal and every real word within a few units in the last place (energies and
  densities pass through eV and back). The charged-particle blocks come back with them
  (HPD as the table stated it). Not written: NJOY's unused room after some law tables,
  so the XSS is shorter by that room.
- **Model:** the evaluation behind the table is not in the file and is reported as lost;
  QM and LR have no ACE counterpart (reported); ACE has no covariances (reported).

## Report

Entries a user meets after `kika.read` of an ACE file: the absent evaluation (a loss, on
every table); no LQR block (a loss); a reaction with no Q (a loss, normal for sums); QM,
LR and covariances (unsupported, on every table); energy laws without a model form (laws
22, 24, 67); photon entries without a model form; MFTYPE 13 photons stated as
multiplicities, and orphan photons (approximations); a charged particle of a type other
than the five light ones, or an MTRH entry kika cannot hang on a reaction (unsupported or
lost, stating that cross-sections are unaffected). A block absent from the table produces
no entry. `encodeAce` adds the loss of a Q-value it writes as 0, the sums it derives
because the suite has none, the charged-particle blocks it does not write because the
suite carries no light charged products, an HPD rebuilt because its yields or
cross-sections changed, and a particle's mean energy it could not compute (its heating
counted as 0).

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
  The encoder reports, and does not write, particle blocks for a suite that carries no
  light charged products
  (`::test_the_charged_particle_blocks_are_reported_not_written`).
- Light charged particles, on the O-16 table of ENDF/B-VIII.1 that kika's processing
  writes (`kika/ace/model_adapter/tests/test_ace_particles.py`): protons and alphas are
  written (`::test_protons_and_alphas_are_written`), each particle's production is
  $\sum y\,\sigma$ of the reactions its MTRH names
  (`::test_each_particle_is_produced_by_the_reactions_mtrh_names`), and a level
  reaction's particle is law 33 (`::test_a_level_reactions_particle_is_law_33`); read
  back, the particles are products and the table comes back to $10^{-12}$ relative,
  word for word in its counts and locators, while an edited yield rebuilds the HPD
  (`test_ace_particles_read_back.py`); these need the O-16 tape. An MT repeated in MTRH
  keeps both entries through the parser and the encoders, on a synthetic table
  (`kika/ace/tests/test_repeated_ixs_mt.py`), and through the model, on the synthetic
  table and, when the tape is present, on He-3's MT 650 of ENDF/B-VIII.1
  (`kika/ace/model_adapter/tests/test_ace_repeated_ixs_mt.py`).
- Every unconverted block is named and the conversion stays clean for cross-sections; an
  absent block says nothing
  (`kika/ace/model_adapter/tests/test_ace_decode_reports_what_it_drops.py`).
- Patched samples are byte-identical to `write_ace` and a layout change falls back to it
  (`kika/ace/tests/test_patch_writer_matches_write_ace.py::test_patched_samples_are_byte_identical`,
  `::test_a_layout_change_falls_back_to_write_ace`).
- A thermal table reads into the processed forms and encodes as OpenMC lays them out
  (`kika/hdf5/tests/test_ace_thermal.py::test_the_table_reads_into_the_processed_forms`,
  `::test_the_forms_encode_as_openmc_lays_them_out`), and `kika.read` returns its suite
  (`kika/tsl/tests/test_tsl_formats.py::test_an_ace_thermal_table_is_read_into_the_model`).
- Processing → ACE: one table per temperature named by ZAID and suffix, its `xsdir`
  line, the refusal of a repeated table, the refusals by category, and `kika.write` of a
  read table giving the same file
  (`kika/ace/model_adapter/tests/test_ace_output.py`); a photon cosine table cut into 32
  equiprobable bins, each carrying 1/32 (`::test_a_photon_cosine_table_is_cut_into_32_equiprobable_bins`);
  an isomer's ZAID (`::test_an_isomer_takes_lanls_zaid`); the MFTYPE of each photon
  (`::test_a_photon_keeps_the_file_it_was_stated_in`).
- Conversions: an OpenMC file to ACE and back with the same cross-sections, the delayed
  $\bar\nu$ of a fissile, the heating number, MT 4 for the unresolved-range competition,
  OpenMC thermal files to `.t`, THERMR's incoherent cosines, and FUDGE's photon flags
  (`kika/ace/model_adapter/tests/test_ace_conversions.py`).
- Perturbation samples without NJOY, and the unperturbed processing equal to
  `write_ace`'s table word for word
  (`kika/sampling/tests/test_ace_kika_engine.py`).
- Thermal tables: each elastic mode written and read back to a fixed point, the locator
  convention, and, with `KIKA_ACE_SAB` pointing to a directory of thermal tables, each
  real table recomposed byte for byte (`kika/tsl/tests/test_ace_thermal_writer.py`).
- `TODO(check)`: the comparison of kika's tables with NJOY's of the same tape and
  temperature is a development check, not a public test.
- `kika/ace/comparison` has no tests.

## Implementation

- `kika/ace/parsers/`: `parse_ace.py` (`read_ace`) and one parser per block.
- `kika/ace/classes/`: `Ace`, `Header`, the read report, and one class per block.
- `kika/ace/encoders/`: one encoder per block, the energy laws, and the layout that places
  the blocks and fills NXS and JXS; `kika/ace/contract.py`: what a table must satisfy to
  be laid out.
- `kika/ace/writers/write_ace.py`: writing, and `write_ace_checked`, which reads the file
  back before it replaces one; `kika/ace/xsdir.py`: `xsdir` lines (`add_xsdir_entry`).
- `kika/endf/process.py`: `EndfProcessing.write_ace`; `kika/_write.py`: the `ace` door;
  `kika/sampling/model_perturbation.py`: `AceOptions(engine="kika")`.
- `kika/tsl/ace_writer.py`: thermal tables (`write_ace_thermal`, `write_thermal_suite`,
  THERMR's incoherent elastic).
- `kika/ace/model_adapter/decode.py`: `decodeAce`, the styles and labels, Q-values, the
  probability tables and where each reaction goes; `secondaries.py`: neutron products,
  fission and delayed neutrons, the light charged particles; `photons.py`: photon
  products; `encode.py`: `encodeAce`, the charged-particle blocks included;
  `kika/ace/classes/ixs_keys.py`: the keys of the IXS entries of a repeated MT.
- `kika/ace/comparison/compare_ace.py`: block-by-block comparison of two tables.

## References

- J. L. Conlin (ed.) and P. Romano, *A Compact ENDF (ACE) Format Specification*,
  LA-UR-19-29016, Los Alamos National Laboratory (2019), doi:10.2172/1561065.
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: ACER.
