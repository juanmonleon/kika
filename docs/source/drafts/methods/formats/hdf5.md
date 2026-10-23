# Reading and writing OpenMC HDF5

```{admonition} At a glance
:class: note
**Reads:** OpenMC's HDF5 incident-neutron files (`data_neutron`) and thermal-scattering
files (`data_thermal`), every temperature they hold, and the `cross_sections.xml` index of
a library.

**Writes:** the same files from the model, either from a suite already in processed form
or from an evaluation, which is processed first; `cross_sections.xml` and an overlay index.

**Into:** a {py:class}`~kika.nuclear_data.model.suite.ReactionSuite` whose styles say it is
processed data: one `heated` style per temperature, the 0 K forms as `recon`, the sampling
tables as `MonteCarlo` (see {doc}`../../architecture/canonical-model`).

**Entry points:** {py:func}`kika.read`, {py:func}`kika.write` (`format="hdf5"`),
{py:func}`kika.hdf5.api.read`, {py:func}`kika.hdf5.api.write`, {py:func}`kika.hdf5.library.open`,
{py:func}`kika.hdf5.library.overlay`, {py:func}`kika.hdf5.contract.checkNeutronFile`,
{py:func}`kika.hdf5.compare.compareNeutron`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

OpenMC's continuous-energy transport reads its nuclear data from one HDF5 file per
nuclide, thermal material or element, listed in a `cross_sections.xml` index (Romano et
al. 2015; the OpenMC documentation, "Nuclear Data File Formats"). The files hold
**processed** data, as ACE files do for MCNP: pointwise cross-sections on one energy grid
per temperature, every temperature of a nuclide in the same file, distributions as tables
with their cumulative distribution ready to be sampled, and response functions (heating,
damage, gas production). Energies are in eV, cross-sections in barns, temperatures are
stored as $kT$ in eV.

The layout of a neutron file, as OpenMC 0.16.0's C++ reads it: one top-level group named
after the nuclide (`Fe56`) with a root `version` attribute of major version 3; `kTs/<T>`
(one dataset per temperature, named `f"{round(kT/k_B)}K"`, so 293.6 K is `294K`);
`energy/<T>`, the grid of each temperature; `reactions/reaction_<MT>`, each with a
`<T>/xs` dataset and its `threshold_idx`, a `redundant` flag, and its `product_<i>`
groups holding a yield and one or more distributions; optionally `urr/<T>` with the
probability tables, `total_nu`, and `fission_energy_release`.

OpenMC is not a dependency of kika, optional or otherwise. Files are read and written with
h5py against what OpenMC's C++ reads, and the names the format fixes are copied, with
their source, from OpenMC 0.16.0.

## Coverage

| Part of a neutron file | What kika does |
|---|---|
| `kTs/<T>`, `energy/<T>` | one `heated_<T>` style per temperature, with $kT$ kept exactly in eV/k |
| `reactions/reaction_<MT>/<T>/xs`, `threshold_idx` | the reaction's cross-section under that temperature's style, an `XYs1d` on a suffix of the grid |
| `reaction_002/0K` and `energy/0K` | the 0 K elastic, as the `recon` style (OpenMC keeps it for resonance up-scattering) |
| `redundant = 0` reactions | `suite.reactions` (the partials the C++ sums into the total) |
| redundant sums (MT 3, 4, 16, 103-107; 18 when the chances are given) | `suite.sums`, as `CrossSectionSum` of the partials the file carries |
| gas production 203-207, KERMA 301 and 901, damage 444 | `suite.productions`: response functions, not reactions (GNDS has no node for them) |
| `product_<i>` yields and distributions | products under `MonteCarlo`: `uncorrelated` (angle tables and an energy law), `correlated`, `kalbach-mann`, `nbody` |
| energy laws `continuous`, `level`, `discrete_photon`, `maxwell`, `evaporation`, `watt`, `madland-nix` | the matching model forms |
| fission: `product_0`, `delayed` neutrons, `total_nu`, `fission_energy_release` | the prompt (or total) neutron, `DelayedNeutron` with its decay rate in 1/s, the total-$\bar\nu$ sum, `FissionEnergyRelease` |
| `urr/<T>` | a `URR_<T>` style with the tables inside the forms of the reactions they apply to |
| reaction `label`, `center_of_mass`, product `emission_mode` | kept in the provenance, written back unchanged |
| thermal files (`data_thermal`): elastic and inelastic per temperature | a `thermalNeutronScatteringLaw` suite, MT 2 and MT 4, distributions per temperature |
| photo-atomic files (`data_photon`) | **refused** by the model reader (the model has no photo-atomic suite); the record layer reads and writes them |
| windowed multipole (`data_wmp`) | refused |

## Reading

**1. The records.** {py:mod}`kika.hdf5.records` reads the file into a tree of groups,
datasets and attributes without interpreting anything; strings stay fixed-length bytes and
attribute types are OpenMC's, so the same tree written back gives the same file.

**2. The gate.** The root `filetype` selects the decoder. A neutron file needs a root
`version` of major 3 and exactly one top-level group.

**3. Temperatures and grids.** Each `kTs/<T>` becomes a `heated_<T>` style whose
temperature is the stored value, in eV/k, kept as written: the C++ names its groups from
it, so a value converted to kelvin and back could name a different group. A reaction's
`xs` at $T$ is the tail of `energy/<T>` from `threshold_idx`; the decoder checks
`threshold_idx + len(xs) == len(energy/<T>)` and that every temperature has a grid.

**4. Reactions and responses.** The partials, sums and response functions are sorted as in
the Coverage table, so that the total the C++ rebuilds from the non-redundant reactions is
the sum of `suite.reactions`.

**5. Distributions.** Every table is a cumulative-distribution table and decodes to
`xs_pdf_cdf1d` nodes under `MonteCarlo`; offsets are not kept, since the writer rebuilds
them from the table lengths.

**6. Fission.** `product_0` is the prompt or total neutron, the delayed neutrons follow
with their decay rates, `total_nu` becomes the multiplicity sum, and
`fission_energy_release` keeps its nine functions; the two OpenMC derives (`q_prompt`,
`q_recoverable`) are kept in the provenance.

## Assumptions and defaults

- **A file refers to an evaluation it does not contain.** The `eval` style is declared and
  empty, as for ACE.
- **A Polynomial function has the domain $[0, \infty)$**, since the file states none; the
  domain is not written back.

## Ambiguous and inconsistent input

- A grid that is not strictly increasing, a `threshold_idx` that does not match the length
  of `xs`, a temperature group with no grid, and `urr/` missing a temperature the file
  has are each an `HDF5FormatError`, not a guess: the C++ assumes all four and fails on
  them at transport time, if at all.
- An index whose `<library>` points a material at a file whose top-level group has
  another name is refused by {py:func}`kika.hdf5.library.overlay`, since the C++ opens the
  group the index names and the mismatch fails only when OpenMC loads it.

## Refused

`HDF5FormatError` (a `ValueError`) for a file that breaks the format's contract (above),
and `HDF5UnsupportedError` (a `NotImplementedError`) for a `filetype` other than
`data_neutron` and `data_thermal`, a temperature or a decay rate in a unit OpenMC does not
store, and, writing, a distribution with no `MonteCarlo` form (OpenMC samples tables
with their cumulative; {doc}`../processing/tabulation` produces them).

## Writing

{py:func}`kika.write` with `format="hdf5"` takes two kinds of suite:

- **already processed** (read from an OpenMC file, or built by
  {py:func}`kika.hdf5.processing.processSuite`): serialised as it is;
- **an evaluation** (from ENDF-6 or GNDS): processed first. kika reconstructs it at 0 K
  unless it has a `recon` style, broadens it to each temperature up to where NJOY BROADR
  stops by default, puts every reaction of a temperature on one grid, thinned so that
  the table delivered stays within $10^{-3}$ ($2\cdot10^{-4}$ below 0.4999 eV), the 0 K
  source, the broadening and the thinning sharing that tolerance
  ({doc}`../processing/processing-chain`), keeps the 0 K elastic in `recon` for
  resonance up-scattering, thinned to half the tolerance (with the same thermal factor),
  rebuilds the sums from their levels when the evaluation does not give them,
  computes gas production (MT 203-207, as GASPR does, {doc}`../processing/gas-production`),
  KERMA and damage (MT 301, 901,
  444, {doc}`../processing/heating`) and the URR probability tables
  ({doc}`../processing/probability-tables`), tabulates every distribution of neutrons and
  photons ({doc}`../processing/tabulation`; the light charged particles, which OpenMC's
  neutron files do not carry, are tabulated only for an ACE table), and derives
  `fission_energy_release` as OpenMC
  does (below). The steps are those of {doc}`../processing/processing-chain`; no NJOY is
  run. A thermal evaluation (MF 7) is processed by
  {py:func}`kika.hdf5.thermal_processing.processThermal`
  ({doc}`../processing/thermal-scattering-tables`).

The encoder enforces, and refuses to break, what OpenMC's C++ assumes:

- group names `f"{round(kT/k_B)}K"`, the same in `kTs`, `energy`, every reaction and `urr`;
  for a processed evaluation, $kT$ is stated as an ACE header states it, in MeV to five
  significant figures, which is what the official libraries hold;
- every reaction at every temperature, and no `reaction_001` (the C++ rebuilds the total);
- `redundant` set on every reaction that is not an exclusive partial and on every response
  function, so the total the C++ rebuilds counts each partial once;
- fission products in the C++'s order: the prompt (or total) neutron, then the delayed
  groups, then the rest;
- `urr/<T>` at every temperature or none, with the same competing inelastic MT everywhere,
  written as a reaction.

**Fission energy release.** The evaluation's MF 1/MT 458 is stored with OpenMC's two
processing steps. A component given as one constant gets the energy dependence ENDF-102
§1.5.2.1 recommends (the Sher-Beck rule): $-0.075\,E$ for the delayed photons and betas,
$-0.105\,E$ for the neutrinos, and for the prompt neutrons
$E_\text{FR,0} + 1.307\,E - 8.07\cdot10^{6}\,(\nu(E) - \nu(0))$ (eV). Then

$$
\begin{aligned}
q_\text{prompt}(E) &= E_\text{FR} + E_\text{NP} + E_\text{GP} - E,\\
q_\text{recoverable}(E) &= E_\text{FR} + E_\text{NP} + E_\text{ND} + E_\text{GP} + E_\text{GD} + E_\text{B} - E,
\end{aligned}
$$ (eq-hdf5-fission-q)

with $E_\text{FR}$ the fragments, $E_\text{NP}$, $E_\text{ND}$ the prompt and delayed
neutrons, $E_\text{GP}$, $E_\text{GD}$ the prompt and delayed photons and $E_\text{B}$
the betas, summed on the union of the tabulated components' grids.

**The file.** It is written under a temporary name in the target directory, read back,
compared record by record with what was meant and decoded again; only then is it moved
into place, and `cross_sections.xml` in the directory is created or updated to index it.
{py:func}`kika.hdf5.library.overlay` writes an index in a new directory that sends some
materials to kika's files and every other material to an existing library where it already
is, with relative paths when they share a drive.

## What the round trip keeps and loses

A file read and written back is identical, record for record (groups, attributes, dtypes,
datasets), for neutron and thermal files alike. What the model has no slot for is kept in
the provenance for that purpose. An edited model is written edited. Byte identity of the
HDF5 container itself is not checked: the comparison is of the tree. An evaluation written
as HDF5 is processed data: reading it back does not give the evaluation.

## Report

The conversion report lists what the processing or the encoder approximated or left out:
data the processing does not carry (for instance photons stated on MT 3 that the ENDF-6
reader keeps verbatim), and the products and laws it tabulated. A
{py:func}`kika.hdf5.contract.checkNeutronFile` run on a file returns the list of contract
violations, empty when the file keeps it: version, temperature names, grids, thresholds,
redundancy, fission product order, cumulative distributions that start at 0 and end at 1
and never decrease, the `urr` group, no `madland-nix` (the C++ cannot sample it) and the
parametrised spectra as `Tabulated1D`.

## Validation

On fixtures cut from the official OpenMC libraries (real files with most of each grid and
table removed, nothing invented: Fe-56 with levels, correlated distributions and photons;
U-235 with six delayed groups, `fission_energy_release` and `urr`; Cu-63 with Kalbach-Mann
and more than ten products; H-2 with `nbody`; four thermal materials covering coherent,
incoherent, inelastic-only and mixed elastic):

- HDF5 → model → HDF5 is identical for every neutron and thermal fixture
  (`kika/hdf5/tests/test_neutron_round_trip.py::test_neutron_round_trip_is_identical`,
  `::test_thermal_round_trip_is_identical`), through `kika.read`/`kika.write` as well
  (`::test_the_door_reads_and_writes_hdf5`); the total the C++ rebuilds is the sum of the
  partials (`::test_the_total_the_cpp_rebuilds_is_the_sum_of_the_partials`); an edited
  model is written edited (`::test_an_edited_model_is_written_edited`); what the model
  holds for U-235 and Fe-56 is checked node by node.
- The record layer reads and writes every fixture identically, eagerly and lazily
  (`kika/hdf5/tests/test_records.py`).
- The contract checker passes every fixture and catches each violation it lists
  (`kika/hdf5/tests/test_contract.py`).
- `cross_sections.xml` round-trips as text, the overlay swaps one material and points at
  the rest, and refuses a file whose group is another material
  (`kika/hdf5/tests/test_library.py`).
- Both sides of a step survive the processing (`kika/hdf5/tests/test_a_step_survives_processing.py`),
  and an HDF5 written from a staged run equals the one written from the tape alone
  (`kika/endf/tests/test_process_endf.py::test_hdf5_equals_the_hdf5_writer_from_scratch`).
- `import kika` does not import h5py or this package (`kika/hdf5/tests/test_dormancy.py`).

### Comparison with the official OpenMC library

Against the official OpenMC ENDF/B-VIII.1 library (processed by NJOY), for H-1, O-16,
Fe-56, U-234, U-235, U-238, Pu-239, Pu-240 and Pu-241 written by kika from the ENDF/B-VIII.1
evaluations at 293.6 K with the default tolerance, with
{py:func}`kika.hdf5.compare.compareNeutron` (measured 2026-10-21): the same reactions in
all nine; no structural difference in eight (U-234 has fewer cascade lines in five high
levels, MT 69, 75, 76, 80 and 86, 19 to 45 photon products against 39 to 54); the
resonance integral (0.5 eV to 1 MeV, 1/E) and the Maxwellian average at the file's $kT$ of
MT 2, 18 and 102 within 0.041 % (O-16 MT 102), and within 0.021 % for the other eight.

For JEFF-4.0 (H-1, O-16, Fe-56, U-235, U-238, Pu-239) against the official OpenMC JEFF-4.0
library, the same reactions, no structural difference, and the same integrals within
0.021 %, except the capture of O-16 (0.17 mb at 0.0253 eV): resonance integral −0.11 %,
Maxwellian average −0.070 % (measured 2026-10-21). The Maxwellian difference is the
official table's. Below 1 eV both files hold the capture as $1/v$ at their nodes (to
$2.5\cdot10^{-4}$), and against the exact $1/v$ law the Maxwellian average of kika's table
is low by 0.0065 % and the official one high by 0.064 %: read lin-lin, a chord of a convex
curve lies above it, and the official grid has 234 points below 1.27 eV ($50\,kT$) against
kika's 405.

In OpenMC 0.16.0, the official ENDF/B-VIII.1 library with kika's files for those nine
nuclides (and, in the UO₂ pin cell, kika's H in H₂O, processed by kika from the ENDF/B-VIII.1
thermal evaluation) against the official library alone, 20 000 particles × 120 batches
(20 inactive), every material at 293.6 K (measured 2026-10-21), $\Delta k$ with the URR
probability tables off / on, $\pm$ the combined standard deviation of the two runs:

| Model | URR tables off (pcm) | URR tables on (pcm) |
|---|---|---|
| UO₂ pin cell | $+125 \pm 91$ | $-102 \pm 89$ |
| Godiva | $+2 \pm 68$ | $+20 \pm 69$ |
| Jezebel | $-72 \pm 65$ | $-39 \pm 59$ |
| Iron | $+17 \pm 70$ | $-27 \pm 78$ |

All eight are within $1.4\sigma$. What is compared is the same model under two libraries,
so the difference is the data; two of the models are not published benchmarks, and none of
the $k$ values is compared with an experiment:

- **Godiva**: the simplified model of ICSBEP HEU-MET-FAST-001 (ICSBEP Handbook), a bare
  sphere of uranium of radius 8.7407 cm at 18.74 g/cm³, 1.02 / 93.71 / 5.27 wt% U-234 /
  U-235 / U-238.
- **Jezebel**: the sphere of ICSBEP PU-MET-FAST-001 (radius 6.3849 cm) of plutonium at
  15.61 g/cm³, 95.2 / 4.5 / 0.3 at.% Pu-239 / Pu-240 / Pu-241, with the benchmark's
  gallium (1.02 wt%) left out: a variant of the benchmark, whose $k$ is not the benchmark's.
- **UO₂ pin cell**: not a published benchmark. An infinite square lattice (pitch 1.26 cm,
  reflective) of a bare UO₂ pin of radius 0.4096 cm at 10.4 g/cm³, 4 at.% U-235 in the
  uranium, no gap and no cladding, in water at 1.0 g/cm³ with the H in H₂O thermal table.
- **Iron**: not a published benchmark. Godiva's sphere inside a 10 cm shell of Fe-56 alone
  at 7.874 g/cm³ (outer radius 18.7407 cm), so that Fe-56's inelastic levels and its photon
  production enter.

The size of kika's files, as {py:func}`~kika.hdf5.compare.compareNeutron` measures it (the
data at the temperatures both files hold and the data that does not depend on the
temperature), is 0.91 (Pu-241) to 1.65 (U-238) times the official one for eight of the nine
ENDF/B-VIII.1 nuclides, and 3.0 times for H-1 (measured 2026-10-21). Three choices make
kika's files larger, each for a precision NJOY's tables do not have: the cross sections of
a temperature are thinned to the share of the $10^{-3}$ tolerance the 0 K source and the
broadening leave, where NJOY thins to the whole $10^{-3}$ after reconstructing and broadening
to it; tabulated angular distributions hold the density to $10^{-3}$, where NJOY's ACER
thins its Legendre tables to $2\cdot10^{-3}$ (O-16's elastic tables hold 1.7 times the
official cosines); and H-1's capture photon carries the angular distribution the
evaluation gives at 153 energies, which the official file writes isotropic.

## Implementation

- `kika.hdf5.records`: the HDF5 tree as plain records; `kika.hdf5.api`: the doors and the
  dispatch on `filetype`.
- `kika.hdf5.neutron_decode`, `neutron_encode`: neutron files ↔ model;
  `kika.hdf5.distributions`, `functions`: OpenMC's distributions and `Function1D` ↔ model
  forms; `kika.hdf5.names`: the names the format fixes.
- `kika.hdf5.thermal_decode`, `thermal_encode`, `ace_thermal`: thermal files and thermal
  ACE tables ↔ model.
- `kika.hdf5.processing`: an evaluation → the processed suite;
  `kika.hdf5.thermal_processing`: the same for a thermal evaluation;
  `kika.hdf5.fission_energy`: the Sher-Beck step and the derived Q's.
- `kika.hdf5.writer`: the write, read-back and indexing; `kika.hdf5.library`:
  `cross_sections.xml` and the overlay.
- `kika.hdf5.contract`, `kika.hdf5.compare`: the contract checker and the comparison with
  another file.

## References

- P. K. Romano, N. E. Horelik, B. R. Herman, A. G. Nelson, B. Forget and K. Smith,
  "OpenMC: A state-of-the-art Monte Carlo code for research and development", *Annals of
  Nuclear Energy* 82 (2015) 90-97, doi:10.1016/j.anucene.2014.07.048.
- The OpenMC Development Team, *OpenMC documentation*, version 0.16.0, "Nuclear Data File
  Formats", <https://docs.openmc.org/en/v0.16.0/io_formats/nuclear_data.html>.
- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: MF 1/MT 458.
- NEA Nuclear Science Committee, *International Handbook of Evaluated Criticality Safety
  Benchmark Experiments*, NEA/NSC/DOC(95)03, OECD Nuclear Energy Agency, Paris:
  HEU-MET-FAST-001 (Godiva) and PU-MET-FAST-001 (Jezebel).
- R. Sher and C. Beck, EPRI NP-1771, Electric Power Research Institute (1981): the energy
  dependence of the fission energy release, as the ENDF-6 manual (CSEWG document
  ENDF-102) recommends it for MF 1/MT 458.
