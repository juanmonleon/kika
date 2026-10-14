# Thermal scattering libraries

A thermal scattering (TSL) library holds one evaluation per *atom in a compound*: H in
H₂O, O in UO₂, C in graphite. {py:mod}`kika.tsl` reads such libraries in five formats
and answers the same questions about each material whatever format it came in:

| format | what it is | detected by |
|---|---|---|
| ENDF-6 | an evaluation tape with NSUB=12 (MF 7) | MF 1/MT 451 |
| GNDS | a `reactionSuite` with the TNSL interaction | the XML root |
| ACE | thermal tables `.NNt`, one file per temperature | the header line |
| OpenMC HDF5 | a `data_thermal` file from OpenMC's `thermal/` folder | the `filetype` attribute |
| G4NDL | the `ThermalScattering` folder of a Geant4 library | its `Coherent`/`Incoherent` subfolders |

The format is decided on the file's content, never on its name or extension. The ENDF-6
and GNDS files are evaluations, with $S(\alpha,\beta)$; the other three are processed
tables (what NJOY THERMR and ACER made of an evaluation), with the inelastic
cross-section and the outgoing-neutron tables instead.

## Index a folder of thermal libraries

{py:func}`kika.tsl.index_tsl` lists every TSL material in a folder, grouped by compound,
reading only headers (of an ENDF-6 tape, MF 1, MF 7/MT 2, the head of MF 7/MT 4 up to
its first $eta$ block, and MF 7/MT 451 from the end of the file).

```python
from kika.tsl import index_tsl

index = index_tsl("endfb81/tsl")           # a folder, any of the five formats
for compound in index.compounds():
    print(compound.name, compound.atoms, compound.variants)
    for tape in compound.tapes:
        print("   ", tape.key, tape.format, tape.elastic, len(tape.temperatures))
```

Each entry is a {py:class}`kika.tsl.TSLTape`. Its `key` is unique in the library and
stable across re-indexing (`<MAT>:<file name>` for an ENDF-6 tape,
`<format>:<material>` otherwise). Besides the labels it is grouped by (`principal`,
`compound`, `variant`), it carries what a listing needs: `elastic` (`"coherent"`,
`"incoherent"`, `"both"` or `None`), `temperatures`, `n_alpha` and `n_beta`, `lat`,
`lasym`, the principal atom's `free_xs`, `principal_awr` and `principal_atoms`, the
`secondaries`, and for ENDF-6 tapes the MF 7/MT 451 `composition`. `problems` lists any
part of a header that could not be read.

**Compounds and variants.** The labels come from the file name first
(`tsl-<principal>in<compound>[-<variant>]`, and JEFF's `tsl_Be_BeO.txt` by the same
rules), and from ZSYMAM, the material's own label, when the name says nothing. Four
uranium dioxide tapes that all write `U(UO2)` as their ZSYMAM stay four tapes, with
variants `5P`, `10P`, `HALEU` and so on. {py:func}`kika.tsl.tsl_naming` is the rule on its own:

```python
from kika.tsl import tsl_naming

naming = tsl_naming("tsl-UinUO2-5P.endf")
naming.principal, naming.compound, naming.variant      # ('U', 'UO2', '5P')
```

**Files beside the tapes.** Files that came with a tape are attached to it by name in
`tape.companions`, by kind (`leapr`, `flassh-dos`, `flassh-control`, `readme`, `njoy`, …;
{py:func}`kika.tsl.companion_kind`). A readme named after the compound goes to the
compound (`compound.companions`). `tape.has_phonon_spectrum` says whether a LEAPR input or
a FLASSH density of states was found. What is neither a tape nor a companion is in
`index.unreadable`, and ENDF-6 tapes of other sublibraries in `index.other_endf`.

**Other layouts.** An OpenMC library root (with `cross_sections.xml`) is followed to its
`thermal/` folder, and a G4NDL root to its `ThermalScattering` folder. ACE tables of one
material at several temperatures (`al-27.70t`, `al-27.71t`) are one entry, with every file
in `tape.paths`. A single file can be indexed on its own; for an ENDF-6 tape the folder it
sits in is still searched for its companions:

```python
index = index_tsl("endfb81/tsl/tsl-Be-metal.endf")
index.single_file                    # True
index.tapes[0].companions            # {'leapr': ['.../tsl-Be-metal.leapr']}
```

`index.describe()` returns the whole index as plain dictionaries (compounds with their
tapes, the temperatures present, counts by format), ready to serialise.

To read only the header of one ENDF-6 tape, without indexing a folder, use
{py:func}`kika.tsl.read_tsl_header`; it returns the same {py:class}`kika.tsl.TSLTape`.

## Read one material

{py:func}`kika.tsl.read_tsl` reads one material in any of the five formats and returns a
{py:class}`kika.tsl.TSLMaterial`:

```python
from kika.tsl import read_tsl

water = read_tsl("endfb81/tsl/tsl-HinH2O.endf")         # ENDF-6
yh2   = read_tsl("openmc/endfb81/thermal/c_Y_in_YH2.h5")  # OpenMC HDF5
al    = read_tsl(["al-27.70t", "al-27.71t"])            # ACE: one file per temperature
alg4  = read_tsl("G4NDL4.7", member="al_metal")         # G4NDL: the root or ThermalScattering
```

A G4NDL folder holds many materials, so `member=` names the one to read (`h_water`,
`al_metal`); it is required there. The format can be forced with `format=` (one of
`kika.tsl.TSL_FORMATS`); {py:func}`kika.tsl.detect_tsl_format` tells it without
reading the file, and returns `None` for a file that is not a TSL file kika reads.

`material.suite` is the model `reactionSuite` (the same object {py:func}`kika.read`
returns) and `material.report` its conversion report. For the processed formats
(`material.is_processed`), `material.processed` keeps the tables the model has no node
for.

### Elastic and inelastic cross-sections

The questions are the same for every format; a part the file does not hold at that
temperature comes back as `None`:

```python
material = read_tsl("al-27.70t")
material.temperatures()          # {'coherent': [293.6], 'incoherent': [], 'inelastic': [293.6]}

sigma = material.elastic(293.6, [0.003, 0.005, 0.02])    # energies in eV
sigma["coherent"], sigma["incoherent"]                    # barns, or None

bragg = material.bragg(293.6)    # Bragg edges (eV) and cumulative S (eV·b)
bragg.edges, bragg.s

energies, xs = material.inelastic(293.6)                  # processed files only
```

{py:meth}`kika.tsl.TSLMaterial.elastic` answers from the model's laws for an evaluation and
for the exact parts of a processed file (OpenMC keeps the bound cross-section and the
Debye-Waller integral of an incoherent elastic), and from the processed table otherwise.
The temperature must be one the file holds.

{py:meth}`kika.tsl.TSLMaterial.inelastic` returns `None` for an ENDF-6 or GNDS evaluation:
the evaluation states $S(\alpha,\beta)$, and the inelastic cross-section is its integral.
To compute it, see *Compute the inelastic tables* below.

### Outgoing-neutron tables

The processed formats carry, at each incident energy, the outgoing energies with their
probabilities and a row of equiprobable cosines per outgoing energy. They are read when
first asked for:

```python
table = material.outgoing(293.6)          # a kika.tsl.processed.OutgoingTable, or None
table.kind                                # 'continuous', 'discrete' or 'skewed'
i = table.nearest(0.0253)                 # the tabulated incident energy closest to 0.0253 eV
table.incident[i], table.energies[i]      # eV
table.probabilities(i)                    # sums to 1
table.cosines[i]                          # one row of cosines per outgoing energy

material.elastic_cosines(293.6)           # (incident energies, cosines) of an incoherent elastic, or None
```

These tables are sampling tables: GNDS has no node for a set of equiprobable cosines, so
they are not in `material.suite`, and its report says so.

## Read a thermal table with `kika.read`

The same files open through {py:func}`kika.read`, which returns the model suite alone:

```python
import kika

suite = kika.read("al-27.70t")                    # an ACE thermal table
suite.interaction                                 # 'thermalNeutronScatteringLaw'
[r.label for r in suite.reactions]                # ['coherent-elastic', 'incoherent-inelastic']

suite = kika.read("G4NDL4.7", format="g4ndl", target="al_metal")
suite = kika.read("tsl-NinUN.endf")               # an ENDF-6 evaluation
```

For a processed file each temperature is a `heated` style (`heated_000`, `heated_001`, …
in the file's order) and the `eval` style is declared with no data behind it: the
evaluation the file was made from is not in it. An OpenMC thermal file opened with
`kika.read` goes through kika's OpenMC reader ({py:func}`kika.hdf5.api.read`), which keeps the whole
file and writes it back unchanged with `kika.write(suite, folder, format="hdf5")`; use
`read_tsl` for the per-format questions above.

## Compute the inelastic tables

kika computes the incoherent inelastic tables of an evaluation itself, as THERMR and ACER
do, and writes them as an OpenMC thermal file. The method, its conventions and its
comparison with the official files are in {doc}`../methods/processing/thermal-scattering-tables`.

```python
import kika

tape = "endfb81/tsl/tsl-HinH2O.endf"
kika.write(kika.read(tape), "openmc_out", format="hdf5",
           endf=tape, name="c_H_in_H2O", temperatures=[293.6, 600.0])
```

`name` is OpenMC's table name, which fixes the atoms the table applies to and how many
are mixed in it; `temperatures` must be ones the evaluation tabulates. The same function
is {py:func}`kika.hdf5.thermal_processing.processThermal`, which returns the processed
suite without writing it; with `options=ThermalInelasticOptions(iinc=1)`
({py:class}`kika.processing.thermal.ThermalInelasticOptions`) it makes the free-gas table
of a nuclide from its neutron tape instead.

## Run NJOY THERMR

{py:func}`kika.tsl.thermr.run_thermr` runs THERMR through NJOY over the free-atom
evaluation of the principal atom, in the chain MODER, RECONR, BROADR, THERMR, MODER, and
returns the cross-sections THERMR writes in MF 3:

```python
from kika.tsl.thermr import run_thermr

result = run_thermr("njoy", "tsl-HinH2O.endf", "n-001_H_001.endf", [293.6, 600.0])
for temperature, sections in result.sections.items():
    E, inelastic = sections[result.mtref]          # MT 222: incoherent inelastic
    E2, free = sections[2]                         # MT 2: the free-atom elastic
    if result.mtref + 1 in sections:
        E3, elastic = sections[result.mtref + 1]   # MT 223: the bound elastic, when there is one
```

Energies are in eV and cross-sections in barns. The inelastic is written under `mtref`
(222 by default) and the elastic, when the tape has one, under `mtref + 1`; both are
THERMR's, with its normalisation per principal atom and B(6) of MF 7/MT 4 as `natom`
(`result.natom`). The keyword arguments are `emax` (THERMR's upper energy, 4 eV by
default), `mtref`, `tol` (0.001), `workdir` (kept when given, a temporary folder
otherwise) and two callbacks: `on_stage`, called with each NJOY module as it starts, and
`should_stop`, which kills NJOY when it returns true (raising
{py:class}`kika.tsl.thermr.ThermrCancelled`). `result.deck` and `result.listing` are
NJOY's input and output.

What it checks and refuses:

- the TSL file must be an NSUB=12 tape, or a GNDS TNSL file (written to ENDF-6 first,
  since THERMR reads only ENDF);
- the neutron file must be an incident-neutron evaluation;
- every temperature must be one the TSL tape tabulates (within 0.05 K), since THERMR does
  not interpolate in temperature;
- NJOY builds on a Windows system with a comma decimal separator can misread their input;
  the run is checked against NJOY's echo of the tolerances and repeated, up to 12 times,
  before a `RuntimeError`;
- a run that writes no output tape raises `RuntimeError` with the end of NJOY's listing.

{py:func}`kika.tsl.thermr.thermr_deck` returns the NJOY input alone, to inspect or run by
hand.
