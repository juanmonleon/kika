# The canonical model

```{admonition} At a glance
:class: note
**Decides:** every format is decoded into the same in-memory tree, whose structure and
names are those of the GNDS 2.1 specification, and which carries, beside the data, what
the source format said that GNDS has no node for and what the conversion lost.

**Packages:** `kika.nuclear_data.model`.

**Enforced by:** `kika/nuclear_data/model/tests/test_gnds_naming.py` (names),
`kika/tests/test_layering.py` (no format imports),
`kika/nuclear_data/model/tests/test_dormancy.py` (not loaded by `import kika`),
`kika/nuclear_data/model/tests/test_functions_reproduce_the_flat_path.py` (evaluation).
```

## The structure

The root of one evaluation is a {py:class}`~kika.nuclear_data.model.suite.ReactionSuite`
(GNDS-2.1 §14.1.1): the reactions of one projectile on one target. Its attributes are
the specification's (`evaluation`, `projectile`, `target`, `projectileFrame`,
`interaction`, `format`) and so are its children:

| Child | Holds |
|---|---|
| `styles` | How each form was derived: evaluated, reconstructed, and so on (§9-10). |
| `PoPs` | The particles: masses, spins, levels (§12). |
| `resonances` | Resolved and unresolved resonance parameters, by formalism (§19). |
| `reactions` | One `Reaction` per reaction channel, with its cross-section and products. |
| `sums` | Cross-sections and multiplicities that are sums of others (§21). |
| `orphanProducts`, `fissionComponents`, `productions`, `incompleteReactions` | As in the specification. |
| `externalFiles`, `applicationData` | As in the specification. |

Three further attributes are not GNDS nodes:

- `covarianceSuite`: the covariances, a
  {py:class}`~kika.nuclear_data.model.covariances.CovarianceSuite` (§25). In GNDS they are a
  separate root that the reaction suite points at; kika keeps the one it read beside the
  evaluation, and its GNDS writer emits it as a separate document.
- `provenance`: what the source format stated that the model has no place for (for ENDF,
  the MF 1/451 header), kept so that the format can be written back.
- `report`: the {py:class}`~kika.nuclear_data.model.conversion.ConversionReport` of the
  conversion that produced the suite.

Decay data and fission-product yields decode to a `PoPs` tree of their own instead of a
reaction suite.

**Data containers hold forms, each under a label.** A cross-section, for instance, can
hold an evaluated form labelled `eval` and a reconstructed one labelled `recon`; the
`styles` node records that `recon` was derived from `eval`. The labels kika uses are
`eval` (what the file states), `recon` (the resonance region reconstructed, from kika's
own reconstruction or from an NJOY PENDF file) and, for a perturbed realisation,
`realization-NNNN`.

**Functions are the GNDS functional containers** (§6). The two that matter most:

- {py:class}`~kika.nuclear_data.model.functions.xys1d.XYs1d`: a table of $(x, y)$ with
  one interpolation law for the whole table. The values are two numpy arrays.
- {py:class}`~kika.nuclear_data.model.functions.regions1d.Regions1d`: several `XYs1d`,
  one per interpolation region, each with its own law. Neighbouring regions share their
  boundary point.

Two-dimensional distributions (`XYs2d`, `Regions2d`), Legendre series (`Legendre`) and
the other containers of §6 follow the same pattern.

**Units live on axes and scalars**, never on array elements (§2.3.3, §5.1.2): an
`XYs1d` carries an `axes` object whose entries name the unit of each column, and the
arrays stay plain numbers. Cross-sections are in barns against incident energy in eV.

## Design decisions

### Shaped and named after GNDS 2.1

The classes are the GNDS node names with the first letter capitalised, and attributes
keep the specification's spelling exactly (`XYs1d`, `outerDomainValue`,
`interpolationQualifier`). The GNDS reader maps nodes onto classes by these names, so a
"tidier" snake_case name would silently break it; the test enforces the convention and
lists every deliberate departure. The model targets GNDS 2.1 (`GNDS_FORMAT = "2.1"`); the
GNDS reader accepts files written in 2.0 and 2.1, which is what the distributed
libraries use.

Using the specification's structure instead of one designed for kika means the hard
modelling questions (where a product's distribution lives, how a sum is linked to its
parts, how a derived form names its origin) have answers that evaluators and the authors
of FUDGE and other processing codes already share (Mattoon et al., 2025).

### Every child exists, even when empty

A child that the file has nothing for is an empty container, not `None`, and the
containers are always true in a boolean test. An empty slot says "kika models this and
this file has none"; a missing attribute says nothing and fails far from its cause.
`resonances` is the one child that is `None` when the evaluation has no resonance
parameters.

### Several interpolation regions are kept as several regions

A table with more than one interpolation region is a `Regions1d`, one function per
region, and never one table with a single "dominant" law. Interpolating part of a table
under the wrong law gives wrong numbers with no error, and reading the regions from a
side field is a step every caller would have to remember. A gap between regions cannot
be written as one ENDF table without inventing data, and raises.

### Evaluation has one implementation

`XYs1d` and `Regions1d` evaluate through the same interpolator that kika's array-based
code uses, which rests on `kika.algebra`. The model therefore returns, bit for bit, what
the rest of kika computes from the same arrays.

### Units are named, and never converted on read

kika keeps its own registry of the GNDS unit syntax (§3.5): adopting a general unit
library would mean translating on every read and write, and the translation is where
unit errors live. The model does not rescale on read; an accessor asked for a unit the
data is not in raises instead of converting. Masses are never expressed in MeV, as §3.5
requires.

### Provenance keeps what GNDS has no node for

Each format has a typed provenance record (`EndfProvenance`, `AceProvenance`,
`GndsProvenance`, and those for G4NDL). It holds what the source stated and the model
does not represent: ENDF header flags, the interpolation pairs of a section, the
material number. With it, a file read and written back keeps fields that have no
meaning in GNDS. The rule for the ENDF decoder: every field of a section is either
mapped to a model node or kept in its provenance.

### The report travels with the data

The {py:class}`~kika.nuclear_data.model.conversion.ConversionReport` sorts what a
conversion did beyond producing its result into four lists:

- `warnings`: something looked odd, and nothing was lost;
- `losses`: data in the source that is not in the result;
- `approximations`: data that is in the result but changed on the way;
- `unsupported`: constructs the reader recognised and cannot represent.

An approximation is the dangerous kind: a loss shows as a missing field, an
approximation looks like data. `isClean` is true when the last three lists are empty
(warnings do not count). An entry can be declared irrelevant to a named consumer, and
`isCleanFor(scope)` asks whether a conversion is clean for that purpose. Entries are
messages, not codes. The report is stored on the suite (`suite.report`), so it cannot be
lost by a caller who drops the second element of a returned tuple.

### Stated sums are kept, and linked to their parts

A cross-section the file states as a sum (the total, MT 1; the non-elastic, MT 3; the
inelastic, MT 4, and the others of §21) is kept with the values the file gives and a
link to its summands; it is not recomputed on read. The file's numbers are the
evaluation, and silently replacing them by a recomputed sum would change it.

### Derived forms are added, not substituted

A reconstruction or a perturbed realisation is added to the same containers under a new
label, next to the evaluated form, which stays. A suite can therefore carry the
evaluation and what was computed from it, and a writer chooses which label to write.

## What it rules out

- A format import anywhere in the model (`kika/tests/test_layering.py`, which scans the
  whole of `kika.nuclear_data`).
- A snake_case attribute beside a GNDS-named one, and a class name that is not a GNDS
  node, unless declared as a divergence
  (`kika/nuclear_data/model/tests/test_gnds_naming.py::test_no_snake_case_twin_of_a_camel_attribute`,
  `::test_class_names_are_gnds_node_names_capitalised`, `::test_every_divergence_is_declared`).
- A function container that evaluates differently from kika's array code
  (`kika/nuclear_data/model/tests/test_functions_reproduce_the_flat_path.py::test_xys1d_equals_the_flat_interpolator`,
  `::test_regions1d_equals_the_flat_interpolator_on_a_multi_region_table`).
- Loading the model on `import kika` (`kika/nuclear_data/model/tests/test_dormancy.py`).

## Known exceptions

- **The model's containers are mutable.** Only small value objects (axes, units,
  physical quantities, reaction identifiers) are frozen; a `ReactionSuite`, its
  functions and its arrays can be edited in place. Code that computes on a suite does not
  edit it, by convention, and the resonance reconstruction checks that the suite did not
  change before it attaches its result.
- **`resonances` is `None` when absent**, unlike every other child.
- **A realisation's label is not registered as a style.** `perturbFromModel` adds each
  sample as forms labelled `realization-NNNN`, but adds no `realization` style (§9.3)
  naming the evaluated style it came from; the label is removed once the sample is
  written.
- **The flat classes.** `kika.nuclear_data` still offers `CrossSection`,
  `AngularDistribution`, `ResonanceParameters` and `NuclideInfo`, which predate the
  model. They are deprecated, warn once when used, and are removed in kika-nd 1.0.
  `kika.nuclear_data.model.interop` projects model nodes onto their plain-data shape, in
  that direction only.

## Extending it

A new kind of data gets a class named after its GNDS node, under the specification's
section, with its attributes spelled as there; a departure goes into the divergence list
of `test_gnds_naming.py` with its reason. What a format states and GNDS cannot hold goes
into that format's provenance record, not into a new model attribute. Every decoder that
fills the new node reports what it could not map.

## Implementation

- `suite.py`: `ReactionSuite`, and `GNDS_FORMAT`.
- `styles.py`: the style classes and the `derivedFrom` chain (a dangling or circular
  chain raises).
- `functions/`: the §6 containers (`XYs1d`, `Regions1d`, `XYs2d`, `Regions2d`,
  `Legendre`, and the rest).
- `axes.py`, `units.py`, `quantities.py`, `values.py`: axes, the unit registry, physical
  quantities.
- `reactions.py`, `distributions.py`, `energy_spectra.py`, `resonances/`, `pops.py`,
  `decay.py`, `fission_yields.py`, `thermal_scattering.py`: the data nodes.
- `sums.py`: cross-section and multiplicity sums.
- `covariances.py`: the `CovarianceSuite`.
- `provenance.py`: the per-format provenance records.
- `conversion.py`: the `ConversionReport`.
- `compare.py`: `modelDifferences`, which walks two trees and lists every leaf that
  differs; the round-trip tests use it.
- `interop.py`: the projection onto the flat classes' plain data.

## References

- C. Mattoon, G. Gert, A. Holcomb, D. Brown, D. Wiarda, C. Chapman, W. Haeck and
  M. Staley, *GNDS-2.1 Specifications*, OECD Nuclear Energy Agency (2025). Section
  numbers on this page are this document's.
- C. Mattoon, B. Beck and G. Gert, "Managing and Processing Nuclear Data Libraries with
  FUDGE", *EPJ Web of Conferences* 284, 14010 (2023), doi:10.1051/epjconf/202328414010.
