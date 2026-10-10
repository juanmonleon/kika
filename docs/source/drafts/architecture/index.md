# How kika is built

```{admonition} At a glance
:class: note
**Decides:** every format is read into one model, shaped and named after GNDS 2.1, and
code depends only downwards: leaves, then the calculation layer, then the format layer,
then the doors and consumers.

**Packages:** `kika.algebra`, `kika.nuclear_data`, `kika.processing`, `kika.cov`,
`kika.endf`, `kika.gnds`, `kika.ace`, `kika.g4ndl`, `kika.sampling`, and the top-level
`kika.read` / `kika.write`.

**Enforced by:** `kika/tests/test_layering.py`, `kika/algebra/tests/test_algebra.py`,
`kika/nuclear_data/model/tests/test_dormancy.py`,
`kika/endf/model_adapter/tests/test_nothing_imports_the_adapter.py` and
`kika/tests/test_read_front_door.py`.
```

## The structure

kika reads several nuclear-data formats: ENDF (Evaluated Nuclear Data File), GNDS
(Generalised Nuclear Database Structure), ACE (A Compact ENDF, the format of continuous-energy
Monte Carlo codes) and G4NDL (the Geant4 neutron data library). Whatever the file, the
data ends up in one in-memory model, the **canonical model**: a
{py:class}`~kika.nuclear_data.model.suite.ReactionSuite` for the reactions of one
projectile on one target, with its covariances beside it. Everything that computes
(reconstruction, group averages, perturbation) works on that model or on plain arrays,
and a format is only the way an answer is read in or written out.

The code is arranged in four layers, and an import may only point downwards:

```{figure} /_static/architecture/layers.*
:alt: Four stacked layers: doors and consumers, format layer, calculation layer, leaves, with arrows pointing down.
:width: 100%

The layers of kika's core. Read the arrows as "may import". Nothing in a lower layer
imports a higher one: the calculation layer never names a format, and the leaves import
nothing from kika at all.
```

- **Leaves** import nothing from kika. `kika.algebra` is the mathematics of tabulated
  functions; `kika._records` is the fixed-width record grammar that ENDF and the
  covariance formats share; `kika._legendre_frames` is the CM/LAB kinematics of Legendre
  coefficients.
- **The calculation layer** is the canonical model (`kika.nuclear_data.model`) and the
  code that computes on it or on arrays: `kika.processing` and `kika.cov`. It imports no
  format package.
- **The format layer** holds one package per format. Each reads its files, decodes them
  into the model, and encodes the model back.
- **Doors and consumers** sit on top. `kika.read` and `kika.write` are the single entry
  and exit for any format; `kika.sampling` and `kika.njoy` use both the model and the
  formats.

The packages for MCNP, sensitivities, uncertainty propagation, plotting and the other
application areas build on this core. The layering tests do not cover them.

## Design decisions

### One model for every format

Every decoder produces the same kind of object, so an operation is written once, on the
model, and holds for every format. A perturbation, for instance, is applied to a node of
the model and not to the records of a file; writing the perturbed node as ENDF, GNDS or
ACE is a separate step. Without one model, each operation would need one version per
format, and the versions could disagree.

### The model is GNDS-shaped and GNDS-named

The model follows the GNDS 2.1 specification (Mattoon et al., 2025): its classes are the
GNDS node names and its attributes keep the specification's spelling (`XYs1d`,
`outerDomainValue`). GNDS is the community's hierarchical description of evaluated data,
so its structure has already been argued over by evaluators and processing-code authors,
and a GNDS file maps onto the model with no intermediate layer. The cost is camelCase
names inside a Python package, accepted on purpose; see {doc}`canonical-model`.

### Formats are doors, not namespaces

{py:func}`kika.read` recognises the format from the file's content and returns a model
object; {py:func}`kika.write` writes one in the format asked for. A user does not have
to learn a format-shaped object for each format. The format-specific readers (such as
`kika.read_endf`) stay first-class underneath: the model does not cover every part of
every file, evaluators legitimately work in a format's own terms, and decoding into the
model costs time that a high-throughput pipeline need not pay. See {doc}`format-adapters`.

### Imports point one way

The calculation layer imports no format, so a computation is defined in terms of the
model or of arrays and cannot depend on how one file spelled its data. Formats may import
the calculation layer. The rule is held by a test that counts the existing exceptions and
lets the count go down but never up. See {doc}`layers`.

### Mathematics is a leaf

Evaluating, integrating, refining and summing a tabulated function is done once, in
`kika.algebra`, which imports nothing from kika. The model, the formats and the physics
can all use it without an import cycle, and two parts of kika cannot compute the same
integral two ways. See {doc}`algebra-layer`.

### Nothing converts silently

Every decoder and encoder fills a
{py:class}`~kika.nuclear_data.model.conversion.ConversionReport` that says what was lost,
what was approximated and what was not understood, and the report travels with the suite
it describes. A model that looks complete and is not would carry an authority it has not
earned.

### Refuse instead of guessing

Where the input allows no single reading (an interpolation law with no closed form, a
missing material number, a covariance form that cannot be combined), kika raises with
the reason, or reports, instead of picking a value. A guess looks like data.

### Results go beside the data they come from

A computed form (a reconstruction, a heated cross-section, a perturbed realisation) is
added to the model under its own label and style, next to the evaluated one, which is
left as it was. Functions that compute on a suite do not change their input.

### The model is not loaded by `import kika`

`import kika` does not import the model; it is loaded the first time something needs
it, such as {py:func}`kika.read`. Code that never uses the model (the MCNP tools, the
existing pipelines) pays nothing for it in import time, and an error inside the model
cannot break them.

## The layers

What each layer contains, the rule between them, how the tests enforce it, and the
exceptions that still exist: {doc}`layers`.

## The canonical model

The `ReactionSuite` and its children, styles and labels, the function containers, units,
provenance and the conversion report: {doc}`canonical-model`.

## Format adapters

How each format is read into the model and written back, the single door, and what makes
a conversion correct: {doc}`format-adapters`. The reading of each format in detail is in
the methods chapter, starting with {doc}`../methods/formats/endf`.

## The algebra layer

The representation of a tabulated function, the rules every operation keeps, and what
belongs in the layer and what does not: {doc}`algebra-layer`.

## Implementation

- `kika/__init__.py`: the public surface, with `read`, `write` and `identify` resolved
  lazily so that the model stays unloaded.
- `kika/_read.py`, `kika/_write.py`, `kika/_identify.py`: the doors.
- `kika/nuclear_data/model/`: the canonical model.
- `kika/endf/`, `kika/gnds/`, `kika/ace/`, `kika/g4ndl/`: the format layer.
- `kika/algebra/`: the mathematics.
- `kika/tests/`: the tests that hold the structure of the repository as a whole.

## References

- C. Mattoon, G. Gert, A. Holcomb, D. Brown, D. Wiarda, C. Chapman, W. Haeck and
  M. Staley, *GNDS-2.1 Specifications*, OECD Nuclear Energy Agency (2025).
