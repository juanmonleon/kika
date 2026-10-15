# The layers and the direction of imports

```{admonition} At a glance
:class: note
**Decides:** the calculation layer (`kika.nuclear_data`, {py:mod}`kika.processing`, {py:mod}`kika.cov`)
imports no format package ({py:mod}`kika.endf`, {py:mod}`kika.ace`, {py:mod}`kika.gnds`, `kika.g4ndl`, `kika.hdf5`), and the
leaves import nothing from kika.

**Packages:** the core of kika (see the figure in {doc}`index`).

**Enforced by:** `kika/tests/test_layering.py` (the import ratchet and the import-time
checks), `kika/algebra/tests/test_algebra.py` (the algebra imports nothing from kika),
`kika/nuclear_data/model/tests/test_dormancy.py` (the model is not loaded by
`import kika`), `kika/endf/model_adapter/tests/test_nothing_imports_the_adapter.py`
(who may import a format adapter).
```

## The structure

| Layer | Packages | May import |
|---|---|---|
| Leaves | {py:mod}`kika.algebra`, `kika._records`, `kika._legendre_frames`, `kika._covariance_forms` | nothing from kika |
| Calculation | `kika.nuclear_data` (with the model, `kika.nuclear_data.model`), `kika.processing`, `kika.cov` | the leaves, and each other |
| Format | `kika.endf`, `kika.gnds`, `kika.ace`, `kika.g4ndl`, `kika.hdf5` | the calculation layer and the leaves |
| Doors and consumers | {py:func}`kika.read`, {py:func}`kika.write`, {py:func}`kika.identify` (in `kika/_read.py`, `_write.py`, `_identify.py`), `kika.sampling`, `kika.njoy`, {py:mod}`kika.tsl` | everything below |

Two things are checked about every import. **Where it points**: a module of the
calculation layer may not import a format package at all, at run time or under
`if TYPE_CHECKING:`. **When it runs**: `kika.processing` and `kika.cov` may not import a
format package at module scope even where the import is an allowed exception, and they
may not load the model at module scope either; such imports are deferred into the
function that needs them.

The packages outside this table (MCNP, sensitivities, uncertainty propagation,
benchmarks, plotting, EXFOR and the rest) are applications of the core and are not
checked by these tests.

## Design decisions

### The calculation layer does not know about formats

A cross-section, an angular distribution or a covariance is the same physical object
whichever file it came from, and the code that computes on it is written in those terms:
model nodes, or arrays. If the calculation layer imported a format, a computation could
come to depend on how one format spells its data (its record layout, its flags, its
units), and the same computation on data from another format would have to be written
again. Formats depend on the calculation layer instead: a decoder builds model nodes, an
encoder reads them.

### A ratchet, and not a rule

Some modules of the calculation layer still import a format. A rule that forbade every
such import would fail today; a rule with no list would let new imports in unnoticed. So
the test writes down every existing exception, file by file, with the number of format
imports it may contain. The number may go **down** without touching the test; it may
never go up, and a module not on the list may have none. A second test fails when an
entry is no longer needed, so the list cannot turn into a wish list. A third runs the
check on a small synthetic tree to prove it catches a violation, relative imports
included.

### Run-time and type-checking imports are counted apart

An import under `if TYPE_CHECKING:` costs nothing when the code runs, but it still makes
the calculation layer describe itself in a format's types. Both kinds are frozen. The
run-time count is the one that couples the layers.

### Imports are checked for when they run, not only for where they point

`kika.processing` and `kika.cov` must not import a format package when they are
themselves imported. When they did, nothing in `kika.endf` could import from
`kika.processing` at module scope without an import cycle, so numerics that belonged in
the calculation layer could not move there. The check is static (an import at module
scope or in a class body counts; one inside a function does not), because
`import kika` already loads the ENDF reader and a run-time check would see it loaded
whatever processing does.

### The model is loaded only when it is used

A plain `import kika` does not import `kika.nuclear_data.model`. The model is several
thousand lines of classes; the cluster pipeline, the desktop application and notebooks
that never touch it would otherwise pay for it on every import, and an error inside it
would break them all. `kika.read` and `kika.write` are reached through a lazy attribute
of the `kika` package for the same reason. Deferred imports of the model are allowed and
expected to grow; module-scope ones from `kika.processing` and `kika.cov` are not.

### Shared, format-free code lives at the top level

Some code is used by packages that may not import each other. It goes to a private
top-level module that imports nothing from kika, and both sides import it:

- `kika._records`: the fixed-width record grammar (11-column fields, ENDF-style floats).
  It is not ENDF's property: the COVERX, COVFIL and BOXER covariance formats in
  `kika.cov` use the same convention.
- `kika._legendre_frames`: the CM/LAB cosine maps, their Jacobian, and the change of frame of Legendre
  coefficients, needed by the model's perturbation code and by the sampler, which may
  import neither each other nor `kika.endf`.
- `kika._covariance_forms`: what a covariance form must be before it can be computed on,
  needed by `kika.cov` and `kika.sampling`.

### The doors are private modules with public names

`kika.read`, `kika.write` and `kika.identify` live in `kika/_read.py`, `_write.py` and
`_identify.py`. A module named `kika/read.py` would collide with the function
`kika.read`: importing the submodule binds the module onto the package, and after the
first access the name would return the module instead of the function.

### Only listed modules may import a format adapter

`kika.endf.model_adapter` and `kika.ace.model_adapter` (the code that converts between a
format and the model) may be imported only by the modules on an explicit list: the
doors, the writers, the sampling pipelines and a few others. Any new importer has to be
added on purpose, which keeps the conversion in one place.

## What it rules out

- A format import, at run time or under `TYPE_CHECKING`, in any module of
  `kika.nuclear_data`, `kika.processing` or `kika.cov` that is not on the allowlist, or
  more of them than the allowlist states
  (`kika/tests/test_layering.py::test_no_new_format_imports_in_the_calculation_layer`).
- A format import at module scope in `kika.processing` or `kika.cov`
  (`::test_processing_pulls_no_format_package_in_at_import_time`), and a module-scope
  import of the model from them (`::test_processing_does_not_wake_the_model_at_import_time`).
- Any import of `kika` from `kika.algebra`, and a new algebra submodule that is not
  added to the test's frozen list
  (`kika/algebra/tests/test_algebra.py::test_algebra_imports_nothing_from_kika`,
  `::test_algebra_holds_only_its_mathematical_submodules`).
- Loading the model from `import kika` or `import kika.nuclear_data`, and the model
  pulling in format modules of its own
  (`kika/nuclear_data/model/tests/test_dormancy.py`).
- An unlisted importer of a format adapter
  (`kika/endf/model_adapter/tests/test_nothing_imports_the_adapter.py::test_only_the_facades_and_the_front_door_import_the_adapter`).
- By convention, and not by a test: no allowlist entry is ever added for
  `kika.nuclear_data.model`, and the leaves other than `kika.algebra` (`kika._records`,
  `kika._legendre_frames`, `kika._covariance_forms`) import nothing from kika. None
  of them imports kika today.

## Known exceptions

Run-time format imports the ratchet allows, all of them deferred to call time:

| Module | Imports | Why it stands |
|---|---|---|
| `kika/cov/legendre_covariance.py` | 4 | Three are its MF 34 encoder, format code that stays where it is; one reads an ENDF file. |
| `kika/nuclear_data/angular_distribution.py` | 13 | A deprecated flat class whose `from_endf` / `to_endf` route through the ENDF adapter. Removed with the flat classes in 1.0. |
| `kika/nuclear_data/cross_section.py` | 3 | The same, plus reading Q-values from ACE through the ACE adapter. |
| `kika/nuclear_data/nuclide_info.py` | 2 | A deprecated flat class that builds an ENDF MF 1 header. |
| `kika/nuclear_data/resonance_parameters.py` | 2 | A deprecated flat class. |
| `kika/processing/derived_covariance.py`, `njoy_pendf_cache.py`, `njoy_reconstruct.py` | 1 each | Each reads a PENDF written by NJOY with the ENDF reader: a real dependency on a file format. |

Type-checking imports allowed: `legendre_covariance.py` 1, `angular_distribution.py` 2,
`cross_section.py` 3, `nuclide_info.py` 2, `resonance_parameters.py` 1.

`kika.sampling` is a consumer and imports both the model and the formats. No test
checks which direction its own modules import in.

## Extending it

- **New numerics** with no physics in them go to `kika.algebra`, after asking whether
  they are mathematics (see {doc}`algebra-layer`); the frozen submodule list in its test
  has to be updated on purpose.
- **A new computation** on nuclear data goes to the calculation layer and takes model
  nodes or arrays. If it seems to need a format, the format-specific part belongs in that
  format's adapter, which hands the computation a model node.
- **A new format** is a new package in the format layer, added to `FORBIDDEN_ROOTS` in
  `kika/tests/test_layering.py` when it is created, with a decoder and an encoder to the
  model (see {doc}`format-adapters`).
- **Code shared by packages that may not import each other** becomes a private top-level
  leaf that imports nothing from kika.

## Implementation

- `kika/tests/test_layering.py`: the guarded and forbidden packages, both allowlists, and
  the import-time checks.
- `kika/algebra/tests/test_algebra.py`: the isolation of the algebra and its frozen
  submodule set.
- `kika/nuclear_data/model/tests/test_dormancy.py`: the model stays unloaded on
  `import kika`.
- `kika/endf/model_adapter/tests/test_nothing_imports_the_adapter.py`: the list of
  modules that may import a format adapter.
- `kika/__init__.py`: the lazy attributes `read`, `write` and `identify`.

## References

None: the structure is kika's own.
