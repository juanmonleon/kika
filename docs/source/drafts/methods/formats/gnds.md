# Reading and writing GNDS

```{admonition} At a glance
:class: note
**Reads:** GNDS 2.0 and 2.1 XML: `reactionSuite`, `covarianceSuite`, `PoPs` (with decay
data) and fission-fragment yields.

**Writes:** the same documents from the model, the covariances as a separate file linked
to the evaluation.

**Into:** a {py:class}`~kika.nuclear_data.model.suite.ReactionSuite` with its
`CovarianceSuite`, or a `PoPs` tree (see {doc}`../../architecture/canonical-model`).

**Entry points:** {py:func}`kika.read`, {py:func}`kika.write`,
`kika.gnds.decode.readReactionSuite`, `kika.gnds.covariances.readCovarianceSuite`,
`kika.gnds.encode.writeReactionSuite`, `kika.gnds.capabilities`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

GNDS, the Generalised Nuclear Database Structure (Mattoon et al., 2025), describes an
evaluation as a tree. The root of the reaction data is a `reactionSuite`; its
covariances are a second root, a `covarianceSuite`, in its own file, which the
`reactionSuite` points to through an `externalFile` entry with a checksum. Particle
properties and decay data are a `PoPs` document. A node refers to another by an `href`,
an XPath into the same or another document. Data containers hold several forms, each
tagged with the label of a style that records how it was derived. Libraries publish
GNDS as XML; the specification also admits JSON and HDF5.

## Coverage

kika answers what it supports node by node, for all 300 elements that `gnds.xsd` and
`covariances.xsd` declare plus the 12 names kika uses that they do not:

```python
import kika.gnds
print(kika.gnds.capabilities().summary())
print(kika.gnds.capabilities(coverage="unsupported").text())
```

On 2026-10-11 it reported 218 nodes read and written in full, 17 in part and 65 not at
all, 13 of those without a line in a file's report. Each row gives the reason, citing the
specification or the source. In outline:

| Part of GNDS | What kika does |
|---|---|
| `reactionSuite`, styles, reactions, cross-sections, sums | read and written |
| Distributions (angular, energy, energy-angle, photon production) | read and written, except laws listed as unsupported by `capabilities()` |
| Resonances (resolved, unresolved, scattering radii) | read and written |
| `covarianceSuite` (§25) | read and written |
| `PoPs`, decay data, fission-fragment yields | read and written |
| Thermal neutron scattering law | read (`kika.gnds.thermal_scattering`) |
| Documentation, `applicationData` | read; kika's own resonance extensions written on request |
| JSON and HDF5 encodings | not read |

`capabilities()` says what the library can lose without opening a file; `suite.report`
says what one file lost.

## Reading

**1. Version.** The `format` attribute is checked first: 2.0 and 2.1 are read through one
code path; 1.9 and 1.10 (older) and 2.2 or 2.2.rc1 (newer) are refused by name, with the
reason. Version 2.1 is a modest revision of 2.0, centred on thermal scattering, and no
node kika reads differs between them.

**2. XML into the model, with no intermediate layer.** The GNDS tree already has the
model's shape, so each node is mapped onto the model class of the same name; there is no
format-shaped object between the XML and the model, unlike ENDF.

**3. Links.** `href` values are resolved in both dialects GNDS uses, and an
`externalFile` is followed with its checksum verified. `kika.read` of a `reactionSuite`
follows its link to the `covarianceSuite` beside it unless `covariances=False`; a missing
sibling is a report entry, not an error.

**4. Units and values** are kept as the file states them, on the axes; nothing is
rescaled on read.

## Assumptions and defaults

- None beyond the file: values, units and labels are read as stated. A file without a
  `format` attribute is refused (see *Refused*), not read as some default version.

## Ambiguous and inconsistent input

- **Covariance forms that combine several matrices** (`mixed`, `sum`) are read as stated;
  a consumer that needs one matrix asks `kika._covariance_forms` for it, and a form that
  cannot be reduced to one matrix raises with an explanation instead of being merged.
- **A missing covariance file** named by `externalFile`: reported, and the suite is
  returned without covariances.

## Refused

- GNDS 1.9, 1.10, 2.2 and 2.2.rc1, by name, and a file with no `format` attribute
  (`UnsupportedGndsVersion`).
- JSON and HDF5 files: `kika.read` recognises GNDS by its XML.

## Writing

`kika.write(suite, path, format="gnds")` writes the `reactionSuite` and, when the suite
has covariances, the `covarianceSuite` as a second file, with the `externalFile` entry and
its SHA-1 written to match the file actually written. The writer:

- **invents nothing**: a product whose distribution kika did not read is written with an
  empty `distribution` node, which fails schema validation and so announces itself, and
  never with `unspecified`, which would state that the evaluator gave none;
- **recomputes nothing**: a stated sum is written with its stated values, a resonance
  background with its own pieces;
- **writes every number so that it reads back as the same double** (Python's shortest
  round-tripping representation);
- writes the version the suite came in as, and 2.0 for a suite with no GNDS origin.

Byte identity with FUDGE's output is not a goal: number spelling, indentation and
attribute order differ.

## What the round trip keeps and loses

Read, write and read again gives the same numbers; write, read and write again gives the
same tree. What the model does not carry (the 65 unsupported nodes) is not written back,
and the report of the read names each one that the file contained, except the 13 that
`capabilities()` lists as silent.

## Report

`suite.report` after a read: unsupported nodes the file contained, a missing covariance
sibling, links that could not be resolved. After a write: every product written with an
empty distribution, every form the writer could not serialise.

## Validation

All tests below passed on 2026-10-11 unless marked.

- **The same evaluation in two encodings.** Fe-56 from ENDF/B-VIII.1 read as GNDS
  (FUDGE's translation) and as ENDF-6, through paths that share no parser: every shared
  cross-section agrees to the last bit, the 312 resolved resonances and the scattering
  radius agree, the covariances agree
  (`kika/gnds/tests/test_cross_section_oracle.py::test_every_shared_cross_section_agrees_to_the_last_bit`,
  `kika/gnds/tests/test_resonance_oracle.py::test_both_encodings_hold_the_same_312_resonances`,
  `kika/gnds/tests/test_covariance_oracle.py::test_fe56_covariances_agree_with_the_endf_path`).
  `TODO(check)`: these need the full GNDS and ENDF files (`tape`, `gnds` markers); not
  re-run for this page.
- **Writing.** Writing twice gives the same file; cross-sections, resonance parameters,
  unresolved averages and covariance matrices survive the round trip; every committed
  fixture validates against the schema; an unreadable distribution is left empty and never
  called `unspecified` (`kika/gnds/tests/test_encode.py`).
- **Version and links** (`kika/gnds/tests/test_version_and_xpath.py`), decay data
  (`test_pops_decay.py`), thermal scattering (`test_thermal_scattering.py`), heated styles
  (`test_heated_styles.py`).
- **The capability table** (`kika/gnds/tests/test_capabilities.py`): two of its tests fail
  on the default branch on 2026-10-11 (`test_the_registry_bounds_the_capability`,
  `test_a_paired_entry_turned_neither_fails_the_bridge`: the `heated` style is listed as
  partial where the node registry allows full, without a caveat).

## Implementation

- `kika/gnds/version.py`: the version gate; `xpath.py`: links and checksums.
- `kika/gnds/decode.py` (`readReactionSuite`), `distributions.py`, `resonances.py`,
  `covariances.py` (`readCovarianceSuite`), `decay.py`, `fission_yields.py`,
  `thermal_scattering.py`, `documentation.py`, `styles.py`, `primitives.py`: reading.
- `kika/gnds/encode.py` (`writeReactionSuite`, `writeCovarianceSuite`),
  `encode_resonances.py`, `encode_pops.py`: writing.
- `kika/gnds/_capabilities.py`: the capability table; `nodes.py`: the node registry it is
  checked against.
- `kika/gnds/endf_conversion.py`: FUDGE's ENDF conversion flags, used when a GNDS suite is
  written as ENDF; `resonance_extensions.py`: kika's opt-in extensions in
  `applicationData`.
- `kika/gnds/library.py`: a directory of GNDS files as a library; `tables.py`: summary
  tables.

## References

- C. Mattoon, G. Gert, A. Holcomb, D. Brown, D. Wiarda, C. Chapman, W. Haeck and
  M. Staley, *GNDS-2.1 Specifications*, OECD Nuclear Energy Agency (2025).
- NEA, *Specifications for the Generalised Nuclear Database Structure (GNDS): Version
  2.0*, OECD (2023), doi:10.1787/5f05e3db-en.
- C. Mattoon, B. Beck and G. Gert, "Managing and Processing Nuclear Data Libraries with
  FUDGE", *EPJ Web of Conferences* 284, 14010 (2023), doi:10.1051/epjconf/202328414010.
