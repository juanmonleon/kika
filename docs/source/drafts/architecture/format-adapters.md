# Format adapters

```{admonition} At a glance
:class: note
**Decides:** each format has one adapter that decodes it into the canonical model and
encodes the model back; one door, {py:func}`kika.read`, routes any file to the right
decoder, and one door, {py:func}`kika.write`, to the right encoder.

**Packages:** `kika.endf` (with `kika.endf.model_adapter` and `kika.endf.writers`),
`kika.gnds`, `kika.ace` (with `kika.ace.model_adapter`), `kika.g4ndl`, and the doors in
`kika/_read.py`, `kika/_write.py`, `kika/_identify.py`.

**Enforced by:** `kika/tests/test_read_front_door.py` (the door is a route),
`kika/endf/model_adapter/tests/test_nothing_imports_the_adapter.py` (who may import an
adapter), and each format's round-trip tests.
```

## The structure

| Format | Into the model | Out of the model | Correctness gate |
|---|---|---|---|
| ENDF-6 | `kika.read_endf` parses the tape into ENDF objects; `decodeReactionSuite` and `decodeCovarianceSuite` (`kika.endf.model_adapter`) decode them | `writeEndfTape` (`kika.endf.writers.assemble`), built on one encoder per section (`encodeMF3MT`, `encodeMF4MT`, ...) | Read, write, read: the model is a fixed point; each section encoder is byte-identical to the file |
| GNDS | `readReactionSuite`, `readCovarianceSuite` (`kika.gnds`), straight from the XML tree | `writeReactionSuite`, `writeCovarianceSuite` | Read, write, read is a fixed point on the numbers; write, read, write is a fixed point on the tree |
| ACE | `read_ace` parses; `decodeAce` (`kika.ace.model_adapter`) decodes | None from the model: ACE is produced by NJOY from an ENDF tape | The parsed ACE object, read and written back, returns every XSS entry bitwise identical |
| G4NDL | `kika.g4ndl.open(root).read(target)`: elastic, inelastic, capture and fission | `writeSuite`, and `patch_isotope` / `patch_elastic` on an existing library | Read and write over whole libraries, per channel |

The doors:

- {py:func}`kika.read` `(path, format=None, covariances=True, target=None)` recognises
  the format from the file's content (`kika.sniff_format`), calls that format's decoder,
  and returns a `ReactionSuite` (a `PoPs` tree for decay and fission-yield data, a
  photo-atomic object for photo-atomic ACE). The conversion report is on `suite.report`.
  `format=` forces a format past the recognition; `covariances=False` skips them;
  `target=` names the isotope in a G4NDL library, which is a directory.
- {py:func}`kika.write` `(suite, path, format="gnds", ...)` writes a suite as GNDS, ENDF
  or G4NDL and returns the conversion report of the writing. A suite with covariances is
  written to GNDS as two files, the evaluation and its covariances, linked as the
  specification requires.
- {py:func}`kika.identify` `(path)` names what kind of file a path is, for every format
  kika reads (covariance libraries, SDF, MCNP and Serpent files, NJOY decks as well as the
  four above), from the head of its content and without running a parser.

## Design decisions

### One door, several decoders, no logic in the door

`kika.read` only recognises and routes. It computes nothing that the decoder called
directly would not compute, so there is one answer to "what does this file contain",
whichever way it was asked. The door's test asserts this for ENDF, GNDS and G4NDL, and
that the report the door attaches is the one the decoder produced.

### Recognise a file by its content, never its name

The format is decided from what the file contains, because extensions are inconsistent
across libraries and sites. A file that matches no format is refused with the list of
what was tried; a binary file is refused instead of being handed to a text decoder.

### The parse layer stays first-class

For ENDF and ACE, reading has two steps: a parser that turns the file into objects
shaped like the format (sections, records), and a decoder that turns those into the
model. The format-shaped objects (`kika.read_endf`, `kika.read_ace`) are public and not
deprecated. The model does not cover every part of every file (ENDF MF 8 to 10, for
instance), evaluators legitimately work in the format's own terms, and decoding into the
model costs about 2.5 times a bare parse, which a high-throughput pipeline does not pay.
GNDS has no such layer: its tree already has the model's shape, so the XML goes straight
into the model.

### Only the adapter converts

Code outside a format package does not decode or encode that format by hand; it calls
the adapter, through the doors or through the modules on the adapter's import list
(the writers, the sampling pipelines). The conversion rules of a format are then in one
place, and the test that guards the list makes each new importer a deliberate choice.

### Correct means the model survives the trip

The ENDF whole-tape gate is a fixed point of the model: read a tape, write it, read the
result, and the two models are the same. Byte identity with the original tape is
deliberately not the gate: whether a field is padded, or whether `1e-5` is spelled
`1.0-5`, carries no information. What the fixed point catches is a quantity that does
not survive the trip. What it cannot see is anything the model does not carry, which is
absent on both sides; the conversion report has to say what that is. The per-section
encoders are held to a stricter rule: each section that came from a file is written back
byte for byte. GNDS is held to two fixed points (on the numbers, and on the tree).
FUDGE (Mattoon et al., 2023), the reference GNDS toolkit, is used as an independent
reader and writer in the oracle tests.

### Write only what the model holds, and say what is missing

A written file is indistinguishable from an evaluation, so anything missing from it is
asserted absent. The writers emit only what the model holds and report every gap. They
invent nothing to fill a hole (a GNDS product whose distribution kika did not read gets
an empty distribution node, which fails validation, instead of one that claims no
distribution is given), and they recompute nothing (a stated sum is written with its
stated values).

### Bookkeeping a format needs is derived, or refused

ENDF needs fields that a GNDS file does not carry (the material number, MF 4 flags,
Q-values per reaction). When a suite has no ENDF provenance, the ENDF writer derives each
such field from the model, from FUDGE's ENDF conversion flags stored in the GNDS file, or
from a named default, and reports what it derived. A field with no neutral default (the
material number, for instance) is refused instead of stamped with a guess. A suite read
from ENDF is never derived: its own fields are written back.

### Covariances are a separate root

GNDS makes the covariance suite a document of its own (§25.1.1). The ENDF decoder
decodes the covariance files (MF 31 to 35) into a separate
{py:class}`~kika.nuclear_data.model.covariances.CovarianceSuite`, which `kika.read` puts
on `suite.covarianceSuite`; the GNDS writer writes it as a sibling file.

## What it rules out

- A door that decodes differently from the decoder it routes to
  (`kika/tests/test_read_front_door.py::test_the_door_is_a_route_not_a_second_decoder`,
  `::test_the_gnds_door_is_a_route_not_a_second_decoder`,
  `::test_the_g4ndl_door_is_a_route_not_a_second_decoder`), and a report emptied on the
  way (`::test_the_front_door_and_the_low_road_agree_on_the_report`).
- An MF with no parser passing unreported
  (`::test_an_mf_with_no_parser_is_reported_rather_than_passed_over`).
- An unlisted importer of an adapter
  (`kika/endf/model_adapter/tests/test_nothing_imports_the_adapter.py::test_only_the_facades_and_the_front_door_import_the_adapter`),
  and reading a file with the parse layer building the model as a side effect
  (`::test_reading_an_endf_file_does_not_build_the_model`,
  `::test_reading_an_ace_file_does_not_build_the_model`).
- An ENDF tape whose model changes on the trip
  (`kika/endf/model_adapter/tests/test_endf_tape_fixed_point.py::test_the_reaction_suite_is_a_fixed_point`),
  and an ENDF write that stamps a guessed material number
  (`::test_a_suite_with_no_mat_is_refused_rather_than_stamped_with_a_guess`).

## Known exceptions

- **No model-to-ACE encoder.** ACE is written from the parsed ACE object, or produced by
  NJOY from an ENDF tape; `kika.write` does not offer ACE.
- **ACE decodes into the `gridded` style**, not `eval`: its cross-sections are values on
  ACE's unionised grid at the file's temperature, not the evaluation.
- **Only GNDS has a capability census.** `kika.gnds.capabilities()` states, for every
  node of the GNDS schemas, whether kika reads and writes it. The other formats state
  their coverage on their methods page, and in the report of each conversion.
- **The deprecated flat classes** in `kika.nuclear_data` import the ENDF and ACE adapters
  directly; they are on the import list until they are removed in kika-nd 1.0.

## Extending it

A new format is a package in the format layer:

1. a parser, if the format needs a format-shaped layer, and a decoder into the model
   that fills a `ConversionReport` for everything it cannot map;
2. a provenance record in the model for what the format states and GNDS cannot hold;
3. an encoder, if kika writes the format, that writes only what the model holds;
4. its recognition in `kika.sniff_format` and its entry in the formats of `kika.read`
   (and of `kika.write`);
5. a round-trip test (the model is a fixed point), and an oracle against another
   implementation where one exists;
6. its package name in `FORBIDDEN_ROOTS` in `kika/tests/test_layering.py`, from the
   day it is created.

The reading of each format, with its assumptions and refusals, is documented in the
methods chapter: {doc}`../methods/formats/endf`.

## Implementation

- `kika/_read.py`: `read`, `sniff_format`; `kika/_write.py`: `write`;
  `kika/_identify.py`: `identify`.
- `kika/endf/parsers/`, `kika/endf/classes/`: the ENDF parse layer.
- `kika/endf/model_adapter/`: ENDF decoders and section encoders; `derive/` for ENDF
  bookkeeping a GNDS-origin suite lacks; `pendf.py` to attach an NJOY PENDF as the
  `recon` form.
- `kika/endf/writers/assemble.py`: a whole tape from the model.
- `kika/gnds/decode.py`, `encode.py`, `covariances.py`, `_capabilities.py`: GNDS.
- `kika/ace/model_adapter/`: ACE decoding.
- `kika/g4ndl/`: G4NDL reading, writing and patching.

## References

- C. Mattoon, G. Gert, A. Holcomb, D. Brown, D. Wiarda, C. Chapman, W. Haeck and
  M. Staley, *GNDS-2.1 Specifications*, OECD Nuclear Energy Agency (2025).
- C. Mattoon, B. Beck and G. Gert, "Managing and Processing Nuclear Data Libraries with
  FUDGE", *EPJ Web of Conferences* 284, 14010 (2023), doi:10.1051/epjconf/202328414010.
