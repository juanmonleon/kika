# Working with ENDF files

What you can do with an ENDF-6 tape in kika, and the call for each task: identify it,
read all of it or only what you need, decode it into the model, edit it, check its
covariances, process it for transport, download it, and index a whole library. How kika reads and writes the
format (what it assumes, what it refuses) is in {doc}`../methods/formats/endf`.

The examples use a tape called `Fe56.endf`; any ENDF-6 tape works.

## Identify a tape without reading it

`tape_inventory` reads the MF 1/451 header and directory only:

```python
from kika.endf import tape_inventory

inv = tape_inventory("Fe56.endf")
inv.mat, inv.za, inv.library_tag     # 2631, 26056, 'JEFF-4.0 Incident Neutron File'
inv.sections                         # ((1, 451), (2, 151), (3, 1), (3, 2), ...)
```

The section list is what the directory states. `tape_inventory(path, scan=True)` scans
the tape itself instead, for a tape whose directory is wrong.

## Read the tape in its own terms

`kika.read_endf` parses the whole tape into objects shaped like the format: one per MF,
one per MT.

```python
import kika

endf = kika.read_endf("Fe56.endf")
capture = endf.mf[3].mt[102]
capture.energies, capture.cross_sections    # the TAB1, as lists
capture.q_reaction                           # QI, in eV
endf.mf[33].mt[102]                          # a covariance section, as ENDF structures it
```

When a few MF files are enough, two cheaper roads:

```python
from kika.endf import open_endf, read_mf3_mt

lazy = open_endf("Fe56.endf")     # indexes the tape; parses nothing yet
34 in lazy.files                  # presence check, no parsing
lazy.files[34]                    # parses MF 34 on first access

elastic = read_mf3_mt("Fe56.endf", 2)   # one MF 3 section, nothing else
```

`kika.endf` also offers `read_mt451`, `read_mf2`, `read_mf4_mt` and `read_mf7_mt` for
single sections.

Use the format's own objects when you work as an evaluator in ENDF's vocabulary, when
you need a part of the file the model does not decode, or when you read thousands of
tapes (decoding into the model costs about 2.5 times a bare parse).

## Read it into the model

```python
ev = kika.read("Fe56.endf")
print(ev.summary())
ev.report.isClean              # True when nothing was lost, approximated or refused
E, xs = ev.cross_section(102)  # two numpy arrays
ev.covarianceSuite             # MF 31-35, decoded
```

The model is the same whatever the format; see {doc}`../architecture/canonical-model`
and {doc}`/user_guide/reading_files`.

## Process it for transport

{py:func}`kika.endf.process_endf` reconstructs the resonances, broadens to each
temperature, computes the unresolved-range probability tables (and, on request, the
heating), and writes a PENDF, an OpenMC HDF5 file or a GNDS file with heated styles, with
no NJOY; {py:func}`kika.endf.reconstruct_endf` is its first stage on its own, at 0 K. The
calls are in {doc}`processing-an-evaluation`.

```python
from kika.endf import process_endf, ProcessingPlan

run = process_endf("Fe56.endf", ProcessingPlan(temperatures=(293.6, 600.0)))
run.write_pendf("Fe56.pendf")
print(run.report_document().to_text())
```

## Edit a tape

Edits work on the text of the tape and keep everything else byte for byte.

Transfer a section from another tape, rebuilding the stated sums (MT 1, 3, 4, ...) that
the change invalidates:

```python
from kika.endf.writers.endf_writer import replace_mt_section

donor = kika.read_endf("Fe56_other.endf").mf[3].mt[102]
replace_mt_section("Fe56.endf", donor, 3, output_filepath="Fe56_new.endf",
                   resum_redundant=True)
```

Remove sections, then rebuild the MF 1/451 directory to match:

```python
from kika.endf.writers.section_ops import remove_sections
from kika.endf.writers.update_directory import update_mf1_directory

text = open("Fe56_new.endf").read()
text, n_removed = remove_sections(text, [(4, 2)])      # (MF, MT); MT None = whole MF
open("Fe56_noMF4.endf", "w", newline="").write(text)
update_mf1_directory("Fe56_noMF4.endf")
```

`replace_mf_section` replaces a whole MF; `kika.endf.writers.redundant.recompute_redundant_mf3`
rebuilds the stated sums of an edited tape on its own; `create_mf33_from_covariance` and
`create_mf34_from_covariance` (in `kika.endf.writers`) write a covariance object as MF 33
or MF 34. To write a tape from the model instead (after a computation on it), use
`kika.write(ev, path, format="endf")`.

## Check the covariances

```python
from kika.endf import check_covariances, check_covariance_library

report = check_covariances(kika.read_endf("Fe56.endf"))    # one tape, already read
library = check_covariance_library("evaluations/")         # a directory of tapes
```

`check_covariances` needs the cross-sections, angular distributions and spectra beside the
covariances (MF 3 for MF 33, MF 2 for MF 32, MF 4 for MF 34, MF 5 for MF 35); read the
whole tape, or the checks that need central values say they could not run.

What is checked, with which thresholds, is in {doc}`../methods/covariance/covariance-checks`.

## Download an evaluation

`kika.endf.remote` downloads evaluations from the IAEA Nuclear Data Services and caches
them:

```python
from kika.endf.remote import fetch_endf, download_endf, get_catalog

endf = fetch_endf("Fe56", library="endfb8.1")                  # parsed, cached
path = download_endf("U235", save_to="tapes/", library="jeff4.0")
get_catalog().search("Fe")                                     # what is available
```

When the IAEA site blocks scripted access, `check_access` says so; download the file in a
browser and `import_entry` / `find_browser_download` put it in the cache. `clear_cache`
and `get_cache_info` manage the cache.

## Index a library

```python
from kika.library_index import index_endf

idx = index_endf("evaluations/", recursive=True)
idx.files          # one entry per tape: target, Z, A, MAT, temperature, path
idx.duplicates     # tapes for a target already indexed
idx.unreadable     # tapes whose header could not be read
```

`index_ace` does the same for a directory of ACE files.

## Plot what you read

Every section with data offers `to_plot_data()`, and `kika.plotting.plottable` makes a
plottable series from an ENDF tape, a PENDF, an ACE file, a model or a covariance; see
the plotting guide.
