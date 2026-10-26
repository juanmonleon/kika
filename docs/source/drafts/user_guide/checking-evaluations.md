# Checking evaluations

## The short way

{py:func}`kika.check` checks one evaluation, a folder of them or a list of files, against
the ENDF-6 format, mathematics and physics: the data of every file (MF 1 to MF 28) and the
covariances (MF 31-35, MF 40). Nothing in the files is changed.

```python
import kika

report = kika.check("n-094_Pu_243.endf")
print(report)                         # one line per finding, worst first
report.to_html("pu243.check.html")    # the report, one self-contained page
```

```text
Evaluation check of n-094_Pu_243.endf (MAT 9449): 2 defects, ... notes
  [defect] MF1 MT455: delayed_spectra_missing -- MF1/455 gives a delayed nu-bar (0.0162) but there is no delayed neutron spectrum (MF5/455) ...
  [defect] MF2 MT151 range[1]: resonance_spin -- unresolved range 1: 1 (l, J) state a neutron on spin 3.5 cannot form; first J = 3.5 at l = 0 (allowed: 3, 4)
  ...
```

Each finding has a level: **defect** (the data are not what the format or physics allows,
and using them gives a wrong result), **warning** (it can change a result and needs
judgement) or **note** (true and worth knowing, not a fault). What each check tests and
when it gives each level is in the report's legend.

## Choosing what to check

| Call | Checks |
|---|---|
| `kika.check(path)` | data and covariances |
| `kika.check(path, what="data")` | MF 1-28 only |
| `kika.check(path, what="covariances")` | MF 31-35, 40 only |
| `kika.check(path, mf=[3])` | MF 3 only (and what MF 3 is checked against, read but not checked) |
| `kika.check(path, mf=[3, 33])` | MF 3's data and MF 33's covariances |
| `kika.check(path, deep=True)` | also the cross-sections reconstructed at 0 K: positivity, sums, $1/v$ |

The tape is read lazily: only the files the asked checks need are parsed, so
`mf=[3]` on a large tape costs its MF 3. `deep=True` costs a reconstruction (seconds for
most tapes).

A GNDS file is checked as the ENDF-6 tape kika writes from it; the report opens with a
note that says so.

## Reading the report

A {py:class}`~kika.endf.checks.CheckReport` is a tuple of findings with:

```python
report.defects                     # the findings at that level
report.at_least("warn")            # warnings and defects
report.by_check("xs_sum")          # one check's findings
report.counts()                    # {(level, check): n}
f = report.defects[0]
f.check, f.level, f.summary        # what, how bad, in one line
f.location.mf, f.location.mt       # where (also k, zap, t and the covariance indices)
f.evidence                         # the numbers the rule measured
report.to_html("page.html", level="note")   # include the notes too
report.to_dict()                   # plain data, for a program
```

The page opens with the verdict and the count per level, then the findings grouped by
check, the method (how each level is drawn) and the legend.

## A whole library

A folder, or a list of files, gives a {py:class}`~kika.endf.checks.LibraryReport` with
one row per tape:

```python
if __name__ == "__main__":
    library = kika.check("endfb81/neutrons/", library="endfb81")
    print(library)                          # census by check: findings and tapes
    library.to_html("endfb81.check.html")   # index of files, summaries, findings by tape
    library.write("tables/")                # summary, files and findings as TSV
    library.report("n-026_Fe_056.endf")     # one tape's report
```

Tapes are checked on several processes at once. The count is chosen as for processing:
`workers=N` if given, otherwise the `KIKA_WORKERS` environment variable, otherwise the
machine's cores up to eight, and no more than one per GB of free memory when `psutil` is
installed. `workers=1` checks in series in the calling process. The report is the same
whatever the count. On Windows and macOS each worker imports the calling script again, so
the call must be under `if __name__ == "__main__":`; started without it, the check stops
with a message that says so.

Files of the folder that are not ENDF-6 (README files, LEAPR inputs beside thermal
scattering tapes) are left out. A tape that cannot be read is a row of its own, and the
walk goes on.

`progress=False` silences the progress line on stderr; `on_tape`, `on_result` and
`should_stop` are callbacks for a caller that shows its own progress or offers a cancel
button.

## From the command line

```bash
kika check n-094_Pu_243.endf                       # data and covariances
kika check n-026_Fe_056.endf --what data --mf 3 4
kika check endfb81/neutrons/ --library endfb81 -o reports/ --workers 4
kika check n-092_U_235.endf --deep --level note    # list the notes too
```

Each file, or each folder as a library, gets one HTML page in the output folder
(`<name>.check.html`). One line per evaluation says what was found. The exit status is 1
when any evaluation has a finding at `--fail-on` (`defect` by default; `warn`, `note` or
`never`) or could not be checked, 0 otherwise, 2 for a usage error, so a script or a CI job
can stop on a defective evaluation.

## Checks when processing

{py:func}`kika.process` runs the static data checks on the evaluation it processes, and its
report lists their defects and warnings under *Reading the evaluation* (see
{doc}`processing-an-evaluation`). The processing still runs and verifies every file it
writes; the check says what in the evaluation a result may inherit.

## Covariances alone

{py:func}`kika.endf.check_covariances` and {py:func}`kika.endf.check_covariance_library`
remain the entry points of the covariance checks on their own, with the reports they always
gave; `kika.check(path, what="covariances")` runs the same checks through the same door as
the data checks.
