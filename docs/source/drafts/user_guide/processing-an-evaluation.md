# Processing an evaluation

## The short way

{py:func}`kika.process` processes any evaluation kika knows how to turn into transport
files, with nothing to choose. It reads the tape's header to tell what it is, takes that
kind's route and its tested settings, and verifies every stage:

```python
import kika

def main():
    run = kika.process("n-026_Fe_056.endf")       # 293.6 K unless temperatures=[...]
    run.write("out/", ["ace", "hdf5"])            # out/26056.02c + xsdir, out/Fe56.h5 + cross_sections.xml
    run.report_document().save("out/Fe56.html")   # did it go well, and what was done

if __name__ == "__main__":
    main()
```

| Evaluation | How kika tells | `write` formats |
|---|---|---|
| incident neutron (ENDF or GNDS) | MF 1/451 NSUB = 10, or a GNDS file | `ace`, `hdf5`, `pendf`, `gnds`, `dosimetry` |
| thermal scattering | NSUB = 12 | `ace` (`.t`, one per temperature), `hdf5` |
| photo-atomic | NSUB = 3; its relaxation and electron tapes are found beside it | `ace` (`.p`) |
| photonuclear | NSUB = 0 | `ace` (`.u`) |
| dosimetry (IRDFF-II and the like) | asked for: `kind="dosimetry"` | `ace` (`.y`) |

The options are few, and each is physical: `temperatures`, `heating` (neutron), `name`
and `neutron` (thermal: the OpenMC table name when the tape does not make it plain, and
the principal nuclide's neutron tape), `relaxation` and `electrons` (photo-atomic, when
they are elsewhere), `workers`. An option that does not apply to the evaluation is
refused, not ignored. A tape kika cannot route, or whose thermal table name it cannot
tell, is refused with what it found; kika does not guess.

The report opens with the answer to *did it go well?* and then says what was done: the
checks of every stage, the findings behind any note or warning, the files written and how
each was verified, and the choices kika made for you. Ask for it after writing, since each
file written is one of the stages it reports.

## From the command line

```
kika process n-026_Fe_056.endf --to ace hdf5 -o out/
kika process endfb81/neutrons/ --to ace -o lib/ --temperatures 293.6 600
kika process tsl-HinH2O.endf -o thermal/
```

Each evaluation (or every evaluation in a folder; anything else there is skipped) gets one
line saying how it went, and its report beside the files, `<name>.report.html`. The exit
status is 1 when any evaluation was incomplete. `python -m kika process ...` is the same.

## The staged API

{py:func}`kika.endf.process_endf` takes an ENDF-6 neutron tape through kika's own
processing chain (reconstruction, Doppler broadening, unresolved-resonance probability
tables, optionally heating) and returns a result from which a PENDF, an OpenMC HDF5 file,
ACE tables or a GNDS file with heated styles are written. {py:func}`kika.gnds.process_gnds` does the
same from a GNDS file. No NJOY is involved. What each stage computes, and how it differs
from NJOY, is in {doc}`../methods/processing/processing-chain`.

## Look before processing

{py:func}`kika.endf.inspect_endf_processing` reads MF 1 and MF 2 only and says, in
seconds, which stages apply and which the material is known to fail:

```python
from kika.endf import inspect_endf_processing

info = inspect_endf_processing("n-026_Fe_056.endf")
info["resolved"]          # {'formalisms': ['Reich-Moore'], 'resonances': 320, 'range_eV': [1e-05, 850000.0]}
info["stages"]["urr"]     # {'applicable': False, 'predicted_rejection': 'urr-absent', 'reason': ...}
```

The prediction covers the flags and formalisms kika refuses (LRP = 2, several isotopes in
MF 2, Adler-Adler); a run can still be rejected on its budgets or its verification.

## Process a tape

```python
from kika.endf import process_endf, ProcessingPlan

def main():
    run = process_endf("n-026_Fe_056.endf",
                       ProcessingPlan(temperatures=(293.6, 900.0)))
    for stage in run.stages:
        print(stage.stage, stage.status, stage.category, stage.summary)

    E, sigma = run.cross_sections(900.0)[102]   # eV, b: capture at 900 K
    E0, sigma0 = run.cross_sections()[102]       # the 0 K reconstruction

if __name__ == "__main__":
    main()
```

The broadening and the URR tables run in several processes (up to eight; set
`ProcessingPlan(workers=...)`). **On Windows and macOS, call `process_endf` from under an
`if __name__ == "__main__":` guard**, as above: those platforms start a worker by
importing the main script again, and without the guard the broadening stage is rejected
with `doppler-failed`.

What the plan controls:

- `temperatures`: kelvin, each > 0. 0 K is the reconstruction and is always there;
  `temperatures=()` stops after it.
- `thnmax`: the top of the broadening, in eV, or `"njoy"` (the default) for where NJOY
  BROADR stops by default, the top of the resolved range. Above it the 0 K values are
  used.
- `thinning`: the relative tolerance each temperature's cross-sections are stated to,
  $10^{-3}$ by default (five times tighter below 0.4999 eV, as in NJOY), shared between
  the 0 K source, the broadening and the thinning so that the result stays within it;
  `None` keeps every point of the union grid.
- `urr`, `urr_bins`, `urr_ladders`: the probability tables (on by default, 20 bands; at
  least 64 ladders per energy and more, up to `urr_ladders` = 4096, until the table's
  statistical error is $5\cdot10^{-3}$), computed at the same temperatures.
- `broaden=False`: no broadening; the temperatures then serve the URR tables only.
- `heating=True`: MT 301 (KERMA) and MT 444 (damage) at every temperature, 0 K included,
  and the displacement cross-sections. It tabulates the secondary distributions of the
  whole tape, and is the slow stage.
- `reconstruction`: the {py:class}`~kika.processing.resonances.ReconstructionOptions` of the 0 K
  stage.

A stage after the reconstruction never raises: it is `rejected` with a `category` and a
sentence, and every earlier result is kept. Only the reconstruction raises
(`UnsupportedResonanceError`, `ReconstructionConvergenceError`), since nothing later has
a source without it. `progress=lambda stage, fraction: ...` receives `read`, `decode`,
`reconstruct`, `broaden:293.6`, … , `urr`, `heating` and `write`, with `None` when the
stage starts and then a fraction in (0, 1].

## What comes back

- `run.cross_sections(T)`: `{MT: (E, sigma)}`, float64, every reaction with an MT,
  sums included, each from its threshold. `T` omitted or 0 gives the reconstruction.
- `run.probability_tables`: `{T: tables}` when the URR stage ran, each with `energies`
  and the `cdf`, `total`, `elastic`, `fission`, `capture` and `heating` rows by band;
  `None` otherwise.
- `run.heating`: `{T: {301: (E, h), 444: (E, d)}}` in eV·b when the heating stage ran.
- `run.displacements`: `{T: {"nrt": (E, n), "arc-dpa": (E, n)}}`, the displacement
  cross-sections in b·displacements on MT 444's grid when the heating stage ran (arc-dpa
  only for Fe, Ni, Cu, Pd, Pt and W). No file format carries them.
- `run.temperatures`, `run.thnmax`, `run.stages`, `run.to_dict()` (plain types and
  arrays; the result is also picklable).

## Write the files

```python
run.write_pendf("fe56.pendf")      # one material per temperature, 0 K first
run.write_hdf5("Fe56.h5")          # OpenMC; index=True also updates cross_sections.xml
run.write_ace("ace/")              # one table per temperature: 26056.02c, 26056.09c
run.write_gnds("fe56.gnds.xml")    # eval, recon and one heated style per temperature
```

Each file is written to a temporary name, read back and checked, and only then moved into
place. Each call adds a `write` stage to `run.stages`; its `details["omitted"]` lists what
the format could not carry (the GNDS file has no URR tables and no covariances). The
HDF5 and the ACE tables need at least one broadened temperature.
`write_pendf(temperatures=[0, 900])` writes a subset; so does
`write_ace(directory, temperatures=[900])`. `write_ace` names each table by its ZAID and
the suffix of its temperature, adds its line to `directory/xsdir`, refuses a table the
`xsdir` already names unless `replace=True`, and takes `extension=".80c"` for a single
temperature.

## The report

```python
doc = run.report_document()
doc.save("fe56-report.html")  # the verdict first, then the checks stage by stage
```

The verdict is the worst outcome of every stage and every file written. A **failed**
check is a result not to use, or a file not written. A **warning** is something that can
change a result and needs your judgement (a material disagreement in the evaluation,
negative heating, a probability table that stopped short of its statistical target). A
**note** is something of the evaluation that kika resolved in a fixed, documented way,
with no effect on the output. The numbers behind the checks are folded under *Details*.

## From GNDS

```python
from kika.gnds import process_gnds

run = process_gnds("n-026_Fe_056.endf.gnds.xml", ProcessingPlan(temperatures=(293.6,)))
run.stage("decode").details.get("source_reconstruction")   # e.g. "recon in the file, ignored"
```

The stages, the result and the outputs are those of `process_endf`. A reconstruction or
heated style the file carries is set aside and kika reconstructs from the evaluated
resonance parameters, so that every table has kika's tolerance and verification.

## Dosimetry tables

A run writes a dosimetry table (`.NNy`) beside its continuous-energy one: every
cross-section it processed, the level sums the tape leaves out, KERMA, damage and gas
production, and the MF 9 and MF 10 production cross-sections, numbered
`MT + 1000 (10 + LFS)`.

```python
paths = run.write_dosimetry_ace("ace/")          # ace/26056.02y and its xsdir line
```

A tape with no resonance parameters (a dosimetry evaluation) is written as it is stated,
at its own temperature, without processing:

```python
from kika.endf import dosimetry_ace

dosimetry_ace("IRDFF-II/Nb093.endf", "ace/", extension=".00y")
```

What goes in, and why: {doc}`../methods/processing/dosimetry-tables`.

## Photo-atomic tables

An element's photo-atomic table (`.p`, the EPRDATA layout MCNP and OpenMC read) needs its
three atomic evaluations:

```python
from kika.endf import process_photoatomic

run = process_photoatomic("photoat-026_Fe_000.endf", "atom-026_Fe_000.endf", "e-026_Fe_000.endf")
run.write_ace("photon/")                          # photon/26000.00p and its xsdir line
run.notes                                         # Compton shells joined or dropped, and the like
```

`write_ace` takes `extension=".25p"` for another suffix and `replace=True` to overwrite a
table the `xsdir` already names. The method: {doc}`../methods/processing/photoatomic-tables`.

## Photonuclear tables

A nuclide's photonuclear table (`.u`) is written from its evaluation of the `g`
sublibrary, which states its cross sections pointwise:

```python
from kika.endf import photonuclear_ace

photonuclear_ace("g-026_Fe_056.endf", "photonuclear/")   # photonuclear/26056.01u and its xsdir line
```

`extension=".24u"` gives another suffix and `replace=True` overwrites a table the `xsdir`
already names. What the table leaves out or approximates is the report of
`kika.ace.processing.photonuclear.photonuclearTable`. The method:
{doc}`../methods/processing/photonuclear-tables`.
