# Processing an evaluation

{py:func}`kika.endf.process_endf` takes an ENDF-6 neutron tape through kika's own
processing chain (reconstruction, Doppler broadening, unresolved-resonance probability
tables, optionally heating) and returns a result from which a PENDF, an OpenMC HDF5 file
or a GNDS file with heated styles is written. {py:func}`kika.gnds.process_gnds` does the
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
- `thinning`: the relative tolerance each temperature's grid is thinned to, $10^{-3}$ by
  default; `None` keeps every point.
- `urr`, `urr_bins`, `urr_ladders`: the probability tables (on by default, 20 bands and
  64 ladders), computed at the same temperatures.
- `broaden=False`: no broadening; the temperatures then serve the URR tables only.
- `heating=True`: MT 301 (KERMA) and MT 444 (damage) at every temperature, 0 K included.
  It tabulates the secondary distributions of the whole tape, and is the slow stage.
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
- `run.temperatures`, `run.thnmax`, `run.stages`, `run.to_dict()` (plain types and
  arrays; the result is also picklable).

## Write the files

```python
run.write_pendf("fe56.pendf")      # one material per temperature, 0 K first
run.write_hdf5("Fe56.h5")          # OpenMC; index=True also updates cross_sections.xml
run.write_gnds("fe56.gnds.xml")    # eval, recon and one heated style per temperature
```

Each file is written to a temporary name, read back and checked, and only then moved into
place. Each call adds a `write` stage to `run.stages`; its `details["omitted"]` lists what
the format could not carry (the GNDS file has no URR tables and no covariances). The
HDF5 needs at least one broadened temperature. `write_pendf(temperatures=[0, 900])`
writes a subset. A PENDF cannot be written from a GNDS evaluation.

## The report

```python
doc = run.report_document()
print(doc.to_text())          # the reconstruction report and a "Processing stages" section
doc.save("fe56-report.html")
```

## From GNDS

```python
from kika.gnds import process_gnds

run = process_gnds("n-026_Fe_056.endf.gnds.xml", ProcessingPlan(temperatures=(293.6,)))
run.stage("decode").details.get("source_reconstruction")   # e.g. "recon in the file, ignored"
```

The stages, the result and the outputs are those of `process_endf`. A reconstruction or
heated style the file carries is set aside and kika reconstructs from the evaluated
resonance parameters, so that every table has kika's tolerance and verification.
