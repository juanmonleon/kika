# Quickstart

## Installation

kika needs Python 3.12 or later:

```bash
pip install kika-nd
```

`pip install --upgrade kika-nd` upgrades it, and `pip show kika-nd` says which version
is installed.

## Read an evaluation

{py:func}`kika.read` opens an ENDF (Evaluated Nuclear Data File), GNDS, ACE, G4NDL or
OpenMC HDF5 file, telling the format from the content, and gives back the same model
every time:

```python
import kika

ev = kika.read("n-026_Fe_056.endf")
E, xs = ev.cross_section(2)     # elastic, as two NumPy arrays (eV, b)
print(ev.summary())             # what the file contained
ev.report.isClean               # False when something could not be read into the model
```

{doc}`reading_files` explains the model and the report.

## Check it

{py:func}`kika.check` checks a file, a folder or a list of files against the format,
mathematics and physics, covariances included, and changes nothing:

```python
report = kika.check("n-026_Fe_056.endf")
print(report)                         # one line per finding, worst first
report.to_html("Fe56.check.html")
```

## Process it for transport

{py:func}`kika.process` takes an evaluation to transport files with kika's own
processing (no NJOY), checks every stage, and writes a report of the run:

```python
def main():
    run = kika.process("n-026_Fe_056.endf")        # 293.6 K unless temperatures=[...]
    run.write("out/", ["ace", "hdf5"])             # ACE + xsdir, OpenMC HDF5 + cross_sections.xml
    run.report_document().save("out/Fe56.html")

if __name__ == "__main__":
    main()
```

The processing runs in several processes, so on Windows a script that calls it needs the
`if __name__ == "__main__":` guard. From the command line, the same is
`kika process n-026_Fe_056.endf --to ace hdf5 -o out/`.

## Convert it

{py:func}`kika.write` writes the model in another format:

```python
kika.write(ev, "Fe56.gnds.xml", format="gnds")
```

## Perturb it

{py:func}`kika.sampling.model_perturbation.perturbFromModel` draws perturbed
evaluations from the covariances the file states:

```python
from kika.sampling.model_perturbation import perturbFromModel

run = perturbFromModel("n-026_Fe_056.endf", {33: None, 34: [2]}, 100, seed=1,
                       outputDir="samples")
run.paths("endf-delta")    # samples/0000/n-026_Fe_056_0000.endf, ...
```

On a tape whose resolved resonances are stated as parameters, the run first
reconstructs them with NJOY, so that the factors act on the resonance cross-section;
point the `NJOY_EXECUTABLE` environment variable at NJOY, or pass `njoyExe=`.

## Transport-code files

```python
inp = kika.read_mcnp("model.i")          # materials and PERT cards
mctal = kika.read_mctal("mctal")         # tallies as labelled xarray tables
mesh = kika.read_meshtal("meshtal")      # FMESH, text or the MCNP 6.3 HDF5 runtape
sens = kika.compute_sensitivity(inputfile="model.i", mctalfile="mctal", tally=4,
                                zaid=26056, label="Fe-56")
```

The tutorials take these one by one: {doc}`tutorials/input_files`,
{doc}`tutorials/mctal_files`, {doc}`tutorials/mesh_tallies`, {doc}`tutorials/pert_cards`
and {doc}`tutorials/sensitivity_analysis`.
