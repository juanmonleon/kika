# Working with the model

Whatever format a file is in (ENDF-6, GNDS, ACE, G4NDL, OpenMC HDF5, or a
thermal-scattering table in any of them), {py:func}`kika.read` returns the same kind of
object, a `ReactionSuite`. A processed file (ACE, HDF5) reads into its processed styles: one `heated`
style per temperature and sampling tables under `MonteCarlo`, with no `eval` form. This
page is the everyday use of it.

## Find what you need

```python
import kika

ev = kika.read("Fe56.endf")
print(ev.summary())

capture = ev.reactions[102]            # by MT
capture = ev.reactions["MT102"]        # by label: "MT<n>" for a suite read from ENDF
capture.crossSection.keys()            # the forms it holds, by label: ['eval']
form = capture.crossSection["eval"]    # a Regions1d (one function per interpolation region)
E, xs = ev.cross_section(102)          # arrays, from the 'eval' form unless form= says otherwise
```

Names inside the model are GNDS's (`crossSection`, `outerDomainValue`, `XYs1d`); methods
are Python's (`cross_section`, `summary`). A stated sum (MT 1, MT 3, MT 4, ...) is in
`ev.sums`, linked to its parts, not in `ev.reactions`; `ev.reactionByENDF_MT(1)` searches
every container. The covariances are on
`ev.covarianceSuite`, the resonance parameters on `ev.resonances`, the particles in
`ev.PoPs`.

## Read the report

```python
ev.report.isClean          # True when nothing was lost, approximated or refused
ev.report.approximations   # read these first: an approximation looks like data
ev.report.losses
ev.report.unsupported
```

## Forms and labels

A container can hold several forms: the evaluation (`eval`), a reconstruction (`recon`),
a perturbed realisation (`realization-0001`). `ev.styles` records how each was derived.
A computation adds its result as a new form and leaves `eval` as it was.

## Write it out

```python
report = kika.write(ev, "Fe56.gnds.xml")              # GNDS; covariances as a second file
report = kika.write(ev, "Fe56_out.endf", format="endf")
report = kika.write(ev, "g4ndl_lib/", format="g4ndl")    # needs a pointwise 'recon' form
report = kika.write(ev, "openmc_lib/", format="hdf5", temperatures=[293.6])   # processed first
```

`kika.write` returns the conversion report of the writing: what the target format could not
carry. Writing an evaluation as HDF5
processes it first; {doc}`processing-an-evaluation` does the same in stages and also
writes a PENDF.

## Moving from the flat classes

`kika.nuclear_data.CrossSection`, `AngularDistribution`, `ResonanceParameters` and
`NuclideInfo` predate the model. They are deprecated, warn once when used, and are removed
in kika-nd 1.0. Read the file with `kika.read` instead and take the data from the suite;
`kika.nuclear_data.model.interop` gives a model node's data in the plain shape the flat
classes used, for code that still expects it.
