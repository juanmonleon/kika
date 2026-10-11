# Plotting nuclear data

kika separates **what** is plotted from **how** it looks. `plottable` turns a source (an
ENDF or PENDF tape, an ACE file, a model, a covariance object, an EXFOR experiment, or a
path to one) into a `PlotItem`: one curve of a physical quantity, in canonical units, with
its provenance. A builder draws items in a style and in the display units you choose.

## One curve, from any source

```python
from kika.plotting import plottable, PlotBuilder

elastic = plottable("Fe56.endf", "cross_section", mt=2)
capture = plottable("Fe56.endf", "cross_section", mt=102)

fig = (PlotBuilder(style="journal")
       .add_data(elastic)
       .add_data(capture)
       .set_scales(log_x=True, log_y=True)
       .set_units(x="MeV")
       .build())
fig.savefig("fe56.pdf")
```

The quantities are `cross_section`, `legendre_coefficient`, `angular_distribution`,
`differential_cross_section`, `relative_uncertainty`, and for photons
`photon_production`, `photon_multiplicity`, `photon_energy`; `spectrum_relative_uncertainty`
for fission spectra. `supported_quantities(source)` lists what a given source can give.
The selector keywords say what to take: `mt`, `order` (Legendre order), `energy` (incident
energy, in eV), `uncertainty=True` to attach the covariance band, `sigma` for its width.

The label is built from the provenance (`JEFF-4.0 Fe56 (n,γ) · MF3 background`), so a curve
says where it came from: here, that the MF 3 of a tape with resonance parameters is only the
background. A quantity that needs the reconstructed cross-section (a differential
cross-section in the resonance region, for instance) raises `ReconstructionRequired`
instead of using the background as if it were the cross-section; attach a PENDF, or pass
`reconstructed=False` to plot the background knowingly.

## Compare evaluations

```python
from kika.plotting import ComparisonBuilder

fig = (ComparisonBuilder(grid_strategy="union")
       .set_reference(plottable("Fe56_jeff.endf", "cross_section", mt=2))
       .add_comparison(plottable("Fe56_endfb.endf", "cross_section", mt=2))
       .build())
```

The difference is computed on the union of the two grids (`grid_strategy`: also the
reference's or the comparison's), interpolating each curve under the law its data states,
or lin-lin when it states none (the law of a PENDF or an ACE file; `interpolation=` to
choose); `set_difference_panel` chooses a relative or
absolute difference panel.
`set_group_average()` on either builder overlays group averages (see
{doc}`../methods/processing/group-averaging`), which is how to compare evaluations in the
resonance region.

## Covariances

```python
from kika.cov.legendre_covariance import LegendreCovariance
from kika.plotting import HeatmapBuilder

cov = LegendreCovariance.from_endf("Fe56.endf")        # MF 34
fig = HeatmapBuilder().add_heatmap(cov.to_heatmap_data(26056, 2, [1, 2, 3])).build()
fig = cov.plot_uncertainties(26056, 2, [1, 2])         # relative uncertainty of a_1, a_2
```

`CrossSectionCovariance` (MF 33 and multigroup files) offers the same `to_heatmap_data`,
`plot_covariance_heatmap` and `plot_uncertainties`. `heatmap_grid` puts several heatmaps
on one colour scale.

## Styles and units

`style_names()` lists the built-in styles (`classic`, `signature`, `journal`, `isotope`,
`mineral`, and dark variants); `get_style`, `compose_style` and `register_style` derive and
register your own. `set_units(x=..., y=...)` converts the display units; the data stay in
canonical units (eV, b). Every section object of the ENDF parse layer also offers
`to_plot_data()`, for the lower-level plot-data classes.
