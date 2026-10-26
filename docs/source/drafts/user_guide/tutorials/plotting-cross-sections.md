# Plotting cross-sections

How to plot the cross-sections of an evaluation, compare two evaluations, read the
resonance region correctly, and draw the uncertainty the evaluation declares. The
examples use the Fe-56 evaluations of ENDF/B-VIII.1 and JEFF-4.0; the outputs shown are
what they printed. What each plotting object does is in {doc}`../plotting`.

## Getting the evaluations

{py:func}`kika.endf.download_endf` fetches an evaluation from the IAEA and keeps a copy in
a local cache, so a second call works offline:

```python
from kika.endf import download_endf

b81_path = download_endf("Fe56", save_to=".", library="endfb8.1")
jeff_path = download_endf("Fe56", save_to=".", library="jeff4.0")
print(b81_path, jeff_path)
```

```text
Fe56_endfb8.1_n.endf Fe56_jeff4.0_n.endf
```

## Three reactions from one evaluation

Read the tape once with {py:func}`kika.endf.read_endf` and ask `plottable` for one curve
per reaction. A `PlotBuilder` draws them:

```python
from kika.endf import read_endf
from kika.plotting import plottable, PlotBuilder

b81 = read_endf(b81_path)
elastic = plottable(b81, "cross_section", mt=2)
capture = plottable(b81, "cross_section", mt=102)
inelastic = plottable(b81, "cross_section", mt=4)
for item in (elastic, capture, inelastic):
    print(item.data.label)

fig = (PlotBuilder(style="journal")
       .add_data(elastic).add_data(capture).add_data(inelastic)
       .set_scales(log_x=True, log_y=True)
       .set_units(x="MeV")
       .build())
fig.savefig("fe56-mf3.png")
```

```text
ENDF/B-VIII.1 Fe56 (n,el) · MF3 background
ENDF/B-VIII.1 Fe56 (n,γ) · MF3 background
ENDF/B-VIII.1 Fe56 (n,n') · MF3 background
```

`plottable` also takes a path (`plottable("Fe56_endfb8.1_n.endf", ...)`), but then reads
the tape on every call; a tape read once serves every curve.

## The resonance region: background or cross-section

The labels say `MF3 background`, and the figure shows why that matters: below 0.85 MeV,
the resolved-resonance region of this evaluation, the elastic curve is 0 b at 1 eV,
$5.4\cdot10^{-4}$ b at 1 keV and $1.1\cdot10^{-4}$ b at 100 keV, where the elastic
cross-section of Fe-56 is 12 b, 9.3 b and 3.6 b. Fe-56 gives its resolved resonances as
parameters in MF 2, and MF 3 then holds only the background that is added to the
resonance contribution. The cross-section itself has to be reconstructed from the
parameters.

`plottable` does not reconstruct on its own. Asking for the reconstructed cross-section
of a tape that does not carry one raises
`ReconstructionRequired`:

```python
from kika.plotting import ReconstructionRequired

try:
    plottable(b81, "cross_section", mt=2, reconstructed=True)
except ReconstructionRequired as exc:
    print(exc)
```

```text
MT2: this evaluation has resonance parameters (MF2), so MF3 is only the background. Reconstruct it (kika.processing.njoy_reconstruct) and set endf.pendf, or pass reconstructed=False to plot the background.
```

`reconstructed=False` plots the background knowingly; the default (`"auto"`) plots it
and labels it as such. A differential cross-section, which is the angular distribution
times the cross-section, is refused on the background by default, since it would not be
a physical quantity.

kika reconstructs the cross-sections itself with {py:func}`kika.process`, without NJOY.
`temperatures=()` stops at the 0 K reconstruction; each PENDF (pointwise ENDF) took
about 20 s here (measured 2026-10-26):

```python
import kika

def main():
    for path, out in ((b81_path, "pendf_b81"), (jeff_path, "pendf_jeff")):
        run = kika.process(path, temperatures=())
        print(run.write(out, ["pendf"]))

if __name__ == "__main__":
    main()
```

```text
[WindowsPath('pendf_b81/Fe56.pendf')]
[WindowsPath('pendf_jeff/Fe56.pendf')]
```

The `if __name__ == "__main__":` guard is needed on Windows and macOS, where the
processing starts worker processes that import the script again (see
{doc}`/user_guide/processing-an-evaluation`).

Attach the reconstruction to the tape with
{py:func}`kika.processing.njoy_pendf_cache.read_pendf_mf3_sections`, and every
`plottable` call on that tape uses it:

```python
from kika.processing.njoy_pendf_cache import read_pendf_mf3_sections

b81.pendf = read_pendf_mf3_sections("pendf_b81/Fe56.pendf")
elastic = plottable(b81, "cross_section", mt=2)
print(elastic.data.label)
```

```text
ENDF/B-VIII.1 Fe56 (n,el) · reconstructed
```

Attaching keeps the tape's covariances (MF 33) at hand for the uncertainty band below.
A PENDF can also be plotted on its own, `plottable("pendf_b81/Fe56.pendf",
"cross_section", mt=2)`; a PENDF written by kika does not name its library, so pass
`evaluation="ENDF/B-VIII.1"` to have it in the label.

## Comparing ENDF/B-VIII.1 with JEFF-4.0

`ComparisonBuilder` draws a reference and one or more comparisons, and
`set_difference_panel` adds their difference below:

```python
from kika.plotting import ComparisonBuilder

jeff = read_endf(jeff_path)
jeff.pendf = read_pendf_mf3_sections("pendf_jeff/Fe56.pendf")

capture_jeff = plottable(jeff, "cross_section", mt=102)
capture_b81 = plottable(b81, "cross_section", mt=102)
print(capture_jeff.data.label)

fig = (ComparisonBuilder(style="journal")
       .set_reference(capture_jeff)
       .add_comparison(capture_b81)
       .set_difference_panel(mode="relative")
       .set_scales(log_x=True, log_y=True)
       .set_labels(x_label="Incident energy (eV)")
       .build())
fig.savefig("fe56-capture-b81-vs-jeff.png")
```

```text
INDL/A Fe56 (n,γ) · reconstructed
```

The label reads `INDL/A`, not `JEFF-4.0`: JEFF-4.0 adopts the IAEA INDEN evaluation of
Fe-56, and the tape says so in its MF 1 header. The label states what the file declares.
To show the library name instead, pass it: `plottable(jeff, "cross_section", mt=102,
evaluation="JEFF-4.0")`, or `label=` for the whole legend entry.

The two curves differ by at most 3.4 % below 1 keV and are identical above 0.86 MeV. In between, the difference panel is
a band of spikes down to −100 %: the resonances of the two evaluations sit at slightly
different energies, and a point-by-point difference compares the peak of one with the
valley of the other.

`ComparisonBuilder` has no `set_units`: the axes stay in the reference's units (eV and
b). Set the x label by hand, as above: without `set_labels(x_label=...)` the difference
panel is labelled `Energy (MeV)` although its axis is in eV.

## Group averages in the resonance region

Through the resonances, compare averages over energy groups instead of points.
{py:func}`kika.processing.group_averaging.resonance_group_average` computes them from
the pointwise curve; attached to a curve as its `group_average_overlay`, the builders
draw them and the difference panel compares them group by group:

```python
import numpy as np
from kika.processing.group_averaging import resonance_group_average

edges = np.geomspace(1e3, 8.5e5, 31)          # 30 groups, 1 keV to 0.85 MeV, in eV

def with_group_averages(item, edges):
    edges, averages = resonance_group_average(item.data.x, item.data.y, edges,
                                              weighting="lethargy")
    item.data.metadata["group_average_overlay"] = {
        "edges": edges, "xs": averages,
        "bounds_used": (edges[0], edges[-1]), "weighting": "lethargy",
    }
    print(item.data.label, np.round(averages[:4], 4))
    return item

fig = (ComparisonBuilder(style="journal")
       .set_reference(with_group_averages(capture_jeff, edges))
       .add_comparison(with_group_averages(capture_b81, edges))
       .set_group_average(main_display="average")
       .set_difference_panel(mode="relative")
       .set_scales(log_x=True, log_y=True)
       .set_limits(x_lim=(1e3, 8.5e5))
       .set_labels(x_label="Incident energy (eV)")
       .build())
fig.savefig("fe56-capture-groups.png")
```

```text
INDL/A Fe56 (n,γ) · reconstructed [0.7996 0.0074 0.0058 0.0063]
ENDF/B-VIII.1 Fe56 (n,γ) · reconstructed [0.7993 0.0071 0.0055 0.006 ]
```

`weighting="lethargy"` weights each group by a $1/E$ flux; `"constant"` by a flat one.
How the average is computed is in {doc}`../../methods/processing/group-averaging`.
`main_display="average"` draws the steps in place of the pointwise curve inside the
averaged span; `"both"` (the default) keeps the pointwise curve faded underneath. The
difference panel now shows one step per group: ENDF/B-VIII.1 is 4 % to 10 % below
JEFF-4.0 from 1.3 keV to 15 keV, 27 % above it in the group from 23 keV to 29 keV, and
within 5 % of it above 29 keV.

## The uncertainty band

`uncertainty=True` attaches the relative standard deviation that MF 33 declares for the
reaction, per covariance group, and the builder draws it as a band around the curve.
`sigma` sets its width in standard deviations:

```python
inelastic = plottable(b81, "cross_section", mt=4, uncertainty=True, sigma=2)
for energy in (2e6, 5e6, 14e6):
    k = np.searchsorted(inelastic.band.x, energy)
    print(f"{energy / 1e6:g} MeV: {100 * inelastic.band.relative_uncertainty[k]:.1f} %")

fig = (PlotBuilder(style="journal", figsize=(7, 4))
       .add_data(inelastic)
       .set_units(x="MeV", y="mb")
       .set_limits(x_lim=(0.8, 20), y_lim=(0, None))
       .build())
fig.savefig("fe56-inelastic-band.pdf")
```

```text
2 MeV: 4.5 %
5 MeV: 4.2 %
14 MeV: 6.9 %
```

The figure shows the inelastic cross-section from 0.8 MeV to 20 MeV in mb, with a grey
band of two standard deviations around it. The band is the relative uncertainty of the
group each energy falls in, applied to the curve. To plot the relative uncertainty itself, ask for the `relative_uncertainty`
quantity: `plottable(b81, "relative_uncertainty", mt=4)` gives it per group, in percent.

## Units, styles and saving

`set_units(x=..., y=...)` converts the axes for display (`"eV"`, `"keV"`, `"MeV"`, `"b"`,
`"mb"`); the curves stay in eV and b, and limits passed to `set_limits` are in the display
units. `style_names()` lists the built-in styles:

```python
from kika.plotting import style_names

print(style_names())
```

```text
['classic', 'signature', 'journal', 'isotope', 'mineral', 'classic-dark', 'signature-dark']
```

`build()` returns a Matplotlib `Figure`, saved with `fig.savefig`: a `.pdf` or `.svg`
for a manuscript, a `.png` for a screen. `PlotBuilder(figsize=(7, 4), dpi=150)` sets
the size, in inches, and the resolution of a raster file.
