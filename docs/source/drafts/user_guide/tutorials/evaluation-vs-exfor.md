# Comparing an evaluation with experiment (EXFOR)

How to find the measurements of a reaction in EXFOR, look at one dataset, and plot it
against an evaluation: the elastic angular distributions of Fe-56 between 1 MeV and
5 MeV against ENDF/B-VIII.1, then the measured capture cross-section against the
evaluated one. The outputs shown are what the code printed with the X4Pro release of
2025-11-05. The EXFOR classes are described in {doc}`../exfor`, the plotting objects in
{doc}`../plotting`.

## The database

kika reads EXFOR from X4Pro, the SQLite form of the EXFOR library distributed by the
IAEA Nuclear Data Section. kika does not ship it: download it from the IAEA and point
kika at the file once per session (or set the environment variable
`KIKA_X4PRO_DB_PATH`):

```python
import kika.exfor as exfor

exfor.configure(db_path="x4sqlite1.db")
```

## Finding the angular distributions

`X4ProDatabase.search_datasets` lists every dataset that matches a set of filters, with
its metadata, without reading the data. Here, Fe-56 angular distributions (`DA`) of
elastic scattering (MT 2) whose incident energies overlap 1 MeV to 5 MeV:

```python
import pandas as pd

db = exfor.X4ProDatabase()
hits = db.search_datasets(targets=["Fe-56"], quantities=["DA"], mts=[2],
                          energy_min_mev=1.0, energy_max_mev=5.0)
print(pd.DataFrame(hits)[["dataset_id", "author", "year", "mf", "e_min", "e_max", "ndat"]])
```

```text
  dataset_id      author  year   mf  e_min  e_max  ndat
0   11276009     Rodgers  1967    4   2.33   2.33     5
1   11708006      Kinney  1968    4   4.60   7.55    85
2   14462002     Ramirez  2017    4   1.30   7.96   192
3   30463020  Schweitzer  1978    4   3.40   3.40    12
4   30463022  Schweitzer  1978  154   3.40   3.40     6
5  400750021     Morozov  1972    4   1.80   1.80    17
6  400750022     Morozov  1972    4   1.80   1.80    17
```

Energies are in MeV. A target is spelled as X4Pro spells it, `Fe-56`; `Fe56` finds
nothing here. `30463022` is a set of Legendre coefficients fitted to the measurement
(its reaction code ends in `DA,,LEG`), not points. `read_all_exfor` loads the
point-wise angular distributions; `group_by_energy=False` keys them by dataset, and the
energy window, in MeV, keeps every dataset whose energies overlap it:

```python
data = exfor.read_all_exfor(target="Fe-56", mt=2, energy_range=(1.0, 5.0),
                            group_by_energy=False)
print(list(data))
```

```text
['11276009', '11708006', '14462002', '30463020', '400750022', '400750021']
```

With the default `group_by_energy=True`, the same call returns a dictionary from each
incident energy, in MeV, to the datasets measured there.

## One dataset: metadata, frame and units

Each item is an `ExforAngularDistribution`:

```python
ramirez = data["14462002"]
print(ramirez)
print(ramirez.exfor_url)
print(ramirez.angle_frame, ramirez.units)
print(ramirez.method)
```

```text
ExforAngularDistribution
  Entry:       14462002
  Label:       Ramirez (2017)
  Target:      Fe56 (ZAID: 26056)
  Reaction:    26-FE-56(N,EL)26-FE-56,,DA
  Frame:       LAB
  Energies:    15 (1.3 - 7.96 MeV)
  Data points: 192
https://www-nds.iaea.org/exfor/14462.002
LAB {'energy': 'MeV', 'angle': 'deg', 'cross_section': 'b/sr'}
{'type': 'TOF', 'energy_resolution_input': {'distance': {'value': 4.0, 'unit': 'm'}, 'time_resolution': {'value': 1.0, 'unit': 'ns'}, 'source': 'file', 'geometry_source': 'file'}}
```

The measurement is in the laboratory frame, in degrees and b/sr. `method` holds what the
dataset says about the experiment: a time-of-flight measurement with a 4 m flight path
and a 1 ns timing resolution. `to_dataframe` gives the points at one energy (within
0.05 MeV by default), in the units asked for:

```python
print(ramirez.to_dataframe(energy=2.0).head(4))
```

```text
   energy  angle    value    error
0     2.0  30.52  0.54848  0.05030
1     2.0  40.66  0.38323  0.03515
2     2.0  50.79  0.22779  0.02089
3     2.0  60.89  0.12873  0.01181
```

`angle_unit="cos"` and `cross_section_unit="mb/sr"` convert the columns;
`decompose_uncertainty=True` splits `error` into its statistical and systematic parts.

ENDF/B-VIII.1 gives the elastic angular distribution of Fe-56 in the centre-of-mass
frame. `convert_to_cm` returns a copy in that frame, with the angles and the cross-sections
transformed by two-body kinematics:

```python
ramirez_cm = ramirez.convert_to_cm()
print(ramirez_cm.angle_frame)
print(ramirez_cm.to_dataframe(energy=2.0).head(4))
```

```text
CM
   energy      angle     value     error
0     2.0  31.044709  0.531850  0.048775
1     2.0  41.333220  0.372982  0.034210
2     2.0  51.590588  0.222705  0.020424
3     2.0  61.792735  0.126516  0.011607
```

## Against the evaluation

The evaluated differential cross-section is the angular distribution (MF 4) times the
cross-section (MF 3), so it needs the reconstructed cross-section: Fe-56 gives its
resonances as parameters, and its MF 3 alone is only a background
({doc}`plotting-cross-sections` shows the reconstruction; here the PENDF it writes is
attached with {py:func}`kika.processing.njoy_pendf_cache.read_pendf_mf3_sections`, to the
tape read by {py:func}`kika.endf.read_endf`). `plottable` gives both curves, the measurement converted to the centre-of-mass
frame by `frame="CM"`. The incident energy is in eV:

```python
from kika.endf import read_endf
from kika.plotting import plottable, PlotBuilder
from kika.processing.njoy_pendf_cache import read_pendf_mf3_sections

b81 = read_endf("Fe56_endfb8.1_n.endf")
b81.pendf = read_pendf_mf3_sections("pendf_b81/Fe56.pendf")

evaluated = plottable(b81, "differential_cross_section", mt=2, energy=2.0e6)
measured = plottable(ramirez, "differential_cross_section", energy=2.0e6, frame="CM")
print(evaluated.data.label, "|", evaluated.data.provenance.frame)
print(measured.data.label, "|", measured.data.provenance.frame)

fig = (PlotBuilder(style="journal")
       .add_data(evaluated).add_data(measured)
       .set_scales(log_y=True)
       .set_units(y="mb/sr")
       .build())
fig.savefig("fe56-elastic-2MeV.png")
```

```text
ENDF/B-VIII.1 Fe56 (n,el) @ 2 MeV · reconstructed | CM
Ramirez (2017) | CM
```

The figure shows the evaluated curve against $\mu = \cos\theta$ in the centre-of-mass
frame and the 14 measured points with their error bars. Without the attached
reconstruction the first call raises `ReconstructionRequired`.
`plottable(b81, "angular_distribution", mt=2, energy=2.0e6)` gives the normalised
angular distribution instead, which does not need it.

Both curves are arrays on the same abscissa, so their ratio at the measured angles is one
interpolation:

```python
import numpy as np

ratio = measured.data.y / np.interp(measured.data.x, evaluated.data.x, evaluated.data.y)
print(np.round(ratio, 2))
```

```text
[1.27 1.11 0.94 0.91 0.97 1.   0.94 0.78 0.71 0.88 1.32 1.45 1.42 1.28]
```

A time-of-flight measurement at 2 MeV averages over a spread of incident energies.
`resolution=(flight_path_m, delta_t_ns)` folds the evaluated curve with that spread,
reading the timing as a FWHM (see {doc}`../../methods/processing/differential-cross-sections`);
the values recorded in `ramirez.method` serve:

```python
folded = plottable(b81, "differential_cross_section", mt=2, energy=2.0e6,
                   resolution=(4.0, 1.0))
print(folded.data.label)
ratio = measured.data.y / np.interp(measured.data.x, folded.data.x, folded.data.y)
print(np.round(ratio, 2))
```

```text
ENDF/B-VIII.1 Fe56 (n,el) @ 2 MeV · reconstructed, folded, TOF 4 m / 1 ns
[0.81 0.75 0.7  0.69 0.73 0.75 0.73 0.65 0.62 0.74 1.04 1.12 1.12 1.04]
```

## Against an ACE file

When the evaluation is at hand as an ACE file, the comparison needs no reconstruction:
the ACE file carries the cross-section. {py:func}`kika.process` writes one (it took
3 minutes for Fe-56 here, measured 2026-10-26):

```python
import kika

def main():
    kika.process("Fe56_endfb8.1_n.endf").write("ace", ["ace"])    # ace/26056.02c

if __name__ == "__main__":
    main()
```

{py:func}`kika.ace.parsers.parse_ace.read_ace` reads it, and `compare_with_ace` draws
the measurement, converted to the centre-of-mass frame, over the ACE angular distribution
at one energy, here in **MeV**:

```python
from kika.ace.parsers.parse_ace import read_ace

ace = read_ace("ace/26056.02c")
ax = ramirez.compare_with_ace(ace, energy=2.0)
ax.figure.savefig("fe56-elastic-2MeV-ace.png")
```

It returns the Matplotlib axes, with the ACE curve labelled by its cross-section at that
energy (`ACE (sigma=1.8521 b)`) and the measured points with their error bars.
`plottable(ace, "differential_cross_section", mt=2, energy=2.0e6)` gives the same ACE
curve as a `PlotItem`, for the builders.

## Cross-sections

`query_cross_sections` returns the cross-section datasets of a reaction as
`ExforCrossSection` objects. The energy window, in MeV, keeps every dataset whose
energies overlap it:

```python
db = exfor.X4ProDatabase()
capture = db.query_cross_sections(target="Fe-56", mt=102, energy_range=(1e-8, 1.0))
for xs in sorted(capture, key=lambda xs: xs.dataset_id):
    print(xs.dataset_id, xs.label, xs.num_data_points, xs.reaction["notation"])
```

```text
11575010 Macklin (1964) 33 26-FE-56(N,G)26-FE-57,,SIG
20370006 Ernst (1970) 464 26-FE-56(N,G)26-FE-57,,SIG,,TTA
20370007 Ernst (1970) 370 26-FE-56(N,G)26-FE-57,,SIG,,TTA
23103002 Wang (2010) 5 26-FE-56(N,G)26-FE-57,,SIG,,SPA
30355003 Allen (1976) 27 26-FE-56(N,G)26-FE-57,,SIG,,AV
31772004 Firestone (2017) 1 26-FE-56(N,G)26-FE-57,,SIG
322340091 Litvinskij (1994) 1 26-FE-56(N,G)26-FE-57,,SIG,,,DERIV
322340092 Litvinskij (1994) 1 26-FE-56(N,G)26-FE-57,,SIG,,,DERIV
40544002 Shcherbakov (1977) 136 26-FE-56(N,G)26-FE-57,,SIG
40544004 Shcherbakov (1977) 1 26-FE-56(N,G)26-FE-57,,SIG
V10012021 Mughabghab (2006) 1 26-FE-56(N,G)26-FE-57,,SIG,,,RECOM
```

Read the reaction code before comparing: `TTA` (thick-target average), `AV` (averaged
over an energy interval) and `SPA` (spectrum-averaged) are averages, to be compared with
averages of the evaluation; `DERIV` and `RECOM` are derived or recommended values, not
measurements. `Shcherbakov (1977)` is point-wise, from 7.3 meV to 1.4 keV:

```python
shcherbakov = next(xs for xs in capture if xs.dataset_id == "40544002")
print(shcherbakov.to_dataframe().head(3))

evaluated = plottable(b81, "cross_section", mt=102)
measured = plottable(shcherbakov, "cross_section", mt=102)
ratio = measured.data.y / np.interp(measured.data.x, evaluated.data.x, evaluated.data.y)
print(np.round(np.median(ratio), 3))

fig = (PlotBuilder(style="journal")
       .add_data(evaluated).add_data(measured)
       .set_scales(log_x=True, log_y=True)
       .set_limits(x_lim=(1e-3, 2e3))
       .build())
fig.savefig("fe56-capture-exfor.png")
```

```text
         energy  cross_section  error
0  7.310000e-09           4.79   1.20
1  7.500000e-09           5.18   1.29
2  7.690000e-09           4.45   1.07
1.006
```

`to_dataframe` is in MeV and b; `plottable` converts to eV, so the measured points and
the evaluated curve share an axis. The figure shows the $1/v$ capture of the evaluation
from 1 meV to 1 keV with the measured points on it, and the resonance at 1.15 keV; the
median ratio of measurement to evaluation over the 136 points is 1.006.
