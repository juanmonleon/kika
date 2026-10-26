# Total Monte Carlo with perturbed evaluations

How to propagate an evaluation's covariances by random sampling: draw perturbed
evaluations with {py:func}`~kika.sampling.model_perturbation.perturbFromModel`, compute the
response with each one, and read the spread with {py:func}`kika.UQ.fastTMC`. The example
perturbs the elastic cross-section of ENDF/B-VIII.1 Fe-56. The spread is checked on a
response that can be computed without a transport code, the elastic cross-section averaged
over two energy groups, where the sandwich rule gives the same quantity from the
covariance directly. The outputs shown are what the run printed. How the run draws and
applies the factors is in {doc}`../running-a-perturbation`.

## 1. Draw the evaluations

```python
from kika.endf.remote import download_endf
from kika.sampling.model_perturbation import perturbFromModel

def main():
    tape = download_endf("Fe56", save_to=".", library="endfb8.1")
    run = perturbFromModel(tape, {33: [2]}, 50, seed=2026, outputDir="tmc", nWorkers=4)
    print(run.description)

if __name__ == "__main__":
    main()
```

```text
1 independent draw(s), grouping='mf'; no section states a block across two files
  [0] MF33: 1 component(s), stride 630, dimension 630, union=global
```

The request `{33: [2]}` draws the elastic cross-section from its MF 33 covariance, here
stated on 630 energy bins. Each sample is a full ENDF tape, `tmc/NNNN/Fe56_endfb8.1_n_NNNN.endf`,
identical to the evaluation except the perturbed sections. Fe-56 states its resolved
resonances as parameters, so the run first reconstructs them with NJOY (from the
`NJOY_EXECUTABLE` environment variable) and perturbs the reconstructed cross-section, not
the MF 3 background; `run.notes` says so. The 50 samples took 173 s with 4 processes and
take 2 GB of disk (measured 2026-10-26).

## 2. With a transport code

For a transport calculation, ask for the files the code reads: ACE tables that kika
processes and writes itself (no NJOY ACER), or OpenMC HDF5 files:

```python
from kika.sampling.model_perturbation import AceOptions, HDF5Options

run = perturbFromModel(tape, {33: [2]}, 50, seed=2026, outputDir="tmc",
                       formats=("ace",), ace=AceOptions(engine="kika", temperatures=(293.6,)))
# or: formats=("hdf5",), hdf5=HDF5Options(temperatures=(293.6,))
```

Run the transport code once per sample, then collect the tally of each run and its
relative statistical error, and hand both to `fastTMC`:

```python
import numpy as np
import kika
from kika.UQ import fastTMC

totals = [kika.read_mctal(f"runs/{i:04d}/mctal")[4].total("energy")       # your runs
          for i in range(50)]
y = np.array([float(t.result.item()) for t in totals])      # the response of each run
sigma = np.array([float(t.error.item()) for t in totals])   # its relative 1-sigma error
result = fastTMC(y_val=y, y_sigma=sigma, random_seed=1)
```

The spread of the 50 responses holds both the nuclear-data uncertainty and the statistics
of each run; `fastTMC` separates the two. `percent_unc_nuclear_data` is the first,
`percent_unc_statistical` the second, `ci` their bootstrap confidence intervals, and
`is_reliable` is False when the statistics are too large a share of the spread for the
separation to mean anything (then quote `percent_unc_observed`, and run longer). This
section is not run here: it needs a transport code and its output.

## 3. A check against the sandwich rule

The elastic cross-section averaged over an energy group is a response every sample gives
directly. With it, the spread of the samples can be compared with what the covariance
itself says. kika averages each sample's cross-section with a 1/E weight over two of the
SCALE 44 groups, 0.9-1.4 and 1.4-1.85 MeV:

```python
import numpy as np
from pathlib import Path
import kika
from kika.algebra import group_averages, interval_laws

edges = [0.9e6, 1.4e6, 1.85e6]          # eV

def elastic_group_means(path):
    ev = kika.read(path, covariances=False)
    form = ev.findReactionByENDF_MT(2).crossSection.evaluated
    x, y, interpolation = form.toEndfRegions()
    return group_averages(x, y, interval_laws(len(x), interpolation), edges, "1/x")

samples = np.array([elastic_group_means(p) for p in sorted(Path("tmc").glob("*/*.endf"))])
samples.mean(axis=0)                                  # array([2.19048126, 2.24600258]) b
samples.std(axis=0, ddof=1) / samples.mean(axis=0)    # array([0.02332322, 0.02494329])
```

The response is computed exactly, with no statistical error, so `fastTMC` is told so with
zero sigmas, and the whole spread is nuclear data:

```python
for g in range(2):
    r = fastTMC(y_val=samples[:, g], y_sigma=np.zeros(len(samples)), random_seed=1, verbose=False)
    print(round(r["percent_unc_nuclear_data"], 2), r["ci"]["percent_unc_nuclear_data"])
```

```text
OK - Use Nuclear Data uncertainty (2.33%)
2.33 (1.91, 2.686)
OK - Use Nuclear Data uncertainty (2.49%)
2.49 (2.0, 2.904)
```

Without `y_sigma`, `fastTMC` cannot separate the two parts and gives `nan` for both,
reporting only the observed spread.

The sandwich rule gives the same quantity from the ERRORR covariance of
{doc}`sandwich-uncertainty`, which averages over each group with NJOY's `iwt=6` weight, 1/E
between 0.1 eV and 2.1 MeV, so the two use the same weight. The sensitivity of a group
average to itself is 1, so $S^\mathsf{T} C S$ is the diagonal of the covariance:

```python
from kika.cov import CrossSectionCovariance

cov = CrossSectionCovariance.from_gendf("errorr_Fe56/tape26", energy_unit="MeV")
grid = np.asarray(cov.energy_grid)
groups = [int(np.argmin(abs(grid - e / 1e6))) for e in edges[:-1]]
np.asarray(cov.get_uncertainty(26056, 2))[groups]     # array([0.02363978, 0.02339885])
```

| Group | Sampling (50 draws) | 95 % interval | Covariance |
|---|---|---|---|
| 0.9-1.4 MeV | 2.33 % | 1.91-2.69 % | 2.36 % |
| 1.4-1.85 MeV | 2.49 % | 2.00-2.90 % | 2.34 % |

The two routes agree within the sampling interval. With 50 draws that interval is about
±20 % of the value; it narrows as $1/\sqrt{N}$, so a few hundred draws are needed to
quote a sampled uncertainty to two figures.
