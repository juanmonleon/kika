# The nuclear-data uncertainty of a response, by the sandwich rule

How to propagate the covariances of an evaluation to a calculated response: sensitivities
from MCNP, covariance matrices from the evaluation on the same energy groups, and the
sandwich rule $\sigma_R^2 = S^\mathsf{T} C S$ with its breakdown by nuclide and reaction.
The response is the neutron flux in a detector shell outside an iron-and-water sphere
(tally 4 of the files in `examples/data/`); the evaluations are ENDF/B-VIII.1 Fe-56 and
H-1. The outputs shown are what the run printed. Getting sensitivities from MCNP is
{doc}`/user_guide/tutorials/sensitivity_analysis`, and the rule itself is
{doc}`../../methods/uq/sandwich`.

## 1. The sensitivities

Two MCNP runs of the same model perturbed the iron (Fe-56) and the water (H-1); each gives
the sensitivity of the tally to one nuclide, on the SCALE 44-group structure:

```python
import kika

fe = kika.compute_sensitivity(inputfile="inputfile_example_1.i", mctalfile="mctalfile_example_1.m",
                              tally=4, zaid=26056, label="Fe-56")
h = kika.compute_sensitivity(inputfile="inputfile_example_2.i", mctalfile="mctalfile_example_2.m",
                             tally=4, zaid=1001, label="H-1")

sdf = kika.create_sdf_data([fe, h], energy="integral", title="Shell detector")
sdf.group_inelastic_reactions(replace=True)
[(d.nuclide, d.mt) for d in sdf.data]
```

```text
Skipping H-1 (n,n') (MT=4): All sensitivity coefficients are zero
...
[('Fe-56', 1), ('Fe-56', 2), ('Fe-56', 4), ('Fe-56', 102), ('Fe-56', 103), ('Fe-56', 107),
 ('H-1', 1), ('H-1', 2), ('H-1', 102)]
```

`create_sdf_data` puts both nuclides in one SDF. It checks that the two runs give the same
unperturbed response (here both 1.13584·10⁻⁸ with a 0.57 % relative error) and refuses
otherwise, unless `response_values=(r0, e0)` says which to use. The reactions MCNP was
asked for and the nuclide does not have (inelastic levels of H-1) have zero sensitivity
and are left out. `group_inelastic_reactions` sums the discrete inelastic levels of Fe-56
(MT 51 and 52 here) into MT 4, because the covariance below states MT 4, not its levels.

## 2. The covariances, on the same groups

The sandwich rule needs the relative covariance of each cross-section on the 44 groups of
the sensitivities. NJOY's ERRORR module computes it from the evaluation's MF 33. kika
writes the NJOY input with {py:class}`~kika.njoy.deck.InputDeck`; the run below expects NJOY
in the `NJOY_EXECUTABLE` environment variable:

```python
import os, shutil, subprocess
from pathlib import Path
from kika.endf.remote import download_endf
from kika.njoy.deck import InputDeck

edges_ev = [e * 1e6 for e in kika.energy_grids.SCALE44]     # SCALE44 is in MeV, ERRORR wants eV

for nuclide in ("Fe56", "H1"):
    tape = download_endf(nuclide, save_to=".", library="endfb8.1")
    work = Path(f"errorr_{nuclide}")
    work.mkdir(exist_ok=True)
    shutil.copy(tape, work / "tape20")
    deck = InputDeck()
    deck.moder(nin=20)
    deck.reconr(isotope=nuclide, err=0.001)
    deck.errorr(isotope=nuclide, ign=1, egn=edges_ev, iwt=6, tempin=0.0, mfcov=33, nout=26)
    deck.save(work / "input")
    with open(work / "input") as stdin:
        subprocess.run([os.environ["NJOY_EXECUTABLE"]], stdin=stdin, cwd=work,
                       stdout=subprocess.DEVNULL, check=True)
```

Three choices in the ERRORR card matter:

- `ign=1` with `egn`: the group boundaries given explicitly, those of the sensitivities.
  The sandwich rule refuses grids that do not coincide; it does not regroup either side.
- `iwt=6`: the weight used to average within a group (NJOY's thermal Maxwellian, 1/E, and
  fission spectrum). Between 0.1 eV and 2.1 MeV it is 1/E.
- `tempin=0.0`: the temperature of the PENDF ERRORR reads, here RECONR's at 0 K. NJOY's
  own default, 300 K, stops ERRORR with `unable to find temp` on a chain without BROADR.

{py:meth}`CrossSectionCovariance.from_gendf <kika.cov.CrossSectionCovariance.from_gendf>`
reads each output tape:

```python
from kika.cov import CrossSectionCovariance

covariances = [CrossSectionCovariance.from_gendf(f"errorr_{n}/tape26", energy_unit="MeV")
               for n in ("Fe56", "H1")]
[(sorted(c.isotopes), sorted(c.reactions)) for c in covariances]
```

```text
[([26056], [1, 2, 4, 5, 16, 102, 103]), ([1001], [1, 2, 102])]
```

## 3. The sandwich rule

{py:func}`~kika.UQ.sandwich.sandwich_uncertainty_propagation` aligns the sensitivities with
the covariances and propagates. Alignment is strict: a sensitivity with no covariance is an
error, so the first call refuses:

```python
from kika.UQ import sandwich_uncertainty_propagation

sandwich_uncertainty_propagation(sdf, covariances)
```

```text
MissingCovarianceError: No covariance data are available for: ZAID=26056 MT=107. To continue
explicitly with the covered subset, repeat the call with missing="drop". The returned
AlignmentReport will record the loss.
```

ENDF/B-VIII.1 states no covariance for Fe-56 (n,α). Leaving it out is a decision, so it is
asked for, and recorded:

```python
result = sandwich_uncertainty_propagation(sdf, covariances, missing="drop", bootstrap_seed=1)
print(result)
```

```text
Response value:                    1.135840e-08

NUCLEAR DATA UNCERTAINTY  (sigma_ND = sqrt(S^T C S))
  Nominal:                         ± 4.905%
  Absolute:                        ± 5.571629e-10
  Bootstrap mean:                  ± 4.908%
  95% CI (bootstrap, N=1000):      [4.846%, 4.976%]
...
TOTAL VARIANCE (includes cross-reaction correlations):
Rank Nuclide      Reaction        MT     Variance     % Total
--------------------------------------------------------------------
1    Fe-56        (n,n')          4      1.5720e-03  65.33%
2    Fe-56        (n,el)          2      7.2620e-04  30.18%
3    H-1          (n,el)          2      1.0197e-04   4.24%
4    Fe-56        (n,γ)           102    6.0306e-06   0.25%
5    H-1          (n,γ)           102    9.3261e-11   0.00%
6    Fe-56        (n,p)           103    0.0000e+00   0.00%
```

The flux in the detector has a 4.9 % uncertainty from these two evaluations, two thirds of
it from the inelastic scattering of Fe-56. The confidence interval comes from drawing the
sensitivities again within their own statistical errors and recomputing $S^\mathsf{T} C S$
each time, so it says how much the 4.9 % itself depends on the statistics of the MCNP run,
not how uncertain the nuclear data are. The run's own
statistical error (0.57 %) is reported apart and is not added in.

The same numbers are attributes of the result: `result.relative_uncertainty` (0.04905),
`result.total_variance`, `result.bootstrap_ci_low` and `bootstrap_ci_high`, and one
`result.contributions` entry per reaction (its variance, its share, its nuclide and
reaction name).

## 4. What was left out, and why

`result.alignment_report` records every decision between the sensitivities and the
covariances:

```python
result.alignment_report.dropped
result.alignment_report.assumptions
```

```text
[ParameterKey(kind='xs', zaid=26056, mt=107, order=None)]
['MT=1 excluded to avoid total/reaction double counting', "nu-bar redundancy resolved with nubar_mode='total'"]
```

MT 1 is always left out: the total is the sum of the partial reactions already propagated,
so including it would count each of them twice. Read the dropped list before quoting the
number: a dropped reaction contributes nothing, whatever its real uncertainty.

## Next

The same profile and covariances give the similarity of two systems,
{py:func}`~kika.UQ.similarity.similarity_ck`, the basis for choosing the benchmarks that
validate a calculation. {doc}`total-monte-carlo` checks the sandwich rule against random
sampling on a quantity both can compute.
