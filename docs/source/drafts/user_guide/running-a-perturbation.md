# Running a perturbation

How to draw an ensemble of perturbed evaluations from the covariances an ENDF
(Evaluated Nuclear Data File) tape states, with
{py:func}`kika.sampling.model_perturbation.perturbFromModel`: how to write the request,
pick the output formats, rehearse a run, read what it did, and get its factors back. What
the run computes (the blocks, the draw, how each quantity is perturbed) is in
{doc}`../methods/sampling/perturbation-pipeline` and the pages of the perturbation
chapter it links.

The examples run on `micro_fe56_xs_and_angular.endf`, a small Fe-56 tape committed under
`kika/endf/tests/data/` (MF 3, MF 4 and the MF 33 and MF 34 of elastic scattering),
copied to the working directory as `Fe56.endf`, and on `micro_u235_nubar.endf` from the
same directory. The outputs shown are what they printed.

## The first run

```python
from kika.sampling.model_perturbation import perturbFromModel

run = perturbFromModel(
    "Fe56.endf",
    {33: None, 34: {"mt": [2], "index": [1, 2, 3]}},   # what to perturb
    4,                                                 # number of samples
    seed=1,
    outputDir="run1",
    resonanceRegion="evaluated",
)
print(run.description)
```

```text
2 independent draw(s), grouping='mf'; no section states a block across two files
  [0] MF33: 1 component(s), stride 3, dimension 3, union=global
  [1] MF34: 3 component(s), stride 42, dimension 126, union=per-component
```

The request asks for every cross-section MF 33 states and for Legendre orders 1-3 of the
elastic angular distribution. The run read the tape, assembled one block per covariance
file, inspected and conditioned them, drew four samples, put each on the model and wrote
it as an ENDF tape (the default format). It returns a
{py:class}`~kika.sampling.model_perturbation.RunResult`.

**About `resonanceRegion`.** By default (`"reconstructed"`) a tape whose resonance region
is stated as parameters is first reconstructed by NJOY RECONR, so that the factors act on
the resonance cross-section and not on the MF 3 background (see
{doc}`../methods/perturbation/cross-section-perturbation`). That needs NJOY and a full
tape. The committed micro-tapes are cut section by section and NJOY cannot read them, so
the examples on this page pass `resonanceRegion="evaluated"`, which perturbs MF 3 as
stated and says so in a warning. On a real evaluation, leave the default.

## Writing the request

A request maps what to perturb to a selection. Two spellings are accepted, and may be
mixed in one request.

**By covariance file**, keyed by MF number:

| Request | Perturbs |
|---|---|
| `{33: None}` | every cross-section MF 33 states |
| `{33: [2, 102]}` | MT 2 and MT 102 |
| `{34: {"mt": [2], "index": [1, 2, 3]}}` | Legendre orders 1-3 of MT 2 |
| `{31: [455, 456]}` | prompt and delayed $\bar\nu$ |
| `{35: None}` | every band of the fission spectrum |
| `{35: {"index": [0, 1]}}` | the two lowest bands |

**By quantity**, in the model's vocabulary (`crossSection`, `angularDistribution`,
`multiplicity`, `energyDistribution`), naming reactions by MT number or by label, and
calling the third coordinate `order` (MF 34) or `band` (MF 35):

```python
request = {"crossSection": None,
           "angularDistribution": {"reaction": "MT2", "order": [1, 2, 3]}}
dry = perturbFromModel("Fe56.endf", request, 4, seed=1, outputDir="dry1",
                       dryRun=True, resonanceRegion="evaluated")
dry.request
```

```text
{33: None, 34: Selection(mf=34, mt=[2], index=[1, 2, 3], relative=True)}
```

`run.request` is always the MF spelling, whichever was given. The quantity spelling is
the one to use for a GNDS source, where reactions have labels and nothing is called
MF 34. A key that is neither an MF number nor one of the four quantities, or MF named
twice, is refused.

A request for a sum the file decomposes (MT 4 when MT 51-91 are present) is a request for
the sections that decompose it; see {doc}`../methods/perturbation/cross-section-perturbation`.

### When the evaluation does not state what you ask for

By default (`onMissing="raise"`) a request for something the evaluation has no covariance
for is refused:

```python
perturbFromModel("Fe56.endf", {33: [2, 102]}, 2, dryRun=True,
                 resonanceRegion="evaluated")
```

```text
ValueError: MT102: asked for, and this evaluation states no cross-section covariance for it.
A covariance perturbs only the cross section it was stated for, so no other block -- a sum's
included -- stands in. Drop it from the request, or pass onMissing='skip' to perturb the
rest and record the omission
```

For a request written once for many files (a directory where only some evaluations state
MF 34, say), pass `onMissing="skip"`. The run perturbs what is there and records every
dropped selection as a warning in the log, a line in `run.notes` and an entry in
`run_metadata.json`:

```python
run = perturbFromModel("U235_nubar.endf", {31: None, 34: None}, 2,
                       dryRun=True, onMissing="skip")
run.request      # {31: Selection(mf=31, mt=None, index=None, relative=True)}
run.notes[...]   # 'angularDistribution was asked for and is not stated by this
                 #  evaluation, so it was not perturbed (...)'
```

A request where nothing matches still raises, under either setting.

A request is relative by default, and the appliers multiply, so an **absolute** MF 31 or
MF 33 section the request reaches is made relative before the draw: divided by the
evaluation's cross-section (or $\bar\nu$) averaged over each covariance bin. The run says
which sections it converted, and how many bins had no cross-section to divide by (those
get zero relative covariance), in `run.notes` and the log:

```text
1 absolute covariance section(s) made relative, divided by the central value averaged
over each bin (1/E weighted): MF33 MT4 x MT4
```

On a tape whose resonance region is stated as parameters and `resonanceRegion="evaluated"`,
the cross-section it divides by is reconstructed by NJOY RECONR, so NJOY is needed. A
section with no central value to divide by (a GNDS source, or a reaction the evaluation
states neither in MF 3 nor as a sum of its partials) follows `onMissing`: a `ValueError`,
or left out with a line in `run.notes`. An absolute MF 34 section is not drawn, and the run
names it. How the conversion is done is in
{doc}`../methods/sampling/perturbation-pipeline`.

## Choosing the output formats

`formats` takes any of the emitters in `kika.sampling.model_perturbation.EMITTERS`:

| Emitter | Writes, per sample | Needs |
|---|---|---|
| `endf-delta` (default) | `NNNN/<stem>_NNNN.endf`: the source tape with only the perturbed sections re-encoded | an ENDF source, given as a path |
| `endf-tape` | `NNNN/<stem>_NNNN.tape.endf`: the whole tape written from the model | |
| `gnds` | `NNNN/<stem>_NNNN.gnds.xml`: the evaluated form and the realisation, each under its label | |
| `ace` | `NNNN/ace/`, one ACE file per temperature, with NJOY's files under `NNNN/njoy/` | `ace=AceOptions(...)` and NJOY |

```python
run = perturbFromModel("Fe56.endf", {33: None}, 2, seed=1, outputDir="run3",
                       formats=("endf-delta", "endf-tape", "gnds"),
                       writeSets=True, resonanceRegion="evaluated")
sorted(p.name for p in (run.outputDir / "0001").iterdir())
```

```text
['Fe56_0001.endf', 'Fe56_0001.gnds.xml', 'Fe56_0001.tape.endf', 'perturbation.json']
```

Use `endf-delta` for transport: two samples differ only in the sections they perturbed,
and every section the model does not decode survives byte for byte. `endf-tape` and
`gnds` carry what the model carries, which can be less than the tape. `run.paths(fmt)`
lists one emitter's files in sample order. `writeSets=True` adds each sample's factors as
`perturbation.json`, which the run's factors table already holds.

An unknown emitter is refused before anything is read. From a GNDS source, a run that
writes files refuses `endf-delta` (there is no tape to patch). `gnds` works, and so do
`endf-tape` and `ace`: the tape is written from the model, with its MAT from ENDF-6's table
for the target (or `mat=`) and the rest of its bookkeeping derived from what the GNDS file
states (see {doc}`../methods/formats/endf`). They are refused when the suite has resolved or
unresolved resonance parameters, which are not written to MF 2: without MF 2 the tape's
MF 3 in the resonance region would be only the background.

### ACE files

`ace` runs NJOY on each sample's ENDF tape. It needs a
{py:class}`~kika.sampling.model_perturbation.AceOptions`:

```python
import os
from kika.sampling.model_perturbation import AceOptions

ace = AceOptions(temperatures=(293.6, 600.0),
                 njoyExe=os.environ["NJOY_EXECUTABLE"],
                 libraryName="jeff40", extensions=("02c", "03c"))
run = perturbFromModel("Fe56.endf", {33: None, 34: {"mt": [2], "index": [1, 2, 3]}},
                       2, seed=1, outputDir="acedry", dryRun=True,
                       formats=("endf-delta", "ace"), ace=ace,
                       resonanceRegion="evaluated")
```

The options are checked before anything is drawn, dry run included: at least one
temperature, one extension per temperature when extensions are given, and an NJOY
executable that exists. `formats=("ace",)` without `ace=` raises `ValueError`; an
`njoyExe` that does not exist raises `FileNotFoundError`. When `njoyExe` is not given,
`AceOptions` takes the environment variable `NJOY_EXECUTABLE`, else `njoy` on the path,
the same lookup the resonance reconstruction uses. With extensions, each ACE file is
`<ZAID>.<ext>` with its xsdir line beside it. `keepNjoyFiles=True` (the default) keeps
NJOY's input, output and listing. After a run, `run.aceFailures()` lists every
`(sample, temperature, return code)` whose NJOY run failed.

When `ace` is asked for without an ENDF emitter, the sample's tape is written anyway
(`endf-delta` from a tape source) because NJOY needs it.

## Rehearsing a run: `dryRun`

`dryRun=True` does everything except write per-sample files: it reads, assembles,
inspects, conditions, draws, puts every realisation on the model and checks it. With an
`outputDir`, the run-level files are still written:

```python
sorted(p.name for p in dry.outputDir.iterdir())
```

```text
['conditioning_plan.json', 'factors.parquet', 'run.log', 'run.log.jsonl', 'run_metadata.json']
```

The factors are the ones the full run draws with the same arguments. Sample 2 of the dry
run and of `run1` above are equal, bit for bit:

```python
import numpy as np
from kika.sampling.perturbation_set import readFactorsTable

a, b = readFactorsTable("run1", 2), readFactorsTable("dry1", 2)
all(np.array_equal(a.factors[c], b.factors[c]) for c in a.components())   # True
```

So a run can be checked on a laptop, and launched where the time is, with the same
arguments.

## Reading the run log

Every stage of a run is a timed event with its numbers, kept on `run.log` (a
`kika.sampling.run_log.RunLog`) and written twice from the same events: `run.log.jsonl`
for a program and `run.log` for a person.

```python
print(dry.log.verdict())
print(dry.log.summary())
```

```text
DRY RUN OK, 9 warning(s) -- read them below
49 event(s): 1 started, 1 read, 3 warning, 1 request, 3 assembled, 11 inspected,
3 conditioned, 3 drawn, 4 applied, 16 checked, 2 written, 1 finished; 9 warning(s)/error(s)
```

The text file starts with the verdict and a summary of the run, then one section per
stage. The opening of `run1/run.log`:

```text
kika perturbation run  [realization]
============================================================
RUN OK, 9 warning(s) -- read them below

Source     Fe56.endf (endf)
Request    crossSection: every reaction stated; angularDistribution: MT2 orders 1, 2, 3
Draw       4 sample(s), seed 1, log space, sobol/svd, psd_method=none
Output     run1  (endf-delta)
Elapsed    3.4 s

1  Read           0.355 s
...
2  Blocks         2 independent draw(s), grouping='mf'; no section states a block across two files
     MF33: MT2 (ZA 26056)                         3x3  union=global
     MF34: MT2 L=1, MT2 L=2, MT2 L=3 (ZA 26056)   126x126  union=per-component

3  Pre-flight     (auto) 2 block(s) samplable; 1 would not reproduce the stated covariance without a repair
   ! MF34: MT2 L=1, MT2 L=2, MT2 L=3 (ZA 26056): definiteness: not PSD: λ_min/λ_max = -3.445e-05 ...  -> clip (stated variances moved up to 4.4%)
     MF34: MT2 L=1, MT2 L=2, MT2 L=3 (ZA 26056): rank: rank 119 of 126 — 7 null direction(s) at tol 1e-10
     MF33: MT2 (ZA 26056): none (no change)
```

Lines marked `!` are warnings; they are repeated at the end under
`WARNINGS AND ERRORS`. The event kinds are a closed set
(`kika.sampling.run_log.EVENT_KINDS`): `started`, `read`, `request`, `assembled`,
`inspected`, `conditioned`, `drawn`, `applied`, `checked`, `emitted`, `written`, `note`,
`warning`, `error`, `finished`. To select events in a program:

```python
from kika.sampling.run_log import RunLog

log = RunLog.read("dry1/run.log.jsonl")     # the same events, back from the file
log.of("drawn")                              # one per block, payload with rank, n_null, ...
log.problems()                               # every warning and error
```

A `logging.Logger` passed as `logger=` receives every event as one line as well.

**Read `run.notes`.** They are the facts true of every sample that no output file states:
a sum rebuilt from its partials, a resonance region perturbed as stated, a selection
dropped under `onMissing="skip"`, a table perturbed as a table. From `run1`:

```text
MF3 sums re-derived from their moved partials: MT1
MF4/MT[2] tabulated and perturbed as a table: the correction is added on the evaluator's
own cosine nodes and the lin-lin integral is not renormalised (per-realisation numbers
under 'tables')
```

## The run directory

```text
run1/
├── 0000/Fe56_0000.endf        one directory per sample, its files
├── 0001/ ...
├── factors.parquet            every sample's factors, one table
├── conditioning_plan.json     the repairs made before the draw
├── run_metadata.json          the arguments, the blocks, the notes, the files
├── run.log                    the log, for a person
└── run.log.jsonl              the log, one JSON event per line
```

- **`factors.parquet`**: one row per (sample, component, bin), with columns `sample`,
  `za`, `mf`, `mt`, `index` (Legendre order or band, else 0), `bin` and `value` (the
  factor, or for MF 35 the absolute delta). Its **index** (the bin edges, the semantics,
  the groups drawn together, the sample labels, the shared provenance) is stored inside
  the parquet file as schema metadata, so the table describes itself. There is no
  separate `factors_index.json`: read the index with
  `kika.sampling.perturbation_set.readFactorsIndex`. `writeFactors=False` skips the table.
- **`conditioning_plan.json`**: one step per block, with the repair applied and the
  reason, and the findings left unrepaired. It is what the run applied, written so that
  it can be edited and handed back (see below).
- **`run_metadata.json`**: the source, the request as given and as normalised, the seed,
  the space, the grouping, the sum and resonance-region settings, the blocks, the notes,
  the conditioning, the structural findings on the covariance sections, the files and
  the decode reports.

The same paths are on `run.files` (`factors`, `conditioning-plan`, `metadata`, `log`,
`log-text`), and each sample's on `run.samples[i]["files"]`.

## Getting the factors back

One sample, back as the `kika.sampling.perturbation_set.PerturbationSet` that was applied:

```python
from kika.sampling.perturbation_set import (PerturbationSet, readFactorsIndex,
                                            readFactorsTable)

pset = readFactorsTable("run1", 2)          # or PerturbationSet.fromRun("run1", 2)
pset                                        # PerturbationSet('realization-0002', 4 component(s),
                                            #   angularDistribution+crossSection)
factors, edges = pset.block(2)              # MF 33, MT 2
factors                                     # [0.86178889 1.18159197 0.98114893]
edges                                       # [ 850000. 1500000. 2500000. 4000000.]

index = readFactorsIndex("run1")
index["nSamples"], index["semantics"], index["edgeRule"]
# (4, 'multiplicative-relative', 'endf-step-duplicate')
```

`readFactorsTable` reads only the rows of the sample asked for. `pset.block(mt, mf=34,
order=1)` addresses an MF 34 component, and `pset.describe()` says which components were
drawn together. A set written with `writeSets=True` is read with
`PerturbationSet.read("run3/0001/perturbation.json")`. For the whole table at once,
`pandas.read_parquet("run1/factors.parquet")`.

To put one sample's factors on a model node yourself (to look at a perturbed
cross-section without writing a tape), `PerturbationSet.apply` takes the node's function
and the component, and returns what
{py:func}`~kika.nuclear_data.model.perturbation.applyFactors` returns:

```python
import kika
from kika.sampling.joint_blocks import ComponentKey

ev = kika.read("Fe56.endf")
xs = ev.reactionByENDF_MT(2).crossSection.evaluated
perturbed, info = pset.apply(xs, ComponentKey(za=26056, mf=33, mt=2, index=0))
info
```

```text
{'min_factor': 0.8617888934556927, 'max_factor': 1.1815919703066156,
 'frac_out_of_coverage': 0.11414523951896882, 'n_inserted': 3}
```

`apply` handles one cross-section; it does not rebuild the sums (MT 1 here) and raises
`KeyError` for a component the set does not carry. `PerturbationSet.applyToSuite` applies
the whole set to a suite as a run does.

## Handing in a conditioning plan

By default the run inspects its blocks and applies the recommended plan, which repairs
definiteness and nothing else (see {doc}`../methods/covariance/conditioning-and-psd`).
To change a repair, edit the plan the run wrote and hand it back; to draw every block
exactly as stated, pass `False`:

```python
import json
from kika.cov.conditioning import ConditioningPlan

REQUEST = {33: None, 34: {"mt": [2], "index": [1, 2, 3]}}
plan = ConditioningPlan.from_dict(json.load(open("run1/conditioning_plan.json")))
run = perturbFromModel("Fe56.endf", REQUEST, 4, seed=1, dryRun=True,
                       conditioningPlan=plan, resonanceRegion="evaluated")
run.conditioningMode        # 'explicit'

run = perturbFromModel("Fe56.endf", REQUEST, 4, seed=1, dryRun=True,
                       conditioningPlan=False, resonanceRegion="evaluated")
run.conditioningMode        # 'none'
```

## Log or linear, per quantity

`space` sets how relative factors are drawn: `"log"` (the default) or `"linear"`, for the
whole run or per quantity, keyed by MF number or by quantity name:

```python
run = perturbFromModel("Fe56.endf", REQUEST, 2, dryRun=True,
                       space={"crossSection": "log", "angularDistribution": "linear"},
                       resonanceRegion="evaluated")
{key[1]: d["space"] for key, d in run.diagnostics.items()}
# {'MF33': 'log', 'MF34': 'linear'}
```

Use linear for Legendre coefficients whose uncertainty straddles zero; see
{doc}`../methods/sampling/perturbation-pipeline` for why. MF 35 is always drawn linearly,
as absolute deltas, whatever `space` says. A misspelt space is refused before the tape is
read.

## Taking the covariance from another tape

`covarianceSource` takes the covariance from a second ENDF tape and the evaluation from
the first. Use it only when the uncertainty you have lives in a different file from the
evaluation you perturb. The run warns every time, because nothing in either file says the
two belong together:

```python
run = perturbFromModel("Fe56.endf", {33: None}, 1, seed=1, dryRun=True,
                       covarianceSource="Fe56_other.endf",
                       resonanceRegion="evaluated")
```

```text
covariance taken from Fe56_other.endf and not from the evaluation itself. Nothing in
either file states that they belong together, so the uncertainty applied here is the one
you asserted, not one the evaluation declares
```

A covariance for a different nuclide is refused (`ValueError: Cf252.endf states
covariance for ZAID 98252, and the evaluation being perturbed is ZAID 26056...`), and so
is a covariance file that does not exist. When the second tape states a cross-section
covariance for a reaction the evaluation does not carry at all (neither the section nor
any partial of it, MT 16 on a tape without it, say), its factor would have nothing to
move: under `onMissing="raise"` the run refuses it with a `ValueError` naming the MT, and
under `onMissing="skip"` it leaves those blocks out and says so in `run.notes`.

## Using several processes: `nWorkers`

With `nWorkers > 1`, the samples are applied and written by a pool of worker processes.
The read, the pre-flight and the draw still happen once, in the calling process, so the
factors are the same whatever the number of workers:

```python
par = perturbFromModel("Fe56.endf", REQUEST, 4, seed=1, outputDir="par",
                       nWorkers=2, resonanceRegion="evaluated")
```

```text
4 sample(s) over 2 worker process(es); the draw was made once here, each worker
re-reads Fe56.endf once and writes whole samples
```

`par/0003/Fe56_0003.endf` is byte-identical to `run1/0003/Fe56_0003.endf`. Each worker
re-reads the source, so the source must be a path. A dry run, or a run of one sample,
ignores `nWorkers` and says so in a note. On a cluster node, pass the CPUs the job was
given; with `ace` among the formats, one NJOY run per worker is what fills them. Because
the pool starts new processes, a script that calls `perturbFromModel` with `nWorkers > 1`
on Windows needs the usual `if __name__ == "__main__":` guard.

## Other arguments

- `grouping`: `"mf"` (default) draws all requested components of one covariance file as
  one block, joined across files only where the evaluation states a covariance between
  them; `"stated"` splits further, into the blocks the file states.
- `samplingMethod` (`"sobol"`, `"lhs"`, `"random"`), `decompositionMethod` (`"svd"`,
  `"eigen"`), `nullTol`, `seed`: passed to the draw.
- `crossSectionSums`, `sumBlocks`: how stated sums are kept consistent; see
  {doc}`../methods/perturbation/cross-section-perturbation`.
- `angularPositivity`: `"report"` (default) or `"repair"` for negative angular
  distributions; see {doc}`../methods/perturbation/legendre-angular-perturbation` and
  {doc}`../methods/perturbation/tabulated-angular-perturbation`.
- `maxOutgoingPoints`: a cap on the size of a perturbed fission-spectrum table; see
  {doc}`../methods/perturbation/fission-spectrum-perturbation`.
- `labelPrefix`: the label of each realisation (`realization-0000`, ...).
- `mat`: the MAT number of a tape written from a GNDS source.

## Perturbing a multiplicity

The same call perturbs $\bar\nu$. The total is rebuilt from its components on every
sample (see {doc}`../methods/perturbation/multiplicity-perturbation`):

```python
run = perturbFromModel("U235_nubar.endf", {"multiplicity": [455, 456]}, 2,
                       seed=3, outputDir="nubar")
run.notes
```

```text
["MT452 was rebuilt from the sum rule on 185 points, from MT[455, 456]: its own factor
block is discarded, which is the convention perturb_nubar_family states. Rebuilding also
repairs the input file's own residual, which was 1.61e-06 at worst per bin"]
```

This tape states no resonance region, so the default `resonanceRegion` needs no NJOY.

## Where this is tested

All passed on 2026-10-11 unless marked:

- the log and the dry run:
  `kika/sampling/tests/test_a_run_keeps_a_log_and_can_rehearse.py`;
- the factors table: `kika/sampling/tests/test_factors_are_one_table_per_run.py`;
- worker processes: `kika/sampling/tests/test_samples_are_written_by_worker_processes.py`;
- the two spellings: `kika/sampling/tests/test_a_request_in_the_models_vocabulary.py`;
- `onMissing`: `kika/sampling/tests/test_a_request_can_be_written_for_many_files.py`;
- `space` per quantity: `kika/sampling/tests/test_a_space_can_be_stated_per_quantity.py`;
- `covarianceSource`: `kika/sampling/tests/test_covariance_can_come_from_another_file.py`;
- absolute sections: `kika/sampling/tests/test_an_absolute_section_is_made_relative.py`;
- the ACE emitter: `kika/sampling/tests/test_ace_is_an_emitter.py` (its NJOY run on a
  full tape, `::test_a_real_ace_comes_out_of_a_real_perturbed_tape`, runs only with
  `NJOY_EXECUTABLE` and `KIKA_FULL_TAPE` set; passed 2026-10-21 on JEFF-4.0 U-235, and on
  Fe-56 the pre-flight refuses its every-block request before NJOY runs);
- the emitters end to end, $\bar\nu$ included:
  `kika/sampling/tests/test_perturbing_from_the_model_end_to_end.py`.

The file-level samplers kept for reproducing earlier ensembles are described in
{doc}`legacy-samplers`.
