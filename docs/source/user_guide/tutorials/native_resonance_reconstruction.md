# Native resonance reconstruction

`kika.processing.resonances` reconstructs the cross sections represented in a
neutron, laboratory-frame `ReactionSuite` at 0 K across its evaluated energy
domain. It combines supported resolved resonances, dilute unresolved averages,
backgrounds, fast-region reactions and the model's sum graph.

```python
from kika.processing.resonances import (
    ReconstructionOptions, reconstruct_suite, attach_reconstruction,
)

# suite is a kika.nuclear_data.model.ReactionSuite.
result = reconstruct_suite(
    suite, source_style="eval", label="recon",
    options=ReconstructionOptions(rtol=1e-3, atol=1e-8),
)
attach_reconstruction(suite, result)
```

When resonance physics requires a target mass ratio and spin, the processor
uses the model's particle data or accepts an explicit
`NeutronContext(target_mass_ratio, target_spin)`. The mass ratio is relative
to the neutron mass. Materials without resonance parameters require neither.
Use the evaluation's declared values when supplying a context.

The reconstruction preserves the evaluated forms and attaches a separate
`CrossSectionReconstructed` style. Totals and other additive aggregates are
derived from their exclusive partials. Disagreements with evaluated aggregates
are recorded in `result.report["source_sum_discrepancies"]`, including the
observed worst energy and both cross sections. Missing graphs, double counting
and unavailable resonance partials are errors.

The relative and absolute tolerances control empirical checks of each final
reaction. They are not a rigorous global bound on unsampled physics. Refinement
budgets and iteration limits are explicit; exhausting one raises
`ReconstructionConvergenceError`. Unsupported physics raises
`UnsupportedResonanceError`; these exceptions have a `category` attribute.
The processor does not silently retry with another engine or looser tolerances.

## Select the same style for plots and group averages

Choose `recon` explicitly. Plot each region separately to retain discontinuities.
The same selected form can be collapsed with the shared algebra layer:

```python
from kika.algebra import group_averages, interval_laws
from kika.nuclear_data.model import Regions1d

reaction = suite.findReactionByENDF_MT(2)
form = reaction.crossSection["recon"]
children = form.function1ds if isinstance(form, Regions1d) else [form]
for child in children:
    ax.plot(child.xs, child.ys)  # ax is a matplotlib Axes

x, y, interpolation = form.toEndfRegions()
laws = interval_laws(len(x), interpolation)
constant_flux = group_averages(x, y, laws, group_edges)
lethargy_flux = group_averages(x, y, laws, group_edges, "1/x")
```

Group means divide the weighted cross-section integral by the weight integral
over each group. Keep group edges within the verified domain. URR averages are
at infinite dilution; self-shielding and probability tables require additional
processing.

## Verify published cross sections

For a model carrying ENDF provenance, use the verified publication writer after
attachment:

```python
from kika.endf.writers.assemble import writeReconstructedEndfTape

conversion = writeReconstructedEndfTape(suite, result, "material.pendf")
```

It writes provisionally, reloads the file and checks reaction coverage, units,
domain boundaries, cross sections and sums before replacing the destination.
It preserves only sections represented by the input model; a MF1/2/3-only
model cannot publish omitted distributions or covariances.

GNDS publication can retain both styles:

```python
import kika
from kika.gnds.decode import readReactionSuite
from kika.gnds.xpath import Document

conversion = kika.write(suite, "material.xml", resonance_extensions=True)
reloaded, read_report = readReactionSuite(Document.parse("material.xml"))
errors = result.verify_suite(reloaded, label="recon")
```

Inspect both conversion reports. Verifying cross sections does not establish
lossless conversion of all resonance parameters or unrelated model data.
The optional KIKA resonance extensions preserve supported conventions that
generic GNDS does not represent. `result.report["source_conversion_report"]`
retains the source conversion report, including limitations explicitly
classified as irrelevant to cross sections. Unknown conversion losses block
reconstruction.

## Resource budgets and repeated verification

`ReconstructionOptions.max_work_bytes` defaults to 64 MiB. It targets temporary
energy batches and refinement chunks; input data, final tables, adaptive growth
and backend allocations are additional memory. It is not a process RSS limit.
An exceptional dense solve that exceeds its estimated workspace raises
`ReconstructionConvergenceError` with category `memory-budget-exhausted`.
The point and iteration budgets remain strict; exceeding them does not relax
the requested tolerance. Large evaluations can require an explicitly larger
`max_points` budget than the default 200,000.

Verification keeps every original node and four independent locations per
panel. A previous successful check can be reused only when all selected table
contents, units and the physical reference match exactly. Editing a table
forces another check. ENDF rounding usually changes the contents, so the
reloaded file receives its own physical verification.

Physics is refined on a common grid within each segment. Stored curves compact
only exactly constant spans, preserving boundaries and steps. Verification
retains an independent, immutable copy of the original computation grids,
including probes for reactions whose stored curves become constant.
`result.report["points"]` counts computation nodes;
`result.report["stored_points"]` counts nodes across all stored curves.
