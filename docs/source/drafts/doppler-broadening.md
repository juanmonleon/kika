# Heating scalar neutron cross-sections

Use {py:func}`kika.processing.doppler.prepare_curve` to snapshot a linear table,
then {py:func}`kika.processing.doppler.broaden_curve` to produce a heated table.
The source must include the complete cross-section, with known source temperature,
explicit target-to-neutron mass ratio and explicit support outside its endpoints.
The operator applies to neutron laboratory-frame scalar cross-sections in a
free-gas target model. It does not heat angular distributions or covariances.

## Heat a synthetic curve

This example uses a synthetic constant source to demonstrate the API. Its mass
ratio does not identify a measured nuclide evaluation. Array inputs use eV and b;
source and destination temperatures use K.

```python
from kika.processing.doppler import (
    DopplerContext, DopplerOptions, SupportPolicy,
    prepare_curve, evaluate_broadened, broaden_curve,
)

source = prepare_curve(
    (([0.001, 2.0], [2.0, 2.0]),),
    context=DopplerContext(56.0, mass_origin="synthetic-example"),
    source_temperature=0.0,
    support=SupportPolicy(lower="constant", upper="constant"),
)
options = DopplerOptions(rtol=1e-3, atol=1e-8)
pointwise = evaluate_broadened(source, [0.01, 0.1, 1.0],
                              temperature=600.0, options=options)
heated = broaden_curve(source, temperature=600.0, options=options)
print(pointwise.values)
print(heated.form)
print(heated.verify_table())
```

Pointwise evaluation returns values with kernel-error estimates, tail bounds
and diagnostics. Tabulation adds adaptive mesh checks. The selected output
window restricts the table; integration still uses the complete source support.
For an input beginning at zero, supply a strictly positive output minimum.
Cooling is rejected. Equal source/destination temperatures preserve the source
function and its semantic boundaries.

## Heat and attach a reaction suite

{py:func}`kika.processing.doppler.broaden_suite` snapshots the selected source
style, heats exclusive scalar leaves and derives sums from their explicit graph.
Supply a support policy for every leaf using `ReactionKey(container, label)`.
A reconstructed source requires an explicit source temperature: the style name
alone does not establish 0 K. A target mass may come from the suite particle
data or an explicit context; conflicting declarations are rejected.

```python
from kika.processing.doppler import (
    ReactionKey, broaden_suite, attach_broadening,
)

# suite is a complete reconstructed neutron/lab scalar ReactionSuite.
# policies contains every exclusive source leaf; sums have no separate policy.
policies = {
    ReactionKey("reactions", reaction.label):
        SupportPolicy(lower="one_over_v", upper="constant")
    for reaction in suite.reactions
}
result = broaden_suite(
    suite, source_style="recon", source_temperature=0.0,
    temperature=600.0, support=policies, options=options, label="hot",
)
attach_broadening(suite, result, date="2026-10-11")
print(result.verify_suite(suite))
```

Choose endpoint policies from the source semantics: the policies in this snippet
are examples, not defaults suitable for every reaction. An unresolved-resonance
source needs a complete dilute mean and an explicit infinite-dilution declaration;
finite-dilution processing is unsupported. The source style, existing forms and
existing styles remain available after attachment.

{py:func}`kika.processing.doppler.attach_broadening` checks source freshness,
coverage, temperature, provenance and the editable result forms before adding the
new forms and `Heated` style. If the caller edits a result, source or graph,
attachment can fail. The attached forms are independent copies.

## Publish and check a scalar result

The attachment date defaults to the local calendar date. Set `date` explicitly
for a reproducible record; it is normalised to an ISO calendar date.

The verified ENDF (Evaluated Nuclear Data File) publication entry point is:

```python
from kika.endf.writers.assemble import writeHeatedEndfTape

publication = writeHeatedEndfTape(suite, result, "heated.pendf")
assert publication.isClean
```

This writes a temporary derived tape, rereads its scalar cross-sections, and
checks serialization and the full accuracy budget before replacing the destination.
A detected conversion loss, approximation or failed accuracy check preserves an
existing destination. Temperature and coverage describe the processed result.

For GNDS (Generalised Nuclear Database Structure), check the conversion report
and the represented heated forms after rereading:

```python
import kika

conversion = kika.write(suite, "heated.xml", format="gnds")
if not conversion.isClean:
    raise RuntimeError(conversion.summary())
reread = kika.read("heated.xml", covariances=False)
published = {
    ReactionKey(container, reaction.label): reaction.crossSection[result.label]
    for container in ("reactions", "sums")
    for reaction in getattr(reread, container)
}
result.verify_table(forms=published, total_budget=True)
```

Also validate the XML against the schema matching its GNDS format, for example
with `xmllint --noout --schema /path/to/gnds.xsd heated.xml`. A clean conversion
report and numerical rereading do not substitute for schema validation. Confirm
the reread heated temperature and coverage. Secondary distributions and
covariances keep their evaluated treatment; these scalar publication checks do
not establish a complete transport library at the heated temperature.

## Interpret accuracy and resources

Accuracy applies to the supplied input table; it excludes unknown physical
reconstruction error. `verify_table()` compares interpolation against the
production evaluator using distinct probes. `verify_reference()` accepts values
and uncertainty estimates from an external reference. These are empirical checks;
`global_error_bound` is false and is not a rigorous whole-domain certificate.

The report records temperatures, mass and support, source hashes, thresholds,
regions, error-budget fractions and verification ratios. A failed resource or
accuracy check raises an exception; it does not silently widen the tolerance.
`max_work_bytes` limits estimated scratch memory, not the whole process RSS or
retained models. Meshes retain their structural seeds and are not thinned.

Distributions and covariances retain their original treatment. A scalar `Heated`
style alone does not constitute a complete transport library at that temperature.
Use explicit output-style selection when publishing a heated result; read it
back and check temperature, represented coverage and interpolation precision.

See {doc}`doppler` for function signatures and result types.

The defining operator, conventions and limits are in {doc}`doppler-methods`.
