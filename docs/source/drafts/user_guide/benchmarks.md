# Benchmarks: ICSBEP/DICE and SINBAD

kika reads two benchmark collections: criticality benchmarks (ICSBEP), through a local
database built from DICE sensitivity data, and shielding benchmarks (SINBAD), from their
XML entries. The methods used on them (similarity $c_k$, the uncertainty of a benchmark's
response) are in {doc}`../methods/uq/similarity-and-benchmarks`.

## Criticality benchmarks (ICSBEP, DICE)

kika does not ship ICSBEP data. Build a local SQLite database from your own DICE export
once, then query it:

```python
import kika.benchmarks as bm

bm.build_benchmarks_db(source_dir="dice_export/", db_path="benchmarks.db")
bm.configure(db_path="benchmarks.db")

hits = bm.find_sensitive_benchmarks("Fe56", reaction=2, energy_region="fast",
                                    sensitivity_threshold=0.01)
top = bm.rank_benchmarks("Fe56", reaction=2, energy_region="fast", limit=20)
```

`energy_region` is `thermal`, `epithermal`, `fast` or `total`. For one benchmark,
`get_benchmark_dashboard`, `get_experimental_keff`, `get_sensitivity_profile`,
`get_spectrum`, `get_balance` and `get_input_deck` return what the database holds.

To rank benchmarks by their similarity to an application, with a covariance:

```python
import kika

app = kika.read_sdf("application.sdf")      # the application's sensitivities
ranking = bm.rank_benchmarks_by_ck(app, covariance=cov, limit=20)   # cov: a CrossSectionCovariance
```

## Shielding benchmarks (SINBAD)

```python
import kika.sinbad as sinbad

b = sinbad.read("asp_fe88/")        # the entry's XML file, or the directory holding it
print(b.summary())                  # detectors, positions, data objects, calculations
df = b.to_dataframe()               # every measured point, one row each
b.ce()                              # calculated over experimental, per calculation
b.covariance(relative=True)         # the covariance the uncertainty budget implies
b.check()                           # links, checksums and consistency of the entry
```

`b.measurements`, `b.calculations` and `b.issues` give the parts of the entry; the
uncertainty budget (`UncertaintyBudget`) keeps each component with its correlation scope.
`sinbad.plot_profile`, `plot_ce` and `plot_correlation` draw them, and
`kika.sinbad.gnds.to_kika_materials` turns the entry's materials into kika materials (see
{doc}`materials-and-energy-grids`).
