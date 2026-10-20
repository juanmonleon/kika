# Experimental data from EXFOR

`kika.exfor` reads experimental reaction data from the IAEA's X4Pro database (EXFOR in
SQLite form) or from per-dataset JSON files: angular distributions, cross-sections and any
other quantity, converted to kika's units, with frame conversion, energy resolution and
comparison against evaluated data.

The X4Pro SQLite file is distributed by the IAEA Nuclear Data Section; kika does not ship
it. Point kika at it once per session:

```python
import kika.exfor as exfor

exfor.configure(db_path="x4sqlite1.db")     # or set KIKA_X4PRO_DB_PATH
```

## Angular distributions

```python
data = exfor.read_all_exfor(target="Fe-56", mt=2)                      # grouped by energy
data = exfor.read_all_exfor(target="Fe-56", mt=2, energy_range=(1.0, 5.0))  # MeV
```

Each item is an `ExforAngularDistribution`: the angles, the differential cross-section and
its uncertainties at one incident energy, with the dataset's metadata (`label`, `dataset_id`,
`exfor_url`, `detector`, `sample`). It converts:

- **frame**: `convert_to_cm`, `convert_to_lab` (two-body kinematics, with the Jacobian);
- **units**: `convert_angle` (degrees or cosines), `convert_energy`, `convert_cross_section`;
- **energy resolution**: `compute_energy_resolution`, from the time-of-flight parameters
  kika keeps for curated datasets (`kika/exfor/exfor_tof_parameters.json`);

and compares: `compare_with_ace(ace)` against an ACE file, `plot`, `to_plot_data` (for the
plotting builders), `to_dataframe`.

## Cross-sections and other quantities

```python
db = exfor.X4ProDatabase()                        # uses the configured path
xs = db.query_cross_sections(target="Fe-56", mt=102, energy_range=(1e-3, 20.0))
hits = db.search_datasets(targets=["Fe-56"], quantities=["DA"], year_min=1990,
                          energy_min_mev=1.0, energy_max_mev=5.0)
exp = db.load_experiment_general(hits[0]["dataset_id"])     # any quantity
```

`ExforCrossSection` offers `filter`, `interpolate`, `summary`, `to_dataframe`;
`ExforExperiment` holds a dataset of any quantity in canonical units. `list_targets`,
`list_unique_quantities` and `get_statistics` describe the database;
`create_subset_database` writes a small database for one target or reaction, convenient to
carry around.

## Plots

`kika.exfor.plotting` draws angular distributions (`plot_exfor_angular`,
`plot_multiple_energies`) and their comparison with an ACE file
(`plot_exfor_ace_comparison`); every EXFOR object also goes through
`kika.plotting.plottable` (see the plotting guide).

## Legacy functions

`kika.exfor.AD_utils` holds the older function-based interface. It is deprecated; the
classes above replace it.

A target is named as X4Pro names it, `Fe-56`. `read_all_exfor` also accepts `Fe56`; the
queries of `X4ProDatabase` do not, and find nothing under that spelling.
