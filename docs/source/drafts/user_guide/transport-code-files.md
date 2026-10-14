# MCNP, Serpent and weight-window files

kika reads the input and output files of the transport codes that consume nuclear data,
so that a calculation's results can be taken back into a sensitivity or uncertainty
analysis.

## MCNP

- **Input**: `kika.read_mcnp(path)` reads materials, cells and PERT cards; see
  {doc}`/user_guide/tutorials/input_files`.
- **Tally output (MCTAL)**: `kika.read_mctal(path)` returns the tallies. Each tally holds
  MCNP's eight bin axes as the file writes them, totals included: `to_xarray()` gives the
  labelled bins, `total(*axes)` the total bins over the axes named, and `to_dataframe()`
  a flat table; each takes `perturbation=N` for the PERT tallies. See
  {doc}`/user_guide/tutorials/mctal_files`.
- **Mesh tallies (FMESH)**: `kika.read_meshtal(path)` reads a MESHTAL text file or the
  HDF5 runtape of MCNP 6.3, rectangular or cylindrical, into the same object, with the
  same `to_xarray`, `total` and `to_dataframe` views; see
  {doc}`/user_guide/tutorials/mesh_tallies`.
- **PERT cards** for sensitivities: `kika.mcnp.generate_pert_cards` and
  `perturb_material`; see {doc}`/user_guide/tutorials/pert_cards` and, for what is computed
  from them, {doc}`../methods/uq/sensitivities`.
- `kika.mcnp.detect_mcnp_file_type(path)` tells input from output; `kika.identify(path)`
  names any file kind kika knows.

## Serpent

```python
import kika.serpent as serpent

inp  = serpent.read_serpent_input("model.inp")      # materials, detectors, grids, sensitivity setup
sens = serpent.read_sensitivity_file("model_sens0.m")
det  = serpent.read_detector_file("model_det0.m")
res  = serpent.read_results_file("model_res.m")
his  = serpent.read_history_file("model_his0.m")
dep  = serpent.read_depletion_file("model_dep.m")
```

A `SensitivityFile` gives sensitivities by reaction (`by_mt`) or Legendre moment
(`by_legendre`), integrated (`get_integrated`) or energy-dependent
(`get_energy_dependent`), as tables (`to_dataframe`, `to_xarray`) or plots
(`plot_energy_sensitivity`). `kika.sensitivities.sensitivity_processing.create_sdf_from_serpent`
turns it into an SDF file for the sandwich rule ({doc}`../methods/uq/sandwich`).

## Weight windows (WWINP)

```python
from kika.wwinp import read_wwinp, write_wwinp

ww = read_wwinp("wwinp")
ww.multiply(1.2)                    # scale every bound
ww.soften(0.6)                      # raise the bounds to a power (< 1 softens, > 1 hardens)
n = ww.apply_ratio_threshold(10.0)  # zero the cells whose ratio to a neighbour exceeds 10
ww.query(particle=1, energy=1.0, x=5.0, y=(0, 10))   # bounds at a point or over ranges
write_wwinp(ww, "wwinp.new")
```

Each operation can be restricted to some particles (`particles=`). `kika.wwinp.plotter.WWPlotter`
draws the map in three dimensions or as slices (it needs PyVista).

Spatial coordinates in `query` are keyword arguments named after the mesh axes (`x`, `y`,
`z` for a Cartesian mesh). The Serpent readers have almost no tests in the repository,
and the WWINP package has none.
