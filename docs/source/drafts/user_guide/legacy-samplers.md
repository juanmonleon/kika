# The file-level samplers (legacy)

```{admonition} Legacy
:class: warning
The samplers on this page are kept so that ensembles drawn with them can be reproduced
and extended. For new work use {py:func}`kika.sampling.model_perturbation.perturbFromModel`
({doc}`running-a-perturbation`), which perturbs any combination of quantities in one run,
keeps a stated sum consistent with its partials, records every repair in a plan and
writes a run log. The file-level samplers are not changed when `perturbFromModel` is,
so their numbers stay where they were.
```

Each sampler reads and writes one format directly (an ENDF (Evaluated Nuclear Data
File) tape, a PENDF tape, an ACE file), without going through the model, and handles one
covariance file. They take lists of files, write their factors to a parquet file and log
through a text log file (`<name>_<timestamp>.log` in `output_dir` unless `log_file` is
given). Their arguments are snake_case (`num_samples`, `output_dir`, `dry_run`,
`nprocs`).

## How they differ from `perturbFromModel`

The log-space draw is the same on both paths: the relative covariance is transformed
element by element, $\Sigma^{\log}_{kl} = \ln(1 + \Sigma_{kl})$, and
$f_k = \exp(y_k - \tfrac12\Sigma^{\log}_{kk})$ is drawn with $y$ of covariance
$\Sigma^{\log}$, so the factors reproduce $\Sigma$ exactly (see
{doc}`../methods/sampling/perturbation-pipeline`; on the file-level samplers,
`kika.sampling.multigroup_draw.draw_relative_factors`). Four differences move numbers, so
an ensemble drawn one way is not reproduced the other way:

- **Repairs inside the draw.** The same samplers apply, by default and in a fixed order:
  a rescale of bins that straddle a reaction threshold (when the thresholds are known:
  the PENDF and ACE samplers know them, the $\bar\nu$ one does not), a rescale of
  statistical-outlier variances, a cap on the relative standard deviation
  (`max_relative_std=3.0`, that is 300 %), then the positive semi-definite repair of
  `psd_method="auto"`. Rows with no stated variance are dropped from the decomposition
  and their factors set to 1, and the factors are cast to `float32`. `perturbFromModel`
  repairs definiteness only, before the draw, and writes the plan
  ({doc}`../methods/covariance/conditioning-and-psd`).
- **Stated sums.** `perturb_PENDF_files` scales each requested MT by its own block and
  does not rebuild the sums from their partials. `perturbFromModel` reproduces that only
  with `crossSectionSums=False`.
- **Angular positivity.** `perturb_ENDF_files` projects a perturbed Legendre series that
  goes negative onto the nearest non-negative one by default (`enforce_positivity=True`,
  checked on 101 cosines, skipped in a dry run); `perturbFromModel` reports it by default
  (`angularPositivity="report"`).
- **Negative eigenvalues.** A direction the matrix being decomposed states with a negative
  eigenvalue is drawn by `perturbFromModel` with zero variance. The file-level samplers
  draw it with the magnitude the SVD returns, $|\lambda|$
  (`negative_directions="fold"`), which matters for `perturb_ENDF_files`, whose default
  `psd_method="none"` decomposes MF 34 as stated; the samplers that repair definiteness
  first leave only numerical residue there.

The draw itself (scrambled Sobol by default, SVD, one seed per block) is the same code,
`kika.sampling.core`, on both paths.

## The samplers

### `perturb_PENDF_files`: cross-sections on a PENDF

`kika.sampling.pendf_perturbation.perturb_PENDF_files(endf_files, mt_list, num_samples,
...)` reconstructs each tape with NJOY RECONR (cached), reads the MF 33 covariance on its
own energy grid, draws relative factors (log space by default) and multiplies the PENDF's
pointwise MF 3 by them, bin by bin. With `output_formats=("pendf", "ace")` (the default)
it then runs NJOY on the original ENDF tape and the perturbed PENDF, skipping RECONR, to
produce ACE files and xsdir lines. An empty `mt_list` means every MT that MF 33 states.
`energy_ranges` restricts the perturbation to energy windows.

### `perturb_ENDF_files`: angular distributions in MF 4

`kika.sampling.endf_perturbation.perturb_ENDF_files(endf_files, mt_list,
legendre_coeffs, num_samples, ...)` reads MF 34 (from the tape, or from `mf34_cov_files`),
draws factors for the requested Legendre orders and multiplies the MF 4 Legendre
coefficients, writing one ENDF tape per sample; `generate_ace=True` runs NJOY on each.
It draws in linear space only: `space="log"` raises `NotImplementedError`, and so does
any decomposition other than `"svd"` or `"eigen"`. A covariance stated in the other
frame from MF 4 is converted, as in {doc}`../methods/perturbation/legendre-angular-perturbation`.
Where an order's bins end inside the distribution, the change runs lin-lin from the last
perturbed energy to the first unperturbed one (the `"ramp"` of
{doc}`../methods/perturbation/tabulated-angular-perturbation`), where `perturbFromModel`
steps. A tabulated MF 4 (LTT=2) is refused, not skipped: this sampler only scales
Legendre coefficients. An order-0 factor never scales a Legendre coefficient.

### `perturb_nubar_files`: neutron multiplicities

`kika.sampling.nubar_perturbation.perturb_nubar_files(endf_files, num_samples, ...)`
reads MF 31, draws relative factors (log space by default) and rewrites the MF 1
$\bar\nu$ tables (MT 452, 455, 456) of each tape, keeping
$\bar\nu_\text{total} = \bar\nu_p + \bar\nu_d$: one member of the family (MT 452 unless
`derived_mt` says otherwise) is recomputed from the others. `generate_ace=True` (the
default) runs NJOY on each tape. See {doc}`../methods/perturbation/multiplicity-perturbation`.

### `perturb_pfns_files`: the prompt fission neutron spectrum

`kika.sampling.pfns_perturbation.perturb_pfns_files(endf_files, num_samples, ...)` reads
the MF 35 covariance of MT 18, draws absolute deltas of the group probabilities
linearly, applies them to the MF 5/MT 18 tables (with the projection and renormalisation
of {doc}`../methods/perturbation/fission-spectrum-perturbation`) and writes one tape per
sample; `generate_ace=True` (the default) runs NJOY. MF 5/MT 455 (delayed) is written
back unchanged, since no evaluation states an MF 35 for it. The output tape keeps the
original MF 35 beside the moved spectrum, and the run summary and the parquet metadata
say so (`"mf35_unchanged": True`).

### `perturb_ENDF_PENDF_combined`: cross-sections and angular distributions, independently

`kika.sampling.combined_perturbation.perturb_ENDF_PENDF_combined(endf_files,
mt_list_mf33, mt_list_mf34, legendre_coeffs, num_samples, ...)` runs the two samplers
above one after the other: `perturb_ENDF_files` for MF 4 (seed `seed + 1`, linear by
default) and `perturb_PENDF_files` for MF 3 (seed `seed`, log by default), then pairs
sample $i$ of one with sample $i$ of the other in one NJOY run per pair. **The two files
are drawn independently**, so any covariance an evaluation states between the
cross-section and the angular coefficients (the MF 34 L=0 blocks) is absent from the
ensemble. And pairing two balanced Sobol sets by index correlates them, so the spread of
a response that mixes the two can come out narrower than independent draws give; how
much narrower depends on the response, and no test in this repository measures it.
`perturbFromModel` draws the two files in one block where the evaluation
correlates them, and in independent blocks with different seeds otherwise.

### `perturb_joint_mf33_mf34`: one draw of the joint covariance

`kika.sampling.joint_perturbation.perturb_joint_mf33_mf34(covariance_endf, base_endf,
num_samples, ...)` assembles the joint covariance of $\sigma$ (MF 33) and the Legendre
coefficients (MF 34, with its L=0 cross blocks) of one reaction, draws it once (linear by
default) and splits each sample into factors on MF 4 (written on `base_endf`) and on
MF 3 (written on the PENDF), then runs NJOY. `which` selects the ensemble: `"joint"` (the
whole matrix), `"mf34"` (the shape block alone, MF 3 left nominal) or `"mf33"` (the
magnitude block alone, MF 4 left nominal). The frozen half is not written once per
sample. It covers one reaction of one nuclide (`mt=2` and `isotope=26056` by default), and
a tape that states no L=0 cross blocks is refused under `which="joint"`.

### `perturb_ACE_files`: multigroup factors on ACE cross-sections

`kika.sampling.ace_perturbation.perturb_ACE_files(ace_files, cov_files, mt_list,
num_samples, ...)` reads a multigroup covariance matrix from a processed file
(SCALE or NJOY format), draws group factors (log space by default) and multiplies each
ACE cross-section value by the factor of the group its energy falls in. It never reads an
ENDF tape. A factor on a lumped MT is applied to every partial of the group present in
the ACE file, unless the request names only some of those partials. A reaction with any
non-positive factor is left unperturbed, with a warning.

### `reprocess_endf_to_ace`: new temperatures for an existing ensemble

`kika.sampling.reprocess_endf_to_ace.reprocess_endf_to_ace(output_dir, zaid,
num_samples, base_filename, temperatures, njoy_exe, library_name, ...)` runs NJOY again on
the perturbed ENDF tapes already in `output_dir` (the directory layout
`perturb_ENDF_files` writes) to make ACE files at other temperatures. It re-runs RECONR
on each tape, so it carries only a perturbation that lives in the ENDF tape (MF 4, MF 1).
An ensemble whose cross-sections were perturbed on the PENDF (`perturb_PENDF_files`, the
MF 3 half of the combined sampler) keeps that perturbation in its PENDF files, and
reprocessing it from the ENDF tapes would give unperturbed cross-sections.

## What ties them to `perturbFromModel`

These tests gate the equivalence of the two paths, where there is one; all passed on
2026-10-11 unless marked:

| Sampler | Gate | Test |
|---|---|---|
| `perturb_PENDF_files` | the draw is bit-identical to the sampler it replaced, and its repair sequence is the plan | `kika/sampling/tests/test_mf33_migration_equivalence.py::test_the_draw_is_bit_identical`, `::test_the_plan_is_generate_samples_repair_sequence` |
| | the model applier gives the same abscissae, values and encoded bytes as the PENDF applier | `kika/nuclear_data/model/tests/test_applying_a_perturbation_to_a_node.py::test_the_model_applier_reproduces_the_format_applier`, `::test_the_encoded_section_is_byte_identical` |
| | end to end with NJOY | `kika/sampling/tests/test_pendf_perturbation.py::test_pendf_only_smoke` (needs NJOY; passed 2026-10-21) |
| `perturb_ENDF_files` | the MF 34 draw is bit for bit the draw it replaced | `kika/sampling/tests/test_mf34_migration_equivalence.py::test_the_draw_itself_is_bit_for_bit` |
| | the model applier refines the same energies and gives the same coefficients | `kika/nuclear_data/model/tests/test_applying_a_perturbation_to_an_angular_distribution.py::test_the_two_appliers_refine_the_same_energies`, `::test_the_two_appliers_produce_the_same_coefficients` |
| `perturb_nubar_files` | the model applier gives the same $\bar\nu$ table | `kika/nuclear_data/model/tests/test_applying_a_perturbation_to_a_multiplicity.py::test_the_two_appliers_produce_the_same_table` |
| | the family sum rule | `kika/sampling/tests/test_mf31_sampling.py::test_family_prompt_only_derives_total_and_keeps_delayed` and the other `test_family_*` tests; on JEFF-4.0 U-235, Th-232 and Pu-241, `::test_end_to_end_write_satisfies_sum_rule` (passed 2026-10-20) |
| `perturb_pfns_files` | the model draws MF 35 as the sampler does | `kika/sampling/tests/test_the_model_draws_a_fission_spectrum_as_the_legacy_does.py::test_the_model_draw_is_the_legacy_draw` |
| | the two write the same MF 5/MT 18 bytes | `kika/sampling/tests/test_the_model_writes_a_fission_spectrum_as_the_legacy_does.py::test_the_two_writers_write_the_same_spectrum_on_the_micro_tape`; on JEFF-4.0 U-235, ENDF/B-VIII.1 Cf-252 and U-235 and JEFF-4.0 Pu-242, `::test_the_two_writers_write_the_same_spectrum_on_jeff40_u235` and its siblings (passed 2026-10-20) |
| `perturb_joint_mf33_mf34` | the shape block is what the MF 34 sampler assembles; the cross reaches the ensemble | `kika/sampling/tests/test_joint_mf33_mf34.py::test_the_shape_block_is_bit_for_bit_what_the_shipped_sampler_assembles`, `::test_the_cross_reaches_the_ensemble` |
| `perturb_ENDF_PENDF_combined` | none against the model path; argument checks pass, and the NJOY smoke test `::test_combined_smoke` makes ACE files (passed 2026-10-21) | `kika/sampling/tests/test_combined_perturbation.py` |
| `perturb_ACE_files`, `reprocess_endf_to_ace` | none against the model path; `perturb_ACE_files` is covered for its argument types only | `kika/sampling/tests/test_normalize_mt_list.py` |

`perturbFromModel` has no ACE-file input and no reprocessing entry: it perturbs an
evaluation, and produces ACE through NJOY with its `ace` emitter.
