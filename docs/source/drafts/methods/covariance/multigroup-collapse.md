# Collapsing a covariance onto a multigroup structure

```{admonition} At a glance
:class: note
**Computes:** the covariance matrix of group-averaged quantities on a chosen group
structure, from a covariance stated on the evaluator's own energy bins: the Legendre
coefficients of MF 34 (relative and absolute, weighted by $\sigma(E)\,\phi(E)$), and
cross-section covariances (MF 33, MF 31) projected onto another grid.

**Applies to:** MF 34 with its MF 4 and a pointwise $\sigma(E)$; any covariance matrix
stated on bins (`CrossSectionCovariance`, the MF 31 and MF 33 samplers' union grids).

**Entry points:** `kika.cov.multigroup.collapse.MF34_to_MG` (alias
`collapse_to_multigroup`), `kika.processing.multigroup.compute_rebin_operator` with
`kika.processing.multigroup.collapse_covariance`,
`kika.cov.CrossSectionCovariance.project_to_grid`.

**Status:** experimental — since 0.4.0.
```

## Background

An ENDF (Evaluated Nuclear Data File) covariance is stated on energy bins chosen by the
evaluator: an element $R_{ij}$ of MF 34 is the relative covariance of the Legendre
coefficients $a_\ell$ and $a_{\ell'}$ (ENDF-6 §34.1),

$$
R_{ij} = \frac{\operatorname{Cov}\big(a_\ell(E), a_{\ell'}(E')\big)}{a_\ell(E)\,a_{\ell'}(E')},
\qquad E \in [E_i, E_{i+1}),\; E' \in [E_j, E_{j+1}),
$$ (eq-multigroup-collapse-mf34)

constant over each pair of bins, with $f(\mu, E) = \sum_\ell \frac{2\ell+1}{2} a_\ell(E)
P_\ell(\mu)$ the angular distribution of MF 4. A sub-subsection (L, L1) of MF 34 may be
made of several LIST records with different LB flags and grids (ENDF-6 Ch. 34, with the
LB flags of MF 33, ENDF-6 Ch. 33); kika sums them, each projected onto the union of
their grids, into one matrix per (MT, L, MT1, L1) before any collapse. An LB=0 record
states an absolute covariance and every other LB a relative one, so in a sub-subsection
that mixes the two the LB=0 part is first divided by $\bar a_{\ell,i}\,\bar a_{\ell',j}$,
the exact cell means of the coefficients from MF 4 on that record's own grid (a cell whose
mean is zero gets zero), and the sum is relative; a sub-subsection stated wholly in LB=0
stays absolute. Negative variances are kept as the file states them. A matrix with
L $\ne$ L1 is a cross-covariance between two orders and need not be symmetric. The flag
LCT says whether the coefficients are in the laboratory (LAB) or centre-of-mass (CM)
frame, or in the frame of MF 4 (LCT=0).

A multigroup calculation (a sensitivity profile, a transport code's group library) needs
the covariance of the **group averages** of the same quantity on its own group
structure $G_0 < G_1 < \dots < G_n$. A group average of a quantity $x(E)$ under a weight
$w(E)$ is

$$
\bar x_g = \frac{\int_{G_g}^{G_{g+1}} w(E)\,x(E)\,dE}{\int_{G_g}^{G_{g+1}} w(E)\,dE},
$$ (eq-multigroup-collapse-group-average)

and the weight for a reaction's angular data is the reaction rate density,
$w(E) = \sigma(E)\,\phi(E)$, with $\phi$ a weighting flux. Collapsing a covariance is
finding the covariance of the $\bar x_g$ from the covariance of $x$ on the evaluator's
bins.

## What it computes

**For MF 34** (`MF34_to_MG`). Input: an ENDF object with MF 4 and MF 34, a pointwise
$\sigma(E)$ per reaction in its `pendf` attribute, a group structure and a weighting flux
$\phi$. Output: a `kika.cov.multigroup.MultigroupLegendreCovariance` holding, for every
(MT, L, MT1, L1) block of MF 34, the relative and the absolute covariance matrix of the
group-averaged coefficients, $n \times n$, and the group means $\bar a_{\ell,g}$. For a
relative block,

$$
R^{\mathrm{MG}}_{gg'} =
\frac{\sum_{i \in g}\sum_{j \in g'} w_i A_{\ell,i}\, R_{ij}\, A_{\ell',j}\, w_j}
     {W_g\,\bar a_{\ell,g}\; W_{g'}\,\bar a_{\ell',g'}},
\qquad
C^{\mathrm{MG}}_{gg'} = \bar a_{\ell,g}\, R^{\mathrm{MG}}_{gg'}\, \bar a_{\ell',g'},
$$ (eq-multigroup-collapse-result)

where $i$ and $j$ run over the cells of the union grid that fall in groups $g$ and $g'$,
$w_i$ is the integral of $\sigma\phi$ over cell $i$, $A_{\ell,i}$ the $\sigma\phi$-weighted
mean of $a_\ell$ over it, $W_g = \sum_{i \in g} w_i$ and $\bar a_{\ell,g}$ the group mean
(all defined in *Method*). This is the covariance of the group averages
$\bar a_{\ell,g}$, {eq}`eq-multigroup-collapse-group-average`, if the relative covariance
{eq}`eq-multigroup-collapse-mf34` holds between the cell means $A_{\ell,i}$, divided by the
product of the group means.

**For a covariance on bins** (`compute_rebin_operator`, `collapse_covariance`). Input: a
covariance $C$ on bins $h$ of one grid and a target grid of bins $g$. Output:

$$
C' = M\,C\,M^\top, \qquad
M_{gh} = \frac{\int_{g \cap h} \phi(E)\,dE}{\int_{g} \phi(E)\,dE}.
$$ (eq-multigroup-collapse-rebin)

## Method

### MF 34: the coefficients on their own mesh

**1. The source.** For each reaction, the coefficients $a_\ell(E)$ are read once into a
`LegendreSource` (`kika.cov.multigroup.collapse.LegendreSource`): the values of every
order at the section's own incident energies, the frame and the AWR. From an ENDF MF 4
section (`legendre_source_from_endf`) the energies are the Legendre energies, joined for
LTT=3 with the tabulated ones (sorted, duplicates removed), and the coefficients come
from the MF 4 reader: a Legendre table under its energy interpolation law, a tabulated
$f(\mu)$ projected onto $P_\ell$ by the reader and normalised so that $a_0 = 1$. An order
the file does not carry at an energy is zero. From the canonical model
(`legendre_source_from_model`), the coefficients of each Legendre child of an
`AngularTwoBody` are read at its own incident energy, repeated energies kept, with no
interpolation between energies.

**2. The frame.** With `output_frame="lab"` and a CM distribution, the coefficients are
taken to the LAB frame before averaging. At every source energy, $f$ is rebuilt from all
the orders the file carries (up to 64) and projected onto the LAB polynomials by a
64-point Gauss-Legendre rule in $\mu_C$:

$$
a^{\mathrm{LAB}}_\ell = \sum_{q=1}^{64} w_q\, f(\mu_q)\, P_\ell\big(\mu_L(\mu_q)\big),
\qquad
\mu_L = \frac{\mu_C + \gamma}{\sqrt{1 + 2\gamma\mu_C + \gamma^2}},
$$ (eq-multigroup-collapse-cm-lab)

with $(\mu_q, w_q)$ the Gauss nodes and weights. This is
$\int f_L(\mu_L) P_\ell(\mu_L)\,d\mu_L$ after the change of variable
$f_L\,d\mu_L = f_C\,d\mu_C$. The speed ratio $\gamma$ is that of two-body kinematics for a
neutron of energy $E$ on a target of mass ratio $A$ (the AWR) with reaction Q-value $Q$:

$$
\gamma(E) = \frac{1}{A}\sqrt{\frac{E}{E + Q\,(A+1)/A}},
$$ (eq-multigroup-collapse-gamma)

which is $1/A$ at every energy for elastic scattering (MT 2, $Q = 0$) and depends on $E$
for a discrete inelastic level (MT 51-90, with $Q$ the level's QI read from MF 3). At or
below the threshold, where the root is not real, the coefficients are left as they are.

**3. Cell means.** On a grid of cells $[E_i, E_{i+1}]$ (the group structure, or the union
grid of step 5), the weighted mean of $a_\ell$ over each cell is

$$
A_{\ell,i} = \frac{\int_{E_i}^{E_{i+1}} \sigma(E)\,\phi(E)\,a_\ell(E)\,dE}{w_i},
\qquad
w_i = \int_{E_i}^{E_{i+1}} \sigma(E)\,\phi(E)\,dE .
$$ (eq-multigroup-collapse-cell-mean)

Inside each cell the integration mesh is the two cell edges, the source energies inside
the cell and the energies of the $\sigma(E)$ table inside the cell. $a_\ell$ is
interpolated linearly in $E$ between source energies; outside the source's range $a_0$ is
held at its end value and every $a_{\ell \ge 1}$ is zero. $\sigma(E)$ is evaluated under
its own interpolation laws, held at its end values outside its table and floored at
zero. Both integrals are trapezoidal on that mesh. When the reaction has no $\sigma(E)$ in
`pendf`, the weight is $\phi$ alone, with a warning; then $\phi = 1$ gives the exact
integral of the linear $a_\ell$, and $\phi = 1/E$ is integrated analytically on every
linear segment,

$$
\int_{e_0}^{e_1} \frac{a_0 + m (E - e_0)}{E}\,dE
= a_0 \ln\frac{e_1}{e_0} + m\Big[(e_1 - e_0) - e_0 \ln\frac{e_1}{e_0}\Big],
$$ (eq-multigroup-collapse-lethargy)

with $m$ the slope of $a_\ell$ on the segment and $w_i = \ln(E_{i+1}/E_i)$.

**4. The weighting flux.** $\phi$ is one of:

| `weighting_function` | $\phi(E)$, $E$ in eV |
|---|---|
| `"constant"` (default) | $1$ |
| `"lethargy"` | $1/E$ |
| `"maxwellian"` | $\sqrt{E}\,e^{-E/kT}$, $kT = 0.0253$ eV |
| `"fission"` | $\sqrt{E}\,e^{-E/\theta}$, $\theta = 1.29\cdot10^{6}$ eV |

### MF 34: a relative block

**5. The union grid.** The MF 34 bin edges and the group edges are merged (sorted, unique),
so every cell of the union grid lies inside at most one MF 34 bin and at most one group.
The relative matrix is carried onto it by copying: a union cell takes the row and column
of the MF 34 bin that contains its midpoint. A union cell outside the MF 34 grid (where
the group structure reaches further) gets a zero row and column, since MF 34 states no
covariance there.

**6. Group means.** Steps 3 and 4 run on the union grid, for the row order $\ell$ and the
column order $\ell'$. Each union cell is assigned to the group that contains its midpoint;
a cell outside the group structure (where the MF 34 grid reaches further) belongs to no
group and enters no sum. Then

$$
W_g = \sum_{i \in g} w_i,
\qquad
\bar a_{\ell,g} = \frac{\sum_{i \in g} w_i A_{\ell,i}}{W_g},
$$ (eq-multigroup-collapse-group-means)

with $\bar a_{\ell,g} = 0$ where $W_g \le 10^{-15}$.

**7. The collapse.** $R^{\mathrm{MG}}$ is {eq}`eq-multigroup-collapse-result`. An element
whose denominator is at most $10^{-15}$ in magnitude is set to zero, and so is any
non-finite one.

**8. The absolute matrix.** With `relative_normalization="mf34_cell"` (the default),
$C^{\mathrm{MG}}$ uses the $\bar a_{\ell,g}$ of step 6. With `"mg_cell"` it uses
group means computed by steps 3 and 4 directly on the group structure. The group means
stored with the result are always the latter.

### MF 34: an absolute block

**9.** A sub-subsection stated wholly in LB=0 records (absolute covariance) is collapsed
with the operator of {eq}`eq-multigroup-collapse-rebin`, from the MF 34 bins to the
groups, with $\phi$ replaced by $\sigma\phi$ (its integral taken as the cumulative
trapezoid of $\sigma\phi$ on the $\sigma(E)$ table, interpolated linearly). The relative
matrix is then $C^{\mathrm{MG}}_{gg'} / (\bar a_{\ell,g} \bar a_{\ell',g'})$ with the
group means of step 8; an element whose mean is at most $10^{-15}$ in magnitude is NaN
until step 10, which treats it as zero.

### MF 34: conditioning each block

**10.** Only a self block (the same isotope, MT and $L$ on both sides) is conditioned, and
once, in the form the file states it: a relative block in relative form, an absolute
block in absolute form. It is symmetrised, $C \leftarrow \frac12 (C + C^\top)$, and, with
`project_to_psd=True` (the default), made positive semi-definite by the automatic rule of
`kika.cov.decomposition`: with $\lambda_{\min}$, $\lambda_{\max}$ the extreme eigenvalues
(of a copy whose non-finite entries are zero), the eigenvalues are clipped at zero,
$C \leftarrow V \max(\Lambda, 0) V^\top$, when
$\max(-\lambda_{\min}, 0)/\lambda_{\max} < 10^{-2}$, and the matrix is replaced by the
nearest positive semi-definite matrix with the same diagonal
({py:func}`kika.cov.decomposition.nearest_psd_higham`) otherwise. Both repairs are defined
in {doc}`conditioning-and-psd`. The other form is then derived from the conditioned one
with the same group means (step 12), so the relative and the absolute block remain one
covariance. A cross block ($L \ne L_1$, or two reactions) is an off-diagonal part of a
larger covariance matrix, has no symmetry or sign of its own, and is stored as collapsed.

### A covariance on bins

**11. The rebin operator.** For every pair of target bin $g$ and source bin $h$, the
overlap $[\max(g_{\mathrm{lo}}, h_{\mathrm{lo}}), \min(g_{\mathrm{hi}}, h_{\mathrm{hi}})]$
is integrated with the antiderivative of $\phi$ and divided by the integral of $\phi$ over
$g$, {eq}`eq-multigroup-collapse-rebin`. A target bin whose $\phi$ integral is below
$10^{-15}$ gets a row of zeros, and so does a target bin outside the source grid. The
covariance is then $M C M^\top$. When the target grid refines the source grid (every
target bin inside one source bin), $M$ has one 1 per row, and $M C M^\top$ copies the
source covariance onto the finer bins; this is how the MF 31 and MF 33 samplers bring
the records of a section onto one union grid (`kika.sampling.mf33_sampling`).

**12. Relative and absolute.** `relative_to_absolute` is
$C = \operatorname{diag}(x)\,R\,\operatorname{diag}(x')$; `absolute_to_relative` is its
inverse, with NaN in every row or column whose mean is at most $10^{-15}$ in magnitude.

**13. A cross-section covariance on another grid.** `CrossSectionCovariance.project_to_grid`
takes every self-covariance block of one reaction (MT row = MT column), turns a relative
block absolute with $\sigma$ evaluated at the **midpoint** of each source bin, collapses it
with $\phi = 1$ by step 11, sums the blocks and symmetrises the sum. The result is an
absolute covariance in b².

## Conventions and choices

- **The weight is $\sigma\phi$, always, and $\sigma$ is never guessed.** The angular
  distribution is a distribution per reaction, so its group average is weighted by the
  reaction rate, as in NJOY's ERRORR (TODO(ref)). `MF34_to_MG` refuses to run without a
  $\sigma(E)$ in `endf_object.pendf`, because a wrong weight leaves no trace in the result.
  The caller fills it, for example from NJOY (`kika.processing.njoy_reconstruct`).
- **A relative block is collapsed as relative, with $a_\ell$ in the weight**
  ({eq}`eq-multigroup-collapse-result`), and not converted to absolute with the MF 34 bin
  mean of $a_\ell$ first. When $a_\ell$ changes sign inside an MF 34 bin, its bin mean is
  small and the uncertainty it carries would cancel.
- **The cross-section of the row reaction weights the whole block.** For a block between
  two reactions, $\sigma$ of MT is used for both the row and the column.
- **$a_\ell$ is linear in $E$ between the source's energies** whatever MF 4's own energy
  law, and a repeated incident energy (a step) is collapsed into one point on the ENDF path.
  The averaging has no way to represent the jump.
- **Outside the source's range, $a_{\ell\ge1} = 0$ and $a_0$ is held**, the ENDF
  convention for an order the file does not state.
- **The two normalisations differ only in how the group means are integrated.** Both are
  $\sigma\phi$-weighted means of $a_\ell$ over the group; `"mf34_cell"` sums them from the
  union cells, `"mg_cell"` integrates them on the group structure. They agree to the
  trapezoidal error of the two meshes, except in the edge case listed under *Limits*.
- **Each block is conditioned on its own.** A block is the unit MF 34 states, and the
  result keeps the blocks separate; a positive semi-definite projection of the assembled
  matrix (`MultigroupLegendreCovariance.covariance_matrix`) is left to the caller (see {doc}`conditioning-and-psd` and
  `kika.cov.conditioning`). A self block is projected in the form the file states and the
  other form is derived from the projected one, so
  $C^{\mathrm{MG}} = \operatorname{diag}(\bar a)\,R^{\mathrm{MG}}\operatorname{diag}(\bar a')$
  holds after a projection too (a congruence of a PSD matrix is PSD).
- **Cross blocks are not conditioned.** Symmetrising or projecting an off-diagonal part on
  its own changes it into something MF 34 does not state; whether the joint matrix is PSD
  is a property of the whole, checked where it is assembled.
- **Nothing outside the group structure, no covariance outside MF 34.** A group average is
  an integral over the group, so the weight of energies outside every group is not added
  to the end groups; and a group, or part of one, outside the MF 34 grid has no stated
  covariance, so it gets none, and no copy of the nearest MF 34 bin.
- **In the LAB output frame, only the coefficients change frame.** The relative block of
  a CM covariance is kept as the relative covariance of the LAB coefficients, and the
  block is labelled LAB. A relative covariance is not invariant under the frame change
  (each LAB coefficient mixes every CM order), so this is an approximation.
  TODO(check): its size has not been measured.
- **The CM-to-LAB map is two-body.** {eq}`eq-multigroup-collapse-gamma` holds for elastic
  scattering and for a discrete inelastic level, the reactions whose MF 4 is a two-body
  angular distribution; for any other reaction a CM distribution has no single LAB
  counterpart at a given angle, and `output_frame="lab"` refuses it.
- **A bin-centre $\sigma$ in `project_to_grid`.** The relative-to-absolute step uses
  $\sigma$ at the midpoint of each source bin, not its bin average; on a bin that spans a
  resonance the two differ.

## Limits and failure modes

Refused by `MF34_to_MG`:

- no MF 4 or no MF 34 in the object; a named grid that `kika.energy_grids.grids` does not
  have; a grid with fewer than two edges or not strictly increasing; an unknown
  `weighting_function` name; `output_frame` other than `"same"` or `"lab"`;
  `relative_normalization` other than `"mf34_cell"` or `"mg_cell"`; `pendf` not set
  (`ValueError`);
- with `output_frame="lab"`, a reaction stated in CM that is neither elastic (MT 2) nor a
  discrete inelastic level (MT 51-90), and a level whose QI cannot be read because the tape
  has no MF 3 section for it (`ValueError`);
- a `weighting_function` given as a callable (`NotImplementedError`): the collapse needs
  its antiderivative and has no way to receive one;
- a `pendf` entry that exposes neither `energies`/`values` nor `energies`/`cross_sections`
  nor the ENDF MF 3 arrays (`TypeError`);
- an MF 34 block whose frame (LCT=1 or 2) differs from the MF 4 frame (`ValueError`). LCT=0
  is always accepted.

Refused by `legendre_source_from_model`: an isotropic distribution (`NotImplementedError`,
it has no energy mesh); a tabulated child under `on_tabulated="raise"`, the default
(`NotImplementedError`), since projecting a table onto Legendre orders is the MF 4
reader's job; `on_tabulated` other than `"raise"` or `"skip"` (`ValueError`). Under
`"skip"` the source covers the Legendre region only.

Reported instead of done (as Python warnings): a block whose reaction has no MF 4 section
is skipped; an MF 4 section without a Legendre mesh is integrated on the MF 34 grid; a
reaction with no $\sigma(E)$ in `pendf` is weighted by $\phi$ alone. In the MF 34 reader,
negative diagonal entries below $-10^{-9}$ of an L=L1 matrix are kept as stated, with a
logged warning; repairing them is the conditioning step's decision. The reader refuses
(`ValueError`) a sub-subsection that mixes LB=0 and relative records when it is given no
MF 4 to divide by; `MF34_to_MG` always passes the MF 4 of the object.

Limits of the method and of the code:

- **A group outside the MF 34 grid has zero covariance**, and a group that only partly
  overlaps it carries the covariance of the overlapping part only: its relative variance
  is that of the stated part diluted by the group's whole weight.
- With $\sigma$ in the weight the integrals of step 3 are trapezoidal on a mesh that
  includes every energy of the $\sigma(E)$ table, so their accuracy is that of the table.
- A relative element is undefined where a group mean $\bar a_{\ell,g}$ is near zero; it is
  set to zero (relative blocks) or NaN before conditioning (absolute input), and a small but non-zero group
  mean can give a large relative variance.

## Validation

All tests below were run on 2026-10-11 and passed.

- **The rebin operator and the congruence.** For a fixed 4-bin covariance collapsed onto
  8 bins with $\phi = 1$, the operator rows sum to 1 to $10^{-12}$, the result is symmetric
  to $10^{-12}$, and operator and result are frozen to a relative $10^{-9}$
  (`kika/processing/tests/test_numeric_goldens.py::test_multigroup_collapse_golden`).
- **The MF 34 collapse, frozen.** MF 34 MT 2 (orders 1-6) of a slice of the JEFF-4.0
  Fe-56 tape, collapsed onto four groups from 0.1 MeV to 20 MeV with $\sigma$ set to a
  constant 1 b: every relative and absolute matrix is frozen to a relative $10^{-9}$
  (`kika/cov/tests/test_mf34_to_mg_golden.py::test_collapsed_matrices_golden`), and the
  weight used is frozen beside it (`::test_the_sigma_weight_golden`). These are
  characterisation tests: they detect a change and do not say the numbers are right. The
  slice's MF 34 grid starts at $10^{-5}$ eV, far below the first group, so they also
  freeze that the cells below it carry no weight. Every self block is symmetric to
  $10^{-10}$ (`::test_the_collapse_is_symmetric`, which step 10 guarantees); the result is
  on the grid asked for (`::test_the_grid_is_what_was_asked_for`), is refused without a
  $\sigma$ (`::test_it_refuses_to_guess_a_sigma`) and with an unknown normalisation
  (`::test_unknown_normalisation_is_refused`).
- **The edges.** Union cells outside the groups carry no weight, and the covariance is zero
  outside the MF 34 grid
  (`kika/cov/tests/test_collapse_edges_and_frames.py::test_cells_outside_the_groups_carry_no_weight`,
  `::test_the_covariance_stops_where_mf34_does`).
- **The MF 34 reader.** An LB=0 part of a mixed sub-subsection is divided by the MF 4 cell
  means before the sum, a mixed sub-subsection without MF 4 is refused, a wholly LB=0 one
  stays absolute, and a negative variance is kept
  (`kika/endf/tests/test_mf34_reader_mixed_and_negative.py`).
- **The $\sigma$ table's shape does not move the result.** $\sigma$ given as an ENDF MF 3
  section and as a `CrossSection` gives bit-identical matrices
  (`kika/tests/test_duck_typed_consumers.py::test_mf34_to_mg_gives_the_same_answer_for_both_pendf_shapes`);
  an unknown shape is refused (`::test_a_pendf_section_of_an_unknown_shape_is_refused`);
  the same holds for `project_to_grid` (`::test_project_to_grid_agrees_for_both_types`).
- **Cell means at every order.** Constant coefficients come back as their values to
  $10^{-9}$ under lethargy weighting, whether the orders are asked for at once or one by
  one through a shared cache
  (`kika/cov/tests/test_collapse_coeff_cache.py::test_shared_cache_preserves_higher_orders`,
  `::test_single_call_all_orders_consistent_with_incremental`).
- **The two producers of the coefficients.** On the Fe-56 slice (MT 2, LTT=3, orders 0-8),
  the model path returns the ENDF path's coefficients bit for bit at every Legendre
  energy (`kika/cov/tests/test_legendre_source_from_model.py::test_the_two_producers_agree_on_every_order`),
  with $a_0 = 1$ exactly (`::test_a_0_is_one_and_is_stated_rather_than_implied`), the same
  frame (`::test_the_frame_survives_the_trip`), the repeated energy the ENDF path drops
  (`::test_the_model_keeps_the_repeated_energy_the_endf_mesh_drops`), and a tabulated
  child refused (`::test_a_tabulated_child_is_refused_rather_than_projected`).
- **The CM-to-LAB transform.** {eq}`eq-multigroup-collapse-cm-lab` agrees with an
  independent transfer matrix to $10^{-13}$ absolute for $A$ = 1.9, 11.9, 55.45 and 236
  (`kika/sampling/tests/test_mf34_frame_conversion.py::test_the_transfer_matrix_agrees_with_the_collapse_transform`).
  The test uses 256 quadrature points; the collapse uses 64. The collapse's conversion
  agrees with the frame module's transfer matrix to $10^{-10}$ for $\gamma = 1/55.45$ and
  0.3, one $\gamma$ per energy gives the conversion at each energy, and below threshold the
  coefficients are unchanged
  (`kika/cov/tests/test_collapse_edges_and_frames.py::test_the_lab_coefficients_match_the_frame_module`,
  `::test_one_gamma_per_energy_is_one_conversion_per_energy`); elastic and a level get their
  own kinematics, and a reaction that is not two-body is refused
  (`::test_elastic_and_a_level_get_their_own_kinematics`,
  `::test_a_reaction_that_is_not_two_body_is_refused`).

TODO(check): no test compares `MF34_to_MG` with NJOY ERRORR's multigroup MF 34 output,
nor the lethargy, Maxwellian or fission weighting against an independent integral.

## Diagnostics

`MF34_to_MG` returns a `MultigroupLegendreCovariance` with:

- `energy_grid`: the group edges; `energy_unit`, taken from the MF 34 carrier;
- per block, `isotope_rows`, `reaction_rows`, `l_rows` and their column twins,
  `relative_matrices`, `absolute_matrices`, and `frame` (`"LAB"` when the coefficients were
  taken to LAB, else the MF 34 frame, `"same-as-MF4"` for LCT=0);
- `legendre_coefficients`: the group means $\bar a_{\ell,g}$ per (isotope, MT, L), from
  the group structure;
- `weighting_function`, a label such as `"constant (sigma-weighted)"`, and
  `relative_normalization`. **When a reaction fell back to $\phi$ alone, the label names
  it**, for example `"constant (sigma-weighted; flux-only for MT51, which had no PENDF)"`;
  its blocks are weighted differently from the others.

**Worth a look:** a relative variance much larger than the MF 34 one marks a group whose
mean $\bar a_{\ell,g}$ is near zero; a row of zeros marks a group with no weight or no
covariance. `summary()` lists the blocks.

## Implementation

- `kika.cov.multigroup.collapse`: `MF34_to_MG` (validation, weights, the loop over
  blocks); `LegendreSource`, `legendre_source_from_endf`, `legendre_source_from_model`
  and `resolve_legendre_source` (the coefficients, memoised per collapse);
  `cm_to_lab_legendre`, with the two-body $\gamma$ of `kika._legendre_frames.twoBodyGamma`;
  `compute_base_cell_means`; `build_union_grid`,
  `expand_matrix_to_union_grid` and `compute_mg_means`; `collapse_relative_covariance`
  and `collapse_absolute_covariance`; `compute_overlap_weights`, which builds the
  overlap integrals of $\phi$ for a base and a group grid; `enforce_matrix_quality`
  (symmetrisation and the projection of a self block).
- `kika.processing.multigroup`: `WeightingFunction` (the fluxes and their
  antiderivatives), `compute_rebin_operator`, `collapse_covariance`,
  `relative_to_absolute`, `absolute_to_relative`; format-free, also used by
  `kika.sampling.mf31_sampling`, `kika.sampling.mf33_sampling` and
  `kika.cov.CrossSectionCovariance.project_to_grid`.
- `kika.cov.decomposition`: the clip and Higham projections and the automatic rule.
- `kika.cov.multigroup.mg_legendre_covariance`: the result class and its assembly of the
  full matrix from the blocks.
- `kika.endf.classes.mf34`: the MF 34 reader that sums the LIST records of a
  sub-subsection, dividing an LB=0 part of a mixed one by the MF 4 cell means
  (`to_ang_covmat`).

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: Ch. 33 (LB flags), Ch. 34 and §34.1 (MF 34).
- TODO(ref): the NJOY ERRORR module, for the $\sigma\phi$-weighted collapse of relative
  covariances.
