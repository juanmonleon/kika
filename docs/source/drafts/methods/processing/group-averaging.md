# Averaging a cross-section over energy groups

```{admonition} At a glance
:class: note
**Computes:** flux-weighted averages of a pointwise cross-section over energy groups, or
over a window of fixed lethargy width sliding along the energy axis.

**Applies to:** any lin-lin pointwise table: a PENDF, a reconstruction, an ACE
cross-section.

**Entry points:** {py:func}`kika.processing.group_averaging.resonance_group_average`,
{py:func}`kika.processing.group_averaging.resonance_window_average`; the integrals:
{py:func}`kika.algebra.group_averages`, {py:func}`kika.algebra.log_window_averages`.

**Status:** stable.
```

## Background

In the resonance region two evaluations, or an evaluation and its reconstruction, can
differ by large factors point by point while their reaction rates barely differ, because
narrow peaks sit at slightly different energies. Averaging over energy groups, with a
weighting spectrum $\phi(E)$, compares what a transport calculation sees. The usual
spectrum in the slowing-down range is $\phi(E) = 1/E$ (flat in lethargy $u = \ln E$).

## What it computes

For a pointwise cross-section $\sigma(E)$ and groups with boundaries
$E_0 < E_1 < \dots < E_G$:

$$
\bar\sigma_g = \frac{\int_{E_{g-1}}^{E_g} \sigma(E)\,\phi(E)\,dE}
                    {\int_{E_{g-1}}^{E_g} \phi(E)\,dE},
\qquad \phi(E) = \frac1E \;\text{(lethargy)} \;\text{or}\; 1 \;\text{(constant)}.
$$ (eq-group-averaging-group)

The window average at a centre $E_c$ is the same ratio over
$[E_c e^{-w/2},\, E_c e^{w/2}]$ for a lethargy width $w$.

## Method

**1. Clip.** Each group is averaged over its overlap with the table: the group edges are
clipped to $[E_\text{min}, E_\text{max}]$ of the pointwise grid. A group that lies wholly
outside the table, or whose weight integral is zero, gets `NaN`.

**2. Integrate exactly.** The table is lin-lin, so on each panel
$\sigma(E) = a + bE$ and both integrals of {eq}`eq-group-averaging-group` are closed
form. For $\phi = 1/E$,

$$
\int_{E_1}^{E_2} \frac{a + bE}{E}\,dE = a \ln\frac{E_2}{E_1} + b\,(E_2 - E_1),
\qquad \int_{E_1}^{E_2} \frac{dE}{E} = \ln\frac{E_2}{E_1},
$$ (eq-group-averaging-panel)

and for $\phi = 1$ they are the trapezoid and the panel width. Panels split by a group
edge are integrated over their two parts. A repeated energy (a step) is a panel of zero
width and contributes nothing.

**3. Windows.** The sliding window is clipped to the table in the same way, so near either
end it is one-sided. At the geometric centre of a group of an equal-lethargy grid with
$\ln(E_g/E_{g-1}) = w$, the window average equals that group's average.

## Conventions and choices

- **Exact integrals, not quadrature.** A trapezoid in lethargy is exact only for a constant
  cross-section; on a coarse table it can be off by several per cent.
- **Lethargy weighting by default**, the physical spectrum of the slowing-down range;
  constant weighting is the alternative.
- **Windows for display, groups for numbers.** A group average depends on where the edges
  fall (a resonance on an edge is split between two groups); a window average is a
  continuous curve with no edges to place.
- **The integrals are those of {py:mod}`kika.algebra`**, shared with every other group average in
  kika (see {doc}`../../architecture/algebra-layer`).

## Limits and failure modes

Refused (`ValueError`): arrays of different shape or not one-dimensional; fewer than two
points; energies that decrease; group boundaries that do not strictly increase; an unknown
weighting; lethargy weighting with a non-positive energy; a non-positive window width.

The table is taken as lin-lin. A table under another law should be linearised first
({py:func}`kika.algebra.to_linlin`).

## Validation

All passed on 2026-10-11 (`kika/processing/tests/test_group_averaging.py`, 17 tests): a
constant cross-section averages to itself under both weights
(`::test_constant_xs_returns_same_value_lethargy`,
`::test_constant_xs_returns_same_value_constant`); $\sigma(E) = E$ matches the analytic
average under both weights (`::test_sigma_equals_E_with_lethargy_weight_matches_analytic`,
`::test_sigma_equals_E_with_constant_weight_matches_analytic`); a group outside the table
is `NaN` (`::test_group_outside_pointwise_range_is_nan`); the window average at a group's
centre is the group average (`::test_window_average_at_group_centres_is_the_group_average`)
and stays inside the bounds (`::test_window_average_stays_inside_the_bounds`).

## Diagnostics

None: the functions return the averages. `NaN` marks a group the table does not reach.

## Implementation

- `kika/processing/group_averaging.py`: the two entry points, which validate their input and
  hand the table to the algebra.
- `kika/algebra/integrate.py`: `group_averages`, `log_window_averages`, the closed-form panel
  integrals with weight 1 or $1/x$.
- `kika/endf/dcs.py`: the bin averages of {doc}`differential-cross-sections` call the same
  `group_averages`, under the interpolation law of the table they average.
- `kika/energy_grids/`: standard group structures and `equal_lethargy_grid`.

## References

None: the integrals are elementary.
