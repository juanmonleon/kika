# Differential cross-sections and energy resolution

```{admonition} At a glance
:class: note
**Computes:** $d\sigma/d\Omega(\mu, E)$ from an angular distribution (MF 4) and a
cross-section (MF 3), in the laboratory or centre-of-mass frame, and the three readings of
$\sigma(E)$ a comparison with a measurement needs: nominal, averaged over an energy bin,
and folded with a time-of-flight resolution.

**Applies to:** elastic scattering (two-body kinematics), Legendre or tabulated MF 4.

**Entry points:** {py:func}`kika.endf.dcs.differential_xs_vs_angle`,
{py:func}`kika.endf.dcs.differential_xs_vs_energy`, {py:func}`kika.endf.dcs.resolve_sigma`,
{py:class}`kika.endf.dcs.TofResolution`, {py:func}`kika.utils.energy_folding.tof_energy_resolution`.

**Status:** stable.
```

## Background

An evaluation states the angular distribution of the emitted neutron as a normalised
$f(\mu, E)$, $\int_{-1}^1 f\,d\mu = 1$, either as Legendre coefficients,

$$
f(\mu, E) = \sum_\ell \frac{2\ell+1}{2}\, a_\ell(E)\, P_\ell(\mu), \qquad a_0 = 1,
$$ (eq-differential-cross-sections-legendre)

or as tables (ENDF-6 Ch. 4). A differential measurement counts neutrons scattered into a
solid angle at a nominal incident energy, with an energy resolution set by the experiment;
in a time-of-flight experiment the resolution grows with energy.

## What it computes

$$
\frac{d\sigma}{d\Omega}(\mu, E) = \frac{\sigma(E)}{2\pi}\, f(\mu, E)
\quad [\text{b/sr}],
\qquad
\frac{d\sigma}{d\mu} = \sigma(E)\, f(\mu, E) \quad [\text{b}],
$$ (eq-differential-cross-sections-dcs)

against $\mu$ at a fixed $E$, or against $E$ at a fixed $\mu$, with $\sigma(E)$ in one of
three readings.

## Method

**1. $f(\mu)$ at an energy.** The Legendre coefficients are interpolated in incident
energy under the MF 4 interpolation law of the section (its own (NBT, INT) pairs, not an
assumed lin-lin), then summed with the Legendre polynomials (Bonnet's recurrence).

**2. Frame.** For elastic scattering, with $\alpha = 1/A$ ($A$ the atomic weight ratio,
or the mass number when the AWR is unknown),

$$
\mu_L = \frac{\mu_C + \alpha}{\sqrt{1 + 2\alpha\mu_C + \alpha^2}}, \qquad
\frac{d\mu_L}{d\mu_C} = \frac{1 + \alpha\mu_C}{(1 + 2\alpha\mu_C + \alpha^2)^{3/2}},
$$ (eq-differential-cross-sections-frame)

and a distribution changes frame as a density, $f_L(\mu_L) = f_C(\mu_C)\, d\mu_C/d\mu_L$
(ENDF-6 App. D). $\alpha = 0$ means no transform.

**3. The readings of $\sigma(E)$.**

- *Nominal*: the table at $E$, interpolated under the table's own (NBT, INT) law
  (`xs_interpolation`; one lin-lin region when it is not given). Outside the table the
  end values are held.
- *Bin-averaged* over $[E_\text{lo}, E_\text{hi}]$ with weight $w(E) = 1/E$ (lethargy) or 1:
  $\langle\sigma\rangle = \int \sigma w\,dE / \int w\,dE$, integrated exactly under the
  same law by the group average of {doc}`group-averaging` ({py:func}`kika.algebra.group_averages`),
  over the part of the bin the table covers.
- *Folded* with a Gaussian resolution of width $\sigma_E(E)$:
  $\tilde\sigma(E_0) = \int \sigma(E)\, \mathcal{N}(E; E_0, \sigma_E^2)\, dE$, computed
  exactly on the lin-lin interpolant of the table ({py:func}`kika.algebra.fold_tabulated`).

**4. Time-of-flight resolution.** With flight path $L$ and timing spread $\delta t$, the
time of flight is $t = L/v$ with $v = c\sqrt{2E/m_n c^2}$ (non-relativistic), and
$\Delta E/E = 2\,\Delta t/t$, so

$$
\sigma_E(E) = \frac{2E\,\delta t}{t} = \frac{2E\,\delta t\, v}{L} \;\propto\; E^{3/2},
$$ (eq-differential-cross-sections-tof)

divided by $2\sqrt{2\ln 2} = 2.3548$ when $\delta t$ is a full width at half maximum (the
default). An experiment that declares its resolution instead (a constant FWHM in eV, or a
fraction of $E$) replaces the geometry.

**5. What a resolution-limited measurement sees.** The detector at $E_0$ counts
$\frac{1}{2\pi}\int R(E;E_0)\,\sigma(E)\,f(\mu,E)\,dE$; since $f$ is linear in the
$a_\ell$, that is a Legendre series with the cross-section-weighted coefficients
$a_\ell^\text{eff} = \langle\sigma a_\ell\rangle / \langle\sigma\rangle$
(`coefficients_sigma_weighted_folded`). Averaging the $a_\ell$ without the weight
(`resolve_coefficients(mode="folded")`) is a different quantity, equal to it only where
$\sigma$ is flat across the window.

## Conventions and choices

- **Energies in eV, $d\sigma/d\Omega$ in b/sr**; a name ending in `_mev` is in MeV.
- **$\delta t$ is a FWHM by default.** Pulse widths and timing are usually quoted so; pass
  `delta_t_is_fwhm=False` for a standard deviation. The two readings differ by a factor
  2.35 in the width of the kernel.
- **No floor on $\sigma_E$.** The fold integrates on the table's own points, so a narrow
  kernel tends to the interpolated value, as it should.
- **A reading whose inputs are missing falls back to the nominal one**, so a curve never
  has a hole.
- **$\sigma$ is read under the law its table states, lin-lin by default.** A PENDF and an
  ACE file are lin-lin, and so is an evaluation's MF 3 above the resonance region; reading
  such a table log-log changes its values between points. An MF 3 with other laws is read
  under them by passing its (NBT, INT) pairs. The fold always uses the lin-lin interpolant,
  which is where its closed form holds.
- **One averaging implementation.** The bin averages of $\sigma$ and of the $a_\ell$
  (`coefficients_bin_averaged`, under the MF 4 law) are the exact group average of
  {doc}`group-averaging`, not a separate quadrature.

## Limits and failure modes

- The frame transform is the elastic two-body one; it is not valid for inelastic or
  multi-body emission.
- The lethargy weight diverges at $E = 0$; a bin that reaches zero is truncated at the first
  positive grid point inside it.
- Folding needs $E > 0$, $L > 0$ and $\delta t > 0$; otherwise $\sigma_E = 0$ and the fold
  returns the lin-lin value of the table at $E$, which equals the nominal reading of a
  lin-lin table and differs from it between points when the table states another law.

## Validation

`kika/endf/tests/test_dcs.py` and `kika/algebra/tests/test_fold.py`, 141 tests passed on
2026-10-11, among them: the bin width matches the TOF resolution
(`test_dcs.py::test_bin_edges_for_width_tof_matches_the_resolution`); folding is exact on a
linear coefficient and returns a constant unchanged
(`::test_coefficients_folded_is_exact_on_a_linear_coefficient`,
`::test_coefficients_folded_returns_a_constant_unchanged`); the bin-averaged coefficients
match the bin-averaged cross-section row by row
(`::test_coefficients_bin_averaged_matches_sigma_bin_averaged_row_by_row`); the four readings
are independent (`::test_the_four_readings_are_independent`); $\sigma$ is read lin-lin by
default and under a stated law when given, and the bin average is exact for a lin-lin
table under both weights (`test_dcs.py::TestSigmaUnderTheTablesLaw`).

## Diagnostics

`legendre_provenance(section)` and `max_legendre_order(section)` state the Legendre order
an MF 4 section carries and where it came from; the plotting functions return arrays.

## Implementation

- `kika/endf/dcs.py`: the angular reconstruction, the frame transform, the readings of
  $\sigma$, the coefficient readings, the differential cross-sections.
- `kika/utils/energy_folding.py`: `tof_energy_resolution`, the one definition of
  {eq}`eq-differential-cross-sections-tof`, and the folding helpers.
- `kika/algebra/fold.py`: the exact Gaussian fold of a lin-lin table.
- `kika/_legendre_frames.py`: the cosine maps.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: Ch. 4 (MF 4), App. D (two-body kinematics).
