# Computing heating (KERMA) and damage cross-sections

```{admonition} At a glance
:class: note
**Computes:** the heating number (KERMA factor times cross-section, MT 301) with the
photons carried away, the same with the photons deposited locally (MT 901), and the
damage-energy production cross-section (MT 444), in eV·b, on the energy grid of each
temperature.

**Applies to:** an incident-neutron evaluation processed for Monte Carlo (its reactions
with their sampled neutron and photon distributions), with MF 1/MT 458 for a fissionable
target; written to a PENDF as MF 3/MT 301 and MT 444, and to an OpenMC HDF5 file as
MT 301, 901 and 444.

**Entry points:** {py:func}`kika.processing.heating.heatingOf`, with
{py:class}`kika.processing.heating.HeatingResult`; through {py:func}`kika.endf.process_endf`
(the `heating` stage of a {py:class}`kika.endf.ProcessingPlan` with `heating=True`) and
{py:func}`kika.hdf5.processing.processSuite`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

A neutron collision deposits locally the kinetic energy of the charged particles and
recoil nuclei it produces; the neutrons and photons it emits carry their energy away. The
KERMA factor (kinetic energy released in matter) of a reaction is that local energy per
collision, and the heating number is its sum over reactions weighted by their
cross-sections, in eV·b. Dividing by the total cross-section gives the energy deposited
per collision. Of the recoil energy, only part goes into displacing atoms of the lattice;
the rest is lost to electronic excitation. The damage energy is the part that displaces
atoms, given by the Lindhard partition of each recoil (Robinson's fit, Norgett et al.
1975), and the damage-energy production cross-section is its sum over reactions. NJOY
HEATR computes both (MacFarlane et al. 2017), and the OpenMC libraries carry them as
MT 301, 901 and 444.

HEATR computes the KERMA by energy balance where the evaluation states the photons: the
available energy $E + Q$ less the energy of every emitted neutron and photon. Capture is
the exception: there the energy balance is replaced by the recoil from momentum
conservation. kika follows
HEATR's method, with its option `local = 0` (photons carried away) for MT 301 and MT 444.

## What it computes

Input: the processed suite (exclusive reactions with the Monte Carlo distributions of
their emitted neutrons and photons, the form {py:func}`kika.hdf5.processing.processSuite`
builds), each reaction's cross-section $\sigma_r(E)$ on the grid, the target's mass ratio
$A$ (AWR) and charge $Z$, each reaction's ground-state Q value $Q_r$ (ENDF's QM, else the
reaction's Q), the displacement threshold $E_d$, and the evaluated suite the processed one
came from, for the charged particles and recoils the processed suite does not keep.
Output, on the grid, with fission excluded:

$$
k(E) = \sum_r \sigma_r(E)\,h_r(E),\qquad
k^\text{loc}(E) = \sum_r \sigma_r(E)\,\big[h_r(E) + G_r(E)\big],\qquad
D(E) = \sum_r \sigma_r(E)\,d_r(E),
$$ (eq-heating-sums)

where $h_r$ is the energy a reaction deposits per collision with its photons carried away,
$G_r = \sum_\gamma y_\gamma \bar E_\gamma$ the energy of its photons, and $d_r$ its damage
energy per collision. The fission part of MT 301 and 901 comes from the fission energy
release of MF 1/MT 458 and is added by each output in the definition that output carries
(*Conventions and choices*).

## Method

### Energy deposited per reaction

For a reaction $r$ with emitted neutrons and photons of yields $y$ and mean laboratory
energies $\bar E$,

$$
h_r(E) = E + Q_r - \sum_n y_n(E)\,\bar E_n(E) - \sum_\gamma y_\gamma(E)\,\bar E_\gamma(E) ,
$$ (eq-heating-balance)

with three exceptions:

- **Capture (MT 102) with photons.** The compound nucleus' recoil plus the photons' kicks,
  $h_{102} = E/(A+1) + \sum_\gamma y_\gamma \langle E_\gamma^2\rangle / [2(A+1)m_nc^2]$, with
  $m_nc^2 = 939.565$ MeV (NJOY's constant). HEATR keeps $E/(A+1)$ for MF 12 photons
  (`gheat`) and leaves it out for MF 6 ones (`tabsq6`); kika adds it always
  (*Conventions and choices*). Capture without photons follows {eq}`eq-heating-balance`, $h_{102} = E + Q_{102}$.
- **MT 5.** HEATR closes no energy balance there: the deposit is the energy of the
  charged particles and recoils the evaluation states with distributions,
  $h_5 = \sum_c y_c \bar E_c$, tabulated from the evaluated suite. Without any, $h_5 = 0$.
- **Photons stated on a sum** (MF 13 on MT 4 or MT 103-107). Their energy
  $\sigma_s\sum_\gamma y_\gamma\bar E_\gamma$ is taken out of $k$ (MT 301) and left in
  $k^\text{loc}$, as HEATR's photon pass does.

A term enters the sums only where $\sigma_r > 0$. Fission reactions (MT 18-21, 38) are
excluded from $k$ and $k^\text{loc}$, and kept in $D$. Negative values are kept: HEATR
does not clip a negative KERMA, and neither does kika.

### Mean outgoing energies

The means come from the Monte Carlo tables of the processed suite, the distributions a
transport code samples, and not from each ENDF law again: one integral over a tabulated
density covers every MF 4, 5 and 6 law the tabulation produced. For a particle of mass
ratio $a$ (1 for the neutron, 0 for a photon) with $x = a/(A+1)^2$, a centre-of-mass
distribution goes to the laboratory by

$$
\bar E'_\text{lab} = \bar E'_\text{cm} + xE + 2\sqrt{xE}\,\big\langle\sqrt{E'_\text{cm}}\,\mu_\text{cm}\big\rangle ;
$$ (eq-heating-cm-to-lab)

a photon has no mass, so its centre-of-mass spectrum is its laboratory one. By form:

- **Two-body** (elastic, a discrete level): $E'_\text{cm} = [A/(A+1)]^2 E\,(1 - E_\text{th}/E)$
  with $E_\text{th} = \tfrac{A+1}{A}\max(-Q, 0)$, and $\langle\sqrt{E'}\mu\rangle =
  \sqrt{E'_\text{cm}}\,\bar\mu(E)$ from the angular tables.
- **Uncorrelated energy and angle:** the spectrum's mean, and in the centre of mass
  $\langle\sqrt{E'}\rangle\,\bar\mu$. A discrete photon has its line energy, a primary
  photon $E_\gamma + \tfrac{A}{A+1}E$.
- **N-body phase space** of $n$ products and total mass ratio $A_p$:
  $\bar E'_\text{cm} = \tfrac{3}{2}E_\text{max}/(\tfrac{3}{2}n - \tfrac{3}{2})$ with
  $E_\text{max} = \tfrac{A_p - a}{A_p}\big(\tfrac{A}{A+1}E + Q\big)$, and no
  $\mu$ term.
- **Correlated energy-angle tables:** $\langle\sqrt{E'}\,\bar\mu(E')\rangle$ over the
  outgoing spectrum. **Kalbach-Mann:** $\bar\mu(E') = r(E')\,[\coth a(E') - 1/a(E')]$.

Each table is integrated over its discrete lines and then its continuum, by four-point
Gauss-Legendre quadrature on each panel of a lin-lin density (exact for the density
times a polynomial of degree up to six), or on the constant of a histogram. For the
laboratory spectra of a reaction the evaluation states in MF 6, the mean is taken by
trapezoids over $E' f(E')$ at the tabulated points, as HEATR does; the photon energy
$G_r$ of $k^\text{loc}$ uses the exact integral in every case. The moments known at the
tables' incident energies are interpolated to the grid lin-lin (step-wise for a histogram
in incident energy), held at the end values outside. An incident energy stated twice is
a step: its first table holds below it and its second at and above it. A law whose
incident table is all zeros (a spectrum stated null at a threshold) is read from the
nearest incident table that emits, as the ends are; a mixture of laws read outside the
domain of its weights is read at the nearest end of that domain, where its
cross-section is zero.

A continuous centre-of-mass law in unit-base interpolation between incident energies
(the interpolation OpenMC samples) is integrated as that blend, since the moment of a
unit-base blend is not the blend of the moments: each incident table is mapped to the
interpolated support, the cross moment $\langle\sqrt{E'}\mu\rangle$ is integrated by
Gauss-Legendre with its order doubled until the laboratory term agrees to
$10^{-3}$ eV $+\,10^{-8}E$, and the laboratory mean is tabulated on the tables' incident
energies and refined at the fractions 0.25, 0.5 and 0.75 of each interval until linear
interpolation reproduces it to $0.1$ eV $+\,10^{-7}E$. A step in incident energy splits
this refinement: each side is refined on its own, the left side up to the step's left
limit.

### Damage energy

The damage energy of a recoil of charge $Z_r$, mass ratio $A_r$ and energy $E_R$ in a
lattice $(Z_l, A_l)$ is

$$
\begin{aligned}
d(E_R) &= \frac{E_R}{1 + F_L\big(3.4008\,\varepsilon^{1/6} + 0.40244\,\varepsilon^{3/4} + \varepsilon\big)},
\qquad \varepsilon = \frac{E_R}{E_L},\\
E_L &= 30.724\,Z_rZ_l\sqrt{Z_r^{2/3} + Z_l^{2/3}}\;\frac{A_r + A_l}{A_l}\ \text{eV},\\
F_L &= \frac{0.0793\,Z_r^{2/3}Z_l^{1/2}(A_r + A_l)^{3/2}}
{\big(Z_r^{2/3} + Z_l^{2/3}\big)^{3/4}A_r^{3/2}A_l^{1/2}} ,
\end{aligned}
$$ (eq-heating-lindhard)

for $E_R \ge E_d$, and zero below $E_d$ or for $Z_r = 0$. $E_d$ is HEATR's table by $Z$
(25 eV unless listed: 31 eV for Be and C, 27 eV for Al, 40 eV for Ca, Ti to Cu, Zr and
Nb, 60 eV for Mo and Ag, 90 eV for Ta and W, 30 eV for Au, 25 eV for Mg, Si and Pb),
or the value given. $d_r$ follows HEATR's path for each kind of reaction:

- **Elastic and discrete inelastic levels (MT 2, 51-90),** two-body: the recoil
  $E_R(\mu) = E\,\tfrac{A}{(A+1)^2}\,(1 - 2r\mu + r^2)$, $r^2 = 1 - E_\text{th}/E$,
  averaged over the centre-of-mass angular distribution by 64-point Gauss-Legendre in
  $\mu$, the target's $(Z, A)$ as recoil, on the tables' incident energies and a
  geometric mesh of ratio 1.1, then interpolated to the grid; negative values are set to
  zero, and the elastic damage is zero below $E = 25(A+1)^2/(4A)$ eV.
- **Capture (MT 102):** the compound nucleus' recoil $E/(A+1)$ and each photon's recoil
  $E_\gamma^2/[2(A+1)m_nc^2]$, each an independent primary through
  {eq}`eq-heating-lindhard` with recoil $(Z, A+1)$, the photons weighted by their yields,
  $\mathrm{df}(E/(A+1)) + \sum_\gamma y_\gamma\langle\mathrm{df}(E_\gamma^2/2(A+1)m_nc^2)\rangle$;
  without photons, HEATR's single photon,
  $E_R = \tfrac{E}{A+1} + \big(\tfrac{A}{A+1}E + Q\big)^2/[2(A+1)m_nc^2]$.
- **Other reactions whose charged particles and recoils the evaluation states with
  distributions** (in ENDF, MF 6): each
  particle's own damage, averaged over its laboratory spectrum (a centre-of-mass spectrum
  over an isotropic centre-of-mass angle, eight-point Gauss-Legendre).
- **The residual of such a reaction when the evaluation lists its light particles
  (p, d, t, ³He, α) and not the residual, or states the residual only as a two-body
  recoil (ENDF MF 6 LAW 4):** the residual $(Z - z, A + 1 - a)$ of the heaviest light
  particle recoils with that particle's momentum,
  $E_R = (E - 2\sqrt{E\,aE'}\,\mu + aE')/(A + 1 - a)$ for a laboratory law and
  $E_R = \alpha + \beta - 2\sqrt{\alpha\beta}\,\mu$,
  $\alpha = (A+1-a)E/(A+1)^2$, $\beta = aE'/(A+1-a)$, for a centre-of-mass law,
  averaged over the particle's own energy-angle distribution (four-point Gauss-Legendre
  on eight sub-panels of each outgoing-energy and angular panel; for a level, one
  outgoing energy, the angle on 64 sub-panels and the incident energy on a geometric
  mesh of ratio 1.02), on the tables' incident energies and interpolated lin-lin.
  This is HEATR's one-particle recoil (`hinit`, `getsix` with `irec > 0`), exact for
  one emitted particle and for a level. HEATR evaluates the recoil at the mean
  laboratory cosine of each laboratory energy (`h6cm`); kika averages the damage over
  the angle. No recoil is added when the residual is itself a light particle
  ($Z - z \le 2$).
- **A discrete charged level (MT 600-648, 650-698, 700-748, 750-798, 800-848) stated in
  MF 4 or by MF 3 alone:** MF 4 gives the particle's angle and no yield, which is one;
  the level is two bodies, the particle's centre-of-mass energy fixed by $Q$, the residual
  recoiling as above from the stated angular distribution, isotropic in the centre of
  mass without one. HEATR does the same kinematics (`disbar`) but takes the target's
  $(Z, A)$ for the recoil's partition, where its MF 6 path takes the residual's; kika takes
  the residual's in both. Read without their yield, these levels would fall to the
  absorption fallback below, a proton capped at the Coulomb barrier.
- **The same sections when they emit more than one particle (MT 22, 24, 28, ...,
  neutrons included, each with a whole yield):** the residual
  $(Z - \sum z_i, A + 1 - \sum a_i)$ takes the momentum balance of all of them,
  $E_R = |\mathbf{p}_n - \sum_i \mathbf{p}_i|^2 / 2M$, each $\mathbf{p}_i$ drawn from
  its particle's own laboratory energy-angle distribution and the azimuths
  independent. Each distribution is gathered on a $24 \times 24$ mesh in
  $(p_z, p_\perp)$ keeping every bin's weight, centroid and second moments, and the
  sums are convolved one particle at a time with sixteen midpoints in the relative
  azimuth, so that $\langle E_R \rangle$ is exact and only the curvature of the damage
  within a bin is lost (below $10^{-3}$ against a Monte Carlo of the same model). The
  evaluation states each particle's distribution and not their correlation, so this
  is a model, not HEATR's: against the residuals that ENDF/B-VIII.1 does state for
  these reactions (sequential-emission cascades), it is closer than the heaviest
  particle's recoil in all six MT 28 sections measured (O-16, Al-27, Si-28, Mn-54,
  Fe-55, Mo-95) and in MT 22 of the light targets (O-16: 1.04-1.06 of the stated
  residual's damage, HEATR's approximation 1.13-1.15), and equal within 0.5 % in MT 22
  of the medium-mass ones, where the alpha carries the momentum.
- **Sections that state their neutrons and not the residual (MT 16, 17, 91, ...):** the
  same momentum balance, from all $n$ neutrons with whole yields, the residual
  $(Z, A + 1 - n)$; for one neutron (MT 91) it is the exact recoil, averaged over the
  points of the neutron's own energy-angle law without binning. A yield that is not
  whole takes HEATR's one-particle recoil (the first neutron, mass $A$). HEATR recoils an
  MF 6 section's residual from one neutron with the target's mass at the mean laboratory
  cosine (`h6cm`), and an MF 4/MF 5 one by one isotropic laboratory emission,
  $E_R = (E - 2\sqrt{EE'}\mu + E')/A$ (`tabdam`). Measured against HEATR itself (NJOY 2016.78 rebuilt without one MT's generated
  recoil, Ni-60 of ENDF/B-VIII.1): HEATR's MT 91 is 32.4 % of MT 444 at 14 MeV and its
  MT 16 17.1 %; HEATR's rule emulated on kika's tables gives 32.4 % and 17.0 %, the exact
  recoil 31.0 % and 17.2 %, the isotropic emission 34.4 % and 17.4 %. The isotropic
  emission is therefore not HEATR's number on an MF 6 section, and the exact recoil differs
  from it by the mean cosine alone. Against the residuals ENDF/B-VIII.1 states, the exact
  MT 91 recoil of Fe-56 is 17 % below at 20 MeV: those residuals are computed for a
  neutron isotropic in the centre of mass (⟨E_R⟩ 537 keV; the neutron's stated law gives
  419 keV, the same spectrum isotropic 515 keV), not for the evaluation's own neutron.
- **Absorption with a charged particle and no neutron (MT > 102), stated by MF 3 alone:**
  a single particle of charge $z$ and mass $a$ (proton, and for MT 104-107 and the
  continua 699, 749, 799, 849 the deuteron, triton, helium-3, alpha; HEATR takes the
  four continua as protons) at the smaller of $E_a = Q + \tfrac{A}{A+1}E$ and the Coulomb barrier
  $1.029\cdot10^6\,zZ/(a^{1/3} + A^{1/3})$ eV, the residual of charge $Z - z$ and mass
  $A + 1 - a$ recoiling with
  $E_R = \big(E_t - 2\sqrt{E_t\,a\,E_a}\,\mu + a E_a\big)/(A+1)$,
  $E_t = \tfrac{A+1-a}{A+1}E$, over $\mu$ by four-point Gauss-Legendre; zero where
  $E_a < 0$.
- **Fission:** one isotropic laboratory emission from the prompt-neutron spectrum,
  $E_R = (E - 2\sqrt{EE'}\mu + E')/A$ over $\mu$ by four-point Gauss-Legendre, the
  target's $(Z, A)$ as recoil (one neutron, not $\bar\nu$; the fragments are not
  followed).

### Displacements (NRT and arc-dpa)

{py:func}`kika.endf.process_endf` also gives the displacements per reaction,
$\sum_r \sigma_r \langle N_d \rangle_r$ in b·displacements on MT 444's grid
(`EndfProcessing.displacements`; `heatingOf(..., displacements=True)`). No other
processor computes them: NJOY stops at the damage energy, FRENDY and FUDGE compute no
damage, OpenMC scores MT 444 only. With $T_d$ the Lindhard-Robinson damage energy of a
recoil (the partition above, without HEATR's threshold on the recoil energy),

$$
N_d(T_d) = \begin{cases} 0 & T_d < E_d \\ 1 & E_d \le T_d < 2E_d/0.8 \\
\dfrac{0.8\,T_d}{2E_d}\,\xi(T_d) & T_d \ge 2E_d/0.8 \end{cases},\qquad
\xi_\text{arc}(T_d) = \frac{1 - c}{(2E_d/0.8)^b}\,T_d^{\,b} + c ,
$$ (eq-heating-displacements)

with $\xi = 1$ for NRT (Norgett, Robinson and Torrens 1975) and the arc-dpa efficiency
of Nordlund et al. (2018). $N_d$ is not linear in $T_d$, so it is applied to each recoil
inside the same quadratures as the damage energy, never to MT 444 itself. NRT uses the
damage's $E_d$ (HEATR's table by default); arc-dpa uses the $E_d$, $b$ and $c$ of the
paper's table 1, the only elements it fits: Fe (40 eV, −0.568, 0.286), Cu (33, −0.68,
0.16), Ni (39, −1.01, 0.23), Pd (41, −0.88, 0.15), Pt (42, −1.12, 0.11), W (70, −0.56,
0.12). For any other element there is no arc-dpa, and the notes say so. The adaptive
quadratures split at the recoil energies of $N_d$'s step and kink and take a relative
tolerance of $10^{-4}$; the step and kink would otherwise defeat their convergence.

For Fe-56 (ENDF/B-VIII.1, 293.6 K), NRT is $0.8\,\text{MT444}/(2E_d)$ to $10^{-3}$ above
100 keV, and higher near 1 keV, where the elastic recoils of ~70 eV fall in the
$N_d = 1$ step; arc-dpa over NRT is 0.67 at thermal energies, 0.39 at 100 keV and 0.30
at 14 MeV, towards $c$ = 0.286.

### Fission

The fission part of MT 301 and 901 is not computed by energy balance. It comes from the
components of MF 1/MT 458 (ENDF-6 §1.5): the kinetic energy of the fragments $E_\text{FR}$,
the prompt and delayed photon energies $E_\text{GP}$ and $E_\text{GD}$, and the delayed
beta energy $E_\text{B}$, each a function of $E$. With $\sigma_f$ the fission
cross-section (MT 18, else the sum of the chances 19, 20, 21, 38):

$$
k_f^\text{PENDF} = \sigma_f\big(E_\text{FR} + E_\text{GP} - \textstyle\sum_\gamma y_\gamma \bar E_\gamma\big),
$$ (eq-heating-fission-pendf)

with the MF 1/MT 458 components as the evaluation states them and the sum over the prompt
photons stated for fission (on MT 18, or on the first chance that states them), and

$$
k_f^\text{HDF5} = \sigma_f\big(E_\text{FR} + E_\text{B}\big),\qquad
k_f^\text{loc,HDF5} = \sigma_f\big(E_\text{FR} + E_\text{GP} + E_\text{GD} + E_\text{B}\big),
$$ (eq-heating-fission-hdf5)

with the components after OpenMC's processing of MF 1/MT 458 (a component stated as a
constant receives the Sher-Beck energy dependence, ENDF-102 §1.5). MT 301 of the PENDF is
$k + k_f^\text{PENDF}$; MT 301 and 901 of the HDF5 file are $k + k_f^\text{HDF5}$ and
$k^\text{loc} + k_f^\text{loc,HDF5}$.

### Grids and outputs

- **PENDF** ({py:func}`kika.endf.process_endf`, `heating=True`). The distributions come
  from {py:func}`kika.hdf5.processing.processSuite`, reusing the run's broadening, and
  {py:func}`kika.processing.heating.heatingOf` runs on the MT 1 grid of every temperature,
  0 K included, as HEATR reads its PENDF: at a broadened temperature the union grid of the
  material, at 0 K the reconstruction's MT 1 grid with every reaction read on it lin-lin.
  Every material of {py:meth}`kika.endf.EndfProcessing.write_pendf` then carries MF 3/MT 301
  and MT 444, inserted in MT order with QM = QI = 0, as HEATR writes them; the MF 1
  directory is rebuilt and both sections are read back and compared with the tables
  (relative $10^{-6}$) before the file is replaced.
- **OpenMC HDF5** ({py:func}`kika.hdf5.processing.processSuite`). On each temperature's
  grid, MT 301, 901 and 444 as productions (`heating`, `heating-local`,
  `damage-energy`), redundant. `omit` can leave out `kerma` or `damage`.

## Conventions and choices

- **The two outputs carry different definitions of MT 301, on purpose.** The PENDF's
  MT 301 is HEATR's with `local = 0` ({eq}`eq-heating-fission-pendf`): fission deposits
  its fragments and prompt photons, less the prompt photons the evaluation emits. The
  HDF5 file's MT 301 is OpenMC's ({eq}`eq-heating-fission-hdf5`): OpenMC's own
  processing takes HEATR's fission KERMA out and puts back
  $(E_\text{FR} + E_\text{B})\,\sigma_f$ from MF 1/MT 458. Each file holds what the code
  that reads it expects: a PENDF is read as NJOY's would be, and an OpenMC file is
  compared with OpenMC's own libraries. Both outputs share $k$, $k^\text{loc}$ and $D$;
  for a target that does not fission they are the same.
- **MT 901 is MT 301 plus the photons' energy,** $h_r + G_r$ with the exact photon means,
  which is what the official OpenMC files carry. It is not HEATR's `local = 1` balance,
  which differs only in capture: there `local = 1` deposits $E + Q$, and kika deposits
  the recoil plus the energy of the photons as the evaluation states them.
- **Capture carries the compound nucleus' recoil, in the KERMA and in the damage.** The
  official OpenMC files lack $\tfrac{E}{A+1}\sigma_\gamma$ where the photons are MF 6 (HEATR's
  `tabsq6`): Ni-58's MT 301 is 2.3 times lower at the 59 keV capture valley. kika adds it
  always, whatever file states the photons, because the compound nucleus keeps the
  neutron's momentum. The damage takes it as one more independent primary next to HEATR's
  photon kicks, $\mathrm{df}(E/(A+1)) + \sum_\gamma y_\gamma\langle\mathrm{df}(E_\gamma^2/2Mc^2)\rangle$:
  at thermal energies $E/(A+1)$ is far below $E_d$ and the damage is HEATR's to the bit;
  at 1 MeV it is the compound that displaces (Fe-56: 13.2 keV per capture against the
  kicks' 0.38 keV). MT 444 then moves where capture dominates the damage: Ni-58 +174 %
  at 59 keV, Fe-56 +119 % at 73 keV, U-238 +26 % at 10.9 keV, nothing at thermal
  energies. Folding all the kicks into one recoil of the mean energy is not done: the
  mean kick of U-238 (24 eV) is below $E_d$ = 25 eV while its hard primaries are not, so
  the thermal capture damage of U-238 would vanish.
- **Mean energies from the sampled tables.** HEATR reads each ENDF law and
  interpolates its means on its own mesh (a ratio of 1.1 between incident energies in
  its two-body routine); kika takes the moments of the distributions OpenMC samples and
  interpolates them lin-lin between the tables' incident energies, as the transport code
  interpolates the tables themselves. The deposited energy is then the one consistent
  with what the transport code emits.
- **HEATR's approximations are kept and reported.** One isotropic laboratory emission
  with the target's $(Z, A)$ for the damage of fission; each photon's
  recoil as an independent primary in capture (with the compound's as one more); the single
  charged particle at the Coulomb barrier for absorption stated by MF 3 alone; trapezoids for the mean of a laboratory MF 6 spectrum, lower
  than the exact mean on wide panels. These are equivalences with HEATR; none is claimed
  to be better physics.
- **The damage keeps HEATR's fission recoil,** one emission from the prompt spectrum,
  while the KERMA's fission part comes from MF 1/MT 458. The MT 444 of both outputs is
  HEATR's `local = 0` damage, the one OpenMC's files carry.

## Limits and failure modes

In {py:func}`kika.endf.process_endf` the stage never raises. It is `skipped` when not
requested, or when the broadening was rejected (the heating needs every requested
temperature); it is `rejected` with `heating-unsupported` for a fissionable target without
MF 1/MT 458 (HEATR's fission heating needs it), and with `heating-failed` for any other
error, keeping every earlier result.

In {py:func}`kika.processing.heating.heatingOf`, a product whose form has no mean energy
(`ValueError`), and a second moment the method does not define (of a two-body neutron or
of a centre-of-mass spectrum, `NotImplementedError`), stop the calculation; a charged
particle or recoil of the evaluation that cannot be tabulated, or has no mean energy, is
left out with a note.

Known limits:

- The processed suite of the HDF5 file keeps only neutrons and photons, as OpenMC's files
  do (an ACE table also keeps the light charged particles). For the heating, the charged
  particles and recoils are tabulated again from the evaluation where it states them
  (MT 5 and the damage of MF 6 reactions); elsewhere HEATR's fallbacks apply.
- In the HDF5 file, a fissionable target without MF 1/MT 458 gets no fission term in
  MT 301 and 901.
- The kinematic limits of HEATR are not computed, and a KERMA outside them is not
  flagged.
- On Au-197 of ENDF/B-VIII.1, MT 301 lies 3-4 % below the official OpenMC library
  between 1 eV and 100 keV, with larger ratios above 100 keV where the KERMA is a small
  difference (measured 2026-10-13; `TODO(check)`: the NJOY version that produced the
  official file, which neither the file nor its archive records; it needs the library's
  release notes). The cause is not identified.

## Validation

The tests below passed on 2026-10-14.

- **Closed forms.** HEATR's displacement thresholds
  (`kika/processing/tests/test_heating.py::test_displacement_energy_is_heatrs_table`);
  the Lindhard partition is zero below $E_d$ and without charge, and equals Robinson's
  formula computed by hand at 100 keV for Fe in Fe
  (`::test_lindhard_has_a_hard_step_and_no_damage_without_charge`); the expectation over
  a lin-lin table is exact for polynomials (`::test_expectations_over_a_table_are_exact_for_polynomials`)
  and counts a discrete line with its weight (`::test_a_line_counts_with_its_weight`);
  isotropic elastic scattering deposits $2AE/(A+1)^2$ (`::test_isotropic_elastic_deposits_two_a_over_a_plus_one_squared`);
  a level takes its excitation out of the neutron, $\bar E' = E(1 + A^2 r^2)/(A+1)^2$
  (`::test_a_level_takes_the_excitation_out_of_the_neutron`); photon lines ignore the
  frame and primaries take $\tfrac{A}{A+1}E$
  (`::test_photon_lines_ignore_the_frame_and_primaries_take_the_neutron_energy`).
- **Reading the laws** (passed on 2026-10-20). A unit-base centre-of-mass mean comes from
  the joint law and not from blended laboratory moments
  (`kika/processing/tests/test_heating.py::test_unit_base_cm_mean_comes_from_the_joint_law_not_blended_lab_moments`);
  a null incident table is read from the nearest one that emits
  (`::test_a_null_table_is_read_from_the_nearest_table_that_emits`); a mixture below its
  weights' domain is read at its first energy
  (`::test_a_mixture_below_its_weights_domain_is_read_at_its_first_energy`). The
  evaluations with incident steps in a unit-base law are processed whole by
  `kika/processing/tests/test_t3_tapes.py::test_the_tape_t3_refused_is_processed`, which
  needs the library tapes.
- **The stage and the PENDF** (the committed Fe-56 micro-tape at 293.6 K). MT 301 and
  MT 444 at 0 K and at the temperature, on that temperature's MT 1 grid (relative
  $10^{-11}$), finite, and MT 301 positive above 1 eV
  (`kika/endf/tests/test_process_endf.py::test_the_heating_stage_gives_mt301_and_mt444_on_each_mt1_grid`);
  every material of the PENDF carries both sections, and the MF 1 directory lists them
  (`::test_the_pendf_carries_the_heating_in_every_material`).

No test in this repository compares the KERMA or the damage with an independent
calculation on a whole evaluation; the comparison with HEATR below is external.

### Comparison with NJOY HEATR

This comparison is not a test of this repository: NJOY is neither a dependency of kika
nor the truth. U-235 of ENDF/B-VIII.1 at 294 K: kika's MT 301 and MT 444
({py:func}`kika.endf.process_endf` with `heating=True`, the reconstruction and the
thinning at their default $10^{-3}$) against NJOY 2016.78 (Linux build) RECONR
(`err` = $10^{-3}$), BROADR (`errthn` = $10^{-3}$) and HEATR with `local` = 0, the same
definition on both sides. kika's values are read lin-lin at NJOY's 75 895 energies, and
the metric is $(h_\text{kika} - h_\text{NJOY})/\lvert h_\text{NJOY}\rvert$ where NJOY's
value is not zero (measured 2026-10-20). For MT 301 its median magnitude is
$1.7\cdot10^{-4}$ and its largest 0.30 % (at 3.9 MeV); below 0.4999 eV it is at most
$7.3\cdot10^{-5}$, from 0.5 eV to 25 keV at most 0.13 %, from 25 keV to 1 MeV at most
0.031 %. For MT 444 it is at most 0.058 % below 0.4999 eV, 0.38 % up to 25 keV and 1.0 %
up to 1 MeV, and above 1 MeV kika lies below NJOY by a median of 4.5 % and at most 6.3 %
(at 14.5 MeV). Below 1 MeV the differences are of the size of the two codes' tolerances
on the cross-sections themselves.

MT 444 of ENDF/B-VIII.1 against OpenMC's official library (NJOY HEATR) with the
neutron-only sections' exact recoil, measured 2026-10-18 at 0.5-20 MeV: Ni-58 0.948-1.001
(0.956 at 14 MeV), Ni-60 0.949-1.004 (0.958), Nb-93 0.992-1.046 (1.046). The nickels fall
by HEATR's mean cosine in MT 91 and MT 16 and in the charged sections (on Ni-60, HEATR's
own MT 91 and MT 16 are reproduced to $10^{-3}$ of MT 444 when its rule is emulated);
Nb-93, an MF 4/MF 5 evaluation, rises because HEATR gives its (n,2n) one isotropic
emission and kika the momenta of both neutrons. Leaving out the neutron-only sections'
exact recoil (the residual recoiling from the charged particles only), the same
comparison gives Ni-58 0.980-1.001, Ni-60 0.988-1.004, Cu-63 0.986-1.014, Cr-50
0.991-1.014, Ti-46 0.991-1.004 and Fe-56 0.977-1.025; with no recoil of the residual at
all, Ni-58 falls to 0.80 at 5 MeV and Ni-60 to 0.69 at 20 MeV. On Ni-58, HEATR's
mean-cosine recoil emulated on kika's tables moves MT 444 by up to +4 % at 20 MeV; the
remaining difference is of the size of HEATR's approximations. The independent recoil
itself moves MT 444 by at most +0.4 % against the heaviest particle's (Ni-58 MT 28 at
14-17 MeV); a yield of 2, as in (n,2p), is one residual, as in HEATR (`sixbar`, recoil
yield 1).

With the charged levels stated in MF 4 read as two bodies, with the kinematics of HEATR's
`disbar`, O-16 is at 0.949 of HEATR at 5 MeV and N-14 at 0.910 (0.962 and 0.926 when
those levels fall to the absorption fallback instead). HEATR's partition with the
target's $(Z, A)$ for the residual raises the alpha levels' recoil damage by 18 % (O-16
MT 800 at 14 MeV), which is the difference. In an
evaluation that leaves MT 16 and 17 to MF 4/MF 5 (158 of ENDF/B-VIII.1, 6 of JEFF-4.0,
22 of JENDL-5) the neutron yield is the MT's two and three, as in OpenMC's files; a
yield of 1 would emit one neutron per (n,2n) in transport and keep one neutron's energy
in the KERMA. HEATR's own MT 301 of Nb-93 is negative at 12 MeV and is no reference
there.

MT 301 (photons transported) is MT 901 less the energy of the photons the file itself
carries, in kika's files to the rounding of the tables. In OpenMC's official Fe-56 of
ENDF/B-VIII.1 it is not: its MT 901 less MT 301 falls short of the photon energy its own
photon products carry by 3.6 keV b at 8 MeV, 31 keV b at 14.1 MeV and 12.6 keV b at
20 MeV (measured 2026-10-18), because HEATR computes the photons' energy itself (by
trapezoids over MF 6) and ACER writes the photons from the evaluation. kika's MT 301 is
therefore 1-2 % below the official one above 8 MeV while the photons and MT 901 agree to
$10^{-3}$, and energy is conserved in transport with kika's file and not with the
official one. In N-14 the official MT 301 subtracts the MF 13 photons of MT 28 and MT 32,
145 keV b at 14 MeV and 357 keV b at 20 MeV, which no file in OpenMC's format can carry
(the evaluation gives those MTs no cross section in MF 3), so kika's MT 301 is 4-8 % above
it there (measured 2026-10-18).

### Integral comparison

The same OpenMC model under the official library and under one in which kika's files
replace some nuclides, with the heating (MT 301), local heating (MT 901) and damage
energy (MT 444) tallied per nuclide and the flux tallied per material on 402 bins. Each
file's responses folded with its own run's flux predict the ratio of the two tallies;
the tallied ratio is within 2 standard deviations of the predicted one for every
nuclide and score (measured 2026-10-18, 4·10⁶ to 6·10⁶ histories per run): the transport code reads
kika's responses as they are written, and the integral difference is the tables'.
In a HEU sphere inside 10 cm of iron (fission spectrum), kika/official is 0.998-1.005 for
the heating of U-234, U-235, U-238 and Fe-56, and 0.980 (U-234), 0.990 (U-235), 1.010
(U-238) and 1.002 (Fe-56) for the damage energy. In 20 cm of Fe-56 around a 14.1 MeV point
source, the local heating is 0.9999 ± 0.0004, the damage energy 1.0010 ± 0.0003 and the
heating 0.984, the official file's photon deficit above.

## Diagnostics

- **Stage report** of {py:func}`kika.endf.process_endf` (`stage("heating")`): `status`,
  `category`, a one-line `summary` (the temperatures, and for a fissionable target the
  fission definition), and `details`: `temperatures` (0 K included), `mts` (301, 444),
  `local` (0), `n_points` per temperature, `notes`, and `error` when rejected. The write
  stage of the PENDF lists the sections under `heating`.
- **`EndfProcessing.heating`**: per temperature (0.0 for 0 K), `{301: (E, h), 444: (E, d)}`
  in eV·b.
- **Notes** (`HeatingResult.notes`, the stage's `notes`, and warnings of the HDF5
  conversion report): the displacement threshold used, each of HEATR's approximations
  kept, an MT 5 without charged particles (deposit left at zero), the photons stated on
  a sum taken out of MT 301, and every product left out of the heating with its reason.
  A product left out lowers that reaction's deposit or damage, and is the note to read
  first.
- **`HeatingResult.perReaction`**: per MT, $h_r$, $h_r + G_r$ and $d_r$ in eV per
  collision, to see which reaction makes a difference.

## Implementation

- `kika.processing.heating`: the energy balance, the capture recoil, the mean energies
  of every Monte Carlo form, the damage paths, the Lindhard partition
  ({py:func}`kika.processing.heating.lindhard`) and HEATR's thresholds
  ({py:func}`kika.processing.heating.displacementEnergy`), in
  {py:func}`kika.processing.heating.heatingOf`.
- `kika.hdf5.processing`: the processed suite whose distributions the heating reads, and
  the HDF5 MT 301, 901 and 444 with OpenMC's fission terms.
- `kika.hdf5.fission_energy`: the evaluation of the MF 1/MT 458 components, and OpenMC's
  Sher-Beck step for the HDF5 file.
- `kika.endf.process`: the `heating` stage, the PENDF's fission term
  {eq}`eq-heating-fission-pendf`, and the insertion of MF 3/MT 301 and MT 444 in each
  PENDF material.

## References

- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: HEATR.
- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: §1.5 (MF 1/MT 458, fission energy release).
- M. J. Norgett, M. T. Robinson and I. M. Torrens, A proposed method of calculating
  displacement dose rates, *Nucl. Eng. Des.* 33 (1975) 50-54.
- K. Nordlund, S. J. Zinkle, A. E. Sand et al., Improving atomic displacement and
  replacement calculations with physically realistic damage models, *Nat. Commun.* 9
  (2018) 1084, doi:10.1038/s41467-018-03415-5: arc-dpa, table 1.
- J. Lindhard, V. Nielsen, M. Scharff and P. V. Thomsen, "Integral equations governing
  radiation effects", *Matematisk-fysiske Meddelelser, Det Kongelige Danske
  Videnskabernes Selskab* 33(10) (1963): the partition of the recoil energy.
- R. Sher and C. Beck, EPRI NP-1771, Electric Power Research Institute (1981): the energy
  dependence of MF 1/MT 458.
