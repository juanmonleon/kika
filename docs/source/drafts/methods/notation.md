# Notation

The symbols the methods pages share, defined once. A page uses them exactly as defined
here. A symbol a page needs and no other page uses is defined on that page, at first use,
and may then mean something else on another page (the last section lists the letters that
do).

## Units

| Quantity | Unit | Note |
|---|---|---|
| Energy (incident, outgoing, Q-values, binding energies) | eV | ENDF's unit; the ACE format's is MeV, converted on reading and writing |
| Cross-section | b | $1\ \text{b} = 10^{-24}\ \text{cm}^2$ |
| Temperature | K | $k_B T$ in eV where a formula needs an energy, $k_B = 8.617333262\cdot10^{-5}$ eV/K |
| Mass | neutron masses | $A$, $A_b$ below; an absolute mass is a rest energy in eV |
| Time | s | decay constants in 1/s |
| Heating (KERMA) | eV·b | a heating number per collision in eV, MeV in ACE |

## Energies and frames

| Symbol | Meaning |
|---|---|
| $E$ | incident energy, in the laboratory (eV) |
| $E'$ | energy of an emitted particle (eV); laboratory unless the page says centre of mass |
| $E'_{lab}$, $E'_{cm}$ | the same, stated in the laboratory or in the centre of mass |
| $\mu = \cos\theta$ | cosine of the emission angle, in the frame the page states |
| $\mu_L$, $\mu_C$ | the cosine in the laboratory and in the centre of mass |
| CM, LAB | centre of mass and laboratory frames |
| $\gamma$ | ratio of the centre of mass's speed to the emitted particle's speed in the centre of mass (two-body kinematics). As a subscript, $\gamma$ names the photon (or capture, in $\Gamma_\gamma$, $\sigma_\gamma$) |
| $T$ | temperature (K) |

## Targets, reactions and products

| Symbol | Meaning |
|---|---|
| $A$ | the target's mass in neutron masses (ENDF's AWR) |
| $A_b$, $A_p$ | mass of an emitted particle, in neutron masses (ENDF's AWP) |
| $m_n$ | neutron mass; $m_n c^2$ in eV where it multiplies an energy |
| $Z$ | atomic number; $ZA = 1000Z + A$ (A the mass number) names a nuclide |
| MF, MT | ENDF file and reaction numbers, written `MF 3`, `MT 102` in prose |
| $Q$, $Q_r$ | Q-value of a reaction $r$ (eV): the kinetic energy it releases, negative for an endothermic reaction |
| $r$ | index of a reaction |
| $y(E)$ | multiplicity (yield) of a product: the mean number emitted per reaction |
| $\bar\nu(E)$ | mean number of neutrons per fission; $\bar\nu_p$, $\bar\nu_d$, $\bar\nu_t$ prompt, delayed and total |
| $\chi(E \to E')$ | a fission or emission spectrum, normalised to 1 over $E'$ |

## Cross-sections and distributions

| Symbol | Meaning |
|---|---|
| $\sigma(E)$ | a cross-section; $\sigma_r$ reaction $r$'s, $\sigma_t$ the total |
| $\sigma_x$, $x \in \{t, n, \gamma, f\}$ | total, elastic, capture and fission (resonance pages) |
| $f(\mu, E)$ | an angular distribution, normalised to 1 over $\mu \in [-1, 1]$ |
| $P_\ell(\mu)$ | Legendre polynomial of order $\ell$ |
| $a_\ell(E)$ | Legendre coefficient of order $\ell$: $f(\mu, E) = \sum_\ell \frac{2\ell + 1}{2}\, a_\ell(E)\, P_\ell(\mu)$, $a_0 = 1$ |
| $f(E \to E', \mu)$ | a joint energy-angle distribution, normalised to 1 |
| $\langle x \rangle$, $\bar x$ | the mean of $x$ over the distribution the page states |

## Grids, groups and tolerances

| Symbol | Meaning |
|---|---|
| $E_i$, $E_j$ | nodes of an energy grid, in increasing order |
| $E_g$, $[E_g, E_{g+1}]$ | a group boundary and group $g$ |
| $\phi(E)$ | a weighting flux or spectrum |
| $\varepsilon$ | a relative tolerance: the largest relative error a tabulation, a reconstruction or a thinning is allowed |

## Covariances and statistics

| Symbol | Meaning |
|---|---|
| $C_{ij}$ | covariance of quantities $i$ and $j$, relative or absolute as the page states |
| $\rho_{ij}$ | correlation, $C_{ij}/\sqrt{C_{ii}C_{jj}}$ |
| $\Sigma$ | a covariance matrix as a whole |
| $\delta$, $\Delta$ | a relative perturbation (a factor is $1 + \delta$) and a difference, as each page states |
| $n\sigma$ | in a comparison of two statistical estimates, $n$ standard deviations of their difference; everywhere else $\sigma$ is a cross-section |

## Letters that are page-local

These letters are defined on the page that uses them and mean different things on
different pages; read each page's own definition:

- $\alpha$, $\beta$: the thermal-scattering momentum and energy transfers, the kinematic
  terms of the heating, $\alpha = 1/A$ of the elastic frame transform, ratios of panel
  widths (tabulated functions), and a velocity over $c$;
- $E_c$: a window's centre (group averaging), a channel's energy (resonance
  reconstruction), a product component's mean energy (heating), the centre of mass's
  kinetic energy per emitted particle (photonuclear tables);
- $C$: a constant of a law, or a covariance with indices;
- $S$: a scattering law, a sensitivity, a statistic;
- $\kappa$, $\eta$, $\lambda$, $\tau$, $\xi$: as each page defines them.
