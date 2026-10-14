# Reconstructing cross-sections from resolved resonance parameters

```{admonition} At a glance
:class: note
**Computes:** pointwise, linearly interpolable cross-sections at 0 K over the whole
evaluated energy domain of a neutron evaluation, from its resolved resonance parameters
(MF 2), its backgrounds (MF 3) and its sum graph, checked against the resonance physics
to a stated tolerance; optionally the PENDF (pointwise ENDF) tape, written, reloaded and
checked again.

**Applies to:** incident-neutron, laboratory-frame model suites (from ENDF-6 or GNDS)
whose resolved ranges are SLBW (LRF=1), MLBW (LRF=2), Reich-Moore (LRF=3) or R-Matrix
Limited (LRF=7); the unresolved range is on its own page,
{doc}`unresolved-resonances`.

**Entry points:** `kika.endf.reconstruct_endf`,
`kika.endf.reconstruction.EndfReconstruction.write_pendf`,
`kika.endf.reconstruction.EndfReconstruction.report_document`; on a model suite
`kika.processing.resonances.reconstruct_suite`,
`kika.processing.resonances.attach_reconstruction`; the low-level pieces
`kika.processing.resonances.prepare_resonances` and
`kika.processing.resonances.tabulate_resonances`.

**Status:** experimental — not yet in a release (after 0.4.0).
```

## Background

An ENDF (Evaluated Nuclear Data File) evaluation states the cross-sections of its
resolved resonance range as parameters: for each resonance an energy $E_\lambda$, a
total angular momentum $J$ and partial widths $\Gamma_{\lambda c}$ per channel, grouped
by orbital angular momentum $\ell$ or by spin group, in MF 2 (ENDF-6 Ch. 2). The
formula that turns them into a cross-section is part of the evaluation: the LRF flag
names it, and the evaluator fitted the parameters with that formula. MF 3 then holds only
a background in the resonance range (LRP=1), which is added to the resonance
contribution. A processing code (NJOY RECONR, MacFarlane et al. 2017) evaluates the
formula on an energy grid fine enough that linear interpolation between the points
reproduces it, and writes the result as a PENDF: MF 3 holding the full cross-sections,
LRP=2.

The formalisms are approximations of R-matrix theory (ENDF-6 App. D). Each channel $c$
(an entrance or exit pair of particles with orbital angular momentum $\ell$ and channel
spin $s$) carries three functions of $\rho = k a$, with $k$ the channel wave number and
$a$ a radius: the penetrability $P_\ell(\rho)$, the shift factor $S_\ell(\rho)$ and the
hard-sphere phase $\varphi_\ell(\rho)$. In terms of the regular and irregular Coulomb
functions $F_\ell$, $G_\ell$ (spherical Riccati-Bessel functions for a neutral pair),
$P_\ell = \rho/(F_\ell^2 + G_\ell^2)$, $S_\ell = \rho(F_\ell F_\ell' +
G_\ell G_\ell')/(F_\ell^2+G_\ell^2)$ and $\varphi_\ell = \arctan(F_\ell/G_\ell)$.
SLBW and MLBW are level expansions of the collision matrix that neglect level-level
interference in the reaction channels; Reich-Moore keeps all interference and eliminates
the many capture channels into a level width; R-Matrix Limited is the general formalism
with explicit channels, which may be charged.

kika evaluates these formulas itself (it does not run NJOY for this; for the NJOY path see
{doc}`njoy`). The worked example of the calls is
{doc}`../../../user_guide/tutorials/native_resonance_reconstruction`.

## What it computes

Input: a model `ReactionSuite` with an evaluated style (by default `eval`), a target mass
ratio $A$ (target mass over neutron mass) and target spin $I$, and the tolerances
`rtol`, `atol` of `kika.processing.resonances.ReconstructionOptions`. Output: for every
reaction of the suite a new form, labelled `recon` by default and carried by a new
`CrossSectionReconstructed` style, which is a lin-lin table (one per energy segment) such
that, at every node and at the interior probes of every panel,

$$
r(E) = \frac{\lvert \sigma(E) - \sigma_\text{lin}(E) \rvert}
{\text{atol} + \text{rtol}\,\max\big(\lvert\sigma(E)\rvert, \lvert\sigma_\text{lin}(E)\rvert\big)}
\le 1 ,
$$ (eq-resonance-reconstruction-budget)

where $\sigma$ is the cross-section the formalism and the background define (in b, $E$
in eV) and $\sigma_\text{lin}$ the table. Tabulation targets $r \le 1/2$, leaving the
other half of the budget for writing the table to a file. In each resonance region, for
a reaction $x$ the formalism produces,

$$
\sigma_x(E) = \sigma_x^\text{bg}(E) + \sigma_x^\text{res}(E),
\qquad
\sigma_S(E) = \sum_{x \in \text{leaves}(S)} \sigma_x(E),
$$ (eq-resonance-reconstruction-assembly)

with $\sigma_x^\text{bg}$ the evaluated MF 3 table and $\sigma_x^\text{res}$ the
resonance contribution of this page; every sum $S$ of the suite (MT 1, MT 3, MT 4, MT 18
with chance fission, ...) is rebuilt from its exclusive leaves, never taken from the file.
Outside the resonance regions the evaluated tables are kept.

## Method

### Common kinematics and channel functions

For a neutron incident with laboratory energy $E$ on a target of mass ratio $A$, the
entrance wave number and the factor that turns a squared amplitude into barns are

$$
k^2 = \frac{2 m_n c^2}{(\hbar c)^2}\left(\frac{A}{A+1}\right)^2 E,
\qquad
\beta(E) = \frac{\pi}{k^2}, \qquad
g_J = \frac{2J+1}{2(2I+1)},
$$ (eq-resonance-reconstruction-kinematics)

with $m_n c^2 = 939.56542052\,\text{MeV}$ and $\hbar c = 197.3269804\,\text{MeV fm}$ (the
values of `kika.processing.resonances.NeutronContext`; $k$ in fm$^{-1}$, $\beta$
converted from fm$^2$ to b by $10^{-2}$). When a radius is to be computed from the mass
(NAPS=0, below), kika uses the ENDF-6 formula

$$
a_\text{mass} = 1.23\, (A\, m_n/\text{u})^{1/3} + 0.8 \quad \text{fm}.
$$ (eq-resonance-reconstruction-mass-radius)

For a neutral channel, $P_\ell$ and $S_\ell$ come from the recurrence

$$
P_0 = \rho,\; S_0 = 0;\qquad
P_\ell = \frac{\rho^2 P_{\ell-1}}{(\ell - S_{\ell-1})^2 + P_{\ell-1}^2},\quad
S_\ell = \frac{\rho^2 (\ell - S_{\ell-1})}{(\ell - S_{\ell-1})^2 + P_{\ell-1}^2} - \ell ,
$$ (eq-resonance-reconstruction-recurrence)

for $\ell = 0, \dots, 64$ (so $P_1 = \rho^3/(1+\rho^2)$, $S_1 = -1/(1+\rho^2)$). The
phase is $\varphi_0 = \rho$, and for $\ell \ge 1$ $\varphi_\ell =
\arctan(F_\ell/G_\ell)$ from the Riccati-Bessel functions; for $\ell = 1, 2$ the
numerator is a power series below $\rho = 1$ ($\varphi_1 = \rho - \arctan\rho$ without
the cancellation), so that $\sin^2\varphi_\ell$ keeps its relative accuracy at low
energy. Where a formula needs $S_\ell(\rho_\lambda) - S_\ell(\rho)$ between two close
arguments, kika evaluates the difference directly (exact rational forms for $\ell=1,2$,
divided differences through the recurrence above), and not by subtracting two values
near $-\ell$.

### The radii and the radius flags

Each $\ell$ uses two radii: the **channel radius** $a_c$, for $P_\ell$ and $S_\ell$, and
the **phase radius** $a_\varphi$, for $\varphi_\ell$. Model radii are in fm; the ENDF
fields AP, APL, APT, APE (in $10^{-12}$ cm) are multiplied by 10 when the file is read.
For an ENDF range with NAPS and NRO (ENDF-6 §2.2.1):

| Flags | $a_\varphi$ | $a_c$ |
|---|---|---|
| NAPS=0 | AP, or AP($E$) if NRO=1 | {eq}`eq-resonance-reconstruction-mass-radius` |
| NAPS=1 | AP, or AP($E$) if NRO=1 | equal to $a_\varphi$ |
| NAPS=2 (with NRO=1) | AP($E$) | the constant AP |

For Reich-Moore, a non-zero APL of an $\ell$-block replaces AP for that $\ell$ (as
$a_\varphi$, and as $a_c$ under NAPS=1); an energy-dependent AP($E$) replaces the
constant AP but never a declared APL. An APL field of 0 means "use AP". For R-Matrix
Limited, every channel declares its own APT (as $a_c$) and APE (as $a_\varphi$). A
tabulated radius AP($E$) is evaluated with its own interpolation law; a histogram
(INT=1) radius cuts the energy grid at its steps, so no table panel straddles a jump.

### Resonance widths at the evaluation energy

Neutron widths are stated at the resonance energy. For all formalisms with widths (not
reduced amplitudes), the neutron width at $E$ and the resonance energy shifted by the
shift factor are

$$
\Gamma_{\lambda n}(E) = \Gamma_{\lambda n}\,\frac{P_\ell(k a_c)}{P_\ell(k_\lambda a_c)},
\qquad
E'_\lambda = E_\lambda + \Gamma_{\lambda n}\,
\frac{S_\ell(k_\lambda a_c) - S_\ell(k a_c)}{2 P_\ell(k_\lambda a_c)},
$$ (eq-resonance-reconstruction-widths)

with $k_\lambda$ the wave number at $\lvert E_\lambda \rvert$: a bound level
($E_\lambda < 0$) takes its reference at the absolute energy. A tabulated $a_c$ is
evaluated at each of the two energies. Capture and fission widths are constant.

### SLBW and MLBW (LRF=1, 2)

**What it computes.** For every $\ell$-block, the elastic, capture, fission and (when a
competitive channel is modelled) competitive cross-sections, and their sum. With
$d_\lambda = (E - E'_\lambda)^2 + \Gamma_\lambda^2/4$ and $\Gamma_\lambda = \Gamma_{\lambda
n}(E) + \Gamma_{\lambda\gamma} + \Gamma_{\lambda f} + \Gamma_{\lambda x}(E)$, the
reaction cross-sections of both approximations are

$$
\sigma_\gamma = \beta \sum_{\ell}\sum_{\lambda} g_{J_\lambda}
\frac{\Gamma_{\lambda n}\Gamma_{\lambda\gamma}}{d_\lambda},\qquad
\sigma_f = \beta \sum_{\ell}\sum_{\lambda} g_{J_\lambda}
\frac{\Gamma_{\lambda n}\Gamma_{\lambda f}}{d_\lambda},\qquad
\sigma_x = \beta \sum_{\ell}\sum_{\lambda} g_{J_\lambda}
\frac{\Gamma_{\lambda n}\Gamma_{\lambda x}}{d_\lambda}.
$$ (eq-resonance-reconstruction-bw-reactions)

**Method.** The elastic cross-section of SLBW sums single levels,

$$
\sigma_n^\text{SLBW} = \beta \sum_\ell \Big[ 4(2\ell+1)\sin^2\varphi_\ell
+ \sum_\lambda g_{J_\lambda} \frac{\Gamma_{\lambda n}\big(\Gamma_{\lambda n}
- 2\Gamma_\lambda \sin^2\varphi_\ell + 2(E - E'_\lambda)\sin 2\varphi_\ell\big)}
{d_\lambda}\Big],
$$ (eq-resonance-reconstruction-slbw)

and that of MLBW sums the amplitudes of all levels of one $J$ before squaring,

$$
\sigma_n^\text{MLBW} = \beta \sum_\ell \Big[ \sum_{J \in \ell} g_J
\Big( \big(2\sin^2\varphi_\ell - T_1^{\ell J}\big)^2
+ \big(\sin 2\varphi_\ell + T_2^{\ell J}\big)^2 \Big)
+ 4\Big(2\ell+1 - \sum_{J \in \ell} g_J\Big)\sin^2\varphi_\ell \Big],
$$ (eq-resonance-reconstruction-mlbw)

$$
T_1^{\ell J} = \sum_{\lambda \in (\ell, J)} \frac{\Gamma_{\lambda n}\Gamma_\lambda}{2 d_\lambda},
\qquad
T_2^{\ell J} = \sum_{\lambda \in (\ell, J)} \frac{\Gamma_{\lambda n}(E - E'_\lambda)}{d_\lambda}.
$$ (eq-resonance-reconstruction-mlbw-sums)

The first sum runs over the values of $J$ that have at least one level; the last term
adds hard-sphere scattering for the $(J, s)$ sectors of the $\ell$ that have none, so
that the potential weight of every $\ell$ is $2\ell+1$. The total is
$\sigma_t = \sigma_n + \sigma_\gamma + \sigma_f + \sigma_x$.

A competitive width (LRX=1) is the width of an inelastic exit with
orbital momentum $\ell_x$, Q-value $Q$, exit mass ratio $A_x$ and exit radius $a_x$
(default {eq}`eq-resonance-reconstruction-mass-radius` with $A_x$):

$$
\Gamma_{\lambda x}(E) = \Gamma_{\lambda x}\, \frac{P_{\ell_x}(k_x a_x)}{P_{\ell_x}(k_{x\lambda} a_x)},
\qquad
k_x^2 = \frac{2 m_n c^2}{(\hbar c)^2}\frac{A_x}{A_x+1}\Big(\frac{A}{A+1}E + Q\Big),
$$ (eq-resonance-reconstruction-competitive)

zero below threshold, with the reference $k_{x\lambda}$ taken at
$\lvert A E_\lambda/(A+1) + Q \rvert$. The stated width is
$\Gamma_{\lambda x} = \Gamma_{\lambda t} - \Gamma_{\lambda n} - \Gamma_{\lambda\gamma} -
\Gamma_{\lambda f}$ (GT minus the partial widths). The level shift
{eq}`eq-resonance-reconstruction-widths` is applied for $\ell \ge 1$ (it is identically
zero for $\ell = 0$).

**Conventions and choices.**

- **$\lvert J \rvert$ is used, its sign ignored**: under LRF=1 and 2 the sign of AJ
  carries no channel-spin information, and $g_J$ depends only on $\lvert J\rvert$.
- **GT is checked, not trusted.** A zero GT means "not given". A non-zero GT must equal
  GN+GG+GF to a relative $5\cdot10^{-6}$ (rounding of the six-digit fields), and then
  GN+GG+GF is used and the difference noted; a larger excess is competitive physics, and
  without a modelled competitive channel the range is refused.
- **Competition already in MF 3 is not added twice.** ENDF puts the competitive
  cross-section of LRF=1, 2 in the MF 3 of its reaction (the ENDF reader assigns it to
  MT 51); the competitive width then enters $\Gamma_\lambda$ but its
  $\sigma_x^\text{res}$ is set to zero, and the MF 3 background is the cross-section.

**Limits and failure modes.** Refused (`UnsupportedResonanceError`): $\ell > 64$; two
blocks with the same $\ell$; negative widths; a level at exactly $E_\lambda = 0$; a $J$
not reachable from $\ell$ and $I$; an LRX=1 block without an exit $\ell_x$, which ENDF
does not state and the caller must supply in the model; a competitive reference exactly
at threshold; an underflowing reference penetrability.

### Reich-Moore (LRF=3)

**What it computes.** Elastic, capture and fission cross-sections per spin sector
$(\ell, J, s)$, with all level-level interference in the neutron and fission channels and
the capture channels eliminated into $\Gamma_{\lambda\gamma}$.

**Method.** The amplitudes of a level, with $P_\ell$ at $\rho = k a_c$, are

$$
A_{\lambda n}(E) = \operatorname{sgn}(\Gamma_{\lambda n})
\sqrt{\frac{\lvert\Gamma_{\lambda n}\rvert}{2 P_\ell(k_\lambda a_c)}}\,\sqrt{P_\ell(k a_c)},
\qquad
A_{\lambda f} = \operatorname{sgn}(\Gamma_{\lambda f})\sqrt{\frac{\lvert\Gamma_{\lambda f}\rvert}{2}},
$$ (eq-resonance-reconstruction-rm-amplitudes)

for the neutron channel and the two fission channels (GFA, GFB) of the record; a fission
column that is zero for every level of a sector is dropped. With
$\tilde d_\lambda = E_\lambda - E - \tfrac{i}{2}\Gamma_{\lambda\gamma}$, the reduced
R-matrix of the sector and the solution for entrance $n$ are

$$
R_{cc'}(E) = \sum_\lambda \frac{A_{\lambda c} A_{\lambda c'}}{\tilde d_\lambda},
\qquad
(\mathbb{1} - i R)\, y = e_n,\qquad w = R\, y,
\qquad
X_\lambda = \frac{1}{\tilde d_\lambda}\sum_c A_{\lambda c}\, y_c ,
$$ (eq-resonance-reconstruction-rm-solve)

the collision matrix elements are $U_{nn} = e^{-2i\varphi_\ell}(1 + 2i w_n)$ and
$\lvert U_{nf}\rvert^2 = 4\lvert w_f\rvert^2$, and the cross-sections are

$$
\sigma_n = \beta g_J \big\lvert 1 - U_{nn} \big\rvert^2,\qquad
\sigma_f = \beta g_J \sum_f \lvert U_{nf}\rvert^2,\qquad
\sigma_\gamma = \beta g_J\, 2\sum_\lambda \Gamma_{\lambda\gamma}\lvert X_\lambda\rvert^2 ,
$$ (eq-resonance-reconstruction-rm-sigma)

summed over the sectors; $\sigma_t = \sigma_n + \sigma_\gamma + \sigma_f$. This is the
ENDF-6 Reich-Moore collision matrix $U = e^{-2i\varphi}\big[2(\mathbb{1}-K)^{-1} -
\mathbb{1}\big]$ with $K = iR$, since $2(\mathbb{1}-iR)^{-1} - \mathbb{1} = \mathbb{1} +
2iR(\mathbb{1}-iR)^{-1}$. $1 - U_{nn}$ is formed as $2i e^{-i\varphi}\sin\varphi -
2i e^{-2i\varphi} w_n$, which avoids cancelling the hard-sphere phase.

The spin sector of a level comes from the sign of its AJ: $s = I - \tfrac12$ for
AJ $< 0$ and $s = I + \tfrac12$ otherwise (and $s = I - \tfrac12$ when AJ $\ge 0$ is
reachable only from that channel spin). Sectors with the same $(\ell, J)$ and different
$s$ are incoherent. Every $(\ell, J, s)$ sector that the declared $\ell$ values allow and
that has no level adds its hard-sphere term $4\beta g_J \sin^2\varphi_\ell$; NLSC is not
used to invent further $\ell$ values.

The linear system is solved per energy. When a level is a "naked pole"
($\lvert\tilde d_\lambda\rvert \le 10^{-6}\sum_c A_{\lambda c}^2$, for instance a level
with $\Gamma_{\lambda\gamma} = 0$ at $E = E_\lambda$), the division by $\tilde
d_\lambda$ is replaced by an augmented channel-and-level system that keeps $X_\lambda$ as
unknowns. Every solve must have a relative residual $\le 10^{-11}$, and the flux balance

$$
\Big\lvert 1 - \lvert U_{nn}\rvert^2 - \sum_f \lvert U_{nf}\rvert^2
- 2\sum_\lambda \Gamma_{\lambda\gamma}\lvert X_\lambda\rvert^2 \Big\rvert \le 10^{-9}
$$ (eq-resonance-reconstruction-flux)

must hold at every energy; otherwise the evaluation raises.

**Conventions and choices.**

- **Capture is computed directly**, as the positive sum
  {eq}`eq-resonance-reconstruction-rm-sigma`, and not as $1 - \sum\lvert U\rvert^2$:
  the subtraction loses all accuracy where capture is small next to scattering.
  {eq}`eq-resonance-reconstruction-flux` is a check, never a source of the result.
  During tabulation (not during verification) kika may instead use the equivalent
  channel-space form $4\, y^\dagger\,\operatorname{Im}(R)\, y$, only where a rounding
  bound shows it within $\min(10^{-8}, 10^{-5}\,\text{rtol})$ relative (and $10^{-10}$
  absolute) of the direct sum.
- **No shift factor.** The ENDF-6 Reich-Moore convention sets the boundary condition
  equal to the shift ($L_c = iP_c$), and kika implements only that convention.
- **No energy displacement, no small artificial width and no clipping** at a pole; the
  augmented system gives the limit, and a degenerate pole is accepted only through a
  least-squares solution whose residual passes the same check.

**Limits and failure modes.** Refused: a negative capture width; a negative neutron
width in an LRF=3 record; a zero-energy level with a neutron width; a boundary condition
other than the eliminated shift; relativistic kinematics; additional phases or external
R-matrices on a Reich-Moore range (they belong to LRF=7); two groups that state
different radii or masses for the same $\ell$; a duplicate $(\ell, J, s)$ sector.

### R-Matrix Limited (LRF=7)

**What it computes.** Cross-sections for every channel of every spin group $(J, \pi)$:
the elastic entrance, inelastic and charged-particle exits (MT 51-91, 103-107,
600-849), and effective fission and capture channels; with KRM=3 the capture channel is
eliminated into $\Gamma_{\lambda\gamma}$, with KRM=4 every channel is explicit.

**Method.** Each channel $c$ has its threshold, channel energy and wave number. With
$\mu_c$ the reduced mass of the pair (from the particle-pair masses, in units of $m_n$)
and $Q_c$ its Q-value,

$$
E_c = \frac{A}{A+1}E + Q_c,\qquad
k_c^2 = \frac{2\mu_c c^2}{(\hbar c)^2}E_c,\qquad
\eta_c = \frac{Z_a Z_b\,\alpha\,\mu_c c^2}{\hbar c\; k_c},
$$ (eq-resonance-reconstruction-rml-kinematics)

and, under KRL=1, $k_c$ from the invariant mass,
$(\hbar c\, k_c)^2 = t(t+2M)(t+2m_a)(t+2m_b)/\big(4(t+M)^2\big)$ with $t$ the kinetic
energy of the pair in its centre of mass and $M = m_a + m_b$ (the Sommerfeld parameter
keeps the rest reduced mass). The channel functions are those of a neutral or Coulomb
pair at $\rho_c = k_c a_c$; for a closed channel ($E_c < 0$) $P_c = 0$ and $S_c$ is the
logarithmic derivative of the decaying solution (for a neutral pair, the exponential
times a finite polynomial in $1/\kappa\rho$; for a charged pair, the Whittaker function
$W_{-\eta, \ell+1/2}(2\kappa\rho)$), with its exact threshold limit. Effective (fission,
capture) channels have $P_c = 1$, $S_c = 0$, $\varphi_c = 0$. A channel whose
penetrability flag is "unity" has $P_c = 1$. Then

$$
L_c = S_c - B_c + iP_c \;\;(\text{SHF}=1),\qquad L_c = -B_c + iP_c \;\;(\text{SHF}=0),
$$ (eq-resonance-reconstruction-rml-l)

$$
R_{cc'} = \sum_\lambda \frac{\gamma_{\lambda c}\gamma_{\lambda c'}}{\tilde d_\lambda}
+ \delta_{cc'} R^\text{ext}_c(E),
\quad
(\mathbb{1} - L R)\,y = e_n,\quad W = R y,\quad
X_\lambda = \frac{1}{\tilde d_\lambda}\sum_c \gamma_{\lambda c} y_c,
$$ (eq-resonance-reconstruction-rml-solve)

with $\tilde d_\lambda = E_\lambda - E - \tfrac{i}{2}\Gamma_{\lambda\gamma}$
($\Gamma_{\lambda\gamma} = 0$ under KRM=4) and the reduced amplitudes

$$
\gamma_{\lambda c} = \operatorname{sgn}(\Gamma_{\lambda c})
\sqrt{\frac{\lvert\Gamma_{\lambda c}\rvert}{2P_c(\lvert E_\lambda\rvert)}}
\quad (\text{IFG}=0),\qquad
\gamma_{\lambda c} \text{ as written} \quad (\text{IFG}=1),
$$ (eq-resonance-reconstruction-rml-amplitudes)

where $P_c(\lvert E_\lambda\rvert)$ is the penetrability at the channel energy of the
absolute laboratory resonance energy (1 for effective channels), and, under KRM=3,
$\Gamma_{\lambda\gamma} = \lvert\Gamma_{\lambda,\text{capture}}\rvert$ (IFG=0) or
$2\gamma_{\lambda,\text{capture}}^2$ (IFG=1). For an entrance channel $n$ (every elastic
channel of the group is an entrance in turn),

$$
U_{cn} = e^{-i(\varphi_c + \varphi_n)}\Big[\delta_{cn} + 2i\sqrt{P_c P_n}\,W_c\Big],
\qquad
\sigma_{n\to c} = \beta_n g_J \big\lvert \delta_{cn} - U_{cn} \big\rvert^2,
$$ (eq-resonance-reconstruction-rml-sigma)

with $\beta_n = \pi/k_n^2$, and the cross-section of a reaction MT is the sum over its
channels; a channel stated as MT 19 is counted as fission. Capture is the direct sum

$$
\sigma_\gamma = \beta_n g_J \Big[2P_n\sum_\lambda \Gamma_{\lambda\gamma}\lvert X_\lambda\rvert^2
+ 4P_n\sum_c \operatorname{Im}R^\text{ext}_c\,\lvert y_c\rvert^2\Big]
$$ (eq-resonance-reconstruction-rml-capture)

(plus the explicit capture channel under KRM=4), and $\sigma_t$ is the sum of all
partials. The flux balance {eq}`eq-resonance-reconstruction-flux`, written with all the
$U_{cn}$ of the group and the absorption of
{eq}`eq-resonance-reconstruction-rml-capture`, must hold to $10^{-9}$; the solve residual
to $10^{-11}$; naked poles use the augmented system as for Reich-Moore.

The options of the format that change the result:

- **Boundary conditions.** $B_c$ is the channel's BND, or the value of the formalism, or
  $-\ell$ under the GNDS "NegativeOrbitalMomentum" convention, or 0. Under the
  eliminated-shift convention $S_c - B_c = 0$.
- **External R-matrix (KBK, KRM=3 only).** The SAMMY form,
  $R^\text{ext} = R_0 + R_1 E + R_2 E^2 - s_1(E_u - E_d) - (s_0 + s_1 E)\ln\frac{E_u -
  E}{E - E_d}$, the Fröhner form, $R^\text{ext} = R_0 + 2 s_c \operatorname{artanh}u +
  i\,\bar\Gamma_\gamma\big/\big((E_u - E_d)(1-u^2)\big)$ with $u = (2E - E_u - E_d)/(E_u
  - E_d)$, or a tabulated complex function; the singularities $E_d < E_u$ must bracket
  the range.
- **Additional phase (KPS).** A tabulated complex phase replaces $\varphi_c$ of its
  channel. Its imaginary part must be $\le 0$ (passive); the flux it absorbs,
  $(1 - e^{2\operatorname{Im}\varphi_n}) + e^{2\operatorname{Im}\varphi_n}\sum_c (1 -
  e^{2\operatorname{Im}\varphi_c})\lvert\delta_{cn} + 2i\sqrt{P_cP_n}W_c\rvert^2$, times
  $\beta_n g_J$, is assigned to the reaction the model names for it, and capture is
  multiplied by $e^{2\operatorname{Im}\varphi_n}$.
- **Brune parameters.** With the Brune boundary convention (C. R. Brune, Phys. Rev. C 66,
  044611 (2002)) the parameters $(\tilde E_\lambda, \gamma_{\lambda c})$ are alternative
  ones, with $S_c$ computed and $B_c = 0$. With $S_{\lambda c} =
  S_c(\tilde E_\lambda)$ and $E_0 = \tilde E_1$, kika builds the level metric $M$ and
  matrix $N$,
  $M_{\lambda\mu} = \delta_{\lambda\mu} - (1-\delta_{\lambda\mu})\sum_c
  \gamma_{\lambda c}\gamma_{\mu c}\frac{S_{\lambda c} - S_{\mu c}}{\tilde E_\lambda -
  \tilde E_\mu}$,
  $N_{\lambda\lambda} = \tilde E_\lambda - E_0 + \sum_c \gamma_{\lambda c}^2 S_{\lambda
  c}$,
  $N_{\lambda\mu} = \tfrac12(\tilde E_\lambda + \tilde E_\mu - 2E_0) M_{\lambda\mu} +
  \tfrac12\sum_c \gamma_{\lambda c}\gamma_{\mu c}(S_{\lambda c} + S_{\mu c})$, and solves
  $x = \big(N - (E - E_0)M - \tfrac{i}{2}\operatorname{diag}\Gamma_\gamma - \gamma L
  \gamma^\top\big)^{-1}\gamma_{\cdot n}$, $W_c = \sum_\lambda \gamma_{\lambda c}x_\lambda$
  (plus $R^\text{ext}$), in place of {eq}`eq-resonance-reconstruction-rml-solve`. $M$ must
  be positive definite and the $\tilde E_\lambda$ distinct.
- **Signed channel spin.** ENDF-6 lets SCH be negative, the sign only telling apart two
  channels with the same $\lvert s\rvert$ (JEFF-4.0 O-16 writes its α channels with
  $s = -1/2$). The coupling of $\ell$, $s$ and $J$ is checked with $\lvert s\rvert$, and
  the sign stays in the channel's identity.
- **Split sectors.** ENDF may write one $(J, \pi)$ as several spin groups (one per
  channel spin); kika merges them into one sector over the union of their channels, so
  that shared channels interfere and hard-sphere scattering is counted once.
- **Missing sectors.** As for Reich-Moore, every $(\ell, J, s)$ allowed by an $\ell$ of
  the elastic channels and absent from the file adds its hard-sphere term.

The Coulomb functions are computed in double precision from the continued fractions of
DLMF §33.8 (modified Lentz algorithm), from the Tricomi series (DLMF §13.2.9) or a
Laplace integral at small $\rho$, and, below the turning point, by integrating the Riccati
equations of $S$ and $\ln P$ in $\ln\rho$,

$$
\frac{dS}{d\ln\rho} = S - \rho^2 + 2\eta\rho + \ell(\ell+1) - S^2 + P^2,\qquad
\frac{d\ln P}{d\ln\rho} = 1 - 2S ,
$$ (eq-resonance-reconstruction-coulomb-transport)

inward from $\max(\rho, \rho_\text{turn} + 8)$, with $\rho_\text{turn} = \eta +
\sqrt{\eta^2 + \ell(\ell+1)}$. A penetrability below the smallest normal double is carried
as $\ln P$, and the solve then bounds the error that its omission makes in every
observable (relative $2\cdot10^{-12}$); an unbounded case raises.

**Limits and failure modes.** Refused: KRM=1, 2 (SLBW, MLBW written as LRF=7); a
formalism whose approximation is neither `ReichMoore` (KRM=3) nor `FullRMatrix` (KRM=4),
the GNDS-2.0 names (an ENDF LRF=7 range decodes to them from KRM, and the GNDS reader maps
the spellings `Reich_Moore` and `Full R-Matrix` that some published files carry to the
same two); an unknown boundary condition; a global radius policy on LRF=7 (the
channels carry their radii); an attractive charged exit; a charged pair on a neutron MT;
an inconsistent parity, spin or channel spin; a missing Q; an elastic pair that disagrees
with the incident target; a capture channel not eliminated under KRM=3, or an external
R-matrix under KRM=4; an active KPS phase (imaginary part $> 0$) or a KPS absorption
without an owning reaction; a degenerate Brune level set; an open, active channel whose
penetrability underflows without a representable logarithm.

### Adler-Adler and other formats

Adler-Adler (LRF=4) and LRF values other than 1, 2, 3, 7 are not read: the ENDF reader
records the range as unsupported, the conversion report is then not clean for
cross-sections, and reconstruction refuses the material (`conversion-not-clean`). GNDS has
no node for LRF=4, and no evaluation in ENDF/B-VIII.1, JEFF-4.0 or JENDL-5 uses it. A material with
several isotopes in MF 2 (NIS > 1) is refused for the same reason.

### Assembling the material

1. **Context.** $A$ and $I$ come from the model's particle data, or are given as a
   `NeutronContext`. `reconstruct_endf` takes them as the evaluation declares them: the
   elastic pair of an LRF=7 range; otherwise the AWRI of the first MF 2 block that states
   one (else MF 1 AWR) and the SPI of the first range. A block's own AWRI applies to that
   block.
2. **Ownership.** Each MF 2 partial (MT 2, 18, 102, a competitive or channel MT) is added
   to the model reaction with that MT. MT 18 is owned by MT 19 when the suite represents
   fission by its chances (MT 18 a sum of MT 19, 20, 21, 38), so that the resonance
   fission enters MT 18 once, through MT 19 (ENDF-6 §3.4.5). A partial with no exclusive
   reaction, or omitted from the total's graph while non-zero, is an error.
3. **Kinematic thresholds.** For every exclusive reaction with $Q < 0$, the threshold
   $E_\text{thr} = -Q\,(A+1)/A$ is rounded to seven significant figures and raised by
   one unit in the seventh, $E^{(7)}_\text{thr}$ (with $A$ from the model's particle
   data, else from the context). A table whose first energy lies below
   $E^{(7)}_\text{thr}$ has that energy replaced by it, and each following energy the
   move overtakes is raised one unit in its seventh figure above the one before it. The
   values are not changed, so the evaluator's first value (normally 0) now sits at
   $E^{(7)}_\text{thr}$; sums follow from their leaves.
4. **Segments.** The domain (the union of the evaluated tables, which must equal the
   projectile domain the evaluated style declares, when it declares one) is cut at every region boundary, every end and
   discontinuity of a table (a repeated energy, a histogram knot), and every step of a
   radius or parameter table. At a shared boundary the region on the right owns the point;
   the segment on the left is evaluated at the next representable energy below it, so the
   output keeps a repeated energy with both one-sided values.
5. **Seeds.** In a resonance segment: 65 energies equally spaced in $\ln E$, the ends,
   every knot of the backgrounds, radii and parameter tables, competitive thresholds,
   and for every level with $E_\lambda > 0$ its energy, its shifted peak (for SLBW/MLBW
   four fixed-point iterations of {eq}`eq-resonance-reconstruction-widths`; for R-Matrix
   Limited eight of $E_\lambda - \sum_c\gamma_{\lambda c}^2\operatorname{Re}L_c$), and
   $E_\lambda \pm m\,\Gamma_\lambda$ for 35 values of $m$ equally spaced in $\ln m$ from
   $1/64$ to $128$. Outside resonance regions, the knots of the evaluated tables.
6. **Refinement.** All reactions of a segment are refined on one shared grid
   (`kika.algebra.refine`). Each panel is probed at the fractions 0.25, 0.5, 0.75; a
   panel that passes is also probed at 0.1127, 0.2764, 0.7236, 0.8873. A panel fails when
   any reaction has $r > 1/2$ ({eq}`eq-resonance-reconstruction-budget`) at any probe; it
   then gains its worst probe and the probe nearest its midpoint. Refinement stops when
   every panel passes, and raises when it exceeds `max_points` (default 200 000;
   4 000 000 in `reconstruct_endf`), `max_iterations` passes (default 40), or a failing
   panel can no longer be split in double precision.
7. **Untouched reactions.** A reaction with no resonance contribution in a segment keeps
   its evaluated lin-lin knots there instead of the shared grid; an exactly constant span
   is stored as its two ends. Sums are rebuilt from their leaves at every node.
8. **Verification.** `attach_reconstruction` re-evaluates the physics exactly (direct
   capture, no channel-space shortcut) at every node and at four other fractions of every
   panel (0.1732, 0.3820, 0.6180, 0.8268), checks {eq}`eq-resonance-reconstruction-budget`
   with the full budget for every reaction, and checks that each stored sum equals the
   sum of its stored leaves to the same budget. Only then is the new style attached (it
   carries the date it was made, as GNDS requires of every style); the evaluated forms
   are untouched.
9. **Publication.** `write_pendf` checks that the source has not changed since the
   reconstruction (a hash of the model), verifies again, resamples each table at the
   energies the ENDF eleven-character fields can represent (the linear table evaluated at
   the rounded energies; colliding knots keep their original values), writes MF 1, 2 and
   3 with LRP=2 and LDRV=1 (ENDF-6 §1.1: MF 2 is kept but its contribution is already in
   MF 3), reloads the file, requires a clean conversion, and verifies the reloaded tables
   against the physics with the full budget. The destination is replaced only if all of
   this passes. Two rules of the writing keep the file readable by NJOY. In MF 3,
   neighbouring interpolation regions of the same law are written as one (NBT, INT)
   pair, with each jump as a repeated energy: the function is the same, and NJOY holds
   at most 20 regions per table, while a reconstructed table has one per discontinuity.
   The checks that read the file back cut it again at their own region edges, and fail
   if an edge is not a point of the table. In MF 1, the descriptive text keeps only the
   five-line identification block of ENDF-6 §1.1, as RECONR does; the evaluation's
   documentation stays in the evaluation.

## Conventions and choices

- **The tolerance is mixed, and split in two.** A relative tolerance alone fails at the
  zeros of interference minima; `atol` (default $10^{-8}$ b) bounds the work there. Half
  of the budget is spent on tabulation so that rounding the energies to ENDF fields
  cannot, by itself, push a correct table out of tolerance. NJOY RECONR tests the
  partials in turn against `err` (with `errmax` and a local criterion `errint` in b);
  FUDGE tests each node against the chord to its neighbours and inserts midpoints. Both
  spend the whole tolerance on the grid, so the ENDF rounding of the written file comes on
  top of it.
- **The check is empirical, and every reaction is checked at points the grid did not
  use.** The probes are a finite set of energies; the report says
  `empirical_verification=True` and `global_error_bound=False`. The verification probes
  differ from the refinement probes, so the grid was not built to pass them. FUDGE never
  re-examines a reaction it has marked converged and takes no interior probes, which is
  where its tables leave the tolerance in the comparison under Validation; NJOY does not
  re-check its grid either. The check is what kika's run costs most: on U-238 of
  ENDF/B-VIII.1 the verification took 912 s against 109 s for the reconstruction (one
  thread, measured 2026-10-10), a cost neither other code pays.
- **Seeds per level instead of a dense pattern.** FUDGE seeds hundreds of offsets around
  every resonance; kika seeds the level, its shifted peak and 35 offsets in units of its
  width, and lets the refinement do the rest. A minimal-seed variant was measured to be
  much slower and is not used.
- **Sums are derived, never kept.** An evaluated MT 1 that disagrees with the sum of its
  partials is reported (`source_sum_discrepancies`) and replaced, so the PENDF closes and
  a sum cannot double count a partial. MT 1 present without an explicit sum graph is
  refused instead of guessed. NJOY accumulates the partials into the sums as it goes;
  FUDGE may add a reconstructed contribution to the background of an evaluated sum. kika
  follows the model's sum graph, so which reactions a sum holds is stated by the data and
  not by a list of MT numbers.
- **Negative values are kept and reported**, never clipped (a background can be negative
  in the resonance range by construction); `minima` in the report gives the smallest
  value per reaction. NJOY clips negative partials and floors the elastic; FUDGE clips
  after assembly. Clipping makes the table depart from the evaluation without saying
  where; a transport code that needs non-negative values can clip on reading, with the
  report saying where it matters.
- **Values in double precision, rounding only on export.** NJOY rounds to seven figures
  while it assembles (SLBW before adding the potential term). kika keeps float64 until the
  file is written, then resamples the verified table at the energies the ENDF fields can
  represent and verifies the file again; on a steep flank of U-238 near 16 689 eV,
  rounding alone moved a correct table to 1.3 times its budget, which this catches.
- **A range boundary is an exact step.** At a boundary between two ranges, or at a
  discontinuity of a table, kika writes a repeated energy with both one-sided values.
  FUDGE replaces the step by a ramp $10^{-8}$ wide in relative energy, and NJOY by a ramp
  of about $5\cdot10^{-7}$; the PENDF of NJOY for Al-27 of ENDF/B-VIII.0 then differs
  from the physics by 62 % in capture just below 845 keV, at the one energy inside the
  ramp.
- **Thresholds are raised to the kinematic one, as RECONR does.** Evaluations round
  $-Q(A+1)/A$ to seven figures, sometimes down, and so open a reaction a few parts per
  million below the energy at which it can occur, where its kinematics has no solution
  (ACER reports it as a threshold error). kika applies RECONR's rule (step 3 of the
  assembly): on Au-197 of ENDF/B-VIII.1 it raises 15 reactions, the first energies of
  MT 17 and 51-63 then equal those of the official OpenMC HDF5 library (processed with
  NJOY) to $10^{-13}$ relative, and
  $\sigma_{51}(77\,800\ \text{eV})$ moves from 0.002660 b to NJOY's 0.002546 b. It is a
  correction of consistency with the energetics, not a more physical shape near the
  threshold: the linear ramp above it is still the evaluator's, and the move is of parts
  per million in energy.
- **APL with NAPS=1 enters the phase and the penetrability.** For a Reich-Moore range
  with NAPS=1, kika uses a non-zero APL of an $\ell$-block as $a_c$ and $a_\varphi$, as
  NJOY does, because NAPS=1 states that the two radii are the same. FUDGE's ENDF to GNDS
  translation puts APL into the phase only, and the published GNDS files of
  ENDF/B-VIII.1 carry that reading; on Fe-56 the two differ in the resonance part of the
  elastic by a median of $2.6\cdot10^{-4}$ and up to about 1.7 %. Which one the evaluator
  meant is not settled by the format; kika states its choice here.
- **One radius policy throughout NAPS=2.** NJOY prepares the widths of a NAPS=2 range
  with the mass-formula radius and evaluates $P_\ell(E)$ with the constant AP; kika uses
  the radii of the table under *The radii and the radius flags* everywhere. On a NAPS=2
  test case with a tabulated AP($E$) NJOY differs from both kika and FUDGE by about 97 %
  of the budget, while those two agree to $10^{-10}$.
- **Level normalisation at the absolute energy.** For a bound level of an R-Matrix
  Limited channel, kika takes the reference penetrability at the channel energy of
  $\lvert E_\lambda\rvert$ (ENDF-6 App. D); NJOY and FUDGE take
  $\lvert E_\lambda - E_\text{channel}\rvert$. On FUDGE's own LRF=7 test tapes this alone
  makes MT 600 differ by 0.55 % from NJOY, and 2.3·10⁻⁶ with the convention changed; it
  is a choice of convention, and neither result is shown to be wrong.
- **Capture as a positive sum.** kika computes capture directly from the level
  amplitudes ({eq}`eq-resonance-reconstruction-rml-capture`) and uses the flux balance
  only as a check. FUDGE obtains capture from the flux deficit, which loses its relative
  accuracy when capture is small: on a synthetic Reich-Moore level with a tiny capture
  width near a pole its NumPy path returns up to $10^{29}$ b, while kika's stays finite
  and within $10^{-13}$ of the closed form.
- **No fallback.** There is no retry with looser tolerances, another formalism or NJOY;
  a material either reconstructs within its budget or raises with a category.
- **Physical constants are fixed and recorded** in the context, so a comparison with
  another processor can match them.
- **0 K only.** No Doppler broadening is applied; the PENDF is at 0 K. Heating the
  reconstructed tables is {doc}`doppler-broadening`.

## Limits and failure modes

Every refusal raises `kika.processing.resonances.UnsupportedResonanceError` (the input is
outside what the engine implements) or `ReconstructionConvergenceError` (a numerical
budget or a check failed), each with a `category` listed in
`kika.processing.resonances.REJECTION_CATEGORIES`:

- `already-reconstructed`: the tape is LRP=2;
- `conversion-not-clean`: the conversion to the model lost or approximated data that
  affects cross-sections (an unread LRF, several isotopes, ...);
- `context-missing`: no target mass or spin declared;
- `missing-resonance-regions`: LRP=1 or a resonance background without usable MF 2;
- `incomplete-reactions`, `domain-not-covered`, `internal-gap`: the material is not
  completely represented;
- `total-without-graph`, `sum-without-graph`, `incomplete-total-graph`: a sum cannot be
  certified;
- `missing-radius-policy`: an unresolved range without a radius policy;
- `unsupported-physics`: the refusals listed per formalism above;
- `budget-exhausted`, `memory-budget-exhausted`: `max_points`, or a dense pole system
  larger than `max_work_bytes` (default 64 MiB);
- `verification-failed`, `publication-failed`: a check above failed, or the file could
  not be written and reloaded cleanly;
- `invalid-J`: an unresolved spin $J$ that the target spin and $\ell$ cannot reach;
- `iterations-exhausted`: the refinement passes ran out before the table converged;
- `float64-separation`: two energies the refinement needs cannot be separated in double
  precision.

Limits of the method: a resonance narrower than the spacing of representable doubles at
its energy cannot be tabulated; a resonance narrow enough that rounding its energy to an
ENDF field moves the peak by more than the budget allows makes publication fail (the
tables are still available); the probes cannot detect a feature that falls between all
seeds and probes of a panel, which is what the per-level seeds are for.

## Validation

The tests below passed on 2026-10-11 (656 passed, 22 skipped, with the compiled kernel
loaded) unless marked. Every comparison is with an independent calculation written in the
test.

- **Channel functions.** $P_\ell$, $S_\ell$ and $\sin^2\varphi_\ell$ against the
  Riccati-Bessel definitions for $\ell = 0..3$, $\rho = 10^{-7}..100$, relative
  $2\cdot10^{-13}$ to $2\cdot10^{-12}$
  (`kika/processing/tests/test_resonance_foundations.py::test_ps_against_bessel_wronskian`);
  the shift difference against a 100-digit recurrence
  (`kika/processing/tests/test_resonance_stability.py::test_shift_difference_against_100_digit_recurrence`).
- **SLBW and MLBW.** An isolated s-wave level against the complex collision amplitude,
  relative $2\cdot10^{-12}$ (elastic) and $2\cdot10^{-14}$ (capture)
  (`kika/processing/tests/test_resonance_foundations.py::test_isolated_swave_partial_widths_and_complex_elastic`);
  two interfering levels (`::test_multilevel_interference_and_block_order_invariance`);
  a shifted p-wave level against an independent collision matrix, $2\cdot10^{-12}$
  (`kika/processing/tests/test_resonance_extended.py::test_shifted_pwave_against_independent_complex_collision_matrix`);
  competition (`::test_competition_against_complex_swave`); absent $J$ sectors
  (`kika/processing/tests/test_resonance_foundations.py::test_potential_only_includes_absent_spin_sectors`).
- **Reich-Moore.** An isolated level against the closed form, $3\cdot10^{-14}$ (capture)
  and $3\cdot10^{-12}$ (elastic), $\ell = 0..3$
  (`kika/processing/tests/test_reich_moore.py::test_isolated_level_against_rm_closed_form_without_bw_shift`);
  capture from $0$ to $0.2$ eV widths to $10^{-13}$ with flux error $< 2\cdot10^{-14}$
  (`::test_capture_is_positive_and_accurate_even_below_flux_subtraction`); exact undamped
  poles (`::test_exact_undamped_pole_uses_limit_and_preserves_unitarity`); an LRF=3 range
  and its LRF=7 equivalent give the same collision physics
  (`::test_lrf3_and_equivalent_lrf7_decode_to_identical_collision_physics`).
- **R-Matrix Limited.** Channel solve against a dense level-space solve, $8\cdot10^{-14}$
  (`kika/processing/tests/test_r_matrix.py::test_channel_solve_against_independent_dense_level_solve`);
  model cross-sections with closed channels against level space, $3\cdot10^{-12}$
  (`::test_model_cross_sections_against_level_space_including_closed_channels`);
  Coulomb $P$, $S$ against 60-digit references, $5\cdot10^{-10}$
  (`kika/processing/tests/test_r_matrix_advanced.py::test_coulomb_against_60_digit_reference`);
  a relativistic charged exit against a 60-digit level matrix
  (`::test_relativistic_charged_against_independent_60_digit_level_matrix`); Brune
  parameters against Brune's equation 33 and the formal transform, $3\cdot10^{-11}$
  (`kika/processing/tests/test_r_matrix_completion.py::test_brune_collision_against_independent_equation_33_and_formal_transform`);
  threshold Coulomb shifts against 70- and 90-digit Whittaker references
  (`::test_threshold_coulomb_against_independent_70_and_90_digit_whittaker`); an ENDF
  LRF=7 section with KRM=4 reconstructs under the approximation it decodes to
  (`kika/processing/tests/test_r_matrix_advanced.py::test_an_endf_krm4_section_reconstructs_without_overriding_the_approximation`).
- **Tabulation.** The table's integral against an independent adaptive quadrature of the
  physics, within the budget
  (`kika/processing/tests/test_resonance_tabulation.py::test_tabulated_integral_against_independent_adaptive_quadrature`);
  ENDF decimal write and parse stay within the budget
  (`::test_endf_actual_decimal_write_and_parse_are_within_budget`); a peak that rounding
  destroys is rejected (`::test_endf_rounding_can_destroy_a_narrow_peak_and_is_rejected`);
  steep flanks are resampled at the rounded energies
  (`kika/processing/tests/test_resonance_suite.py::test_endf_rounding_resamples_steep_flanks_without_mutating_source`);
  competition in MF 3 is not added twice
  (`::test_competition_already_in_background_is_not_added_twice`); fission is owned by
  the first chance (`kika/processing/tests/test_resonance_material.py::test_mf2_fission_is_owned_by_first_chance_and_summed_once`).
- **Compiled kernel against NumPy.** The exact compiled path against the NumPy solver on
  the Fe-56 Reich-Moore range of the test tape, $10^{-11}$
  (`kika/processing/tests/test_rm_native.py::test_the_exact_native_witness_is_the_numpy_reference_on_a_real_region`);
  the compiled Breit-Wigner sums, $5\cdot10^{-13}$
  (`kika/processing/tests/test_resonance_workspace.py::test_native_bw_matches_reference_all_levels_and_missing_spin_sectors`).
- **End to end on a tape.** `reconstruct_endf` on
  `kika/endf/tests/data/micro_fe56_structural.endf`, its tables and its report
  (`kika/endf/tests/test_reconstruct_endf.py`); the PENDF written, reloaded and not
  reconstructed twice
  (`kika/endf/tests/test_reconstruct_endf.py::test_pendf_is_written_reloaded_and_not_reconstructed_twice`,
  deselected by default, run with `-k`, passed 2026-10-11); threaded verification equals
  the serial one (`::test_threaded_verification_is_the_serial_one`); the ENDF field
  rounding (`kika/endf/tests/test_reconstruction_precision.py`); each check of the
  report flags what it should (`::test_each_check_flags_what_it_should`).
- **Kinematic thresholds.** A table that opens below the threshold is raised to
  $E^{(7)}_\text{thr}$, is zero below it and keeps its values
  (`kika/processing/tests/test_kinematic_thresholds.py::test_a_table_below_the_kinematic_threshold_is_raised_to_reconr_s_value`);
  overtaken points move up one place each
  (`::test_points_the_move_overtakes_are_pushed_up_one_place_each`); a table already at
  the threshold is left alone (`::test_a_table_already_at_reconr_s_threshold_is_left_alone`).
- **A PENDF NJOY can read.** Neighbouring regions of one law are written as one, state
  the same function, and are cut back at the original edges without inventing one
  (`kika/endf/model_adapter/tests/test_a_kika_pendf_is_readable_by_njoy.py`); the MF 1
  identification block is kept (`::test_a_pendf_keeps_the_identification_block_of_mf1`).
  These tests, with `kika/endf/tests/test_reconstruct_endf.py` and
  `kika/processing/tests/test_resonance_publication.py`, passed on 2026-10-14.
- **Parameters read by two parsers.** The 312 Fe-56 resonances read from ENDF-6 by kika
  and from GNDS written by FUDGE agree to $10^{-12}$
  (`kika/gnds/tests/test_resonance_oracle.py::test_both_encodings_hold_the_same_312_resonances`);
  this checks the input, not the cross-sections. `TODO(check)`: needs the tape and GNDS
  data (`--deep`); not re-run for this page.

### Comparison with NJOY and FUDGE

These comparisons are not tests of this repository: NJOY and FUDGE are neither
dependencies of kika nor the truth, and a test pinned to them would fail the day kika
improves on them. NJOY 2016.78 (Linux build), FUDGE 6.14.0
with its compiled extensions; the evaluations of ENDF/B-VIII.1. All three codes ran at
the same nominal tolerance: kika `rtol` $=10^{-3}$, `atol` $=10^{-8}$ b; NJOY `err` $=$
`errmax` $=10^{-3}$, `errint` $=10^{-8}$; FUDGE `accuracy` $=10^{-3}$ unless stated. The
metric is the ratio $r$ of {eq}`eq-resonance-reconstruction-budget` with that tolerance
(1 uses all of it), of each code's linear table against kika's exact evaluation of the
physics (direct per-level sums, not its table), at probes none of the codes chose: points
equally spaced in $\ln E$, the centre and flanks of every level, and the 16 worst nodes
of each code and reaction, MT 2, 18 and 102 in the resolved range, with a margin of
$10^{-6}$ relative around range boundaries (measured 2026-10-10).

| Material | kika | FUDGE | NJOY |
|---|---|---|---|
| Au-197 | 0.50 | 0.25 | 1.2 |
| Fe-56 | 0.50 | 13 | 1.3 |
| Pu-239 | 0.50 | 6.0 | 2.0 |
| U-235 | 0.50 | 7.4 | 1.6 |
| U-238 | 0.50 | 0.61 | 1.8 |
| Zr-90 | 0.50 | 0.55 | 1.5 |

kika's maxima sit at 0.5 because it tabulates to half the budget. The reference is
kika's physics, so the table answers only if the codes agree on the physics, which was
checked by evaluating each code's own kernel at the worst probes:

- **FUDGE, U-235.** Its kernel agrees with kika's to $5\cdot10^{-9}$ relative in elastic
  and $8\cdot10^{-9}$ in fission and capture, so the excess is tabulation: FUDGE checks
  each node against the chord to its neighbours and never returns to a reaction it marked
  converged. At `accuracy` $=10^{-4}$ it passes a set of 25 891 probes of the same kind
  (maximum 0.95, against kika's 0.50 at $10^{-3}$, measured 2026-10-09) but not an
  adversarial set of 26 044 energies around the worst panels (fission 1.6, capture 1.7, 21 and 22 probes above 1).
- **FUDGE, Fe-56.** At `accuracy` $=10^{-4}$ the ratio stays 13. Here the kernels
  themselves differ, by 1.3 % in elastic at the worst probe, in the deep interference
  minima near 73 keV and 219 keV, where kika and NJOY agree with each other to
  $5\cdot10^{-4}$. The cause in FUDGE is not established.
- **NJOY.** Its excesses (up to twice the budget, 99th percentile of the ratio below
  0.9) were not attributed to a single cause, since NJOY's kernel was not evaluated
  separately. Where a cause is known it is a convention named under *Conventions and
  choices* (range-boundary ramps, the NAPS=2 radius).

In time, FUDGE is faster than kika on every one of these materials (reconstruction only,
excluding its ENDF to GNDS translation, one thread on the same machine, measured
2026-10-10): from about 10 times on Zr-90 and Fe-56 to 1.2 times on U-235 and U-238. NJOY
(wall time including its MODER pass) is faster on Pu-239 and Zr-90, comparable on Fe-56,
and slower on U-235 and U-238. At equal achieved error on U-235 (FUDGE at
`accuracy` $=10^{-4}$), kika takes about 60 % of FUDGE's time. These times exclude kika's
verification, which on its own costs several times the reconstruction.

## Diagnostics

`EndfReconstruction.report` (the `report` of the `SuiteReconstructionResult`) holds:

- `regions`: one entry per segment, with `domain`, `points`, `iterations`, `evaluations`,
  `refinement_maxima` and `verification_maxima` per reaction (largest $r$ of
  {eq}`eq-resonance-reconstruction-budget` at the refinement and verification probes;
  the first should be $\le 0.5$, the second $\le 1$), `verification_worst` (energy, exact
  and tabulated values) and `integrals` (estimates of $\int\sigma\,dE$ and
  $\int\sigma\,dE/E$ by three-point Gauss-Legendre on each panel, with the integral of
  $\lvert\sigma - \sigma_\text{lin}\rvert$);
- `source_sum_error_ratios`, `source_sum_discrepancies`: how far each evaluated sum is
  from the sum of its leaves, and where. **A ratio above 1** means the evaluation's own
  MT 1 (or other sum) is inconsistent, and the output differs from it;
- `minima`: the smallest value per reaction; **a negative minimum** is a negative
  cross-section kept as computed;
- `preparation_notes`: what preparation assumed or corrected (a rounded GT, a declared
  AWRI, a zeroed negative competitive width);
- `raised_thresholds`: for each reaction whose first energy was moved to its kinematic
  threshold (step 3 of the assembly), the evaluated and the new first energy. An entry
  says the evaluation opens the reaction a few parts per million below the energy at
  which it can occur; it is a correction of the abscissa, not of the values;
- `source_conversion_report`: the conversion report of the source, whose losses would
  have blocked the reconstruction;
- `points`, `stored_points`, `domain`, `context`, `options`, `resonance_partial_owners`,
  `engine` (`kika-native-suite-2`; it names the reconstruction physics and changes
  whenever a change of kernel, grid or publication alters a reconstructed value, so a
  cache keyed on it is invalidated exactly then), `source_sha256`.

`report_document()` renders the same, with the tape's MF 1 and MF 2 flags, as a
dictionary, text or an HTML page (`to_dict`, `to_text`, `to_html`, `save`; the
dictionary's `schema` key names its version). It opens with a verdict that follows from
a list of checks, each `passed`, `note`, `warning`, `failed` or `not run`, and with the
meaning of each level printed under it:

| Check | What it compares | Should worry the user when |
|---|---|---|
| `tabulation` | the tables against the reference at four interior points per panel | `failed` (the run normally raises before) |
| `pendf` | the reloaded PENDF against the reference, at every node and four interior points | `failed` (the file is then not kept); `not run` without `write_pendf` |
| `source_sums` | each sum the evaluation stores against the sum of its parts | `warning`: the evaluation is inconsistent, and the reconstructed sum differs from the one it prints |
| `negatives` | the sign of every value | `warning`: negative values kept as computed |
| `grid` | ascending energies; a repeated energy only at a real jump | `note` for a jump of the evaluation's own MF 3 away from a range boundary; `failed` for decreasing energies |
| `integrals` | $\int\sigma\,dE$ of each table against an accurate estimate of the reference's | `note` when the absolute tolerance dominates a small reaction |
| `parameters` | the adjustments preparation made (a width negative by rounding, a block's own AWRI) | `note`, each adjustment listed |
| `conversion` | what reading the file lost or approximated in MF 2 and MF 3 | `warning`: the result may not reflect the evaluation there |

It then lists the findings, what was done, the resonance ranges (LRU, LRF, NRO, NAPS and
AP, shown in fm), one row per MT (points, jumps, finest $\Delta E/E$, negatives, worst
verification ratio, integral difference), the sum graph, the conversion notes and the
timings, with the engine's segments and iterations folded under the run details. A
caller of `PreparedResonances.evaluate` can also pass a `diagnostics` dictionary, which
collects the largest solve residual and flux error of each formalism.

## Implementation

- `kika.endf.reconstruction`: `reconstruct_endf` (read MF 1, 2, 3; decode; context;
  reconstruct), `EndfReconstruction` (`tables_by_mt`, `write_pendf`, `report_document`).
- `kika.endf.reconstruction_report`: the report document and its checks.
- `kika.endf.writers.assemble`: `writeReconstructedEndfTape` (verify, resample, write
  LRP=2, reload, verify again); `kika.endf.model_adapter.encode` merges the
  interpolation regions of one law, and `kika.endf.model_adapter.pendf` trims the MF 1
  text.
- `kika.processing.resonances.suite`: `reconstruct_suite`, `attach_reconstruction`,
  ownership, kinematic thresholds, sum graph, segments, verification.
- `kika.processing.resonances.prepare`, `prepare_reich_moore`, `prepare_r_matrix`: the
  checks and the normalisation of each formalism into immutable groups, radii and
  amplitudes; `radii`: radius functions.
- `kika.processing.resonances.breit_wigner`, `reich_moore`, `r_matrix`, `brune`: the
  formalisms; `channel_functions`, `coulomb`, `kinematics`: channel functions and
  kinematics.
- `kika.processing.resonances.assemble`, `tabulate`, `grid`: backgrounds, sums,
  refinement and the low-level `tabulate_resonances`.
- `kika.processing.resonances._rm_native` (C, built with the package) and
  `_rm_acceleration`: the compiled level sums for Reich-Moore, diagonal R-Matrix Limited
  and Breit-Wigner, and the unresolved integrand. They are used only for normal-range
  inputs (for instance at least 32 levels and at most 3 channels for Reich-Moore, at
  least 8 levels, $\ell \le 2$, a constant radius and no competition for Breit-Wigner),
  during tabulation only when `rtol` $\ge 10^{-3}$, and during verification in their
  exact form; otherwise, or when the extension is not loaded
  (`kika.processing.resonances.native_available`), the NumPy solver gives the same
  quantities. Nothing is compiled at run time.
- `kika.processing.resonances.parallel`: the verification physics on at most
  $\min(4, \text{cores}/2)$ threads at low priority, for large materials (more than
  $10^9$ energy-level products); results are bit-identical to the serial path.

## References

- D. A. Brown (ed.), *ENDF-6 Formats Manual*, Brookhaven National Laboratory (2023),
  doi:10.2172/2007538: Ch. 2 (MF 2, §2.2.1 NRO and NAPS), §1.1 (LRP), §3.4.5 (MT 19),
  App. D (the resonance formulas).
- R. E. MacFarlane, D. W. Muir, R. M. Boicourt, A. C. Kahler and J. L. Conlin, *The NJOY
  Nuclear Data Processing System, Version 2016*, Los Alamos National Laboratory (2017),
  doi:10.2172/1338791: RECONR.
- C. R. Brune, "Alternative parametrization of R-matrix theory", Phys. Rev. C 66, 044611
  (2002).
- *NIST Digital Library of Mathematical Functions*, <https://dlmf.nist.gov/>: §10.29,
  §13.2, §33.8.
- A. M. Lane and R. G. Thomas, "R-matrix theory of nuclear reactions", *Reviews of
  Modern Physics* 30, 257-353 (1958).
- C. W. Reich and M. S. Moore, "Multilevel formula for the fission process", *Physical
  Review* 111, 929-933 (1958).
