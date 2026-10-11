"""KERMA and damage energy from a processed suite (roadmap H8; replaces NJOY HEATR).

What OpenMC's files carry as MT301 (heating, photons transported), MT901
(heating with the photons deposited locally) and MT444 (damage energy), minus
fission, which the writer adds from MF1/458 as ``from_njoy`` does. The method
is HEATR's (``heatr.f90``; the reading is ``docs/library/heating_kerma_damage_plan.md``
in kika-workspace):

- **energy balance** (MT301): a reaction deposits ``E + Q_M − Σ_n y_n Ē_n −
  Σ_γ y_γ Ē_γ`` -- everything but the neutrons and photons it emits (HEATR's
  ``nheat`` and then ``gheat``, ``local = 0``). Capture is the exception: with
  its photons known, MT301 keeps only the recoil,
  ``E/(A+1) + Σ_γ y_γ ⟨E_γ²⟩/(2(A+1)m_nc²)``.
- **MT901** is MT301 plus the energy of the photons emitted, ``Σ_γ y_γ Ē_γ`` --
  what OpenMC's files carry (measured on Fe-56: at thermal 901 − 301 is the
  capture photons' 7.489 MeV, where HEATR's ``local = 1`` balance would give
  ``E + Q`` = 7.646 MeV).
- **damage**: the Lindhard–Robinson partition of each recoil (``df``) with
  HEATR's displacement threshold per Z, integrated the way HEATR does for each
  path: two-body kinematics over the angular distribution (``disbar``), one
  isotropic emission over the outgoing spectrum (``tabdam``), the photon
  recoils for capture (``gheat``), particle emission up to the Coulomb barrier
  for absorption (``capdam``).

The mean energies come from the processed suite's Monte Carlo tables -- the
distributions OpenMC samples -- and not from re-reading each ENDF law: one
integral over a tabulated density covers every MF4/MF5/MF6 law the
tabulation produced. HEATR's own interpolation between nodes (×1.1 in
``disbar``) is not copied; the moments are interpolated between the tables'
incident energies, lin-lin, as OpenMC interpolates the tables themselves.

Approximations kika keeps from HEATR are listed on :data:`HEATR_APPROXIMATIONS`
and reported. Where HEATR uses the explicit recoil and charged-particle
distributions of an MF6 section for the damage, kika -- whose processed suite
keeps only neutrons and photons, as OpenMC's files do -- uses HEATR's own
fallback for a section without them (one isotropic emission for neutrons,
``capdam`` for absorption), and says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["HeatingResult", "heatingOf", "displacementEnergy", "lindhard", "NEUTRON_REST_ENERGY"]

#: m_n c² in eV, from NJOY's constants (``phys.f90``: amassn·amu·c²/ev).
NEUTRON_REST_ENERGY = 1.00866491595 * 931.49410242e6
#: Mass ratios to the neutron of p, d, t, ³He, α (``phys.f90``).
PARTICLE_MASS_RATIO = {"p": 0.99862347812, "d": 1.99626900810, "t": 2.99071370040,
                       "h": 2.98931830190, "a": 3.97213205420}
#: Below this incident energy the elastic recoil cannot pass 25 eV and HEATR
#: writes no elastic damage (``disbar``, ``enx = 25·(A+1)²/(4A)``; 25 eV fixed).
ELASTIC_DAMAGE_THRESHOLD = 25.0
FISSION_MTS = (18, 19, 20, 21, 38)

HEATR_APPROXIMATIONS = (
    "damage of the continuum neutron emission: one isotropic emission in the lab, the "
    "target's Z and A for the recoil (HEATR's tabdam)",
    "capture: the recoil of each photon on its own, Σ y⟨E_γ²⟩/(2(A+1)m_nc²), without the "
    "compound nucleus' recoil E/(A+1), in the KERMA and in the damage (HEATR's output)",
    "damage of particle emission: capdam's single particle at min(Q + AE/(A+1), Coulomb "
    "barrier), deuterons and alphas of a continuum taken as protons",
)

_GL4 = np.polynomial.legendre.leggauss(4)
_GL8 = np.polynomial.legendre.leggauss(8)
_GL64 = np.polynomial.legendre.leggauss(64)


@dataclass
class HeatingResult:
    """KERMA factors in eV·b on ``grid``, fission excluded."""

    grid: np.ndarray
    #: MT301 without fission: the photons carried away (``local = 0``).
    kerma: np.ndarray
    #: MT901 without fission: the photons deposited (``local = 1``).
    kermaLocal: np.ndarray
    #: MT444 (``local = 0``, the one OpenMC's files carry).
    damage: np.ndarray
    notes: List[str] = field(default_factory=list)
    #: ``mt -> (kerma, kermaLocal, damage)`` per reaction, eV per reaction
    #: (multiply by its cross section for its share of the KERMA).
    perReaction: Dict[int, tuple] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Lindhard
# ----------------------------------------------------------------------

def displacementEnergy(Z: int) -> float:
    """HEATR's default displacement threshold ``ed`` in eV (``heatr.f90:329-366``)."""
    table = {4: 31, 6: 31, 12: 25, 13: 27, 14: 25, 20: 40, 40: 40, 41: 40, 42: 60, 47: 60,
             73: 90, 74: 90, 79: 30, 82: 25}
    if 22 <= Z <= 29:
        return 40.0
    return float(table.get(int(Z), 25))


def lindhard(E, Zr, Ar, Zl, Al, ed):
    """Damage energy of a recoil ``(Zr, Ar)`` of energy ``E`` in a lattice ``(Zl, Al)`` (``df``)."""
    E = np.asarray(E, dtype=float)
    if Zr == 0:
        return np.zeros_like(E)
    el = 30.724 * Zr * Zl * np.sqrt(Zr ** (2 / 3) + Zl ** (2 / 3)) * (Ar + Al) / Al
    denom = (Zr ** (2 / 3) + Zl ** (2 / 3)) ** 0.75 * Ar ** 1.5 * np.sqrt(Al)
    fl = 0.0793 * Zr ** (2 / 3) * np.sqrt(Zl) * (Ar + Al) ** 1.5 / denom
    ep = np.maximum(E, 0.0) / el
    out = E / (1.0 + fl * (3.4008 * ep ** (1 / 6) + 0.40244 * ep ** 0.75 + ep))
    return np.where(E >= ed, out, 0.0)


# ----------------------------------------------------------------------
# Expectations over a tabulated density
# ----------------------------------------------------------------------

def _isFlat(table) -> bool:
    from kika.nuclear_data.model import Interpolation

    return table.interpolation == Interpolation.flat


def _expect(table, g: Callable[[np.ndarray], np.ndarray]) -> float:
    """``∫ g(x) p(x) dx`` over an ``Xs_pdf_cdf1d`` (lines first, then the continuum), normalised."""
    x = np.asarray(table.xs, dtype=float)
    p = np.asarray(table.pdf, dtype=float)
    nd = int(getattr(table, "numberOfDiscrete", 0) or 0)
    total = 0.0
    value = 0.0
    if nd:
        w = p[:nd]
        total += float(w.sum())
        value += float(np.sum(w * g(x[:nd])))
    cx, cp = x[nd:], p[nd:]
    if len(cx) > 1:
        a, b = cx[:-1], cx[1:]
        nodes, weights = _GL4
        half = 0.5 * (b - a)
        mid = 0.5 * (b + a)
        pts = mid[:, None] + half[:, None] * nodes[None, :]
        if _isFlat(table):
            dens = np.broadcast_to(cp[:-1, None], pts.shape)
        else:
            slope = np.where(b > a, (cp[1:] - cp[:-1]) / np.where(b > a, b - a, 1.0), 0.0)
            dens = cp[:-1, None] + slope[:, None] * (pts - a[:, None])
        w = half[:, None] * weights[None, :] * dens
        total += float(w.sum())
        value += float(np.sum(w * g(pts)))
    return value / total if total > 0 else 0.0


def _pdfAt(table, x):
    """The continuum density of a table at ``x`` (0 outside)."""
    xs = np.asarray(table.xs, dtype=float)
    p = np.asarray(table.pdf, dtype=float)
    nd = int(getattr(table, "numberOfDiscrete", 0) or 0)
    xs, p = xs[nd:], p[nd:]
    if _isFlat(table):
        i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 1)
        out = p[i]
    else:
        out = np.interp(x, xs, p)
    return np.where((x >= xs[0]) & (x <= xs[-1]), out, 0.0)


def _tables(form):
    """``(incident energies, tables, flat)`` of an XYs2d or Regions2d of tables."""
    from kika.nuclear_data.model import Interpolation

    regions = getattr(form, "function2ds", None)
    if regions is None:
        tables = list(form.function1ds)
        flat = getattr(form, "interpolation", None) == Interpolation.flat
    else:
        tables = [t for region in regions for t in region.function1ds]
        flat = all(getattr(r, "interpolation", None) == Interpolation.flat for r in regions)
    energies = np.array([float(t.outerDomainValue) for t in tables])
    return energies, tables, flat


def _onGrid(energies, values, grid, flat=False):
    """Moments known at the tables' incident energies, read at ``grid`` (clamped at the ends)."""
    values = np.asarray(values, dtype=float)
    if len(energies) == 1:
        return np.full(len(grid), values[0])
    if flat:
        i = np.clip(np.searchsorted(energies, grid, side="right") - 1, 0, len(energies) - 1)
        return values[i]
    return np.interp(grid, energies, values)


# ----------------------------------------------------------------------
# Mean outgoing energies of a product
# ----------------------------------------------------------------------

@dataclass
class _Kinematics:
    A: float               # target mass ratio
    Q: float               # the reaction's QI
    awp: float = 1.0       # emitted particle's mass ratio


def _cmToLab(grid, ecm, sqrtMu, k: _Kinematics):
    """``⟨E'_lab⟩ = ⟨E'_cm⟩ + x E + 2√(xE)⟨√E'_cm μ⟩``, ``x = awp/(A+1)²``."""
    x = k.awp / (k.A + 1.0) ** 2
    return ecm + x * grid + 2.0 * np.sqrt(x * grid) * sqrtMu


def _meanMu(form, grid):
    if form is None or type(form).__name__ == "Isotropic2d":
        return np.zeros(len(grid))
    energies, tables, flat = _tables(form)
    return _onGrid(energies, [_expect(t, lambda m: m) for t in tables], grid, flat)


def _meanEnergyOf(form, grid, k: _Kinematics, moment=1):
    """``⟨E'^moment⟩`` in the lab of one Monte Carlo form, on ``grid``."""
    from kika.nuclear_data import model as m

    if isinstance(form, m.DistributionMixture):
        out = np.zeros(len(grid))
        for term in form.weighted:
            weight = (np.asarray(term.weight.evaluate(grid), dtype=float)
                      if term.weight is not None else 1.0)
            out = out + weight * _meanEnergyOf(term.functional, grid, k, moment)
        return out
    frame = getattr(form, "productFrame", None)
    cm = frame is not None and str(getattr(frame, "value", frame)) in ("centerOfMass", "Frame.centerOfMass")
    if isinstance(form, m.AngularTwoBody):
        # Elastic or a level, two-body in the centre of mass.
        thr = (k.A + 1.0) / k.A * max(-k.Q, 0.0)
        r2 = np.clip(1.0 - thr / np.maximum(grid, 1e-300), 0.0, None)
        ecm = (k.A / (k.A + 1.0)) ** 2 * grid * r2
        mu = _meanMu(form.angular, grid)
        if moment != 1:
            raise NotImplementedError("second moment of a two-body neutron")
        return _cmToLab(grid, ecm, np.sqrt(ecm) * mu, k)
    if isinstance(form, m.Uncorrelated):
        energy = form.energy
        if isinstance(energy, m.DiscretePhotonEnergy):
            return np.asarray(energy.outgoingEnergy(grid), dtype=float) ** moment
        if isinstance(energy, m.LevelEnergy):
            ecm = np.clip(energy.outgoingEnergy(grid), 0.0, None)
            mu = _meanMu(form.angular, grid)
            return _cmToLab(grid, ecm, np.sqrt(ecm) * mu, k)
        if isinstance(energy, m.NBodyPhaseSpace):
            n = int(energy.numberOfProducts)
            from kika._constants import NEUTRON_MASS_AMU

            Ap = float(energy.mass.value) / NEUTRON_MASS_AMU
            emax = (Ap - k.awp) / Ap * (k.A / (k.A + 1.0) * grid + k.Q)
            p = 1.5 * n - 4.0
            ecm = np.clip(emax, 0.0, None) * 1.5 / (p + 2.5)
            return _cmToLab(grid, ecm, 0.0, k)
        if energy is None:
            raise ValueError("an uncorrelated form without an energy distribution")
        energies, tables, flat = _tables(energy)
        mean = _onGrid(energies, [_expect(t, lambda e: e ** moment) for t in tables], grid, flat)
        if not cm or k.awp == 0.0:
            return mean
        if moment != 1:
            raise NotImplementedError("second moment of a centre-of-mass spectrum")
        root = _onGrid(energies, [_expect(t, np.sqrt) for t in tables], grid, flat)
        return _cmToLab(grid, mean, root * _meanMu(form.angular, grid), k)
    if isinstance(form, m.EnergyAngularMC):
        energies, tables, flat = _tables(form.energy)
        mean = _onGrid(energies, [_expect(t, lambda e: e ** moment) for t in tables], grid, flat)
        if not cm:
            return mean
        inner = list(form.energyAngular.function2ds)
        cross = []
        for t, mus in zip(tables, inner):
            mubar = np.array([_expect(u, lambda x: x) for u in mus.function1ds])
            nodes = np.asarray(t.xs, dtype=float)
            cross.append(_expect(t, lambda e, nodes=nodes, mubar=mubar:
                                 np.sqrt(e) * np.interp(e, nodes, mubar)))
        return _cmToLab(grid, mean, _onGrid(energies, cross, grid, flat), k)
    if isinstance(form, m.KalbachMann):
        energies, tables, flat = _tables(form.f)
        mean = _onGrid(energies, [_expect(t, lambda e: e ** moment) for t in tables], grid, flat)
        rs = list(m.toEndfTab2(form.r)[0])
        As = list(m.toEndfTab2(form.a)[0])
        cross = []
        for t, r, a in zip(tables, rs, As):
            def g(e, r=r, a=a):
                rr = np.interp(e, r.xs, r.ys)
                aa = np.interp(e, a.xs, a.ys)
                safe = np.where(np.abs(aa) > 1e-6, aa, 1.0)
                mu = np.where(np.abs(aa) > 1e-6, 1.0 / np.tanh(safe) - 1.0 / safe, aa / 3.0)
                return np.sqrt(e) * rr * mu
            cross.append(_expect(t, g))
        return _cmToLab(grid, mean, _onGrid(energies, cross, grid, flat), k)
    raise ValueError(f"no mean energy for a {type(form).__name__}")


#: The style label of the processed (sampled) forms.
MONTE_CARLO = "MonteCarlo"


def _yield(product, grid):
    """The product's multiplicity on ``grid``: its Monte Carlo form, else the evaluated one."""
    multiplicity = product.multiplicity
    if multiplicity is None:
        return np.zeros(len(grid))
    function = multiplicity.forms.get(MONTE_CARLO) or multiplicity.form
    if function is None:
        return np.zeros(len(grid))
    return np.asarray(function.evaluate(grid), dtype=float)


def _distribution(product):
    component = product.distribution
    if component is None or MONTE_CARLO not in component.forms:
        return None
    return component.forms[MONTE_CARLO]


# ----------------------------------------------------------------------
# Damage per path
# ----------------------------------------------------------------------

def _twoBodyDamage(form, grid, A, Q, Z, ed, elastic):
    """``disbar``: ``E_R(μ) = E·A/(A+1)²·(1 − 2rμ + r²)`` over the angular tables, GL-64."""
    angular = form.angular if hasattr(form, "angular") else None
    thr = (A + 1.0) / A * max(-Q, 0.0)
    nodesMu, weightsMu = _GL64
    if angular is None or type(angular).__name__ == "Isotropic2d":
        energies, pdfs = np.array([grid[0], grid[-1]]), [np.full(64, 0.5)] * 2
    else:
        energies, tables, _ = _tables(angular)
        pdfs = [_pdfAt(t, nodesMu) for t in tables]
        pdfs = [p / max(float(np.sum(weightsMu * p)), 1e-300) for p in pdfs]
    nodes = np.unique(np.concatenate([energies, _geometric(grid[0], grid[-1])]))
    nodes = nodes[(nodes >= grid[0]) & (nodes <= grid[-1])]
    j = np.clip(np.searchsorted(energies, nodes, side="right") - 1, 0, len(energies) - 1)
    jn = np.minimum(j + 1, len(energies) - 1)
    span = energies[jn] - energies[j]
    w = np.where(span > 0, (nodes - energies[j]) / np.where(span > 0, span, 1.0), 0.0)
    w = np.clip(w, 0.0, 1.0)
    pdf = np.array([(1 - wi) * pdfs[a] + wi * pdfs[b] for wi, a, b in zip(w, j, jn)])
    r2 = np.clip(1.0 - thr / np.maximum(nodes, 1e-300), 0.0, None)
    r = np.sqrt(r2)
    er = nodes[:, None] * A / (A + 1.0) ** 2 * (1.0 - 2.0 * r[:, None] * nodesMu[None, :] + r2[:, None])
    d = np.sum(weightsMu[None, :] * pdf * lindhard(er, Z, A, Z, A, ed), axis=1)
    d = np.where(r2 > 0, np.maximum(d, 0.0), 0.0)
    if elastic:
        d = np.where(nodes >= ELASTIC_DAMAGE_THRESHOLD * (A + 1.0) ** 2 / (4.0 * A), d, 0.0)
    return np.interp(grid, nodes, d)


def _geometric(lo, hi, step=1.1):
    lo = max(float(lo), 1.0e-5)
    n = int(np.ceil(np.log(hi / lo) / np.log(step))) + 1
    return lo * step ** np.arange(n)


def _emissionDamage(form, grid, A, Z, ed):
    """``tabdam``: one isotropic emission, ``E_R = (E − 2√(EE')μ + E')/A``, per outgoing table."""
    from kika.nuclear_data import model as m

    if isinstance(form, m.DistributionMixture):
        out = np.zeros(len(grid))
        for term in form.weighted:
            weight = (np.asarray(term.weight.evaluate(grid), dtype=float)
                      if term.weight is not None else 1.0)
            out = out + weight * _emissionDamage(term.functional, grid, A, Z, ed)
        return out
    if isinstance(form, m.Uncorrelated):
        spectrum = form.energy
    elif isinstance(form, m.EnergyAngularMC):
        spectrum = form.energy
    elif isinstance(form, m.KalbachMann):
        spectrum = form.f
    else:
        return np.zeros(len(grid))
    if spectrum is None or isinstance(spectrum, (m.LevelEnergy, m.DiscretePhotonEnergy,
                                                 m.NBodyPhaseSpace)):
        return np.zeros(len(grid))
    energies, tables, flat = _tables(spectrum)
    u, wu = _GL4
    values = []
    for E, t in zip(energies, tables):
        def g(ep, E=E):
            er = (E - 2.0 * np.sqrt(E * ep)[..., None] * u + ep[..., None]) / A
            return np.sum(wu * lindhard(er, Z, A, Z, A, ed), axis=-1) / 2.0
        values.append(_expect(t, g))
    return _onGrid(energies, values, grid, flat)


def _captureDamage(photons, grid, A, Z, ed, Q):
    """Capture: ``Σ_γ y_γ ⟨df(E_γ²/2tm)⟩`` (``gheat``), or ``capdam``'s single photon."""
    tm = NEUTRON_REST_ENERGY * (A + 1.0)
    if not photons:
        er = grid / (A + 1.0)
        ea = A * er + Q
        return lindhard(er + ea * ea / (2.0 * tm), Z, A + 1.0, Z, A, ed)
    total = np.zeros(len(grid))
    for product, form in photons:
        y = _yield(product, grid)
        total = total + y * _photonRecoilDamage(form, grid, A, Z, ed, tm)
    return total


def _photonRecoilDamage(form, grid, A, Z, ed, tm):
    from kika.nuclear_data import model as m

    if isinstance(form, m.DistributionMixture):
        out = np.zeros(len(grid))
        for term in form.weighted:
            weight = (np.asarray(term.weight.evaluate(grid), dtype=float)
                      if term.weight is not None else 1.0)
            out = out + weight * _photonRecoilDamage(term.functional, grid, A, Z, ed, tm)
        return out
    energy = form.energy
    if isinstance(energy, m.DiscretePhotonEnergy):
        eg = np.asarray(energy.outgoingEnergy(grid), dtype=float)
        return lindhard(eg * eg / (2.0 * tm), Z, A + 1.0, Z, A, ed)
    energies, tables, flat = _tables(energy)
    values = [_expect(t, lambda e: lindhard(e * e / (2.0 * tm), Z, A + 1.0, Z, A, ed))
              for t in tables]
    return _onGrid(energies, values, grid, flat)


def _absorptionDamage(mt, grid, A, Z, ed, Q):
    """``capdam`` for a reaction that emits a charged particle and no neutron."""
    zx, ax = 1.0, 1.0
    if mt == 104:
        ax = 2.0
    elif mt == 105:
        ax = 3.0
    elif mt == 106:
        zx, ax = 2.0, 3.0
    elif mt == 107:
        zx, ax = 2.0, 4.0
    ec = 1.029e6 * zx * Z / (ax ** (1 / 3) + A ** (1 / 3))
    ea = Q + A * grid / (A + 1.0)
    et = (A + 1.0 - ax) * grid / (A + 1.0)
    ea_c = np.minimum(ea, ec)
    u, wu = _GL4
    er = (et[:, None] - 2.0 * np.sqrt(np.clip(et * ax * ea_c, 0.0, None))[:, None] * u[None, :]
          + ax * ea_c[:, None]) / (A + 1.0)
    d = np.sum(wu[None, :] * lindhard(er, Z - zx, A + 1.0 - ax, Z, A, ed), axis=1) / 2.0
    return np.where(ea >= 0.0, d, 0.0)


# ----------------------------------------------------------------------
# Charged particles and recoils, from the evaluation
# ----------------------------------------------------------------------

def _massRatio(pid, pops):
    from kika._constants import NEUTRON_MASS_AMU

    light = {"H1": "p", "H2": "d", "H3": "t", "He3": "h", "He4": "a"}
    if pid in light:
        return PARTICLE_MASS_RATIO[light[pid]]
    if pops is not None and pid in pops:
        mass = getattr(pops[pid], "mass", None)
        if mass is not None and getattr(mass, "value", None) is not None:
            return float(mass.convertedTo("amu").value) / NEUTRON_MASS_AMU
    from kika.nuclear_data.model import zaFromPid

    return float(zaFromPid(pid) % 1000) * 0.99167


def _heavyProducts(reaction, evaluation, grid, A, Z, zaTarget, ed, notes):
    """``(deposit, damage)`` of a reaction's charged particles and recoils.

    What HEATR takes from an MF6 section's ``izap > 1`` subsections: each
    particle's mean lab energy and its own Lindhard damage. ``None`` when the
    evaluation states none of them with a distribution.
    """
    from kika.nuclear_data.model import zaFromPid
    from kika.processing.tabulation import TabulationContext, tabulateDistribution

    source = evaluation.findReactionByENDF_MT(reaction.ENDF_MT) if evaluation is not None else None
    if source is None:
        return None
    Q = float(source.outputChannel.Q.value or 0.0)
    deposit = np.zeros(len(grid))
    damage = np.zeros(len(grid))
    found = False
    for product in source.outputChannel.products:
        if product.pid in ("n", "photon") or product.distribution is None:
            continue
        forms = product.distribution.forms
        form = forms.get("eval") or (next(iter(forms.values())) if forms else None)
        multiplicity = product.multiplicity.form if product.multiplicity is not None else None
        if form is None or multiplicity is None:
            continue
        za = zaFromPid(product.pid)
        context = TabulationContext(awr=A, zaTarget=zaTarget, mt=reaction.ENDF_MT, Q=Q,
                                    pid=product.pid, zaEmitted=za)
        try:
            tabulated = tabulateDistribution(form, context)
        except (TypeError, ValueError, NotImplementedError) as exc:
            notes.append(f"MT{reaction.ENDF_MT} {product.pid}: {type(form).__name__} not "
                         f"tabulated for the heating ({exc}); left out")
            continue
        if tabulated is None:
            continue
        awp = _massRatio(product.pid, evaluation.PoPs)
        k = _Kinematics(A=A, Q=Q, awp=awp)
        y = np.asarray(multiplicity.evaluate(grid), dtype=float)
        try:
            mean = _meanEnergyOf(tabulated, grid, k)
        except (ValueError, NotImplementedError) as exc:
            notes.append(f"MT{reaction.ENDF_MT} {product.pid}: no mean energy ({exc}); left out")
            continue
        found = True
        deposit = deposit + y * mean
        damage = damage + y * _particleDamage(tabulated, grid, k, za // 1000, awp, Z, A, ed)
    return (deposit, damage) if found else None


def _particleDamage(form, grid, k, zp, ap, Z, A, ed):
    """``⟨df(E'_lab)⟩`` of a particle ``(zp, ap)`` in the lattice; a centre-of-mass
    spectrum taken to the lab over an isotropic centre-of-mass angle."""
    from kika.nuclear_data import model as m

    if isinstance(form, m.DistributionMixture):
        out = np.zeros(len(grid))
        for term in form.weighted:
            weight = (np.asarray(term.weight.evaluate(grid), dtype=float)
                      if term.weight is not None else 1.0)
            out = out + weight * _particleDamage(term.functional, grid, k, zp, ap, Z, A, ed)
        return out
    spectrum = form.f if isinstance(form, m.KalbachMann) else getattr(form, "energy", None)
    if spectrum is None or isinstance(spectrum, (m.LevelEnergy, m.DiscretePhotonEnergy,
                                                 m.NBodyPhaseSpace)):
        return lindhard(_meanEnergyOf(form, grid, k), zp, ap, Z, A, ed)
    frame = getattr(form, "productFrame", None)
    cm = str(getattr(frame, "value", frame)) in ("centerOfMass", "Frame.centerOfMass")
    x = k.awp / (k.A + 1.0) ** 2
    energies, tables, flat = _tables(spectrum)
    u, wu = _GL8

    def labDamage(e, E):
        if not cm:
            return lindhard(e, zp, ap, Z, A, ed)
        # E'_lab = E'_cm + xE + 2√(xE·E'_cm)·μ, averaged over an isotropic μ.
        lab = (e[..., None] + x * E + 2.0 * np.sqrt(x * E * e)[..., None] * u)
        return np.sum(wu * lindhard(lab, zp, ap, Z, A, ed), axis=-1) / 2.0

    values = [_expect(t, lambda e, E=E: labDamage(e, E)) for E, t in zip(energies, tables)]
    return _onGrid(energies, values, grid, flat)


# ----------------------------------------------------------------------
# The whole suite
# ----------------------------------------------------------------------

def heatingOf(suite, grid, sigma: Dict[int, np.ndarray], *, awr: float, Z: int,
              qm: Optional[Dict[int, float]] = None, ed: Optional[float] = None,
              evaluation=None, zaTarget: Optional[int] = None,
              report=None) -> HeatingResult:
    """KERMA (local and not) and damage of a processed suite on ``grid``.

    ``suite`` is the processed suite (:func:`kika.hdf5.processing.processSuite`
    builds it): exclusive reactions with their Monte Carlo product
    distributions. ``sigma[mt]`` is each reaction's cross section on ``grid``
    at the temperature. ``qm[mt]`` is the evaluation's ground-state Q (ENDF
    QM; default the reaction's Q). ``ed`` defaults to HEATR's table.
    ``evaluation`` is the evaluated suite the processed one came from: its
    charged particles and recoils (which the processed suite, like OpenMC's
    files, does not keep) give MT5's deposit and the damage of every reaction
    that states them, as HEATR takes them from MF6.
    """
    grid = np.asarray(grid, dtype=float)
    A = float(awr)
    ed = displacementEnergy(Z) if not ed else float(ed)
    qm = qm or {}
    tm = NEUTRON_REST_ENERGY * (A + 1.0)
    perReaction: Dict[int, tuple] = {}
    kerma = np.zeros(len(grid))
    local = np.zeros(len(grid))
    damage = np.zeros(len(grid))
    notes: List[str] = [f"heating by kika (energy balance on the Monte Carlo tables; "
                        f"damage with ed = {ed:g} eV)"] + [f"as HEATR: {a}" for a in HEATR_APPROXIMATIONS]
    for reaction in suite.reactions:
        mt = reaction.ENDF_MT
        if mt is None or mt in FISSION_MTS or mt not in sigma:
            continue
        s = np.asarray(sigma[mt], dtype=float)
        Q = float(reaction.outputChannel.Q.value or 0.0)
        Qbal = float(qm.get(mt, Q))
        k = _Kinematics(A=A, Q=Q)
        # A photon has no mass: a centre-of-mass photon spectrum is its lab one.
        photon = _Kinematics(A=A, Q=Q, awp=0.0)
        neutrons, photons = [], []
        for product in reaction.outputChannel.products:
            if product.pid not in ("n", "photon"):
                continue
            form = _distribution(product)
            if form is None:
                continue
            (neutrons if product.pid == "n" else photons).append((product, form))
        emitted = np.zeros(len(grid))
        for product, form in neutrons:
            emitted = emitted + _yield(product, grid) * _meanEnergyOf(form, grid, k)
        deposit = grid + Qbal - emitted
        gammas = np.zeros(len(grid))
        for product, form in photons:
            gammas = gammas + _yield(product, grid) * _meanEnergyOf(form, grid, photon)
        heavy = (_heavyProducts(reaction, evaluation, grid, A, Z, zaTarget or 1000 * Z, ed, notes)
                 if evaluation is not None and mt not in (2, 102) and not 51 <= mt <= 90 else None)
        if mt == 5:
            # HEATR closes no energy balance on MT5 (ebal6 = 0, heatr.f90:1446-1461):
            # it deposits what the charged particles and recoils carry.
            if heavy is None:
                notes.append("MT5: no charged particles or recoils with distributions; "
                             "its deposit is left at zero, as HEATR leaves it")
            # ``deposit`` is with the photons in, as everywhere else.
            deposit = (heavy[0] if heavy is not None else np.zeros(len(grid))) + gammas
        if mt == 102 and photons:
            second = np.zeros(len(grid))
            for product, form in photons:
                second = second + _yield(product, grid) * _meanEnergyOf(form, grid, photon, moment=2)
            # HEATR's output keeps only the photons' recoil: at Fe-56's capture
            # resonances (1.15, 73, 106 keV) the official MT301 is short of
            # E/(A+1)·σ_γ to the eV, though gheat's code reads as if it kept it.
            carried = second / (2.0 * tm)
        else:
            carried = deposit - gammas
        live = s > 0
        kerma += np.where(live, s * carried, 0.0)
        # MT901 is MT301 plus the energy of the photons the reaction emits, as
        # OpenMC's files carry it (Fe-56 capture at 0.0253 eV: 901 - 301 is the
        # photons' 7.489 MeV, not HEATR's local=1 balance E + Q = 7.646 MeV).
        # It differs from the balance only where the balance is not used: capture.
        local += np.where(live, s * (carried + gammas), 0.0)
        # damage
        if mt == 2 or (51 <= mt <= 90 and neutrons):
            d = _twoBodyDamage(neutrons[0][1], grid, A, Q, Z, ed, elastic=(mt == 2)) if neutrons \
                else np.zeros(len(grid))
        elif mt == 102:
            d = _captureDamage(photons, grid, A, Z, ed, Q)
        elif heavy is not None:
            d = heavy[1]
        elif neutrons:
            d = np.zeros(len(grid))
            for product, form in neutrons[:1]:
                d = _emissionDamage(form, grid, A, Z, ed)
        elif mt > 102:
            d = _absorptionDamage(mt, grid, A, Z, ed, Q)
        else:
            d = np.zeros(len(grid))
        damage += np.where(live, s * d, 0.0)
        perReaction[mt] = (carried, carried + gammas, d)
    # Photons stated on a sum (MF13 on MT4, 103-107): carried away for MT301 and
    # deposited for MT901, as HEATR's gheat does with σ_sum·y·Ē (= σ_γ·Ē).
    for reaction in getattr(suite, "sums", ()):
        mt = reaction.ENDF_MT
        if mt is None or mt not in sigma:
            continue
        photons = [(p, _distribution(p)) for p in reaction.outputChannel.products
                   if p.pid == "photon"]
        photons = [(p, f) for p, f in photons if f is not None]
        if not photons:
            continue
        s = np.asarray(sigma[mt], dtype=float)
        photon = _Kinematics(A=A, Q=0.0, awp=0.0)
        gammas = np.zeros(len(grid))
        for product, form in photons:
            gammas = gammas + _yield(product, grid) * _meanEnergyOf(form, grid, photon)
        kerma -= np.where(s > 0, s * gammas, 0.0)
        notes.append(f"MT{mt}: {len(photons)} photons stated on the sum, taken out of MT301")
    if report is not None:
        for note in notes:
            report.warn(note)
    return HeatingResult(grid=grid, kerma=kerma, kermaLocal=local, damage=damage, notes=notes,
                         perReaction=perReaction)
