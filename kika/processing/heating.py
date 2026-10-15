"""KERMA and damage energy from a processed suite (replaces NJOY HEATR).

What OpenMC's files carry as MT301 (heating, photons transported), MT901
(heating with the photons deposited locally) and MT444 (damage energy), minus
fission, which the writer adds from MF1/458 as ``from_njoy`` does. The method
is HEATR's (``heatr.f90``):

- **energy balance** (MT301): a reaction deposits ``E + Q_M − Σ_n y_n Ē_n −
  Σ_γ y_γ Ē_γ`` -- everything but the neutrons and photons it emits (HEATR's
  ``nheat`` and then ``gheat``, ``local = 0``). Capture is the exception: with
  its photons known, MT301 follows the stated capture branch: independent
  MF6 cascade-photon kicks, or for MF12 and an explicit two-body primary
  photon the compound recoil ``E/(A+1)`` plus the stated photon kicks.
  MF12 primary-line kicks use their fixed binding energy, as gheat does.
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

The mean energies integrate the processed suite's Monte Carlo densities exactly
for polynomial moments on flat/linear panels (including MF6), from the
distributions OpenMC samples, rather than from re-reading each ENDF law: one
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
    "MF6 capture cascades: the recoil of each photon on its own, Σ y⟨E_γ²⟩/(2(A+1)m_nc²), without the "
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
    #: Photon energy per reaction, including photons declared on redundant sums.
    photonEnergy: Dict[int, np.ndarray] = field(default_factory=dict)

    #: HEATR diagnostic upper energy per collision; absent for unresolved recoils.
    kinematicUpper: Dict[int, np.ndarray] = field(default_factory=dict)

    def partialsOf(self, sigma, requested, *, fissionKerma=None,
                   fissionPhotonEnergy=None, fissionKinematicEnergy=None, local=False):
        """Requested heating/damage responses in eV·b, using already computed moments.

        ENDF-6 HEATR's selection rules apply, including charged-particle level
        families in 401/403–407/447. Photon corrections declared on sums are
        counted once. ``local`` deposits transported photons as well.

        ``fissionKerma`` supplies independently calculated fission energy per
        collision, keyed by the exclusive fission MTs. It is required for a
        heating selection containing active fission; this class deliberately
        does not fabricate release energies from neutron spectra. MT443 uses
        HEATR's diagnostic convention (not a rigorous physical upper bound).
        Undetermined active MF6 recoils raise rather than supplying a zero;
        fission requires ``fissionKinematicEnergy``. MT442 needs
        ``fissionPhotonEnergy`` on active fission channels. When ``local`` is
        true, the supplied fissionKerma must itself include local photon energy.
        """
        ids = tuple(requested)
        if any(isinstance(mt, (bool, np.bool_)) or not isinstance(mt, (int, np.integer))
               for mt in ids):
            raise ValueError("partial response identifiers must be integers")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate partial response identifiers")
        allowed = {301, 303, 304, 318, 401, 442, 443, 444, 445, 446, 447, 901}
        allowed.update(300 + mt for mt in self.perReaction if 1 < mt <= 141)
        allowed.update(range(403, 408))
        unknown = set(ids) - allowed
        if unknown:
            raise ValueError(f"unsupported partial responses: {sorted(unknown)}")
        cross = {}
        for mt in set(self.perReaction) | set(self.photonEnergy):
            if mt not in sigma:
                raise ValueError(f"missing cross section for MT{mt}")
            values = np.asarray(sigma[mt], dtype=float)
            if values.shape != self.grid.shape or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"invalid cross section for MT{mt}")
            cross[mt] = values
        fissionKerma = fissionKerma or {}
        fissionPhotonEnergy = fissionPhotonEnergy or {}
        fissionKinematicEnergy = fissionKinematicEnergy or {}

        def selected(response, mt):
            if response in (301, 901, 442, 443, 444):
                return True
            if response == 303:
                return mt not in (2, 3)
            if response in (304, 446):
                return 51 <= mt <= 91
            if response == 318:
                return mt in FISSION_MTS
            if response in (401, 447):
                return 102 <= mt <= 120 or 600 <= mt <= 849
            if response == 445:
                return mt == 2
            if 403 <= response <= 407:
                first = 600 + 50 * (response - 403)
                return mt == response - 300 or first <= mt <= first + 49
            return mt == response - 300

        output = {}
        for response in ids:
            if response == 444:
                output[response] = self.damage.copy()
                continue
            values = np.zeros_like(self.grid)
            damage = response in (445, 446, 447)
            localResponse = local or response == 901
            for mt, moments in self.perReaction.items():
                if not selected(response, mt):
                    continue
                if response == 442:
                    continue
                if response == 443:
                    if not np.any(cross[mt] > 0):
                        continue
                    bounds = fissionKinematicEnergy if mt in FISSION_MTS else self.kinematicUpper
                    if mt not in bounds:
                        raise NotImplementedError(f"MT443 has no determined recoil energy for MT{mt}")
                    energy = np.asarray(bounds[mt], dtype=float)
                    if energy.shape != self.grid.shape or not np.all(np.isfinite(energy)):
                        raise ValueError(f"invalid kinematic energy for MT{mt}")
                elif damage:
                    energy = moments[2]
                elif mt in FISSION_MTS:
                    if not np.any(cross[mt] > 0):
                        continue
                    if mt not in fissionKerma:
                        raise ValueError(f"fission heating requires release energy for MT{mt}")
                    energy = np.asarray(fissionKerma[mt], dtype=float)
                    if energy.shape != self.grid.shape or not np.all(np.isfinite(energy)):
                        raise ValueError(f"invalid fission release energy for MT{mt}")
                else:
                    energy = moments[1 if localResponse else 0]
                values += cross[mt] * energy
            # Per-reaction moments already subtract their own photons. Sum
            # photons, absent from perReaction, must be subtracted separately.
            for mt, energy in (() if response == 443 else self.photonEnergy.items()):
                # HEATR gheat maps MT4 photon production onto MT51 for
                # partial selection, and MT3 onto the total. These are
                # photon-production labels, never extra neutron reactions.
                selectionMt = (51 if mt == 4 else 1 if mt == 3 else mt)
                if not selected(response, selectionMt):
                    continue
                if response == 442:
                    values += cross[mt] * energy
                elif not damage and not localResponse and mt not in self.perReaction:
                    values -= cross[mt] * energy
            if response == 442:
                for mt in self.perReaction:
                    if mt not in FISSION_MTS or not np.any(cross[mt] > 0):
                        continue
                    if mt not in fissionPhotonEnergy:
                        raise ValueError(f"fission photon response requires photon energy for MT{mt}")
                    energy = np.asarray(fissionPhotonEnergy[mt], dtype=float)
                    if energy.shape != self.grid.shape or not np.all(np.isfinite(energy)):
                        raise ValueError(f"invalid fission photon energy for MT{mt}")
                    values += cross[mt] * energy
            output[response] = values
        return output


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


def _expect(table, g: Callable[[np.ndarray], np.ndarray], trapezoid: bool = False) -> float:
    """``∫ g(x) p(x) dx`` over an ``Xs_pdf_cdf1d`` or ``XYs1d`` (lines first, then the continuum), normalised.

    ``trapezoid`` takes the continuum by trapezoids over ``g·p`` at the
    tabulated points, as HEATR does for MF6 spectra in the lab.
    """
    x = np.asarray(table.xs, dtype=float)
    p = np.asarray(table.pdf if hasattr(table, "pdf") else table.ys, dtype=float)
    nd = int(getattr(table, "numberOfDiscrete", 0) or 0)
    total = 0.0
    value = 0.0
    if nd:
        w = p[:nd]
        total += float(w.sum())
        value += float(np.sum(w * g(x[:nd])))
    cx, cp = x[nd:], p[nd:]
    if len(cx) > 1 and trapezoid and not _isFlat(table):
        dx = np.diff(cx)
        gp = g(cx) * cp
        total += float(np.sum(0.5 * (cp[:-1] + cp[1:]) * dx))
        value += float(np.sum(0.5 * (gp[:-1] + gp[1:]) * dx))
    elif len(cx) > 1:
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
    p = np.asarray(table.pdf if hasattr(table, "pdf") else table.ys, dtype=float)
    nd = int(getattr(table, "numberOfDiscrete", 0) or 0)
    xs, p = xs[nd:], p[nd:]
    if _isFlat(table):
        i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 1)
        out = p[i]
    else:
        out = np.interp(x, xs, p)
    return np.where((x >= xs[0]) & (x <= xs[-1]), out, 0.0)


def _tables(form):
    """``(incident energies, tables, flat)`` of an XYs2d or Regions2d of tables.

    A parametrised spectrum (MF5 LF=7/9/11/12; OpenMC keeps them as laws, so the
    Monte Carlo form is the law) is tabulated first, at the incident energies
    its parameters are given at (``_Spectrum.toPointwise``).
    """
    from kika.nuclear_data.model import Interpolation

    if not hasattr(form, "function1ds") and not hasattr(form, "function2ds") and hasattr(form, "toPointwise"):
        form = form.toPointwise()
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
    #: HEATR's mean of a lab-frame MF6 spectrum: trapezoids over E'·f(E')
    #: (``getsix``), not the exact integral of the lin-lin density. On wide
    #: panels it is lower: U-235 MT91's photons at 5.382 MeV average 1.473 MeV
    #: by trapezoids and 1.515 MeV exactly.
    trapezoid: bool = False


def _qAt(value, grid):
    """An auxiliary Q on its declared domain; never extrapolate a user table."""
    grid = np.asarray(grid, dtype=float)
    if hasattr(value, "evaluate"):
        lo, hi = float(value.domainMin), float(value.domainMax)
        if np.any(grid < lo) or np.any(grid > hi):
            raise ValueError("qbar does not cover the heating energy grid")
        result = np.asarray(value.evaluate(grid), dtype=float)
    else:
        result = np.full(grid.shape, float(value))
    if result.shape != grid.shape or not np.all(np.isfinite(result)):
        raise ValueError("auxiliary Q must give one finite value per incident energy")
    return result


def _cmToLab(grid, ecm, sqrtMu, k: _Kinematics):
    """``⟨E'_lab⟩ = ⟨E'_cm⟩ + x E + 2√(xE)⟨√E'_cm μ⟩``, ``x = awp/(A+1)²``."""
    x = k.awp / (k.A + 1.0) ** 2
    return ecm + x * grid + 2.0 * np.sqrt(x * grid) * sqrtMu


def _meanMu(form, grid):
    if form is None or type(form).__name__ == "Isotropic2d":
        return np.zeros(len(grid))
    energies, tables, flat = _tables(form)
    return _onGrid(energies, [_expect(t, lambda m: m) for t in tables], grid, flat)


def _unitBaseCMMean(form, grid, k):
    """Integrate a continuous unit-base joint law, before the lab transformation.

    A blend of moments is not the moment of a unit-base blend. Each source
    density is mapped to the interpolated support. Substitution sqrt(E')
    makes the square-root cross moment polynomial on every linear panel.
    Interpolation of the resulting lab mean is refined on three interior
    probes to 0.1 eV + 1e-7 times incident energy; this is a sampled criterion.
    """
    energies,tables,flat=_tables(form.energy)
    if flat:return None
    if any(getattr(t,"numberOfDiscrete",0) for t in tables):return None
    angular=list(form.energyAngular.function2ds)
    mus=[np.array([_expect(u,lambda x:x) for u in a.function1ds]) for a in angular]
    means=[_expect(t,lambda e:e) for t in tables]
    ranges=[(float(t.xs[0]),float(t.xs[-1]-t.xs[0])) for t in tables]
    if any(w<=0 for _,w in ranges):return None
    for t,mu in zip(tables,mus):
        if len(mu)!=len(t.xs):raise ValueError("energy-angle grids differ in unit-base moment")
    def at(E):
        j=int(np.clip(np.searchsorted(energies,E,side="right")-1,0,len(energies)-1))
        upper=min(j+1,len(energies)-1)
        theta=(E-energies[j])/(energies[upper]-energies[j]) if upper!=j else 0.
        theta=float(np.clip(theta,0,1));weights=(1-theta,theta)
        qa=sum(w*ranges[i][0] for w,i in zip(weights,(j,upper)))
        qw=sum(w*ranges[i][1] for w,i in zip(weights,(j,upper)))
        mean=cross=0.
        for weight,i in zip(weights,(j,upper)):
            if weight==0:continue
            a,width=ranges[i];table=tables[i];x=np.asarray(table.xs);p=np.asarray(table.pdf);mu=mus[i]
            mapped=qa+qw*(x-a)/width
            root=np.sqrt(np.maximum(mapped,0));left=root[:-1,None];right=root[1:,None]
            nodes,ww=_GL4;r=(left+right)/2+(right-left)/2*nodes
            original=a+(r*r-qa)*width/qw
            frac=(original-x[:-1,None])/np.maximum(np.diff(x)[:,None],1e-300)
            if _isFlat(table):
                # A correlated histogram uses its lower-bin angular law too
                # (OpenMC secondary_correlated.cpp); no endpoint blend.
                density=p[:-1,None]
                angle=mu[:-1,None]
                norm=float(np.sum(np.diff(x)*p[:-1]))
            else:
                density=p[:-1,None]+np.diff(p)[:,None]*frac
                angle=mu[:-1,None]+np.diff(mu)[:,None]*frac
                norm=float(np.sum(np.diff(x)*(p[:-1]+p[1:])/2))
            integral=float(np.sum((right-left)/2*ww*2*r*r*width/qw*density*angle))/norm
            cross+=weight*integral
            mean+=weight*(qa+qw*(means[i]-a)/width)
        boost=k.awp/(k.A+1)**2
        return mean+boost*E+2*np.sqrt(boost*E)*cross
    lo=max(float(np.min(grid)),float(energies[0]));hi=min(float(np.max(grid)),float(energies[-1]))
    if hi<lo:return np.array([at(float(E)) for E in grid])
    nodes=np.unique(np.r_[lo,energies[(energies>lo)&(energies<hi)],hi])
    values=np.array([at(float(E)) for E in nodes])
    for iteration in range(30):
        additions=[]
        for a,b,va,vb in zip(nodes[:-1],nodes[1:],values[:-1],values[1:]):
            if b==a:continue
            for fraction in (.25,.5,.75):
                E=a+(b-a)*fraction;v=at(float(E))
                if abs(v-(va+(vb-va)*fraction))>.1+1e-7*E:additions.append((E,v))
        if not additions:break
        if len(nodes)+len(additions)>20000:raise ValueError("unit-base mean refinement exceeds capacity")
        # Each new node was already integrated while checking the error.
        # Keep those exact values and the previous nodes' values; do not
        # repeat quadrature on the enlarged mesh.
        known = dict(zip(nodes, values))
        known.update(additions)
        nodes = np.array(sorted(known))
        values = np.array([known[E] for E in nodes])
    else:raise ValueError("unit-base mean refinement did not converge")
    output=np.interp(grid,nodes,values)
    outside=(grid<lo)|(grid>hi)
    for index in np.flatnonzero(outside):output[index]=at(float(grid[index]))
    return output


def _unitBaseSpectrumMean(energy, grid, moment):
    """OpenMC's unit-base continuum mapping, with unmapped discrete lines."""
    energies, tables, flat = _tables(energy)
    if flat:
        return _onGrid(energies, [_expect(t, lambda e: e**moment) for t in tables], grid, True)
    parts = []
    low, width = [], []
    nodes, weights = _GL4
    for table in tables:
        x, p = np.asarray(table.xs), np.asarray(table.pdf if hasattr(table, "pdf") else table.ys)
        nd = int(getattr(table, "numberOfDiscrete", 0) or 0)
        discrete = np.array([np.sum(p[:nd]*x[:nd]**k) for k in range(3)])
        cx, cp = x[nd:], p[nd:]
        if len(cx) < 2 or cx[-1] <= cx[0]:
            return _onGrid(energies, [_expect(t, lambda e: e**moment) for t in tables], grid)
        half = np.diff(cx)/2
        points = (cx[:-1]+cx[1:])[:, None]/2 + half[:, None]*nodes
        if _isFlat(table):
            density = cp[:-1, None]
        else:
            density = cp[:-1, None]+np.diff(cp)[:, None]*(nodes+1)/2
        weighted = half[:, None]*weights*density
        continuous = np.array([np.sum(weighted*points**k) for k in range(3)])
        mass = discrete[0]+continuous[0]
        if mass <= 0:
            raise ValueError("unit-base spectrum has no positive mass")
        parts.append(np.r_[discrete, continuous]/mass)
        low.append(float(cx[0])); width.append(float(cx[-1]-cx[0]))
    parts, low, width = np.asarray(parts), np.array(low), np.array(width)
    j = np.clip(np.searchsorted(energies, grid, side="right")-1, 0, len(energies)-1)
    k = np.minimum(j+1, len(energies)-1)
    span = energies[k]-energies[j]
    fraction = np.clip(np.divide(grid-energies[j], span, out=np.zeros(len(grid)), where=span>0), 0., 1.)
    qa = (1-fraction)*low[j]+fraction*low[k]
    qw = (1-fraction)*width[j]+fraction*width[k]
    output = np.zeros(len(grid))
    for weight, ids in ((1-fraction, j), (fraction, k)):
        mass = parts[ids, 3]
        u1 = (parts[ids, 4]-low[ids]*mass)/width[ids]
        value = parts[ids, 1]+qa*mass+qw*u1
        if moment == 2:
            u2 = (parts[ids, 5]-2*low[ids]*parts[ids, 4]+low[ids]**2*mass)/width[ids]**2
            value = parts[ids, 2]+qa*qa*mass+2*qa*qw*u1+qw*qw*u2
        output += weight*value
    return output


def _analyticSpectrumMoment(spectrum, grid, power):
    """Truncated gamma moment with the source's incident parameter law.

    The small-support form avoids ratios of underflowing incomplete gamma
    functions; neither branch needs a rendered outgoing-energy grid.
    """
    from scipy.special import gammainc, gammaln, hyp1f1
    from kika.nuclear_data import model as m

    shape = 1.5 if isinstance(spectrum, m.SimpleMaxwellianFission) else 2.
    grid = np.asarray(grid, dtype=float)
    top = np.array([spectrum.upperBound(float(E)) for E in grid])
    theta = np.asarray(spectrum.theta.evaluate(grid, outOfRange="hold"), dtype=float)
    active = top > 0
    if (np.any(~np.isfinite(top)) or
            np.any(active & ((theta <= 0) | ~np.isfinite(theta)))):
        raise ValueError("analytic energy moments require positive theta and finite support")
    result = np.zeros(len(grid))
    x = top[active] / theta[active]
    small = x < 1.
    value = np.empty(len(x))
    value[small] = (top[active][small] ** power * shape / (shape + power)
        * hyp1f1(shape + power, shape + power + 1, -x[small])
        / hyp1f1(shape, shape + 1, -x[small]))
    value[~small] = (theta[active][~small] ** power
        * np.exp(gammaln(shape + power) - gammaln(shape))
        * gammainc(shape + power, x[~small]) / gammainc(shape, x[~small]))
    result[active] = value
    return result


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
        if isinstance(energy, (m.SimpleMaxwellianFission, m.Evaporation)):
            mean = _analyticSpectrumMoment(energy, grid, moment)
            if not cm or k.awp == 0.0:
                return mean
            if moment != 1:
                raise NotImplementedError("second moment of a centre-of-mass spectrum")
            mu = _meanMu(form.angular, grid)
            root = _analyticSpectrumMoment(energy, grid, .5) if np.any(mu) else 0.
            return _cmToLab(grid, mean, root * mu, k)
        energies, tables, flat = _tables(energy)
        lab = not cm or k.awp == 0.0
        # HEATR's trapezoids are for the mean; the photons' second moment
        # (capture recoil) it integrates by Gauss points (tabsqr, tabsq6).
        mean = _onGrid(energies, [_expect(t, lambda e: e ** moment,
                                          k.trapezoid and lab and moment == 1)
                                  for t in tables], grid, flat)
        qualifier = getattr(energy, "interpolationQualifier", None)
        if (lab and moment in (1, 2) and
                (str(getattr(qualifier, "value", qualifier)) == "unitBase"
                 or all(hasattr(t, "pdf") for t in tables))):
            return _unitBaseSpectrumMean(energy, grid, moment)
        if lab:
            return mean
        if moment != 1:
            raise NotImplementedError("second moment of a centre-of-mass spectrum")
        root = _onGrid(energies, [_expect(t, np.sqrt) for t in tables], grid, flat)
        return _cmToLab(grid, mean, root * _meanMu(form.angular, grid), k)
    if isinstance(form, m.EnergyAngularMC):
        qualifier=getattr(form.energy,"interpolationQualifier",None)
        if cm and moment==1 and str(getattr(qualifier,"value",qualifier))=="unitBase":
            exact=_unitBaseCMMean(form,grid,k)
            if exact is not None:return exact
        energies, tables, flat = _tables(form.energy)
        mean = _onGrid(energies, [_expect(t, lambda e: e ** moment, k.trapezoid and not cm)
                                  for t in tables], grid, flat)
        if not cm or k.awp == 0.0:
            if moment in (1, 2):
                return _unitBaseSpectrumMean(form.energy, grid, moment)
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
    thr = (A + 1.0) / A * max(-Q, 0.0)
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
    if isinstance(spectrum, (m.SimpleMaxwellianFission, m.Evaporation)):
        return _analyticEmissionDamage(spectrum, grid, A, Z, ed)
    energies, tables, flat = _tables(spectrum)
    u, wu = _GL4
    values = []
    for E, t in zip(energies, tables):
        def g(ep, E=E):
            er = (E - 2.0 * np.sqrt(E * ep)[..., None] * u + ep[..., None]) / A
            return np.sum(wu * lindhard(er, Z, A, Z, A, ed), axis=-1) / 2.0
        values.append(_expect(t, g))
    return _onGrid(energies, values, grid, flat)


def _analyticEmissionDamage(spectrum, grid, A, Z, ed):
    """Integrate LF7/9 directly, retaining the parameter interpolation law.

    A display grid at parameter knots loses logarithmic incident dependence.
    Gaussian orders are doubled on sqrt(E'/theta), splitting the displacement
    threshold in angle. Incident interpolation is checked at three interior
    points; this is a sampled error criterion, with explicit resource caps.
    """
    from functools import lru_cache
    gauss = {n: np.polynomial.legendre.leggauss(n) for n in (16, 32, 64, 128)}
    @lru_cache(maxsize=30000)
    def at(E):
        theta = float(spectrum.theta.evaluate(E, outOfRange="hold"))
        top = float(spectrum.upperBound(E))
        if top <= 0:
            return 0.0
        if not theta > 0 or not np.isfinite(top):
            raise ValueError("analytic damage requires positive theta and finite support")
        vmax = np.sqrt(top / theta)
        cuts = np.unique(np.r_[0., vmax, 1., 2., 4., 8.,
            abs(np.sqrt(E) - np.sqrt(A * ed)) / np.sqrt(theta),
            (np.sqrt(E) + np.sqrt(A * ed)) / np.sqrt(theta)])
        cuts = cuts[(cuts >= 0) & (cuts <= vmax)]
        previous = None
        for order in (16, 32, 64, 128):
            nodes, weights = gauss[order]
            half = np.diff(cuts) / 2
            v = ((cuts[:-1] + cuts[1:])[:, None] / 2 + half[:, None] * nodes).ravel()
            w = (half[:, None] * weights).ravel()
            ep = theta * v * v
            root = np.sqrt(E * ep)
            last = np.clip((E + ep - A * ed) / np.maximum(2 * root, 1e-300), -1., 1.)
            mu = (last[:, None] - 1) / 2 + (last[:, None] + 1) / 2 * nodes
            recoil = np.maximum(0., (E + ep[:, None] - 2 * root[:, None] * mu) / A)
            angular = np.sum(weights * lindhard(recoil, Z, A, Z, A, ed), axis=1) * (last + 1) / 4
            density = spectrum.evaluate(E, ep) * (2 * theta * v)
            mass = float(np.sum(w * density))
            value = float(np.sum(w * density * angular))
            if (previous is not None and abs(mass - 1) <= 1e-9
                    and abs(value - previous) <= .01 + 1e-7 * abs(value)):
                return value
            previous = value
        raise ValueError("analytic emission damage quadrature did not converge")
    active = np.array([spectrum.upperBound(float(E)) > 0 for E in grid])
    output = np.zeros(len(grid))
    if not np.any(active):
        return output
    validGrid = np.asarray(grid)[active]
    lo, hi = float(np.min(validGrid)), float(np.max(validGrid))
    if lo == hi:
        output[active] = at(lo)
        return output
    parameters = spectrum.incidentEnergies()
    nodes = np.unique(np.r_[lo, hi, parameters[(parameters > lo) & (parameters < hi)],
                             _geometric(lo, hi, step=4.)])
    nodes = nodes[(nodes >= lo) & (nodes <= hi)]
    values = np.array([at(float(E)) for E in nodes])
    for iteration in range(30):
        additions = {}
        for a, b, va, vb in zip(nodes[:-1], nodes[1:], values[:-1], values[1:]):
            for f in (.25, .5, .75):
                E = float(a + (b - a) * f)
                value = at(E)
                if abs(value - (va + (vb - va) * f)) > .05 + 1e-5 * abs(value):
                    additions[E] = value
        if not additions:
            output[active] = np.interp(validGrid, nodes, values)
            return output
        if len(nodes) + len(additions) > 20000:
            raise ValueError("analytic emission damage interpolation exceeds capacity")
        known = dict(zip(nodes, values)); known.update(additions)
        nodes = np.array(sorted(known)); values = np.array([known[E] for E in nodes])
    raise ValueError("analytic emission damage interpolation did not converge")


def _captureDamage(photons, grid, A, Z, ed, Q, fixedPrimary=False):
    """Capture: ``Σ_γ y_γ ⟨df(E_γ²/2tm)⟩`` (``gheat``), or ``capdam``'s single photon."""
    tm = NEUTRON_REST_ENERGY * (A + 1.0)
    if not photons:
        er = grid / (A + 1.0)
        ea = A * er + Q
        return lindhard(er + ea * ea / (2.0 * tm), Z, A + 1.0, Z, A, ed)
    total = np.zeros(len(grid))
    for product, form in photons:
        y = _yield(product, grid)
        total = total + y * _photonRecoilDamage(form, grid, A, Z, ed, tm, fixedPrimary)
    return total


def _photonRecoilDamage(form, grid, A, Z, ed, tm, fixedPrimary=False):
    from kika.nuclear_data import model as m

    if isinstance(form, m.DistributionMixture):
        out = np.zeros(len(grid))
        for term in form.weighted:
            weight = (np.asarray(term.weight.evaluate(grid), dtype=float)
                      if term.weight is not None else 1.0)
            out = out + weight * _photonRecoilDamage(term.functional, grid, A, Z, ed, tm, fixedPrimary)
        return out
    energy = form.energy
    if isinstance(energy, m.DiscretePhotonEnergy):
        eg = (np.full(len(grid),float(energy.energy)) if fixedPrimary and energy.primaryFlag==2
              else np.asarray(energy.outgoingEnergy(grid), dtype=float))
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

def _twoBodyCapture(photons, evaluation, grid, A):
    """HEATR hgam102: compound recoil plus a discrete photon's fixed kick.

    This is the explicitly declared two-body capture photon (MF6 LAW2),
    not the MF12/13 gamma-cascade approximation. The line binding energy
    comes from the processed physical distribution, never raw sidecars.
    """
    from kika.nuclear_data import model as m
    source = evaluation.findReactionByENDF_MT(102) if evaluation is not None else None
    if source is None or len(photons) != 1:
        return None
    original = next((p for p in source.outputChannel.products if p.pid == "photon"), None)
    forms = getattr(getattr(original, "distribution", None), "forms", {})
    evaluated = forms.get("eval") or (next(iter(forms.values())) if forms else None)
    explicitPrimary = (
        isinstance(evaluated, m.Uncorrelated)
        and isinstance(evaluated.energy, m.PrimaryGamma)
        and (evaluated.productFrame == m.Frame.centerOfMass
             or _statedInMF6(evaluation, 102)))
    if not isinstance(evaluated,m.AngularTwoBody) and not explicitPrimary:
        return None
    product, form = photons[0]
    if not isinstance(form, m.Uncorrelated):
        raise ValueError("two-body capture photon has no uncorrelated line distribution")
    if isinstance(form.energy, m.DiscretePhotonEnergy):
        if form.energy.primaryFlag != 2:
            raise ValueError("two-body capture photon must be a primary line")
        binding = float(form.energy.energy)
    else:
        energies, tables, flat = _tables(form.energy)
        if not tables or any(getattr(t, "numberOfDiscrete", 0) != 1 or len(t.xs) != 1 for t in tables):
            raise ValueError("two-body capture requires exactly one discrete photon line")
        binding = float(tables[0].xs[0]) - A / (A + 1) * energies[0]
        predicted = binding + A / (A + 1) * np.asarray(energies)
        if not np.allclose([t.xs[0] for t in tables], predicted, rtol=1e-6, atol=1e-6):
            raise ValueError("two-body capture line does not follow the declared primary kinematics")
    if not np.isfinite(binding) or binding <= 0:
        raise ValueError("two-body capture binding energy must be positive and finite")
    return _yield(product, grid) * (grid / (A + 1) + binding * binding /
                                   (2 * NEUTRON_REST_ENERGY * (A + 1)))


def _statedInMF6(evaluation, mt) -> bool:
    """The evaluation states this reaction's products in MF6 (its ENDF provenance says so)."""
    if evaluation is None:
        return False
    source = evaluation.findReactionByENDF_MT(mt)
    fields = getattr(getattr(source, "provenance", None), "headerFields", None) or {}
    return bool(fields.get("mf6"))



def _mf12Capture(evaluation, mt):
    """Source branch only; numerical photon values always come from the model."""
    if evaluation is None:return False
    source=evaluation.findReactionByENDF_MT(mt)
    fields=getattr(getattr(source,"provenance",None),"headerFields",None) or {}
    return bool(fields.get("mf12")) and not bool(fields.get("mf6"))


def _mf12CaptureDomain(photons, grid):
    """gheat closes its correction on the last MF12 subsection's domain."""
    if not photons:return np.zeros(len(grid),dtype=bool)
    multiplicity=getattr(photons[-1][0],"multiplicity",None)
    forms=getattr(multiplicity,"forms",None) or {}
    function=forms.get(MONTE_CARLO) or getattr(multiplicity,"form",None)
    lo=getattr(function,"domainMin",None);hi=getattr(function,"domainMax",None)
    xs=getattr(function,"xs",None);ys=getattr(function,"ys",None)
    if xs is not None and ys is not None:
        xs=np.asarray(xs);live=np.flatnonzero(np.asarray(ys)!=0)
        if not len(live):return np.zeros(len(grid),dtype=bool)
        first,last=int(live[0]),int(live[-1])
        lo=float(xs[first]);hi=float(xs[min(last+1,len(xs)-1)])
        if first and not _isFlat(function):lo=float(xs[first-1])
        # A repeated terminal x states a zero tail while gheat's support
        # still includes that x. The recoil uses the actual (right) yield.
        return (grid>=lo)&(grid<=hi)
    return ((grid>=lo) if lo is not None else np.ones(len(grid),dtype=bool)) & ((grid<=hi) if hi is not None else True) & (_yield(photons[-1][0],grid)>0)


def _kinematicUpperOf(mt, grid, A, Q, Qbal, emitted, neutronYield, carried,
                      deposit, heavy, mf6, lr, twoBodyCapture, photons, captureCompound=False, captureDomain=None):
    """HEATR nheat/gheat MT443 diagnostic, in eV per collision.

    The historical high limit is a diagnostic approximation, not a certified
    physical bound. Explicit MF6 recoil energy is mandatory on the MF6 path.
    """
    if mt == 2:
        return carried
    if mf6:
        if mt == 102 and twoBodyCapture is not None:
            return carried
        if heavy is not None:
            return heavy[0]
        if 51 <= mt <= 90:
            return grid + Q - emitted
        return None
    if mt == 102:
        if photons:
            # gheat replaces the single-photon kick of nheat with the
            # independent kicks of the stated gamma cascade.
            if captureCompound:
                available=Q+A*grid/(A+1)
                fallback=grid/(A+1)+available*available/(2*NEUTRON_REST_ENERGY*(A+1))
                return np.where(captureDomain,carried,fallback+carried-deposit)
            return grid / (A + 1) + carried
        available = Q + A * grid / (A + 1)
        return grid / (A + 1) + available * available / (2 * NEUTRON_REST_ENERGY * (A + 1))
    if 51 <= mt <= 90:
        return grid + (Qbal if lr not in (0, 31) else Q) - emitted
    mean = np.divide(emitted, neutronYield, out=np.zeros_like(grid), where=neutronYield > 0)
    if mt == 91:
        return grid + Qbal - mean if lr not in (0, 31) else (grid + mean) / (A + 1)
    if mt == 16 and A >= 10:
        return np.minimum(deposit, (grid + mean) / (A - 1))
    if mt == 17 and A >= 10:
        return np.minimum(deposit, (grid + 2 * mean) / (A - 2))
    return deposit


def heatingOf(suite, grid, sigma: Dict[int, np.ndarray], *, awr: float, Z: int,
              qm: Optional[Dict[int, float]] = None, ed: Optional[float] = None,
              qa: Optional[Dict[int, float]] = None, qbar: Optional[Dict[int, object]] = None,
              evaluation=None, zaTarget: Optional[int] = None,
              report=None, kinematic: bool = False) -> HeatingResult:
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
    Scalar ``qa`` replaces the ground-state balance Q on HEATR's supported
    non-fission paths, without changing evaluated kinematics. ``qbar`` maps MTs
    to model Function1d objects. It replaces the balance Q and the Q used by
    the absorption/capture recoil fallback; evaluated outgoing distributions
    and the two-body threshold remain unchanged, as HEATR disbar actually
    initialises that threshold before reading qbar. Tables must cover the whole
    requested grid. Capture with explicit photons and MT5 explicit
    deposition keep their evaluated recoil/deposit convention.
    ``kinematic=True`` computes the additional MT443 diagnostic arrays.
    They are omitted by default, saving time and memory for normal heating.
    For fission with MF1/458, that release retains authority as in HEATR;
    auxiliary fission Q is accepted and reported but does not replace its
    released-energy components. Missing release information raises explicitly.
    """
    grid = np.asarray(grid, dtype=float)
    A = float(awr)
    ed = displacementEnergy(Z) if not ed else float(ed)
    qm = qm or {}
    qa, qbar = dict(qa or {}), dict(qbar or {})
    if set(qa) & set(qbar):
        raise ValueError("a reaction cannot have both scalar qa and tabulated qbar")
    physical = list(suite.reactions) + list(getattr(suite, "fissionComponents", ()))
    fissionSources = physical + [r for r in getattr(suite, "sums", ()) if r.ENDF_MT == 18]
    known = {r.ENDF_MT for r in fissionSources}
    if (set(qa) | set(qbar)) - known:
        raise ValueError("auxiliary Q names an absent reaction")
    auxiliaryFissions = (set(qa) | set(qbar)) & set(FISSION_MTS)
    if auxiliaryFissions:
        releases = [release for r in fissionSources if r.ENDF_MT in FISSION_MTS
                    for release in getattr(getattr(r.outputChannel, "fissionFragmentData", None),
                                           "fissionEnergyReleases", ())]
        if not any(release.promptProductKE is not None and release.promptGammaEnergy is not None
                   for release in releases):
            raise NotImplementedError("fission auxiliary Q requires fission-release processing")
    from kika.nuclear_data import model as m
    for mt, value in qa.items():
        if not np.isscalar(value) or not np.isfinite(float(value)):
            raise ValueError(f"MT{mt} scalar qa must be finite")
    for mt, value in qbar.items():
        if not isinstance(value, m.Function1d):
            raise TypeError(f"MT{mt} qbar must be a model Function1d")
        _qAt(value, grid)
    tm = NEUTRON_REST_ENERGY * (A + 1.0)
    perReaction: Dict[int, tuple] = {}
    photonEnergy: Dict[int, np.ndarray] = {}
    kinematicUpper: Dict[int, np.ndarray] = {}
    kerma = np.zeros(len(grid))
    local = np.zeros(len(grid))
    damage = np.zeros(len(grid))
    notes: List[str] = [f"heating by kika (energy balance on the Monte Carlo tables; "
                        f"damage with ed = {ed:g} eV)"] + [f"as HEATR: {a}" for a in HEATR_APPROXIMATIONS]
    for mt in sorted(auxiliaryFissions):
        notes.append(f"MT{mt}: auxiliary fission Q accepted; MF1/458 retains authority for released energy, as HEATR")
    for reaction in suite.reactions:
        mt = reaction.ENDF_MT
        if mt is None or mt in FISSION_MTS or mt not in sigma:
            continue
        s = np.asarray(sigma[mt], dtype=float)
        Q = float(reaction.outputChannel.Q.value or 0.0)
        Qbal = float(qm.get(mt, Q))
        Qkin = qbar.get(mt, Q)
        if mt in qbar:
            Qbal = _qAt(Qkin, grid)
        elif mt in qa:
            if mt <= 100 or _statedInMF6(evaluation, mt):
                Qbal = float(qa[mt])
            else:
                notes.append(f"MT{mt}: scalar qa does not change this absorption balance, as HEATR")
        if mt in qa or mt in qbar:
            notes.append(f"MT{mt}: user auxiliary Q ({'qbar' if mt in qbar else 'qa'})")
        # HEATR disbar initialises its threshold from evaluated Q before
        # qbar is read at each energy. Auxiliary heating Q does not rewrite
        # the processed Monte Carlo neutron distribution.
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
        neutronYield = np.zeros(len(grid)) if kinematic else None
        for product, form in neutrons:
            multiplicity = _yield(product, grid)
            if kinematic:
                neutronYield += multiplicity
            emitted = emitted + multiplicity * _meanEnergyOf(form, grid, k)
        deposit = grid + Qbal - emitted
        gammas = np.zeros(len(grid))
        exact = np.zeros(len(grid))
        for product, form in photons:
            y = _yield(product, grid)
            gammas = gammas + y * _meanEnergyOf(form, grid, photon)
            exact = gammas
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
        twoBodyCapture = None
        captureCompound = mt == 102 and _mf12Capture(evaluation, mt)
        captureDomain = _mf12CaptureDomain(photons,grid) if captureCompound else None
        if mt == 102 and photons:
            second = np.zeros(len(grid))
            for product, form in photons:
                energy = getattr(form,"energy",None)
                if captureCompound and isinstance(energy,m.DiscretePhotonEnergy) and energy.primaryFlag==2:
                    kick = np.full(len(grid),float(energy.energy)**2)
                else:
                    kick = _meanEnergyOf(form, grid, photon, moment=2)
                second = second + _yield(product, grid) * kick
            # MF6 cascades retain the measured photon-kick approximation.
            # MF12 goes through gheat instead: its compound recoil and fixed
            # primary-line kicks are observed separately in real HEATR.
            carried = second / (2.0 * tm)
            if captureCompound:
                carried = carried + np.where(captureDomain,grid/(A+1),deposit)
                notes.append("MT102: MF12 gheat branch; compound recoil and fixed primary-line photon kicks")
            twoBodyCapture = _twoBodyCapture(photons, evaluation, grid, A)
            if twoBodyCapture is not None:
                carried = twoBodyCapture
                notes.append("MT102: explicit two-body primary photon; compound recoil "
                             "plus fixed photon kick as HEATR hgam102")
        else:
            carried = deposit - gammas
        live = s > 0
        kerma += np.where(live, s * carried, 0.0)
        # MT901 is HEATR's MT301 plus the energy of the photons the reaction
        # emits, taken exactly from the tables -- what OpenMC's files carry
        # (Fe-56 capture at 0.0253 eV: 901 - 301 is the photons' 7.489 MeV, not
        # HEATR's local=1 balance E + Q = 7.646 MeV).
        local += np.where(live, s * (carried + exact), 0.0)
        # damage
        if mt == 2 or (51 <= mt <= 90 and neutrons):
            d = _twoBodyDamage(neutrons[0][1], grid, A, Q, Z, ed, elastic=(mt == 2)) if neutrons \
                else np.zeros(len(grid))
        elif mt == 102:
            if photons and twoBodyCapture is not None:
                y = _yield(photons[0][0], grid)
                recoil = np.divide(carried, y, out=np.zeros_like(carried), where=y > 0)
                d = y * lindhard(recoil, Z, A + 1, Z, A, ed)
            else:
                d = _captureDamage(photons, grid, A, Z, ed, _qAt(Qkin, grid), fixedPrimary=captureCompound)
                if captureCompound and photons:
                    fallback=_captureDamage([],grid,A,Z,ed,_qAt(Qkin,grid))
                    d=d+np.where(captureDomain,0.,fallback)
        elif heavy is not None:
            d = heavy[1]
        elif neutrons:
            d = np.zeros(len(grid))
            for product, form in neutrons[:1]:
                d = _emissionDamage(form, grid, A, Z, ed)
        elif mt > 102:
            d = _absorptionDamage(mt, grid, A, Z, ed, _qAt(Qkin, grid))
        else:
            d = np.zeros(len(grid))
        damage += np.where(live, s * d, 0.0)
        perReaction[mt] = (carried, carried + exact, d)
        photonEnergy[mt] = exact
        if kinematic:
            fields = getattr(getattr(reaction, "provenance", None), "headerFields", None) or {}
            mf6 = _statedInMF6(evaluation, mt) or bool(fields.get("mf6"))
            lr = int(getattr(getattr(reaction, "provenance", None), "lr", 0) or 0)
            upper = _kinematicUpperOf(mt, grid, A, _qAt(Qkin, grid), Qbal, emitted,
                neutronYield, carried, deposit, heavy, mf6, lr, twoBodyCapture, photons, captureCompound, captureDomain)
            if upper is not None:
                kinematicUpper[mt] = upper
    # Fission's damage: HEATR's conbar gives a fission the recoil of one
    # emission from its prompt-neutron spectrum (tabdam, one neutron, not ν̄;
    # the fragments are not followed): U-235 at thermal, ~6.9 keV per fission.
    # MT444 keeps it -- unlike the KERMA, whose fission part comes from MF1/458.
    chances = [r for r in getattr(suite, "fissionComponents", ()) if r.ENDF_MT in FISSION_MTS]
    fissions = chances or [r for r in suite.reactions if r.ENDF_MT in FISSION_MTS]
    for reaction in fissions:
        mt = reaction.ENDF_MT
        if mt not in sigma:
            continue
        prompt = next(((p, _distribution(p)) for p in reaction.outputChannel.products
                       if p.pid == "n" and _distribution(p) is not None), None)
        if prompt is None:
            continue
        s = np.asarray(sigma[mt], dtype=float)
        d = _emissionDamage(prompt[1], grid, A, Z, ed)
        damage += np.where(s > 0, s * d, 0.0)
        perReaction[mt] = (np.zeros(len(grid)), np.zeros(len(grid)), d)
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
        photonEnergy[mt] = gammas
        notes.append(f"MT{mt}: {len(photons)} photons stated on the sum, taken out of MT301")
    if report is not None:
        for note in notes:
            report.warn(note)
    return HeatingResult(grid=grid, kerma=kerma, kermaLocal=local, damage=damage, notes=notes,
                         perReaction=perReaction, photonEnergy=photonEnergy, kinematicUpper=kinematicUpper)
