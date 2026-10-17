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

import contextvars
import copy
import os
import weakref
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["HeatingResult", "heatingOf", "displacementEnergy", "lindhard", "NEUTRON_REST_ENERGY",
           "DisplacementModel", "displacementModels", "ARC_DPA"]

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

from importlib import import_module
try:
    _heating_native = import_module("kika.processing._heating_native")
except ModuleNotFoundError as exc:  # the NumPy reference path stays the definition
    if exc.name != "kika.processing._heating_native":
        raise
    _heating_native = None
if _heating_native is not None and _heating_native.API_VERSION != 2:
    raise ImportError("rebuild kika's native heating extension (API version mismatch)")

#: Threads for the independent incident energies of a damage quadrature: set by
#: heatingOf from the processing's ``workers`` for the length of one call.
_energyThreads = contextvars.ContextVar("kika_heating_threads", default=1)


@dataclass(frozen=True)
class DisplacementModel:
    """Displacements per primary recoil, from its damage energy ``T_d`` (Nordlund 2018).

    ``N_d = 0`` below ``ed``, 1 up to ``2 ed/0.8``, and ``0.8 T_d/(2 ed) xi(T_d)``
    above, with ``xi = 1`` for NRT (``b = 0, c = 1``) and, for arc-dpa,
    ``xi = (1 - c)/(2 ed/0.8)^b T_d^b + c`` (K. Nordlund et al., Nat. Commun. 9,
    1084 (2018), eqs. 4-5). ``T_d`` is the Lindhard-Robinson damage energy of the
    recoil; ``ed`` here is the model's threshold on it, not HEATR's threshold on
    the recoil energy that MT444 keeps.
    """

    name: str
    ed: float
    b: float = 0.0
    c: float = 1.0

    @property
    def mode(self) -> int:
        """1 for NRT, 2 for arc-dpa: the native kernel's switch (0 is MT444)."""
        return 1 if self.b == 0.0 and self.c == 1.0 else 2

    def count(self, T):
        T = np.asarray(T, dtype=float)
        high = 2.0 * self.ed / 0.8
        n = (0.8 * T) / (2.0 * self.ed)
        if self.mode == 2:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                n = n * ((1.0 - self.c) / np.power(high, self.b) * np.power(T, self.b) + self.c)
        return np.where(T < self.ed, 0.0, np.where(T < high, 1.0, n))


#: arc-dpa constants ``Z -> (E_d, b, c)``, Nordlund et al. 2018, table 1 (MD fits for
#: elemental metals; no other element has them, and kika gives no arc-dpa for it).
ARC_DPA = {26: (40.0, -0.568, 0.286), 29: (33.0, -0.68, 0.16), 28: (39.0, -1.01, 0.23),
           46: (41.0, -0.88, 0.15), 78: (42.0, -1.12, 0.11), 74: (70.0, -0.56, 0.12)}


def displacementModels(Z: int, ed: float) -> Dict[str, DisplacementModel]:
    """NRT with the processing's ``ed``; arc-dpa where its constants exist."""
    models = {"nrt": DisplacementModel("nrt", float(ed))}
    if int(Z) in ARC_DPA:
        edArc, b, c = ARC_DPA[int(Z)]
        models["arc-dpa"] = DisplacementModel("arc-dpa", edArc, b, c)
    return models


#: The displacement model the damage quadratures evaluate (None: MT444's damage
#: energy). Set by _heatingOf for its displacement passes; _evaluateAll carries it
#: into its threads.
_DISPLACEMENTS = contextvars.ContextVar("kika_displacement_model", default=None)


def _evaluateAll(at, energies) -> np.ndarray:
    """``[at(E) for E in energies]``, the distinct energies computed on threads.

    ``at`` is a pure, memoised function of one energy whose heavy part runs in
    the native kernel without the GIL, so the values are those of the
    sequential loop whatever thread computes them. An energy that fails raises
    as it would there.
    """
    energies = [float(E) for E in energies]
    distinct = list(dict.fromkeys(energies))
    threads = _energyThreads.get()
    if threads > 1 and len(distinct) > 1:
        from concurrent.futures import ThreadPoolExecutor
        # A pool thread does not inherit this context (the displacement model):
        # each task runs in a copy made here.
        contexts = [contextvars.copy_context() for _ in distinct]
        with ThreadPoolExecutor(max_workers=min(threads, len(distinct))) as pool:
            for _ in pool.map(lambda job: job[0].run(at, job[1]), zip(contexts, distinct)):
                pass
    return np.array([at(E) for E in energies])


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
    #: ``model -> Σ_r σ_r N_d,r`` in b·displacements on ``grid``, fission included:
    #: ``"nrt"`` always, ``"arc-dpa"`` where its constants exist (:class:`DisplacementModel`).
    #: A dpa rate is this times the flux and the time.
    displacements: Dict[str, np.ndarray] = field(default_factory=dict)
    #: ``model -> mt -> N_d`` displacements per reaction.
    displacementsPerReaction: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)

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


def _lindhardConstants(Zr, Ar, Zl, Al):
    """``(el, fl)`` of :func:`lindhard`'s partition, which the native kernel takes as given."""
    el = 30.724 * Zr * Zl * np.sqrt(Zr ** (2 / 3) + Zl ** (2 / 3)) * (Ar + Al) / Al
    denom = (Zr ** (2 / 3) + Zl ** (2 / 3)) ** 0.75 * Ar ** 1.5 * np.sqrt(Al)
    fl = 0.0793 * Zr ** (2 / 3) * np.sqrt(Zl) * (Ar + Al) ** 1.5 / denom
    return el, fl


def lindhard(E, Zr, Ar, Zl, Al, ed):
    """Damage energy of a recoil ``(Zr, Ar)`` of energy ``E`` in a lattice ``(Zl, Al)`` (``df``).

    Under a :class:`DisplacementModel` (``_DISPLACEMENTS``), its displacements instead.
    """
    E = np.asarray(E, dtype=float)
    if Zr == 0:
        return np.zeros_like(E)
    el, fl = _lindhardConstants(Zr, Ar, Zl, Al)
    model = _DISPLACEMENTS.get()
    if E.ndim == 0:
        ep = np.maximum(E, 0.0) / el
        out = E / (1.0 + fl * (3.4008 * ep ** (1 / 6) + 0.40244 * ep ** 0.75 + ep))
        if model is not None:
            return model.count(out)
        return np.where(E >= ed, out, 0.0)
    # The same expression in place, in its order of operations: the damage
    # quadratures call this on millions of recoils at a time, and the
    # temporaries cost a fifth of it.
    ep = np.maximum(E, 0.0)
    ep /= el
    out = np.power(ep, 1 / 6)
    out *= 3.4008
    term = np.power(ep, 0.75)
    term *= 0.40244
    out += term
    out += ep
    out *= fl
    out += 1.0
    np.divide(E, out, out=out)
    if model is not None:
        # Displacements from the damage energy itself: the model's threshold is
        # on T_d, so HEATR's on the recoil energy does not apply.
        return model.count(out)
    out[~(E >= ed)] = 0.0
    return out


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
    makes the tabulated square-root cross moment polynomial on every linear
    panel. Kalbach parameters are read before unit-base mapping, as OpenMC
    samples them; Gaussian orders are doubled until their lab cross term
    agrees to .001 eV + 1e-8 incident energy.
    Interpolation of the resulting lab mean is refined on three interior
    probes to 0.1 eV + 1e-7 times incident energy; this is a sampled criterion.
    """
    from functools import lru_cache
    from kika.nuclear_data import model as m
    kalbach = isinstance(form,m.KalbachMann)
    energies,tables,flat=_tables(form.f if kalbach else form.energy)
    if flat:return None
    if any(getattr(t,"numberOfDiscrete",0) for t in tables):return None
    gauss = {4:_GL4}
    if kalbach:
        rs=list(m.toEndfTab2(form.r)[0]);slopes=list(m.toEndfTab2(form.a)[0])
        if len(rs)!=len(tables) or len(slopes)!=len(tables):
            raise ValueError("Kalbach parameter incident grids differ")
        parameters=[]
        for table,rr,aa in zip(tables,rs,slopes):
            if not np.array_equal(table.xs,rr.xs) or not np.array_equal(table.xs,aa.xs):
                raise ValueError("Kalbach parameter outgoing grids differ")
            parameters.append((np.asarray(rr.ys),np.asarray(aa.ys)))
        gauss.update({n:np.polynomial.legendre.leggauss(n) for n in (8,16,32)})
        mus=None
    else:
        angular=list(form.energyAngular.function2ds)
        mus=[np.array([_expect(u,lambda x:x) for u in a.function1ds]) for a in angular]
    means=[_expect(t,lambda e:e) for t in tables]
    ranges=[(float(t.xs[0]),float(t.xs[-1]-t.xs[0])) for t in tables]
    if any(w<=0 for _,w in ranges):return None
    if not kalbach:
        for t,mu in zip(tables,mus):
            if len(mu)!=len(t.xs):raise ValueError("energy-angle grids differ in unit-base moment")
    if not kalbach and all(a==0. for a,width in ranges):
        # A zero lower support factors sqrt(width) out of the cross moment.
        # Only normalized source moments remain; the incident mixture and
        # CM boost can then be evaluated exactly, without an incident mesh.
        unitCross=[]
        for table,mu,(a,width) in zip(tables,mus,ranges):
            x=np.asarray(table.xs);p=np.asarray(table.pdf)
            roots=np.sqrt(np.maximum(x/width,0.));left=roots[:-1,None];right=roots[1:,None]
            nodes,weights=_GL4;r=(left+right)/2+(right-left)/2*nodes
            frac=(width*r*r-x[:-1,None])/np.maximum(np.diff(x)[:,None],1e-300)
            if _isFlat(table):
                density=p[:-1,None];angle=mu[:-1,None]
                norm=float(np.sum(np.diff(x)*p[:-1]))
            else:
                density=p[:-1,None]+np.diff(p)[:,None]*frac
                angle=mu[:-1,None]+np.diff(mu)[:,None]*frac
                norm=float(np.sum(np.diff(x)*(p[:-1]+p[1:])/2))
            unitCross.append(float(np.sum((right-left)/2*weights*2*r*r*width*density*angle))/norm)
        query=np.asarray(grid,dtype=float);widths=np.array([w for a,w in ranges]);unitMean=np.asarray(means)/widths;unitCross=np.asarray(unitCross)
        j=np.clip(np.searchsorted(energies,query,side="right")-1,0,len(energies)-1);upper=np.minimum(j+1,len(energies)-1)
        span=energies[upper]-energies[j]
        fraction=np.clip(np.divide(query-energies[j],span,out=np.zeros(len(query)),where=span>0),0.,1.)
        width=(1-fraction)*widths[j]+fraction*widths[upper]
        mean=width*((1-fraction)*unitMean[j]+fraction*unitMean[upper])
        cross=np.sqrt(width)*((1-fraction)*unitCross[j]+fraction*unitCross[upper])
        boost=k.awp/(k.A+1)**2
        return mean+boost*query+2*np.sqrt(boost*query)*cross
    @lru_cache(maxsize=30000)
    def at(E, components=False):
        j=int(np.clip(np.searchsorted(energies,E,side="right")-1,0,len(energies)-1))
        upper=min(j+1,len(energies)-1)
        theta=(E-energies[j])/(energies[upper]-energies[j]) if upper!=j else 0.
        theta=float(np.clip(theta,0,1));weights=(1-theta,theta)
        qa=sum(w*ranges[i][0] for w,i in zip(weights,(j,upper)))
        qw=sum(w*ranges[i][1] for w,i in zip(weights,(j,upper)))
        mean=cross=0.
        for weight,i in zip(weights,(j,upper)):
            if weight==0:continue
            a,width=ranges[i];table=tables[i];x=np.asarray(table.xs);p=np.asarray(table.pdf)
            mapped=qa+qw*(x-a)/width
            root=np.sqrt(np.maximum(mapped,0));left=root[:-1,None];right=root[1:,None]
            previous=None
            for order,(nodes,ww) in gauss.items():
                r=(left+right)/2+(right-left)/2*nodes
                original=a+(r*r-qa)*width/qw
                frac=(original-x[:-1,None])/np.maximum(np.diff(x)[:,None],1e-300)
                if _isFlat(table):
                    density=p[:-1,None]
                    norm=float(np.sum(np.diff(x)*p[:-1]))
                else:
                    density=p[:-1,None]+np.diff(p)[:,None]*frac
                    norm=float(np.sum(np.diff(x)*(p[:-1]+p[1:])/2))
                if kalbach:
                    rr,aa=parameters[i]
                    if _isFlat(table):
                        rr=rr[:-1,None];aa=aa[:-1,None]
                    else:
                        rr=rr[:-1,None]+np.diff(rr)[:,None]*frac
                        aa=aa[:-1,None]+np.diff(aa)[:,None]*frac
                    safe=np.where(abs(aa)>.01,aa,1.)
                    langevin=np.where(abs(aa)>.01,1/np.tanh(safe)-1/safe,
                                      aa*(1/3+aa*aa*(-1/45+aa*aa*(2/945-aa*aa/4725))))
                    angle=rr*langevin
                else:
                    mu=mus[i]
                    angle=(mu[:-1,None] if _isFlat(table) else
                           mu[:-1,None]+np.diff(mu)[:,None]*frac)
                integral=float(np.sum((right-left)/2*ww*2*r*r*width/qw*density*angle))/norm
                if not kalbach:break
                error=2*np.sqrt(k.awp*max(E,0))/(k.A+1)*abs(integral-previous) if previous is not None else np.inf
                if error <= .001+1e-8*max(E,0):break
                previous=integral
            else:raise ValueError("Kalbach cross-moment quadrature did not converge")
            cross+=weight*integral
            mean+=weight*(qa+qw*(means[i]-a)/width)
        if components:return mean,cross
        boost=k.awp/(k.A+1)**2
        return mean+boost*E+2*np.sqrt(boost*E)*cross
    lo=max(float(np.min(grid)),float(energies[0]));hi=min(float(np.max(grid)),float(energies[-1]))
    def held(query, first):
        source=float(np.nextafter(energies[0],-np.inf)) if first else float(energies[-1])
        mean,cross=at(source,True)
        boost=k.awp/(k.A+1)**2
        return mean+boost*query+2*np.sqrt(boost*query)*cross
    if hi<lo:return held(np.asarray(grid),bool(np.max(grid)<energies[0]))
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
    below=grid<lo;above=grid>hi
    if np.any(below):output[below]=held(grid[below],True)
    if np.any(above):output[above]=held(grid[above],False)
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
    top = grid + spectrum.upperBound(0.)
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
        if not cm or k.awp==0.:
            return _unitBaseSpectrumMean(form.f,grid,moment)
        if moment==1:
            exact=_unitBaseCMMean(form,grid,k)
            if exact is not None:return exact
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

def _damageScale() -> float:
    """What one eV of damage energy is worth in the active units: 1 for MT444,
    ``0.8/(2 E_d)`` displacements under a model, so that the quadratures' absolute
    tolerances (stated in eV) keep their relative weight."""
    model = _DISPLACEMENTS.get()
    return 1.0 if model is None else 0.8 / (2.0 * model.ed)


#: Relative tolerance of the displacement quadratures: N_d's own constants carry
#: 2-10 % (Nordlund et al. 2018, table 1), so 1e-4 is ample; MT444 keeps its own.
DISPLACEMENT_RTOL = 1e-4


def _relTol(rtol: float) -> float:
    """A damage quadrature's relative tolerance: its own for MT444, at least
    DISPLACEMENT_RTOL under a displacement model."""
    return rtol if _DISPLACEMENTS.get() is None else max(rtol, DISPLACEMENT_RTOL)


def _recoilThreshold(ed, Zr, Ar, Zl, Al) -> float:
    """The recoil energy at which a quadrature splits: HEATR's ``ed`` for MT444; under
    a displacement model, the recoil energy whose damage energy is the model's
    ``E_d``, where ``N_d`` steps from 0 to 1 (a cut there is what lets the
    quadrature converge on the step)."""
    model = _DISPLACEMENTS.get()
    if model is None or Zr == 0:
        return ed
    return _thresholdOf(float(model.ed), float(Zr), float(Ar), float(Zl), float(Al))


from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=4096)
def _thresholdOf(target, Zr, Ar, Zl, Al) -> float:
    el, fl = _lindhardConstants(Zr, Ar, Zl, Al)

    def T(E):
        ep = E / el
        return E / (1.0 + fl * (3.4008 * ep ** (1 / 6) + 0.40244 * ep ** 0.75 + ep))

    lo, hi = target, 2.0 * target
    while T(hi) < target:
        lo, hi = hi, 2.0 * hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if T(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * hi:
            break
    return hi


def _recoilKink(Zr, Ar, Zl, Al) -> List[float]:
    """Under a displacement model, the recoil energy where ``T_d = 2 E_d/0.8``: N_d's
    kink (constant 1 below, linear above), where a quadrature must split too."""
    model = _DISPLACEMENTS.get()
    if model is None or Zr == 0:
        return []
    return [_thresholdOf(2.0 * float(model.ed) / 0.8, float(Zr), float(Ar), float(Zl), float(Al))]


def _nativeModel():
    """``(mode, ed, b, c)`` of the active displacement model for the native kernel."""
    model = _DISPLACEMENTS.get()
    return (0, 0.0, 0.0, 1.0) if model is None else (model.mode, float(model.ed), float(model.b), float(model.c))


def _twoBodyChunk(cuts,u,w,x,p,offset,slope,partition,ed,hist):
    """One angular quadrature of :func:`_twoBodyDamage` before its mass division:
    Gauss nodes ``u``/``w`` on each panel between ``cuts``, the recoil
    ``offset - slope*mu`` through :func:`lindhard`. Native when built;
    :func:`_twoBodyChunkNumpy` is the definition."""
    if _heating_native is None:
        return _twoBodyChunkNumpy(cuts,u,w,x,p,offset,slope,partition,ed,hist)
    el,fl=(float(c) for c in _lindhardConstants(*partition))
    return _heating_native.twobody_chunk(cuts,u,w,x,p,float(offset),float(slope),el,fl,float(ed),bool(hist),
        *_nativeModel())


def _twoBodyChunkNumpy(cuts,u,w,x,p,offset,slope,partition,ed,hist):
    half=np.diff(cuts)/2;mid=(cuts[:-1]+cuts[1:])/2
    mu=mid[:,None]+half[:,None]*u
    if hist:density=p[np.clip(np.searchsorted(x,mu,side='right')-1,0,len(p)-1)]
    else:density=np.interp(mu,x,p)
    return float(np.sum(half[:,None]*w*density*lindhard(np.maximum(offset-slope*mu,0.),*partition,ed)))


def _twoBodyDamage(form,grid,A,Q,Z,ed,elastic):
 """Two-body recoil damage on normalized angular panels.

 Split at the displacement threshold and recoil decades, double Gaussian
 orders to .001 eV + 1e-8 relative agreement, then refine incident-energy
 interpolation at three interior probes to .01 eV + 1e-5 relative error.
 These are sampled convergence criteria, with explicit resource limits.
 The historical fixed 25 eV elastic activation convention is retained.
 """
 from functools import lru_cache
 grid=np.asarray(grid,dtype=float)
 kink=_recoilKink(Z,A,Z,A);ed=_recoilThreshold(ed,Z,A,Z,A);scale=_damageScale();rt8=_relTol(1e-8);rt5=_relTol(1e-5)
 angular=getattr(form,'angular',None);isotropic=angular is None or type(angular).__name__=='Isotropic2d'
 if isotropic:energies=np.array([grid[0]]);tables=None;flat=False
 else:energies,tables,flat=_tables(angular)
 gauss={n:np.polynomial.legendre.leggauss(n) for n in (8,16,32,64)}
 data=[]
 for t in tables or []:
  x=np.asarray(t.xs);p=np.asarray(t.pdf if hasattr(t,'pdf') else t.ys);hist=_isFlat(t)
  norm=float(np.sum(np.diff(x)*(p[:-1] if hist else (p[:-1]+p[1:])/2)))
  if not norm>0:raise ValueError('angular damage requires a positive density mass')
  data.append((x,p,hist,norm))
 cutoff=25*(A+1)**2/(4*A) if elastic else 0.
 threshold=(A+1)/A*max(-Q,0.)
 @lru_cache(maxsize=30000)
 def at(E):
  if E<cutoff or E<threshold:return 0.
  r=np.sqrt(max(0.,1-threshold/max(E,1e-300)));c=E*A/(A+1)**2;offset=c*(1+r*r);slope=2*c*r
  if slope==0:return float(lindhard(offset,Z,A,Z,A,ed))
  last=min(1.,(offset-ed)/slope)
  if last<=-1:return 0.
  j=int(np.clip(np.searchsorted(energies,E,side='right')-1,0,len(energies)-1));k=min(j+1,len(energies)-1)
  f=float(np.clip((E-energies[j])/(energies[k]-energies[j]),0,1)) if k!=j and not flat else 0.
  rcuts=np.r_[max(ed,1e-12)*10.**np.arange(13),kink];mcuts=(offset-rcuts)/slope;mcuts=mcuts[(mcuts>-1)&(mcuts<last)]
  value=0.
  for idx,weight in ((j,1-f),(k,f)):
   if weight==0:continue
   if isotropic:x=np.array([-1.,1.]);p=np.array([.5,.5]);hist=False;norm=1.
   else:x,p,hist,norm=data[idx]
   cuts=np.unique(np.r_[-1.,x[(x>-1)&(x<last)],mcuts,last])
   previous=None
   for n in (8,16,32,64):
    u,w=gauss[n]
    integral=_twoBodyChunk(cuts,u,w,x,p,offset,slope,(Z,A,Z,A),ed,hist)/norm
    if previous is not None and abs(integral-previous)<=.001*scale+rt8*abs(integral):break
    previous=integral
   else:raise ValueError('two-body damage angular quadrature did not converge')
   value+=weight*integral
  return value
 lo,hi=float(grid.min()),float(grid.max())
 if hi==lo:return np.full(len(grid),at(lo))
 nodes=np.unique(np.r_[lo,hi,energies[(energies>lo)&(energies<hi)],_geometric(lo,hi,step=2.),cutoff,threshold]);nodes=nodes[(nodes>=lo)&(nodes<=hi)]
 for edge in (cutoff,threshold):
  if lo<edge<=hi:nodes=np.unique(np.r_[nodes,np.nextafter(edge,-np.inf)])
 values=_evaluateAll(at,nodes)
 for _ in range(30):
  additions={}
  _evaluateAll(at,[float(a+(b-a)*fraction) for a,b in zip(nodes[:-1],nodes[1:]) for fraction in (.25,.5,.75)])
  for a,b,va,vb in zip(nodes[:-1],nodes[1:],values[:-1],values[1:]):
   for fraction in (.25,.5,.75):
    e=float(a+(b-a)*fraction)
    if not a<e<b:continue
    v=at(e)
    if abs(v-(va+(vb-va)*fraction))>.01*scale+rt5*abs(v):additions[e]=v
  if not additions:return np.interp(grid,nodes,values)
  if len(nodes)+len(additions)>20000:raise ValueError('two-body damage interpolation exceeds resource cap')
  known=dict(zip(nodes,values));known.update(additions);nodes=np.array(sorted(known));values=np.array([known[e] for e in nodes])
 raise ValueError('two-body damage interpolation did not converge')


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
    return _tabulatedEmissionDamage(spectrum, grid, A, Z, ed)


def _emissionChunk(cuts,g,w,mu,wm,x,p,low,width,oldwidth,incident,denominator,ed,partition,hist,lab):
    """One chunk of :func:`_tabulatedEmissionDamage`'s quadrature: the panels between
    ``cuts`` (outgoing energies; Gauss nodes ``g``/``w`` in sqrt(E')) and, unless
    ``lab``, the recoil angle (nodes ``mu``/``wm``), each recoil through
    :func:`lindhard` with ``partition = (Zr, Ar, Zl, Al)``. The native kernel does
    it operation for operation; :func:`_emissionChunkNumpy` is the definition."""
    if _heating_native is None:
        return _emissionChunkNumpy(cuts,g,w,mu,wm,x,p,low,width,oldwidth,incident,denominator,ed,partition,hist,lab)
    noRecoil=partition[0]==0
    el,fl=(1.,1.) if noRecoil else (float(c) for c in _lindhardConstants(*partition))
    # The kernel reads C-contiguous doubles; a model table's xs can be a strided
    # view (every ENDF-decoded MF6 recoil spectrum: Fe-56 B-VIII.1 refused).
    cuts,g,w,mu,wm,x,p=(np.ascontiguousarray(a,dtype=float) for a in (cuts,g,w,mu,wm,x,p))
    return _heating_native.emission_chunk(cuts,g,w,mu,wm,x,p,float(low),float(width),float(oldwidth),
        float(incident),float(denominator),float(denominator*ed),el,fl,float(ed),bool(hist),bool(lab),noRecoil,
        *_nativeModel())


def _emissionChunkNumpy(cuts,g,w,mu,wm,x,p,low,width,oldwidth,incident,denominator,ed,partition,hist,lab):
    aa=np.sqrt(cuts[:-1]);bb=np.sqrt(cuts[1:]);half=(bb-aa)/2
    v=(aa[:,None]+bb[:,None])/2+half[:,None]*g;ep=v*v
    source=x[0]+(ep-low)*oldwidth/width
    if hist:
        pos=np.clip(np.searchsorted(x,source,side='right')-1,0,len(x)-2);density=p[pos]
    else:density=np.interp(source,x,p)
    if lab:
        # In the laboratory frame the particle energy is angle independent.
        angular=lindhard(ep,*partition,ed)
    else:
        root=np.sqrt(incident*ep);last=np.clip(np.divide(incident+ep-denominator*ed,2*root,out=np.ones_like(ep),where=root>0),-1.,1.)
        angle=(last[...,None]-1)/2+(last[...,None]+1)/2*mu
        recoil=np.maximum(0.,(incident+ep[...,None]-2*root[...,None]*angle)/denominator)
        angular=np.sum(wm*lindhard(recoil,*partition,ed),axis=-1)*(last+1)/4
    return float(np.sum(w*half[:,None]*2*v*density*oldwidth/width*angular))


def _tabulatedEmissionDamage(spectrum,grid,A,Z,ed,*,particle=None):
    """Controlled tabulated recoil or particle damage with actual incident energy.

    The isotropic single-emission recoil approximation is retained. Gaussian
    orders double after splitting displacement-threshold crossings. Successive
    orders must agree to .01 eV + 1e-5 relative; three interior interpolation
    probes must agree to .05 eV + 1e-4 relative. These allocate a fraction of
    the 1 eV + 1e-3 validation budget, with explicit resource caps. The sampled
    convergence criterion is not a uniform mathematical error certificate.
    """
    from functools import lru_cache
    grid=np.asarray(grid,dtype=float)
    if not len(grid):return np.empty(0)
    # A particle uses its own Z/mass and an optional isotropic CM boost;
    # a neutron emission instead uses the target recoil divided by A.
    if particle is None:recoilZ,recoilA,denominator,incidentFactor=Z,A,A,1.
    else:recoilZ,recoilA,incidentFactor=particle;denominator=1.
    kink=_recoilKink(recoilZ,recoilA,Z,A);ed=_recoilThreshold(ed,recoilZ,recoilA,Z,A);scale=_damageScale()
    rt5,rt4=_relTol(1e-5),_relTol(1e-4)
    energies,tables,flat=_tables(spectrum)
    if any(getattr(t,'numberOfDiscrete',0) for t in tables):
        raise NotImplementedError('tabulated neutron recoil with discrete outgoing lines')
    prepared=[]
    for t in tables:
        x=np.asarray(t.xs,dtype=float);p=np.asarray(t.pdf if hasattr(t,'pdf') else t.ys,dtype=float)
        hist=_isFlat(t);width=x[-1]-x[0]
        norm=float(np.sum(np.diff(x)*(p[:-1] if hist else (p[:-1]+p[1:])/2)))
        if not width>0 or not norm>0:raise ValueError('tabulated recoil spectrum requires positive width and mass')
        prepared.append((x,p/norm,hist,width))
    gauss={n:np.polynomial.legendre.leggauss(n) for n in (8,16,32,64,128)}
    lab=particle is not None and incidentFactor==0.
    @lru_cache(maxsize=30000)
    def at(E):
        j=int(np.clip(np.searchsorted(energies,E,side='right')-1,0,len(tables)-1));k=min(j+1,len(tables)-1)
        f=float(np.clip((E-energies[j])/(energies[k]-energies[j]),0.,1.)) if j!=k and not flat else 0.
        low=(1-f)*prepared[j][0][0]+f*prepared[k][0][0]
        width=(1-f)*prepared[j][3]+f*prepared[k][3]
        previous=None
        for order in (8,16,32,64):
            g,w=gauss[order];mu,wm=gauss[order*2];value=0.
            for i,weight in ((j,1-f),(k,f)):
                if weight==0:continue
                x,p,hist,oldwidth=prepared[i]
                mapped=low+(x-x[0])*width/oldwidth
                incident=incidentFactor*E
                cuts=np.unique(np.r_[mapped,[(np.sqrt(incident)+sign*np.sqrt(denominator*r))**2
                                             for r in [ed,*kink] for sign in (-1.,1.)]])
                cuts=cuts[(cuts>=mapped[0])&(cuts<=mapped[-1])]
                for start in range(0,len(cuts)-1,256):
                    stop=min(start+256,len(cuts)-1)
                    value+=weight*_emissionChunk(cuts[start:stop+1],g,w,mu,wm,x,p,low,width,oldwidth,incident,denominator,ed,
                                                 (recoilZ,recoilA,Z,A),bool(hist),lab)
            if previous is not None and abs(value-previous)<=.01*scale+rt5*abs(value):return value
            previous=value
        raise ValueError('tabulated recoil damage quadrature did not converge')
    lo=float(grid.min());hi=float(grid.max())
    if lo==hi:return np.full(len(grid),at(lo))
    nodes=np.unique(np.r_[lo,hi,energies[(energies>lo)&(energies<hi)],_geometric(lo,hi,step=4.)]);nodes=nodes[(nodes>=lo)&(nodes<=hi)]
    if flat:
        left=np.nextafter(energies[1:],-np.inf);nodes=np.unique(np.r_[nodes,left[(left>lo)&(left<hi)]])
    values=_evaluateAll(at,nodes)
    for iteration in range(30):
        additions={}
        _evaluateAll(at,[float(a+(b-a)*f) for a,b in zip(nodes[:-1],nodes[1:]) for f in (.25,.5,.75)])
        for a,b,va,vb in zip(nodes[:-1],nodes[1:],values[:-1],values[1:]):
            for f in (.25,.5,.75):
                E=float(a+(b-a)*f)
                if not a<E<b:continue
                value=at(E)
                if abs(value-(va+(vb-va)*f))>.05*scale+rt4*abs(value):additions[E]=value
        if not additions:return np.interp(grid,nodes,values)
        if len(nodes)+len(additions)>20000:raise ValueError('tabulated recoil damage interpolation exceeds resource cap')
        known=dict(zip(nodes,values));known.update(additions);nodes=np.array(sorted(known));values=np.array([known[E] for E in nodes])
    raise ValueError('tabulated recoil damage interpolation did not converge')

def _analyticEmissionDamage(spectrum, grid, A, Z, ed):
    """Integrate LF7/9 directly, retaining the parameter interpolation law.

    A display grid at parameter knots loses logarithmic incident dependence.
    Gaussian orders are doubled on sqrt(E'/theta), splitting the displacement
    threshold in angle. Incident interpolation is checked at three interior
    points; this is a sampled error criterion, with explicit resource caps.
    """
    from functools import lru_cache
    gauss = {n: np.polynomial.legendre.leggauss(n) for n in (16, 32, 64, 128)}
    kink = _recoilKink(Z, A, Z, A); ed = _recoilThreshold(ed, Z, A, Z, A); scale = _damageScale()
    rt7, rt5 = _relTol(1e-7), _relTol(1e-5)
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
            [abs(np.sqrt(E) + sign * np.sqrt(A * r)) / np.sqrt(theta)
             for r in [ed, *kink] for sign in (-1., 1.)]])
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
                    and abs(value - previous) <= .01 * scale + rt7 * abs(value)):
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
                if abs(value - (va + (vb - va) * f)) > .05 * scale + rt5 * abs(value):
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
    """Capture: ``df(E/(A+1)) + Σ_γ y_γ ⟨df(E_γ²/2tm)⟩``, or ``capdam``'s single photon.

    HEATR's ``gheat`` sums the damage of each photon's recoil as an independent
    primary and leaves out the compound's own motion, E/(A+1), which kika's
    KERMA carries since D49. Here the compound recoil is one more independent
    primary, in the same superposition: at thermal energies E/(A+1) is far below
    ``ed`` and the damage is HEATR's to the bit; above a few keV it is the
    compound recoil that displaces (Fe-56 at 1 MeV: 13.2 keV against gheat's
    0.38 keV). Folding every kick into one recoil of mean energy E/(A+1) + E_ph
    was measured and rejected: the mean of the photon kicks can sit below
    ``ed`` while the hard primaries do not (U-238 thermal capture damage went
    to zero; P3 plan §5, capture_damage_recoil.py).
    """
    tm = NEUTRON_REST_ENERGY * (A + 1.0)
    if not photons:
        er = grid / (A + 1.0)
        ea = A * er + Q
        return lindhard(er + ea * ea / (2.0 * tm), Z, A + 1.0, Z, A, ed)
    total = lindhard(grid / (A + 1.0), Z, A + 1.0, Z, A, ed)
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


#: id(form) -> (weak reference to the form, {context key: (table, error)}).
_HEAVY_TABULATIONS: Dict[int, tuple] = {}


def _tabulateHeavy(form, context):
    """``tabulateDistribution(form, context)`` once per evaluated form and context.

    A charged particle's tabulation depends on its evaluated law and the reaction,
    not on the temperature or the energy grid, and the heating of every
    temperature asks for it again: U-235's take ~46 s each time. The table is
    read, never changed, by the moments and the damage. A failure is kept too and
    raised again, so each temperature's notes are the same. The entry goes with
    the form (model forms are compared by value, so the key is their identity).
    """
    from kika.processing.tabulation import tabulateDistribution

    key = (context.awr, context.zaTarget, context.mt, context.Q, context.pid, context.zaEmitted,
           context.options)
    entry = _HEAVY_TABULATIONS.get(id(form))
    if entry is None or entry[0]() is not form:
        formId = id(form)

        def forget(reference, formId=formId):
            if _HEAVY_TABULATIONS.get(formId, (None,))[0] is reference:
                del _HEAVY_TABULATIONS[formId]

        entry = _HEAVY_TABULATIONS[formId] = (weakref.ref(form, forget), {})
    tables = entry[1]
    if key not in tables:
        try:
            tables[key] = (tabulateDistribution(form, context), None)
        except (TypeError, ValueError, NotImplementedError) as exc:
            # A copy carries no traceback: the frames would keep the form alive.
            tables[key] = (None, copy.copy(exc))
    tabulated, error = tables[key]
    if error is not None:
        raise copy.copy(error)
    return tabulated


def _heavyProducts(reaction, evaluation, grid, A, Z, zaTarget, ed, notes, damageOnly=False):
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
    from kika.nuclear_data.model.endf_conversion import reactionQI
    Q = float(reactionQI(source, evaluation) or 0.0)
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
            tabulated = _tabulateHeavy(form, context)
        except (TypeError, ValueError, NotImplementedError) as exc:
            notes.append(f"MT{reaction.ENDF_MT} {product.pid}: {type(form).__name__} not "
                         f"tabulated for the heating ({exc}); left out")
            continue
        if tabulated is None:
            continue
        awp = _massRatio(product.pid, evaluation.PoPs)
        k = _Kinematics(A=A, Q=Q, awp=awp)
        y = np.asarray(multiplicity.evaluate(grid), dtype=float)
        if not damageOnly:
            # The displacement passes need the damage alone; the deposit is MT444's pass's.
            try:
                mean = _meanEnergyOf(tabulated, grid, k)
            except (ValueError, NotImplementedError) as exc:
                notes.append(f"MT{reaction.ENDF_MT} {product.pid}: no mean energy ({exc}); left out")
                continue
            deposit = deposit + y * mean
        found = True
        active=y!=0
        if np.any(active):
            damage[active]+=y[active]*_particleDamage(tabulated,grid[active],k,za//1000,awp,Z,A,ed)
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
    if isinstance(spectrum,m.XYs2d):
        return _tabulatedEmissionDamage(spectrum,grid,A,Z,ed,particle=(zp,ap,x if cm else 0.))
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
    """Where any MF12 subsection emits: gheat's correction applies there, and
    the neutron balance stays where none does (D47).

    The last subsection alone is not the support: ENDF/B-VIII.1 Cl-35 MT102
    states 383, the last from 100 keV, and taking it left the whole capture
    Q deposited below (thermal KERMA x1163 against NJOY's 17035.10c).
    """
    domain=np.zeros(len(grid),dtype=bool)
    for photon in photons:
        domain|=_photonSupport(photon,grid)
    return domain


def _photonSupport(photon, grid):
    """One MF12 subsection's domain, as gheat reads it."""
    multiplicity=getattr(photon[0],"multiplicity",None)
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
    return ((grid>=lo) if lo is not None else np.ones(len(grid),dtype=bool)) & ((grid<=hi) if hi is not None else True) & (_yield(photon[0],grid)>0)


def _generatedMF6Recoil(evaluation, mt, form, grid, k):
    """HEATR's one-particle diagnostic when MF6 omits the residual.

    Product identity/law provenance selects this branch; both energy moments
    come from the canonical processed law. The generated residual has mass A,
    even for multiple neutron emission: this is HEATR's stated approximation.
    """
    from kika.nuclear_data import model as m
    source = evaluation.findReactionByENDF_MT(mt) if evaluation is not None else None
    fields = getattr(getattr(source,"provenance",None),"headerFields",None) or {}
    records = (fields.get("mf6") or {}).get("products",[])
    neutrons = [r for r in records if r.get("zap")==1]
    if (len(neutrons)!=1 or neutrons[0].get("law")!=1
            or any(r.get("zap") not in (0,1) for r in records)):
        return None
    if (not isinstance(form,(m.EnergyAngularMC,m.KalbachMann))
            or form.productFrame!=m.Frame.centerOfMass):
        return None
    lab = _unitBaseCMMean(form,grid,k)
    if lab is None:return None
    energy = form.f if isinstance(form,m.KalbachMann) else form.energy
    cm = _unitBaseSpectrumMean(energy,grid,1)
    recoil = grid/(k.A+1)+(k.A+1)/k.A*cm-lab
    if np.any(recoil < -(.001+1e-10*grid)):
        raise ValueError("generated MF6 recoil energy is negative")
    return np.maximum(recoil,0.)


def _kinematicUpperOf(mt, grid, A, Q, Qbal, emitted, neutronYield, carried,
                      deposit, heavy, mf6, lr, twoBodyCapture, photons, captureCompound=False, captureDomain=None):
    """HEATR nheat/gheat MT443 diagnostic, in eV per collision.

    The historical high limit is a diagnostic approximation, not a certified
    physical bound. MF6 capture uses its generated photon recoil; other MF6
    paths require explicit recoil energy.
    """
    if mt == 2:
        return carried
    if mf6:
        if mt == 102 and photons:
            # nheat generates the residual subsection when MF6 capture only
            # states photons: its hmax is the sum of independent photon
            # momentum kicks (hgam102 supplies the explicit primary case).
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
            return carried
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


def fissionPhotonEnergy(suite, grid, *, awr: float) -> Dict[int, np.ndarray]:
    """Prompt-photon energy per fission (eV) that a processed suite's tables carry.

    HEATR's fission KERMA is ``(EFR + EGP)·σ_f`` from MF1/458 less the photons the
    evaluation states, wherever it states them (``heatr.f90`` nheat, gheat): the
    release keeps authority, and only the photons actually transported leave.
    Fission's own photons are stated once, on MT18 (as a reaction or as the sum of
    its chances) or else on each chance; this gives ``{18: E}`` in the first case,
    ``{mt: E}`` per chance in the second, ``{}`` when fission states none (Pu-238 of
    ENDF/B-VIII.1: all of EGP stays local). Photons on MT3 are not fission's here:
    :func:`heatingOf` takes them out of MT301 as any photons stated on a sum.
    Delayed photons are left out, as EGD is not in MT301.
    """
    grid = np.asarray(grid, dtype=float)
    photon = _Kinematics(A=float(awr), Q=0.0, awp=0.0)
    fissions = [r for c in (suite.reactions, getattr(suite, "fissionComponents", ()),
                            getattr(suite, "sums", ()))
                for r in c if r.ENDF_MT in FISSION_MTS]
    found: Dict[int, np.ndarray] = {}
    for reaction in sorted(fissions, key=lambda r: r.ENDF_MT != 18):
        energy, stated = np.zeros(len(grid)), False
        for product in reaction.outputChannel.products:
            if product.pid != "photon":
                continue
            provenance = getattr(product, "provenance", None)
            mode = (getattr(provenance, "attributes", None) or {}).get("emission_mode", "prompt")
            form = _distribution(product) if mode == "prompt" else None
            if form is not None:
                energy = energy + _yield(product, grid) * _meanEnergyOf(form, grid, photon)
                stated = True
        if stated:
            found[reaction.ENDF_MT] = energy
            if reaction.ENDF_MT == 18:
                break
    return found


def fissionPhotonKerma(suite, grid, sigma: Dict[int, np.ndarray], *, awr: float) -> np.ndarray:
    """``Σ σ·E_γ`` (eV·b) of :func:`fissionPhotonEnergy` on ``grid``: MT18's photons
    with the whole fission cross section, a chance's with its own."""
    energies = fissionPhotonEnergy(suite, grid, awr=awr)
    term = np.zeros(len(np.asarray(grid)))
    if 18 in energies:
        sigmaF = sigma.get(18)
        if sigmaF is None:
            sigmaF = np.sum([sigma[mt] for mt in FISSION_MTS if mt != 18 and mt in sigma], axis=0)
        return np.where(sigmaF > 0, sigmaF * energies[18], 0.0)
    for mt, energy in energies.items():
        if mt in sigma:
            term = term + np.where(sigma[mt] > 0, sigma[mt] * energy, 0.0)
    return term


def heatingOf(suite, grid, sigma: Dict[int, np.ndarray], *, workers: Optional[int] = None,
              **options) -> HeatingResult:
    """Run :func:`_heatingOf` with its damage quadratures on ``workers`` threads.

    ``workers`` is the processing's own count; when not given, ``KIKA_WORKERS``
    or ``min(cpus, 8)``, as for the Doppler broadening (:mod:`kika.processing.workers`);
    1 keeps every energy on this thread. The result does not depend on it.
    """
    from kika.processing.workers import pool_default, resolve
    count = resolve(workers, pool_default())
    token = _energyThreads.set(count)
    try:
        return _heatingOf(suite, grid, sigma, **options)
    finally:
        _energyThreads.reset(token)


def _heatingOf(suite, grid, sigma: Dict[int, np.ndarray], *, awr: float, Z: int,
              qm: Optional[Dict[int, float]] = None, ed: Optional[float] = None,
              qa: Optional[Dict[int, float]] = None, qbar: Optional[Dict[int, object]] = None,
              evaluation=None, zaTarget: Optional[int] = None,
              report=None, kinematic: bool = False, displacements: bool = False) -> HeatingResult:
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
    ``displacements=True`` adds the NRT and arc-dpa displacements
    (:attr:`HeatingResult.displacements`): one more pass of the damage quadratures
    per model, so only the outputs that carry them ask for them.
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
    #: (mt, sigma, live, damage function, shared key) of every damage term.
    passes: List[tuple] = []
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
            else:
                # The compound nucleus keeps the neutron's momentum whatever file
                # states the photons (D49). HEATR's tabsq6 drops it for MF6, so a
                # GNDS evaluation, which never says MF12, lost it too (Cl-35 at
                # 22.4 keV: KERMA 16 % below NJOY's, which went through gheat).
                carried = carried + grid / (A + 1.0)
                notes.append("MT102: compound recoil E/(A+1) added to the MF6 photon kicks; "
                             "HEATR's tabsq6 leaves it out")
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
        # damage: MT444 now, and again under each displacement model below
        def reactionDamage(mt=mt, Q=Q, Qkin=Qkin, neutrons=neutrons, photons=photons,
                           twoBodyCapture=twoBodyCapture, carried=carried, heavy=heavy,
                           captureCompound=captureCompound, captureDomain=captureDomain,
                           reaction=reaction):
            if mt == 2 or (51 <= mt <= 90 and neutrons):
                return _twoBodyDamage(neutrons[0][1], grid, A, Q, Z, ed, elastic=(mt == 2)) if neutrons \
                    else np.zeros(len(grid))
            if mt == 102:
                if photons and twoBodyCapture is not None:
                    y = _yield(photons[0][0], grid)
                    recoil = np.divide(carried, y, out=np.zeros_like(carried), where=y > 0)
                    return y * lindhard(recoil, Z, A + 1, Z, A, ed)
                d = _captureDamage(photons, grid, A, Z, ed, _qAt(Qkin, grid), fixedPrimary=captureCompound)
                if captureCompound and photons:
                    fallback=_captureDamage([],grid,A,Z,ed,_qAt(Qkin,grid))
                    d=d+np.where(captureDomain,0.,fallback)
                return d
            if heavy is not None:
                if _DISPLACEMENTS.get() is None:
                    return heavy[1]
                again = _heavyProducts(reaction, evaluation, grid, A, Z, zaTarget or 1000 * Z, ed, [], damageOnly=True)
                return again[1] if again is not None else np.zeros(len(grid))
            if neutrons:
                d = np.zeros(len(grid))
                for product, form in neutrons[:1]:
                    d = _emissionDamage(form, grid, A, Z, ed)
                return d
            if mt > 102:
                return _absorptionDamage(mt, grid, A, Z, ed, _qAt(Qkin, grid))
            return np.zeros(len(grid))
        d = reactionDamage()
        passes.append((mt, s, live, reactionDamage, None))
        damage += np.where(live, s * d, 0.0)
        perReaction[mt] = (carried, carried + exact, d)
        photonEnergy[mt] = exact
        if kinematic:
            fields = getattr(getattr(reaction, "provenance", None), "headerFields", None) or {}
            mf6 = _statedInMF6(evaluation, mt) or bool(fields.get("mf6"))
            lr = int(getattr(getattr(reaction, "provenance", None), "lr", 0) or 0)
            upper = _kinematicUpperOf(mt, grid, A, _qAt(Qkin, grid), Qbal, emitted,
                neutronYield, carried, deposit, heavy, mf6, lr, twoBodyCapture, photons, captureCompound, captureDomain)
            if upper is None and mf6 and len(neutrons)==1:
                upper = _generatedMF6Recoil(evaluation,mt,neutrons[0][1],grid,k)
                if upper is not None:
                    notes.append(f"MT{mt}: MT443 uses HEATR's generated one-particle MF6 recoil approximation")
            if upper is not None:
                kinematicUpper[mt] = upper
    # Fission's damage: HEATR's conbar gives a fission the recoil of one
    # emission from its prompt-neutron spectrum (tabdam, one neutron, not ν̄;
    # the fragments are not followed): U-235 at thermal, ~6.9 keV per fission.
    # MT444 keeps it -- unlike the KERMA, whose fission part comes from MF1/458.
    chances = [r for r in getattr(suite, "fissionComponents", ()) if r.ENDF_MT in FISSION_MTS]
    if not chances:
        chances = [r for r in suite.reactions if r.ENDF_MT in FISSION_MTS and r.ENDF_MT != 18]
    fissions = chances or [r for r in suite.reactions if r.ENDF_MT == 18]
    aggregate = next((r for r in getattr(suite, "sums", ()) if r.ENDF_MT == 18), None)
    commonPrompt = (next(((p, _distribution(p)) for p in aggregate.outputChannel.products
                         if p.pid == "n" and _distribution(p) is not None), None)
                    if aggregate is not None else None)
    fissionDamage = {}
    for reaction in fissions:
        mt = reaction.ENDF_MT
        if mt not in sigma:
            continue
        prompt = next(((p, _distribution(p)) for p in reaction.outputChannel.products
                       if p.pid == "n" and _distribution(p) is not None), None)
        if prompt is None and commonPrompt is not None:
            prompt = commonPrompt
            notes.append(f"MT{mt}: fission damage uses the common MT18 prompt spectrum")
        if prompt is None:
            continue
        s = np.asarray(sigma[mt], dtype=float)
        key = id(prompt[1])
        if key not in fissionDamage:
            fissionDamage[key] = _emissionDamage(prompt[1], grid, A, Z, ed)
        d = fissionDamage[key]
        passes.append((mt, s, s > 0, lambda form=prompt[1]: _emissionDamage(form, grid, A, Z, ed), key))
        damage += np.where(s > 0, s * d, 0.0)
        perReaction[mt] = (np.zeros(len(grid)), np.zeros(len(grid)), d)
    # Photons stated on a sum (MF13 on MT3, MT4, 103-107): carried away for MT301
    # and deposited for MT901, as HEATR's gheat does with σ_sum·y·Ē (= σ_γ·Ē).
    # A summed MT18's own photons are fission's: fissionPhotonEnergy takes them,
    # against the release's EGP, so they are not taken out here a second time.
    for reaction in getattr(suite, "sums", ()):
        mt = reaction.ENDF_MT
        if mt is None or mt not in sigma or mt in FISSION_MTS:
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
    # Displacements: the same damage terms, with each model's N_d(T_d) for each
    # recoil in place of T_d (NRT always; arc-dpa where Nordlund's constants exist).
    wanted = displacements
    displacements: Dict[str, np.ndarray] = {}
    displacementsPerReaction: Dict[str, Dict[int, np.ndarray]] = {}
    models = displacementModels(Z, ed) if wanted else {}

    def displacementPass(model):
        _DISPLACEMENTS.set(model)
        total, per, shared = np.zeros(len(grid)), {}, {}
        for mt, s, live, function, key in passes:
            if key is None:
                n = function()
            else:
                if key not in shared:
                    shared[key] = function()
                n = shared[key]
            per[mt] = n
            total += np.where(live, s * n, 0.0)
        return total, per

    # One pass per model, in a copy of this context (the energy threads already
    # share the work; side-by-side passes measured no faster, PXB-045).
    for name, model in models.items():
        total, per = contextvars.copy_context().run(displacementPass, model)
        displacements[name], displacementsPerReaction[name] = total, per
    if models:
        notes.append("displacements: " + "; ".join(
            f"{name} with E_d = {m_.ed:g} eV" + (f", b = {m_.b:g}, c = {m_.c:g} (Nordlund et al. 2018)"
                                                 if m_.mode == 2 else "") for name, m_ in models.items()))
    if models and "arc-dpa" not in models:
        notes.append(f"no arc-dpa: Nordlund et al. 2018 give its constants for Fe, Ni, Cu, Pd, Pt "
                     f"and W only, not Z = {int(Z)}")
    if report is not None:
        for note in notes:
            report.warn(note)
    return HeatingResult(grid=grid, kerma=kerma, kermaLocal=local, damage=damage, notes=notes,
                         perReaction=perReaction, photonEnergy=photonEnergy, kinematicUpper=kinematicUpper,
                         displacements=displacements, displacementsPerReaction=displacementsPerReaction)
