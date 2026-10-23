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
  isotropic emission over the prompt spectrum for fission (``tabdam``), the photon
  recoils for capture (``gheat``), particle emission up to the Coulomb barrier
  for absorption (``capdam``).
- **one-particle recoil**: the residual of an MF6 section that lists its light
  particles (p, d, t, ³He, α) and not the residual, or states it only as a
  two-body recoil (LAW=4), recoils with the momentum of the heaviest of them,
  averaged over that particle's own energy-angle law (HEATR's ``hinit``/``getsix``
  with ``irec > 0``). It is exact for one emitted particle and for a level; HEATR
  takes the recoil at the mean lab cosine of each lab energy (``h6cm``), kika
  averages the damage over the angle.
- **independent recoil** (beyond HEATR): when such a section emits more than one
  particle (MT22, 24, 28, ..., neutrons included, whole yields), the residual takes
  the momentum balance of all of them, each drawn from its own lab energy-angle
  law with independent azimuths. The evaluation gives no correlation, so this is
  a model; against the residuals ENDF/B-VIII.1 does state it is closer than
  HEATR's one particle in every MT28 measured and equal in MT22 of medium-mass
  targets.
- **neutron recoil**: a section that states its neutrons and not the residual
  (MT16, 17, 91, ...) recoils the same way, from all its neutrons (mass A+1−n),
  exact for one. HEATR recoils an MF6 section's residual from one neutron with the
  target's mass at the mean lab cosine (``h6cm``), and an MF4/MF5 one by ``tabdam``,
  one isotropic lab emission; ``tabdam`` on an MF6 section is not HEATR's number
  (Ni-60 MT91 at 14 MeV 6 % above it).

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
fallback for a section without them (the residual's recoil from the neutrons,
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
    "damage of fission: one isotropic emission from the prompt spectrum in the lab, the "
    "target's Z and A for the recoil, the fragments not followed (HEATR's tabdam)",
    "damage of particle emission stated by MF3 alone: capdam's single particle at "
    "min(Q + AE/(A+1), Coulomb barrier)",
)

from kika._native import load as _load_native
# The NumPy reference path stays the definition.
_heating_native = _load_native("kika.processing._heating_native", 6)

#: Threads for the independent incident energies of a damage quadrature: set by
#: heatingOf from the processing's ``workers`` for the length of one call.
_energyThreads = contextvars.ContextVar("kika_heating_threads", default=1)

#: A process pool kika.hdf5.processing.processSuite opens for the length of one call, on
#: which the tabulations that are pure Python start early: the charged particles' here
#: (:func:`_prefetchHeavy`), the Monte Carlo products' there. ``None`` outside it.
_PREFETCH = contextvars.ContextVar("kika_processing_prefetch", default=None)


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


def _evaluateAllRows(at, energies):
    """:func:`_evaluateAll` for an ``at`` that returns a row (a joint quadrature)."""
    energies = [float(E) for E in energies]
    distinct = list(dict.fromkeys(energies))
    threads = _energyThreads.get()
    if threads > 1 and len(distinct) > 1:
        from concurrent.futures import ThreadPoolExecutor
        contexts = [contextvars.copy_context() for _ in distinct]
        with ThreadPoolExecutor(max_workers=min(threads, len(distinct))) as pool:
            for _ in pool.map(lambda job: job[0].run(at, job[1]), zip(contexts, distinct)):
                pass
    return [at(E) for E in energies]


def _threadMap(function, items) -> list:
    """``[function(item) for item in items]``, the items computed on the processing's
    threads (:data:`_energyThreads`). Each item is independent and its heavy part runs
    in the native kernel without the GIL, so the list is the sequential one, item for
    item, whatever thread computed it; an item that fails raises as it would there."""
    items = list(items)
    threads = _energyThreads.get()
    if threads > 1 and len(items) > 1:
        from concurrent.futures import ThreadPoolExecutor
        contexts = [contextvars.copy_context() for _ in items]
        with ThreadPoolExecutor(max_workers=min(threads, len(items))) as pool:
            return list(pool.map(lambda job: job[0].run(function, job[1]), zip(contexts, items)))
    return [function(item) for item in items]


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


def _lindhardPowers(ep):
    """``(ep^(1/6), ep^(3/4))`` of :func:`lindhard`'s partition, as sqrt(cbrt(ep)) and
    s*sqrt(s) with s = sqrt(ep): the same powers to an ulp, at about half the cost of two
    ``pow`` (sqrt is exact; the native kernel takes the same cbrt)."""
    s = np.sqrt(ep)
    return np.sqrt(np.cbrt(ep)), s * np.sqrt(s)


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
        sixth, threeQuarters = _lindhardPowers(ep)
        out = E / (1.0 + fl * (3.4008 * sixth + 0.40244 * threeQuarters + ep))
        if model is not None:
            return model.count(out)
        return np.where(E >= ed, out, 0.0)
    # The same expression in place, in its order of operations: the damage
    # quadratures call this on millions of recoils at a time, and the
    # temporaries cost a fifth of it.
    ep = np.maximum(E, 0.0)
    ep /= el
    out, term = _lindhardPowers(ep)
    out *= 3.4008
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


def _expect(table, g: Callable[[np.ndarray], np.ndarray]) -> float:
    """``∫ g(x) p(x) dx`` over an ``Xs_pdf_cdf1d`` or ``XYs1d`` (lines first, then the continuum), normalised.

    The continuum is integrated on its own law, not by trapezoids over ``g·p``
    at the tabulated points as HEATR's ``getsix`` does for MF6 spectra in the
    lab (U-235 MT91's photons at 5.382 MeV: 1.473 MeV by trapezoids, 1.515
    exactly; the trapezoids were replaced in 78693cf).
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
    photon: bool = False   # the projectile is a photon (else a neutron)

    def boost(self, E):
        """The centre of mass's kinetic energy per emitted particle, ``m_b v_cm²/2`` (eV), at ``E``.

        A neutron: ``awp E/(A+1)²``. A photon, which brings momentum ``E/c`` and no
        mass: ``½ awp m_n c² β²`` with ``β = E/(E + A m_n c²)``, exact for the
        centre of mass's velocity.
        """
        if self.photon:
            beta = np.asarray(E, dtype=float) / (np.asarray(E, dtype=float) + self.A * NEUTRON_REST_ENERGY)
            return 0.5 * self.awp * NEUTRON_REST_ENERGY * beta * beta
        return self.awp / (self.A + 1.0) ** 2 * E


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
    """``⟨E'_lab⟩ = ⟨E'_cm⟩ + E_c + 2√E_c⟨√E'_cm μ⟩``, ``E_c = k.boost(E)`` (``awp E/(A+1)²`` for a neutron)."""
    ec = k.boost(grid)
    return ecm + ec + 2.0 * np.sqrt(ec) * sqrtMu


def _meanMu(form, grid):
    if form is None or type(form).__name__ == "Isotropic2d":
        return np.zeros(len(grid))
    energies, tables, flat = _tables(form)
    return _onGrid(energies, [_expect(t, lambda m: m) for t in tables], grid, flat)


_crtTanh = np.vectorize(__import__("math").tanh, otypes=[float])


def _unitBaseIntegral(x, p, angular, nodes, ww, a, width, qa, qw, flat, kalbach):
    """One quadrature of :func:`_unitBaseCMMean`'s cross moment: the native kernel when it is
    built, :func:`_unitBaseIntegralNumpy` (the definition) otherwise."""
    if _heating_native is None:
        return _unitBaseIntegralNumpy(x, p, angular, nodes, ww, a, width, qa, qw, flat, kalbach)
    vector = lambda v: np.ascontiguousarray(v, dtype=float)  # noqa: E731
    rr, slope = angular if kalbach else (angular, angular)
    return _heating_native.unit_base_integral(vector(x), vector(p), vector(rr), vector(slope), vector(nodes),
                                              vector(ww), float(a), float(width), float(qa), float(qw),
                                              bool(flat), bool(kalbach))


def _unitBaseIntegralNumpy(x, p, angular, nodes, ww, a, width, qa, qw, flat, kalbach):
    mapped=qa+qw*(x-a)/width
    root=np.sqrt(np.maximum(mapped,0));left=root[:-1,None];right=root[1:,None]
    r=(left+right)/2+(right-left)/2*nodes
    original=a+(r*r-qa)*width/qw
    frac=(original-x[:-1,None])/np.maximum(np.diff(x)[:,None],1e-300)
    if flat:
        density=p[:-1,None]
        norm=float(np.sum(np.diff(x)*p[:-1]))
    else:
        density=p[:-1,None]+np.diff(p)[:,None]*frac
        norm=float(np.sum(np.diff(x)*(p[:-1]+p[1:])/2))
    if kalbach:
        rr,aa=angular
        if flat:
            rr=rr[:-1,None];aa=aa[:-1,None]
        else:
            rr=rr[:-1,None]+np.diff(rr)[:,None]*frac
            aa=aa[:-1,None]+np.diff(aa)[:,None]*frac
        safe=np.where(abs(aa)>.01,aa,1.)
        # tanh of the C runtime (math.tanh), as the native kernel takes it: NumPy's own
        # vectorised tanh differs from it in the last bit.
        langevin=np.where(abs(aa)>.01,1/_crtTanh(safe)-1/safe,
                          aa*(1/3+aa*aa*(-1/45+aa*aa*(2/945-aa*aa/4725))))
        angle=rr*langevin
    else:
        mu=angular
        angle=(mu[:-1,None] if flat else mu[:-1,None]+np.diff(mu)[:,None]*frac)
    return float(np.sum((right-left)/2*ww*2*r*r*width/qw*density*angle))/norm


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
        boost=k.boost(query)
        return mean+boost+2*np.sqrt(boost)*cross
    @lru_cache(maxsize=30000)
    def at(E, components=False):
        j=int(np.clip(np.searchsorted(energies,E,side="right")-1,0,len(energies)-1))
        upper=min(j+1,len(energies)-1)
        # a repeated energy (a step) has no width: its first table, as the vector form above
        theta=(E-energies[j])/(energies[upper]-energies[j]) if energies[upper]>energies[j] else 0.
        theta=float(np.clip(theta,0,1));weights=(1-theta,theta)
        qa=sum(w*ranges[i][0] for w,i in zip(weights,(j,upper)))
        qw=sum(w*ranges[i][1] for w,i in zip(weights,(j,upper)))
        mean=cross=0.
        for weight,i in zip(weights,(j,upper)):
            if weight==0:continue
            a,width=ranges[i];table=tables[i];x=np.asarray(table.xs);p=np.asarray(table.pdf)
            previous=None
            for order,(nodes,ww) in gauss.items():
                integral=_unitBaseIntegral(x,p,parameters[i] if kalbach else mus[i],nodes,ww,a,width,qa,qw,
                                           _isFlat(table),kalbach)
                if not kalbach:break
                error=(2*np.sqrt(k.boost(max(E,0))) if k.photon else 2*np.sqrt(k.awp*max(E,0))/(k.A+1))*abs(integral-previous) if previous is not None else np.inf
                if error <= .001+1e-8*max(E,0):break
                previous=integral
            else:raise ValueError("Kalbach cross-moment quadrature did not converge")
            cross+=weight*integral
            mean+=weight*(qa+qw*(means[i]-a)/width)
        if components:return mean,cross
        boost=k.boost(E)
        return mean+boost+2*np.sqrt(boost)*cross
    lo=max(float(np.min(grid)),float(energies[0]));hi=min(float(np.max(grid)),float(energies[-1]))
    def held(query, first):
        source=float(np.nextafter(energies[0],-np.inf)) if first else float(energies[-1])
        mean,cross=at(source,True)
        boost=k.boost(query)
        return mean+boost+2*np.sqrt(boost)*cross
    if hi<lo:return held(np.asarray(grid),bool(np.max(grid)<energies[0]))
    # An incident energy stated twice is a step (JEFF-4.0 B-11 MT16, two tables at
    # 20 MeV): each side is refined on its own, the left one up to the step's left
    # limit, and the step's own energy takes the right side's value, as at() does.
    unique,counts=np.unique(energies,return_counts=True)
    steps=[float(e) for e,c in zip(unique,counts) if c>1 and lo<e<hi]
    bounds=[lo,*steps,hi]
    meshes=[];total=0
    for a0,b0 in zip(bounds[:-1],bounds[1:]):
        nodes=np.unique(np.r_[a0,energies[(energies>a0)&(energies<b0)],b0])
        right=float(np.nextafter(b0,-np.inf)) if b0 in steps else float(b0)
        def value(E,right=right,b0=b0):
            return at(right) if E==b0 else at(float(E))
        values=np.array([value(float(E)) for E in nodes])
        for iteration in range(30):
            additions=[]
            for a,b,va,vb in zip(nodes[:-1],nodes[1:],values[:-1],values[1:]):
                if b==a:continue
                for fraction in (.25,.5,.75):
                    E=a+(b-a)*fraction;v=at(float(E))
                    if abs(v-(va+(vb-va)*fraction))>.1+1e-7*E:additions.append((E,v))
            if not additions:break
            if total+len(nodes)+len(additions)>20000:raise ValueError("unit-base mean refinement exceeds capacity")
            # Each new node was already integrated while checking the error.
            # Keep those exact values and the previous nodes' values; do not
            # repeat quadrature on the enlarged mesh.
            known = dict(zip(nodes, values))
            known.update(additions)
            nodes = np.array(sorted(known))
            values = np.array([known[E] for E in nodes])
        else:raise ValueError("unit-base mean refinement did not converge")
        meshes.append((nodes,values));total+=len(nodes)
    side=np.searchsorted(np.asarray(steps,dtype=float),grid,side="right")
    output=np.empty(len(grid))
    for index,(nodes,values) in enumerate(meshes):
        mask=side==index
        output[mask]=np.interp(np.asarray(grid)[mask],nodes,values)
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
        mean = _onGrid(energies, [_expect(t, lambda e: e ** moment) for t in tables], grid, flat)
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
        mean = _onGrid(energies, [_expect(t, lambda e: e ** moment) for t in tables], grid, flat)
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

#: The displacement models the MT444 pass computes alongside it (joint quadratures),
#: and where it keeps their values for the models' own passes to take: set by
#: _heatingOf when displacements are asked for. A quadrature that has a joint form
#: then evaluates the damage energy once per recoil for all of them.
_JOINT = contextvars.ContextVar("kika_joint_displacements", default=None)
_JOINT_STORE = contextvars.ContextVar("kika_joint_store", default=None)


def _jointModels() -> tuple:
    """The models to compute jointly now: only on MT444's pass, when asked."""
    if _DISPLACEMENTS.get() is not None:
        return ()
    return _JOINT.get() or ()


def _jointKey(name, grid, *parts):
    grid = np.ascontiguousarray(grid, dtype=float)
    return (name, len(grid), hash(grid.tobytes())) + tuple(parts)


def _jointRecall(key):
    """A model pass's value, kept by MT444's joint quadrature (None if there is none)."""
    model, store = _DISPLACEMENTS.get(), _JOINT_STORE.get()
    if model is None or store is None or key not in store:
        return None
    return store[key].get(model.name)


def _jointKeep(key, models, values):
    store = _JOINT_STORE.get()
    if store is not None:
        store[key] = {m.name: np.array(values[:, 1 + k]) for k, m in enumerate(models)}


def _jointPacked(models) -> tuple:
    """(mode, ed, b, c) of up to two models for the joint native kernels."""
    packed = []
    for m_ in list(models)[:2]:
        packed += [m_.mode, float(m_.ed), float(m_.b), float(m_.c)]
    while len(packed) < 8:
        packed += [0, 0.0, 0.0, 1.0]
    return tuple(packed)


def _damageEnergy(E, Zr, Ar, Zl, Al):
    """:func:`lindhard`'s partition with no threshold, in its vector order of operations."""
    E = np.atleast_1d(np.asarray(E, dtype=float))
    if Zr == 0:
        return np.zeros_like(E)
    el, fl = _lindhardConstants(Zr, Ar, Zl, Al)
    ep = np.maximum(E, 0.0)
    ep /= el
    out, term = _lindhardPowers(ep)
    out *= 3.4008
    term *= 0.40244
    out += term
    out += ep
    out *= fl
    out += 1.0
    np.divide(E, out, out=out)
    return out


def _jointValues(E, partition, ed, models):
    """``(3, ...)``: MT444's damage energy (cut at ``ed``) and up to two models' N_d."""
    shape = np.shape(E)
    E = np.asarray(E, dtype=float).reshape(-1)
    T = _damageEnergy(E, *partition)
    out = np.zeros((3,) + E.shape)
    out[0] = np.where(E >= ed, T, 0.0)
    for k, m_ in enumerate(list(models)[:2]):
        out[1 + k] = m_.count(T)
    return out.reshape((3,) + shape)


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
        sixth, threeQuarters = _lindhardPowers(ep)
        return E / (1.0 + fl * (3.4008 * sixth + 0.40244 * threeQuarters + ep))

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


def _twoBodyChunkJoint(cuts,u,w,x,p,offset,slope,partition,ed,hist,models):
    """:func:`_twoBodyChunk` for MT444 and the models at once: an array of 3."""
    if _heating_native is None:
        return _twoBodyChunkJointNumpy(cuts,u,w,x,p,offset,slope,partition,ed,hist,models)
    el,fl=(float(c) for c in _lindhardConstants(*partition))
    return np.array(_heating_native.twobody_chunk_joint(cuts,u,w,x,p,float(offset),float(slope),el,fl,float(ed),
        bool(hist),*_jointPacked(models)))


def _twoBodyChunkJointNumpy(cuts,u,w,x,p,offset,slope,partition,ed,hist,models):
    half=np.diff(cuts)/2;mid=(cuts[:-1]+cuts[1:])/2
    mu=mid[:,None]+half[:,None]*u
    if hist:density=p[np.clip(np.searchsorted(x,mu,side='right')-1,0,len(p)-1)]
    else:density=np.interp(mu,x,p)
    values=_jointValues(np.maximum(offset-slope*mu,0.),partition,ed,models)
    factor=half[:,None]*w*density
    return np.array([float(np.sum(factor*values[c])) for c in range(3)])


def _emissionChunkJoint(cuts,g,w,mu,wm,x,p,low,width,oldwidth,incident,denominator,thresholds,ed,partition,hist,lab,models):
    """:func:`_emissionChunk` for MT444 and the models at once: an array of 3. The
    angular range runs to the lowest recoil threshold of the three (``thresholds``,
    up to 3), split at the others so no Gauss panel holds a step; ``ed`` is MT444's cut."""
    thresholds=(list(thresholds)+[thresholds[-1]]*3)[:3]
    if _heating_native is None:
        return _emissionChunkJointNumpy(cuts,g,w,mu,wm,x,p,low,width,oldwidth,incident,denominator,thresholds,ed,
                                        partition,hist,lab,models)
    noRecoil=partition[0]==0
    el,fl=(1.,1.) if noRecoil else (float(c) for c in _lindhardConstants(*partition))
    cuts,g,w,mu,wm,x,p=(np.ascontiguousarray(a,dtype=float) for a in (cuts,g,w,mu,wm,x,p))
    return np.array(_heating_native.emission_chunk_joint(cuts,g,w,mu,wm,x,p,float(low),float(width),float(oldwidth),
        float(incident),float(denominator),*(float(denominator*r) for r in thresholds),el,fl,float(ed),bool(hist),
        bool(lab),noRecoil,*_jointPacked(models)))


def _emissionChunkJointNumpy(cuts,g,w,mu,wm,x,p,low,width,oldwidth,incident,denominator,thresholds,ed,partition,hist,lab,models):
    aa=np.sqrt(cuts[:-1]);bb=np.sqrt(cuts[1:]);half=(bb-aa)/2
    v=(aa[:,None]+bb[:,None])/2+half[:,None]*g;ep=v*v
    source=x[0]+(ep-low)*oldwidth/width
    if hist:
        pos=np.clip(np.searchsorted(x,source,side='right')-1,0,len(x)-2);density=p[pos]
    else:density=np.interp(source,x,p)
    if partition[0]==0:
        angular=np.zeros((3,)+ep.shape)
    elif lab:
        angular=_jointValues(ep,partition,ed,models)
    else:
        root=np.sqrt(incident*ep)
        angular=np.zeros((3,)+ep.shape)
        start=np.full(ep.shape,-1.)
        for s,r in enumerate(sorted(thresholds,reverse=True)):
            if s and r==sorted(thresholds,reverse=True)[s-1]:continue
            stop=np.clip(np.divide(incident+ep-denominator*r,2*root,out=np.ones_like(ep),where=root>0),-1.,1.)
            stop=np.where(stop>start,stop,start)
            angle=(start[...,None]+stop[...,None])/2+((stop[...,None]-start[...,None])/2)*mu
            recoil=np.maximum(0.,(incident+ep[...,None]-2*root[...,None]*angle)/denominator)
            values=_jointValues(recoil,partition,ed,models)
            for c in range(3):
                angular[c]+=np.sum(wm*values[c],axis=-1)*(stop-start)/4
            start=stop
    factor=w*half[:,None]*2*v*density*oldwidth/width
    return np.array([float(np.sum(factor*angular[c])) for c in range(3)])


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
 key=_jointKey('twoBody',grid,id(form),A,Q,Z,ed,elastic)
 kept=_jointRecall(key)
 if kept is not None:return kept
 models=_jointModels()
 if models:
  values=_twoBodyDamageJoint(form,grid,A,Q,Z,ed,elastic,models)
  _jointKeep(key,models,values)
  return values[:,0]
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


def _twoBodyDamageJoint(form,grid,A,Q,Z,ed,elastic,models):
 """:func:`_twoBodyDamage` for MT444 and the models at once, ``(len(grid), 3)``.

 One quadrature: the cuts are the union of the three (MT444's ``ed``, each
 model's step and kink), the angular range starts at the lowest threshold, each
 component converges to its own tolerance (MT444's, and for N_d the scaled
 absolute and DISPLACEMENT_RTOL relative), and the incident-energy refinement
 adds a node where any component asks for it. MT444 is then within its
 tolerance of the separate quadrature, not bit for bit.
 """
 from functools import lru_cache
 partition=(Z,A,Z,A)
 thresholds=[ed]+[_thresholdOf(float(m_.ed),*map(float,partition)) for m_ in models]
 kinks=[_thresholdOf(2.*float(m_.ed)/0.8,*map(float,partition)) for m_ in models]
 edLow=min(thresholds)
 scales=np.array([1.]+[0.8/(2.*m_.ed) for m_ in models]+[1.]*(2-len(models)))
 rel8=np.array([1e-8]+[max(1e-8,DISPLACEMENT_RTOL)]*2);rel5=np.array([1e-5]+[max(1e-5,DISPLACEMENT_RTOL)]*2)
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
 zero=np.zeros(3)
 @lru_cache(maxsize=30000)
 def at(E):
  if E<cutoff or E<threshold:return zero
  r=np.sqrt(max(0.,1-threshold/max(E,1e-300)));c=E*A/(A+1)**2;offset=c*(1+r*r);slope=2*c*r
  if slope==0:return _jointValues(np.array([offset]),partition,ed,models)[:,0]
  last=min(1.,(offset-edLow)/slope)
  if last<=-1:return zero
  j=int(np.clip(np.searchsorted(energies,E,side='right')-1,0,len(energies)-1));k=min(j+1,len(energies)-1)
  f=float(np.clip((E-energies[j])/(energies[k]-energies[j]),0,1)) if k!=j and not flat else 0.
  rcuts=np.r_[max(edLow,1e-12)*10.**np.arange(13),thresholds,kinks];mcuts=(offset-rcuts)/slope;mcuts=mcuts[(mcuts>-1)&(mcuts<last)]
  value=np.zeros(3)
  for idx,weight in ((j,1-f),(k,f)):
   if weight==0:continue
   if isotropic:x=np.array([-1.,1.]);p=np.array([.5,.5]);hist=False;norm=1.
   else:x,p,hist,norm=data[idx]
   cuts=np.unique(np.r_[-1.,x[(x>-1)&(x<last)],mcuts,last])
   previous=None
   for n in (8,16,32,64):
    u,w=gauss[n]
    integral=_twoBodyChunkJoint(cuts,u,w,x,p,offset,slope,partition,ed,hist,models)/norm
    if previous is not None and np.all(np.abs(integral-previous)<=.001*scales+rel8*np.abs(integral)):break
    previous=integral
   else:raise ValueError('two-body damage angular quadrature did not converge')
   value=value+weight*integral
  return value
 lo,hi=float(grid.min()),float(grid.max())
 if hi==lo:return np.tile(at(lo),(len(grid),1))
 nodes=np.unique(np.r_[lo,hi,energies[(energies>lo)&(energies<hi)],_geometric(lo,hi,step=2.),cutoff,threshold]);nodes=nodes[(nodes>=lo)&(nodes<=hi)]
 for edge in (cutoff,threshold):
  if lo<edge<=hi:nodes=np.unique(np.r_[nodes,np.nextafter(edge,-np.inf)])
 values=np.array([at(float(e)) for e in nodes]) if _energyThreads.get()<=1 else np.array(_evaluateAllRows(at,nodes))
 for _ in range(30):
  additions={}
  _evaluateAllRows(at,[float(a+(b-a)*fraction) for a,b in zip(nodes[:-1],nodes[1:]) for fraction in (.25,.5,.75)])
  for a,b,va,vb in zip(nodes[:-1],nodes[1:],values[:-1],values[1:]):
   for fraction in (.25,.5,.75):
    e=float(a+(b-a)*fraction)
    if not a<e<b:continue
    v=at(e)
    if np.any(np.abs(v-(va+(vb-va)*fraction))>.01*scales+rel5*np.abs(v)):additions[e]=v
  if not additions:return np.stack([np.interp(grid,nodes,values[:,c]) for c in range(3)],axis=1)
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
    key=_jointKey('tabulatedEmission',grid,id(spectrum),A,Z,ed,particle)
    kept=_jointRecall(key)
    if kept is not None:return kept
    models=_jointModels()
    if models:
        values=_tabulatedEmissionDamageJoint(spectrum,grid,A,Z,ed,particle,models)
        _jointKeep(key,models,values)
        return values[:,0]
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


def _tabulatedEmissionDamageJoint(spectrum,grid,A,Z,ed,particle,models):
    """:func:`_tabulatedEmissionDamage` for MT444 and the models at once, ``(len(grid), 3)``:
    the outgoing-energy cuts of all three thresholds and kinks, the angular range from
    the lowest threshold, each component to its own tolerance."""
    from functools import lru_cache
    if particle is None:recoilZ,recoilA,denominator,incidentFactor=Z,A,A,1.
    else:recoilZ,recoilA,incidentFactor=particle;denominator=1.
    partition=(recoilZ,recoilA,Z,A)
    if recoilZ==0:
        thresholds=[ed];kinks=[]
    else:
        thresholds=[ed]+[_thresholdOf(float(m_.ed),*map(float,partition)) for m_ in models]
        kinks=[_thresholdOf(2.*float(m_.ed)/0.8,*map(float,partition)) for m_ in models]
    edLow=min(thresholds)
    scales=np.array([1.]+[0.8/(2.*m_.ed) for m_ in models]+[1.]*(2-len(models)))
    rel5=np.array([1e-5]+[max(1e-5,DISPLACEMENT_RTOL)]*2);rel4=np.array([1e-4]+[max(1e-4,DISPLACEMENT_RTOL)]*2)
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
            g,w=gauss[order];mu,wm=gauss[order*2];value=np.zeros(3)
            for i,weight in ((j,1-f),(k,f)):
                if weight==0:continue
                x,p,hist,oldwidth=prepared[i]
                mapped=low+(x-x[0])*width/oldwidth
                incident=incidentFactor*E
                cuts=np.unique(np.r_[mapped,[(np.sqrt(incident)+sign*np.sqrt(denominator*r))**2
                                             for r in [*thresholds,*kinks] for sign in (-1.,1.)]])
                cuts=cuts[(cuts>=mapped[0])&(cuts<=mapped[-1])]
                for start in range(0,len(cuts)-1,256):
                    stop=min(start+256,len(cuts)-1)
                    value=value+weight*_emissionChunkJoint(cuts[start:stop+1],g,w,mu,wm,x,p,low,width,oldwidth,incident,
                                                           denominator,thresholds,ed,partition,bool(hist),lab,models)
            if previous is not None and np.all(np.abs(value-previous)<=.01*scales+rel5*np.abs(value)):return value
            previous=value
        raise ValueError('tabulated recoil damage quadrature did not converge')
    lo=float(grid.min());hi=float(grid.max())
    if lo==hi:return np.tile(at(lo),(len(grid),1))
    nodes=np.unique(np.r_[lo,hi,energies[(energies>lo)&(energies<hi)],_geometric(lo,hi,step=4.)]);nodes=nodes[(nodes>=lo)&(nodes<=hi)]
    if flat:
        left=np.nextafter(energies[1:],-np.inf);nodes=np.unique(np.r_[nodes,left[(left>lo)&(left<hi)]])
    values=np.array(_evaluateAllRows(at,nodes))
    for iteration in range(30):
        additions={}
        _evaluateAllRows(at,[float(a+(b-a)*f) for a,b in zip(nodes[:-1],nodes[1:]) for f in (.25,.5,.75)])
        for a,b,va,vb in zip(nodes[:-1],nodes[1:],values[:-1],values[1:]):
            for f in (.25,.5,.75):
                E=float(a+(b-a)*f)
                if not a<E<b:continue
                value=at(E)
                if np.any(np.abs(value-(va+(vb-va)*f))>.05*scales+rel4*np.abs(value)):additions[E]=value
        if not additions:return np.stack([np.interp(grid,nodes,values[:,c]) for c in range(3)],axis=1)
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
    key = _jointKey('analyticEmission', np.asarray(grid, dtype=float), id(spectrum), A, Z, ed)
    kept = _jointRecall(key)
    if kept is not None:
        return kept
    models = _jointModels()
    if models:
        values = _analyticEmissionDamageJoint(spectrum, grid, A, Z, ed, models)
        _jointKeep(key, models, values)
        return values[:, 0]
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


def _analyticEmissionDamageJoint(spectrum, grid, A, Z, ed, models):
    """:func:`_analyticEmissionDamage` for MT444 and the models at once, ``(len(grid), 3)``."""
    from functools import lru_cache
    partition = (Z, A, Z, A)
    thresholds = [ed] + [_thresholdOf(float(m_.ed), *map(float, partition)) for m_ in models]
    kinks = [_thresholdOf(2. * float(m_.ed) / 0.8, *map(float, partition)) for m_ in models]
    edLow = min(thresholds)
    scales = np.array([1.] + [0.8 / (2. * m_.ed) for m_ in models] + [1.] * (2 - len(models)))
    rel7 = np.array([1e-7] + [max(1e-7, DISPLACEMENT_RTOL)] * 2)
    rel5 = np.array([1e-5] + [max(1e-5, DISPLACEMENT_RTOL)] * 2)
    gauss = {n: np.polynomial.legendre.leggauss(n) for n in (16, 32, 64, 128)}
    zero = np.zeros(3)

    @lru_cache(maxsize=30000)
    def at(E):
        theta = float(spectrum.theta.evaluate(E, outOfRange="hold"))
        top = float(spectrum.upperBound(E))
        if top <= 0:
            return zero
        if not theta > 0 or not np.isfinite(top):
            raise ValueError("analytic damage requires positive theta and finite support")
        vmax = np.sqrt(top / theta)
        cuts = np.unique(np.r_[0., vmax, 1., 2., 4., 8.,
            [abs(np.sqrt(E) + sign * np.sqrt(A * r)) / np.sqrt(theta)
             for r in [*thresholds, *kinks] for sign in (-1., 1.)]])
        cuts = cuts[(cuts >= 0) & (cuts <= vmax)]
        previous = None
        for order in (16, 32, 64, 128):
            nodes, weights = gauss[order]
            half = np.diff(cuts) / 2
            v = ((cuts[:-1] + cuts[1:])[:, None] / 2 + half[:, None] * nodes).ravel()
            w = (half[:, None] * weights).ravel()
            ep = theta * v * v
            root = np.sqrt(E * ep)
            angular = np.zeros((3, len(ep)))
            start = np.full(len(ep), -1.)
            for r in sorted(set(thresholds), reverse=True):
                stop = np.clip((E + ep - A * r) / np.maximum(2 * root, 1e-300), -1., 1.)
                stop = np.where(stop > start, stop, start)
                mu = (start[:, None] + stop[:, None]) / 2 + (stop[:, None] - start[:, None]) / 2 * nodes
                recoil = np.maximum(0., (E + ep[:, None] - 2 * root[:, None] * mu) / A)
                joint = _jointValues(recoil, partition, ed, models)
                for c in range(3):
                    angular[c] += np.sum(weights * joint[c], axis=1) * (stop - start) / 4
                start = stop
            density = spectrum.evaluate(E, ep) * (2 * theta * v)
            mass = float(np.sum(w * density))
            value = np.array([float(np.sum(w * density * angular[c])) for c in range(3)])
            if (previous is not None and abs(mass - 1) <= 1e-9
                    and np.all(np.abs(value - previous) <= .01 * scales + rel7 * np.abs(value))):
                return value
            previous = value
        raise ValueError("analytic emission damage quadrature did not converge")
    active = np.array([spectrum.upperBound(float(E)) > 0 for E in grid])
    output = np.zeros((len(grid), 3))
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
                if np.any(np.abs(value - (va + (vb - va) * f)) > .05 * scales + rel5 * np.abs(value)):
                    additions[E] = value
        if not additions:
            for c in range(3):
                output[active, c] = np.interp(validGrid, nodes, values[:, c])
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
    KERMA carries. Here the compound recoil is one more independent
    primary, in the same superposition: at thermal energies E/(A+1) is far below
    ``ed`` and the damage is HEATR's to the bit; above a few keV it is the
    compound recoil that displaces (Fe-56 at 1 MeV: 13.2 keV against gheat's
    0.38 keV). Folding every kick into one recoil of mean energy E/(A+1) + E_ph
    was measured and rejected: the mean of the photon kicks can sit below
    ``ed`` while the hard primaries do not (U-238 thermal capture damage went
    to zero).
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
    """``capdam`` for a reaction that emits a charged particle and no neutron.

    The continuum of a charged-level range (649, 699, 749, 799, 849) emits its own
    particle; HEATR takes all five as protons (``capdam``: ``zx = ax = 1`` above MT107).
    """
    zx, ax = 1.0, 1.0
    if mt in (104, 699):
        ax = 2.0
    elif mt in (105, 749):
        ax = 3.0
    elif mt in (106, 799):
        zx, ax = 2.0, 3.0
    elif mt in (107, 849):
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


#: The light charged particles; any other non-neutron, non-photon product is a residual.
_LIGHT_PARTICLES = ("H1", "H2", "H3", "He3", "He4")

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
    if key in tables and hasattr(tables[key], "result"):
        tables[key] = tables[key].result()                    # started on the prefetch pool
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


def _tabulateJob(job):
    """``(tabulated, error)`` of one ``(form, context)``, in a worker process."""
    from kika.processing.tabulation import tabulateDistribution

    form, context = job
    try:
        return tabulateDistribution(form, context), None
    except (TypeError, ValueError, NotImplementedError) as exc:
        return None, exc


def _heavyJobs(suite, evaluation, mts, A, zaTarget):
    """The ``(form, context)`` pairs :func:`_heavyProducts` and :func:`_independentParticles`
    will tabulate, as they build them: each reaction's charged particles and recoils, and
    the neutrons of one that has some."""
    from kika.nuclear_data.model import Isotropic2d, zaFromPid
    from kika.nuclear_data.model.endf_conversion import reactionQI
    from kika.processing.tabulation import TabulationContext

    jobs = []
    for reaction in suite.reactions:
        mt = reaction.ENDF_MT
        if mt is None or mt in FISSION_MTS or mt not in mts or mt in (2, 102) or 51 <= mt <= 90:
            continue
        source = evaluation.findReactionByENDF_MT(mt)
        if source is None:
            continue
        Q = float(reactionQI(source, evaluation) or 0.0)
        level = _levelParticle(mt)
        charged, neutrons = [], []
        for product in source.outputChannel.products:
            forms = product.distribution.forms if product.distribution is not None else {}
            form = forms.get("eval") or (next(iter(forms.values())) if forms else None)
            if form is None or isinstance(form, Isotropic2d) or product.multiplicity is None:
                continue
            if product.pid == "n":
                neutrons.append((form, TabulationContext(awr=A, zaTarget=zaTarget, mt=mt, Q=Q, pid="n",
                                                         zaEmitted=1)))
            elif product.pid != "photon":
                charged.append((form, TabulationContext(awr=A, zaTarget=zaTarget, mt=mt, Q=Q, pid=product.pid,
                                                        zaEmitted=zaFromPid(product.pid))))
        jobs += charged + (neutrons if charged else [])
    return jobs


def _prefetchHeavy(jobs, workers):
    """Fill :func:`_tabulateHeavy`'s cache for ``jobs`` on ``workers`` processes: each table is
    the one the sequential call would make (the same function on the same form), and the
    tabulation is pure Python, which threads would not run at once."""
    missing = []
    for form, context in jobs:
        key = (context.awr, context.zaTarget, context.mt, context.Q, context.pid, context.zaEmitted,
               context.options)
        entry = _HEAVY_TABULATIONS.get(id(form))
        if entry is None or entry[0]() is not form or key not in entry[1]:
            missing.append((form, context, key))
    pool = _PREFETCH.get()
    if pool is not None:
        # The processing's pool: the jobs start now and are collected when asked for.
        results = [pool.submit(_tabulateJob, (form, context)) for form, context, _ in missing]
    elif workers < 2 or len(missing) < 4:
        return
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as local:
            results = list(local.map(_tabulateJob, [(form, context) for form, context, _ in missing]))
    for (form, _context, key), result in zip(missing, results):
        entry = _HEAVY_TABULATIONS.get(id(form))
        if entry is None or entry[0]() is not form:
            formId = id(form)

            def forget(reference, formId=formId):
                if _HEAVY_TABULATIONS.get(formId, (None,))[0] is reference:
                    del _HEAVY_TABULATIONS[formId]

            entry = _HEAVY_TABULATIONS[formId] = (weakref.ref(form, forget), {})
        entry[1].setdefault(key, result)


def _heavyProducts(reaction, evaluation, grid, A, Z, zaTarget, ed, notes, damageOnly=False):
    """``(deposit, damage)`` of a reaction's charged particles and recoils.

    What HEATR takes from an MF6 section's ``izap > 1`` subsections: each
    particle's mean lab energy and its own Lindhard damage. ``None`` when the
    evaluation states none of them with a distribution.
    """
    from kika.nuclear_data.model import AngularTwoBody, Isotropic2d, zaFromPid
    from kika.processing.tabulation import TabulationContext, tabulateDistribution

    source = evaluation.findReactionByENDF_MT(reaction.ENDF_MT) if evaluation is not None else None
    if source is None:
        return None
    from kika.nuclear_data.model.endf_conversion import reactionQI
    Q = float(reactionQI(source, evaluation) or 0.0)
    deposit = np.zeros(len(grid))
    damage = np.zeros(len(grid))
    found = False
    light, residual, twoBody, neutrons = [], False, False, []
    level = _levelParticle(reaction.ENDF_MT)
    for product in source.outputChannel.products:
        if product.pid == "n" and product.distribution is not None and product.multiplicity is not None:
            neutrons.append(product)
        ownLevel = level is not None and product.pid == level
        if product.pid in ("n", "photon") or (product.distribution is None and not ownLevel):
            continue
        forms = product.distribution.forms if product.distribution is not None else {}
        form = forms.get("eval") or (next(iter(forms.values())) if forms else None)
        multiplicity = product.multiplicity.form if product.multiplicity is not None else None
        if ownLevel and multiplicity is None:
            # MF4 states a level's particle by its angle alone: one of it, two bodies
            # (HEATR's disbar). Left out, the level fell to capdam's proton below the
            # Coulomb barrier (O-16 MT800-848, all of MT444 at 5 MeV).
            multiplicity = _UNIT_YIELD
        if form is None and ownLevel:
            form = Isotropic2d()
        if form is None or multiplicity is None:
            continue
        za = zaFromPid(product.pid)
        context = TabulationContext(awr=A, zaTarget=zaTarget, mt=reaction.ENDF_MT, Q=Q,
                                    pid=product.pid, zaEmitted=za)
        if ownLevel and isinstance(form, Isotropic2d):
            tabulated = _isotropicLevel(product.pid, A, Q)
        else:
            try:
                tabulated = _tabulateHeavy(form, context)
            except (TypeError, ValueError, NotImplementedError) as exc:
                notes.append(f"MT{reaction.ENDF_MT} {product.pid}: {type(form).__name__} not "
                             f"tabulated for the heating ({exc}); left out")
                continue
        if tabulated is None:
            if isinstance(form, AngularTwoBody) and product.pid not in _LIGHT_PARTICLES:
                twoBody = True                  # LAW=4: the partner's recoil, by momentum
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
        if product.pid in _LIGHT_PARTICLES:
            light.append((awp, tabulated, k, y, za // 1000))
        else:
            residual = True
        active=y!=0
        if np.any(active):
            damage[active]+=y[active]*_particleDamage(tabulated,grid[active],k,za//1000,awp,Z,A,ed)
    if light and not residual:
        # The section lists the light particles and not the residual: HEATR's
        # one-particle recoil from the heaviest of them (hinit: "does not give
        # recoil za=..., one-particle recoil approx. used"; getsix, irec > 0).
        # Without it only the particle's own damage is left, 100-300 times
        # smaller (ENDF/B-VIII.1 Ni-60 MT649: MT444 14-32 % below HEATR's at
        # 10-20 MeV). A residual of Z <= 2 is itself a light particle. A
        # residual stated as a two-body recoil (LAW=4) is the same momentum
        # balance, exact for a level: HEATR's sixbar backs up to the partner
        # (irec = j6 - 1). Without it a charged level kept only the particle's
        # damage (Ni-58 MT600-609, MT444 20 % below HEATR's at 5 MeV).
        # With more than one particle emitted (MT22, 28, 24, ...), the residual takes
        # the momentum of all of them, drawn independently: against the cascades that
        # ENDF/B-VIII.1 does state, closer than HEATR's recoil from the one particle
        # (O-16 MT22 1.04-1.06 against 1.13-1.15).
        independent = _independentParticles(light, neutrons, grid, A, Z, zaTarget, reaction.ENDF_MT, Q,
                                            notes) if not twoBody else None
        if independent is not None:
            particles, zr, ar = independent
            if zr >= 3:
                try:
                    damage = damage + _independentRecoilDamage(particles, grid, A, Z, zr, ar, ed)
                except NotImplementedError as exc:
                    notes.append(f"MT{reaction.ENDF_MT}: no independent recoil ({exc}); "
                                 f"one-particle recoil instead")
                else:
                    notes.append(f"MT{reaction.ENDF_MT}: residual not given; its recoil from the "
                                 f"momenta of all {sum(int(round(np.max(y))) for _, _, y in particles)} "
                                 f"particles, independent")
                    return (deposit, damage) if found else None
        awp, tabulated, k, y, zp = max(light, key=lambda item: item[0])
        zr, ar = Z - zp, A + 1.0 - awp
        if zr >= 3:
            try:
                recoil = _recoilDamage(tabulated, grid, k, zr, ar, Z, A, ed)
            except NotImplementedError as exc:
                notes.append(f"MT{reaction.ENDF_MT}: no residual recoil ({exc}); left out")
            else:
                # One residual per reaction, whatever the particle's yield: HEATR's
                # sixbar sets the recoil's yield to 1 (irec > 0: pe = 1). With y the
                # (n,2p) residual counted twice (Ni-58 MT111, MT444 +2.5 % at 20 MeV).
                damage = damage + np.where(y > 0, recoil, 0.0)
                notes.append(f"MT{reaction.ENDF_MT}: " + (
                    "two-body residual (LAW=4): its recoil from the particle's angular law" if twoBody
                    else "two-body level: the residual's recoil from the particle's angular law "
                         "(HEATR's disbar)" if level is not None
                    else "residual not given; HEATR's one-particle recoil"))
    return (deposit, damage) if found else None


class _UnitYield:
    """The yield of one: what an MF4 level states for its particle without saying it."""

    @staticmethod
    def evaluate(grid):
        return np.ones(len(np.atleast_1d(grid)))


_UNIT_YIELD = _UnitYield()

#: The particle of each charged-level range (ENDF-6 MT600-848); the continuum of
#: each (649, 699, ...) is not a level.
_LEVEL_RANGES = ((600, "H1"), (650, "H2"), (700, "H3"), (750, "He3"), (800, "He4"))


def _levelParticle(mt):
    """The light particle a discrete charged-level MT emits, else ``None``."""
    for first, pid in _LEVEL_RANGES:
        if first <= mt <= first + 48:
            return pid
    return None


def _isotropicLevel(pid, A, Q):
    """A level's particle isotropic in the centre of mass, its energy from Q."""
    from kika.nuclear_data import model as m
    from kika.processing.tabulation import levelEnergy
    from kika.processing.tabulation.distribution import LIGHT_PARTICLE_AWP

    return m.Uncorrelated(angular=m.Isotropic2d(), energy=levelEnergy(Q, A, LIGHT_PARTICLE_AWP[pid]),
                          productFrame=m.Frame.centerOfMass)


def _levelDamage(mt, grid, A, Z, ed, Q):
    """A charged level stated by MF3 alone: two bodies, isotropic in the centre of mass
    (HEATR's disbar without MF4), the particle's damage and the residual's."""
    pid = _levelParticle(mt)
    form = _isotropicLevel(pid, A, Q)
    awp = PARTICLE_MASS_RATIO[{"H1": "p", "H2": "d", "H3": "t", "He3": "h", "He4": "a"}[pid]]
    zp = 2 if pid in ("He3", "He4") else 1
    k = _Kinematics(A=A, Q=Q, awp=awp)
    return (_particleDamage(form, grid, k, zp, awp, Z, A, ed)
            + _recoilDamage(form, grid, k, Z - zp, A + 1.0 - awp, Z, A, ed))


def _independentParticles(light, neutrons, grid, A, Z, zaTarget, mt, Q, notes):
    """``(particles, zr, ar)`` for :func:`_independentRecoilDamage`, or ``None`` when the
    section emits one particle (its recoil is then exact) or a yield is not a whole number."""
    from kika.processing.tabulation import TabulationContext

    particles = [(tabulated, awp, y) for awp, tabulated, _, y, _ in light]
    for product in neutrons:
        forms = product.distribution.forms
        form = forms.get("eval") or (next(iter(forms.values())) if forms else None)
        if form is None:
            return None
        context = TabulationContext(awr=A, zaTarget=zaTarget, mt=mt, Q=Q, pid="n", zaEmitted=1)
        try:
            tabulated = _tabulateHeavy(form, context)
        except (TypeError, ValueError, NotImplementedError) as exc:
            notes.append(f"MT{mt} n: {type(form).__name__} not tabulated for the recoil ({exc})")
            return None
        if tabulated is None:
            return None
        particles.append((tabulated, 1.0, np.asarray(product.multiplicity.form.evaluate(grid), dtype=float)))
    counts = [y[y > 0] for _, _, y in particles]
    if any(np.any(np.abs(c - np.round(c)) > 1e-6) for c in counts):
        return None
    if sum(int(round(np.max(c))) if len(c) else 0 for c in counts) < 2:
        return None
    mass = sum(awp * int(round(np.max(c))) for (_, awp, _), c in zip(particles, counts) if len(c))
    charge = sum(zp * int(round(np.max(y))) for awp, _, _, y, zp in light)
    return particles, Z - charge, A + 1.0 - mass


def _neutronResidualDamage(mt, neutrons, grid, A, Z, ed, notes=None):
    """The residual's damage in a section that states its neutrons and not the residual.

    ``neutrons`` holds ``(product, Monte Carlo form)``. With whole yields the residual
    ``(Z, A+1−n)`` recoils from the momenta of all ``n`` neutrons, each on its own
    energy-angle law (:func:`_independentRecoilDamage`): for one neutron (MT91) the exact
    recoil, HEATR's one-particle recoil for an MF6 neutron (``hinit``, ``getsix`` with
    ``irec > 0``) taken over the angle instead of at its mean cosine. With a yield that
    is not whole, HEATR's one-particle recoil: the first neutron, the residual of mass A,
    one per reaction. HEATR's ``tabdam`` -- one isotropic lab emission, mass A, kika's rule
    before -- only when the form has no lab momenta.

    Against HEATR itself (NJOY 2016.78 rebuilt without the generated recoil of one MT,
    ENDF/B-VIII.1 Ni-60), HEATR's MT91 is 32.4 % of MT444 at 14 MeV; its rule
    emulated on kika's tables gives 32.4 %, the exact recoil 31.0 % (the mean cosine,
    Jensen), ``tabdam`` 34.4 %. Against the residual Fe-56 states, the exact MT91 is
    −17 % at 20 MeV: that residual is a neutron isotropic in the centre of mass (⟨E_R⟩
    537 keV; the neutron's own law gives 419, isotropic 515), not the evaluation's own
    neutron.
    """
    particles = [(form, 1.0, _yield(product, grid)) for product, form in neutrons]
    counts = [y[y > 0] for _, _, y in particles]
    n = sum(int(round(np.max(c))) for c in counts if len(c))
    whole = all(not np.any(np.abs(c - np.round(c)) > 1e-6) for c in counts)
    if whole and n >= 1:
        how, ar = (f"residual not given; its recoil from the momenta of all {n} neutrons, independent"
                   if n > 1 else "residual not given; its recoil from the neutron's energy-angle law"), A + 1.0 - n
    else:
        form, _, y = particles[0]
        particles = [(form, 1.0, np.where(y > 0, 1.0, 0.0))]
        how, ar = "neutron yield not whole; HEATR's one-particle recoil (mass A)", A
    try:
        damage = _independentRecoilDamage(particles, grid, A, Z, Z, ar, ed)
    except NotImplementedError as exc:
        if notes is not None:
            notes.append(f"MT{mt}: no residual recoil ({exc}); HEATR's tabdam instead")
        return _emissionDamage(neutrons[0][1], grid, A, Z, ed)
    if notes is not None:
        notes.append(f"MT{mt}: {how}")
    return damage


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


#: The angular rule of an isotropic emission: Gauss-Legendre on [-1, 1], weights summing to 1.
_ISOTROPIC_MU = (np.polynomial.legendre.leggauss(16)[0], np.polynomial.legendre.leggauss(16)[1] / 2.0)


def _ruleOf(table, split=1):
    """``(nodes, weights)`` with ``Σ w g(x) = ∫ g p dx``, normalised: lines first, then the
    continuum by 4-point Gauss on each panel cut in ``split`` (``split=1`` is :func:`_expect`'s)."""
    x = np.asarray(table.xs, dtype=float)
    p = np.asarray(table.pdf if hasattr(table, "pdf") else table.ys, dtype=float)
    nd = int(getattr(table, "numberOfDiscrete", 0) or 0)
    nodes, weights = [x[:nd]], [p[:nd]]
    cx, cp = x[nd:], p[nd:]
    if len(cx) > 1:
        width = cx[1:] - cx[:-1]
        cuts = np.linspace(0.0, 1.0, split + 1)
        a = (cx[:-1, None] + width[:, None] * cuts[None, :-1]).ravel()
        b = (cx[:-1, None] + width[:, None] * cuts[None, 1:]).ravel()
        lo = np.repeat(cx[:-1], split)
        start = np.repeat(cp[:-1], split)
        slope = np.repeat(np.where(width > 0, (cp[1:] - cp[:-1]) / np.where(width > 0, width, 1.0), 0.0), split)
        u, wu = _GL4
        half, mid = 0.5 * (b - a), 0.5 * (b + a)
        pts = mid[:, None] + half[:, None] * u[None, :]
        if _isFlat(table):
            dens = np.broadcast_to(start[:, None], pts.shape)
        else:
            dens = start[:, None] + slope[:, None] * (pts - lo[:, None])
        keep = b > a
        nodes.append(pts[keep].ravel())
        weights.append((half[:, None] * wu[None, :] * dens)[keep].ravel())
    nodes, weights = np.concatenate(nodes), np.concatenate(weights)
    total = weights.sum()
    return nodes, (weights / total if total > 0 else weights)


def _mixedRule(tables, rules, at):
    """The rule of the distribution interpolated lin-lin (as a mixture) between ``tables``' nodes at ``at``."""
    nodes = np.array([float(t.outerDomainValue) for t in tables])
    if at <= nodes[0] or len(nodes) == 1:
        return rules[0]
    if at >= nodes[-1]:
        return rules[-1]
    i = int(np.searchsorted(nodes, at, side="right") - 1)
    f = (at - nodes[i]) / (nodes[i + 1] - nodes[i])
    (x0, w0), (x1, w1) = rules[i], rules[i + 1]
    return np.concatenate([x0, x1]), np.concatenate([(1.0 - f) * w0, f * w1])


#: Sub-panels per panel of the recoil quadrature, in E' and in μ: the recoil goes as
#: √E' near E' = 0, and a two-point μ table is one panel; one 4-point Gauss panel
#: integrates either to ~1e-3 only.
_RECOIL_SPLIT = 8

#: Ratio between the incident energies at which a level's two-body recoil is computed.
_LEVEL_STEP = 1.02

#: Sub-panels per angular panel of a level's two-body recoil.
_LEVEL_SPLIT = 64


def _recoilDamage(form, grid, k, zr, ar, Z, A, ed):
    """``⟨df(E_R)⟩`` of the residual ``(zr, ar)`` recoiling from one emission of a particle.

    HEATR's one-particle recoil for an MF6 section that lists the particle but not the
    residual (``hinit``, ``getsix`` with ``irec > 0``): momentum conservation with the
    particle's own energy-angle distribution. In the lab, ``E_R = (E − 2√(E m E')μ + m E')/M``;
    in the centre of mass, ``E_R = a + b − 2√(ab)μ``, ``a = M E/(A+1)²``, ``b = m E'/M``
    (``m = k.awp``, ``M = ar``). For a single emitted particle it is the exact recoil.
    The damage is computed at the incident energies of the particle's tables and read
    lin-lin between them, as HEATR's ``sixbar`` does.
    """
    from kika.nuclear_data import model as m

    if isinstance(form, m.DistributionMixture):
        out = np.zeros(len(grid))
        for term in form.weighted:
            weight = (np.asarray(term.weight.evaluate(grid), dtype=float)
                      if term.weight is not None else 1.0)
            out = out + weight * _recoilDamage(term.functional, grid, k, zr, ar, Z, A, ed)
        return out
    frame = getattr(form, "productFrame", None)
    cm = str(getattr(frame, "value", frame)) in ("centerOfMass", "Frame.centerOfMass")
    mp, M = k.awp, ar

    def damageAt(E, e, mus, weights):
        """``Σ_μ w df(E_R(E, e, μ))`` for outgoing energies ``e`` (n,) and a rule in μ."""
        return _recoilRows(E, e, mus, weights, cm, M, mp, A, zr, Z, ed)

    if isinstance(form, m.EnergyAngularMC):
        energies, tables, flat = _tables(form.energy)

        def valueAt(job):
            E, t, mus = job
            rules = [_ruleOf(u, _RECOIL_SPLIT) for u in mus.function1ds]
            nodes = np.asarray(t.xs, dtype=float)
            nd = int(getattr(t, "numberOfDiscrete", 0) or 0)
            e, w = _ruleOf(t, _RECOIL_SPLIT)
            value = 0.0
            for j in range(nd):                               # lines: their own angular table
                value += w[j] * damageAt(E, e[j:j + 1], *rules[j])[0]
            ec, wc, cont = e[nd:], w[nd:], nodes[nd:]
            if len(cont) == 1 and len(ec):
                value += float(wc @ damageAt(E, ec, *rules[nd]))
            elif len(ec):
                # P(mu|E') lin-lin between the tabulated E': the mixture of its two neighbours.
                interval = np.clip(np.searchsorted(cont, ec, side="right") - 1, 0, len(cont) - 2)
                for i in np.unique(interval):
                    inside = interval == i
                    lo, hi = cont[i], cont[i + 1]
                    f = np.clip((ec[inside] - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros(inside.sum())
                    left = damageAt(E, ec[inside], *rules[nd + i])
                    right = damageAt(E, ec[inside], *rules[nd + i + 1])
                    value += float(wc[inside] @ ((1.0 - f) * left + f * right))
            return value

        values = _threadMap(valueAt, zip(energies, tables, form.energyAngular.function2ds))
        return _onGrid(energies, values, grid, flat)
    if isinstance(form, m.KalbachMann):
        energies, tables, flat = _tables(form.f)
        rs = list(m.toEndfTab2(form.r)[0])
        As = list(m.toEndfTab2(form.a)[0])
        mu, wmu = np.polynomial.legendre.leggauss(32)

        def valueAt(job):
            E, t, r, a = job
            e, w = _ruleOf(t, _RECOIL_SPLIT)
            rr = np.interp(e, r.xs, r.ys)[:, None]
            aa = np.interp(e, a.xs, a.ys)[:, None]
            safe = np.where(np.abs(aa) > 1e-6, aa, 1.0)
            norm = np.where(np.abs(aa) > 1e-6, safe / (2.0 * np.sinh(safe)), 0.5)
            p = norm * (np.cosh(aa * mu) + rr * np.sinh(aa * mu)) * wmu
            return float(w @ _recoilRows(E, e, mu, p, cm, M, mp, A, zr, Z, ed))

        values = _threadMap(valueAt, zip(energies, tables, rs, As))
        return _onGrid(energies, values, grid, flat)
    if isinstance(form, m.Uncorrelated):
        spectrum = form.energy
        if spectrum is None or isinstance(spectrum, (m.DiscretePhotonEnergy, m.NBodyPhaseSpace)):
            return np.zeros(len(grid))
        angular = getattr(form, "angular", None)
        if angular is None or isinstance(angular, m.Isotropic2d):
            angularTables = angularRules = None
        else:
            _, angularTables, _ = _tables(angular)
            angularRules = [_ruleOf(u, _RECOIL_SPLIT) for u in angularTables]
        if isinstance(spectrum, m.LevelEnergy):
            # A level: two bodies, the residual's centre-of-mass momentum the particle's.
            # One outgoing energy per incident energy, so the damage is read on a fine
            # geometric mesh and the angular tables' energies, not on those alone.
            lo, hi = float(np.min(grid)), float(np.max(grid))
            threshold = float(spectrum.threshold)
            nodes = [_geometric(max(lo, threshold, 1e-5), hi, step=_LEVEL_STEP), [lo, hi]]
            if angularTables is not None:
                nodes.append([float(t.outerDomainValue) for t in angularTables])
            if lo < threshold < hi:
                nodes.append([np.nextafter(threshold, -np.inf), threshold])
            energies = np.unique(np.concatenate(nodes))
            energies = energies[(energies >= lo) & (energies <= hi)]
            if angularTables is not None:
                # One outgoing energy: the angle alone carries the quadrature, and the
                # damage steps at E_R = ed inside it.
                angularRules = [_ruleOf(u, _LEVEL_SPLIT) for u in angularTables]
            else:
                cuts = np.linspace(-1.0, 1.0, _LEVEL_SPLIT + 1)
                u, wu = _GL4
                half = 0.5 * np.diff(cuts)
                isotropic = (((cuts[:-1] + cuts[1:]) / 2)[:, None] + half[:, None] * u).ravel(),                     (half[:, None] * wu / 2.0 * np.ones_like(u)).ravel()
            def levelAt(E):
                e = float(spectrum.outgoingEnergy(E))
                if e <= 0.0:
                    return 0.0
                x, wx = (isotropic if angularTables is None
                         else _mixedRule(angularTables, angularRules, float(E)))
                return float(damageAt(E, np.array([e]), x, wx)[0])

            return _onGrid(energies, _threadMap(levelAt, energies), grid)
        energies, tables, flat = _tables(spectrum)

        def tableAt(job):
            E, t = job
            x, wx = (_ISOTROPIC_MU if angularTables is None
                     else _mixedRule(angularTables, angularRules, float(E)))
            e, w = _ruleOf(t, _RECOIL_SPLIT)
            return float(w @ damageAt(E, e, x, wx))

        return _onGrid(energies, _threadMap(tableAt, zip(energies, tables)), grid, flat)
    raise NotImplementedError(f"one-particle recoil of a {type(form).__name__}")


#: Bins per axis of a momentum distribution (p_z, p_⊥) in the independent recoil:
#: each bin keeps its weight and centroid, so the first moments are exact and the
#: second lose ~(1/bins)²/12 of the spread.
_MOMENTUM_BINS = 24

#: Pairs times azimuths per block of :func:`_addMomenta`, to bound its memory.
_MOMENTUM_BLOCK = 1 << 21

#: Midpoints in the relative azimuth of two independent momenta, on [0, π]: the
#: integrand is smooth and periodic, where the midpoint rule converges fastest.
_AZIMUTH_NODES = 16


def _incidentNodes(form):
    """The incident energies of a Monte Carlo form's tables."""
    from kika.nuclear_data import model as m

    if isinstance(form, m.DistributionMixture):
        return np.unique(np.concatenate([_incidentNodes(t.functional) for t in form.weighted]))
    if isinstance(form, m.EnergyAngularMC):
        return _tables(form.energy)[0]
    if isinstance(form, m.KalbachMann):
        return _tables(form.f)[0]
    if isinstance(form, m.Uncorrelated) and form.energy is not None and \
            not isinstance(form.energy, (m.LevelEnergy, m.DiscretePhotonEnergy, m.NBodyPhaseSpace)):
        return _tables(form.energy)[0]
    raise NotImplementedError(f"independent recoil of a {type(form).__name__}")


def _emissionRule(form, i, E, cache=None):
    """``(E', μ, w)`` of incident table ``i`` of a Monte Carlo form, in its own frame.

    ``cache``, a dict for the length of one computation, keeps the rules that depend on the
    table alone (correlated and Kalbach-Mann laws): every incident energy between two tables
    asks for both. The arrays are read, never changed."""
    from kika.nuclear_data import model as m

    if cache is not None and isinstance(form, (m.EnergyAngularMC, m.KalbachMann)):
        key = (id(form), i)
        if key not in cache:
            cache[key] = (form, _emissionRule(form, i, E))     # the form kept alive with its key
        return cache[key][1]
    if isinstance(form, m.EnergyAngularMC):
        _, tables, _ = _tables(form.energy)
        t, angular = tables[i], form.energyAngular.function2ds[i]
        rules = [_ruleOf(u, 2) for u in angular.function1ds]
        nodes = np.asarray(t.xs, dtype=float)
        nd = int(getattr(t, "numberOfDiscrete", 0) or 0)
        e, w = _ruleOf(t, 2)
        parts = [(np.full(len(rules[j][0]), e[j]), rules[j][0], w[j] * rules[j][1]) for j in range(nd)]
        ec, wc, cont = e[nd:], w[nd:], nodes[nd:]
        if len(cont) == 1:
            parts += [(np.full(len(rules[nd][0]), a), rules[nd][0], b * rules[nd][1]) for a, b in zip(ec, wc)]
        elif len(ec):
            # P(μ|E') lin-lin between the tabulated E': the mixture of its two neighbours.
            k = np.clip(np.searchsorted(cont, ec, side="right") - 1, 0, len(cont) - 2)
            for a, b, j in zip(ec, wc, k):
                lo, hi = cont[j], cont[j + 1]
                f = min(max((a - lo) / (hi - lo), 0.0), 1.0) if hi > lo else 0.0
                for g, (x, wx) in ((1.0 - f, rules[nd + j]), (f, rules[nd + j + 1])):
                    if g > 0:
                        parts.append((np.full(len(x), a), x, b * g * wx))
        return tuple(np.concatenate(c) for c in zip(*parts))
    if isinstance(form, m.KalbachMann):
        _, tables, _ = _tables(form.f)
        r, a = list(m.toEndfTab2(form.r)[0])[i], list(m.toEndfTab2(form.a)[0])[i]
        mu, wmu = np.polynomial.legendre.leggauss(32)
        e, w = _ruleOf(tables[i], 2)
        rr = np.interp(e, r.xs, r.ys)[:, None]
        aa = np.interp(e, a.xs, a.ys)[:, None]
        safe = np.where(np.abs(aa) > 1e-6, aa, 1.0)
        norm = np.where(np.abs(aa) > 1e-6, safe / (2.0 * np.sinh(safe)), 0.5)
        p = norm * (np.cosh(aa * mu) + rr * np.sinh(aa * mu)) * wmu
        return np.repeat(e, len(mu)), np.tile(mu, len(e)), (w[:, None] * p).ravel()
    if isinstance(form, m.Uncorrelated):
        _, tables, _ = _tables(form.energy)
        angular = getattr(form, "angular", None)
        if angular is None or isinstance(angular, m.Isotropic2d):
            x, wx = _ISOTROPIC_MU
        else:
            _, at, _ = _tables(angular)
            x, wx = _mixedRule(at, [_ruleOf(u, 2) for u in at], float(E))
        e, w = _ruleOf(tables[i], 2)
        return np.repeat(e, len(x)), np.tile(x, len(e)), (w[:, None] * wx[None, :]).ravel()
    raise NotImplementedError(f"independent recoil of a {type(form).__name__}")


def _labMomenta(form, E, awp, A, cache=None):
    """``(p_z, p_⊥, w)`` of a particle at incident energy ``E``, in the lab.

    Momenta in neutron masses and eV (``p² = 2 m E'``), ``z`` along the beam; the
    weights sum to 1. Between two incident tables the distribution is their
    lin-lin mixture, as the damage is read between them everywhere else.
    """
    from kika.nuclear_data import model as m

    if isinstance(form, m.DistributionMixture):
        weights = [t.weight for t in form.weighted if t.weight is not None]
        if weights:
            # Outside its weights' domain (O-17 MT16 below 4.39 MeV, where its
            # cross section is zero) a mixture is read at the nearest end, as
            # a single law is read at its first or last table.
            E = min(max(E, min(float(w.domainMin) for w in weights)),
                    max(float(w.domainMax) for w in weights))
        parts = []
        for term in form.weighted:
            g = float(np.asarray(term.weight.evaluate([E]), dtype=float)[0]) if term.weight is not None else 1.0
            if g > 0:
                pz, pp, w = _labMomenta(term.functional, E, awp, A, cache)
                parts.append((pz, pp, g * w))
        pz, pp, w = (np.concatenate(c) for c in zip(*parts))
        return pz, pp, w / w.sum()
    nodes = _incidentNodes(form)
    if E <= nodes[0] or len(nodes) == 1:
        mix = [(0, 1.0)]
    elif E >= nodes[-1]:
        mix = [(len(nodes) - 1, 1.0)]
    else:
        i = int(np.searchsorted(nodes, E, side="right") - 1)
        f = (E - nodes[i]) / (nodes[i + 1] - nodes[i])
        mix = [(i, 1.0 - f), (i + 1, f)]
    parts = []
    for i, g in mix:
        if g > 0:
            e, mu, w = _emissionRule(form, i, E, cache)
            total = w.sum()
            if total > 0:
                parts.append((e, mu, g * w / total))
    if not parts:
        # Tables of zeros (Ra-225 ENDF/B-VIII.1: a spectrum stated null at its
        # threshold): the nearest table that emits, as the ends are read.
        for i in sorted(range(len(nodes)), key=lambda j: abs(nodes[j] - E)):
            e, mu, w = _emissionRule(form, i, E, cache)
            if w.sum() > 0:
                parts.append((e, mu, w / w.sum()))
                break
        else:
            raise ValueError("a law whose every incident table is zero emits nothing")
    e, mu, w = (np.concatenate(c) for c in zip(*parts))
    mu = np.clip(mu, -1.0, 1.0)
    v = np.sqrt(2.0 * np.clip(e, 0.0, None) / awp)
    vz, vp = v * mu, v * np.sqrt(1.0 - mu * mu)
    frame = getattr(form, "productFrame", None)
    if str(getattr(frame, "value", frame)) in ("centerOfMass", "Frame.centerOfMass"):
        vz = vz + np.sqrt(2.0 * E) / (A + 1.0)
    return awp * vz, awp * vp, w / w.sum()


class _MomentumBins:
    """A momentum distribution on a fixed :data:`_MOMENTUM_BINS`² mesh in ``(p_z, p_⊥)``.

    Each bin keeps its weight, centroid and second moments: ``E_R`` is quadratic in
    the momentum, so with them ``⟨E_R⟩`` is exact, and only the curvature of the
    damage within a bin is lost.
    """

    def __init__(self, zlo, zhi, plo, phi):
        n = _MOMENTUM_BINS
        self.edges = (zlo, max(zhi - zlo, 1e-300), plo, max(phi - plo, 1e-300))
        self.sums = np.zeros((5, n * n))                     # w, w p_z, w p_⊥, w p_z², w p_⊥²

    def add(self, pz, pp, w, vz=0.0):
        """Points of weight ``w``, each carrying a variance ``vz`` in ``p_z`` from before."""
        n = _MOMENTUM_BINS
        zlo, zw, plo, pw = self.edges
        i = np.clip(((pz - zlo) / zw * n).astype(np.int64), 0, n - 1)
        k = np.clip(((pp - plo) / pw * n).astype(np.int64), 0, n - 1)
        cell = i * n + k
        for row, values in enumerate((w, w * pz, w * pp, w * (pz * pz + vz), w * pp * pp)):
            self.sums[row] += np.bincount(cell, weights=values, minlength=n * n)

    def moments(self):
        """``(p̄_z, p̄_⊥, w, var p_z, ⟨p_⊥²⟩)`` of the occupied bins."""
        w, z, p, zz, pp = self.sums
        live = w > 0
        w = w[live]
        z, p = z[live] / w, p[live] / w
        return z, p, w, np.clip(zz[live] / w - z * z, 0.0, None), pp[live] / w


def _binMomenta(pz, pp, w):
    """Lab momenta ``(p_z, p_⊥, w)`` as :meth:`_MomentumBins.moments`."""
    keep = w > 0
    pz, pp, w = pz[keep], pp[keep], w[keep]
    if len(w) <= _MOMENTUM_BINS:
        return pz, pp, w, np.zeros(len(w)), pp * pp
    bins = _MomentumBins(pz.min(), pz.max(), pp.min(), pp.max())
    if _heating_native is not None:
        zlo, zw, plo, pw = bins.edges
        raw = _heating_native.points_bin(*(np.ascontiguousarray(v, dtype=float) for v in (pz, pp, w)),
                                         float(zlo), float(zw), float(plo), float(pw), int(_MOMENTUM_BINS))
        bins.sums = np.frombuffer(raw, dtype=float).reshape(5, -1).copy()
    else:
        bins.add(pz, pp, w)
    return bins.moments()


def _recoilRows(E, e, mus, weights, cm, M, mp, A, zr, Z, ed):
    """``Σ_k weights[., k] df(max(E_R(E, e_i, mu_k), 0))`` for each outgoing energy ``e_i``:
    :func:`_recoilDamage`'s rows. ``weights`` is one rule shared by every ``e`` or one row
    per ``e``. The native kernel does it operation for operation, a row summed pairwise as
    np.sum does; the NumPy statements below are the definition."""
    e = np.ascontiguousarray(e, dtype=float)
    mus = np.ascontiguousarray(mus, dtype=float)
    weights = np.ascontiguousarray(weights, dtype=float)
    a = M * E / (A + 1.0) ** 2
    if _heating_native is not None and len(e):
        noRecoil = zr == 0
        el, fl = (1.0, 1.0) if noRecoil else (float(c) for c in _lindhardConstants(zr, M, Z, A))
        raw = _heating_native.recoil_rows(e, mus, weights.ravel(), float(E), float(a), float(a * mp),
                                          float(E * mp), float(mp), float(M), el, fl, float(ed),
                                          bool(cm), noRecoil, *_nativeModel())
        return np.frombuffer(raw, dtype=float).copy()
    ec, mc = e[:, None], mus[None, :]
    if cm:
        er = a + mp * ec / M - 2.0 * np.sqrt(a * mp * ec / M) * mc
    else:
        er = (E - 2.0 * np.sqrt(E * mp * ec) * mc + mp * ec) / M
    rule = weights if weights.ndim == 2 else weights[None, :]
    return np.sum(lindhard(np.clip(er, 0.0, None), zr, M, Z, A, ed) * rule, axis=1)


def _momentaDamage(a, b, P, ar, zr, Z, A, ed, damage):
    """``_addMomenta(a, b, last=damage)`` of :func:`_independentRecoilDamage`, natively
    when the kernel is built: the same points in the same order, each block's np.sum."""
    if _heating_native is None:
        return _addMomenta(a, b, last=damage)
    (az, ap, aw, avz, ap2), (bz, bp, bw, bvz, bp2) = a, b
    vectors = [np.ascontiguousarray(v, dtype=float) for v in (az, avz, ap, ap2, aw, bz, bvz, bp, bp2, bw)]
    cos = np.cos((np.arange(_AZIMUTH_NODES) + 0.5) * np.pi / _AZIMUTH_NODES)
    rows = max(1, _MOMENTUM_BLOCK // (len(bz) * _AZIMUTH_NODES))
    noRecoil = zr == 0
    el, fl = (1.0, 1.0) if noRecoil else (float(c) for c in _lindhardConstants(zr, ar, Z, A))
    return _heating_native.momenta_damage(*vectors, cos, float(P), float(2.0 * ar), el, fl, float(ed),
                                          int(rows), noRecoil, *_nativeModel())


def _addMomenta(a, b, last=None):
    """The distribution of the sum of two independent, azimuthally symmetric momenta, binned.

    ``p_⊥² = ⟨a_⊥²⟩ + ⟨b_⊥²⟩ + 2 a̅_⊥ b̅_⊥ cos φ`` over the relative azimuth φ: its mean
    over φ and the bins' spreads is exact, as are those of ``p_z = a_z + b_z``. With
    ``last(p_z, var p_z, p_⊥², w)``, the sum of ``last`` over the points, not binned:
    the final sum's bins would put the damage at each bin's mean recoil."""
    (az, ap, aw, avz, ap2), (bz, bp, bw, bvz, bp2) = a, b
    cos = np.cos((np.arange(_AZIMUTH_NODES) + 0.5) * np.pi / _AZIMUTH_NODES)
    top = np.sqrt(ap2.max()) + np.sqrt(bp2.max())
    bins = _MomentumBins(az.min() + bz.min(), az.max() + bz.max(), 0.0, top) if last is None else None
    total = 0.0
    rows = max(1, _MOMENTUM_BLOCK // (len(bz) * _AZIMUTH_NODES))
    if last is None and _heating_native is not None:
        vectors = [np.ascontiguousarray(v, dtype=float) for v in (az, avz, ap, ap2, aw, bz, bvz, bp, bp2, bw)]
        zlo, zw, plo, pw = bins.edges
        raw = _heating_native.momenta_bin(*vectors, cos, float(zlo), float(zw), float(plo), float(pw),
                                          int(_MOMENTUM_BINS), int(rows))
        bins.sums = np.frombuffer(raw, dtype=float).reshape(5, -1).copy()
        return bins.moments()
    for s in range(0, len(az), rows):
        z = (az[s:s + rows, None] + bz[None, :]).ravel()
        vz = (avz[s:s + rows, None] + bvz[None, :]).ravel()
        w = (aw[s:s + rows, None] * bw[None, :]).ravel() / _AZIMUTH_NODES
        ab = 2.0 * (ap[s:s + rows, None] * bp[None, :]).ravel()
        sq = (ap2[s:s + rows, None] + bp2[None, :]).ravel()
        perp2 = np.clip(sq[:, None] + ab[:, None] * cos[None, :], 0.0, None).ravel()
        z, vz, w = (np.repeat(x, _AZIMUTH_NODES) for x in (z, vz, w))
        if last is not None:
            total += last(z, vz, perp2, w)
        else:
            bins.add(z, np.sqrt(perp2), w, vz)
    return total if last is not None else bins.moments()


def _independentRecoilDamage(particles, grid, A, Z, zr, ar, ed):
    """``⟨df(E_R)⟩`` of the residual ``(zr, ar)`` of a multi-particle emission, its
    momentum the balance of all the particles, each drawn independently.

    ``particles`` holds ``(form, awp, yields)``: a Monte Carlo form, the mass ratio,
    and the yield on ``grid`` (a whole number above the threshold). With ``p_in``
    the neutron's momentum, ``E_R = |p_in − Σ p_i|²/2M``; each ``p_i`` follows the
    particle's own energy-angle law in the lab, the azimuths independent. The
    evaluation states each particle's law and not their correlation, which the
    cascade does carry, so this is a model: measured against the residuals
    ENDF/B-VIII.1 does state (TALYS, GNASH cascades), it is closer than one
    particle's recoil in all six MT28 sections measured and equal in MT22 of
    medium-mass targets, where the alpha carries the momentum. The damage is computed at the particles'
    incident energies and read lin-lin between them.
    """
    lo, hi = float(np.min(grid)), float(np.max(grid))
    nodes = np.unique(np.concatenate([_incidentNodes(form) for form, _, _ in particles] + [[lo, hi]]))
    nodes = nodes[(nodes >= lo) & (nodes <= hi)]
    rules = {}

    def nodeAt(E):
        # A yield read between the grid's points around a threshold is a fraction.
        counts = [int(round(np.interp(E, grid, y))) for _, _, y in particles]
        if any(c <= 0 for c in counts):
            return 0.0                                       # below a threshold
        P = np.sqrt(2.0 * E)

        def damage(qz, vz, qp2, w, P=P):
            er = ((P - qz) ** 2 + vz + qp2) / (2.0 * ar)
            return float(np.sum(w * lindhard(er, zr, ar, Z, A, ed)))

        if sum(counts) == 1:
            # One particle: the recoil of each point of its law, nothing binned.
            (form, awp, _), = (p for p, c in zip(particles, counts) if c)
            qz, qp, w = _labMomenta(form, float(E), awp, A, rules)
            return damage(qz, 0.0, qp * qp, w)
        # Two neutrons of one law have the same distribution: binned once, added twice.
        momenta = []
        for (form, awp, _), count in zip(particles, counts):
            if count:
                momenta += [_binMomenta(*_labMomenta(form, float(E), awp, A, rules))] * count
        total = momenta[0]
        for one in momenta[1:-1]:
            total = _addMomenta(total, one)
        return _momentaDamage(total, momenta[-1], P, ar, zr, Z, A, ed, damage)

    values = np.array(_threadMap(nodeAt, nodes), dtype=float)
    return _onGrid(nodes, values, grid)


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
    the neutron balance stays where none does.

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
    # With displacements asked for, MT444's own pass computes them jointly where a
    # quadrature can (two-body, tabulated and analytic emission) and keeps them for
    # the models' passes below, which then only compute the rest.
    jointModels = tuple(displacementModels(Z, ed).values()) if displacements else ()
    jointTokens = (_JOINT.set(jointModels), _JOINT_STORE.set({})) if jointModels else None
    try:
        return _heatingBody(suite, grid, sigma, A, Z, ed, qm, qa, qbar, evaluation, zaTarget,
                            report, kinematic, displacements, tm, auxiliaryFissions)
    finally:
        if jointTokens is not None:
            _JOINT.reset(jointTokens[0]); _JOINT_STORE.reset(jointTokens[1])


def _heatingBody(suite, grid, sigma, A, Z, ed, qm, qa, qbar, evaluation, zaTarget, report, kinematic,
                 displacements, tm, auxiliaryFissions):
    """The body of :func:`_heatingOf` (its arguments checked and prepared there)."""
    from kika.nuclear_data import model as m
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
    if evaluation is not None:
        _prefetchHeavy(_heavyJobs(suite, evaluation, set(sigma), A, zaTarget or 1000 * Z), _energyThreads.get())
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
                # states the photons. HEATR's tabsq6 drops it for MF6, so a
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
                return _neutronResidualDamage(mt, neutrons, grid, A, Z, ed,
                                              notes if _DISPLACEMENTS.get() is None else None)
            if _levelParticle(mt) is not None:
                return _levelDamage(mt, grid, A, Z, ed, Q)
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
    # share the work; side-by-side passes measured no faster).
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
