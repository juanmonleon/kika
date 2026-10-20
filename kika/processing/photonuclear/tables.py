r"""The model of a photonuclear evaluation → the contents of a ``.u`` ACE table.

What NJOY2016's ACER (``acephn``) does, with the choices below. Every number is
computed from the model; nothing is an option.

**Cross sections.** Each reaction's MF3 is made lin-lin to
:data:`~kika.algebra.refine.LINEARIZATION_TOLERANCE` (only a law other than
lin-lin adds points), the grid is the union of the reactions' grids, and each
reaction is stated from its first MF3 energy (IE), as ACER states it. The
particle-production sums MT 201-207 are not reactions and stay out of MTR and of
the total, which is the sum of the reactions written: ACER adds them in, and
Zr-90's total comes out twice MT 5 at the giant resonance. So do the sums ENDF
defines (MT 1, 3, MT 4 beside its levels, 103-107 beside theirs): H-2 states
MT 3 beside MT 50, its only partial.

**Particles.** Neutrons, photons, protons, deuterons, tritons, helium-3 and
alphas (IPT 1, 2, 9, 31, 32, 33, 34), each with the reactions that emit it,
MF4/MF5 reactions first and then MF6 ones ascending, as ACER orders them. A
product's yield is its multiplicity (MFTYPE 6, or 12 for an MF4/MF5 neutron),
made lin-lin. The production cross section (PXS) starts one grid point below
the first energy where the particle is produced, as ACER does.

**Distributions** are tabulated by :mod:`kika.processing.tabulation` with the
photon as projectile:

- Kalbach-Mann is ENDF-6 Eq. 6.5-6.6, as an ``energyAngularMC`` (law 61);
- a two-body particle (MF6 LAW 2), and the light recoil that comes with it (LAW 4:
  H-2's proton, its angle the partner's with mu -> -mu), has its centre-of-mass energy from the
  exact relativistic kinematics of a massless photon on a target at rest,
  :func:`photonTwoBodyEnergy`, written as a chain of law-33 pieces, each the
  chord of that energy, split until every piece is within
  :data:`TWO_BODY_TOLERANCE` of it (ACER's single law 33 neglects the photon's
  momentum: up to 2.6 % off for Be-9). The rest energies are the nuclear masses
  of the 2020 Atomic Mass Evaluation (``data/ame2020_masses.npz``: atomic less
  Z electrons), and Q is the reaction's, except in MT 5, which lumps channels
  and states Q = 0: there it is the channel's, from those masses and the
  residual level's energy (He-3's p + d: -5.49 MeV, where Q = 0 gives the
  deuteron the binding energy, as ACER's law 33 does);
- a photon line (a discrete energy) is a law-4 table of one discrete line, the
  form a ``.u`` reader takes (``phnout`` refuses law 2); and the products of one
  kind a reaction states separately (N-14's MF 12 lines) are one entry, their
  yields added and their laws a chain weighted by each one's share of the yield,
  so that no reaction is listed twice for a particle;
- the de-excitation photons a residual level states once (MF12/MF14 LO = 2,
  in PoPs) are written on every reaction that leaves that level, one line per
  transition with its yield, as the neutron tables do (ACER drops them);
- fission's delayed neutrons go with the prompt ones on MT 18: the yield is
  ν̄_p + ν̄_d and the energy law a chain, the prompt law with probability
  ν̄_p/ν̄ and each delayed family's spectrum with ν̄_i/ν̄ (ACER drops them).
  A family without a spectrum (Th-232, Pu-238 of ENDF/B-VIII.1) is not written,
  and the report says so.

**Heating** (MeV per collision in the table, eV here): what is not carried by a
transported neutron or photon stays where the collision happens. THN is
:math:`\sum_r \sigma_r h_r / \sigma_t`, with :math:`h_r` the sum over the light
charged particles and the residual nuclei of :math:`y \langle E'_{lab}\rangle`,
centre-of-mass energies taken to the laboratory with the photon's own
kinematics (:class:`kika.processing.heating._Kinematics`). A residual whose
evaluation gives no distribution (MF6 LAW 0, the residuals of MT 5 in most of
ENDF/B-VIII.1) recoils against what was emitted: in the centre of mass its
momentum balances theirs, so with the emissions taken as independent
:math:`\langle E_R\rangle = E_c(M_R) + \sum_i y_i m_i \langle E'_{i,cm}\rangle / M_R`
(a photon counting :math:`E^2/2` for :math:`m E`), :math:`M_R` the residuals' mean
mass. ACER adds E + Q for such a reaction instead, Q = 0 in MT 5, which keeps
every emitted nucleon's binding energy local: 5 MeV per collision in Bi-209 at
8 MeV, where only the recoil, keV, stays. Fission adds its fragments: (EFR + EGP)(E) from MF1/458
when the evaluation has it, else the reaction's Q less the neutrons' energy,
which also keeps the delayed betas and gammas local, and the report says so.
PHN of a particle is :math:`\sum y \sigma \langle E'_{lab}\rangle / \sigma_t`.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from kika._constants import NEUTRON_MASS_AMU
from kika.ace.classes.photonuclear import (ParticleProduction, PhotonuclearAce, PhotonuclearReaction,
                                           ProductReaction, ProductYield)

__all__ = ["photonuclearTable", "photonTwoBodyEnergy", "TWO_BODY_TOLERANCE"]

#: Relative error of the chain of law-33 pieces against the exact two-body energy.
TWO_BODY_TOLERANCE = 1.0e-4
#: The particles a photonuclear table carries, in ACER's order, with their IPT.
PARTICLES = (("n", 1), ("photon", 2), ("H1", 9), ("H2", 31), ("H3", 32), ("He3", 33), ("He4", 34))
#: Light particles' mass ratios to the neutron, as ENDF states them (AWP).
from kika.processing.tabulation.distribution import LIGHT_PARTICLE_AWP  # noqa: E402

#: The particle-production sums of MF3, which are not reactions.
PRODUCTION_SUMS = range(201, 208)
#: ENDF's sums (MT 1, 3 always; 4 and 103-107 when their components are there).
SUMS_OF = {1: None, 3: None, 4: range(50, 92), 103: range(600, 650), 104: range(650, 700),
           105: range(700, 750), 106: range(750, 800), 107: range(800, 850)}
#: The reaction that lumps channels and states no Q of its own.
LUMPED_MT = 5
#: Electron mass (u) and the energy of 1 u (eV), CODATA 2018.
ELECTRON_MASS_U = 5.48579909065e-4
U_EV = 931.49410242e6


def _ame():
    global _AME
    if _AME is None:
        from pathlib import Path

        d = np.load(Path(__file__).with_name("data") / "ame2020_masses.npz")
        _AME = dict(zip(d["za"].tolist(), d["mass_u"].tolist()))
    return _AME


_AME = None


def nuclearRestEnergy(za: int) -> float:
    """The rest energy (eV) of the nucleus ``za`` (1000 Z + A; 1 the neutron), from AME2020."""
    za = int(za)
    if za not in _ame():
        raise KeyError(f"ZA {za} is not in the 2020 Atomic Mass Evaluation")
    return (_ame()[za] - (za // 1000) * ELECTRON_MASS_U) * U_EV


def _m():
    from kika.nuclear_data import model

    return model


# ----------------------------------------------------------------------
# Two-body kinematics of a photon
# ----------------------------------------------------------------------

def photonTwoBodyEnergy(E, *, targetMass: float, emittedMass: float, Q: float) -> np.ndarray:
    """The emitted particle's kinetic energy in the centre of mass, eV, for a photon of ``E`` (eV).

    Exact and relativistic: a massless photon of momentum ``E/c`` on a target of
    rest energy ``targetMass`` (eV), leaving the particle (``emittedMass``) and a
    residual of rest energy ``targetMass - emittedMass - Q``. Zero below the
    threshold.
    """
    E = np.asarray(E, dtype=float)
    M_A, m_b = float(targetMass), float(emittedMass)
    M_B = M_A - m_b - float(Q)
    s = M_A * M_A + 2.0 * E * M_A
    root = np.sqrt(s)
    p2 = np.maximum((s - (m_b + M_B) ** 2) * (s - (m_b - M_B) ** 2), 0.0) / (4.0 * s)
    # T = sqrt(p² + m²) − m, written without the cancellation of a small p
    return p2 / (np.sqrt(p2 + m_b * m_b) + m_b) if m_b > 0 else np.sqrt(p2) + 0.0 * root


def _twoBodyThreshold(targetMass, emittedMass, Q) -> float:
    M_B = targetMass - emittedMass - Q
    return max(((emittedMass + M_B) ** 2 - targetMass ** 2) / (2.0 * targetMass), 0.0)


def _chordPieces(T, lo: float, hi: float, tol: float) -> List[float]:
    """Energies cutting [lo, hi] into pieces where the chord of ``T`` is within ``tol`` (relative)."""
    edges = [lo, hi]
    fractions = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    for _ in range(60):
        added = []
        for a, b in zip(edges[:-1], edges[1:]):
            q = a + (b - a) * fractions
            ta, tb, tq = T(np.array([a]))[0], T(np.array([b]))[0], T(q)
            chord = ta + (tb - ta) * fractions
            if np.any(np.abs(tq - chord) > tol * np.abs(tq)):
                added.append(0.5 * (a + b))
        if not added:
            return edges
        edges = sorted(edges + added)
    raise ValueError("the two-body energy did not come within its tolerance in 60 halvings")


def _twoBodyForm(form, *, targetMass: float, emittedMass: float, Q: float, hi: float, options):
    """A two-body (MF6 LAW 2) particle → a mixture of ``LevelEnergy`` pieces, centre of mass.

    Each piece is ``C (E - Ec)``, the chord of :func:`photonTwoBodyEnergy` between
    its ends, applying (probability 1) from its lower end to the next one's; the
    last one up to ``hi`` included.
    """
    m = _m()
    from kika.processing.tabulation.angular import tabulateAngular

    M_A, m_b = float(targetMass), float(emittedMass)
    lo = _twoBodyThreshold(M_A, m_b, Q)
    if not hi > lo:
        return None

    def T(e):
        return photonTwoBodyEnergy(e, targetMass=M_A, emittedMass=m_b, Q=Q)

    edges = _chordPieces(T, lo, hi, TWO_BODY_TOLERANCE)
    angular, _ = tabulateAngular(form.angular, options)
    if isinstance(angular, m.Isotropic2d):
        angular, _ = tabulateAngular(m.XYs2d(function1ds=[
            m.XYs1d(xs=[-1.0, 1.0], ys=[0.5, 0.5], outerDomainValue=lo),
            m.XYs1d(xs=[-1.0, 1.0], ys=[0.5, 0.5], outerDomainValue=hi)]), options)
    terms = []
    pieces = list(zip(edges[:-1], edges[1:]))
    for k, (a, b) in enumerate(pieces):
        ta, tb = float(T(np.array([a]))[0]), float(T(np.array([b]))[0])
        slope = (tb - ta) / (b - a)
        ec = a - ta / slope
        level = m.Uncorrelated(angular=angular, energy=m.LevelEnergy(threshold=ec, massRatio=slope),
                               label="MonteCarlo", productFrame=m.Frame.centerOfMass)
        last = k == len(pieces) - 1
        xs = ([lo] if k else []) + [a, b] + ([] if last else [hi])
        ys = ([0.0] if k else []) + ([1.0, 1.0] if last else [1.0, 0.0]) + ([] if last else [0.0])
        xs, ys = _distinct(xs, ys)
        weight = m.XYs1d(xs=xs, ys=ys, interpolation=m.Interpolation.flat)
        terms.append(m.Weighted(weight=weight, functional=level))
    return m.DistributionMixture(weighted=terms, label="MonteCarlo")


def _distinct(xs, ys):
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if out_x and x <= out_x[-1]:
            out_y[-1] = y
            continue
        out_x.append(x)
        out_y.append(y)
    if len(out_x) == 1:
        out_x.append(out_x[0] * (1.0 + 1e-12))
        out_y.append(out_y[0])
    return out_x, out_y


def _twoBodyRecoil(form, grid, *, A: float, targetMass: float, emittedMass: float, Q: float):
    """⟨E_lab⟩ of the residual a two-body photon reaction leaves (eV), on ``grid``."""
    from kika.processing.heating import NEUTRON_REST_ENERGY, _Kinematics, _meanMu

    M_A, m_b = float(targetMass), float(emittedMass)
    M_B = M_A - m_b - Q
    tb = photonTwoBodyEnergy(grid, targetMass=M_A, emittedMass=m_b, Q=Q)
    p2 = tb * (tb + 2.0 * m_b)
    tr = p2 / (np.sqrt(p2 + M_B * M_B) + M_B)
    k = _Kinematics(A=A, Q=Q, awp=M_B / NEUTRON_REST_ENERGY, photon=True)
    ec = k.boost(grid)
    mu = -_meanMu(form.angular, grid)
    return tr + ec + 2.0 * np.sqrt(ec * tr) * mu


# ----------------------------------------------------------------------
# The table
# ----------------------------------------------------------------------

def _linlin(form) -> Tuple[np.ndarray, np.ndarray]:
    """A 1-d function as lin-lin points (a step is a repeated abscissa)."""
    m = _m()
    from kika.algebra import interval_laws, to_linlin
    from kika.algebra.refine import LINEARIZATION_TOLERANCE

    whole = form if isinstance(form, m.Regions1d) else m.Regions1d(function1ds=[form])
    x, y, pairs = whole.toEndfRegions()
    return to_linlin(np.asarray(x, dtype=float), np.asarray(y, dtype=float), interval_laws(len(x), pairs),
                     tol=LINEARIZATION_TOLERANCE)


def _onGrid(x, y, grid, start):
    from kika.algebra.evaluate import sample_on_union
    from kika.algebra.laws import LINLIN

    out = np.zeros(grid.size)
    out[start:] = sample_on_union(x, y, LINLIN, grid[start:])
    return out


def _multiplicity(product, mt, lo, hi, report):
    """``(x, y)`` lin-lin of a product's multiplicity, or ``None``."""
    m = _m()
    from kika.hdf5.processing import _evaluated

    form = _evaluated(product.multiplicity) if product.multiplicity is not None else None
    if isinstance(form, (m.Constant1d, m.Polynomial1d)) and (
            isinstance(form, m.Constant1d) or len(form.coefficients) == 1):
        c = float(form.constant if isinstance(form, m.Constant1d) else form.coefficients[0])
        return np.array([lo, hi]), np.array([c, c])
    if isinstance(form, (m.XYs1d, m.Regions1d)):
        return _linlin(form)
    count = m.impliedMultiplicity(mt, product.pid)
    if count:
        return np.array([lo, hi]), np.array([float(count)] * 2)
    report.lost(f"MT{mt} {product.label}: multiplicity {type(form).__name__} is not a number; "
                f"product not written")
    return None


def _massRatio(pops, pid) -> float:
    return float(pops[pid].mass.convertedTo("amu").value) / NEUTRON_MASS_AMU


class _Builder:
    def __init__(self, suite, report, options):
        m = _m()
        self.m, self.suite, self.report, self.options = m, suite, report, options
        target = suite.PoPs[suite.target]
        self.Z, self.A = int(target.Z), int(target.A)
        self.za = 1000 * self.Z + self.A
        awr = getattr(getattr(suite, "provenance", None), "awr", None)
        self.awr = float(awr) if awr else _massRatio(suite.PoPs, suite.target)

    # --- cross sections ----------------------------------------------------

    def reactions(self):
        s = self.suite
        chosen, skipped = [], []
        present = {int(r.ENDF_MT) for r in s.reactions}
        for reaction in list(s.reactions):
            mt = int(reaction.ENDF_MT)
            parts = SUMS_OF.get(mt, ())
            summed = mt in SUMS_OF and (parts is None and len(present) > 1
                                        or parts is not None and bool(present & set(parts)))
            (skipped if mt in PRODUCTION_SUMS or summed else chosen).append(reaction)
        if not any(int(r.ENDF_MT) == 18 for r in chosen):
            chosen += list(getattr(s, "fissionComponents", ()))
        if skipped:
            self.report.warn(f"MT{sorted(int(r.ENDF_MT) for r in skipped)} are sums (of particle productions "
                             f"or of partials the table has): not in MTR nor in the total")
        return sorted(chosen, key=lambda r: int(r.ENDF_MT))

    def crossSections(self, reactions):
        from kika.algebra.grid import union
        from kika.hdf5.processing import _evaluated

        tables = {}
        for r in reactions:
            form = _evaluated(r.crossSection)
            if form is None:
                raise ValueError(f"MT{r.ENDF_MT} has no cross section")
            tables[int(r.ENDF_MT)] = _linlin(form)
        grid = union([x for x, _ in tables.values()])
        values = {}
        for mt, (x, y) in tables.items():
            start = int(np.searchsorted(grid, x[0], side="left"))
            values[mt] = (start, _onGrid(x, y, grid, start))
        return grid, tables, values

    # --- products ----------------------------------------------------------

    def tabulated(self, product, reaction, mt, Q):
        """``(Monte Carlo form, mftype)`` of one light product, or ``None``."""
        m = self.m
        from kika.hdf5.processing import _evaluated, _mf6Record
        from kika.processing.tabulation import TabulationContext, tabulateDistribution

        form = _evaluated(product.distribution)
        record = _mf6Record(reaction, product)
        mftype = 6 if record is not None or product.pid != "n" else 12
        awp = LIGHT_PARTICLE_AWP.get(product.pid, 1.0 if product.pid == "n" else 0.0)
        if isinstance(form, m.AngularTwoBody):
            hi = float(self.ranges[mt][1])
            if form.isRecoil:
                # the light recoil of a two-body reaction (H-2's proton): its partner's
                # kinematics with its own mass, and the partner's angle reversed
                partner = next((o for o in reaction.outputChannel.products if o is not product
                                and isinstance(_evaluated(o.distribution), m.AngularTwoBody)
                                and not _evaluated(o.distribution).isRecoil), None)
                if partner is None:
                    self.report.lost(f"MT{mt} {product.label}: a recoil with no two-body partner")
                    return None
                masses = self.twoBodyMasses(reaction, partner, mt, Q)
                masses = dict(targetMass=masses["targetMass"], Q=masses["Q"],
                              emittedMass=masses["targetMass"] - masses["emittedMass"] - masses["Q"])
                form = _Reversed(_evaluated(partner.distribution))
            else:
                masses = self.twoBodyMasses(reaction, product, mt, Q)
            two = _twoBodyForm(form, hi=hi, options=self.options, **masses)
            return (two, mftype, (two, masses)) if two is not None else None
        if form is None or isinstance(form, (m.Unspecified, m.Branching3d)):
            self.report.lost(f"MT{mt} {product.label}: no distribution; product not written")
            return None
        discrete = (record.get("law_fields") or {}).get("nd") if record is not None else None
        context = TabulationContext(awr=self.awr, zaTarget=self.za, mt=mt, Q=Q, pid=product.pid,
                                    zaEmitted=0 if product.pid == "photon" else m.zaFromPid(product.pid),
                                    discrete=discrete, options=self.options, report=self.report,
                                    projectile="photon")
        tabulated = tabulateDistribution(form, context)
        if tabulated is None:
            self.report.lost(f"MT{mt} {product.label}: a {type(form).__name__} has no Monte Carlo form")
            return None
        if product.pid == "photon":
            from kika.hdf5.processing import _explicitPhotonAngle

            _explicitPhotonAngle(tabulated, reaction)
            tabulated = _linesAsTables(tabulated, *self.ranges[mt])
        return tabulated, mftype, form

    def twoBodyMasses(self, reaction, product, mt, Q):
        """``targetMass``, ``emittedMass`` (eV) and the channel's ``Q`` of a two-body product."""
        m = self.m
        emitted = 1 if product.pid == "n" else m.zaFromPid(product.pid)
        M_A, m_b = nuclearRestEnergy(self.za), nuclearRestEnergy(emitted)
        if mt != LUMPED_MT:
            return dict(targetMass=M_A, emittedMass=m_b, Q=float(Q))
        residual = self.za - emitted
        level = 0.0
        for other in reaction.outputChannel.products:
            if other is product or other.pid in ("photon",):
                continue
            za = 1 if other.pid == "n" else m.zaFromPid(other.pid)
            if za == residual and other.pid in self.suite.PoPs:
                energy = getattr(self.suite.PoPs[other.pid], "energy", None)
                level = float(energy.value) if energy is not None else 0.0
                break
        return dict(targetMass=M_A, emittedMass=m_b, Q=M_A - m_b - nuclearRestEnergy(residual) - level)

    def build(self) -> PhotonuclearAce:
        m, report = self.m, self.report
        from kika.nuclear_data.model.endf_conversion import reactionQI

        reactions = self.reactions()
        grid, tables, values = self.crossSections(reactions)
        self.ranges = {mt: (x[0], x[-1]) for mt, (x, _) in tables.items()}
        total = np.sum([v for _, v in values.values()], axis=0)
        table = PhotonuclearAce(za=self.za, isomeric_state=int(getattr(self.suite.PoPs[self.suite.target],
                                                                       "index", 0) or 0))
        table.energies, table.total = grid, total
        for r in reactions:
            mt = int(r.ENDF_MT)
            start, v = values[mt]
            table.reactions.append(PhotonuclearReaction(mt=mt, q_value=float(reactionQI(r, self.suite) or 0.0),
                                                        first_index=start + 1, cross_section=v[start:]))

        entries: Dict[str, list] = {pid: [] for pid, _ in PARTICLES}
        local = np.zeros(grid.size)
        lost_residuals, estimated = [], []
        for r in reactions:
            mt = int(r.ENDF_MT)
            Q = float(reactionQI(r, self.suite) or 0.0)
            sigma = values[mt][1]
            lo, hi = self.ranges[mt]
            explicitPhoton = False
            twoBody = None
            before = {pid: len(v) for pid, v in entries.items()}
            for product in r.outputChannel.products:
                if product.pid in entries:
                    made = self.tabulated(product, r, mt, Q)
                    if made is None:
                        continue
                    form, mftype, evaluated = made
                    if isinstance(evaluated, tuple):
                        twoBody = evaluated
                    mult = _multiplicity(product, mt, lo, hi, report)
                    if mult is None:
                        continue
                    entries[product.pid].append(dict(mt=mt, form=form, mftype=mftype, yields=mult,
                                                     sigma=sigma, Q=Q, reaction=r))
                    explicitPhoton |= product.pid == "photon"
            # residuals: their energy stays local
            unstated = []
            for product in r.outputChannel.products:
                if product.pid in entries:
                    continue
                heat = self.residualHeating(product, r, mt, Q, grid, twoBody, lost_residuals)
                if heat is None:
                    unstated.append(product)
                else:
                    local += sigma * heat
            if not explicitPhoton:
                from kika.hdf5.processing import _cascadePhotons

                # every line of the level's cascade in one entry: the yield is their sum,
                # the energy law a table of discrete lines with probabilities y_i / sum y
                lines = [(float(p.distribution["MonteCarlo"].energy.energy), float(p.multiplicity.form.ys[0]))
                         for p in _cascadePhotons(self.suite.PoPs, _NoLightResiduals(r), self.awr)]
                if lines:
                    form = m.Uncorrelated(angular=_isotropicLab(lo, hi), energy=_lineTable(lines, lo, hi),
                                          label="MonteCarlo", productFrame=m.Frame.lab)
                    count = sum(y for _, y in lines)
                    entries["photon"].append(dict(mt=mt, form=form, mftype=12,
                                                  yields=(np.array([lo, hi]), np.array([count, count])),
                                                  sigma=sigma, Q=Q, reaction=r))
            if unstated:
                emitted = [(pid, e) for pid, v in entries.items() for e in v[before[pid]:]]
                local += sigma * self.momentumRecoil(unstated, emitted, mt, grid)
                estimated.append(mt)
            if mt == 18:
                local += sigma * self.fission(r, entries["n"], grid, Q)

        if lost_residuals:
            report.warn(f"{len(lost_residuals)} residual nuclei with no distribution in MT "
                        f"{sorted(set(lost_residuals))}: their recoil adds nothing to the heating")
        if estimated:
            report.approximated(f"MT{sorted(set(estimated))}: residuals with no distribution (MF6 LAW 0) heat by "
                                f"their recoil from momentum balance, the emissions taken as independent")

        thn = np.array(local)
        for pid, ipt in PARTICLES:
            if not entries[pid]:
                continue
            block = self.particle(pid, ipt, entries[pid], grid, total)
            table.particles.append(block)
            if pid not in ("n", "photon"):
                thn[block.heating_first_index - 1:] += block.heating * total[block.heating_first_index - 1:]
        table.heating = np.divide(thn, total, out=np.zeros(grid.size), where=total > 0)
        return table

    # --- heating pieces ----------------------------------------------------

    def residualHeating(self, product, reaction, mt, Q, grid, twoBody, lost):
        m = self.m
        from kika.hdf5.processing import _evaluated
        from kika.processing.heating import _Kinematics, _meanEnergyOf
        from kika.processing.tabulation import TabulationContext, tabulateDistribution

        form = _evaluated(product.distribution)
        if isinstance(form, m.AngularTwoBody) and form.isRecoil:
            if twoBody is None:
                lost.append(mt)
                return np.zeros(grid.size)
            partner, masses = twoBody
            return _twoBodyRecoil(partner.weighted[0].functional, grid, A=self.awr, **masses)
        if form is None or isinstance(form, (m.Unspecified, m.Branching3d)):
            return None
        lo, hi = self.ranges[mt]
        mult = _multiplicity(product, mt, lo, hi, self.report)
        if mult is None:
            return np.zeros(grid.size)
        context = TabulationContext(awr=self.awr, zaTarget=self.za, mt=mt, Q=Q, pid=product.pid,
                                    zaEmitted=m.zaFromPid(product.pid), options=self.options,
                                    report=self.report, projectile="photon")
        tabulated = tabulateDistribution(form, context)
        if tabulated is None:
            lost.append(mt)
            return np.zeros(grid.size)
        awp = _massRatio(self.suite.PoPs, product.pid) if product.pid in self.suite.PoPs else float(
            m.zaFromPid(product.pid) % 1000)
        mean = _meanEnergyOf(tabulated, grid, _Kinematics(A=self.awr, Q=Q, awp=awp, photon=True))
        return np.interp(grid, mult[0], mult[1], left=0.0, right=0.0) * mean

    def meanOf(self, pid, e, grid):
        """A product entry's ⟨E'_lab⟩ on ``grid`` (eV), computed once."""
        if "mean" not in e:
            from kika.processing.heating import _Kinematics, _meanEnergyOf

            awp = LIGHT_PARTICLE_AWP.get(pid, 1.0 if pid == "n" else 0.0)
            e["mean"] = _meanEnergyOf(e["form"], grid, _Kinematics(A=self.awr, Q=e["Q"], awp=awp, photon=True))
        return e["mean"]

    def momentumRecoil(self, residuals, emitted, mt, grid):
        """⟨E_R⟩ per reaction (eV) of residuals without a distribution, from momentum balance."""
        m = self.m
        from kika.processing.heating import NEUTRON_REST_ENERGY, _Kinematics

        lo, hi = self.ranges[mt]
        yR, massR = np.zeros(grid.size), np.zeros(grid.size)
        for product in residuals:
            mult = _multiplicity(product, mt, lo, hi, self.report)
            if mult is None:
                continue
            za = m.zaFromPid(product.pid)
            try:
                M = nuclearRestEnergy(za)
            except KeyError:
                M = float(self.suite.PoPs[product.pid].mass.convertedTo("amu").value) * U_EV
            y = np.interp(grid, mult[0], mult[1], left=0.0, right=0.0)
            yR += y
            massR += y * M
        M_R = np.divide(massR, yR, out=np.full(grid.size, self.awr * NEUTRON_REST_ENERGY), where=yR > 0)
        balance = np.zeros(grid.size)
        for pid, e in emitted:
            y = np.interp(grid, e["yields"][0], e["yields"][1], left=0.0, right=0.0)
            mean = self.meanOf(pid, e, grid)
            if pid == "photon":
                balance += y * 0.5 * mean * mean
                continue
            mass = nuclearRestEnergy(1 if pid == "n" else m.zaFromPid(pid))
            ec = _Kinematics(A=self.awr, Q=0.0, awp=mass / NEUTRON_REST_ENERGY, photon=True).boost(grid)
            balance += y * mass * np.clip(mean - ec, 0.0, None)
        ecR = _Kinematics(A=self.awr, Q=0.0, awp=1.0, photon=True).boost(grid) * M_R / NEUTRON_REST_ENERGY
        return yR * ecR + np.divide(balance, M_R, out=np.zeros(grid.size), where=M_R > 0) * (yR > 0)

    def fission(self, reaction, neutrons, grid, Q):
        """The fragments' energy per fission (eV) on ``grid``."""
        data = reaction.outputChannel.fissionFragmentData
        releases = list(getattr(data, "fissionEnergyReleases", ()) or [])
        release = releases[0] if releases else None
        if release is not None and release.promptProductKE is not None and release.promptGammaEnergy is not None:
            efr = np.array([float(release.promptProductKE.evaluate(e)) for e in grid])
            egp = np.array([float(release.promptGammaEnergy.evaluate(e)) for e in grid])
            return efr + egp
        from kika.processing.heating import _Kinematics, _meanEnergyOf

        carried = np.zeros(grid.size)
        for e in neutrons:
            if e["mt"] != 18:
                continue
            mean = _meanEnergyOf(e["form"], grid, _Kinematics(A=self.awr, Q=Q, awp=1.0, photon=True))
            carried += np.interp(grid, e["yields"][0], e["yields"][1], left=0.0, right=0.0) * mean
        self.report.warn("MT18: no MF1/458; the fragments' share is Q less the neutrons' energy, which "
                         "keeps the delayed betas and gammas local too")
        return np.clip(Q + grid * 0.0 - carried, 0.0, None) if Q > 0 else np.zeros(grid.size)

    # --- one particle ------------------------------------------------------

    def particle(self, pid, ipt, entries, grid, total) -> ParticleProduction:
        m = self.m
        from kika.ace.model_adapter.encode import (_angularOf, _angularRecord, _chain, _correlatedRecord,
                                                   _frameOf, CORRELATED_LAWS)
        from kika.processing.heating import _Kinematics, _meanEnergyOf

        if pid == "n":
            entries = self.withDelayed(entries)
        entries = self.merged(pid, entries)
        entries.sort(key=lambda e: (e["mftype"] != 12 or pid != "n", e["mt"]))
        awp = LIGHT_PARTICLE_AWP.get(pid, 1.0 if pid == "n" else 0.0)
        production, heat = np.zeros(grid.size), np.zeros(grid.size)
        thresholds = []
        reactions = []
        for e in entries:
            mt, form = e["mt"], e["form"]
            x, y = e["yields"]
            on = np.interp(grid, x, y, left=0.0, right=0.0)
            production += on * e["sigma"]
            heat += on * e["sigma"] * self.meanOf(pid, e, grid)
            lo = self.ranges[mt][0]
            positive = x[np.asarray(y) > 0]
            thresholds.append(max(lo, float(positive[0])) if positive.size else lo)
            attributes = {"laws": [33] * len(form.weighted)} if _isLevelChain(form) else {}
            laws, numbers = _chain(form, attributes, f"{pid} MT{mt}", self.ranges[mt])
            if any(n in CORRELATED_LAWS for n in numbers):
                angular, kind = _correlatedRecord(mt), -1
            else:
                angular = _angularRecord(_angularOf(form), mt, photon=pid == "photon")
                stated = getattr(angular, "_energies", None)
                kind = 0 if stated is None or len(stated) == 0 else 1
                angular = angular if kind == 1 else None
            frame = -1 if _frameOf(form) else 1
            yields = ProductYield(mftype=e["mftype"], mt_mult=mt, energies=np.asarray(x, dtype=float),
                                  values=np.asarray(y, dtype=float))
            reactions.append(ProductReaction(mt=mt, frame=frame, yield_=yields, angular_locator=kind,
                                             angular=angular, laws=laws))
        threshold = min(thresholds)
        first = int(np.searchsorted(grid, threshold * (1.0 - 1e-10), side="left"))
        start = max(first - 1, 0)
        heating = np.divide(heat, total, out=np.zeros(grid.size), where=total > 0)
        return ParticleProduction(ipt=ipt, production_first_index=start + 1, production=production[start:],
                                  heating_first_index=start + 1, heating=heating[start:],
                                  reactions=reactions)

    def merged(self, pid, entries):
        """Entries of one reaction joined: yields added, laws a chain weighted by yield share."""
        m = self.m
        out, groups = [], {}
        for e in entries:
            groups.setdefault(e["mt"], []).append(e)
        for mt in sorted(groups, key=lambda k: entries.index(groups[k][0])):
            group = groups[mt]
            if len(group) == 1:
                out.append(group[0])
                continue
            from kika.ace.model_adapter.encode import _frameOf

            frames = {_frameOf(e["form"]) for e in group}
            if len(frames) > 1:
                raise NotImplementedError(f"MT{mt} {pid}: products in both frames cannot share one entry")
            xs = np.unique(np.concatenate([e["yields"][0] for e in group]))
            each = [np.interp(xs, *e["yields"], left=0.0, right=0.0) for e in group]
            total = np.sum(each, axis=0)
            terms = []
            for e, y in zip(group, each):
                if isinstance(e["form"], m.DistributionMixture):
                    raise NotImplementedError(f"MT{mt} {pid}: a law chain inside a joined entry")
                weight = m.XYs1d(xs=xs, ys=np.divide(y, total, out=np.zeros(xs.size), where=total > 0))
                terms.append(m.Weighted(weight=weight, functional=e["form"]))
            joined = {k: v for k, v in group[0].items() if k != "mean"}
            joined["form"] = m.DistributionMixture(weighted=terms, label="MonteCarlo")
            joined["yields"] = (xs, total)
            out.append(joined)
            self.report.warn(f"MT{mt}: {len(group)} {pid} products joined in one entry (a chain of laws)")
        return out

    def withDelayed(self, entries):
        """MT18's prompt neutrons joined with the delayed families into one entry (law chain)."""
        m = self.m
        out = []
        for e in entries:
            if e["mt"] != 18:
                out.append(e)
                continue
            data = e["reaction"].outputChannel.fissionFragmentData
            families = []
            for family in (getattr(data, "delayedNeutrons", ()) or []):
                product = family.product
                if product is None or product.distribution is None or not len(product.distribution):
                    self.report.warn(f"MT18: delayed family {family.label} has no spectrum; its neutrons "
                                     f"are not written")
                    continue
                made = self.tabulated(product, e["reaction"], 18, e["Q"])
                mult = _multiplicity(product, 18, *self.ranges[18], self.report)
                if made is None or mult is None:
                    continue
                families.append((made[0], mult))
            if not families:
                out.append(e)
                continue
            xs = np.unique(np.concatenate([e["yields"][0]] + [f[1][0] for f in families]))
            prompt = np.interp(xs, *e["yields"])
            each = [np.interp(xs, *f[1]) for f in families]
            totalNu = prompt + np.sum(each, axis=0)
            forms = [e["form"]] + [f[0] for f in families]
            if _correlated(e["form"]):
                forms = [e["form"]] + [_isotropic61(f, self.options) for f in forms[1:]]
            terms = []
            for form, nu in zip(forms, [prompt] + each):
                weight = m.XYs1d(xs=xs, ys=np.divide(nu, totalNu, out=np.zeros(xs.size), where=totalNu > 0))
                functional = form
                if isinstance(form, m.DistributionMixture):
                    raise NotImplementedError("MT18: a prompt law chain with delayed families")
                terms.append(m.Weighted(weight=weight, functional=functional))
            joined = {k: v for k, v in e.items() if k != "mean"}
            joined["form"] = m.DistributionMixture(weighted=terms, label="MonteCarlo")
            joined["yields"] = (xs, totalNu)
            out.append(joined)
            self.report.warn(f"MT18: {len(families)} delayed families written with the prompt neutrons, "
                             f"yield nu_p + nu_d and one law each")
        return out


class _Reversed:
    """A two-body form seen from the other body: the cosine reversed (mu -> -mu)."""

    def __init__(self, form):
        m = _m()
        self.isRecoil = False
        self.productFrame = form.productFrame
        angular = form.angular
        if isinstance(angular, m.Isotropic2d) or angular is None:
            self.angular = angular
            return
        flipped = []
        for f in angular.function1ds:
            if isinstance(f, m.Legendre):
                c = np.array(f.coefficients, dtype=float)
                c[1::2] *= -1.0
                flipped.append(m.Legendre(coefficients=c, outerDomainValue=f.outerDomainValue))
            else:
                x, y = np.asarray(f.xs, dtype=float), np.asarray(f.ys, dtype=float)
                flipped.append(m.XYs1d(xs=-x[::-1], ys=y[::-1], interpolation=f.interpolation,
                                       outerDomainValue=f.outerDomainValue))
        self.angular = m.XYs2d(function1ds=flipped, interpolation=angular.interpolation)


class _NoLightResiduals:
    """A reaction seen without its light products, for :func:`_cascadePhotons`."""

    def __init__(self, reaction):
        self._r = reaction
        self.ENDF_MT = reaction.ENDF_MT
        self.crossSection = reaction.crossSection
        light = {pid for pid, _ in PARTICLES}
        channel = copy.copy(reaction.outputChannel)
        products = [p for p in reaction.outputChannel.products if p.pid not in light]
        self.outputChannel = _Channel(products)


class _Channel:
    def __init__(self, products):
        self.products = products


def _lineTable(lines, lo, hi):
    """Photon lines ``[(energy, yield)]`` as the outgoing-energy tables of law 4 at ``lo`` and ``hi``.

    Each line is discrete with probability ``yield / sum(yield)``; a line repeated
    in the cascade (the same energy from two paths) is one line with both yields.
    """
    m = _m()
    merged = {}
    for energy, y in lines:
        merged[energy] = merged.get(energy, 0.0) + y
    energies = np.array(sorted(merged))
    p = np.array([merged[e] for e in energies])
    p = p / p.sum()
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    tables = [m.Xs_pdf_cdf1d(xs=energies, pdf=p, cdf=cdf, interpolation=m.Interpolation.flat,
                             numberOfDiscrete=energies.size, outerDomainValue=float(e)) for e in (lo, hi)]
    return m.XYs2d(function1ds=tables)


def _isotropicLab(lo, hi):
    from kika.hdf5.processing import _isotropicPhotonAngle

    return _isotropicPhotonAngle(lo, hi)


def _linesAsTables(form, lo, hi):
    """Photon lines stated as discrete energies → law-4 tables of one discrete line each."""
    m = _m()
    if isinstance(form, m.DistributionMixture):
        for term in form.weighted:
            term.functional = _linesAsTables(term.functional, lo, hi)
        return form
    if isinstance(form, m.Uncorrelated) and isinstance(form.energy, (m.DiscretePhotonEnergy, m.DiscreteGamma)):
        energy = form.energy
        if isinstance(energy, m.DiscretePhotonEnergy) and int(energy.primaryFlag) != 0:
            raise NotImplementedError("a primary photon (its energy moves with the incident energy) in a .u table")
        value = float(energy.energy if isinstance(energy, m.DiscretePhotonEnergy) else energy.value)
        return m.Uncorrelated(angular=form.angular, energy=_lineTable([(value, 1.0)], lo, hi),
                              label=form.label, productFrame=form.productFrame)
    if isinstance(form, m.Uncorrelated) and isinstance(form.energy, m.PrimaryGamma):
        raise NotImplementedError("a primary photon (its energy moves with the incident energy) in a .u table")
    return form


def _isLevelChain(form) -> bool:
    m = _m()
    return isinstance(form, m.DistributionMixture) and all(
        isinstance(w.functional, m.Uncorrelated) and isinstance(w.functional.energy, m.LevelEnergy)
        for w in form.weighted)


def _correlated(form) -> bool:
    m = _m()
    return isinstance(form, (m.EnergyAngularMC, m.KalbachMann))


def _isotropic61(form, options):
    """An uncorrelated, isotropic spectrum as an energyAngularMC with isotropic cosine tables."""
    m = _m()
    from kika.processing.tabulation.correlated import _isotropicTable

    energy = form.energy
    tables = []
    for t in energy.function1ds:
        tables.append(m.XYs2d(function1ds=[_isotropicTable(float(x)) for x in t.xs],
                              outerDomainValue=t.outerDomainValue))
    return m.EnergyAngularMC(energy=energy, energyAngular=m.XYs3d(function2ds=tables), label="MonteCarlo",
                             productFrame=form.productFrame)


def photonuclearTable(suite, *, report=None, options=None) -> PhotonuclearAce:
    """The contents of the ``.u`` table of a photonuclear evaluation (``suite.projectile == 'photon'``).

    Returns a :class:`PhotonuclearAce` without header or XSS; the report says
    what was left out or approximated, and why.
    """
    m = _m()
    from kika.processing.tabulation import TabulationOptions

    if getattr(suite, "projectile", None) != "photon":
        raise ValueError(f"a photonuclear table needs a photon projectile, not {suite.projectile!r}")
    report = report if report is not None else m.ConversionReport()
    builder = _Builder(suite, report, options or TabulationOptions())
    table = builder.build()
    table.report = report
    table.awr = builder.awr
    return table
