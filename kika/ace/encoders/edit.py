"""Add or remove a reaction of a neutron ACE table, and get a consistent table back.

An ACE table is a flat array held together by locators: NXS counts, JXS block
positions, and inside the blocks LSIG, LAND, LDLW, IDAT, LNW, the laws' L and
LC... Adding a reaction moves almost every one of them. These functions only
edit the block objects -- one entry in MTR, LQR, TYR, SIG and, for a reaction
with secondary neutrons, AND and DLW -- on a copy of the table, and
:func:`~kika.ace.encoders.layout.rebuild` lays the XSS out again, so every
count and locator is recomputed from what is there.

The ESZ total (and the absorption, for a reaction without secondary neutrons)
is adjusted by the reaction's cross section unless ``adjust_esz=False``: ACE
states the total explicitly and a transport code samples collisions from it,
so a reaction added without it would never happen.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from kika.ace.classes.cross_section.cross_section_data import ReactionCrossSection
from kika.ace.encoders.layout import rebuild

__all__ = ["add_reaction", "remove_reaction"]


def _adjust_esz(ace, energy_idx: int, xs: np.ndarray, sign: float, absorption: bool) -> None:
    esz = ace.esz_block
    stop = energy_idx + len(xs)
    esz.total_xs[energy_idx:stop] += sign * xs
    if absorption:
        esz.absorption_xs[energy_idx:stop] += sign * xs


def add_reaction(ace, mt: int, xs, *, energy_idx: int, q_value: float, ty: int = 0,
                 angular=None, energy_laws: Optional[List] = None, neutron_yield=None,
                 adjust_esz: bool = True, strict: bool = True):
    """A new ``Ace``: ``ace`` with reaction ``mt`` added.

    Parameters
    ----------
    xs : array
        The cross section, on the ESZ grid from index ``energy_idx`` (0-based)
        to its end, in barns.
    q_value : float
        Q in MeV (LQR).
    ty : int
        TY: 0 for a reaction without secondary neutrons, else their number,
        negative in the centre-of-mass frame; 19 for fission, or ±(>100) with
        ``neutron_yield`` (an ``EnergyDependentYield``) for an energy-dependent
        number.
    angular, energy_laws
        For ``ty != 0``: the reaction's AND distribution (an angular
        distribution object, or ``IsotropicAngularDistribution``) and its DLW
        law chain (a list of energy-law objects with their applicability).
        A reaction with secondary neutrons is placed last among those, as
        NXS(5) requires.

    The input table is not modified.
    """
    mt = int(mt)
    new = ace.copy()
    rmt = new.reaction_mt_data
    mts = [int(m) for m in rmt.incident_neutron]
    if mt in mts or mt in (1, 2, 101):
        raise ValueError(f"MT={mt} is already in the table")
    xs = np.asarray(xs, dtype=float)
    nes = len(new.esz_block.energies)
    if energy_idx < 0 or energy_idx + len(xs) != nes:
        raise ValueError(f"MT={mt}: {len(xs)} values from index {energy_idx} do not end at the last of "
                         f"the {nes} grid points, as every ACE cross section does")
    secondary = int(ty) != 0
    if secondary and (angular is None or not energy_laws):
        raise ValueError(f"MT={mt} with TY={ty} needs its angular distribution and energy laws")
    if abs(int(ty)) > 100 and neutron_yield is None:
        raise ValueError(f"MT={mt} with TY={ty} needs its energy-dependent yield")

    at = len(rmt.secondary_neutron_mt) if secondary else len(mts)
    rmt.incident_neutron = np.insert(np.asarray(rmt.incident_neutron, dtype=float), at, mt)
    new.q_values.q_values = np.insert(np.asarray(new.q_values.q_values, dtype=float), at, q_value)
    new.particle_release.incident_neutron = np.insert(
        np.asarray(new.particle_release.incident_neutron, dtype=float), at, ty)
    if secondary:
        rmt.secondary_neutron_mt = np.append(np.asarray(rmt.secondary_neutron_mt, dtype=float), mt)
        new.angular_distributions.incident_neutron[mt] = angular
        new.energy_distributions.incident_neutron[mt] = list(energy_laws)
        if neutron_yield is not None:
            new.energy_distributions.neutron_yields[mt] = neutron_yield
    new.cross_section.reaction[mt] = ReactionCrossSection(
        mt=mt, energy_idx=energy_idx, num_energies=len(xs), _xs_entries=xs,
        _energy_entries=new.esz_block.energies[energy_idx:])
    if adjust_esz:
        _adjust_esz(new, energy_idx, xs, +1.0, absorption=not secondary)
    return rebuild(new, strict=strict)


def remove_reaction(ace, mt: int, *, adjust_esz: bool = True, strict: bool = True):
    """A new ``Ace``: ``ace`` without reaction ``mt`` (and its photons, if any).

    The input table is not modified.
    """
    mt = int(mt)
    new = ace.copy()
    rmt = new.reaction_mt_data
    mts = [int(m) for m in rmt.incident_neutron]
    if mt not in mts:
        raise ValueError(f"MT={mt} is not in MTR")
    keep = [k for k, m in enumerate(mts) if m != mt]
    secondary = mt in {int(m) for m in rmt.secondary_neutron_mt}
    r = new.cross_section.reaction[mt]
    if adjust_esz:
        _adjust_esz(new, r.energy_idx, np.asarray(r.xs_values, dtype=float), -1.0, absorption=not secondary)
    rmt.incident_neutron = np.asarray(rmt.incident_neutron)[keep]
    new.q_values.q_values = np.asarray(new.q_values.q_values)[keep]
    new.particle_release.incident_neutron = np.asarray(new.particle_release.incident_neutron)[keep]
    rmt.secondary_neutron_mt = np.asarray([m for m in rmt.secondary_neutron_mt if int(m) != mt], dtype=float)
    new.cross_section.reaction.pop(mt)
    new.angular_distributions.incident_neutron.pop(mt, None)
    new.energy_distributions.incident_neutron.pop(mt, None)
    new.energy_distributions.neutron_yields.pop(mt, None)
    photons = [int(m) for m in rmt.photon_production
               if int(m) // 1000 == mt or getattr(new.photon_production_xs.cross_sections.get(int(m)), "mtmult", None) == mt]
    if photons:
        _drop_photons(new, photons)
    return rebuild(new, strict=strict)


def _drop_photons(new, photons):
    rmt = new.reaction_mt_data
    keep = [k for k, m in enumerate(rmt.photon_production) if int(m) not in photons]
    rmt.photon_production = np.asarray(rmt.photon_production)[keep]
    for m in photons:
        new.photon_production_xs.cross_sections.pop(m, None)
        new.angular_distributions.photon_production.pop(m, None)
        new.energy_distributions.photon_production.pop(m, None)
    y = new.photon_yield_multipliers
    if y is not None:
        y.multiplier_mts = [m for m in y.multiplier_mts if m not in {p // 1000 for p in photons}]
