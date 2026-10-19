"""What a neutron ACE table must satisfy, read or about to be written.

Two kinds of check, each returning a list of :class:`Issue` (empty = fine):

* :func:`structure_issues`: the blocks agree with each other, which is what
  the encoder needs to lay them out (ACEtk checks these): one Q, TY and cross
  section per MTR reaction; an angular and an energy distribution for each
  reaction with secondary neutrons; ν whenever MT=18 is there; each photon and
  particle production entry with its cross section, angle and energy law;
  SIG records inside the energy grid (IE + NE - 1 <= NES).
* :func:`physics_issues`: what the numbers must look like, which ACEtk does
  not check: ascending grids, non-negative cross sections, CDFs from 0 to 1,
  cosines in [-1, 1], law probabilities in [0, 1].

``encode_xss`` refuses a table with structure issues (:class:`AceContractError`
lists them); physics issues are reported, since NJOY itself writes a few
(tiny negative cross sections in some evaluations, for instance).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from kika.ace.classes.ixs_keys import ixs_keys


@dataclass(frozen=True)
class Issue:
    block: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.block} {self.where}: {self.message}" if self.where else f"{self.block}: {self.message}"


class AceContractError(ValueError):
    """The blocks of an Ace do not agree with each other: it cannot be written."""

    def __init__(self, issues: List[Issue]):
        self.issues = list(issues)
        super().__init__("; ".join(str(i) for i in self.issues[:10])
                         + (f" (and {len(self.issues) - 10} more)" if len(self.issues) > 10 else ""))


def structure_issues(ace) -> List[Issue]:
    out: List[Issue] = []
    add = lambda b, w, m: out.append(Issue(b, w, m))  # noqa: E731
    rmt = ace.reaction_mt_data
    if ace.esz_block is None or len(ace.esz_block.energies) == 0:
        add("ESZ", "", "no energy grid")
        return out
    nes = len(ace.esz_block.energies)
    for name in ("total_xs", "absorption_xs", "elastic_xs", "heating_numbers"):
        if len(getattr(ace.esz_block, name)) != nes:
            add("ESZ", name, f"{len(getattr(ace.esz_block, name))} values on a grid of {nes}")
    mts = [int(m) for m in rmt.incident_neutron]
    if len(set(mts)) != len(mts):
        add("MTR", "", "repeated MT")
    if len(ace.q_values.q_values) != len(mts):
        add("LQR", "", f"{len(ace.q_values.q_values)} Q values for {len(mts)} MTR reactions")
    if len(ace.particle_release.incident_neutron) != len(mts):
        add("TYR", "", f"{len(ace.particle_release.incident_neutron)} TY for {len(mts)} MTR reactions")
    xs = ace.cross_section.reaction if ace.cross_section is not None else {}
    for mt in mts:
        r = xs.get(mt)
        if r is None:
            add("SIG", f"MT={mt}", "no cross section")
        elif r.energy_idx < 0 or r.energy_idx + len(r._xs_entries) > nes:
            add("SIG", f"MT={mt}", f"IE={r.energy_idx + 1}, NE={len(r._xs_entries)} run past the grid of {nes}")
    sec = [int(m) for m in rmt.secondary_neutron_mt]
    if any(m not in mts for m in sec):
        add("MTR", "", "a reaction with secondary neutrons is not in MTR")
    if mts[:len(sec)] != sec:
        add("MTR", "", "the reactions with secondary neutrons must come first in MTR (NXS(5))")
    ad, ed = ace.angular_distributions, ace.energy_distributions
    if ad is None or ad.elastic is None:
        add("AND", "elastic", "no elastic angular distribution")
    for mt in sec:
        if ad is None or mt not in ad.incident_neutron:
            add("AND", f"MT={mt}", "no angular distribution")
        if ed is None or not ed.incident_neutron.get(mt):
            add("DLW", f"MT={mt}", "no energy distribution")
    tys = dict(zip(mts, (int(t) for t in ace.particle_release.incident_neutron)))
    for mt, ty in tys.items():
        if abs(ty) > 100 and (ed is None or mt not in ed.neutron_yields):
            add("DLW", f"MT={mt}", f"TY={ty} but no energy-dependent yield")
    if 18 in mts and (ace.nubar is None or not ace.nubar.has_nubar):
        add("NU", "", "MT=18 without nu")
    for mt in (int(m) for m in rmt.photon_production):
        if ace.photon_production_xs is None or mt not in ace.photon_production_xs.cross_sections:
            add("SIGP", f"MT={mt}", "no production cross section")
        if ad is None or mt not in ad.photon_production:
            add("ANDP", f"MT={mt}", "no angular distribution")
        if ed is None or not ed.photon_production.get(mt):
            add("DLWP", f"MT={mt}", "no energy distribution")
    for i, pm in enumerate(rmt.particle_production or []):
        # one key per MTRH entry: an MT may repeat (kika.ace.classes.ixs_keys)
        for mt in ixs_keys(pm):
            where = f"particle {i + 1} MT={mt}"
            if mt not in ace.particle_production_xs.particle_cross_sections.get(i + 1, {}):
                add("SIGH", where, "no production cross section")
            if mt not in ad.particle_production[i]:
                add("ANDH", where, "no angular distribution")
            if not ed.particle_production[i].get(mt):
                add("DLWH", where, "no energy distribution")
    return out


def _ascending(x, strict=True) -> bool:
    d = np.diff(np.asarray(x, dtype=float))
    return bool(np.all(d > 0) if strict else np.all(d >= 0))


def physics_issues(ace) -> List[Issue]:
    out: List[Issue] = []
    add = lambda b, w, m: out.append(Issue(b, w, m))  # noqa: E731
    esz = ace.esz_block
    if not _ascending(esz.energies, strict=False):
        add("ESZ", "", "energy grid not ascending")
    for name in ("total_xs", "absorption_xs", "elastic_xs"):
        a = np.asarray(getattr(esz, name))
        if a.size and a.min() < 0:
            add("ESZ", name, f"negative values (min {a.min():.3g})")
    for mt, r in (ace.cross_section.reaction.items() if ace.cross_section else ()):
        a = np.asarray(r._xs_entries)
        if a.size and a.min() < 0:
            add("SIG", f"MT={mt}", f"negative cross section (min {a.min():.3g})")
    ad = ace.angular_distributions
    if ad is not None:
        dists = [("AND", "elastic", ad.elastic)] + [("AND", f"MT={m}", d) for m, d in ad.incident_neutron.items()]
        dists += [("ANDP", f"MT={m}", d) for m, d in ad.photon_production.items()]
        for i, p in enumerate(ad.particle_production or []):
            dists += [("ANDH", f"particle {i + 1} MT={m}", d) for m, d in p.items()]
        for block, where, d in dists:
            for k, form in enumerate(getattr(d, "energy_forms", []) or []):
                if form == "tabulated":
                    mu, cdf = np.asarray(d._cosine_grid[k]), np.asarray(d._cdf[k])
                    if mu[0] < -1 - 1e-9 or mu[-1] > 1 + 1e-9 or not _ascending(mu, strict=False):
                        add(block, f"{where} E#{k + 1}", "cosines not ascending in [-1, 1]")
                    if abs(cdf[0]) > 1e-6 or abs(cdf[-1] - 1) > 1e-6:
                        add(block, f"{where} E#{k + 1}", f"CDF from {cdf[0]:.6g} to {cdf[-1]:.6g}, not 0 to 1")
                elif form == "equiprobable":
                    mu = np.asarray(_bins(d, k))
                    if mu[0] < -1 - 1e-9 or mu[-1] > 1 + 1e-9 or not _ascending(mu, strict=False):
                        add(block, f"{where} E#{k + 1}", "equiprobable cosines not ascending in [-1, 1]")
    ed = ace.energy_distributions
    if ed is not None:
        chains = [("DLW", f"MT={m}", c) for m, c in ed.incident_neutron.items()]
        chains += [("DLWP", f"MT={m}", c) for m, c in ed.photon_production.items()]
        for i, p in enumerate(ed.particle_production or []):
            chains += [("DLWH", f"particle {i + 1} MT={m}", c) for m, c in p.items()]
        chains += [("DNED", f"group {g + 1}", c) for g, c in enumerate(ed.delayed_neutron)]
        for block, where, chain in chains:
            for law in chain:
                p = np.asarray(law.applicability_probabilities, dtype=float)
                if p.size and (p.min() < 0 or p.max() > 1 + 1e-9):
                    add(block, where, f"LAW={law.law} probability outside [0, 1]")
                for t in getattr(law, "distributions", None) or []:
                    if not isinstance(t, dict) or "cdf" not in t:
                        continue
                    cdf = np.asarray(t["cdf"], dtype=float)
                    if cdf.size and abs(cdf[-1] - 1) > 1e-6:
                        add(block, where, f"LAW={law.law} outgoing CDF ends at {cdf[-1]:.6g}")
                        break
    return out


def _bins(d, k):
    return d._cosine_bins[k] if hasattr(d, "_cosine_bins") and d._cosine_bins else d._cosine_grid[k]


def check(ace) -> List[Issue]:
    """Structure, then physics (physics only if the structure holds)."""
    s = structure_issues(ace)
    return s if s else physics_issues(ace)
