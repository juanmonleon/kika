"""What OpenMC's C++ assumes of a neutron file and never checks (roadmap H4, gate 1).

``openmc_hdf5_roadmap.md`` §1.2 lists it; each point is a check here, run on
the file as written:

- root ``version`` with major 3, one top-level group;
- ``kTs`` names are ``f"{round(kT/k_B)}K"`` of their own values, and every
  temperature has ``energy/<T>``, strictly increasing;
- every reaction has every temperature, with ``threshold_idx + len(xs) ==
  len(energy/<T>)``; no ``reaction_001``;
- ``redundant`` set on the sums and response functions (3, 4, 16, 103-107 when
  the levels are present, 18 when the chances are, 203-207, 301, 444, 901),
  and the partials it leaves count each part once;
- fission products in the C++'s order: a neutron first (prompt or total), the
  delayed groups right after it, each with a ``decay_rate``;
- every stored CDF starts at 0 (or at the lines' weight), ends at 1, never
  decreases; ``applicability`` only where ``n_distribution > 1`` matters;
- a ``urr`` group for every temperature or none, its ``inelastic`` the same
  everywhere and written as a reaction;
- no ``madland-nix`` (the C++ cannot sample it), and ``maxwell``/
  ``evaporation``/``watt`` parameters as ``Tabulated1D``.

:func:`checkNeutronFile` returns the list of violations; empty means the file
keeps the contract.
"""
from __future__ import annotations

from typing import List

import numpy as np

from . import records as R
from .names import FISSION_MTS, HDF5_VERSION, K_BOLTZMANN

__all__ = ["checkNeutronFile", "checkNeutronTree"]

REDUNDANT_ALWAYS = (1, 3, 27, 101, 203, 204, 205, 206, 207, 301, 444, 901)
SUM_LEVELS = {4: range(51, 92), 16: range(875, 892), 103: range(600, 650),
              104: range(650, 700), 105: range(700, 750), 106: range(750, 800),
              107: range(800, 850), 18: (19, 20, 21, 38)}


def checkNeutronFile(path) -> List[str]:
    with R.openFile(path) as tree:
        return checkNeutronTree(tree)


def _tables(node, where, out):
    """Every (3+, N) table with offsets: CDF row monotone, starting at 0, ending at 1."""
    for name, child in node.children.items():
        path = f"{where}/{name}"
        if isinstance(child, R.Group):
            _tables(child, path, out)
            continue
        offsets = R.attr(child, "offsets")
        if offsets is None or name in ("energy",):
            continue
        data = child.data
        if data.ndim != 2 or data.shape[0] < 3:
            continue
        ends = list(np.asarray(offsets)[1:]) + [data.shape[1]]
        lines = R.attr(child, "n_discrete_lines")
        for k, (a, b) in enumerate(zip(np.asarray(offsets), ends)):
            c = data[2, int(a):int(b)]
            if c.size == 0:
                out.append(f"{path}: empty table {k}")
                continue
            nd = int(np.asarray(lines)[k]) if lines is not None else 0
            if nd == 0 and abs(c[0]) > 1e-12:
                out.append(f"{path}: table {k} CDF starts at {c[0]}")
            if abs(c[-1] - 1.0) > 1e-6:
                out.append(f"{path}: table {k} CDF ends at {c[-1]}")
            if np.any(np.diff(c) < -1e-12):
                out.append(f"{path}: table {k} CDF decreases")
            if np.any(data[1, int(a):int(b)] < 0):
                out.append(f"{path}: table {k} negative density")


def checkNeutronTree(tree) -> List[str]:
    out: List[str] = []
    version = R.attr(tree, "version")
    if version is None or int(np.asarray(version)[0]) != HDF5_VERSION[0]:
        out.append(f"root version {version}")
    names = [n for n, _ in tree.groups()]
    if len(names) != 1:
        return out + [f"{len(names)} top-level groups"]
    name = names[0]
    g = tree[name]
    for attr in ("Z", "A", "metastable", "atomic_weight_ratio"):
        if R.attr(g, attr) is None:
            out.append(f"missing @{attr}")
    temps = []
    for T, d in g["kTs"].datasets():
        kT = float(d.data)
        if f"{int(round(kT / K_BOLTZMANN))}K" != T:
            out.append(f"kTs/{T} holds kT={kT}, which the C++ names {round(kT / K_BOLTZMANN)}K")
        temps.append(T)
        if T not in g["energy"].children:
            out.append(f"no energy/{T}")
            continue
        e = g[f"energy/{T}"].data
        if np.any(np.diff(e) <= 0):
            out.append(f"energy/{T} not strictly increasing")
    reactions = g["reactions"].children
    if "reaction_001" in reactions:
        out.append("reaction_001 present: the C++ would read it as the elastic")
    mts = {int(R.attr(r, "mt")): k for k, r in reactions.items()}
    redundant = {mt: int(R.attr(reactions[k], "redundant", 0)) for mt, k in mts.items()}
    for mt, key in mts.items():
        r = reactions[key]
        for T in temps:
            if T not in r.children:
                out.append(f"{key} has no {T}")
                continue
            xs = r[f"{T}/xs"]
            thr = int(R.attr(xs, "threshold_idx"))
            if thr + xs.shape[0] != g[f"energy/{T}"].shape[0]:
                out.append(f"{key}/{T}: threshold_idx {thr} + {xs.shape[0]} != grid")
        if mt in REDUNDANT_ALWAYS and not redundant[mt]:
            out.append(f"{key} must be redundant")
        if mt in SUM_LEVELS and any(c in mts and not redundant[c] for c in SUM_LEVELS[mt]) \
                and not redundant[mt]:
            out.append(f"{key} is a sum of partials that are written, and is not redundant")
        products = sorted((k for k in r.children if k.startswith("product_")),
                          key=lambda k: int(k.split("_")[1]))
        if mt in FISSION_MTS and products:
            first = r[products[0]]
            if R.text(R.attr(first, "particle")) != "neutron" or \
                    R.text(R.attr(first, "emission_mode")) not in ("prompt", "total"):
                out.append(f"{key}: product_0 is not the prompt/total neutron")
            modes = [R.text(R.attr(r[p], "emission_mode")) for p in products]
            delayed = [i for i, m in enumerate(modes) if m == "delayed"]
            if delayed and delayed != list(range(1, 1 + len(delayed))):
                out.append(f"{key}: delayed groups are not product_1..G")
            for i in delayed:
                if R.attr(r[products[i]], "decay_rate") is None:
                    out.append(f"{key}/{products[i]}: delayed without decay_rate")
            if len(products) > 10 and delayed and max(delayed) >= 10:
                out.append(f"{key}: a delayed group past product_9 is read out of order")
        for p in products:
            pr = r[p]
            for d, dist in pr.groups():
                if not d.startswith("distribution_"):
                    continue
                kind = R.text(R.attr(dist, "type"))
                if kind not in ("uncorrelated", "correlated", "kalbach-mann", "nbody"):
                    out.append(f"{key}/{p}/{d}: type {kind!r} the C++ skips")
                if kind == "uncorrelated" and "energy" in dist.children:
                    etype = R.text(R.attr(dist["energy"], "type"))
                    if etype == "madland-nix":
                        out.append(f"{key}/{p}/{d}: madland-nix, which the C++ cannot sample")
                    if etype in ("maxwell", "evaporation", "watt"):
                        for f in ("theta", "a", "b"):
                            if f in dist["energy"].children and \
                                    R.text(R.attr(dist["energy"][f], "type")) != "Tabulated1D":
                                out.append(f"{key}/{p}/{d}: {etype} {f} not Tabulated1D")
            _tables(pr, f"{key}/{p}", out)
    if "urr" in g.children:
        missing = [T for T in temps if T not in g["urr"].children]
        if missing:
            out.append(f"urr/ lacks {missing}")
        inelastic = {int(R.attr(g[f"urr/{T}"], "inelastic")) for T in temps if T in g["urr"].children}
        if len(inelastic) > 1:
            out.append(f"urr inelastic differs between temperatures: {inelastic}")
        for mt in inelastic:
            if mt > 0 and mt not in mts:
                out.append(f"urr names MT{mt} and it is not written")
    return out
