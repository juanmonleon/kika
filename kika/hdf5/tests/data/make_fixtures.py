"""Cut the committed HDF5 fixtures out of the official OpenMC libraries.

Run by hand, never by the tests:

    python kika/hdf5/tests/data/make_fixtures.py [ROOT]

``ROOT`` holds the official libraries (``endfb81/``, ``jeff40/``, ``jendl5/``);
by default ``C:\\Users\\Usuario\\BaradDur\\NuclearData\\openmc``.

**What a fixture is.** A real file with most of it removed and nothing
invented: a few temperatures, a few reactions, every N-th point of each energy
grid (threshold points always kept, so ``threshold_idx`` stays exact), and a
few incident energies of each distribution table, with offsets rebuilt. The
structure -- the groups, attributes, dtypes, layouts and the contract of
``openmc_hdf5_format_survey.md`` §2 -- is the official file's; the physics is
deliberately damaged (a thinned grid is not a converged grid). They exist to
test the format, so the size is kB and not MB.

Cases (roadmap H0):

- ``Fe56.h5``: levels, ``correlated`` in MT16/91, photons with a discrete line
  and the incident-energy code 22, redundant 103, 203, 301, 444, 901; two
  temperatures and the 0 K elastic.
- ``U235.h5``: fission with six delayed groups, ``total_nu``,
  ``fission_energy_release``, ``urr`` and the redundant MT4 it names.
- ``Cu63.h5``: ``kalbach-mann``, and a reaction with more than ten products
  (D10, the lexicographic order of ``product_10``).
- ``H2.h5``: ``nbody`` and a primary ``discrete_photon``.
- thermal ``c_Graphite`` (coherent elastic), ``c_H_in_ZrH`` (incoherent
  elastic), ``c_H_in_H2O`` (inelastic only), ``c_Zr_in_ZrH2`` (``mixed_elastic``
  with a ``Sum``), one temperature each.
- photon ``Fe``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from kika.hdf5 import records as R  # noqa: E402

DEFAULT_ROOT = Path(r"C:\Users\Usuario\BaradDur\NuclearData\openmc")
GRID_POINTS = 400        # roughly how many points each energy grid keeps
INCIDENT_POINTS = 4      # incident energies kept per distribution table
ROW_POINTS = 12          # outgoing points kept per incident energy
BRAGG_EDGES = 60         # Bragg edges kept by a coherent elastic


def _materialise(node):
    """Load every lazy dataset below ``node`` so it survives the file closing."""
    for _, child in R.walk(node):
        if isinstance(child, R.Dataset) and child._data is None:
            child.data = child.data
    return node


def _pick(n, keep):
    """Indices of ``n`` incident energies: first, last and evenly spaced in between."""
    if n <= keep:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, keep).round().astype(int))


def _blocks(offsets, total):
    ends = list(offsets[1:]) + [total]
    return [(int(a), int(b)) for a, b in zip(offsets, ends)]


def _subsetInterp(interp, chosen, n):
    """A (2, NR) breakpoint table re-expressed on the kept incident energies."""
    interp = np.asarray(interp)
    bps, codes = interp[0], interp[1]
    newBps = [int(np.searchsorted(chosen, bp - 1, side="right")) for bp in bps]
    newBps[-1] = len(chosen)
    keep = [i for i in range(len(newBps)) if i == 0 or newBps[i] > newBps[i - 1]]
    return np.array([[newBps[i] for i in keep], [codes[i] for i in keep]], dtype=interp.dtype)


def _cutAngle(group):
    energy = group["energy"].data
    mu = group["mu"]
    chosen = _pick(len(energy), INCIDENT_POINTS)
    offsets = R.attr(mu, "offsets")
    interp = R.attr(mu, "interpolation")
    blocks = _blocks(offsets, mu.data.shape[1])
    data = np.concatenate([mu.data[:, slice(*blocks[i])] for i in chosen], axis=1)
    newOffsets = np.cumsum([0] + [blocks[i][1] - blocks[i][0] for i in chosen[:-1]]).astype(offsets.dtype)
    group["energy"].data = energy[chosen]
    mu.data = data
    mu.attrs["offsets"] = R.Attribute.of(newOffsets, offsets.dtype)
    mu.attrs["interpolation"] = R.Attribute.of(np.asarray(interp)[chosen], np.asarray(interp).dtype)


def _cutTable(group, name, muName=None):
    """``continuous`` / ``kalbach-mann`` (``distribution``) or ``correlated`` (``energy_out`` + ``mu``)."""
    energy = group["energy"]
    table = group[name]
    chosen = _pick(len(energy.data), INCIDENT_POINTS)
    offsets = R.attr(table, "offsets")
    blocks = _blocks(offsets, table.data.shape[1])
    discrete = np.asarray(R.attr(table, "n_discrete_lines"))
    pieces, muPieces, muStart = [], [], 0
    for i in chosen:
        start, stop = blocks[i]
        nd = int(discrete[i])
        rows = list(range(start, start + nd)) + [start + nd + int(k) for k in
                                                 _pick(stop - start - nd, ROW_POINTS)]
        block = table.data[:, rows].copy()
        if muName is not None:
            mu = group[muName].data
            allMu = table.data[4].astype(int)
            for k, j in enumerate(rows):
                end = int(allMu[j + 1]) if j + 1 < table.data.shape[1] else mu.shape[1]
                segment = mu[:, int(allMu[j]):end]
                block[4, k] = muStart
                muStart += segment.shape[1]
                muPieces.append(segment)
        pieces.append(block)
    table.data = np.concatenate(pieces, axis=1)
    newOffsets = np.cumsum([0] + [b.shape[1] for b in pieces[:-1]]).astype(offsets.dtype)
    table.attrs["offsets"] = R.Attribute.of(newOffsets, offsets.dtype)
    for key in ("interpolation", "n_discrete_lines"):
        values = np.asarray(R.attr(table, key))
        table.attrs[key] = R.Attribute.of(values[chosen], values.dtype)
    if muName is not None:
        group[muName].data = np.concatenate(muPieces, axis=1)
    energy.attrs["interpolation"] = R.Attribute.of(
        _subsetInterp(R.attr(energy, "interpolation"), chosen, len(energy.data)),
        np.asarray(R.attr(energy, "interpolation")).dtype)
    energy.data = energy.data[chosen]


def _cutDistribution(dist):
    kind = R.text(R.attr(dist, "type"))
    if kind == "uncorrelated":
        if "angle" in dist:
            _cutAngle(dist["angle"])
        if "energy" in dist and R.text(R.attr(dist["energy"], "type")) == "continuous":
            _cutTable(dist["energy"], "distribution")
    elif kind == "kalbach-mann":
        _cutTable(dist, "distribution")
    elif kind in ("correlated", "incoherent_inelastic"):
        _cutTable(dist, "energy_out", "mu")


def _thinGrid(n, keep):
    stride = max(1, n // GRID_POINTS)
    return sorted(set(range(0, n, stride)) | {n - 1} | set(keep))


def cutNeutron(source, target, mts, temperatures, extra=()):
    with R.openFile(source) as root:
        name = next(iter(root.children))
        nuclide = root[name]
        out = R.Group(attrs=dict(root.attrs))
        g = out.require(name)
        g.attrs = dict(nuclide.attrs)
        reactions = nuclide["reactions"]
        kept = {k: v for k, v in reactions.children.items() if int(R.attr(v, "mt")) in mts}
        for T in temperatures + ["0K"]:
            if T == "0K" and "0K" not in nuclide["energy"].children:
                continue
            full = nuclide["energy"][T].data
            thresholds = [int(R.attr(r[T]["xs"], "threshold_idx")) for r in kept.values() if T in r.children]
            if T != "0K" and "urr" in nuclide.children:
                pass
            idx = np.array(_thinGrid(len(full), thresholds))
            g.require("energy")[T] = R.Dataset.of(full[idx])
            if T != "0K":
                g.require("kTs")[T] = R.Dataset.of(nuclide["kTs"][T].data)
            for rname, r in kept.items():
                if T not in r.children:
                    continue
                xs = r[T]["xs"]
                thr = int(R.attr(xs, "threshold_idx"))
                values = xs.data
                sel = idx[idx >= thr]
                newThr = int(np.searchsorted(idx, thr))
                dataset = R.Dataset.of(values[sel - thr])
                dataset.attrs["threshold_idx"] = R.Attribute.of(newThr, xs.attrs["threshold_idx"].dtype)
                g.require(f"reactions/{rname}")[T] = R.Group(children={"xs": dataset})
        for rname, r in kept.items():
            target_r = g.require(f"reactions/{rname}")
            target_r.attrs = dict(r.attrs)
            for cname, child in r.children.items():
                if cname.startswith("product_"):
                    product = _materialise(child)
                    for dname, dist in product.groups():
                        if dname.startswith("distribution_"):
                            _cutDistribution(dist)
                    target_r.children[cname] = product
        for key in extra:
            if key == "urr":
                urr = g.require("urr")
                for T in temperatures:
                    urr.children[T] = _materialise(nuclide["urr"][T])
            elif key in nuclide.children:
                g.children[key] = _materialise(nuclide[key])
        # Children in the order h5py lists them, as the official file has them.
        g.children = dict(sorted(g.children.items()))
        g["reactions"].children = dict(sorted(g["reactions"].children.items()))
        for _, r in g["reactions"].groups():
            r.children = dict(sorted(r.children.items()))
        R.writeFile(out, target)


def _thinBragg(block):
    """Every k-th Bragg edge of a coherent elastic, applied alike to its copies."""
    if "elastic" not in block.children:
        return
    elastic = block["elastic"]
    targets = []
    for path, node in R.walk(elastic):
        if isinstance(node, R.Dataset) and R.text(R.attr(node, "type", b"")) == "CoherentElastic":
            targets.append(node)
        if path.endswith("coherent_xs"):
            targets.append(node)
    for node in targets:
        edges = node.data
        node.data = edges[:, _pick(edges.shape[1], BRAGG_EDGES)]


def cutThermal(source, target, temperature=None):
    with R.openFile(source) as root:
        name = next(iter(root.children))
        material = root[name]
        temps = sorted(material["kTs"].children)
        T = temperature if temperature in temps else temps[0]
        out = R.Group(attrs=dict(root.attrs))
        g = out.require(name)
        g.attrs = dict(material.attrs)
        g.require("kTs")[T] = R.Dataset.of(material["kTs"][T].data)
        block = _materialise(material[T])
        _thinBragg(block)
        for kind in ("elastic", "inelastic"):
            if kind not in block.children:
                continue
            dist = block[kind]["distribution"]
            if R.text(R.attr(dist, "type")) == "incoherent_inelastic":
                _cutTable(dist, "energy_out", "mu")
                xs = block[kind]["xs"]
                energies = dist["energy"].data
                table = xs.data
                keep = np.isin(table[0], energies)
                if keep.sum() == len(energies):
                    xs.data = table[:, keep]
                    xs.attrs["breakpoints"] = R.Attribute.of(np.array([len(energies)]), np.asarray(R.attr(xs, "breakpoints")).dtype)
        g.children[T] = block
        R.writeFile(out, target)


def cutPhoton(source, target):
    with R.openFile(source) as root:
        name = next(iter(root.children))
        element = _materialise(root[name])
        energy = element["energy"].data
        subshells = element["subshells"]
        thresholds = [int(R.attr(s["xs"], "threshold_idx")) for _, s in subshells.groups()]
        idx = np.array(_thinGrid(len(energy), thresholds))
        element["energy"].data = energy[idx]
        for key in ("coherent", "incoherent", "pair_production_electron",
                    "pair_production_nuclear", "photoelectric", "heating"):
            if key in element.children:
                element[key]["xs"].data = element[key]["xs"].data[idx]
        for _, s in subshells.groups():
            xs = s["xs"]
            thr = int(R.attr(xs, "threshold_idx"))
            sel = idx[idx >= thr]
            xs.data = xs.data[sel - thr]
            xs.attrs["threshold_idx"] = R.Attribute.of(int(np.searchsorted(idx, thr)),
                                                       xs.attrs["threshold_idx"].dtype)
        out = R.Group(attrs=dict(root.attrs), children={name: element})
        R.writeFile(out, target)


def _klalbachMTs(path, count=2):
    import h5py
    found, many = [], None
    with h5py.File(path, "r") as f:
        reactions = f[list(f)[0]]["reactions"]
        for name, r in reactions.items():
            products = [k for k in r if k.startswith("product_")]
            types = {r[p][d].attrs["type"] for p in products for d in r[p] if d.startswith("distribution_")}
            if b"kalbach-mann" in types and len(found) < count:
                found.append(int(r.attrs["mt"]))
            if 10 < len(products) and (many is None or len(products) < many[1]):
                many = (int(r.attrs["mt"]), len(products))
    return found + ([many[0]] if many else [])


def main(root=DEFAULT_ROOT):
    root = Path(root)
    lib = root / "endfb81"
    neutron = lib / "neutron"
    cutNeutron(neutron / "Fe56.h5", HERE / "Fe56.h5",
               {2, 16, 51, 91, 102, 103, 600, 203, 301, 444, 901}, ["294K", "600K"])
    cutNeutron(neutron / "U235.h5", HERE / "U235.h5",
               {2, 4, 18, 102, 301, 444, 901}, ["294K"],
               extra=("urr", "total_nu", "fission_energy_release"))
    cutNeutron(neutron / "Cu63.h5", HERE / "Cu63.h5",
               {2} | set(_klalbachMTs(neutron / "Cu63.h5")), ["294K"])
    cutNeutron(neutron / "H2.h5", HERE / "H2.h5", {2, 16, 102}, ["294K"])
    thermal = lib / "thermal"
    for name in ("c_Graphite", "c_H_in_ZrH", "c_H_in_H2O", "c_Zr_in_ZrH2"):
        cutThermal(thermal / f"{name}.h5", HERE / f"{name}.h5", "294K")
    cutPhoton(lib / "photon" / "Fe.h5", HERE / "Fe.h5")
    for path in sorted(HERE.glob("*.h5")):
        print(f"{path.name:22s} {path.stat().st_size / 1024:8.1f} kB")


if __name__ == "__main__":
    main(*sys.argv[1:])
