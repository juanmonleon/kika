"""OpenMC HDF5 thermal files (``thermal/c_H_in_H2O.h5``), format version 3.

One file is one material at every temperature it was processed at. Under the
material's group: ``kTs/<T>K`` (kT in eV), and per temperature ``elastic`` and
``inelastic``, each a ``xs`` dataset and a ``distribution`` group:

- ``elastic/xs`` is ``CoherentElastic`` (Bragg edges and cumulative S, eV·b),
  ``IncoherentElastic`` (the bound σ and W', the evaluation's own numbers),
  ``Tabulated1D`` (THERMR's σ(E)) or a ``Sum`` of the first two (mixed);
- ``inelastic/xs`` is ``Tabulated1D``, and its distribution is either
  ``incoherent_inelastic`` (continuous: incident ``energy``, then ``energy_out``
  rows E', p, c, μ-interpolation, μ-offset, and ``mu`` rows μ, p, c) or
  ``incoherent_inelastic_discrete`` (``energy_out`` and ``mu_out`` arrays).

Reading needs ``h5py``, which kika does not require: it is imported here,
on first use, and its absence is a clear error.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .processed import (BraggTable, IncoherentElasticTable, OutgoingTable, ProcessedTemperature,
                        ProcessedTSL, merge_temperatures)

__all__ = ["read_openmc_thermal", "is_openmc_thermal", "HDF5_SIGNATURE"]

#: HDF5's eight-byte signature.
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

#: Boltzmann constant in eV/K.
K_EV = 8.617333262e-5


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Reading OpenMC HDF5 thermal files needs h5py (pip install h5py)") from exc
    return h5py


def is_openmc_thermal(path: Union[str, Path]) -> bool:
    """True when *path* is an HDF5 file whose ``filetype`` is ``data_thermal``."""
    try:
        with open(path, "rb") as handle:
            if handle.read(8) != HDF5_SIGNATURE:
                return False
        h5py = _h5py()
        with h5py.File(path, "r") as f:
            kind = f.attrs.get("filetype")
            return (kind.decode() if isinstance(kind, bytes) else str(kind)) == "data_thermal"
    except (OSError, ImportError):
        return False


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _elastic(group, block: ProcessedTemperature) -> None:
    xs = group["xs"]
    kind = _text(xs.attrs.get("type", b""))
    parts = []
    if kind == "Sum":
        parts = [xs[name] for name in sorted(k for k in xs if k.startswith("func_"))]
    else:
        parts = [xs]
    for part in parts:
        part_kind = _text(part.attrs.get("type", b""))
        values = part[()]
        if part_kind == "CoherentElastic":
            block.coherent = BraggTable(edges=np.asarray(values[0], float),
                                        s=np.asarray(values[1], float))
        elif part_kind == "IncoherentElastic":
            block.incoherent = IncoherentElasticTable(bound_xs=float(values[0]),
                                                      debye_waller=float(values[1]))
        elif part_kind == "Tabulated1D":
            block.incoherent = IncoherentElasticTable(energies=np.asarray(values[0], float),
                                                      xs=np.asarray(values[1], float))
    distribution = group.get("distribution")
    if distribution is not None and block.incoherent is not None:
        node = distribution.get("incoherent", distribution)
        if "mu_out" in node:
            mu = np.asarray(node["mu_out"][()], float)
            block.incoherent.cosines = mu
            if block.incoherent.energies is not None and len(block.incoherent.energies) == mu.shape[0]:
                block.incoherent.cosine_energies = block.incoherent.energies


def _outgoing_loader(path: str, material: str, temperature_key: str):
    def load() -> Optional[OutgoingTable]:
        h5py = _h5py()
        with h5py.File(path, "r") as f:
            node = f[material][temperature_key]["inelastic"].get("distribution")
            if node is None:
                return None
            kind = _text(node.attrs.get("type", b""))
            if kind == "incoherent_inelastic_discrete":
                energy_out = np.asarray(node["energy_out"][()], float)
                mu_out = np.asarray(node["mu_out"][()], float)
                skewed = bool(node.attrs.get("skewed", False))
                incident = np.asarray(f[material][temperature_key]["inelastic"]["xs"][0], float)
                n, m = energy_out.shape
                weights = None
                if skewed and m >= 4:
                    w = np.ones(m)
                    w[[0, 1, -2, -1]] = [0.1, 0.4, 0.4, 0.1]
                    weights = [w] * n
                return OutgoingTable(kind="skewed" if skewed else "discrete",
                                     incident=incident[:n], energies=list(energy_out),
                                     cosines=list(mu_out), weights=weights)
            incident = np.asarray(node["energy"][()], float)
            eout = node["energy_out"]
            offsets = np.asarray(eout.attrs["offsets"], int)
            table = np.asarray(eout[()], float)
            mu = np.asarray(node["mu"][()], float)
            ends = np.concatenate([offsets[1:], [table.shape[1]]])
            # Row 4 is each outgoing energy's offset into ``mu``; its cosines
            # run to the next one's.
            mu_starts = table[4].astype(int)
            mu_ends = np.concatenate([mu_starts[1:], [mu.shape[1]]])
            energies, pdf, cdf, cosines = [], [], [], []
            for start, end in zip(offsets, ends):
                rows = table[:, start:end]
                energies.append(rows[0])
                pdf.append(rows[1])
                cdf.append(rows[2])
                cosines.append([mu[0, a:b] for a, b in zip(mu_starts[start:end], mu_ends[start:end])])
            return OutgoingTable(kind="continuous", incident=incident, energies=energies,
                                 cosines=cosines, pdf=pdf, cdf=cdf)
    return load


def read_openmc_thermal(path: Union[str, Path]) -> ProcessedTSL:
    """The material in an OpenMC thermal HDF5 file, at every temperature it holds."""
    h5py = _h5py()
    path = str(path)
    with h5py.File(path, "r") as f:
        material = next(iter(f.keys()))
        group = f[material]
        nuclides = [_text(n) for n in group.attrs.get("nuclides", [])]
        awr = float(group.attrs.get("atomic_weight_ratio", 0.0)) or None
        emax = float(group.attrs.get("energy_max", 0.0)) or None
        blocks: List[ProcessedTemperature] = []
        kts = group["kTs"]
        for key in kts:
            temperature = round(float(kts[key][()]) / K_EV, 1)
            node = group[key]
            block = ProcessedTemperature(temperature=temperature)
            if "elastic" in node:
                _elastic(node["elastic"], block)
            if "inelastic" in node:
                xs = np.asarray(node["inelastic"]["xs"][()], float)
                block.inelastic_energies, block.inelastic_xs = xs[0], xs[1]
                dist = node["inelastic"].get("distribution")
                if dist is not None:
                    kind = _text(dist.attrs.get("type", b""))
                    block.outgoing_kind = ("continuous" if kind == "incoherent_inelastic"
                                           else "skewed" if dist.attrs.get("skewed", False)
                                           else "discrete")
                    block.outgoing_loader = _outgoing_loader(path, material, key)
            blocks.append(block)
    name = material[2:] if material.startswith("c_") else material
    return ProcessedTSL(format="openmc", name=name.replace("_", " "), awr=awr,
                        temperatures=merge_temperatures(blocks),
                        nuclides=[n for n in nuclides if not n.endswith("0") or len(nuclides) == 1],
                        title=material, emax=emax, paths=[path],
                        details={"group": material, "nuclides_listed": nuclides})
