"""G4NDL ``ThermalScattering``: Geant4's thermal data, one folder per process.

``ThermalScattering/{Coherent,Incoherent,Inelastic}/{CrossSection,FS}/<name>.z``,
each a zlib stream of text: a two-line header (``G4NDL`` and the source tag)
and then one block per temperature, ``a b T n`` followed by the data:

- ``CrossSection`` (``3 MT``): *n* pairs (E eV, σ b), THERMR's MF3;
- ``Coherent/FS`` (``7 2``): the Bragg edges. The first block is *n* pairs
  (E_i, S_i); every later block repeats only the *n* S values, on the first
  block's edges (Geant4's ``readACoherentFSDATA`` reads it so);
- ``Incoherent/FS`` and ``Inelastic/FS`` (``6 MT``): *n* records, each an
  ENDF LIST head ``0 E 0 0 NW NL`` and NW values in rows of NL: E', its
  probability density, then NL − 2 equiprobable cosines. That is THERMR's
  MF6 (E' = E and density 1 for incoherent elastic).

A material is a file *name* across the six folders; ``h_water`` has only
``Inelastic``, ``al_metal`` ``Coherent`` and ``Inelastic``.
"""
from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .processed import (BraggTable, IncoherentElasticTable, OutgoingTable, ProcessedTemperature,
                        ProcessedTSL, merge_temperatures)

__all__ = ["G4NDL_PROCESSES", "g4ndl_thermal_root", "list_g4ndl_thermal", "read_g4ndl_thermal",
           "g4ndl_thermal_naming"]

G4NDL_PROCESSES = ("Coherent", "Incoherent", "Inelastic")


def g4ndl_thermal_root(path: Union[str, Path]) -> Optional[Path]:
    """The ``ThermalScattering`` folder at or under *path* (a G4NDL root), or None."""
    path = Path(path)
    # A G4NDL root has Inelastic/CrossSection too (the neutron data), so the
    # child is tried first and a folder counts only with Coherent or Incoherent.
    for candidate in (path / "ThermalScattering", path):
        if candidate.is_dir() and any((candidate / p / "CrossSection").is_dir()
                                      for p in ("Coherent", "Incoherent")):
            return candidate
    return None


def list_g4ndl_thermal(root: Union[str, Path]) -> Dict[str, Dict[str, Path]]:
    """``{name: {process: CrossSection file}}`` for every material in *root*."""
    root = Path(root)
    found: Dict[str, Dict[str, Path]] = {}
    for process in G4NDL_PROCESSES:
        folder = root / process / "CrossSection"
        if not folder.is_dir():
            continue
        for item in sorted(folder.glob("*.z")):
            found.setdefault(item.stem, {})[process] = item
    return found


def _tokens(path: Path) -> Tuple[str, List[str]]:
    text = zlib.decompress(path.read_bytes()).decode("ascii", errors="replace")
    lines = text.split("\n", 2)
    source = lines[1].strip() if len(lines) > 1 else ""
    return source, (lines[2] if len(lines) > 2 else "").split()


def _pairs_blocks(path: Path) -> Tuple[str, List[Tuple[float, np.ndarray, np.ndarray]]]:
    """``CrossSection``: per temperature (T, E, σ), cut where THERMR's data stops.

    THERMR writes the bound-atom reactions up to 20 MeV and zero above its
    E max (4 eV, say); those zeros are not data, so the table ends at the
    last non-zero value.
    """
    source, tok = _tokens(path)
    values = np.array(tok, dtype=float)
    out, i = [], 0
    while i + 4 <= values.size:
        temperature, n = float(values[i + 2]), int(values[i + 3])
        data = values[i + 4:i + 4 + 2 * n]
        energies, sigma = data[0::2].copy(), data[1::2].copy()
        nonzero = np.flatnonzero(sigma > 0)
        if nonzero.size:
            energies, sigma = energies[:nonzero[-1] + 1], sigma[:nonzero[-1] + 1]
        out.append((temperature, energies, sigma))
        i += 4 + 2 * n
    return source, out


def _bragg_blocks(path: Path) -> List[Tuple[float, BraggTable]]:
    _source, tok = _tokens(path)
    values = np.array(tok, dtype=float)
    out, i, edges = [], 0, None
    while i + 4 <= values.size:
        temperature, n = float(values[i + 2]), int(values[i + 3])
        i += 4
        if edges is None:
            data = values[i:i + 2 * n]
            edges, s = data[0::2].copy(), data[1::2].copy()
            i += 2 * n
        else:
            s = values[i:i + n].copy()
            i += n
        out.append((temperature, BraggTable(edges=edges, s=s)))
    return out


def _mf6_blocks(path: Path) -> List[Tuple[float, np.ndarray, List[np.ndarray]]]:
    """``FS`` (``6 MT``): per temperature (T, incident E, [rows of E', p, μ…])."""
    _source, tok = _tokens(path)
    values = np.array(tok, dtype=float)
    out, i = [], 0
    while i + 4 <= values.size:
        temperature, n = float(values[i + 2]), int(values[i + 3])
        i += 4
        incident, records = np.empty(n), []
        for k in range(n):
            incident[k] = values[i + 1]
            nw, nl = int(values[i + 4]), int(values[i + 5])
            i += 6
            records.append(values[i:i + nw].reshape(nw // nl, nl))
            i += nw
        out.append((temperature, incident, records))
    return out


class _LazyOutgoing:
    """Reads an inelastic ``FS`` file once, on first use, for all its temperatures."""

    def __init__(self, path: Path):
        self.path = path
        self._tables: Optional[Dict[float, OutgoingTable]] = None

    def table(self, temperature: float) -> Optional[OutgoingTable]:
        if self._tables is None:
            self._tables = {}
            for t, incident, records in _mf6_blocks(self.path):
                energies, pdf, cosines = [], [], []
                for rows in records:
                    keep = rows[:, 1] > 0  # the leading E' = 0 row has no weight
                    keep[0] = keep[0] or rows.shape[0] == 1
                    energies.append(rows[keep, 0])
                    pdf.append(rows[keep, 1])
                    cosines.append(rows[keep, 2:])
                cdf = []
                for e, p in zip(energies, pdf):
                    if e.size > 1:
                        c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(e))])
                        cdf.append(c / c[-1] if c[-1] > 0 else c)
                    else:
                        cdf.append(np.ones(e.size))
                self._tables[round(t, 3)] = OutgoingTable(kind="continuous", incident=incident,
                                                          energies=energies, cosines=cosines,
                                                          pdf=pdf, cdf=cdf)
        return self._tables.get(round(temperature, 3))


#: G4NDL's words for compounds, as chemical formulas where there is one.
_COMPOUNDS = {
    "water": "H2O", "heavy_water": "D2O", "polyethylene": "CH2", "ice": "H2O-ice",
    "metal": "metal", "benzen": "C6H6", "toluene": "C7H8",
    "ortho_h2": "ortho-H2", "para_h2": "para-H2", "ortho_d2": "ortho-D2", "para_d2": "para-D2",
    "l_ch4": "CH4-liquid", "s_ch4": "CH4-solid", "magnesium": "Mg",
}


def _formula(text: str) -> Optional[str]:
    """``sio2`` → ``SiO2``, ``zrh`` → ``ZrH``: a lower-case formula in chemical case, or None."""
    from kika._constants import ATOMIC_NUMBER_TO_SYMBOL

    symbols = {s.lower(): s for s in ATOMIC_NUMBER_TO_SYMBOL.values()}
    out, i = [], 0
    while i < len(text):
        if text[i].isdigit():
            out.append(text[i])
            i += 1
        elif text[i:i + 2] in symbols:
            out.append(symbols[text[i:i + 2]])
            i += 2
        elif text[i] in symbols:
            out.append(symbols[text[i]])
            i += 1
        else:
            return None
    return "".join(out)


def g4ndl_thermal_naming(name: str) -> Tuple[str, Optional[str], str, Optional[str]]:
    """``(name, principal, compound, variant)`` for a G4NDL thermal file stem.

    ``h_water`` is H in H2O, ``si28_sio2_alpha`` Si28 in SiO2 (alpha),
    ``graphite_10p`` graphite (10P), ``u235_uo2`` U235 in UO2.
    """
    if name.startswith("graphite"):
        variant = name[len("graphite_"):].upper() if name.startswith("graphite_") else None
        return name, "C", "Graphite", variant
    head, _, rest = name.partition("_")
    match = re.match(r"^([a-z]{1,2})(\d*)$", head)
    principal = (match.group(1).capitalize() + match.group(2)) if match else head
    variant = None
    for suffix in ("alpha", "beta", "singlecrystal", "phaseII"):
        if rest.endswith("_" + suffix):
            rest, variant = rest[: -len(suffix) - 1], suffix
            break
    compound = _COMPOUNDS.get(rest) or _formula(rest) or rest
    if compound == "metal":
        compound = f"{re.sub(r'[0-9]+$', '', principal)} metal"
    return name, principal, compound, variant


def read_g4ndl_thermal(root: Union[str, Path], name: str) -> ProcessedTSL:
    """Material *name* (``h_water``) of the ``ThermalScattering`` folder *root*."""
    root = g4ndl_thermal_root(root) or Path(root)
    files = list_g4ndl_thermal(root).get(name)
    if not files:
        raise FileNotFoundError(f"No thermal material {name!r} under {root}")
    blocks: Dict[float, ProcessedTemperature] = {}

    def block(t: float) -> ProcessedTemperature:
        return blocks.setdefault(round(t, 3), ProcessedTemperature(temperature=t))

    source = ""
    paths: List[str] = []
    if "Coherent" in files:
        fs = root / "Coherent" / "FS" / f"{name}.z"
        paths.append(str(fs))
        for t, table in _bragg_blocks(fs):
            block(t).coherent = table
    if "Incoherent" in files:
        source, xs_blocks = _pairs_blocks(files["Incoherent"])
        paths.append(str(files["Incoherent"]))
        fs = root / "Incoherent" / "FS" / f"{name}.z"
        cosines = {}
        if fs.is_file():
            for t, incident, records in _mf6_blocks(fs):
                cosines[round(t, 3)] = (incident, np.array([r[0, 2:] for r in records]))
        for t, e, s in xs_blocks:
            cos = cosines.get(round(t, 3))
            block(t).incoherent = IncoherentElasticTable(
                energies=e, xs=s, cosine_energies=cos[0] if cos else None,
                cosines=cos[1] if cos else None)
    if "Inelastic" in files:
        source, xs_blocks = _pairs_blocks(files["Inelastic"])
        paths.append(str(files["Inelastic"]))
        fs = root / "Inelastic" / "FS" / f"{name}.z"
        lazy = _LazyOutgoing(fs) if fs.is_file() else None
        for t, e, s in xs_blocks:
            b = block(t)
            b.inelastic_energies, b.inelastic_xs = e, s
            if lazy is not None:
                b.outgoing_kind = "continuous"
                b.outgoing_loader = (lambda lazy=lazy, t=t: lazy.table(t))
    stem, principal, compound, variant = g4ndl_thermal_naming(name)
    temperatures = merge_temperatures(blocks.values())
    emax = max((float(b.inelastic_energies[-1]) for b in temperatures
                if b.inelastic_energies is not None and b.inelastic_energies.size), default=None)
    if compound.endswith(" metal"):
        label = f"{principal} metal"
    elif principal and compound != "Graphite":
        label = f"{principal} in {compound}"
    else:
        label = name.replace("_", " ")
    return ProcessedTSL(format="g4ndl", name=label, awr=None, temperatures=temperatures,
                        nuclides=[principal] if principal else [], title=name, library=source or None,
                        emax=emax, paths=paths,
                        details={"name": name, "processes": sorted(files), "source": source})
