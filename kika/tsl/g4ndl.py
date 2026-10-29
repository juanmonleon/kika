"""G4NDL ``ThermalScattering``: Geant4's thermal data, one folder per process.

``ThermalScattering/{Coherent,Incoherent,Inelastic}/{CrossSection,FS}/<name>.z``,
each a zlib stream of text: an optional header (the token ``G4NDL`` and one
source token, which Geant4's ``G4ParticleHPManager::GetDataStream`` skips) and
then one block per temperature, ``MF MT T n`` followed by the data:

- ``CrossSection`` (``3 MT``): *n* pairs (E eV, σ b), THERMR's MF3, read by
  ``G4ParticleHPThermalScatteringData::readData`` into a lin-lin vector. Every
  distributed table ends in zeros up to 20 MeV (``4 σ``, ``4.000001 0``,
  ``2e7 0``): Geant4 holds the last value past the end of a vector, so those
  zeros are what turns the process off above THERMR's E max;
- ``Coherent/FS`` (``7 2``): the Bragg edges. The first block is *n* pairs
  (E_i, S_i); every later block repeats only the *n* S values, on the first
  block's edges (Geant4's ``readACoherentFSDATA`` reads it so);
- ``Incoherent/FS`` and ``Inelastic/FS`` (``6 MT``): *n* records, each an
  ENDF LIST head ``0 E 0 0 NW NL`` and NW values in rows of NL: E', its
  probability density, then NL − 2 equiprobable cosines. That is THERMR's
  MF6 (E' = E and density 1 for incoherent elastic). Geant4 builds the
  outgoing-energy cdf from the densities by the trapezoid rule over **every**
  row, the zero-density ones included (``readAnE_P_E_isoAng``).

A material is a file *name* across the six folders; ``h_water`` has only
``Inelastic``, ``al_metal`` ``Coherent`` and ``Inelastic``.

**Reading** (:func:`read_g4ndl_thermal`) keeps everything the consumer reads:
the tables are cut at their last non-zero σ (the zeros past it are not data),
and the cut energies go to ``details["tails"]``; the coherent σ table rides on
the :class:`~kika.tsl.processed.BraggTable`; MF/MT go to ``details["mf_mt"]``.

**Writing** (:func:`write_g4ndl_thermal`) is the way back, so a material read
here is written to files Geant4 reads as the same numbers. Floats are written
with ``repr`` (the shortest string that reads back to the same double, never a
rounding; the distributed files print 7 digits), integers as integers (Geant4
reads MF, MT, *n*, NW and NL with ``>> G4int``). Every file is read back
before the call returns and must give back the arrays that went in.
"""
from __future__ import annotations

import math
import re
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from kika.g4ndl.exceptions import G4NDLUnsupportedError

from .processed import (BraggTable, IncoherentElasticTable, OutgoingTable, ProcessedTemperature,
                        ProcessedTSL, merge_temperatures)

__all__ = ["G4NDL_PROCESSES", "g4ndl_thermal_root", "list_g4ndl_thermal", "read_g4ndl_thermal",
           "g4ndl_thermal_naming", "write_g4ndl_thermal"]

G4NDL_PROCESSES = ("Coherent", "Incoherent", "Inelastic")
HEADER_TAG = "G4NDL"


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


def _tokens(path: Path) -> Tuple[Optional[str], List[str]]:
    """``(source, data tokens)``; the header is optional, as in ``GetDataStream``."""
    tok = zlib.decompress(path.read_bytes()).decode("ascii", errors="replace").split()
    if tok and tok[0] == HEADER_TAG:
        return (tok[1] if len(tok) > 1 else ""), tok[2:]
    return None, tok


def _head(values: np.ndarray, i: int, path: Path) -> Tuple[int, int, float, int]:
    if i + 4 > values.size:
        raise ValueError(f"{path}: a block head is cut short at token {i}")
    return int(values[i]), int(values[i + 1]), float(values[i + 2]), int(values[i + 3])


def _check_end(path: Path, i: int, size: int):
    if i != size:
        raise ValueError(f"{path}: {size - i:+d} tokens left over after the last block")


def _pairs_blocks(path: Path) -> Tuple[Optional[str], Tuple[int, int],
                                       List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]]:
    """``CrossSection``: ``(source, (MF, MT), [(T, E, σ, tail)])``.

    THERMR writes the bound-atom reactions up to 20 MeV and zero above its
    E max (4 eV, say); those zeros are not data, so the table ends at the
    last non-zero value and *tail* holds the energies cut off (their σ is 0).
    """
    source, tok = _tokens(path)
    values = np.array(tok, dtype=float)
    out, i, mfmt = [], 0, (3, 0)
    while i < values.size:
        mf, mt, temperature, n = _head(values, i, path)
        if not out:
            mfmt = (mf, mt)
        data = values[i + 4:i + 4 + 2 * n]
        if data.size != 2 * n:
            raise ValueError(f"{path}: {temperature} K declares {n} pairs and has {data.size / 2}")
        energies, sigma = data[0::2].copy(), data[1::2].copy()
        tail = np.empty(0)
        nonzero = np.flatnonzero(sigma > 0)
        if nonzero.size:
            cut = nonzero[-1] + 1
            energies, sigma, tail = energies[:cut], sigma[:cut], energies[cut:]
        out.append((temperature, energies, sigma, tail))
        i += 4 + 2 * n
    _check_end(path, i, values.size)
    return source, mfmt, out


def _bragg_blocks(path: Path) -> Tuple[Tuple[int, int], List[Tuple[float, BraggTable]]]:
    """``Coherent/FS`` (``7 2``): ``((MF, MT), [(T, Bragg table)])``."""
    _source, tok = _tokens(path)
    values = np.array(tok, dtype=float)
    out, i, edges, mfmt = [], 0, None, (7, 2)
    while i < values.size:
        mf, mt, temperature, n = _head(values, i, path)
        if not out:
            mfmt = (mf, mt)
        i += 4
        if edges is None:
            data = values[i:i + 2 * n]
            edges, s = data[0::2].copy(), data[1::2].copy()
            i += 2 * n
        else:
            if n != edges.size:
                raise ValueError(f"{path}: {temperature} K has {n} S values on {edges.size} edges")
            s = values[i:i + n].copy()
            i += n
        out.append((temperature, BraggTable(edges=edges, s=s)))
    _check_end(path, i, values.size)
    return mfmt, out


def _mf6_blocks(path: Path) -> Tuple[Tuple[int, int],
                                     List[Tuple[float, np.ndarray, List[np.ndarray]]]]:
    """``FS`` (``6 MT``): ``((MF, MT), [(T, incident E, [rows of E', p, μ…])])``."""
    _source, tok = _tokens(path)
    values = np.array(tok, dtype=float)
    out, i, mfmt = [], 0, (6, 0)
    while i < values.size:
        mf, mt, temperature, n = _head(values, i, path)
        if not out:
            mfmt = (mf, mt)
        i += 4
        incident, records = np.empty(n), []
        for k in range(n):
            incident[k] = values[i + 1]
            nw, nl = int(values[i + 4]), int(values[i + 5])
            i += 6
            if nl < 2 or nw % nl:
                raise ValueError(f"{path}: {temperature} K record {k} has NW={nw}, NL={nl}")
            records.append(values[i:i + nw].reshape(nw // nl, nl))
            i += nw
        out.append((temperature, incident, records))
    _check_end(path, i, values.size)
    return mfmt, out


def _trapezoid_cdf(energies: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    """Geant4's ``secondary_energy_cdf``: trapezoids over every row, normalised."""
    if energies.size < 2:
        return np.ones(energies.size)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(energies))])
    return c / c[-1] if c[-1] > 0 else c


class _LazyOutgoing:
    """Reads an inelastic ``FS`` file once, on first use, for all its temperatures.

    Every row is kept, the zero-density ones too: Geant4's cdf is the trapezoid
    rule over all of them, so dropping the leading E' = 0 row (or an interior
    zero) would move probability between outgoing-energy bins.
    """

    def __init__(self, path: Path):
        self.path = path
        self._tables: Optional[Dict[float, OutgoingTable]] = None

    def table(self, temperature: float) -> Optional[OutgoingTable]:
        if self._tables is None:
            self._tables = {}
            for t, incident, records in _mf6_blocks(self.path)[1]:
                energies = [rows[:, 0].copy() for rows in records]
                pdf = [rows[:, 1].copy() for rows in records]
                cosines = [rows[:, 2:].copy() for rows in records]
                cdf = [_trapezoid_cdf(e, p) for e, p in zip(energies, pdf)]
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

    source: Optional[str] = None
    paths: List[str] = []
    mf_mt: Dict[str, List[int]] = {}
    tails: Dict[str, Dict[float, List[float]]] = {}

    def cross_sections(process: str):
        nonlocal source
        file_source, mfmt, xs_blocks = _pairs_blocks(files[process])
        source = source if source is not None else file_source
        paths.append(str(files[process]))
        mf_mt[f"{process}/CrossSection"] = list(mfmt)
        tails[process] = {round(t, 3): tail.tolist() for t, _e, _s, tail in xs_blocks}
        return xs_blocks

    if "Coherent" in files:
        fs = root / "Coherent" / "FS" / f"{name}.z"
        paths.append(str(fs))
        tables = {round(t, 3): (e, s) for t, e, s, _tail in cross_sections("Coherent")}
        mfmt, bragg = _bragg_blocks(fs)
        mf_mt["Coherent/FS"] = list(mfmt)
        for t, table in bragg:
            table.energies, table.xs = tables.get(round(t, 3), (None, None))
            block(t).coherent = table
    if "Incoherent" in files:
        xs_blocks = cross_sections("Incoherent")
        fs = root / "Incoherent" / "FS" / f"{name}.z"
        cosines = {}
        if fs.is_file():
            mfmt, fs_blocks = _mf6_blocks(fs)
            mf_mt["Incoherent/FS"] = list(mfmt)
            for t, incident, records in fs_blocks:
                cosines[round(t, 3)] = (incident, np.array([r[0, 2:] for r in records]))
        for t, e, s, _tail in xs_blocks:
            cos = cosines.get(round(t, 3))
            block(t).incoherent = IncoherentElasticTable(
                energies=e, xs=s, cosine_energies=cos[0] if cos else None,
                cosines=cos[1] if cos else None)
    if "Inelastic" in files:
        xs_blocks = cross_sections("Inelastic")
        fs = root / "Inelastic" / "FS" / f"{name}.z"
        lazy = None
        if fs.is_file():
            lazy = _LazyOutgoing(fs)
            mf_mt["Inelastic/FS"] = list(_fs_mfmt(fs))
        for t, e, s, _tail in xs_blocks:
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
                        details={"name": name, "processes": sorted(files), "source": source or "",
                                 "mf_mt": mf_mt, "tails": tails})


def _fs_mfmt(path: Path) -> Tuple[int, int]:
    """MF and MT of an ``FS`` file, without reading its records."""
    _source, tok = _tokens(path)
    return (int(tok[0]), int(tok[1])) if len(tok) >= 2 else (6, 0)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

#: Where every distributed table ends: σ = 0 from just above E max to here.
TAIL_END = 2.0e7
#: The relative step from E max to the first zero, and from a Bragg edge back
#: to the end of the plateau below it. Under 1e-6, so Geant4's guard against
#: doubled-up energies (``G4ParticleHPVector::GetXsec``) holds the lower value
#: across the step instead of interpolating a ramp.
STEP = 1.0e-7
#: Lin-lin interpolation of S/E between two points a ratio r apart is off by
#: at most about (r − 1)²/4, at their geometric mean; 1.02 keeps a built σ_coh
#: within 1e-4 of S/E (measured on the 26 coherent materials of G4NDL 4.7.1:
#: 9.8e-5, where THERMR's own 7-digit tables are off by up to 1.9e-2).
BRAGG_RATIO = 1.02
#: THERMR's MT when the material does not carry its own (Geant4 discards it).
DEFAULT_MT = {"Coherent": 230, "Incoherent": 230, "Inelastic": 229}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


def _unsupported(message: str):
    raise G4NDLUnsupportedError(message)


def _float(x: float) -> str:
    x = float(x)
    if not math.isfinite(x):
        _unsupported(f"{x!r} cannot be written for Geant4")
    return repr(x)


def _head_text(*items) -> str:
    return "\n".join(f"{(str(int(v)) if isinstance(v, (int, np.integer)) else _float(v)):>13}"
                     for v in items) + "\n"


def _rows(values: Sequence[float], per_line: int) -> str:
    text = [_float(v) for v in values]
    return "".join(" ".join(text[k:k + per_line]) + "\n" for k in range(0, len(text), per_line))


def _check_table(what: str, energies, values, positive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    e = np.asarray(energies, dtype=float)
    y = np.asarray(values, dtype=float)
    if e.ndim != 1 or e.shape != y.shape or e.size == 0:
        _unsupported(f"{what}: {e.shape} energies against {y.shape} values")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(y))):
        _unsupported(f"{what}: a non-finite number")
    if np.any(np.diff(e) < 0) or (positive and e[0] <= 0):
        _unsupported(f"{what}: the energies are not positive and non-decreasing")
    if np.any(y < 0):
        _unsupported(f"{what}: a negative value")
    return e, y


def _tail(e: np.ndarray, carried: Optional[Sequence[float]]) -> np.ndarray:
    """The zero points after a table: the carried ones, else E max (1 + STEP) and 20 MeV."""
    if carried is not None:
        tail = np.asarray(carried, dtype=float)
        if tail.size and tail[0] < e[-1]:
            _unsupported(f"the carried zero tail starts at {tail[0]} eV, below E max {e[-1]} eV")
        return tail
    first = e[-1] * (1.0 + STEP)
    return np.array([first, TAIL_END]) if first < TAIL_END else np.empty(0)


def bragg_cross_section_table(table: BraggTable, emax: float, emin: float = 1.0e-5
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """σ_coh(E) = S_i/E tabulated for a lin-lin consumer, from *emin* to *emax*.

    Each edge gets two points, the end of the plateau below it (E_i (1 − STEP),
    S_{i−1}/E) and the edge itself (S_i/E_i), so Geant4 jumps there instead of
    ramping; between edges the 1/E decay is sampled at ratio BRAGG_RATIO.
    """
    edges = np.asarray(table.edges, dtype=float)
    s = np.asarray(table.s, dtype=float)
    keep = edges <= emax
    edges, s = edges[keep], s[keep]
    knots = [(min(emin, edges[0] if edges.size else emin), 0.0)]
    previous = 0.0
    for k, edge in enumerate(edges):
        stop = edges[k + 1] if k + 1 < edges.size else emax
        below = edge * (1.0 - STEP)
        if below > knots[-1][0]:
            knots.append((below, previous / below))
        if edge > knots[-1][0]:
            knots.append((edge, s[k] / edge))
        span = stop * (1.0 - STEP) if k + 1 < edges.size else stop
        n = int(math.ceil(math.log(span / edge) / math.log(BRAGG_RATIO))) if span > edge else 0
        for x in edge * (span / edge) ** (np.arange(1, n + 1) / n):
            if x > knots[-1][0]:
                knots.append((float(x), s[k] / x))
        previous = s[k]
    if knots[-1][0] < emax:
        knots.append((emax, previous / emax))
    e, y = np.array(knots).T
    return e, y


def union_bragg_s(edges: np.ndarray, s: np.ndarray, union: np.ndarray) -> np.ndarray:
    """S on the *union* of every temperature's edges: the value held from the own edge below.

    Geant4 reads the edges once, from the first temperature, and every later
    block repeats only S. NJOY's ACER thins the edges per temperature (Be in
    JEFF-3.3: 110 at 293.6 K, 63 at 1200 K), so the blocks are put on the
    union. That is exact: S is a step function, an added edge repeats the S
    below it, and Geant4 samples the first edge whose S reaches its random
    fraction of S_j (``ApplyYourself``), which an edge adding nothing never is.
    """
    index = np.searchsorted(edges, union, side="right") - 1
    return np.where(index >= 0, s[np.clip(index, 0, None)], 0.0)


def _cs_file(source: str, mf: int, mt: int,
             blocks: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]) -> str:
    out = [f"{HEADER_TAG:>13}\n{source}\n"]
    for t, e, s, tail in blocks:
        energies = np.concatenate([e, tail])
        values = np.concatenate([s, np.zeros(tail.size)])
        pairs = np.empty(2 * energies.size)
        pairs[0::2], pairs[1::2] = energies, values
        out.append(_head_text(mf, mt, float(t), int(energies.size)))
        out.append(_rows(pairs, 6))
    return "".join(out)


def _bragg_file(source: str, mf: int, mt: int, blocks: List[Tuple[float, BraggTable]]) -> str:
    out = [f"{HEADER_TAG:>13}\n{source}\n"]
    for k, (t, table) in enumerate(blocks):
        out.append(_head_text(mf, mt, float(t), int(table.edges.size)))
        if k == 0:
            pairs = np.empty(2 * table.edges.size)
            pairs[0::2], pairs[1::2] = table.edges, table.s
            out.append(_rows(pairs, 6))
        else:
            out.append(_rows(table.s, 6))
    return "".join(out)


def _mf6_file(source: str, mf: int, mt: int,
              blocks: List[Tuple[float, np.ndarray, List[np.ndarray]]]) -> str:
    out = [f"{HEADER_TAG:>13}\n{source}\n"]
    for t, incident, records in blocks:
        out.append(_head_text(mf, mt, float(t), int(len(records))))
        for energy, rows in zip(incident, records):
            out.append(_head_text_line(energy, rows.size, rows.shape[1]))
            out.append("".join(" ".join(_float(v) for v in row) + "\n" for row in rows))
    return "".join(out)


def _head_text_line(energy: float, nw: int, nl: int) -> str:
    return " ".join(f"{v:>12}" for v in ("0", _float(energy), "0", "0", str(int(nw)),
                                         str(int(nl)))) + "\n"


def _incoherent_records(what: str, table: IncoherentElasticTable
                        ) -> Tuple[np.ndarray, List[np.ndarray]]:
    if table.cosine_energies is None or table.cosines is None:
        _unsupported(f"{what}: no equiprobable cosines (an exact bound σ and Debye-Waller form "
                     "has to be tabulated first); Geant4 samples the angle from them")
    incident = np.asarray(table.cosine_energies, dtype=float)
    cosines = np.atleast_2d(np.asarray(table.cosines, dtype=float))
    if cosines.shape[0] != incident.size or not np.all(np.abs(cosines) <= 1.0):
        _unsupported(f"{what}: {cosines.shape} cosines on {incident.size} energies, or |μ| > 1")
    records = [np.concatenate([[e, 1.0], mu])[None, :] for e, mu in zip(incident, cosines)]
    return incident, records


def _inelastic_records(what: str, table: Optional[OutgoingTable]
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
    if table is None:
        _unsupported(f"{what}: σ_inel has no outgoing-neutron table; Geant4 samples from one")
    if table.kind != "continuous" or table.pdf is None:
        _unsupported(f"{what}: a {table.kind!r} outgoing table; G4NDL holds only continuous "
                     "(E', pdf, equiprobable cosines) rows")
    records = []
    for k, (e, p, mu) in enumerate(zip(table.energies, table.pdf, table.cosines)):
        e = np.asarray(e, dtype=float)
        p = np.asarray(p, dtype=float)
        mu = np.asarray(mu, dtype=float).reshape(e.size, -1)
        if p.shape != e.shape or np.any(p < 0) or np.any(np.diff(e) < 0) \
                or not np.all(np.abs(mu) <= 1.0):
            _unsupported(f"{what}: incident energy {k} has an unusable (E', pdf, μ) table")
        records.append(np.column_stack([e, p, mu]))
    return np.asarray(table.incident, dtype=float), records


def write_g4ndl_thermal(material, root: Union[str, Path], name: Optional[str] = None, *,
                        source: Optional[str] = None,
                        mf_mt: Optional[Dict[str, Sequence[int]]] = None,
                        processes: Optional[Sequence[str]] = None,
                        overwrite: bool = False) -> List[Path]:
    """Write one thermal material as G4NDL ``ThermalScattering`` files; returns the paths.

    *material* is a :class:`~kika.tsl.processed.ProcessedTSL` or the
    :class:`~kika.tsl.read.TSLMaterial` :func:`kika.tsl.read_tsl` returns, from
    any processed format (G4NDL, ACE, OpenMC). *root* is the G4NDL root or its
    ``ThermalScattering`` folder; *name* the file stem (``h_water``), by default
    the one a G4NDL material was read under. A process is written when the
    material has it at some temperature, as its ``CrossSection`` and ``FS``
    pair: coherent from the Bragg tables, incoherent elastic from σ(E) and the
    equiprobable cosines, inelastic from σ(E) and a continuous outgoing table.

    What a material read from G4NDL carries comes back as it was: the coherent
    σ table, the zero tails, MF/MT and the source token. Otherwise σ_coh is
    built from the edges (:func:`bragg_cross_section_table`), each table ends
    in zeros from E max (1 + 1e-7) to 20 MeV (without them Geant4 would hold
    the last σ above E max), MT takes :data:`DEFAULT_MT` and the source
    ``kika``. Refused, with :class:`~kika.g4ndl.exceptions.G4NDLUnsupportedError`:
    what Geant4 would read as something else (discrete or skewed outgoing
    tables, incoherent elastic without cosines, non-finite or negative numbers,
    a source with whitespace). Bragg edges that change with temperature (ACER
    thins them) are put on their union, exactly (:func:`union_bragg_s`).

    *processes* limits the writing to some of ``Coherent``, ``Incoherent`` and
    ``Inelastic`` (each one asked for must be in the material); the files of the
    others are left as they are, which is how one process of a library's
    material is replaced.

    Geant4 opens a material by the name its ``G4ParticleHPThermalScatteringNames``
    maps a G4 material to; a new *name* needs ``AddThermalElement`` there.
    Existing files of *name* are refused unless *overwrite*, which also removes
    the files of processes written for and absent from the material (Geant4
    would read them).
    """
    tsl = getattr(material, "processed", material)
    if not isinstance(tsl, ProcessedTSL):
        raise TypeError(f"expected a ProcessedTSL or a processed TSLMaterial, got {type(material)}")
    carried = tsl.details if tsl.format == "g4ndl" else {}
    name = name or carried.get("name")
    if not name or not _NAME.match(str(name)):
        raise ValueError(f"name={name!r}: give the G4NDL file stem, e.g. 'h_water'")
    source = source if source is not None else (carried.get("source") or "kika")
    if not source or re.search(r"\s", source):
        _unsupported(f"source={source!r}: Geant4 reads the source as one token")
    mf_mt = {**(carried.get("mf_mt") or {}), **(mf_mt or {})}
    tails = carried.get("tails") or {}
    wanted = tuple(processes) if processes else G4NDL_PROCESSES
    unknown = sorted(set(wanted) - set(G4NDL_PROCESSES))
    if unknown:
        raise ValueError(f"processes={unknown}: G4NDL has {list(G4NDL_PROCESSES)}")
    root = Path(root)
    folder = root if root.name == "ThermalScattering" else root / "ThermalScattering"

    def mfmt(key: str, mf: int, mt: int) -> Tuple[int, int]:
        got = mf_mt.get(key)
        return (int(got[0]), int(got[1])) if got else (mf, mt)

    def tail_of(process: str, t: float):
        by_t = tails.get(process)
        return None if by_t is None else by_t.get(round(t, 3))

    temps = sorted(tsl.temperatures, key=lambda b: b.temperature)
    texts: Dict[Tuple[str, str], str] = {}
    expected: Dict[str, list] = {}

    coherent = [b for b in temps if b.coherent is not None] if "Coherent" in wanted else []
    if coherent:
        own = [_check_table(f"Bragg edges at {b.temperature} K", b.coherent.edges, b.coherent.s)
               for b in coherent]
        union = np.unique(np.concatenate([edges for edges, _s in own]))
        cs_blocks, fs_blocks = [], []
        for b, (edges, s) in zip(coherent, own):
            t, table = b.temperature, b.coherent
            if not np.array_equal(edges, union):
                s = union_bragg_s(edges, s, union)
                edges = union
            if table.energies is not None and table.xs is not None:
                e, y = _check_table(f"σ_coh at {t} K", table.energies, table.xs)
            else:
                emax = tsl.emax or float(edges[-1])
                e, y = bragg_cross_section_table(BraggTable(edges=edges, s=s), emax)
            cs_blocks.append((t, e, y, _tail(e, tail_of("Coherent", t))))
            fs_blocks.append((t, BraggTable(edges=edges, s=s)))
        mf, mt = mfmt("Coherent/CrossSection", 3, DEFAULT_MT["Coherent"])
        texts[("Coherent", "CrossSection")] = _cs_file(source, mf, mt, cs_blocks)
        mf, mt = mfmt("Coherent/FS", 7, 2)
        texts[("Coherent", "FS")] = _bragg_file(source, mf, mt, fs_blocks)
        expected["Coherent"] = [(cs, fs) for cs, fs in zip(cs_blocks, fs_blocks)]

    incoherent = [b for b in temps if b.incoherent is not None] if "Incoherent" in wanted else []
    if incoherent:
        cs_blocks, fs_blocks = [], []
        for b in incoherent:
            t, table = b.temperature, b.incoherent
            if table.energies is None or table.xs is None:
                _unsupported(f"incoherent elastic at {t} K is the exact bound σ and Debye-Waller "
                             "form; G4NDL holds σ(E) tabulated")
            e, y = _check_table(f"σ_inc at {t} K", table.energies, table.xs)
            cs_blocks.append((t, e, y, _tail(e, tail_of("Incoherent", t))))
            incident, records = _incoherent_records(f"incoherent elastic at {t} K", table)
            fs_blocks.append((t, incident, records))
        mf, mt = mfmt("Incoherent/CrossSection", 3, DEFAULT_MT["Incoherent"])
        texts[("Incoherent", "CrossSection")] = _cs_file(source, mf, mt, cs_blocks)
        mf, mt = mfmt("Incoherent/FS", 6, mt)
        texts[("Incoherent", "FS")] = _mf6_file(source, mf, mt, fs_blocks)
        expected["Incoherent"] = list(zip(cs_blocks, fs_blocks))

    inelastic = [b for b in temps if b.inelastic_xs is not None] if "Inelastic" in wanted else []
    if inelastic:
        cs_blocks, fs_blocks = [], []
        for b in inelastic:
            t = b.temperature
            e, y = _check_table(f"σ_inel at {t} K", b.inelastic_energies, b.inelastic_xs)
            cs_blocks.append((t, e, y, _tail(e, tail_of("Inelastic", t))))
            incident, records = _inelastic_records(f"inelastic at {t} K", b.outgoing())
            fs_blocks.append((t, incident, records))
        mf, mt = mfmt("Inelastic/CrossSection", 3, DEFAULT_MT["Inelastic"])
        texts[("Inelastic", "CrossSection")] = _cs_file(source, mf, mt, cs_blocks)
        mf, mt = mfmt("Inelastic/FS", 6, mt)
        texts[("Inelastic", "FS")] = _mf6_file(source, mf, mt, fs_blocks)
        expected["Inelastic"] = list(zip(cs_blocks, fs_blocks))

    if not texts:
        _unsupported(f"{tsl.name}: none of {list(wanted)} to write")
    missing = sorted(set(processes or ()) - {key[0] for key in texts})
    if missing:
        _unsupported(f"{tsl.name}: processes={missing} asked for and not in the material")

    targets = {key: folder / key[0] / key[1] / f"{name}.z"
               for key in [(p, k) for p in wanted for k in ("CrossSection", "FS")]}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{name!r} is already in {folder} ({len(existing)} files); "
                              "pass overwrite=True to replace it")
    for key, path in targets.items():
        if key not in texts and path.exists():
            path.unlink()
    written = []
    for key, text in texts.items():
        path = targets[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(zlib.compress(text.encode("ascii")))
        written.append(path)
    _verify(folder, name, expected)
    return written


def _verify(folder: Path, name: str, expected: Dict[str, list]):
    """Read the files back with the reader; every array must be the one that went in."""
    def same(a, b, what):
        if not np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float)):
            raise ValueError(f"{folder}/{name}: {what} does not read back as written")

    for process, items in expected.items():
        _source, _mfmt, cs_blocks = _pairs_blocks(folder / process / "CrossSection" / f"{name}.z")
        if len(cs_blocks) != len(items):
            raise ValueError(f"{folder}/{name}: {process} reads back {len(cs_blocks)} temperatures")
        fs = folder / process / "FS" / f"{name}.z"
        fs_back = _bragg_blocks(fs)[1] if process == "Coherent" else _mf6_blocks(fs)[1]
        for (cs_in, fs_in), cs_out, fs_out in zip(items, cs_blocks, fs_back):
            t = cs_in[0]
            same([cs_out[0]], [t], f"{process} temperature")
            # Reading cuts at the last non-zero σ, so compare the full tables.
            same(np.concatenate([cs_out[1], cs_out[3]]), np.concatenate([cs_in[1], cs_in[3]]),
                 f"{process} σ energies at {t} K")
            same(np.concatenate([cs_out[2], np.zeros(cs_out[3].size)]),
                 np.concatenate([cs_in[2], np.zeros(cs_in[3].size)]), f"{process} σ at {t} K")
            same([fs_out[0]], [fs_in[0]], f"{process} FS temperature")
            if process == "Coherent":
                same(fs_out[1].edges, fs_in[1].edges, f"Bragg edges at {t} K")
                same(fs_out[1].s, fs_in[1].s, f"Bragg S at {t} K")
            else:
                same(fs_out[1], fs_in[1], f"{process} incident energies at {t} K")
                if len(fs_out[2]) != len(fs_in[2]):
                    raise ValueError(f"{folder}/{name}: {process} FS records at {t} K")
                for k, (a, b) in enumerate(zip(fs_out[2], fs_in[2])):
                    if a.shape != b.shape:
                        raise ValueError(f"{folder}/{name}: {process} FS record {k} at {t} K "
                                         f"reads back {a.shape}, wrote {b.shape}")
                    same(a, b, f"{process} FS record {k} at {t} K")
