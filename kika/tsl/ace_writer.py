"""ACE thermal tables (``.NNt``) from kika's processed thermal data, as ACER lays them out.

The inverse of :mod:`kika.tsl.ace`: one :class:`~kika.tsl.processed.ProcessedTemperature`
becomes NXS, JXS and XSS, in NJOY 2016 ACER's order (``acesix``/``thermal``),
and :func:`write_ace_thermal` writes the file with the neutron tables' writer
(legacy header, I20 for the integer words, E20.11 for the reals).

=========  =========================================================
ITIE       NE, E(NE) in MeV; ITIX = ITIE + 1 + NE: σ_inel(NE)
ITXE       continuous (IFENG = 2): NE locators, then NE counts; each
           incident energy's block starts one word after its locator
           (the format's historical offset), NIL + 2 words per outgoing
           energy: E', pdf (1/MeV), cdf, NIL - 1 equiprobable cosines.
           Discrete or skewed (IFENG = 0, 1): NE × NIEB rows of E' and
           NIL + 1 cosines.
ITCE       N, E(N) in MeV, then P(N): the cumulative S in MeV·b for a
           coherent elastic (IDPNC = 4), σ in b for an incoherent one
           (IDPNC = 3, cosines at ITCA, NCL + 1 per energy). A mixed
           elastic (IDPNC = 5) writes the coherent half here and the
           incoherent half at JXS(7..9) with NXS(8) = NCLI.
=========  =========================================================
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .processed import ProcessedTemperature, ProcessedTSL

__all__ = ["encode_ace_thermal", "write_ace_thermal"]

MEV = 1.0e6
K_MEV = 8.617333262e-11
#: NXS(2): ACER's inelastic mode in every modern table
IDPNI = 3


class _Words:
    """XSS being built, 1-based, with the integer mask beside it."""

    def __init__(self):
        self.values: List[np.ndarray] = []
        self.ints: List[np.ndarray] = []
        self.n = 0

    @property
    def next(self) -> int:
        return self.n + 1

    def add(self, values, integer=False) -> int:
        at = self.next
        v = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
        self.values.append(v)
        self.ints.append(np.full(v.size, bool(integer)))
        self.n += v.size
        return at

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        xss = np.concatenate([[0.0]] + self.values)
        return xss, np.concatenate(self.ints) if self.ints else np.zeros(0, dtype=bool)


def _outgoing(words: _Words, table, nil: Optional[int]):
    """ITXE: the outgoing tables. Returns ``(ITXE, NIL, IFENG, NIEB)``."""
    n = table.incident.size
    if table.kind == "continuous":
        ncos = {np.asarray(c).shape[1] for c in table.cosines}
        if len(ncos) != 1:
            raise ValueError("a continuous ACE table has one cosine count, NIL - 1")
        ncos = ncos.pop()
        counts = [len(e) for e in table.energies]
        itxe = words.next
        stride = ncos + 3
        start = itxe + 2 * n
        locators, at = [], start
        for m in counts:
            locators.append(at - 1)
            at += m * stride
        words.add(locators, integer=True)
        words.add(counts, integer=True)
        for e, p, c, mu in zip(table.energies, table.pdf, table.cdf, table.cosines):
            rows = np.column_stack([np.asarray(e) / MEV, np.asarray(p) * MEV, np.asarray(c), np.asarray(mu)])
            words.add(rows)
        return itxe, ncos + 1, 2, None
    m = {len(e) for e in table.energies}
    if len(m) != 1:
        raise ValueError("a discrete ACE table has NIEB outgoing energies at every incident energy")
    m = m.pop()
    ncos = {np.asarray(c).shape[1] for c in table.cosines}.pop()
    itxe = words.next
    for e, mu in zip(table.energies, table.cosines):
        words.add(np.column_stack([np.asarray(e) / MEV, np.asarray(mu)]))
    return itxe, ncos - 1, 1 if table.kind == "skewed" else 0, m


def _incoherent(words: _Words, table) -> Tuple[int, int, int, int]:
    """``(ITCE, ITCX, ITCA, NCL)`` of an incoherent elastic σ(E) with its equiprobable cosines."""
    if table.energies is None or table.xs is None or table.cosines is None:
        raise ValueError("an ACE incoherent elastic needs σ(E) on a grid and its equiprobable "
                         "cosines (the bound σ and W' alone are OpenMC's form)")
    e = np.asarray(table.energies, dtype=float)
    itce = words.add([e.size], integer=True)
    words.add(e / MEV)
    itcx = words.add(np.asarray(table.xs, dtype=float))
    cos = np.asarray(table.cosines, dtype=float)
    itca = words.add(cos)
    return itce, itcx, itca, cos.shape[1] - 1


def encode_ace_thermal(block: ProcessedTemperature, *, nieb: Optional[int] = None):
    """``(nxs, jxs, xss, is_int)`` of one temperature: NXS and JXS 1-based
    (``[0]`` unused), XSS with ``xss[0]`` the padding and ``is_int`` the mask of
    XSS(1..n). ``nieb`` is NXS(4) for a continuous table, which ACER fills
    though nothing reads it (the discrete NIEB is the table's)."""
    words = _Words()
    nxs, jxs = [0] * 17, [0] * 33
    if block.inelastic_energies is None or block.inelastic_xs is None:
        raise ValueError("an ACE thermal table starts with the incoherent inelastic σ(E)")
    e = np.asarray(block.inelastic_energies, dtype=float)
    jxs[1] = words.add([e.size], integer=True)
    words.add(e / MEV)
    jxs[2] = words.add(np.asarray(block.inelastic_xs, dtype=float))
    table = block.outgoing()
    if table is None:
        raise ValueError("an ACE thermal table needs the outgoing-neutron tables")
    jxs[3], nil, ifeng, table_nieb = _outgoing(words, table, None)
    nxs[2], nxs[3], nxs[7] = IDPNI, nil, ifeng
    nxs[4] = table_nieb if table_nieb is not None else int(nieb or 0)

    if block.coherent is not None:
        edges = np.asarray(block.coherent.edges, dtype=float)
        jxs[4] = words.add([edges.size], integer=True)
        words.add(edges / MEV)
        jxs[5] = words.add(np.asarray(block.coherent.s, dtype=float) / MEV)
        nxs[5], nxs[6] = 4, -1
        if block.incoherent is not None:
            jxs[7], jxs[8], jxs[9], nxs[8] = _incoherent(words, block.incoherent)
            nxs[5] = 5
    elif block.incoherent is not None:
        jxs[4], jxs[5], jxs[6], nxs[6] = _incoherent(words, block.incoherent)
        nxs[5] = 3
    xss, is_int = words.arrays()
    nxs[1] = xss.size - 1
    return nxs, jxs, xss, is_int


def write_ace_thermal(data: Union[ProcessedTSL, ProcessedTemperature], path, *, temperature=None,
                      zaid: str, awr: Optional[float] = None, kt: Optional[float] = None,
                      title: Optional[str] = None, mat: Optional[int] = None, date: Optional[str] = None,
                      izaw: Optional[Sequence[int]] = None, nieb: Optional[int] = None) -> Path:
    """Write one temperature of ``data`` as the ACE thermal table ``zaid`` (``"h-h2o.80t"``).

    ``data`` is a :class:`ProcessedTSL` (``temperature`` picks the block when
    it has several) or one :class:`ProcessedTemperature`. The header takes
    ``awr``, ``izaw`` (the ZA of each scattering nuclide), ``title``, ``mat``
    and ``date`` from the arguments, else from a ProcessedTSL read from ACE;
    ``kt`` (MeV) defaults to the block's temperature times Boltzmann's
    constant. The file is read back with :func:`kika.tsl.ace.read_ace_thermal`
    before it replaces ``path``.
    """
    import datetime
    import os

    from kika.ace.classes.header import Header
    from kika.ace.writers.write_ace import format_xss_lines
    from kika.ace.writers.write_header import write_header

    details = {}
    if isinstance(data, ProcessedTSL):
        block = data.at(temperature) if temperature is not None else (
            data.temperatures[0] if len(data.temperatures) == 1 else None)
        if block is None:
            raise ValueError(f"{len(data.temperatures)} temperatures; say which (temperature=)")
        details = data.details or {}
        awr = awr if awr is not None else data.awr
        title = title if title is not None else data.title
        if izaw is None and data.nuclides:
            izaw = [_za(n) for n in data.nuclides]
    else:
        block = data
    if awr is None or izaw is None:
        raise ValueError("an ACE thermal header needs the AWR and the scattering nuclides' ZA (izaw=)")
    nieb = nieb if nieb is not None else details.get("nieb")
    nxs, jxs, xss, is_int = encode_ace_thermal(block, nieb=nieb)

    stem, dot, extension = str(zaid).partition(".")
    h = Header()
    h.format_version = "legacy"
    h.zaid, h.extension = stem, (dot + extension) if dot else None
    h.atomic_weight_ratio = float(awr)
    h.temperature = float(kt) if kt is not None else block.temperature * K_MEV
    h.date = date or details.get("date") or datetime.date.today().strftime("%m/%d/%y")
    h.comment = (title or "")[:70]
    matid = mat if mat is not None else _mat(details.get("mat"))
    h.matid = int(matid) if matid is not None else None
    h.izaw_array = [(0, 0.0)] + [(int(z), 0.0) for z in izaw] + [(0, 0.0)] * (16 - len(izaw))
    h.nxs_array = nxs
    h.jxs_array = jxs

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", newline="\n", buffering=1024 * 1024) as f:
        f.write(write_header(h) + "\n")
        for text in format_xss_lines(xss[1:], is_int):
            f.write(text)
    try:
        from .ace import read_ace_thermal

        read_ace_thermal(temporary).temperatures[0].outgoing()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    return path


def _za(nuclide: str) -> int:
    """``"H1"`` → 1001, ``"C0"`` → 6000."""
    import re

    from kika._constants import ATOMIC_NUMBER_TO_SYMBOL

    m = re.match(r"^([A-Z][a-z]?)(\d+)$", nuclide)
    if not m:
        raise ValueError(f"{nuclide!r} is not a nuclide name like 'H1'")
    z = {v: k for k, v in ATOMIC_NUMBER_TO_SYMBOL.items()}[m.group(1)]
    return 1000 * z + int(m.group(2))


def _mat(text) -> Optional[int]:
    if text is None:
        return None
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    return int(digits) if digits else None


# ----------------------------------------------------------------------
# From kika's processed thermal suite (kika.hdf5.thermal_processing.processThermal)
# ----------------------------------------------------------------------

def incoherent_elastic_table(energies, bound_xs: float, debye_waller: float, *, bins: int = 16,
                             nmix: int = 1):
    """THERMR's incoherent elastic (``iel``) as ACER writes it: σ(E)/nmix and the
    mean cosine of each of ``bins`` equiprobable bins, on ``energies`` (eV).

    σ = σ_b/2 (1 − e^(−2c))/c and p(μ) ∝ e^(−c(1−μ)), with c = 2EW'.
    """
    from .processed import IncoherentElasticTable

    e = np.asarray(energies, dtype=float)
    c2 = 2.0 * e * debye_waller
    x1 = np.exp(-2.0 * c2)
    xs = 0.5 * bound_xs * (1.0 - x1) / c2 / nmix
    cos = np.empty((e.size, bins))
    for i, c in enumerate(c2):
        u, r1x1 = -1.0, 1.0 / (1.0 - x1[i])
        for k in range(bins):
            x2 = np.exp(-c * (1.0 - u))
            unow = 1.0 + np.log((1.0 - x1[i]) / bins + x2) / c
            cos[i, k] = bins / c * (np.exp(-c * (1.0 - unow)) * (c * unow - 1.0) - x2 * (c * u - 1.0)) * r1x1
            u = unow
    return IncoherentElasticTable(energies=e, xs=xs, cosine_energies=e, cosines=cos)


def processed_temperature(suite, temperature: float, *, nmix: int = 1, bins: int = 16) -> ProcessedTemperature:
    """One temperature of a processed thermal suite as ACE holds it.

    The inelastic σ and continuous outgoing tables are the suite's (THERMR +
    ACER ``iwt = 2`` by kika); the leading E' = 0 line OpenMC's form carries
    when the first CDF is not 0 is not ACE's and is dropped. The coherent
    elastic is the suite's ACER Bragg edges; the incoherent elastic, which the
    suite states as σ_b and W', is THERMR's ``iel`` on the inelastic incident
    grid with ``bins`` equiprobable cosines.
    """
    from kika.nuclear_data import model as m

    from .processed import BraggTable, OutgoingTable

    K_EV = 8.617333262e-5
    label = None
    for style in suite.styles:
        if isinstance(style, m.Heated) and style.temperature is not None:
            kT = float(style.temperature.value)
            if abs(kT / K_EV - float(temperature)) <= 1e-3 * max(1.0, float(temperature)):
                label = style.label
    if label is None:
        raise ValueError(f"the suite has no heated style at {temperature} K")
    reactions = {r.label if hasattr(r, "label") else str(r.id.label): r for r in suite.reactions}
    inelastic = next(r for r in suite.reactions if int(r.ENDF_MT) == 4)
    elastic = next((r for r in suite.reactions if int(r.ENDF_MT) == 2), None)
    del reactions

    block = ProcessedTemperature(temperature=float(temperature), outgoing_kind="continuous")
    xs = inelastic.crossSection[label]
    block.inelastic_energies = np.asarray(xs.xs, dtype=float)
    block.inelastic_xs = np.asarray(xs.ys, dtype=float)
    form = inelastic.outputChannel.products[0].distribution[label]
    energies, pdf, cdf, cosines = [], [], [], []
    for spectrum, angular in zip(form.energy.function1ds, form.energyAngular.function2ds):
        x, p, c = (np.asarray(v, dtype=float) for v in (spectrum.xs, spectrum.pdf, spectrum.cdf))
        mu = np.array([np.asarray(t.xs, dtype=float) for t in angular.function1ds])
        if x.size > 1 and x[0] == 0.0 and c[0] == 0.0 and p[0] == 0.0:
            x, p, c, mu = x[1:], p[1:], c[1:], mu[1:]
        energies.append(x)
        pdf.append(p)
        cdf.append(c)
        cosines.append(mu)
    table = OutgoingTable(kind="continuous", incident=block.inelastic_energies, energies=energies,
                          cosines=cosines, pdf=pdf, cdf=cdf)
    block._outgoing = table

    if elastic is not None:
        exs = elastic.crossSection[label]
        parts = exs.functions if isinstance(exs, m.SumOfFunctions) else [exs]
        for part in parts:
            if isinstance(part, m.CoherentElasticXS):
                block.coherent = BraggTable(edges=np.asarray(part.braggEdges, dtype=float),
                                            s=np.asarray(part.structureFactors, dtype=float))
            elif isinstance(part, m.IncoherentElasticXS):
                block.incoherent = incoherent_elastic_table(
                    block.inelastic_energies, float(part.boundCrossSection), float(part.debyeWaller),
                    bins=bins, nmix=nmix)
            else:
                raise ValueError(f"an elastic {type(part).__name__} has no ACE form")
    return block


def write_thermal_suite(suite, path, temperature=None, *, zaid: Optional[str] = None,
                        extension: Optional[str] = None, bins: int = 16) -> Path:
    """``kika.write(thermal suite, path, format="ace")``: one temperature as a ``.NNt`` table.

    The suite is :func:`kika.hdf5.thermal_processing.processThermal`'s (or one
    read from an OpenMC thermal file with the same forms). ``zaid`` names the
    table (``"hh2o.02t"``); by default the registry's short name with the
    temperature's suffix (``ace_temperature_suffix``), or ``extension``.
    """
    from kika.hdf5.names import K_BOLTZMANN
    from kika.hdf5.thermal_names import THERMAL_DATA

    temps = []
    from kika.nuclear_data import model as m
    for style in suite.styles:
        if isinstance(style, m.Heated) and style.temperature is not None:
            temps.append(float(style.temperature.value) / K_BOLTZMANN)
    if temperature is None:
        if len(temps) != 1:
            raise ValueError(f"the suite has {len(temps)} temperatures {temps}; an ACE table holds one")
        temperature = temps[0]
    entry = THERMAL_DATA.get(suite.target)
    nmix = entry.nmix if entry is not None else 1
    block = processed_temperature(suite, float(temperature), nmix=nmix, bins=bins)
    attributes = getattr(suite.provenance, "attributes", None) or {}
    nuclides = attributes.get("nuclides") or []
    izaw = list(entry.zaids) if entry is not None else [_za(n) for n in nuclides]
    if zaid is None:
        if entry is None:
            raise ValueError(f"{suite.target!r} is not in OpenMC's thermal registry; pass zaid=")
        if extension is None:
            from kika._utils import ace_temperature_suffix
            extension = ace_temperature_suffix(float(temperature)) + "t"
        zaid = f"{entry.name}{extension if extension.startswith('.') else '.' + extension}"
    title = f"{suite.target} {suite.evaluation or ''}".strip()
    return write_ace_thermal(block, path, zaid=zaid, awr=attributes.get("atomic_weight_ratio"),
                             kt=float(temperature) * K_MEV, title=title, izaw=izaw, nieb=64)
