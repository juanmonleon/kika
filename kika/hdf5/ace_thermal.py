"""A thermal-scattering ACE table (``.NNt``) → the processed thermal forms.

kika's ACE reader covers continuous-energy neutron and photo-atomic tables; the
thermal tables THERMR + ACER write (ITIE/ITCE/ITXE/ITCA blocks) are read here,
the way ``openmc.data.ThermalScattering.from_ace`` reads them
(``thermal.py:751-965``), into the forms
:mod:`kika.hdf5.thermal_decode` produces from an OpenMC file -- so the writer
handles both alike:

- incoherent inelastic: the cross section (ITIE) and, with ``nxs[7] == 2``
  (ACER ``iwt=2``, what OpenMC uses), the continuous outgoing-energy tables with
  equiprobable cosines (ITXE) as an :class:`EnergyAngularMC`, the cosines
  sorted; a table whose cumulative does not start at 0 gets a 0 eV point with
  isotropic cosines, as OpenMC adds one; otherwise the discrete form;
- elastic (ITCE/ITCA): coherent (Bragg edges and cumulative factors, MeV·b →
  eV·b), incoherent (cross section and equiprobable cosines), or both summed.

The raw table is ``(header, IZ/AW pairs, NXS, JXS, XSS)`` of a type-1 ACE file;
``XSS`` is kept 1-based as the JXS locators are.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .names import EV_PER_MEV

__all__ = ["AceTable", "readAceTable", "thermalForms"]


@dataclass
class AceTable:
    name: str
    awr: float
    kT: float             # MeV, as the header states it
    pairs: List[Tuple[int, float]]
    nxs: np.ndarray       # 1-based: nxs[1] is NXS(1)
    jxs: np.ndarray       # 1-based
    xss: np.ndarray       # 1-based


def readAceTable(path) -> AceTable:
    """The first table of a type-1 ACE file."""
    lines = Path(path).read_text().splitlines()
    head = lines[0].split()
    name, awr, kT = head[0], float(head[1]), float(head[2])
    pairs = []
    for line in lines[2:6]:
        for k in range(4):
            field = line[18 * k:18 * k + 18]
            if field.strip():
                za, aw = int(field[:7]), float(field[7:18])
                if za:
                    pairs.append((za, aw))
    ints = " ".join(lines[6:12]).split()
    nxs = np.concatenate([[0], np.array(ints[:16], dtype=int)])
    jxs = np.concatenate([[0], np.array(ints[16:48], dtype=int)])
    length = nxs[1]
    values: List[float] = []
    for line in lines[12:]:
        values.extend(float(v) for v in line.split())
        if len(values) >= length:
            break
    xss = np.concatenate([[0.0], np.array(values[:length], dtype=float)])
    return AceTable(name, awr, kT, pairs, nxs, jxs, xss)


def _m():
    from kika.nuclear_data import model

    return model


def _discrete(values, outer):
    m = _m()
    n = len(values)
    p = np.full(n, 1.0 / n)
    return m.Xs_pdf_cdf1d(xs=np.sort(values), pdf=p, cdf=np.cumsum(p),
                          interpolation=m.Interpolation.flat, numberOfDiscrete=n,
                          outerDomainValue=float(outer))


def thermalForms(ace: AceTable):
    """``(inelastic xs, inelastic distribution, elastic xs or None, elastic distribution or None, energy_max)``."""
    m = _m()
    xss, nxs, jxs = ace.xss, ace.nxs, ace.jxs
    idx = jxs[1]
    n = int(xss[idx])
    energy = xss[idx + 1: idx + 1 + n] * EV_PER_MEV
    inelasticXs = m.XYs1d(xs=energy, ys=xss[idx + 1 + n: idx + 1 + 2 * n])
    energyMax = float(energy[-1])

    if nxs[7] == 2:
        nMu = nxs[3] - 1
        idx = jxs[3]
        locc = xss[idx: idx + n].astype(int)
        nOut = xss[idx + n: idx + 2 * n].astype(int)
        spectra, angular = [], []
        for i in range(n):
            j0 = locc[i]
            stride = nMu + 3
            e = xss[j0 + 1: j0 + 1 + nOut[i] * stride: stride] * EV_PER_MEV
            p = xss[j0 + 2: j0 + 2 + nOut[i] * stride: stride] / EV_PER_MEV
            c = xss[j0 + 3: j0 + 3 + nOut[i] * stride: stride]
            tables = []
            for k in range(nOut[i]):
                start = j0 + 4 + k * stride
                tables.append(_discrete(xss[start: start + nMu], e[k]))
            if c[0] > 0.0:
                e, p, c = np.insert(e, 0, 0.0), np.insert(p, 0, 0.0), np.insert(c, 0, 0.0)
                dmu = 2.0 / nMu
                tables.insert(0, _discrete(np.linspace(-1 + dmu / 2, 1 - dmu / 2, nMu), 0.0))
            spectra.append(m.Xs_pdf_cdf1d(xs=e, pdf=p, cdf=c, interpolation=m.Interpolation.linlin,
                                          outerDomainValue=float(energy[i])))
            angular.append(m.XYs2d(function1ds=tables, outerDomainValue=float(energy[i])))
        inelastic = m.EnergyAngularMC(energy=m.XYs2d(function1ds=spectra),
                                      energyAngular=m.XYs3d(function2ds=angular))
    else:
        nMu, nOut = nxs[3], nxs[4]
        idx = jxs[3]
        block = xss[idx: idx + n * nOut * (nMu + 2)].reshape(n, nOut, nMu + 2)
        inelastic = m.IncoherentInelasticDiscreteMC(energyOut=block[:, :, 0] * EV_PER_MEV,
                                                    muOut=block[:, :, 1:], skewed=nxs[7] == 1)

    elasticXs = elasticDist = None
    idx = jxs[4]
    if idx:
        coherent = incoherent = None
        if nxs[5] in (4, 5):
            ne = int(xss[idx])
            edges = xss[idx + 1: idx + 1 + ne] * EV_PER_MEV
            factors = xss[idx + 1 + ne: idx + 1 + 2 * ne] * EV_PER_MEV
            coherent = (m.CoherentElasticXS(braggEdges=edges, structureFactors=factors),
                        m.CoherentElasticMC())
        if nxs[5] in (3, 5):
            mixed = nxs[5] == 5
            j = jxs[7] if mixed else jxs[4]
            ne = int(xss[j])
            e = xss[j + 1: j + 1 + ne] * EV_PER_MEV
            values = xss[j + 1 + ne: j + 1 + 2 * ne]
            nMu = (nxs[8] if mixed else nxs[6]) + 1
            k = jxs[9] if mixed else jxs[6]
            mu = xss[k: k + ne * nMu].reshape(ne, nMu)
            incoherent = (m.XYs1d(xs=e, ys=values), m.IncoherentElasticDiscreteMC(muOut=mu))
        if coherent and incoherent:
            elasticXs = m.SumOfFunctions(functions=[coherent[0], incoherent[0]])
            elasticDist = m.MixedElasticMC(coherent=coherent[1], incoherent=incoherent[1])
        else:
            elasticXs, elasticDist = coherent or incoherent
    return inelasticXs, inelastic, elasticXs, elasticDist, energyMax


def nuclidesFromPairs(pairs) -> List[str]:
    """The ``nuclides`` a table applies to, from its ZAIDs, as OpenMC lists them.

    H and Fe by isotope; any other element as its natural element (``C0``)
    followed by its naturally occurring isotopes (``thermal.py:945-960``).
    """
    from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, NATURAL_ABUNDANCE

    out: List[str] = []
    for zaid, _ in pairs:
        if zaid <= 0:
            continue
        Z, A = divmod(int(zaid), 1000)
        element = ATOMIC_NUMBER_TO_SYMBOL[Z]
        if element in ("H", "Fe"):
            if f"{element}{A}" not in out:
                out.append(f"{element}{A}")
            continue
        if f"{element}0" not in out:
            out.append(f"{element}0")
        for isotope in _naturalIsotopes(element, NATURAL_ABUNDANCE):
            if isotope not in out:
                out.append(isotope)
    return out


def _naturalIsotopes(element, table):
    """``["C12", "C13"]``: the naturally occurring isotopes, by mass number."""
    from kika._constants import SYMBOL_TO_ATOMIC_NUMBER

    Z = SYMBOL_TO_ATOMIC_NUMBER[element]
    return [f"{element}{za % 1000}" for za in sorted(table.get(1000 * Z, {}))]
