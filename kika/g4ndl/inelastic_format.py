"""Inelastic records → G4NDL text, and the exact comparison the writer checks against.

The mirror of :mod:`kika.g4ndl.inelastic_parse`: :func:`formatInelasticFS`
emits the tokens of an :class:`~kika.g4ndl.inelastic_records.InelasticFSRecord`
in the order the consumer reads them, and :func:`formatGammas` those of a
:class:`~kika.g4ndl.inelastic_records.GammasRecord`. The numbers follow
:mod:`kika.g4ndl.encode`: floats as ``repr`` (the shortest string that reads
back to the same double, never a rounding), integers as integers, since
Geant4 reads every count, flag and code with ``>> G4int``. Line breaks mean
nothing to the consumer and are chosen for reading.

:func:`inelasticDifferences` walks two records field by field — dataclasses,
tuples, arrays element for element, floats by ``==`` — and lists every place
they differ; ``path`` is ignored. It is what the writer's read-back check and
the fixed-point tests use.
"""
from __future__ import annotations

import dataclasses
from typing import List

import numpy as np

from kika.g4ndl.inelastic_records import (
    DT_ANGULAR, DT_CROSS_SECTION, DT_ENERGY, DT_ENERGY_ANGLE, DT_PHOTON_ANGULAR,
    DT_PHOTON_ENERGY, DT_PHOTON_MULTIPLICITY, DT_PHOTON_PARTIALS,
    AngularBody, ContinuumBody, CrossSectionBody, DiscreteTwoBodyBody, EnergyAngleBody,
    EnergyBody, GammasRecord, InelasticFSRecord, LabAngleEnergyBody, NBodyBody,
    PhotonAngularBody, PhotonCascadeBody, PhotonEnergyBody, PhotonMultiplicityBody,
    PhotonPartialsBody, Tab1,
)
from kika.g4ndl.records import AngularBlock, Interpolation, LegendreRecord

__all__ = ["formatInelasticFS", "formatGammas", "formatSectionBody", "inelasticDifferences"]

_PER_LINE = 6


def _f(x) -> str:
    return repr(float(x))


def _i(x) -> str:
    return str(int(x))


def _values(values) -> List[str]:
    flat = [repr(v) for v in np.asarray(values, dtype=np.float64).ravel().tolist()]
    return [" ".join(flat[k:k + _PER_LINE]) for k in range(0, len(flat), _PER_LINE)]


def _interleaved(x, y) -> List[str]:
    flat = np.empty(2 * len(x), dtype=np.float64)
    flat[0::2] = x
    flat[1::2] = y
    return _values(flat)


def _interp(interp: Interpolation) -> List[str]:
    return [_i(interp.nRegions),
            " ".join(f"{int(b)} {int(c)}" for b, c in zip(interp.nbt, interp.codes))]


def _tab1(t: Tab1) -> List[str]:
    return [_i(len(t))] + _interp(t.interpolation) + _interleaved(t.x, t.y)


# ------------------------------------------------------------------ bodies

def _angularBlock(block: AngularBlock) -> List[str]:
    out = [_i(len(block.records))] + _interp(block.interpolation)
    for r in block.records:
        if isinstance(r, LegendreRecord):
            out.append(f"{_f(r.temperature)} {_f(r.energy)} {_i(r.tempdep)} {len(r.coefficients)}")
            out += _values(r.coefficients)
        else:
            out.append(f"{_f(r.temperature)} {_f(r.energy)} {_i(r.tempdep)} {len(r.mu)}")
            out += _interp(r.interpolation) + _interleaved(r.mu, r.probability)
    return out


def _angular(b: AngularBody) -> List[str]:
    out = [f"{_i(b.repFlag)} {_f(b.targetMass)} {_i(b.frameFlag)}"]
    for block in (b.legendre, b.tabulated):
        if block is not None:
            out += _angularBlock(block)
    return out


def _energy(b: EnergyBody) -> List[str]:
    out = [f"{_f(b.dummy)} {len(b.partials)}"]
    for p in b.partials:
        out.append(_i(p.law))
        out += _tab1(p.probability)
        if p.law == 1:
            out.append(_i(p.nDistFunc))
            out += _interp(p.interpolation)
            for e, g in p.spectra:
                out.append(_f(e))
                out += _tab1(g)
        elif p.law == 12:
            out.append(f"{_f(p.scalars[0])} {_f(p.scalars[1])}")
            out += _tab1(p.parameters[0])
        else:
            for t in p.parameters:
                out += _tab1(t)
    return out


def _continuum(b: ContinuumBody) -> List[str]:
    out = [f"{_f(b.targetCode)} {_i(b.angularRep)} {_i(b.secondaryInterpolation)} "
           f"{len(b.energies)}"] + _interp(b.interpolation)
    for e in b.energies:
        out.append(f"{_f(e.energy)} {e.rows.shape[0]} {_i(e.nDiscrete)} {_i(e.nAngularParameters)}")
        for row in e.rows:
            out += _values(row)
    return out


def _twoBody(b: DiscreteTwoBodyBody) -> List[str]:
    out = [_i(b.nEnergy)] + _interp(b.interpolation)
    for e in b.energies:
        out.append(f"{_f(e.energy)} {_i(e.representation)} {_i(e.nCoefficients)}")
        out += _values(e.values)
    return out


def _labAngleEnergy(b: LabAngleEnergyBody) -> List[str]:
    out = [_i(len(b.energies))] + _interp(b.interpolation)
    for e in b.energies:
        out.append(f"{_f(e.energy)} {_i(e.nCosTh)}")
        out += _interp(e.interpolation)
        for a in e.angles:
            out.append(_f(a.mu))
            out += _tab1(a.spectrum)
    return out


def _energyAngle(b: EnergyAngleBody) -> List[str]:
    out = [f"{_f(b.targetMass)} {_i(b.frameFlag)} {_i(b.nProducts)}"]
    for p in b.products:
        out.append(f"{_f(p.massCode)} {_f(p.mass)} {_i(p.isomerFlag)} {_i(p.distLaw)} "
                   f"{_f(p.groundStateQ)} {_f(p.actualStateQ)}")
        out += _tab1(p.yield_)
        body = p.body
        if isinstance(body, ContinuumBody):
            out += _continuum(body)
        elif isinstance(body, DiscreteTwoBodyBody):
            out += _twoBody(body)
        elif isinstance(body, NBodyBody):
            out.append(f"{_f(body.totalMass)} {_i(body.totalCount)}")
        elif isinstance(body, LabAngleEnergyBody):
            out += _labAngleEnergy(body)
        else:
            assert body is None, type(body)
    return out


def _photonMean(b) -> List[str]:
    out = [f"{_i(b.repFlag)} {_f(b.targetMass)}"]
    if isinstance(b, PhotonMultiplicityBody):
        out.append(_i(b.nDiscrete))
        for line in b.lines:
            out.append(f"{_i(line.disType)} {_f(line.energy)}")
            out += _tab1(line.yield_)
        return out
    assert isinstance(b, PhotonCascadeBody)
    out.append(f"{_i(b.conversionFlag)} {_f(b.baseEnergy)} {_i(b.conversionFlag2)} "
               f"{_i(b.nGammaEnergies)}")
    for t in b.transitions:
        vals = [t.levelEnergy, t.probability] + ([t.photonFraction]
                                                 if b.conversionFlag2 == 2 else [])
        out.append(" ".join(_f(v) for v in vals))
    return out


def _photonPartials(b: PhotonPartialsBody) -> List[str]:
    out = [f"{_i(b.nDiscrete)} {_f(b.targetMass)}"]
    if b.total is not None:
        out += _tab1(b.total)
    for p in b.partials:
        out.append(f"{_f(p.energy)} {_f(p.shell)} {_i(p.isPrimary)} {_i(p.disType)}")
        out += _tab1(p.sigma)
    return out


def _photonAngular(b: PhotonAngularBody) -> List[str]:
    out = [_i(b.isoFlag)]
    if b.isoFlag == 1:
        return out
    out.append(f"{_i(b.tabulationType)} {_i(b.nDiscrete2)} {_i(b.nIso)}")
    for e, es in b.isotropic:
        out.append(f"{_f(e)} {_f(es)}")
    for line in b.lines:
        out.append(f"{_f(line.energy)} {_f(line.shell)} {_i(line.nNeu)}")
        if b.tabulationType == 1:
            out += _interp(line.interpolation)
            for r in line.records:
                out.append(f"{_f(r.energy)} {len(r.coefficients)}")
                out += _values(r.coefficients)
        else:
            for r in line.records:
                out.append(f"{_f(r.energy)} {len(r.mu)}")
                out += _interp(r.interpolation) + _interleaved(r.mu, r.probability)
    return out


def _photonEnergies(b: PhotonEnergyBody) -> List[str]:
    if not b.needed:
        return []
    out = [_i(len(b.spectra))]
    for s in b.spectra:
        out.append(_i(s.dummy))
        out += _tab1(s.probability)
        out.append(_i(len(s.spectra)))
        out += _interp(s.interpolation)
        for at in s.spectra:
            out.append(_f(at.energy))
            out += _tab1(at.spectrum)
    return out


_BODY = {
    DT_ANGULAR: _angular, DT_ENERGY: _energy, DT_ENERGY_ANGLE: _energyAngle,
    DT_PHOTON_MULTIPLICITY: _photonMean, DT_PHOTON_PARTIALS: _photonPartials,
    DT_PHOTON_ANGULAR: _photonAngular, DT_PHOTON_ENERGY: _photonEnergies,
}


def formatSectionBody(section, composite: bool) -> str:
    """The text of one section's body, header excluded: what a verbatim section keeps."""
    b = section.body
    if section.dataType == DT_CROSS_SECTION:
        out = [f"{_f(b.QI)} {_i(b.LR)}"] if composite else []
        out.append(_i(len(b.points)))
        return "\n".join(out + _interleaved(b.points.x, b.points.y))
    return "\n".join(_BODY[section.dataType](b))


def formatInelasticFS(record: InelasticFSRecord) -> str:
    """``Inelastic/Fxx`` text, section by section, in the consumer's reading order."""
    out = [f"{record.header[0]} {record.header[1]}"] if record.header is not None else []
    for k, s in enumerate(record.sections):
        head = f"{_i(s.infoType)} {_i(s.dataType)}"
        if record.composite:
            head += f" {_i(s.sfType)} {_i(s.dummy)}"
        elif k == 0:
            head += f" {_f(record.Qvalue)} {_i(record.Qdummy)}"
        out.append(head)
        b = s.body
        if s.dataType == DT_CROSS_SECTION:
            assert isinstance(b, CrossSectionBody)
            if record.composite:
                out.append(f"{_f(b.QI)} {_i(b.LR)}")
            out.append(_i(len(b.points)))
            out += _interleaved(b.points.x, b.points.y)
        else:
            out += _BODY[s.dataType](b)
    return "\n".join(out) + "\n"


def formatGammas(record: GammasRecord) -> str:
    """``Inelastic/Gammas/z<Z>.a<A>`` text: one triple per line, as distributed."""
    rows = zip(record.levels.tolist(), record.gammas.tolist(), record.probabilities.tolist())
    return "".join(f"{repr(a)} {repr(b)} {repr(c)}\n" for a, b, c in rows)


# ------------------------------------------------------------------ comparison

def _differences(a, b, where: str, out: List[str]) -> None:
    if len(out) > 50:
        return
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        if type(a) is not type(b):
            out.append(f"{where}: {type(a).__name__} != {type(b).__name__}")
            return
        for f in dataclasses.fields(a):
            if f.name == "path":
                continue
            _differences(getattr(a, f.name), getattr(b, f.name), f"{where}.{f.name}", out)
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape or not np.all((x == y) | (np.isnan(x) & np.isnan(y))):
            out.append(f"{where}: arrays differ")
        return
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            out.append(f"{where}: {len(a)} != {len(b)} items")
            return
        for k, (x, y) in enumerate(zip(a, b)):
            _differences(x, y, f"{where}[{k}]", out)
        return
    if a != b:
        out.append(f"{where}: {a!r} != {b!r}")


def inelasticDifferences(a, b) -> List[str]:
    """Every field where two inelastic records differ (at most ~50); ``path`` ignored."""
    out: List[str] = []
    _differences(a, b, type(a).__name__, out)
    return out
