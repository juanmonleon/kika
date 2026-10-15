"""OpenMC's angle-energy distributions ↔ the model's Monte Carlo forms.

Every table OpenMC stores is tabulated with its cumulative, and decodes to
``xs_pdf_cdf1d`` nodes:

=================================  ===========================================
``distribution_<i>/@type``         model form (label ``MonteCarlo``)
=================================  ===========================================
``uncorrelated``, angle only       :class:`AngularTwoBody` (elastic)
``uncorrelated``                   :class:`Uncorrelated`: angle an ``XYs2d`` of
                                   ``xs_pdf_cdf1d`` in mu (absent:
                                   :class:`Isotropic2d`); energy per its type
``correlated``                     :class:`EnergyAngularMC`
``kalbach-mann``                   :class:`KalbachMann`: ``f`` of
                                   ``xs_pdf_cdf1d``, ``r`` and ``a`` of ``XYs1d``
``nbody``                          :class:`Uncorrelated` with
                                   :class:`NBodyPhaseSpace`
=================================  ===========================================

Energy types of ``uncorrelated``: ``continuous`` (a function of incident
energy of ``xs_pdf_cdf1d``, discrete lines first), ``level``
(:class:`LevelEnergy`), ``discrete_photon`` (:class:`DiscretePhotonEnergy`),
``maxwell`` / ``evaporation`` / ``watt`` / ``madland-nix`` (the §18.3
parametrised spectra).

The layouts are ``correlated.py:107-200``, ``kalbach_mann.py:360-423``,
``energy_distribution.py:1066-1126`` and ``angle_distribution.py:61-102`` of
OpenMC 0.16.0. Offsets are rebuilt on write from the table lengths, as OpenMC
builds them.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import records as R
from .exceptions import HDF5FormatError, HDF5UnsupportedError
from .functions import decodeFunction, encodeFunction
from .names import MONTE_CARLO_LABEL

__all__ = ["decodeDistribution", "encodeDistribution", "decodeAngle", "encodeAngle"]

#: The flag OpenMC writes on a correlated mu table that is a set of discrete cosines.
MU_DISCRETE = 0


def _m():
    from kika.nuclear_data import model

    return model


def _code(interpolation) -> int:
    m = _m()
    return 1 if m.Interpolation(interpolation) == m.Interpolation.flat else 2


def _interp(code: int):
    m = _m()
    if code == 1:
        return m.Interpolation.flat
    if code == 2:
        return m.Interpolation.linlin
    raise HDF5FormatError(f"table interpolation {code}; OpenMC writes 1 or 2")


def _blocks(offsets, total: int):
    offsets = [int(o) for o in offsets]
    return list(zip(offsets, offsets[1:] + [total]))


def _pdfCdf(table, start, stop, code, discrete, outer):
    m = _m()
    return m.Xs_pdf_cdf1d(xs=table[0, start:stop], pdf=table[1, start:stop],
                          cdf=table[2, start:stop], interpolation=_interp(code),
                          numberOfDiscrete=int(discrete), outerDomainValue=float(outer))


def _incidentForm(functions, interpolation, label=None):
    """``energy@interpolation`` (2, NR) over the per-energy tables → XYs2d / Regions2d."""
    m = _m()
    interpolation = np.asarray(interpolation)
    pairs = list(zip(interpolation[0].tolist(), interpolation[1].tolist()))
    return m.fromEndfTab2(functions, pairs, label=label)


def _incidentAttr(form):
    m = _m()
    functions, pairs = m.toEndfTab2(form)
    table = np.array([[p[0] for p in pairs], [p[1] for p in pairs]], dtype=np.int64)
    return functions, table


def _energyDataset(functions, interpolationTable):
    dataset = R.Dataset.of(np.array([f.outerDomainValue for f in functions], dtype=float))
    dataset.attrs["interpolation"] = R.Attribute.of(interpolationTable)
    return dataset


def _tableAttrs(dataset, offsets, codes, discrete):
    dataset.attrs["offsets"] = R.Attribute.of(np.asarray(offsets, dtype=np.int64))
    dataset.attrs["interpolation"] = R.Attribute.of(np.asarray(codes, dtype=np.int64))
    dataset.attrs["n_discrete_lines"] = R.Attribute.of(np.asarray(discrete, dtype=np.int64))


# ----------------------------------------------------------------------
# Angle
# ----------------------------------------------------------------------

def decodeAngle(group, where=""):
    """``angle/`` → an ``XYs2d`` of ``xs_pdf_cdf1d`` in mu."""
    m = _m()
    energy = group["energy"].data
    mu = group["mu"]
    table = mu.data
    offsets = R.attr(mu, "offsets")
    codes = R.attr(mu, "interpolation")
    functions = [_pdfCdf(table, a, b, codes[i], 0, energy[i])
                 for i, (a, b) in enumerate(_blocks(offsets, table.shape[1]))]
    return m.XYs2d(function1ds=functions, label=MONTE_CARLO_LABEL)


def encodeAngle(form) -> R.Group:
    m = _m()
    if not isinstance(form, m.XYs2d):
        raise HDF5UnsupportedError(
            f"an angular distribution must be tabulated as an XYs2d of xs_pdf_cdf1d to be "
            f"written; got {type(form).__name__}. Tabulate it first "
            f"(kika.processing.tabulation)")
    tables = list(form.function1ds)
    _checkTables(tables, "angle")
    group = R.Group()
    group["energy"] = R.Dataset.of(np.array([t.outerDomainValue for t in tables], dtype=float))
    data = np.vstack([np.concatenate([t.xs for t in tables]),
                      np.concatenate([t.pdf for t in tables]),
                      np.concatenate([t.cdf for t in tables])])
    mu = R.Dataset.of(data)
    lengths = [len(t) for t in tables]
    mu.attrs["offsets"] = R.Attribute.of(np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64))
    mu.attrs["interpolation"] = R.Attribute.of(np.array([_code(t.interpolation) for t in tables],
                                                        dtype=np.int64))
    group["mu"] = mu
    return group


def _checkTables(tables, what):
    m = _m()
    for t in tables:
        if not isinstance(t, m.Xs_pdf_cdf1d):
            raise HDF5UnsupportedError(
                f"{what}: a {type(t).__name__} is not a table with its cumulative; OpenMC "
                f"samples the stored CDF as it is, so every table must be an xs_pdf_cdf1d")


# ----------------------------------------------------------------------
# Tables of outgoing energy (continuous, kalbach-mann, correlated)
# ----------------------------------------------------------------------

def _decodeEnergyTables(group, tableName, where):
    energy = group["energy"]
    incident = energy.data
    table = group[tableName]
    data = table.data
    offsets = R.attr(table, "offsets")
    codes = R.attr(table, "interpolation")
    discrete = R.attr(table, "n_discrete_lines")
    blocks = _blocks(offsets, data.shape[1])
    if len(blocks) != len(incident):
        raise HDF5FormatError(f"{len(blocks)} tables for {len(incident)} incident energies", node=where)
    return incident, R.attr(energy, "interpolation"), data, blocks, codes, discrete


def _encodeEnergyTables(form, rows):
    """Concatenate the per-energy tables; ``rows(t)`` gives the extra rows of table ``t``."""
    functions, interpolationTable = _incidentAttr(form)
    _checkTables(functions, "outgoing energy")
    lengths = [len(t) for t in functions]
    columns = [np.vstack([t.xs, t.pdf, t.cdf] + rows(i, t)) for i, t in enumerate(functions)]
    data = np.concatenate(columns, axis=1)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    codes = [_code(t.interpolation) for t in functions]
    discrete = [t.numberOfDiscrete for t in functions]
    return functions, interpolationTable, data, offsets, codes, discrete


def decodeContinuous(group, where=""):
    incident, interpolation, data, blocks, codes, discrete = _decodeEnergyTables(group, "distribution", where)
    functions = [_pdfCdf(data, a, b, codes[i], discrete[i], incident[i])
                 for i, (a, b) in enumerate(blocks)]
    return _incidentForm(functions, interpolation)


def encodeContinuous(form) -> R.Group:
    functions, interp, data, offsets, codes, discrete = _encodeEnergyTables(form, lambda i, t: [])
    group = R.Group()
    group.setAttr("type", "continuous")
    group["energy"] = _energyDataset(functions, interp)
    dataset = R.Dataset.of(data)
    _tableAttrs(dataset, offsets, codes, discrete)
    group["distribution"] = dataset
    return group


def decodeKalbachMann(group, frame, where=""):
    m = _m()
    incident, interpolation, data, blocks, codes, discrete = _decodeEnergyTables(group, "distribution", where)
    f, r, a = [], [], []
    for i, (lo, hi) in enumerate(blocks):
        f.append(_pdfCdf(data, lo, hi, codes[i], discrete[i], incident[i]))
        law = _interp(int(codes[i]))
        r.append(m.XYs1d(xs=data[0, lo:hi], ys=data[3, lo:hi], interpolation=law,
                         outerDomainValue=float(incident[i])))
        a.append(m.XYs1d(xs=data[0, lo:hi], ys=data[4, lo:hi], interpolation=law,
                         outerDomainValue=float(incident[i])))
    return m.KalbachMann(f=_incidentForm(f, interpolation), r=_incidentForm(r, interpolation),
                         a=_incidentForm(a, interpolation), label=MONTE_CARLO_LABEL,
                         productFrame=frame)


def encodeKalbachMann(form) -> R.Group:
    m = _m()
    r, _ = m.toEndfTab2(form.r)
    a = m.toEndfTab2(form.a)[0] if form.a is not None else None
    if a is None:
        raise HDF5UnsupportedError("a Kalbach-Mann form without its slope a cannot be written: "
                                   "OpenMC stores a per outgoing point (tabulate it with the "
                                   "systematics first, kika.processing.tabulation)")

    def rows(i, t):
        if len(r[i]) != len(t) or len(a[i]) != len(t):
            raise HDF5FormatError("Kalbach-Mann r and a must be on the outgoing grid of f")
        return [r[i].ys, a[i].ys]

    functions, interp, data, offsets, codes, discrete = _encodeEnergyTables(form.f, rows)
    group = R.Group()
    group.setAttr("type", "kalbach-mann")
    group["energy"] = _energyDataset(functions, interp)
    dataset = R.Dataset.of(data)
    _tableAttrs(dataset, offsets, codes, discrete)
    group["distribution"] = dataset
    return group


def decodeCorrelated(group, frame, where="", label=MONTE_CARLO_LABEL):
    m = _m()
    incident, interpolation, data, blocks, codes, discrete = _decodeEnergyTables(group, "energy_out", where)
    mu = group["mu"].data
    muStarts = data[4].astype(int)
    energy, inner = [], []
    for i, (lo, hi) in enumerate(blocks):
        energy.append(_pdfCdf(data, lo, hi, codes[i], discrete[i], incident[i]))
        tables = []
        for j in range(lo, hi):
            start = muStarts[j]
            stop = muStarts[j + 1] if j + 1 < data.shape[1] else mu.shape[1]
            flag = int(data[3, j])
            n = stop - start
            tables.append(m.Xs_pdf_cdf1d(
                xs=mu[0, start:stop], pdf=mu[1, start:stop], cdf=mu[2, start:stop],
                interpolation=m.Interpolation.flat if flag in (MU_DISCRETE, 1) else _interp(flag),
                numberOfDiscrete=n if flag == MU_DISCRETE else 0,
                outerDomainValue=float(data[0, j])))
        inner.append(m.XYs2d(function1ds=tables, outerDomainValue=float(incident[i])))
    energyForm = _incidentForm(energy, interpolation)
    xys3d = m.XYs3d(function2ds=inner)
    if isinstance(energyForm, m.XYs2d):
        # The incident-energy law is stated once, on the energy table; the
        # angular one repeats it, for a reader of the XYs3d alone.
        xys3d.interpolation = energyForm.interpolation
        xys3d.interpolationQualifier = energyForm.interpolationQualifier
    return m.EnergyAngularMC(energy=energyForm, energyAngular=xys3d, label=label, productFrame=frame)


def encodeCorrelated(form, typeName="correlated") -> R.Group:
    m = _m()
    inner = list(form.energyAngular.function2ds)
    muTables: List = []
    muStart = [0]

    def rows(i, t):
        tables = list(inner[i].function1ds)
        if len(tables) != len(t):
            raise HDF5FormatError(f"incident energy {i}: {len(tables)} mu tables for "
                                  f"{len(t)} outgoing energies")
        _checkTables(tables, "mu")
        flags, starts = [], []
        for table in tables:
            allDiscrete = table.numberOfDiscrete and table.numberOfDiscrete == len(table)
            flags.append(MU_DISCRETE if allDiscrete else _code(table.interpolation))
            starts.append(muStart[0])
            muStart[0] += len(table)
            muTables.append(table)
        return [np.asarray(flags, float), np.asarray(starts, float)]

    functions, interp, data, offsets, codes, discrete = _encodeEnergyTables(form.energy, rows)
    group = R.Group()
    group.setAttr("type", typeName)
    group["energy"] = _energyDataset(functions, interp)
    dataset = R.Dataset.of(data)
    _tableAttrs(dataset, offsets, codes, discrete)
    group["energy_out"] = dataset
    group["mu"] = R.Dataset.of(np.vstack([np.concatenate([t.xs for t in muTables]),
                                          np.concatenate([t.pdf for t in muTables]),
                                          np.concatenate([t.cdf for t in muTables])])
                               if muTables else np.zeros((3, 0)))
    return group


# ----------------------------------------------------------------------
# Uncorrelated energy laws
# ----------------------------------------------------------------------

def _pq(value, unit="eV"):
    return _m().PhysicalQuantity(value=float(value), unit=unit)


def decodeEnergy(group, where=""):
    m = _m()
    kind = R.text(R.attr(group, "type"))
    if kind == "continuous":
        return decodeContinuous(group, where)
    if kind == "level":
        return m.LevelEnergy(threshold=float(R.attr(group, "threshold")),
                             massRatio=float(R.attr(group, "mass_ratio")))
    if kind == "discrete_photon":
        return m.DiscretePhotonEnergy(primaryFlag=int(R.attr(group, "primary_flag")),
                                      energy=float(R.attr(group, "energy")),
                                      atomicWeightRatio=float(R.attr(group, "atomic_weight_ratio")))
    if kind == "maxwell":
        return m.SimpleMaxwellianFission(U=_pq(R.attr(group, "u")),
                                         theta=decodeFunction(group["theta"], where))
    if kind == "evaporation":
        return m.Evaporation(U=_pq(R.attr(group, "u")), theta=decodeFunction(group["theta"], where))
    if kind == "watt":
        return m.Watt(U=_pq(R.attr(group, "u")), a=decodeFunction(group["a"], where),
                      b=decodeFunction(group["b"], where))
    if kind == "madland-nix":
        return m.MadlandNix(EFL=_pq(R.attr(group, "efl")), EFH=_pq(R.attr(group, "efh")),
                            T_M=decodeFunction(group["xy"], where))
    raise HDF5UnsupportedError(f"energy distribution type {kind!r} [{where}]")


def encodeEnergy(energy) -> R.Group:
    m = _m()
    group = R.Group()
    if isinstance(energy, (m.XYs2d, m.Regions2d)):
        return encodeContinuous(energy)
    if isinstance(energy, m.LevelEnergy):
        group.setAttr("type", "level")
        group.setAttr("threshold", float(energy.threshold))
        group.setAttr("mass_ratio", float(energy.massRatio))
        return group
    if isinstance(energy, m.DiscretePhotonEnergy):
        group.setAttr("type", "discrete_photon")
        group.setAttr("primary_flag", int(energy.primaryFlag))
        group.setAttr("energy", float(energy.energy))
        group.setAttr("atomic_weight_ratio", float(energy.atomicWeightRatio))
        return group
    if isinstance(energy, (m.SimpleMaxwellianFission, m.Evaporation)):
        group.setAttr("type", "maxwell" if isinstance(energy, m.SimpleMaxwellianFission) else "evaporation")
        group.setAttr("u", float(energy.U.value))
        group["theta"] = _tabulatedOnly(energy.theta)
        return group
    if isinstance(energy, m.Watt):
        group.setAttr("type", "watt")
        group.setAttr("u", float(energy.U.value))
        group["a"] = _tabulatedOnly(energy.a)
        group["b"] = _tabulatedOnly(energy.b)
        return group
    if isinstance(energy, m.MadlandNix):
        # The C++ cannot sample it (secondary_uncorrelated.cpp:64-65); the
        # writer tabulates Madland-Nix as continuous before it gets here.
        raise HDF5UnsupportedError("madland-nix is not sampled by OpenMC's C++; write it as a "
                                   "continuous table (kika.processing.tabulation)")
    raise HDF5UnsupportedError(f"a {type(energy).__name__} has no OpenMC energy layout; "
                               f"tabulate it first (kika.processing.tabulation)")


def _tabulatedOnly(function):
    """theta, a and b are read as Tabulated1D by the C++ (distribution_energy.cpp:274-345)."""
    dataset = encodeFunction(function)
    if R.text(R.attr(dataset, "type")) != "Tabulated1D":
        raise HDF5UnsupportedError("the parameters of maxwell/evaporation/watt must be "
                                   "tabulated: the C++ reads them as Tabulated1D")
    return dataset


# ----------------------------------------------------------------------
# One distribution_<i>
# ----------------------------------------------------------------------

def decodeDistribution(group, frame, where="", nbody=None):
    """``distribution_<i>`` (without its ``applicability``) → a model form.

    ``nbody`` collects the raw attributes of an ``nbody`` distribution, which
    the model states as a number of products and a mass; the atomic weight
    ratio and Q value it repeats belong to the nuclide and the reaction.
    """
    m = _m()
    kind = R.text(R.attr(group, "type"))
    if kind == "uncorrelated":
        angular = decodeAngle(group["angle"], where) if "angle" in group else None
        if "energy" not in group:
            if angular is None:
                raise HDF5FormatError("uncorrelated with neither angle nor energy", node=where)
            return m.AngularTwoBody(angular=angular, productFrame=frame, label=MONTE_CARLO_LABEL)
        return m.Uncorrelated(angular=angular if angular is not None else m.Isotropic2d(),
                              energy=decodeEnergy(group["energy"], where),
                              label=MONTE_CARLO_LABEL, productFrame=frame)
    if kind == "correlated":
        return decodeCorrelated(group, frame, where)
    if kind == "kalbach-mann":
        return decodeKalbachMann(group, frame, where)
    if kind == "nbody":
        if nbody is not None:
            nbody.update({k: R.attr(group, k) for k in
                          ("total_mass", "n_particles", "atomic_weight_ratio", "q_value")})
        return m.Uncorrelated(angular=m.Isotropic2d(),
                              energy=m.NBodyPhaseSpace(
                                  numberOfProducts=int(R.attr(group, "n_particles")),
                                  mass=_pq(float(R.attr(group, "total_mass")) * _neutronMass(), "amu")),
                              label=MONTE_CARLO_LABEL, productFrame=frame)
    raise HDF5UnsupportedError(f"distribution type {kind!r} [{where}]")


def _neutronMass():
    from kika._constants import NEUTRON_MASS_AMU

    return NEUTRON_MASS_AMU


def encodeDistribution(form, *, awr=None, qValue=None, nbody=None) -> R.Group:
    """A Monte Carlo form → ``distribution_<i>`` (without ``applicability``)."""
    m = _m()
    if isinstance(form, m.AngularTwoBody):
        group = R.Group()
        group.setAttr("type", "uncorrelated")
        group["angle"] = encodeAngle(form.angular)
        return group
    if isinstance(form, m.Uncorrelated):
        if isinstance(form.energy, m.NBodyPhaseSpace):
            group = R.Group()
            group.setAttr("type", "nbody")
            stored = nbody or {}
            mass = stored.get("total_mass")
            if mass is None or int(stored.get("n_particles", -1)) != form.energy.numberOfProducts:
                mass = form.energy.mass.value / _neutronMass()
            group.setAttr("total_mass", float(mass))
            group.setAttr("n_particles", int(form.energy.numberOfProducts))
            group.setAttr("atomic_weight_ratio", float(stored.get("atomic_weight_ratio", awr)))
            group.setAttr("q_value", float(stored.get("q_value", qValue)))
            return group
        group = R.Group()
        group.setAttr("type", "uncorrelated")
        if form.angular is not None and not isinstance(form.angular, m.Isotropic2d):
            group["angle"] = encodeAngle(form.angular)
        if form.energy is not None:
            group["energy"] = encodeEnergy(form.energy)
        return group
    if isinstance(form, m.EnergyAngularMC):
        return encodeCorrelated(form)
    if isinstance(form, m.KalbachMann):
        return encodeKalbachMann(form)
    raise HDF5UnsupportedError(
        f"a {type(form).__name__} has no OpenMC layout; the writer needs the Monte Carlo "
        f"forms (xs_pdf_cdf1d tables) that kika.processing.tabulation produces")
