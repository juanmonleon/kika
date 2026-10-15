"""Model → OpenMC incident-neutron HDF5.

Writes what the C++ reads, directly with h5py -- not through ``openmc.data``,
whose exporter refuses ENDF-derived data, and which is not a dependency. The encoder serialises; it does not process. It needs the suite
in the processed shape :mod:`kika.hdf5.neutron_decode` produces and
:func:`kika.hdf5.writer.writeSuite` builds from an evaluation:

- one ``heated_<T>`` style per temperature, each carrying its kT;
- every reaction's cross section at every temperature as an ``XYs1d`` on that
  temperature's common grid -- each one a suffix of the grid, which is what
  ``threshold_idx`` says (``len(xs) + threshold_idx == len(energy/<T>)``);
- every distribution and yield as a ``MonteCarlo`` form (tables with their CDF).

The contract it enforces, and refuses to break:

- group names ``f"{round(kT/k_B)}K"``, the same in ``kTs``, ``energy``, every
  reaction and ``urr``;
- every reaction has every temperature; no ``reaction_001``;
- ``redundant`` set on every reaction that is not an exclusive partial, so the
  C++'s rebuilt total counts each partial once;
- fission products in the order the C++ assumes: the prompt (or total) neutron
  first, then the delayed groups, then the rest (``nuclide.cpp:529-570``);
- ``urr/<T>`` for every temperature when present, and the MT its ``inelastic``
  names written.
"""
from __future__ import annotations

import string
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import records as R
from .distributions import encodeDistribution
from .exceptions import HDF5FormatError, HDF5UnsupportedError
from .functions import encodeFunction
from .names import (FISSION_MTS, HDF5_VERSION, K_BOLTZMANN, MONTE_CARLO_LABEL, RECON_LABEL,
                    gndsName, heatedLabel, reactionLabel, temperatureName, temperatureOfLabel,
                    urrLabel)
from .neutron_decode import FISSION_ENERGY_TERMS, PID_TO_PARTICLE, RESPONSE_MTS, TOTAL_NUBAR_LABEL

__all__ = ["encodeNeutron", "heatedTemperatures", "kTOf", "monteCarloForm", "yieldForm"]


def _m():
    from kika.nuclear_data import model

    return model


def kTOf(style) -> float:
    """The kT in eV a ``heated`` style states, whatever unit it states it in."""
    temperature = style.temperature
    if temperature is None:
        raise HDF5FormatError(f"heated style {style.label!r} has no temperature")
    value, unit = float(temperature.value), (temperature.unit or "K")
    if unit == "eV/k":
        return value
    if unit == "MeV/k":
        return value * 1.0e6
    if unit == "K":
        return value * K_BOLTZMANN
    raise HDF5UnsupportedError(f"heated temperature in {unit!r}")


def heatedTemperatures(suite, temperatures=None) -> Dict[str, float]:
    """``{group name: kT}`` for the ``heated_<T>`` styles to write, in name order."""
    m = _m()
    found = {}
    for style in suite.styles:
        if isinstance(style, m.Heated) and style.label.startswith("heated_"):
            kT = kTOf(style)
            name = temperatureName(kT)
            if name != temperatureOfLabel(style.label):
                raise HDF5FormatError(
                    f"style {style.label!r} states kT={kT!r} eV, which OpenMC names {name!r}; "
                    f"the C++ opens groups by that name (nuclide.cpp:226)")
            found[name] = kT
    if temperatures is not None:
        wanted = [t if isinstance(t, str) else temperatureName(float(t) * K_BOLTZMANN)
                  for t in temperatures]
        missing = [t for t in wanted if t not in found]
        if missing:
            raise HDF5FormatError(f"no heated style for {missing}; the suite has {sorted(found)}")
        found = {t: found[t] for t in wanted}
    if not found:
        raise HDF5FormatError("the suite has no heated_<T> style; OpenMC needs at least one "
                              "temperature (broaden first, kika.processing.doppler)")
    return dict(sorted(found.items()))


def monteCarloForm(component, what: str):
    """The ``MonteCarlo`` form of a distribution, or a refusal that says what to do."""
    if component is None:
        return None
    if MONTE_CARLO_LABEL in component.forms:
        return component.forms[MONTE_CARLO_LABEL]
    raise HDF5UnsupportedError(
        f"{what} has no {MONTE_CARLO_LABEL!r} form ({sorted(component.forms)}); OpenMC samples "
        f"tables with their CDF, which kika.processing.tabulation produces")


def yieldForm(multiplicity):
    """The multiplicity to write: the ``MonteCarlo`` form, else the evaluated one."""
    if multiplicity is None:
        return None
    if MONTE_CARLO_LABEL in multiplicity.forms:
        return multiplicity.forms[MONTE_CARLO_LABEL]
    return multiplicity.form


def _provenance(node, key, default=None):
    m = _m()
    provenance = getattr(node, "provenance", None)
    if isinstance(provenance, m.Hdf5Provenance):
        return provenance.attributes.get(key, default)
    return default


#: A radionuclide production (MF8-10) shares its MT with the reaction that makes
#: it, and OpenMC names a group by MT alone, so it has no place in this file:
#: neither the NJOY/ACE route nor ``openmc.data`` writes one.
from kika.nuclear_data.model import isRadionuclideProduction  # noqa: E402


def _allReactions(suite):
    for container in (suite.reactions, suite.fissionComponents, suite.sums,
                      suite.productions, suite.incompleteReactions):
        for reaction in container:
            if container is suite.productions and isRadionuclideProduction(reaction):
                continue
            yield container, reaction


def _redundant(suite, container, reaction) -> int:
    stated = _provenance(reaction, "redundant")
    if stated is not None:
        return int(stated)
    if container is suite.reactions:
        return 0
    if container is suite.fissionComponents:
        return 0
    return 1


# ----------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------

def _grid(suite, label) -> np.ndarray:
    """The common grid of one temperature: the longest cross section, all others its suffixes."""
    forms = [r.crossSection.forms[label] for _, r in _allReactions(suite)
             if label in r.crossSection.forms]
    if not forms:
        raise HDF5FormatError(f"no reaction has a {label!r} cross section")
    grid = max(forms, key=len).xs
    for form in forms:
        tail = grid[len(grid) - len(form):]
        if len(form) > len(grid) or not np.array_equal(tail, form.xs):
            raise HDF5FormatError(
                f"the {label!r} cross sections are not on one common grid; OpenMC stores one "
                f"grid per temperature and each reaction from a threshold index on it "
                f"(kika.algebra builds the union grid)")
    if np.any(np.diff(grid) <= 0):
        raise HDF5FormatError(f"the {label!r} grid is not strictly increasing; the C++ assumes it is")
    return grid


# ----------------------------------------------------------------------
# The file
# ----------------------------------------------------------------------

def encodeNeutron(suite, temperatures=None, *, awr: Optional[float] = None,
                  name: Optional[str] = None, zeroKelvinElastic: bool = True,
                  report=None) -> R.Group:
    """The record tree of one incident-neutron file.

    ``temperatures`` restricts the ``heated_<T>`` styles written (names or
    kelvins); by default all of them. ``awr`` overrides the atomic weight ratio
    (the writer passes the one the official library uses).
    """
    m = _m()
    report = report if report is not None else m.ConversionReport()
    temps = heatedTemperatures(suite, temperatures)
    target = suite.PoPs[suite.target] if suite.target in suite.PoPs else None
    Z = _provenance(suite, "Z", getattr(target, "Z", None))
    A = _provenance(suite, "A", getattr(target, "A", None))
    metastable = _provenance(suite, "metastable", 0)
    if Z is None or A is None:
        raise HDF5FormatError(f"the target {suite.target!r} has no Z and A in PoPs")
    if awr is None:
        awr = _provenance(suite, "atomic_weight_ratio")
    if awr is None:
        awr = _awrFromModel(suite, target)
    name = name or (suite.target if _provenance(suite, "Z") is not None else gndsName(Z, A, metastable))

    root = R.Group()
    root.setAttr("filetype", "data_neutron")
    root.setAttr("version", np.array(HDF5_VERSION, dtype=np.int64))
    g = root.require(name)
    g.setAttr("Z", int(Z))
    g.setAttr("A", int(A))
    g.setAttr("metastable", int(metastable))
    g.setAttr("atomic_weight_ratio", float(awr))

    grids = {}
    for T, kT in temps.items():
        g.require("kTs")[T] = R.Dataset.of(np.float64(kT))
        grids[T] = _grid(suite, heatedLabel(T))
        g.require("energy")[T] = R.Dataset.of(grids[T])
    elastic = suite.findReactionByENDF_MT(2)
    zeroK = (zeroKelvinElastic and elastic is not None
             and RECON_LABEL in elastic.crossSection.forms)
    if zeroK:
        g.require("energy")["0K"] = R.Dataset.of(elastic.crossSection.forms[RECON_LABEL].xs)

    skipped = sum(1 for p in suite.productions if isRadionuclideProduction(p))
    if skipped:
        report.lost(f"{skipped} radionuclide production(s) (MF8-10) have no group in "
                    f"OpenMC's format, which names reactions by MT alone; not written",
                    unaffectedScopes=("cross-sections",))

    reactions = g.require("reactions")
    written = {}
    for container, reaction in _allReactions(suite):
        mt = reaction.ENDF_MT
        if mt is None:
            report.lost(f"reaction {reaction.label!r} has no MT and OpenMC names groups by MT; "
                        f"not written")
            continue
        if mt == 1:
            # The C++ takes reactions_[0] as elastic and reads groups in name
            # order, so a reaction_001 would become "elastic" (nuclide.cpp:581).
            report.warn("MT1 is never written: the C++ rebuilds the total and a reaction_001 "
                        "would be read as the elastic")
            continue
        key = f"reaction_{mt:03d}"
        if key in written:
            raise HDF5FormatError(f"two reactions with MT{mt}")
        written[key] = reaction
        reactions[key] = _encodeReaction(suite, container, reaction, temps, grids,
                                         zeroK and mt == 2, awr, report)
    reactions.children = dict(sorted(reactions.children.items()))

    totalNu = suite.sums.multiplicitySums.byLabel(TOTAL_NUBAR_LABEL) \
        or suite.sums.multiplicitySums.byENDF_MT(452)
    fission = next((suite.findReactionByENDF_MT(mt) for mt in FISSION_MTS
                    if suite.findReactionByENDF_MT(mt) is not None), None)
    if totalNu is not None and fission is not None:
        g["total_nu"] = _totalNuGroup(totalNu)
    if fission is not None:
        release = _fissionEnergyRelease(fission)
        if release is not None:
            g["fission_energy_release"] = _encodeFissionEnergy(release)
    urr = _urrTables(suite, temps)
    if urr:
        _checkUrrInelastic(urr, written)
        g["urr"] = R.Group(children={T: _encodeUrr(tables) for T, tables in urr.items()})
    g.children = dict(sorted(g.children.items()))
    return root


def _awrFromModel(suite, target):
    m = _m()
    from kika._constants import NEUTRON_MASS_AMU

    provenance = suite.provenance
    stated = getattr(provenance, "awr", None)
    if stated:
        return float(stated)
    if target is not None and target.mass is not None:
        return float(target.mass.value) / NEUTRON_MASS_AMU
    raise HDF5FormatError("no atomic weight ratio: pass awr=, or give the target a mass")


# ----------------------------------------------------------------------
# One reaction
# ----------------------------------------------------------------------

def _centerOfMass(reaction) -> int:
    stated = _provenance(reaction, "center_of_mass")
    if stated is not None:
        return int(stated)
    m = _m()
    for product in _orderedProducts(reaction):
        if product.pid == "n" and product.distribution is not None and product.distribution.forms:
            form = next(iter(product.distribution.forms.values()))
            frame = getattr(form, "productFrame", None)
            if frame is None and isinstance(form, m.DistributionMixture) and form.weighted:
                frame = getattr(form.weighted[0].functional, "productFrame", None)
            return int(frame == m.Frame.centerOfMass)
    return 0


def _encodeReaction(suite, container, reaction, temps, grids, zeroK, awr, report) -> R.Group:
    mt = reaction.ENDF_MT
    group = R.Group()
    group.setAttr("mt", int(mt))
    group.setAttr("label", _provenance(reaction, "label") or reactionLabel(mt))
    Q = reaction.outputChannel.Q.value if reaction.outputChannel.Q is not None else None
    group.setAttr("Q_value", float(Q) if Q is not None else 0.0)
    group.setAttr("center_of_mass", _centerOfMass(reaction))
    group.setAttr("redundant", _redundant(suite, container, reaction))
    for T in temps:
        label = heatedLabel(T)
        form = reaction.crossSection.forms.get(label)
        if form is None:
            raise HDF5FormatError(
                f"MT{mt} has no {label!r} cross section; the C++ needs every reaction at every "
                f"temperature it loads (no per-reaction fallback)")
        group[T] = _xsGroup(form, grids[T])
    if zeroK:
        form = reaction.crossSection.forms[RECON_LABEL]
        group["0K"] = _xsGroup(form, form.xs)
    for i, product in enumerate(_orderedProducts(reaction)):
        group[f"product_{i}"] = _encodeProduct(product, reaction, awr)
    return group


def _xsGroup(form, grid) -> R.Group:
    threshold = len(grid) - len(form)
    dataset = R.Dataset.of(np.asarray(form.ys, dtype=float))
    dataset.attrs["threshold_idx"] = R.Attribute.of(np.int64(threshold))
    return R.Group(children={"xs": dataset})


def _orderedProducts(reaction) -> List:
    """Products in the order the C++ reads them: for fission, prompt, delayed, rest."""
    channel = reaction.outputChannel
    products = list(channel.products)
    data = channel.fissionFragmentData
    delayed = [d.product for d in data.delayedNeutrons] if data is not None else []
    if reaction.ENDF_MT not in FISSION_MTS:
        return products + delayed
    neutrons = [p for p in products if p.pid == "n"]
    rest = [p for p in products if p.pid != "n"]
    return neutrons[:1] + delayed + neutrons[1:] + rest


def _emissionMode(product, reaction, delayedProducts) -> str:
    stated = _provenance(product, "emission_mode")
    if stated:
        return stated
    if any(product is d for d in delayedProducts):
        return "delayed"
    # A fission neutron with no delayed families beside it carries the total
    # nu (OpenMC's ``whichnu``: prompt only when delayed data exist).
    channel = getattr(reaction, "outputChannel", None)
    ffd = getattr(channel, "fissionFragmentData", None)
    if (product.pid == "n" and reaction.ENDF_MT in FISSION_MTS
            and (ffd is None or not ffd.delayedNeutrons.delayedNeutrons)):
        return "total"
    return "prompt"


def _encodeProduct(product, reaction, awr) -> R.Group:
    m = _m()
    data = reaction.outputChannel.fissionFragmentData
    delayed = [d for d in data.delayedNeutrons] if data is not None else []
    group = R.Group()
    particle = _provenance(product, "particle") or PID_TO_PARTICLE.get(product.pid, product.pid)
    group.setAttr("particle", particle)
    mode = _emissionMode(product, reaction, [d.product for d in delayed])
    group.setAttr("emission_mode", mode)
    rate = next((d.rate for d in delayed if d.product is product), None)
    if rate is not None and rate.value is not None and float(rate.value) > 0:
        if rate.unit not in ("1/s", "s**-1"):
            raise HDF5UnsupportedError(f"a decay rate in {rate.unit!r}; OpenMC stores 1/s")
        group.setAttr("decay_rate", float(rate.value))
    function = yieldForm(product.multiplicity)
    if function is None:
        raise HDF5FormatError(f"MT{reaction.ENDF_MT} {product.label}: no multiplicity to write "
                              f"as the yield")
    group["yield"] = encodeFunction(function)
    form = monteCarloForm(product.distribution, f"MT{reaction.ENDF_MT} {product.label}")
    Q = reaction.outputChannel.Q.value
    nbody = _provenance(product, "nbody")
    if form is None:
        group.setAttr("n_distribution", 0)
        return group
    weighted = form.weighted if isinstance(form, m.DistributionMixture) else [m.Weighted(functional=form)]
    group.setAttr("n_distribution", len(weighted))
    for k, term in enumerate(weighted):
        dgroup = encodeDistribution(term.functional, awr=awr, qValue=Q, nbody=nbody)
        if term.weight is not None:
            dgroup["applicability"] = encodeFunction(term.weight)
        group[f"distribution_{k}"] = dgroup
    return group


# ----------------------------------------------------------------------
# Nuclide-level groups
# ----------------------------------------------------------------------

def _totalNuGroup(total) -> R.Group:
    group = R.Group()
    group.setAttr("particle", "neutron")
    group.setAttr("emission_mode", "total")
    group["yield"] = encodeFunction(yieldForm(total.multiplicity))
    group.setAttr("n_distribution", 0)
    return group


def _fissionEnergyRelease(fission):
    data = fission.outputChannel.fissionFragmentData
    if data is None or not data.fissionEnergyReleases:
        return None
    for release in data.fissionEnergyReleases:
        if release.label == MONTE_CARLO_LABEL:
            return release
    return data.fissionEnergyReleases[0]


def _encodeFissionEnergy(release) -> R.Group:
    group = R.Group()
    for dname, term in FISSION_ENERGY_TERMS.items():
        function = getattr(release, term)
        if function is not None:
            group[dname] = encodeFunction(function)
    extra = _provenance(release, "q_prompt"), _provenance(release, "q_recoverable")
    if extra[0] is None or extra[1] is None:
        from .fission_energy import derivedQ

        extra = derivedQ(release)
    group["q_prompt"], group["q_recoverable"] = (encodeFunction(f) for f in extra)
    group.children = dict(sorted(group.children.items()))
    return group


def _urrTables(suite, temps) -> Dict[str, object]:
    out = {}
    for T in temps:
        label = urrLabel(T)
        for _, reaction in _allReactions(suite):
            form = reaction.crossSection.forms.get(label)
            if form is not None and getattr(form, "table", None) is not None:
                out[T] = form.table
                break
    if out and len(out) != len(temps):
        raise HDF5FormatError(f"URR tables for {sorted(out)} but not for "
                              f"{sorted(set(temps) - set(out))}; the C++ needs every temperature "
                              f"or none (nuclide.cpp:273-284)")
    return out


def _checkUrrInelastic(urr, written):
    values = {int(t.inelastic) for t in urr.values()}
    if len(values) != 1:
        raise HDF5FormatError(f"URR inelastic differs between temperatures {values}; the C++ "
                              f"requires one value (nuclide.cpp:298-306)")
    inelastic = values.pop()
    if inelastic > 0 and f"reaction_{inelastic:03d}" not in written:
        raise HDF5FormatError(f"URR tables name MT{inelastic} for the inelastic competition and "
                              f"it is not written; the C++ requires it (nuclide.cpp:308-320)")


def _encodeUrr(tables) -> R.Group:
    group = R.Group()
    group.setAttr("interpolation", int(tables.interpolation))
    group.setAttr("inelastic", int(tables.inelastic))
    group.setAttr("absorption", int(tables.absorption))
    group.setAttr("multiply_smooth", int(bool(tables.multiplySmooth)))
    group["energy"] = R.Dataset.of(np.asarray(tables.energies, dtype=float))
    group["table"] = R.Dataset.of(np.asarray(tables.table, dtype=float))
    return group
