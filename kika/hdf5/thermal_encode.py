"""A ``thermalNeutronScatteringLaw`` suite in processed form → OpenMC thermal HDF5.

The inverse of :mod:`kika.hdf5.thermal_decode`: per ``heated_<T>`` style, the
elastic and inelastic cross sections and distributions, laid out as
``ThermalScattering.export_to_hdf5`` lays them out (``thermal.py:611-643``).
The C++ requirements it keeps (``src/thermal.cpp``): a ``coherent_elastic``
distribution needs a ``CoherentElastic`` cross section, ``mixed_elastic`` a
``Sum`` whose first function is ``CoherentElastic``, the discrete laws a
``Tabulated1D``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import records as R
from .distributions import encodeCorrelated
from .exceptions import HDF5FormatError, HDF5UnsupportedError
from .functions import encodeFunction
from .names import HDF5_VERSION, heatedLabel
from .neutron_encode import heatedTemperatures
from .thermal_decode import THERMAL_MT

__all__ = ["encodeThermal", "writeThermal"]


def _m():
    from kika.nuclear_data import model

    return model


def _xs(form):
    m = _m()
    if isinstance(form, m.CoherentElasticXS):
        dataset = R.Dataset.of(np.vstack([form.braggEdges, form.structureFactors]))
        dataset.attrs["type"] = R.Attribute.of("CoherentElastic")
        return dataset
    if isinstance(form, m.IncoherentElasticXS):
        dataset = R.Dataset.of(np.array([form.boundCrossSection, form.debyeWaller], dtype=float))
        dataset.attrs["type"] = R.Attribute.of("IncoherentElastic")
        return dataset
    if isinstance(form, m.SumOfFunctions):
        group = R.Group()
        group.setAttr("n", len(form.functions))
        group.setAttr("type", "Sum")
        for i, function in enumerate(form.functions, start=1):
            group[f"func_{i}"] = _xs(function)
        return group
    return encodeFunction(form)


def _coherentPart(xs):
    m = _m()
    if isinstance(xs, m.CoherentElasticXS):
        return xs
    if isinstance(xs, m.SumOfFunctions) and xs.functions and isinstance(xs.functions[0], m.CoherentElasticXS):
        return xs.functions[0]
    raise HDF5FormatError("a coherent elastic distribution needs a CoherentElastic cross section "
                          "(thermal.cpp: the C++ requires it)")


def _distribution(form, xs):
    m = _m()
    group = R.Group()
    if isinstance(form, m.CoherentElasticMC):
        group.setAttr("type", "coherent_elastic")
        group["coherent_xs"] = _xs(_coherentPart(xs))
        return group
    if isinstance(form, m.IncoherentElasticMC):
        group.setAttr("type", "incoherent_elastic")
        group["debye_waller"] = R.Dataset.of(np.float64(form.debyeWaller))
        return group
    if isinstance(form, m.IncoherentElasticDiscreteMC):
        group.setAttr("type", "incoherent_elastic_discrete")
        group["mu_out"] = R.Dataset.of(np.asarray(form.muOut, dtype=float))
        return group
    if isinstance(form, m.IncoherentInelasticDiscreteMC):
        group.setAttr("type", "incoherent_inelastic_discrete")
        group["energy_out"] = R.Dataset.of(np.asarray(form.energyOut, dtype=float))
        group["mu_out"] = R.Dataset.of(np.asarray(form.muOut, dtype=float))
        group["skewed"] = R.Dataset.of(np.bool_(form.skewed))
        return group
    if isinstance(form, m.MixedElasticMC):
        group.setAttr("type", "mixed_elastic")
        group["coherent"] = _distribution(form.coherent, xs)
        group["incoherent"] = _distribution(form.incoherent, xs)
        return group
    if isinstance(form, m.EnergyAngularMC):
        return encodeCorrelated(form, typeName="incoherent_inelastic")
    raise HDF5UnsupportedError(f"a {type(form).__name__} is not a thermal distribution OpenMC reads")


def encodeThermal(suite, temperatures=None) -> R.Group:
    m = _m()
    provenance = suite.provenance if isinstance(suite.provenance, m.Hdf5Provenance) else None
    attributes = provenance.attributes if provenance is not None else {}
    temps = heatedTemperatures(suite, temperatures)
    root = R.Group()
    root.setAttr("filetype", "data_thermal")
    root.setAttr("version", np.array(HDF5_VERSION, dtype=np.int64))
    g = root.require(suite.target)
    for key in ("atomic_weight_ratio", "energy_max"):
        if key not in attributes:
            raise HDF5FormatError(f"a thermal table needs {key}; give the suite an Hdf5Provenance "
                                  f"with it (the writer from an evaluation does)")
        g.setAttr(key, float(attributes[key]))
    nuclides = np.array(attributes.get("nuclides", []), dtype=attributes.get("nuclides_dtype", "S"))
    g.attrs["nuclides"] = R.Attribute(value=nuclides, dtype=nuclides.dtype, shape=nuclides.shape)
    for T, kT in temps.items():
        g.require("kTs")[T] = R.Dataset.of(np.float64(kT))
    for T in temps:
        label = heatedLabel(T)
        for kind, mt in THERMAL_MT.items():
            reaction = suite.findReactionByENDF_MT(mt)
            if reaction is None or label not in reaction.crossSection.forms:
                continue
            xs = reaction.crossSection.forms[label]
            product = reaction.outputChannel.products[0]
            block = g.require(f"{T}/{kind}")
            block["xs"] = _xs(xs)
            block["distribution"] = _distribution(product.distribution.forms[label], xs)
    g.children = dict(sorted(g.children.items()))
    return root


def writeThermal(suite, directory, temperatures=None, *, report=None, index=True, fileName=None):
    from .thermal_decode import decodeThermal
    from .writer import indexFile

    directory = Path(directory)
    tree = encodeThermal(suite, temperatures)
    target = directory / (fileName or f"{suite.target}.h5")
    temporary = target.with_name(f".{target.name}.tmp")
    R.writeFile(tree, temporary)
    try:
        written = R.readFile(temporary)
        differences = R.compareTrees(tree, written, limit=10)
        if differences:
            raise HDF5FormatError(f"read back differs: {differences}", path=temporary)
        decodeThermal(written, path=temporary)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, target)
    if index:
        indexFile(directory, target, suite.target, "thermal")
    return report if report is not None else _m().ConversionReport()
