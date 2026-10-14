"""OpenMC thermal-scattering HDF5 → a ``thermalNeutronScatteringLaw`` suite.

A processed thermal file states, per temperature, the elastic and inelastic
cross sections and how to sample the outgoing neutron. The suite read from it:

- has ``interaction = "thermalNeutronScatteringLaw"`` and ``target`` the table
  name (``c_H_in_H2O``); the ``eval`` style is declared and empty, and each
  temperature is a ``heated_<T>`` style carrying the file's kT;
- holds an elastic reaction (MT 2) when the file has elastic data and an
  inelastic one (MT 4), each with its cross section and its outgoing neutron's
  distribution under the ``heated_<T>`` label of every temperature -- per
  temperature because, unlike a neutron file, here the distributions change
  with temperature;
- keeps ``atomic_weight_ratio``, ``energy_max`` and ``nuclides`` (the
  materials the table applies to) in its :class:`Hdf5Provenance`.

Cross sections: ``Tabulated1D`` → ``XYs1d``; ``CoherentElastic`` →
:class:`CoherentElasticXS`; ``IncoherentElastic`` → :class:`IncoherentElasticXS`;
``Sum`` → :class:`SumOfFunctions`. Distributions: ``coherent_elastic``,
``incoherent_elastic``, ``incoherent_elastic_discrete``,
``incoherent_inelastic_discrete`` and ``mixed_elastic`` to the processed
thermal forms of :mod:`kika.nuclear_data.model.processed_forms`, and
``incoherent_inelastic`` -- the ``correlated`` layout -- to
:class:`EnergyAngularMC`.
"""
from __future__ import annotations

import numpy as np

from . import records as R
from .distributions import decodeCorrelated
from .exceptions import HDF5FormatError, HDF5UnsupportedError
from .functions import decodeFunction
from .names import EVAL_LABEL, HDF5_VERSION, heatedLabel

__all__ = ["decodeThermal", "readThermal", "THERMAL_MT"]

#: The reactions a thermal suite holds, by OpenMC's group name.
THERMAL_MT = {"elastic": 2, "inelastic": 4}


def _m():
    from kika.nuclear_data import model

    return model


def readThermal(path):
    with R.openFile(path) as tree:
        return decodeThermal(tree, path=path)


def _xs(node, where):
    m = _m()
    if isinstance(node, R.Group):
        if R.text(R.attr(node, "type", b"")) != "Sum":
            raise HDF5UnsupportedError(f"cross-section group of type "
                                       f"{R.text(R.attr(node, 'type', b''))!r} [{where}]")
        count = int(R.attr(node, "n"))
        return m.SumOfFunctions(functions=[_xs(node[f"func_{i}"], f"{where}/func_{i}")
                                           for i in range(1, count + 1)])
    kind = R.text(R.attr(node, "type", b""))
    data = node.data
    if kind == "CoherentElastic":
        return m.CoherentElasticXS(braggEdges=data[0], structureFactors=data[1])
    if kind == "IncoherentElastic":
        return m.IncoherentElasticXS(boundCrossSection=float(data[0]), debyeWaller=float(data[1]))
    return decodeFunction(node, where)


def _distribution(group, where, label):
    m = _m()
    kind = R.text(R.attr(group, "type"))
    if kind == "coherent_elastic":
        return m.CoherentElasticMC()
    if kind == "incoherent_elastic":
        return m.IncoherentElasticMC(debyeWaller=float(group["debye_waller"].data))
    if kind == "incoherent_elastic_discrete":
        return m.IncoherentElasticDiscreteMC(muOut=group["mu_out"].data)
    if kind == "incoherent_inelastic_discrete":
        return m.IncoherentInelasticDiscreteMC(energyOut=group["energy_out"].data,
                                               muOut=group["mu_out"].data,
                                               skewed=bool(group["skewed"].data))
    if kind == "incoherent_inelastic":
        return decodeCorrelated(group, m.Frame.lab, where, label=label)
    if kind == "mixed_elastic":
        return m.MixedElasticMC(coherent=_distribution(group["coherent"], f"{where}/coherent", label),
                                incoherent=_distribution(group["incoherent"], f"{where}/incoherent",
                                                         label))
    raise HDF5UnsupportedError(f"thermal distribution type {kind!r} [{where}]")


def decodeThermal(tree, path=None, report=None):
    """The record tree of one thermal file → ``(suite, report)``."""
    m = _m()
    report = report if report is not None else m.ConversionReport()
    if R.text(R.attr(tree, "filetype", b"")) != "data_thermal":
        raise HDF5FormatError("filetype is not 'data_thermal'", path=path)
    if int(np.asarray(R.attr(tree, "version"))[0]) != HDF5_VERSION[0]:
        raise HDF5FormatError("unsupported major version", path=path)
    names = [n for n, _ in tree.groups()]
    if len(names) != 1:
        raise HDF5FormatError(f"{len(names)} top-level groups; a thermal file holds one table",
                              path=path)
    name = names[0]
    g = tree[name]
    nuclides = [R.text(n) for n in np.atleast_1d(R.attr(g, "nuclides"))]
    nuclidesDtype = R.Attribute.of(np.array(nuclides, dtype="S")).dtype
    suite = m.ReactionSuite(evaluation="", projectile="n", target=name,
                            interaction="thermalNeutronScatteringLaw")
    suite.PoPs.add(m.Particle(id="n"))
    suite.provenance = m.Hdf5Provenance(path=str(path) if path else None, node=f"/{name}", attributes={
        "filetype": "data_thermal", "atomic_weight_ratio": float(R.attr(g, "atomic_weight_ratio")),
        "energy_max": float(R.attr(g, "energy_max")), "nuclides": nuclides,
        "nuclides_dtype": str(g.attrs["nuclides"].dtype)})
    suite.styles.add(m.Evaluated(label=EVAL_LABEL))
    temperatures = sorted(g["kTs"].children)
    for T in temperatures:
        suite.styles.add(m.Heated(label=heatedLabel(T), derivedFrom=EVAL_LABEL,
                                  temperature=m.PhysicalQuantity(value=float(g[f"kTs/{T}"].data),
                                                                 unit="eV/k")))
    reactions = {}
    for T in temperatures:
        if T not in g.children:
            raise HDF5FormatError(f"kTs names {T} and there is no {T} group", path=path)
        label = heatedLabel(T)
        for kind, mt in THERMAL_MT.items():
            if kind not in g[T].children:
                continue
            where = f"/{name}/{T}/{kind}"
            reaction = reactions.get(kind)
            if reaction is None:
                product = m.Product(pid="n", label="n", distribution=m.Distribution())
                channel = m.OutputChannel(process=f"thermalNeutronScatteringLaw {kind}")
                channel.products.products.append(product)
                reaction = reactions[kind] = m.Reaction(
                    id=m.ReactionId(label=kind, ENDF_MT=mt), outputChannel=channel,
                    provenance=m.Hdf5Provenance(node=f"/{name}/<T>/{kind}"))
            block = g[f"{T}/{kind}"]
            reaction.crossSection[label] = _xs(block["xs"], f"{where}/xs")
            reaction.outputChannel.products[0].distribution[label] = _distribution(
                block["distribution"], f"{where}/distribution", label)
    for kind in THERMAL_MT:
        if kind in reactions:
            suite.reactions.append(reactions[kind])
    report.lost("a processed thermal file holds no S(alpha, beta): the law it was made from is "
                "the evaluation it refers to, and the 'eval' style is declared empty")
    suite.report = report
    return suite, report
