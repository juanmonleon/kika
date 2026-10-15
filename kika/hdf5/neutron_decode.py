"""OpenMC incident-neutron HDF5 → :class:`~kika.nuclear_data.model.suite.ReactionSuite`.

**A processed file, and the model says so**, as the ACE decoder
does (:mod:`kika.ace.model_adapter.decode`):

- the ``eval`` style is declared and empty: the file refers to an evaluation it
  does not contain;
- each temperature group is a ``heated`` style, ``heated_<T>`` (``heated_294K``),
  whose ``temperature`` is the file's kT in ``eV/k`` -- the value in
  ``kTs/<T>``, kept exactly, since the C++ names its groups from it;
- the 0 K grid and the elastic's ``0K/xs`` are the ``recon`` style
  (``crossSectionReconstructed``);
- every distribution and yield is under ``MonteCarlo`` (``MonteCarlo_cdf``):
  tables with their cumulative, as OpenMC samples them;
- the URR probability tables of a temperature are ``URR_<T>``
  (``URR_probabilityTables``), the tables themselves inside the
  :class:`URR_probabilityTables1d` form of the reactions they apply to.

**Where the redundant reactions go.** The C++ rebuilds the total from the
reactions with ``redundant = 0`` (``nuclide.cpp:401-424``); those are
``suite.reactions``. A redundant reaction that sums partials -- MT 3, 4, 16,
103-107, or 18 when the chances are given -- is a :class:`CrossSectionSum` in
``suite.sums``, its summands the partials the file carries. The gas production
203-207, the KERMA 301 and 901 and the damage energy 444 are not reactions:
they are response functions of the processed library, which GNDS has no node
for (FUDGE keeps the energy deposit as ``averageProductData`` per product). They
go to ``suite.productions``, where ``suite.reactionByENDF_MT`` finds them.

**Fission** follows OpenMC's contract (``nuclide.cpp:529-570``): ``product_0``
is the prompt (or total) neutron and becomes the fission channel's neutron
product; each ``delayed`` neutron becomes a :class:`DelayedNeutron` with its
decay rate in 1/s; ``total_nu`` becomes the total-ν̄ ``multiplicitySum``; and
``fission_energy_release`` a :class:`FissionEnergyRelease`, whose two OpenMC
derived terms ``q_prompt`` and ``q_recoverable`` are kept in its provenance.

What the file states and the model has no slot for -- a reaction's ``label`` and
``center_of_mass``, a product's ``emission_mode`` -- is kept on
:class:`Hdf5Provenance` so the encoder writes it back unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from kika._constants import mf3SumComponents

from . import records as R
from .distributions import decodeDistribution
from .exceptions import HDF5FormatError
from .functions import decodeFunction
from .names import (EVAL_LABEL, FISSION_MTS, HDF5_VERSION, MONTE_CARLO_LABEL, RECON_LABEL,
                    heatedLabel, urrLabel)

__all__ = ["decodeNeutron", "readNeutron", "PARTICLE_TO_PID", "PID_TO_PARTICLE",
           "RESPONSE_MTS", "FISSION_ENERGY_TERMS", "TOTAL_NUBAR_LABEL"]

#: OpenMC's particle names → GNDS particle ids. Anything else is already a GNDS
#: nuclide name (``He4``) and is used as it is.
PARTICLE_TO_PID = {"neutron": "n", "photon": "photon", "electron": "e-", "positron": "e+"}
PID_TO_PARTICLE = {pid: name for name, pid in PARTICLE_TO_PID.items()}

#: Redundant MTs that are response functions, not reactions.
RESPONSE_MTS = (203, 204, 205, 206, 207, *range(301, 448), 901)

#: ``fission_energy_release/<dataset>`` → FissionEnergyRelease term.
FISSION_ENERGY_TERMS = {
    "fragments": "promptProductKE",
    "prompt_neutrons": "promptNeutronKE",
    "delayed_neutrons": "delayedNeutronKE",
    "prompt_photons": "promptGammaEnergy",
    "delayed_photons": "delayedGammaEnergy",
    "betas": "delayedBetaEnergy",
    "neutrinos": "neutrinoEnergy",
}

#: As FUDGE and the ENDF adapter spell it.
TOTAL_NUBAR_LABEL = "total fission neutron multiplicity"


def _m():
    from kika.nuclear_data import model

    return model


def readNeutron(path) -> Tuple[object, object]:
    """Read a ``.h5`` file into a suite. Returns ``(suite, report)``."""
    tree = R.readFile(path)
    return decodeNeutron(tree, path=path)


def _checkHeader(tree, path) -> str:
    filetype = R.text(R.attr(tree, "filetype", b""))
    if filetype != "data_neutron":
        raise HDF5FormatError(f"filetype is {filetype!r}, not 'data_neutron'", path=path)
    version = R.attr(tree, "version")
    if version is None:
        raise HDF5FormatError("no root 'version' attribute (the C++ requires it)", path=path)
    if int(np.asarray(version)[0]) != HDF5_VERSION[0]:
        raise HDF5FormatError(f"major version {int(np.asarray(version)[0])}, kika reads "
                              f"{HDF5_VERSION[0]}", path=path)
    groups = [name for name, _ in tree.groups()]
    if len(groups) != 1:
        raise HDF5FormatError(f"{len(groups)} top-level groups {groups}; a neutron file holds one "
                              f"nuclide", path=path)
    return groups[0]


def decodeNeutron(tree, path=None, report=None):
    """The record tree of one incident-neutron file → ``(suite, report)``."""
    m = _m()
    from kika._constants import NEUTRON_MASS_AMU

    report = report if report is not None else m.ConversionReport()
    name = _checkHeader(tree, path)
    g = tree[name]
    awr = float(R.attr(g, "atomic_weight_ratio"))
    Z, A, metastable = (int(R.attr(g, k)) for k in ("Z", "A", "metastable"))

    suite = m.ReactionSuite(evaluation="", projectile="n", target=name)
    suite.PoPs.add(m.Nuclide(id=name, Z=Z, A=A,
                             mass=m.PhysicalQuantity(value=awr * NEUTRON_MASS_AMU, unit="amu")))
    suite.PoPs.add(m.Particle(id="n"))
    suite.provenance = m.Hdf5Provenance(path=str(path) if path else None, node=f"/{name}",
                                        attributes={"filetype": "data_neutron",
                                                    "version": [int(v) for v in R.attr(tree, "version")],
                                                    "Z": Z, "A": A, "metastable": metastable,
                                                    "atomic_weight_ratio": awr})

    temperatures = sorted(g["kTs"].children)
    kTs = {T: float(g[f"kTs/{T}"].data) for T in temperatures}
    suite.styles.add(m.Evaluated(label=EVAL_LABEL))
    for T in temperatures:
        suite.styles.add(m.Heated(label=heatedLabel(T), derivedFrom=EVAL_LABEL,
                                  temperature=m.PhysicalQuantity(value=kTs[T], unit="eV/k")))
    grids = {T: g[f"energy/{T}"].data for T in temperatures}
    if "0K" in g["energy"].children:
        suite.styles.add(m.CrossSectionReconstructed(label=RECON_LABEL, derivedFrom=EVAL_LABEL))
        grids["0K"] = g["energy/0K"].data
    suite.styles.add(m.MonteCarlo_cdf(label=MONTE_CARLO_LABEL, derivedFrom=EVAL_LABEL))
    report.lost("an OpenMC HDF5 file refers to an evaluation it does not contain, so the "
                "'eval' style is declared with no data behind it")

    decoded = []
    for rname, rgroup in g["reactions"].groups():
        decoded.append(_decodeReaction(rgroup, f"/{name}/reactions/{rname}", grids, awr, report))
    _place(suite, decoded)

    if "total_nu" in g.children:
        _totalNu(suite, g["total_nu"], f"/{name}/total_nu")
    if "fission_energy_release" in g.children:
        _fissionEnergy(suite, g["fission_energy_release"], f"/{name}/fission_energy_release", report)
    if "urr" in g.children:
        _urr(suite, g["urr"], temperatures, report)

    suite.report = report
    return suite, report


# ----------------------------------------------------------------------
# Reactions
# ----------------------------------------------------------------------

def _frame(centerOfMass: bool):
    m = _m()
    return m.Frame.centerOfMass if centerOfMass else m.Frame.lab


def _decodeReaction(group, where, grids, awr, report):
    m = _m()
    mt = int(R.attr(group, "mt"))
    centerOfMass = bool(int(R.attr(group, "center_of_mass")))
    redundant = bool(int(R.attr(group, "redundant", 0)))
    crossSection = m.CrossSection()
    for T, child in group.groups():
        if T.startswith("product_"):
            continue
        if T not in grids:
            raise HDF5FormatError(f"temperature group {T} has no energy grid", node=where)
        xs = child["xs"]
        threshold = int(R.attr(xs, "threshold_idx"))
        grid = grids[T]
        values = xs.data
        if threshold + len(values) != len(grid):
            raise HDF5FormatError(f"{T}: threshold_idx {threshold} + {len(values)} values != "
                                  f"{len(grid)} grid points", node=where)
        label = RECON_LABEL if T == "0K" else heatedLabel(T)
        crossSection[label] = m.XYs1d(xs=grid[threshold:], ys=values,
                                      interpolation=m.Interpolation.linlin,
                                      axes=m.crossSectionAxes(), label=label)

    products, delayed = [], []
    productNames = sorted((k for k in group.children if k.startswith("product_")),
                          key=lambda k: int(k.split("_")[1]))
    for pname in productNames:
        product, mode, rate = _decodeProduct(group[pname], f"{where}/{pname}", _frame(centerOfMass))
        if mode == "delayed":
            delayed.append((product, rate))
        else:
            products.append(product)

    channel = m.OutputChannel(Q=m.Q(value=float(R.attr(group, "Q_value")), unit="eV"))
    for product in products:
        channel.products.products.append(product)
    if delayed:
        channel.fissionFragmentData = m.FissionFragmentData()
        for index, (product, rate) in enumerate(delayed, start=1):
            channel.fissionFragmentData.delayedNeutrons.append(m.DelayedNeutron(
                label=str(index), rate=m.PhysicalQuantity(value=rate, unit="1/s"), product=product))
    label = R.text(R.attr(group, "label"))
    reactionId = m.ReactionId(label=label, ENDF_MT=mt,
                              fissionGenre="total" if mt in FISSION_MTS else None)
    provenance = m.Hdf5Provenance(node=where, attributes={
        "label": label, "center_of_mass": int(centerOfMass), "redundant": int(redundant)})
    return mt, redundant, (reactionId, crossSection, channel, provenance)


def _place(suite, decoded):
    """Each reaction into the container its role puts it in."""
    m = _m()
    present = {mt for mt, _, _ in decoded}
    for mt, redundant, (reactionId, crossSection, channel, provenance) in decoded:
        if not redundant:
            reaction = m.Reaction(id=reactionId, crossSection=crossSection, outputChannel=channel,
                                  provenance=provenance)
            container = suite.fissionComponents if (mt in (19, 20, 21, 38) and 18 in present) \
                else suite.reactions
            container.append(reaction)
            continue
        if mt in RESPONSE_MTS:
            suite.productions.append(m.Reaction(id=reactionId, crossSection=crossSection,
                                                outputChannel=channel, provenance=provenance))
            continue
        reaction = m.CrossSectionSum(id=reactionId, crossSection=crossSection,
                                     outputChannel=channel, provenance=provenance)
        for component in _components(mt, present):
            label = next(rid.label for c, _, (rid, *_rest) in decoded if c == component)
            reaction.summands.append(m.Add(
                href=f"/reactionSuite/reactions/reaction[@label='{label}']/crossSection"))
        suite.sums.append(reaction)


def _components(mt, present):
    """The MTs a sum adds up, as far as the file carries them."""
    return mf3SumComponents(mt, present)


# ----------------------------------------------------------------------
# Products
# ----------------------------------------------------------------------

def _decodeProduct(group, where, frame):
    m = _m()
    particle = R.text(R.attr(group, "particle"))
    pid = PARTICLE_TO_PID.get(particle, particle)
    mode = R.text(R.attr(group, "emission_mode"))
    rate = R.attr(group, "decay_rate")
    product = m.Product(pid=pid, label=pid)
    product.multiplicity = m.Multiplicity(form=decodeFunction(group["yield"], f"{where}/yield",
                                                              label=MONTE_CARLO_LABEL))
    count = int(R.attr(group, "n_distribution"))
    nbody: Dict[str, object] = {}
    weighted, anyApplicability = [], False
    for k in range(count):
        dgroup = group[f"distribution_{k}"]
        form = decodeDistribution(dgroup, frame, f"{where}/distribution_{k}", nbody=nbody)
        applicability = None
        if "applicability" in dgroup.children:
            anyApplicability = True
            applicability = decodeFunction(dgroup["applicability"], f"{where}/distribution_{k}")
        weighted.append(m.Weighted(weight=applicability, functional=form))
    if weighted:
        product.distribution = m.Distribution()
        if count == 1 and not anyApplicability:
            product.distribution[MONTE_CARLO_LABEL] = weighted[0].functional
        else:
            product.distribution[MONTE_CARLO_LABEL] = m.DistributionMixture(
                weighted=weighted, label=MONTE_CARLO_LABEL)
    attributes = {"emission_mode": mode, "particle": particle}
    if nbody:
        attributes["nbody"] = nbody
    product.provenance = m.Hdf5Provenance(node=where, attributes=attributes)
    return product, mode, (float(rate) if rate is not None else None)


# ----------------------------------------------------------------------
# Nuclide-level groups
# ----------------------------------------------------------------------

def _fissionReaction(suite):
    for mt in FISSION_MTS:
        reaction = suite.findReactionByENDF_MT(mt)
        if reaction is not None:
            return reaction
    return None


def _totalNu(suite, group, where):
    m = _m()
    yield_ = decodeFunction(group["yield"], f"{where}/yield", label=MONTE_CARLO_LABEL)
    total = m.MultiplicitySum(label=TOTAL_NUBAR_LABEL, multiplicity=m.Multiplicity(form=yield_),
                              ENDF_MT=452)
    fission = _fissionReaction(suite)
    if fission is not None:
        total.summands.append(m.Add(
            href=f"/reactionSuite/reactions/reaction[@label='{fission.label}']"
                 f"/outputChannel/products/product[@label='n']/multiplicity"))
    suite.sums.multiplicitySums.append(total)


def _fissionEnergy(suite, group, where, report):
    m = _m()
    fission = _fissionReaction(suite)
    if fission is None:
        report.lost(f"{where}: fission_energy_release in a file with no fission reaction; not read")
        return
    release = m.FissionEnergyRelease(label=MONTE_CARLO_LABEL)
    extra = {}
    for dname, dataset in group.datasets():
        function = decodeFunction(dataset, f"{where}/{dname}")
        if dname in FISSION_ENERGY_TERMS:
            setattr(release, FISSION_ENERGY_TERMS[dname], function)
        else:
            extra[dname] = function
    release.provenance = m.Hdf5Provenance(node=where, attributes=extra)
    channel = fission.outputChannel
    if channel.fissionFragmentData is None:
        channel.fissionFragmentData = m.FissionFragmentData()
    channel.fissionFragmentData.fissionEnergyReleases.append(release)


def _urr(suite, group, temperatures, report):
    m = _m()
    missing = [T for T in temperatures if T not in group.children]
    if missing:
        raise HDF5FormatError(f"urr/ has no group for {missing}; the C++ needs one per temperature")
    for T in temperatures:
        tg = group[T]
        tables = m.ProbabilityTables(
            energies=tg["energy"].data, table=tg["table"].data,
            interpolation=int(R.attr(tg, "interpolation")),
            inelastic=int(R.attr(tg, "inelastic")), absorption=int(R.attr(tg, "absorption")),
            multiplySmooth=bool(int(R.attr(tg, "multiply_smooth"))))
        label = urrLabel(T)
        suite.styles.add(m.URR_probabilityTables(label=label, derivedFrom=heatedLabel(T)))
        for mt in (2, 18, 102):
            reaction = suite.findReactionByENDF_MT(mt)
            if reaction is not None:
                reaction.crossSection[label] = m.URR_probabilityTables1d(label=label, table=tables)
