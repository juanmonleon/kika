"""A thermal scattering evaluation → the processed suite the thermal encoder writes (H9b, H9c).

As OpenMC's ``ThermalScattering.from_njoy`` builds its official files
(``thermal.py:966-1038``, ``njoy.py:595-760``):

- the **inelastic** part comes from NJOY (roadmap D7): RECONR and BROADR on the
  principal nuclide's evaluation, THERMR twice (free gas, MT 221; bound, MT 222)
  and ACER per temperature (``iwt=2``: continuous outgoing energies with
  equiprobable cosines), read with :mod:`kika.hdf5.ace_thermal`. kika has no
  THERMR of its own yet (``openmc_hdf5_roadmap.md`` §2.1);
- the **elastic** part is ACER's for a coherent elastic, and the evaluation's own
  for an incoherent or mixed one (``use_endf_data``): σ_b and W'(T) of MF7/MT2
  (:class:`IncoherentElasticXS`), with the coherent half of a mixed one from
  the evaluation's Bragg edges too, at the temperatures the evaluation states.

The table's name (``c_H_in_H2O``) keys OpenMC's registry
(:mod:`kika.hdf5.thermal_names`) for the ACE name, the ZAIDs and ``nmix`` ACER
is given.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .exceptions import HDF5FormatError, MissingNJOYPieceError
from .names import EVAL_LABEL, heatedLabel, temperatureName

__all__ = ["processThermal", "thermalDeck"]


def _m():
    from kika.nuclear_data import model

    return model


def thermalDeck(*, mat: int, matThermal: int, temperatures: Sequence[float], aceName: str,
                zaids: Sequence[int], nmix: int, elastic: int, elasticType: int, natom: int,
                energyMax: float, library: str = "kika", nbin: int = 16, iwt: int = 2,
                error: float = 0.001) -> str:
    """The NJOY input of ``make_ace_thermal``, card for card."""
    temps = " ".join(str(t) for t in temperatures)
    mtElastic = 223 if elastic else 0
    lines = ["reconr", "20 22", f"'{library} PENDF'/", f"{mat} 2/",
             f"{error} 0. {10 * error} 5.0e-8/", "'kika'/", "'Processed by NJOY for kika'/", "0/",
             "broadr", "20 22 23", f"{mat} {len(temperatures)} 0 0 0. /", f"{error}/", temps, "0/",
             "thermr", "0 23 24", f"0 {mat} {nbin} {len(temperatures)} 1 0 0 1 221 1/", temps,
             f"{error} {energyMax}",
             "thermr", "21 24 25",
             f"{matThermal} {mat} {nbin} {len(temperatures)} 2 {elastic} 0 {natom} 222 1/", temps,
             f"{error} {energyMax}"]
    for i, temperature in enumerate(temperatures):
        nace = 26 + 2 * i
        lines += ["acer", f"20 25 0 {nace} {nace + 1}", f"2 0 1 .{i + 1:02d}/",
                  f"'{library}: thermal scattering processed by NJOY'/",
                  f"{mat} {temperature} '{aceName}' {len(zaids)} /",
                  " ".join(str(z) for z in zaids) + " /",
                  f"222 64 {mtElastic} {elasticType} {nmix} {energyMax} {iwt}/"]
    lines.append("stop")
    return "\n".join(lines) + "\n"


def _law(suite, kind):
    m = _m()
    for reaction in suite.reactions:
        ddcs = reaction.doubleDifferentialCrossSection
        if ddcs is None:
            continue
        for form in ddcs.forms.values():
            if isinstance(form, kind):
                return form
    return None


def _endfElastic(suite, T: float):
    """The evaluation's incoherent (or mixed) elastic at ``T`` as processed forms, or ``None``."""
    m = _m()
    coherent = _law(suite, m.CoherentElastic)
    incoherent = _law(suite, m.IncoherentElastic)
    if incoherent is None:
        return None
    temps = incoherent.temperatures
    if not np.any(np.isclose(temps, T, rtol=0, atol=0.05)):
        return None
    W = incoherent.debyeWaller(float(T))
    sigma = float(incoherent.boundAtomCrossSection.convertedTo("b").value)
    inc = (m.IncoherentElasticXS(boundCrossSection=sigma, debyeWaller=W), m.IncoherentElasticMC(debyeWaller=W))
    if coherent is None:
        return inc
    match = [t for t in coherent.temperatures if abs(t - T) < 0.05]
    if not match:
        return None
    coh = m.CoherentElasticXS(braggEdges=coherent.braggEdges(),
                              structureFactors=coherent.cumulativeS(match[0]))
    return (m.SumOfFunctions(functions=[coh, inc[0]]),
            m.MixedElasticMC(coherent=m.CoherentElasticMC(), incoherent=inc[1]))


def _evaluationTemperatures(suite) -> List[float]:
    m = _m()
    law = _law(suite, m.IncoherentInelastic)
    if law is None:
        raise HDF5FormatError("the thermal evaluation has no incoherent inelastic law")
    kernel = law.principal.selfScatteringKernel.kernel
    grid = kernel.grids[0]
    from kika.nuclear_data.model.units import conversion_factor

    return [float(t) * conversion_factor(grid.unit or "K", "K") for t in grid.values]


def processThermal(thermal, neutron, *, name: str, temperatures=None, njoy=None,
                   useEndfData: bool = True, report=None, workdir=None, keep: bool = False):
    """The processed thermal suite of the evaluation at ``thermal`` (path), its
    principal nuclide's evaluation at ``neutron`` (path).

    ``name`` is the OpenMC table name (``c_H_in_H2O``); ``temperatures`` in
    kelvin, by default every temperature the evaluation tabulates.
    """
    m = _m()
    import kika
    from .ace_thermal import nuclidesFromPairs, readAceTable, thermalForms
    from .njoy_pieces import _runCommand
    from .thermal_names import THERMAL_DATA

    report = report if report is not None else m.ConversionReport()
    if name not in THERMAL_DATA:
        raise HDF5FormatError(f"{name!r} is not in OpenMC's thermal registry; known: "
                              f"{sorted(THERMAL_DATA)[:5]}...")
    entry = THERMAL_DATA[name]
    tsl = kika.read(str(thermal), covariances=False)
    matThermal, mat = _matOf(thermal), _matOf(neutron)
    inelastic = _law(tsl, m.IncoherentInelastic)
    principal = inelastic.principal
    energyMax = float(principal.e_max.convertedTo("eV").value)
    natom = int(principal.numberPerMolecule)
    coherent, incoherent = _law(tsl, m.CoherentElastic), _law(tsl, m.IncoherentElastic)
    elastic = int(coherent is not None or incoherent is not None)
    lhtr = 3 if (coherent and incoherent) else (1 if coherent else (2 if incoherent else 0))
    elasticType = max(lhtr - 1, 0)
    temps = [float(t) for t in (temperatures or _evaluationTemperatures(tsl))]
    deck = thermalDeck(mat=mat, matThermal=matThermal, temperatures=temps, aceName=entry.name,
                       zaids=entry.zaids, nmix=entry.nmix, elastic=elastic, elasticType=elasticType,
                       natom=natom, energyMax=energyMax)

    base = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="kika_thermr_"))
    try:
        base.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(neutron, base / "tape20")
        shutil.copyfile(thermal, base / "tape21")
        (base / "input").write_text(deck)
        if isinstance(njoy, (list, tuple)):
            result = _runCommand(njoy, base, deck)
        else:
            from kika.njoy.run_njoy import _run_njoy_deck
            from kika.processing.njoy_pendf_cache import find_njoy_executable

            result = _run_njoy_deck(find_njoy_executable(njoy, why="thermal inelastic (D7)"), base, deck)
        (base / "output.txt").write_bytes(result.stdout or b"")
        suite = m.ReactionSuite(evaluation=tsl.evaluation or "", projectile="n", target=name,
                                interaction="thermalNeutronScatteringLaw")
        suite.PoPs.add(m.Particle(id="n"))
        suite.styles.add(m.Evaluated(label=EVAL_LABEL))
        reactions = {}
        awr = nuclides = maxE = None
        for i, T in enumerate(temps):
            acePath = base / f"tape{26 + 2 * i}"
            if not acePath.is_file():
                raise MissingNJOYPieceError(f"ACER wrote no table at {T} K; see {base / 'output.txt'}",
                                            ("thermal inelastic",))
            ace = readAceTable(acePath)
            inXs, inDist, elXs, elDist, energyMaxT = thermalForms(ace)
            kT = ace.kT * 1.0e6
            label = heatedLabel(temperatureName(kT))
            suite.styles.add(m.Heated(label=label, derivedFrom=EVAL_LABEL,
                                      temperature=m.PhysicalQuantity(value=kT, unit="eV/k")))
            if useEndfData:
                exact = _endfElastic(tsl, T)
                if exact is not None:
                    elXs, elDist = exact
            for kind, mt, xs, dist in (("elastic", 2, elXs, elDist), ("inelastic", 4, inXs, inDist)):
                if xs is None:
                    continue
                reaction = reactions.get(kind)
                if reaction is None:
                    product = m.Product(pid="n", label="n", distribution=m.Distribution())
                    channel = m.OutputChannel(process=f"thermalNeutronScatteringLaw {kind}")
                    channel.products.products.append(product)
                    reaction = reactions[kind] = m.Reaction(id=m.ReactionId(label=kind, ENDF_MT=mt),
                                                            outputChannel=channel)
                reaction.crossSection[label] = xs
                reaction.outputChannel.products[0].distribution[label] = dist
            if awr is None:
                awr, nuclides, maxE = ace.awr, nuclidesFromPairs(ace.pairs), energyMaxT
        for kind in ("elastic", "inelastic"):
            if kind in reactions:
                suite.reactions.append(reactions[kind])
        suite.provenance = m.Hdf5Provenance(attributes={
            "atomic_weight_ratio": float(awr), "energy_max": float(maxE), "nuclides": nuclides})
        report.warn(f"thermal inelastic from NJOY THERMR + ACER (roadmap D7)"
                    + ("; incoherent elastic from the evaluation (use_endf_data)" if useEndfData else ""))
        suite.report = report
        return suite
    finally:
        if not keep and workdir is None:
            shutil.rmtree(base, ignore_errors=True)


def _matOf(path) -> int:
    with open(path, errors="replace") as handle:
        for line in handle:
            if len(line) >= 75 and line[70:72].strip() == "1" and line[72:75].strip() == "451":
                return int(line[66:70])
    raise HDF5FormatError(f"no MF1/MT451 in {path}")
