"""A thermal scattering evaluation → the processed suite the thermal encoder writes (H9b, H9c).

As OpenMC's ``ThermalScattering.from_njoy`` builds its official files
(``thermal.py:966-1038``, ``njoy.py:595-760``), without NJOY:

- the **inelastic** part by kika's THERMR + ACER
  (:func:`kika.processing.thermal.inelasticTables`): continuous outgoing
  energies with 16 equiprobable cosines (ACER ``iwt=2``), on THERMR's incident
  grid, at each temperature the evaluation tabulates;
- the **elastic** part from the evaluation (``use_endf_data``): σ_b and W'(T)
  of MF7/MT2 for an incoherent one (:class:`IncoherentElasticXS`), the Bragg
  edges and the cumulative structure factors at the temperature for a
  coherent one, both for a mixed one.

The table's name (``c_H_in_H2O``) keys OpenMC's registry
(:mod:`kika.hdf5.thermal_names`) for the ZAIDs and ``nmix``.

With ``ThermalInelasticOptions(iinc=1)`` the table is THERMR's **free gas** of
a nuclide instead (:func:`kika.processing.thermal.freeGasTables`): the input is
the nuclide's neutron tape, the cross section its elastic broadened by kika to
each temperature (as THERMR copies BROADR's MT2), no elastic part, and the
name is the caller's (it is not in OpenMC's registry).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .exceptions import HDF5FormatError
from .names import EVAL_LABEL, heatedLabel, temperatureName

__all__ = ["processThermal"]


def _m():
    from kika.nuclear_data import model

    return model


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


def _evaluationTemperatures(suite) -> List[float]:
    m = _m()
    law = _law(suite, m.IncoherentInelastic)
    if law is None:
        raise HDF5FormatError("the thermal evaluation has no incoherent inelastic law")
    kernel = law.principal.selfScatteringKernel.kernel
    grid = kernel.grids[0]
    from kika.nuclear_data.model.units import conversion_factor

    return [float(t) * conversion_factor(grid.unit or "K", "K") for t in grid.values]


def processThermal(thermal, neutron=None, *, name: str, temperatures=None,
                   useEndfData: bool = True, report=None, options=None):
    """The processed thermal suite of the evaluation at ``thermal`` (path).

    ``name`` is the OpenMC table name (``c_H_in_H2O``); ``temperatures`` in
    kelvin, by default every temperature the evaluation tabulates. The
    incoherent inelastic tables are kika's (:func:`kika.processing.thermal.inelasticTables`,
    THERMR + ACER, roadmap H9c); the elastic comes from the evaluation, as
    OpenMC's ``from_njoy`` takes it with ``use_endf_data``: incoherent from
    σ_b and the Debye-Waller integral, coherent from the Bragg edges and the
    cumulative structure factors at the temperature. ``neutron`` (the
    principal nuclide's tape) is no longer read; it is accepted so a caller of
    the NJOY-era signature keeps working.
    """
    m = _m()
    import kika
    from kika.processing.thermal import ThermalInelasticOptions, inelasticTables
    from .ace_thermal import nuclidesFromPairs
    from .processing import aceKT
    from .thermal_names import THERMAL_DATA

    report = report if report is not None else m.ConversionReport()
    if options is not None and options.iinc == 1:
        return _processFreeGas(thermal, name=name, temperatures=temperatures, report=report,
                               options=options)
    if name not in THERMAL_DATA:
        raise HDF5FormatError(f"{name!r} is not in OpenMC's thermal registry; known: "
                              f"{sorted(THERMAL_DATA)[:5]}...")
    entry = THERMAL_DATA[name]
    tsl = kika.read(str(thermal), covariances=False)
    # The low-level ENDF reader can report a malformed MT2 and continue with
    # MT4. A valid inelastic kernel must not hide the missing elastic physics.
    if (_declaresElastic(thermal) and _law(tsl, m.CoherentElastic) is None
            and _law(tsl, m.IncoherentElastic) is None):
        raise HDF5FormatError("the evaluation declares MF7/MT2 but its elastic law "
                              "was not decoded; refusing an incomplete thermal file")
    inelastic = _law(tsl, m.IncoherentInelastic)
    if inelastic is None:
        issues = getattr(getattr(tsl, "report", None), "unsupported", [])
        detail = "; decoder: " + "; ".join(issues[:3]) if issues else ""
        raise HDF5FormatError("the thermal evaluation has no incoherent inelastic law" + detail)
    temps = [float(t) for t in (temperatures or _evaluationTemperatures(tsl))]
    awr = _principalAwr(tsl, inelastic)
    # The table's AWR is ACER's: the principal nuclide's, read from its neutron
    # tape (graphite 11.89365, Zr 89.1324), where the scattering law's B(3)
    # differs (11.898, 90.436). THERMR computes with B(3).
    headerAwr = _tapeAwr(neutron) if neutron is not None else awr
    suite = m.ReactionSuite(evaluation=tsl.evaluation or "", projectile="n", target=name,
                            interaction="thermalNeutronScatteringLaw")
    suite.PoPs.add(m.Particle(id="n"))
    suite.styles.add(m.Evaluated(label=EVAL_LABEL))
    reactions = {}
    energyMax = None
    for T in temps:
        kT = aceKT(T)
        label = heatedLabel(temperatureName(kT))
        suite.styles.add(m.Heated(label=label, derivedFrom=EVAL_LABEL,
                                  temperature=m.PhysicalQuantity(value=kT, unit="eV/k")))
        inXs, inDist, energyMaxT = inelasticTables(inelastic, T, nmix=entry.nmix, awr=awr,
                                                   options=options or ThermalInelasticOptions())
        elXs, elDist = _evaluatedElastic(tsl, T, report, entry.nmix)
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
        if energyMax is None:
            energyMax = energyMaxT
    for kind in ("elastic", "inelastic"):
        if kind in reactions:
            suite.reactions.append(reactions[kind])
    suite.provenance = m.Hdf5Provenance(attributes={
        # The ACE header's AWR, written with six decimals (ACER's f12.6).
        "atomic_weight_ratio": round(float(headerAwr), 6), "energy_max": float(energyMax),
        "nuclides": nuclidesFromPairs([(z, 0.0) for z in entry.zaids])})
    report.warn("thermal inelastic by kika (THERMR's calcem and ACER's iwt=2, roadmap H9c); "
                "elastic from the evaluation (use_endf_data)")
    suite.report = report
    return suite


def _processFreeGas(path, *, name: str, temperatures, report, options):
    """THERMR ``iinc=1`` + ACER for the nuclide on the neutron tape ``path``."""
    m = _m()
    from kika.endf.process import ProcessingPlan, process_endf
    from kika.processing.thermal import freeGasTables
    from .ace_thermal import nuclidesFromPairs
    from .processing import aceKT

    if not temperatures:
        raise HDF5FormatError("a free-gas table needs its temperatures: there is no scattering "
                              "law on a neutron tape to take them from")
    temps = [float(t) for t in temperatures]
    awr = _tapeAwr(path)
    za = _tapeZa(path)
    processed = process_endf(str(path), ProcessingPlan(temperatures=tuple(temps), urr=False,
                                                       workers=options.workers))
    emax = options.emax if options.emax is not None else 10.0
    suite = m.ReactionSuite(evaluation="", projectile="n", target=name,
                            interaction="thermalNeutronScatteringLaw")
    suite.PoPs.add(m.Particle(id="n"))
    suite.styles.add(m.Evaluated(label=EVAL_LABEL))
    product = m.Product(pid="n", label="n", distribution=m.Distribution())
    channel = m.OutputChannel(process="thermalNeutronScatteringLaw inelastic")
    channel.products.products.append(product)
    reaction = m.Reaction(id=m.ReactionId(label="inelastic", ENDF_MT=4), outputChannel=channel)
    energyMax = None
    for T in temps:
        kT = aceKT(T)
        label = heatedLabel(temperatureName(kT))
        suite.styles.add(m.Heated(label=label, derivedFrom=EVAL_LABEL,
                                  temperature=m.PhysicalQuantity(value=kT, unit="eV/k")))
        elastic = processed.cross_sections(T)[2]
        xs, dist, energyMaxT = freeGasTables(awr, T, elastic, emax=emax, options=options,
                                             report=report)
        reaction.crossSection[label] = xs
        product.distribution[label] = dist
        if energyMax is None:
            energyMax = energyMaxT
    suite.reactions.append(reaction)
    suite.provenance = m.Hdf5Provenance(attributes={
        "atomic_weight_ratio": round(float(awr), 6), "energy_max": float(energyMax),
        "nuclides": nuclidesFromPairs([(za, 0.0)])})
    report.warn("thermal free gas by kika (analytic free-gas law, corrected singularity and tails, consistent PDF/CDF; cross section the elastic "
                "broadened by kika)")
    suite.report = report
    return suite


def _tapeZa(path) -> int:
    """MF1/MT451's ZA (the HEAD record's C1) of an ENDF tape."""
    import re

    with open(path, errors="replace") as handle:
        for line in handle:
            if len(line) >= 75 and line[70:72].strip() == "1" and line[72:75].strip() == "451":
                field = line[0:11].strip()
                match = re.fullmatch(r"([+-]?[\d.]+)([+-]\d+)", field)
                return int(round(float(f"{match.group(1)}e{match.group(2)}") if match else float(field)))
    raise HDF5FormatError(f"no MF1/MT451 in {path}")


def _tapeAwr(path) -> float:
    """MF1/MT451's AWR (the HEAD record's C2) of an ENDF tape."""
    import re

    with open(path, errors="replace") as handle:
        for line in handle:
            if len(line) >= 75 and line[70:72].strip() == "1" and line[72:75].strip() == "451":
                field = line[11:22].strip()
                match = re.fullmatch(r"([+-]?[\d.]+)([+-]\d+)", field)
                return float(f"{match.group(1)}e{match.group(2)}") if match else float(field)
    raise HDF5FormatError(f"no MF1/MT451 in {path}")


def _principalAwr(tsl, law) -> float:
    """The principal atom's mass ratio as the tape states it (B(3)), else its mass."""
    from kika._constants import NEUTRON_MASS_AMU

    fields = getattr(getattr(tsl, "provenance", None), "headerFields", None) or {}
    for key in ("mf7", "MF7"):
        block = fields.get(key)
        if isinstance(block, dict) and block.get("b"):
            return float(block["b"][2])
    return float(law.principal.mass.convertedTo("amu").value) / NEUTRON_MASS_AMU


def inelasticEmax(tsl) -> float:
    m = _m()
    return float(_law(tsl, m.IncoherentInelastic).principal.e_max.convertedTo("eV").value)


def _acerBraggEdges(edges, cumulative, emax, nmix=1):
    """The Bragg edges ACER keeps (``aceth.f90:192-214``): walking up the edges, one
    is written when the cumulative E·σ grows by more than ``(c_last + 1)/1e6``
    (eV·b), and the walk stops at ``emax``. An edge of zero weight never passes;
    graphite's edges above ~1.1 eV add less than the threshold and are dropped."""
    from kika.processing._thermal_native import endf11, sigfig

    edges = np.asarray(edges, dtype=float)
    cumulative = np.asarray(cumulative, dtype=float)
    keptE, keptS = [], []
    last = 0.0
    for i, (e, c) in enumerate(zip(edges, cumulative)):
        following = edges[i + 1] if i + 1 < len(edges) else np.inf
        if following > emax or abs(following - emax) <= 1e-5:
            break
        # ACER reads THERMR's coherent σ at the point just above the edge,
        # sigfig(E, 7, +1), as the tape stores it (NJOY's a11), and takes E·σ.
        shaded = sigfig(e, 7, +1)
        sigma = np.array([sigfig(c / shaded, 7, 0)])
        endf11(sigma)
        c = shaded * float(sigma[0])
        if c - last > (last + 1) / 1.0e6:
            keptE.append(shaded)
            keptS.append(c)
            last = c
    # ACE stores both in MeV to 12 figures, the factors divided by nmix
    # (aceth.f90:729-730); OpenMC multiplies by 1e6 on read.
    keptE = [float(f"{e * 1e-6:.11e}") * 1e6 for e in keptE]
    keptS = [float(f"{c * 1e-6 / nmix:.11e}") * 1e6 for c in keptS]
    return np.asarray(keptE), np.asarray(keptS)


def _evaluatedElastic(tsl, T: float, report, nmix: int = 1):
    """The evaluation's elastic at ``T`` as processed forms: ``(xs, distribution)`` or Nones."""
    m = _m()
    coherent = _law(tsl, m.CoherentElastic)
    incoherent = _law(tsl, m.IncoherentElastic)
    inc = coh = None
    if incoherent is not None:
        W = incoherent.debyeWaller(float(T))
        sigma = float(incoherent.boundAtomCrossSection.convertedTo("b").value)
        inc = (m.IncoherentElasticXS(boundCrossSection=sigma, debyeWaller=W),
               m.IncoherentElasticMC(debyeWaller=W))
    if coherent is not None:
        match = [t for t in coherent.temperatures if abs(t - T) < 0.05]
        if not match:
            raise HDF5FormatError(f"the coherent elastic is not tabulated at {T} K")
        if incoherent is not None:
            # OpenMC from_njoy(use_endf_data=True) replaces a mixed elastic
            # law from the evaluated source, including its coherent part.
            # Keep all physical edges and factors rather than truncating and
            # rounding this part through ACE's coherent-only serialization.
            edges = np.asarray(coherent.braggEdges(), dtype=float)
            factors = np.asarray(coherent.cumulativeS(match[0]), dtype=float) / nmix
        else:
            edges, factors = _acerBraggEdges(coherent.braggEdges(), coherent.cumulativeS(match[0]),
                                             float(inelasticEmax(tsl)), nmix)
        coh = (m.CoherentElasticXS(braggEdges=edges, structureFactors=factors),
               m.CoherentElasticMC())
    if coh and inc:
        return (m.SumOfFunctions(functions=[coh[0], inc[0]]),
                m.MixedElasticMC(coherent=coh[1], incoherent=inc[1]))
    return coh or inc or (None, None)


def _declaresElastic(path) -> bool:
    """Presence of MF7/MT2 in an ENDF source, only as a completeness guard.

    Numerical values still come exclusively from the model. GNDS sources
    have no ENDF directory and are not scanned as ENDF records.
    """
    with open(path, errors="replace") as handle:
        first = handle.readline()
        if first.lstrip("\ufeff \t\r\n").startswith("<"):
            return False
        for line in handle:
            if (len(line) >= 75 and line[70:72].strip() == "7"
                    and line[72:75].strip() == "2"):
                return True
    return False


def _matOf(path) -> int:
    with open(path, errors="replace") as handle:
        for line in handle:
            if len(line) >= 75 and line[70:72].strip() == "1" and line[72:75].strip() == "451":
                return int(line[66:70])
    raise HDF5FormatError(f"no MF1/MT451 in {path}")
