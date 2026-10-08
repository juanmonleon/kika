"""G4NDL ``Fission/`` ↔ the model: MT18, its chances, and their writer.

The grammar is :mod:`kika.g4ndl.fission`; this module puts what it reads in
the nodes the ENDF adapter puts the same evaluation in
(``kika/endf/model_adapter/{multiplicity,fission_energy,energy}.py``), so a
G4NDL suite and an ENDF suite of one fissile isotope come out in the same
shape. The labels, the hrefs and ``p_k·ν̄_d`` are restated here rather than
imported, as :mod:`kika.g4ndl.inelastic_decode` restates the MF4-MF6 mapping:
the two formats do not depend on each other, and only the façades and the
front door may import an adapter
(``kika/endf/model_adapter/tests/test_nothing_imports_the_adapter.py``). That
the two still agree is what ``test_fission.py`` (``nubarNode`` finds the
nu-bar) and ``test_fission_full_libraries.py`` (the model against the ENDF
adapter's) check.

**MT18** is one :class:`~kika.nuclear_data.model.reactions.Reaction`, its σ
under ``recon`` (lin-lin, 0 K) from ``Fission/CrossSection``. Its Q is left
*unknown*: both libraries write ``0 0`` as the bookkeeping, and the fission Q
is not in the file (``ET`` of the energy release is the nearest thing, and it
is a different statement). ``Fission/FS``, section by section:

=========  ===================================================================
2, 1       ν̄ total (MF1/452). With a prompt ν̄ it is the §21.3
           ``multiplicitySum`` "total fission neutron multiplicity" over the
           prompt and the delayed; without, the fission neutron's own
           multiplicity — :func:`~kika.endf.model_adapter.multiplicity.nubarHref`'s
           ``separatePrompt`` rule
4, 1       ν̄ prompt (MF1/456): the fission neutron's multiplicity
3, 1       ν̄ delayed (MF1/455): the delayed ``multiplicitySum``, and one §18.4
           ``delayedNeutron`` per decay constant on ``fissionFragmentData``
5, 1       the energy release (MF1/458): a ``fissionEnergyRelease``, nine
           constant ``polynomial1d`` (G4NDL keeps no uncertainty and no energy
           dependence)
1, 4       the prompt neutron's angular half (MF4/18)
1, 5       its energy half (MF5/18): a table for one LF=1 partial, the §18.3
           parametrised spectrum for one LF=5/7/9/11/12, ``weightedFunctionals``
           for several. The two make the neutron's ``uncorrelated``
3, 5       the delayed families' spectra (MF5/455): partial *k* goes on family
           *k*'s product, isotropic in the lab, with the multiplicity
           ``p_k(E)·ν̄_d(E)`` — :func:`~kika.endf.model_adapter.fission_energy.
           attachDelayedSpectra`'s placement
1, 12-15   **not modelled**: the photons, as for the inelastic and the capture
=========  ===================================================================

What the model has no slot for travels in the reaction's
:class:`~kika.nuclear_data.model.provenance.G4NDLFissionProvenance`, one entry
per section in file order: ``targetMass`` and ``iflag`` of a ν̄ body, the
energy release's dummy, ``p(E)`` of a single partial, the weights ``p_k`` and
a digest of the multiplicity they made. A section the model cannot hold
exactly goes there whole, as G4NDL text, and the report says so.

**The chances** (``Fission/FC`` … ``LC``) are MT19, 20, 21 and 38: σ under
``recon``, the Q the file prints as the first bookkeeping integer, and, when
the file has a final state, the neutron's ``uncorrelated`` from its MF4 and MF5
(no multiplicity: Geant4 takes ν̄ from ``FS``).

**Fragment yields** (``Fission/FF``, G4NDL 4.7.1 only): the model's
``productYields`` slot has nothing that fills it yet, for ENDF either, so the
file travels whole in MT18's provenance and is written back unchanged.

**Writing** is the inverse. What a section needs and the model does not say
(``targetMass``, ``iflag``, the dummies) comes from the provenance while it
still describes the model, otherwise from ``targetMass=`` or the suite's AWR
and the format's defaults. Every text is parsed back with the strict reader
and compared before anything is written.

Two things a suite from an ENDF tape carries and Geant4 cannot: an energy
release that depends on E (Geant4 keeps one number per term, so the term's
value at thermal is written, and the report says so), and a spectrum's ``U``
(no G4NDL law reads one; dropped, said).
"""
from __future__ import annotations

import dataclasses
import hashlib
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from kika.g4ndl.exceptions import G4NDLError, G4NDLFormatError, G4NDLUnsupportedError
from kika.g4ndl.fission import (
    CHANCE_MT, CHANCES, CROSS_SECTION_DIR, FF_DIR, FISSION_MT, FS_DIR, INFO_DELAYED,
    INFO_ENERGY_RELEASE, INFO_NEUTRONS, INFO_NU_PROMPT, INFO_NU_TOTAL, DT_YIELD,
    ChanceFissionRecord, EnergyReleaseBody, FissionFSRecord, FissionSection,
    FragmentYieldsRecord, NuDelayedBody, NuPromptBody, NuTotalBody, fissionDifferences,
    formatChanceFission, formatFissionFS, formatFissionSectionBody, formatFragmentYields,
    parse_chance_fission, parse_fission_fs, parse_fragment_yields, parseFissionSectionBody,
)
from kika.g4ndl.inelastic_records import (
    DT_ANGULAR, DT_ENERGY, DT_PHOTON_MULTIPLICITY, EnergyBody, EnergyLaw, Tab1,
)
from kika.g4ndl.records import CrossSectionRecord, Interpolation
from kika.g4ndl.tokens import TokenStream

__all__ = ["CHANCE_DIRECTORY", "decodeFission", "encodeFission", "writeFission",
           "fissionReactions"]

#: :data:`kika.g4ndl.decode.RECONSTRUCTED_LABEL` and ``EVALUATED_LABEL``,
#: restated so that importing this module does not import the decoder.
RECONSTRUCTED_LABEL = "recon"
EVALUATED_LABEL = "eval"

#: §21.3 labels, spelt as :mod:`kika.endf.model_adapter.multiplicity` (and FUDGE) spell them.
TOTAL_NUBAR_LABEL = "total fission neutron multiplicity"
DELAYED_NUBAR_LABEL = "delayed fission neutron multiplicity"


def _fissionProductMultiplicityHref() -> str:
    """The fission neutron's multiplicity: ``multiplicity.fissionProductMultiplicityHref``."""
    return (f"/reactionSuite/reactions/reaction[@label='MT{FISSION_MT}']"
            f"/outputChannel/products/product[@label='n']/multiplicity")


def _multiplicitySumHref(label: str) -> str:
    """One §21.3 ``multiplicitySum``'s multiplicity: ``multiplicity.multiplicitySumHref``."""
    return (f"/reactionSuite/sums/multiplicitySums"
            f"/multiplicitySum[@label='{label}']/multiplicity")

#: MT -> the chance directory that holds it.
CHANCE_DIRECTORY: Dict[int, str] = {mt: ch for ch, mt in CHANCE_MT.items()}

#: The order the sections are written in when the provenance gives none: the
#: order of the 23 G4NDL 4.7.1 files that have every non-photon section.
_DEFAULT_ORDER = ((INFO_NU_TOTAL, DT_YIELD), (INFO_DELAYED, DT_YIELD),
                  (INFO_NU_PROMPT, DT_YIELD), (INFO_ENERGY_RELEASE, DT_YIELD),
                  (INFO_NEUTRONS, DT_ANGULAR), (INFO_NEUTRONS, DT_ENERGY),
                  (INFO_DELAYED, DT_ENERGY))

#: The cross-section bookkeeping integers of ``Fission/CrossSection`` in both libraries.
_MT18_BOOKKEEPING = (0, 0)

#: As :data:`kika.g4ndl.capture._Q_BOOKKEEPING_TOLERANCE`: the files print Q to the eV.
_Q_BOOKKEEPING_TOLERANCE = 1.0

#: Where a constant energy-release term is read when the model's term is not one.
_THERMAL_EV = 0.0253

_LAW_NAMES = {1: "a table", 5: "general evaporation", 7: "a simple Maxwellian",
              9: "evaporation", 11: "Watt", 12: "Madland-Nix"}


# ------------------------------------------------------------------ helpers

def _sha256(path) -> Optional[str]:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path is not None else None


def _digest(function) -> str:
    """A table's points and laws, for "has the model changed since it was read"."""
    from kika.nuclear_data.model import Regions1d

    if function is None:
        return "none"
    form = function if isinstance(function, Regions1d) else Regions1d(function1ds=[function])
    x, y, pairs = form.toEndfRegions()
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    h.update(repr([(int(b), int(c)) for b, c in pairs]).encode())
    return h.hexdigest()


def _crossSectionForm(record: CrossSectionRecord):
    from kika.nuclear_data.model import XYs1d, crossSectionAxes
    from kika.nuclear_data.model.enums import Interpolation as ModelInterpolation

    return XYs1d(xs=record.energy.copy(), ys=record.sigma.copy(),
                 interpolation=ModelInterpolation.linlin, axes=crossSectionAxes(),
                 label=RECONSTRUCTED_LABEL)


def _parameterAxes(name: str, unit: str, x: str = "energy_in", xUnit: str = "eV"):
    from kika.nuclear_data.model.axes import Axes

    return Axes.forFunction1d(name, unit, x, xUnit)


# ------------------------------------------------------------------ decode: MF5

def _lawForm(law: EnergyLaw, where: str, code1: dict, k: int):
    """One partial → its model form, as :func:`kika.endf.model_adapter.energy._member`."""
    from kika.g4ndl.inelastic_decode import _function1d, _tab2Pairs
    from kika.nuclear_data.model import (
        Evaporation, GeneralEvaporation, MadlandNix, PhysicalQuantity,
        SimpleMaxwellianFission, Watt, energyAxes, fromEndfTab2,
    )

    def parameter(tab, name, unit, x="energy_in", xUnit="eV"):
        return _function1d(tab, f"{where} partial {k + 1} {name}",
                           axes=_parameterAxes(name, unit, x, xUnit),
                           code1=code1, key=f"{k}:{name}")

    if law.law == 1:
        axes = energyAxes()
        functions = []
        for j, (e, g) in enumerate(law.spectra):
            f = _function1d(g, f"{where} partial {k + 1} spectrum {j + 1}",
                            code1=code1, key=f"{k}:spectrum {j + 1}")
            f.outerDomainValue = float(e)
            f.index = j
            functions.append(f)
        pairs = _tab2Pairs(law.interpolation, f"{where} partial {k + 1}", code1=code1,
                           key=f"{k}:incident")
        return fromEndfTab2(functions, pairs, axes=axes)
    if law.law == 5:
        theta, g = law.parameters
        return GeneralEvaporation(U=None, theta=parameter(theta, "theta", "eV"),
                                  g=parameter(g, "g", "", "energy_out / theta(energy_in)", ""))
    if law.law == 7:
        return SimpleMaxwellianFission(U=None, theta=parameter(law.parameters[0], "theta", "eV"))
    if law.law == 9:
        return Evaporation(U=None, theta=parameter(law.parameters[0], "theta", "eV"))
    if law.law == 11:
        a, b = law.parameters
        return Watt(U=None, a=parameter(a, "a", "eV"), b=parameter(b, "b", "1/eV"))
    efl, efh = law.scalars
    return MadlandNix(EFL=PhysicalQuantity(float(efl), "eV"),
                      EFH=PhysicalQuantity(float(efh), "eV"),
                      T_M=parameter(law.parameters[0], "T_M", "eV"))


def _weight(tab: Tab1, where: str, code1: dict, k: int):
    from kika.g4ndl.inelastic_decode import _function1d

    return _function1d(tab, f"{where} partial {k + 1} p(E)",
                       axes=_parameterAxes("weight", ""), code1=code1, key=f"{k}:p")


def _partialEntry(law: EnergyLaw) -> dict:
    out = {"law": int(law.law)}
    if law.law == 1:
        out["nDistFunc"] = int(law.nDistFunc)
    return out


def _energyForm(body: EnergyBody, entry: dict, where: str):
    """A ``dataType=5`` body → the prompt spectrum: one law, or their weighted sum."""
    from kika.g4ndl.inelastic_decode import _tab1Dict
    from kika.nuclear_data.model import Weighted, WeightedFunctionals

    code1: dict = {}
    forms = [_lawForm(law, where, code1, k) for k, law in enumerate(body.partials)]
    entry["dummy5"] = float(body.dummy)
    entry["partials"] = [_partialEntry(law) for law in body.partials]
    if len(forms) == 1:
        entry["probability"] = _tab1Dict(body.partials[0].probability)
        form = forms[0]
    else:
        form = WeightedFunctionals(weighted=[
            Weighted(weight=_weight(law.probability, where, code1, k), functional=f)
            for k, (law, f) in enumerate(zip(body.partials, forms))])
    if code1:
        entry["code1"] = code1
    return form


# ------------------------------------------------------------------ decode: MF1

def _multiplicity(body, where: str, domain, entry: dict):
    """A ν̄ body → a :class:`Multiplicity` and what the body says beyond it."""
    from kika.g4ndl.inelastic_decode import _function1d
    from kika.nuclear_data.model import Multiplicity, Polynomial1d, multiplicityAxes

    entry["targetMass"] = float(body.targetMass)
    entry["iflag"] = int(body.iflag)
    code1: dict = {}
    if body.table is not None:
        form = _function1d(body.table, f"{where} nu(E)", axes=multiplicityAxes(),
                           label=EVALUATED_LABEL, code1=code1, key="nu")
    else:
        coefficients = (body.coefficients if isinstance(body, NuTotalBody)
                        else np.array([body.value]))
        form = Polynomial1d(coefficients=np.array(coefficients, dtype=float),
                            domainMin_=float(domain[0]), domainMax_=float(domain[1]),
                            axes=multiplicityAxes(), label=EVALUATED_LABEL)
    if code1:
        entry["code1"] = code1
    return Multiplicity(form=form)


def _energyRelease(body: EnergyReleaseBody, domain, entry: dict):
    from kika.nuclear_data.model import FissionEnergyRelease, Polynomial1d, fissionEnergyReleaseAxes

    entry["dummy"] = float(body.values[0])
    node = FissionEnergyRelease(label=EVALUATED_LABEL)
    for name, value in zip(FissionEnergyRelease.TERMS, body.values[1:]):
        setattr(node, name, Polynomial1d(coefficients=np.array([float(value)]),
                                         domainMin_=float(domain[0]),
                                         domainMax_=float(domain[1]),
                                         axes=fissionEnergyReleaseAxes()))
    return node


# ------------------------------------------------------------------ decode

class _Kept(Exception):
    """A section kept as G4NDL text rather than modelled, and why."""


def _neutronDistribution(channel, angularBody, energyBody, angularEntry, energyEntry,
                         where: str) -> None:
    """MF4 (+ MF5) → the fission neutron's distribution, as the inelastic decoder builds it."""
    from kika.g4ndl.inelastic_decode import _angular
    from kika.nuclear_data.model import Distribution, Isotropic2d, Uncorrelated, XYs2d

    _angular(angularBody, channel, "n", angularEntry, f"{where} MF4")
    channel.genre = "NBody"
    if energyBody is None:
        return
    product = channel.ensureProduct("n")
    existing = product.distribution[EVALUATED_LABEL]
    if not isinstance(existing.angular, (XYs2d, Isotropic2d)):
        raise _Kept(f"the angular half is a {type(existing.angular).__name__}; "
                    f"uncorrelated admits only XYs2d or isotropic2d")
    energy = _energyForm(energyBody, energyEntry, f"{where} MF5")
    product.distribution = Distribution()
    product.distribution[EVALUATED_LABEL] = Uncorrelated(
        angular=existing.angular, energy=energy, productFrame=existing.productFrame)


def _keep(entry: dict, text: str) -> None:
    for key in list(entry):
        if key not in ("position", "infoType", "dataType"):
            del entry[key]
    entry["verbatim"] = text


def decodeFission(crossSection: CrossSectionRecord, finalState: Optional[FissionFSRecord],
                  suite, *, chances: Optional[Dict[str, ChanceFissionRecord]] = None,
                  fragmentYields: Optional[FragmentYieldsRecord] = None, library=None,
                  report=None):
    """Add MT18 (and its chances) to ``suite`` from the ``Fission/`` records of one isotope.

    ``chances`` maps ``"FC"`` … ``"LC"`` to the parsed chance files the isotope
    has. Returns the report, which is also ``suite.report``'s.
    """
    from kika.g4ndl.inelastic_decode import _Verbatim
    from kika.nuclear_data.model import (
        ConversionReport, CrossSection, G4NDLFissionProvenance,
        OutputChannel, Q, Reaction, ReactionId,
    )

    report = report if report is not None else ConversionReport()
    libraryRoot = str(library.root) if library is not None else None
    libraryName = library.root.name if library is not None else None
    where = (crossSection.path.name if crossSection.path is not None else "Fission") + " MT18"
    provenance = G4NDLFissionProvenance(
        library=libraryRoot, libraryName=libraryName, directory="CrossSection",
        crossSectionPath=str(crossSection.path) if crossSection.path is not None else None,
        crossSectionSha256=_sha256(crossSection.path), crossSectionHeader=crossSection.header,
        bookkeeping=tuple(int(b) for b in crossSection.bookkeeping))
    reaction = Reaction(id=ReactionId(label=f"MT{FISSION_MT}", ENDF_MT=FISSION_MT),
                        crossSection=CrossSection(), outputChannel=OutputChannel(Q=Q()),
                        provenance=provenance)
    reaction.crossSection[RECONSTRUCTED_LABEL] = _crossSectionForm(crossSection)
    channel = reaction.outputChannel
    channel.genre = "NBody"
    domain = (float(crossSection.energy[0]), float(crossSection.energy[-1]))

    if finalState is None:
        report.warn(f"{where}: the library has no Fission/FS for this isotope; Geant4 "
                    f"would emit no fission neutron")
    else:
        provenance.finalStatePath = (str(finalState.path) if finalState.path is not None
                                     else None)
        provenance.finalStateSha256 = _sha256(finalState.path)
        provenance.finalStateHeader = finalState.header
        _decodeFS(finalState, reaction, suite, provenance, domain, where, report,
                  libraryRoot, libraryName, _Verbatim)
    if fragmentYields is not None:
        provenance.fragmentYields = formatFragmentYields(
            dataclasses.replace(fragmentYields, header=None))
        provenance.fragmentYieldsPath = (str(fragmentYields.path)
                                         if fragmentYields.path is not None else None)
        provenance.fragmentYieldsSha256 = _sha256(fragmentYields.path)
        provenance.fragmentYieldsHeader = fragmentYields.header
        report.unsupportedNode(
            f"{where} Fission/FF: the fission-product yields (MF8/454, 459) have a "
            f"productYields slot in the model and nothing that fills it yet; kept as "
            f"G4NDL text in the provenance and written back unchanged")
    suite.reactions.append(reaction)
    for chance in CHANCES:
        record = (chances or {}).get(chance)
        if record is not None:
            suite.reactions.append(_decodeChance(record, libraryRoot, libraryName, report,
                                                 _Verbatim))
    return report


def _decodeFS(record: FissionFSRecord, reaction, suite, provenance, domain, where, report,
              libraryRoot, libraryName, verbatimError) -> None:
    from kika.nuclear_data.model import (
        Add, DelayedNeutron, FissionFragmentData, G4NDLFissionProvenance, MultiplicitySum,
        PhysicalQuantity, Product,
    )

    channel = reaction.outputChannel
    last = {}
    for k, s in enumerate(record.sections):
        last[(s.infoType, s.dataType)] = k
    nu: Dict[int, object] = {}
    rates = None
    release = None
    angular = energy = delayedSpectra = None
    entries = []
    for k, s in enumerate(record.sections):
        entry = {"position": k, "infoType": int(s.infoType), "dataType": int(s.dataType)}
        entries.append(entry)
        tag = f"{where} ({s.infoType}, {s.dataType})"
        try:
            if last[(s.infoType, s.dataType)] != k:
                raise _Kept("repeated later in the file, and the consumer keeps the last")
            if s.dataType >= DT_PHOTON_MULTIPLICITY:
                raise _Kept("photon production (MF12-15) has no model form yet")
            if s.infoType in (INFO_NU_TOTAL, INFO_NU_PROMPT) or (
                    s.infoType == INFO_DELAYED and s.dataType == DT_YIELD):
                nu[s.infoType] = _multiplicity(s.body, tag, domain, entry)
                if s.infoType == INFO_DELAYED:
                    rates = [float(r) for r in s.body.decayConstants]
            elif s.infoType == INFO_ENERGY_RELEASE:
                release = _energyRelease(s.body, domain, entry)
            elif s.dataType == DT_ANGULAR:
                angular = (s.body, entry)
            elif s.infoType == INFO_NEUTRONS:
                energy = (s.body, entry)
            else:
                delayedSpectra = (s.body, entry)
        except (_Kept, verbatimError, G4NDLUnsupportedError) as exc:
            if s.dataType < DT_PHOTON_MULTIPLICITY:
                report.unsupportedNode(f"{tag}: {exc}; kept as G4NDL text in the provenance")
            _keep(entry, formatFissionSectionBody(s))
    provenance.sections = entries
    photons = sorted({s.dataType for s in record.sections
                      if s.dataType >= DT_PHOTON_MULTIPLICITY})
    if photons:
        report.unsupportedNode(
            f"{where} (1, {'/'.join(str(d) for d in photons)}): fission photon production "
            f"(MF12-15) has no model form yet; kept as G4NDL text in the provenance and "
            f"written back unchanged")

    # The prompt neutron: MF4, and MF5 only beside a modelled MF4.
    if angular is not None:
        try:
            _neutronDistribution(channel, angular[0], energy[0] if energy else None,
                                 angular[1], energy[1] if energy else {}, where)
        except (_Kept, verbatimError, G4NDLUnsupportedError) as exc:
            report.unsupportedNode(f"{where} (1, 4) and (1, 5): {exc}; kept as G4NDL text "
                                   f"in the provenance")
            channel.products.products[:] = [p for p in channel.products if p.pid != "n"]
            for (body, entry), dt in ((angular, DT_ANGULAR), (energy, DT_ENERGY)):
                if entry is not None:
                    _keep(entry, formatFissionSectionBody(FissionSection(1, dt, body)))
    elif energy is not None:
        report.unsupportedNode(f"{where} (1, 5): a spectrum with no angular half in the "
                               f"model; kept as G4NDL text in the provenance")
        _keep(energy[1], formatFissionSectionBody(FissionSection(1, DT_ENERGY, energy[0])))

    # ν̄, where attachNubar puts it.
    nuProvenance = G4NDLFissionProvenance(library=libraryRoot, libraryName=libraryName,
                                          directory="FS")
    for m in nu.values():
        m.provenance = nuProvenance
    separatePrompt = INFO_NU_PROMPT in nu
    primitive = nu.get(INFO_NU_PROMPT) or nu.get(INFO_NU_TOTAL)
    if primitive is not None:
        channel.ensureProduct("n").multiplicity = primitive
    if INFO_DELAYED in nu or release is not None:
        channel.fissionFragmentData = FissionFragmentData()
    if INFO_DELAYED in nu:
        for i, rate in enumerate(rates, start=1):
            channel.fissionFragmentData.delayedNeutrons.append(DelayedNeutron(
                label=str(i), rate=PhysicalQuantity(value=rate, unit="1/s"),
                product=Product(pid="n", label="n")))
        suite.sums.multiplicitySums.append(MultiplicitySum(
            label=DELAYED_NUBAR_LABEL, multiplicity=nu[INFO_DELAYED], ENDF_MT=455))
    if INFO_NU_TOTAL in nu and separatePrompt:
        total = MultiplicitySum(label=TOTAL_NUBAR_LABEL, multiplicity=nu[INFO_NU_TOTAL],
                                ENDF_MT=452)
        total.summands.append(Add(href=_fissionProductMultiplicityHref()))
        if INFO_DELAYED in nu:
            total.summands.append(Add(href=_multiplicitySumHref(DELAYED_NUBAR_LABEL)))
        suite.sums.multiplicitySums.append(total)
    if release is not None:
        channel.fissionFragmentData.fissionEnergyReleases.append(release)
        report.approximated(
            f"{where} (5, 1): G4NDL keeps one number per energy-release term, with no "
            f"uncertainty; each is a constant polynomial1d over the fission cross "
            f"section's range {domain[0]:g}-{domain[1]:g} eV")

    _reportCode1(entries, where, report)

    if delayedSpectra is not None:
        body, entry = delayedSpectra
        try:
            _delayedFamilies(body, entry, channel, nu.get(INFO_DELAYED), where, report)
            _reportCode1([entry], where, report)
        except (_Kept, verbatimError, G4NDLUnsupportedError) as exc:
            report.unsupportedNode(f"{where} (3, 5): {exc}; kept as G4NDL text in the "
                                   f"provenance")
            for family in (channel.fissionFragmentData.delayedNeutrons
                           if channel.fissionFragmentData is not None else ()):
                if family.product is not None:
                    family.product.multiplicity = None
                    family.product.distribution = None
            _keep(entry, formatFissionSectionBody(FissionSection(INFO_DELAYED, DT_ENERGY, body)))


def _reportCode1(entries, where: str, report) -> None:
    """Say which tables declared INT=1 and were read lin-lin (roadmap D10-3)."""
    for entry in entries:
        n = len(entry.get("code1") or ())
        if n and entry.get("verbatim") is None:
            report.approximated(
                f"{where} ({entry['infoType']}, {entry['dataType']}): {n} table(s) with "
                f"interpolation code 1 read lin-lin, which is how Geant4 evaluates it "
                f"(G4ParticleHPInterpolator.hh); ENDF would read a histogram. The code is "
                f"kept and written back")


def _familyMultiplicity(nubar, weight, label: str, report):
    """``p_k(E) · ν̄_d(E)``, as ``fission_energy._familyMultiplicity`` (and FUDGE) build it.

    Exact when ``p_k`` is constant: the delayed nu-bar's own table, scaled.
    Otherwise the product of two tables is a table of neither law, so it is
    sampled on the union of both grids, steps kept, and joined lin-lin, and the
    report says so.
    """
    from kika.algebra import interval_laws, sample_on_union, union
    from kika.nuclear_data.model import Multiplicity, Regions1d, XYs1d, multiplicityAxes

    nform = nubar if isinstance(nubar, Regions1d) else Regions1d(function1ds=[nubar])
    wform = weight if isinstance(weight, Regions1d) else Regions1d(function1ds=[weight])
    nx, ny, npairs = nform.toEndfRegions()
    wx, wy, wpairs = wform.toEndfRegions()
    wy = np.asarray(wy, dtype=float)
    if wy.size and np.all(wy == wy[0]):
        table = Regions1d.fromEndfRegions(nx, np.asarray(ny, dtype=float) * float(wy[0]),
                                          npairs, axes=multiplicityAxes())
        return Multiplicity(form=table.function1ds[0] if len(table.function1ds) == 1 else table)
    nx, ny = np.asarray(nx, dtype=float), np.asarray(ny, dtype=float)
    wx = np.asarray(wx, dtype=float)
    grid = union([nx, wx[(wx >= nx[0]) & (wx <= nx[-1])]], steps=wx[1:-1])
    values = (sample_on_union(nx, ny, interval_laws(nx.size, npairs), grid)
              * sample_on_union(wx, wy, interval_laws(wx.size, wpairs), grid))
    report.approximated(
        f"delayedNeutron {label!r}: its multiplicity is p_k(E) * nu_d(E) with a p_k that is "
        f"not constant, sampled on the {grid.size}-point union of both grids and joined "
        f"lin-lin -- the product of two tables is not a table of either law")
    return Multiplicity(form=XYs1d(xs=grid, ys=values, axes=multiplicityAxes()))


def _delayedFamilies(body: EnergyBody, entry: dict, channel, delayed, where, report) -> None:
    """MF5/455 → each family's product, as ``attachDelayedSpectra`` places it."""
    from kika.g4ndl.inelastic_decode import _tab1Dict
    from kika.nuclear_data.model import Distribution, Frame, Isotropic2d, Uncorrelated

    data = channel.fissionFragmentData
    families = list(data.delayedNeutrons) if data is not None else []
    if len(families) != len(body.partials):
        raise _Kept(f"{len(body.partials)} family spectra and {len(families)} decay "
                    f"constants; which spectrum is whose is not in the file")
    if delayed is None:
        raise _Kept("no delayed nu-bar to weigh the families' multiplicities by")
    code1: dict = {}
    forms = [_lawForm(law, f"{where} (3, 5)", code1, k) for k, law in enumerate(body.partials)]
    weights = [_weight(law.probability, f"{where} (3, 5)", code1, k)
               for k, law in enumerate(body.partials)]
    entry["dummy5"] = float(body.dummy)
    entry["partials"] = [_partialEntry(law) for law in body.partials]
    entry["weights"] = [_tab1Dict(law.probability) for law in body.partials]
    if code1:
        entry["code1"] = code1
    digests = []
    for family, weight, form in zip(families, weights, forms):
        product = family.product
        product.multiplicity = _familyMultiplicity(delayed.form, weight, family.label, report)
        digests.append(_digest(product.multiplicity.form))
        product.distribution = Distribution()
        product.distribution[EVALUATED_LABEL] = Uncorrelated(
            angular=Isotropic2d(), energy=form, productFrame=Frame.lab)
    entry["multiplicityDigests"] = digests
    entry["nubarDigest"] = _digest(delayed.form)


def _decodeChance(record: ChanceFissionRecord, libraryRoot, libraryName, report,
                  verbatimError):
    from kika.nuclear_data.model import (
        CrossSection, G4NDLFissionProvenance, OutputChannel, Q, Reaction, ReactionId,
    )

    mt = CHANCE_MT[record.chance]
    where = f"{record.path.name if record.path is not None else record.chance} MT{mt}"
    cs = record.crossSection
    provenance = G4NDLFissionProvenance(
        library=libraryRoot, libraryName=libraryName, directory=record.chance,
        crossSectionPath=str(record.path) if record.path is not None else None,
        crossSectionSha256=_sha256(record.path), crossSectionHeader=cs.header,
        bookkeeping=tuple(int(b) for b in cs.bookkeeping))
    reaction = Reaction(id=ReactionId(label=f"MT{mt}", ENDF_MT=mt), crossSection=CrossSection(),
                        outputChannel=OutputChannel(Q=Q(value=float(cs.bookkeeping[0]),
                                                        unit="eV")),
                        provenance=provenance)
    reaction.crossSection[RECONSTRUCTED_LABEL] = _crossSectionForm(cs)
    reaction.outputChannel.genre = "NBody"
    if np.any(cs.sigma < 0):
        report.warn(f"{where}: {int(np.sum(cs.sigma < 0))} negative cross-section value(s), "
                    f"down to {float(cs.sigma.min())!r} b; kept as written, which is how "
                    f"Geant4 reads them")
    if record.hasFinalState:
        angularEntry = {"position": 0, "infoType": int(record.angularHead[0]),
                        "dataType": int(record.angularHead[1])}
        energyEntry = {"position": 1, "infoType": int(record.energyHead[0]),
                       "dataType": int(record.energyHead[1])}
        provenance.sections = [angularEntry, energyEntry]
        try:
            _neutronDistribution(reaction.outputChannel, record.angular, record.energy,
                                 angularEntry, energyEntry, where)
            _reportCode1(provenance.sections, where, report)
        except (_Kept, verbatimError, G4NDLUnsupportedError) as exc:
            report.unsupportedNode(f"{where}: its final state: {exc}; kept as G4NDL text "
                                   f"in the provenance")
            reaction.outputChannel.products.products[:] = []
            _keep(angularEntry, formatFissionSectionBody(FissionSection(1, DT_ANGULAR,
                                                                        record.angular)))
            _keep(energyEntry, formatFissionSectionBody(FissionSection(1, DT_ENERGY,
                                                                       record.energy)))
    return reaction


# ------------------------------------------------------------------ encode

def fissionReactions(suite) -> List[object]:
    """MT18 and the chance reactions ``suite`` has, in the order they are written."""
    out = []
    for mt in (FISSION_MT,) + tuple(CHANCE_MT.values()):
        reaction = suite.findReactionByENDF_MT(mt)
        if reaction is not None:
            out.append(reaction)
    return out


def _provenance(reaction):
    from kika.nuclear_data.model import G4NDLFissionProvenance

    p = getattr(reaction, "provenance", None)
    return p if isinstance(p, G4NDLFissionProvenance) else None


def _crossSection(reaction, provenance, header, bookkeeping, where) -> CrossSectionRecord:
    from kika.g4ndl.capture import _header
    from kika.g4ndl.encode import _checkUnits

    form = reaction.crossSection.get(RECONSTRUCTED_LABEL) if reaction.crossSection else None
    if form is None:
        raise G4NDLUnsupportedError(
            f"{where} has no '{RECONSTRUCTED_LABEL}' cross section: G4NDL is pointwise "
            f"sigma at 0 K, lin-lin. For an ENDF tape, add it with "
            f"kika.endf.model_adapter.pendf.readReconstructed (NJOY RECONR, 0 K)")
    x, y, pairs = form.toEndfRegions()
    if {int(c) for _, c in pairs} != {2}:
        raise G4NDLUnsupportedError(
            f"{where}: the cross section has interpolation codes "
            f"{sorted({int(c) for _, c in pairs})}; G4NDL is read lin-lin. Linearise it")
    _checkUnits(form.axes, {"energy_in": "eV", "crossSection": "b"}, f"{where} sigma")
    carried = provenance.crossSectionHeader if provenance is not None else None
    return CrossSectionRecord(None, _header(header, carried), bookkeeping,
                              np.array(x, dtype=np.float64), np.array(y, dtype=np.float64))


def _chanceBookkeeping(reaction, provenance) -> Tuple[int, int]:
    q = reaction.outputChannel.Q if reaction.outputChannel is not None else None
    q = float(q.value) if q is not None and q.value is not None else 0.0
    carried = provenance.bookkeeping if provenance is not None else None
    if carried and abs(float(carried[0]) - q) <= _Q_BOOKKEEPING_TOLERANCE:
        return tuple(int(b) for b in carried)
    return (int(round(q)), 0)


def _evalForm(product):
    d = product.distribution if product is not None else None
    if d is None or EVALUATED_LABEL not in d.keys():
        return None
    return d[EVALUATED_LABEL]


def _neutron(channel):
    found = [p for p in (channel.products if channel is not None else ()) if p.pid == "n"]
    if len(found) > 1:
        raise G4NDLUnsupportedError(f"the fission channel has {len(found)} neutron products; "
                                    f"G4NDL has one prompt neutron")
    return found[0] if found else None


def _constant(function) -> Optional[float]:
    from kika.nuclear_data.model import Polynomial1d

    if isinstance(function, Polynomial1d) and len(function.coefficients) == 1:
        return float(function.coefficients[0])
    return None


def _nuBody(cls, multiplicity, entry: dict, mass: float, rates, where: str):
    """A :class:`Multiplicity` -> a nu-bar body of type ``cls``."""
    from kika.g4ndl.inelastic_encode import _tab1
    from kika.nuclear_data.model import Polynomial1d

    form = multiplicity.form
    extra = (np.array(rates, dtype=np.float64),) if cls is NuDelayedBody else ()
    if isinstance(form, Polynomial1d):
        if cls is NuTotalBody:
            return NuTotalBody(mass, 1, coefficients=np.array(form.coefficients, dtype=np.float64))
        value = _constant(form)
        if value is None:
            raise G4NDLUnsupportedError(f"{where}: a polynomial of degree "
                                        f"{len(form.coefficients) - 1}; G4NDL holds a "
                                        f"constant or a table here")
        return cls(mass, 1, *extra, value=value)
    table = _tab1(form, f"{where} nu(E)", (entry.get("code1") or {}).get("nu"))
    return cls(mass, 2, *extra, table=table)


def _uniformProbability(energies) -> Tab1:
    e = np.array([float(energies[0]), float(energies[-1])])
    return Tab1(Interpolation((2,), (2,)), e, np.array([1.0, 1.0]))


def _lawOf(form, probability, partial: dict, code1: dict, k: int, where: str, report):
    """A model spectrum -> one ``EnergyLaw`` (the inverse of :func:`_lawForm`)."""
    from kika.g4ndl.inelastic_encode import _interp2, _tab1
    from kika.nuclear_data.model import (
        Evaporation, GeneralEvaporation, MadlandNix, Regions2d, SimpleMaxwellianFission,
        Watt, XYs2d,
    )
    from kika.nuclear_data.model.energy_spectra import _energyValue

    tag = f"{where} partial {k + 1}"
    if isinstance(form, (XYs2d, Regions2d)):
        functions, interp = _interp2(form, tag, code1.get(f"{k}:incident"))
        spectra = tuple((float(f.outerDomainValue),
                         _tab1(f, f"{tag} spectrum {j + 1}", code1.get(f"{k}:spectrum {j + 1}")))
                        for j, f in enumerate(functions))
        nd = partial.get("nDistFunc")
        nd = int(nd) if nd is not None and (int(nd) == len(spectra) or
                                            (int(nd) == 0 and len(spectra) == 1)) else len(spectra)
        prob = probability if probability is not None else _uniformProbability(
            [spectra[0][0], spectra[-1][0]])
        return EnergyLaw(1, prob, nDistFunc=nd, interpolation=interp, spectra=spectra)
    u = getattr(form, "U", None)
    if u is not None and _energyValue(u) != 0.0:
        report.approximated(f"{tag}: U = {_energyValue(u):g} eV is dropped: no G4NDL energy "
                            f"law reads one, and Geant4 samples the law up to its own cut")

    def tab(function, name):
        return _tab1(function, f"{tag} {name}", code1.get(f"{k}:{name}"))

    if isinstance(form, GeneralEvaporation):
        law, params, scalars = 5, (tab(form.theta, "theta"), tab(form.g, "g")), ()
    elif isinstance(form, SimpleMaxwellianFission):
        law, params, scalars = 7, (tab(form.theta, "theta"),), ()
    elif isinstance(form, Evaporation):
        law, params, scalars = 9, (tab(form.theta, "theta"),), ()
    elif isinstance(form, Watt):
        law, params, scalars = 11, (tab(form.a, "a"), tab(form.b, "b")), ()
    elif isinstance(form, MadlandNix):
        law, params = 12, (tab(form.T_M, "T_M"),)
        scalars = (_energyValue(form.EFL), _energyValue(form.EFH))
    else:
        raise G4NDLUnsupportedError(f"{tag}: a {type(form).__name__} has no G4NDL energy law")
    prob = probability if probability is not None else _uniformProbability(params[0].x)
    return EnergyLaw(law, prob, parameters=params, scalars=scalars)


def _energyBody(form, entry: dict, where: str, report) -> EnergyBody:
    """The prompt spectrum -> a ``dataType=5`` body."""
    from kika.g4ndl.inelastic_encode import _tab1, _tab1FromDict
    from kika.nuclear_data.model import WeightedFunctionals

    code1 = entry.get("code1") or {}
    partials = list(entry.get("partials") or [])
    if isinstance(form, WeightedFunctionals):
        if len(partials) != len(form.weighted):
            partials = [{}] * len(form.weighted)
        laws = tuple(
            _lawOf(w.functional, _tab1(w.weight, f"{where} partial {k + 1} p(E)",
                                       code1.get(f"{k}:p")),
                   partials[k], code1, k, where, report)
            for k, w in enumerate(form.weighted))
    else:
        prob = (_tab1FromDict(entry["probability"]) if entry.get("probability") is not None
                else None)
        laws = (_lawOf(form, prob, partials[0] if len(partials) == 1 else {}, code1, 0, where,
                       report),)
    return EnergyBody(float(entry.get("dummy5", 0.0)), laws)


def _delayedBody(families, delayedSum, entry: dict, where: str, report) -> EnergyBody:
    """The families' spectra -> MF5/455, ``p_k`` kept while the multiplicities are."""
    from kika.g4ndl.inelastic_encode import _tab1FromDict

    code1 = entry.get("code1") or {}
    partials = list(entry.get("partials") or [])
    if len(partials) != len(families):
        partials = [{}] * len(families)
    kept = list(entry.get("weights") or [])
    digests = list(entry.get("multiplicityDigests") or [])
    nubar = delayedSum.multiplicity.form if delayedSum is not None else None
    unchanged = (len(kept) == len(families) == len(digests)
                 and entry.get("nubarDigest") == _digest(nubar)
                 and all(_digest(getattr(f.product.multiplicity, "form", None)) == d
                         for f, d in zip(families, digests)))
    laws = []
    for k, family in enumerate(families):
        form = getattr(_evalForm(family.product), "energy", None)
        if form is None:
            raise G4NDLUnsupportedError(f"{where}: delayed family {family.label!r} has no "
                                        f"spectrum, and MF5/455 needs one per family")
        if unchanged:
            weight = _tab1FromDict(kept[k])
        else:
            weight = _weightFrom(family, nubar, f"{where} family {family.label}", report)
        laws.append(_lawOf(form, weight, partials[k], code1, k, where, report))
    return EnergyBody(float(entry.get("dummy5", 0.0)), tuple(laws))


def _weightFrom(family, nubar, where: str, report) -> Tab1:
    """``p_k(E) = m_k(E) / nu_d(E)`` on the family multiplicity's own grid."""
    from kika.nuclear_data.model import Regions1d

    m = getattr(getattr(family.product, "multiplicity", None), "form", None)
    if m is None or nubar is None:
        raise G4NDLUnsupportedError(f"{where}: no family multiplicity and delayed nu-bar to "
                                    f"take p_k(E) from, and no p_k kept from a G4NDL read")
    form = m if isinstance(m, Regions1d) else Regions1d(function1ds=[m])
    x, y, _ = form.toEndfRegions()
    x = np.asarray(x, dtype=np.float64)
    nu = np.asarray(nubar.evaluate(x, outOfRange="hold"), dtype=np.float64)
    p = np.divide(np.asarray(y, dtype=np.float64), nu, out=np.zeros_like(x), where=nu != 0)
    report.approximated(f"{where}: p_k(E) is the family multiplicity over the delayed nu-bar, "
                        f"on the multiplicity's {x.size}-point grid, lin-lin")
    return Tab1(Interpolation((int(x.size),), (2,)), x, p)


def _releaseBody(node, entry: dict, where: str, report) -> EnergyReleaseBody:
    from kika.nuclear_data.model import FissionEnergyRelease

    values = [float(entry.get("dummy", 0.0))]
    varying = []
    for name in FissionEnergyRelease.TERMS:
        function = getattr(node, name)
        if function is None:
            values.append(0.0)
            report.approximated(f"{where}: {name} absent from the model; written as zero")
            continue
        value = _constant(function)
        if value is None:
            value = float(np.asarray(function.evaluate(_THERMAL_EV)))
            varying.append(name)
        values.append(value)
    if varying:
        report.approximated(
            f"{where}: {', '.join(varying)} depend on the incident energy, and Geant4 keeps "
            f"one number per term (G4ParticleHPFissionERelease); each is written at "
            f"{_THERMAL_EV} eV")
    return EnergyReleaseBody(np.array(values, dtype=np.float64))


def _fsEntries(provenance) -> List[dict]:
    if provenance is None or provenance.directory != "CrossSection":
        return []
    return list(provenance.sections)


def _entryFor(entries, key) -> dict:
    for e in reversed(entries):
        if (e["infoType"], e["dataType"]) == key and e.get("verbatim") is None:
            return e
    return {}


def _modelledFS(reaction, suite, entries, targetMass, where, report):
    """The bodies of the ``Fission/FS`` sections the model states for MT18."""
    from kika.g4ndl.inelastic_encode import _angular, _mass, _split

    out: Dict[Tuple[int, int], object] = {}
    channel = reaction.outputChannel
    neutron = _neutron(channel)
    sums = suite.sums.multiplicitySums
    totalSum, delayedSum = sums.byENDF_MT(452), sums.byENDF_MT(455)
    primitive = neutron.multiplicity if neutron is not None else None
    if primitive is not None and not primitive.isEvaluable:
        primitive = None
    data = channel.fissionFragmentData

    def mass(key):
        return _mass(_entryFor(entries, key).get("targetMass"), targetMass, suite,
                     f"{where} {key}")

    nus = []
    if totalSum is not None and totalSum.multiplicity is not None:
        nus.append(((INFO_NU_TOTAL, DT_YIELD), NuTotalBody, totalSum.multiplicity, None))
        if primitive is not None:
            nus.append(((INFO_NU_PROMPT, DT_YIELD), NuPromptBody, primitive, None))
    elif primitive is not None:
        nus.append(((INFO_NU_TOTAL, DT_YIELD), NuTotalBody, primitive, None))
    families = list(data.delayedNeutrons) if data is not None else []
    if delayedSum is not None and delayedSum.multiplicity is not None:
        rates = []
        for f in families:
            rate = f.rate
            if rate is None or rate.unit not in ("1/s", "s**-1"):
                raise G4NDLUnsupportedError(f"{where}: delayed family {f.label!r} has no "
                                            f"decay constant in 1/s")
            rates.append(float(rate.value))
        if not rates:
            raise G4NDLUnsupportedError(f"{where}: a delayed nu-bar and no precursor family; "
                                        f"MF1/455 states at least one decay constant")
        nus.append(((INFO_DELAYED, DT_YIELD), NuDelayedBody, delayedSum.multiplicity, rates))
    for key, cls, multiplicity, rates in nus:
        out[key] = _nuBody(cls, multiplicity, _entryFor(entries, key), mass(key), rates,
                           f"{where} {key}")

    releases = list(data.fissionEnergyReleases) if data is not None else []
    if releases:
        node = next((r for r in releases if r.label == EVALUATED_LABEL), releases[0])
        key = (INFO_ENERGY_RELEASE, DT_YIELD)
        out[key] = _releaseBody(node, _entryFor(entries, key), f"{where} {key}", report)

    form = _evalForm(neutron)
    if form is not None:
        angular, energy, frame = _split(form, f"{where} neutron")
        key = (INFO_NEUTRONS, DT_ANGULAR)
        e4 = _entryFor(entries, key)
        m = float(e4["targetMass"]) if "targetMass" in e4 else mass(key)
        out[key] = _angular(angular, frame, m, e4, f"{where} MF4", report)
        if energy is not None:
            key = (INFO_NEUTRONS, DT_ENERGY)
            out[key] = _energyBody(energy, _entryFor(entries, key), f"{where} MF5", report)

    if any(_evalForm(f.product) is not None for f in families):
        key = (INFO_DELAYED, DT_ENERGY)
        out[key] = _delayedBody(families, delayedSum, _entryFor(entries, key),
                                f"{where} MF5/455", report)
    return out


def _fsRecord(reaction, suite, header, targetMass, report) -> Optional[FissionFSRecord]:
    from kika.g4ndl.capture import _header

    provenance = _provenance(reaction)
    entries = _fsEntries(provenance)
    where = "Fission/FS MT18"
    modelled = _modelledFS(reaction, suite, entries, targetMass, where, report)
    sections: List[FissionSection] = []
    used = set()
    photons: Dict[str, object] = {}
    lastOf = {}
    for i, e in enumerate(entries):
        lastOf[(e["infoType"], e["dataType"])] = i
    for i, entry in enumerate(entries):
        key = (entry["infoType"], entry["dataType"])
        if entry.get("verbatim") is not None:
            if key in modelled and lastOf[key] == i:
                continue  # the model now holds what the file held verbatim
            try:
                body = parseFissionSectionBody(entry["verbatim"], key[0], key[1], photons,
                                               tag=f"{where} {key}")
            except G4NDLFormatError as exc:
                raise G4NDLError(f"{where} {key}: the text kept in the provenance does not "
                                 f"read back: {exc}") from None
            sections.append(FissionSection(key[0], key[1], body))
            continue
        if key in modelled and key not in used:
            sections.append(FissionSection(key[0], key[1], modelled[key]))
            used.add(key)
        elif key not in modelled:
            report.warn(f"{where}: the file had a {key} section and the model no longer "
                        f"holds what it said; it is not written")
    if any(s.dataType >= DT_PHOTON_MULTIPLICITY for s in sections):
        report.warn(f"{where}: the fission photons are written back as they were read; kika "
                    f"does not model photon production yet")
    for key in _DEFAULT_ORDER:
        if key in modelled and key not in used:
            sections.append(FissionSection(key[0], key[1], modelled[key]))
    if not sections:
        return None
    carried = provenance.finalStateHeader if provenance is not None else None
    return FissionFSRecord(None, _header(header, carried), tuple(sections))


def _chanceRecord(reaction, header, targetMass, suite, report) -> ChanceFissionRecord:
    from kika.g4ndl.inelastic_encode import _angular, _mass, _split

    mt = reaction.id.ENDF_MT
    chance = CHANCE_DIRECTORY[mt]
    where = f"Fission/{chance} MT{mt}"
    provenance = _provenance(reaction)
    cs = _crossSection(reaction, provenance, header, _chanceBookkeeping(reaction, provenance),
                       where)
    entries = list(provenance.sections) if provenance is not None else []
    heads = [(int(e["infoType"]), int(e["dataType"])) for e in entries] or [(1, 4), (3, 5)]
    form = _evalForm(_neutron(reaction.outputChannel))
    if form is not None:
        angular, energy, frame = _split(form, f"{where} neutron")
        if energy is None:
            raise G4NDLUnsupportedError(f"{where}: a chance's final state is MF4 and MF5 "
                                        f"together, and the neutron has no spectrum")
        e4 = entries[0] if entries and entries[0].get("verbatim") is None else {}
        e5 = entries[1] if len(entries) > 1 and entries[1].get("verbatim") is None else {}
        m = (float(e4["targetMass"]) if "targetMass" in e4
             else _mass(None, targetMass, suite, where))
        return ChanceFissionRecord(None, chance, cs, heads[0],
                                   _angular(angular, frame, m, e4, f"{where} MF4", report),
                                   heads[1], _energyBody(energy, e5, f"{where} MF5", report))
    if len(entries) == 2 and all(e.get("verbatim") is not None for e in entries):
        bodies = [parseFissionSectionBody(e["verbatim"], 1, int(e["dataType"]),
                                          tag=f"{where} {h}") for e, h in zip(entries, heads)]
        return ChanceFissionRecord(None, chance, cs, heads[0], bodies[0], heads[1], bodies[1])
    return ChanceFissionRecord(None, chance, cs)


def encodeFission(suite, *, header=None, targetMass: Optional[float] = None, report=None):
    """MT18 and its chances -> ``(crossSection, fs, {chance: record}, ff, report)``.

    ``fs`` is ``None`` when the model states nothing ``Fission/FS`` holds, and
    ``ff`` when no fragment yields were kept.
    """
    from kika.g4ndl.capture import _header
    from kika.g4ndl.encode import KEEP
    from kika.nuclear_data.model import ConversionReport

    report = report if report is not None else ConversionReport()
    header = KEEP if header is None else header
    reaction = suite.findReactionByENDF_MT(FISSION_MT)
    if reaction is None:
        raise G4NDLUnsupportedError("the suite has no MT18 to write to Fission/")
    provenance = _provenance(reaction)
    bookkeeping = (tuple(int(b) for b in provenance.bookkeeping)
                   if provenance is not None and provenance.bookkeeping else _MT18_BOOKKEEPING)
    cs = _crossSection(reaction, provenance, header, bookkeeping, "Fission/CrossSection MT18")
    fs = _fsRecord(reaction, suite, header, targetMass, report)
    if fs is None:
        report.warn("Fission MT18: the suite states no nu-bar, spectrum or energy release, so "
                    "no Fission/FS is written and Geant4 would emit no fission neutron")
    chances = {CHANCE_DIRECTORY[r.id.ENDF_MT]: _chanceRecord(r, header, targetMass, suite,
                                                             report)
               for r in fissionReactions(suite)[1:]}
    ff = None
    if provenance is not None and provenance.fragmentYields:
        record = parse_fragment_yields(TokenStream(provenance.fragmentYields))
        ff = dataclasses.replace(record, header=_header(header, provenance.fragmentYieldsHeader))
    return cs, fs, chances, ff, report


# ------------------------------------------------------------------ write

def _verify(text: str, record, parse, what: str) -> None:
    try:
        back = parse(TokenStream(text))
    except G4NDLFormatError as exc:
        raise G4NDLUnsupportedError(f"{what}: the model holds something the G4NDL grammar "
                                    f"refuses, so the file was not written. {exc}") from None
    diffs = fissionDifferences(record, back)
    if diffs:
        raise G4NDLError(f"{what}: the written text does not read back: {'; '.join(diffs[:5])}")


def writeFission(suite, root, *, compressed: bool = False, header=None,
                 targetMass: Optional[float] = None, elementName: Optional[str] = None):
    """Write MT18 and its chances into the library directory ``root``.

    ``Fission/CrossSection``, ``Fission/FS``, one ``Fission/FC`` ... ``LC`` per
    chance reaction and ``Fission/FF`` when fragment yields were kept, ``.z``
    when ``compressed``. The isotope's file in a ``Fission/`` directory the
    suite has nothing for is removed, and so is a twin of the other variant:
    Geant4 would otherwise read a chance (or yields) the suite does not have.
    """
    from kika.g4ndl.encode import _replaceFile, formatCrossSection, recordDifferences, targetKey
    from kika.g4ndl.names import file_name
    from kika.g4ndl.parse import parse_cross_section

    cs, fs, chances, ff, report = encodeFission(suite, header=header, targetMass=targetMass)
    stem = file_name(targetKey(suite))
    if elementName is not None:
        stem = stem.rsplit("_", 1)[0] + "_" + elementName
    payloads = {}
    text = formatCrossSection(cs)
    if recordDifferences(cs, parse_cross_section(TokenStream(text))):
        raise G4NDLError(f"{CROSS_SECTION_DIR}/{stem}: the text does not read back")
    payloads[CROSS_SECTION_DIR] = text
    if fs is not None:
        text = formatFissionFS(fs)
        _verify(text, fs, parse_fission_fs, f"{FS_DIR}/{stem}")
        payloads[FS_DIR] = text
    for chance, record in chances.items():
        text = formatChanceFission(record)
        _verify(text, record, lambda s, c=chance: parse_chance_fission(s, c),
                f"Fission/{chance}/{stem}")
        payloads[f"Fission/{chance}"] = text
    if ff is not None:
        text = formatFragmentYields(ff)
        _verify(text, ff, parse_fragment_yields, f"{FF_DIR}/{stem}")
        payloads[FF_DIR] = text
    root = Path(root)
    for sub, text in payloads.items():
        data = text.encode("ascii")
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        _replaceFile(d / (stem + (".z" if compressed else "")),
                     zlib.compress(data, 9) if compressed else data)
        twin = d / (stem if compressed else stem + ".z")
        if twin.exists():
            twin.unlink()
            report.warn(f"removed {twin}: Geant4 reads the .z when both exist")
    for sub in (FS_DIR,) + tuple(f"Fission/{c}" for c in CHANCES) + (FF_DIR,):
        if sub in payloads:
            continue
        for name in (stem, stem + ".z"):
            stale = root / sub / name
            if stale.exists():
                stale.unlink()
                report.warn(f"removed {stale}: the suite has nothing for {sub}")
    return report
