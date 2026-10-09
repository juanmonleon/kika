"""G4NDL photon bodies ↔ ENDF MF12-15 sections ↔ the model (roadmap D10-2, E5f.3).

Geant4's photon reader (``G4ParticleHPPhotonDist``) reads ENDF's photon files
almost field for field: ``InitMean`` is MF12 (``repFlag`` = LO), ``InitPartials``
MF13, ``InitAngular`` MF14 and ``InitEnergies`` MF15. So the G4NDL records
(:mod:`kika.g4ndl.inelastic_records`, ``Photon*Body``) are converted to the flat
ENDF sections (:mod:`kika.endf.classes.mf12` …) and the ENDF adapter
(:mod:`kika.endf.model_adapter.photons`), measured byte for byte on the three
libraries, puts them in the model -- and the way back is the same two steps
reversed. There is one place the photons of an ENDF tape and of a G4NDL file
reach the model, not two.

What G4NDL does not write and ENDF needs is filled, never guessed beyond the
format's own default, and kept so the record comes back as read:

- ``InitMean`` drops each photon's ES and LP. ES is taken from ``InitAngular``,
  which writes it, and LP is 0. Geant4 pairs the two **by index** (photon *i*
  gets ``InitAngular``'s entry *i*, the isotropic ones first), not by (EG, ES)
  as ENDF does -- and G4NDL 4.7.1's N-14 writes 9172252 in one and 9172250 in
  the other. So the ENDF MF14 carries MF12's (EG, ES) photon by photon, which
  makes the adapter pair them as Geant4 does, and ``InitAngular``'s own values
  are kept in the bookkeeping and written back.
- ``InitMean`` drops MF12's total multiplicity (NK > 1). None is invented:
  the model gets no ``multiplicitySum``.
- ``InitAngular`` with ``isoFlag=1`` reads nothing more; ENDF's LI=1 section
  needs NK, the number of photons, which is ``InitMean``'s.
- A tabulated ``InitAngular`` line has no incident-energy interpolation; the
  ENDF TAB2 gets lin-lin.

The G4NDL bookkeeping the model has no slot for -- ``nDiscrete`` as written,
the cascade's first conversion flag, ``nDiscrete2``/``nIso`` -- goes into the
G4NDL provenance entry beside the ENDF ``headerFields`` the adapter fills.

A cascade (``repFlag=2``, MF12 LO=2) names the levels it decays to by their
energies, so it is attached once every level section of the isotope is known
(:func:`attachCascades`): the inelastic decoder collects the cascades of all
its channels and attaches them together, and the ENDF adapter puts each on its
residual level in PoPs, as it does for a tape. A cascade with nothing to name
its levels (a capture or fission final state) stays G4NDL text.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from kika.g4ndl.inelastic_records import (
    PhotonAngularBody, PhotonAngularLine, PhotonCascadeBody, PhotonEnergyBody, PhotonLegendre,
    PhotonLine, PhotonMultiplicityBody, PhotonPartial, PhotonPartialsBody, PhotonSpectrum,
    PhotonSpectrumAt, PhotonTabulated, Tab1,
)
from kika.g4ndl.records import Interpolation

__all__ = ["photonsToEndf", "endfToPhotons", "attachPhotonBodies", "photonBodiesFromModel",
           "photonLabels"]


def photonLabels(fields) -> List[str]:
    """The labels of the photon products the ENDF photon bookkeeping describes."""
    fields = fields or {}
    for key in ("mf12", "mf13"):
        if key in fields and fields[key].get("photons"):
            return [p["label"] for p in fields[key]["photons"]]
    return []


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _interp(interpolation: Optional[Interpolation], n: int) -> List[Tuple[int, int]]:
    if interpolation is None or not interpolation.nbt:
        return [(max(n, 1), 2)]
    return [(int(b), int(c)) for b, c in zip(interpolation.nbt, interpolation.codes)]


def _interpolation(pairs) -> Interpolation:
    pairs = list(pairs or [])
    return Interpolation(tuple(int(b) for b, _ in pairs), tuple(int(c) for _, c in pairs))


def _table(tab: Tab1, c1=0.0, c2=0.0, l1=0, l2=0):
    from kika.endf.classes.mf12.base import PhotonTable

    return PhotonTable(c1=float(c1), c2=float(c2), l1=int(l1), l2=int(l2),
                       interp=_interp(tab.interpolation, len(tab.x)),
                       x=[float(v) for v in tab.x], y=[float(v) for v in tab.y])


def _tab1(table) -> Tab1:
    return Tab1(_interpolation(table.interp), np.asarray(table.x, dtype=float),
                np.asarray(table.y, dtype=float))


# ---------------------------------------------------------------------------
# G4NDL → ENDF
# ---------------------------------------------------------------------------

def photonsToEndf(mean, angular: Optional[PhotonAngularBody],
                  energies: Optional[PhotonEnergyBody], mt: int, za: float
                  ) -> Tuple[Dict[int, object], dict]:
    """The photon bodies of one reaction → ``({MF: section}, g4ndl bookkeeping)``."""
    from kika.endf.classes.mf12.base import MF12MT, MF13MT
    from kika.endf.classes.mf14.base import (AngularNode, AnisotropicPhoton, IsotropicPhoton,
                                             MF14MT)
    from kika.endf.classes.mf15.base import MF15MT, PhotonSpectrum as EndfSpectrum

    sections: Dict[int, object] = {}
    book: dict = {}
    shells = _shellsFrom(angular)            # [(EG, ES)] in InitAngular's order
    pairs14 = [list(pair) for pair in shells]
    nPhotons = (len(mean.lines) if isinstance(mean, PhotonMultiplicityBody)
                else len(mean.partials) if isinstance(mean, PhotonPartialsBody) else -1)
    byIndex = bool(pairs14) and len(pairs14) == nPhotons
    if isinstance(mean, PhotonCascadeBody):
        book.update(kind="cascade", repFlag=mean.repFlag, ic=mean.conversionFlag,
                    nGammaEnergies=mean.nGammaEnergies)
        section = MF12MT(number=mt, _za=float(za), _awr=float(mean.targetMass), _lo=2,
                         _l2=int(mean.conversionFlag2), _ns=len(mean.transitions))
        section.es_ns = float(mean.baseEnergy)
        section.nt = len(mean.transitions)
        values = []
        for t in mean.transitions:
            values += [float(t.levelEnergy), float(t.probability)]
            if int(mean.conversionFlag2) == 2:
                values.append(float(t.photonFraction))
        section.transition_values = values
        sections[12] = section
        count = len(mean.transitions)
    elif isinstance(mean, PhotonMultiplicityBody):
        book.update(kind="mean", nDiscrete=mean.nDiscrete, repFlag=mean.repFlag)
        section = MF12MT(number=mt, _za=float(za), _awr=float(mean.targetMass), _lo=1)
        for index, line in enumerate(mean.lines):
            es = pairs14[index][1] if byIndex else _shellOf(shells, line.energy)
            section.photons.append(_table(line.yield_, c1=line.energy, c2=es, l2=line.disType))
        sections[12] = section
        count = len(mean.lines)
    elif isinstance(mean, PhotonPartialsBody):
        book.update(kind="partials", nDiscrete=mean.nDiscrete)
        section = MF13MT(number=mt, _za=float(za), _awr=float(mean.targetMass))
        if mean.total is not None:
            section.total = _table(mean.total)
        for partial in mean.partials:
            section.photons.append(_table(partial.sigma, c1=partial.energy, c2=partial.shell,
                                          l1=partial.isPrimary, l2=partial.disType))
        sections[13] = section
        count = len(mean.partials)
    else:
        raise _Unmodelled(f"a photon body of type {type(mean).__name__}")

    if angular is not None:
        mf14 = MF14MT(number=mt, _za=float(za), _awr=float(mean.targetMass))
        if angular.isoFlag == 1:
            mf14._li, mf14._ltt, mf14._nk = 1, 0, count
        else:
            mf14._li, mf14._ltt = 0, int(angular.tabulationType)
            mf14._nk, mf14._ni = int(angular.nDiscrete2), int(angular.nIso)
            production = sections.get(12) or sections.get(13)
            # MF12's (EG, ES) by index when the counts agree: Geant4's pairing.
            own = ([(p.eg, p.es) for p in production.photons] if byIndex
                   else [tuple(pair) for pair in pairs14])
            if byIndex and [list(pair) for pair in own] != pairs14:
                book["mf14Pairs"] = pairs14
            niso = int(angular.nIso)
            mf14.isotropic = [IsotropicPhoton(eg=float(e), es=float(s)) for e, s in own[:niso]]
            for line, (eg, es) in zip(angular.lines, own[niso:]):
                photon = AnisotropicPhoton(eg=float(eg), es=float(es),
                                           tab2_interp=_interp(line.interpolation, line.nNeu))
                for record in line.records:
                    if isinstance(record, PhotonLegendre):
                        photon.nodes.append(AngularNode(
                            energy=float(record.energy),
                            coefficients=[float(c) for c in record.coefficients]))
                    else:
                        photon.nodes.append(AngularNode(
                            energy=float(record.energy),
                            interp=_interp(record.interpolation, len(record.mu)),
                            mu=[float(v) for v in record.mu],
                            p=[float(v) for v in record.probability]))
                mf14.anisotropic.append(photon)
            book["tabulatedHasNoTab2Interp"] = int(angular.tabulationType) == 2
        sections[14] = mf14

    if energies is not None and energies.needed:
        mf15 = MF15MT(number=mt, _za=float(za), _awr=float(mean.targetMass))
        for spectrum in energies.spectra:
            mf15.spectra.append(EndfSpectrum(
                weight=_table(spectrum.probability, l2=spectrum.dummy),
                tab2_interp=_interp(spectrum.interpolation, len(spectrum.spectra)),
                distributions=[_table(at.spectrum, c2=at.energy) for at in spectrum.spectra]))
        sections[15] = mf15
    book["energiesNeeded"] = bool(energies is not None and energies.needed)
    book["angular"] = angular is not None
    return sections, book


def _shellsFrom(angular: Optional[PhotonAngularBody]) -> List[list]:
    if angular is None or angular.isoFlag == 1:
        return []
    return ([[float(e), float(s)] for e, s in angular.isotropic]
            + [[float(l.energy), float(l.shell)] for l in angular.lines])


def _shellOf(shells: List[list], energy: float) -> float:
    """ES of the first unused ``InitAngular`` photon with this EG (0 if none):
    ENDF pairs MF12 and MF14 by (EG, ES), and ``InitMean`` does not write ES."""
    for entry in shells:
        if entry[0] == float(energy):
            shells.remove(entry)
            return entry[1]
    return 0.0


#: The pid an MF6 photon wears while the ENDF adapter places MF12's photons.
_ASIDE = "photon (MF6)"


class _Unmodelled(Exception):
    """A photon body the model cannot hold; it stays G4NDL text."""


# ---------------------------------------------------------------------------
# ENDF → G4NDL
# ---------------------------------------------------------------------------

def endfToPhotons(sections: Dict[int, object], book: dict, targetMass: Optional[float] = None):
    """``{MF: section}`` → ``(mean, angular, energies)`` G4NDL bodies."""
    production = sections.get(12) or sections.get(13)
    mass = float(targetMass if targetMass is not None else production._awr)
    if 13 in sections:
        mf13 = sections[13]
        n = book.get("nDiscrete", len(mf13.photons))
        mean = PhotonPartialsBody(
            int(n), mass, _tab1(mf13.total) if mf13.total is not None and n != 1 else None,
            tuple(PhotonPartial(float(p.eg), float(p.es), int(p.lp), int(p.lf), _tab1(p))
                  for p in mf13.photons))
    elif sections[12].lo == 2:
        from kika.g4ndl.inelastic_records import PhotonTransition

        mf12 = sections[12]
        lg = int(mf12.lg)
        transitions = tuple(PhotonTransition(float(row[0]), float(row[1]),
                                             float(row[2]) if lg == 2 else None)
                            for row in mf12.transitions)
        mean = PhotonCascadeBody(2, mass, int(book.get("ic", lg)), float(mf12.es_ns), lg,
                                 int(book.get("nGammaEnergies", len(transitions))), transitions)
    else:
        mf12 = sections[12]
        n = book.get("nDiscrete", len(mf12.photons))
        mean = PhotonMultiplicityBody(int(book.get("repFlag", 1)), mass, int(n),
                                      tuple(PhotonLine(int(p.lf), float(p.eg), _tab1(p))
                                            for p in mf12.photons))
    angular = None
    mf14 = sections.get(14)
    if mf14 is not None:
        if mf14._li == 1:
            angular = PhotonAngularBody(1)
        else:
            lines = []
            for photon in mf14.anisotropic:
                if mf14._ltt == 1:
                    records = tuple(PhotonLegendre(float(node.energy),
                                                   np.asarray(node.coefficients, dtype=float))
                                    for node in photon.nodes)
                    interpolation = _interpolation(photon.tab2_interp)
                else:
                    records = tuple(PhotonTabulated(float(node.energy), _interpolation(node.interp),
                                                    np.asarray(node.mu, dtype=float),
                                                    np.asarray(node.p, dtype=float))
                                    for node in photon.nodes)
                    interpolation = None
                lines.append(PhotonAngularLine(float(photon.eg), float(photon.es),
                                               len(photon.nodes), interpolation, records))
            pairs = book.get("mf14Pairs")
            if pairs is not None and len(pairs) == len(mf14.isotropic) + len(lines):
                # InitAngular's own (EG, ES), as the file wrote them.
                niso = len(mf14.isotropic)
                isotropic = tuple((float(e), float(s)) for e, s in pairs[:niso])
                lines = [PhotonAngularLine(float(e), float(s), line.nNeu, line.interpolation,
                                           line.records)
                         for (e, s), line in zip(pairs[niso:], lines)]
            else:
                isotropic = tuple((float(p.eg), float(p.es)) for p in mf14.isotropic)
            angular = PhotonAngularBody(0, int(mf14._ltt), int(mf14._nk), int(mf14._ni),
                                        isotropic, tuple(lines))
    mf15 = sections.get(15)
    if mf15 is None:
        energies = PhotonEnergyBody(False)
    else:
        energies = PhotonEnergyBody(True, tuple(
            PhotonSpectrum(int(spectrum.weight.l2), _tab1(spectrum.weight),
                           _interpolation(spectrum.tab2_interp),
                           tuple(PhotonSpectrumAt(float(d.c2), _tab1(d))
                                 for d in spectrum.distributions))
            for spectrum in mf15.spectra))
    return mean, angular, energies


# ---------------------------------------------------------------------------
# The model, through the ENDF adapter
# ---------------------------------------------------------------------------

def attachPhotonBodies(suite, reaction, mean, angular, energies, entry: dict, where: str,
                       report, za: float, mt: Optional[int] = None) -> bool:
    """Put one reaction's G4NDL photons on its products. ``False`` if they stay text.

    The ENDF adapter is called with a borrowed ENDF provenance on the reaction;
    the ``headerFields`` it fills move into ``entry["endf"]`` and the G4NDL
    bookkeeping into ``entry["g4ndl"]``.
    """
    from kika.endf.model_adapter import photons as adapter
    from kika.nuclear_data.model import EVAL_LABEL, EndfProvenance

    mt = int(mt if mt is not None else reaction.ENDF_MT)
    if isinstance(mean, PhotonCascadeBody):
        report.unsupportedNode(f"{where}: a cascade (repFlag=2, ENDF MF12 LO=2) names levels "
                               f"this final state does not carry; kept as G4NDL text in the "
                               f"provenance and written back unchanged")
        return False
    try:
        sections, book = photonsToEndf(mean, angular, energies, mt, za)
    except _Unmodelled as exc:
        report.unsupportedNode(f"{where}: {exc}; kept as G4NDL text in the provenance and "
                               f"written back unchanged")
        return False
    why = adapter._whyNotModelled(sections)
    if why is not None:
        report.unsupportedNode(f"{where}: {why}; kept as G4NDL text")
        return False
    sigma = _sigma(reaction)
    host = adapter._Host(reaction, SimpleNamespace(crossSection={EVAL_LABEL: sigma}),
                         f"/reactionSuite/reactions/reaction[@label='{reaction.label}']",
                         "reaction")
    own = reaction.provenance
    reaction.provenance = EndfProvenance(za=za)
    # A photon the file's MF6 section already states (LAW=0, U-236 F04-F10 of
    # G4NDL 4.7.1) is MF6's: the ENDF adapter would merge MF12's photons into
    # it, which a tape can afford -- its MF6 is written back from its own
    # provenance -- and a G4NDL record cannot. It is set aside under another
    # pid while the adapter runs, keeping its label so no new one collides.
    others = [product for product in reaction.outputChannel.products if product.pid == "photon"]
    for product in others:
        product.pid = _ASIDE
    try:
        why = adapter._attach(suite, host, sections, mt, report)
        fields = dict(reaction.provenance.headerFields)
    finally:
        reaction.provenance = own
        for product in others:
            product.pid = "photon"
    if why is not None:
        report.unsupportedNode(f"{where}: {why}; kept as G4NDL text")
        return False
    entry["endf"] = fields
    entry["g4ndl"] = book
    return True


def attachCascades(suite, items, report, za: float) -> List[bool]:
    """Attach the cascades of several level reactions at once. ``items`` are
    ``(reaction, mean, angular, energies, bag, where)``; returns, per item,
    whether it reached the model (``bag`` then holds ``endf``/``g4ndl``)."""
    from kika.endf.model_adapter import photons as adapter
    from kika.nuclear_data.model import EndfProvenance

    converted = []
    sections12 = {}
    for reaction, mean, angular, energies, bag, where in items:
        sections, book = photonsToEndf(mean, angular, energies, int(reaction.ENDF_MT), za)
        converted.append((sections, book))
        sections12[int(reaction.ENDF_MT)] = sections[12]
    done = []
    for (reaction, mean, angular, energies, bag, where), (sections, book) in zip(items, converted):
        own = reaction.provenance
        # G4NDL states QI only; the level a cascade leaves is ES_NS above the
        # ground state, so QM = QI + ES_NS is what the ENDF route would read.
        qi = getattr(reaction.outputChannel.Q, "value", None)
        esNs = float(sections[12].es_ns)
        reaction.provenance = EndfProvenance(za=za, qm=None if qi is None else float(qi) + esNs,
                                             lr=0)
        try:
            why = adapter._attachCascade(suite, sections, sections12, int(reaction.ENDF_MT),
                                         report)
            fields = dict(reaction.provenance.headerFields)
        finally:
            reaction.provenance = own
        if why is not None:
            report.unsupportedNode(f"{where}: {why}; kept as G4NDL text")
            done.append(False)
            continue
        # (QI + ES) - QI may miss ES by an ulp; the level is ES_NS, as read.
        level = suite.PoPs.particles.get(fields["mf12"]["level"])
        if level is not None and level.energy is not None and level.energy.value != esNs:
            from kika.nuclear_data.model import PhysicalQuantity
            level.energy = PhysicalQuantity(value=esNs, unit="eV")
            fields["mf12"] = dict(fields["mf12"], levelEnergy=esNs)
        bag["endf"] = fields
        bag["g4ndl"] = book
        done.append(True)
    return done


def _sigma(reaction):
    from kika.nuclear_data.model import EVAL_LABEL

    for label in (EVAL_LABEL, "recon"):
        try:
            return reaction.crossSection[label]
        except (KeyError, TypeError):
            continue
    return None


def photonBodiesFromModel(suite, reaction, entry: dict, mat: int, report,
                          targetMass: Optional[float] = None):
    """The model's photons of *reaction* → ``(mean, angular, energies)``, or ``None``.

    Uses the ENDF ``headerFields`` the decode kept (a G4NDL-read suite) or the
    reaction's own ENDF provenance (an ENDF-read suite).
    """
    import copy

    from kika.endf.model_adapter.photons import encodePhotonSections
    from kika.nuclear_data.model import EVAL_LABEL, EndfProvenance

    fields = entry.get("endf") if entry is not None else None
    if fields is None and getattr(reaction.provenance, "sourceFormat", None) != "endf":
        return None
    # A shallow copy carries the ENDF bookkeeping and exposes G4NDL's σ (under
    # ``recon``) as the ``eval`` MF13 divides by; the model is not touched.
    view = copy.copy(reaction)
    if fields is not None:
        view.provenance = EndfProvenance(headerFields=dict(fields))
    sigma = _sigma(reaction)
    if sigma is not None:
        view.crossSection = {EVAL_LABEL: sigma}
    shell = SimpleNamespace(reactions=[view], orphanProducts=[], provenance=None,
                            sums=suite.sums, PoPs=suite.PoPs,
                            findReactionByENDF_MT=lambda mt: view if int(mt) == int(
                                reaction.ENDF_MT) else suite.findReactionByENDF_MT(mt))
    encoded, _ = encodePhotonSections(shell, mat, report)
    sections = {mf: section for mf, mt, section in encoded if mt == int(reaction.ENDF_MT)}
    if not sections or not (12 in sections or 13 in sections):
        return None
    return endfToPhotons(sections, (entry or {}).get("g4ndl") or {}, targetMass)
