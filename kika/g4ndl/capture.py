"""G4NDL ``Capture/``: radiative capture (MT102), read, modelled and written.

The third process after the elastic (Phases 1-9) and the inelastic (Phase
10). One isotope's capture is a cross section and at most one final state,
and every piece of grammar is one kika already reads:

``Capture/CrossSection/<name>``
    the elastic cross-section grammar (``G4ParticleHPIsoData::Init``), whose
    two bookkeeping integers Geant4 discards. In capture both libraries write
    the reaction's Q value there, in eV and rounded, and 0 (measured over the
    559 + 592 files on 2026-10-08).
``Capture/FSMF6/<name>``
    an ENDF MF6 body, ``G4ParticleHPEnAngCorrelation::Init``: the inelastic
    ``dataType=6`` body with no section header. 508 (G4NDL 4.7.1) and 564
    (JEFF-4.0) files, all one photon product under LAW=1 but H-1's (a LAW=2
    photon and its LAW=4 deuteron) and JEFF-4.0 Cu-63's (a LAW=0 Cu-64).
``Capture/FS/<name>``
    the photon body of an inelastic reaction without its three section
    headers: ``G4ParticleHPPhotonDist::InitMean``, ``InitAngular`` and
    ``InitEnergies`` in that order (MF12/13, MF14, MF15). 48 and 22 files.

``G4NeutronHPCaptureFS::Init`` opens ``FSMF6`` first, by exact name, and
reads ``FS`` only when there is none; with neither it samples the photons
from its own ``G4PhotonEvaporation``. No isotope of either library has both
files, and kika refuses a library that does rather than pick one silently.

**The model.** MT102 is one :class:`~kika.nuclear_data.model.reactions.Reaction`
with its σ under ``recon`` (lin-lin, 0 K, as every G4NDL σ) and its Q. An
``FSMF6`` body becomes its products, by the same mapping as the inelastic MF6
(:func:`kika.g4ndl.inelastic_decode._energyAngle`): the capture γ spectrum
reaches the model. An ``FS`` body does **not**: the model has no
photon-production form yet (roadmap Fase 10, D10-2), so it travels whole as
G4NDL text in the reaction's
:class:`~kika.nuclear_data.model.provenance.G4NDLCaptureProvenance`, the
report says so, and the encoder writes it back unchanged.

**Writing.** σ, Q and the products come from the model; the bookkeeping the
model has no slot for comes from the provenance while it still describes the
model. A suite read from ENDF carries its MF6 bookkeeping
(:func:`kika.g4ndl.inelastic_encode._fromEndfMF6`), so an evaluation with an
MF6 for MT102 (JEFF-4.0) gives an ``FSMF6``; one whose capture photons are
MF12-15 gives no final state, since the ENDF adapter does not read those
files either, and the report says that Geant4 will fall back to photon
evaporation. Every text is parsed back with the strict reader and compared
record for record before anything is written.
"""
from __future__ import annotations

import dataclasses
import hashlib
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from kika.g4ndl.exceptions import G4NDLError, G4NDLFormatError, G4NDLUnsupportedError
from kika.g4ndl.inelastic_records import (
    EnergyAngleBody, PhotonAngularBody, PhotonCascadeBody, PhotonEnergyBody,
    PhotonMultiplicityBody,
)
from kika.g4ndl.records import CrossSectionRecord
from kika.g4ndl.tokens import HEADER_TAG, TokenStream

__all__ = ["CAPTURE_MT", "parseFinalState", "captureFinalState", "finalStateDifferences", "CROSS_SECTION_DIR", "FS_DIR", "FSMF6_DIR", "FINAL_STATE_DIRS",
           "CapturePhotonsRecord", "CaptureMF6Record", "parse_capture_photons",
           "parse_capture_mf6", "formatCaptureFS", "decodeCapture", "encodeCapture",
           "writeCapture"]

CAPTURE_MT = 102
#: :data:`kika.g4ndl.decode.RECONSTRUCTED_LABEL`, restated so that parsing a
#: capture file does not import the decoder, and with it the model.
RECONSTRUCTED_LABEL = "recon"
CROSS_SECTION_DIR = "Capture/CrossSection"
FSMF6_DIR = "Capture/FSMF6"
FS_DIR = "Capture/FS"
#: In the order ``G4NeutronHPCaptureFS::Init`` tries them.
FINAL_STATE_DIRS = (FSMF6_DIR, FS_DIR)

#: How far apart, in eV, the model's Q and the cross-section file's first
#: bookkeeping integer may be for the integer to be carried as read: the files
#: print Q rounded to the eV.
_Q_BOOKKEEPING_TOLERANCE = 1.0


# ------------------------------------------------------------------ records

@dataclass(frozen=True, eq=False)
class CapturePhotonsRecord:
    """``Capture/FS/<name>``: ``InitMean``, ``InitAngular``, ``InitEnergies``.

    ``mean`` is a :class:`~kika.g4ndl.inelastic_records.PhotonMultiplicityBody`
    (repFlag 1) or a :class:`~kika.g4ndl.inelastic_records.PhotonCascadeBody`
    (repFlag 2); its ``targetMass`` is the mass Geant4 uses for the capture
    kinematics. ``energies.needed`` is whether ``InitEnergies`` read anything.
    """

    path: Optional[Path]
    header: Optional[Tuple[str, str]]
    mean: Union[PhotonMultiplicityBody, PhotonCascadeBody]
    angular: PhotonAngularBody
    energies: PhotonEnergyBody


@dataclass(frozen=True, eq=False)
class CaptureMF6Record:
    """``Capture/FSMF6/<name>``: one ENDF MF6 body (``targetMass frameFlag nProducts``…)."""

    path: Optional[Path]
    header: Optional[Tuple[str, str]]
    body: EnergyAngleBody


CaptureFSRecord = Union[CapturePhotonsRecord, CaptureMF6Record]


# ------------------------------------------------------------------ parse

def parse_capture_photons(stream: TokenStream) -> CapturePhotonsRecord:
    """``Capture/FS/<name>``, read to its last token.

    An empty file is refused: Geant4's ``InitMean`` would return ``false`` and
    sample the photons from its evaporation model instead, which is a fact
    about the library a reader should hear, not a final state.
    """
    from kika.g4ndl.inelastic_parse import _photonAngular, _photonEnergies, _photonMean

    tag = "Capture/FS"
    if stream.atEnd():
        stream._fail("the file holds no photon data; Geant4 would fall back to "
                     "G4PhotonEvaporation", f"{tag}: repFlag", stream.position)
    mean = _photonMean(stream, f"{tag} (MF12/13)")
    angular = _photonAngular(stream, f"{tag} (MF14)")
    energies = _photonEnergies(stream, f"{tag} (MF15)", mean)
    stream.expectEnd(f"{tag}: end of file")
    return CapturePhotonsRecord(stream.path, stream.header, mean, angular, energies)


def parse_capture_mf6(stream: TokenStream) -> CaptureMF6Record:
    """``Capture/FSMF6/<name>``, read to its last token."""
    from kika.g4ndl.inelastic_parse import _energyAngle

    tag = "Capture/FSMF6"
    body = _energyAngle(stream, tag)
    stream.expectEnd(f"{tag}: end of file")
    return CaptureMF6Record(stream.path, stream.header, body)


_PARSERS = {FSMF6_DIR: parse_capture_mf6, FS_DIR: parse_capture_photons}


def parseFinalState(text: str, kind: str) -> CaptureFSRecord:
    """A final-state body from its text; ``kind`` is ``"FSMF6"`` or ``"FS"``."""
    return _PARSERS[f"Capture/{kind}"](TokenStream(text))


# ------------------------------------------------------------------ format

def _bodyLines(record: CaptureFSRecord):
    from kika.g4ndl.inelastic_format import (
        _energyAngle, _photonAngular, _photonEnergies, _photonMean,
    )

    if isinstance(record, CaptureMF6Record):
        return _energyAngle(record.body)
    return _photonMean(record.mean) + _photonAngular(record.angular) + \
        _photonEnergies(record.energies)


def formatCaptureFS(record: CaptureFSRecord) -> str:
    """The text of a ``Capture/FS`` or ``Capture/FSMF6`` file."""
    out = [f"{record.header[0]} {record.header[1]}"] if record.header is not None else []
    return "\n".join(out + _bodyLines(record)) + "\n"


def _bodyText(record: CaptureFSRecord) -> str:
    """The body alone, header excluded: what a verbatim final state keeps."""
    return "\n".join(_bodyLines(record))


def _kind(record: CaptureFSRecord) -> str:
    return "FSMF6" if isinstance(record, CaptureMF6Record) else "FS"


# ------------------------------------------------------------------ decode

def _sha256(path: Optional[Path]) -> Optional[str]:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path is not None else None


def decodeCapture(crossSection: CrossSectionRecord, finalState: Optional[CaptureFSRecord],
                  suite, *, library=None, report=None):
    """Add MT102 to ``suite`` from its ``Capture/CrossSection`` and final-state records.

    Returns the report, which is also ``suite.report``'s.
    """
    from kika.g4ndl.inelastic_decode import _energyAngle, _Verbatim
    from kika.nuclear_data.model import (
        ConversionReport, CrossSection, G4NDLCaptureProvenance, OutputChannel, Q, Reaction,
        ReactionId, XYs1d, crossSectionAxes,
    )
    from kika.nuclear_data.model.enums import Interpolation as ModelInterpolation

    report = report if report is not None else ConversionReport()
    where = (crossSection.path.name if crossSection.path is not None else "Capture") + " MT102"
    provenance = G4NDLCaptureProvenance(
        library=str(library.root) if library is not None else None,
        libraryName=library.root.name if library is not None else None,
        crossSectionPath=str(crossSection.path) if crossSection.path is not None else None,
        crossSectionSha256=_sha256(crossSection.path),
        crossSectionHeader=crossSection.header,
        bookkeeping=tuple(int(b) for b in crossSection.bookkeeping))
    q = float(crossSection.bookkeeping[0])
    reaction = Reaction(id=ReactionId(label=f"MT{CAPTURE_MT}", ENDF_MT=CAPTURE_MT),
                        crossSection=CrossSection(),
                        outputChannel=OutputChannel(Q=Q(value=q, unit="eV")),
                        provenance=provenance)
    reaction.outputChannel.genre = "NBody"
    reaction.crossSection[RECONSTRUCTED_LABEL] = XYs1d(
        xs=crossSection.energy.copy(), ys=crossSection.sigma.copy(),
        interpolation=ModelInterpolation.linlin, axes=crossSectionAxes(),
        label=RECONSTRUCTED_LABEL)
    if finalState is None:
        report.warn(f"{where}: the library has no capture final state for this isotope; "
                    f"Geant4 samples its photons from G4PhotonEvaporation")
    else:
        kind = _kind(finalState)
        provenance.finalState = kind
        provenance.finalStatePath = str(finalState.path) if finalState.path is not None else None
        provenance.finalStateSha256 = _sha256(finalState.path)
        provenance.finalStateHeader = finalState.header
        entry = provenance.finalStateEntry
        if kind == "FSMF6":
            try:
                _energyAngle(finalState.body, reaction.outputChannel, entry,
                             f"{where} MF6", report)
            except (_Verbatim, G4NDLUnsupportedError) as exc:
                entry.clear()
                entry["verbatim"] = _bodyText(finalState)
                report.unsupportedNode(f"{where} FSMF6: {exc}; kept as G4NDL text in the "
                                       f"provenance and written back unchanged")
            qs = {float(p.actualStateQ) for p in finalState.body.products}
            if len(qs) == 1:
                # The same Q as the bookkeeping, to the eV, in every real file,
                # but printed to full precision.
                reaction.outputChannel.Q.value = qs.pop()
        else:
            entry["verbatim"] = _bodyText(finalState)
            report.unsupportedNode(
                f"{where} FS: capture photon production (MF12-15) has no model form yet; "
                f"kept as G4NDL text in the provenance and written back unchanged")
    suite.reactions.append(reaction)
    return report


# ------------------------------------------------------------------ encode

def _captureReaction(suite):
    reaction = suite.findReactionByENDF_MT(CAPTURE_MT)
    if reaction is None:
        raise G4NDLUnsupportedError("the suite has no MT102 to write to Capture/")
    return reaction


def _captureProvenance(reaction):
    from kika.nuclear_data.model import G4NDLCaptureProvenance

    p = reaction.provenance
    return p if isinstance(p, G4NDLCaptureProvenance) else None


def _header(header, carried):
    from kika.g4ndl.encode import KEEP

    if header == KEEP:
        return tuple(carried) if carried else None
    if header is None:
        return None
    label = str(header)
    if not label or len(label.split()) != 1:
        raise ValueError(f"header label {label!r} must be one non-empty token")
    return (HEADER_TAG, label)


def _crossSectionRecord(reaction, provenance, header, where, report) -> CrossSectionRecord:
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
    q = reaction.outputChannel.Q if reaction.outputChannel is not None else None
    q = float(q.value) if q is not None and q.value is not None else 0.0
    carried = provenance.bookkeeping if provenance is not None else None
    if carried and abs(float(carried[0]) - q) <= _Q_BOOKKEEPING_TOLERANCE:
        bookkeeping = tuple(int(b) for b in carried)
    else:
        bookkeeping = (int(round(q)), 0)
    return CrossSectionRecord(None, _header(header, provenance.crossSectionHeader
                                            if provenance is not None else None),
                              bookkeeping, np.array(x, dtype=np.float64),
                              np.array(y, dtype=np.float64))


def captureFinalState(suite, *, header=None, targetMass: Optional[float] = None,
                      report=None) -> Optional[CaptureFSRecord]:
    """MT102's final state as a record, or ``None`` when the suite states none.

    The model's products win: when MT102 has a product with a multiplicity or
    a distribution, the final state is the ``FSMF6`` they make. Otherwise a
    final state kept verbatim in the provenance is written back as it was.
    The cross section plays no part, so this needs no ``recon`` σ.
    """
    from kika.g4ndl.encode import KEEP
    from kika.g4ndl.inelastic_encode import _energyAngle, _fromEndfMF6
    from kika.nuclear_data.model import ConversionReport

    report = report if report is not None else ConversionReport()
    header = KEEP if header is None else header
    reaction = _captureReaction(suite)
    where = "Capture MT102"
    provenance = _captureProvenance(reaction)
    entry = dict(provenance.finalStateEntry) if provenance is not None else {}
    carriedHeader = provenance.finalStateHeader if provenance is not None else None
    products = [p for p in (reaction.outputChannel.products if reaction.outputChannel else ())
                if p.multiplicity is not None or p.distribution]
    if products:
        if entry.get("verbatim") is not None:
            report.warn(f"{where}: the {provenance.finalState} final state the file had is "
                        f"replaced by the model's products, written as FSMF6")
            entry = {}
        if not entry.get("products"):
            entry = _fromEndfMF6(reaction) or entry
        body = _energyAngle(products, entry, suite, targetMass, f"{where} FSMF6", report)
        return CaptureMF6Record(None, _header(header, carriedHeader), body)
    if entry.get("verbatim") is not None:
        kind = provenance.finalState
        try:
            record = parseFinalState(entry["verbatim"], kind)
        except G4NDLFormatError as exc:
            raise G4NDLError(f"{where}: the {kind} text kept in the provenance does not "
                             f"read back: {exc}") from None
        if kind == "FS":
            report.warn(f"{where}: the capture photons (FS) are written back as they were "
                        f"read; kika does not model photon production yet")
        return dataclasses.replace(record, path=None, header=_header(header, carriedHeader))
    report.warn(f"{where}: the suite has no capture final state (an ENDF tape's capture "
                f"photons are MF12-15, which kika does not read yet), so none is written "
                f"and Geant4 samples the photons from G4PhotonEvaporation")
    return None


def encodeCapture(suite, *, header=None, targetMass: Optional[float] = None, report=None
                  ) -> Tuple[CrossSectionRecord, Optional[CaptureFSRecord], object]:
    """MT102 of ``suite`` → ``(crossSection, finalState or None, report)``."""
    from kika.g4ndl.encode import KEEP
    from kika.nuclear_data.model import ConversionReport

    report = report if report is not None else ConversionReport()
    header = KEEP if header is None else header
    reaction = _captureReaction(suite)
    cs = _crossSectionRecord(reaction, _captureProvenance(reaction), header,
                             "Capture MT102", report)
    fs = captureFinalState(suite, header=header, targetMass=targetMass, report=report)
    return cs, fs, report


# ------------------------------------------------------------------ write

def finalStateDifferences(a, b):
    """Every field where two final-state records differ; ``path`` is ignored."""
    from kika.g4ndl.inelastic_format import inelasticDifferences

    return inelasticDifferences(a, b)


def _verify(text: str, record: CaptureFSRecord, what: str) -> None:
    try:
        back = parseFinalState(text, _kind(record))
    except G4NDLFormatError as exc:
        raise G4NDLUnsupportedError(f"{what}: the model holds something the G4NDL grammar "
                                    f"refuses, so the file was not written. {exc}") from None
    diffs = finalStateDifferences(record, back)
    if diffs:
        raise G4NDLError(f"{what}: the written text does not read back: {'; '.join(diffs[:5])}")


def writeCapture(suite, root, *, compressed: bool = False, header=None,
                 targetMass: Optional[float] = None, elementName: Optional[str] = None):
    """Write MT102 into the library directory ``root``.

    ``root/Capture/CrossSection/<name>`` and the final state, in ``FSMF6`` or
    ``FS``, ``.z`` when ``compressed``. The isotope's file in the other
    final-state directory is removed, and so is a twin of the other variant
    (``.z`` or plain): Geant4 reads ``FSMF6`` before ``FS`` and a ``.z``
    before its text, so a stale one would be what it reads.
    """
    from kika.g4ndl.encode import (
        _replaceFile, formatCrossSection, recordDifferences, targetKey,
    )
    from kika.g4ndl.names import file_name
    from kika.g4ndl.parse import parse_cross_section

    cs, fs, report = encodeCapture(suite, header=header, targetMass=targetMass)
    stem = file_name(targetKey(suite))
    if elementName is not None:
        stem = stem.rsplit("_", 1)[0] + "_" + elementName
    payloads = {}
    text = formatCrossSection(cs)
    if recordDifferences(cs, parse_cross_section(TokenStream(text))):
        raise G4NDLError(f"{CROSS_SECTION_DIR}/{stem}: the text does not read back")
    payloads[CROSS_SECTION_DIR] = text
    if fs is not None:
        sub = f"Capture/{_kind(fs)}"
        text = formatCaptureFS(fs)
        _verify(text, fs, f"{sub}/{stem}")
        payloads[sub] = text
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
    for sub in FINAL_STATE_DIRS:
        if sub in payloads:
            continue
        for name in (stem, stem + ".z"):
            stale = root / sub / name
            if stale.exists():
                stale.unlink()
                report.warn(f"removed {stale}: the suite's capture final state is not {sub}")
    return report
