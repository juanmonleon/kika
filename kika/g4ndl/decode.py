"""G4NDL elastic records → :class:`~kika.nuclear_data.model.suite.ReactionSuite`.

Phase 4 of the G4NDL roadmap. The parsed records (:mod:`kika.g4ndl.records`)
go straight into the common model, the way :mod:`kika.gnds` does it: no
G4NDL-shaped object in between, and the tokens the model has no slot for in a
:class:`~kika.nuclear_data.model.provenance.G4NDLProvenance`.

**What the suite says, and why.** A G4NDL library is processed data, so the
styles state which half is processed:

* the cross section is an ``XYs1d`` lin-lin under ``recon``, a
  ``crossSectionReconstructed`` style derived from ``eval``. That is the
  consumer's contract, not a guess about the file: Geant4 reads it as
  pointwise σ at 0 K and Doppler-broadens it itself
  (``G4ParticleHPElasticData.cc:150``), and the file has no interpolation
  record, so lin-lin is what it means;
* the angular distribution is under ``eval``. It is MF4 carried over: the
  ``repFlag`` 1/2/3 blocks are ENDF's LTT 1/2/3 with the same records (Fe-56
  JEFF-4.0 has the 3 960 Legendre energies of the tape). The file does not
  record which evaluation it came from, and the report says so.

The physics mapping is the ENDF MF4 one (``kika/endf/model_adapter/angular.py``):
``a_0 = 1`` is prepended to the stored ``a_1 … a_NL``, a tabulated record with
one region is an ``XYs1d`` and with more a ``Regions1d``, a block with more
than one energy region a ``Regions2d``, and ``repFlag=3`` a ``Regions2d`` of
the Legendre block and the table. It is written again here rather than
imported, so that the two formats do not depend on each other.

**Two decisions made with Juan (2026-10-06).**

* **INT=1 in μ is read as lin-lin** (revised in Fase 10, D10-3; it was
  refused on 2026-10-06). Geant4 evaluates code 1 as lin-lin — the
  ``Histogram`` call is commented out (``G4ParticleHPInterpolator.hh:91-95``)
  — and the model holds what is transported. ENDF would read a histogram, so
  the report calls it an approximation; the declared code is kept in the
  provenance (``tabulatedCode1``) and written back. The inelastic reader does
  the same. No real elastic file uses it. A code 1 in **incident energy** is
  still refused: neither library has one, and it has no slot.
* **``tempdep != 0`` is accepted.** Geant4 reads it and discards it; refusing
  would be stricter than the consumer for no gain. It is kept in the
  provenance and reported as a warning when it is not zero.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from kika.g4ndl.exceptions import G4NDLUnsupportedError
from kika.g4ndl.names import IsotopeKey
from kika.g4ndl.records import (
    FRAME_CM, FRAME_LAB, REP_ISOTROPIC, REP_LEGENDRE, REP_MIXED, REP_TABULATED,
    AngularBlock, CrossSectionRecord, ElasticFSRecord, Interpolation,
)
from kika.nuclear_data.model import (
    AngularTwoBody,
    ConversionReport,
    CrossSection,
    CrossSectionReconstructed,
    Distribution,
    Evaluated,
    Frame,
    G4NDLProvenance,
    Isotropic2d,
    Legendre,
    Nuclide,
    OutputChannel,
    PoPs,
    Q,
    Reaction,
    ReactionId,
    ReactionSuite,
    Regions1d,
    Regions2d,
    XYs1d,
    angularAxes,
    crossSectionAxes,
    fromEndfTab2,
    pidFromZA,
)
from kika.nuclear_data.model.enums import Interpolation as ModelInterpolation

__all__ = ["decodeElastic", "newSuite", "EVALUATED_LABEL", "RECONSTRUCTED_LABEL",
           "ELASTIC_MT", "FRAME_FROM_FLAG", "targetId"]

EVALUATED_LABEL = "eval"
RECONSTRUCTED_LABEL = "recon"
ELASTIC_MT = 2

#: G4NDL ``frameFlag`` → GNDS §3.4.2 frame. Same numbers as ENDF's LCT.
FRAME_FROM_FLAG = {FRAME_LAB: Frame.lab, FRAME_CM: Frame.centerOfMass}

#: Geant4 evaluates pairs this close as a step to the lower node
#: (``G4ParticleHPVector::GetXsec``); quoted in the report, not applied.
_GEANT4_STEP_RTOL = 1.0e-6


def targetId(key: IsotopeKey) -> str:
    """The GNDS id of the target: ``Fe56``, ``C`` for natural carbon, ``Am242_m1``.

    A metastable target takes GNDS's metastable alias spelling (``_m<M>``),
    not the excited-level one (``_e<level>``): G4NDL names the isomer by
    ``M`` and does not say which nuclear level it is.
    """
    za = key.Z * 1000 + (key.A or 0)
    pid = pidFromZA(za)
    return f"{pid}_m{key.M}" if key.M else pid


def newSuite(key: IsotopeKey, *, library=None, provenance: Optional[G4NDLProvenance] = None,
             report: Optional[ConversionReport] = None,
             readDirectories: Tuple[str, ...] = ("Elastic",),
             absentDirectories: Tuple[str, ...] = ()):
    """An empty suite for ``key``: target nuclide, the two styles, the provenance.

    What the elastic and the inelastic decoders share. ``readDirectories``
    are the library's top-level directories the caller reads; every other one
    present is reported as not read; ``absentDirectories`` are those kika
    reads that hold nothing for this isotope.
    """
    report = report if report is not None else ConversionReport()
    if provenance is None:
        provenance = G4NDLProvenance(
            library=str(library.root) if library is not None else None,
            libraryName=library.root.name if library is not None else None)
    pops = PoPs()
    target = targetId(key)
    pops.add(Nuclide(id=target, Z=key.Z, A=key.A))
    if key.M:
        report.warn(
            f"{target}: G4NDL names the metastable state M={key.M} but not its "
            f"nuclear level, so the nuclide's level index is left at 0")
    suite = ReactionSuite(
        evaluation=provenance.libraryName or "",
        projectile="n",
        target=target,
        projectileFrame=Frame.lab,
    )
    suite.PoPs = pops
    suite.provenance = provenance
    suite.styles.add(Evaluated(label=EVALUATED_LABEL))
    suite.styles.add(CrossSectionReconstructed(label=RECONSTRUCTED_LABEL,
                                               derivedFrom=EVALUATED_LABEL))
    _reportPartialRead(library, report, readDirectories, absentDirectories)
    report.lost(
        "G4NDL does not record which evaluation it was translated from, nor its "
        "resonance parameters: the 'eval' style holds the distributions only, "
        "and the cross sections are the 0 K pointwise ones under 'recon'")
    suite.report = report
    return suite, report


def decodeElastic(crossSection: CrossSectionRecord, finalState: ElasticFSRecord,
                  key: IsotopeKey, *, library=None,
                  report: Optional[ConversionReport] = None,
                  readDirectories: Tuple[str, ...] = ("Elastic",),
                  absentDirectories: Tuple[str, ...] = ()):
    """One isotope's elastic records → ``(suite, report)``.

    Parameters
    ----------
    crossSection, finalState
        The parsed ``Elastic/CrossSection`` and ``Elastic/FS`` files.
    key
        The isotope, from the library index.
    library
        The :class:`~kika.g4ndl.library.G4NDLLibrary` they came from, when
        there is one: it names the library and lists what else it holds, so
        the report can say the read is partial.

    Raises
    ------
    G4NDLUnsupportedError
        For an interpolation code 1 (histogram) between incident energies.
    """
    provenance = _provenance(crossSection, finalState, library)
    where = finalState.path.name if finalState.path is not None else str(key)
    suite, report = newSuite(key, library=library, provenance=provenance, report=report,
                             readDirectories=readDirectories,
                             absentDirectories=absentDirectories)

    reaction = Reaction(
        id=ReactionId(label=f"MT{ELASTIC_MT}", ENDF_MT=ELASTIC_MT),
        crossSection=_crossSection(crossSection, where, report),
        # Elastic: Q = 0 by the physics of the channel, not by default.
        outputChannel=OutputChannel(Q=Q(value=0.0, unit="eV")),
        provenance=provenance,
    )
    channel = reaction.outputChannel
    channel.genre = "twoBody"
    product = channel.ensureProduct("n")
    product.provenance = provenance
    product.distribution = Distribution()
    product.distribution[EVALUATED_LABEL] = _angular(finalState, where, report,
                                                     provenance.tabulatedCode1)
    suite.reactions.append(reaction)
    return suite, report


# ------------------------------------------------------------ cross section

def _crossSection(record: CrossSectionRecord, where: str,
                  report: ConversionReport) -> CrossSection:
    out = CrossSection()
    out[RECONSTRUCTED_LABEL] = XYs1d(
        xs=record.energy.copy(), ys=record.sigma.copy(),
        interpolation=ModelInterpolation.linlin,
        axes=crossSectionAxes(), label=RECONSTRUCTED_LABEL,
    )
    step = np.diff(record.energy)
    repeated = int(np.sum(step == 0))
    close = int(np.sum((step > 0) & (step < _GEANT4_STEP_RTOL * record.energy[1:])))
    if repeated or close:
        report.warn(
            f"{where}: the cross section has {repeated} repeated energies (kept, "
            f"in file order, as discontinuities) and {close} more node pairs "
            f"closer than {_GEANT4_STEP_RTOL:g} relative. Geant4 evaluates both "
            f"as a step to the lower node (G4ParticleHPVector::GetXsec), so "
            f"between them it and lin-lin disagree")
    return out


# ------------------------------------------------------------ final state

def _angular(record: ElasticFSRecord, where: str, report: ConversionReport,
             code1: Optional[dict] = None):
    frame = FRAME_FROM_FLAG[record.frameFlag]
    if record.repFlag == REP_ISOTROPIC:
        # The consumer reads frameFlag again and keeps the second one.
        if record.frameFlag2 != record.frameFlag:
            report.warn(
                f"{where}: repFlag=0 with frameFlag={record.frameFlag} and a "
                f"second frameFlag={record.frameFlag2}; Geant4 uses the second")
        return Isotropic2d(productFrame=FRAME_FROM_FLAG[record.frameFlag2])

    _checkTemperatures(record, where, report)
    axes = angularAxes()  # one object: see the note in endf/model_adapter/angular.py
    code1 = {} if code1 is None else code1
    if record.repFlag == REP_LEGENDRE:
        angular = _block(record.legendre, axes, where, "Legendre")
    elif record.repFlag == REP_TABULATED:
        angular = _block(record.tabulated, axes, where, "table", code1)
    else:
        assert record.repFlag == REP_MIXED
        angular = Regions2d(function2ds=[
            _block(record.legendre, axes, where, "Legendre"),
            _block(record.tabulated, axes, where, "table", code1),
        ], axes=axes)
    if code1:
        report.approximated(
            f"{where}: {len(code1)} table record(s) with interpolation code 1 in mu read "
            f"lin-lin, which is how Geant4 evaluates it (G4ParticleHPInterpolator.hh); "
            f"ENDF would read a histogram. The code is kept and written back")
    return AngularTwoBody(angular=angular, productFrame=frame)


def _block(block: AngularBlock, axes, where: str, tag: str, code1=None):
    _refuseHistogram(block.interpolation, f"{where}: {tag} block, incident energy")
    functions = []
    for i, rec in enumerate(block.records):
        if tag == "Legendre":
            coefficients = np.empty(len(rec.coefficients) + 1)
            coefficients[0] = 1.0
            coefficients[1:] = rec.coefficients
            functions.append(Legendre(coefficients=coefficients,
                                      outerDomainValue=float(rec.energy), index=i))
        else:
            functions.append(_tabulated(rec, i, code1))
    return fromEndfTab2(functions, _pairs(block.interpolation), axes=axes)


def _tabulated(rec, index: int, code1=None):
    """One region is an ``XYs1d``; a ``regions1d`` needs two (``gnds.xsd``).

    A code 1 is held lin-lin and its declared pairs go to ``code1[str(index)]``.
    """
    declared = _pairs(rec.interpolation)
    if any(c == 1 for _, c in declared):
        code1[str(index)] = [[int(b), int(c)] for b, c in declared]
    function = Regions1d.fromEndfRegions(rec.mu.copy(), rec.probability.copy(),
                                         [(b, 2 if c == 1 else c) for b, c in declared])
    if len(function.function1ds) == 1:
        function = function.function1ds[0]
    function.outerDomainValue = float(rec.energy)
    function.index = index
    return function


def _pairs(interp: Interpolation) -> List[Tuple[int, int]]:
    return list(zip(interp.nbt, interp.codes))


def _refuseHistogram(interp: Interpolation, what: str) -> None:
    if 1 in interp.codes:
        raise G4NDLUnsupportedError(
            f"{what}: interpolation code 1 (histogram). Geant4 evaluates it as "
            f"lin-lin (G4ParticleHPInterpolator.hh, the Histogram call is "
            f"commented out), so the file and its consumer disagree on what it "
            f"means; kika refuses it until a real file needs a decision")


def _checkTemperatures(record: ElasticFSRecord, where: str,
                       report: ConversionReport) -> None:
    if record.tempdeps.any() or record.temperatures.any():
        report.warn(
            f"{where}: {int(np.count_nonzero(record.temperatures))} incident "
            f"energies carry a non-zero temperature T and "
            f"{int(np.count_nonzero(record.tempdeps))} a non-zero tempdep. Both "
            f"are ENDF MF4's T and LT; Geant4 ignores them. Kept in the "
            f"provenance, not applied")


# ------------------------------------------------------------ bookkeeping

def _sha256(path: Optional[Path]) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None else None


def _provenance(cs: CrossSectionRecord, fs: ElasticFSRecord, library) -> G4NDLProvenance:
    leg, tab = fs.legendre, fs.tabulated
    return G4NDLProvenance(
        library=str(library.root) if library is not None else None,
        libraryName=library.root.name if library is not None else None,
        crossSectionPath=str(cs.path) if cs.path is not None else None,
        finalStatePath=str(fs.path) if fs.path is not None else None,
        crossSectionSha256=_sha256(cs.path),
        finalStateSha256=_sha256(fs.path),
        crossSectionHeader=cs.header,
        finalStateHeader=fs.header,
        bookkeeping=cs.bookkeeping,
        repFlag=fs.repFlag,
        targetMass=fs.targetMass,
        frameFlag=fs.frameFlag,
        frameFlag2=fs.frameFlag2,
        legendreTemperatures=[r.temperature for r in leg.records] if leg else [],
        legendreTempdeps=[r.tempdep for r in leg.records] if leg else [],
        tabulatedTemperatures=[r.temperature for r in tab.records] if tab else [],
        tabulatedTempdeps=[r.tempdep for r in tab.records] if tab else [],
    )


_PROCESS_NAMES = {"Elastic": "the elastic channel (MT2)",
                  "Inelastic": "the inelastic channels",
                  "Capture": "the capture (MT102)",
                  "Fission": "the fission (MT18 and its chances)"}


def _reportPartialRead(library, report: ConversionReport,
                       readDirectories: Tuple[str, ...] = ("Elastic",),
                       absentDirectories: Tuple[str, ...] = ()) -> None:
    if library is None:
        return
    held = " and ".join(_PROCESS_NAMES.get(d, d) for d in readDirectories)
    for name in library.presentTopLevel():
        if name not in readDirectories and name not in absentDirectories:
            what = ("which kika reads but was not asked for" if name in _PROCESS_NAMES
                    else "which kika does not read yet")
            report.unsupportedNode(
                f"the library also holds {name}/, {what}: this suite is {held} only")
