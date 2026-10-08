"""Typed, per-source-format provenance — the replacement for the ``metadata`` dict.

**What is wrong with the dict.** The flat classes carry ``metadata: Dict``, and
four packages reach into it by string key, so it is public API without being
declared as any. Worse, it mixes namespaces: ENDF puts ``mat, awr, qm, qi, lr,
interpolation_regions`` in it and ACE puts ``source_format, ace_zaid,
ace_extension, ...`` in the same dict, so "does this section have a Q value" is
answered by a ``.get`` that cannot distinguish *absent* from *zero*. That is
half of the Q = 0 defect.

**The int/float trap, recorded before it bites.** ENDF's fixed-format floats are
parsed to ``int`` whenever the value is integral, so on the committed Fe-56
slice ``emax`` is ``150000000`` and ``abundance`` is ``1``, both ``int``, while
on another tape the same fields are ``float``. Every numeric field below is
therefore annotated ``float`` and **coerced at construction**, which is the only
honest option: annotating ``int`` would be wrong on the next tape, and
annotating ``float`` without coercing would be a lie about what is stored.
``kika/nuclear_data/tests/test_metadata_contract.py`` pins the underlying
behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["Provenance", "EndfProvenance", "AceProvenance",
           "GndsProvenance", "G4NDLProvenance", "G4NDLInelasticProvenance"]


@dataclass
class Provenance:
    """Base: where a piece of data came from, in that format's own terms."""

    sourceFormat: str = "unknown"

    def asDict(self) -> Dict[str, object]:
        """Flat view, for the interop bridge and for diagnostics."""
        from dataclasses import asdict

        return asdict(self)


def _asFloat(value: object) -> Optional[float]:
    """Coerce an ENDF numeric field, keeping ``None`` distinct from ``0.0``."""
    return None if value is None else float(value)


def _asEndfInt(value: object) -> Optional[int]:
    """Round, do not truncate, an ENDF field that is conceptually an integer.

    ENDF writes ZA as a fixed-format float and kika's reader rebuilds it as
    ``mantissa * 10**exponent``, which is not exact: Th-232's ``9.023200+4``
    comes back as ``90231.99999999999``. ``int()`` on that is 90231, so a
    section decoded and re-encoded names a different nuclide — and because the
    flat classes truncate the same way, a gate that compares the model against
    them agrees on the wrong answer. Caught by comparing against the *file*.
    """
    if value is None:
        return None
    return int(round(float(value)))


@dataclass
class EndfProvenance(Provenance):
    """ENDF-6 bookkeeping that the physics model has no home for.

    Fields that *are* physics live in the model proper: the reaction Q value is
    ``outputChannel.Q``, the interpolation regions are the ``regions1d``, and
    the ZA is the nuclide in ``PoPs``. What remains here is genuinely
    format-specific.

    ``qm`` is the one arguable case. ENDF carries two Q values: ``QM``, the
    mass-difference Q for the ground state, and ``QI``, the Q of the particular
    level. GNDS §17.1.1 has a single ``Q``, which corresponds to ``QI``; ``QM``
    is derivable from the masses in ``PoPs``. So ``QI`` becomes the model's Q
    and ``QM`` stays here, to be written back byte-identically rather than
    recomputed.
    """

    sourceFormat: str = "endf"
    mat: Optional[int] = None
    awr: Optional[float] = None
    #: ENDF ZA = 1000*Z + A, as the section header wrote it.
    za: Optional[int] = None
    #: ENDF QM — the mass-difference Q. See the class docstring.
    qm: Optional[float] = None
    #: ENDF LR — the complex-breakup flag.
    lr: Optional[int] = None
    #: The (NBT, INT) pairs exactly as the file wrote them. The model's
    #: ``regions1d`` is built from these; keeping the originals means a
    #: round-trip does not depend on the reconstruction being lossless.
    interpolationRegions: List[Tuple[int, int]] = field(default_factory=list)
    #: Section header fields with no GNDS counterpart, kept verbatim so the
    #: encoder writes them back rather than recomputing them: MF1/451's
    #: nineteen, or MF34's ``ltt``. Untyped by design — this is the one place
    #: an ENDF-only field can live without the model growing a slot for it.
    headerFields: Dict[str, object] = field(default_factory=dict)
    evaluationInfo: Dict[str, str] = field(default_factory=dict)
    #: MF1/451's NWD descriptive records, data columns only (the ID columns are
    #: regenerated on the way out). ``evaluationInfo`` holds the seven fields
    #: the first two of these records are *parsed into*; this holds all NWD of
    #: them verbatim, because the rest — the free-text comment block an
    #: evaluator wrote — is parsed into nothing at all and is most of the
    #: section. Without it MF1/451 cannot be written back, only approximated.
    descriptiveText: List[str] = field(default_factory=list)
    #: MF1/451's NXC directory entries, ``(MF, MT, NC, MOD)``. NC is a **line
    #: count**, so this is only valid for the tape it was read from; after a
    #: section is replaced, ``kika.endf.writers.update_directory`` rebuilds it
    #: from the written file. Kept here so a round trip that changes nothing
    #: writes back what it read rather than a recomputation of it.
    directory: List[Tuple[int, int, int, int]] = field(default_factory=list)
    #: What :func:`kika.endf.check_covariances` (layer 1) found in the MF31/33/34
    #: section a covariance was decoded from: faults of the section *as written*
    #: -- an LS=1 triangle in a cross block, a partner block that is missing --
    #: which the assembled matrix can no longer show. They only inform; the
    #: sampling pre-flight shows them and traces its own findings back to them.
    #: Empty when the checks were not run, and on every other kind of section.
    covarianceFindings: Tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.awr = _asFloat(self.awr)
        self.qm = _asFloat(self.qm)
        if self.lr is not None:
            self.lr = int(self.lr)
        self.mat = _asEndfInt(self.mat)
        self.za = _asEndfInt(self.za)


@dataclass
class AceProvenance(Provenance):
    """ACE bookkeeping.

    ACE is a *processed* representation, not a peer evaluation: it has been
    reconstructed, Doppler-broadened and put on a grid, and it records no
    reaction Q values at all. The absence is declared here rather than defaulted
    to zero, which is the structural version of the phase 1 fix.
    """

    sourceFormat: str = "ace"
    zaid: Optional[int] = None
    extension: Optional[str] = None
    awr: Optional[float] = None
    comment: Optional[str] = None
    date: Optional[str] = None
    matid: Optional[int] = None
    formatVersion: Optional[str] = None
    temperatureMeV: Optional[float] = None

    def __post_init__(self) -> None:
        self.awr = _asFloat(self.awr)
        self.temperatureMeV = _asFloat(self.temperatureMeV)

    @property
    def carriesQValues(self) -> bool:
        """Always ``False``. Stated as a property so a caller can ask."""
        return False


@dataclass
class GndsProvenance(Provenance):
    """Where a GNDS-read suite came from, and — the load-bearing part — *that* it
    did.

    Thin on purpose: a GNDS file's own bookkeeping is already GNDS nodes, so
    unlike ENDF's MF1/451 header there is nothing here that the model cannot
    hold. What this exists for is the question the writer has to answer and
    could not: **did this evaluation come from a GNDS file, and if so which
    version did it declare?**

    ``ReactionSuite.format`` cannot answer it. Its default is ``"2.1"``, so an
    ENDF-decoded suite and a suite read from a 2.1 file are indistinguishable by
    that field, and ``kika.write`` would mirror a version nothing ever declared.
    "Has no provenance" cannot answer it either, and that was the first attempt:
    the ENDF adapter only fills ``suite.provenance`` when MF1/451 parses, so a
    tape with a damaged header decoded to a suite that then claimed a GNDS
    origin. This is the positive statement, made by the only reader entitled to
    make it.
    """

    sourceFormat: str = "gnds"
    #: The ``format`` attribute the file declared — ``"2.0"`` or ``"2.1"``.
    formatVersion: Optional[str] = None
    #: The file it was read from, when there was one.
    path: Optional[str] = None


@dataclass
class G4NDLProvenance(Provenance):
    """Where a G4NDL-read suite came from, and the tokens the model has no slot for.

    G4NDL is a *processed* library (Geant4 ParticleHP): one directory per
    process, one file per isotope. What is here is what Phase 6's writer needs
    to write an isotope back token for token, and what a reader needs to know
    which files were read:

    * the two bookkeeping integers of the cross section (``0 0`` in every real
      file) and the optional ``G4NDL <source>`` header of each file;
    * ``repFlag``, ``frameFlag`` and, for ``repFlag=0``, the second frame flag
      Geant4 reads (and uses, since it overwrites the first);
    * per incident energy, the temperature ``T`` and ``tempdep``. Both come
      from ENDF MF4's ``T`` and ``LT``; Geant4 reads ``tempdep`` into a local
      and never uses it, and stores ``T`` without reading it back
      (``G4ParticleHPElasticFS.cc:104-215``). Kept so they survive a round trip.

    ``targetMass`` is the consumer's mass ratio (ENDF's AWR), not ``A``.
    """

    sourceFormat: str = "g4ndl"
    #: The library root that was opened, and its directory name (``JEFF-4.0``).
    library: Optional[str] = None
    libraryName: Optional[str] = None
    crossSectionPath: Optional[str] = None
    finalStatePath: Optional[str] = None
    #: sha256 of each file as stored on disk (the ``.z`` when compressed).
    crossSectionSha256: Optional[str] = None
    finalStateSha256: Optional[str] = None
    crossSectionHeader: Optional[Tuple[str, str]] = None
    finalStateHeader: Optional[Tuple[str, str]] = None
    bookkeeping: Optional[Tuple[int, int]] = None
    repFlag: Optional[int] = None
    targetMass: Optional[float] = None
    frameFlag: Optional[int] = None
    frameFlag2: Optional[int] = None
    #: ``T`` and ``tempdep`` per incident energy, Legendre block then table.
    legendreTemperatures: List[float] = field(default_factory=list)
    legendreTempdeps: List[int] = field(default_factory=list)
    tabulatedTemperatures: List[float] = field(default_factory=list)
    tabulatedTempdeps: List[int] = field(default_factory=list)
    #: The declared ``(NBT, INT)`` of each table record (by index, from 0)
    #: whose μ interpolation had a code 1, which the model holds lin-lin as
    #: Geant4 evaluates it; the encoder writes it back (roadmap Fase 10, D10-3).
    tabulatedCode1: Dict[str, List[List[int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.targetMass = _asFloat(self.targetMass)


@dataclass
class G4NDLInelasticProvenance(Provenance):
    """One inelastic reaction read from G4NDL, and the tokens the model has no slot for.

    The inelastic counterpart of :class:`G4NDLProvenance`, attached to each
    reaction (and to the ``Inelastic/CrossSection`` sum) rather than to the
    suite, because one isotope's inelastic data is many files: one per
    channel directory ``F01`` … ``F36``, and a composite channel (``F01``,
    ``F23``-``F27``) holds several reactions in one file.

    ``sections`` lists this reaction's sections of the file in file order, one
    plain ``dict`` each: its ``position`` among all the file's sections, the
    header integers, and the bookkeeping of its body that the model has no
    place for (``targetMass`` of an angular or energy-angle section, ``T`` and
    ``tempdep`` per incident energy, the ZAP/AWP/LIP/LAW of each product...).
    A section the model does not carry at all — the photon sections 12-15 and
    any law kika has no form for — is kept whole as ``verbatim`` G4NDL text:
    plain data, not a parsed object, so that the model holds no format-shaped
    class (``kika/g4ndl/inelastic_decode.py``).
    """

    sourceFormat: str = "g4ndl"
    library: Optional[str] = None
    libraryName: Optional[str] = None
    #: ``"F01"`` … ``"F36"``, or ``"CrossSection"`` for the total.
    channel: Optional[str] = None
    path: Optional[str] = None
    sha256: Optional[str] = None
    header: Optional[Tuple[str, str]] = None
    composite: Optional[bool] = None
    #: A base-FS file's ``Qvalue`` and ``dummy``, read once after its first section.
    Qvalue: Optional[float] = None
    Qdummy: Optional[int] = None
    sfType: Optional[int] = None
    #: How many sections the whole file has, to tell a complete set from a subset.
    nSections: Optional[int] = None
    sections: List[Dict[str, object]] = field(default_factory=list)
    #: ``Inelastic/CrossSection`` only: the two bookkeeping integers.
    bookkeeping: Optional[Tuple[int, int]] = None
    #: A sum only (the ``inelastic`` total, MT4, MT103-107): SHA-256 of its σ
    #: and of its parts' σ as read. What tells the writer later whether the
    #: sum or its parts were edited (roadmap G4NDL Fase 10, D10-1).
    crossSectionDigest: Optional[str] = None
    partsDigest: Optional[str] = None
