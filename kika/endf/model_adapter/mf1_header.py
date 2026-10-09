"""MF1/451's numeric header, derived from the model when no ENDF read kept it.

A suite decoded from ENDF carries the nineteen header fields verbatim in
``EndfProvenance.headerFields`` and :func:`encodeMF1MT451` writes those back
unchanged. A suite decoded from **GNDS** has no such record, and this module
derives the fields from what the model does state — the way FUDGE's
``toENDF6`` does (``brownies/legacy/toENDF6/reactionSuite.py``), because there
is no other place they could come from:

=========================  =================================================
``NLIB``                   ``evaluated.library`` through ENDF-102's NLIB
                           table (``"JEFF"`` → 2), FUDGE's own convention
``NVER``/``LREL``/``NMOD`` ``evaluated.version`` spelled ``"NVER.LREL.NMOD"``
``AWR``                    the target's PoPs mass over the neutron's; kika's
                           mass table when PoPs carries none
``AWI``                    the same for the projectile (1 for a neutron)
``EMAX``/``TEMP``          ``projectileEnergyDomain.max``, ``temperature``
``LRP``                    what ``resonances`` holds
``LFI``                    whether any reaction is fission
``LIS``/``ELIS``/``STA``   the target's nuclear level and half-life
``NSUB``                   10·IPART + ITYPE from the projectile
``NFOR``/``LDRV``          6, and 0 for an ``evaluated`` style
=========================  =================================================

**Where this departs from FUDGE, and why.** FUDGE prints a line and carries on
when a field has to be guessed; here every field that is not derived from data
goes into the :class:`ConversionReport` by name, so a reader of the tape can
tell what the file said from what the converter assumed. ``AWR`` falls back to
kika's mass table, reported, where FUDGE falls back silently. The comment block
is the file's own (``documentation/endfCompatible``) and, when there is none,
the five identification records with blank columns rather than FUDGE's
``LLNL``/``Unknown`` placeholders. And the derivation is gated against real tapes:
``test_header_synthesis.py`` derives the header of ENDF-decoded suites with the
provenance hidden and compares it field by field with what the file states.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

__all__ = ["NLIB_LIBRARIES", "libraryFromNlib", "nlibFromLibrary",
           "versionFromHeader", "synthesiseMF1Header"]

#: ENDF-102 §1.1's NLIB table, spelled the way FUDGE's ``endf_endl.NLIBs``
#: spells it, so the ``library`` attribute of a style means the same thing in a
#: GNDS file written by kika and in one written by FUDGE.
NLIB_LIBRARIES: Dict[int, str] = {
    0: "ENDF/B", 1: "ENDF/A", 2: "JEFF", 3: "EFF", 4: "ENDF/B (HE)",
    5: "CENDL", 6: "JENDL", 17: "TENDL", 18: "ROSFOND", 21: "SG-23",
    31: "INDL/V", 32: "INDL/A", 33: "FENDL", 34: "IRDF",
    35: "BROND (IAEA version)", 36: "INGDB-90", 37: "FENDL/A", 38: "IAEA/PD",
    41: "BROND",
}

#: Reactions whose presence makes LFI=1 (ENDF-102 §1.1: "fission data given").
_FISSION_MTS = {18, 19, 20, 21, 38}


def libraryFromNlib(nlib) -> str:
    """NLIB → the style's ``library``. An NLIB the table lacks keeps its number."""
    if nlib is None:
        return ""
    return NLIB_LIBRARIES.get(int(nlib), str(int(nlib)))


def nlibFromLibrary(library) -> Optional[int]:
    """The inverse; also reads the bare number kika wrote before 2026-10-08."""
    if library is None:
        return None
    text = str(library).strip()
    for nlib, name in NLIB_LIBRARIES.items():
        if name.lower() == text.lower():
            return nlib
    try:
        return int(text)
    except ValueError:
        return None


def versionFromHeader(nver, lrel, nmod) -> str:
    """``"NVER.LREL.NMOD"``, the spelling FUDGE writes and reads back."""
    return ".".join(str(int(value or 0)) for value in (nver, lrel, nmod))


def _parseVersion(version) -> Optional[Tuple[int, int, int]]:
    parts = str(version or "").strip().split(".")
    try:
        numbers = [int(part) for part in parts if part != ""]
    except ValueError:
        return None
    if not numbers or len(numbers) > 3:
        return None
    return tuple(numbers + [0] * (3 - len(numbers)))


def _massAmu(particle) -> Optional[float]:
    mass = getattr(particle, "mass", None)
    if mass is None:
        return None
    try:
        return float(mass.convertedTo("amu").value)
    except Exception:
        return None


def _awr(pops, pid: str, za: int, neutronAmu: float, what: str, report,
         nuclear: bool = False) -> float:
    """Mass ratio to the neutron, from PoPs when it says, else kika's table.

    ``nuclear`` for a projectile: ENDF's AWI of a hydrogen or helium isotope is
    its nuclear mass, while PoPs carries the atomic one (``light_masses``).
    """
    from kika._constants import ATOMIC_MASS

    particle = pops.particles.get(pid) if pops is not None else None
    amu = _massAmu(particle)
    if amu is None:
        amu = ATOMIC_MASS.get(za)
        if amu is None:
            raise ValueError(
                f"no mass for the {what} {pid!r}: PoPs carries none and kika's "
                f"mass table has no ZA {za}, so MF1/451's mass ratio cannot be "
                f"written"
            )
        report.approximated(
            f"MF1/451: the {what} mass ({pid}) is not in PoPs, so it was taken "
            f"from kika's atomic mass table ({amu} amu)"
        )
    if nuclear:
        from .light_masses import ELECTRON_MASS_AMU, _BINDING_AMU, isLight

        if isLight(za):
            amu -= ELECTRON_MASS_AMU * (za // 1000) + _BINDING_AMU[za]
    return amu / neutronAmu


def _evaluatedStyle(suite):
    from kika.nuclear_data.model import Evaluated

    for style in getattr(suite, "styles", None) or []:
        if isinstance(style, Evaluated):
            return style
    return None


def _ipart(projectile: str) -> int:
    from kika.nuclear_data.model.pops import zaFromPid

    return zaFromPid(projectile)


def _lrp(resonances) -> int:
    """ENDF-102 §1.1's LRP from what ``resonances`` holds — FUDGE's rule.

    -1: no MF2 at all. 0: MF2 holds only the scattering radius. 1: resolved or
    unresolved parameters whose cross sections are to be added to MF3 — which
    is what kika's model means by carrying them, since it reconstructs from
    them. FUDGE's LRP=2 ("for information only") comes from a
    ``reconstructCrossSection=false`` the model has no field for.
    """
    if resonances is None:
        return -1
    if not getattr(resonances, "resolved", None) and getattr(resonances, "unresolved", None) is None:
        return 0
    return 1


def _level(suite, target: str, report) -> Tuple[int, float, int, int]:
    """(LIS, ELIS, STA, LISO) for the target."""
    particle = suite.PoPs.particles.get(target) if suite.PoPs is not None else None
    lis = int(getattr(particle, "nuclearLevel", 0) or 0)
    if not lis and "_e" in target:
        lis = int(target.split("_e")[1] or 0)

    # FUDGE's rule, and deliberately not "any finite half-life": STA is the
    # evaluator's flag, not a decay fact. U-235 and U-238 are unstable and every
    # library writes STA=0 for them, so a numeric half-life must not set it. The
    # ENDF decoder writes the literal "unstable" for STA=1, as FUDGE does, which
    # is what makes this field go round.
    halflife = getattr(particle, "halflife", None)
    marked = isinstance(halflife, str) and halflife.lower() == "unstable"
    sta = 1 if marked or lis else 0

    elis = 0.0
    liso = 0
    if lis:
        energy = getattr(particle, "energy", None)
        if energy is not None:
            elis = float(energy.convertedTo("eV").value)
        else:
            report.approximated(
                f"MF1/451: the target {target} is level {lis}, but PoPs gives no "
                f"level energy, so ELIS is written as 0"
            )
        liso = 1
        report.approximated(
            f"MF1/451: the target {target} is an excited level and kika does "
            f"not read PoPs' metastable aliases, so LISO is written as 1"
        )
    return lis, elis, sta, liso


def synthesiseMF1Header(suite, report, *, targetZA: Optional[int] = None
                        ) -> Tuple[Dict[str, object], int, float]:
    """``(headerFields, ZA, AWR)`` derived from *suite*, for :func:`encodeMF1MT451`.

    *targetZA* overrides the ZA spelled by the target id. A thermal-scattering
    target (``tnsl-…``) spells none, so for one it is required.

    Raises when the suite has no ``evaluated`` style: TEMP, EMAX and the library
    all live there, and a header without them would be invented, not derived.
    """
    from kika._constants import NEUTRON_MASS_AMU
    from kika.nuclear_data.model.pops import zaFromPid

    style = _evaluatedStyle(suite)
    if style is None:
        raise ValueError(
            "the suite has no evaluated style, so MF1/451 cannot be derived: "
            "the library, version, temperature and energy domain all live there"
        )

    from kika.nuclear_data.model.thermal_scattering import TNSL_INTERACTION

    target = suite.target
    thermal = getattr(suite, "interaction", None) == TNSL_INTERACTION
    if thermal and targetZA is None:
        raise ValueError(
            f"the thermal-scattering target {target!r} spells no ZA: its MF1/451 "
            f"pseudo-ZA has to be given (derive/suite.py writes MAT + 100)")
    za = int(targetZA) if targetZA is not None else zaFromPid(target)
    neutronAmu = _massAmu(suite.PoPs.particles.get("n")) or NEUTRON_MASS_AMU
    awr = _awr(suite.PoPs, target, za, neutronAmu, "target", report)

    projectile = suite.projectile or "n"
    ipart = _ipart(projectile)
    awi = 1.0 if projectile == "n" else (
        0.0 if ipart == 0 else _awr(suite.PoPs, projectile, ipart, neutronAmu,
                                     "projectile", report, nuclear=True))
    mts = {getattr(reaction, "ENDF_MT", None) for reaction in suite.reactions}
    # ENDF-102 §1.1: ITYPE 2 is thermal neutron scattering (NSUB = 12).
    itype = 2 if thermal else (
        3 if any(mt is not None and 500 <= mt < 573 for mt in mts) else 0)

    nlib = nlibFromLibrary(style.library)
    if nlib is None:
        nlib = -1
        report.approximated(
            f"MF1/451: library {style.library!r} is not in ENDF-102's NLIB table, "
            f"so NLIB is written as -1 (FUDGE's 'unknown')"
        )
    version = _parseVersion(style.version)
    if version is None:
        version = (0, 0, 0)
        report.approximated(
            f"MF1/451: version {style.version!r} is not 'NVER.LREL.NMOD', so "
            f"NVER, LREL and NMOD are written as 0"
        )
    nver, lrel, nmod = version

    domain = style.projectileEnergyDomain
    if domain is not None:
        from kika.nuclear_data.model import PhysicalQuantity

        emax = float(PhysicalQuantity(value=domain.max, unit=domain.unit or "eV")
                     .convertedTo("eV").value)
    else:
        maxima = [getattr(reaction.crossSection[label], "domainMax", None)
                  for reaction in suite.reactions for label in reaction.crossSection]
        maxima = [value for value in maxima if value is not None]
        if not maxima:
            raise ValueError("MF1/451 needs EMAX and the suite states no energy domain")
        emax = float(max(maxima))
        report.approximated(
            "MF1/451: the evaluated style has no projectileEnergyDomain, so EMAX "
            "is the largest cross-section domain"
        )

    temperature = style.temperature
    temp = 0.0 if temperature is None else float(temperature.convertedTo("K").value)
    if temperature is None:
        report.approximated("MF1/451: the evaluated style has no temperature; TEMP is 0 K")

    lis, elis, sta, liso = _level(suite, target, report)
    lfi = int(any(mt in _FISSION_MTS for mt in mts) or bool(len(suite.fissionComponents)))

    fields = {
        "lrp": _lrp(suite.resonances), "lfi": lfi, "nlib": nlib, "nmod": nmod,
        "elis": elis, "sta": sta, "lis": lis, "liso": liso,
        "nfor": 6, "awi": awi, "emax": emax, "lrel": lrel,
        "nsub": 10 * ipart + itype, "nver": nver, "ldrv": 0, "temp": temp,
    }
    return fields, za, awr
