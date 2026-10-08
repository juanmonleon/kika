"""G1: the suite's own ENDF bookkeeping — MAT, ZA, AWR and the MF1/451 header.

Everything else takes these three from here, because ENDF stamps them on every
section and a tape that disagreed with itself about AWR would be worse than one
that refused.
"""
from __future__ import annotations

from typing import Optional

from . import DERIVED, DerivationContext, register

__all__ = ["matFromTable", "deriveSuite"]


def matFromTable(suite) -> Optional[int]:
    """The ENDF MAT of the target from ENDF-6's assignment table, or ``None``.

    The ground state's MAT, or the isomer table's when the target id carries an
    ``_e<n>`` suffix. This is the number every library uses for the material,
    so it is derived, not assumed; a target the table does not know is refused.
    """
    from kika._constants import ZAID_TO_ENDF_MAT, ZAID_TO_ENDF_MAT_ISOMER
    from kika.nuclear_data.model.pops import zaFromPid

    target = str(getattr(suite, "target", "") or "")
    try:
        za = zaFromPid(target)
    except ValueError:
        return None
    suffix = target.split("_e", 1)[1] if "_e" in target else ""
    if suffix.isdigit() and int(suffix) > 0:
        return ZAID_TO_ENDF_MAT_ISOMER.get(za) or ZAID_TO_ENDF_MAT.get(za)
    return ZAID_TO_ENDF_MAT.get(za)


def _suiteFlag(context: DerivationContext, name: str) -> Optional[int]:
    """An integer flag FUDGE's ``ENDFconversionFlags`` state for ``/reactionSuite``."""
    flag = context.conversionFlags.get("/reactionSuite", "")
    for item in flag.split(","):
        key, _, value = item.strip().partition("=")
        if key == name and value.strip().lstrip("-").isdigit():
            return int(value)
    return None


def _isThermal(suite) -> bool:
    from kika.nuclear_data.model.thermal_scattering import TNSL_INTERACTION

    return getattr(suite, "interaction", None) == TNSL_INTERACTION


def _mat(context: DerivationContext, report) -> int:
    """MAT, in the order: the caller's, FUDGE's ``MAT=`` flag, the ENDF table."""
    if context.mat is not None:
        return int(context.mat)
    mat = _suiteFlag(context, "MAT")
    if mat is not None:
        return mat
    if _isThermal(context.suite):
        raise ValueError(
            f"no ENDF MAT for the thermal-scattering evaluation "
            f"{context.suite.target!r}: its file states none (FUDGE's MAT=…,ZA=… "
            f"note) and a TSL material has no entry in ENDF-6's MAT table. "
            f"Pass mat= explicitly.")
    mat = matFromTable(context.suite)
    if mat is None:
        raise ValueError(
            f"no ENDF MAT for the target {context.suite.target!r}: the suite was "
            f"not read from ENDF, its file states none, and ENDF-6's MAT table "
            f"has no entry for it. Pass mat= explicitly."
        )
    return mat


def _thermalZA(context: DerivationContext, report) -> Optional[int]:
    """The pseudo-ZA of a TSL evaluation's MF1/MF7 headers; ``None`` for any other.

    MAT + 100, FUDGE's rule (``toENDF6/reactionSuite.py``) and ENDF/B's
    convention. The ``ZA=`` of FUDGE's note is **not** this number: it is the
    principal scatterer's (1001 for s-CH4), which FUDGE looks a mass up by.
    GNDS has nowhere to keep the tape's own pseudo-ZA, so a library that spells
    it otherwise -- JEFF-4.0 writes Be metal as 4000 at MAT 26 -- does not come
    back as it was, and the report says so every time.
    """
    if not _isThermal(context.suite):
        return None
    report.approximated(
        f"MF1/451: the thermal-scattering pseudo-ZA is written as MAT + 100 = "
        f"{context.mat + 100} (ENDF/B's convention, FUDGE's rule); GNDS does not "
        f"carry the tape's own, and a library that spells it otherwise (JEFF-4.0: "
        f"Z·1000) does not get it back")
    return context.mat + 100


def deriveSuite(suite, path, context: DerivationContext, report):
    """The suite's provenance: MAT, ZA, AWR and the nineteen MF1/451 fields."""
    from kika.nuclear_data.model import EndfProvenance

    from ..mf1_header import synthesiseMF1Header

    context.mat = _mat(context, report)
    fields, za, awr = synthesiseMF1Header(suite, report,
                                          targetZA=_thermalZA(context, report))
    context.za, context.awr = za, awr
    return EndfProvenance(sourceFormat=DERIVED, mat=context.mat, za=za, awr=awr,
                          headerFields=fields)


def _isSuite(node) -> bool:
    from kika.nuclear_data.model import ReactionSuite

    return isinstance(node, ReactionSuite)


register("suite", _isSuite, deriveSuite)
