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


def _mat(context: DerivationContext, report) -> int:
    """MAT, in the order: the caller's, FUDGE's ``MAT=`` flag, the ENDF table."""
    if context.mat is not None:
        return int(context.mat)
    flag = context.conversionFlags.get("/reactionSuite", "")
    for item in flag.split(","):
        key, _, value = item.strip().partition("=")
        if key == "MAT" and value.strip().lstrip("-").isdigit():
            return int(value)
    mat = matFromTable(context.suite)
    if mat is None:
        raise ValueError(
            f"no ENDF MAT for the target {context.suite.target!r}: the suite was "
            f"not read from ENDF, its file states none, and ENDF-6's MAT table "
            f"has no entry for it. Pass mat= explicitly."
        )
    return mat


def deriveSuite(suite, path, context: DerivationContext, report):
    """The suite's provenance: MAT, ZA, AWR and the nineteen MF1/451 fields."""
    from kika.nuclear_data.model import EndfProvenance

    from ..mf1_header import synthesiseMF1Header

    fields, za, awr = synthesiseMF1Header(suite, report)
    context.mat = _mat(context, report)
    context.za, context.awr = za, awr
    return EndfProvenance(sourceFormat=DERIVED, mat=context.mat, za=za, awr=awr,
                          headerFields=fields)


def _isSuite(node) -> bool:
    from kika.nuclear_data.model import ReactionSuite

    return isinstance(node, ReactionSuite)


register("suite", _isSuite, deriveSuite)
