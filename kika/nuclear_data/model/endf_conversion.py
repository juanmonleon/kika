"""FUDGE's ``ENDFconversionFlags``: what a GNDS file says about getting back to ENDF.

GNDS states the physics and not ENDF's bookkeeping, so FUDGE, which writes
most of the GNDS in circulation (556 of the 558 neutron evaluations carry this
node), leaves itself notes for the few things it cannot derive on the way back:
the ``QI`` of a continuum reaction whose GNDS ``Q`` is the ground-state one, the
MAT of an isomer, a product that goes to MF6 rather than MF4/MF5. Each note is a
``flags`` string attached to an ``href`` into the suite.

Modelled rather than dropped because the ENDF writer needs it
(``docs/library/gnds_to_endf_plan.md`` D2), and typed rather than kept as XML
(``gnds_endf_conflicts.md`` §6.3). It lives in the suite's ``applicationData``,
as it does in the file, under FUDGE's institution label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["ENDF_CONVERSION_INSTITUTION", "EndfConversionFlags", "parseFlags"]

#: The ``institution`` label FUDGE files the node under.
ENDF_CONVERSION_INSTITUTION = "LLNL"


def parseFlags(text: str) -> Dict[str, Optional[str]]:
    """``"QI=-4999990.0,MF6"`` → ``{"QI": "-4999990.0", "MF6": None}``."""
    flags: Dict[str, Optional[str]] = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        flags[key.strip()] = value.strip() if sep else None
    return flags


@dataclass
class EndfConversionFlags:
    """The ``(href, flags)`` pairs of one ``ENDFconversionFlags``, in file order."""

    conversions: List[Tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.conversions)

    def flagsFor(self, href: str) -> Dict[str, Optional[str]]:
        """Every flag stated for *href*, merged; empty when there are none."""
        merged: Dict[str, Optional[str]] = {}
        for target, flags in self.conversions:
            if target == href:
                merged.update(parseFlags(flags))
        return merged

    @staticmethod
    def of(suite) -> Optional["EndfConversionFlags"]:
        """The suite's flags, when its ``applicationData`` holds them."""
        for entry in getattr(getattr(suite, "applicationData", None), "entries", ()):
            if isinstance(entry, EndfConversionFlags):
                return entry
        return None
