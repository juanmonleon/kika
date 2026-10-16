"""FUDGE's ``ENDFconversionFlags``: what a GNDS file says about getting back to ENDF.

GNDS states the physics and not ENDF's bookkeeping, so FUDGE, which writes
most of the GNDS in circulation (556 of the 558 neutron evaluations carry this
node), leaves itself notes for the few things it cannot derive on the way back:
the ``QI`` of a continuum reaction whose GNDS ``Q`` is the ground-state one, the
MAT of an isomer, a product that goes to MF6 rather than MF4/MF5. Each note is a
``flags`` string attached to an ``href`` into the suite.

Modelled rather than dropped because the ENDF writer needs it, and typed
rather than kept as XML. It lives in the suite's ``applicationData``,
as it does in the file, under FUDGE's institution label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["ENDF_CONVERSION_INSTITUTION", "EndfConversionFlags", "parseFlags",
           "QI_FLAGGED_MTS", "reactionHref", "reactionQI"]

#: The ``institution`` label FUDGE files the node under.
ENDF_CONVERSION_INSTITUTION = "LLNL"

#: The MTs whose GNDS ``Q`` is the ground-state Q (ENDF's QM) and whose QI is a
#: ``QI=`` flag: every continuum and MT103-107. Measured on the 558 neutron files
#: of NNDC's ENDF/B-VIII.1 GNDS (2026-10-16): every reaction with one of these
#: MTs carries the flag, whether or not QI equals QM (2 041 of 2 041), and no
#: other MT does.
QI_FLAGGED_MTS = frozenset({91, 103, 104, 105, 106, 107, 649, 699, 749, 799, 849, 891})


def reactionHref(reaction) -> str:
    """The xPath FUDGE's ``ENDFconversionFlags`` use to name *reaction*."""
    from .reactions import CrossSectionSum

    if isinstance(reaction, CrossSectionSum):
        container = "sums/crossSectionSums/crossSectionSum"
    else:
        container = "reactions/reaction"
    return f"/reactionSuite/{container}[@label='{reaction.label}']"


def reactionQI(reaction, suite) -> Optional[float]:
    """The reaction's QI: the Q of the state it leaves the residual in.

    What kinematics, thresholds and a Monte Carlo library's ``Q_value`` need.
    A GNDS ``Q`` is that QI except where FUDGE's convention makes it the
    ground-state Q (:data:`QI_FLAGGED_MTS`: Fe-56 MT91 has ``Q`` 0 and QI
    -2.334 MeV), and there the ``QI=`` flag holds it. ``None`` when the
    reaction states no Q.
    """
    flags = EndfConversionFlags.of(suite) if suite is not None else None
    if flags:
        stated = flags.qiFor(reactionHref(reaction))
        if stated is not None:
            return stated
    q = getattr(getattr(reaction, "outputChannel", None), "Q", None)
    value = getattr(q, "value", None)
    return None if value is None else float(value)


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

    def qiFor(self, reactionHref: str) -> Optional[float]:
        """The ``QI=`` flag stated for a reaction's ``outputChannel/Q``, else ``None``."""
        value = self.flagsFor(reactionHref + "/outputChannel/Q").get("QI")
        return None if value is None else float(value)

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
