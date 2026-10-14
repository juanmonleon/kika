"""What a read of an ACE table could not take, or took with a caveat.

``read_ace`` is strict by default: a block or reaction that cannot be read
raises :class:`AceReadError` naming it, because a table that silently lacks a
reaction is worse than one that does not load. ``read_ace(..., strict=False)``
keeps going with what could be read, and every gap lands in
``ace.read_report`` as well as in the log, so it can be inspected rather than
scrolled past.

Notes are different: they are things the file legitimately does that a user
may want to know about (a repeated incident energy, an interpolation law kika
treats like MCNP and OpenMC do rather than as ENDF says). They never raise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


class AceReadError(ValueError):
    """A part of an ACE table could not be read (``read_ace(strict=True)``)."""


@dataclass(frozen=True)
class ReadIssue:
    block: str            # ACE block, as the manual names it: "DLW", "SIG", "DNED"...
    where: str            # what in the block: "MT=16", "group 3", "particle 1 MT=5"
    message: str

    def __str__(self) -> str:
        return f"{self.block} {self.where}: {self.message}" if self.where else f"{self.block}: {self.message}"


@dataclass
class ReadReport:
    """The gaps (``lost``) and caveats (``notes``) of one read."""
    strict: bool = True
    lost: List[ReadIssue] = field(default_factory=list)
    notes: List[ReadIssue] = field(default_factory=list)
    # (block, where) being read, for the law parsers, which do not know it
    context: Optional[tuple] = field(default=None, repr=False, compare=False)

    def __bool__(self) -> bool:
        return bool(self.lost or self.notes)

    def __repr__(self) -> str:
        if not self:
            return "ReadReport: complete, no notes"
        lines = [f"ReadReport: {len(self.lost)} not read, {len(self.notes)} notes"]
        lines += [f"  lost  {issue}" for issue in self.lost]
        lines += [f"  note  {issue}" for issue in self.notes]
        return "\n".join(lines)


def _report_of(ace) -> Optional[ReadReport]:
    return getattr(ace, "read_report", None)


def _issue(report, block, where, message) -> ReadIssue:
    if block is None:
        block, where = (report.context if report is not None and report.context else ("?", ""))
    return ReadIssue(block, where, message)


def lost(ace, block: Optional[str], where: Optional[str], message: str) -> None:
    """Part of the table could not be read: raise if strict, else record it.

    ``block=None`` takes the block and place being read from the report's
    ``context`` (set by the block reader around a law).
    """
    report = _report_of(ace)
    issue = _issue(report, block, where, message)
    if report is None or report.strict:
        raise AceReadError(str(issue))
    report.lost.append(issue)
    logger.warning(str(issue))


def note(ace, block: Optional[str], where: Optional[str], message: str) -> None:
    """Something legitimate but worth knowing; recorded, never raised."""
    report = _report_of(ace)
    if report is not None:
        report.notes.append(_issue(report, block, where, message))
