"""``ConversionReport`` — nothing converts silently.

A decoder or an encoder records in one what it did beyond producing its result.
The rule it serves: while MF coverage is partial, the writer emits only what kika
owns and declares the gaps, because a structurally valid, physically incomplete
reactionSuite carries authority it has not earned. This is the object that does
the declaring.

Four kinds, and the distinction between them is the point:

``warnings``       something looked odd but nothing was lost
``losses``         data in the source that is not in the result
``approximations`` data that is in the result but changed on the way
``unsupported``    constructs the reader recognised and cannot handle at all

The one that matters most is ``approximations``. A loss is visible — the field
is missing. An approximation looks like data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = ["ConversionReport"]


@dataclass
class ConversionReport:
    """What a conversion did, beyond producing its result.

    An instance is built empty, filled by the decoder or encoder through the methods
    below, and found afterwards on ``ReactionSuite.report``. ``len(report)`` is the
    number of entries in the four lists, and a report is always true in a boolean test.

    Attributes
    ----------
    warnings : list of str
        Something looked odd but nothing was lost.
    losses : list of str
        Data in the source that is not in the result.
    approximations : list of str
        Data that is in the result but changed on the way.
    unsupported : list of str
        Constructs the reader recognised and cannot handle at all.
    scopeExclusions : list of dict
        One ``{"kind", "message", "scopes"}`` entry per loss, approximation or
        unsupported node declared with ``unaffectedScopes``, naming the consumers
        for which that entry is irrelevant. ``kind`` is ``"losses"``,
        ``"approximations"`` or ``"unsupported"``.
    """

    warnings: List[str] = field(default_factory=list)
    losses: List[str] = field(default_factory=list)
    approximations: List[str] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)
    scopeExclusions: List[dict] = field(default_factory=list)

    def warn(self, message: str) -> None:
        """Record a warning (something odd, nothing lost).

        Parameters
        ----------
        message
            What looked odd.
        """
        self.warnings.append(message)

    def lost(self, message: str, *, unaffectedScopes=()) -> None:
        """Record data of the source that is missing from the result.

        Parameters
        ----------
        message
            What was lost.
        unaffectedScopes
            Names of the consumers for which this loss is irrelevant. Empty means
            it counts against every scope in :meth:`isCleanFor`.
        """
        self.losses.append(message)
        self._exclude('losses', message, unaffectedScopes)

    def approximated(self, message: str, *, unaffectedScopes=()) -> None:
        """Record data that is in the result but changed on the way.

        Parameters
        ----------
        message
            What was approximated.
        unaffectedScopes
            As in :meth:`lost`.
        """
        self.approximations.append(message)
        self._exclude('approximations', message, unaffectedScopes)

    def _exclude(self, kind: str, message: str, scopes) -> None:
        if scopes:
            self.scopeExclusions.append(dict(kind=kind, message=message, scopes=list(scopes)))

    def unsupportedNode(self, message: str, *, unaffectedScopes=()) -> None:
        """Declare a limitation, optionally proven irrelevant to named consumers.

        The global report remains unclean. Unannotated entries always block
        scoped checks; consumers must not infer scope by parsing messages.

        Parameters
        ----------
        message
            The construct that cannot be handled.
        unaffectedScopes
            As in :meth:`lost`.
        """
        self.unsupported.append(message)
        self._exclude('unsupported', message, unaffectedScopes)

    def extend(self, other: "ConversionReport") -> None:
        """Append every entry of *other*, scope exclusions included, to this report.

        Parameters
        ----------
        other
            The report to take the entries from; it is not modified.
        """
        self.warnings.extend(other.warnings)
        self.losses.extend(other.losses)
        self.approximations.extend(other.approximations)
        self.unsupported.extend(other.unsupported)
        self.scopeExclusions.extend(other.scopeExclusions)

    def isCleanFor(self, scope: str) -> bool:
        """Whether every issue is absent or explicitly irrelevant to ``scope``.

        Only losses, approximations and unsupported nodes count; warnings do not.
        Occurrences are counted, so a tagged entry cannot exempt another, identical
        but unannotated issue added by a different converter.

        Parameters
        ----------
        scope
            The name of a consumer, as given in ``unaffectedScopes`` when the issue
            was recorded.

        Returns
        -------
        bool
            True when each loss, approximation and unsupported message occurs no more
            often than it was declared irrelevant to *scope*.
        """
        from collections import Counter
        excluded = Counter((entry['kind'], entry['message'])
                           for entry in self.scopeExclusions
                           if scope in entry['scopes'])
        for kind in ('losses', 'approximations', 'unsupported'):
            for message, count in Counter(getattr(self, kind)).items():
                if count > excluded[(kind, message)]:
                    return False
        return True

    @property
    def isClean(self) -> bool:
        """Nothing was lost, approximated or refused. Warnings do not count."""
        return not (self.losses or self.approximations or self.unsupported)

    @property
    def isEmpty(self) -> bool:
        """True when the report has no entry at all, warnings included."""
        return self.isClean and not self.warnings

    def __len__(self) -> int:
        return len(self.warnings) + len(self.losses) + len(self.approximations) + len(self.unsupported)

    def __bool__(self) -> bool:
        # A report is an object that always exists; `if report:` must not read
        # "nothing to report" as "no report".
        return True

    def summary(self) -> str:
        """The counts per kind, such as ``"2 losses, 1 warnings"``, or ``"clean"``.

        Kinds are listed in the order losses, approximations, unsupported, warnings,
        and only the non-empty ones. ``"clean"`` is returned when all four are empty.
        """
        parts = []
        for name in ("losses", "approximations", "unsupported", "warnings"):
            entries = getattr(self, name)
            if entries:
                parts.append(f"{len(entries)} {name}")
        return ", ".join(parts) if parts else "clean"

    def __repr__(self) -> str:
        return f"ConversionReport({self.summary()})"
