"""The controlled vocabularies of the SINBAD data format, and what they mean here.

These are the lists of ``Structures/Data_structures.md`` §5, transcribed. They
are **not** enforced on read: a file written against a later draft must still
open, and a value outside a list is passed through unchanged. They exist so
that code (and tab-completion) can name a value without spelling a string, and
so that :func:`kika.sinbad.vocabularies` can answer "what can this attribute
say?" without a copy of the specification.

The one place a vocabulary changes behaviour is
:data:`CORRELATION_SCOPE_IS_GLOBAL`, which decides how a budget component
correlates two points; see :mod:`kika.sinbad.uncertainty`.
"""

from typing import Dict, Tuple

__all__ = [
    "FORMAT_VERSIONS",
    "BENCHMARK_ROOT",
    "CALCULATIONS_ROOT",
    "CALCULATIONS_DIRNAME",
    "QUANTITIES",
    "NATURES",
    "COLUMN_ROLES",
    "COLUMN_KINDS",
    "CORRELATION_SCOPES",
    "CORRELATION_SCOPE_IS_GLOBAL",
    "PROVENANCES",
    "REFERENCE_STATUSES",
    "EXTERNAL_FILE_ROLES",
    "CORRECTION_DIRECTIONS",
    "COMPARISON_OPERATORS",
    "ABSENCE_KINDS",
    "CONTENT_TAGS",
    "vocabularies",
]

#: Draft versions this reader has been written against. A file declaring
#: anything else still opens; :attr:`SinbadBenchmark.format_version` reports
#: what it said.
FORMAT_VERSIONS: Tuple[str, ...] = ("0.1", "0.2", "0.3")

#: Root element of a benchmark file and of a calculations file (§1).
BENCHMARK_ROOT = "sinbad"
CALCULATIONS_ROOT = "sinbadCalculations"

#: Calculations files live in this directory beside the benchmark file.
CALCULATIONS_DIRNAME = "calculations"

#: §5. What a data object holds, as a phrase.
QUANTITIES: Tuple[str, ...] = (
    "reaction rate", "particle spectrum", "conventional flux", "dose rate",
    "activity", "count rate", "pulse-height distribution",
    "multiplicity distribution", "event list", "ratio",
    "sensitivity coefficient", "detector response function",
    "energy group structure", "material composition", "assembly geometry",
    "source distribution", "source strength", "detector specification",
    "correction factor", "transport model file", "covariance matrix",
)

#: §5. Where the numbers came from.
NATURES: Tuple[str, ...] = ("measured", "evaluated", "calculated", "derived")

#: §2.3. What a table column is for. ``value`` is the default.
COLUMN_ROLES: Tuple[str, ...] = ("independent", "value", "uncertainty")

#: §2.3. An uncertainty column is one contribution or the combined figure.
COLUMN_KINDS: Tuple[str, ...] = ("component", "total")

#: §3.2. How far a single uncertainty component correlates.
CORRELATION_SCOPES: Tuple[str, ...] = (
    "none", "within-detector", "within-entry", "within-campaign",
    "within-configuration",
)

#: Of those, the ones that correlate every point in the file with every other.
#: ``within-detector`` is the interesting case and is resolved per point, by
#: the object the component points at; ``none`` contributes on the diagonal
#: only.
CORRELATION_SCOPE_IS_GLOBAL: Dict[str, bool] = {
    "none": False,
    "within-detector": False,
    "within-entry": True,
    "within-campaign": True,
    "within-configuration": True,
}

#: §5. Whether a value was read off the entry or worked out from it.
PROVENANCES: Tuple[str, ...] = ("reported", "inferred", "missing")

#: §5. Why something a reference points at is not here.
REFERENCE_STATUSES: Tuple[str, ...] = (
    "stated, absent", "not distributed", "not reported",
)

#: §5. What a file of the entry is.
EXTERNAL_FILE_ROLES: Tuple[str, ...] = (
    "abstract", "description", "reference", "figure", "inputDeck",
    "outputListing", "archive", "fileList", "library", "rawData",
)

#: §3.1. How a correction is applied to the values.
CORRECTION_DIRECTIONS: Tuple[str, ...] = ("subtract", "add", "multiply")

#: §4.9. How a calculation is compared with the measurement.
COMPARISON_OPERATORS: Tuple[str, ...] = ("ratio", "difference", "relative")

#: §4.10. Why something is not in the entry at all.
ABSENCE_KINDS: Tuple[str, ...] = (
    "not stated", "not reported", "measured, not reported",
)

#: §3. The child elements a ``dataObject`` may hold, and which class reads each.
#: The mapping itself lives in :mod:`kika.sinbad.content`; this is the list, in
#: the order the specification gives it, so that an unknown child can be named
#: as such.
CONTENT_TAGS: Tuple[str, ...] = (
    "double", "grid", "table", "gridded1d", "gridded2d", "gridded3d",
    "XYs1d", "polynomial1d", "externalFile", "materials", "geometry",
    "functionalForm", "factorChain",
)


def vocabularies() -> Dict[str, Tuple[str, ...]]:
    """
    Return the controlled vocabularies of the format, keyed by attribute.

    Returns
    -------
    dict
        Attribute name -> the values ``Data_structures.md`` §5 allows.

    Examples
    --------
    >>> from kika.sinbad import vocabularies
    >>> vocabularies()["nature"]
    ('measured', 'evaluated', 'calculated', 'derived')
    """
    return {
        "quantity": QUANTITIES,
        "nature": NATURES,
        "role": COLUMN_ROLES,
        "kind": COLUMN_KINDS,
        "correlationScope": CORRELATION_SCOPES,
        "provenance": PROVENANCES,
        "status": REFERENCE_STATUSES,
        "externalFile/role": EXTERNAL_FILE_ROLES,
        "direction": CORRECTION_DIRECTIONS,
        "operator": COMPARISON_OPERATORS,
        "absence/kind": ABSENCE_KINDS,
    }
