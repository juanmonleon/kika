"""Errors raised while reading a SINBAD benchmark file.

All of them derive from :class:`SinbadError`, so a caller that only wants to
know whether a file could be read can catch one name.
"""

__all__ = [
    "SinbadError",
    "SinbadFormatError",
    "LabelNotFoundError",
    "AmbiguousLabelError",
    "ContentTypeError",
    "BenchmarkMismatchError",
    "IncompleteBudgetError",
]


class SinbadError(Exception):
    """Base class for everything this subpackage raises."""


class SinbadFormatError(SinbadError):
    """The file is not a SINBAD benchmark or calculations file, or is malformed."""


class LabelNotFoundError(SinbadError, KeyError):
    """No object in the benchmark (or its calculations) carries this label.

    Derives from :class:`KeyError` as well, because labels are looked up with
    ``benchmark["label"]`` and a caller may reasonably catch either name.
    """

    def __str__(self) -> str:  # KeyError would quote the message
        return self.args[0] if self.args else ""


class AmbiguousLabelError(SinbadError, KeyError):
    """A fragment matched more than one object and no single one was meant.

    Look-up accepts a fragment of a label as a convenience, so this is what a
    caller gets for ``b["reactionRate"]`` when the entry holds five of them.
    Like :class:`LabelNotFoundError` it is a :class:`KeyError` as well, so
    code that guards a look-up with one name still catches it.
    """

    def __str__(self) -> str:  # KeyError would quote the message
        return self.args[0] if self.args else ""


class ContentTypeError(SinbadError, TypeError):
    """A data object was asked for content it does not hold.

    ``dataObject`` is one container for every kind of number the format
    carries, so asking a geometry for its table is a type error, not a missing
    key.
    """


class BenchmarkMismatchError(SinbadError):
    """A calculations file names a benchmark that is not the one it was opened with.

    The format ties a calculations file to its benchmark by sha1 precisely so
    this can be detected. A *changed* checksum is a warning, not an error --
    see :attr:`kika.sinbad.Calculations.matches_benchmark`; this exception is
    for a different entry altogether.
    """


class IncompleteBudgetError(SinbadError):
    """An uncertainty budget does not cover every point it would have to.

    §3.2 (v0.4) lets an entry give a component only at the positions where it
    was published -- AEA-RS-1231 Table 18 gives the uncertainty of the McBEND
    rates at two positions per detector -- and name one it never quantified.
    A covariance over the whole table would need the missing values, and
    filling them in would be a calculation the entry did not authorise, so
    the reader stops instead.

    The message says which components are missing where. The same facts are on
    the exception, for a caller that wants to act on them:

    Attributes
    ----------
    label : str
        The data object.
    missing : dict of str to list of str
        Component name to the positions it is not given at.
    complete_at : list of str
        The positions where every quantified component is given.
    """

    def __init__(self, message: str, label: str = "", missing=None, complete_at=None):
        super().__init__(message)
        self.label = label
        self.missing = dict(missing or {})
        self.complete_at = list(complete_at or [])
