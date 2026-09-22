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
