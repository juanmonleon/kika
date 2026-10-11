"""What the OpenMC HDF5 reader and writer raise.

Every failure names the file and the HDF5 path it happened at. OpenMC's C++
reader checks almost nothing (``openmc_hdf5_format_survey.md`` §2): a file that
breaks its assumptions loads and then indexes out of bounds. A reader that is to
be trusted as a yardstick, and a writer whose files are to be trusted at all,
have to stop instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = ["HDF5Error", "HDF5FormatError", "HDF5UnsupportedError",
           "NuclideNotFoundError", "MissingNJOYPieceError"]


class HDF5Error(Exception):
    """Base class for every OpenMC-HDF5 error kika raises."""


class HDF5FormatError(HDF5Error, ValueError):
    """A file is not what the OpenMC format says it must be.

    ``path`` is the file and ``node`` the HDF5 path inside it, e.g.
    ``/U235/reactions/reaction_018/product_0``.
    """

    def __init__(self, message: str, *, path: Optional[Path] = None,
                 node: Optional[str] = None):
        self.path = Path(path) if path is not None else None
        self.node = node
        where = [str(part) for part in (self.path, node) if part]
        super().__init__(f"{message} [{'; '.join(where)}]" if where else message)


class HDF5UnsupportedError(HDF5Error, NotImplementedError):
    """Well formed, and something kika will not decide on.

    Raised rather than approximating, so the caller sees what is missing
    instead of receiving a file that looks complete.
    """


class NuclideNotFoundError(HDF5Error, LookupError):
    """``cross_sections.xml`` names no file for this exact material."""


class MissingNJOYPieceError(HDF5Error, RuntimeError):
    """A piece the writer takes from NJOY (D7) could not be produced.

    ``pieces`` names them. Writing without them is a request
    (``omit=("urr", ...)``), never a silent fallback.
    """

    def __init__(self, message: str, pieces=()):
        self.pieces = tuple(pieces)
        super().__init__(message)
