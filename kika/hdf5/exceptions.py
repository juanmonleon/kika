"""What the OpenMC HDF5 reader and writer raise.

Every failure names the file and the HDF5 path it happened at. OpenMC's C++
reader checks almost nothing: a file that
breaks its assumptions loads and then indexes out of bounds. A reader that is to
be trusted as a yardstick, and a writer whose files are to be trusted at all,
have to stop instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = ["HDF5Error", "HDF5FormatError", "HDF5UnsupportedError",
           "NuclideNotFoundError"]


class HDF5Error(Exception):
    """Base class for every OpenMC-HDF5 error kika raises.

    Catch it to handle the three errors below (:class:`HDF5FormatError`,
    :class:`HDF5UnsupportedError` and :class:`NuclideNotFoundError`) together.
    """


class HDF5FormatError(HDF5Error, ValueError):
    """A file is not what the OpenMC format says it must be.

    Raised by the readers and writers when the HDF5 content breaks the layout
    OpenMC's C++ reader assumes (a missing ``filetype`` attribute, an unknown
    interpolation code, tables whose counts disagree). It is a ``ValueError``.

    ``path`` is the file and ``node`` the HDF5 path inside it, e.g.
    ``/U235/reactions/reaction_018/product_0``. Each is None when the raiser did
    not know it, and the message ends with the ones that are set in brackets.
    """

    def __init__(self, message: str, *, path: Optional[Path] = None,
                 node: Optional[str] = None):
        self.path = Path(path) if path is not None else None
        self.node = node
        where = [str(part) for part in (self.path, node) if part]
        super().__init__(f"{message} [{'; '.join(where)}]" if where else message)


class HDF5UnsupportedError(HDF5Error, NotImplementedError):
    """The file or distribution is well formed, but kika does not handle it.

    Raised where the alternative would be an approximation: a photon or ``wmp``
    file read into the model, a ``filetype`` kika does not read, a distribution
    with no OpenMC layout to write. The caller sees what is missing and does not
    receive a file that looks complete. It is a ``NotImplementedError``.
    """


class NuclideNotFoundError(HDF5Error, LookupError):
    """``cross_sections.xml`` lists no file of the requested type for this material.

    Raised by the library look-ups. The match is exact: OpenMC names metastables
    ``Am242_m1`` and thermal tables ``c_H_in_H2O``. It is a ``LookupError``.
    """
