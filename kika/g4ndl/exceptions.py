"""What the G4NDL reader raises.

Every failure names the file it happened in, and a token-level failure also
names the record being read and the position of the offending token. The
consumer (Geant4) reads with ``operator>>`` and does not notice most of these:
a short file leaves the stream in a fail state and the remaining fields at
whatever they held. A reader that is to be trusted as a yardstick has to stop
instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = ["G4NDLError", "G4NDLFormatError", "G4NDLUnsupportedError",
           "IsotopeNotFoundError"]


class G4NDLError(Exception):
    """Base class for every G4NDL error kika raises."""


class G4NDLFormatError(G4NDLError, ValueError):
    """A file is not what the G4NDL grammar says it must be.

    Attributes
    ----------
    path
        The file, as given to the reader (the ``.z`` when it was compressed).
    record
        What was being read, e.g. ``"FS repFlag=1, energy 3 of 96: NL"``.
    token
        Zero-based index of the offending token in the decompressed stream,
        counting the optional two-token header; ``None`` for byte-level errors.
    """

    def __init__(self, message: str, *, path: Optional[Path] = None,
                 record: Optional[str] = None, token: Optional[int] = None):
        self.path = Path(path) if path is not None else None
        self.record = record
        self.token = token
        where = []
        if self.path is not None:
            where.append(str(self.path))
        if record:
            where.append(record)
        if token is not None:
            where.append(f"token {token}")
        super().__init__(f"{message} [{'; '.join(where)}]" if where else message)


class IsotopeNotFoundError(G4NDLError, LookupError):
    """The library has no file for this exact isotope.

    Geant4 would fall back to the natural element or to a neighbouring mass
    number (``G4ParticleHPNames::GetName``); kika does not, because data from
    a different nuclide arriving under the requested name is the one error a
    user cannot see afterwards.
    """


class G4NDLUnsupportedError(G4NDLError, NotImplementedError):
    """The file is well formed, but uses something kika will not decide on.

    Not a format error: the consumer accepts it. Raised where reading it into
    the model would require choosing between two meanings — today only an
    interpolation code 1, which ENDF calls histogram and Geant4 evaluates as
    lin-lin.
    """
