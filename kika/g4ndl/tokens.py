"""The token stream, read the way the consumer reads it.

Geant4 reads every G4NDL file with ``operator>>`` on an ``istringstream``:
whitespace-separated tokens, **no notion of a line**. The number of values per
line means nothing and the reader must not rely on it.

Two things the consumer does that this class reproduces exactly:

* **The optional header** (``G4ParticleHPManager.cc:197-208``): if and only if
  the first token is literally ``G4NDL``, it and the token after it (the data
  source label) are consumed; otherwise reading starts at token 0. A file
  starting with a number never loses two tokens.
* **Integers are integers.** ``>> G4int`` on ``"2.0"`` reads ``2`` and leaves
  ``.0`` in the stream, misaligning every field after it without an error.
  Here a token read as an integer must be one; ``"2.0"`` raises.

And one thing it does not: running off the end raises, naming what was being
read, instead of leaving the remaining fields at whatever they held.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from kika.g4ndl.exceptions import G4NDLFormatError

__all__ = ["HEADER_TAG", "TokenStream"]

HEADER_TAG = "G4NDL"

_INT_RE = re.compile(r"^[+-]?\d+$")


class TokenStream:
    """A cursor over the whitespace-separated tokens of one G4NDL file.

    Parameters
    ----------
    text
        The decompressed file content (:func:`kika.g4ndl.stream.read_text`).
    path
        Only for error messages.
    """

    def __init__(self, text: str, path: Optional[Path] = None):
        self._tokens = text.split()
        self.path = Path(path) if path is not None else None
        self.header: Optional[Tuple[str, str]] = None
        self._i = 0
        if self._tokens and self._tokens[0] == HEADER_TAG:
            if len(self._tokens) < 2:
                self._fail("header 'G4NDL' without a source label", "header", 0)
            self.header = (self._tokens[0], self._tokens[1])
            self._i = 2

    # ------------------------------------------------------------ position
    @property
    def position(self) -> int:
        """Index of the next token, counting the header tokens."""
        return self._i

    @property
    def remaining(self) -> int:
        return len(self._tokens) - self._i

    def atEnd(self) -> bool:
        return self._i >= len(self._tokens)

    # ------------------------------------------------------------- scalars
    def _take(self, what: str) -> str:
        if self._i >= len(self._tokens):
            self._fail("unexpected end of data", what, self._i)
        tok = self._tokens[self._i]
        self._i += 1
        return tok

    def int(self, what: str) -> int:
        """Next token as an integer; a token like ``2.0`` is an error."""
        tok = self._take(what)
        if not _INT_RE.match(tok):
            self._fail(f"expected an integer, found {tok!r}", what, self._i - 1)
        return int(tok)

    def float(self, what: str) -> float:
        """Next token as a finite float."""
        tok = self._take(what)
        try:
            value = float(tok)
        except ValueError:
            self._fail(f"expected a number, found {tok!r}", what, self._i - 1)
        if not np.isfinite(value):
            self._fail(f"non-finite value {tok!r}", what, self._i - 1)
        return value

    # -------------------------------------------------------------- arrays
    def floats(self, n: int, what: str) -> np.ndarray:
        """The next ``n`` tokens as a float64 array, all finite.

        Vectorised because a CrossSection carries up to ~350 000 pairs; the
        error path falls back to a scan so the message names the bad token.
        """
        if n < 0:
            self._fail(f"negative count {n}", what, self._i)
        start = self._i
        if self.remaining < n:
            self._fail(f"unexpected end of data: {n} values needed, "
                       f"{self.remaining} left", what, len(self._tokens))
        chunk = self._tokens[start:start + n]
        try:
            values = np.array(chunk, dtype=np.float64)
        except ValueError:
            for k, tok in enumerate(chunk):
                try:
                    float(tok)
                except ValueError:
                    self._fail(f"expected a number, found {tok!r}", what, start + k)
            raise  # pragma: no cover - the scan above always finds it
        bad = np.flatnonzero(~np.isfinite(values))
        if bad.size:
            k = int(bad[0])
            self._fail(f"non-finite value {chunk[k]!r}", what, start + k)
        self._i = start + n
        return values

    def pairs(self, n: int, what: str) -> Tuple[np.ndarray, np.ndarray]:
        """``n`` interleaved ``(x, y)`` pairs as two arrays."""
        flat = self.floats(2 * n, what)
        return flat[0::2].copy(), flat[1::2].copy()

    # ----------------------------------------------------------------- end
    def expectEnd(self, what: str = "end of file") -> None:
        """Raise if any token is left. Geant4 ignores trailing tokens; kika does not."""
        if not self.atEnd():
            self._fail(f"{self.remaining} unread tokens after the last record",
                       what, self._i)

    def _fail(self, message: str, what: str, token: int):
        raise G4NDLFormatError(message, path=self.path, record=what, token=token)
