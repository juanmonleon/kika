"""From a file on disk to the text the consumer tokenises.

``G4ParticleHPManager::GetDataStream`` (``src/G4ParticleHPManager.cc:146``):
a ``.z`` is a raw **zlib** stream (``uncompress()``), not gzip and not zip; a
file without the suffix is plain text. Which of the two to open when both
exist is the library's decision (:mod:`kika.g4ndl.library`), not this one's.
"""
from __future__ import annotations

import zlib
from pathlib import Path

from kika.g4ndl.exceptions import G4NDLFormatError

__all__ = ["DEFAULT_MAX_BYTES", "read_text"]

#: Ceiling on the decompressed size of one file. The largest elastic file in
#: JEFF-4.0 or G4NDL 4.7.1 is U-238's CrossSection, 9.8 MB plain (measured
#: 2026-10-05); 512 MB leaves a factor of fifty and still refuses a
#: decompression bomb.
DEFAULT_MAX_BYTES = 512 * 1024 * 1024


def read_text(path, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Return the decompressed ASCII text of one G4NDL file.

    Raises :class:`G4NDLFormatError` for a damaged or truncated zlib stream,
    a stream larger than ``max_bytes``, or bytes that are not ASCII.
    """
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".z":
        d = zlib.decompressobj()
        try:
            plain = d.decompress(raw, max_bytes)
        except zlib.error as exc:
            raise G4NDLFormatError(f"not a valid zlib stream: {exc}", path=path) from None
        if d.unconsumed_tail:
            raise G4NDLFormatError(
                f"decompresses to more than max_bytes={max_bytes}", path=path)
        if not d.eof:
            raise G4NDLFormatError("zlib stream is truncated", path=path)
        if d.unused_data:
            raise G4NDLFormatError(
                f"{len(d.unused_data)} bytes after the end of the zlib stream", path=path)
    else:
        if len(raw) > max_bytes:
            raise G4NDLFormatError(f"larger than max_bytes={max_bytes}", path=path)
        plain = raw
    try:
        return plain.decode("ascii")
    except UnicodeDecodeError as exc:
        raise G4NDLFormatError(f"non-ASCII byte at offset {exc.start}", path=path) from None
