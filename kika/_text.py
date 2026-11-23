"""Decoding of the text files kika reads, the same way in the detector and in every reader.

Files come from other tools and other editors: UTF-8 with or without a BOM, or a stray
cp1252 byte in a title. A UTF-16 file is refused with a message saying how to fix it.
"""
from __future__ import annotations

import os

UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


class UTF16FileError(ValueError):
    """The file is UTF-16 text, which none of kika's formats are written in."""


def utf16_message(path) -> str:
    return (f"{os.fspath(path)} is UTF-16 text (PowerShell 5.1 writes it with '>' or Out-File). "
            f"Save it as UTF-8 or ASCII and open it again.")


def decode_text(raw: bytes, path="<bytes>") -> str:
    """UTF-8 without its BOM; an undecodable byte becomes U+FFFD instead of raising."""
    if raw.startswith(UTF16_BOMS):
        raise UTF16FileError(utf16_message(path))
    return raw.decode("utf-8-sig", errors="replace")


def read_text(path) -> str:
    with open(path, "rb") as handle:
        return decode_text(handle.read(), path)


CSV_DELIMITERS = (";", "\t", ",")


def sniff_csv(text: str, max_rows: int = 20) -> tuple[str, int] | None:
    """``(delimiter, comment_lines)`` of a delimited table, or None when it is not one.

    Leading ``#`` lines are comments. A delimiter qualifies when every row of the head
    splits into the same number of fields, at least two, honouring quotes. ``;`` and tab
    are tried before ``,`` because a ``;`` table with decimal commas also splits evenly
    on ``,``.
    """
    import csv
    import io

    lines = text.splitlines()
    comments = 0
    while comments < len(lines) and lines[comments].lstrip().startswith("#"):
        comments += 1
    rows = [ln for ln in lines[comments:] if ln.strip()][:max_rows]
    if len(rows) < 2 or rows[0].lstrip().startswith(("{", "[")):
        return None
    for delimiter in CSV_DELIMITERS:
        widths = {len(r) for r in csv.reader(io.StringIO("\n".join(rows)), delimiter=delimiter)}
        if len(widths) == 1 and widths.pop() >= 2:
            return delimiter, comments
    return None


def read_lines(path) -> list[str]:
    """``read_text`` split exactly as ``open(path).readlines()`` would (universal newlines)."""
    import io

    return io.StringIO(read_text(path), newline=None).readlines()
