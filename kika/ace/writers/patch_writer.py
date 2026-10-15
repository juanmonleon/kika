"""Write many variants of one ACE table by patching only what changed.

A perturbed ACE sample differs from its source table in a few hundred thousand
XSS values out of several million (Fe-56 Lib81: 0.29 M of 5.8 M for five
reactions), yet :func:`~kika.ace.writers.write_ace.write_ace` formats every one
of them again, which made the writer ~80 % of a sample's time. In the XSS
section every value has a fixed place -- four 20-column fields per line -- so a
sample can be written as the source table's text with the changed lines
re-formatted in place.

The output is **byte-identical** to ``write_ace``: the template *is* a
``write_ace`` file of the unperturbed table, the lines are re-formatted with
the format string ``write_ace``'s ``np.savetxt`` applies, and every case in
which ``write_ace`` would lay the file out differently -- another header, an
XSS of another length, another integer-spelling mask (``xss_int_words``) or,
without one, the all-integer ``%20d`` format switching on or off -- falls back
to ``write_ace`` itself.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

import numpy as np

from kika.ace.classes.xss import unpack_int_words
from kika.ace.writers.write_ace import format_xss_lines, write_ace
from kika.ace.writers.write_header import write_header

_FLOAT_FMT = "%20.11E"
_ROW_FMT = _FLOAT_FMT * 4


def _same_int_words(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a is b or np.array_equal(a, b)


def _all_integer(values: np.ndarray) -> bool:
    """``write_ace``'s own test for the ``%20d`` layout."""
    return bool(np.all(np.abs(values - np.rint(values)) < 1e-12))


class AcePatchWriter:
    """Writes variants of *ace* that differ from it only in XSS values.

    ``ace`` is the table as read, before any perturbation; its XSS is copied
    here, so the caller may then perturb ``ace`` in place, :meth:`write` it,
    and :meth:`restore` it for the next sample. ``template_path`` is where the
    unperturbed table is written once (the file is owned by this object and
    removed by :meth:`close`). With ``write_template=False`` the file is taken
    as already written -- a worker process sharing the template its parent
    wrote from the same table.
    """

    def __init__(self, ace, template_path: str, write_template: bool = True):
        self.base_xss = np.array(ace.xss_data, dtype=float, copy=True)
        self.base_header = write_header(ace.header)
        self.template_path = str(template_path)
        if write_template:
            write_ace(ace, self.template_path, overwrite=True)

        values = self.base_xss[1:]
        packed = getattr(ace, "xss_int_words", None)
        self.base_int_words = None if packed is None else np.array(packed, copy=True)
        # With the spelling known, write_ace formats word by word and leaves the
        # last line unpadded; without it, the all-integer %20d layout is the
        # one case it lays out differently.
        self.is_int = unpack_int_words(self.base_int_words, values.size)
        self.usable = values.size > 0 and (self.is_int is not None or not _all_integer(values))
        if not self.usable:
            return
        size = os.path.getsize(self.template_path)
        with open(self.template_path, "rb") as handle:
            head = handle.read(4096)
        newline = b"\r\n" if b"\r\n" in head else b"\n"
        self.line = 80 + len(newline)
        n = values.size
        self.full_rows = n // 4
        self.remainder = n % 4
        last = 0
        if self.remainder:
            last = (20 * self.remainder + len(newline)) if self.is_int is not None else self.line
        self.data_start = size - self.full_rows * self.line - last
        # The XSS starts on the line after the header; anything else means the
        # layout is not the one assumed here.
        with open(self.template_path, "rb") as handle:
            handle.seek(self.data_start - len(newline))
            self.usable = handle.read(len(newline)) == newline

    def restore(self, ace) -> None:
        """Put the unperturbed XSS back into *ace*, keeping its block views."""
        np.copyto(ace.xss_data, self.base_xss)

    def write(self, ace, path: str) -> bool:
        """Write *ace* to *path*. ``True`` if patched, ``False`` if it fell back."""
        xss = np.asarray(ace.xss_data, dtype=float)
        if (not self.usable or xss.shape != self.base_xss.shape
                or write_header(ace.header) != self.base_header
                or not _same_int_words(getattr(ace, "xss_int_words", None), self.base_int_words)
                or (self.is_int is None and _all_integer(xss[1:]))):
            write_ace(ace, path, overwrite=True)
            return False

        values = xss[1:]
        # NaN != NaN would mark an unchanged NaN as changed; re-formatting it
        # gives the same text, so that only costs time.
        changed = np.flatnonzero(values != self.base_xss[1:])
        shutil.copyfile(self.template_path, path)
        if changed.size == 0:
            return True
        rows = np.unique(changed // 4)
        # Contiguous runs of changed lines go out as one write each.
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        with open(path, "r+b") as handle:
            for run in np.split(rows, breaks):
                first, last = int(run[0]), int(run[-1])
                handle.seek(self.data_start + first * self.line)
                handle.write(self._format_rows(values, first, last))
        return True

    def _format_rows(self, values: np.ndarray, first: int, last: int) -> bytes:
        newline = "\r\n" if self.line == 82 else "\n"
        if self.is_int is not None:
            text = "".join(format_xss_lines(values, self.is_int, first, last))
            return (text.replace("\n", newline) if newline != "\n" else text).encode("ascii")
        out = []
        for row in range(first, last + 1):
            if row < self.full_rows:
                out.append(_ROW_FMT % tuple(values[4 * row:4 * row + 4]))
            else:
                tail = values[4 * row:]
                out.append("".join(_FLOAT_FMT % v for v in tail).ljust(80))
        return (newline.join(out) + newline).encode("ascii")

    def close(self) -> None:
        try:
            os.remove(self.template_path)
        except OSError:
            pass
