"""A block of XSS words being written, with the spelling of each.

Every encoder writes one ACE block into a :class:`Words`. Locators inside a
block are relative to the block's first word (LSIG to SIG, LAND to AND, LOCC
and IDAT and the laws' L/LC to JED = the DLW start...), which is exactly
``Words.next``: the 1-based position the next word will take. So a block is
encoded without knowing where the layout will put it, and the layout only
places blocks and fills JXS.

Each word carries whether NJOY writes it as an integer (I20: counts, MT
numbers, locators, interpolation codes) or a real (E20.11): that becomes
``Ace.xss_int_words`` and lets ``write_ace`` give NJOY's bytes back.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


class Words:
    __slots__ = ("values", "is_int")

    def __init__(self):
        self.values: list = []
        self.is_int: list = []

    def __len__(self) -> int:
        return len(self.values)

    @property
    def next(self) -> int:
        """Block-relative 1-based position of the next word: the locator of what is written next."""
        return len(self.values) + 1

    def int(self, *values) -> None:
        """Integer words (counts, MTs, locators, interpolation codes)."""
        for v in values:
            self.values.append(float(int(round(float(v)))))
            self.is_int.append(True)

    def real(self, *values) -> None:
        """Real words, copied bit for bit."""
        for v in values:
            self.values.append(float(v))
            self.is_int.append(False)

    def ints(self, values: Iterable) -> None:
        self.int(*np.asarray(values, dtype=float).ravel().tolist())

    def reals(self, values: Iterable) -> None:
        a = np.asarray(values, dtype=float).ravel().tolist()
        self.values.extend(a)
        self.is_int.extend([False] * len(a))

    def reserve(self, n: int = 1) -> int:
        """Reserve ``n`` integer words to fill later with :meth:`set`; returns the 0-based index of the first."""
        i = len(self.values)
        self.int(*([0] * n))
        return i

    def set(self, index: int, value) -> None:
        """Fill a reserved integer word (0-based index from :meth:`reserve`)."""
        self.values[index] = float(int(round(float(value))))

    def extend(self, other: "Words") -> None:
        self.values.extend(other.values)
        self.is_int.extend(other.is_int)

    def arrays(self):
        """``(values float64, is_int bool)`` as numpy arrays."""
        return np.asarray(self.values, dtype=np.float64), np.asarray(self.is_int, dtype=bool)
