"""The XSS array: one float64 numpy array per table.

``ace.xss_data`` is a 1-D ``float64`` array of length ``NXS(1) + 1``. Index 0
is a placeholder (0.0), so ``xss_data[i]`` is the FORTRAN ``XSS(i)`` of the ACE
manual and every JXS pointer indexes it unchanged.

Every block the parsers carve out of it (energy grid, reaction cross sections,
locators, distributions) holds a **view** — a basic slice — not a copy. Writing
into a block therefore writes into ``xss_data``, which is what ``write_ace``
serialises; that is how ``Ace.update_cross_sections`` and the samplers perturb
a table. Fancy indexing (a boolean mask, an index list) returns a copy, so
anything meant to be written back must stay a basic slice.
"""
from typing import Optional

import numpy as np


def xss_position(view: np.ndarray) -> Optional[int]:
    """1-based XSS position of the first element of a view of ``xss_data``.

    Returns None when ``view`` is not a view of a float64 array (a copy, or an
    array a parser built itself).
    """
    base = getattr(view, "base", None)
    if not isinstance(base, np.ndarray) or base.dtype != np.float64:
        return None
    offset = view.__array_interface__["data"][0] - base.__array_interface__["data"][0]
    return offset // base.itemsize


def xss_ints(xss: np.ndarray, start: int, n: int) -> list:
    """The ``n`` integer words ``XSS(start..start+n-1)`` as a list of ints.

    One array operation instead of ``int(xss[i])`` per word: NBT/INT and
    locator arrays are read this way. ACE writes these words as integers, so
    rounding and truncating agree; rounding is used so that a value written
    as ``2.99999999999E+00`` still reads as 3. Raises ``IndexError`` when the
    words run past the array.
    """
    if n < 0 or start + n > len(xss):
        raise IndexError(f"XSS({start}:{start + n - 1}) is past the XSS array ({len(xss)})")
    return np.rint(xss[start:start + n]).astype(np.int64).tolist()


# --- how each word was spelled ----------------------------------------------
#
# ACE has no types: MT numbers, counts and locators sit in the same REAL array
# as the cross sections, and only their spelling tells them apart -- NJOY
# writes them as I20, the rest as E20.11. ``Ace.xss_int_words`` keeps that
# spelling as one bit per word (``np.packbits``, 1/64 of the XSS memory), bit k
# for XSS(k+1), so that ``write_ace`` gives back the bytes it read. A table
# built from scratch gets its mask from the encoder, which knows which words
# are counts; without a mask every word is written as a real, which MCNP reads
# the same.

def pack_int_words(is_int: np.ndarray) -> np.ndarray:
    """Pack a bool array over XSS(1..n) into the ``xss_int_words`` bit mask."""
    return np.packbits(np.asarray(is_int, dtype=bool), bitorder="little")


def unpack_int_words(packed: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    """The bool array over XSS(1..n), or None without a mask or with one of another length."""
    if packed is None or len(packed) != (n + 7) // 8:
        return None
    return np.unpackbits(packed, count=n, bitorder="little").astype(bool)


def integer_fields(fields: np.ndarray) -> np.ndarray:
    """Which 20-byte fields (a ``uint8`` array ``(..., 20)``) are spelled as integers.

    An E20.11 real always has a decimal point; an I20 integer never does.
    """
    return ~(fields == ord(".")).any(axis=-1)


def table_padding(xss: np.ndarray, jed: int, locations, lengths):
    """Words between consecutive tables of a law: ``(padding, unread)``.

    ``locations`` are the tables' JED-relative locators and ``lengths`` their
    sizes in words. ``padding[i]`` counts the zero words after table i before
    table i+1 (0 when they touch); ``unread`` lists the gaps that hold
    something other than zeros, which no locator points at.
    """
    padding, unread = [], []
    for i in range(len(locations) - 1):
        end = int(locations[i]) + int(lengths[i])
        gap = int(locations[i + 1]) - end
        if gap <= 0:
            padding.append(0)
            continue
        if np.any(xss[jed + end - 1:jed + end - 1 + gap]):
            unread.append((i + 1, gap))
            padding.append(0)
        else:
            padding.append(gap)
    return padding, unread
