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
