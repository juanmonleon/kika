"""Number parsing shared by the MCNP output readers (MCTAL and MESHTAL).

MCNP writes reals with Fortran edit descriptors, which has two consequences a
plain ``float(token)`` does not survive:

* a three-digit exponent drops the ``E``: ``8.15942-132`` is ``8.15942e-132``;
* two numbers can touch with no blank between them when the second one is
  negative and the field is full: ``1.00E+00-2.00E+00``.

:func:`numbers` scans a line with one regular expression that handles both, so
the readers never split on whitespace and then guess.
"""

from __future__ import annotations

import re

import numpy as np

# A mantissa followed by an optional exponent, written either with a letter
# (E/D, any sign) or, for three-digit exponents, as a bare sign and three digits.
# The bare-sign branch must demand exactly three digits; otherwise the minus of
# a following number (``1.0-2.0``) would be swallowed as an exponent.
_NUMBER = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)"
    r"(?:[eEdD][-+]?\d+|[-+]\d{3}(?!\d|\.))?"
)


def fortran_float(token: str) -> float:
    """Convert one Fortran-formatted real to ``float``.

    :param token: A single number as written by MCNP, e.g. ``"1.23E-04"``,
        ``"8.15942-132"`` or ``"1.0D+00"``.
    :returns: The value.
    :raises ValueError: If *token* is not one number.
    """
    match = _NUMBER.fullmatch(token.strip())
    if match is None:
        raise ValueError(f"not a Fortran real: {token!r}")
    return _to_float(match.group(0))


def numbers(line: str) -> list[float]:
    """Every real on *line*, in order, including run-together ones.

    :param line: A line of MCNP output holding only numbers.
    :returns: The values.
    """
    return [_to_float(t) for t in _NUMBER.findall(line)]


def numbers_array(lines) -> np.ndarray:
    """Concatenate :func:`numbers` over *lines* into one float array."""
    out: list[float] = []
    for line in lines:
        out.extend(numbers(line))
    return np.asarray(out, dtype=float)


_FIELD = re.compile(r"Total|" + _NUMBER.pattern)


def fields(line: str) -> list[float]:
    """Like :func:`numbers`, but a literal ``Total`` (a total-bin label in a
    meshtal row) is returned as ``nan`` in its position."""
    return [np.nan if t == "Total" else _to_float(t) for t in _FIELD.findall(line)]


def _to_float(text: str) -> float:
    text = text.replace("d", "e").replace("D", "e")
    try:
        return float(text)
    except ValueError:
        pass
    # Bare-sign exponent: the sign closest to the end that is not the leading one.
    cut = max(text.rfind("-"), text.rfind("+"))
    return float(f"{text[:cut]}e{text[cut:]}")
