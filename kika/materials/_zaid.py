"""The nuclide a transport-code ZAID names, isomers included, and back."""
from typing import Tuple

from .._constants import ATOMIC_MASS


def identity(zaid: int) -> Tuple[int, int, int]:
    """``(Z, A, isomer)`` of *zaid*; A is 0 for a natural element.

    Reads kika's own ``IZZZAAA`` (``1095242``), LANL's ``ZZZ(A+300+100m)``
    (``95642``) and the NEA/JEFF ``ZZZ(300 + A mod 100)`` (``95342``). The two
    isomer forms drop information, so A is the one candidate the AME2020 table
    knows as a nuclide of that Z.
    """
    zaid = int(zaid)
    if zaid // 1000 > 118:
        return (zaid % 1000000) // 1000, zaid % 1000, zaid // 1000000
    z, raw = divmod(zaid, 1000)
    if raw < 300:
        return z, raw, 0
    if raw < 400:
        candidates = [(a, 1) for a in range(raw - 300, 300, 100)]
    else:
        candidates = [(raw - 300 - 100 * m, m) for m in range(1, (raw - 300) // 100 + 1)]
    known = [(a, m) for a, m in candidates if a > 0 and z * 1000 + a in ATOMIC_MASS]
    if len(known) != 1:
        raise ValueError(
            f"ZAID {zaid} does not name one nuclide: as an isomer it could be "
            f"{[z * 1000 + a for a, _ in known] or 'none'} (Z={z})"
        )
    a, m = known[0]
    return z, a, m


def canonical(zaid: int) -> int:
    """*zaid* in kika's own form: ``ZZZAAA``, or ``IZZZAAA`` for an isomer."""
    z, a, m = identity(zaid)
    return m * 1000000 + z * 1000 + a


def mcnp(zaid: int) -> int:
    """*zaid* as an m card writes it: as read when it came from a card, LANL's ``A + 300 + 100 m`` for kika's own isomer form."""
    if int(zaid) // 1000 <= 118:
        return int(zaid)
    z, a, m = identity(zaid)
    return z * 1000 + a + 300 + 100 * m
