"""Keys of the per-particle IXS entries (MTRH, SIGH, ANDH, DLWH).

The reactions producing a particle are listed in MTRH, and an MT may appear
there more than once: n + He-3 -> d + d (MT650) is written by NJOY as two
deuteron entries, each with its own yield record, angular and energy
distribution. The per-particle containers are dicts, so each entry needs a
key of its own.

The first entry of an MT is keyed by the plain ``int`` MT, as before; the
k-th entry of an MT already seen (k >= 2) by ``RepeatedMT(mt, k)``. A table
whose MTs are unique (almost all of them) has the same keys it always had,
and ``ixs_keys`` gives the keys of an MTRH list in its order, for the readers
and the writers to agree on.
"""
from __future__ import annotations

from typing import Iterable, List


class RepeatedMT(int):
    """The key of the ``occurrence``-th IXS entry of an MT (``occurrence`` >= 2).

    It is an ``int`` whose value is the MT, so ``int(key)``, arithmetic and
    formatting with a spec give the MT; it is not equal to the plain MT (which
    keys the first entry) nor to another occurrence, and it sorts right after
    the earlier entries of its MT. ``str`` gives ``650#2``.
    """

    def __new__(cls, mt: int, occurrence: int):
        occurrence = int(occurrence)
        if occurrence < 2:
            raise ValueError(f"RepeatedMT is for the second and later entries of an MT, not {occurrence}")
        self = super().__new__(cls, int(mt))
        self.occurrence = occurrence
        return self

    def __getnewargs__(self):
        return int(self), self.occurrence

    def _order(self):
        return int(self), self.occurrence

    @staticmethod
    def _other(other):
        if isinstance(other, RepeatedMT):
            return other._order()
        if isinstance(other, int) and not isinstance(other, bool):
            return int(other), 1
        return None

    def __eq__(self, other):
        o = self._other(other)
        return NotImplemented if o is None else self._order() == o

    def __ne__(self, other):
        o = self._other(other)
        return NotImplemented if o is None else self._order() != o

    def __lt__(self, other):
        o = self._other(other)
        return NotImplemented if o is None else self._order() < o

    def __le__(self, other):
        o = self._other(other)
        return NotImplemented if o is None else self._order() <= o

    def __gt__(self, other):
        o = self._other(other)
        return NotImplemented if o is None else self._order() > o

    def __ge__(self, other):
        o = self._other(other)
        return NotImplemented if o is None else self._order() >= o

    def __hash__(self):
        return hash(self._order())

    def __repr__(self):
        return f"RepeatedMT({int(self)}, {self.occurrence})"

    def __str__(self):
        return f"{int(self)}#{self.occurrence}"


def ixs_keys(mts: Iterable) -> List[int]:
    """The keys of an MTRH list's entries, in its order.

    The plain MT for the first entry of each MT, ``RepeatedMT(mt, k)`` for
    its k-th.
    """
    seen = {}
    keys = []
    for mt in mts:
        mt = int(mt)
        seen[mt] = seen.get(mt, 0) + 1
        keys.append(mt if seen[mt] == 1 else RepeatedMT(mt, seen[mt]))
    return keys


def ixs_key(value) -> int:
    """``value`` as an IXS key: a ``RepeatedMT`` is kept, anything else is ``int``."""
    return value if isinstance(value, RepeatedMT) else int(value)
