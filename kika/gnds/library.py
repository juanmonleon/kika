"""A directory of GNDS files, as a library: index it, then read one target.

The published GNDS libraries are directories of one ``reactionSuite`` per
target, with the covariances in a sibling directory the files link to
(ENDF/B-VIII.1: ``neutrons/n-026_Fe_056.endf.gnds.xml`` and
``neutrons/Covariances/n-026_Fe_056.endf.gnds-covar.xml``). A viewer wants the
list of targets before it reads any of them, and on a network share it must
not decode 557 files to get it. :func:`open` therefore reads only the root
element of each file -- the few hundred bytes that carry ``projectile``,
``target``, ``evaluation`` and ``format`` -- and :meth:`GNDSLibrary.read`
hands one of them to :func:`kika.read`.

Nothing here imports the model at module scope (see :mod:`kika.gnds`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

__all__ = ["GNDSLibrary", "GNDSLibraryError", "IndexedTarget", "open", "parseTargetId"]

#: Suffixes of a ``reactionSuite`` file; the covariance files end in
#: ``-covar.xml`` and are reached through the suite's ``externalFile``.
_SUITE_SUFFIXES = (".gnds.xml", ".xml")
_COVARIANCE_MARK = "-covar"

#: How much of a file is read to find its root element.
_HEADER_BYTES = 4096

_ROOT_RE = re.compile(r"<reactionSuite\b([^>]*)>", re.S)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
#: A GNDS nuclide id: ``Fe56``, ``Am242_m1``, ``C0`` (the natural element).
_ID_RE = re.compile(r"^([A-Z][a-z]{0,2})(\d+)(?:_m(\d+))?$")


class GNDSLibraryError(ValueError):
    """The directory holds no GNDS reaction suite, or the target is not in it."""


def parseTargetId(target: str):
    """``(symbol, A, M)`` of a GNDS nuclide id; ``A`` is ``None`` for a
    natural element (``C0``), ``None`` overall when the id is not a nuclide
    (``n``, ``photon``, a thermal-scattering material)."""
    m = _ID_RE.match(target)
    if not m:
        return None
    a = int(m.group(2))
    return m.group(1), (a or None), int(m.group(3) or 0)


@dataclass(frozen=True)
class IndexedTarget:
    """One file of the library, from its root element alone."""

    target: str
    projectile: str
    evaluation: Optional[str]
    format: Optional[str]
    interaction: Optional[str]
    path: Path
    covariance: Optional[Path]
    Z: Optional[int]
    A: Optional[int]
    M: int
    symbol: Optional[str]


def _header(path: Path) -> Optional[Dict[str, str]]:
    with path.open("rb") as handle:
        head = handle.read(_HEADER_BYTES).decode("utf-8", errors="replace")
    m = _ROOT_RE.search(head)
    if not m:
        return None
    attrs = dict(_ATTR_RE.findall(m.group(1)))
    ext = re.search(r'<externalFile\b[^>]*label="covariances"[^>]*path="([^"]*)"', head)
    if ext:
        attrs["_covariance"] = ext.group(1)
    return attrs


class GNDSLibrary:
    """The targets of one directory of GNDS ``reactionSuite`` files.

    ``root`` may also be **one file**: a single evaluation (one's own, a FUDGE
    output, a file kika wrote) is then a library of one target, so a viewer
    built for libraries opens it with no second code path. Its covariance file
    is found as for any other, through the suite's ``externalFile``.
    """

    def __init__(self, root: Union[str, Path]):
        from kika._constants import SYMBOL_TO_ATOMIC_NUMBER as SYMBOL_TO_Z

        self.root = Path(root)
        if self.root.is_file():
            candidates = [self.root]
        elif self.root.is_dir():
            candidates = sorted(self.root.iterdir())
        else:
            raise GNDSLibraryError(f"Neither a file nor a directory: {self.root}")
        self._index: Dict[str, IndexedTarget] = {}
        self.duplicates: List[Path] = []
        self.unreadable: List[Path] = []
        for path in candidates:
            name = path.name
            if path is not self.root and (
                    not path.is_file() or _COVARIANCE_MARK in name or not name.endswith(_SUITE_SUFFIXES)):
                continue
            try:
                attrs = _header(path)
            except OSError:
                attrs = None
            if not attrs or "target" not in attrs:
                self.unreadable.append(path)
                continue
            target = attrs["target"]
            parsed = parseTargetId(target)
            symbol, a, m = parsed if parsed else (None, None, 0)
            z = SYMBOL_TO_Z.get(symbol) if symbol else None
            cov = attrs.get("_covariance")
            entry = IndexedTarget(
                target=target, projectile=attrs.get("projectile", ""),
                evaluation=attrs.get("evaluation"), format=attrs.get("format"),
                interaction=attrs.get("interaction"), path=path,
                covariance=(path.parent / cov) if cov else None,
                Z=z, A=a, M=m, symbol=symbol)
            if target in self._index:
                self.duplicates.append(path)
                continue
            self._index[target] = entry
        if not self._index:
            raise GNDSLibraryError(
                f"{self.root} is not a GNDS reactionSuite file" if self.root.is_file()
                else f"{self.root} holds no GNDS reactionSuite file (*.gnds.xml)")

    @property
    def name(self) -> str:
        """The directory's name, or the file's without ``.gnds.xml``/``.xml``."""
        if self.root.is_file():
            for suffix in _SUITE_SUFFIXES:
                if self.root.name.endswith(suffix):
                    return self.root.name[: -len(suffix)]
        return self.root.name

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, target: str) -> bool:
        return target in self._index

    def targets(self) -> List[str]:
        """Every target id, nuclides first by (Z, A, M), then the rest by name."""
        def key(t: str):
            e = self._index[t]
            return (e.Z is None, e.Z or 0, e.A or 0, e.M, t)
        return sorted(self._index, key=key)

    def locate(self, target: str) -> IndexedTarget:
        try:
            return self._index[target]
        except KeyError:
            raise GNDSLibraryError(f"{target!r} is not in {self.root.name}") from None

    @property
    def evaluations(self) -> List[str]:
        return sorted({e.evaluation for e in self._index.values() if e.evaluation})

    @property
    def projectiles(self) -> List[str]:
        return sorted({e.projectile for e in self._index.values()})

    def read(self, target: str, covariances: bool = False):
        """Decode ``target`` into a :class:`ReactionSuite` (:func:`kika.read`)."""
        from kika._read import read

        return read(str(self.locate(target).path), format="gnds", covariances=covariances)

    def describe(self) -> dict:
        """The library as a viewer lists it, from the index alone."""
        entries = []
        for t in self.targets():
            e = self._index[t]
            entries.append(dict(
                target=t, id=t, Z=e.Z, A=e.A, M=e.M, element=e.symbol,
                file_name=e.path.name, size=e.path.stat().st_size,
                covariance=bool(e.covariance and e.covariance.is_file()),
                evaluation=e.evaluation, format=e.format, projectile=e.projectile))
        return dict(root=str(self.root), name=self.name, single_file=self.root.is_file(),
                    isotopes=entries,
                    evaluations=self.evaluations, projectiles=self.projectiles,
                    unreadable=len(self.unreadable), duplicates=len(self.duplicates))


def open(root: Union[str, Path]) -> GNDSLibrary:  # noqa: A001 -- mirrors kika.g4ndl.open
    """Index the GNDS directory ``root``, or the single GNDS file ``root``."""
    return GNDSLibrary(root)
