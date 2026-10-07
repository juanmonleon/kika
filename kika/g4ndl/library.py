"""A G4NDL library: a directory, indexed by isotope, read one file at a time.

A G4NDL library is not one file. It is the tree ``G4NEUTRONHPDATA`` points at
— ``Elastic/``, ``Capture/``, ``Inelastic/``, ``Fission/`` and more, each with
per-isotope files — and Geant4 opens files in it by building a name from
``Z``, ``A`` and ``M``. :class:`G4NDLLibrary` does the same, with three
deliberate differences, each because the consumer's behaviour hides an error
from the user:

* **Exact lookup.** An isotope the library lacks raises
  :class:`~kika.g4ndl.exceptions.IsotopeNotFoundError`; Geant4 would quietly
  substitute the natural element or a neighbouring ``A``.
* **Duplicates are reported.** When ``<name>`` and ``<name>.z`` both exist,
  the ``.z`` wins, as in ``GetDataStream``, and the library lists the pair in
  :attr:`G4NDLLibrary.duplicates`.
* **Only the processes kika reads are indexed.** Today that is
  ``Elastic/CrossSection`` and ``Elastic/FS``, at the library root and nowhere
  else: ``JENDL_HE/neutron/Elastic`` is a different, high-energy data set that
  ships in the same tarball, and walking every directory named ``Elastic``
  would mix the two.

Nothing is read until asked for; building the index lists two directories.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from kika.g4ndl.exceptions import G4NDLError, IsotopeNotFoundError
from kika.g4ndl.names import IsotopeKey, TargetLike, parse_file_name, parse_target
from kika.g4ndl.parse import parse_cross_section, parse_elastic_fs
from kika.g4ndl.records import CrossSectionRecord, ElasticFSRecord
from kika.g4ndl.stream import DEFAULT_MAX_BYTES, read_text
from kika.g4ndl.tokens import TokenStream

__all__ = ["G4NDLLibrary", "IndexedFile", "open"]

#: Process name -> the subdirectories kika indexes for it, in reading order.
PROCESSES: Dict[str, Tuple[str, ...]] = {
    "elastic": ("Elastic/CrossSection", "Elastic/FS"),
}

@dataclass(frozen=True)
class IndexedFile:
    """One isotope's file in one subdirectory.

    ``path`` is what gets read; ``shadowed`` is the plain-text twin a ``.z``
    took precedence over, when there is one.
    """

    key: IsotopeKey
    subdir: str
    path: Path
    elementName: str
    shadowed: Optional[Path] = None

    @property
    def compressed(self) -> bool:
        return self.path.suffix == ".z"


class G4NDLLibrary:
    """Index of a G4NDL library directory.

    Parameters
    ----------
    root
        The directory ``G4NEUTRONHPDATA`` would point at, e.g. ``.../G4NDL4.7.1``.
    max_bytes
        Ceiling on the decompressed size of any one file.

    Raises
    ------
    G4NDLError
        If ``root`` is not a directory or holds none of the indexed
        subdirectories — most often a path one level too high, since the
        IAEA tarballs nest ``JEFF-4.0/`` inside ``JEFF-4.0/``.
    """

    def __init__(self, root, max_bytes: int = DEFAULT_MAX_BYTES):
        self.root = Path(root)
        self.max_bytes = max_bytes
        if not self.root.is_dir():
            raise G4NDLError(f"not a directory: {self.root}")
        self._index: Dict[Tuple[str, IsotopeKey], IndexedFile] = {}
        #: Files in an indexed subdirectory whose name is not ``Z_A[mM]_Element``.
        self.unindexed: List[Path] = []
        #: ``(text, .z)`` pairs where the ``.z`` won.
        self.duplicates: List[Tuple[Path, Path]] = []
        found = False
        for subdirs in PROCESSES.values():
            for sub in subdirs:
                d = self.root / sub
                if d.is_dir():
                    found = True
                    self._indexDirectory(sub, d)
        if not found:
            nested = [p for p in self.root.iterdir()
                      if p.is_dir() and (p / "Elastic").is_dir()]
            hint = f"; did you mean {nested[0]}?" if nested else ""
            raise G4NDLError(f"{self.root} holds no Elastic/CrossSection or "
                             f"Elastic/FS directory{hint}")

    def _indexDirectory(self, sub: str, d: Path) -> None:
        plain: Dict[str, Path] = {}
        compressed: Dict[str, Path] = {}
        for p in d.iterdir():
            if not p.is_file():
                continue
            if p.suffix == ".z":
                compressed[p.name[:-2]] = p
            else:
                plain[p.name] = p
        for stem in sorted(set(plain) | set(compressed)):
            parsed = parse_file_name(stem)
            if parsed is None:
                self.unindexed.extend(x for x in (plain.get(stem), compressed.get(stem)) if x)
                continue
            key, element = parsed
            path = compressed.get(stem) or plain[stem]
            shadowed = plain.get(stem) if stem in compressed else None
            if shadowed is not None:
                self.duplicates.append((shadowed, path))
            self._index[(sub, key)] = IndexedFile(key, sub, path, element, shadowed)

    # ------------------------------------------------------------- queries
    def isotopes(self, process: str = "elastic") -> List[IsotopeKey]:
        """Isotopes with **every** file ``process`` needs, sorted by (Z, A, M).

        ``0_0_Zero``, which both libraries ship in ``Elastic/CrossSection``
        only, is therefore not listed.
        """
        subdirs = self._subdirs(process)
        keys = {k for (s, k) in self._index if s == subdirs[0]}
        for s in subdirs[1:]:
            keys &= {k for (s2, k) in self._index if s2 == s}
        return sorted(keys, key=lambda k: (k.Z, -1 if k.A is None else k.A, k.M))

    def __contains__(self, target) -> bool:
        try:
            key = parse_target(target)
        except ValueError:
            return False
        return all((s, key) in self._index for s in PROCESSES["elastic"])

    def __len__(self) -> int:
        return len(self.isotopes())

    def __iter__(self) -> Iterator[IsotopeKey]:
        return iter(self.isotopes())

    def presentTopLevel(self) -> List[str]:
        """Top-level directories present, indexed or not — for partial-read reports."""
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def describe(self) -> dict:
        """The library as a viewer lists it, from the index alone: no file is
        opened, so it stays cheap on a 560-isotope library on a network share.

        ``isotopes`` holds one entry per isotope :meth:`isotopes` returns, with
        the ``target`` name :meth:`read` takes (``Fe56``, ``Cnat``, ``Co58m1``),
        its GNDS id (``Fe56``, ``C``, ``Co58_m1``), ``Z``, ``A`` (``None`` for
        a natural element), ``M``, the element name of its file and whether
        its files are ``.z``. ``unread`` names the top-level directories kika
        does not read yet (``Capture``, ``Inelastic``, ...).
        """
        from kika.g4ndl.decode import targetId

        entries = []
        for key in self.isotopes():
            files = [self._index[(s, key)] for s in PROCESSES["elastic"]]
            entries.append(dict(target=str(key), id=targetId(key), Z=key.Z, A=key.A, M=key.M,
                                element=files[0].elementName,
                                compressed=all(f.compressed for f in files)))
        read = {s.split("/")[0] for subs in PROCESSES.values() for s in subs}
        return dict(root=str(self.root), name=self.root.name, isotopes=entries,
                    processes=sorted(PROCESSES),
                    unread=[d for d in self.presentTopLevel() if d not in read],
                    unindexed=len(self.unindexed), duplicates=len(self.duplicates))

    def locate(self, target: TargetLike, subdir: str) -> IndexedFile:
        """The file holding ``target`` in ``subdir``, e.g. ``"Elastic/FS"``."""
        if subdir not in {s for subs in PROCESSES.values() for s in subs}:
            raise ValueError(f"subdir {subdir!r} is not indexed; "
                             f"indexed: {sorted(s for v in PROCESSES.values() for s in v)}")
        key = parse_target(target)
        try:
            return self._index[(subdir, key)]
        except KeyError:
            raise IsotopeNotFoundError(
                f"{key} ({key.Z}_{'nat' if key.A is None else key.A}"
                f"{f'm{key.M}' if key.M else ''}) has no file in "
                f"{self.root / subdir}; kika does not substitute the natural "
                "element or a neighbouring isotope") from None

    def tokens(self, target: TargetLike, subdir: str) -> TokenStream:
        """Open ``target``'s file in ``subdir`` as a :class:`TokenStream`."""
        f = self.locate(target, subdir)
        return TokenStream(read_text(f.path, self.max_bytes), path=f.path)

    def crossSection(self, target: TargetLike) -> CrossSectionRecord:
        """``target``'s ``Elastic/CrossSection``, parsed to its last token."""
        return parse_cross_section(self.tokens(target, "Elastic/CrossSection"))

    def elasticFinalState(self, target: TargetLike) -> ElasticFSRecord:
        """``target``'s ``Elastic/FS``, parsed to its last token."""
        return parse_elastic_fs(self.tokens(target, "Elastic/FS"))

    def read(self, target: TargetLike):
        """``target`` as a :class:`~kika.nuclear_data.model.suite.ReactionSuite`.

        **Elastic only (MT2)**: the other processes the library holds are not
        read, and ``suite.report`` lists them. This is what
        ``kika.read(root, format="g4ndl", target=...)`` calls. Importing the
        decoder here, not at module scope, keeps ``import kika.g4ndl`` from
        waking the model.
        """
        from kika.g4ndl.decode import decodeElastic

        key = parse_target(target)
        suite, _ = decodeElastic(self.crossSection(key), self.elasticFinalState(key),
                                 key, library=self)
        return suite

    def _subdirs(self, process: str) -> Tuple[str, ...]:
        try:
            return PROCESSES[process]
        except KeyError:
            raise ValueError(f"process {process!r} is not read by kika yet; "
                             f"supported: {sorted(PROCESSES)}") from None

    def __repr__(self) -> str:
        return f"G4NDLLibrary({str(self.root)!r}, {len(self)} elastic isotopes)"


def open(root, max_bytes: int = DEFAULT_MAX_BYTES) -> G4NDLLibrary:  # noqa: A001
    """Index the G4NDL library at ``root``. See :class:`G4NDLLibrary`."""
    return G4NDLLibrary(root, max_bytes=max_bytes)
