"""``cross_sections.xml``: the index of an OpenMC library, and the overlay.

The index plays the part ``xsdir`` plays for MCNP (:mod:`kika.ace.xsdir`):

.. code-block:: xml

    <cross_sections>
      <directory>/abs/path</directory>                       <!-- optional -->
      <library materials="U235" path="neutron/U235.h5" type="neutron"/>
      <depletion_chain path="chain.xml" type="depletion_chain"/>      <!-- optional -->
    </cross_sections>

The C++ accepts ``type`` in ``neutron``, ``thermal``, ``photon`` and ``wmp`` and
iterates only ``<library>`` nodes; a relative ``path`` is joined to
``<directory>`` when there is one and to the XML's own directory otherwise
(``cross_sections.cpp:47-79, 305-317``). ``<depletion_chain>`` is read by
OpenMC's Python only, and kept here so it round trips.

**The overlay** (:func:`overlay`) is what H6 and H7 need: an index in a new
directory that sends a few materials to kika's files and every other one to the
official library *where it already is*, so a benchmark can swap U-235 without
copying 32 GB. Paths to the base library are written relative to the new index
when they can be (the same drive), so one overlay works from Windows and from
the WSL alike.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .exceptions import HDF5FormatError, NuclideNotFoundError

__all__ = ["LibraryEntry", "CrossSectionsXml", "Library", "open", "overlay",
           "LIBRARY_TYPES", "registerFile"]

#: What the C++ accepts (``cross_sections.cpp:47-61``); ``wmp`` is reserved for
#: the windowed multipole, parked by Juan on 2026-10-11 (roadmap D1).
LIBRARY_TYPES = ("neutron", "thermal", "photon", "wmp")


@dataclass
class LibraryEntry:
    """One ``<library>``: the materials one file holds, where it is, and its kind."""

    materials: Tuple[str, ...]
    path: str
    type: str

    def __post_init__(self) -> None:
        if isinstance(self.materials, str):
            self.materials = tuple(self.materials.split())
        self.materials = tuple(self.materials)
        if self.type not in LIBRARY_TYPES:
            raise ValueError(f"library type must be one of {LIBRARY_TYPES}, got {self.type!r}")


@dataclass
class CrossSectionsXml:
    """The parsed index, kept as written: relative paths stay relative."""

    entries: List[LibraryEntry] = field(default_factory=list)
    directory: Optional[str] = None
    depletionChain: Optional[str] = None
    #: Where it was read from, which relative paths resolve against.
    source: Optional[Path] = None

    @classmethod
    def read(cls, path) -> "CrossSectionsXml":
        path = Path(path)
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise HDF5FormatError(f"not well-formed XML: {exc}", path=path) from None
        if root.tag != "cross_sections":
            raise HDF5FormatError(f"root element is <{root.tag}>, not <cross_sections>", path=path)
        index = cls(source=path)
        directory = root.find("directory")
        if directory is not None and (directory.text or "").strip():
            index.directory = directory.text.strip()
        for node in root.findall("library"):
            index.entries.append(LibraryEntry(
                materials=tuple((node.get("materials") or "").split()),
                path=node.get("path"), type=node.get("type")))
        chain = root.find("depletion_chain")
        if chain is not None:
            index.depletionChain = chain.get("path")
        return index

    def write(self, path) -> Path:
        """Write the index in the layout ``DataLibrary.export_to_xml`` produces."""
        path = Path(path)
        root = ET.Element("cross_sections")
        if self.directory:
            ET.SubElement(root, "directory").text = self.directory
        for entry in self.entries:
            ET.SubElement(root, "library", materials=" ".join(entry.materials),
                          path=entry.path, type=entry.type)
        if self.depletionChain:
            ET.SubElement(root, "depletion_chain", path=self.depletionChain,
                          type="depletion_chain")
        ET.indent(root, space="  ")
        path.parent.mkdir(parents=True, exist_ok=True)
        # lxml (what OpenMC writes with) closes empty elements as "/>", the
        # standard library as " />"; matched so an index read and written back
        # is the same text.
        payload = ET.tostring(root, encoding="unicode").replace(" />", "/>")
        path.write_text("<?xml version='1.0' encoding='UTF-8'?>\n" + payload + "\n",
                        encoding="utf-8", newline="\n")
        self.source = path
        return path

    def base(self) -> Path:
        """What a relative ``path`` is joined to."""
        if self.directory:
            return Path(self.directory)
        if self.source is None:
            raise ValueError("an index that was never read or written has no directory")
        return self.source.parent

    def resolve(self, entry: LibraryEntry) -> Path:
        candidate = Path(entry.path)
        return candidate if candidate.is_absolute() else self.base() / candidate

    def find(self, material: str, type: str = "neutron") -> LibraryEntry:
        for entry in self.entries:
            if entry.type == type and material in entry.materials:
                return entry
        raise NuclideNotFoundError(
            f"no {type} library holds {material!r} in {self.source or 'this index'}; "
            f"the look-up is exact (OpenMC names metastables Am242_m1, thermal "
            f"tables c_H_in_H2O, photon files by element)")

    def materials(self, type: str = "neutron") -> List[str]:
        return [m for entry in self.entries if entry.type == type for m in entry.materials]


def registerFile(path, relativeTo: Optional[Path] = None) -> LibraryEntry:
    """The entry for one ``.h5`` file, read off the file as ``DataLibrary.register_file`` does.

    ``type`` is the root ``filetype`` without its ``data_`` prefix and
    ``materials`` the top-level group names (``library.py:60-75``).
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        filetype = handle.attrs.get("filetype")
        if filetype is None:
            raise HDF5FormatError("no root 'filetype' attribute", path=path)
        kind = (filetype.decode() if isinstance(filetype, bytes) else str(filetype))
        if not kind.startswith("data_"):
            raise HDF5FormatError(f"filetype {kind!r} does not start with 'data_'", path=path)
        materials = tuple(handle.keys())
    shown = _relative(path, relativeTo) if relativeTo is not None else str(path)
    return LibraryEntry(materials=materials, path=shown, type=kind[5:])


def _relative(target: Path, start: Path) -> str:
    """``target`` relative to ``start`` in POSIX form, or absolute across drives."""
    try:
        return PurePosixPath(*Path(os.path.relpath(Path(target).resolve(), Path(start).resolve())).parts).as_posix()
    except ValueError:
        return Path(target).resolve().as_posix()


class Library:
    """An opened OpenMC library: its index and the files it names."""

    def __init__(self, index: CrossSectionsXml) -> None:
        self.index = index

    def __repr__(self) -> str:
        counts = {t: len(self.index.materials(t)) for t in LIBRARY_TYPES}
        shown = ", ".join(f"{n} {t}" for t, n in counts.items() if n)
        return f"<Library {self.index.source} | {shown or 'empty'}>"

    def path(self, material: str, type: str = "neutron") -> Path:
        return self.index.resolve(self.index.find(material, type))

    def materials(self, type: str = "neutron") -> List[str]:
        return self.index.materials(type)

    def read(self, material: str, type: str = "neutron", **kwargs):
        """The material decoded into the model (neutron: a ``ReactionSuite``)."""
        from .api import read

        return read(self.path(material, type), **kwargs)

    def describe(self) -> dict:
        """The library as a viewer lists it, from the index and the file sizes
        alone (no ``.h5`` is opened): one row per neutron nuclide, with its
        ``Z``/``A``/``M`` parsed from the OpenMC name (``A`` is ``None`` for a
        natural element, ``C0``), and the thermal and photon materials by name.
        ``missing`` counts index entries whose file is not on disk;
        ``single_file`` is true for a library opened from one ``.h5``."""
        from .names import ATOMIC_NUMBER_TO_SYMBOL, parseGndsName

        nuclides, unparsed, missing = [], [], 0
        for entry in self.index.entries:
            if entry.type != "neutron":
                continue
            path = self.index.resolve(entry)
            try:
                size = path.stat().st_size
            except OSError:
                missing += 1
                continue
            for material in entry.materials:
                try:
                    Z, A, M = parseGndsName(material)
                except ValueError:
                    unparsed.append(material)
                    continue
                nuclides.append(dict(target=material, id=material, Z=Z, A=A or None, M=M,
                                     element=ATOMIC_NUMBER_TO_SYMBOL[Z], file_name=path.name,
                                     size=size))
        nuclides.sort(key=lambda n: (n["Z"], n["A"] or 0, n["M"]))
        source = self.index.source
        single = source is not None and source.suffix.lower() in (".h5", ".hdf5")
        root = source.parent if source is not None else None
        name = source.stem if single else (root.name if root else "")
        return dict(root=str(source if single else root) if root else "",
                    index="" if single else (str(source) if source else ""),
                    name=name, single_file=single, isotopes=nuclides,
                    thermal=self.materials("thermal"), photon=self.materials("photon"),
                    unparsed=unparsed, missing=missing,
                    depletion_chain=self.index.depletionChain)


def open(path) -> Library:  # noqa: A001 - mirrors kika.g4ndl.open
    """A library from its ``cross_sections.xml``, the directory holding one,
    or a single ``.h5`` file: a library of that one file, indexed in memory
    from the file itself (:func:`registerFile`), as GNDS opens one ``.gnds.xml``.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "cross_sections.xml"
    if not path.is_file():
        raise FileNotFoundError(f"no cross_sections.xml at {path}")
    if path.suffix.lower() in (".h5", ".hdf5"):
        entry = registerFile(path, relativeTo=path.parent)
        return Library(CrossSectionsXml(entries=[entry], directory=str(path.parent), source=path))
    return Library(CrossSectionsXml.read(path))


def overlay(base, replacements: Mapping[str, object], output, *,
            types: Optional[Mapping[str, str]] = None) -> CrossSectionsXml:
    """An index at ``output`` that sends ``replacements`` to new files and the rest to ``base``.

    ``replacements`` maps a material name to the ``.h5`` kika wrote for it.
    A replaced material keeps its place in the index; one the base library does
    not have is appended. ``types`` gives the library type per material when it
    is not ``neutron``. The new file's own top-level group is checked against the
    material name, because the C++ opens the group named in the index
    (``nuclide.cpp:1104-1170``) and a mismatch fails only at load time.
    """
    import h5py

    base = open(base).index if not isinstance(base, CrossSectionsXml) else base
    output = Path(output)
    outputDir = output.parent
    types = dict(types or {})
    remaining = dict(replacements)
    new = CrossSectionsXml(depletionChain=None)

    for name, file in remaining.items():
        with h5py.File(file, "r") as handle:
            if name not in handle:
                raise HDF5FormatError(f"holds {list(handle)} and not {name!r}", path=Path(file))

    for entry in base.entries:
        replaced = [m for m in entry.materials if m in remaining and types.get(m, "neutron") == entry.type]
        if replaced and len(entry.materials) == 1:
            file = remaining.pop(replaced[0])
            new.entries.append(LibraryEntry(entry.materials, _relative(Path(file), outputDir), entry.type))
            continue
        if replaced:
            raise NotImplementedError(f"{entry.path} holds several materials {entry.materials}; "
                                      f"replacing one of them is not supported")
        new.entries.append(LibraryEntry(entry.materials, _relative(base.resolve(entry), outputDir), entry.type))
    for name, file in remaining.items():
        new.entries.append(LibraryEntry((name,), _relative(Path(file), outputDir), types.get(name, "neutron")))
    if base.depletionChain:
        chain = Path(base.depletionChain)
        chain = chain if chain.is_absolute() else base.base() / chain
        new.depletionChain = _relative(chain, outputDir)
    new.write(output)
    return new
