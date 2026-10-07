"""The XML layer: one parsed file, its labels, and the primitives §2 defines.

Everything above this module works in objects and numpy arrays; this is the
only place that knows the format is XML at all. It deliberately uses nothing
but the standard library and numpy, so the reader can be handed to the SINBAD
subgroup on its own, and so ``import kika.sinbad`` stays cheap -- pandas and
the GNDS model are reached at call time, in :mod:`kika.sinbad.content` and
:mod:`kika.sinbad.gnds`.

Three things here are worth knowing about.

**Labels are global, and shared across files.** §1: a calculations file may use
the labels of its benchmark. :class:`Document` indexes every element that
carries a ``label`` an object is referred to by, and
:class:`LabelIndex` chains the benchmark's index with those of its calculations
files, so ``benchmark["reactionRate-S32"]`` and
``calculations["enea-tort-3.2"]["reactionRate-S32"]`` find the same element.
``grid``, ``axis`` and ``double`` are excluded: their ``label`` is an axis name
or a scalar's name, not an identifier anything points at.

**A grid may be written once and linked.** §2.2: ``<link href="..."/>`` in
place of ``<values>``, and the href is an XPath -- the only one in the format.
:func:`resolve_href` translates it into what :mod:`xml.etree` accepts.

**A blank table cell is a missing value, not a zero.** §5.4.4 writes it
``<td/>``, and the specification's whitespace form cannot express it at all.
:func:`read_table` returns it as ``''`` and the float conversion in
:mod:`kika.sinbad.content` turns it into ``nan``.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from kika.sinbad._constants import BENCHMARK_ROOT, CALCULATIONS_ROOT
from kika.sinbad.exceptions import LabelNotFoundError, SinbadFormatError

__all__ = [
    "Document",
    "LabelIndex",
    "resolve_href",
    "read_values",
    "read_table",
    "sha1_of",
    "as_float",
    "as_floats",
]

#: Elements whose ``label`` names the element itself, so that another object
#: may point at it. ``grid``/``axis`` label an axis, ``double`` labels a scalar
#: inside its parent, ``range`` likewise -- none of them is a reference target.
_NOT_AN_IDENTIFIER = frozenset({"grid", "axis", "double", "range", "magnitude"})


def _parse(path: Path) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise SinbadFormatError(f"{path} is not well-formed XML: {exc}") from exc
    except OSError as exc:
        raise SinbadFormatError(f"cannot read {path}: {exc}") from exc


class Document:
    """
    One parsed SINBAD XML file: a benchmark or a set of calculations.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to read.

    Attributes
    ----------
    path : pathlib.Path
        Where it was read from.
    root : xml.etree.ElementTree.Element
        The root element, ``sinbad`` or ``sinbadCalculations``.
    kind : str
        ``"benchmark"`` or ``"calculations"``.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path).expanduser().resolve()
        self.root = _parse(self.path)
        if self.root.tag == BENCHMARK_ROOT:
            self.kind = "benchmark"
        elif self.root.tag == CALCULATIONS_ROOT:
            self.kind = "calculations"
        else:
            raise SinbadFormatError(
                f"{self.path} has root <{self.root.tag}>; a SINBAD file has "
                f"<{BENCHMARK_ROOT}> or <{CALCULATIONS_ROOT}>"
            )
        self.labels: Dict[str, ET.Element] = {
            e.get("label"): e
            for e in self.root.iter()
            if e.get("label") and e.tag not in _NOT_AN_IDENTIFIER
        }

    @property
    def format_version(self) -> Optional[str]:
        return self.root.get("formatVersion")

    def find(self, path: str) -> Optional[ET.Element]:
        """``Element.find`` on the root, spelled once."""
        return self.root.find(path)

    def findall(self, path: str) -> List[ET.Element]:
        """``Element.findall`` on the root, spelled once."""
        return self.root.findall(path)

    def checksum(self) -> str:
        """The sha1 of the file as it is on disk -- what a calculations file cites."""
        return sha1_of(self.path)

    def __repr__(self) -> str:
        return f"<Document {self.path.name} ({self.kind}), {len(self.labels)} labels>"


class LabelIndex:
    """
    Label look-up across a benchmark and the calculations files beside it.

    The benchmark is searched first, then each calculations file in the order
    it was added, which is the precedence §1 describes: a calculations file
    *uses* the benchmark's labels and defines only what the benchmark has not.

    Parameters
    ----------
    documents : sequence of Document
        The benchmark first.
    """

    def __init__(self, documents: Sequence[Document]):
        self._documents = list(documents)

    def add(self, document: Document) -> None:
        self._documents.append(document)

    def element(self, label: str) -> ET.Element:
        """Return the element carrying ``label``, or raise."""
        for doc in self._documents:
            found = doc.labels.get(label)
            if found is not None:
                return found
        raise LabelNotFoundError(
            f"no object labelled {label!r} in "
            + ", ".join(d.path.name for d in self._documents)
        )

    def optional(self, label: Optional[str]) -> Optional[ET.Element]:
        """Like :meth:`element`, but ``None`` for a missing or absent label."""
        if not label:
            return None
        for doc in self._documents:
            found = doc.labels.get(label)
            if found is not None:
                return found
        return None

    def document_of(self, label: str) -> Document:
        """Which file defines ``label``."""
        for doc in self._documents:
            if label in doc.labels:
                return doc
        raise LabelNotFoundError(f"no object labelled {label!r}")

    def __contains__(self, label: object) -> bool:
        return any(label in d.labels for d in self._documents)

    def __iter__(self) -> Iterator[str]:
        seen = set()
        for doc in self._documents:
            for label in doc.labels:
                if label not in seen:
                    seen.add(label)
                    yield label

    def resolve_href(self, href: str) -> ET.Element:
        """Resolve a ``link/@href`` against every document in the index."""
        for doc in self._documents:
            found = resolve_href(doc.root, href)
            if found is not None:
                return found
        raise SinbadFormatError(f"link href {href!r} resolves to nothing")


def resolve_href(root: ET.Element, href: str) -> Optional[ET.Element]:
    """
    Resolve a GNDS ``@href`` XPath against ``root``.

    :mod:`xml.etree` implements a subset of XPath that covers what the format
    uses -- ``label`` predicates and child steps -- but it will not accept a
    pattern starting at the document root. ``//x`` becomes ``.//x`` and ``/x``
    becomes ``./x``; anything else is passed through.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
    href : str
        For example ``"//*[@label='groupStructure-BUGLE47']/grid/values"``.

    Returns
    -------
    xml.etree.ElementTree.Element or None
    """
    path = href
    if path.startswith("//"):
        path = "." + path
    elif path.startswith("/"):
        path = "." + path
    try:
        return root.find(path)
    except SyntaxError as exc:  # ElementTree's own error for an unsupported XPath
        raise SinbadFormatError(f"cannot follow href {href!r}: {exc}") from exc


def read_values(element: ET.Element, index: "LabelIndex | None" = None) -> np.ndarray:
    """
    Read a ``values`` body, or follow the ``link`` standing in for one.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        A ``values`` element, or a parent holding one (``grid``, ``array``).
    index : LabelIndex, optional
        Needed only to follow a ``link``.

    Returns
    -------
    numpy.ndarray
        One-dimensional, float.
    """
    if element.tag != "values":
        link = element.find("link")
        if link is not None:
            href = link.get("href", "")
            if index is None:
                raise SinbadFormatError(
                    f"values are linked ({href!r}) and no index was given to follow it"
                )
            return read_values(index.resolve_href(href), index)
        found = element.find("values")
        if found is None:
            raise SinbadFormatError(
                f"<{element.tag}> holds neither <values> nor <link>"
            )
        element = found
    return as_floats(element.text)


def read_table(table: ET.Element) -> Tuple[List[ET.Element], List[List[str]]]:
    """
    Read a GNDS ``table`` into its column headers and its cells, as text.

    GNDS-2.1 §5.4.4 gives two bodies: a whitespace-separated block of numbers
    read row by row, and explicit ``<tr sep="td">`` rows of ``<td>`` cells,
    which is the only one that can carry text or a blank. Both are returned the
    same way, and no conversion is done here -- a blank cell is ``''``.

    Parameters
    ----------
    table : xml.etree.ElementTree.Element

    Returns
    -------
    columns : list of Element
        The ``column`` elements, in document order.
    rows : list of list of str
        One list of cells per row.
    """
    headers = table.find("columnHeaders")
    columns = list(headers.findall("column")) if headers is not None else []
    data = table.find("data")
    if data is None:
        return columns, []
    if data.get("sep") == "tr":
        rows = []
        for tr in data.findall("tr"):
            if tr.get("sep") == "td":
                rows.append([(td.text or "").strip() for td in tr.findall("td")])
            else:
                rows.append((tr.text or "").split())
        return columns, rows
    ncol = int(table.get("columns") or len(columns) or 1)
    flat = (data.text or "").split()
    return columns, [flat[i:i + ncol] for i in range(0, len(flat), ncol)]


def sha1_of(path: Union[str, Path]) -> str:
    """The sha1 of a file, as the format writes checksums."""
    digest = hashlib.sha1()  # noqa: S324 - the format specifies sha1
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def as_float(text: Optional[str]) -> float:
    """A cell or attribute as a float; blank and absent are ``nan``."""
    if text is None:
        return float("nan")
    text = text.strip()
    if not text:
        return float("nan")
    return float(text)


def as_floats(text: Optional[str]) -> np.ndarray:
    """A whitespace-separated body as a 1-D float array."""
    return np.array([float(x) for x in (text or "").split()], dtype=float)
