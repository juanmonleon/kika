"""The entry blocks: who the benchmark is, what it was measured on, what was run.

§4 of ``Data_structures.md``. Everything here is small, flat and frozen where
it can be: these are the descriptive blocks of an entry, and a reader wants
them to read like the entry reads.

Two of them do more than describe.

:class:`ExternalFile` knows the entry it belongs to, so
:meth:`ExternalFile.verify` can check the sha1 the format stores against the
file on disk. That is the only mechanism the format has for noticing that a
benchmark and the repository it was transcribed from have drifted apart, and it
costs one line to use.

:class:`Comparison` holds published C/E values *and* can recompute them from
the two data objects it names, in the convention it declares. The entry's own
numbers stay the reference; :meth:`Comparison.recompute` says whether they can
be reproduced from what the files hold, which is a different and more useful
question than whether they were copied correctly.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from kika.sinbad._xml import as_float, sha1_of
from kika.sinbad.content import Double, Range, Table, read_doubles
from kika.sinbad.exceptions import LabelNotFoundError, SinbadError

__all__ = [
    "LegacyId",
    "RelatedEntry",
    "Identification",
    "Person",
    "Date",
    "BibItem",
    "ComputerCode",
    "Documentation",
    "QualityAssessment",
    "Issue",
    "Status",
    "ExternalFile",
    "Normalisation",
    "ValueConvention",
    "Direction",
    "NamedPoint",
    "CoordinateFrame",
    "Position",
    "RadiationSource",
    "Cover",
    "Detector",
    "Absence",
    "Calculation",
    "Comparison",
]


def _text(element: Optional[ET.Element]) -> str:
    if element is None or element.text is None:
        return ""
    return " ".join(element.text.split())


# --------------------------------------------------------------------------
# §4.1 identification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyId:
    """An identifier the entry carried before SINBAD v2."""

    scheme: str
    value: str
    source: Optional[str] = None


@dataclass(frozen=True)
class RelatedEntry:
    """Another SINBAD entry, and how it relates to this one."""

    short_code: str
    title: str
    relation: str


@dataclass(frozen=True)
class Identification:
    """
    §4.1. The SINBAD nomenclature keys, and the entry's other names.

    Attributes
    ----------
    domain : str
        ``FIS``, ``FUS`` or ``ACC``.
    purpose, geometry, material, source, sequence, measurement : str
        The remaining nomenclature keys.
    volume : str or None
    legacy_ids : tuple of LegacyId
    related : tuple of RelatedEntry
    """

    domain: str = ""
    purpose: str = ""
    geometry: str = ""
    material: str = ""
    source: str = ""
    sequence: str = ""
    measurement: str = ""
    volume: Optional[str] = None
    legacy_ids: Tuple[LegacyId, ...] = ()
    related: Tuple[RelatedEntry, ...] = ()

    @classmethod
    def _read(cls, element: Optional[ET.Element]) -> "Identification":
        if element is None:
            return cls()
        return cls(
            domain=element.get("domain", ""),
            purpose=element.get("purpose", ""),
            geometry=element.get("geometry", ""),
            material=element.get("material", ""),
            source=element.get("source", ""),
            sequence=element.get("sequence", ""),
            measurement=element.get("measurement", ""),
            volume=element.get("volume"),
            legacy_ids=tuple(
                LegacyId(e.get("scheme", ""), e.get("value", ""), e.get("source"))
                for e in element.findall("legacyId")
            ),
            related=tuple(
                RelatedEntry(e.get("shortCode", ""), e.get("title", ""), e.get("relation", ""))
                for e in element.findall("relatedEntry")
            ),
        )


# --------------------------------------------------------------------------
# §4.2 documentation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Person:
    """An author or a contributor, with their affiliations."""

    name: str
    affiliations: Tuple[str, ...] = ()
    note: str = ""
    contributor_type: Optional[str] = None

    @classmethod
    def _read(cls, element: ET.Element) -> "Person":
        return cls(
            name=element.get("name", ""),
            affiliations=tuple(
                a.get("name", "") for a in element.findall("affiliations/affiliation")
            ),
            note=_text(element.find("note")),
            contributor_type=element.get("contributorType"),
        )


@dataclass(frozen=True)
class Date:
    """A date, as far as the entry knows it, and what happened then."""

    date_type: str
    value: str
    event: str = ""

    @property
    def year(self) -> Optional[int]:
        try:
            return int(self.value[:4])
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class BibItem:
    """One numbered reference. ``xref`` is what other nodes point at."""

    xref: str
    text: str


@dataclass(frozen=True)
class ComputerCode:
    """A code a calculation was run with."""

    label: str
    name: str
    version: str = ""
    note: str = ""


@dataclass(frozen=True)
class Documentation:
    """
    §4.2. Who made it, when, what it is called, and what it cites.

    Attributes
    ----------
    title, abstract : str
    authors, contributors : tuple of Person
    dates : tuple of Date
    computer_codes : tuple of ComputerCode
    bibliography : tuple of BibItem
    """

    title: str = ""
    abstract: str = ""
    authors: Tuple[Person, ...] = ()
    contributors: Tuple[Person, ...] = ()
    dates: Tuple[Date, ...] = ()
    computer_codes: Tuple[ComputerCode, ...] = ()
    bibliography: Tuple[BibItem, ...] = ()

    @classmethod
    def _read(cls, element: Optional[ET.Element]) -> "Documentation":
        if element is None:
            return cls()
        return cls(
            title=_text(element.find("title")),
            abstract=_text(element.find("abstract")),
            authors=tuple(Person._read(a) for a in element.findall("authors/author")),
            contributors=tuple(
                Person._read(c) for c in element.findall("contributors/contributor")
            ),
            dates=tuple(
                Date(d.get("dateType", ""), d.get("value", ""), d.get("event", ""))
                for d in element.findall("dates/date")
            ),
            computer_codes=tuple(
                ComputerCode(
                    c.get("label", ""), c.get("name", ""), c.get("version", ""),
                    _text(c.find("note")),
                )
                for c in element.findall("computerCodes/computerCode")
            ),
            bibliography=tuple(
                BibItem(b.get("xref", ""), _text(b))
                for b in element.findall("bibliography/bibitem")
            ),
        )

    def bibitem(self, xref: str) -> BibItem:
        """
        One reference, by the ``xref`` other nodes cite.

        Parameters
        ----------
        xref : str
            The reference number, compared as text (``str(xref)``), so ``3`` and
            ``"3"`` find the same item.

        Returns
        -------
        ``BibItem``
            The item, with its ``xref`` and its ``text`` (the citation as the
            entry prints it).

        Raises
        ------
        KeyError
            If the bibliography has no item with that ``xref``.
        """
        for item in self.bibliography:
            if item.xref == str(xref):
                return item
        raise KeyError(f"no bibitem {xref!r}")

    @property
    def year(self) -> Optional[int]:
        """The earliest year any date gives -- when the entry was made."""
        years = [d.year for d in self.dates if d.year]
        return min(years) if years else None


# --------------------------------------------------------------------------
# §4.3 status
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityAssessment:
    """The rating the entry was given, and what the assessor was not sure of."""

    rating: str = ""
    xref: Optional[str] = None
    reservations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Issue:
    """
    §4.3. Something wrong or inconsistent in the sources, as the entry records it.

    Attributes
    ----------
    label : str
        ``F1``, ``F2``, ... -- cited from elsewhere in the file.
    kind : str
        ``inconsistency``, ``metadata``, ``unit``, ``frame``, ...
    about : tuple of str
        Labels or names the issue is about; empty when it is about the entry.
    text : str
    """

    label: str
    kind: str
    about: Tuple[str, ...]
    text: str

    @classmethod
    def _read(cls, element: ET.Element) -> "Issue":
        return cls(
            label=element.get("label", ""),
            kind=element.get("kind", ""),
            about=tuple((element.get("about") or "").split()),
            text=_text(element),
        )

    def __repr__(self) -> str:
        about = f" about {' '.join(self.about)}" if self.about else ""
        return f"<Issue {self.label} ({self.kind}){about}: {self.text[:60]}...>"


@dataclass(frozen=True)
class Status:
    """
    §4.3. Availability, maturity, the quality assessment and the known issues.

    Attributes
    ----------
    availability : str
        As the file states it; ``""`` when the entry has no status block.
    maturity_level : str or None
        As the file states it.
    quality : ``QualityAssessment`` or None
        The rating the entry was given (``rating``, the bibliography ``xref`` it
        cites, and the assessor's ``reservations`` as a tuple of text).
    issues : tuple of Issue
        What the entry says is wrong or inconsistent in its own sources.
    """

    availability: str = ""
    maturity_level: Optional[str] = None
    quality: Optional[QualityAssessment] = None
    issues: Tuple[Issue, ...] = ()

    @classmethod
    def _read(cls, element: Optional[ET.Element]) -> "Status":
        if element is None:
            return cls()
        assessment = element.find("qualityAssessment")
        return cls(
            availability=element.get("availability", ""),
            maturity_level=element.get("maturityLevel"),
            quality=(
                QualityAssessment(
                    rating=assessment.get("rating", ""),
                    xref=assessment.get("xref"),
                    reservations=tuple(
                        _text(r) for r in assessment.findall("reservation")
                    ),
                )
                if assessment is not None else None
            ),
            issues=tuple(Issue._read(i) for i in element.findall("issues/issue")),
        )


# --------------------------------------------------------------------------
# §2.6 external files
# --------------------------------------------------------------------------


@dataclass
class ExternalFile:
    """
    §2.6. A file of the entry repository, by path and checksum.

    The path is relative to the entry directory -- the ``00_``-``05_`` tree the
    benchmark was transcribed from -- not to the XML file. Point
    :meth:`resolve` at that directory to get an absolute path, and
    :meth:`verify` to check the sha1.

    Attributes
    ----------
    label, path, checksum, algorithm, role, format : str
    xref : str or None
        The bibliography item this file is.
    archive : str or None
        Label of the archive this file is inside; then ``path`` is the path
        within it.
    """

    label: str
    path: str
    checksum: Optional[str] = None
    algorithm: str = "sha1"
    role: str = ""
    format: str = ""
    xref: Optional[str] = None
    archive: Optional[str] = None

    @classmethod
    def _read(cls, element: ET.Element) -> "ExternalFile":
        return cls(
            label=element.get("label", ""),
            path=element.get("path", ""),
            checksum=element.get("checksum"),
            algorithm=element.get("algorithm", "sha1"),
            role=element.get("role", ""),
            format=element.get("format", ""),
            xref=element.get("xref"),
            archive=element.get("archive"),
        )

    def resolve(self, entry_root) -> Path:
        """
        The absolute path of this file under an entry directory.

        Parameters
        ----------
        entry_root : str or pathlib.Path
            The entry directory the ``path`` of the file is relative to. A leading
            ``~`` is expanded.

        Returns
        -------
        pathlib.Path
            ``entry_root / path``. The file is not checked to exist.

        Raises
        ------
        SinbadError
            If the file is inside an archive; its path is then relative to the
            archive, which this reader does not open.
        """
        if self.archive:
            raise SinbadError(
                f"{self.label!r} is inside the archive {self.archive!r}; "
                f"its path is relative to that archive, not to the entry"
            )
        return Path(entry_root).expanduser() / self.path

    def verify(self, entry_root) -> bool:
        """
        Whether the file on disk still has the checksum the entry recorded.

        Parameters
        ----------
        entry_root : str or pathlib.Path
            The entry directory the ``path`` of the file is relative to.

        Returns
        -------
        bool
            ``True`` when the sha1 of the file's bytes, as it is on disk now,
            equals the recorded checksum.

        Raises
        ------
        SinbadError
            If the file is inside an archive, the entry stored no checksum, the
            file is not there, or the checksum algorithm is not ``sha1``.
        """
        path = self.resolve(entry_root)
        if not self.checksum:
            raise SinbadError(f"{self.label!r} carries no checksum to verify")
        if not path.exists():
            raise SinbadError(f"{path} does not exist")
        if self.algorithm != "sha1":
            raise SinbadError(f"checksum algorithm {self.algorithm!r} is not supported")
        return sha1_of(path) == self.checksum

    def __repr__(self) -> str:
        where = f" in {self.archive}" if self.archive else ""
        return f"<ExternalFile {self.label!r} {self.role} {self.path}{where}>"


# --------------------------------------------------------------------------
# §4.4 definitions
# --------------------------------------------------------------------------


@dataclass
class Normalisation:
    """
    §4.4. What a set of numbers is *per*: 30 kW of reactor power, 1 W of plate.

    The format keeps this separate from the unit on purpose -- ``units`` holds
    the dimension, the convention holds *per watt of plate*. Mixing the two is
    how normalisations get lost.

    Attributes
    ----------
    label : str
        What a data object's ``normalisation`` points at.
    basis : str
        What the numbers are per, as the file states it.
    provenance : str or None
        As the file states it.
    value : Double or None
        The size of the normalisation (for example the power), in the unit the
        file declares; ``None`` when the entry gives none.
    """

    label: str
    basis: str = ""
    provenance: Optional[str] = None
    value: Optional[Double] = None

    @classmethod
    def _read(cls, element: ET.Element) -> "Normalisation":
        scalar = element.find("double")
        return cls(
            label=element.get("label", ""),
            basis=element.get("basis", ""),
            provenance=element.get("provenance"),
            value=Double._read(scalar) if scalar is not None else None,
        )

    def __repr__(self) -> str:
        size = f" = {self.value.value:g} {self.value.unit}" if self.value else ""
        return f"<Normalisation {self.label!r} ({self.basis}){size}>"


@dataclass(frozen=True)
class ValueConvention:
    """
    §4.4. What the stored values include -- background in, or background out.

    A data object names one by ``convention``, and a correction goes between
    two of them (``from`` and ``to``).

    Attributes
    ----------
    label : str
        What ``convention`` and a correction's ``from`` and ``to`` point at.
    basis : str
        What the values include, as the file states it.
    """

    label: str
    basis: str = ""

    @classmethod
    def _read(cls, element: ET.Element) -> "ValueConvention":
        return cls(label=element.get("label", ""), basis=element.get("basis", ""))


# --------------------------------------------------------------------------
# §4.5 frames and positions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Direction:
    """One axis of a coordinate frame, and where its zero is."""

    name: str
    description: str = ""


@dataclass(frozen=True)
class NamedPoint:
    """A named point of a frame, e.g. the nuclear centre."""

    name: str
    x: float = float("nan")
    y: float = float("nan")
    z: float = float("nan")


@dataclass(frozen=True)
class CoordinateFrame:
    """
    §4.5. The frame every coordinate in the entry is given in.

    Attributes
    ----------
    label : str
        What a data object's ``frame`` points at.
    unit : str
        The unit of the coordinates, as the file declares it; kika does not
        convert it.
    source : str
        Where the frame comes from, as the file states it.
    directions : tuple of ``Direction``
        One per axis, each with its ``name`` and ``description``.
    points : tuple of ``NamedPoint``
        The named points of the frame (for example the nuclear centre), each
        with ``name`` and ``x``, ``y``, ``z`` as the file gives them (NaN for a
        coordinate it omits). A point keeps no unit, and kika does not tie its
        coordinates to ``unit``; in the format they are in the frame's unit.
    """

    label: str
    unit: str = ""
    source: str = ""
    directions: Tuple[Direction, ...] = ()
    points: Tuple[NamedPoint, ...] = ()

    @classmethod
    def _read(cls, element: ET.Element) -> "CoordinateFrame":
        return cls(
            label=element.get("label", ""),
            unit=element.get("unit", ""),
            source=element.get("source", ""),
            directions=tuple(
                Direction(d.get("name", ""), d.get("description", ""))
                for d in element.findall("direction")
            ),
            points=tuple(
                NamedPoint(
                    p.get("name", ""), as_float(p.get("x")), as_float(p.get("y")),
                    as_float(p.get("z")),
                )
                for p in element.findall("point")
            ),
        )


@dataclass(frozen=True)
class Position:
    """
    §4.5. A measuring position: which layer it is in, and the shield in front of it.

    ``label`` is what every table's ``position`` column holds, which is how
    rows from different files are matched.

    Attributes
    ----------
    label : str
        The position name.
    layer : str or None
        The name of the geometry layer it is in.
    shield_thickness : float
        The shield in front of it, as the number the file gives; this object
        keeps no unit for it. NaN when the file gives none.
    """

    label: str
    layer: Optional[str] = None
    shield_thickness: float = float("nan")

    @classmethod
    def _read(cls, element: ET.Element) -> "Position":
        return cls(
            label=element.get("label", ""),
            layer=element.get("layer"),
            shield_thickness=as_float(element.get("shieldThickness")),
        )


# --------------------------------------------------------------------------
# §4.6-4.7 sources and detectors
# --------------------------------------------------------------------------


@dataclass
class RadiationSource:
    """
    §4.6. What produced the radiation, where it sat, and how it is distributed.

    Attributes
    ----------
    label, kind, particle : str
        The source name, its kind and the particle it emits, as the file states
        them.
    layer, material : str or None
        Names of the geometry layer and the material it sits in.
    spatial_distribution, z_dependence : str or None
        As the file states them.
    strength : str or None
        In the format, the label of the data object holding the source
        strength. kika stores the string and does not resolve it.
    description : str
        Free text of the file.
    spectrum_ref : str or None
        In the format, the label of the data object holding the energy
        spectrum. kika stores the string and does not resolve it.
    spectrum_status : str or None
        Why there is no spectrum, when the entry says so.
    spectrum_description : str
        Free text about the spectrum.
    """

    label: str
    kind: str = ""
    particle: str = ""
    layer: Optional[str] = None
    material: Optional[str] = None
    spatial_distribution: Optional[str] = None
    z_dependence: Optional[str] = None
    strength: Optional[str] = None
    description: str = ""
    spectrum_ref: Optional[str] = None
    spectrum_status: Optional[str] = None
    spectrum_description: str = ""

    @classmethod
    def _read(cls, element: ET.Element) -> "RadiationSource":
        spectrum = element.find("energySpectrum")
        return cls(
            label=element.get("label", ""),
            kind=element.get("kind", ""),
            particle=element.get("particle", ""),
            layer=element.get("layer"),
            material=element.get("material"),
            spatial_distribution=element.get("spatialDistribution"),
            z_dependence=element.get("zDependence"),
            strength=element.get("strength"),
            description=element.get("description", ""),
            spectrum_ref=spectrum.get("ref") if spectrum is not None else None,
            spectrum_status=spectrum.get("status") if spectrum is not None else None,
            spectrum_description=(
                spectrum.get("description", "") if spectrum is not None else ""
            ),
        )


@dataclass(frozen=True)
class Cover:
    """§4.7. What a foil was wrapped in -- cadmium, and how thick."""

    material: str = ""
    material_provenance: Optional[str] = None
    dimensions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Detector:
    """
    §4.7. One detector: what reaction it measures, and its dimensions.

    Dimensions are whatever the entry gives -- diameter, thickness, mass --
    keyed by their label, as :class:`~kika.sinbad.Double` or ``Range`` (an
    interval, with ``min``, ``max`` and ``unit``). Each keeps the unit the file
    declares for it.

    Attributes
    ----------
    label, kind : str
        The detector name and its kind, as the file states them.
    reaction : str or None
        The reaction it measures, e.g. ``"Au197(n,g)Au198"``.
    counting_system, form : str or None
        As the file states them.
    dimensions : dict
        Label to :class:`~kika.sinbad.Double` or ``Range``.
    cover : ``Cover`` or None
        What the detector was wrapped in: its ``material``, the provenance of
        that material and its own ``dimensions`` (same form as above).

    Examples
    --------
    >>> d = b.detectors["det-Au197"]                     # doctest: +SKIP
    >>> d.reaction, d.dimensions["thickness"].value      # doctest: +SKIP
    ('Au197(n,g)Au198', 0.05)
    """

    label: str
    kind: str = ""
    reaction: Optional[str] = None
    counting_system: Optional[str] = None
    form: Optional[str] = None
    dimensions: Dict[str, Any] = field(default_factory=dict)
    cover: Optional[Cover] = None

    @classmethod
    def _read(cls, element: ET.Element) -> "Detector":
        cover_node = element.find("cover")
        return cls(
            label=element.get("label", ""),
            kind=element.get("kind", ""),
            reaction=element.get("reaction"),
            counting_system=element.get("countingSystem"),
            form=element.get("form"),
            dimensions=read_doubles(element),
            cover=(
                Cover(
                    material=cover_node.get("material", ""),
                    material_provenance=cover_node.get("materialProvenance"),
                    dimensions=read_doubles(cover_node),
                )
                if cover_node is not None else None
            ),
        )

    @property
    def target(self) -> Optional[str]:
        """The target nuclide of the reaction, e.g. ``"Au197"``."""
        if not self.reaction:
            return None
        return self.reaction.split("(")[0]

    def __repr__(self) -> str:
        return f"<Detector {self.label!r} {self.kind} {self.reaction or ''}>"


@dataclass(frozen=True)
class Absence:
    """
    §4.10. Something the entry does not have, and why.

    *"An absence is a datum."* A missing value inside a table is a blank cell;
    this is for what has no place in the structures at all.

    Attributes
    ----------
    kind : str
        Why it is not there, as the file states it (``not stated``,
        ``not reported`` or ``measured, not reported`` in the specification's
        vocabulary, which is not enforced).
    text : str
        What is missing, as the entry says it.
    """

    kind: str
    text: str

    def __repr__(self) -> str:
        return f"<Absence ({self.kind}) {self.text}>"


# --------------------------------------------------------------------------
# §4.8-4.9 calculations and comparisons
# --------------------------------------------------------------------------


@dataclass
class Calculation:
    """
    §4.8. One run: the code, the library, the options, its input and output files.

    Attributes
    ----------
    label, code, evaluated_library : str
    library : str or None
        The processed library actually used (``BUGJEFF311.BOLIB``), as against
        the evaluation it came from (``JEFF-3.1.1``).
    input_files : tuple of str
        In the format, labels of ``externalFile`` entries (the ``ref`` of each
        ``inputFile``). kika stores the strings and does not resolve them.
    options : str
        The code options, as free text; ``""`` when the file gives none.
    xref : str or None
        In the format, the ``xref`` of the bibliography item describing the
        run. kika stores the string and does not resolve it.
    source_distribution, source_spectrum : str or None
        In the format, labels of the data objects used as the source
        distribution and the source spectrum. kika stores the strings and
        does not resolve them.
    output_files : tuple of str
        In the format, labels of ``externalFile`` entries holding the output
        (the ``ref`` of each ``outputFile``). kika stores the strings and does
        not resolve them.
    output_status : str or None
        Why the output is not distributed, when it is not. It is the first status
        found among the output files.
    """

    label: str
    code: Optional[str] = None
    library: Optional[str] = None
    evaluated_library: Optional[str] = None
    options: str = ""
    xref: Optional[str] = None
    source_distribution: Optional[str] = None
    source_spectrum: Optional[str] = None
    input_files: Tuple[str, ...] = ()
    output_files: Tuple[str, ...] = ()
    output_status: Optional[str] = None

    @classmethod
    def _read(cls, element: ET.Element) -> "Calculation":
        outputs = element.findall("outputFile")
        return cls(
            label=element.get("label", ""),
            code=element.get("code"),
            library=element.get("library"),
            evaluated_library=element.get("evaluatedLibrary"),
            options=element.get("options", ""),
            xref=element.get("xref"),
            source_distribution=element.get("sourceDistribution"),
            source_spectrum=element.get("sourceSpectrum"),
            input_files=tuple(
                f.get("ref", "") for f in element.findall("inputFile") if f.get("ref")
            ),
            output_files=tuple(f.get("ref", "") for f in outputs if f.get("ref")),
            output_status=next(
                (f.get("status") for f in outputs if f.get("status")), None
            ),
        )

    def __repr__(self) -> str:
        return f"<Calculation {self.label!r} {self.code} {self.library or self.evaluated_library or ''}>"


class Comparison:
    """
    §4.9. A published comparison of a calculated table with a measured one --
    or, from v0.4, with another calculated one of the same file (the ratio of
    two calculations). :attr:`against` says which.

    Row by row on ``position``, in the convention it declares. The published
    numbers are in ``table`` (a :class:`~kika.sinbad.Table`, or ``None`` when
    the file gives none), in the unit each column declares; :meth:`recompute`
    forms the same comparison from the two data objects, which is how a reader
    finds out whether the entry's C/E can be reproduced from the entry's own
    numbers.

    ``operator`` is ``ratio`` (calculated over measured), ``difference``
    (calculated minus measured) or ``relative`` (the difference over the
    measured), and ``ratio`` when the file gives none. The label attributes
    ``numerator_label``, ``denominator_label`` and ``convention`` are what the
    file names.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        The ``comparison`` element.
    entry : Calculations, optional
        The calculations file the comparison belongs to, which resolves
        :attr:`numerator` and :attr:`denominator`. Without it they are ``None``
        and :meth:`recompute` raises.

    Examples
    --------
    >>> c = calc.comparisons["CE-S32-flat"]              # doctest: +SKIP
    >>> c.to_dataframe().head(2)                         # doctest: +SKIP
    >>> c.recompute().head(2)                            # doctest: +SKIP
    """

    def __init__(self, element: ET.Element, entry: Any = None):
        self._element = element
        self._entry = entry
        self.label: str = element.get("label", "")
        self.operator: str = element.get("operator", "ratio")
        self.numerator_label: Optional[str] = element.get("numerator")
        self.denominator_label: Optional[str] = element.get("denominator")
        self.convention: Optional[str] = element.get("convention")
        self.source: str = element.get("source", "")
        table = element.find("table")
        self.table: Optional[Table] = Table(table) if table is not None else None

    @property
    def numerator(self):
        """The calculated data object."""
        return self._entry[self.numerator_label] if self._entry is not None else None

    @property
    def denominator(self):
        """The measured data object of the benchmark, or a calculated one of this file."""
        return self._entry[self.denominator_label] if self._entry is not None else None

    @property
    def against(self) -> Optional[str]:
        """
        ``"measured"`` for a C/E, ``"calculated"`` for a ratio between two calculations.

        Read from the denominator's ``@nature``; ``None`` when the denominator
        cannot be resolved.
        """
        if self._entry is None or not self.denominator_label:
            return None
        try:
            nature = self.denominator.nature
        except LabelNotFoundError:
            return None
        return "calculated" if nature == "calculated" else "measured"

    def to_dataframe(self):
        """
        The published comparison.

        Returns
        -------
        pandas.DataFrame
            The comparison's table as the file gives it; see
            :meth:`Table.to_dataframe`.

        Raises
        ------
        SinbadError
            If the comparison holds no table.
        """
        if self.table is None:
            raise SinbadError(f"comparison {self.label!r} holds no table")
        return self.table.to_dataframe()

    def recompute(self):
        """
        Form the comparison again from the two data objects it names.

        Rows are matched on ``position``, and the denominator is taken in the
        comparison's convention, applying whatever corrections it declares --
        which is exactly what §4.9 says the published numbers mean.

        Returns
        -------
        pandas.DataFrame
            One row per position and value column of the calculated table, for
            the positions that the denominator also has and where neither value
            is NaN. Columns ``position``, ``column`` (the calculated column),
            ``recomputed`` (by ``operator``) and, when the published table
            has a matching row and column, ``published`` and ``ratio``
            (recomputed over published; NaN when the published value is 0).

        Raises
        ------
        SinbadError
            If the comparison was not opened from a benchmark, or the
            denominator cannot be brought to the convention (see
            :meth:`~kika.sinbad.DataObject.corrected`).
        """
        import pandas as pd  # noqa: PLC0415

        if self._entry is None:
            raise SinbadError("this comparison was not opened from a benchmark")
        numerator, denominator = self.numerator, self.denominator
        measured = denominator.corrected(self.convention)
        measured_positions = [str(p) for p in denominator.table.positions]
        calculated_table = numerator.table
        calculated_positions = [str(p) for p in calculated_table.positions]
        published = self.table

        rows = []
        for column in calculated_table.value_columns:
            calculated = calculated_table[column.name]
            for i, position in enumerate(calculated_positions):
                if position not in measured_positions:
                    continue
                j = measured_positions.index(position)
                if np.isnan(measured[j]) or np.isnan(calculated[i]):
                    continue
                if self.operator == "ratio":
                    value = calculated[i] / measured[j]
                elif self.operator == "difference":
                    value = calculated[i] - measured[j]
                else:  # relative
                    value = (calculated[i] - measured[j]) / measured[j]
                row = {
                    "position": position,
                    "column": column.name,
                    "recomputed": value,
                }
                if published is not None:
                    published_positions = published.positions
                    match = (
                        [] if published_positions is None
                        else [k for k, p in enumerate(published_positions) if str(p) == position]
                    )
                    name = _matching_column(published, column.name)
                    if match and name:
                        row["published"] = published[name][match[0]]
                        row["ratio"] = value / row["published"] if row["published"] else np.nan
                rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"<Comparison {self.label!r} {self.operator} "
            f"{self.numerator_label}/{self.denominator_label}>"
        )


def _matching_column(published: Table, calculated_name: str) -> Optional[str]:
    """Pair a calculated column with the published one from the same library.

    The names differ by prefix -- ``reactionRate-BUGJEFF311`` against
    ``CE-BUGJEFF311`` -- and agree on the suffix after the first dash, which is
    the library. Falls back to the sole value column when there is only one.
    """
    suffix = calculated_name.split("-", 1)[-1]
    for column in published.value_columns:
        if column.name.split("-", 1)[-1] == suffix:
            return column.name
    values = published.value_columns
    return values[0].name if len(values) == 1 else None
