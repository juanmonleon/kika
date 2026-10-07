"""The data object: one set of numbers, and everything needed to use it.

§3 of ``Data_structures.md``. A ``dataObject`` is the format's universal
container -- the measured reaction rates of one foil, the source distribution,
the assembly geometry, a calculated table, a group structure are all one. What
distinguishes them is the metadata on the object and the container inside it,
so this class is thin by design: the attributes of §3, the
:class:`~kika.sinbad.content.Content` it holds, its corrections and its
uncertainty budget.

**References are resolved, not just reported.** ``normalisation="nestor30kW"``
is a label; :attr:`DataObject.normalisation` hands back the
:class:`~kika.sinbad.entry.Normalisation` it names, from whichever file defines
it. A calculations file that normalises to the benchmark's ``nestor30kW``
resolves across the file boundary, which is what §1 means by *"a calculations
file may use the labels of its benchmark"*.

**The convention is the part that is easy to get wrong.** The pilot's sulphur
table is stored ``asMeasured``, with a 2 % background correction declared and
*not* applied; the gold table is stored ``backgroundSubtracted``, with the
correction applied and its size not reported. Comparing the two without asking
for a convention compares two different things. :meth:`DataObject.corrected`
is the one call that makes them comparable, and it applies only what the object
itself declares.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from kika.sinbad._xml import LabelIndex
from kika.sinbad.content import Content, Table, read_content
from kika.sinbad.exceptions import (
    AmbiguousLabelError,
    ContentTypeError,
    LabelNotFoundError,
    SinbadError,
)
from kika.sinbad.uncertainty import Correction, UncertaintyBudget, covariance

__all__ = ["DataObject", "DataObjectCollection"]


class DataObject:
    """
    One set of numbers from a SINBAD entry, with its metadata.

    Built by the reader; not constructed directly.

    Attributes
    ----------
    label : str
        Its identifier -- what everything else points at.
    quantity : str
        What it is (§5): ``"reaction rate"``, ``"assembly geometry"``, ...
    nature : str
        ``measured``, ``evaluated``, ``calculated`` or ``derived``.
    source : str
        Where in the entry it was transcribed from.
    reaction : str or None
        GNDS reaction id, e.g. ``"S32(n,p)P32"``.
    content : Content
        The container: a :class:`~kika.sinbad.content.Table`, a
        :class:`~kika.sinbad.content.Gridded`, a
        :class:`~kika.sinbad.content.Geometry`, ...
    corrections : list of Correction
    uncertainty_budget : UncertaintyBudget or None
    note : str or None

    Examples
    --------
    >>> d = b["reactionRate-S32"]                        # doctest: +SKIP
    >>> d.quantity, d.nature, d.convention.label         # doctest: +SKIP
    ('reaction rate', 'measured', 'asMeasured')
    >>> d.to_dataframe().head(2)                         # doctest: +SKIP
    >>> d.corrected("backgroundSubtracted")[:2]          # doctest: +SKIP
    """

    def __init__(self, element: ET.Element, index: LabelIndex, entry: Any = None):
        self._element = element
        self._index = index
        self._entry = entry

        self.label: str = element.get("label", "")
        self.quantity: str = element.get("quantity", "")
        self.nature: str = element.get("nature", "")
        self.source: str = element.get("source", "")
        self.reaction: Optional[str] = element.get("reaction")
        self.unit_provenance: Optional[str] = element.get("unitProvenance")
        self.library: Optional[str] = element.get("library")
        self.weighting: Optional[str] = element.get("weighting")
        self.weighting_library: Optional[str] = element.get("weightingLibrary")
        self.variant: Optional[str] = element.get("variant")

        #: Raw label references, as the file writes them.
        self.normalisation_label: Optional[str] = element.get("normalisation")
        self.convention_label: Optional[str] = element.get("convention")
        self.frame_label: Optional[str] = element.get("frame")
        self.detector_labels: List[str] = (element.get("measuredBy") or "").split()
        #: v0.4: the one position a whole object is at (a spectrum at A2), and
        #: the calculation that produced it.
        self.position_label: Optional[str] = element.get("position")
        self.calculated_by_label: Optional[str] = element.get("calculatedBy")

        self.corrections: List[Correction] = [
            Correction._read(c) for c in element.findall("correction")
        ]
        self.uncertainty_budget: Optional[UncertaintyBudget] = UncertaintyBudget._read(
            element.find("uncertaintyBudget")
        )
        note = element.find("note")
        self.note: Optional[str] = (
            " ".join(note.text.split()) if note is not None and note.text else None
        )

        self._content: Optional[Content] = None

    # -- content ----------------------------------------------------------

    @property
    def content(self) -> Content:
        """The container holding the numbers. Read on first use."""
        if self._content is None:
            for child in self._element:
                if child.tag in ("correction", "uncertaintyBudget", "note"):
                    continue
                self._content = read_content(child, self._index)
                break
            else:
                raise SinbadError(f"data object {self.label!r} holds no content")
        return self._content

    @property
    def kind(self) -> str:
        """The tag of the container -- ``table``, ``gridded2d``, ``geometry``, ..."""
        return self.content.tag

    @property
    def table(self) -> Table:
        """
        The table, for an object that holds one.

        Raises
        ------
        ContentTypeError
            If the object holds something else. The message says what.
        """
        content = self.content
        if not isinstance(content, Table):
            raise ContentTypeError(
                f"{self.label!r} holds a <{content.tag}>, not a table"
            )
        return content

    # -- resolved references ----------------------------------------------

    @property
    def normalisation(self):
        """The :class:`~kika.sinbad.entry.Normalisation` this object is per, if any."""
        return self._resolve(self.normalisation_label, "Normalisation")

    @property
    def convention(self):
        """The :class:`~kika.sinbad.entry.ValueConvention` the values are in, if any."""
        return self._resolve(self.convention_label, "ValueConvention")

    @property
    def frame(self):
        """The :class:`~kika.sinbad.entry.CoordinateFrame` coordinates are in, if any."""
        return self._resolve(self.frame_label, "CoordinateFrame")

    @property
    def position(self):
        """
        The :class:`~kika.sinbad.entry.Position` the whole object is at, if it says one.

        Not to be confused with :attr:`positions`, the position column of a
        table: this is for an object that is itself at one place -- a
        calculated spectrum at A2 (v0.4, ``@position``).
        """
        return self._resolve(self.position_label, "Position")

    @property
    def calculated_by(self):
        """The :class:`~kika.sinbad.entry.Calculation` that produced it, if it says one (v0.4)."""
        return self._resolve(self.calculated_by_label, "Calculation")

    @property
    def detectors(self) -> List[Any]:
        """The detectors that measured it, resolved."""
        return [d for d in (self._resolve(label, "Detector") for label in self.detector_labels) if d]

    def _resolve(self, label: Optional[str], _what: str):
        if not label or self._entry is None:
            return None
        try:
            return self._entry.resolve(label)
        except LabelNotFoundError:
            return None

    # -- the numbers ------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """The single value column of the table, or the array of a gridded object."""
        content = self.content
        if isinstance(content, Table):
            return content.values
        values = getattr(content, "values", None)
        if values is None:
            raise ContentTypeError(f"{self.label!r} (<{content.tag}>) holds no value array")
        return values

    @property
    def positions(self) -> Optional[np.ndarray]:
        """The ``position`` column, for a table indexed by detector position."""
        content = self.content
        return content.positions if isinstance(content, Table) else None

    @property
    def uncertainty(self) -> Optional[np.ndarray]:
        """
        The total relative uncertainty at 1 s.d., as a fraction.

        Taken from the table's ``kind="total"`` column when there is one, and
        otherwise combined from the budget. The two agree in the pilot; where
        they do not, the published column is what the entry stands behind.
        A budget with a component given only at some positions (v0.4) gives
        NaN at the others: the total is not known there.
        """
        try:
            table = self.table
        except ContentTypeError:
            return None
        published = table.total_uncertainty
        if published is not None:
            column = next(c for c in table.uncertainty_columns if c.kind == "total")
            scale = 100.0 if column.unit == "%" else 1.0
            return np.asarray(published, dtype=float) / scale
        if self.uncertainty_budget is not None:
            return self.uncertainty_budget.total(table)
        return None

    def corrected(self, to: Optional[str] = None) -> np.ndarray:
        """
        The values, brought to another convention.

        Applies every correction the object declares as *not applied* whose
        ``@to`` is the convention asked for. A correction the entry declares
        without a magnitude cannot be applied and raises, with its stated
        reason -- silently returning the uncorrected numbers would be worse.

        Parameters
        ----------
        to : str, optional
            Label of the target ``valueConvention``. With ``None``, the values
            as stored.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> d.corrected("backgroundSubtracted")[:2]      # doctest: +SKIP
        array([1.98e-17, 4.20e-18])
        """
        values = np.array(self.values, dtype=float, copy=True)
        if to is None or to == self.convention_label:
            return values
        applied_any = False
        for correction in self._corrections_to(to):
            if not correction.is_quantified:
                raise SinbadError(
                    f"{self.label!r}: the {correction.kind!r} correction to {to!r} "
                    f"has no magnitude ({correction.status or 'not given'}), so the "
                    f"values cannot be taken there"
                )
            values = correction.apply(values)
            applied_any = True
        if not applied_any and self.convention_label not in (None, to):
            raise SinbadError(
                f"{self.label!r} is in convention {self.convention_label!r} and "
                f"declares no correction to {to!r}"
            )
        return values

    def _corrections_to(self, to: str) -> List[Correction]:
        """The corrections this object declares but has not applied, that reach ``to``."""
        return [
            c for c in self.corrections
            if not c.applied and c.to_convention == to
        ]

    def covariance(self, relative: bool = False) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        """
        The covariance of this object's points, from its uncertainty budget.

        See :func:`kika.sinbad.covariance` for the rule.
        """
        return covariance([self], relative=relative)

    # -- views ------------------------------------------------------------

    def to_dataframe(self):
        """
        The content as a :class:`pandas.DataFrame`.

        What the frame holds depends on the container: a table gives its
        columns, a gridded array one row per cell, a geometry one row per
        layer, materials one row per nuclide.
        """
        return self.content.to_dataframe()

    def summary(self) -> str:
        """One paragraph: what this object is, and what is attached to it."""
        lines = [f"{self.label}  --  {self.quantity} ({self.nature})"]
        if self.reaction:
            lines.append(f"  reaction        {self.reaction}")
        if self.detector_labels:
            lines.append(f"  measured by     {' '.join(self.detector_labels)}")
        if self.position_label:
            lines.append(f"  at position     {self.position_label}")
        if self.calculated_by_label:
            lines.append(f"  calculated by   {self.calculated_by_label}")
        if self.normalisation_label:
            lines.append(f"  normalisation   {self.normalisation_label}")
        if self.convention_label:
            lines.append(f"  convention      {self.convention_label}")
        lines.append(f"  content         {self.content!r}")
        for correction in self.corrections:
            lines.append(f"  correction      {correction!r}")
        if self.uncertainty_budget is not None:
            for component in self.uncertainty_budget:
                lines.append(f"  uncertainty     {component!r}")
        if self.source:
            lines.append(f"  source          {self.source}")
        if self.note:
            lines.append(f"  note            {self.note}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<DataObject {self.label!r} {self.quantity} ({self.nature}), "
            f"<{self.content.tag}>>"
        )


class DataObjectCollection:
    """
    The data objects of a file, addressable by label, by position and by filter.

    Calling the collection filters it and returns another collection, so
    filters chain::

        b.data(quantity="reaction rate")(nature="measured")

    Examples
    --------
    >>> b.data                                            # doctest: +SKIP
    <DataObjectCollection 9 objects>
    >>> b.data["reactionRate-S32"]                        # doctest: +SKIP
    >>> [d.label for d in b.data(quantity="reaction rate")]   # doctest: +SKIP
    ['reactionRate-S32', 'reactionRate-In115', ...]
    """

    def __init__(self, objects: Sequence[DataObject]):
        self._objects: List[DataObject] = list(objects)

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> Iterator[DataObject]:
        return iter(self._objects)

    def __contains__(self, label: object) -> bool:
        return any(o.label == label for o in self._objects)

    def __getitem__(self, key: Union[str, int, slice]) -> Union[DataObject, "DataObjectCollection"]:
        if isinstance(key, int):
            return self._objects[key]
        if isinstance(key, slice):
            return DataObjectCollection(self._objects[key])
        for obj in self._objects:
            if obj.label == key:
                return obj
        matches = [o for o in self._objects if key in o.label]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AmbiguousLabelError(
                f"{key!r} matches {len(matches)} objects "
                f"({', '.join(m.label for m in matches[:5])}); use the full label"
            )
        raise LabelNotFoundError(f"no data object {key!r}")

    def __call__(
        self,
        quantity: Optional[str] = None,
        nature: Optional[str] = None,
        reaction: Optional[str] = None,
        detector: Optional[str] = None,
        kind: Optional[str] = None,
        library: Optional[str] = None,
        position: Optional[str] = None,
        calculated_by: Optional[str] = None,
    ) -> "DataObjectCollection":
        """
        Filter the collection. Every argument given must match.

        Parameters
        ----------
        quantity, nature, reaction, library : str, optional
            Matched exactly against the object's attribute.
        detector : str, optional
            Matched against the labels in ``measuredBy``; a fragment is enough,
            so ``detector="S32"`` finds both sulphur pellets.
        kind : str, optional
            The container tag: ``"table"``, ``"gridded1d"``, ...
        position, calculated_by : str, optional
            Matched exactly against ``@position`` and ``@calculatedBy`` (v0.4):
            ``position="A2"`` finds the spectra calculated at A2.
        """
        selected = self._objects
        if quantity is not None:
            selected = [o for o in selected if o.quantity == quantity]
        if nature is not None:
            selected = [o for o in selected if o.nature == nature]
        if reaction is not None:
            selected = [o for o in selected if o.reaction == reaction]
        if library is not None:
            selected = [o for o in selected if o.library == library]
        if detector is not None:
            selected = [
                o for o in selected
                if any(detector in label for label in o.detector_labels)
            ]
        if kind is not None:
            selected = [o for o in selected if o.kind == kind]
        if position is not None:
            selected = [o for o in selected if o.position_label == position]
        if calculated_by is not None:
            selected = [o for o in selected if o.calculated_by_label == calculated_by]
        return DataObjectCollection(selected)

    @property
    def labels(self) -> List[str]:
        return [o.label for o in self._objects]

    @property
    def quantities(self) -> List[str]:
        """The distinct quantities present, in order of first appearance."""
        seen: List[str] = []
        for obj in self._objects:
            if obj.quantity not in seen:
                seen.append(obj.quantity)
        return seen

    def to_dataframe(self, convention: Optional[str] = None):
        """
        Every table in the collection, stacked, with an ``object`` column.

        Objects that hold no table are skipped. With ``convention``, values are
        brought there first, and a ``convention`` column records it.

        Parameters
        ----------
        convention : str, optional
            Label of the ``valueConvention`` to bring the values to.

        Raises
        ------
        SinbadError
            If an object would have to be corrected to reach ``convention``
            but holds more than one value column, so that
            :meth:`DataObject.corrected` cannot say which. Stamping the frame
            with a convention the numbers are not in would be worse than
            stopping.
        """
        import pandas as pd  # noqa: PLC0415

        frames = []
        for obj in self._objects:
            try:
                table = obj.table
            except ContentTypeError:
                continue
            frame = table.to_dataframe()
            if convention is not None:
                column = table.value_columns
                if len(column) == 1:
                    frame[column[0].name] = obj.corrected(convention)
                elif obj.convention_label not in (None, convention):
                    raise SinbadError(
                        f"{obj.label!r} is stored in convention "
                        f"{obj.convention_label!r} and has {len(column)} value columns "
                        f"({', '.join(c.name for c in column)}), so it cannot be brought "
                        f"to {convention!r} as one table -- take the columns one at a "
                        f"time. Labelling them {convention!r} untouched would be a lie."
                    )
                frame["convention"] = convention
            frame.insert(0, "object", obj.label)
            if obj.reaction:
                frame.insert(1, "reaction", obj.reaction)
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def covariance(self, relative: bool = False):
        """The joint covariance of every object in the collection."""
        return covariance(list(self._objects), relative=relative)

    def __repr__(self) -> str:
        return f"<DataObjectCollection {len(self._objects)} objects>"
