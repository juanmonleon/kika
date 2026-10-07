"""Corrections and uncertainty budgets, and the covariance they imply.

§3 of ``Data_structures.md`` puts both inside the ``dataObject``: *"Everything
about one set of numbers is in its dataObject: the values, how they were
corrected, and their uncertainty budget."* This module reads those two blocks
and does the one calculation the format asks a reader to do.

**The format stores components, not matrices.** §2.8: a covariance matrix is
written only when the entry publishes one; otherwise each uncertainty component
carries a *correlation scope*, and the covariance between two points is the sum
of the components they share::

    cov(i, j) = sum over shared components c of  r_c(i) * r_c(j)

with ``none`` contributing on the diagonal only, ``within-detector`` between
points measured by the same detector, and ``within-entry`` (and the wider
scopes) between every pair in the entry. That rule is the whole of
:func:`covariance`, and it is why the pilot's 55 measured points have a full
correlation matrix without a single matrix being stored.

**Everything is brought to 1 s.d. and to a fraction.** A component published as
``8 %`` at ``2 s.d.`` enters as 0.04. ``confidenceLevel`` is read for the
coverage factor; a level that is not ``k s.d.`` is taken at face value and
reported by :attr:`UncertaintyComponent.coverage_factor` as 1, because
inventing a conversion from *"95 % confidence"* would be a calculation the
entry did not authorise.

**A correction is declared whether or not it was applied.** §3.1: with
``applied="true"`` the values already went from ``@from`` to the object's
convention; with ``applied="false"`` applying it would take them to ``@to``.
:meth:`Correction.factor` is what the arithmetic needs, and
:meth:`kika.sinbad.DataObject.corrected` is what a user calls.
"""

from __future__ import annotations

import math
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kika.sinbad._constants import CORRELATION_SCOPE_IS_GLOBAL
from kika.sinbad._xml import as_float
from kika.sinbad.content import Double, Table
from kika.sinbad.exceptions import ContentTypeError, IncompleteBudgetError, SinbadError

__all__ = [
    "Correction",
    "UncertaintyComponent",
    "UncertaintyBudget",
    "PublishedTotal",
    "covariance",
    "correlation",
]

#: ``"1 s.d."``, ``"2 s.d."`` -- the only confidence levels with a coverage
#: factor a reader may act on without inventing a distribution.
_SD = re.compile(r"^\s*([\d.]+)\s*s\.d\.\s*$")


def _coverage_factor(level: Optional[str]) -> float:
    if not level:
        return 1.0
    match = _SD.match(level)
    return float(match.group(1)) if match else 1.0


# --------------------------------------------------------------------------
# §3.1 correction
# --------------------------------------------------------------------------


@dataclass
class Correction:
    """
    One correction the entry declares for a set of numbers.

    Attributes
    ----------
    kind : str
        What it corrects for, e.g. ``"background"``.
    direction : str
        ``subtract``, ``add`` or ``multiply``.
    applied : bool
        Whether the stored values already carry it.
    from_convention, to_convention : str or None
        Labels of the ``valueConvention`` it goes between.
    magnitude : Double or None
        ``None`` when the entry declares the correction but not its size; then
        :attr:`status` says why.
    status : str or None
        For example ``"not reported"``.
    method, source : str
        How it was measured, and where the entry says so.
    """

    kind: str = ""
    direction: str = "subtract"
    applied: bool = False
    from_convention: Optional[str] = None
    to_convention: Optional[str] = None
    magnitude: Optional[Double] = None
    status: Optional[str] = None
    method: str = ""
    source: str = ""

    @classmethod
    def _read(cls, element: ET.Element) -> "Correction":
        scalar = element.find("double")
        magnitude_node = element.find("magnitude")
        return cls(
            kind=element.get("kind", ""),
            direction=element.get("direction", "subtract"),
            applied=element.get("applied") == "true",
            from_convention=element.get("from"),
            to_convention=element.get("to"),
            magnitude=Double._read(scalar) if scalar is not None else None,
            status=magnitude_node.get("status") if magnitude_node is not None else None,
            method=element.get("method", ""),
            source=element.get("source", ""),
        )

    @property
    def is_quantified(self) -> bool:
        """Whether the entry says how big the correction is."""
        return self.magnitude is not None and not math.isnan(self.magnitude.value)

    @property
    def relative(self) -> float:
        """The magnitude as a fraction. A ``%`` magnitude is divided by 100."""
        if not self.is_quantified:
            raise SinbadError(
                f"the {self.kind!r} correction has no magnitude "
                f"({self.status or 'not given'})"
            )
        value = self.magnitude.value
        return value / 100.0 if self.magnitude.unit == "%" else value

    def factor(self) -> float:
        """
        The multiplicative factor that applies this correction.

        ``subtract`` of 2 % gives 0.98, ``add`` gives 1.02, ``multiply`` gives
        the magnitude itself. Only defined for a relative magnitude -- an
        absolute one is not a factor, and :meth:`apply` handles it.
        """
        if self.direction == "multiply":
            return self.relative
        if self.magnitude is not None and self.magnitude.unit not in ("%", ""):
            raise SinbadError(
                f"the {self.kind!r} correction is absolute ({self.magnitude.unit}); "
                f"it has no factor -- use apply()"
            )
        return 1.0 - self.relative if self.direction == "subtract" else 1.0 + self.relative

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Return ``values`` with this correction applied."""
        if self.direction == "multiply":
            return values * self.relative
        if self.magnitude is not None and self.magnitude.unit not in ("%", ""):
            step = self.magnitude.value
            return values - step if self.direction == "subtract" else values + step
        return values * self.factor()

    def __repr__(self) -> str:
        size = (
            f"{self.magnitude.value:g}{self.magnitude.unit}"
            if self.is_quantified else (self.status or "size not given")
        )
        state = "applied" if self.applied else "declared, not applied"
        return f"<Correction {self.kind} {self.direction} {size} ({state})>"


# --------------------------------------------------------------------------
# §3.2 uncertainty budget
# --------------------------------------------------------------------------


def _row_positions(table: Table) -> List[str]:
    """The position of every row, as text; the row number when there is no position column."""
    positions = table.positions
    if positions is None:
        return [str(i) for i in range(table.nrows)]
    return [str(p) for p in positions]


def _valued_positions(table: Table) -> List[str]:
    """The positions whose value cell is filled -- the points the entry has."""
    positions = _row_positions(table)
    try:
        values = table.values
    except ContentTypeError:
        return positions  # several value columns: every row counts
    return [p for p, v in zip(positions, values) if not np.isnan(v)]


def _read_at(element: ET.Element) -> Dict[str, float]:
    return {a.get("position", ""): as_float(a.get("value")) for a in element.findall("at")}


@dataclass
class UncertaintyComponent:
    """
    One contribution to the uncertainty of a set of numbers.

    It is given in one of four ways (§3.2):

    * per point, as a column of the object's table (:attr:`column`);
    * as one value for all points, as published (:attr:`value`);
    * only at the positions where the entry gives it (:attr:`at`, v0.4) --
      AEA-RS-1231 Table 18 gives the uncertainty of the calculated rates at
      two positions per detector, and nothing in between;
    * not at all: the entry names it and does not quantify it
      (:attr:`status`, v0.4).

    Attributes
    ----------
    name : str
    correlation_scope : str
        ``none``, ``within-detector``, ``within-entry``, ``within-campaign`` or
        ``within-configuration``.
    column : str or None
        The column holding it, when it is per point.
    value, unit : float, str
        The single value, when it is one for all points.
    at : dict of str to float
        Position label to value, in :attr:`unit` at :attr:`confidence_level`,
        when it is given only at some positions.
    status : str or None
        Why there is no number, when there is none -- ``"not quantified"``.
    confidence_level : str or None
        As published, e.g. ``"2 s.d."``.
    ref : str or None
        What the component belongs to -- the detector, the source strength.
        This is what ``within-detector`` compares.
    source : str
    """

    name: str
    correlation_scope: str = "none"
    column: Optional[str] = None
    value: float = float("nan")
    unit: str = ""
    confidence_level: Optional[str] = None
    confidence_level_provenance: Optional[str] = None
    ref: Optional[str] = None
    source: str = ""
    at: Dict[str, float] = field(default_factory=dict)
    status: Optional[str] = None

    @classmethod
    def _read(cls, element: ET.Element) -> "UncertaintyComponent":
        return cls(
            name=element.get("name", ""),
            correlation_scope=element.get("correlationScope", "none"),
            column=element.get("column"),
            value=as_float(element.get("value")),
            unit=element.get("unit", ""),
            confidence_level=element.get("confidenceLevel"),
            confidence_level_provenance=element.get("confidenceLevelProvenance"),
            ref=element.get("ref"),
            source=element.get("source", ""),
            at=_read_at(element),
            status=element.get("status"),
        )

    @property
    def is_per_point(self) -> bool:
        """Whether it varies point by point, as a column of the table."""
        return self.column is not None

    @property
    def is_partial(self) -> bool:
        """Whether it is given only at some positions (``<at>``)."""
        return self.column is None and bool(self.at)

    @property
    def is_quantified(self) -> bool:
        """Whether the entry gives any number for it. ``False`` for a component named by :attr:`status` only."""
        return self.is_per_point or self.is_partial or not math.isnan(self.value)

    @property
    def coverage_factor(self) -> float:
        """``k`` in ``k s.d.``; 1 when the level is absent or not stated that way."""
        return _coverage_factor(self.confidence_level)

    def missing_at(self, table: Table) -> List[str]:
        """
        The points of ``table`` this component gives no number for.

        Empty for a per-point or single-value component; every point for an
        unquantified one.
        """
        if self.is_per_point or (self.is_quantified and not self.is_partial):
            return []
        return [p for p in _valued_positions(table) if p not in self.at]

    def relative(self, table: Optional[Table] = None) -> np.ndarray:
        """
        This component as a relative standard deviation at 1 s.d.

        Parameters
        ----------
        table : Table, optional
            The object's table -- needed for a per-point component and for one
            given only at some positions.

        Returns
        -------
        numpy.ndarray
            One value per row of the table, or a single-element array. A
            component given only at some positions is NaN at the others, and
            an unquantified one is ``[nan]``: a number the entry does not have
            is not replaced by one it did not publish.
        """
        if self.is_per_point or self.is_partial:
            if table is None:
                how = f"column {self.column!r}" if self.is_per_point else "given at some positions"
                raise ContentTypeError(
                    f"component {self.name!r} is per point ({how}) and no table was given"
                )
        if self.is_per_point:
            column = table.column(self.column)
            raw = np.asarray(table[self.column], dtype=float)
            scale = 100.0 if column.unit == "%" else 1.0
            # The column states the level for a per-point component (§2.3); the
            # component's own attribute is the fallback for a file that puts it
            # there instead, so that a 2 s.d. column is never read as 1 s.d.
            level = column.confidence_level or self.confidence_level
            return raw / scale / _coverage_factor(level)
        scale = 100.0 if self.unit == "%" else 1.0
        if self.is_partial:
            raw = np.array([self.at.get(p, np.nan) for p in _row_positions(table)], dtype=float)
            return raw / scale / self.coverage_factor
        return np.array([self.value / scale / self.coverage_factor])

    def scope_key(self, default_ref: str) -> Optional[frozenset]:
        """
        What two points must share for this component to correlate them.

        ``None`` means "never off the diagonal". A frozenset means "these
        labels must intersect" -- a set because a component may name more than
        one object, as the pilot's sulphur does with its two pellets.
        """
        if CORRELATION_SCOPE_IS_GLOBAL.get(self.correlation_scope, False):
            return frozenset({"*"})
        if self.correlation_scope == "none":
            return None
        return frozenset((self.ref or default_ref).split())

    def _as_published(self) -> str:
        if self.is_per_point:
            return f"column {self.column}"
        if self.is_partial:
            return f"at {', '.join(self.at)}"
        if not self.is_quantified:
            return self.status or "not given"
        return f"{self.value:g}{self.unit}"

    def __repr__(self) -> str:
        if self.is_per_point:
            size = f"column {self.column!r}"
        else:
            size = self._as_published()
            if self.is_quantified and self.confidence_level:
                size += f" @ {self.confidence_level}"
        return f"<{self.name} {size}, {self.correlation_scope}>"


@dataclass
class PublishedTotal:
    """
    The total uncertainty the entry prints at some positions (v0.4).

    ``uncertaintyBudget/total``: for a table with no total column, the total
    the entry gives where it gives one -- AEA-RS-1231 Table 18 at two
    positions per detector. It is what the recombined components are checked
    against, not an input.

    Attributes
    ----------
    at : dict of str to float
        Position label to value, in :attr:`unit` at :attr:`confidence_level`.
    unit : str
    confidence_level : str or None
    source : str
    """

    at: Dict[str, float] = field(default_factory=dict)
    unit: str = ""
    confidence_level: Optional[str] = None
    source: str = ""

    @classmethod
    def _read(cls, element: Optional[ET.Element]) -> Optional["PublishedTotal"]:
        if element is None:
            return None
        return cls(
            at=_read_at(element),
            unit=element.get("unit", ""),
            confidence_level=element.get("confidenceLevel"),
            source=element.get("source", ""),
        )

    def relative(self, position: str) -> float:
        """The total at ``position``, as a relative standard deviation at 1 s.d.; NaN where it is not given."""
        scale = 100.0 if self.unit == "%" else 1.0
        return self.at.get(position, np.nan) / scale / _coverage_factor(self.confidence_level)

    def __repr__(self) -> str:
        level = f" @ {self.confidence_level}" if self.confidence_level else ""
        return f"<PublishedTotal at {', '.join(self.at)}{level}>"


@dataclass
class UncertaintyBudget:
    """
    The components of a set of numbers, and how they combine.

    Attributes
    ----------
    components : list of UncertaintyComponent
    basis : str
        ``components`` (the default), ``observed-dispersion`` or
        ``quoted-without-basis``.
    combination_rule : str
        ``quadratic``, ``linear`` or ``not stated``.
    covariance : str
        ``from components``, ``reported`` or ``not applicable``.
    published_total : PublishedTotal or None
        The total the entry prints at some positions, when the table has no
        total column (v0.4).

    Examples
    --------
    >>> budget = b["reactionRate-S32"].uncertainty_budget    # doctest: +SKIP
    >>> budget.to_dataframe()                                # doctest: +SKIP
    >>> budget.total()[:2]                                   # doctest: +SKIP
    array([0.065, 0.065])
    """

    components: List[UncertaintyComponent] = field(default_factory=list)
    basis: str = "components"
    combination_rule: str = "quadratic"
    combination_rule_provenance: Optional[str] = None
    covariance: str = "from components"
    published_total: Optional[PublishedTotal] = None

    @classmethod
    def _read(cls, element: Optional[ET.Element]) -> Optional["UncertaintyBudget"]:
        if element is None:
            return None
        return cls(
            components=[
                UncertaintyComponent._read(c) for c in element.findall("component")
            ],
            basis=element.get("basis", "components"),
            combination_rule=element.get("combinationRule", "quadratic"),
            combination_rule_provenance=element.get("combinationRuleProvenance"),
            covariance=element.get("covariance", "from components"),
            published_total=PublishedTotal._read(element.find("total")),
        )

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __getitem__(self, name: str) -> UncertaintyComponent:
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(f"no component {name!r}; have {[c.name for c in self.components]}")

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.components]

    @property
    def unquantified(self) -> List[UncertaintyComponent]:
        """The components the entry names without giving a number."""
        return [c for c in self.components if not c.is_quantified]

    @property
    def partial(self) -> List[UncertaintyComponent]:
        """The components given only at some positions."""
        return [c for c in self.components if c.is_partial]

    def missing(self, table: Table) -> Dict[str, List[str]]:
        """
        Which quantified components are missing at which points of ``table``.

        Empty when every quantified component reaches every point, which is
        what :func:`covariance` needs. Unquantified components are not listed
        here -- see :attr:`unquantified`.
        """
        return {
            c.name: gaps for c in self.components
            if c.is_quantified and (gaps := c.missing_at(table))
        }

    def complete_at(self, table: Table) -> List[str]:
        """The points of ``table`` where every quantified component is given."""
        gaps = {p for positions in self.missing(table).values() for p in positions}
        return [p for p in _valued_positions(table) if p not in gaps]

    def total(self, table: Optional[Table] = None) -> np.ndarray:
        """
        Combine the quantified components, by the rule the entry declares.

        Returns
        -------
        numpy.ndarray
            Relative standard deviation at 1 s.d., one value per point when any
            component is per point or given at some positions. NaN at a point
            where a component is missing: the total is not known there. A
            component the entry names but does not quantify is left out, as
            the entry's own totals leave it out.
        """
        parts = [c.relative(table) for c in self.components if c.is_quantified]
        if not parts:
            return np.array([])
        size = max(p.size for p in parts)
        stacked = np.vstack([np.broadcast_to(p, (size,)) for p in parts])
        if self.combination_rule == "linear":
            return stacked.sum(axis=0)
        return np.sqrt((stacked ** 2).sum(axis=0))

    def by_position(self, table: Table):
        """
        The budget at the positions where some component is given only there.

        This is how a partial budget (v0.4, ``component/at``) is read: one row
        per such position, one column per quantified component at 1 s.d. as a
        fraction, the recombined ``total`` and the ``published_total`` the
        entry prints there, when it prints one.

        Returns
        -------
        pandas.DataFrame
            Indexed by position. Empty when no component is partial.
        """
        import pandas as pd  # noqa: PLC0415

        rows = _row_positions(table)
        wanted = [p for p in rows if any(p in c.at for c in self.partial)]
        quantified = [c for c in self.components if c.is_quantified]
        sizes = {c.name: np.broadcast_to(c.relative(table), (len(rows),)) for c in quantified}
        total = np.broadcast_to(self.total(table), (len(rows),))
        frame = pd.DataFrame(
            {name: [values[rows.index(p)] for p in wanted] for name, values in sizes.items()},
            index=pd.Index(wanted, name="position"),
        )
        frame["total"] = [total[rows.index(p)] for p in wanted]
        if self.published_total is not None:
            frame["published_total"] = [self.published_total.relative(p) for p in wanted]
        return frame

    def to_dataframe(self, table: Optional[Table] = None):
        """
        One row per component: its size at 1 s.d., its scope, its source.

        ``relative_1sd`` is the mean over the points where the component is
        given; ``as_published`` says how the entry gives it -- a column, one
        value, some positions, or why there is no number.
        """
        import pandas as pd  # noqa: PLC0415

        rows = []
        for component in self.components:
            needs_table = component.is_per_point or component.is_partial
            relative = component.relative(table) if (table is not None or not needs_table) else None
            finite = relative[~np.isnan(relative)] if relative is not None else np.array([])
            rows.append({
                "component": component.name,
                "per_point": component.is_per_point,
                "relative_1sd": float(np.mean(finite)) if finite.size else np.nan,
                "as_published": component._as_published(),
                "confidence_level": component.confidence_level,
                "correlation_scope": component.correlation_scope,
                "ref": component.ref,
                "source": component.source,
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"<UncertaintyBudget {len(self.components)} components "
            f"({', '.join(self.names)}), {self.combination_rule}>"
        )


# --------------------------------------------------------------------------
# the covariance the components imply
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Point:
    """One row of one data object, with the components that reach it."""

    label: str
    position: str
    value: float
    components: Dict[str, Tuple[float, Optional[frozenset]]]


def _require_complete(label: str, budget: UncertaintyBudget, table: Table) -> None:
    """
    Stop, and say why, when a budget does not reach every point of its table.

    A covariance over the whole table needs every component at every point.
    A component given only at some positions (v0.4) leaves the others
    unknown, and interpolating it would put numbers in the entry that the
    entry never published -- so this raises, with what is missing where, what
    the budget does cover, and what the caller can do instead.
    """
    missing = budget.missing(table)
    if not missing:
        return
    points = _valued_positions(table)
    complete = budget.complete_at(table)
    given: Dict[Tuple[str, ...], List[UncertaintyComponent]] = {}
    for component in budget.components:
        if component.name in missing:
            given.setdefault(tuple(p for p in points if p not in missing[component.name]), []).append(component)
    lines = [
        f"{label!r}: its uncertainty budget does not reach every point of the table, "
        f"so no covariance can be built from it without inventing numbers."
    ]
    for positions, components in given.items():
        names = ", ".join(c.name for c in components)
        sources = sorted({c.source for c in components if c.source})
        where = ", ".join(positions) if positions else "no point of the table"
        lines.append(
            f"  - {names}: given only at {where}"
            + (f" ({'; '.join(sources)})" if sources else "")
            + f", of the {len(points)} points ({points[0]} ... {points[-1]})."
        )
    for component in budget.unquantified:
        lines.append(f"  - {component.name}: named by the entry, {component.status or 'with no number'}.")
    lines.append(
        f"  The budget is complete at: {', '.join(complete) if complete else 'no point'}."
    )
    lines.append(
        "  Leave this object out of the covariance. To see the budget where it is given: "
        "obj.uncertainty_budget.by_position(obj.table)."
    )
    raise IncompleteBudgetError(
        "\n".join(lines), label=label, missing=missing, complete_at=complete
    )


def _points(objects: Sequence) -> List[_Point]:
    points: List[_Point] = []
    for obj in objects:
        table = obj.table
        budget = obj.uncertainty_budget
        if budget is None:
            raise SinbadError(
                f"{obj.label!r} declares no uncertainty budget; "
                f"there is nothing to build a covariance from"
            )
        _require_complete(obj.label, budget, table)
        if budget.unquantified:
            warnings.warn(
                f"{obj.label!r}: the covariance leaves out "
                + ", ".join(f"{c.name} ({c.status or 'not quantified'})" for c in budget.unquantified)
                + " -- named by the entry, with no number to put in",
                stacklevel=3,
            )
        values = table.values
        positions = table.positions
        default_ref = " ".join(obj.detector_labels) if obj.detector_labels else obj.label
        per_component = {
            c.name: (c.relative(table), c.scope_key(default_ref))
            for c in budget if c.is_quantified
        }
        for row in range(table.nrows):
            if np.isnan(values[row]):
                continue  # a blank cell is a point the entry does not have
            components = {
                name: (float(sizes[row] if sizes.size > 1 else sizes[0]), key)
                for name, (sizes, key) in per_component.items()
            }
            points.append(
                _Point(
                    label=obj.label,
                    position=str(positions[row]) if positions is not None else str(row),
                    value=float(values[row]),
                    components=components,
                )
            )
    return points


def covariance(objects: Sequence, relative: bool = False) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Build the covariance of a set of data objects from their uncertainty budgets.

    This is the calculation §2.8 leaves to the reader: the entry stores
    components with a correlation scope, and the covariance between two points
    is the sum of the components they share.

    Parameters
    ----------
    objects : sequence of DataObject
        Each must hold a table and declare an uncertainty budget.
    relative : bool, default False
        Return the relative covariance instead of the absolute one.

    Returns
    -------
    matrix : numpy.ndarray
        Square, one row per point, points in the order the objects were given.
    index : list of tuple
        ``(object label, position)`` for each row -- what the matrix is of.

    Raises
    ------
    IncompleteBudgetError
        If a budget gives some component only at some positions (v0.4,
        ``component/at``), so the table's other points have no value for it.
        The message lists what is missing where and where the budget is
        complete; nothing is interpolated.

    Notes
    -----
    Points whose value is blank in the table are left out: the entry has no
    measurement there, so it has no uncertainty either. A component the entry
    names but does not quantify (``@status``) is left out too, with a warning.

    Examples
    --------
    >>> matrix, index = b.covariance()                   # doctest: +SKIP
    >>> matrix.shape                                     # doctest: +SKIP
    (55, 55)
    """
    points = _points(objects)
    size = len(points)
    matrix = np.zeros((size, size))

    # One outer product per component, not one sum per cell: the rule is
    # cov = sum_c r_c r_c^T masked by what c correlates, and N**2 Python
    # iterations is a cliff an entry with a few thousand points would fall off.
    names: List[str] = []
    for point in points:
        for name in point.components:
            if name not in names:
                names.append(name)

    for name in names:
        sizes = np.zeros(size)
        keys: List[Optional[frozenset]] = [None] * size
        present = np.zeros(size, dtype=bool)
        for i, point in enumerate(points):
            entry = point.components.get(name)
            if entry is None:
                continue
            sizes[i], keys[i] = entry
            present[i] = True
        if not present.any():
            continue

        shared = np.outer(present, present)
        # ``none`` contributes on the diagonal only, so its key is None and it
        # never enters the membership matrix; every other scope correlates two
        # points whose refs intersect, and a scope that reaches the whole entry
        # says so with the single ref "*", which every point then shares. Set
        # intersection is a boolean matrix product, which is what keeps this
        # off a second N**2 loop in Python -- and it is symmetric by
        # construction, so a file that gives one component two different scopes
        # cannot produce a covariance that disagrees with its own transpose.
        refs: Dict[str, int] = {}
        for key in keys:
            for ref in key or ():
                refs.setdefault(ref, len(refs))
        membership = np.zeros((size, len(refs)), dtype=bool)
        for i, key in enumerate(keys):
            if present[i] and key:
                membership[i, [refs[ref] for ref in key]] = True
        reaches = membership.astype(np.uint8) @ membership.astype(np.uint8).T > 0
        np.fill_diagonal(reaches, True)

        matrix += np.outer(sizes, sizes) * (shared & reaches)

    if not relative:
        scale = np.array([p.value for p in points])
        matrix = matrix * np.outer(scale, scale)
    return matrix, [(p.label, p.position) for p in points]


def correlation(objects: Sequence) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    The correlation matrix of a set of data objects.

    Same as :func:`covariance`, normalised by the diagonal.
    """
    matrix, index = covariance(objects, relative=True)
    sigma = np.sqrt(np.diag(matrix))
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = matrix / np.outer(sigma, sigma)
    return np.nan_to_num(normalised), index
