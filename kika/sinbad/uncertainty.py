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
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kika.sinbad._constants import CORRELATION_SCOPE_IS_GLOBAL
from kika.sinbad._xml import as_float
from kika.sinbad.content import Double, Table
from kika.sinbad.exceptions import ContentTypeError, SinbadError

__all__ = [
    "Correction",
    "UncertaintyComponent",
    "UncertaintyBudget",
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


@dataclass
class UncertaintyComponent:
    """
    One contribution to the uncertainty of a set of numbers.

    It is given either per point, as a column of the object's table, or as one
    value for all points, as published.

    Attributes
    ----------
    name : str
    correlation_scope : str
        ``none``, ``within-detector``, ``within-entry``, ``within-campaign`` or
        ``within-configuration``.
    column : str or None
        The column holding it, when it is per point.
    value, unit : float, str
        The single value, when it is not.
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
        )

    @property
    def is_per_point(self) -> bool:
        """Whether it varies point by point."""
        return self.column is not None

    @property
    def coverage_factor(self) -> float:
        """``k`` in ``k s.d.``; 1 when the level is absent or not stated that way."""
        return _coverage_factor(self.confidence_level)

    def relative(self, table: Optional[Table] = None) -> np.ndarray:
        """
        This component as a relative standard deviation at 1 s.d.

        Parameters
        ----------
        table : Table, optional
            The object's table -- needed for a per-point component.

        Returns
        -------
        numpy.ndarray
            One value per row of the table, or a single-element array.
        """
        if self.is_per_point:
            if table is None:
                raise ContentTypeError(
                    f"component {self.name!r} is per point (column {self.column!r}) "
                    f"and no table was given"
                )
            column = table.column(self.column)
            raw = np.asarray(table[self.column], dtype=float)
            scale = 100.0 if column.unit == "%" else 1.0
            # The column states the level for a per-point component (§2.3); the
            # component's own attribute is the fallback for a file that puts it
            # there instead, so that a 2 s.d. column is never read as 1 s.d.
            level = column.confidence_level or self.confidence_level
            return raw / scale / _coverage_factor(level)
        scale = 100.0 if self.unit == "%" else 1.0
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

    def __repr__(self) -> str:
        if self.is_per_point:
            size = f"column {self.column!r}"
        else:
            size = f"{self.value:g}{self.unit}"
            if self.confidence_level:
                size += f" @ {self.confidence_level}"
        return f"<{self.name} {size}, {self.correlation_scope}>"


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

    def total(self, table: Optional[Table] = None) -> np.ndarray:
        """
        Combine the components, by the rule the entry declares.

        Returns
        -------
        numpy.ndarray
            Relative standard deviation at 1 s.d., one value per point when any
            component is per point.
        """
        parts = [c.relative(table) for c in self.components]
        if not parts:
            return np.array([])
        size = max(p.size for p in parts)
        stacked = np.vstack([np.broadcast_to(p, (size,)) for p in parts])
        if self.combination_rule == "linear":
            return stacked.sum(axis=0)
        return np.sqrt((stacked ** 2).sum(axis=0))

    def to_dataframe(self, table: Optional[Table] = None):
        """One row per component: its size at 1 s.d., its scope, its source."""
        import pandas as pd  # noqa: PLC0415

        rows = []
        for component in self.components:
            relative = component.relative(table) if (table is not None or not component.is_per_point) else None
            rows.append({
                "component": component.name,
                "per_point": component.is_per_point,
                "relative_1sd": (
                    float(np.mean(relative)) if relative is not None and relative.size else np.nan
                ),
                "as_published": (
                    f"column {component.column}" if component.is_per_point
                    else f"{component.value:g}{component.unit}"
                ),
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
        values = table.values
        positions = table.positions
        default_ref = " ".join(obj.detector_labels) if obj.detector_labels else obj.label
        per_component = {
            c.name: (c.relative(table), c.scope_key(default_ref)) for c in budget
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

    Notes
    -----
    Points whose value is blank in the table are left out: the entry has no
    measurement there, so it has no uncertainty either.

    Examples
    --------
    >>> matrix, index = b.covariance()                   # doctest: +SKIP
    >>> matrix.shape                                     # doctest: +SKIP
    (55, 55)
    """
    points = _points(objects)
    size = len(points)
    matrix = np.zeros((size, size))
    for i, left in enumerate(points):
        for j, right in enumerate(points):
            total = 0.0
            for name, (ri, key) in left.components.items():
                if name not in right.components:
                    continue
                rj, key_right = right.components[name]
                if i == j:
                    total += ri * rj
                elif key is None or key_right is None:
                    continue
                elif "*" in key or (key & key_right):
                    total += ri * rj
            matrix[i, j] = total
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
