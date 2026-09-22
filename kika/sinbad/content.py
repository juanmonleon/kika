"""What a ``dataObject`` holds: one container per kind of number the format carries.

§3 of ``Data_structures.md``: *"Every set of numbers is one dataObject: one G or
S container, what it is, and how to use it."* The metadata -- what it is, how
to use it -- lives on :class:`kika.sinbad.DataObject`. This module is the
container: the table, the grid, the gridded array, the materials, the geometry,
the chain of factors.

**The classes are named after the XML tags, and after GNDS where the tag is.**
``table``, ``grid``, ``axes``, ``gridded1d`` and ``XYs1d`` are GNDS-2.1 nodes
that kika already models in :mod:`kika.nuclear_data.model`; ``materials`` and
``geometry`` are SFCOMPO's. Keeping the names means that a reader who knows one
knows the other, and it is what makes :mod:`kika.sinbad.gnds` a translation of
a few lines rather than a mapping table.

**They are not the model classes themselves**, and that is deliberate. The
model is thirty modules that describe an evaluated nuclear-data file; a SINBAD
entry is a shielding experiment, and most of what it holds has no evaluated
counterpart. Reading an entry must not cost a model import, and the subgroup
must be able to take this reader with nothing but numpy. So the containers here
hold numpy arrays and unit *strings*, and :meth:`Gridded.to_gnds` builds the
model object on request, at call time.

**Units are carried, not checked.** The format says GNDS notation, but the
entry is the authority on what it printed: ``1/(cm**3*s)``, ``inch`` and ``%``
all appear in the pilot and not all of them parse under
:mod:`kika.nuclear_data.model.units`. A reader that refused them would refuse
valid files. :func:`kika.sinbad.gnds.check_units` reports which ones a GNDS
consumer would reject, as a question about the file rather than an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from kika.sinbad._xml import LabelIndex, as_float, as_floats, read_table, read_values
from kika.sinbad.exceptions import ContentTypeError, SinbadFormatError

__all__ = [
    "Content",
    "Double",
    "Uncertainty",
    "Range",
    "Axis",
    "Grid",
    "Axes",
    "Column",
    "Table",
    "Gridded",
    "GridContent",
    "DoubleContent",
    "XYs1d",
    "Polynomial1d",
    "Nuclide",
    "Material",
    "Materials",
    "Layer",
    "Shape",
    "Geometry",
    "Factor",
    "FactorChain",
    "FunctionalForm",
    "UnknownContent",
    "read_content",
]


def _pandas():
    """pandas, imported at call time: reading a file must not need it."""
    import pandas as pd  # noqa: PLC0415 - deliberate, see the module docstring

    return pd


def _text(element: Optional[ET.Element]) -> Optional[str]:
    if element is None or element.text is None:
        return None
    return " ".join(element.text.split())


# --------------------------------------------------------------------------
# §2.1 scalars
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Uncertainty:
    """GNDS §4.2.1. The uncertainty of one scalar, in the unit of its value."""

    standard: Optional[float] = None
    confidence_intervals: Tuple[Tuple[str, float, float], ...] = ()

    @classmethod
    def _read(cls, parent: ET.Element) -> Optional["Uncertainty"]:
        node = parent.find("uncertainty")
        if node is None:
            return None
        standard = node.find("standard/double")
        intervals = tuple(
            (i.get("confidence", ""), as_float(i.get("lower")), as_float(i.get("upper")))
            for i in node.findall("confidenceIntervals/interval")
        )
        return cls(
            standard=as_float(standard.get("value")) if standard is not None else None,
            confidence_intervals=intervals,
        )


@dataclass(frozen=True)
class Double:
    """
    GNDS §4.1.1. One number with a unit, and optionally its uncertainty.

    Examples
    --------
    >>> b["sourceStrength"].content.result.value        # doctest: +SKIP
    43201380.0
    """

    label: Optional[str]
    value: float
    unit: str = ""
    uncertainty: Optional[Uncertainty] = None

    @classmethod
    def _read(cls, element: ET.Element) -> "Double":
        return cls(
            label=element.get("label"),
            value=as_float(element.get("value")),
            unit=element.get("unit", ""),
            uncertainty=Uncertainty._read(element),
        )

    def __repr__(self) -> str:
        name = f"{self.label}=" if self.label else ""
        return f"<{name}{self.value:g} {self.unit}>".replace(" >", ">")


@dataclass(frozen=True)
class Range:
    """A value the entry gives only as an interval (§2.1, SINBAD's own)."""

    label: Optional[str]
    min: float
    max: float
    unit: str = ""

    @classmethod
    def _read(cls, element: ET.Element) -> "Range":
        return cls(
            label=element.get("label"),
            min=as_float(element.get("min")),
            max=as_float(element.get("max")),
            unit=element.get("unit", ""),
        )


def read_doubles(parent: ET.Element) -> Dict[str, Any]:
    """Every ``double`` and ``range`` directly under ``parent``, keyed by label.

    Detectors, covers and normalisations all carry their dimensions this way.
    """
    out: Dict[str, Any] = {}
    for node in parent:
        if node.tag == "double":
            scalar = Double._read(node)
            out[scalar.label or "value"] = scalar
        elif node.tag == "range":
            span = Range._read(node)
            out[span.label or "value"] = span
    return out


# --------------------------------------------------------------------------
# §2.2 axes and grids -- the GNDS names, because they are the GNDS nodes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    """GNDS §5.1.2. An index, a label and a unit. Index 0 is the dependent axis."""

    index: int
    label: str = ""
    unit: str = ""

    @classmethod
    def _read(cls, element: ET.Element) -> "Axis":
        return cls(
            index=int(element.get("index", 0)),
            label=element.get("label", ""),
            unit=element.get("unit", ""),
        )


@dataclass(frozen=True)
class Grid(Axis):
    """
    GNDS §5.1.3. An :class:`Axis` that carries the values it is evaluated at.

    ``style`` says what they mean: ``boundaries`` for the N+1 edges of N
    groups, ``points`` for N abscissae, ``parameters`` for something like a
    Legendre order.

    Attributes
    ----------
    values : numpy.ndarray
    style : str
    interpolation : str or None
    """

    values: np.ndarray = field(default_factory=lambda: np.array([]))
    style: str = "none"
    interpolation: Optional[str] = None

    @classmethod
    def _read(cls, element: ET.Element, index: Optional[LabelIndex] = None) -> "Grid":
        return cls(
            index=int(element.get("index", 1)),
            label=element.get("label", ""),
            unit=element.get("unit", ""),
            values=read_values(element, index),
            style=element.get("style", "none"),
            interpolation=element.get("interpolation"),
        )

    @property
    def size(self) -> int:
        """How many cells the grid describes: N boundaries mean N-1 cells."""
        if self.style == "boundaries":
            return max(len(self.values) - 1, 0)
        return len(self.values)

    @property
    def centers(self) -> np.ndarray:
        """Mid-points of a ``boundaries`` grid; the values themselves otherwise."""
        if self.style != "boundaries" or len(self.values) < 2:
            return self.values
        return 0.5 * (self.values[:-1] + self.values[1:])

    @property
    def widths(self) -> np.ndarray:
        """Widths of a ``boundaries`` grid."""
        if self.style != "boundaries":
            raise ContentTypeError(f"grid {self.label!r} is {self.style}, not boundaries")
        return np.diff(self.values)


@dataclass
class Axes:
    """
    GNDS §5.1.1. The axes of a function: the independent ones and the dependent one.

    Look-up is by ``index``, never by position -- the specification's own
    examples list them highest first.
    """

    axes: List[Axis] = field(default_factory=list)

    @classmethod
    def _read(cls, parent: ET.Element, index: Optional[LabelIndex] = None) -> "Axes":
        node = parent.find("axes")
        if node is None:
            return cls([])
        out: List[Axis] = []
        for child in node:
            if child.tag == "grid":
                out.append(Grid._read(child, index))
            elif child.tag == "axis":
                out.append(Axis._read(child))
        return cls(out)

    def __len__(self) -> int:
        return len(self.axes)

    def __iter__(self):
        return iter(self.axes)

    def __getitem__(self, index: int) -> Axis:
        for axis in self.axes:
            if axis.index == index:
                return axis
        raise KeyError(f"no axis with index {index}; have {[a.index for a in self.axes]}")

    @property
    def dependent(self) -> Axis:
        """Index 0 -- what the array *is*."""
        return self[0]

    @property
    def independent(self) -> List[Grid]:
        """The grids, highest index first, as GNDS writes them."""
        return sorted(
            (a for a in self.axes if isinstance(a, Grid)),
            key=lambda a: -a.index,
        )

    def by_label(self, label: str) -> Axis:
        for axis in self.axes:
            if axis.label == label:
                return axis
        raise KeyError(f"no axis labelled {label!r}")


# --------------------------------------------------------------------------
# the content base class
# --------------------------------------------------------------------------


class Content:
    """
    Base of everything a ``dataObject`` can hold.

    Subclasses add the accessors that make sense for their shape; all of them
    answer :attr:`tag` and, where it means anything, ``to_dataframe()``.
    """

    #: The XML tag this class reads.
    tag: str = ""

    def to_dataframe(self):
        """Return the content as a :class:`pandas.DataFrame`."""
        raise ContentTypeError(
            f"{type(self).__name__} has no single table form; "
            f"read its attributes instead"
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


# --------------------------------------------------------------------------
# §2.3 table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """
    GNDS §5.4.2-3 plus the SINBAD attributes of §2.3.

    Attributes
    ----------
    index, name, unit : int, str, str
    role : str
        ``independent``, ``value`` (the default) or ``uncertainty``.
    kind : str or None
        For an uncertainty column: ``component`` or ``total``.
    confidence_level : str or None
        As published, e.g. ``"1 s.d."``.
    types : str or None
        GNDS §5.4.3, as written: ``UTF8Text``, ``Integer32``, ...
    is_text : bool
        True when :attr:`types` says the cells are text. Any other declared
        type is still read as a number -- ``types="Integer32"`` is a column of
        numbers, and treating every typed column as text would turn one into
        strings.
    calculated_by : str or None
        Label of the calculation that produced it.
    response_function : str or None
        Label of the detector response function it was folded with.
    """

    index: int
    name: str
    unit: str = ""
    role: str = "value"
    kind: Optional[str] = None
    confidence_level: Optional[str] = None
    types: Optional[str] = None
    calculated_by: Optional[str] = None
    response_function: Optional[str] = None

    #: The GNDS type names whose cells are text rather than numbers.
    TEXT_TYPES = frozenset({"utf8text", "string", "text"})

    @property
    def is_text(self) -> bool:
        """Whether the cells of this column are text."""
        return (self.types or "").strip().lower() in self.TEXT_TYPES

    @classmethod
    def _read(cls, element: ET.Element) -> "Column":
        return cls(
            index=int(element.get("index", 0)),
            name=element.get("name", ""),
            unit=element.get("unit", ""),
            role=element.get("role", "value"),
            kind=element.get("kind"),
            confidence_level=element.get("confidenceLevel"),
            types=element.get("types"),
            calculated_by=element.get("calculatedBy"),
            response_function=element.get("responseFunction"),
        )


class Table(Content):
    """
    GNDS §5.4.1. Columns with a role, and rows that may have blanks.

    Values at detector positions start with ``position`` and
    ``shieldThickness``, both independent, and rows are matched on
    ``position`` -- which is what makes two tables from different files
    comparable without an index.

    Examples
    --------
    >>> t = b["reactionRate-S32"].table                  # doctest: +SKIP
    >>> t["reactionRate"][:2]                            # doctest: +SKIP
    array([2.02e-17, 4.29e-18])
    >>> t.to_dataframe().head(2)                         # doctest: +SKIP
    """

    tag = "table"

    def __init__(self, element: ET.Element):
        raw_columns, rows = read_table(element)
        self.columns: List[Column] = [Column._read(c) for c in raw_columns]
        self._rows = rows
        self._cells: Dict[str, np.ndarray] = {}
        for position, column in enumerate(self.columns):
            cells = [row[position] if position < len(row) else "" for row in rows]
            if column.is_text:
                self._cells[column.name] = np.array(cells, dtype=object)
            else:
                self._cells[column.name] = np.array(
                    [as_float(c) for c in cells], dtype=float
                )

    # -- shape ------------------------------------------------------------

    @property
    def names(self) -> List[str]:
        """The column names, in order."""
        return [c.name for c in self.columns]

    @property
    def nrows(self) -> int:
        return len(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, name: object) -> bool:
        return name in self._cells

    def __getitem__(self, name: str) -> np.ndarray:
        """One column, by name: floats, or objects for a text column."""
        try:
            return self._cells[name]
        except KeyError:
            raise KeyError(
                f"no column {name!r} in this table; have {self.names}"
            ) from None

    def column(self, name: str) -> Column:
        """The header of one column."""
        for candidate in self.columns:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no column {name!r}; have {self.names}")

    # -- roles ------------------------------------------------------------

    def by_role(self, role: str) -> List[Column]:
        """Every column with this role."""
        return [c for c in self.columns if c.role == role]

    @property
    def independent(self) -> List[Column]:
        return self.by_role("independent")

    @property
    def value_columns(self) -> List[Column]:
        return self.by_role("value")

    @property
    def uncertainty_columns(self) -> List[Column]:
        return self.by_role("uncertainty")

    @property
    def values(self) -> np.ndarray:
        """
        The single value column.

        Raises
        ------
        ContentTypeError
            If the table has none, or several -- a table of four libraries has
            four, and which one is meant is the caller's to say.
        """
        columns = self.value_columns
        if len(columns) != 1:
            raise ContentTypeError(
                f"this table has {len(columns)} value columns "
                f"({[c.name for c in columns]}); ask for one by name"
            )
        return self[columns[0].name]

    @property
    def total_uncertainty(self) -> Optional[np.ndarray]:
        """The ``kind="total"`` uncertainty column, if the table has one."""
        for column in self.uncertainty_columns:
            if column.kind == "total":
                return self[column.name]
        return None

    @property
    def positions(self) -> Optional[np.ndarray]:
        """The ``position`` column -- the key rows are matched on."""
        return self._cells.get("position")

    def to_dataframe(self):
        """
        The table as a :class:`pandas.DataFrame`, columns in document order.

        Units are not in the column names; they are in :attr:`columns`, and
        :meth:`units` gives them as a mapping.
        """
        pd = _pandas()
        return pd.DataFrame({c.name: self._cells[c.name] for c in self.columns})

    def units(self) -> Dict[str, str]:
        """Column name -> unit, as published."""
        return {c.name: c.unit for c in self.columns}

    def __repr__(self) -> str:
        return f"<Table {self.nrows}x{len(self.columns)} {self.names}>"


# --------------------------------------------------------------------------
# §2.4-2.5 gridded arrays
# --------------------------------------------------------------------------


class Gridded(Content):
    """
    GNDS §6.5. An array on one or more grids: ``gridded1d``, ``gridded2d``, ``gridded3d``.

    §2.5: the array is row-major with the **highest** grid index first, so for
    a ``gridded2d`` with grids 2 and 1, ``values[i, j]`` is grid-2 cell ``i``
    and grid-1 cell ``j``.

    Examples
    --------
    >>> g = b["sourceDistribution-xy"].content           # doctest: +SKIP
    >>> g.values.shape                                   # doctest: +SKIP
    (15, 15)
    >>> g.grid("x").centers[:3]                          # doctest: +SKIP
    """

    def __init__(self, element: ET.Element, index: Optional[LabelIndex] = None):
        self.tag = element.tag
        self.axes = Axes._read(element, index)
        array = element.find("array")
        if array is None:
            raise SinbadFormatError(f"<{element.tag}> holds no <array>")
        flat = read_values(array, index)
        shape = array.get("shape")
        self.shape: Tuple[int, ...] = (
            tuple(int(n) for n in shape.split(",")) if shape else (flat.size,)
        )
        self.symmetry = array.get("symmetry")
        if flat.size == int(np.prod(self.shape)):
            self.values = flat.reshape(self.shape)
        else:  # a symmetric array stores one triangle; leave it flat and say so
            self.values = flat

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def unit(self) -> str:
        """The unit of the dependent axis -- what the numbers are in."""
        return self.axes.dependent.unit

    def grid(self, which) -> Grid:
        """One grid, by label (``"energy"``) or by index (``1``)."""
        if isinstance(which, int):
            axis = self.axes[which]
        else:
            axis = self.axes.by_label(which)
        if not isinstance(axis, Grid):
            raise ContentTypeError(f"axis {which!r} is an axis, not a grid")
        return axis

    def to_dataframe(self):
        """
        One row per cell: the grid centers, then the value.

        For a 1-D array this is the natural table. For 2-D and 3-D it is the
        long form -- one row per cell, which is what a plot or a group-by
        wants; :attr:`values` is there when the matrix itself is.
        """
        pd = _pandas()
        grids = self.axes.independent  # highest index first, as the array is laid out
        if not grids:
            return pd.DataFrame({self.axes.dependent.label or "value": self.values.ravel()})
        centers = [g.centers for g in grids]
        mesh = np.meshgrid(*centers, indexing="ij")
        frame = {g.label or f"axis{g.index}": m.ravel() for g, m in zip(grids, mesh)}
        frame[self.axes.dependent.label or "value"] = np.asarray(self.values).ravel()
        return pd.DataFrame(frame)

    def to_gnds(self):
        """
        The same array as :mod:`kika.nuclear_data.model` objects.

        Returns a ``(Axes, ndarray)`` pair built with the model's own
        :class:`~kika.nuclear_data.model.Axes`, :class:`~kika.nuclear_data.model.Grid`
        and :class:`~kika.nuclear_data.model.Axis`, so a SINBAD group structure
        or spectrum can be handed to code that speaks GNDS. The import happens
        here, not at module scope; see :mod:`kika.sinbad.gnds`.
        """
        from kika.sinbad.gnds import to_model_axes  # noqa: PLC0415

        return to_model_axes(self.axes), self.values

    def __repr__(self) -> str:
        shape = "x".join(str(n) for n in self.shape)
        return f"<{self.tag} {shape} {self.axes.dependent.label or ''} [{self.unit}]>"


class GridContent(Content):
    """A bare ``grid`` as the whole content -- an energy group structure (§2.2)."""

    tag = "grid"

    def __init__(self, element: ET.Element, index: Optional[LabelIndex] = None):
        self.grid = Grid._read(element, index)

    @property
    def values(self) -> np.ndarray:
        return self.grid.values

    @property
    def unit(self) -> str:
        return self.grid.unit

    @property
    def size(self) -> int:
        """The number of groups: one fewer than the number of boundaries."""
        return self.grid.size

    def to_dataframe(self):
        pd = _pandas()
        if self.grid.style == "boundaries" and self.grid.size:
            return pd.DataFrame({
                "group": np.arange(1, self.grid.size + 1),
                "lower": self.grid.values[1:] if self.grid.values[0] > self.grid.values[-1]
                else self.grid.values[:-1],
                "upper": self.grid.values[:-1] if self.grid.values[0] > self.grid.values[-1]
                else self.grid.values[1:],
            })
        return pd.DataFrame({self.grid.label or "value": self.grid.values})

    def __repr__(self) -> str:
        return (
            f"<grid {self.grid.label!r} {len(self.grid.values)} {self.grid.style} "
            f"[{self.grid.unit}]>"
        )


class XYs1d(Content):
    """GNDS §6.1.1. A tabulated function, as interleaved x,y pairs."""

    tag = "XYs1d"

    def __init__(self, element: ET.Element, index: Optional[LabelIndex] = None):
        self.interpolation = element.get("interpolation", "lin-lin")
        self.axes = Axes._read(element, index)
        flat = read_values(element, index)
        self.x = flat[0::2]
        self.y = flat[1::2]

    def to_dataframe(self):
        pd = _pandas()
        xlabel = self.axes[1].label if len(self.axes) > 1 else "x"
        ylabel = self.axes.dependent.label if len(self.axes) else "y"
        return pd.DataFrame({xlabel or "x": self.x, ylabel or "y": self.y})

    def __repr__(self) -> str:
        return f"<XYs1d {len(self.x)} points, {self.interpolation}>"


class Polynomial1d(Content):
    """GNDS §6.2.2. Coefficients, lowest order first, over a domain."""

    tag = "polynomial1d"

    def __init__(self, element: ET.Element, index: Optional[LabelIndex] = None):
        self.axes = Axes._read(element, index)
        self.coefficients = read_values(element, index)
        self.domain_min = as_float(element.get("domainMin"))
        self.domain_max = as_float(element.get("domainMax"))

    def __call__(self, x):
        """Evaluate the polynomial at ``x``."""
        return np.polyval(self.coefficients[::-1], x)

    def __repr__(self) -> str:
        return f"<polynomial1d order {len(self.coefficients) - 1}>"


# --------------------------------------------------------------------------
# §2.10 materials and geometry -- SFCOMPO's structures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Nuclide:
    """One line of an SFCOMPO ``csvData`` block: a nuclide and its amount."""

    name: str
    amount: float


@dataclass(frozen=True)
class Material:
    """
    SFCOMPO ``MaterialType``. A density and a composition.

    Attributes
    ----------
    name : str
    density, density_unit : float, str
    nuclides : tuple of Nuclide
    units : str
        What the amounts are -- ``weight fraction``, ``atom fraction``, ...
    """

    name: str
    density: float
    density_unit: str
    nuclides: Tuple[Nuclide, ...]
    units: str = ""
    nuclide_format: str = "ZA"

    @property
    def total(self) -> float:
        """The sum of the amounts. Should be 1 for a fraction -- and is not always."""
        return float(sum(n.amount for n in self.nuclides))

    @classmethod
    def _read(cls, element: ET.Element) -> "Material":
        density = element.find("materialDensity")
        nuclides_node = element.find("nuclides")
        entries: List[Nuclide] = []
        units = ""
        nuclide_format = "ZA"
        if nuclides_node is not None:
            units = nuclides_node.get("nuclideDensityUnits", "")
            nuclide_format = nuclides_node.get("nuclideFormat", "ZA")
            csv = nuclides_node.find("csvData")
            delimiter = nuclides_node.get("delimiter", "space")
            sep = None if delimiter == "space" else delimiter
            for line in (csv.text or "").splitlines() if csv is not None else []:
                parts = line.split(sep) if sep else line.split()
                if len(parts) >= 2:
                    entries.append(Nuclide(parts[0].strip(), float(parts[1])))
        return cls(
            name=element.get("name", ""),
            density=as_float(density.get("value")) if density is not None else float("nan"),
            density_unit=density.get("unit", "") if density is not None else "",
            nuclides=tuple(entries),
            units=units,
            nuclide_format=nuclide_format,
        )


class Materials(Content):
    """SFCOMPO ``MaterialsType``. The materials of the assembly, by name."""

    tag = "materials"

    def __init__(self, element: ET.Element):
        self.materials: Dict[str, Material] = {
            m.get("name", ""): Material._read(m) for m in element.findall("material")
        }

    def __len__(self) -> int:
        return len(self.materials)

    def __iter__(self):
        return iter(self.materials.values())

    def __contains__(self, name: object) -> bool:
        return name in self.materials

    def __getitem__(self, name: str) -> Material:
        try:
            return self.materials[name]
        except KeyError:
            raise KeyError(
                f"no material {name!r}; have {list(self.materials)}"
            ) from None

    @property
    def names(self) -> List[str]:
        return list(self.materials)

    def to_dataframe(self):
        """One row per nuclide: material, density, nuclide, amount."""
        pd = _pandas()
        rows = [
            {
                "material": material.name,
                "density": material.density,
                "density_unit": material.density_unit,
                "nuclide": nuclide.name,
                "amount": nuclide.amount,
                "units": material.units,
            }
            for material in self.materials.values()
            for nuclide in material.nuclides
        ]
        return pd.DataFrame(rows)

    def to_kika_materials(self):
        """
        The same compositions as :class:`kika.materials.MaterialCollection`.

        kika already models a material as a density plus weight or atom
        fractions, and everything downstream -- MCNP cards, ACE look-ups --
        speaks that. The import is deferred to call time.
        """
        from kika.sinbad.gnds import to_kika_materials  # noqa: PLC0415

        return to_kika_materials(self)

    def __repr__(self) -> str:
        return f"<Materials {len(self.materials)}: {', '.join(list(self.materials)[:4])}...>"


@dataclass(frozen=True)
class Layer:
    """SFCOMPO ``GeoLayerType``. One slice of the stack, between two coordinates."""

    name: str
    material: Optional[str]
    min: float
    max: float
    unit: str = ""

    @property
    def thickness(self) -> float:
        return self.max - self.min

    @classmethod
    def _read(cls, element: ET.Element) -> "Layer":
        low, high = element.find("min"), element.find("max")
        return cls(
            name=element.get("name", ""),
            material=element.get("material"),
            min=as_float(low.get("value")) if low is not None else float("nan"),
            max=as_float(high.get("value")) if high is not None else float("nan"),
            unit=(low.get("unit", "") if low is not None else ""),
        )


@dataclass
class Shape:
    """SFCOMPO ``GeoSlabType`` / ``GeoCylinderType`` / ``GeoSphereType``."""

    kind: str
    name: str
    description: str = ""
    axis: str = ""
    layers: Tuple[Layer, ...] = ()
    lengths: Dict[str, Double] = field(default_factory=dict)

    @classmethod
    def _read(cls, element: ET.Element) -> "Shape":
        layers_node = element.find("layers")
        lengths = {
            child.tag: Double(
                label=child.tag,
                value=as_float(child.get("value")),
                unit=child.get("unit", ""),
            )
            for child in element
            if child.tag in ("length_x", "length_y", "length_z", "radius", "height")
        }
        return cls(
            kind=element.tag,
            name=element.get("name", ""),
            description=element.get("description", ""),
            axis=layers_node.get("axis", "") if layers_node is not None else "",
            layers=tuple(
                Layer._read(layer) for layer in
                (layers_node.findall("layer") if layers_node is not None else [])
            ),
            lengths=lengths,
        )


class Geometry(Content):
    """
    SFCOMPO ``GeometryType``. The assembly, as stacked layers.

    Examples
    --------
    >>> g = b["assemblyGeometry"].content                # doctest: +SKIP
    >>> g.layer("fissionPlate").thickness                # doctest: +SKIP
    2.9
    """

    tag = "geometry"

    def __init__(self, element: ET.Element):
        self.shapes: List[Shape] = [
            Shape._read(child)
            for child in element
            if child.tag in ("slab", "cylinder", "sphere")
        ]

    @property
    def layers(self) -> List[Layer]:
        """Every layer of every shape, in document order."""
        return [layer for shape in self.shapes for layer in shape.layers]

    def layer(self, name: str) -> Layer:
        """One layer, by name -- what a ``position`` points at."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(f"no layer {name!r}")

    def to_dataframe(self):
        """One row per layer: shape, name, material, min, max, thickness."""
        pd = _pandas()
        return pd.DataFrame([
            {
                "shape": shape.name,
                "layer": layer.name,
                "material": layer.material,
                "min": layer.min,
                "max": layer.max,
                "thickness": layer.thickness,
                "unit": layer.unit,
            }
            for shape in self.shapes
            for layer in shape.layers
        ])

    def __repr__(self) -> str:
        return f"<Geometry {len(self.shapes)} shape(s), {len(self.layers)} layers>"


# --------------------------------------------------------------------------
# §2.9 factor chain, §2.7 functional form
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Factor:
    """One term of a :class:`FactorChain`, with where it came from."""

    value: Double
    provenance: str = ""
    uncertainty_as_published: Optional[str] = None

    @property
    def label(self) -> Optional[str]:
        return self.value.label


class FactorChain(Content):
    """
    A number the entry builds as a product (or sum) of stated factors.

    The pilot's source strength is the example: plate power x fissions per watt
    x neutrons per fission. :meth:`recompute` multiplies the factors out, which
    is how the checker verifies the published result -- and how a user finds
    out that a normalisation they are about to apply is the one the entry meant.

    Examples
    --------
    >>> chain = b["sourceStrength"].content              # doctest: +SKIP
    >>> chain.result.value, chain.recompute()            # doctest: +SKIP
    (43201380.0, 43201380.16...)
    """

    tag = "factorChain"

    def __init__(self, element: ET.Element):
        self.operator = element.get("operator", "product")
        self.factors: List[Factor] = []
        for node in element.findall("factor"):
            scalar = node.find("double")
            if scalar is None:
                continue
            self.factors.append(
                Factor(
                    value=Double._read(scalar),
                    provenance=node.get("provenance", ""),
                    uncertainty_as_published=node.get("uncertaintyAsPublished"),
                )
            )
        result = element.find("result")
        scalar = result.find("double") if result is not None else None
        self.result: Optional[Double] = Double._read(scalar) if scalar is not None else None
        self.result_nature = result.get("nature", "") if result is not None else ""

    def __len__(self) -> int:
        return len(self.factors)

    def __getitem__(self, label: str) -> Factor:
        for factor in self.factors:
            if factor.label == label:
                return factor
        raise KeyError(f"no factor {label!r}; have {[f.label for f in self.factors]}")

    def recompute(self) -> float:
        """Combine the factors with the declared operator."""
        values = [f.value.value for f in self.factors]
        if self.operator == "sum":
            return float(np.sum(values))
        return float(np.prod(values))

    def to_dataframe(self):
        pd = _pandas()
        return pd.DataFrame([
            {
                "factor": f.label,
                "value": f.value.value,
                "unit": f.value.unit,
                "provenance": f.provenance,
                "uncertainty": f.uncertainty_as_published,
            }
            for f in self.factors
        ])

    def __repr__(self) -> str:
        result = f" = {self.result.value:g}" if self.result else ""
        return f"<FactorChain {self.operator} of {len(self.factors)}{result}>"


class FunctionalForm(Content):
    """A form GNDS does not have, given as parameters (§2.7). Unused in the pilot."""

    tag = "functionalForm"

    def __init__(self, element: ET.Element):
        self.form = element.get("form", "")
        self.parameters: Dict[str, Double] = {}
        for node in element.findall("parameter"):
            scalar = node.find("double")
            if scalar is not None:
                parameter = Double._read(scalar)
                self.parameters[parameter.label or "value"] = parameter
        domain = element.find("domain")
        self.domain = (
            (as_float(domain.get("min")), as_float(domain.get("max")), domain.get("unit", ""))
            if domain is not None else None
        )
        fitted = element.find("fittedTo")
        self.fitted_to = fitted.get("ref") or fitted.get("status") if fitted is not None else None
        goodness = element.find("goodness")
        self.goodness = (
            (goodness.get("statistic", ""), as_float(goodness.get("value")))
            if goodness is not None else None
        )

    def __repr__(self) -> str:
        return f"<FunctionalForm {self.form!r}, {len(self.parameters)} parameters>"


class DoubleContent(Content):
    """A ``dataObject`` whose whole content is one number."""

    tag = "double"

    def __init__(self, element: ET.Element):
        self.value = Double._read(element)

    def to_dataframe(self):
        pd = _pandas()
        return pd.DataFrame([{
            "label": self.value.label,
            "value": self.value.value,
            "unit": self.value.unit,
        }])

    def __repr__(self) -> str:
        return f"<Double {self.value.value:g} {self.value.unit}>"


class UnknownContent(Content):
    """
    A container this reader does not model yet, kept as the raw element.

    A draft format grows, and a file written against a later one must still
    open. :attr:`element` is the live :mod:`xml.etree` node, so nothing is lost
    -- only the convenience.
    """

    def __init__(self, element: ET.Element):
        self.tag = element.tag
        self.element = element

    def __repr__(self) -> str:
        return f"<UnknownContent <{self.tag}> -- not modelled; use .element>"


#: Tag -> the class that reads it. :func:`read_content` is the only user.
_READERS = {
    "table": Table,
    "grid": GridContent,
    "gridded1d": Gridded,
    "gridded2d": Gridded,
    "gridded3d": Gridded,
    "XYs1d": XYs1d,
    "polynomial1d": Polynomial1d,
    "materials": Materials,
    "geometry": Geometry,
    "factorChain": FactorChain,
    "functionalForm": FunctionalForm,
    "double": DoubleContent,
}

#: Which of them want the label index (to follow a linked grid).
_NEEDS_INDEX = frozenset({"grid", "gridded1d", "gridded2d", "gridded3d", "XYs1d",
                          "polynomial1d"})


def read_content(element: ET.Element, index: Optional[LabelIndex] = None) -> Content:
    """
    Build the right :class:`Content` for one container element.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        The child of a ``dataObject`` that holds its numbers.
    index : LabelIndex, optional
        Used to follow a ``link`` standing in for a grid's values.

    Returns
    -------
    Content
        :class:`UnknownContent` for a tag this version does not model.
    """
    reader = _READERS.get(element.tag)
    if reader is None:
        return UnknownContent(element)
    if element.tag in _NEEDS_INDEX:
        return reader(element, index)
    return reader(element)
