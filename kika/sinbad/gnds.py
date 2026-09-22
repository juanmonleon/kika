"""Handing SINBAD structures to the rest of kika: GNDS axes, units, materials.

Most of what a SINBAD entry stores is a GNDS-2.1 or SFCOMPO node, and kika
already models both -- ``axes``, ``grid`` and ``values`` in
:mod:`kika.nuclear_data.model`, a composition in :mod:`kika.materials`. This
module is the bridge, and it exists as a module of its own for one reason:
**everything in it imports at call time**.

:mod:`kika.nuclear_data.model` is thirty modules describing an evaluated
nuclear-data file, and ``kika/nuclear_data/model/tests/test_dormancy.py``
asserts it stays unreachable from a plain ``import kika``. Reading a shielding
benchmark has no business waking it: a SINBAD entry is an experiment, and the
overlap is the handful of container shapes below. So
:mod:`kika.sinbad.content` holds numpy arrays and unit strings, and the
translation happens when someone asks for it -- which also means the reader can
be handed to the SINBAD subgroup with nothing but numpy underneath.

**Units are reported, not enforced.** The format says GNDS notation, and the
pilot writes ``1/(cm**3*s)``, ``inch`` and ``%``, none of which
:func:`kika.nuclear_data.model.units.parse_unit` accepts today -- the first for
a gap in kika's parser, the others because GNDS §3.5 has no symbol for them.
A reader that validated on open would refuse a valid file, so
:func:`check_units` answers the question instead of imposing the answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - vocabulary only, never imported at run time
    from kika.sinbad.content import Axes, Gridded, Materials

__all__ = ["to_model_axes", "to_model_grid", "to_kika_materials", "check_units"]


def to_model_grid(grid) -> Any:
    """
    One SINBAD ``grid`` as a :class:`kika.nuclear_data.model.Grid`.

    Parameters
    ----------
    grid : kika.sinbad.content.Grid

    Returns
    -------
    kika.nuclear_data.model.Grid
    """
    from kika.nuclear_data.model import Grid as ModelGrid  # noqa: PLC0415
    from kika.nuclear_data.model.enums import GridStyle, Interpolation  # noqa: PLC0415

    style = GridStyle[grid.style] if grid.style in GridStyle.__members__ else GridStyle.none
    interpolation = Interpolation.linlin
    if grid.interpolation:
        key = grid.interpolation.replace("-", "")
        if key in Interpolation.__members__:
            interpolation = Interpolation[key]
    return ModelGrid(
        index=grid.index,
        label=grid.label,
        unit=_model_safe_unit(grid.unit),
        style=style,
        interpolation=interpolation,
        values=np.asarray(grid.values, dtype=float),
    )


def to_model_axes(axes: "Axes") -> Any:
    """
    A SINBAD ``axes`` block as a :class:`kika.nuclear_data.model.Axes`.

    Grids keep their values, axes their labels and units, and index 0 stays the
    dependent axis -- the model's own invariant.

    Parameters
    ----------
    axes : kika.sinbad.content.Axes

    Returns
    -------
    kika.nuclear_data.model.Axes

    Examples
    --------
    >>> model_axes, values = b["sourceSpectrum-BUGJEFF311"].content.to_gnds()  # doctest: +SKIP
    >>> model_axes.dependent.label                                             # doctest: +SKIP
    'chi'
    """
    from kika.nuclear_data.model import Axes as ModelAxes, Axis as ModelAxis  # noqa: PLC0415
    from kika.sinbad.content import Grid  # noqa: PLC0415

    converted: List[Any] = []
    for axis in axes:
        if isinstance(axis, Grid):
            converted.append(to_model_grid(axis))
        else:
            converted.append(
                ModelAxis(index=axis.index, label=axis.label, unit=_model_safe_unit(axis.unit))
            )
    return ModelAxes(converted)


def _model_safe_unit(unit: str) -> str:
    """
    The unit if the model's parser accepts it, and ``""`` if it does not.

    Dropping a unit the model cannot spell is the lesser evil: the alternative
    is that building the axes of a perfectly good spectrum raises. What was
    dropped is not lost -- the SINBAD object keeps the published string, and
    :func:`check_units` lists every one of them.
    """
    from kika.nuclear_data.model.units import UnitError, parse_unit  # noqa: PLC0415

    if not unit:
        return ""
    try:
        parse_unit(unit)
    except UnitError:
        return ""
    return unit


def check_units(benchmark) -> Any:
    """
    Every unit the benchmark uses, and whether a GNDS consumer would accept it.

    The format requires GNDS notation; this says where the file and
    :mod:`kika.nuclear_data.model.units` disagree, which is a question about
    the file, not an error in it.

    Parameters
    ----------
    benchmark : SinbadBenchmark

    Returns
    -------
    pandas.DataFrame
        One row per distinct unit: where it appears, whether it parses, and the
        parser's complaint when it does not.

    Examples
    --------
    >>> from kika.sinbad.gnds import check_units         # doctest: +SKIP
    >>> check_units(b)[lambda f: ~f.parses]              # doctest: +SKIP
    """
    import pandas as pd  # noqa: PLC0415

    from kika.nuclear_data.model.units import UnitError, parse_unit  # noqa: PLC0415
    from kika.sinbad.content import Grid, Gridded, GridContent, Table  # noqa: PLC0415

    seen: Dict[str, List[str]] = {}

    def note(unit: Optional[str], where: str) -> None:
        if not unit:
            return
        seen.setdefault(unit, [])
        if where not in seen[unit]:
            seen[unit].append(where)

    sources = [benchmark] + list(benchmark.calculations)
    for holder in sources:
        for obj in holder.data:
            content = obj.content
            if isinstance(content, Table):
                for column in content.columns:
                    note(column.unit, f"{obj.label}/{column.name}")
            elif isinstance(content, (Gridded,)):
                for axis in content.axes:
                    note(axis.unit, f"{obj.label}/{axis.label or axis.index}")
            elif isinstance(content, GridContent):
                note(content.grid.unit, f"{obj.label}/{content.grid.label}")
        for normalisation in holder.normalisations:
            if normalisation.value is not None:
                note(normalisation.value.unit, f"normalisation/{normalisation.label}")
    for detector in benchmark.detectors:
        for name, dimension in detector.dimensions.items():
            note(getattr(dimension, "unit", ""), f"{detector.label}/{name}")
        if detector.cover is not None:
            for name, dimension in detector.cover.dimensions.items():
                note(getattr(dimension, "unit", ""), f"{detector.label}/cover/{name}")

    rows = []
    for unit, where in sorted(seen.items()):
        try:
            parse_unit(unit)
            rows.append({"unit": unit, "parses": True, "error": "",
                         "used_by": len(where), "first": where[0]})
        except UnitError as exc:
            rows.append({"unit": unit, "parses": False, "error": str(exc),
                         "used_by": len(where), "first": where[0]})
    return pd.DataFrame(rows)


def to_kika_materials(materials: "Materials") -> Any:
    """
    The compositions of an entry as a :class:`kika.materials.MaterialCollection`.

    SFCOMPO writes a natural element with A = 0 (``Fe0``); kika writes it
    without a mass number (``Fe``). Weight fractions become ``'wo'`` and atom
    fractions ``'ao'``; anything else is refused rather than guessed, because a
    fraction read as the wrong kind is a silent factor of the atomic weight.

    Parameters
    ----------
    materials : kika.sinbad.content.Materials

    Returns
    -------
    kika.materials.MaterialCollection

    Examples
    --------
    >>> collection = b["materials"].content.to_kika_materials()   # doctest: +SKIP
    >>> print(collection["mildSteel6"].to_mcnp())                 # doctest: +SKIP
    """
    from kika.materials import Material, MaterialCollection  # noqa: PLC0415
    from kika.sinbad.exceptions import SinbadError  # noqa: PLC0415

    collection = MaterialCollection()
    for number, material in enumerate(materials, start=1):
        units = (material.units or "").lower()
        if "weight" in units:
            fraction_type = "wo"
        elif "atom" in units:
            fraction_type = "ao"
        else:
            raise SinbadError(
                f"material {material.name!r} gives amounts as {material.units!r}; "
                f"kika needs weight or atom fractions to know what they mean"
            )
        converted = Material(
            id=number,
            name=material.name,
            fraction_type=fraction_type,
            density=material.density,
            density_unit="g/cc" if material.density_unit in ("g/cm**3", "g/cc") else material.density_unit,
        )
        for nuclide in material.nuclides:
            name = nuclide.name
            if name.endswith("0") and not name[:-1].isdigit():
                # Fe0 is natural iron; kika spells that Fe. Ca0, Na0 likewise.
                stripped = name[:-1]
                if stripped.isalpha():
                    name = stripped
            converted.add_nuclide(name, nuclide.amount, fraction_type)
        collection.add_material(converted)
    return collection
