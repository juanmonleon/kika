"""
Units and physical quantities for plotting.

kika hands curves to the plotting layer in canonical units (eV, barn, b/sr) and
converts only for display. This module is the table that makes that possible: what
each unit measures, its factor to the canonical unit, and how each quantity is
labelled on an axis.

Plain multiplicative units only. Degrees are not here on purpose: an angle in
degrees is not a rescaled cosine, so converting between them is a change of
variable (and of the density), not a unit conversion.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


class UnitError(ValueError):
    """Two units that do not measure the same thing."""


#: unit -> (dimension, factor to the dimension's canonical unit)
_UNITS: Dict[str, Tuple[str, float]] = {
    # energy (canonical eV)
    'eV': ('energy', 1.0),
    'keV': ('energy', 1.0e3),
    'MeV': ('energy', 1.0e6),
    # cross section (canonical barn)
    'b': ('area', 1.0),
    'mb': ('area', 1.0e-3),
    'ub': ('area', 1.0e-6),
    'µb': ('area', 1.0e-6),
    # differential cross section per solid angle (canonical b/sr)
    'b/sr': ('area/sr', 1.0),
    'mb/sr': ('area/sr', 1.0e-3),
    # differential cross section per unit cosine (canonical b)
    # -- deliberately a separate dimension from area: dsigma/dmu = 2 pi dsigma/dOmega
    # -- per unit cosine, never interchangeable with a plain cross section.
    'b/mu': ('area/mu', 1.0),
    'mb/mu': ('area/mu', 1.0e-3),
    # dimensionless ratios (canonical: a plain fraction)
    '1': ('ratio', 1.0),
    '%': ('ratio', 1.0e-2),
    # cosine of an angle
    'mu': ('cosine', 1.0),
    # probability density per unit cosine, f(mu)
    '1/mu': ('density/mu', 1.0),
}

#: Canonical unit of each dimension.
CANONICAL: Dict[str, str] = {
    'energy': 'eV', 'area': 'b', 'area/sr': 'b/sr', 'area/mu': 'b/mu',
    'ratio': '1', 'cosine': 'mu', 'density/mu': '1/mu',
}

#: How a unit is written in an axis label. The empty string means "no unit shown".
_DISPLAY: Dict[str, str] = {
    '1': '', 'mu': '', '1/mu': '', 'ub': 'µb',
}

# Aliases people actually type; normalised before lookup.
_ALIASES: Dict[str, str] = {
    'barn': 'b', 'barns': 'b', 'millibarn': 'mb', 'ev': 'eV', 'kev': 'keV', 'mev': 'MeV',
    'percent': '%', '': '1', 'fraction': '1', 'dimensionless': '1',
    'cos': 'mu', 'cosine': 'mu', 'μ': 'mu',
}


def normalise_unit(unit: Optional[str]) -> Optional[str]:
    """The registry spelling of ``unit`` (``'MEV'`` -> ``'MeV'``), or None."""
    if unit is None:
        return None
    u = str(unit).strip()
    if u in _UNITS:
        return u
    lowered = u.lower()
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    for known in _UNITS:
        if known.lower() == lowered:
            return known
    raise UnitError(f"Unknown unit {unit!r}. Known: {', '.join(sorted(_UNITS))}")


def dimension(unit: str) -> str:
    """What ``unit`` measures (``'energy'``, ``'area'``, ...)."""
    return _UNITS[normalise_unit(unit)][0]


def compatible(a: Optional[str], b: Optional[str]) -> bool:
    """Whether values in ``a`` can be converted to ``b``."""
    if a is None or b is None:
        return False
    try:
        return dimension(a) == dimension(b)
    except UnitError:
        return False


def factor(from_unit: str, to_unit: str) -> float:
    """Multiply values in ``from_unit`` by this to express them in ``to_unit``."""
    fa, fb = normalise_unit(from_unit), normalise_unit(to_unit)
    da, ka = _UNITS[fa]
    db, kb = _UNITS[fb]
    if da != db:
        raise UnitError(f"Cannot convert {from_unit!r} ({da}) to {to_unit!r} ({db})")
    return ka / kb


def convert(values, from_unit: str, to_unit: str) -> np.ndarray:
    """``values`` (in ``from_unit``) expressed in ``to_unit``."""
    f = factor(from_unit, to_unit)
    arr = np.asarray(values, dtype=float)
    return arr if f == 1.0 else arr * f


def display_unit(unit: Optional[str]) -> str:
    """How ``unit`` appears in an axis label (``'1'`` -> no unit shown)."""
    if unit is None:
        return ''
    u = normalise_unit(unit)
    return _DISPLAY.get(u, u)


@dataclass(frozen=True)
class Quantity:
    """
    A plottable physical quantity: its axes, canonical units and labels.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``'cross_section'``.
    label : str
        Human-readable name.
    x_label, y_label : str
        Axis names without units.
    x_unit, y_unit : str
        Canonical units of the axes.
    interpolation : str
        Default interpolation between points, for comparisons ('log-log', 'lin-lin', ...).
    log_x, log_y : bool
        Natural axis scales.
    """
    name: str
    label: str
    x_label: str
    y_label: str
    x_unit: str
    y_unit: str
    interpolation: str = 'lin-lin'
    log_x: bool = False
    log_y: bool = False

    def axis_label(self, axis: str, unit: Optional[str] = None) -> str:
        """``'Incident energy (MeV)'``: the axis name and ``unit`` (default: canonical)."""
        name = self.x_label if axis == 'x' else self.y_label
        shown = display_unit(unit or (self.x_unit if axis == 'x' else self.y_unit))
        return f'{name} ({shown})' if shown else name


QUANTITIES: Dict[str, Quantity] = {q.name: q for q in (
    Quantity('cross_section', 'Cross section',
             'Incident energy', 'Cross section', 'eV', 'b',
             interpolation='log-log', log_x=True, log_y=True),
    Quantity('legendre_coefficient', 'Legendre coefficient',
             'Incident energy', 'Legendre coefficient $a_L$', 'eV', '1',
             interpolation='lin-lin', log_x=True),
    Quantity('angular_distribution', 'Angular distribution',
             r'$\mu = \cos\theta$', r'$f(\mu)$', 'mu', '1/mu',
             interpolation='lin-lin'),
    Quantity('differential_cross_section', 'Differential cross section',
             r'$\mu = \cos\theta$', r'$d\sigma/d\Omega$', 'mu', 'b/sr',
             interpolation='lin-lin', log_y=True),
    Quantity('relative_uncertainty', 'Relative uncertainty',
             'Incident energy', 'Relative uncertainty', 'eV', '%',
             interpolation='lin-lin', log_x=True),
)}


def get_quantity(name: str) -> Quantity:
    """The :class:`Quantity` called ``name``."""
    try:
        return QUANTITIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown quantity {name!r}. Available: {', '.join(QUANTITIES)}"
        ) from None


class MixedQuantityWarning(UserWarning):
    """Curves of different quantities, units or frames were put on one axes."""


def data_to_units(data, x_unit: Optional[str] = None, y_unit: Optional[str] = None):
    """A copy of ``data`` (PlotData) expressed in ``x_unit`` / ``y_unit``.

    Axes whose unit is unknown on either side are left alone, so data without
    unit metadata (every PlotData made before units existed) passes through
    untouched. Incompatible units raise :class:`UnitError`.
    """
    import copy

    fx = factor(data.x_unit, x_unit) if (data.x_unit and x_unit) else 1.0
    fy = factor(data.y_unit, y_unit) if (data.y_unit and y_unit) else 1.0
    if fx == 1.0 and fy == 1.0 and (not x_unit or data.x_unit in (None, x_unit)) \
            and (not y_unit or data.y_unit in (None, y_unit)):
        return data
    new = copy.copy(data)
    new.metadata = dict(data.metadata)
    if fx != 1.0:
        new.x = np.asarray(data.x, dtype=float) * fx
        if 'xerr' in new.metadata and new.metadata['xerr'] is not None:
            new.metadata['xerr'] = np.asarray(new.metadata['xerr'], dtype=float) * fx
    if fy != 1.0:
        new.y = np.asarray(data.y, dtype=float) * fy
        if 'yerr' in new.metadata and new.metadata['yerr'] is not None:
            new.metadata['yerr'] = np.asarray(new.metadata['yerr'], dtype=float) * fy
    if data.x_unit and x_unit:
        new.x_unit = normalise_unit(x_unit)
    if data.y_unit and y_unit:
        new.y_unit = normalise_unit(y_unit)
    return new


def band_to_units(band, fx: float, fy: float):
    """A copy of an UncertaintyBand scaled by ``fx`` in x and ``fy`` in y.

    Relative bands are unit-free in y and only follow x."""
    import copy

    if band is None or (fx == 1.0 and fy == 1.0):
        return band
    new = copy.copy(band)
    new.x = np.asarray(band.x, dtype=float) * fx
    if not band.is_relative() and fy != 1.0:
        new.y_lower = np.asarray(band.y_lower, dtype=float) * fy
        new.y_upper = np.asarray(band.y_upper, dtype=float) * fy
    return new
