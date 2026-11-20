"""Standard multigroup energy structures and a generator for equal-lethargy grids.

Each preset (``SCALE44``, ``XMAS172``, ``WIMS69`` and the others) is a list of group
boundaries in MeV in ascending order, ``N + 1`` values for ``N`` groups.
:func:`equal_lethargy_grid` builds a log-spaced grid between two bounds.
"""
from .grids import (
    VITAMINJ174, VITAMINJ175, SCALE44, SCALE56, SCALE238, SCALE252,
    WIMS69, ECCO33, ONEGROUP20, CASMO12, CASMO40, CASMO70, LANL30, XMAS172
)
from .equal_lethargy import equal_lethargy_grid

__all__ = [
    'VITAMINJ174',
    'VITAMINJ175',
    'SCALE44',
    'SCALE56',
    'SCALE238',
    'SCALE252',
    'WIMS69',
    'ECCO33',
    'ONEGROUP20',
    'CASMO12',
    'CASMO40',
    'CASMO70',
    'LANL30',
    'XMAS172',
    'equal_lethargy_grid',
]