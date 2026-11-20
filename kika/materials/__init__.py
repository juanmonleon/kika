"""Material compositions for the Monte Carlo codes, independent of the code.

A material holds nuclides (ZAID, fraction, library overrides), a density and a
temperature. It is built by hand or read from an MCNP or Serpent input, converted
between atomic and weight fractions, and written back as an MCNP or Serpent card.

Examples
--------
>>> from kika.materials import Material, MaterialCollection

>>> # Create a simple material
>>> mat = Material(id=1)
>>> mat.add_nuclide('H', 2.0, 'ao')
>>> mat.add_nuclide('O', 1.0, 'ao')
>>> mat.normalize()
>>> mat.set_density(1.0, 'g/cc')

>>> # Export to different codes
>>> print(mat.to_mcnp())
>>> print(mat.to_serpent())
"""

from .material import Nuclide, NuclideAccessor, Material, MaterialCollection
from .parse_mcnp import read_material
from .parse_serpent import read_serpent_materials

__all__ = [
    'Nuclide',
    'NuclideAccessor',
    'Material',
    'MaterialCollection',
    'read_material',
    'read_serpent_materials',
]
