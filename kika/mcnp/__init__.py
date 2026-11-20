"""MCNP input decks, MCTAL and MESHTAL tally files, and PERT cards.

The readers return labelled objects: :func:`~kika.mcnp.parse_input.read_mcnp` keeps
the PERT cards and the material cards of a deck, and
:func:`~kika.mcnp.parse_mctal.read_mctal` and
:func:`~kika.mcnp.parse_meshtal.read_meshtal` the tallies of a run. The generators write the PERT
cards of a sensitivity run and build the perturbed materials they refer to.
"""

from .parse_input import read_mcnp
from .parse_mctal import read_mctal
from .parse_meshtal import read_meshtal
from .detect_file_type import detect_mcnp_file_type
from .pert_generator import generate_pert_cards, generate_PERTcards, perturb_material, perturb_materials
from kika.materials import Material, MaterialCollection, Nuclide, NuclideAccessor

__all__ = [
    'read_mcnp', 'read_mctal', 'read_meshtal', 'detect_mcnp_file_type',
    'generate_pert_cards', 'generate_PERTcards', 'perturb_material', 'perturb_materials',
    'Material', 'MaterialCollection', 'Nuclide', 'NuclideAccessor',
]
