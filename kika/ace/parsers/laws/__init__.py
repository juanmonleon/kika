"""
Package for ACE energy distribution law parsers.

Each law represents a different method for representing energy distributions
in MCNP ACE format files.
"""

from .law_1 import parse_tabular_energy_distribution
from .law_2 import parse_discrete_energy_distribution
from .law_3 import parse_level_scattering
from .law_33 import parse_particle_level_scattering
from .law_4 import parse_continuous_energy_angle_distribution
from .law_5 import parse_general_evaporation_spectrum
from .law_7 import parse_maxwell_fission_spectrum
from .law_9 import parse_evaporation_spectrum
from .law_11 import parse_energy_dependent_watt_spectrum
from .law_22 import parse_tabular_linear_functions
from .law_24 import parse_tabular_energy_multipliers
from .law_44 import parse_kalbach_mann_distribution
from .law_61 import parse_tabulated_angle_energy_distribution
from .law_66 import parse_nbody_phase_space_distribution
from .law_67 import parse_laboratory_angle_energy_distribution
from .energy_dependent_yields import parse_energy_dependent_yield

__all__ = [
    'parse_tabular_energy_distribution',
    'parse_discrete_energy_distribution',
    'parse_level_scattering',
    'parse_particle_level_scattering',
    'parse_continuous_energy_angle_distribution',
    'parse_general_evaporation_spectrum',
    'parse_maxwell_fission_spectrum',
    'parse_evaporation_spectrum',
    'parse_energy_dependent_watt_spectrum',
    'parse_tabular_linear_functions',
    'parse_tabular_energy_multipliers',
    'parse_kalbach_mann_distribution',
    'parse_tabulated_angle_energy_distribution',
    'parse_nbody_phase_space_distribution',
    'parse_laboratory_angle_energy_distribution',
    'parse_energy_dependent_yield'
]
