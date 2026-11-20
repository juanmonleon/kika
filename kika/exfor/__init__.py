"""
Experimental EXFOR data, read from an X4Pro database or from JSON files.

The classes hold angular distributions, cross-sections and general EXFOR quantities
with their metadata, convert between the laboratory (LAB) and centre-of-mass (CM)
frames and between units, and plot the data against an evaluation. The database
location is set once with :func:`~kika.exfor.configure`.
"""

# Configuration (call configure() to set defaults for the session)
from kika.exfor.config import configure, get_config

# Constants
from kika.exfor._constants import (
    EXFOR_QUANTITY_CODES,
    EXFOR_QUANTITY_WILDCARDS,
    EXFOR_URL_BASE,
    QUANTITY_FAMILIES,
)

# Primary API - Main exports
from kika.exfor.io import read_exfor, read_all_exfor
from kika.exfor.exfor_entry import ExforEntry
from kika.exfor.angular_distribution import ExforAngularDistribution
from kika.exfor.cross_section import ExforCrossSection
from kika.exfor.experiment import ExforExperiment

# Database API
from kika.exfor.database import (
    X4ProDatabase,
    X4ProDataset,
    read_exfor_from_database,
    parse_reacode_fields,
    exfor_quantity,
    canonical_unit_factor,
)

# Transform functions
from kika.exfor.transforms import (
    cos_cm_from_cos_lab,
    cos_lab_from_cos_cm,
    jacobian_cm_to_lab,
    jacobian_lab_to_cm,
    transform_lab_to_cm,
    transform_cm_to_lab,
    angle_deg_to_cos,
    cos_to_angle_deg,
)

# Plotting functions
from kika.exfor.plotting import (
    plot_exfor_angular,
    plot_exfor_ace_comparison,
    plot_multiple_energies,
)

# Legacy API - Deprecated (imported for backward compatibility)
from kika.exfor.AD_utils import (
    # ACE Data Processing
    extract_angular_distribution,
    calculate_differential_cross_section,
    cosine_to_angle_degrees,
    angle_degrees_to_cosine,
    # EXFOR Data Processing (deprecated - use read_exfor instead)
    load_exfor_data,
    extract_experiment_info,
    load_all_exfor_data,
    # Plotting Functions (deprecated - use new plotting module)
    plot_angular_distribution,
    plot_combined_angular_distribution,
    plot_individual_energy_comparisons,
    plot_combined_angular_distribution_multi_ace,
    plot_all_energies_comparison,
)

__all__ = [
    # Configuration
    "configure",
    "get_config",
    # Constants
    "EXFOR_QUANTITY_CODES",
    "EXFOR_QUANTITY_WILDCARDS",
    "EXFOR_URL_BASE",
    "QUANTITY_FAMILIES",
    # Primary API
    "read_exfor",
    "read_all_exfor",
    "ExforEntry",
    "ExforAngularDistribution",
    "ExforCrossSection",
    "ExforExperiment",
    # Database API
    "X4ProDatabase",
    "parse_reacode_fields",
    "exfor_quantity",
    "canonical_unit_factor",
    "X4ProDataset",
    "read_exfor_from_database",
    # Transforms
    "cos_cm_from_cos_lab",
    "cos_lab_from_cos_cm",
    "jacobian_cm_to_lab",
    "jacobian_lab_to_cm",
    "transform_lab_to_cm",
    "transform_cm_to_lab",
    "angle_deg_to_cos",
    "cos_to_angle_deg",
    # Plotting
    "plot_exfor_angular",
    "plot_exfor_ace_comparison",
    "plot_multiple_energies",
    # Legacy API (deprecated)
    "extract_angular_distribution",
    "calculate_differential_cross_section",
    "cosine_to_angle_degrees",
    "angle_degrees_to_cosine",
    "load_exfor_data",
    "extract_experiment_info",
    "load_all_exfor_data",
    "plot_angular_distribution",
    "plot_combined_angular_distribution",
    "plot_individual_energy_comparisons",
    "plot_combined_angular_distribution_multi_ace",
    "plot_all_energies_comparison",
]
