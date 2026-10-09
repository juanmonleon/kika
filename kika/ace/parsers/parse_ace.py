import logging
from kika.ace.classes.ace import Ace
import numpy as np
from kika.ace.classes.header import Header
from kika.ace.parsers.parse_esz import read_esz_block
from kika.ace.parsers import read_header, read_nubar_data
from kika.ace.parsers.parse_delayed import read_delayed_neutron_data
from kika.ace.parsers.parse_mtr import read_mtr_blocks
from kika.ace.parsers.parse_lqr import read_lqr_block
from kika.ace.parsers.parse_tyr import read_tyr_blocks
from kika.ace.parsers.parse_xs_locators import read_xs_locator_blocks
from kika.ace.parsers.parse_xs_data import read_xs_data_block
from kika.ace.parsers.parse_angular_locators import read_angular_locator_blocks
from kika.ace.parsers.parse_angular_distribution import read_angular_distribution_blocks
from kika.ace.parsers.parse_energy_distribution_locators import read_energy_locator_blocks
from kika.ace.parsers.parse_energy_distributions import read_energy_distribution_blocks
from kika.ace.classes.energy_distribution.container import EnergyDistributionContainer
from kika.ace.parsers.parse_gpd import read_gpd_block
from kika.ace.parsers.parse_photon_production_xs import read_production_xs_blocks
from kika.ace.parsers.parse_yield_multipliers import read_yield_multiplier_blocks
from kika.ace.parsers.parse_fission_xs import read_fission_xs_block
from kika.ace.parsers.parse_unresolved_resonance import read_unresolved_resonance_block
from kika.ace.parsers.parse_secondary_particle_types import parse_ptype_block
from kika.ace.parsers.parse_secondary_reaction_counts import parse_ntro_block
from kika.ace.parsers.parse_secondary_data_locators import parse_ixs_block
from kika.ace.parsers.parse_secondary_cross_sections import parse_hpd_block

# Setup logger
logger = logging.getLogger(__name__)

def read_ace(filename, debug=False):
    """
    Read and parse an ACE format file.
    
    This implementation eagerly loads all data except energy distribution data
    which is still loaded on-demand when accessed.
    
    Parameters
    ----------
    filename : str
        Path to the ACE file
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    ace : Ace
        An Ace object containing the parsed data
    """

    if debug:
        logger.debug(f"Reading ACE file: {filename}")
        
    ace = Ace()
    ace.filename = filename
    ace.header = Header()
    ace._debug = debug  # Store debug flag for later use by parsers
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Determine if it's a legacy or 2.0.1 header and read it
    if lines and "2.0.1" in lines[0][:10]:
        ace.header.format_version = "2.0.1"
    else:
        ace.header.format_version = "legacy"
    
    if debug:
        logger.debug(f"ACE format version: {ace.header.format_version}")
    
    # Read the entire header (opening and arrays)
    line_idx = read_header(ace.header, lines, debug=debug)
    
    # Read the XSS array - essential data needed for all parsers
    ace.xss_data = read_xss(lines[line_idx:], ace.header.nxs_array[1])
    
    # Eagerly load all components except energy distribution data
    
    # ESZ Block
    ace.esz_block = read_esz_block(ace, debug)
    
    # Nubar data
    ace.nubar = read_nubar_data(ace, debug)
    
    # Delayed neutron data
    ace.delayed_neutron_data = read_delayed_neutron_data(ace, debug)
    
    # MT Reaction data
    ace.reaction_mt_data = read_mtr_blocks(ace, debug)
    
    # Q values
    ace.q_values = read_lqr_block(ace, debug)
    
    # Particle release data
    ace.particle_release = read_tyr_blocks(ace, debug)
    
    # Cross section locators
    ace.xs_locators = read_xs_locator_blocks(ace, debug)
    
    # Cross section data
    ace.xs_data = read_xs_data_block(ace, debug)
    
    # Angular distribution locators - Fix: Assign the return value instead of direct modification
    ace.angular_locators = read_angular_locator_blocks(ace, debug)
    
    # Angular distribution data - Fix: Assign the return value instead of direct modification
    ace.angular_distributions = read_angular_distribution_blocks(ace, debug)
    
    # Energy distribution locators
    ace.energy_distribution_locators = read_energy_locator_blocks(ace, debug)
    
    # Energy distribution data
    ace.energy_distributions = read_energy_distribution_blocks(ace, debug)

    # Photon production data
    ace.photon_production_data = read_gpd_block(ace, debug)
    
    # Secondary particle types (PTYPE block)
    # This must be read first as other secondary particle blocks depend on it
    ace.secondary_particle_types = parse_ptype_block(ace, debug)
    
    # Photon production cross sections
    photon_xs, particle_xs = read_production_xs_blocks(ace, debug)
    ace.photon_production_xs = photon_xs
    ace.particle_production_xs = particle_xs
    
    # Fission cross section
    ace.fission_xs = read_fission_xs_block(ace, debug)
    
    # Unresolved resonance tables
    ace.unresolved_resonance = read_unresolved_resonance_block(ace, debug)
    
    # Secondary particle reaction counts (NTRO block)
    ace.secondary_particle_reactions = parse_ntro_block(ace, debug)
    
    # Secondary particle data locations (IXS block)
    ace.secondary_particle_data_locations = parse_ixs_block(ace, debug)
    
    # Photon and secondary particle yield multipliers
    photon_yield_multipliers, particle_yield_multipliers = read_yield_multiplier_blocks(ace, debug)
    ace.photon_yield_multipliers = photon_yield_multipliers
    ace.particle_yield_multipliers = particle_yield_multipliers
    
    # Secondary particle cross sections (HPD block)
    ace.secondary_particle_cross_sections = parse_hpd_block(ace, debug)
    
    return ace

#: Lines of XSS parsed per chunk (4 values each).
_XSS_CHUNK_LINES = 65536


def read_xss(lines, n_values=None):
    """
    Read the XSS array of an ACE table into a float64 numpy array.

    The array uses 1-based indexing to match the FORTRAN ``XSS(i)`` of the ACE
    manual: index 0 holds a 0.0 placeholder and ``xss[i]`` is ``XSS(i)``.
    Blocks parsed from it are views of this array (see
    :mod:`kika.ace.classes.xss`).

    Parameters
    ----------
    lines : list of str
        The lines of the file after the header.
    n_values : int, optional
        ``NXS(1)``, the declared length of the XSS array. When given, only the
        lines that hold it are read, and the count is checked.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of length ``n + 1``.
    """
    if n_values:
        lines = lines[:-(-n_values // 4)]  # 4 values per line, last one partial

    # Whitespace split is ~4x faster than slicing 4E20 fields and agrees with it
    # whenever adjacent fields are separated by a blank, which every E20.11 /
    # I20 value is. A glued pair, or a token float() rejects, changes the count,
    # and that falls back to the fixed-width reading.
    # Parsed in chunks so the transient token strings stay bounded (a single
    # split of a 5.8 M-value table holds ~350 MB of str objects at once).
    try:
        values = np.concatenate([np.empty(0)] + [
            np.array(" ".join(lines[i:i + _XSS_CHUNK_LINES]).split(), dtype=np.float64)
            for i in range(0, len(lines), _XSS_CHUNK_LINES)
        ])
    except ValueError:
        values = None
    if values is None or (n_values and values.size != n_values):
        values = _read_xss_fixed_width(lines)
    if n_values and values.size != n_values:
        raise ValueError(
            f"XSS array has {values.size} values, NXS(1) declares {n_values}"
        )

    xss = np.empty(values.size + 1, dtype=np.float64)
    xss[0] = 0.0
    xss[1:] = values
    return xss


def _read_xss_fixed_width(lines):
    """4E20 field-by-field reading; skips blank and non-numeric fields."""
    values = []
    for line in lines:
        for i in range(4):
            field = line[i * 20:(i + 1) * 20].strip()
            if field:
                try:
                    values.append(float(field))
                except ValueError:
                    pass
    return np.array(values, dtype=np.float64)
