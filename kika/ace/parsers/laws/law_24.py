import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.tabular_functions import TabularEnergyMultipliers

# Setup logger
logger = logging.getLogger(__name__)

def parse_tabular_energy_multipliers(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> TabularEnergyMultipliers:
    """
    Parse a tabular energy multipliers distribution (Law 24).
    
    According to Table 42, LAW=24 contains:
    - LDAT(1): N_R - Number of interpolation regions
    - LDAT(2) to LDAT(1+N_R): NBT - Interpolation parameters
    - LDAT(2+N_R) to LDAT(1+2*N_R): INT - Interpolation schemes
    - LDAT(2+2*N_R): N_E - Number of incident energies
    - LDAT(3+2*N_R) to LDAT(2+2*N_R+N_E): E_in(l) - Incident energy table
    - LDAT(3+2*N_R+N_E): NET - Number of outgoing values in each table
    - LDAT(4+2*N_R+N_E) to end: Tables of multipliers for each incident energy
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    base_dist : EnergyDistribution
        Base distribution with common properties
    idat_idx : int
        Starting index for the law data in the XSS array
    debug : bool, optional
        If True, enables debug logging
        
    Returns
    -------
    TabularEnergyMultipliers
        Tabular energy multipliers distribution object
    """
    if debug:
        logger.debug(f"Parsing tabular energy multipliers (Law 24) starting at index {idat_idx}")
    
    # Create a new distribution object using the base properties
    distribution = TabularEnergyMultipliers(
        law=base_dist.law,
        idat=base_dist.idat
    )
    
    # Copy applicability data from base_dist
    distribution.applicability_energies = base_dist.applicability_energies
    distribution.applicability_probabilities = base_dist.applicability_probabilities
    distribution.applicability_nbt = base_dist.applicability_nbt
    distribution.applicability_interp = base_dist.applicability_interp
    
    # Check if we have data to parse
    if idat_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Index {idat_idx} out of bounds for XSS data with length {len(ace.xss_data)}")
        return distribution
    
    # Read the number of interpolation regions (N_R)
    distribution.n_interp_regions = int(ace.xss_data[idat_idx])
    if debug:
        logger.debug(f"Number of interpolation regions (N_R): {distribution.n_interp_regions}")
    idx = idat_idx + 1
    n_r = distribution.n_interp_regions
    
    # Read the interpolation parameters if present
    if n_r > 0 and idx + 2*n_r - 1 < len(ace.xss_data):
        # Read NBT values
        distribution.nbt = [int(ace.xss_data[idx + i]) for i in range(n_r)]
        if debug:
            logger.debug(f"NBT values: {distribution.nbt}")
        idx += n_r
        
        # Read INT values
        distribution.interp = [int(ace.xss_data[idx + i]) for i in range(n_r)]
        if debug:
            logger.debug(f"INT values: {distribution.interp}")
        idx += n_r
    elif n_r > 0 and debug:
        logger.debug(f"Not enough data to read interpolation parameters. Need index up to {idx + 2*n_r - 1}, have {len(ace.xss_data)}")
    
    # Read the number of incident energies (N_E)
    if idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Index {idx} out of bounds for XSS data with length {len(ace.xss_data)}")
        return distribution
    
    distribution.n_energies = int(ace.xss_data[idx])
    if debug:
        logger.debug(f"Number of incident energies (N_E): {distribution.n_energies}")
    idx += 1
    n_e = distribution.n_energies
    
    # Read the incident energies - a view of xss_data
    if idx + n_e - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"Not enough data to read incident energies. Need index up to {idx + n_e - 1}, have {len(ace.xss_data)}")
        return distribution
    
    distribution.incident_energies = ace.xss_data[idx:idx + n_e]
    if debug:
        logger.debug(f"Incident energies range: [{distribution.incident_energies[0] if n_e > 0 else 'N/A'}, {distribution.incident_energies[-1] if n_e > 0 else 'N/A'}]")
    idx += n_e
    
    # Read the number of multiplier values (NET)
    if idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Index {idx} out of bounds for XSS data with length {len(ace.xss_data)}")
        return distribution
    
    distribution.n_mult_values = int(ace.xss_data[idx])
    if debug:
        logger.debug(f"Number of multiplier values (NET): {distribution.n_mult_values}")
    idx += 1
    net = distribution.n_mult_values
    
    # Read the multiplier tables - a view of xss_data
    distribution.multiplier_tables = []
    
    # Each table has NET values, and there are N_E tables
    for i in range(n_e):
        if idx + net - 1 >= len(ace.xss_data):
            if debug:
                logger.debug(f"Not enough data to read multiplier table {i+1}. Need index up to {idx + net - 1}, have {len(ace.xss_data)}")
            break
        
        table = ace.xss_data[idx:idx + net]
        distribution.multiplier_tables.append(table)
        if debug:
            logger.debug(f"Multiplier table {i+1} range: [{table[0] if net > 0 else 'N/A'}, {table[-1] if net > 0 else 'N/A'}]")
        idx += net
    
    if debug:
        logger.debug(f"Completed parsing tabular energy multipliers with {len(distribution.multiplier_tables)} tables")
    return distribution