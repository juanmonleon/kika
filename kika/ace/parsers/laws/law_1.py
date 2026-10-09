import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.tabular import TabularEnergyDistribution

# Setup logger
logger = logging.getLogger(__name__)

def parse_tabular_energy_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> TabularEnergyDistribution:
    """
    Parse a tabular energy distribution (Law 1).
    
    According to Table 32, LAW=1 represents Tabular Equiprobable Energy Bins (From ENDF Law 1):
    - LDAT(1): Number of interpolation regions between tables of E_out (N_R)
    - LDAT(2) to LDAT(1+N_R): NBT interpolation parameters
    - LDAT(2+N_R) to LDAT(1+2*N_R): INT interpolation schemes
    - LDAT(2+2*N_R): Number of incident energies (N_E)
    - LDAT(3+2*N_R) to LDAT(2+2*N_R+N_E): List of incident energies
    - LDAT(3+2*N_R+N_E): Number of outgoing energies in each table (NET)
    - LDAT(4+2*N_R+N_E) to end: E_out tables, each with NET entries
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    base_dist : EnergyDistribution
        Base distribution with common properties
    idat_idx : int
        Starting index for the law data in the XSS array
    debug : bool, optional
        Enable debug logging
        
    Returns
    -------
    TabularEnergyDistribution
        Tabular energy distribution object
    """
    if debug:
        logger.debug(f"Parsing tabular energy distribution (Law 1) starting at index {idat_idx}")
    
    # Create a new distribution object using the base properties
    distribution = TabularEnergyDistribution(
        law=base_dist.law,
        idat=base_dist.idat
    )
    
    # Copy applicability data from base_dist
    distribution.applicability_energies = base_dist.applicability_energies
    distribution.applicability_probabilities = base_dist.applicability_probabilities
    distribution.nbt = base_dist.nbt
    distribution.interp = base_dist.interp
    
    # Check if we have data to parse
    if idat_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Index {idat_idx} out of bounds for XSS data with length {len(ace.xss_data)}")
        return distribution
    
    # Read the number of interpolation regions (N_R)
    n_r_entry = ace.xss_data[idat_idx]
    n_r = int(n_r_entry)
    if debug:
        logger.debug(f"Number of interpolation regions (N_R): {n_r}")
    idx = idat_idx + 1
    
    # Read the interpolation parameters if present
    e_out_nbt = []
    e_out_int = []
    if n_r > 0:
        # Read NBT values - a view of xss_data
        if idx + n_r - 1 < len(ace.xss_data):
            e_out_nbt = ace.xss_data[idx:idx + n_r]
            if debug:
                nbt_values = [float(entry) for entry in e_out_nbt]
                logger.debug(f"NBT values: {nbt_values}")
            idx += n_r
        
            # Read INT values - a view of xss_data
            if idx + n_r - 1 < len(ace.xss_data):
                e_out_int = ace.xss_data[idx:idx + n_r]
                if debug:
                    int_values = [float(entry) for entry in e_out_int]
                    logger.debug(f"INT values: {int_values}")
                idx += n_r
            else:
                if debug:
                    logger.debug(f"Not enough data to read INT values. Need index up to {idx + n_r - 1}, have {len(ace.xss_data)}")
        else:
            if debug:
                logger.debug(f"Not enough data to read NBT values. Need index up to {idx + n_r - 1}, have {len(ace.xss_data)}")
    
    # Read the number of incident energies (N_E)
    if idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Index {idx} out of bounds for XSS data with length {len(ace.xss_data)}")
        return distribution
    
    n_e_entry = ace.xss_data[idx]
    n_e = int(n_e_entry)
    distribution.n_incident_energies = n_e
    if debug:
        logger.debug(f"Number of incident energies (N_E): {n_e}")
    idx += 1
    
    # Check if we have enough data
    if idx + n_e - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"Not enough data to read incident energies. Need index up to {idx + n_e - 1}, have {len(ace.xss_data)}")
        return distribution
    
    # Read the list of incident energies - a view of xss_data
    incident_energies = ace.xss_data[idx:idx + n_e]
    distribution.incident_energies = incident_energies
    if debug and n_e > 0:
        energy_values = [float(entry) for entry in incident_energies]
        logger.debug(f"Incident energies range: [{energy_values[0]}, {energy_values[-1]}]")
    idx += n_e
    
    # Read the number of outgoing energies (NET) in each table
    if idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Index {idx} out of bounds for XSS data with length {len(ace.xss_data)}")
        return distribution
    
    net_entry = ace.xss_data[idx]
    net = int(net_entry)
    if debug:
        logger.debug(f"Number of outgoing energies (NET): {net}")
    idx += 1
    
    # Read the E_out tables for each incident energy
    # Each table has NET entries
    distribution.distribution_data = []
    
    for i in range(n_e):
        if debug:
            logger.debug(f"Processing E_out table {i+1}/{n_e}")
        if idx + net - 1 >= len(ace.xss_data):
            if debug:
                logger.debug(f"Not enough data to read E_out table {i+1}. Need index up to {idx + net - 1}, have {len(ace.xss_data)}")
            break
        
        # Get the outgoing energy boundaries for this incident energy
        e_out = ace.xss_data[idx:idx + net]
        if debug and net > 0:
            e_out_values = [float(entry) for entry in e_out]
            logger.debug(f"E_out table {i+1} range: [{e_out_values[0]}, {e_out_values[-1]}]")
        idx += net
        
        # Calculate the probability (uniform across bins)
        p_out = [1.0 / (net - 1)] * (net - 1) + [0.0]  # NET-1 equal probability bins
        
        # Store with interpolation scheme (always linear-linear for equiprobable bins)
        distribution.distribution_data.append((net, 2, e_out, p_out))
        if debug:
            logger.debug(f"Successfully stored E_out table {i+1}")
    
    if debug:
        logger.debug(f"Completed parsing tabular energy distribution with {len(distribution.distribution_data)} E_out tables")
    return distribution