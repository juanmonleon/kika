import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.tabular_functions import TabularLinearFunctions

# Setup logger
logger = logging.getLogger(__name__)

def parse_tabular_linear_functions(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> TabularLinearFunctions:
    """
    Parse a tabular linear functions distribution (Law 22).
    
    According to Table 41, LAW=22 contains:
    - LDAT(1): N_R - Number of interpolation regions
    - LDAT(2) to LDAT(1+N_R): NBT - Interpolation parameters
    - LDAT(2+N_R) to LDAT(1+2*N_R): INT - Interpolation schemes
    - LDAT(2+2*N_R): N_E - Number of incident energies
    - LDAT(3+2*N_R) to LDAT(2+2*N_R+N_E): E_in(l) - Incident energy table
    - LDAT(3+2*N_R+N_E) to LDAT(2+2*N_R+2*N_E): LOCE(l) - Locators of E_out tables
    
    For each incident energy E_in(i), the data contains:
    - NF_i: Number of functions for this energy
    - P_ik: Probability for each function
    - T_ik: Origin parameter for each function
    - C_ik: Slope parameter for each function
    
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
    TabularLinearFunctions
        Tabular linear functions distribution object
    """
    if debug:
        logger.debug(f"Parsing tabular linear functions (Law 22) starting at index {idat_idx}")
    
    # Create a new distribution object using the base properties
    distribution = TabularLinearFunctions(
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
    
    # Read the table locators
    if idx + n_e - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"Not enough data to read table locators. Need index up to {idx + n_e - 1}, have {len(ace.xss_data)}")
        return distribution
    
    distribution.table_locators = [int(ace.xss_data[idx + i]) for i in range(n_e)]
    if debug:
        logger.debug(f"Table locators: {distribution.table_locators}")
    idx += n_e
    
    # Initialize function_data list
    distribution.function_data = []
    
    # Read the function data for each incident energy
    base_idx = idat_idx  # Base address for locators
    
    for i in range(n_e):
        locator = distribution.table_locators[i]
        if debug:
            logger.debug(f"Processing function data for incident energy {i+1}/{n_e}, locator={locator}")
        if locator <= 0:
            # Skip if locator is invalid
            if debug:
                logger.debug(f"Skipping function data for incident energy {i+1}: invalid locator ({locator})")
            distribution.function_data.append(None)
            continue
        
        # Calculate absolute index
        func_idx = base_idx + locator - 1  # -1 for 0-indexing
        if debug:
            logger.debug(f"Absolute index for function data: {func_idx}")
        
        # Check if we're within bounds
        if func_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"Absolute index {func_idx} out of bounds for XSS data with length {len(ace.xss_data)}")
            distribution.function_data.append(None)
            continue
        
        # Read number of functions (NF_i)
        nf = int(ace.xss_data[func_idx])
        if debug:
            logger.debug(f"Number of functions (NF): {nf}")
        
        # Check if we have enough data
        if func_idx + 1 + 3*nf - 1 >= len(ace.xss_data):
            if debug:
                logger.debug(f"Not enough data to read function parameters. Need index up to {func_idx + 1 + 3*nf - 1}, have {len(ace.xss_data)}")
            distribution.function_data.append(None)
            continue
        
        # Read probability values (P_ik)
        p_values = ace.xss_data[func_idx + 1:func_idx + 1 + nf]
        if debug:
            logger.debug(f"Probability values range: [{p_values[0] if nf > 0 else 'N/A'}, {p_values[-1] if nf > 0 else 'N/A'}]")
        
        # Read origin parameter values (T_ik)
        t_values = ace.xss_data[func_idx + 1 + nf:func_idx + 1 + nf + nf]
        if debug:
            logger.debug(f"Origin parameter values range: [{t_values[0] if nf > 0 else 'N/A'}, {t_values[-1] if nf > 0 else 'N/A'}]")
        
        # Read slope parameter values (C_ik)
        c_values = ace.xss_data[func_idx + 1 + 2*nf:func_idx + 1 + 2*nf + nf]
        if debug:
            logger.debug(f"Slope parameter values range: [{c_values[0] if nf > 0 else 'N/A'}, {c_values[-1] if nf > 0 else 'N/A'}]")
        
        # Store function data
        func_data = {
            'nf': nf,
            'p': p_values,
            't': t_values,
            'c': c_values
        }
        
        distribution.function_data.append(func_data)
        if debug:
            logger.debug(f"Successfully stored function data for incident energy {i+1}")
    
    if debug:
        logger.debug(f"Completed parsing tabular linear functions with {len([fd for fd in distribution.function_data if fd is not None])} valid function data sets")
    return distribution