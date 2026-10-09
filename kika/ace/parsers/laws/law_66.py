import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.phase_space import NBodyPhaseSpaceDistribution

# Setup logger
logger = logging.getLogger(__name__)

def parse_nbody_phase_space_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> NBodyPhaseSpaceDistribution:
    """
    Parse an N-body phase space distribution (Law 66).
    
    According to Table 48, LAW=66 contains:
    - LDAT(1): NPSX - Number of bodies in the phase space
    - LDAT(2): A_P - Total mass ratio for the NPSX particles
    - LDAT(3): INTT - Interpolation parameter
    - LDAT(4): N_P - Number of points in the distribution
    - LDAT(5) to LDAT(4+N_P): ξ_out(j) - ξ grid (between 0 and 1)
    - LDAT(5+N_P) to LDAT(4+2*N_P): PDF(j) - Probability density function
    - LDAT(5+2*N_P) to LDAT(4+3*N_P): CDF(j) - Cumulative density function
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    base_dist : EnergyDistribution
        Base distribution with common properties
    idat_idx : int
        Starting index for the law data in the XSS array
    debug : bool, optional
        Flag to enable or disable debug logging
        
    Returns
    -------
    NBodyPhaseSpaceDistribution
        N-body phase space distribution object
    """
    if debug:
        logger.debug(f"Parsing N-body phase space distribution (Law 66) starting at index {idat_idx}")
    
    # Create a new distribution object using the base properties
    distribution = NBodyPhaseSpaceDistribution(
        law=base_dist.law,
        idat=base_dist.idat
    )
    
    # Copy applicability data from base_dist
    distribution.applicability_energies = base_dist.applicability_energies
    distribution.applicability_probabilities = base_dist.applicability_probabilities
    distribution.applicability_nbt = base_dist.applicability_nbt
    distribution.applicability_interp = base_dist.applicability_interp
    
    # Check if we have data to parse
    if idat_idx + 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"Index {idat_idx+1} out of bounds for XSS data with length {len(ace.xss_data)}")
        return distribution
    
    # Read NPSX (number of bodies in the phase space)
    distribution.npsx = int(ace.xss_data[idat_idx])
    if debug:
        logger.debug(f"NPSX (number of bodies in the phase space): {distribution.npsx}")
    
    # Read A_P (total mass ratio for the NPSX particles)
    if idat_idx + 2 <= len(ace.xss_data):
        distribution.ap = float(ace.xss_data[idat_idx + 1])
        if debug:
            logger.debug(f"A_P (total mass ratio): {distribution.ap}")
    else:
        if debug:
            logger.debug(f"Index {idat_idx+2} out of bounds for XSS data with length {len(ace.xss_data)}")
    
    # Read INTT (interpolation parameter)
    if idat_idx + 3 <= len(ace.xss_data):
        distribution.intt = int(ace.xss_data[idat_idx + 2])
        if debug:
            logger.debug(f"INTT (interpolation parameter): {distribution.intt}")
    else:
        if debug:
            logger.debug(f"Index {idat_idx+3} out of bounds for XSS data with length {len(ace.xss_data)}")
    
    # Read N_P (number of points in the distribution)
    if idat_idx + 4 <= len(ace.xss_data):
        distribution.n_points = int(ace.xss_data[idat_idx + 3])
        if debug:
            logger.debug(f"N_P (number of points in the distribution): {distribution.n_points}")
    else:
        if debug:
            logger.debug(f"Index {idat_idx+4} out of bounds for XSS data with length {len(ace.xss_data)}")
    
    # Get number of points
    n_p = distribution.n_points
    
    # Read ξ grid - a view of xss_data
    if idat_idx + 4 + n_p <= len(ace.xss_data):
        distribution.xi_grid = ace.xss_data[idat_idx + 4:idat_idx + 4 + n_p]
        if debug:
            logger.debug(f"ξ grid range: [{distribution.xi_grid[0] if n_p > 0 else 'N/A'}, {distribution.xi_grid[-1] if n_p > 0 else 'N/A'}]")
    else:
        if debug:
            logger.debug(f"Not enough data to read ξ grid. Need index up to {idat_idx + 4 + n_p}, have {len(ace.xss_data)}")
    
    # Read PDF - a view of xss_data
    if idat_idx + 4 + n_p + n_p <= len(ace.xss_data):
        distribution.pdf = ace.xss_data[idat_idx + 4 + n_p:idat_idx + 4 + n_p + n_p]
        if debug:
            logger.debug(f"PDF range: [{distribution.pdf[0] if n_p > 0 else 'N/A'}, {distribution.pdf[-1] if n_p > 0 else 'N/A'}]")
    else:
        if debug:
            logger.debug(f"Not enough data to read PDF. Need index up to {idat_idx + 4 + n_p + n_p}, have {len(ace.xss_data)}")
    
    # Read CDF - a view of xss_data
    if idat_idx + 4 + n_p + n_p + n_p <= len(ace.xss_data):
        distribution.cdf = ace.xss_data[idat_idx + 4 + 2*n_p:idat_idx + 4 + 2*n_p + n_p]
        if debug:
            logger.debug(f"CDF range: [{distribution.cdf[0] if n_p > 0 else 'N/A'}, {distribution.cdf[-1] if n_p > 0 else 'N/A'}]")
    else:
        if debug:
            logger.debug(f"Not enough data to read CDF. Need index up to {idat_idx + 4 + n_p + n_p + n_p}, have {len(ace.xss_data)}")
    
    if debug:
        logger.debug(f"Completed parsing N-body phase space distribution with {n_p} points")
    return distribution