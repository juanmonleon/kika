import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.kalbach_mann import KalbachMannDistribution

# Setup logger
logger = logging.getLogger(__name__)

def parse_kalbach_mann_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> KalbachMannDistribution:
    """
    Parse a Kalbach-Mann correlated energy-angle distribution (Law 44).
    
    According to Table 43 and 44, LAW=44 contains:
    - LDAT(1): N_R (number of interpolation regions)
    - LDAT(2..1+N_R): NBT interpolation parameters
    - LDAT(2+N_R..1+2*N_R): INT interpolation schemes
    - LDAT(2+2*N_R): N_E (number of incident energies)
    - LDAT(3+2*N_R..2+2*N_R+N_E): E(l) (incident energies)
    - LDAT(3+2*N_R+N_E..2+2*N_R+2*N_E): L(l) (distribution locations)
    
    For each incident energy E(i), the distribution contains:
    - INTT': Combined interpolation parameter (10*N_D + INTT)
    - N_p: Number of points in the distribution
    - E_out(l): Outgoing energy grid
    - PDF(l): Probability density function
    - CDF(l): Cumulative density function
    - R(l): Precompound fraction
    - A(l): Angular distribution slope
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    base_dist : EnergyDistribution
        Base distribution with common properties
    idat_idx : int
        Starting index for the law data in the XSS array
    debug : bool, optional
        If True, enables detailed debug logging
        
    Returns
    -------
    KalbachMannDistribution
        Kalbach-Mann distribution object
    """
    if debug:
        logger.debug(f"Parsing Kalbach-Mann distribution (Law 44) starting at index {idat_idx}")
    
    # Create a new distribution object using the base properties
    distribution = KalbachMannDistribution(
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
    
    # Check if we have enough data
    if idx + n_e - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"Not enough data to read incident energies. Need index up to {idx + n_e - 1}, have {len(ace.xss_data)}")
        return distribution
    
    # Read the incident energies - a view of xss_data
    distribution.incident_energies = ace.xss_data[idx:idx + n_e]
    if debug:
        logger.debug(f"Incident energies range: [{distribution.incident_energies[0] if n_e > 0 else 'N/A'}, {distribution.incident_energies[-1] if n_e > 0 else 'N/A'}]")
    idx += n_e
    
    # Check if we have enough data for the locations
    if idx + n_e - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"Not enough data to read distribution locations. Need index up to {idx + n_e - 1}, have {len(ace.xss_data)}")
        return distribution
    
    # Read the distribution locations (L values)
    distribution.distribution_locations = [int(ace.xss_data[idx + i]) for i in range(n_e)]
    if debug:
        logger.debug(f"Distribution locations: {distribution.distribution_locations}")
    
    # Now read each distribution and store it
    distribution.distributions = []
    # L(l) is relative to JED, the start of the DLW/DLWP/DNED block (manual
    # Tables 28, 36, 38, 42), not to IDAT.
    jed = base_dist.jed
    
    for i in range(n_e):
        # Get the location of this distribution
        loc = distribution.distribution_locations[i]
        if debug:
            logger.debug(f"Processing distribution {i+1}/{n_e}, location={loc}")
        if loc <= 0:
            # Skip if location is invalid
            if debug:
                logger.debug(f"Skipping distribution {i+1}: invalid location ({loc})")
            distribution.distributions.append(None)
            continue
        
        # Convert to absolute index
        dist_idx = jed + loc - 1  # -1 for 0-indexing
        if debug:
            logger.debug(f"Absolute index for distribution {i+1}: {dist_idx}")
        
        # Check if we're within bounds
        if dist_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"Absolute index {dist_idx} out of bounds for XSS data with length {len(ace.xss_data)}")
            distribution.distributions.append(None)
            continue
        
        # Read the combined interpolation parameter (INTT')
        intt_prime = int(ace.xss_data[dist_idx])
        if debug:
            logger.debug(f"INTT' (combined interpolation parameter): {intt_prime}")
        
        # Separate into N_D (number of discrete lines) and INTT (interpolation scheme)
        n_discrete = intt_prime // 10
        intt = intt_prime % 10
        if debug:
            logger.debug(f"N_D (number of discrete lines): {n_discrete}, INTT (interpolation scheme): {intt}")
        
        # Read the number of points (N_p)
        n_points = int(ace.xss_data[dist_idx + 1])
        if debug:
            logger.debug(f"N_p (number of points): {n_points}")
        
        # Check if we have enough data for the full distribution
        # We need 5 arrays of length n_points (E_out, PDF, CDF, R, A)
        if dist_idx + 2 + 5*n_points - 1 >= len(ace.xss_data):
            if debug:
                logger.debug(f"Not enough data for full distribution. Need index up to {dist_idx + 2 + 5*n_points - 1}, have {len(ace.xss_data)}")
            distribution.distributions.append(None)
            continue
        
        # Read the outgoing energy grid (E_out)
        e_out = ace.xss_data[dist_idx + 2:dist_idx + 2 + n_points]
        if debug:
            logger.debug(f"E_out range: [{e_out[0] if n_points > 0 else 'N/A'}, {e_out[-1] if n_points > 0 else 'N/A'}]")
        
        # Read the probability density function (PDF)
        pdf = ace.xss_data[dist_idx + 2 + n_points:dist_idx + 2 + n_points + n_points]
        if debug:
            logger.debug(f"PDF range: [{pdf[0] if n_points > 0 else 'N/A'}, {pdf[-1] if n_points > 0 else 'N/A'}]")
        
        # Read the cumulative density function (CDF)
        cdf = ace.xss_data[dist_idx + 2 + 2*n_points:dist_idx + 2 + 2*n_points + n_points]
        if debug:
            logger.debug(f"CDF range: [{cdf[0] if n_points > 0 else 'N/A'}, {cdf[-1] if n_points > 0 else 'N/A'}]")
        
        # Read the precompound fraction (R)
        r_values = ace.xss_data[dist_idx + 2 + 3*n_points:dist_idx + 2 + 3*n_points + n_points]
        if debug:
            logger.debug(f"R (precompound fraction) range: [{r_values[0] if n_points > 0 else 'N/A'}, {r_values[-1] if n_points > 0 else 'N/A'}]")
        
        # Read the angular distribution slope (A)
        a_values = ace.xss_data[dist_idx + 2 + 4*n_points:dist_idx + 2 + 4*n_points + n_points]
        if debug:
            logger.debug(f"A (angular distribution slope) range: [{a_values[0] if n_points > 0 else 'N/A'}, {a_values[-1] if n_points > 0 else 'N/A'}]")
        
        # Store the distribution
        dist_data = {
            'intt': intt,
            'n_discrete': n_discrete,
            'n_points': n_points,
            'e_out': e_out,
            'pdf': pdf,
            'cdf': cdf,
            'r': r_values,
            'a': a_values
        }
        
        distribution.distributions.append(dist_data)
        if debug:
            logger.debug(f"Successfully stored distribution {i+1}")
    
    if debug:
        logger.debug(f"Completed parsing Kalbach-Mann distribution with {len(distribution.distributions)} valid distributions")
    return distribution