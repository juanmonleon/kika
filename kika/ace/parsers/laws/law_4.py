import logging
from kika.ace.classes.xss import table_padding, xss_ints
from kika.ace.classes.read_report import note
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.tabular import ContinuousTabularDistribution

# Setup logger
logger = logging.getLogger(__name__)

def parse_continuous_energy_angle_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> ContinuousTabularDistribution:
    """
    Parse a continuous energy-angle distribution (Law 4).
    
    According to Table 35 and 36, LAW=4 contains:
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
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    base_dist : EnergyDistribution
        Base distribution with common properties
    idat_idx : int
        Starting index for the law data in the XSS array
    debug : bool, optional
        Enable debug logging (default is False)
        
    Returns
    -------
    ContinuousTabularDistribution
        Continuous tabular distribution object
    """
    if debug:
        logger.debug(f"Parsing continuous energy-angle distribution (Law 4) starting at index {idat_idx}")
    
    # Create a new distribution object using the base properties
    distribution = ContinuousTabularDistribution(
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
        raise ValueError(f"Index {idat_idx} out of bounds for XSS data with length {len(ace.xss_data)}")
    
    # Read the number of interpolation regions (N_R)
    n_interp_regions_entry = ace.xss_data[idat_idx]
    distribution.n_interp_regions = int(n_interp_regions_entry)
    if debug:
        logger.debug(f"Number of interpolation regions (N_R): {distribution.n_interp_regions}")
    idx = idat_idx + 1
    n_r = distribution.n_interp_regions
    
    # Read the interpolation parameters if present
    if n_r > 0 and idx + 2*n_r - 1 < len(ace.xss_data):
        # Read NBT values
        distribution.nbt = xss_ints(ace.xss_data, idx, n_r)
        if debug:
            logger.debug(f"NBT values: {distribution.nbt}")
        idx += n_r
        
        # Read INT values
        distribution.interp = xss_ints(ace.xss_data, idx, n_r)
        if debug:
            logger.debug(f"INT values: {distribution.interp}")
        idx += n_r
    elif n_r > 0:
        raise ValueError(f"NR={n_r} interpolation regions at {idx} run past the XSS array ({len(ace.xss_data)})")
    
    # Read the number of incident energies (N_E)
    if idx >= len(ace.xss_data):
        raise ValueError(f"Index {idx} out of bounds for XSS data with length {len(ace.xss_data)}")
    
    n_energies_entry = ace.xss_data[idx]
    distribution.n_energies = int(n_energies_entry)
    if debug:
        logger.debug(f"Number of incident energies (N_E): {distribution.n_energies}")
    idx += 1
    n_e = distribution.n_energies
    
    # Check if we have enough data
    if idx + n_e - 1 >= len(ace.xss_data):
        raise ValueError(f"Not enough data to read incident energies. Need index up to {idx + n_e - 1}, have {len(ace.xss_data)}")
    
    # Read the incident energies - a view of xss_data
    distribution.incident_energies = ace.xss_data[idx:idx + n_e]
    if debug and n_e > 0:
        logger.debug(f"Incident energies range: [{distribution.incident_energies[0]}, {distribution.incident_energies[-1]}]")
    idx += n_e
    
    # Check if we have enough data for the locations
    if idx + n_e - 1 >= len(ace.xss_data):
        raise ValueError(f"Not enough data to read distribution locations. Need index up to {idx + n_e - 1}, have {len(ace.xss_data)}")
    
    # Read the distribution locations (L values)
    distribution.distribution_locations = xss_ints(ace.xss_data, idx, n_e)
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
            raise ValueError(f"distribution {i+1}: invalid location ({loc})")
        
        # Convert to absolute index
        dist_idx = jed + loc - 1  # -1 for 0-indexing
        if debug:
            logger.debug(f"Absolute index for distribution {i+1}: {dist_idx}")
        
        # Check if we're within bounds
        if dist_idx >= len(ace.xss_data):
            raise ValueError(f"Absolute index {dist_idx} out of bounds for XSS data with length {len(ace.xss_data)}")
        
        # Read the combined interpolation parameter (INTT')
        intt_prime_entry = ace.xss_data[dist_idx]
        intt_prime = int(intt_prime_entry)
        
        # Separate into N_D (number of discrete lines) and INTT (interpolation scheme)
        n_discrete = intt_prime // 10
        intt = intt_prime % 10
        if debug:
            logger.debug(f"INTT' = {intt_prime}, N_D = {n_discrete}, INTT = {intt}")
        
        # Read the number of points (N_p)
        n_points_entry = ace.xss_data[dist_idx + 1]
        n_points = int(n_points_entry)
        if debug:
            logger.debug(f"Number of points (N_p): {n_points}")
        
        # Check if we have enough data for the full distribution
        if dist_idx + 2 + 3*n_points - 1 >= len(ace.xss_data):
            raise ValueError(f"Not enough data for full distribution. Need index up to {dist_idx + 2 + 3*n_points - 1}, have {len(ace.xss_data)}")
        
        # Read the outgoing energy grid (E_out)
        e_out = ace.xss_data[dist_idx + 2:dist_idx + 2 + n_points]
        if debug and n_points > 0:
            logger.debug(f"E_out range: [{e_out[0]}, {e_out[-1]}]")
        
        # Read the probability density function (PDF)
        pdf = ace.xss_data[dist_idx + 2 + n_points:dist_idx + 2 + 2*n_points]
        if debug and n_points > 0:
            logger.debug(f"PDF range: [{pdf[0]}, {pdf[-1]}]")
        
        # Read the cumulative density function (CDF)
        cdf = ace.xss_data[dist_idx + 2 + 2*n_points:dist_idx + 2 + 3*n_points]
        if debug and n_points > 0:
            logger.debug(f"CDF range: [{cdf[0]}, {cdf[-1]}]")
        
        # Store the distribution
        dist_data = {
            'intt': intt,
            'n_discrete': n_discrete,
            'n_points': n_points,
            'e_out': e_out,
            'pdf': pdf,
            'cdf': cdf
        }
        
        distribution.distributions.append(dist_data)
        if debug:
            logger.debug(f"Successfully stored distribution {i+1}")
    
    if debug:
        logger.debug(f"Completed parsing continuous energy-angle distribution with {len([d for d in distribution.distributions if d is not None])} valid distributions")
    distribution.padding, unread = table_padding(
        ace.xss_data, base_dist.jed, distribution.distribution_locations,
        [2 + 3 * d['n_points'] for d in distribution.distributions])
    if distribution.distributions:
        # JED-relative position just past the last table, for the block reader's padding check
        distribution.data_end = (int(distribution.distribution_locations[-1]) + 2
                                 + 3 * distribution.distributions[-1]['n_points'])
    for table, gap in unread:
        note(ace, None, None, f"LAW={distribution.law}: {gap} words after table {table} are not zeros and "
                               f"no locator points at them; they are not kept")
    return distribution
