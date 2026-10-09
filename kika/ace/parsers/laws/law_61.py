import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.angle_energy import TabulatedAngleEnergyDistribution

# Setup logger
logger = logging.getLogger(__name__)

class Law61ParseError(Exception):
    """Exception raised for errors in parsing Law 61 data."""
    pass

def parse_tabulated_angle_energy_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> TabulatedAngleEnergyDistribution:
    """
    Parse a tabulated angle-energy distribution (Law 61).
    
    According to Table 45, 46, and 47:
    
    Table 45: LAW=61
    - LDAT(1): N_R (number of interpolation regions)
    - LDAT(2..1+N_R): NBT(l) interpolation parameters
    - LDAT(2+N_R..1+2*N_R): INT(l) interpolation schemes
    - LDAT(2+2*N_R): N_E (number of incident energies)
    - LDAT(3+2*N_R..2+2*N_R+N_E): E(l) (incident energies)
    - LDAT(3+2*N_R+N_E..2+2*N_R+2*N_E): L(l) (distribution locations)
    
    Table 46: Secondary energy distribution for each incident energy
    - INTT': Combined interpolation parameter (10*N_D + INTT)
    - N_p: Number of points in the distribution
    - E_out(l): Outgoing energy grid
    - PDF(l): Probability density function
    - CDF(l): Cumulative density function
    - LC(l): Location of angular distribution tables
    
    Table 47: Angular distribution
    - JJ: Interpolation flag
    - N_p: Number of points
    - Cos_out(j): Cosine scattering angular grid
    - PDF(j): Probability density function
    - CDF(j): Cumulative density function
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    base_dist : EnergyDistribution
        Base distribution with common properties
    idat_idx : int
        Starting index for the law data in the XSS array
    debug : bool, optional
        Enable debug logging if True
        
    Returns
    -------
    TabulatedAngleEnergyDistribution
        Tabulated angle-energy distribution object
        
    Raises
    ------
    Law61ParseError
        If there are issues with data format or indexing
    """
    if debug:
        logger.debug(f"=== PARSING LAW 61 ===")
        logger.debug(f"Starting at index idat_idx={idat_idx}")
    
    # Create a new distribution object using the base properties
    distribution = TabulatedAngleEnergyDistribution(
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
        raise Law61ParseError(f"Index {idat_idx} out of bounds for XSS data with length {len(ace.xss_data)}")
    
    # Read the number of interpolation regions (N_R) - LDAT(1)
    try:
        distribution.n_interp_regions = int(ace.xss_data[idat_idx])
        if debug:
            logger.debug(f"LDAT(1): N_R={distribution.n_interp_regions} (at index {idat_idx})")
    except (IndexError, AttributeError, ValueError) as e:
        raise Law61ParseError(f"Failed to read N_R at index {idat_idx}: {str(e)}")
    
    idx = idat_idx + 1
    n_r = distribution.n_interp_regions
    
    # Read the interpolation parameters if present
    if n_r > 0:
        if idx + 2*n_r - 1 >= len(ace.xss_data):
            raise Law61ParseError(f"Not enough data for interpolation parameters. Need values up to index {idx + 2*n_r - 1}, array length is {len(ace.xss_data)}")
        
        try:
            # Read NBT values - LDAT(2..1+N_R)
            distribution.nbt = [int(ace.xss_data[idx + i]) for i in range(n_r)]
            if debug:
                logger.debug(f"LDAT(2..1+N_R): NBT values: {distribution.nbt} (at indices {idx}..{idx+n_r-1})")
            idx += n_r
            
            # Read INT values - LDAT(2+N_R..1+2*N_R)
            distribution.interp = [int(ace.xss_data[idx + i]) for i in range(n_r)]
            if debug:
                logger.debug(f"LDAT(2+N_R..1+2*N_R): INT values: {distribution.interp} (at indices {idx}..{idx+n_r-1})")
            idx += n_r
        except (IndexError, AttributeError, ValueError) as e:
            raise Law61ParseError(f"Failed to read NBT/INT values at index range {idx-n_r}..{idx+n_r-1}: {str(e)}")
    
    # Read the number of incident energies (N_E) - LDAT(2+2*N_R)
    try:
        if idx >= len(ace.xss_data):
            raise Law61ParseError(f"Index {idx} out of bounds for XSS data when reading N_E")
        
        distribution.n_energies = int(ace.xss_data[idx])
        if debug:
            logger.debug(f"LDAT(2+2*N_R): N_E={distribution.n_energies} (at index {idx})")
    except (IndexError, AttributeError, ValueError) as e:
        raise Law61ParseError(f"Failed to read N_E at index {idx}: {str(e)}")
    
    idx += 1
    n_e = distribution.n_energies
    
    # Check if we have enough data for the energies
    if n_e <= 0:
        raise Law61ParseError(f"Invalid N_E value: {n_e} (must be positive)")
    
    if idx + n_e - 1 >= len(ace.xss_data):
        raise Law61ParseError(f"Not enough data to read incident energies. Need index up to {idx + n_e - 1}, have {len(ace.xss_data)}")
    
    # Read the incident energies - LDAT(3+2*N_R..2+2*N_R+N_E)
    try:
        distribution.incident_energies = ace.xss_data[idx:idx + n_e]
        if debug:
            energy_values = [float(e) for e in distribution.incident_energies]
            logger.debug(f"LDAT(3+2*N_R..2+2*N_R+N_E): Incident energies: {energy_values[:3]}...{energy_values[-3:] if len(energy_values) > 3 else ''}")
            logger.debug(f"  (at indices {idx}..{idx+n_e-1})")
    except (IndexError, AttributeError) as e:
        raise Law61ParseError(f"Failed to read incident energies at index range {idx}..{idx+n_e-1}: {str(e)}")
    
    idx += n_e
    
    # Check if we have enough data for the locations
    if idx + n_e - 1 >= len(ace.xss_data):
        raise Law61ParseError(f"Not enough data to read distribution locations. Need index up to {idx + n_e - 1}, have {len(ace.xss_data)}")
    
    # Read the distribution locations (L values) - LDAT(3+2*N_R+N_E..2+2*N_R+2*N_E)
    try:
        distribution.distribution_locations = [int(ace.xss_data[idx + i]) for i in range(n_e)]
        if debug:
            logger.debug(f"LDAT(3+2*N_R+N_E..2+2*N_R+2*N_E): L values: {distribution.distribution_locations[:3]}...{distribution.distribution_locations[-3:] if len(distribution.distribution_locations) > 3 else ''}")
            logger.debug(f"  (at indices {idx}..{idx+n_e-1})")
    except (IndexError, AttributeError, ValueError) as e:
        raise Law61ParseError(f"Failed to read distribution locations at index range {idx}..{idx+n_e-1}: {str(e)}")
    
    # Initialize the angular tables list
    distribution.angular_tables = []
    
    # Now read each energy distribution and store it
    distribution.distributions = []
    
    # Get the JXS values for the angular distribution blocks
    try:
        jxs_dlw = ace.header.jxs_array[11]  # JXS(11) for neutron reactions
        jxs_dlwp = ace.header.jxs_array[19]  # JXS(19) for photon-producing reactions
        if debug:
            logger.debug(f"JXS values: JXS(11)={jxs_dlw}, JXS(19)={jxs_dlwp}")
    except (IndexError, AttributeError) as e:
        raise Law61ParseError(f"Failed to read JXS values: {str(e)}")
    

    # Base JED (where data for this reaction starts)
    if not hasattr(base_dist, 'jed') or base_dist.jed is None:
        raise Law61ParseError("Missing JED value required for parsing Law 61 distribution")
        
    jed = base_dist.jed
    if debug:
        logger.debug(f"Using JED={jed} for Law 61 calculations")
        logger.debug(f"base_dist.idat={base_dist.idat}, idat_idx={idat_idx}")


#   # Base JED (where data for this reaction starts)
#   jed = None
#   if hasattr(base_dist, 'jxs_dlw') and base_dist.jxs_dlw is not None:
#       jed = base_dist.jxs_dlw
#   elif hasattr(base_dist, 'jxs_dlwp') and base_dist.jxs_dlwp is not None:
#       jed = base_dist.jxs_dlwp
#   else:
#       # If neither is stored in the base_dist, use the idat_idx directly
#       # This assumes idat_idx is already an absolute position
#       jed = idat_idx - base_dist.idat + 1  # Reconstruct JED 
#   
#   if debug:
#       logger.debug(f"Using JED={jed} as base for locator calculations")
#       logger.debug(f"base_dist.idat={base_dist.idat}, idat_idx={idat_idx}")
    
    # Dictionary to store angular tables
    angular_tables_dict = {}  # loc -> table index
    
    # Process each incident energy point
    for i in range(n_e):
        # Get the location of this distribution (L value from Table 45)
        L = distribution.distribution_locations[i]
        if debug:
            logger.debug(f"\nProcessing distribution for incident energy {i+1}/{n_e}")
            logger.debug(f"  Energy value: {distribution.incident_energies[i]} MeV")
            logger.debug(f"  L={L} (distribution location)")
        
        if L <= 0:
            # Skip if location is invalid
            if debug:
                logger.debug(f"  Skipping distribution {i+1}: invalid location (L={L})")
            distribution.distributions.append(None)
            continue
        
        # Per documentation: To access distribution data, use JED + L
        dist_idx = jed + L - 1
        if debug:
            logger.debug(f"  Secondary energy distribution index: JED + L = {jed} + {L} - 1 = {dist_idx}")
            if dist_idx < len(ace.xss_data):
                logger.debug(f"  Value at this index: {ace.xss_data[dist_idx]}")
        
        # Check if we're within bounds
        if dist_idx >= len(ace.xss_data):
            raise Law61ParseError(f"Distribution index {dist_idx} (JED={jed} + L={L}) out of bounds for XSS data with length {len(ace.xss_data)}")
        
        # Read the combined interpolation parameter (INTT') from Table 46
        try:
            intt_prime = int(ace.xss_data[dist_idx])
            # Separate into N_D (number of discrete lines) and INTT (interpolation scheme)
            n_discrete = intt_prime // 10
            intt = intt_prime % 10
            if debug:
                logger.debug(f"  INTT'={intt_prime} (combined parameter): N_D={n_discrete}, INTT={intt}")
        except (IndexError, AttributeError, ValueError) as e:
            raise Law61ParseError(f"Failed to read INTT' at index {dist_idx}: {str(e)}")
        
        # Read the number of points (N_p) from Table 46
        try:
            n_points = int(ace.xss_data[dist_idx + 1])
            if debug:
                logger.debug(f"  N_p={n_points} (number of points in distribution)")
        except (IndexError, AttributeError, ValueError) as e:
            raise Law61ParseError(f"Failed to read N_p at index {dist_idx + 1}: {str(e)}")
        
        # Check if we have enough data for the energy distribution part
        # We need 4 arrays of length n_points (E_out, PDF, CDF, LC)
        if dist_idx + 2 + 4*n_points > len(ace.xss_data):
            raise Law61ParseError(
                f"Not enough data for energy distribution {i+1}. Need index up to {dist_idx + 2 + 4*n_points}, "
                f"have {len(ace.xss_data)}. n_points={n_points}, dist_idx={dist_idx}"
            )
        
        try:
            # Read the outgoing energy grid (E_out) from Table 46
            e_out = ace.xss_data[dist_idx + 2:dist_idx + 2 + n_points]
            
            # Read the probability density function (PDF) from Table 46
            pdf = ace.xss_data[dist_idx + 2 + n_points:dist_idx + 2 + n_points + n_points]
            
            # Read the cumulative density function (CDF) from Table 46
            cdf = ace.xss_data[dist_idx + 2 + 2*n_points:dist_idx + 2 + 2*n_points + n_points]
            
            # Read the angular distribution location (LC) from Table 46
            LC_values = [int(ace.xss_data[dist_idx + 2 + 3*n_points + j]) for j in range(n_points)]
            
            if debug:
                if n_points > 0:
                    logger.debug(f"  E_out range: [{e_out[0]:.5e} - {e_out[-1]:.5e}]")
                    logger.debug(f"  PDF range: [{pdf[0]:.5e} - {pdf[-1]:.5e}]")
                    logger.debug(f"  CDF range: [{cdf[0]:.5e} - {cdf[-1]:.5e}]")
                logger.debug(f"  LC values: {LC_values[:5]}...{LC_values[-5:] if len(LC_values) > 5 else ''}")
        except (IndexError, AttributeError, ValueError) as e:
            raise Law61ParseError(f"Failed to read energy distribution data for incident energy {i+1}: {str(e)}")
        
        # Store the energy distribution
        dist_data = {
            'intt': intt,
            'n_discrete': n_discrete,
            'n_points': n_points,
            'e_out': e_out,
            'pdf': pdf,
            'cdf': cdf,
            'lc': LC_values
        }
        
        distribution.distributions.append(dist_data)
        if debug:
            logger.debug(f"  Successfully stored energy distribution {i+1}")
        




        for j, LC in enumerate(LC_values):
            # Skip if we've already processed this LC
            if LC in angular_tables_dict:
                if debug:
                    logger.debug(f"  Angular distribution for LC={LC} already processed, reusing")
                continue
            
            if debug:
                logger.debug(f"\nProcessing angular distribution for LC={LC} (from outgoing energy point {j+1})")
            
            # Handle negative LC values
            abs_LC = abs(LC)
            
            # Use the same JED index for angular distributions that we used for energy distributions
            # This ensures consistency within the same block (DLW, DLWP, or DLWH)
            ang_dist_idx = jed + abs_LC - 1
            
            if debug:
                logger.debug(f"  Angular dist index: JED + |LC| = {jed} + {abs_LC} - 1 = {ang_dist_idx}")
                if ang_dist_idx < len(ace.xss_data):
                    logger.debug(f"  Value at this index: {ace.xss_data[ang_dist_idx]}")
            
            # Check if we're within bounds with no fallback
            if ang_dist_idx >= len(ace.xss_data):
                raise Law61ParseError(
                    f"Angular distribution index {ang_dist_idx} (JED={jed} + |LC|={abs_LC} - 1) "
                    f"out of bounds for XSS data with length {len(ace.xss_data)}"
                )
            
            try:
                # Read the interpolation flag (JJ) from Table 47
                jj = int(ace.xss_data[ang_dist_idx])
                if debug:
                    logger.debug(f"  JJ={jj} (interpolation flag for angular distribution)")
                    if jj not in [1, 2]:
                        logger.debug(f"  WARNING: Unexpected JJ value {jj}. Valid values are 1 (histogram) or 2 (linear)")
            except (IndexError, AttributeError, ValueError) as e:
                raise Law61ParseError(f"Failed to read JJ at index {ang_dist_idx}: {str(e)}")
            










        # Now read the angular distributions for each LC value
#        for j, LC in enumerate(LC_values):
#            # Skip if we've already processed this LC
#            if LC in angular_tables_dict:
#                if debug:
#                    logger.debug(f"  Angular distribution for LC={LC} already processed, reusing")
#                continue
#            
#            if debug:
#                logger.debug(f"\nProcessing angular distribution for LC={LC} (from outgoing energy point {j+1})")
#            
#            # Handle negative LC values
#            abs_LC = abs(LC)
#            
#            # Per documentation: 
#            # L = JXS(11) + |LC| for neutron reactions
#            # L = JXS(19) + |LC| for photon-producing reactions
#            # First try with neutron reactions (JXS(11))
#            ang_dist_idx = jxs_dlw + abs_LC - 1
#            if debug:
#                logger.debug(f"  Angular dist index (neutron): JXS(11) + |LC| = {jxs_dlw} + {abs_LC} - 1 = {ang_dist_idx}")
#                if ang_dist_idx < len(ace.xss_data):
#                    logger.debug(f"  Value at this index: {ace.xss_data[ang_dist_idx]}")
#            
#            # Check if we're within bounds
#            angular_data_source = "neutron"
#            if ang_dist_idx >= len(ace.xss_data):
#                # Try with photon production (JXS(19)) if neutron reaction fails
#                ang_dist_idx = jxs_dlwp + abs_LC - 1
#                angular_data_source = "photon"
#                if debug:
#                    logger.debug(f"  Angular dist index (photon): JXS(19) + |LC| - 1 = {jxs_dlwp} + {abs_LC} = {ang_dist_idx}")
#                    if ang_dist_idx < len(ace.xss_data):
#                        logger.debug(f"  Value at this index: {ace.xss_data[ang_dist_idx]}")
#                
#                if ang_dist_idx >= len(ace.xss_data):
#                    raise Law61ParseError(
#                        f"Angular distribution index {ang_dist_idx} out of bounds for XSS data. "
#                        f"Tried both neutron (JXS(11)={jxs_dlw} + |LC|={abs_LC}) and photon (JXS(19)={jxs_dlwp} + |LC|={abs_LC})"
#                    )
#            
#            try:
#                # Read the interpolation flag (JJ) from Table 47
#                jj = int(ace.xss_data[ang_dist_idx])
#                if debug:
#                    logger.debug(f"  JJ={jj} (interpolation flag for angular distribution)")
#                    if jj not in [1, 2]:
#                        logger.debug(f"  WARNING: Unexpected JJ value {jj}. Valid values are 1 (histogram) or 2 (linear)")
#            except (IndexError, AttributeError, ValueError) as e:
#                raise Law61ParseError(f"Failed to read JJ at index {ang_dist_idx}: {str(e)}")
            
            try:
                # Read the number of points (N_p) from Table 47
                ang_n_points = int(ace.xss_data[ang_dist_idx + 1])
                if debug:
                    logger.debug(f"  N_p={ang_n_points} (number of points in angular distribution)")
            except (IndexError, AttributeError, ValueError) as e:
                raise Law61ParseError(f"Failed to read angular N_p at index {ang_dist_idx + 1}: {str(e)}")
            
            # Check if we have enough data for the angular distribution
            if ang_dist_idx + 2 + 3*ang_n_points > len(ace.xss_data):
                raise Law61ParseError(
                    f"Not enough data for angular distribution. Need index up to {ang_dist_idx + 2 + 3*ang_n_points}, "
                    f"have {len(ace.xss_data)}. n_points={ang_n_points}, ang_dist_idx={ang_dist_idx}"
                )
            
            try:
                # Read the cosine scattering grid (Cos_out) from Table 47
                cosines = ace.xss_data[ang_dist_idx + 2:ang_dist_idx + 2 + ang_n_points]
                
                # Read the probability density function (PDF) from Table 47
                ang_pdf = ace.xss_data[ang_dist_idx + 2 + ang_n_points:ang_dist_idx + 2 + ang_n_points + ang_n_points]
                
                # Read the cumulative density function (CDF) from Table 47
                ang_cdf = ace.xss_data[ang_dist_idx + 2 + 2*ang_n_points:ang_dist_idx + 2 + 2*ang_n_points + ang_n_points]
                
                if debug:
                    if ang_n_points > 0:
                        logger.debug(f"  Cosine range: [{cosines[0]:.5f} - {cosines[-1]:.5f}]")
                        logger.debug(f"  Angular PDF range: [{ang_pdf[0]:.5e} - {ang_pdf[-1]:.5e}]")
                        logger.debug(f"  Angular CDF range: [{ang_cdf[0]:.5e} - {ang_cdf[-1]:.5e}]")
            except (IndexError, AttributeError) as e:
                raise Law61ParseError(f"Failed to read angular distribution data for LC={LC}: {str(e)}")
            
            # Validate that the cosines are in the expected range [-1, 1]
            if ang_n_points > 0:
                min_cos = min(float(cos) for cos in cosines)
                max_cos = max(float(cos) for cos in cosines)
                if min_cos < -1.001 or max_cos > 1.001:  # Allow slight numerical error
                    if debug:
                        logger.debug(f"  WARNING: Cosine values outside expected range [-1,1]: min={min_cos}, max={max_cos}")
            
            # Store the angular distribution
            ang_data = {
                'jj': jj,
                'n_points': ang_n_points,
                'cosines': cosines,
                'pdf': ang_pdf,
                'cdf': ang_cdf,
            }
            
            # Add to the list and dictionary
            angular_tables_dict[LC] = len(distribution.angular_tables)
            distribution.angular_tables.append(ang_data)
            if debug:
                logger.debug(f"  Successfully stored angular distribution for LC={LC}")
    
    if debug:
        logger.debug(f"\nCompleted parsing Law 61:")
        logger.debug(f"  Parsed {len(distribution.distributions)} energy distributions")
        logger.debug(f"  Parsed {len(distribution.angular_tables)} angular distributions")
    
    return distribution