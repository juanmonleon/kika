import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.evaporation import EvaporationSpectrum

# Setup logger
logger = logging.getLogger(__name__)

def parse_evaporation_spectrum(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> EvaporationSpectrum:
    """
    Parse an evaporation spectrum (Law 9).
    
    According to Table 39, LAW=9 contains:
    - LDAT(1): N_R - Number of interpolation regions between temperatures
    - LDAT(2) to LDAT(1+N_R): NBT - Interpolation parameters for temperatures
    - LDAT(2+N_R) to LDAT(1+2*N_R): INT - Interpolation scheme for temperatures
    - LDAT(2+2*N_R): N_E - Number of incident energies tabulated
    - LDAT(3+2*N_R) to LDAT(2+2*N_R+N_E): E(l) - Incident energy table
    - LDAT(3+2*N_R+N_E) to LDAT(2+2*N_R+2*N_E): θ(l) - Effective temperature table
    - LDAT(3+2*N_R+2*N_E): U - Restriction energy
    
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
    EvaporationSpectrum
        Evaporation spectrum object
    """
    if debug:
        logger.debug(f"Parsing evaporation spectrum (Law 9) starting at index {idat_idx}")
    
    # Create a new distribution object using the base properties
    distribution = EvaporationSpectrum(
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
    
    # Read the number of interpolation regions for temperature (N_R)
    n_temp_interp_regions_entry = ace.xss_data[idat_idx]
    distribution.n_temp_interp_regions = int(n_temp_interp_regions_entry)
    n_r = distribution.n_temp_interp_regions
    if debug:
        logger.debug(f"Number of interpolation regions for temperature (N_R): {n_r}")
    idx = idat_idx + 1
    
    # Read interpolation parameters for temperature if present
    if n_r > 0:
        # Read NBT values for temperature interpolation
        if idx + n_r <= len(ace.xss_data):
            distribution.temp_nbt = [int(ace.xss_data[idx + i]) for i in range(n_r)]
            if debug:
                logger.debug(f"Temperature NBT values: {distribution.temp_nbt}")
            idx += n_r
        elif debug:
            logger.debug(f"Not enough data to read temperature NBT values. Need index up to {idx + n_r}, have {len(ace.xss_data)}")
        
        # Read INT values for temperature interpolation
        if idx + n_r <= len(ace.xss_data):
            distribution.temp_interp = [int(ace.xss_data[idx + i]) for i in range(n_r)]
            if debug:
                logger.debug(f"Temperature INT values: {distribution.temp_interp}")
            idx += n_r
        elif debug:
            logger.debug(f"Not enough data to read temperature INT values. Need index up to {idx + n_r}, have {len(ace.xss_data)}")
    
    # Read the number of incident energies (N_E)
    if idx < len(ace.xss_data):
        n_incident_energies_entry = ace.xss_data[idx]
        distribution.n_incident_energies = int(n_incident_energies_entry)
        n_e = distribution.n_incident_energies
        if debug:
            logger.debug(f"Number of incident energies (N_E): {n_e}")
        idx += 1
    else:
        if debug:
            logger.debug(f"Index {idx} out of bounds for XSS data with length {len(ace.xss_data)}")
        return distribution
    
    # Read the incident energy table - a view of xss_data
    if idx + n_e <= len(ace.xss_data):
        distribution.incident_energies = ace.xss_data[idx:idx + n_e]
        if debug:
            logger.debug(f"Incident energy table range: [{distribution.incident_energies[0] if n_e > 0 else 'N/A'}, {distribution.incident_energies[-1] if n_e > 0 else 'N/A'}]")
        idx += n_e
    else:
        if debug:
            logger.debug(f"Not enough data to read incident energy table. Need index up to {idx + n_e}, have {len(ace.xss_data)}")
        return distribution
    
    # Read the temperature table - a view of xss_data
    if idx + n_e <= len(ace.xss_data):
        distribution.temperatures = ace.xss_data[idx:idx + n_e]
        if debug:
            logger.debug(f"Temperature table range: [{distribution.temperatures[0] if n_e > 0 else 'N/A'}, {distribution.temperatures[-1] if n_e > 0 else 'N/A'}]")
        idx += n_e
    else:
        if debug:
            logger.debug(f"Not enough data to read temperature table. Need index up to {idx + n_e}, have {len(ace.xss_data)}")
        return distribution
    
    # Read the restriction energy (U)
    if idx < len(ace.xss_data):
        distribution.restriction_energy = float(ace.xss_data[idx])
        if debug:
            logger.debug(f"Restriction energy (U): {distribution.restriction_energy}")
    elif debug:
        logger.debug(f"Index {idx} out of bounds for XSS data with length {len(ace.xss_data)}, cannot read restriction energy")
    
    if debug:
        logger.debug(f"Completed parsing evaporation spectrum with {n_e} incident energies")
    return distribution