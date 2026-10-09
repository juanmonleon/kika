import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.watt import EnergyDependentWattSpectrum

# Setup logger
logger = logging.getLogger(__name__)

def parse_energy_dependent_watt_spectrum(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> EnergyDependentWattSpectrum:
    """
    Parse an energy-dependent Watt spectrum (Law 11).
    
    According to Table 40, LAW=11 contains:
    - First section: parameters for a(E)
      - LDAT(1): N_Ra - Number of interpolation regions for parameter a
      - LDAT(2) to LDAT(1+N_Ra): NBT_a - Interpolation parameters for a
      - LDAT(2+N_Ra) to LDAT(1+2*N_Ra): INT_a - Interpolation scheme for a
      - LDAT(2+2*N_Ra): N_Ea - Number of incident energies for a
      - LDAT(3+2*N_Ra) to LDAT(2+2*N_Ra+N_Ea): E_a - Incident energy table for a
      - LDAT(3+2*N_Ra+N_Ea) to LDAT(2+2*N_Ra+2*N_Ea): a - Parameter a values
    
    - Second section: parameters for b(E)
      - LDAT(L): N_Rb - Number of interpolation regions for parameter b
      - LDAT(L+1) to LDAT(L+N_Rb): NBT_b - Interpolation parameters for b
      - LDAT(L+1+N_Rb) to LDAT(L+1+2*N_Rb): INT_b - Interpolation scheme for b
      - LDAT(L+1+2*N_Rb): N_Eb - Number of incident energies for b
      - LDAT(L+2+2*N_Rb) to LDAT(L+1+2*N_Rb+N_Eb): E_b - Incident energy table for b
      - LDAT(L+2+2*N_Rb+N_Eb) to LDAT(L+1+2*N_Rb+2*N_Eb): b - Parameter b values
    
    - Final value:
      - LDAT(L+2+2*N_Rb+2*N_Eb): U - Restriction energy
    
    where L = 3 + 2 * (N_Ra + N_Ea)
    
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
    EnergyDependentWattSpectrum
        Energy-dependent Watt spectrum object
    """
    if debug:
        logger.debug(f"Parsing energy-dependent Watt spectrum (Law 11) starting at index {idat_idx}")
    
    # Create a new distribution object using the base properties
    distribution = EnergyDependentWattSpectrum(
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
    
    # ---- First section: parameters for a(E) ----
    
    # Read the number of interpolation regions for parameter a (N_Ra)
    n_a_interp_regions_entry = ace.xss_data[idat_idx]
    distribution.n_a_interp_regions = int(n_a_interp_regions_entry)
    n_ra = distribution.n_a_interp_regions
    if debug:
        logger.debug(f"Number of interpolation regions for parameter a (N_Ra): {n_ra}")
    idx = idat_idx + 1
    
    # Read interpolation parameters for a if present
    if n_ra > 0:
        # Read NBT_a values
        if idx + n_ra <= len(ace.xss_data):
            distribution.a_nbt = [int(ace.xss_data[idx + i]) for i in range(n_ra)]
            if debug:
                logger.debug(f"NBT_a values: {distribution.a_nbt}")
            idx += n_ra
        elif debug:
            logger.debug(f"Not enough data to read NBT_a values. Need index up to {idx + n_ra}, have {len(ace.xss_data)}")
        
        # Read INT_a values
        if idx + n_ra <= len(ace.xss_data):
            distribution.a_interp = [int(ace.xss_data[idx + i]) for i in range(n_ra)]
            if debug:
                logger.debug(f"INT_a values: {distribution.a_interp}")
            idx += n_ra
        elif debug:
            logger.debug(f"Not enough data to read INT_a values. Need index up to {idx + n_ra}, have {len(ace.xss_data)}")
    
    # Read the number of incident energies for parameter a (N_Ea)
    if idx < len(ace.xss_data):
        n_a_energies_entry = ace.xss_data[idx]
        distribution.n_a_energies = int(n_a_energies_entry)
        n_ea = distribution.n_a_energies
        if debug:
            logger.debug(f"Number of incident energies for parameter a (N_Ea): {n_ea}")
        idx += 1
    else:
        if debug:
            logger.debug(f"Index {idx} out of bounds for XSS data with length {len(ace.xss_data)}")
        return distribution
    
    # Read the incident energy table for parameter a - a view of xss_data
    if idx + n_ea <= len(ace.xss_data):
        distribution.a_incident_energies = ace.xss_data[idx:idx + n_ea]
        if debug:
            logger.debug(f"Incident energy table for parameter a range: [{distribution.a_incident_energies[0] if n_ea > 0 else 'N/A'}, {distribution.a_incident_energies[-1] if n_ea > 0 else 'N/A'}]")
        idx += n_ea
    else:
        if debug:
            logger.debug(f"Not enough data to read incident energy table for parameter a. Need index up to {idx + n_ea}, have {len(ace.xss_data)}")
        return distribution
    
    # Read the a parameter values - a view of xss_data
    if idx + n_ea <= len(ace.xss_data):
        distribution.a_values = ace.xss_data[idx:idx + n_ea]
        if debug:
            logger.debug(f"Parameter a values range: [{distribution.a_values[0] if n_ea > 0 else 'N/A'}, {distribution.a_values[-1] if n_ea > 0 else 'N/A'}]")
        idx += n_ea
    else:
        if debug:
            logger.debug(f"Not enough data to read parameter a values. Need index up to {idx + n_ea}, have {len(ace.xss_data)}")
        return distribution
    
    # Calculate L = 3 + 2 * (N_Ra + N_Ea)
    l_idx = 3 + 2 * (n_ra + n_ea)
    if debug:
        logger.debug(f"Calculated L index: {l_idx}")
    
    # ---- Second section: parameters for b(E) ----
    
    # Read the number of interpolation regions for parameter b (N_Rb)
    if idx < len(ace.xss_data):
        n_b_interp_regions_entry = ace.xss_data[idx]
        distribution.n_b_interp_regions = int(n_b_interp_regions_entry)
        n_rb = distribution.n_b_interp_regions
        if debug:
            logger.debug(f"Number of interpolation regions for parameter b (N_Rb): {n_rb}")
        idx += 1
    else:
        if debug:
            logger.debug(f"Index {idx} out of bounds for XSS data with length {len(ace.xss_data)}")
        return distribution
    
    # Read interpolation parameters for b if present
    if n_rb > 0:
        # Read NBT_b values
        if idx + n_rb <= len(ace.xss_data):
            distribution.b_nbt = [int(ace.xss_data[idx + i]) for i in range(n_rb)]
            if debug:
                logger.debug(f"NBT_b values: {distribution.b_nbt}")
            idx += n_rb
        elif debug:
            logger.debug(f"Not enough data to read NBT_b values. Need index up to {idx + n_rb}, have {len(ace.xss_data)}")
        
        # Read INT_b values
        if idx + n_rb <= len(ace.xss_data):
            distribution.b_interp = [int(ace.xss_data[idx + i]) for i in range(n_rb)]
            if debug:
                logger.debug(f"INT_b values: {distribution.b_interp}")
            idx += n_rb
        elif debug:
            logger.debug(f"Not enough data to read INT_b values. Need index up to {idx + n_rb}, have {len(ace.xss_data)}")
    
    # Read the number of incident energies for parameter b (N_Eb)
    if idx < len(ace.xss_data):
        n_b_energies_entry = ace.xss_data[idx]
        distribution.n_b_energies = int(n_b_energies_entry)
        n_eb = distribution.n_b_energies
        if debug:
            logger.debug(f"Number of incident energies for parameter b (N_Eb): {n_eb}")
        idx += 1
    else:
        if debug:
            logger.debug(f"Index {idx} out of bounds for XSS data with length {len(ace.xss_data)}")
        return distribution
    
    # Read the incident energy table for parameter b - a view of xss_data
    if idx + n_eb <= len(ace.xss_data):
        distribution.b_incident_energies = ace.xss_data[idx:idx + n_eb]
        if debug:
            logger.debug(f"Incident energy table for parameter b range: [{distribution.b_incident_energies[0] if n_eb > 0 else 'N/A'}, {distribution.b_incident_energies[-1] if n_eb > 0 else 'N/A'}]")
        idx += n_eb
    else:
        if debug:
            logger.debug(f"Not enough data to read incident energy table for parameter b. Need index up to {idx + n_eb}, have {len(ace.xss_data)}")
        return distribution
    
    # Read the b parameter values - a view of xss_data
    if idx + n_eb <= len(ace.xss_data):
        distribution.b_values = ace.xss_data[idx:idx + n_eb]
        if debug:
            logger.debug(f"Parameter b values range: [{distribution.b_values[0] if n_eb > 0 else 'N/A'}, {distribution.b_values[-1] if n_eb > 0 else 'N/A'}]")
        idx += n_eb
    else:
        if debug:
            logger.debug(f"Not enough data to read parameter b values. Need index up to {idx + n_eb}, have {len(ace.xss_data)}")
        return distribution
    
    # ---- Final value: restriction energy ----
    
    # Read the restriction energy (U)
    if idx < len(ace.xss_data):
        distribution.restriction_energy = float(ace.xss_data[idx])
        if debug:
            logger.debug(f"Restriction energy (U): {distribution.restriction_energy}")
    elif debug:
        logger.debug(f"Index {idx} out of bounds for XSS data with length {len(ace.xss_data)}, cannot read restriction energy")
    
    if debug:
        logger.debug(f"Completed parsing energy-dependent Watt spectrum with parameter a ({n_ea} points) and parameter b ({n_eb} points)")
    return distribution