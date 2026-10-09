import logging
from typing import List, Optional
from kika.ace.classes.ace import Ace
from kika.ace.classes.secondary_particles.secondary_particle_data_locators import SecondaryParticleDataLocators

# Setup logger
logger = logging.getLogger(__name__)

def parse_ixs_block(ace: Ace, debug=False) -> SecondaryParticleDataLocators:
    """
    Read the IXS block containing the data locations for each secondary particle type (JXS(32)).
    
    This block contains pointers to where various data for each secondary particle type
    can be found. It includes locations for cross sections, angular distributions,
    energy distributions, and more, specific to each particle type.
    
    This block is present when the PTYPE block is present.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to read from
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    SecondaryParticleDataLocators
        Object containing the location data for each secondary particle type
    """
    if debug:
        logger.debug("\n===== SECONDARY PARTICLE DATA LOCATIONS (IXS) PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize result object
    result = SecondaryParticleDataLocators()
    
    # Check if we have the necessary basic data structures
    if not ace.header or not ace.header.jxs_array or ace.xss_data is None or len(ace.xss_data) == 0:
        if debug:
            logger.debug("Skipping data locations: required base data structures missing")
        return result
    
    # This block should be present if PTYPE block is present
    # First check if PTYPE data is available
    if (not hasattr(ace, "secondary_particle_types") or 
        not ace.secondary_particle_types or 
        not hasattr(ace.secondary_particle_types, "particle_ids") or
        not ace.secondary_particle_types.particle_ids):
        if debug:
            logger.debug("Skipping data locations: no secondary particle types defined")
        return result
    
    # Get the number of particle types
    ntype = ace.secondary_particle_types.num_secondary_particles
    
    if debug:
        logger.debug(f"Number of secondary particle types: {ntype}")
    
    if ntype <= 0:
        if debug:
            logger.debug("No secondary particle types defined, nothing to process")
        return result
    
    # Get the starting index for the IXS block (NEXT = JXS(32))
    # Check if JXS array has at least 33 elements (for JXS(32))
    if len(ace.header.jxs_array) <= 32:
        if debug:
            logger.debug("Skipping data locations: JXS array too short (no JXS(32))")
        return result
    
    next_idx = ace.header.jxs_array[32]
    
    if debug:
        logger.debug(f"JXS(32) = {next_idx} → Locator for data locations")
    
    # Check if the IXS block exists in the XSS array
    if next_idx <= 0:
        if debug:
            logger.debug("Data location info not present (JXS(32) <= 0)")
        return result
    
    if next_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid location: JXS(32)={next_idx} exceeds XSS length {len(ace.xss_data)}")
        return result
    
    if debug:
        logger.debug(f"Data location info starts at index {next_idx}")
    
    # Read the locators for each particle type
    for j in range(1, ntype + 1):
        # Calculate the starting index for this particle type
        # According to documentation: LTYPE = NEXT + 10 × (j - 1)
        ltype = next_idx + 10 * (j - 1)
        
        if debug:
            logger.debug(f"\nProcessing secondary particle type {j}:")
            particle_id = ace.secondary_particle_types.particle_ids[j-1] if j-1 < len(ace.secondary_particle_types.particle_ids) else "?"
            particle_name = ace.secondary_particle_types.get_particle_name(particle_id) if hasattr(ace.secondary_particle_types, "get_particle_name") else f"Type {j}"
            logger.debug(f"  Particle: {particle_name.capitalize()} (ID: {particle_id})")
            logger.debug(f"  Index calculation: {next_idx} + 10*({j}-1) = {ltype}")
        
        # Check if we have enough data for this particle's locators
        if ltype + 9 >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: Not enough data for particle type {j}: need 10 entries, but only {len(ace.xss_data) - ltype} available")
            break  # Not enough data for this particle type
        
        # Create a locator set for this particle type
        locator_set = result.create_locator_set(
            hpd=int(ace.xss_data[ltype]),
            mtrh=int(ace.xss_data[ltype + 1]),
            tyrh=int(ace.xss_data[ltype + 2]),
            lsigh=int(ace.xss_data[ltype + 3]),
            sigh=int(ace.xss_data[ltype + 4]),
            landh=int(ace.xss_data[ltype + 5]),
            andh=int(ace.xss_data[ltype + 6]),
            ldlwh=int(ace.xss_data[ltype + 7]),
            dlwh=int(ace.xss_data[ltype + 8]),
            yh=int(ace.xss_data[ltype + 9])
        )
        
        result.locator_sets.append(locator_set)
        
        if debug:
            logger.debug(f"  Successfully read data locations for {particle_name}")
    
    if result.locator_sets:
        result.has_data = True
        if debug:
            logger.debug(f"Successfully read data locations for {len(result.locator_sets)} particle types")
    
    return result

# Backward compatibility alias
read_production_locators_block = parse_ixs_block