import logging
from typing import List, Optional
from kika.ace.classes.ace import Ace
from kika.ace.classes.secondary_particles.secondary_particles_types import SecondaryParticleTypes

# Setup logger
logger = logging.getLogger(__name__)

def parse_ptype_block(ace: Ace, debug=False) -> SecondaryParticleTypes:
    """
    Read the PTYPE block containing secondary particle type identifiers (JXS(30)).
    
    The PTYPE Block gives a list of particle types (neutrons, protons, alphas, etc.)
    that can be produced in nuclear reactions. The ACE file contains production data
    for these particles, including cross sections, angular distributions,
    and energy distributions.
    
    This block is present when NXS(7) is nonzero.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to read from
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    SecondaryParticleTypes
        Object containing the secondary particle type data
    """
    if debug:
        logger.debug("\n===== SECONDARY PARTICLE TYPES (PTYPE) PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize result object
    result = SecondaryParticleTypes()
    
    # Check if we have the necessary base data structures
    if not ace.header or not ace.header.jxs_array or not ace.header.nxs_array or ace.xss_data is None or len(ace.xss_data) == 0:
        if debug:
            logger.debug("Skipping secondary particle types: required base data structures missing")
        return result
    
    # Get the number of particle types (NTYPE = NXS(7))
    # Check if NXS array has at least 8 elements (for NXS(7))
    if len(ace.header.nxs_array) <= 7:
        if debug:
            logger.debug("Skipping secondary particle types: NXS array too short (no NXS(7))")
        return result
    
    ntype = ace.header.nxs_array[7]
    
    if debug:
        logger.debug(f"NXS(7) = {ntype} → Number of secondary particle types")
    
    # According to documentation, PTYPE block is present when NXS(7) is nonzero
    if ntype <= 0:
        if debug:
            logger.debug("No secondary particle types defined (NXS(7) = 0)")
        return result
    
    # Get the starting index for the PTYPE block (LTYPE = JXS(30))
    # Check if JXS array has at least 31 elements (for JXS(30))
    if len(ace.header.jxs_array) <= 30:
        if debug:
            logger.debug("Skipping secondary particle types: JXS array too short (no JXS(30))")
        return result
    
    ltype_idx = ace.header.jxs_array[30]
    
    if debug:
        logger.debug(f"JXS(30) = {ltype_idx} → Locator for secondary particle types")
    
    # Check if the PTYPE block exists in the XSS array
    if ltype_idx <= 0:
        if debug:
            logger.debug("Secondary particle types data not present (JXS(30) <= 0)")
        return result
    
    if ltype_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid location: JXS(30)={ltype_idx} exceeds XSS length {len(ace.xss_data)}")
        return result
    
    if debug:
        logger.debug(f"Secondary particle types data starts at index {ltype_idx}")
    
    # Check if we have enough data for all particle types
    if ltype_idx + ntype - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data: need {ntype} entries, but only {len(ace.xss_data) - ltype_idx} available")
        return result
    
    # Read the particle type identifiers
    for i in range(ntype):
        particle_id_entry = ace.xss_data[ltype_idx + i]
        particle_id = int(particle_id_entry)
        result.particle_ids.append(particle_id)
        
        if debug:
            particle_name = result.get_particle_name(particle_id) if hasattr(result, "get_particle_name") else f"Unknown({particle_id})"
            logger.debug(f"  Type {i+1}: {particle_name.capitalize()} (ID: {particle_id})")
    
    if result.particle_ids:
        result.has_data = True
        if debug:
            logger.debug(f"Successfully read {len(result.particle_ids)} secondary particle types")
    
    return result
