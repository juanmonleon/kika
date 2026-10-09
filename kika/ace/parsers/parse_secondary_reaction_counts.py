import logging
from typing import List, Optional
from kika.ace.classes.ace import Ace
from kika.ace.classes.secondary_particles.secondary_particle_reactions import SecondaryParticleReactions

# Setup logger
logger = logging.getLogger(__name__)

def parse_ntro_block(ace: Ace, debug=False) -> SecondaryParticleReactions:
    """
    Read the NTRO block containing the number of reactions for each secondary particle type (JXS(31)).
    
    This block tells us how many nuclear reactions will produce each type of secondary particle
    defined in the PTYPE block. For example, it might tell us there are 20 reactions
    that produce neutrons, 5 that produce protons, etc.
    
    This block is present when the PTYPE block is present.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to read from
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    SecondaryParticleReactions
        Object containing the reaction count data for each secondary particle type
    """
    if debug:
        logger.debug("\n===== SECONDARY PARTICLE REACTION COUNTS (NTRO) PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize result object
    result = SecondaryParticleReactions()
    
    # Check if we have the necessary basic data structures
    if not ace.header or not ace.header.jxs_array or ace.xss_data is None or len(ace.xss_data) == 0:
        if debug:
            logger.debug("Skipping reaction counts: required base data structures missing")
        return result
    
    # This block should be present if PTYPE block is present
    # First check if PTYPE data is available
    if (not hasattr(ace, "secondary_particle_types") or 
        not ace.secondary_particle_types or 
        not hasattr(ace.secondary_particle_types, "particle_ids") or
        not ace.secondary_particle_types.particle_ids):
        if debug:
            logger.debug("Skipping reaction counts: no secondary particle types defined")
        return result
    
    # Get the number of particle types
    ntype = ace.secondary_particle_types.num_secondary_particles
    
    if debug:
        logger.debug(f"Number of secondary particle types: {ntype}")
    
    if ntype <= 0:
        if debug:
            logger.debug("No secondary particle types defined, nothing to process")
        return result
    
    # Get the starting index for the NTRO block (LTYPE = JXS(31))
    # Check if JXS array has at least 32 elements (for JXS(31))
    if len(ace.header.jxs_array) <= 31:
        if debug:
            logger.debug("Skipping reaction counts: JXS array too short (no JXS(31))")
        return result
    
    ltype_idx = ace.header.jxs_array[31]
    
    if debug:
        logger.debug(f"JXS(31) = {ltype_idx} → Locator for reaction counts")
    
    # Check if the NTRO block exists in the XSS array
    if ltype_idx <= 0:
        if debug:
            logger.debug("Reaction count data not present (JXS(31) <= 0)")
        return result
    
    if ltype_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid location: JXS(31)={ltype_idx} exceeds XSS length {len(ace.xss_data)}")
        return result
    
    if debug:
        logger.debug(f"Reaction count data starts at index {ltype_idx}")
    
    # Check if we have enough data for all particle types
    if ltype_idx + ntype - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data: need {ntype} entries, but only {len(ace.xss_data) - ltype_idx} available")
        return result
    
    # Read the number of reactions for each particle type
    for i in range(ntype):
        reaction_count = int(ace.xss_data[ltype_idx + i])
        result.reaction_counts.append(reaction_count)
        
        if debug:
            particle_id = ace.secondary_particle_types.particle_ids[i] if i < len(ace.secondary_particle_types.particle_ids) else "?"
            particle_name = ace.secondary_particle_types.get_particle_name(particle_id) if hasattr(ace.secondary_particle_types, "get_particle_name") else f"Type {i+1}"
            logger.debug(f"  {particle_name.capitalize()}: {reaction_count} reactions")
    
    if result.reaction_counts:
        result.has_data = True
        if debug:
            logger.debug(f"Successfully read reaction counts for {len(result.reaction_counts)} particle types")
    
    return result

# Backward compatibility alias
read_reaction_counts_block = parse_ntro_block