import logging
from typing import List, Optional
from kika.ace.classes.ace import Ace
from kika.ace.classes.secondary_particles.secondary_particle_cross_sections import SecondaryParticleCrossSections, ParticleProductionCrossSection

# Setup logger
logger = logging.getLogger(__name__)

def parse_hpd_block(ace: Ace, debug: bool = False) -> SecondaryParticleCrossSections:
    """
    Read the HPD blocks containing production cross section data for each secondary particle type.
    
    This block contains the total cross section for producing each type of secondary particle,
    and associated heating numbers (energy deposition). For example, it includes
    the cross section for (n,p) reactions, (n,α) reactions, etc.
    
    This block is present when:
    - Secondary particle types are defined (NTYPE ≠ 0), and
    - The PTYPE, NTRO, and IXS blocks are present
    
    Parameters
    ----------
    ace : Ace
        The Ace object to read from
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    SecondaryParticleCrossSections
        Object containing the production cross section data for each secondary particle type
    """
    if debug:
        logger.debug("\n===== SECONDARY PARTICLE PRODUCTION CROSS SECTIONS (HPD) PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
        
    # Initialize result object
    result = SecondaryParticleCrossSections()
    
    # Check if we have the necessary basic data structures
    if not ace.header or not ace.header.jxs_array or ace.xss_data is None or len(ace.xss_data) == 0:
        if debug:
            logger.debug("Skipping production cross sections: required base data structures missing")
        return result
    
    # Check if NXS(7) is nonzero
    if len(ace.header.nxs_array) <= 7 or ace.header.nxs_array[7] <= 0:
        if debug:
            logger.debug("Skipping production cross sections: no secondary particle types defined")
        return result
    
    # Check if PTYPE block is present
    if (not hasattr(ace, "secondary_particle_types") or 
        not ace.secondary_particle_types or 
        not hasattr(ace.secondary_particle_types, "particle_ids") or
        not ace.secondary_particle_types.particle_ids):
        if debug:
            logger.debug("Skipping production cross sections: no secondary particle types defined")
        return result
    
    # Check if NTRO block is present
    if (not hasattr(ace, "secondary_particle_reactions") or 
        not ace.secondary_particle_reactions or 
        not hasattr(ace.secondary_particle_reactions, "reaction_counts") or
        not ace.secondary_particle_reactions.reaction_counts):
        if debug:
            logger.debug("Skipping production cross sections: no reaction count data")
        return result
    
    # Check if IXS block is present
    if (not hasattr(ace, "secondary_particle_data_locations") or 
        not ace.secondary_particle_data_locations or 
        not hasattr(ace.secondary_particle_data_locations, "locator_sets") or
        not ace.secondary_particle_data_locations.locator_sets):
        if debug:
            logger.debug("Skipping production cross sections: no data location information")
        return result
    
    # Get the number of particle types
    ntype = ace.secondary_particle_types.num_secondary_particles
    
    if debug:
        logger.debug(f"Processing {ntype} secondary particle types")
    
    # Process each particle type
    for j in range(1, ntype + 1):
        if debug:
            particle_id = ace.secondary_particle_types.particle_ids[j-1] if j-1 < len(ace.secondary_particle_types.particle_ids) else "?"
            particle_name = ace.secondary_particle_types.get_particle_name(particle_id) if hasattr(ace.secondary_particle_types, "get_particle_name") else f"Type {j}"
            logger.debug(f"Processing: {particle_name.capitalize()} (ID: {particle_id})")
            
        # Get the locator set for this particle
        locator_set = ace.secondary_particle_data_locations.get_locators(j)
        if not locator_set:
            if debug:
                logger.debug(f"No data location info for particle type {j}, skipping")
            continue
        
        # Get the HPD index for this particle
        hpd_idx = locator_set.hpd
        if hpd_idx <= 0:
            if debug:
                logger.debug(f"Invalid cross section location {hpd_idx} for particle type {j}, skipping")
            continue
        
        if debug:
            logger.debug(f"Cross section data starts at index {hpd_idx}")
        
        # Check if HPD index is valid
        if hpd_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"Cross section location {hpd_idx} exceeds XSS length {len(ace.xss_data)}")
            continue
        
        # Check if we have enough data to read the basic parameters
        if hpd_idx + 2 >= len(ace.xss_data):
            if debug:
                logger.debug(f"Not enough data at index {hpd_idx}, skipping")
            continue
        
        # Read the energy grid index (IE)
        ie = int(ace.xss_data[hpd_idx])
        
        # Read the number of consecutive energies (N_E)
        ne = int(ace.xss_data[hpd_idx + 1])
        
        if debug:
            logger.debug(f"Energy grid index: {ie}, Number of energy points: {ne}")
        
        # Check if we have enough data for the cross sections and heating numbers
        if hpd_idx + 2 + 2*ne - 1 >= len(ace.xss_data):
            if debug:
                logger.debug(f"Not enough data for cross section values, skipping")
            continue
        
        # Create a data object for this particle
        particle_data = result.create_cross_section_data(
            energy_grid_index=ie,
            num_energies=ne
        )
        
        # Read the cross section values (view of xss_data)
        particle_data.xs_values = ace.xss_data[hpd_idx + 2:hpd_idx + 2 + ne]
        
        # Read the heating numbers (view of xss_data)
        particle_data.heating_numbers = ace.xss_data[hpd_idx + 2 + ne:hpd_idx + 2 + 2 * ne]
        
        # Add the data to the container
        result.particle_data[j] = particle_data
        
        if debug:
            logger.debug(f"Successfully read cross section data: {ne} energy points")
    
    if result.particle_data:
        result.has_data = True
        if debug:
            num_particles = len(result.particle_data)
            logger.debug(f"Finished reading cross section data for {num_particles} particle types")
    elif debug:
        logger.debug("No cross section data found")
    
    return result

# Backward compatibility alias
read_production_cross_sections_block = parse_hpd_block