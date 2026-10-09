import logging
from typing import List, Optional, Tuple
from kika.ace.classes.ace import Ace
from kika.ace.classes.yield_multipliers import PhotonYieldMultipliers, SecondaryParticleYieldMultipliers

# Setup logger
logger = logging.getLogger(__name__)

def read_yield_multiplier_blocks(ace: Ace, debug=False) -> Tuple[PhotonYieldMultipliers, SecondaryParticleYieldMultipliers]:
    """
    Read the YP and YH blocks containing yield multiplier MT numbers.
    
    The YP and YH blocks contain a list of MT identifiers of cross sections used 
    as yield multipliers to calculate photon and secondary particle production 
    cross sections, respectively.
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the data to read
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    Tuple[PhotonYieldMultipliers, SecondaryParticleYieldMultipliers]
        A tuple containing the photon and secondary particle yield multipliers
    """
    if debug:
        logger.debug("\n===== YIELD MULTIPLIER BLOCKS PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize containers
    photon_yield_multipliers = PhotonYieldMultipliers()
    particle_yield_multipliers = SecondaryParticleYieldMultipliers()
    
    # Read photon yield multipliers (YP block)
    photon_yield_multipliers = read_photon_yield_multipliers(ace, debug)
    
    # Read secondary particle yield multipliers (YH block)
    particle_yield_multipliers = read_secondary_particle_yield_multipliers(ace, debug)
    
    return photon_yield_multipliers, particle_yield_multipliers

def read_photon_yield_multipliers(ace: Ace, debug=False) -> PhotonYieldMultipliers:
    """
    Read photon yield multiplier MT numbers (YP block).
    
    For the YP Block (Table 60):
    - LY = JXS(20) (manual Table 55)
    - NYP is at location LY
    - MTY values are at locations LY+1 through LY+NYP
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the data to read
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    PhotonYieldMultipliers
        Object containing the photon yield multiplier data
    """
    # Initialize result
    result = PhotonYieldMultipliers()
    
    # Check if we have the necessary data
    if not ace.header or not ace.header.nxs_array or ace.xss_data is None or len(ace.xss_data) == 0:
        if debug:
            logger.debug("Skipping YP block: required data missing")
        return result
    
    if debug:
        logger.debug("\n----- PHOTON YIELD MULTIPLIERS (YP BLOCK) -----")
    
    # YP starts at JXS(20) (manual Table 55); NXS(6) is the number of
    # photon-production reactions, not a locator
    ly = ace.header.jxs_array[20]
    
    if debug:
        logger.debug(f"JXS(20) = {ly} → Location of YP block (LY)")
    
    # This block is only present if LY is nonzero
    if ly <= 0:
        if debug:
            logger.debug(f"No YP block present: JXS(20)={ly} ≤ 0")
        return result
    
    if ly >= len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid YP block location: LY={ly} exceeds XSS length {len(ace.xss_data)}")
        return result
    
    # Read NYP (number of MTs to follow)
    nyp_entry = ace.xss_data[ly]
    nyp = int(nyp_entry)
    
    if debug:
        logger.debug(f"NYP = {nyp} → Number of yield multiplier MT numbers")
    
    if nyp <= 0:
        if debug:
            logger.debug(f"No yield multiplier MT numbers: NYP={nyp} ≤ 0")
        return result
    
    # Check if we have enough data for all MT numbers
    if ly + nyp >= len(ace.xss_data):
        if debug:
            logger.debug(f"Not enough data: need {nyp} entries, but only {len(ace.xss_data) - ly - 1} available")
        return result
    
    # Read the MT numbers (MTY values)
    if debug:
        logger.debug(f"Reading {nyp} MT numbers from indices {ly+1} to {ly+nyp}")
    
    for i in range(nyp):
        mt_entry = ace.xss_data[ly + 1 + i]  # Skip the first value (which is NYP)
        mt = int(mt_entry)
        result.multiplier_mts.append(mt)
        
        if debug:
            logger.debug(f"  MTY[{i+1}] = {mt}")
    
    if result.multiplier_mts:
        result.has_data = True
        if debug:
            logger.debug(f"Successfully read {len(result.multiplier_mts)} photon yield multiplier MT numbers")
    
    return result

def read_secondary_particle_yield_multipliers(ace: Ace, debug=False) -> SecondaryParticleYieldMultipliers:
    """
    Read secondary particle yield multiplier MT numbers (YH block).
    
    For the YH Block:
    - For each particle type i (1 to NTYPE):
      - LY = XSS(JXS(32) + 10*(i-1) + 9), the YH word of the IXS block
      - LY = JED
      - NYH is at location LY
      - MTY values are at locations LY+1 through LY+NYH
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the data to read
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    SecondaryParticleYieldMultipliers
        Object containing the secondary particle yield multiplier data
    """
    # Initialize result
    result = SecondaryParticleYieldMultipliers()
    
    # Check if we have the necessary data
    if not ace.header or not ace.header.jxs_array or ace.xss_data is None or len(ace.xss_data) == 0:
        if debug:
            logger.debug("Skipping YH block: required data missing")
        return result
    
    if debug:
        logger.debug("\n----- SECONDARY PARTICLE YIELD MULTIPLIERS (YH BLOCK) -----")
    
    # Improved check for secondary particle types data
    # First check for secondary_particles which is the primary attribute
    secondary_particles = getattr(ace, "secondary_particles", None)
    
    # If not found or no data, try the alternate attribute name
    if (secondary_particles is None or 
        not hasattr(secondary_particles, "particle_ids") or 
        not secondary_particles.particle_ids):
        secondary_particles = getattr(ace, "secondary_particle_types", None)
    
    # Final check if we have usable data
    if (secondary_particles is None or 
        not hasattr(secondary_particles, "particle_ids") or 
        not secondary_particles.particle_ids):
        if debug:
            logger.debug("No secondary particle types data available")
        return result
    
    # Get the number of particle types (NTYPE)
    n_types = len(secondary_particles.particle_ids)
    
    if debug:
        logger.debug(f"Number of secondary particle types (NTYPE): {n_types}")
    
    if n_types <= 0:
        if debug:
            logger.debug("No secondary particle types defined, nothing to process")
        return result
    
    # Get JXS(32) index for the IXS block
    if len(ace.header.jxs_array) <= 32:
        if debug:
            logger.debug("Skipping YH block: JXS array too short (no JXS(32))")
        return result
    
    jxs32_idx = ace.header.jxs_array[32]  # JXS(32) - Index for IXS block
    
    if debug:
        logger.debug(f"JXS(32) = {jxs32_idx} → Location of IXS block")
    
    if jxs32_idx <= 0:
        if debug:
            logger.debug(f"No IXS block present: JXS(32)={jxs32_idx} ≤ 0")
        return result
    
    if jxs32_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid IXS block location: JXS(32)={jxs32_idx} exceeds XSS length {len(ace.xss_data)}")
        return result
    
    # Iterate through each particle type (1 to NTYPE)
    for i in range(1, n_types + 1):
        if debug:
            particle_id = secondary_particles.particle_ids[i-1] if i-1 < len(secondary_particles.particle_ids) else "?"
            particle_name = secondary_particles.get_particle_name(particle_id) if hasattr(secondary_particles, "get_particle_name") else f"Type {i}"
            logger.debug(f"\nProcessing particle type {i}: {particle_name} (ID: {particle_id})")
        
        # YH is the 10th IXS word of the particle (HPD, MTRH, TYRH, LSIGH, SIGH,
        # LANDH, ANDH, LDLWH, DLWH, YH); the 9th, +8, is DLWH
        jed_idx = jxs32_idx + 10 * (i - 1) + 9
        
        if debug:
            logger.debug(f"  YH index calculation: JXS(32) + 10*(i-1) + 9 = {jxs32_idx} + 10*({i}-1) + 9 = {jed_idx}")
        
        if jed_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: JED index {jed_idx} exceeds XSS length {len(ace.xss_data)}")
            continue
        
        # Get the value at JED, which is LY
        ly_entry = ace.xss_data[jed_idx]
        ly = int(ly_entry)
        
        if debug:
            logger.debug(f"  LY = {ly} → Location of yield multiplier data")
        
        # This block is only present if LY is nonzero
        if ly <= 0:
            if debug:
                logger.debug(f"  No yield multiplier data for this particle: LY={ly} ≤ 0")
            continue
        
        if ly >= len(ace.xss_data):
            if debug:
                logger.debug(f"  Invalid YH location: LY={ly} exceeds XSS length {len(ace.xss_data)}")
            continue
        
        # Read NYH (number of MTs to follow)
        nyh_entry = ace.xss_data[ly]
        nyh = int(nyh_entry)
        
        if debug:
            logger.debug(f"  NYH = {nyh} → Number of yield multiplier MT numbers")
        
        if nyh <= 0:
            if debug:
                logger.debug(f"  No yield multiplier MT numbers: NYH={nyh} ≤ 0")
            continue
        
        # Check if we have enough data for all MT numbers
        if ly + nyh >= len(ace.xss_data):
            if debug:
                logger.debug(f"  Not enough data: need {nyh} entries, but only {len(ace.xss_data) - ly - 1} available")
            continue
        
        # Read the MT numbers (MTY values)
        if debug:
            logger.debug(f"  Reading {nyh} MT numbers from indices {ly+1} to {ly+nyh}")
        
        particle_mts = []
        for j in range(nyh):
            mt_entry = ace.xss_data[ly + 1 + j]  # Skip the first value (which is NYH)
            mt = int(mt_entry)
            particle_mts.append(mt)
            
            if debug:
                logger.debug(f"    MTY[{j+1}] = {mt}")
        
        if particle_mts:
            result.particle_multipliers[i] = particle_mts
            result.multiplier_mts.extend(particle_mts)  # Add to overall list as well
            result.has_data = True
            
            if debug:
                logger.debug(f"  Successfully read {len(particle_mts)} yield multiplier MT numbers for particle type {i}")
    
    return result
