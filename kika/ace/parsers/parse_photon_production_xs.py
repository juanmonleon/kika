import logging
from typing import List, Optional, Tuple

import numpy as np
from kika.ace.classes.ace import Ace
from kika.ace.classes.photon_production_xs import (
    PhotonProductionCrossSections, ParticleProductionCrossSections,
    YieldBasedCrossSection, DirectCrossSection
)

# Setup logger
logger = logging.getLogger(__name__)

def read_production_xs_blocks(ace: Ace, debug=False) -> Tuple[PhotonProductionCrossSections, ParticleProductionCrossSections]:
    """
    Read the SIGP and SIGH blocks containing production cross section data.
    
    According to the ACE format documentation:
    - The SIGP block contains photon production cross section data
    - The SIGH block contains particle production cross section data
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the data to read
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    Tuple[PhotonProductionCrossSections, ParticleProductionCrossSections]
        A tuple containing the photon and particle production cross sections
    """
    if debug:
        logger.debug("\n===== PRODUCTION CROSS SECTION BLOCKS PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize containers
    photon_production_xs = PhotonProductionCrossSections()
    particle_production_xs = ParticleProductionCrossSections()
    
    # Read photon production cross sections (SIGP block)
    photon_production_xs = read_photon_production_xs(ace, debug)
    
    # Read particle production cross sections (SIGH block)
    particle_production_xs = read_particle_production_xs(ace, debug)
    
    return photon_production_xs, particle_production_xs

def read_photon_production_xs(ace: Ace, debug=False) -> PhotonProductionCrossSections:
    """
    Read photon production cross section data (SIGP block).
    
    According to Table 56:
    - SIG = JXS(15)
    - NMT = NXS(6)
    
    The cross section data starts at SIG + LOCA_i - 1 for each reaction i.
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the data to read
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    PhotonProductionCrossSections
        Object containing the photon production cross section data
    """
    # Initialize result
    result = PhotonProductionCrossSections()
    
    # Check if we have the necessary data
    if (not ace.header or not ace.header.jxs_array or not ace.header.nxs_array or 
        ace.xss_data is None or len(ace.xss_data) == 0):
        if debug:
            logger.debug("Skipping SIGP block: required data missing")
        return result
    
    # Check if reaction MT data is available
    if (not hasattr(ace, "reaction_mt_data") or not ace.reaction_mt_data or
        not hasattr(ace.reaction_mt_data, "photon_production") or 
        len(ace.reaction_mt_data.photon_production) == 0):
        if debug:
            logger.debug("Skipping SIGP block: no photon production MT data available")
        return result
    
    if debug:
        logger.debug("\n----- PHOTON PRODUCTION CROSS SECTIONS (SIGP BLOCK) -----")
    
    # Get block parameters from Table 56
    # Check if JXS array has the necessary index
    if len(ace.header.jxs_array) <= 15:
        if debug:
            logger.debug("Skipping SIGP block: JXS array too short (no JXS(15))")
        return result
    
    if len(ace.header.nxs_array) <= 6:
        if debug:
            logger.debug("Skipping SIGP block: NXS array too short (no NXS(6))")
        return result
    
    sigp_idx = ace.header.jxs_array[15]  # JXS(15) - Starting index of SIGP block (SIG)
    nmt = ace.header.nxs_array[6]  # NXS(6) - Number of photon production reactions (NMT)
    
    if debug:
        logger.debug(f"JXS(15) = {sigp_idx} → Starting index of SIGP block (SIG)")
        logger.debug(f"NXS(6) = {nmt} → Number of photon production reactions (NMT)")
    
    if sigp_idx <= 0 or nmt <= 0:
        if debug:
            logger.debug(f"No SIGP block present: JXS(15)={sigp_idx}, NXS(6)={nmt}")
        return result
    
    if sigp_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid SIGP block location: JXS(15)={sigp_idx} exceeds XSS length {len(ace.xss_data)}")
        return result
    
    # Check if cross section locators are available
    if (not hasattr(ace, "xs_locators") or not ace.xs_locators or 
        not hasattr(ace.xs_locators, "photon_production") or len(ace.xs_locators.photon_production) == 0):
        if debug:
            logger.debug("Skipping SIGP block: no photon production locators available")
        return result
    
    # Get the MT numbers and locators
    mts = ace.reaction_mt_data.photon_production
    locators = ace.xs_locators.photon_production
    
    if len(mts) == 0 or len(locators) == 0 or len(mts) != len(locators):
        if debug:
            logger.debug(f"Invalid data: mts={len(mts) if mts else 0}, locators={len(locators) if locators else 0}")
        return result
    
    if debug:
        logger.debug(f"Processing {len(mts)} photon production reactions")
    
    # Process each reaction
    for i, (mt, loc) in enumerate(zip(mts, locators)):
        mt_value = int(mt)
        loc_value = int(loc)
        
        if debug:
            logger.debug(f"\nReaction {i+1}: MT={mt_value}")
            logger.debug(f"  LOCA = {loc_value}")
        
        # Calculate the index where the cross section data starts
        # According to Table 57: SIG + LOCA_i - 1
        xs_start_idx = sigp_idx + loc_value - 1
        
        if debug:
            logger.debug(f"  Cross section data starts at SIG + LOCA - 1 = {sigp_idx} + {loc_value} - 1 = {xs_start_idx}")
        
        # Check bounds
        if xs_start_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: Index {xs_start_idx} exceeds XSS length {len(ace.xss_data)}")
            continue
        
        # Read the cross section data based on MFTYPE
        mftype_entry = ace.xss_data[xs_start_idx]
        mftype = int(mftype_entry)
        
        if debug:
            logger.debug(f"  MFTYPE = {mftype}")
        
        if mftype in (12, 16):
            if debug:
                logger.debug(f"  Processing yield-based cross section (MFTYPE={mftype})")
            xs = read_yield_based_xs(ace.xss_data, xs_start_idx, mt_value, mftype, debug)
            if xs:
                result.cross_sections[mt_value] = xs
                result.has_data = True
                if debug:
                    logger.debug(f"  Successfully read yield-based cross section for MT={mt_value}")
        elif mftype == 13:
            if debug:
                logger.debug("  Processing direct cross section (MFTYPE=13)")
            xs = read_direct_xs(ace.xss_data, xs_start_idx, mt_value, debug)
            if xs:
                result.cross_sections[mt_value] = xs
                result.has_data = True
                if debug:
                    logger.debug(f"  Successfully read direct cross section for MT={mt_value}")
        else:
            if debug:
                logger.debug(f"  Unsupported MFTYPE: {mftype}")
    
    return result

def read_particle_production_xs(ace: Ace, debug=False) -> ParticleProductionCrossSections:
    """
    Read particle production cross section data (SIGH block).
    
    According to Table 56:
    - For each particle type i (1 to NTYPE):
      - SIG = XSS(JXS(32) + 10*(i-1) + 4)
      - NMT = XSS(JXS(31) + i - 1)
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the data to read
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    ParticleProductionCrossSections
        Object containing the particle production cross section data
    """
    # Initialize result
    result = ParticleProductionCrossSections()
    
    # Check if we have the necessary data
    if not ace.header or not ace.header.jxs_array or ace.xss_data is None or len(ace.xss_data) == 0:
        if debug:
            logger.debug("Skipping SIGH block: required data missing")
        return result
    
    # Enhanced check for secondary particle types
    has_secondary_particles = False
    
    # First try the primary attribute name
    if hasattr(ace, "secondary_particles") and ace.secondary_particles is not None:
        if hasattr(ace.secondary_particles, "particle_ids") and ace.secondary_particles.particle_ids:
            has_secondary_particles = True
            secondary_particles = ace.secondary_particles
    
    # If not found, try the alternate attribute name
    if not has_secondary_particles and hasattr(ace, "secondary_particle_types") and ace.secondary_particle_types is not None:
        if hasattr(ace.secondary_particle_types, "particle_ids") and ace.secondary_particle_types.particle_ids:
            has_secondary_particles = True
            secondary_particles = ace.secondary_particle_types
    
    if not has_secondary_particles:
        if debug:
            logger.debug("Skipping SIGH block: no secondary particle types data available")
            if hasattr(ace, "secondary_particles"):
                logger.debug(f"  secondary_particles exists: {ace.secondary_particles is not None}")
                if ace.secondary_particles is not None:
                    logger.debug(f"  has particle_ids attribute: {hasattr(ace.secondary_particles, 'particle_ids')}")
                    if hasattr(ace.secondary_particles, "particle_ids"):
                        logger.debug(f"  particle_ids not empty: {bool(ace.secondary_particles.particle_ids)}")
        return result
    
    if debug:
        logger.debug("\n----- PARTICLE PRODUCTION CROSS SECTIONS (SIGH BLOCK) -----")
    
    # Get the number of particle types (NTYPE)
    n_types = len(secondary_particles.particle_ids)
    
    if debug:
        logger.debug(f"Number of secondary particle types (NTYPE): {n_types}")
    
    if n_types <= 0:
        if debug:
            logger.debug("No secondary particle types defined, nothing to process")
        return result
    
    # Get JXS(31) and JXS(32) indices
    if len(ace.header.jxs_array) <= 32:
        if debug:
            logger.debug("Skipping SIGH block: JXS array too short (no JXS(31) or JXS(32))")
        return result
    
    jxs31_idx = ace.header.jxs_array[31]  # JXS(31) - Index for NTRO block
    jxs32_idx = ace.header.jxs_array[32]  # JXS(32) - Index for IXS block
    
    if debug:
        logger.debug(f"JXS(31) = {jxs31_idx} → Location of NTRO block")
        logger.debug(f"JXS(32) = {jxs32_idx} → Location of IXS block")
    
    if jxs31_idx <= 0 or jxs32_idx <= 0:
        if debug:
            logger.debug(f"No NTRO or IXS block present: JXS(31)={jxs31_idx}, JXS(32)={jxs32_idx}")
        return result
    
    if jxs31_idx >= len(ace.xss_data) or jxs32_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid block locations: JXS(31)={jxs31_idx}, JXS(32)={jxs32_idx} exceed XSS length {len(ace.xss_data)}")
        return result
    
    # Check if reaction MT data and cross section locators are available
    if (not hasattr(ace, "reaction_mt_data") or not ace.reaction_mt_data or
        not hasattr(ace.reaction_mt_data, "particle_production") or 
        not ace.reaction_mt_data.particle_production):
        if debug:
            logger.debug("Skipping SIGH block: no particle production MT data available")
        return result
    
    if (not hasattr(ace, "xs_locators") or not ace.xs_locators or
        not hasattr(ace.xs_locators, "particle_production") or 
        not ace.xs_locators.particle_production):
        if debug:
            logger.debug("Skipping SIGH block: no particle production locators available")
        return result
    
    # Process each particle type (1 to NTYPE)
    for i in range(1, n_types + 1):
        if debug:
            particle_id = secondary_particles.particle_ids[i-1] if i-1 < len(secondary_particles.particle_ids) else "?"
            particle_name = secondary_particles.get_particle_name(particle_id) if hasattr(secondary_particles, "get_particle_name") else f"Type {i}"
            logger.debug(f"\nProcessing particle type {i}: {particle_name} (ID: {particle_id})")
        
        # Calculate indices according to Table 56
        
        # NMT = XSS(JXS(31) + i - 1)
        nmt_idx = jxs31_idx + i - 1
        
        if debug:
            logger.debug(f"  NMT index: JXS(31) + i - 1 = {jxs31_idx} + {i} - 1 = {nmt_idx}")
        
        if nmt_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: NMT index {nmt_idx} exceeds XSS length {len(ace.xss_data)}")
            continue
        
        nmt_entry = ace.xss_data[nmt_idx]
        nmt = int(nmt_entry)
        
        if debug:
            logger.debug(f"  NMT = {nmt} → Number of reactions for this particle")
        
        # SIG = XSS(JXS(32) + 10*(i-1) + 4)
        sig_idx_loc = jxs32_idx + 10 * (i - 1) + 4
        
        if debug:
            logger.debug(f"  SIG index location: JXS(32) + 10*(i-1) + 4 = {jxs32_idx} + 10*({i}-1) + 4 = {sig_idx_loc}")
        
        if sig_idx_loc >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: SIG index location {sig_idx_loc} exceeds XSS length {len(ace.xss_data)}")
            continue
        
        sig_idx_entry = ace.xss_data[sig_idx_loc]
        sig_idx = int(sig_idx_entry)
        
        if debug:
            logger.debug(f"  SIG = {sig_idx} → Starting index for cross section data")
        
        # Check bounds
        if sig_idx <= 0:
            if debug:
                logger.debug(f"  No cross section data: SIG={sig_idx} ≤ 0")
            continue
            
        if sig_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  Invalid SIG location: {sig_idx} exceeds XSS length {len(ace.xss_data)}")
            continue
        
        # Get MT numbers and locators for this particle type
        # Check array bounds
        if i > len(ace.reaction_mt_data.particle_production):
            if debug:
                logger.debug(f"  No MT numbers available for particle type {i}")
            continue
            
        if i > len(ace.xs_locators.particle_production):
            if debug:
                logger.debug(f"  No XS locators available for particle type {i}")
            continue
        
        mts = ace.reaction_mt_data.particle_production[i-1]
        locators = ace.xs_locators.particle_production[i-1]
        
        if len(mts) == 0 or len(locators) == 0 or len(mts) != len(locators):
            if debug:
                logger.debug(f"  Invalid data: mts={len(mts) if mts else 0}, locators={len(locators) if locators else 0}")
            continue
        
        if debug:
            logger.debug(f"  Processing {len(mts)} reactions for particle type {i}")
        
        # Store the MT numbers for this particle type
        result.particle_types[i] = [int(mt) for mt in mts]
        
        # Process each reaction for this particle type
        for j, (mt, loc) in enumerate(zip(mts, locators)):
            mt_value = int(mt)
            loc_value = int(loc)
            
            if debug:
                logger.debug(f"\n  Reaction {j+1}: MT={mt_value}")
                logger.debug(f"    LOCA = {loc_value}")
            
            # Calculate the index where the cross section data starts
            # According to Table 57: SIG + LOCA_j - 1
            xs_start_idx = sig_idx + loc_value - 1
            
            if debug:
                logger.debug(f"    Cross section data starts at SIG + LOCA - 1 = {sig_idx} + {loc_value} - 1 = {xs_start_idx}")
            
            # Check bounds
            if xs_start_idx >= len(ace.xss_data):
                if debug:
                    logger.debug(f"    ERROR: Index {xs_start_idx} exceeds XSS length {len(ace.xss_data)}")
                continue
            
            # Read the cross section data based on MFTYPE
            mftype_entry = ace.xss_data[xs_start_idx]
            mftype = int(mftype_entry)
            
            if debug:
                logger.debug(f"    MFTYPE = {mftype}")
            
            if mftype in (12, 16):
                if debug:
                    logger.debug(f"    Processing yield-based cross section (MFTYPE={mftype})")
                xs = read_yield_based_xs(ace.xss_data, xs_start_idx, mt_value, mftype, debug)
                if xs:
                    # Keyed by particle type too: the same MT (e.g. MT5)
                    # produces several particles
                    result.particle_cross_sections.setdefault(i, {})[mt_value] = xs
                    result.has_data = True
                    if debug:
                        logger.debug(f"    Successfully read yield-based cross section for MT={mt_value}")
            else:
                if debug:
                    logger.debug(f"    Unsupported MFTYPE: {mftype}")
    
    # Add this check at the end to ensure consistency
    if result.particle_cross_sections:
        result.has_data = True
        if debug:
            n_xs = sum(len(v) for v in result.particle_cross_sections.values())
            logger.debug(f"Successfully processed {n_xs} reaction cross sections for {len(result.particle_types)} particle types")
    else:
        result.has_data = False
        if debug:
            logger.debug("No valid particle production cross section data found")
    
    return result

def read_yield_based_xs(xss: np.ndarray, start_idx: int, mt: int, mftype: int, debug=False) -> Optional[YieldBasedCrossSection]:
    """
    Read yield-based cross section data (MFTYPE = 12 or 16).
    
    According to Table 58, the format for yield-based cross sections (MFTYPE=12 or 16) is:
    - MFTYPE at start_idx
    - MTMULT at start_idx + 1
    - NR (number of interpolation regions) at start_idx + 2
    - NBT and INT arrays for interpolation if NR > 0
    - NE (number of energies) after interpolation data
    - Energy and yield arrays
    
    Parameters
    ----------
    xss : np.ndarray
        The XSS array (ace.xss_data)
    start_idx : int
        Starting index in the XSS array (where MFTYPE is located)
    mt : int
        MT number for this reaction
    mftype : int
        MFTYPE value (12 or 16)
    debug : bool, optional
        Whether to print debug information, defaults to False
    
    Returns
    -------
    Optional[YieldBasedCrossSection]
        The parsed cross section data or None if there was an error
    """
    if start_idx + 2 >= len(xss):
        if debug:
            logger.debug(f"    ERROR: Index {start_idx + 2} exceeds XSS length {len(xss)}")
        return None
    
    xs = YieldBasedCrossSection(mt=mt, mftype=mftype)
    
    # Read MTMULT
    mtmult_entry = xss[start_idx + 1]
    xs.mtmult = int(mtmult_entry)
    
    if debug:
        logger.debug(f"    MTMULT = {xs.mtmult} → MT whose cross section multiplies yield")
    
    # Read number of interpolation regions (NR)
    num_regions_entry = xss[start_idx + 2]
    xs.num_regions = int(num_regions_entry)
    
    if debug:
        logger.debug(f"    NR = {xs.num_regions} → Number of interpolation regions")
    
    current_idx = start_idx + 3
    
    # Read interpolation data if present
    if xs.num_regions > 0:
        if current_idx + 2 * xs.num_regions - 1 >= len(xss):
            if debug:
                logger.debug(f"    ERROR: Not enough data for interpolation regions")
            return None
        
        # Read NBT array (view of xss_data)
        xs.interpolation_bounds = xss[current_idx:current_idx + xs.num_regions]
        current_idx += xs.num_regions
        
        # Read INT array (view of xss_data)
        xs.interpolation_schemes = xss[current_idx:current_idx + xs.num_regions]
        current_idx += xs.num_regions
        
        if debug:
            bounds = [int(b) for b in xs.interpolation_bounds]
            schemes = [int(s) for s in xs.interpolation_schemes]
            logger.debug(f"    NBT (interpolation bounds): {bounds}")
            logger.debug(f"    INT (interpolation schemes): {schemes}")
    
    # Read number of energies (NE)
    if current_idx >= len(xss):
        if debug:
            logger.debug(f"    ERROR: Index {current_idx} exceeds XSS length {len(xss)}")
        return None
        
    num_energies_entry = xss[current_idx]
    xs.num_energies = int(num_energies_entry)
    current_idx += 1
    
    if debug:
        logger.debug(f"    NE = {xs.num_energies} → Number of energy points")
    
    # Check bounds
    if current_idx + 2 * xs.num_energies - 1 >= len(xss):
        if debug:
            logger.debug(f"    ERROR: Not enough data for energy/yield points")
        return None
    
    # Read energies (E array) (view of xss_data)
    xs.energies = xss[current_idx:current_idx + xs.num_energies]
    current_idx += xs.num_energies
    
    # Read yields (Y array) (view of xss_data)
    xs.yields = xss[current_idx:current_idx + xs.num_energies]
    
    if debug:
        logger.debug(f"    Successfully read {xs.num_energies} energy/yield points")
    
    return xs

def read_direct_xs(xss: np.ndarray, start_idx: int, mt: int, debug=False) -> Optional[DirectCrossSection]:
    """
    Read direct cross section data (MFTYPE = 13).
    
    According to Table 59, the format for direct cross sections (MFTYPE=13) is:
    - MFTYPE at start_idx
    - IE (energy grid index) at start_idx + 1
    - NE (number of consecutive entries) at start_idx + 2
    - Cross section values
    
    Parameters
    ----------
    xss : np.ndarray
        The XSS array (ace.xss_data)
    start_idx : int
        Starting index in the XSS array (where MFTYPE is located)
    mt : int
        MT number for this reaction
    debug : bool, optional
        Whether to print debug information, defaults to False
    
    Returns
    -------
    Optional[DirectCrossSection]
        The parsed cross section data or None if there was an error
    """
    if start_idx + 2 >= len(xss):
        if debug:
            logger.debug(f"    ERROR: Index {start_idx + 2} exceeds XSS length {len(xss)}")
        return None
    
    xs = DirectCrossSection(mt=mt, mftype=13)
    
    # Read energy grid index (IE)
    energy_grid_idx_entry = xss[start_idx + 1]
    xs.energy_grid_index = int(energy_grid_idx_entry)
    
    if debug:
        logger.debug(f"    IE = {xs.energy_grid_index} → Energy grid index")
    
    # Read number of entries (NE)
    num_entries_entry = xss[start_idx + 2]
    xs.num_entries = int(num_entries_entry)
    
    if debug:
        logger.debug(f"    NE = {xs.num_entries} → Number of consecutive entries")
    
    # Check bounds
    if start_idx + 3 + xs.num_entries - 1 >= len(xss):
        if debug:
            logger.debug(f"    ERROR: Not enough data for cross section values")
        return None
    
    # Read cross section values (view of xss_data)
    xs.cross_sections = xss[start_idx + 3:start_idx + 3 + xs.num_entries]
    
    if debug:
        logger.debug(f"    Successfully read {xs.num_entries} cross section values")
    
    return xs
