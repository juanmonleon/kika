import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.angular_distribution.angular_locators import AngularDistributionLocators

# Setup logger
logger = logging.getLogger(__name__)

def read_angular_locator_blocks(ace: Ace, debug=False) -> AngularDistributionLocators:
    """
    Read the LAND, LANDP, and LANDH blocks from the ACE file.
    
    These blocks contain locators (indices) to the angular distribution data:
    - LAND: For incident neutron reactions (JXS[8])
    - LANDP: For photon production reactions (JXS[16])
    - LANDH: For other particle production reactions (accessed through particle pointers)
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    AngularDistributionLocators
        The angular distribution locators object
    
    Raises
    ------
    ValueError
        If required data is missing or indices are invalid
    """
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        if debug:
            logger.debug("Skipping angular locator blocks: required data missing")
        raise ValueError("Cannot read angular locator blocks: header or XSS data missing")
    
    if debug:
        logger.debug("\n===== ANGULAR DISTRIBUTION LOCATOR BLOCKS PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Create a new container if not present
    if ace.angular_locators is None:
        ace.angular_locators = AngularDistributionLocators()
    
    # Get the NXS values for number of MT values
    num_reactions = ace.header.nxs_array[4]  # NXS(4)
    num_secondary_neutron_reactions = ace.header.nxs_array[5]  # NXS(5)
    num_photon_production_reactions = ace.header.nxs_array[6]  # NXS(6)
    num_particle_types = ace.header.nxs_array[7]  # NXS(7)
    
    if debug:
        logger.debug(f"NXS(4) = {num_reactions} → Total number of reactions")
        logger.debug(f"NXS(5) = {num_secondary_neutron_reactions} → Number of secondary neutron reactions")
        logger.debug(f"NXS(6) = {num_photon_production_reactions} → Number of photon production reactions")
        logger.debug(f"NXS(7) = {num_particle_types} → Number of particle types")
    
    # Store the number of MT values in the angular_locators container
    ace.angular_locators.num_neutron_reactions = num_reactions
    ace.angular_locators.num_secondary_neutron_reactions = num_secondary_neutron_reactions
    ace.angular_locators.num_photon_production_reactions = num_photon_production_reactions
    ace.angular_locators.num_particle_types = num_particle_types
    
    # Get the JXS pointers for each block
    land_idx = ace.header.jxs_array[8]  # JXS(8)
    landp_idx = ace.header.jxs_array[16]  # JXS(16)
    jxs31_idx = ace.header.jxs_array[31]  # JXS(31)
    jxs32_idx = ace.header.jxs_array[32]  # JXS(32)
    
    if debug:
        logger.debug(f"JXS(8) = {land_idx} → Locator for LAND block (angular locators for neutrons)")
        logger.debug(f"JXS(16) = {landp_idx} → Locator for LANDP block (angular locators for photons)")
        logger.debug(f"JXS(31) = {jxs31_idx} → Locator for NMT values")
        logger.debug(f"JXS(32) = {jxs32_idx} → Base index for particle data blocks")
    
    # Read LAND block. It always exists (manual Table 16): its first locator is
    # the elastic one, followed by one per secondary-neutron reaction, so a
    # table with NXS(5)=0 (H-1, He-4) still carries the elastic locator.
    if debug:
        logger.debug("\n----- LAND Block -----")
    read_land_block(ace, land_idx, num_secondary_neutron_reactions, debug)
    
    # Read LANDP block (photon production)
    if num_photon_production_reactions > 0:
        if debug:
            logger.debug("\n----- LANDP Block -----")
        read_landp_block(ace, landp_idx, num_photon_production_reactions, debug)
    elif debug:
        logger.debug("No LANDP block to process (no photon production reactions)")
    
    # Read LANDH block (other particle production)
    if num_particle_types > 0:
        if debug:
            logger.debug("\n----- LANDH Block -----")
        read_landh_block(ace, jxs31_idx, jxs32_idx, num_particle_types, debug)
    elif debug:
        logger.debug("No LANDH block to process (no particle types)")
    
    return ace.angular_locators


def read_land_block(ace: Ace, land_idx: int, num_reactions: int, debug=False) -> None:
    """
    Read the LAND block for incident neutron reaction angular distribution locators.
    
    This block contains locators (LOCB values) that point to angular distributions:
    - LOCB1 = LAND[0]: Angular distribution for elastic scattering (should be 1)
    - LOCB2 = LAND[1]: Angular distribution for the first reaction
    - ...
    - LOCB_NMT = LAND[NMT-1]: Angular distribution for the last reaction
    
    These locators are relative offsets from AND=JXS(9). To get the actual
    XSS index, use: JXS(9) + LOCB
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    land_idx : int
        Starting index of the LAND block in the XSS array
    num_reactions : int
        Number of reactions excluding elastic scattering (NXS(5))
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the LAND block data is missing or invalid
    """
    if land_idx <= 0:
        if debug:
            logger.debug(f"Invalid LAND block index: {land_idx}")
        raise ValueError(f"Invalid LAND block index: {land_idx}")
    
    if debug:
        logger.debug(f"LAND block starts at index {land_idx}")
        logger.debug(f"Number of reactions (excluding elastic): {num_reactions}")
    
    if land_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"LAND block index out of bounds: {land_idx} >= {len(ace.xss_data)}")
        raise ValueError(f"LAND block index out of bounds: {land_idx} >= {len(ace.xss_data)}")
    
    # Make sure we have enough data (1 for elastic scattering + num_reactions)
    total_entries = num_reactions + 1  
    if land_idx + total_entries > len(ace.xss_data):
        if debug:
            logger.debug(f"LAND block truncated: need {total_entries} entries, but only {len(ace.xss_data) - land_idx} available")
        raise ValueError(f"LAND block truncated: need {total_entries} entries, but only {len(ace.xss_data) - land_idx} available")
    
    # LOCB1 - Angular distribution for elastic scattering (first element)
    elastic_locator = int(ace.xss_data[land_idx])
    ace.angular_locators.elastic_scattering = elastic_locator
    
    if debug:
        logger.debug(f"Elastic scattering locator (LOCB1): {elastic_locator}")
        # Check if LOCB₁ = 1 as per documentation
        if elastic_locator != 1:
            logger.warning(f"Elastic scattering locator value is {elastic_locator}, but documentation indicates it should be 1")
    
    # LOCB2 through LOCB_NMT - Angular distributions for other neutron reactions
    ace.angular_locators.incident_neutron = ace.xss_data[land_idx + 1:land_idx + 1 + num_reactions]
    
    if debug:
        logger.debug(f"Successfully read {len(ace.angular_locators.incident_neutron)} neutron reaction angular distribution locators")
        # Display first 3 LOCB values for verification
        display_count = min(3, len(ace.angular_locators.incident_neutron))
        if display_count > 0:
            locb_values = [int(ace.angular_locators.incident_neutron[i]) for i in range(display_count)]
            logger.debug(f"First {display_count} LOCB values: {locb_values}")
        logger.debug(f"These locators are relative to JXS(9)={ace.header.jxs_array[9]}")


def read_landp_block(ace: Ace, landp_idx: int, num_photon_reactions: int, debug=False) -> None:
    """
    Read the LANDP block for photon production angular distribution locators.
    
    This block contains locators (LOCB values) that point to photon production angular distributions:
    - LOCB1 = LANDP[0]: Angular distribution for first photon production reaction
    - LOCB2 = LANDP[1]: Angular distribution for second photon production reaction
    - ...
    - LOCB_NMT = LANDP[NMT-1]: Angular distribution for the last photon production reaction
    
    These locators are relative offsets from ANDP=JXS(17). To get the actual
    XSS index, use: ANDP + LOCB - 1
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    landp_idx : int
        Starting index of the LANDP block in the XSS array
    num_photon_reactions : int
        Number of photon production reactions (NXS(6))
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the LANDP block data is missing or invalid
    """
    if landp_idx <= 0:
        if debug:
            logger.debug(f"Invalid LANDP block index: {landp_idx}")
        raise ValueError(f"Invalid LANDP block index: {landp_idx}")
    
    if debug:
        logger.debug(f"LANDP block starts at index {landp_idx}")
        logger.debug(f"Number of photon production reactions: {num_photon_reactions}")
    
    if landp_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"LANDP block index out of bounds: {landp_idx} >= {len(ace.xss_data)}")
        raise ValueError(f"LANDP block index out of bounds: {landp_idx} >= {len(ace.xss_data)}")
    
    # Make sure we have enough data
    if landp_idx + num_photon_reactions > len(ace.xss_data):
        if debug:
            logger.debug(f"LANDP block truncated: need {num_photon_reactions} entries, but only {len(ace.xss_data) - landp_idx} available")
        raise ValueError(f"LANDP block truncated: need {num_photon_reactions} entries, but only {len(ace.xss_data) - landp_idx} available")
    
    # Read the photon production  (LOCB1 through LOCB_NMT)
    ace.angular_locators.photon_production = ace.xss_data[landp_idx:landp_idx + num_photon_reactions]
    
    if debug:
        logger.debug(f"Successfully read {len(ace.angular_locators.photon_production)} photon production angular distribution locators")
        # Display first 3 LOCB values for verification
        display_count = min(3, len(ace.angular_locators.photon_production))
        if display_count > 0:
            locb_values = [int(ace.angular_locators.photon_production[i]) for i in range(display_count)]
            logger.debug(f"First {display_count} LOCB values: {locb_values}")


def read_landh_block(ace: Ace, jxs31_idx: int, jxs32_idx: int, num_particle_types: int, debug=False) -> None:
    """
    Read the LANDH block for other particle production angular distribution locators.
    
    This block contains locators (LOCB values) for angular distributions of particle production reactions:
    - For each particle type i (1 to NXS(7)):
      - The number of reactions is stored at XSS(JXS(31)+i-1)
      - The LANDH pointer is at XSS(JXS(32)+10*(i-1)+5)
      - This pointer points to a list of locators, one for each MT number
    
    These locators are relative offsets from ANDH=XSS(JXS(32)+10*(i-1)+6). To get the actual
    XSS index, use: ANDH + LOCB - 1
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    jxs31_idx : int
        Starting index of the NMT values (JXS(31))
    jxs32_idx : int
        Base index for particle data blocks (JXS(32))
    num_particle_types : int
        Number of particle types (NXS(7))
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the LANDH block data is missing or invalid
    """
    if jxs31_idx <= 0:
        if debug:
            logger.debug(f"Invalid particle pointer block index: {jxs31_idx}")
        raise ValueError(f"Invalid particle pointer block index: {jxs31_idx}")
    
    if debug:
        logger.debug(f"Particle pointer index: {jxs31_idx}")
        logger.debug(f"Number of particle types: {num_particle_types}")
    
    if jxs31_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Particle pointer block index out of bounds: {jxs31_idx} >= {len(ace.xss_data)}")
        raise ValueError(f"Particle pointer block index out of bounds: {jxs31_idx} >= {len(ace.xss_data)}")
    
    if jxs32_idx < 0:
        if debug:
            logger.debug(f"Invalid JXS(32) value: {jxs32_idx}")
        raise ValueError(f"Invalid JXS(32) value: {jxs32_idx}")
    
    if debug:
        logger.debug(f"JXS(32) index: {jxs32_idx}")
    
    # Initialize the particle production locators list
    ace.angular_locators.particle_production = []
    
    # For each particle type i, process its angular distribution locators
    for i in range(1, num_particle_types + 1):
        if debug:
            logger.debug(f"\nProcessing particle type {i}:")
        
        # 1. Get the number of MT values for this particle type
        mt_count_idx = jxs31_idx + (i - 1)
        
        if debug:
            logger.debug(f"  MT count index: jxs31_idx + (i-1) = {jxs31_idx} + ({i-1}) = {mt_count_idx}")
        
        if mt_count_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: MT count index out of bounds: {mt_count_idx} >= {len(ace.xss_data)}")
            raise ValueError(f"MT count index out of bounds for particle type {i}: {mt_count_idx} >= {len(ace.xss_data)}")
        
        num_mt_values = int(ace.xss_data[mt_count_idx])
        
        if debug:
            logger.debug(f"  Number of MT values: {num_mt_values}")
        
        # 2. Get the LANDH pointer from the particle's data block
        # LANDH pointer is at XSS(JXS(32)+10*(i-1)+5) according to Table 18
        landh_offset = 10 * (i - 1) + 5
        landh_pointer_idx = jxs32_idx + landh_offset
        
        if debug:
            logger.debug(f"  LANDH pointer index: JXS(32) + 10*(i-1) + 5 = {jxs32_idx} + {landh_offset} = {landh_pointer_idx}")
        
        if landh_pointer_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: LANDH pointer index out of bounds: {landh_pointer_idx} >= {len(ace.xss_data)}")
            raise ValueError(f"LANDH pointer index out of bounds for particle type {i}: {landh_pointer_idx} >= {len(ace.xss_data)}")
        
        # Get the actual pointer value (to the LOCB values)
        landh_locb_ptr = int(ace.xss_data[landh_pointer_idx])
        
        if debug:
            logger.debug(f"  LANDH pointer value: {landh_locb_ptr} → Location of LOCB values for particle type {i}")
        
        # Also get the ANDH pointer (needed for interpreting the locators)
        # ANDH pointer is at XSS(JXS(32)+10*(i-1)+6) according to Table 21
        andh_offset = 10 * (i - 1) + 6
        andh_pointer_idx = jxs32_idx + andh_offset
        
        if debug and andh_pointer_idx < len(ace.xss_data):
            andh_ptr = int(ace.xss_data[andh_pointer_idx])
            logger.debug(f"  ANDH pointer index: JXS(32) + 10*(i-1) + 6 = {jxs32_idx} + {andh_offset} = {andh_pointer_idx}")
            logger.debug(f"  ANDH pointer value: {andh_ptr} → Base for angular distribution data")
        
        # Make sure we have enough data
        if landh_locb_ptr + num_mt_values > len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: LANDH block truncated: need {num_mt_values} entries, but only {len(ace.xss_data) - landh_locb_ptr} available")
            raise ValueError(f"LANDH block truncated for particle type {i}: need {num_mt_values} entries, but only {len(ace.xss_data) - landh_locb_ptr} available")
        
        # Read the locators (LOCB values) for this particle type
        locb_values = ace.xss_data[landh_locb_ptr:landh_locb_ptr + num_mt_values]
        ace.angular_locators.particle_production.append(locb_values)
        
        if debug:
            logger.debug(f"  Successfully read {len(locb_values)} angular distribution locators")
            # Display first 3 LOCB values for verification
            display_count = min(3, len(locb_values))
            if display_count > 0:
                locb_sample = [int(locb_values[j]) for j in range(display_count)]
                logger.debug(f"  First {display_count} LOCB values: {locb_sample}")
    
    if debug:
        logger.debug(f"\nSuccessfully read angular distribution locators for {len(ace.angular_locators.particle_production)} particle types")
