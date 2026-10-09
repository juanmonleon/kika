from typing import List, Dict, Optional
import numpy as np
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.locators import EnergyDistributionLocators
import logging

# Setup logger
logger = logging.getLogger(__name__)

def read_energy_locator_blocks(ace: Ace, debug: bool = False) -> EnergyDistributionLocators:
    """
    Read the LDLW, LDLWP, LDLWH, and DNEDL blocks from the ACE file.
    
    These blocks contain locators (indices) to the energy distribution data:
    - LDLW: For incident neutron reactions (JXS[10])
    - LDLWP: For photon production reactions (JXS[18])
    - LDLWH: For other particle production reactions (accessed through JXS[31])
    - DNEDL: For delayed neutron precursor groups (JXS[26])
    
    All locators (LOCCᵢ) are relative to:
    - JED = JXS(19) for LDLW and LDLWP
    - JED = XSS(JXS(32) + 10*(i-1) + 8) for LDLWH (with particle index i)
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS data
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    EnergyDistributionLocators
        Object containing the energy distribution locators
    """
    if debug:
        logger.debug("\n===== ENERGY DISTRIBUTION LOCATOR BLOCKS PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
        
    # Create a new container
    result = EnergyDistributionLocators()
    
    # Check if we have the necessary base data structures
    if not ace.header or not ace.header.jxs_array or not ace.header.nxs_array or ace.xss_data is None or len(ace.xss_data) == 0:
        if debug:
            logger.debug("Skipping energy distribution locator blocks: required data missing")
        return result
    
    # Get the NXS values for number of MT values
    num_secondary_neutron_reactions = ace.header.num_secondary_neutron_reactions  # NXS(5)
    num_photon_production_reactions = ace.header.num_photon_production_reactions  # NXS(6)
    num_particle_types = ace.header.num_particle_types  # NXS(7)
    num_delayed_neutron_precursors = ace.header.num_delayed_neutron_precursors  # NXS(8)
    
    if debug:
        logger.debug(f"NXS(5) = {num_secondary_neutron_reactions} → Secondary neutron reactions")
        logger.debug(f"NXS(6) = {num_photon_production_reactions} → Photon production reactions")
        logger.debug(f"NXS(7) = {num_particle_types} → Particle types")
        logger.debug(f"NXS(8) = {num_delayed_neutron_precursors} → Delayed neutron precursors")
    
    # Store the number of MT values in the energy_distribution_locators container
    result.num_secondary_neutron_reactions = num_secondary_neutron_reactions
    result.num_photon_production_reactions = num_photon_production_reactions
    result.num_particle_types = num_particle_types
    result.num_delayed_neutron_precursors = num_delayed_neutron_precursors
    
    # Get the JXS pointers for each block
    jxs10 = ace.header.jxs_array[10]  # JXS(10) - LDLW block
    jxs18 = ace.header.jxs_array[18]  # JXS(18) - LDLWP block
    jxs26 = ace.header.jxs_array[26]  # JXS(26) - DNEDL block
    jxs31 = ace.header.jxs_array[31]  # JXS(31) - Particle pointer array
    
    if debug:
        logger.debug(f"JXS(10) = {jxs10} → LDLW block locator for secondary neutrons")
        logger.debug(f"JXS(18) = {jxs18} → LDLWP block locator for photon production")
        logger.debug(f"JXS(26) = {jxs26} → DNEDL block locator for delayed neutrons")
        logger.debug(f"JXS(31) = {jxs31} → Particle pointer for other secondary particles")
    
    # Read LDLW block (incident neutron reactions with secondary neutrons)
    if num_secondary_neutron_reactions > 0 and jxs10 > 0:
        if debug:
            logger.debug(f"Reading LDLW block at index {jxs10}")
        result.incident_neutron = read_ldlw_block(ace, jxs10, num_secondary_neutron_reactions, debug)
    else:
        if debug:
            logger.debug("Skipping LDLW block: no secondary neutron reactions or JXS(10) <= 0")
            
    # Read LDLWP block (photon production)
    if num_photon_production_reactions > 0 and jxs18 > 0:
        if debug:
            logger.debug(f"Reading LDLWP block at index {jxs18}")
        result.photon_production = read_ldlwp_block(ace, jxs18, num_photon_production_reactions, debug)
    else:
        if debug:
            logger.debug("Skipping LDLWP block: no photon production reactions or JXS(18) <= 0")
    
    # Read DNEDL block (delayed neutron precursors)
    if num_delayed_neutron_precursors > 0 and jxs26 > 0:
        if debug:
            logger.debug(f"Reading DNEDL block at index {jxs26}")
        result.delayed_neutron = read_dnedl_block(ace, jxs26, num_delayed_neutron_precursors, debug)
    else:
        if debug:
            logger.debug("Skipping DNEDL block: no delayed neutron precursors or JXS(26) <= 0")
    
    # Read LDLWH block (other particle production)
    if num_particle_types > 0 and jxs31 > 0:
        if debug:
            logger.debug(f"Reading LDLWH block for {num_particle_types} particle types")
        result.particle_production = read_ldlwh_block(ace, jxs31, num_particle_types, debug)
    else:
        if debug:
            logger.debug("Skipping LDLWH block: no particle types or JXS(31) <= 0")
    
    return result


def read_ldlw_block(ace: Ace, jxs10: int, num_reactions: int, debug: bool = False) -> np.ndarray:
    """
    Read the LDLW block for incident neutron reaction energy distribution locators.
    
    According to Table 28, each location in LDLW gives the energy distribution data location 
    for a specific MT reaction (LOCCᵢ). These locators are relative to JXS(19).
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS data
    jxs10 : int
        JXS(10) - Starting index of the LDLW block in the XSS array
    num_reactions : int
        NXS(5) - Number of reactions with secondary neutrons
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    numpy.ndarray
        List of energy distribution locators for incident neutron reactions
    """
    if jxs10 <= 0:
        if debug:
            logger.debug(f"Invalid LDLW block index: {jxs10}")
        return []
    
    if jxs10 >= len(ace.xss_data):
        if debug:
            logger.debug(f"LDLW block index out of bounds: {jxs10} >= {len(ace.xss_data)}")
        return []
    
    # Make sure we have enough data
    if jxs10 + num_reactions > len(ace.xss_data):
        if debug:
            logger.debug(f"LDLW block truncated: need {num_reactions} entries, but only {len(ace.xss_data) - jxs10} available")
        return []
    
    # Read the energy distribution locators - a view of xss_data
    locators = ace.xss_data[jxs10:jxs10 + num_reactions]
    
    if debug:
        logger.debug(f"Successfully read {len(locators)} incident neutron energy distribution locators")
        # Print the first few LOCC values to verify
        sample_size = min(3, len(locators))
        locc_values = [int(locators[i]) for i in range(sample_size)]
        logger.debug(f"First {sample_size} LDLW LOCC values: {locc_values}")
        logger.debug(f"First {sample_size} LDLW indices: {[jxs10 + i for i in range(sample_size)]}")
        
        if len(locators) > 0:
            # Check if locators are monotonically increasing
            is_monotonic = all(locators[i] <= locators[i+1] for i in range(len(locators)-1))
            logger.debug(f"Locators are {'monotonically increasing' if is_monotonic else 'NOT monotonically increasing'}")
            logger.debug(f"Note: These locators are relative to JXS(19) = {ace.header.jxs_array[19]}")
    
    return locators


def read_ldlwp_block(ace: Ace, jxs18: int, num_photon_reactions: int, debug: bool = False) -> np.ndarray:
    """
    Read the LDLWP block for photon production energy distribution locators.
    
    According to Table 28, each location in LDLWP gives the energy distribution data 
    location for a specific photon production reaction (LOCCᵢ).
    These locators are relative to JXS(19).
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS data
    jxs18 : int
        JXS(18) - Starting index of the LDLWP block in the XSS array
    num_photon_reactions : int
        NXS(6) - Number of photon production reactions
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    numpy.ndarray
        List of energy distribution locators for photon production
    """
    if jxs18 <= 0:
        if debug:
            logger.debug(f"Invalid LDLWP block index: {jxs18}")
        return []
    
    if jxs18 >= len(ace.xss_data):
        if debug:
            logger.debug(f"LDLWP block index out of bounds: {jxs18} >= {len(ace.xss_data)}")
        return []
    
    # Make sure we have enough data
    if jxs18 + num_photon_reactions > len(ace.xss_data):
        if debug:
            logger.debug(f"LDLWP block truncated: need {num_photon_reactions} entries, but only {len(ace.xss_data) - jxs18} available")
        return []
    
    # Read the photon production energy distribution locators - a view of xss_data
    locators = ace.xss_data[jxs18:jxs18 + num_photon_reactions]
    
    if debug:
        logger.debug(f"Successfully read {len(locators)} photon production energy distribution locators")
        # Print the first few LOCC values to verify
        sample_size = min(3, len(locators))
        if sample_size > 0:
            locc_values = [int(locators[i]) for i in range(sample_size)]
            logger.debug(f"First {sample_size} LDLWP LOCC values: {locc_values}")
            logger.debug(f"First {sample_size} LDLWP indices: {[jxs18 + i for i in range(sample_size)]}")
        
        if len(locators) > 0:
            # Check if locators are monotonically increasing
            is_monotonic = all(locators[i] <= locators[i+1] for i in range(len(locators)-1))
            logger.debug(f"Locators are {'monotonically increasing' if is_monotonic else 'NOT monotonically increasing'}")
            logger.debug(f"Note: These locators are relative to JXS(19) = {ace.header.jxs_array[19]}")
    
    return locators


def read_ldlwh_block(ace: Ace, jxs31: int, num_particle_types: int, debug: bool = False) -> List[np.ndarray]:
    """
    Read the LDLWH block for other particle production energy distribution locators.
    
    According to Table 27, for each particle type i:
    - The number of MT values is found at XSS(JXS(31) + i - 1)
    - The LDLWH block starts at XSS(JXS(32) + 10*(i-1) + 7)
    - These locators are relative to JED = XSS(JXS(32) + 10*(i-1) + 8)
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS data
    jxs31 : int
        JXS(31) - Particle pointer array starting index in the XSS array
    num_particle_types : int
        NXS(7) - Number of particle types
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    List[numpy.ndarray]
        List of lists of energy distribution locators for each particle type
    """
    if jxs31 <= 0:
        if debug:
            logger.debug(f"Invalid particle pointer block index: {jxs31}")
        return []
    
    if jxs31 >= len(ace.xss_data):
        if debug:
            logger.debug(f"Particle pointer block index out of bounds: {jxs31} >= {len(ace.xss_data)}")
        return []
    
    # Get the base index for JXS(32)
    if len(ace.header.jxs_array) <= 32:
        if debug:
            logger.debug("JXS array too short: missing JXS(32) index")
        return []
    
    jxs32 = ace.header.jxs_array[32]
    
    if jxs32 <= 0:
        if debug:
            logger.debug(f"Invalid JXS(32) value: {jxs32}")
        return []
    
    # Initialize the particle production locators list
    particle_production_locators = []
    
    # For each particle type i, process its energy distribution locators
    for i in range(1, num_particle_types + 1):
        if debug:
            logger.debug(f"Processing energy distribution locators for particle type {i}")
            
        # 1. Get the number of MT values for this particle type at XSS(JXS(31) + i - 1)
        mt_count_idx = jxs31 + (i - 1)
        
        if mt_count_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"MT count index out of bounds for particle type {i}: {mt_count_idx} >= {len(ace.xss_data)}")
            particle_production_locators.append([])
            continue
        
        num_mt_values = int(ace.xss_data[mt_count_idx])
        
        if debug:
            logger.debug(f"Particle type {i} has {num_mt_values} MT values")
            
        # 2. Get the LDLWH pointer from XSS(JXS(32) + 10*(i-1) + 7)
        ldlwh_pointer_idx = jxs32 + 10 * (i - 1) + 7 
        
        if ldlwh_pointer_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"LDLWH pointer index out of bounds for particle type {i}: {ldlwh_pointer_idx} >= {len(ace.xss_data)}")
            particle_production_locators.append([])
            continue
        
        # Get the actual pointer value
        ldlwh_pointer = int(ace.xss_data[ldlwh_pointer_idx])
        
        if debug:
            logger.debug(f"LDLWH pointer for particle type {i}: {ldlwh_pointer}")
            
        # 3. Special case: If LDLWH pointer is 0, there's no energy distribution data for this particle
        if ldlwh_pointer <= 0:
            if debug:
                logger.debug(f"No energy distribution data for particle type {i}")
            particle_production_locators.append([])
            continue
        
        # 5. Make sure we have enough data for all MT values
        if ldlwh_pointer + num_mt_values > len(ace.xss_data):
            if debug:
                logger.debug(f"LDLWH block truncated for particle type {i}: need {num_mt_values} entries, but only {len(ace.xss_data) - ldlwh_pointer} available")
            particle_production_locators.append([])
            continue
        
        # 6. Read the locators for this particle type. A view of xss_data
        locators = ace.xss_data[ldlwh_pointer:ldlwh_pointer + num_mt_values]
        particle_production_locators.append(locators)
        
        if debug:
            logger.debug(f"Successfully read {len(locators)} energy distribution locators for particle type {i}")
            # Print the first few LOCC values to verify
            sample_size = min(3, len(locators))
            if sample_size > 0:
                locc_values = [int(locators[j]) for j in range(sample_size)]
                logger.debug(f"First {sample_size} LDLWH LOCC values for particle {i}: {locc_values}")
                logger.debug(f"First {sample_size} LDLWH indices for particle {i}: {[ldlwh_pointer + j for j in range(sample_size)]}")
            
            if len(locators) > 0:
                # Check if locators are monotonically increasing
                is_monotonic = all(locators[j] <= locators[j+1] for j in range(len(locators)-1))
                logger.debug(f"Locators are {'monotonically increasing' if is_monotonic else 'NOT monotonically increasing'}")
                
                # Get the JED value - these locators are relative to XSS(JXS(32) + 10*(i-1) + 8)
                jed_idx = jxs32 + 10 * (i - 1) + 8
                if jed_idx < len(ace.xss_data):
                    jed = int(ace.xss_data[jed_idx])
                    logger.debug(f"Note: These locators are relative to JED = {jed}")
    
    return particle_production_locators


def read_dnedl_block(ace: Ace, jxs26: int, num_precursors: int, debug: bool = False) -> np.ndarray:
    """
    Read the DNEDL block for delayed neutron energy distribution locators.
    
    According to Table 28, each location in DNEDL gives the energy distribution 
    data location for a specific delayed neutron precursor group (LOCCᵢ).
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS data
    jxs26 : int
        JXS(26) - Starting index of the DNEDL block in the XSS array
    num_precursors : int
        NXS(8) - Number of delayed neutron precursor groups
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    numpy.ndarray
        List of energy distribution locators for delayed neutron groups
    """
    if jxs26 <= 0:
        if debug:
            logger.debug(f"Invalid DNEDL block index: {jxs26}")
        return []
    
    if jxs26 >= len(ace.xss_data):
        if debug:
            logger.debug(f"DNEDL block index out of bounds: {jxs26} >= {len(ace.xss_data)}")
        return []
    
    # Make sure we have enough data
    if jxs26 + num_precursors > len(ace.xss_data):
        if debug:
            logger.debug(f"DNEDL block truncated: need {num_precursors} entries, but only {len(ace.xss_data) - jxs26} available")
        return []
    
    # Read the delayed neutron energy distribution locators - a view of xss_data
    locators = ace.xss_data[jxs26:jxs26 + num_precursors]
    
    if debug:
        logger.debug(f"Successfully read {len(locators)} delayed neutron energy distribution locators")
        # Print the first few LOCC values to verify
        sample_size = min(3, len(locators))
        if sample_size > 0:
            locc_values = [int(locators[i]) for i in range(sample_size)]
            logger.debug(f"First {sample_size} DNEDL LOCC values: {locc_values}")
            logger.debug(f"First {sample_size} DNEDL indices: {[jxs26 + i for i in range(sample_size)]}")
        
        if len(locators) > 0:
            # Check if locators are monotonically increasing
            is_monotonic = all(locators[i] <= locators[i+1] for i in range(len(locators)-1))
            logger.debug(f"Locators are {'monotonically increasing' if is_monotonic else 'NOT monotonically increasing'}")
    
    return locators
