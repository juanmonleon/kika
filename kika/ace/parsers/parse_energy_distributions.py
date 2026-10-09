from typing import List, Optional, Dict
import numpy as np
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.container import EnergyDistributionContainer
from kika.ace.parsers.laws import (
    parse_tabular_energy_distribution,
    parse_discrete_energy_distribution,
    parse_level_scattering,
    parse_continuous_energy_angle_distribution,
    parse_general_evaporation_spectrum,
    parse_maxwell_fission_spectrum,
    parse_evaporation_spectrum,
    parse_energy_dependent_watt_spectrum,
    parse_tabular_linear_functions,
    parse_tabular_energy_multipliers,
    parse_kalbach_mann_distribution,
    parse_tabulated_angle_energy_distribution,
    parse_nbody_phase_space_distribution,
    parse_laboratory_angle_energy_distribution,
    parse_energy_dependent_yield
)
from kika.ace.classes.xss import xss_position
import logging

# Setup logger
logger = logging.getLogger(__name__)

class EnergyDistributionParseError(ValueError):
    """Exception raised for errors in parsing energy distribution data."""
    pass

def read_energy_distribution_blocks(ace: Ace, debug: bool = False) -> EnergyDistributionContainer:
    """
    Read the DLW, DLWP, DLWH, and DNED blocks from the ACE file.
    
    These blocks contain energy distributions for secondary particles:
    - DLW: For incident neutron reactions (secondary neutrons)
    - DLWP: For photon production reactions
    - DLWH: For other particle production reactions
    - DNED: For delayed neutron precursor groups
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS data and headers
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    EnergyDistributionContainer
        Container with the energy distribution data
        
    Raises
    ------
    ValueError
        If required data is missing or indices are invalid
    """

    if debug: 
        logger.debug("\n===== ENERGY DISTRIBUTION BLOCKS PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
        
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None or
        ace.energy_distribution_locators is None):
        raise ValueError("Cannot read energy distribution blocks: header, XSS data, or locators missing")
    
    # Create a new container for the return value
    result = EnergyDistributionContainer()
    
    # Get the JXS values for the energy distribution blocks
    jxs_dlw = ace.header.jxs_array[11]    # JXS(11)
    jxs_dlwp = ace.header.jxs_array[19]   # JXS(19)
    jxs_dned = ace.header.jxs_array[27]   # JXS(27)
    
    if debug:
        logger.debug(f"JXS(11) = {jxs_dlw} → DLW block for incident neutron reactions")
        logger.debug(f"JXS(19) = {jxs_dlwp} → DLWP block for photon production reactions")
        logger.debug(f"JXS(27) = {jxs_dned} → DNED block for delayed neutron precursors")
    
    # Store the container reference to be used by the helper functions
    _process_energy_distributions(ace, result, jxs_dlw, jxs_dlwp, jxs_dned, debug)
    
    return result


def _process_energy_distributions(ace: Ace, result: EnergyDistributionContainer, 
                                 jxs_dlw: int, jxs_dlwp: int, jxs_dned: int, debug: bool = False) -> None:
    """
    Process all energy distribution blocks and fill the result container.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    jxs_dlw : int
        JXS(11) - Starting index of the DLW block 
    jxs_dlwp : int
        JXS(19) - Starting index of the DLWP block
    jxs_dned : int
        JXS(27) - Starting index of the DNED block
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    # 1. Read DLW block (neutron energy distributions)
    if ace.energy_distribution_locators.has_neutron_data and jxs_dlw > 0:
        if debug: logger.debug(f"Reading DLW block starting at index {jxs_dlw}")
        read_dlw_block(ace, result, jxs_dlw, debug=debug)
    elif debug: logger.debug("Skipping DLW block (no neutron data or invalid JXS index)")
    
    # 2. Read DLWP block (photon production energy distributions)
    if ace.energy_distribution_locators.has_photon_production_data and jxs_dlwp > 0:
        if debug: logger.debug(f"Reading DLWP block starting at index {jxs_dlwp}")
        read_dlwp_block(ace, result, jxs_dlwp, debug=debug)
    elif debug: logger.debug("Skipping DLWP block (no photon production data or invalid JXS index)")
    
    # 3. Read DLWH block (other particle production energy distributions)
    if ace.energy_distribution_locators.has_particle_production_data:
        if debug: logger.debug("Reading DLWH block")
        read_dlwh_block(ace, result, debug=debug)
    elif debug: logger.debug("Skipping DLWH block (no particle production data)")
    
    # 4. Read DNED block (delayed neutron energy distributions)
    if ace.energy_distribution_locators.has_delayed_neutron_data and jxs_dned > 0:
        if debug: logger.debug(f"Reading DNED block starting at index {jxs_dned}")
        read_dned_block(ace, result, jxs_dned, debug=debug)
    elif debug: logger.debug("Skipping DNED block (no delayed neutron data or invalid JXS index)")
    
    if debug: logger.debug("Finished read_energy_distribution_blocks")


def read_dlw_block(ace: Ace, result: EnergyDistributionContainer, jxs_dlw: int, debug: bool = False) -> None:
    """
    Read the DLW block for neutron energy distributions.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    jxs_dlw : int
        JXS(11) - Starting index of the DLW block in the XSS array
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug: logger.debug(f"Starting read_dlw_block at index {jxs_dlw}")
    # Get MT numbers and corresponding locators from the previously read LDLW block
    if not ace.reaction_mt_data or not ace.reaction_mt_data.has_neutron_mt_data:
        if debug: logger.debug("Exiting read_dlw_block: no neutron MT data available")
        return
    
    # Get the MT numbers for reactions with secondary neutrons
    neutron_mts = ace.reaction_mt_data.secondary_neutron_mt
    if debug: 
        logger.debug(f"Found {len(neutron_mts)} secondary neutron MT numbers")
        # Print the first few MT values
        sample_size = min(3, len(neutron_mts))
        mt_sample = [int(neutron_mts[i]) for i in range(sample_size)]
        logger.debug(f"First {sample_size} MT values: {mt_sample}")
    
    # Get the corresponding locators from the energy distribution locators
    locators = []
    
    # Map each MT number to its position in the MTR block
    mt_idx_map = {}
    for i, mt_entry in enumerate(ace.reaction_mt_data.incident_neutron):
        mt_value = int(mt_entry)
        mt_idx_map[mt_value] = i
    
    if debug:
        logger.debug(f"Created MT index map with {len(mt_idx_map)} entries")
        logger.debug(f"Total incident_neutron MT entries: {len(ace.reaction_mt_data.incident_neutron)}")
        logger.debug(f"Total energy_distribution_locators: {len(ace.energy_distribution_locators.incident_neutron)}")
    
    # Debugging: print the full incident_neutron and LDLW arrays length
    if debug:
        logger.debug(f"Incident neutron MTs array length: {len(ace.reaction_mt_data.incident_neutron)}")
        logger.debug(f"LDLW locators array length: {len(ace.energy_distribution_locators.incident_neutron)}")
    
    for i, mt_item in enumerate(neutron_mts):
        # Make sure we're using integer MT values
        mt_value = int(mt_item)
        
        if mt_value in mt_idx_map and mt_idx_map[mt_value] < len(ace.energy_distribution_locators.incident_neutron):
            locator_entry = ace.energy_distribution_locators.incident_neutron[mt_idx_map[mt_value]]
            locators.append(locator_entry)
            if debug:
                locator_value = int(locator_entry)
                logger.debug(f"MT={mt_value}, locator={locator_value}, absolute index={jxs_dlw + locator_value - 1 if locator_value > 0 else 'N/A'}")
                # Print additional info about the mapping
                logger.debug(f"  MT index in incident_neutron: {mt_idx_map[mt_value]}")
                ldlw_start = xss_position(ace.energy_distribution_locators.incident_neutron)
                locator_pos = ldlw_start + mt_idx_map[mt_value] if ldlw_start is not None else None
                logger.debug(f"  Locator entry: {locator_pos}:{locator_entry}")
        elif debug:
            logger.debug(f"Could not find locator for MT={mt_value}")
            if mt_value in mt_idx_map:
                logger.debug(f"  MT index in map: {mt_idx_map[mt_value]}")
                logger.debug(f"  But this index is >= {len(ace.energy_distribution_locators.incident_neutron)}")
            else:
                logger.debug(f"  MT not found in incident_neutron array")
    
    # Make sure we have the same number of MT numbers and locators
    num_secondary_neutron_reactions = ace.header.nxs_array[4]  # NXS(5) in 1-indexed notation
    if debug:
        logger.debug(f"NXS(5)={num_secondary_neutron_reactions}, found MT numbers: {len(neutron_mts)}, locators: {len(locators)}")
        if len(neutron_mts) != len(locators):
            logger.debug(f"Warning: Mismatch between MT numbers and locators count")
    
    # Process each MT number with its corresponding locator
    for i, mt_item in enumerate(neutron_mts):
        if i >= len(locators):
            if debug: logger.debug(f"Skipping MT={mt_item}, index {i} out of range for locators array")
            continue
        
        mt_value = int(mt_item)
            
        locator_entry = locators[i]
        locator_value = int(locator_entry)
        
        if locator_value <= 0:
            if debug: logger.debug(f"Skipping MT={mt_value}, invalid locator: {locator_value}")
            continue  # Skip if the locator is invalid
        
        # According to Table 30, the energy distribution starts at JED + LOCC - 1
        offset = jxs_dlw + locator_value - 1  # Subtract 1 to match the documentation
        
        if debug: logger.debug(f"Processing MT={mt_value}, locator={locator_value}, offset={offset} (XSS length: {len(ace.xss_data)})")
        
        if offset >= len(ace.xss_data):
            if debug: logger.debug(f"Warning: Offset {offset} out of range for XSS data of length {len(ace.xss_data)}")
            continue
        
        try:
            if debug: logger.debug(f"Calling read_energy_distribution for MT={mt_value}")
            distributions = read_energy_distribution(ace, offset, jxs_dlw, debug=debug)
            if distributions:
                if debug: logger.debug(f"Successfully read {len(distributions)} distributions for MT={mt_value}")
                result.incident_neutron[mt_value] = distributions
                if debug: logger.debug(f"Stored distributions for MT={mt_value} (type: {type(mt_value)})")
            elif debug: logger.debug(f"No distributions found for MT={mt_value}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing energy distribution for MT={mt_value}: {e}")
    
    # Process energy-dependent yields for neutron reactions
    if ace.particle_release and ace.particle_release.has_neutron_data:
        if debug: logger.debug("Processing energy-dependent yields for neutron reactions")
        for i, ty in enumerate(ace.particle_release.incident_neutron):
            ty_value = int(ty)
            
            # According to documentation, yields are specified for TY values > 100 in absolute value
            if abs(ty_value) > 100 and i < len(neutron_mts):
                mt_item = neutron_mts[i]
                mt_value = int(mt_item)
                
                # Formula from Table 52: KY = JED + |TY_i| - 101
                ky = jxs_dlw + abs(ty_value) - 101
                
                if debug: 
                    logger.debug(f"Energy-dependent yield for MT={mt_value}, TY={ty_value}")
                    logger.debug(f"KY calculation: JED + |TY_i| - 101 = {jxs_dlw} + {abs(ty_value)} - 101 = {ky}")
                
                if ky >= len(ace.xss_data):
                    if debug: logger.debug(f"Warning: KY={ky} out of range for XSS data of length {len(ace.xss_data)}")
                    continue
                
                try:
                    yield_data = parse_energy_dependent_yield(ace, ky)
                    result.neutron_yields[mt_value] = yield_data
                    if debug: 
                        logger.debug(f"Successfully parsed energy-dependent yield data for MT={mt_value}")
                        if yield_data and hasattr(yield_data, 'energies') and len(yield_data.energies) > 0:
                            num_points = len(yield_data.energies)
                            logger.debug(f"Yield has {num_points} energy points")
                            if num_points > 0:
                                first_e = yield_data.energies[0]
                                last_e = yield_data.energies[-1]
                                logger.debug(f"Energy range: {first_e} to {last_e} MeV")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing energy-dependent yield for MT={mt_value}: {e}")
    
    if debug:
        logger.debug(f"Final incident_neutron keys: {list(result.incident_neutron.keys())}")
        logger.debug(f"Types of keys: {[type(k) for k in result.incident_neutron.keys()]}")
    
    if debug:
        logger.debug("Finished read_dlw_block")


def read_dlwp_block(ace: Ace, result: EnergyDistributionContainer, jxs_dlwp: int, debug: bool = False) -> None:
    """
    Read the DLWP block for photon energy distributions.
    
    According to Table 30, the energy distribution for reaction MT starts at:
    JED + LOCC - 1, where JED is JXS(19) and LOCC is provided by the LDLWP block.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    jxs_dlwp : int
        JXS(19) - Starting index of the DLWP block in the XSS array
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug: logger.debug(f"Starting read_dlwp_block at index {jxs_dlwp}")
    if not ace.reaction_mt_data or not ace.reaction_mt_data.has_photon_production_mt_data:
        if debug: logger.debug("Exiting read_dlwp_block: no photon production MT data available")
        return
    
    photon_mts = ace.reaction_mt_data.photon_production
    locators = ace.energy_distribution_locators.photon_production
    
    if debug: logger.debug(f"Found {len(photon_mts)} photon MT numbers and {len(locators)} locators")
    
    if len(photon_mts) != len(locators) and debug:
        logger.debug(f"Warning: Mismatch between photon MT numbers ({len(photon_mts)}) and locators ({len(locators)})")
    
    for i, mt in enumerate(photon_mts):
        if i >= len(locators):
            if debug: logger.debug(f"Skipping MT={mt}, index {i} out of range for locators array")
            continue
            
        locator = locators[i]
        locator_value = int(locator)
        
        mt_value = int(mt)
        
        if locator_value <= 0:
            if debug: logger.debug(f"Skipping MT={mt_value}, invalid locator: {locator_value}")
            continue
        
        # According to Table 30, the energy distribution starts at JED + LOCC - 1
        offset = jxs_dlwp + locator_value - 1  # Subtract 1 to match the documentation
        
        if debug: logger.debug(f"Processing MT={mt_value}, locator={locator_value}, offset={offset} (XSS length: {len(ace.xss_data)})")
        
        if offset >= len(ace.xss_data):
            if debug: logger.debug(f"Warning: Offset {offset} out of range for XSS data of length {len(ace.xss_data)}")
            continue
        
        try:
            if debug: logger.debug(f"Calling read_energy_distribution for MT={mt_value}")
            distributions = read_energy_distribution(ace, offset, jxs_dlwp, debug=debug)
            if distributions:
                if debug: logger.debug(f"Successfully read {len(distributions)} distributions for MT={mt_value}")
                result.photon_production[mt_value] = distributions
            elif debug: logger.debug(f"No distributions found for MT={mt_value}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing photon energy distribution for MT={mt_value}: {e}")
    
    if debug:
        logger.debug("Finished read_dlwp_block")


def read_dlwh_block(ace: Ace, result: EnergyDistributionContainer, debug: bool = False) -> None:
    """
    Read the DLWH block for other particle energy distributions.
    
    According to Table 30, the energy distribution for reaction MT starts at:
    JED + LOCC - 1, where JED is XSS(JXS(32) + 10*(i-1) + 8) and LOCC is 
    provided by the LDLWH block.
    
    Note: According to the documentation, energy-dependent neutron yields are only 
    available in the DLW and DLWP blocks, not in the DLWH block.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug: logger.debug("Starting read_dlwh_block")
    if not ace.reaction_mt_data or not ace.reaction_mt_data.has_particle_production_mt_data:
        if debug: logger.debug("Exiting read_dlwh_block: no particle production MT data available")
        return
    
    jxs32_idx = ace.header.jxs_array[32]
    if jxs32_idx <= 0:
        if debug: logger.debug(f"Invalid JXS(32) index: {jxs32_idx}")
        return
    
    jxs31_idx = ace.header.jxs_array[31]
    if jxs31_idx <= 0:
        if debug: logger.debug(f"Invalid JXS(31) index: {jxs31_idx}")
        return
    
    if debug: logger.debug(f"JXS(31) index: {jxs31_idx}, JXS(32) index: {jxs32_idx}")
    
    num_particle_types = ace.header.num_particle_types
    if debug: logger.debug(f"Number of particle types: {num_particle_types}")
    result.particle_production = [{} for _ in range(num_particle_types)]
    
    for i in range(num_particle_types):
        if debug: logger.debug(f"Processing particle type {i+1}")
        if i >= len(ace.energy_distribution_locators.particle_production):
            if debug: logger.debug(f"Skipping particle type {i+1}: no locators available")
            continue
            
        locators = ace.energy_distribution_locators.particle_production[i]
        if len(locators) == 0:
            if debug: logger.debug(f"Skipping particle type {i+1}: empty locators list")
            continue
            
        particle_mts = ace.reaction_mt_data.particle_production[i] if i < len(ace.reaction_mt_data.particle_production) else []
        if len(particle_mts) == 0 or len(particle_mts) != len(locators):
            if debug: logger.debug(f"Issue with particle type {i+1}: MT count={len(particle_mts)}, locator count={len(locators)}")
            continue
        
        jed_idx = jxs32_idx + 10 * i + 8 
        if jed_idx >= len(ace.xss_data):
            if debug: logger.debug(f"JED index {jed_idx} out of range for XSS data of length {len(ace.xss_data)}")
            continue
            
        jed = int(ace.xss_data[jed_idx])
        if debug: logger.debug(f"Particle type {i+1} JED={jed}, calculated at index {jed_idx}")
        
        for j, mt in enumerate(particle_mts):
            locator = locators[j]
            locator_value = int(locator)
            
            mt_value = int(mt)
            
            if locator_value <= 0:
                if debug: logger.debug(f"Skipping MT={mt_value}, invalid locator: {locator_value}")
                continue
            
            # According to Table 30, the energy distribution starts at JED + LOCC - 1
            offset = jed + locator_value - 1  # Subtract 1 to match the documentation
            
            if debug: logger.debug(f"Processing MT={mt_value}, locator={locator_value}, JED={jed}, offset={offset} (XSS length: {len(ace.xss_data)})")
            
            if offset >= len(ace.xss_data):
                if debug: logger.debug(f"Warning: Offset {offset} out of range for XSS data of length {len(ace.xss_data)}")
                continue
            
            try:
                if debug: logger.debug(f"Calling read_energy_distribution for MT={mt_value}")
                distributions = read_energy_distribution(ace, offset, jed, debug=debug)
                if distributions:
                    if debug: logger.debug(f"Successfully read {len(distributions)} distributions for MT={mt_value}")
                    result.particle_production[i][mt_value] = distributions
                elif debug: logger.debug(f"No distributions found for MT={mt_value}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing particle energy distribution for particle {i+1}, MT={mt_value}: {e}")
    
    if debug:
        logger.debug("Finished read_dlwh_block")


def read_dned_block(ace: Ace, result: EnergyDistributionContainer, jxs_dned: int, debug: bool = False) -> None:
    """
    Read the DNED block for delayed neutron energy distributions.
    
    According to Table 30, the energy distribution for delayed neutron group starts at:
    JED + LOCC - 1, where JED is JXS(27) and LOCC is provided by the DNEDL block.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    jxs_dned : int
        JXS(27) - Starting index of the DNED block in the XSS array
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug: logger.debug(f"Starting read_dned_block at index {jxs_dned}")
    if not ace.energy_distribution_locators.has_delayed_neutron_data:
        if debug: logger.debug("Exiting read_dned_block: no delayed neutron data available")
        return
    
    locators = ace.energy_distribution_locators.delayed_neutron
    num_groups = len(locators)
    if debug: logger.debug(f"Found {num_groups} delayed neutron groups")
    
    for i in range(num_groups):
        locator = locators[i]
        locator_value = int(locator)
        
        if locator_value <= 0:
            if debug: logger.debug(f"Skipping group {i+1}, invalid locator: {locator_value}")
            continue
        
        # According to Table 30, the energy distribution starts at JED + LOCC - 1
        offset = jxs_dned + locator_value - 1  # Subtract 1 to match the documentation
        
        if debug: logger.debug(f"Processing group {i+1}, locator={locator_value}, offset={offset} (XSS length: {len(ace.xss_data)})")
        
        if offset >= len(ace.xss_data):
            if debug: logger.debug(f"Warning: Offset {offset} out of range for XSS data of length {len(ace.xss_data)}")
            continue
        
        try:
            if debug: logger.debug(f"Calling read_energy_distribution for group {i+1}")
            distributions = read_energy_distribution(ace, offset, jxs_dned, debug=debug)
            if distributions and len(distributions) > 0:
                result.delayed_neutron.append(distributions[0])
                if debug: logger.debug(f"Successfully read distribution for group {i+1}")
            elif debug: logger.debug(f"No distributions found for group {i+1}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing delayed neutron energy distribution for group {i+1}: {e}")
    
    if debug: logger.debug("Finished read_dned_block")


def read_energy_distribution(ace: Ace, offset: int, base_jed: int, debug: bool = False) -> List[EnergyDistribution]:
    """
    Read energy distribution data starting at the given offset in the XSS array.
    
    According to Table 31, each energy distribution consists of:
    - LNW: Locator for the next law (0 if this is the last one)
    - LAW: Law number for the distribution
    - IDAT: Locator for the law data (relative to JED)
    - NR: Number of interpolation regions to define law applicability
    - NBT, INT arrays: ENDF interpolation parameters
    - NE: Number of energies for law applicability
    - E, P arrays: Energy points and probability for law validity
    
    For multiple laws:
    - If LNW=0, this is the last law (always used regardless of other conditions)
    - For a given incident energy E, LAW is used only if random_number < P(E)
    - If E < E(1), use P(1)
    - If E > E(NE), use P(NE)
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    offset : int
        Starting index in the XSS array (JED + LOCC - 1)
    base_jed : int
        Base JED index to be used for absolute position calculations
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    List[EnergyDistribution]
        List of energy distribution objects
        
    Raises
    ------
    EnergyDistributionParseError
        If there are issues with data format or indexing
    """
    distributions = []
    
    if debug:
        logger.debug(f"\n=== READING ENERGY DISTRIBUTION ===")
        logger.debug(f"Starting at offset={offset}, base_jed={base_jed}")
    
    if offset >= len(ace.xss_data):
        raise EnergyDistributionParseError(f"Initial offset {offset} out of bounds for XSS data with length {len(ace.xss_data)}")
    
    current_offset = offset
    law_count = 0
    
    while current_offset < len(ace.xss_data):
        law_count += 1
        if debug:
            logger.debug(f"\nParsing law #{law_count} at offset {current_offset}")
        
        try:
            # Read LNW, LAW, IDAT values
            lnw_entry = ace.xss_data[current_offset]
            law_entry = ace.xss_data[current_offset + 1]
            idat_entry = ace.xss_data[current_offset + 2]
            
            lnw = int(lnw_entry)
            law = int(law_entry)
            idat = int(idat_entry)
            
            if debug:
                logger.debug(f"Table 31 values: LNW={lnw}, LAW={law}, IDAT={idat}")
                logger.debug(f"Read from indices: {current_offset}, {current_offset+1}, {current_offset+2}")
        except (IndexError, AttributeError, ValueError) as e:
            raise EnergyDistributionParseError(f"Failed to read LNW/LAW/IDAT at offset {current_offset}: {str(e)}")
        
        # Validate LAW value (should be one of the known laws)
        valid_laws = [1, 2, 3, 4, 5, 7, 9, 11, 22, 24, 44, 61, 66, 67]
        if law not in valid_laws:
            logger.debug(f"WARNING: Unexpected LAW value: {law}. Valid laws are: {valid_laws}")
        
        try:
            # Read NR (number of interpolation regions)
            n_r_entry = ace.xss_data[current_offset + 3]
            n_r = int(n_r_entry)
            if debug:
                logger.debug(f"NR={n_r} (number of interpolation regions)")
                
            idx = current_offset + 4
        except (IndexError, AttributeError, ValueError) as e:
            raise EnergyDistributionParseError(f"Failed to read NR at offset {current_offset + 3}: {str(e)}")
        
        # Read NBT, INT arrays if present
        nbt = []
        interp = []
        
        if n_r > 0:
            if idx + 2*n_r - 1 >= len(ace.xss_data):
                raise EnergyDistributionParseError(
                    f"Not enough data for NBT/INT arrays. Need indices up to {idx + 2*n_r - 1}, have {len(ace.xss_data)}."
                )
            
            try:
                nbt = [int(ace.xss_data[idx + i]) for i in range(n_r)]
                idx += n_r
                interp = [int(ace.xss_data[idx + i]) for i in range(n_r)]
                idx += n_r
                
                if debug:
                    logger.debug(f"NBT={nbt}, INT={interp}")
            except (IndexError, AttributeError, ValueError) as e:
                raise EnergyDistributionParseError(f"Failed to read NBT/INT arrays: {str(e)}")
        
        # Read NE (number of energies for applicability)
        try:
            if idx >= len(ace.xss_data):
                raise EnergyDistributionParseError(f"Index {idx} out of bounds for XSS data when reading NE")
                
            n_e_entry = ace.xss_data[idx]
            n_e = int(n_e_entry)
            idx += 1
            
            if debug:
                logger.debug(f"NE={n_e} (number of energies for applicability)")
        except (IndexError, AttributeError, ValueError) as e:
            raise EnergyDistributionParseError(f"Failed to read NE at index {idx-1}: {str(e)}")
        
        # Read E, P arrays
        energies = []
        probabilities = []
        
        if n_e > 0:
            if idx + 2*n_e - 1 >= len(ace.xss_data):
                raise EnergyDistributionParseError(
                    f"Not enough data for E/P arrays. Need indices up to {idx + 2*n_e - 1}, have {len(ace.xss_data)}."
                )
            
            try:
                energies = ace.xss_data[idx:idx + n_e]
                idx += n_e
                probabilities = ace.xss_data[idx:idx + n_e]
                idx += n_e
                
                if debug and n_e > 0:
                    energy_values = energies.tolist()
                    prob_values = probabilities.tolist()
                    logger.debug(f"E range: [{energy_values[0]:.5e} - {energy_values[-1]:.5e}]")
                    logger.debug(f"P range: [{prob_values[0]:.5e} - {prob_values[-1]:.5e}]")
            except (IndexError, AttributeError) as e:
                raise EnergyDistributionParseError(f"Failed to read E/P arrays: {str(e)}")
        
        # Calculate absolute position using the base JED
        idat_absolute = base_jed + idat - 1
        
        if debug: 
            logger.debug(f"IDAT_absolute = base_jed + idat - 1 = {base_jed} + {idat} - 1 = {idat_absolute}")
            if idat_absolute < len(ace.xss_data):
                logger.debug(f"Value at IDAT_absolute: {ace.xss_data[idat_absolute]}")
        
        if idat_absolute >= len(ace.xss_data):
            raise EnergyDistributionParseError(
                f"IDAT_absolute={idat_absolute} out of bounds for XSS data with length {len(ace.xss_data)}"
            )
        
        # Create the energy distribution object
        distribution = create_energy_distribution(
            ace, law, idat, idat_absolute, 
            energies, probabilities, nbt, interp,
            debug=debug
        )
        
        if distribution:
            # Add relevant JXS values to help Law 61 parser
            if hasattr(distribution, 'jxs_dlw'):
                distribution.jxs_dlw = base_jed
            
            distributions.append(distribution)
            if debug: logger.debug(f"Successfully created distribution for law {law}")
        else:
            if debug: logger.debug(f"No distribution created for law {law}")
        
        # Check if this is the last law
        if lnw == 0:
            if debug: logger.debug("Reached last law (LNW=0)")
            break
            
        # Calculate position of next law using base_jed
        next_offset = base_jed + lnw - 1
        
        if debug:  
            logger.debug(f"Next law at: base_jed + lnw - 1 = {base_jed} + {lnw} - 1 = {next_offset}")
        
        if next_offset <= current_offset:
            raise EnergyDistributionParseError(
                f"Invalid next law offset {next_offset} <= current offset {current_offset}. Possible infinite loop."
            )
            
        if next_offset >= len(ace.xss_data):
            raise EnergyDistributionParseError(
                f"Next law offset {next_offset} out of bounds for XSS data with length {len(ace.xss_data)}"
            )
        
        current_offset = next_offset
    
    if debug: 
        logger.debug(f"\nFinished reading distributions. Found {len(distributions)} laws")
    
    return distributions

def create_energy_distribution(
    ace: Ace, law: int, idat: int, idat_absolute: int, 
    applicability_energies: np.ndarray, 
    applicability_probabilities: np.ndarray,
    nbt: List[int], interp: List[int],
    debug: bool = False
) -> Optional[EnergyDistribution]:
    """
    Create an energy distribution object based on the law number.
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    law : int
        Law number for the distribution
    idat : int
        Locator for the distribution data (relative to JED)
    idat_absolute : int
        Absolute index in XSS array for the law data 
    applicability_energies : np.ndarray
        Energy points for law applicability (view of xss_data)
    applicability_probabilities : np.ndarray
        Probability of law validity at each energy point
    nbt : List[int]
        NBT interpolation parameters
    interp : List[int]
        INT interpolation scheme parameters
    debug: bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    Optional[EnergyDistribution]
        Energy distribution object, or None if the law is not supported
        
    Raises
    ------
    EnergyDistributionParseError
        If there are issues with the parsing process
    """
    if debug: 
        logger.debug(f"\n--- Creating energy distribution for law {law} ---")
        logger.debug(f"IDAT={idat}, IDAT_absolute={idat_absolute}")
    
    # Create base distribution object
    base_distribution = EnergyDistribution(
        law=law, 
        idat=idat
    )
    base_distribution.applicability_energies = applicability_energies
    base_distribution.applicability_probabilities = applicability_probabilities
    base_distribution.applicability_nbt = nbt
    base_distribution.applicability_interp = interp
    

    # Store the actual JED value - this should always be used
    base_distribution.jed = idat_absolute - idat + 1  # This is the actual JED value

    ## Store the JXS values for Law 61
    #try:
    #    base_distribution.jxs_dlw = ace.header.jxs_array[11]  # JXS(11) for neutron reactions
    #    base_distribution.jxs_dlwp = ace.header.jxs_array[19]  # JXS(19) for photon-producing reactions
    #except (IndexError, AttributeError) as e:
    #    if debug:
    #        logger.debug(f"Warning: Could not read JXS values for distribution: {str(e)}")
    
    # Call the appropriate parser based on the law number
    try:
        if law == 1:
            if debug: logger.debug("Parsing tabular energy distribution (Law 1)")
            return parse_tabular_energy_distribution(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 2:
            if debug: logger.debug("Parsing discrete energy distribution (Law 2)")
            return parse_discrete_energy_distribution(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 3:
            if debug: logger.debug("Parsing level scattering (Law 3)")
            return parse_level_scattering(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 4:
            if debug: logger.debug("Parsing continuous energy-angle distribution (Law 4)")
            return parse_continuous_energy_angle_distribution(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 5:
            if debug: logger.debug("Parsing general evaporation spectrum (Law 5)")
            return parse_general_evaporation_spectrum(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 7:
            if debug: logger.debug("Parsing Maxwell fission spectrum (Law 7)")
            return parse_maxwell_fission_spectrum(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 9:
            if debug:  logger.debug("Parsing evaporation spectrum (Law 9)")
            return parse_evaporation_spectrum(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 11:
            if debug: logger.debug("Parsing energy-dependent Watt spectrum (Law 11)")
            return parse_energy_dependent_watt_spectrum(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 22:
            if debug: logger.debug("Parsing tabular linear functions (Law 22)")
            return parse_tabular_linear_functions(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 24:
            if debug: logger.debug("Parsing tabular energy multipliers (Law 24)")
            return parse_tabular_energy_multipliers(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 44:
            if debug: logger.debug("Parsing Kalbach-Mann distribution (Law 44)")
            return parse_kalbach_mann_distribution(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 61:
            if debug: logger.debug("Parsing tabulated angle-energy distribution (Law 61)")
            return parse_tabulated_angle_energy_distribution(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 66:
            if debug: logger.debug("Parsing N-body phase space distribution (Law 66)")
            return parse_nbody_phase_space_distribution(ace, base_distribution, idat_absolute, debug=debug)
        elif law == 67:
            if debug: logger.debug("Parsing laboratory angle-energy distribution (Law 67)")
            return parse_laboratory_angle_energy_distribution(ace, base_distribution, idat_absolute, debug=debug)
        else:
            if debug: logger.debug(f"No specific parser for law {law}, returning base distribution")
            return base_distribution
    except Exception as e:
        raise EnergyDistributionParseError(f"Error parsing law {law} at absolute index {idat_absolute}: {str(e)}")
