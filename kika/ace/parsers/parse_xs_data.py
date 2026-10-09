import logging

import numpy as np
from kika.ace.classes.cross_section.cross_section_data import CrossSectionData, ReactionCrossSection

# Setup logger
logger = logging.getLogger(__name__)

def read_xs_data_block(ace, debug=False):
    """
    Read SIG block (reaction cross sections) from the XSS array if it exists.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    CrossSectionData
        The cross section data object
    """
    # Fix the condition - don't check has_neutron_data yet since we might initialize xs_locators later
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        if debug:
            logger.debug("Skipping XS data block: required header or XSS data missing")
        return None  # Return None explicitly in error case
    
    # Now check for xs_locators separately
    if ace.xs_locators is None:
        if debug:
            logger.debug("Skipping XS data block: xs_locators is None")
        return None
        
    # Now we can check the property since we know xs_locators exists
    if not ace.xs_locators.has_neutron_data:
        if debug:
            logger.debug("Skipping XS data block: no neutron XS locator data available")
        return None
    
    if debug:
        logger.debug("\n===== CROSS SECTION DATA BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize cross_section if it doesn't exist
    if ace.cross_section is None:
        ace.cross_section = CrossSectionData()
    
    # Store energy grid for convenience
    if ace.esz_block and len(ace.esz_block.energies):
        ace.cross_section.set_energy_grid(ace.esz_block.energies)
    
    # Get the starting index for SIG block
    sig_idx = ace.header.jxs_array[7]  # JXS(7)
    
    if debug:
        logger.debug(f"JXS(7) = {sig_idx} → Starting index of SIG block")
    
    if sig_idx <= 0:
        if debug:
            logger.debug("No SIG block present (JXS(7) ≤ 0)")
        return None
    
    if debug:
        logger.debug(f"SIG block starts at index {sig_idx} (0-indexed)")
    
    # Get MT numbers from reaction_mt_data
    if not ace.reaction_mt_data or not ace.reaction_mt_data.has_neutron_mt_data:
        if debug:
            logger.debug("No MT data available for neutron reactions")
        return None
        
    mt_entries = ace.reaction_mt_data.incident_neutron
    locator_entries = ace.xs_locators.incident_neutron
    
    if debug:
        logger.debug(f"Found {len(mt_entries)} MT entries and {len(locator_entries)} locator entries")
    
    if len(mt_entries) != len(locator_entries):
        logger.warning(f"MTR has {len(mt_entries)} reactions but LSIG {len(locator_entries)} locators; SIG not read")
        return None
    
    # Process each reaction
    for i, (mt_entry, locator_entry) in enumerate(zip(mt_entries, locator_entries)):
        mt_value = int(mt_entry)
        locator_value = int(locator_entry)
        
        # Calculate absolute index - FIX: Subtract 1 to match documentation
        # According to Table 16: LXS + LOCA_i - 1
        abs_idx = sig_idx + locator_value - 1

        if debug:
            logger.debug(f"\nReaction {i+1}: MT={mt_value}")
            logger.debug(f"  Locator value: {locator_value}")
            logger.debug(f"  Absolute index: sig_idx + locator - 1 = {sig_idx} + {locator_value} - 1 = {abs_idx}")
        
        if abs_idx >= len(ace.xss_data):
            logger.warning(f"SIG of MT={mt_value}: Absolute index {abs_idx} is out of bounds ({len(ace.xss_data)})")
            continue
            
        try:
            # Read energy grid index and number of energies
            energy_idx = int(ace.xss_data[abs_idx])
            num_energies = int(ace.xss_data[abs_idx + 1])
            
            if debug:
                logger.debug(f"  Energy grid index from ACE: {energy_idx} (1-indexed FORTRAN style)")
                logger.debug(f"  Converting to 0-indexed for Python: {energy_idx-1}")
                logger.debug(f"  Number of energies: {num_energies}")
            
            # Convert from 1-indexed (FORTRAN style) to 0-indexed (Python style)
            python_energy_idx = energy_idx - 1
            
            # Validate that indices make sense
            if energy_idx <= 0:
                logger.warning(f"SIG of MT={mt_value}: Invalid energy index {energy_idx} (must be > 0)")
                continue
                
            if num_energies <= 0:
                logger.warning(f"SIG of MT={mt_value}: Invalid number of energies {num_energies} (must be > 0)")
                continue
                
            # Verify energy index doesn't exceed the energy grid size (use Python-style index for check)
            if python_energy_idx >= len(ace.esz_block.energies):
                logger.warning(f"SIG of MT={mt_value}: Energy index {energy_idx} (0-indexed: {python_energy_idx}) exceeds energy grid size {len(ace.esz_block.energies)}")
                continue
                
            # Check if num_energies would make the cross section extend beyond the energy grid
            if python_energy_idx + num_energies > len(ace.esz_block.energies):
                logger.warning(f"SIG of MT={mt_value} runs past the energy grid; truncated to the grid")
                if debug:
                    logger.debug(f"  WARNING: Cross section would extend beyond energy grid: "
                               f"start={energy_idx} (0-indexed: {python_energy_idx}), length={num_energies}, "
                               f"grid size={len(ace.esz_block.energies)}, end={python_energy_idx+num_energies}")
                    logger.debug(f"  Adjusting num_energies from {num_energies} to {len(ace.esz_block.energies) - python_energy_idx}")
                num_energies = len(ace.esz_block.energies) - python_energy_idx
            
            # Read cross section values
            xs_start = abs_idx + 2
            xs_end = xs_start + num_energies
            
            if debug:
                logger.debug(f"  XS data range: XSS[{xs_start}:{xs_end}]")
            
            if xs_end <= len(ace.xss_data):
                # A view: perturbing the reaction writes the XSS array
                xs_entries = ace.xss_data[xs_start:xs_end]
                
                # Get energy entries for this reaction
                if ace.esz_block and len(ace.esz_block.energies):
                    end_energy_idx = min(python_energy_idx + num_energies, len(ace.esz_block.energies))
                    energy_entries = ace.esz_block.energies[python_energy_idx:end_energy_idx]
                else:
                    energy_entries = np.empty(0)
                
                reaction_xs = ReactionCrossSection(
                    mt=mt_value,
                    energy_idx=python_energy_idx,
                    num_energies=num_energies,
                    _xs_entries=xs_entries,
                    _energy_entries=energy_entries
                )
                
                # Store using the integer MT value as the key for lookup
                ace.cross_section.reaction[mt_value] = reaction_xs
                
                if debug:
                    logger.debug(f"  Successfully read {len(xs_entries)} XS values for MT={mt_value}")
            else:
                logger.warning(f"SIG of MT={mt_value}: XS data would extend beyond XSS array: {xs_end} > {len(ace.xss_data)}")
        except (IndexError, ValueError) as e:
            # Skip reaction if there's an error
            logger.warning(f"SIG of MT={mt_value}: {str(e)}")
            continue
    
    if debug:
        logger.debug(f"\nSuccessfully processed {len(ace.cross_section.reaction)} reaction cross sections")
    
    # Return the cross_section object
    return ace.cross_section
