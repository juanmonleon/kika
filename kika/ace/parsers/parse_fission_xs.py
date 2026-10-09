import logging
from typing import Optional
from kika.ace.classes.ace import Ace
from kika.ace.classes.fission_xs import FissionCrossSection

# Setup logger
logger = logging.getLogger(__name__)

def read_fission_xs_block(ace: Ace, debug=False) -> Optional[FissionCrossSection]:
    """
    Read the FIS block containing total fission cross section data.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update with fission cross section data
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    Optional[FissionCrossSection]
        The fission cross section object if data exists, None otherwise
    """
    if debug:
        logger.debug("\n===== FIS BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Create a new FissionCrossSection object
    fission_xs = FissionCrossSection()
    fission_xs.has_data = False
    
    # Check if we have the necessary data
    if not ace.header or not ace.header.jxs_array or ace.xss_data is None or len(ace.xss_data) == 0:
        if debug:
            logger.debug("Skipping FIS block: required data missing")
        return fission_xs
    
    # Get FIS block index (JXS(21))
    fis_idx = ace.header.jxs_array[21]
    
    if debug:
        logger.debug(f"JXS(21) = {fis_idx} → Locator for FIS block")
    
    # Check if FIS block exists (JXS(21) ≠ 0)
    if fis_idx <= 0:
        if debug:
            logger.debug(f"No FIS block present: JXS(21)={fis_idx}")
        return fission_xs
    
    # Validate index is within bounds
    if fis_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid FIS block index: {fis_idx} >= {len(ace.xss_data)}")
        return fission_xs
    
    if debug:
        logger.debug(f"FIS block starts at index {fis_idx} (FORTRAN 1-indexed)")
    
    # Read the energy grid index (IE)
    energy_grid_index = int(ace.xss_data[fis_idx])
    
    if debug:
        logger.debug(f"Energy grid index (IE): {energy_grid_index}")
    
    # Read the number of consecutive entries (NE)
    if fis_idx + 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read number of entries")
        return fission_xs
    
    num_entries = int(ace.xss_data[fis_idx + 1])
    
    if debug:
        logger.debug(f"Number of cross section entries (NE): {num_entries}")
    
    if num_entries <= 0:
        if debug:
            logger.debug(f"ERROR: Invalid number of entries: {num_entries}")
        return fission_xs
    
    # Ensure we have enough data for the cross sections
    if fis_idx + 2 + num_entries - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data for cross sections: need {num_entries}, available {len(ace.xss_data) - (fis_idx + 2)}")
        return fission_xs
    
    # Read the cross section values (view of xss_data)
    cross_sections = ace.xss_data[fis_idx + 2:fis_idx + 2 + num_entries]
    
    # Store the data
    fission_xs.energy_grid_index = energy_grid_index
    fission_xs.num_entries = num_entries
    fission_xs.cross_sections = cross_sections
    fission_xs.has_data = True
    
    if debug:
        logger.debug(f"Successfully read {num_entries} fission cross section values")
    
    return fission_xs
