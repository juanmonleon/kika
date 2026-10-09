import logging
from kika.ace.classes.delayed_neutron.delayed_neutron import DelayedNeutronPrecursor, DelayedNeutronData
from typing import Tuple

import numpy as np

# Setup logger
logger = logging.getLogger(__name__)

def read_delayed_neutron_data(ace, debug=False):
    """
    Read BDD block (delayed neutron precursor data) from the XSS array if it exists.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    DelayedNeutronData
        The delayed neutron data object
    """
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.xss_data is None):
        raise ValueError("Cannot read delayed neutron data: header or XSS data missing")
    
    # Create a new DelayedNeutronData object
    delayed_neutron_data = DelayedNeutronData()
    
    if debug:
        logger.debug("\n===== DELAYED NEUTRON BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Read BDD block if present
    bdd_idx = ace.header.jxs_array[25]  # JXS(25)
    
    if debug:
        logger.debug(f"JXS(25) = {bdd_idx} → Locator for BDD block (FORTRAN 1-indexed)")
    
    if bdd_idx > 0:
        # Set flag indicating delayed neutron data is present
        delayed_neutron_data.has_delayed_neutron_data = True
        
        if debug:
            logger.debug(f"BDD block found at index {bdd_idx}")
        
        # Check bounds
        if bdd_idx >= len(ace.xss_data):
            raise ValueError(f"Invalid BDD block index: {bdd_idx}")
        
        # Get the number of precursor groups
        num_precursors = ace.header.nxs_array[8]  # NXS(8)
        
        if debug:
            logger.debug(f"Number of precursor groups: {num_precursors}")
        
        if num_precursors <= 0:
            raise ValueError(f"Invalid number of precursor groups: {num_precursors}")
        
        # Parse each precursor group
        current_idx = bdd_idx
        for i in range(num_precursors):
            if debug:
                logger.debug(f"Parsing precursor group {i+1} starting at index {current_idx}")
            
            # Parse precursor data and get the new index position
            precursor, current_idx = parse_precursor_data(ace.xss_data, current_idx, debug)
            delayed_neutron_data.precursors.append(precursor)
            
            if debug:
                logger.debug(f"Precursor group {i+1} parsed, new index: {current_idx}")
    else:
        if debug:
            logger.debug("No BDD block found (JXS(25) <= 0)")
    
    # Always return the DelayedNeutronData object
    return delayed_neutron_data

def parse_precursor_data(xss_data: np.ndarray, idx: int, debug=False) -> Tuple[DelayedNeutronPrecursor, int]:
    """
    Parse data for a single delayed neutron precursor group.
    
    Parameters
    ----------
    xss_data : np.ndarray
        The XSS data array
    idx : int
        Starting index in the XSS array for the precursor data
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    ------- 
    Tuple[DelayedNeutronPrecursor, int]
        A tuple containing the precursor data and the new index position
    """
    # Check valid index
    if idx <= 0 or idx >= len(xss_data):
        raise ValueError(f"Invalid precursor data index: {idx}")
    
    # Create a DelayedNeutronPrecursor object
    precursor = DelayedNeutronPrecursor()
    
    # Get the decay constant
    precursor.decay_constant = float(xss_data[idx])
    
    if debug:
        logger.debug(f"Decay constant: {precursor.decay_constant}")
    
    current_idx = idx + 1
    
    # Get the number of interpolation regions
    n_regions = int(xss_data[current_idx])
    
    if debug:
        logger.debug(f"Number of interpolation regions: {n_regions}")
    
    current_idx += 1
    
    # Read interpolation regions if present
    if n_regions > 0:
        # Read NBT array
        nbt = [int(x) for x in xss_data[current_idx:current_idx + n_regions]]
        current_idx += n_regions
        
        # Read INT array
        interp = [int(x) for x in xss_data[current_idx:current_idx + n_regions]]
        current_idx += n_regions
        
        # Store interpolation regions
        precursor.interpolation_regions = list(zip(nbt, interp))
        
        if debug:
            logger.debug(f"Interpolation regions: {precursor.interpolation_regions}")
    
    # Read number of energy points
    n_energies = int(xss_data[current_idx])
    
    if debug:
        logger.debug(f"Number of energy points: {n_energies}")
    
    current_idx += 1
    
    # Read energy points
    precursor.energies = xss_data[current_idx:current_idx + n_energies]
    current_idx += n_energies
    
    # Read probabilities
    precursor.probabilities = xss_data[current_idx:current_idx + n_energies]
    current_idx += n_energies
    
    return precursor, current_idx
