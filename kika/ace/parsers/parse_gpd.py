import logging
from typing import List
from kika.ace.classes.gpd import PhotonProductionData

# Setup logger
logger = logging.getLogger(__name__)

# Table 54: Discrete neutron energy boundaries 
NEUTRON_ENERGY_BOUNDARIES = [
    1.39e-10, 1.52e-7, 4.14e-7, 1.13e-6, 3.06e-6,
    8.32e-6, 2.26e-5, 6.14e-5, 1.67e-4, 4.54e-4,
    1.235e-3, 3.35e-3, 9.23e-3, 2.48e-2, 6.76e-2,
    0.184, 0.303, 0.500, 0.823, 1.353,
    1.738, 2.232, 2.865, 3.68, 6.07,
    7.79, 10.0, 12.0, 13.5, 15.0
]

def read_gpd_block(ace, debug=False):
    """
    Read GPD block (photon production cross section) from the XSS array if it exists.
    
    The GPD Block contains the total photon production cross section and may include
    outgoing photon energies in an obsolete 30×20 matrix format used in older datasets.
    
    This block only exists when:
    - JXS(12) ≠ 0 (GPD block exists)
    - JXS(13) = 0 (obsolete format indicator)
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    PhotonProductionData or None
        Photon production data object if GPD block exists, otherwise None
    """
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        if debug:
            logger.debug("Skipping GPD block: required data missing")
        return None
    
    if debug:
        logger.debug("\n===== GPD BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Check if GPD block exists: JXS(12) ≠ 0 and JXS(13) = 0
    gpd_idx = ace.header.jxs_array[12]
    jxs_13 = ace.header.jxs_array[13]
    
    if debug:
        logger.debug(f"JXS(12) = {gpd_idx} → Locator for GPD block (S_GPD)")
        logger.debug(f"JXS(13) = {jxs_13} → Must be 0 for obsolete GPD format")
    
    if gpd_idx <= 0 or jxs_13 != 0:
        if debug:
            logger.debug(f"No GPD block present: JXS(12)={gpd_idx}, JXS(13)={jxs_13}")
        return None
    
    # Initialize photon_production_data
    result = PhotonProductionData()
    
    if debug:
        logger.debug(f"GPD block starts at index {gpd_idx} (in XSS array)")
    
    # Get the number of energy points from the ESZ block (NES)
    n_energy = ace.header.num_energies
    
    if debug:
        logger.debug(f"Number of energy points (NES): {n_energy}")
    
    # Check if we have enough data for the total photon production cross section
    if gpd_idx + n_energy > len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: GPD block would extend beyond XSS array: {gpd_idx + n_energy} > {len(ace.xss_data)}")
        return None
    
    # Extract total photon production cross section (view of xss_data)
    # This is σ_γ(l), l = 1,…,NES from Table 53
    result.total_xs = ace.xss_data[gpd_idx:gpd_idx + n_energy]
    
    if debug:
        logger.debug(f"Successfully read {n_energy} total photon production XS values")
    
    # Check if outgoing photon energies are provided
    # According to Table 55, these start at S_GPD + NES (after the cross section data)
    outgoing_start = gpd_idx + n_energy
    outgoing_size = 30*20  # 30 groups with 20 energies each = 600 values
    
    if debug:
        logger.debug(f"Checking for outgoing photon energies at index {outgoing_start}")
        logger.debug(f"Need {outgoing_size} additional values for 30×20 matrix of outgoing energies")
    
    # There should be 30 groups with 20 energies each = 600 values
    if outgoing_start + outgoing_size <= len(ace.xss_data):
        if debug:
            logger.debug("Outgoing photon energies data found (obsolete 30×20 matrix format)")
        
        # Initialize the outgoing energies container
        outgoing_energies = []
        
        # According to Table 55, there are 30 groups of 20 equiprobable outgoing photon energies
        # Each group corresponds to an incident neutron energy range defined by NEUTRON_ENERGY_BOUNDARIES
        for i in range(30):  # 30 incident neutron energy groups
            start_idx = outgoing_start + i*20
            end_idx = start_idx + 20
            
            if debug and i == 0:
                logger.debug(f"Reading first energy group from XSS[{start_idx}:{end_idx}]")
            
            # Extract the 20 equiprobable outgoing photon energies for this neutron energy group
            # View of xss_data
            group_energies = ace.xss_data[start_idx:start_idx + 20]
            outgoing_energies.append(group_energies)
        
        result.outgoing_energies = outgoing_energies
        
        # Also store the neutron energy boundaries for reference
        result.neutron_energy_boundaries = NEUTRON_ENERGY_BOUNDARIES
        
        if debug:
            logger.debug(f"Successfully read outgoing photon energies for {len(outgoing_energies)} neutron energy groups")
    elif debug:
        logger.debug(f"No outgoing photon energies data (would require {outgoing_size} more values)")
    
    return result
