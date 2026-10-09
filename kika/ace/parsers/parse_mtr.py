from typing import List, Optional
import logging
from kika.ace.classes.mt_reaction.mtr import ReactionMTData

# Setup logger
logger = logging.getLogger(__name__)

def read_mtr_blocks(ace, debug=False):
    """
    Read MTR, MTRP, and MTRH blocks from the XSS array if they exist.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    ReactionMTData
        The MT reaction data object
    """
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        raise ValueError("Cannot read MTR blocks: header or XSS data missing")
    
    # Create a new ReactionMTData object
    reaction_mt_data = ReactionMTData()
    
    # Read MTR block (neutron reaction MT numbers) if present
    mtr_idx = ace.header.jxs_array[3]  # JXS(3)
    if mtr_idx > 0:
        num_reactions = ace.header.nxs_array[4]  # NXS(4)
        
        if debug:
            logger.debug("\n===== MTR BLOCK PARSING =====")
            logger.debug(f"JXS(3) = {mtr_idx} → Starting index of MTR block")
            logger.debug(f"NXS(4) = {num_reactions} → Total number of reactions")
        
        if num_reactions > 0:
            if (mtr_idx + num_reactions <= len(ace.xss_data)):
                reaction_mt_data.incident_neutron = ace.xss_data[mtr_idx:mtr_idx + num_reactions]
                
                if debug:
                    logger.debug(f"Read {len(reaction_mt_data.incident_neutron)} MT values from MTR block")
                    logger.debug(f"MT values: {[int(entry) for entry in reaction_mt_data.incident_neutron]}")
                
                # Determine reactions with secondary neutrons
                # First NXS(5) values excluding elastic scattering (MT=2)
                num_secondary_neutron_reactions = ace.header.nxs_array[5]  # NXS(5)
                
                if debug:
                    logger.debug(f"NXS(5) = {num_secondary_neutron_reactions} → Number of secondary neutron reactions")
                
                # Filter out elastic scattering (MT=2) from the secondary neutron reactions
                mts = reaction_mt_data.incident_neutron
                reaction_mt_data.secondary_neutron_mt = (
                    mts[mts != 2][:num_secondary_neutron_reactions]  # Limit to NXS(5) entries
                )
                
                if debug:
                    logger.debug(f"Secondary neutron MT numbers: {[int(entry) for entry in reaction_mt_data.secondary_neutron_mt]}")
    
    # Read MTRP block (photon production MT numbers) if present
    mtrp_idx = ace.header.jxs_array[13]  # JXS(13)
    if mtrp_idx > 0:
        num_photon_reactions = ace.header.nxs_array[6]  # NXS(6)
        
        if debug:
            logger.debug("\n===== MTRP BLOCK PARSING =====")
            logger.debug(f"JXS(13) = {mtrp_idx} → Starting index of MTRP block")
            logger.debug(f"NXS(6) = {num_photon_reactions} → Number of photon production reactions")
        
        if num_photon_reactions > 0:
            if mtrp_idx + num_photon_reactions <= len(ace.xss_data):
                reaction_mt_data.photon_production = ace.xss_data[mtrp_idx:mtrp_idx + num_photon_reactions]
                
                if debug:
                    logger.debug(f"Read {len(reaction_mt_data.photon_production)} MT values from MTRP block")
                    logger.debug(f"Photon production MT values: {[int(entry) for entry in reaction_mt_data.photon_production]}")
    
    # Read MTRH block (particle production MT numbers) if present
    jxs31 = ace.header.jxs_array[31]  # JXS(31)
    jxs32 = ace.header.jxs_array[32]  # JXS(32)
    num_particle_types = ace.header.nxs_array[7]  # NXS(7)
    
    if debug:
        logger.debug("\n===== MTRH BLOCK PARSING =====")
        logger.debug(f"JXS(31) = {jxs31} → Locator for NMT values")
        logger.debug(f"JXS(32) = {jxs32} → Locator for MTRH block")
        logger.debug(f"NXS(7) = {num_particle_types} → Number of particle types")
    
    if jxs31 > 0 and jxs32 > 0 and num_particle_types > 0:
        # Initialize list for each particle type
        reaction_mt_data.particle_production = [[] for _ in range(num_particle_types)]
        
        # Process each particle type
        for i_python in range(num_particle_types):
            # Convert to FORTRAN 1-based indexing for the formula
            i = i_python + 1  # This is the i in the formulas from the documentation
            
            if debug:
                logger.debug(f"\nProcessing particle type {i} (Python index {i_python}):")
            
            # Get the number of MT numbers for this particle type
            # NMT = XSS(JXS(31)+i−1) in FORTRAN / Table 9
            nmt_idx = jxs31 + (i - 1)
            
            if debug:
                logger.debug(f"  NMT index = JXS(31) + (i-1) = {jxs31} + ({i}-1) = {nmt_idx}")
            
            if nmt_idx >= len(ace.xss_data):
                error_msg = f"MTRH particle type {i} NMT index {nmt_idx} is out of bounds"
                if debug:
                    logger.debug(f"  ERROR: {error_msg}")
                raise IndexError(error_msg)
                
            nmt = int(ace.xss_data[nmt_idx])
            
            if debug:
                logger.debug(f"  NMT = XSS[{nmt_idx}] = {nmt} → Number of MT reactions for this particle")
            
            if nmt <= 0:
                if debug:
                    logger.debug(f"  Skipping particle type {i}: NMT={nmt} ≤ 0")
                continue  # Skip if no reactions for this particle
            
            # Get the starting index for the MT numbers
            # LMT = XSS(JXS(32)+10*(i−1)+1) in FORTRAN / Table 9
            offset = 10*(i-1)
            lmt_idx_ptr = jxs32 + offset + 1  # Fixed: Added the missing +1 to match the formula
            
            if debug:
                logger.debug(f"  LMT pointer = JXS(32) + 10*(i-1) + 1 = {jxs32} + {offset} + 1 = {lmt_idx_ptr}")
            
            if lmt_idx_ptr >= len(ace.xss_data):
                error_msg = f"MTRH particle type {i} LMT pointer index {lmt_idx_ptr} is out of bounds"
                if debug:
                    logger.debug(f"  ERROR: {error_msg}")
                raise IndexError(error_msg)
                
            lmt = int(ace.xss_data[lmt_idx_ptr])
            
            if debug:
                logger.debug(f"  LMT = XSS[{lmt_idx_ptr}] = {lmt} → location of MT values")
        
            
            if debug:
                logger.debug(f"  LMT 0-indexed = {lmt}")
            
            if lmt < 0:
                error_msg = f"MTRH particle type {i} LMT value {lmt} is invalid (must be > 0)"
                if debug:
                    logger.debug(f"  ERROR: {error_msg}")
                raise ValueError(error_msg)
                
            if lmt + nmt > len(ace.xss_data):
                error_msg = f"MTRH particle type {i} would read past end of XSS array"
                if debug:
                    logger.debug(f"  ERROR: {error_msg}")
                raise IndexError(error_msg)
                
            try:
                # Read the MT numbers for this particle type
                mt_range = f"{lmt}:{lmt+nmt}"
                mt_values = ace.xss_data[lmt:lmt + nmt]
                
                if debug:
                    logger.debug(f"  Reading MT values from XSS[{mt_range}]: {[int(entry) for entry in mt_values]}")
                
                reaction_mt_data.particle_production[i_python] = mt_values
                
                if debug:
                    logger.debug(f"  Successfully parsed {len(mt_values)} MT values for particle type {i}")
            except (ValueError, TypeError) as e:
                error_msg = f"Error parsing MTRH block for particle type {i}: {str(e)}"
                if debug:
                    logger.debug(f"  ERROR: {error_msg}")
                raise ValueError(error_msg)
    elif debug:
        logger.debug("No MTRH block to process: JXS(31)={jxs31}, JXS(32)={jxs32}, NXS(7)={num_particle_types}")
    
    # Always return the ReactionMTData object
    return reaction_mt_data
