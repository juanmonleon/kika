import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.cross_section.cross_section_locators import CrossSectionLocators

# Setup logger
logger = logging.getLogger(__name__)

def read_xs_locator_blocks(ace: Ace, debug=False):
    """
    Read LSIG, LSIGP, and LSIGH blocks from the XSS array if they exist.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    CrossSectionLocators
        The cross section locators object
    """
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        raise ValueError("Cannot read cross section locator blocks: header or XSS data missing")
    
    if debug:
        logger.debug("\n===== CROSS SECTION LOCATOR BLOCKS PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize xs_locators if it doesn't exist
    if ace.xs_locators is None:
        ace.xs_locators = CrossSectionLocators()
    
    # Read LSIG block (incident neutron cross section locators) if present
    lsig_idx = ace.header.jxs_array[6]  # JXS(6)
    
    if debug:
        logger.debug(f"JXS(6) = {lsig_idx} → Locator for LSIG block")
    
    if lsig_idx > 0:
        num_reactions = ace.header.nxs_array[4]  # NXS(4)
        
        if debug:
            logger.debug(f"NXS(4) = {num_reactions} → Total number of reactions")
        
        if num_reactions > 0:
            
            if debug:
                logger.debug(f"LSIG block range: XSS[{lsig_idx}:{lsig_idx+num_reactions}]")
            
            if (lsig_idx + num_reactions <= len(ace.xss_data)):
                ace.xs_locators.incident_neutron = ace.xss_data[lsig_idx:lsig_idx + num_reactions]
                
                if debug:
                    logger.debug(f"Successfully read {len(ace.xs_locators.incident_neutron)} incident neutron XS locators")
    
    # Read LSIGP block (photon production cross section locators) if present
    lsigp_idx = ace.header.jxs_array[14]  # JXS(14)
    
    if debug:
        logger.debug("\n----- LSIGP Block -----")
        logger.debug(f"JXS(14) = {lsigp_idx} → Locator for LSIGP block")
    
    if lsigp_idx > 0:
        num_photon_reactions = ace.header.nxs_array[6]  # NXS(6)
        
        if debug:
            logger.debug(f"NXS(6) = {num_photon_reactions} → Number of photon production reactions")
        
        if num_photon_reactions > 0:
            
            if debug:
                logger.debug(f"LSIGP block range: XSS[{lsigp_idx}:{lsigp_idx+num_photon_reactions}]")
            
            if lsigp_idx + num_photon_reactions <= len(ace.xss_data):
                ace.xs_locators.photon_production = ace.xss_data[lsigp_idx:lsigp_idx + num_photon_reactions]
                
                if debug:
                    logger.debug(f"Successfully read {len(ace.xs_locators.photon_production)} photon production XS locators")
    
    # Read LSIGH block (particle production cross section locators) if present
    jxs31 = ace.header.jxs_array[31]  # JXS(31)
    jxs32 = ace.header.jxs_array[32]  # JXS(32)
    num_particle_types = ace.header.nxs_array[7]  # NXS(7)
    
    if debug:
        logger.debug("\n----- LSIGH Block -----")
        logger.debug(f"JXS(31) = {jxs31} → Locator for NMT values (FORTRAN 1-indexed)")
        logger.debug(f"JXS(32) = {jxs32} → Locator for LSIGH block (FORTRAN 1-indexed)")
        logger.debug(f"NXS(7) = {num_particle_types} → Number of particle types")
    
    if jxs31 > 0 and jxs32 > 0 and num_particle_types > 0:
        # Initialize list for each particle type
        ace.xs_locators.particle_production = [[] for _ in range(num_particle_types)]
        
        # Process each particle type
        for i_python in range(num_particle_types):
            # Convert to FORTRAN 1-based indexing for the formula
            i = i_python + 1
            
            if debug:
                logger.debug(f"\nProcessing particle type {i} (Python index {i_python}):")
            
            # Get the number of MT numbers for this particle type
            # NMT = XSS(JXS(31)+i-1) in FORTRAN
            nmt_idx = jxs31 + (i - 1)  
            
            if debug:
                logger.debug(f"  NMT index calculation: jxs31 + (i-1) = {jxs31} + ({i}-1) = {nmt_idx}")
            
            if nmt_idx >= len(ace.xss_data):
                if debug:
                    logger.debug(f"  ERROR: NMT index {nmt_idx} is out of bounds ({len(ace.xss_data)})")
                continue
                
            nmt = int(ace.xss_data[nmt_idx])
            
            if debug:
                logger.debug(f"  NMT = XSS[{nmt_idx}] = {nmt} → Number of MT reactions for this particle")
            
            if nmt <= 0:
                if debug:
                    logger.debug(f"  Skipping particle type {i}: NMT={nmt} ≤ 0")
                continue  # Skip if no reactions for this particle
            
            # Get the starting index for the XS locators
            # LSIGH = XSS(JXS(32)+10*(i-1)+3) in FORTRAN
            offset = 10*(i-1) + 3
            lsigh_idx_ptr = jxs32 + 10*(i-1) + 3  # Adjusted for Python indexing
            
            if debug:
                logger.debug(f"  LSIGH pointer calculation: jxs32 + 10*(i-1) + 3 = {jxs32} + 10*({i}-1) + 3 = {jxs32} + {offset} = {lsigh_idx_ptr}")
            
            if lsigh_idx_ptr >= len(ace.xss_data):
                if debug:
                    logger.debug(f"  ERROR: LSIGH pointer {lsigh_idx_ptr} is out of bounds ({len(ace.xss_data)})")
                continue
                
            lsigh = int(ace.xss_data[lsigh_idx_ptr])
            
            if debug:
                logger.debug(f"  LSIGH = XSS[{lsigh_idx_ptr}] = {lsigh} → 1-indexed location of XS locators")

            
            if lsigh < 0:
                if debug:
                    logger.debug(f"  ERROR: LSIGH value {lsigh} is invalid (must be > 0)")
                continue
                
            if lsigh + nmt > len(ace.xss_data):
                if debug:
                    logger.debug(f"  ERROR: LSIGH would read past end of XSS array: {lsigh+nmt} > {len(ace.xss_data)}")
                continue
                
            xs_locators = ace.xss_data[lsigh:lsigh + nmt]
            ace.xs_locators.particle_production[i_python] = xs_locators
            
            if debug:
                logger.debug(f"  Successfully read {len(xs_locators)} XS locators for particle type {i}")
    elif debug:
        logger.debug(f"No LSIGH block to process: JXS(31)={jxs31}, JXS(32)={jxs32}, NXS(7)={num_particle_types}")
    
    # Return the xs_locators object
    return ace.xs_locators
