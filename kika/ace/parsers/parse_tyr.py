import logging
from kika.ace.classes.particle_release.particle_release import ParticleRelease

# Setup logger
logger = logging.getLogger(__name__)

def read_tyr_blocks(ace, debug=False, strict_validation=True):
    """
    Read TYR and TYRH blocks from the XSS array if they exist.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
    strict_validation : bool, optional
        If True, invalid TY values will raise an error If False, 
        invalid values will generate a warning but processing will continue.
        
    Returns
    -------
    ParticleRelease
        The particle release object
    """
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        raise ValueError("Cannot read TYR blocks: header or XSS data missing")
    
    # Initialize particle_release if it doesn't exist
    if ace.particle_release is None:
        ace.particle_release = ParticleRelease()
    
    if debug:
        logger.debug("\n===== TYR BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
        logger.debug(f"NXS array: {ace.header.nxs_array}")
        logger.debug(f"JXS array: {ace.header.jxs_array}")
    
    # Read TYR block (incident neutron reactions) if present
    tyr_idx = ace.header.jxs_array[5]  # JXS(5)
    num_reactions = ace.header.nxs_array[4]  # NXS(4)
    
    if debug:
        logger.debug(f"JXS(5) = {tyr_idx} → Starting index of TYR block")
        logger.debug(f"NXS(4) = {num_reactions} → Total number of reactions including elastic")
    
    # Initialize the incident_neutron list even if we don't find data
    # This prevents the 'NoneType' error
    ace.particle_release.incident_neutron = []
    
    if num_reactions > 0:
        # Make sure we handle files with only elastic scattering correctly
        if num_reactions == 1:
            if debug:
                logger.debug("Only elastic scattering present, no TYR block to read")
            return
            
        # Adjust for elastic scattering (not included in TYR)
        num_reactions_tyr = num_reactions - 1
        
        if debug:
            logger.debug(f"Reactions in TYR block: {num_reactions_tyr} (NXS(4)-1, elastic excluded)")
        
        if tyr_idx > 0 and num_reactions_tyr > 0:
            
            if debug:
                logger.debug(f"TYR starting index: {tyr_idx}")
                logger.debug(f"XSS array length: {len(ace.xss_data)-1}") 
            
            # Double-check array bounds
            if tyr_idx >= len(ace.xss_data):
                error_msg = f"TYR block index {tyr_idx} is out of bounds for XSS array of length {len(ace.xss_data)-1}"
                logger.error(error_msg)
                raise IndexError(error_msg)
                
            # End index calculation
            end_idx = min(tyr_idx + num_reactions_tyr - 1, len(ace.xss_data) - 1)
            
            if debug:
                logger.debug(f"TYR block range: XSS[{tyr_idx}:{end_idx+1}]")
                logger.debug(f"First value at XSS[{tyr_idx}] = {ace.xss_data[tyr_idx]}")
            
            try:
                # Read the TYR block (view of xss_data)
                ty_entries = ace.xss_data[tyr_idx:end_idx+1]
                
                if debug:
                    logger.debug(f"TYR values read: {[int(entry) for entry in ty_entries]}")
                
                # Validate TY values
                for i, entry in enumerate(ty_entries):
                    ty_value = int(entry)
                    valid = _is_valid_ty_value(ty_value)
                    if debug:
                        logger.debug(f"  TY[{i+1}] = {ty_value} → {'VALID' if valid else 'INVALID'}")
                    if not valid:
                        error_msg = f"Invalid TY value {ty_value} at index {i+1} in TYR block"
                        if strict_validation:
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                        else:
                            logger.warning(f"{error_msg} - continuing with processing")
                        
                ace.particle_release.incident_neutron = ty_entries
                if debug:
                    logger.debug(f"Successfully parsed {len(ty_entries)} TY values for incident neutron reactions")
            except (ValueError, TypeError) as e:
                error_msg = f"Error parsing TYR block: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
    
    # Read TYRH block (particle production reactions) if present
    jxs31 = ace.header.jxs_array[31]  # JXS(31)
    jxs32 = ace.header.jxs_array[32]  # JXS(32)
    num_particle_types = ace.header.nxs_array[7]  # NXS(7)
    
    if debug:
        logger.debug("\n===== TYRH BLOCK PARSING =====")
        logger.debug(f"JXS(31) = {jxs31} → Locator for NMT values")
        logger.debug(f"JXS(32) = {jxs32} → Locator for TYRH block")
        logger.debug(f"NXS(7) = {num_particle_types} → Number of particle types")
    
    if jxs31 > 0 and jxs32 > 0 and num_particle_types > 0:
        # Initialize list for each particle type
        ace.particle_release.particle_production = [[] for _ in range(num_particle_types)]
    
        if debug:
            logger.debug(f"XSS array length = {len(ace.xss_data)-1}") 
        
        # Process each particle type (we'll use 1-based iteration to match ACE format)
        for i in range(1, num_particle_types + 1):
            i_python = i - 1  # Only for array access in our Python code
            
            if debug:
                logger.debug(f"\nProcessing particle type {i}:")
            
            # Get the number of MT numbers for this particle type
            # NMT = XSS(JXS(31)+i-1) in FORTRAN
            nmt_idx = jxs31 + i - 1  
            if debug:
                logger.debug(f"  NMT index calculation: jxs31 + (i-1) = {jxs31} + ({i}-1) = {nmt_idx}")
            
            if nmt_idx >= len(ace.xss_data):
                error_msg = f"TYRH particle type {i} NMT index {nmt_idx} is out of bounds ({len(ace.xss_data)-1})"
                logger.error(error_msg)
                raise IndexError(error_msg)
                
            nmt = int(ace.xss_data[nmt_idx])
            if debug:
                logger.debug(f"  NMT = XSS[{nmt_idx}] = {nmt} → Number of MT reactions for this particle")
            
            if nmt <= 0:
                if debug:
                    logger.debug(f"  Skipping particle type {i}: NMT={nmt} ≤ 0")
                continue  # Skip if no reactions for this particle
            
            # Get the starting index for the TY values
            # LTYR = XSS(JXS(32)+10*(i-1)+2) in FORTRAN
            offset = 10*(i-1) + 2
            ltyr_idx_ptr = jxs32 + offset  # Fixed: Removed incorrect "-1" subtraction
            
            if debug:
                logger.debug(f"  LTYR pointer calculation: jxs32 + 10*(i-1) + 2 = {jxs32} + 10*({i}-1) + 2 = {jxs32} + {offset} = {ltyr_idx_ptr}")
            
            if ltyr_idx_ptr >= len(ace.xss_data):
                error_msg = f"TYRH particle type {i} LTYR pointer index {ltyr_idx_ptr} is out of bounds ({len(ace.xss_data)-1})"
                logger.error(error_msg)
                raise IndexError(error_msg)
                
            ltyr = int(ace.xss_data[ltyr_idx_ptr])
            
            if debug:
                logger.debug(f"  LTYR = XSS[{ltyr_idx_ptr}] = {ltyr} → location of TY values")
            
            if debug:
                logger.debug(f"  LTYR = {ltyr}")
            
            if ltyr <= 0:
                error_msg = f"TYRH particle type {i} LTYR value {ltyr} is invalid (must be > 0)"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if ltyr + nmt - 1 >= len(ace.xss_data):
                error_msg = f"TYRH particle type {i} would read past end of XSS array: {ltyr+nmt-1} >= {len(ace.xss_data)}"
                logger.error(error_msg)
                raise IndexError(error_msg)
                
            # Read the TY values for this particle type
            try:
                ty_range = f"{ltyr}:{ltyr+nmt}"
                # View of xss_data
                ty_entries = ace.xss_data[ltyr:ltyr+nmt]
                
                if debug:
                    logger.debug(f"  Reading TY values from XSS[{ty_range}]")
                    logger.debug(f"  TY values: {[int(entry) for entry in ty_entries]}")
                
                # Validate TY values
                for j, entry in enumerate(ty_entries):
                    ty_value = int(entry)
                    valid = _is_valid_ty_value(ty_value)
                    if debug:
                        logger.debug(f"    TY[{j+1}] = {ty_value} → {'VALID' if valid else 'INVALID'}")
                    if not valid:
                        error_msg = f"Invalid TY value {ty_value} for particle type {i}, reaction {j+1}"
                        if strict_validation:
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                        else:
                            logger.warning(f"{error_msg} - continuing with processing")
                        
                ace.particle_release.particle_production[i_python] = ty_entries
                if debug:
                    logger.debug(f"  Successfully parsed {len(ty_entries)} TY values for particle type {i}")
            except (ValueError, TypeError) as e:
                error_msg = f"Error parsing TYRH block for particle type {i}: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
    elif debug:
        logger.debug(f"No TYRH block to process: JXS(31)={jxs31}, JXS(32)={jxs32}, NXS(7)={num_particle_types}")
    
    # Return the particle_release object
    return ace.particle_release

def _is_valid_ty_value(ty: int) -> bool:
    """
    Validate a TY value against allowed values according to the ACE format specification.
    
    According to documentation, allowed values are:
    ±1, ±2, ±3, ±4, ±19, 0, and integers > 100 in absolute value
    
    Not included in documentation but found in some files is ±5
    
    Parameters
    ----------
    ty : int
        TY value to validate
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    # Check absorption case
    if ty == 0:
        return True
        
    # Check common cases: ±1, ±2, ±3, ±4, ±5, ±19
    if abs(ty) in (1, 2, 3, 4, 5, 19):
        return True
        
    # Check large values (energy-dependent multiplicities)
    if abs(ty) > 100:
        return True
        
    return False
