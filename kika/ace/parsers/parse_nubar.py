import logging
import numpy as np
from kika.ace.classes.nubar.nubar import NuData, NuPolynomial, NuTabulated, NuContainer

# Setup logger
logger = logging.getLogger(__name__)

def read_nubar_data(ace, debug=False):
    """
    Read both NU and DNU blocks from the XSS array if they exist.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.xss_data is None):
        raise ValueError("Cannot read nubar data: header or XSS data missing")
    
    # Create a new NuContainer if not present
    if ace.nubar is None:
        ace.nubar = NuContainer()
    
    if debug:
        logger.debug("\n===== NUBAR BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Read NU block (fission nubar data) if present
    jxs2 = ace.header.jxs_array[2]  # JXS(2)
    
    if debug:
        logger.debug(f"JXS(2) = {jxs2} → Locator for NU block")
        logger.debug(f"Type of JXS(2): {type(jxs2)}")
    
    if jxs2 == 0:
        # No NU Block exists
        if debug:
            logger.debug("No NU block present (JXS(2) = 0)")
        ace.nubar.has_nubar = False
    elif jxs2 > 0:
        # Ensure jxs2 is an integer
        jxs2 = int(jxs2)
        
        if debug:
            logger.debug(f"JXS(2) after int conversion: {jxs2}")
        
        # Check bounds
        if jxs2 >= len(ace.xss_data):
            if debug:
                logger.debug(f"Error: JXS(2)={jxs2} is out of bounds for XSS array of length {len(ace.xss_data)}")
            ace.nubar.has_nubar = False
            return
        
        # NU Block exists - check XSS(JXS(2)) value to determine format
        ace.nubar.has_nubar = True
        
        # Read the value at XSS(JXS(2)) to determine if one or two arrays
        try:
            if debug:
                logger.debug(f"Attempting to access xss_data[{jxs2}]")
                logger.debug(f"Type of xss_data: {type(ace.xss_data)}")
                if jxs2 < len(ace.xss_data):
                    logger.debug(f"Value at xss_data[{jxs2}]: {ace.xss_data[jxs2]}")
                    logger.debug(f"Type of xss_data[{jxs2}]: {type(ace.xss_data[jxs2])}")
            
            xss_jxs2_value = ace.xss_data[jxs2]
            
            if debug:
                logger.debug(f"Value at XSS(JXS(2)) = {xss_jxs2_value}")
                logger.debug(f"Type of XSS(JXS(2)): {type(xss_jxs2_value)}")
        except (IndexError, AttributeError) as e:
            if debug:
                logger.debug(f"Error accessing XSS(JXS(2)): {e}")
            ace.nubar.has_nubar = False
            return
        
        if xss_jxs2_value > 0:
            # Case 1: Only one ν array (either prompt or total)
            ace.nubar.has_both_nu_types = False
            
            if debug:
                logger.debug(f"Single NU type present (XSS(JXS(2)) > 0)")
            
            # The NU array begins at location XSS(KNU) where KNU=JXS(2)
            single_nu_idx = jxs2
            
            if debug:
                logger.debug(f"NU data starts at index {single_nu_idx}")
                logger.debug(f"Type of single_nu_idx: {type(single_nu_idx)}")
            
            # Parse the single nubar array (assumed to be total nubar)
            try:
                ace.nubar.total = parse_nubar_array(ace.xss_data, single_nu_idx, debug)
            except Exception as e:
                if debug:
                    logger.debug(f"Error parsing single nubar array: {e}")
                raise
        else:
            # Case 2: Both prompt and total ν arrays are given (XSS(JXS(2)) < 0)
            ace.nubar.has_both_nu_types = True
            
            if debug:
                logger.debug(f"Both prompt and total NU types present (XSS(JXS(2)) < 0)")
            
            # The prompt ν array begins at XSS(KNU) where KNU=JXS(2) + 1
            prompt_nu_idx = jxs2 + 1
            
            # The total ν array begins at XSS(KNU) where KNU = JXS(2) + ABS(XSS(JXS(2))) + 1
            abs_xss_value = abs(float(xss_jxs2_value))
            total_nu_idx = jxs2 + int(abs_xss_value) + 1
            
            if debug:
                logger.debug(f"Prompt NU data starts at index {prompt_nu_idx}")
                logger.debug(f"Total NU data starts at index {total_nu_idx}")
                logger.debug(f"Type of prompt_nu_idx: {type(prompt_nu_idx)}")
                logger.debug(f"Type of total_nu_idx: {type(total_nu_idx)}")
            
            # Parse prompt nubar data
            try:
                ace.nubar.prompt = parse_nubar_array(ace.xss_data, prompt_nu_idx, debug)
            except Exception as e:
                if debug:
                    logger.debug(f"Error parsing prompt nubar array: {e}")
                raise
            
            # Parse total nubar data
            try:
                ace.nubar.total = parse_nubar_array(ace.xss_data, total_nu_idx, debug)
            except Exception as e:
                if debug:
                    logger.debug(f"Error parsing total nubar array: {e}")
                raise
    else:
        # Invalid JXS(2) value
        if debug:
            logger.debug(f"Invalid JXS(2) value: {jxs2}")
        ace.nubar.has_nubar = False
    
    # Read DNU block (delayed fission nubar data) if present
    dnu_idx = ace.header.jxs_array[24]  # JXS(24)
    
    if debug:
        logger.debug("\n===== DELAYED NUBAR BLOCK PARSING =====")
        logger.debug(f"JXS(24) = {dnu_idx} → Locator for DNU block")
        logger.debug(f"Type of JXS(24): {type(dnu_idx)}")
    
    if dnu_idx > 0:
        # Ensure dnu_idx is an integer
        dnu_idx = int(dnu_idx)
        
        if debug:
            logger.debug(f"JXS(24) after int conversion: {dnu_idx}")
        
        # Check bounds
        if dnu_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"Error: JXS(24)={dnu_idx} is out of bounds for XSS array of length {len(ace.xss_data)}")
            return
        
        # Delayed ν is given when JXS(24) > 0
        ace.nubar.has_delayed = True
        
        if debug:
            logger.debug(f"DNU block found at index {dnu_idx}")
        
        # Delayed ν array begins at XSS(KNU) where KNU=JXS(24)
        # Parse the delayed nubar data (always in tabulated form)
        try:
            ace.nubar.delayed = parse_nubar_array(ace.xss_data, dnu_idx, debug)
        except Exception as e:
            if debug:
                logger.debug(f"Error parsing delayed nubar array: {e}")
            raise
        
        # Validate that it's in tabulated form (LNU = 2)
        if ace.nubar.delayed.format != "tabulated":
            raise ValueError("Delayed nubar data must be in tabulated form (LNU = 2)")

    # Add this at the end of the function to confirm data was found and initialized
    if debug:
        logger.debug("\n===== NUBAR SUMMARY =====")
        logger.debug(f"Has nubar data: {ace.nubar.has_nubar}")
        logger.debug(f"Has both types: {ace.nubar.has_both_nu_types}")
        logger.debug(f"Has delayed: {ace.nubar.has_delayed}")
        logger.debug(f"Total nubar: {ace.nubar.total is not None}")
        if ace.nubar.total is not None:
            logger.debug(f"  Format: {ace.nubar.total.format}")
        logger.debug(f"Prompt nubar: {ace.nubar.prompt is not None}")
        if ace.nubar.prompt is not None:
            logger.debug(f"  Format: {ace.nubar.prompt.format}")
        logger.debug(f"Delayed nubar: {ace.nubar.delayed is not None}")
        if ace.nubar.delayed is not None:
            logger.debug(f"  Format: {ace.nubar.delayed.format}")
    
    return ace.nubar  # Ensure we're explicitly returning the nubar object

def parse_nubar_array(xss_data: np.ndarray, idx: int, debug=False) -> NuData:
    """
    Parse a single nubar array from the XSS data.
    
    Parameters
    ----------
    xss_data : List[float]
        The XSS data array
    idx : int
        Starting index in the XSS array for the nubar data
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    NuData
        A NuData object containing the parsed data
    """
    if debug:
        logger.debug(f"Entering parse_nubar_array with idx={idx}, type(idx)={type(idx)}")
    
    # Ensure idx is an integer
    if not isinstance(idx, int):
        if debug:
            logger.debug(f"Converting idx from {type(idx)} to int")
        try:
            idx = int(idx)
        except (ValueError, TypeError) as e:
            if debug:
                logger.debug(f"Error converting idx to int: {e}")
            raise ValueError(f"Invalid nubar array index (not an integer): {idx}")
    
    if idx < 0 or idx >= len(xss_data):
        if debug:
            logger.debug(f"Index check failed: idx={idx}, len(xss_data)={len(xss_data)}")
        raise ValueError(f"Invalid nubar array index (out of bounds): {idx}")
    
    # Create a NuData object to hold the result
    result = NuData()
    
    # Get the format flag (LNU)
    try:
        if debug:
            logger.debug(f"Accessing xss_data[{idx}]")
            logger.debug(f"xss_data[{idx}] = {xss_data[idx]}")
            logger.debug(f"xss_data[{idx}] = {xss_data[idx]}")
        
        format_int = int(xss_data[idx])
        
        if debug:
            logger.debug(f"Format flag (LNU) = {format_int}")
    except (IndexError, AttributeError, TypeError, ValueError) as e:
        if debug:
            logger.debug(f"Error getting format flag: {e}")
            logger.debug(f"Type of xss_data: {type(xss_data)}")
            if isinstance(xss_data, list) and 0 <= idx < len(xss_data):
                logger.debug(f"Type of xss_data[{idx}]: {type(xss_data[idx])}")
        raise
    
    if format_int == 1:
        result.format = "polynomial"
    elif format_int == 2:
        result.format = "tabulated"
    else:
        raise ValueError(f"Unknown nubar format flag (LNU): {format_int}")
    
    if debug:
        logger.debug(f"Nubar format (LNU): {format_int} → {result.format}")
    
    if result.format == "polynomial":  # Polynomial form (LNU = 1)
        n_coeff = int(xss_data[idx + 1])  # Number of coefficients
        
        if debug:
            logger.debug(f"Polynomial form with {n_coeff} coefficients")
        
        # Create a NuPolynomial object and populate it
        result.polynomial = NuPolynomial()
        result.polynomial.coefficients = xss_data[idx + 2:idx + 2 + n_coeff]
        
    elif result.format == "tabulated":  # Tabulated form (LNU = 2)
        n_regions = int(xss_data[idx + 1])  # Number of interpolation regions
        
        if debug:
            logger.debug(f"Tabulated form with {n_regions} interpolation regions")
        
        # Create a NuTabulated object
        result.tabulated = NuTabulated()
        
        # Current position in the XSS array
        pos = idx + 2
        
        # Read interpolation regions if present
        if n_regions > 0:
            # Read NBT array
            nbt = [int(x) for x in xss_data[pos:pos + n_regions]]
            pos += n_regions
            
            # Read INT array
            interp = [int(x) for x in xss_data[pos:pos + n_regions]]
            pos += n_regions
            
            # Store interpolation regions
            result.tabulated.interpolation_regions = list(zip(nbt, interp))
            
            if debug:
                logger.debug(f"Interpolation regions: {result.tabulated.interpolation_regions}")
        
        # Read number of energy points
        n_energies = int(xss_data[pos])
        pos += 1
        
        if debug:
            logger.debug(f"Number of energy points: {n_energies}")
        
        # Read energy points
        result.tabulated.energies = xss_data[pos:pos + n_energies]
        pos += n_energies
        
        # Read nubar values
        result.tabulated.nubar_values = xss_data[pos:pos + n_energies]
    
    return result
