import logging
from typing import List, Optional
from kika.ace.classes.q_values import QValues

# Setup logger
logger = logging.getLogger(__name__)

def read_lqr_block(ace, debug=False):
    """
    Read LQR block from the XSS array if it exists.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    QValues
        The Q-values object
    """
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        raise ValueError("Cannot read LQR block: header or XSS data missing")
    
    if debug:
        logger.debug("\n===== LQR BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize q_values if it doesn't exist
    if ace.q_values is None:
        ace.q_values = QValues()
    
    # Read LQR block (reaction Q-values) if present
    lqr_idx = ace.header.jxs_array[4]  # JXS(4)
    
    if debug:
        logger.debug(f"JXS(4) = {lqr_idx} → Locator for LQR block")
    
    if lqr_idx > 0:
        num_reactions = ace.header.nxs_array[4]  # NXS(4)
        
        if debug:
            logger.debug(f"NXS(4) = {num_reactions} → Total number of reactions")
        
        if num_reactions > 0:
            
            if debug:
                logger.debug(f"LQR block range: XSS[{lqr_idx}:{lqr_idx+num_reactions}]")
            
            if (lqr_idx + num_reactions <= len(ace.xss_data)):
                ace.q_values.q_values = ace.xss_data[lqr_idx:lqr_idx + num_reactions]
                
                if debug:
                    logger.debug(f"Successfully read {len(ace.q_values.q_values)} Q-values")
    
    # Return the q_values object
    return ace.q_values
