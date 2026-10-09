import logging
from kika.ace.classes.esz import EszBlock

# Setup logger
logger = logging.getLogger(__name__)

def read_esz_block(ace, debug=False):
    """
    Read the ESZ block from the XSS array.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    EszBlock
        The parsed ESZ block
    """

    if debug is None:
        debug = ace._debug
        
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        raise ValueError("Cannot read ESZ block: header or XSS data missing")
    
    if debug:
        logger.debug("\n===== ESZ BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")

    esz_block = EszBlock()
    
    # Get the starting index for ESZ block
    esz_idx = ace.header.jxs_array[1]  # JXS(1)
    
    # Get the number of energy points - NXS(3) according to documentation
    n_energy = ace.header.nxs_array[3]  # NXS(3)
    
    if debug:
        logger.debug(f"JXS(1) = {esz_idx} → Starting index of ESZ block (FORTRAN 1-indexed)")
        logger.debug(f"Number of energy points NXS(3) = {n_energy}")
    
    if esz_idx <= 0 or n_energy <= 0:
        logger.warning(f"Warning: 0 number of energies: ESZ={esz_idx}, NE={n_energy}.")
    
    # Check if we have enough data
    if esz_idx + 5*n_energy > len(ace.xss_data) + 1:
        raise ValueError("XSS array is too short for the specified ESZ block")
    
    # Set flag indicating ESZ data is present
    esz_block.has_data = True
    
    if debug:
        logger.debug(f"Reading energy grid from XSS[{esz_idx}:{esz_idx+n_energy}]")
        logger.debug(f"Reading total XS from XSS[{esz_idx+n_energy}:{esz_idx+2*n_energy}]")
        logger.debug(f"Reading absorption XS from XSS[{esz_idx+2*n_energy}:{esz_idx+3*n_energy}]")
        logger.debug(f"Reading elastic XS from XSS[{esz_idx+3*n_energy}:{esz_idx+4*n_energy}]")
        logger.debug(f"Reading heating numbers from XSS[{esz_idx+4*n_energy}:{esz_idx+5*n_energy}]")
    
    # Extract data from XSS array according to Table 5
    esz_block.energies = ace.xss_data[esz_idx:esz_idx+n_energy]
    esz_block.total_xs = ace.xss_data[esz_idx+n_energy:esz_idx+2*n_energy]
    esz_block.absorption_xs = ace.xss_data[esz_idx+2*n_energy:esz_idx+3*n_energy]
    esz_block.elastic_xs = ace.xss_data[esz_idx+3*n_energy:esz_idx+4*n_energy]
    esz_block.heating_numbers = ace.xss_data[esz_idx+4*n_energy:esz_idx+5*n_energy]
    
    if debug:
        logger.debug(f"Read {n_energy} energy points and cross sections")
    
    # Initialize cross_section if needed and add standard cross sections
    if ace.cross_section is None:
        from kika.ace.classes.cross_section.cross_section_data import CrossSectionData
        ace.cross_section = CrossSectionData()
    
    # Set energy grid and add standard cross sections
    ace.cross_section.set_energy_grid(esz_block.energies)
    ace.cross_section.add_standard_xs(esz_block)
    
    if debug and ace.cross_section:
        logger.debug(f"Added standard cross sections (MT=1,2,101) to cross_section object")
    
    # Return the EszBlock for eager loading
    return esz_block
