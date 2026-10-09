import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.level_scattering import LevelScattering

# Setup logger
logger = logging.getLogger(__name__)

def parse_level_scattering(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> LevelScattering:
    """
    Parse a level scattering distribution (Law 3).
    
    According to Table 34, LAW=3 contains:
    - LDAT(1): (A + 1)/A|Q|
    - LDAT(2): (A / (A + 1))^2
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    base_dist : EnergyDistribution
        Base distribution with common properties
    idat_idx : int
        Starting index for the law data in the XSS array
    debug : bool, optional
        If True, enables debug logging
        
    Returns
    -------
    LevelScattering
        Level scattering distribution object
    """
    if debug:
        logger.debug(f"Parsing level scattering distribution (Law 3) starting at index {idat_idx}")
    
    # Create a new distribution object using the base properties
    distribution = LevelScattering(
        law=base_dist.law,
        idat=base_dist.idat
    )
    
    # Copy applicability data from base_dist
    distribution.applicability_energies = base_dist.applicability_energies
    distribution.applicability_probabilities = base_dist.applicability_probabilities
    distribution.applicability_nbt = base_dist.applicability_nbt
    distribution.applicability_interp = base_dist.applicability_interp
    
    # Read the two parameters for Law 3
    if idat_idx + 1 < len(ace.xss_data):
        # LDAT(1): (A + 1)/A|Q|
        aplusoaabsq_entry = ace.xss_data[idat_idx]
        distribution.aplusoaabsq = float(aplusoaabsq_entry)
        if debug:
            logger.debug(f"(A + 1)/A|Q| value: {distribution.aplusoaabsq}")
        
        # LDAT(2): (A / (A + 1))^2
        asquare_entry = ace.xss_data[idat_idx + 1]
        distribution.asquare = float(asquare_entry)
        if debug:
            logger.debug(f"(A / (A + 1))^2 value: {distribution.asquare}")
    else:
        if debug:
            logger.debug(f"Not enough data to read parameters. Need indices {idat_idx} and {idat_idx + 1}, have {len(ace.xss_data)} entries")
    
    if debug:
        logger.debug("Completed parsing level scattering distribution")
    return distribution