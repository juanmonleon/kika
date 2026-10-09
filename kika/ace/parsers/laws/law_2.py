import logging
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.discrete import DiscreteEnergyDistribution

# Setup logger
logger = logging.getLogger(__name__)

def parse_discrete_energy_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> DiscreteEnergyDistribution:
    """
    Parse a discrete energy distribution (Law 2).
    
    According to Table 33, LAW=2 represents Discrete Photon Energy:
    - LDAT(1): LP - Indicator of whether photon is primary or non-primary
    - LDAT(2): EG - Photon energy or binding energy
    
    If LP=0 or LP=1, the photon energy is simply EG.
    If LP=2, the photon energy is EG + (AWR/(AWR+1))*E_N where E_N is the incident energy.
    
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
    DiscreteEnergyDistribution
        Discrete energy distribution object
    """
    if debug:
        logger.debug(f"Parsing discrete energy distribution (Law 2) starting at index {idat_idx}")
    
    # Create a new distribution object using the base properties
    distribution = DiscreteEnergyDistribution(
        law=base_dist.law,
        idat=base_dist.idat
    )
    
    # Copy applicability data from base_dist
    distribution.applicability_energies = base_dist.applicability_energies
    distribution.applicability_probabilities = base_dist.applicability_probabilities
    distribution.applicability_nbt = base_dist.applicability_nbt
    distribution.applicability_interp = base_dist.applicability_interp
    
    # Check if we have data to parse
    if idat_idx + 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"Not enough data to read LP and EG. Need index up to {idat_idx + 1}, have {len(ace.xss_data)}")
        return distribution
    
    # Read LP (indicator of whether photon is primary or non-primary)
    lp_entry = ace.xss_data[idat_idx]
    lp = int(lp_entry)
    if debug:
        logger.debug(f"LP (primary photon indicator): {lp}")
    
    # Read EG (photon energy or binding energy)
    eg_entry = ace.xss_data[idat_idx + 1]
    eg = float(eg_entry)
    if debug:
        logger.debug(f"EG (photon energy or binding energy): {eg}")
    
    # Store this information in the distribution object
    # For Law 2, we only have a single discrete energy and probability
    distribution.lp = lp
    distribution.eg = eg
    
    # For LP=0 or LP=1, the energy is just EG
    # For LP=2, we'll need to calculate it during sampling based on incident energy
    # For now, just store the discrete energy as EG
    distribution.n_discrete_energies = 1
    distribution.discrete_energies = [eg]
    distribution.probabilities = [1.0]  # Only one discrete energy, so probability is 100%
    
    if debug:
        logger.debug(f"Completed parsing discrete energy distribution with LP={lp}, EG={eg}")
    return distribution