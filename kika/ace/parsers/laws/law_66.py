import logging
import numpy as np
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.phase_space import NBodyPhaseSpaceDistribution
from kika.ace.classes.read_report import note

# Setup logger
logger = logging.getLogger(__name__)

def parse_nbody_phase_space_distribution(ace: Ace, base_dist: EnergyDistribution, idat_idx: int, debug: bool = False) -> NBodyPhaseSpaceDistribution:
    """
    Parse an N-body phase space distribution (Law 66).
    
    According to Table 48, LAW=66 contains:
    - LDAT(1): NPSX - Number of bodies in the phase space
    - LDAT(2): A_P - Total mass ratio for the NPSX particles
    - LDAT(3): INTT - Interpolation parameter
    - LDAT(4): N_P - Number of points in the distribution
    - LDAT(5) to LDAT(4+N_P): ξ_out(j) - ξ grid (between 0 and 1)
    - LDAT(5+N_P) to LDAT(4+2*N_P): PDF(j) - Probability density function
    - LDAT(5+2*N_P) to LDAT(4+3*N_P): CDF(j) - Cumulative density function
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    base_dist : EnergyDistribution
        Base distribution with common properties
    idat_idx : int
        Starting index for the law data in the XSS array
    debug : bool, optional
        Flag to enable or disable debug logging
        
    Returns
    -------
    NBodyPhaseSpaceDistribution
        N-body phase space distribution object
    """
    distribution = NBodyPhaseSpaceDistribution(
        law=base_dist.law,
        idat=base_dist.idat,
        applicability_energies=base_dist.applicability_energies,
        applicability_probabilities=base_dist.applicability_probabilities,
        applicability_nbt=base_dist.applicability_nbt,
        applicability_interp=base_dist.applicability_interp,
    )
    xss = ace.xss_data
    if idat_idx + 1 >= len(xss):
        raise ValueError(f"LAW=66 needs NPSX and Ap at {idat_idx}, past the XSS array ({len(xss)})")
    distribution.npsx = int(round(xss[idat_idx]))
    distribution.ap = float(xss[idat_idx + 1])

    # INTT, NP and the ξ table follow in what NJOY 2016 writes (Table 48);
    # older NJOY wrote the two words alone, and the words after them are then
    # the next law. The table is taken only if it reads as one.
    table = _xi_table(xss, idat_idx + 2)
    if table is None:
        note(ace, None, None, f"LAW=66: no tabulated ξ part at {idat_idx + 2} (written before NJOY 2016); "
                               f"NPSX={distribution.npsx} and Ap read, the phase-space density is analytic")
        return distribution
    distribution.intt, distribution.n_points, distribution.xi_grid, distribution.pdf, distribution.cdf = table
    return distribution


def _xi_table(xss, start):
    """``(INTT, NP, ξ, pdf, cdf)`` at ``start`` if it is a ξ table on [0, 1], else None."""
    if start + 1 >= len(xss):
        return None
    intt, n_p = xss[start], xss[start + 1]
    if intt not in (1.0, 2.0) or n_p != int(n_p) or n_p < 2 or start + 2 + 3 * int(n_p) > len(xss):
        return None
    n_p = int(n_p)
    a = start + 2
    xi, pdf, cdf = xss[a:a + n_p], xss[a + n_p:a + 2 * n_p], xss[a + 2 * n_p:a + 3 * n_p]
    if not (xi[0] == 0.0 and abs(xi[-1] - 1.0) < 1e-6 and np.all(np.diff(xi) >= 0)
            and np.all(pdf >= 0) and abs(cdf[0]) < 1e-6 and abs(cdf[-1] - 1.0) < 1e-4):
        return None
    return int(intt), n_p, xi, pdf, cdf
