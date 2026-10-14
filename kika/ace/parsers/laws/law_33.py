from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.level_scattering import ParticleLevelScattering


def parse_particle_level_scattering(ace: Ace, base_dist: EnergyDistribution, idat_idx: int,
                                    debug: bool = False) -> ParticleLevelScattering:
    """
    Parse a level-scattering distribution of a produced particle (Law 33).

    Two words, as law 3: LDAT(1) = Ec and LDAT(2) = C, with E_out = C·(E - Ec)
    (NJOY ``acelcp``; OpenMC reads 3 and 33 alike).
    """
    if idat_idx + 1 >= len(ace.xss_data):
        raise ValueError(f"LAW=33 needs LDAT(1:2) at {idat_idx}, past the XSS array ({len(ace.xss_data)})")
    return ParticleLevelScattering(
        law=33,
        idat=base_dist.idat,
        applicability_energies=base_dist.applicability_energies,
        applicability_probabilities=base_dist.applicability_probabilities,
        applicability_nbt=base_dist.applicability_nbt,
        applicability_interp=base_dist.applicability_interp,
        threshold=float(ace.xss_data[idat_idx]),
        energy_factor=float(ace.xss_data[idat_idx + 1]),
    )
