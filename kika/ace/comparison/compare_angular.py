"""
Module for comparing angular distribution data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.classes.angular_distribution.angular_distribution import AngularDistributionType
from kika.ace.comparison.compare_ace import compare_arrays

def compare_angular_distributions(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare angular distributions between two ACE objects."""
    # Check if both objects have angular distribution data
    if ace1.angular_distributions is None and ace2.angular_distributions is None:
        return True
    
    if ace1.angular_distributions is None or ace2.angular_distributions is None:
        if verbose:
            print("Angular distribution mismatch: One ACE object has no angular distribution data")
        return False
    
    # Compare elastic scattering angular distribution
    if not compare_elastic_angular(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare neutron reaction angular distributions
    if not compare_neutron_angular(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare photon production angular distributions
    if not compare_photon_angular(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle production angular distributions
    if not compare_particle_angular(ace1, ace2, tolerance, verbose):
        return False
    
    return True

def compare_elastic_angular(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare elastic scattering angular distributions."""
    has_elastic1 = (ace1.angular_distributions and ace1.angular_distributions.has_elastic_data)
    has_elastic2 = (ace2.angular_distributions and ace2.angular_distributions.has_elastic_data)
    
    if not has_elastic1 and not has_elastic2:
        return True
    
    if has_elastic1 != has_elastic2:
        if verbose:
            print("Elastic angular distribution mismatch: Presence differs")
        return False
    
    # Now perform a detailed comparison of the elastic distributions
    dist1 = ace1.angular_distributions.elastic
    dist2 = ace2.angular_distributions.elastic
    
    # Compare distribution types
    if dist1.distribution_type != dist2.distribution_type:
        if verbose:
            print(f"Elastic angular distribution mismatch: Types differ "
                  f"({dist1.distribution_type} vs {dist2.distribution_type})")
        return False
    
    # Compare energies
    energy_values1 = [float(e) for e in dist1.energies]
    energy_values2 = [float(e) for e in dist2.energies]
    if not compare_arrays(energy_values1, energy_values2, tolerance, "Elastic angular energy grid", verbose):
        return False
        
    # Compare specific distribution types
    return compare_angular_distribution_data(dist1, dist2, tolerance, "Elastic", verbose)

def compare_neutron_angular(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare neutron reaction angular distributions."""
    has_neutron1 = (ace1.angular_distributions and ace1.angular_distributions.has_neutron_data)
    has_neutron2 = (ace2.angular_distributions and ace2.angular_distributions.has_neutron_data)
    
    if not has_neutron1 and not has_neutron2:
        return True
    
    if has_neutron1 != has_neutron2:
        if verbose:
            print("Neutron angular distribution mismatch: Presence differs")
        return False
    
    # Get MT numbers from both distributions
    mt_numbers1 = set(ace1.angular_distributions.get_neutron_reaction_mt_numbers())
    mt_numbers2 = set(ace2.angular_distributions.get_neutron_reaction_mt_numbers())
    
    # Check if the same MT numbers are present
    if mt_numbers1 != mt_numbers2:
        if verbose:
            print("Neutron angular distribution mismatch: Different MT numbers")
            print(f"MT numbers only in first: {sorted(mt_numbers1 - mt_numbers2)}")
            print(f"MT numbers only in second: {sorted(mt_numbers2 - mt_numbers1)}")
        return False
    
    # Compare each MT reaction's distribution
    for mt in sorted(mt_numbers1):
        dist1 = ace1.angular_distributions.incident_neutron.get(mt)
        dist2 = ace2.angular_distributions.incident_neutron.get(mt)
        
        # Compare distribution types
        if dist1.distribution_type != dist2.distribution_type:
            if verbose:
                print(f"Neutron MT={mt} angular distribution mismatch: Types differ "
                      f"({dist1.distribution_type} vs {dist2.distribution_type})")
            return False
        
        # Compare energies
        energy_values1 = [float(e) for e in dist1.energies]
        energy_values2 = [float(e) for e in dist2.energies]
        if not compare_arrays(energy_values1, energy_values2, tolerance, f"Neutron MT={mt} angular energy grid", verbose):
            return False
        
        # Compare specific distribution data
        if not compare_angular_distribution_data(dist1, dist2, tolerance, f"Neutron MT={mt}", verbose):
            return False
    
    return True

def compare_photon_angular(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare photon production angular distributions."""
    has_photon1 = (ace1.angular_distributions and ace1.angular_distributions.has_photon_production_data)
    has_photon2 = (ace2.angular_distributions and ace2.angular_distributions.has_photon_production_data)
    
    if not has_photon1 and not has_photon2:
        return True
    
    if has_photon1 != has_photon2:
        if verbose:
            print("Photon angular distribution mismatch: Presence differs")
        return False
    
    # Get MT numbers from both distributions
    mt_numbers1 = set(ace1.angular_distributions.get_photon_production_mt_numbers())
    mt_numbers2 = set(ace2.angular_distributions.get_photon_production_mt_numbers())
    
    # Check if the same MT numbers are present
    if mt_numbers1 != mt_numbers2:
        if verbose:
            print("Photon angular distribution mismatch: Different MT numbers")
            print(f"MT numbers only in first: {sorted(mt_numbers1 - mt_numbers2)}")
            print(f"MT numbers only in second: {sorted(mt_numbers2 - mt_numbers1)}")
        return False
    
    # Compare each MT reaction's distribution
    for mt in sorted(mt_numbers1):
        dist1 = ace1.angular_distributions.photon_production.get(mt)
        dist2 = ace2.angular_distributions.photon_production.get(mt)
        
        # Compare distribution types
        if dist1.distribution_type != dist2.distribution_type:
            if verbose:
                print(f"Photon MT={mt} angular distribution mismatch: Types differ "
                      f"({dist1.distribution_type} vs {dist2.distribution_type})")
            return False
        
        # Compare energies
        energy_values1 = [float(e) for e in dist1.energies]
        energy_values2 = [float(e) for e in dist2.energies]
        if not compare_arrays(energy_values1, energy_values2, tolerance, f"Photon MT={mt} angular energy grid", verbose):
            return False
        
        # Compare specific distribution data
        if not compare_angular_distribution_data(dist1, dist2, tolerance, f"Photon MT={mt}", verbose):
            return False
    
    return True

def compare_particle_angular(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare particle production angular distributions."""
    has_particle1 = (ace1.angular_distributions and ace1.angular_distributions.has_particle_production_data)
    has_particle2 = (ace2.angular_distributions and ace2.angular_distributions.has_particle_production_data)
    
    if not has_particle1 and not has_particle2:
        return True
    
    if has_particle1 != has_particle2:
        if verbose:
            print("Particle angular distribution mismatch: Presence differs")
        return False
    
    # Compare number of particle types
    n_particles1 = len(ace1.angular_distributions.particle_production)
    n_particles2 = len(ace2.angular_distributions.particle_production)
    
    if n_particles1 != n_particles2:
        if verbose:
            print(f"Particle angular distribution mismatch: Number of particle types differs "
                  f"({n_particles1} vs {n_particles2})")
        return False
    
    # Compare each particle type
    for particle_idx in range(n_particles1):
        # Get MT numbers for this particle type
        mt_numbers1 = set(ace1.angular_distributions.get_particle_production_mt_numbers(particle_idx) or [])
        mt_numbers2 = set(ace2.angular_distributions.get_particle_production_mt_numbers(particle_idx) or [])
        
        # Check if the same MT numbers are present
        if mt_numbers1 != mt_numbers2:
            if verbose:
                print(f"Particle type {particle_idx} angular distribution mismatch: Different MT numbers")
                print(f"MT numbers only in first: {sorted(mt_numbers1 - mt_numbers2)}")
                print(f"MT numbers only in second: {sorted(mt_numbers2 - mt_numbers1)}")
            return False
        
        # Skip comparison if no MT numbers (empty distributions)
        if not mt_numbers1:
            continue
        
        # Compare each MT reaction's distribution for this particle
        for mt in sorted(mt_numbers1):
            dist1 = ace1.angular_distributions.particle_production[particle_idx].get(mt)
            dist2 = ace2.angular_distributions.particle_production[particle_idx].get(mt)
            
            # Check if both distributions exist
            if dist1 is None and dist2 is None:
                continue  # Both are None, so they match
            
            if dist1 is None or dist2 is None:
                if verbose:
                    print(f"Particle type {particle_idx} MT={mt} angular distribution mismatch: "
                          f"One distribution is None, the other isn't")
                return False
            
            # Compare distribution types
            if dist1.distribution_type != dist2.distribution_type:
                if verbose:
                    print(f"Particle type {particle_idx} MT={mt} angular distribution mismatch: Types differ "
                          f"({dist1.distribution_type} vs {dist2.distribution_type})")
                return False
            
            # Compare energies
            energy_values1 = [float(e) for e in dist1.energies]
            energy_values2 = [float(e) for e in dist2.energies]
            if not compare_arrays(energy_values1, energy_values2, tolerance, 
                                 f"Particle type {particle_idx} MT={mt} angular energy grid", verbose):
                return False
            
            # Compare specific distribution data
            if not compare_angular_distribution_data(dist1, dist2, tolerance, 
                                                   f"Particle type {particle_idx} MT={mt}", verbose):
                return False
    
    return True

def compare_angular_distribution_data(dist1, dist2, tolerance: float, name: str, verbose: bool) -> bool:
    """
    Compare the specific data for angular distributions based on their type.
    
    Parameters
    ----------
    dist1 : AngularDistribution
        First angular distribution
    dist2 : AngularDistribution
        Second angular distribution
    tolerance : float
        Tolerance for floating-point comparisons
    name : str
        Name identifier for the distribution (for reporting)
    verbose : bool
        If True, print detailed information about any differences
        
    Returns
    -------
    bool
        True if distributions are equivalent, False otherwise
    """
    # If both are isotropic, nothing else to compare
    if dist1.distribution_type == AngularDistributionType.ISOTROPIC and \
       dist2.distribution_type == AngularDistributionType.ISOTROPIC:
        return True
    
    # Compare equiprobable distributions
    if dist1.distribution_type == AngularDistributionType.EQUIPROBABLE:
        # Compare cosine bins for each energy
        if len(dist1.cosine_bins) != len(dist2.cosine_bins):
            if verbose:
                print(f"{name} angular distribution mismatch: Different number of energy points for cosine bins "
                      f"({len(dist1.cosine_bins)} vs {len(dist2.cosine_bins)})")
            return False
        
        for i, (bins1, bins2) in enumerate(zip(dist1.cosine_bins, dist2.cosine_bins)):
            bins_values1 = [float(b) for b in bins1]
            bins_values2 = [float(b) for b in bins2]
            
            if not compare_arrays(bins_values1, bins_values2, tolerance, 
                                 f"{name} angular distribution cosine bins at energy point {i}", verbose):
                return False
        
        return True
    
    # Compare tabulated distributions
    if dist1.distribution_type == AngularDistributionType.TABULATED:
        # Compare interpolation flags
        if dist1.interpolation != dist2.interpolation:
            if verbose:
                print(f"{name} angular distribution mismatch: Different interpolation flags "
                      f"({dist1.interpolation} vs {dist2.interpolation})")
            return False
        
        # Compare number of tabulated points
        if len(dist1.cosine_grid) != len(dist2.cosine_grid):
            if verbose:
                print(f"{name} angular distribution mismatch: Different number of energy points for cosine grid "
                      f"({len(dist1.cosine_grid)} vs {len(dist2.cosine_grid)})")
            return False
        
        # Compare cosine grids, PDFs, and CDFs for each energy
        for i in range(len(dist1.cosine_grid)):
            # Cosine grid
            grid1 = [float(c) for c in dist1.cosine_grid[i]]
            grid2 = [float(c) for c in dist2.cosine_grid[i]]
            if not compare_arrays(grid1, grid2, tolerance, 
                                 f"{name} angular distribution cosine grid at energy point {i}", verbose):
                return False
            
            # PDF
            pdf1 = [float(p) for p in dist1.pdf[i]]
            pdf2 = [float(p) for p in dist2.pdf[i]]
            if not compare_arrays(pdf1, pdf2, tolerance, 
                                 f"{name} angular distribution PDF at energy point {i}", verbose):
                return False
            
            # CDF
            cdf1 = [float(c) for c in dist1.cdf[i]]
            cdf2 = [float(c) for c in dist2.cdf[i]]
            if not compare_arrays(cdf1, cdf2, tolerance, 
                                 f"{name} angular distribution CDF at energy point {i}", verbose):
                return False
        
        return True
    
    # Unknown or mismatched distribution types
    if verbose:
        print(f"{name} angular distribution has unknown or mismatched type: {dist1.distribution_type}")
    return False
