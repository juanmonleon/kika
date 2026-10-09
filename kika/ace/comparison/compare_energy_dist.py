"""
Module for comparing energy distribution data in ACE format.
"""

from typing import List
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.comparison.compare_utils import compare_arrays

def compare_energy_distributions(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare energy distributions between two ACE objects."""
    # Check if both objects have energy distribution data
    if ace1.energy_distributions is None and ace2.energy_distributions is None:
        return True
    
    if ace1.energy_distributions is None or ace2.energy_distributions is None:
        if verbose:
            print("Energy distribution mismatch: One ACE object has no energy distribution data")
        return False
    
    # Compare neutron energy distributions
    if not compare_neutron_energy_dist(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare photon production energy distributions
    if not compare_photon_energy_dist(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle production energy distributions
    if not compare_particle_energy_dist(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare delayed neutron energy distributions
    if not compare_delayed_neutron_dist(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare energy-dependent yields
    if not compare_yields(ace1, ace2, tolerance, verbose):
        return False
    
    return True

def compare_neutron_energy_dist(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare neutron energy distributions."""
    has_neutron1 = (ace1.energy_distributions and ace1.energy_distributions.has_neutron_data)
    has_neutron2 = (ace2.energy_distributions and ace2.energy_distributions.has_neutron_data)
    
    if not has_neutron1 and not has_neutron2:
        return True
    
    if has_neutron1 != has_neutron2:
        if verbose:
            print("Neutron energy distribution mismatch: Presence differs")
        return False
    
    # Compare MT numbers
    mt_numbers1 = set(ace1.energy_distributions.get_neutron_reaction_mt_numbers())
    mt_numbers2 = set(ace2.energy_distributions.get_neutron_reaction_mt_numbers())
    
    if mt_numbers1 != mt_numbers2:
        if verbose:
            print("Neutron energy distribution mismatch: Different MT numbers")
            print(f"MT numbers only in first: {sorted(mt_numbers1 - mt_numbers2)}")
            print(f"MT numbers only in second: {sorted(mt_numbers2 - mt_numbers1)}")
        return False
    
    # Compare each MT reaction's distributions
    for mt in sorted(mt_numbers1):
        dist_list1 = ace1.energy_distributions.get_neutron_distribution(mt)
        dist_list2 = ace2.energy_distributions.get_neutron_distribution(mt)
        
        if not compare_distribution_lists(dist_list1, dist_list2, tolerance, f"Neutron MT={mt}", verbose):
            return False
    
    return True

def compare_photon_energy_dist(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare photon production energy distributions."""
    has_photon1 = (ace1.energy_distributions and ace1.energy_distributions.has_photon_production_data)
    has_photon2 = (ace2.energy_distributions and ace2.energy_distributions.has_photon_production_data)
    
    if not has_photon1 and not has_photon2:
        return True
    
    if has_photon1 != has_photon2:
        if verbose:
            print("Photon energy distribution mismatch: Presence differs")
        return False
    
    # Compare MT numbers
    mt_numbers1 = set(ace1.energy_distributions.get_photon_production_mt_numbers())
    mt_numbers2 = set(ace2.energy_distributions.get_photon_production_mt_numbers())
    
    if mt_numbers1 != mt_numbers2:
        if verbose:
            print("Photon energy distribution mismatch: Different MT numbers")
            print(f"MT numbers only in first: {sorted(mt_numbers1 - mt_numbers2)}")
            print(f"MT numbers only in second: {sorted(mt_numbers2 - mt_numbers1)}")
        return False
    
    # Compare each MT reaction's distributions
    for mt in sorted(mt_numbers1):
        dist_list1 = ace1.energy_distributions.get_photon_distribution(mt)
        dist_list2 = ace2.energy_distributions.get_photon_distribution(mt)
        
        if not compare_distribution_lists(dist_list1, dist_list2, tolerance, f"Photon MT={mt}", verbose):
            return False
    
    return True

def compare_particle_energy_dist(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare particle production energy distributions."""
    has_particle1 = (ace1.energy_distributions and ace1.energy_distributions.has_particle_production_data)
    has_particle2 = (ace2.energy_distributions and ace2.energy_distributions.has_particle_production_data)
    
    if not has_particle1 and not has_particle2:
        return True
    
    if has_particle1 != has_particle2:
        if verbose:
            print("Particle energy distribution mismatch: Presence differs")
        return False
    
    # Compare number of particle types
    n_types1 = len(ace1.energy_distributions.particle_production)
    n_types2 = len(ace2.energy_distributions.particle_production)
    
    if n_types1 != n_types2:
        if verbose:
            print(f"Particle energy distribution mismatch: Number of particle types differs ({n_types1} vs {n_types2})")
        return False
    
    # Compare each particle type
    for particle_idx in range(n_types1):
        # Compare MT numbers for this particle type
        mt_numbers1 = set(ace1.energy_distributions.get_particle_production_mt_numbers(particle_idx))
        mt_numbers2 = set(ace2.energy_distributions.get_particle_production_mt_numbers(particle_idx))
        
        if mt_numbers1 != mt_numbers2:
            if verbose:
                print(f"Particle type {particle_idx} energy distribution mismatch: Different MT numbers")
                print(f"MT numbers only in first: {sorted(mt_numbers1 - mt_numbers2)}")
                print(f"MT numbers only in second: {sorted(mt_numbers2 - mt_numbers1)}")
            return False
        
        # Compare each MT reaction's distributions for this particle type
        for mt in sorted(mt_numbers1):
            dist_list1 = ace1.energy_distributions.get_particle_distribution(particle_idx, mt)
            dist_list2 = ace2.energy_distributions.get_particle_distribution(particle_idx, mt)
            
            if not compare_distribution_lists(dist_list1, dist_list2, tolerance, 
                                            f"Particle type {particle_idx} MT={mt}", verbose):
                return False
    
    return True

def compare_delayed_neutron_dist(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare delayed neutron energy distributions."""
    has_delayed1 = (ace1.energy_distributions and ace1.energy_distributions.has_delayed_neutron_data)
    has_delayed2 = (ace2.energy_distributions and ace2.energy_distributions.has_delayed_neutron_data)
    
    if not has_delayed1 and not has_delayed2:
        return True
    
    if has_delayed1 != has_delayed2:
        if verbose:
            print("Delayed neutron energy distribution mismatch: Presence differs")
        return False
    
    # Check number of delayed neutron groups
    n_groups1 = len(ace1.energy_distributions.delayed_neutron)
    n_groups2 = len(ace2.energy_distributions.delayed_neutron)
    
    if n_groups1 != n_groups2:
        if verbose:
            print(f"Delayed neutron mismatch: Number of groups differs ({n_groups1} vs {n_groups2})")
        return False
    
    # Compare each delayed neutron group's distribution
    for i in range(n_groups1):
        dist1 = ace1.energy_distributions.get_delayed_neutron_distribution(i)
        dist2 = ace2.energy_distributions.get_delayed_neutron_distribution(i)
        
        if not compare_energy_distribution(dist1, dist2, tolerance, f"Delayed neutron group {i}", verbose):
            return False
    
    return True

def compare_yields(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare energy-dependent yields."""
    # Compare neutron yields
    has_neutron_yields1 = (ace1.energy_distributions and ace1.energy_distributions.has_neutron_yields)
    has_neutron_yields2 = (ace2.energy_distributions and ace2.energy_distributions.has_neutron_yields)
    
    if has_neutron_yields1 != has_neutron_yields2:
        if verbose:
            print("Energy-dependent neutron yields mismatch: Presence differs")
        return False
    
    if has_neutron_yields1 and has_neutron_yields2:
        # Compare MT numbers for neutron yields
        mt_numbers1 = set(ace1.energy_distributions.neutron_yields.keys())
        mt_numbers2 = set(ace2.energy_distributions.neutron_yields.keys())
        
        if mt_numbers1 != mt_numbers2:
            if verbose:
                print("Neutron yields mismatch: Different MT numbers")
                print(f"MT numbers only in first: {sorted(mt_numbers1 - mt_numbers2)}")
                print(f"MT numbers only in second: {sorted(mt_numbers2 - mt_numbers1)}")
            return False
        
        # Compare yield values for each MT
        for mt in sorted(mt_numbers1):
            yield1 = ace1.energy_distributions.neutron_yields[mt]
            yield2 = ace2.energy_distributions.neutron_yields[mt]
            
            if not compare_energy_distribution(yield1, yield2, tolerance, f"Neutron yield MT={mt}", verbose):
                return False
    
    # Compare particle yields
    has_particle_yields1 = (ace1.energy_distributions and ace1.energy_distributions.has_particle_yields)
    has_particle_yields2 = (ace2.energy_distributions and ace2.energy_distributions.has_particle_yields)
    
    if has_particle_yields1 != has_particle_yields2:
        if verbose:
            print("Energy-dependent particle yields mismatch: Presence differs")
        return False
    
    if has_particle_yields1 and has_particle_yields2:
        # Compare number of particle types
        n_types1 = len(ace1.energy_distributions.particle_yields)
        n_types2 = len(ace2.energy_distributions.particle_yields)
        
        if n_types1 != n_types2:
            if verbose:
                print(f"Particle yields mismatch: Number of particle types differs ({n_types1} vs {n_types2})")
            return False
        
        # Compare each particle type
        for particle_idx in range(n_types1):
            # Skip if this particle type has no yields
            if particle_idx >= len(ace1.energy_distributions.particle_yields) or \
               particle_idx >= len(ace2.energy_distributions.particle_yields):
                continue
            
            # Get MT numbers for this particle type
            mt_dict1 = ace1.energy_distributions.particle_yields[particle_idx]
            mt_dict2 = ace2.energy_distributions.particle_yields[particle_idx]
            
            if not mt_dict1 and not mt_dict2:
                continue
            
            if not mt_dict1 or not mt_dict2:
                if verbose:
                    print(f"Particle type {particle_idx} yields mismatch: One has yields, the other doesn't")
                return False
            
            mt_numbers1 = set(mt_dict1.keys())
            mt_numbers2 = set(mt_dict2.keys())
            
            if mt_numbers1 != mt_numbers2:
                if verbose:
                    print(f"Particle type {particle_idx} yields mismatch: Different MT numbers")
                    print(f"MT numbers only in first: {sorted(mt_numbers1 - mt_numbers2)}")
                    print(f"MT numbers only in second: {sorted(mt_numbers2 - mt_numbers1)}")
                return False
            
            # Compare yield values for each MT
            for mt in sorted(mt_numbers1):
                yield1 = mt_dict1[mt]
                yield2 = mt_dict2[mt]
                
                if not compare_energy_distribution(yield1, yield2, tolerance, 
                                                 f"Particle type {particle_idx} yield MT={mt}", verbose):
                    return False
    
    return True

def compare_distribution_lists(list1: List[EnergyDistribution], list2: List[EnergyDistribution], 
                              tolerance: float, name: str, verbose: bool) -> bool:
    """
    Compare two lists of energy distributions.
    
    Parameters
    ----------
    list1 : List[EnergyDistribution]
        First list of energy distributions
    list2 : List[EnergyDistribution]
        Second list of energy distributions
    tolerance : float
        Tolerance for floating-point comparisons
    name : str
        Name of the distribution for reporting
    verbose : bool
        If True, print detailed information about any differences
        
    Returns
    -------
    bool
        True if the lists are equivalent, False otherwise
    """
    if list1 is None and list2 is None:
        return True
    
    if list1 is None or list2 is None:
        if verbose:
            print(f"{name} energy distribution mismatch: One list is None")
        return False
    
    # Compare list lengths
    if len(list1) != len(list2):
        if verbose:
            print(f"{name} energy distribution mismatch: Different number of distributions "
                  f"({len(list1)} vs {len(list2)})")
        return False
    
    # Compare each distribution
    for i, (dist1, dist2) in enumerate(zip(list1, list2)):
        if not compare_energy_distribution(dist1, dist2, tolerance, f"{name} distribution {i}", verbose):
            return False
    
    return True

def compare_energy_distribution(dist1, dist2, tolerance: float, name: str, verbose: bool) -> bool:
    """
    Compare two energy distributions for equality within tolerance.
    
    Parameters
    ----------
    dist1 : EnergyDistribution or EnergyDependentYield
        First energy distribution
    dist2 : EnergyDistribution or EnergyDependentYield
        Second energy distribution
    tolerance : float
        Tolerance for floating-point comparisons
    name : str
        Name of the distribution for reporting
    verbose : bool
        If True, print detailed information about any differences
        
    Returns
    -------
    bool
        True if distributions are equivalent, False otherwise
    """
    if dist1 is None and dist2 is None:
        return True
    
    if dist1 is None or dist2 is None:
        if verbose:
            print(f"{name} energy distribution mismatch: One distribution is None")
        return False
    
    # Check if the objects are of the same type
    if type(dist1) != type(dist2):
        if verbose:
            print(f"{name} energy distribution mismatch: Different types "
                  f"({type(dist1).__name__} vs {type(dist2).__name__})")
        return False
    
    # Handle EnergyDependentYield objects differently
    if hasattr(dist1, 'law'):
        # Compare law numbers for EnergyDistribution objects
        if dist1.law != dist2.law:
            if verbose:
                print(f"{name} energy distribution mismatch: Different law numbers "
                      f"({dist1.law} vs {dist2.law})")
            return False
    
    # Compare common attributes based on what's available
    common_attrs = []
    
    # Attributes common to EnergyDistribution objects
    if hasattr(dist1, 'law'):
        common_attrs.extend(["idat", "nbt", "interp"])
    
    # Attributes common to both EnergyDistribution and EnergyDependentYield
    if hasattr(dist1, 'energies') and hasattr(dist2, 'energies'):
        # Compare energies
        energies1 = [float(e) for e in dist1.energies]
        energies2 = [float(e) for e in dist2.energies]
        
        if not compare_arrays(energies1, energies2, tolerance, f"{name} energies", verbose):
            return False
    
    # Compare yield values for EnergyDependentYield objects
    if hasattr(dist1, 'yields') and hasattr(dist2, 'yields'):
        yields1 = [float(y) for y in dist1.yields]
        yields2 = [float(y) for y in dist2.yields]
        
        if not compare_arrays(yields1, yields2, tolerance, f"{name} yields", verbose):
            return False
    
    # Compare standard attributes
    for attr in common_attrs:
        if hasattr(dist1, attr) and hasattr(dist2, attr):
            val1 = getattr(dist1, attr)
            val2 = getattr(dist2, attr)
            
            if isinstance(val1, list) and isinstance(val2, list):
                # Compare lists
                if len(val1) != len(val2):
                    if verbose:
                        print(f"{name} energy distribution mismatch: {attr} length differs "
                              f"({len(val1)} vs {len(val2)})")
                    return False
                
                for i, (v1, v2) in enumerate(zip(val1, val2)):
                    if v1 != v2:
                        if verbose:
                            print(f"{name} energy distribution mismatch: {attr}[{i}] differs "
                                  f"({v1} vs {v2})")
                        return False
            else:
                # Compare scalar values
                if val1 != val2:
                    if verbose:
                        print(f"{name} energy distribution mismatch: {attr} differs "
                              f"({val1} vs {val2})")
                    return False
    
    # Compare applicability data for EnergyDistribution objects
    if hasattr(dist1, "applicability_energies") and hasattr(dist2, "applicability_energies"):
        energies1 = [float(e) for e in dist1.applicability_energies]
        energies2 = [float(e) for e in dist2.applicability_energies]
        
        if not compare_arrays(energies1, energies2, tolerance, f"{name} applicability energies", verbose):
            return False
        
        probs1 = [float(p) for p in dist1.applicability_probabilities]
        probs2 = [float(p) for p in dist2.applicability_probabilities]
        
        if not compare_arrays(probs1, probs2, tolerance, f"{name} applicability probabilities", verbose):
            return False
    
    # More detailed comparison would need to check law-specific attributes
    # This would be quite extensive given the many different energy distribution laws
    # For now, we'll just check the common attributes and rely on the law number matching
    
    return True
