"""
Module for comparing MT reaction numbers in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_utils import compare_arrays

def compare_mt_data(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare MT reaction numbers between two ACE objects."""
    # Check if both objects have MT data
    has_mt_data1 = (ace1.reaction_mt_data is not None)
    has_mt_data2 = (ace2.reaction_mt_data is not None)
    
    if not has_mt_data1 and not has_mt_data2:
        return True
    
    if has_mt_data1 != has_mt_data2:
        if verbose:
            print("MT data mismatch: Presence differs")
        return False
    
    # Compare neutron reaction MT numbers
    if not compare_neutron_mt_data(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare photon production MT numbers
    if not compare_photon_mt_data(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle production MT numbers
    if not compare_particle_mt_data(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare secondary neutron MT numbers
    if not compare_secondary_neutron_mt_data(ace1, ace2, tolerance, verbose):
        return False
    
    return True

def compare_neutron_mt_data(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare neutron reaction MT numbers."""
    has_neutron1 = ace1.reaction_mt_data.has_neutron_mt_data
    has_neutron2 = ace2.reaction_mt_data.has_neutron_mt_data
    
    if not has_neutron1 and not has_neutron2:
        return True
    
    if has_neutron1 != has_neutron2:
        if verbose:
            print("Neutron reaction MT data mismatch: Presence differs")
        return False
    
    # Compare the number of MT numbers
    n_mts1 = len(ace1.reaction_mt_data.incident_neutron)
    n_mts2 = len(ace2.reaction_mt_data.incident_neutron)
    
    if n_mts1 != n_mts2:
        if verbose:
            print(f"Neutron reaction MT data mismatch: Count differs ({n_mts1} vs {n_mts2})")
        return False
    
    # Compare MT values
    mt_values1 = [float(mt) for mt in ace1.reaction_mt_data.incident_neutron]
    mt_values2 = [float(mt) for mt in ace2.reaction_mt_data.incident_neutron]
    
    return compare_arrays(mt_values1, mt_values2, tolerance, "Neutron reaction MT numbers", verbose)

def compare_photon_mt_data(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare photon production MT numbers."""
    has_photon1 = ace1.reaction_mt_data.has_photon_production_mt_data
    has_photon2 = ace2.reaction_mt_data.has_photon_production_mt_data
    
    if not has_photon1 and not has_photon2:
        return True
    
    if has_photon1 != has_photon2:
        if verbose:
            print("Photon production MT data mismatch: Presence differs")
        return False
    
    # Compare the number of MT numbers
    n_mts1 = len(ace1.reaction_mt_data.photon_production)
    n_mts2 = len(ace2.reaction_mt_data.photon_production)
    
    if n_mts1 != n_mts2:
        if verbose:
            print(f"Photon production MT data mismatch: Count differs ({n_mts1} vs {n_mts2})")
        return False
    
    # Compare MT values
    mt_values1 = [float(mt) for mt in ace1.reaction_mt_data.photon_production]
    mt_values2 = [float(mt) for mt in ace2.reaction_mt_data.photon_production]
    
    return compare_arrays(mt_values1, mt_values2, tolerance, "Photon production MT numbers", verbose)

def compare_particle_mt_data(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare particle production MT numbers."""
    has_particle1 = ace1.reaction_mt_data.has_particle_production_mt_data
    has_particle2 = ace2.reaction_mt_data.has_particle_production_mt_data
    
    if not has_particle1 and not has_particle2:
        return True
    
    if has_particle1 != has_particle2:
        if verbose:
            print("Particle production MT data mismatch: Presence differs")
        return False
    
    # Compare number of particle types
    n_types1 = ace1.reaction_mt_data.get_num_particle_types()
    n_types2 = ace2.reaction_mt_data.get_num_particle_types()
    
    if n_types1 != n_types2:
        if verbose:
            print(f"Particle production MT data mismatch: Number of particle types differs ({n_types1} vs {n_types2})")
        return False
    
    # Compare each particle type's MT numbers
    for i in range(n_types1):
        mt_list1 = ace1.reaction_mt_data.get_particle_production_mt_numbers(i)
        mt_list2 = ace2.reaction_mt_data.get_particle_production_mt_numbers(i)
        
        # Check if both lists exist
        if (mt_list1 is None) != (mt_list2 is None):
            if verbose:
                print(f"Particle type {i} MT data mismatch: One has data, the other doesn't")
            return False
        
        if mt_list1 is None and mt_list2 is None:
            continue
        
        # Compare number of MT numbers for this particle type
        if len(mt_list1) != len(mt_list2):
            if verbose:
                print(f"Particle type {i} MT data mismatch: Count differs ({len(mt_list1)} vs {len(mt_list2)})")
            return False
        
        # Compare MT values for this particle type
        mt_values1 = [float(mt) for mt in mt_list1]
        mt_values2 = [float(mt) for mt in mt_list2]
        
        if not compare_arrays(mt_values1, mt_values2, tolerance, f"Particle type {i} MT numbers", verbose):
            return False
    
    return True

def compare_secondary_neutron_mt_data(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare secondary neutron MT numbers."""
    has_secondary1 = ace1.reaction_mt_data.has_secondary_neutron_data
    has_secondary2 = ace2.reaction_mt_data.has_secondary_neutron_data
    
    if not has_secondary1 and not has_secondary2:
        return True
    
    if has_secondary1 != has_secondary2:
        if verbose:
            print("Secondary neutron MT data mismatch: Presence differs")
        return False
    
    # Compare the number of MT numbers
    n_mts1 = len(ace1.reaction_mt_data.secondary_neutron_mt)
    n_mts2 = len(ace2.reaction_mt_data.secondary_neutron_mt)
    
    if n_mts1 != n_mts2:
        if verbose:
            print(f"Secondary neutron MT data mismatch: Count differs ({n_mts1} vs {n_mts2})")
        return False
    
    # Compare MT values
    mt_values1 = [float(mt) for mt in ace1.reaction_mt_data.secondary_neutron_mt]
    mt_values2 = [float(mt) for mt in ace2.reaction_mt_data.secondary_neutron_mt]
    
    return compare_arrays(mt_values1, mt_values2, tolerance, "Secondary neutron MT numbers", verbose)
