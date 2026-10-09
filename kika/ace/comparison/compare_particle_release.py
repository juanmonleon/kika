"""
Module for comparing particle release data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_utils import compare_arrays

def compare_particle_release(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare particle release data between two ACE objects."""
    # Check if both objects have particle release data
    has_particle_release1 = (ace1.particle_release is not None and 
                            (ace1.particle_release.has_neutron_data or 
                             ace1.particle_release.has_particle_production_data))
    
    has_particle_release2 = (ace2.particle_release is not None and 
                            (ace2.particle_release.has_neutron_data or 
                             ace2.particle_release.has_particle_production_data))
    
    if not has_particle_release1 and not has_particle_release2:
        return True
    
    if has_particle_release1 != has_particle_release2:
        if verbose:
            print("Particle release data mismatch: Presence differs")
        return False
    
    # Compare neutron reaction particle release data
    if not compare_neutron_particle_release(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle production particle release data
    if not compare_particle_production_release(ace1, ace2, tolerance, verbose):
        return False
    
    return True

def compare_neutron_particle_release(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare neutron reaction particle release data."""
    has_neutron1 = ace1.particle_release.has_neutron_data
    has_neutron2 = ace2.particle_release.has_neutron_data
    
    if not has_neutron1 and not has_neutron2:
        return True
    
    if has_neutron1 != has_neutron2:
        if verbose:
            print("Neutron particle release mismatch: Presence differs")
        return False
    
    # Compare the number of reactions
    n_reactions1 = len(ace1.particle_release.incident_neutron)
    n_reactions2 = len(ace2.particle_release.incident_neutron)
    
    if n_reactions1 != n_reactions2:
        if verbose:
            print(f"Neutron particle release mismatch: Number of reactions differs ({n_reactions1} vs {n_reactions2})")
        return False
    
    # Compare TY values
    ty_values1 = [float(ty) for ty in ace1.particle_release.incident_neutron]
    ty_values2 = [float(ty) for ty in ace2.particle_release.incident_neutron]
    
    if not compare_arrays(ty_values1, ty_values2, tolerance, "Neutron particle release TY values", verbose):
        return False
    
    return True

def compare_particle_production_release(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare particle production particle release data."""
    has_particle1 = ace1.particle_release.has_particle_production_data
    has_particle2 = ace2.particle_release.has_particle_production_data
    
    if not has_particle1 and not has_particle2:
        return True
    
    if has_particle1 != has_particle2:
        if verbose:
            print("Particle production release mismatch: Presence differs")
        return False
    
    # Compare number of particle types
    n_types1 = len(ace1.particle_release.particle_production)
    n_types2 = len(ace2.particle_release.particle_production)
    
    if n_types1 != n_types2:
        if verbose:
            print(f"Particle production release mismatch: Number of particle types differs ({n_types1} vs {n_types2})")
        return False
    
    # Compare each particle type's TY values
    for i, (ty_list1, ty_list2) in enumerate(zip(ace1.particle_release.particle_production,
                                                 ace2.particle_release.particle_production)):
        # Compare number of reactions for this particle type
        if len(ty_list1) != len(ty_list2):
            if verbose:
                print(f"Particle type {i} release mismatch: Number of reactions differs "
                      f"({len(ty_list1)} vs {len(ty_list2)})")
            return False
        
        # Compare TY values for this particle type
        ty_values1 = [float(ty) for ty in ty_list1]
        ty_values2 = [float(ty) for ty in ty_list2]
        
        if not compare_arrays(ty_values1, ty_values2, tolerance, f"Particle type {i} release TY values", verbose):
            return False
    
    return True
