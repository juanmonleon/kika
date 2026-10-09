"""
Module for comparing angular distribution locators in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_utils import compare_arrays, compare_floats

def compare_angular_locators(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare angular distribution locators between two ACE objects."""
    # Check if both objects have angular distribution locators
    has_locators1 = (ace1.angular_locators is not None and 
                    (ace1.angular_locators.has_neutron_data or 
                     ace1.angular_locators.has_elastic_data or
                     ace1.angular_locators.has_photon_production_data or 
                     ace1.angular_locators.has_particle_production_data))
    
    has_locators2 = (ace2.angular_locators is not None and 
                    (ace2.angular_locators.has_neutron_data or 
                     ace2.angular_locators.has_elastic_data or
                     ace2.angular_locators.has_photon_production_data or 
                     ace2.angular_locators.has_particle_production_data))
    
    if not has_locators1 and not has_locators2:
        return True
    
    if has_locators1 != has_locators2:
        if verbose:
            print("Angular distribution locators mismatch: Presence differs")
        return False
    
    # Compare NXS values
    if ace1.angular_locators.num_neutron_reactions != ace2.angular_locators.num_neutron_reactions:
        if verbose:
            print(f"Angular locators mismatch: Number of neutron reactions differs "
                  f"({ace1.angular_locators.num_neutron_reactions} vs {ace2.angular_locators.num_neutron_reactions})")
        return False
    
    if ace1.angular_locators.num_secondary_neutron_reactions != ace2.angular_locators.num_secondary_neutron_reactions:
        if verbose:
            print(f"Angular locators mismatch: Number of secondary neutron reactions differs "
                  f"({ace1.angular_locators.num_secondary_neutron_reactions} vs {ace2.angular_locators.num_secondary_neutron_reactions})")
        return False
    
    if ace1.angular_locators.num_photon_production_reactions != ace2.angular_locators.num_photon_production_reactions:
        if verbose:
            print(f"Angular locators mismatch: Number of photon production reactions differs "
                  f"({ace1.angular_locators.num_photon_production_reactions} vs {ace2.angular_locators.num_photon_production_reactions})")
        return False
    
    if ace1.angular_locators.num_particle_types != ace2.angular_locators.num_particle_types:
        if verbose:
            print(f"Angular locators mismatch: Number of particle types differs "
                  f"({ace1.angular_locators.num_particle_types} vs {ace2.angular_locators.num_particle_types})")
        return False
    
    # Compare elastic scattering locator
    if ace1.angular_locators.has_elastic_data != ace2.angular_locators.has_elastic_data:
        if verbose:
            print("Angular locators mismatch: Elastic scattering locator presence differs")
        return False
    
    if ace1.angular_locators.has_elastic_data and ace2.angular_locators.has_elastic_data:
        elastic_loc1 = ace1.angular_locators.elastic_scattering
        elastic_loc2 = ace2.angular_locators.elastic_scattering
        if not compare_floats(elastic_loc1, elastic_loc2, tolerance, "Elastic scattering angular locator", verbose):
            return False
    
    # Compare neutron reaction angular locators
    if not compare_neutron_angular_locators(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare photon production angular locators
    if not compare_photon_angular_locators(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle production angular locators
    if not compare_particle_angular_locators(ace1, ace2, tolerance, verbose):
        return False
    
    return True

def compare_neutron_angular_locators(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare neutron reaction angular locators."""
    has_neutron1 = ace1.angular_locators.has_neutron_data
    has_neutron2 = ace2.angular_locators.has_neutron_data
    
    if not has_neutron1 and not has_neutron2:
        return True
    
    if has_neutron1 != has_neutron2:
        if verbose:
            print("Neutron reaction angular locators mismatch: Presence differs")
        return False
    
    # Compare the number of locators
    n_locators1 = len(ace1.angular_locators.incident_neutron)
    n_locators2 = len(ace2.angular_locators.incident_neutron)
    
    if n_locators1 != n_locators2:
        if verbose:
            print(f"Neutron reaction angular locators mismatch: Count differs ({n_locators1} vs {n_locators2})")
        return False
    
    # Compare locator values
    locators1 = ace1.angular_locators.incident_neutron.tolist()
    locators2 = ace2.angular_locators.incident_neutron.tolist()
    
    return compare_arrays(locators1, locators2, tolerance, "Neutron reaction angular locators", verbose)

def compare_photon_angular_locators(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare photon production angular locators."""
    has_photon1 = ace1.angular_locators.has_photon_production_data
    has_photon2 = ace2.angular_locators.has_photon_production_data
    
    if not has_photon1 and not has_photon2:
        return True
    
    if has_photon1 != has_photon2:
        if verbose:
            print("Photon production angular locators mismatch: Presence differs")
        return False
    
    # Compare the number of locators
    n_locators1 = len(ace1.angular_locators.photon_production)
    n_locators2 = len(ace2.angular_locators.photon_production)
    
    if n_locators1 != n_locators2:
        if verbose:
            print(f"Photon production angular locators mismatch: Count differs ({n_locators1} vs {n_locators2})")
        return False
    
    # Compare locator values
    locators1 = ace1.angular_locators.photon_production.tolist()
    locators2 = ace2.angular_locators.photon_production.tolist()
    
    return compare_arrays(locators1, locators2, tolerance, "Photon production angular locators", verbose)

def compare_particle_angular_locators(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare particle production angular locators."""
    has_particle1 = ace1.angular_locators.has_particle_production_data
    has_particle2 = ace2.angular_locators.has_particle_production_data
    
    if not has_particle1 and not has_particle2:
        return True
    
    if has_particle1 != has_particle2:
        if verbose:
            print("Particle production angular locators mismatch: Presence differs")
        return False
    
    # Compare number of particle types
    n_types1 = len(ace1.angular_locators.particle_production)
    n_types2 = len(ace2.angular_locators.particle_production)
    
    if n_types1 != n_types2:
        if verbose:
            print(f"Particle production angular locators mismatch: Number of particle types differs ({n_types1} vs {n_types2})")
        return False
    
    # Compare each particle type's locators
    for i, (loc_list1, loc_list2) in enumerate(zip(ace1.angular_locators.particle_production,
                                                  ace2.angular_locators.particle_production)):
        # Compare number of locators for this particle type
        if len(loc_list1) != len(loc_list2):
            if verbose:
                print(f"Particle type {i} angular locators mismatch: Count differs ({len(loc_list1)} vs {len(loc_list2)})")
            return False
        
        # Compare locator values for this particle type
        locators1 = [float(loc) for loc in loc_list1]
        locators2 = [float(loc) for loc in loc_list2]
        
        if not compare_arrays(locators1, locators2, tolerance, f"Particle type {i} angular locators", verbose):
            return False
    
    return True
