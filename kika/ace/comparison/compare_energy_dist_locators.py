"""
Module for comparing energy distribution locators in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_utils import compare_arrays

def compare_energy_dist_locators(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare energy distribution locators between two ACE objects."""
    # Check if both objects have energy distribution locators
    if ace1.energy_distribution_locators is None and ace2.energy_distribution_locators is None:
        return True
    
    if ace1.energy_distribution_locators is None or ace2.energy_distribution_locators is None:
        if verbose:
            print("Energy distribution locators mismatch: One ACE object has no energy distribution locators data")
        return False
    
    # Compare NXS values
    if not compare_nxs_values(ace1, ace2, verbose):
        return False
    
    # Compare neutron reaction locators
    if not compare_neutron_locators(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare photon production locators
    if not compare_photon_locators(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle production locators
    if not compare_particle_locators(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare delayed neutron locators
    if not compare_delayed_locators(ace1, ace2, tolerance, verbose):
        return False
    
    return True

def compare_nxs_values(ace1: Ace, ace2: Ace, verbose: bool) -> bool:
    """Compare NXS values related to energy distributions."""
    ed_loc1 = ace1.energy_distribution_locators
    ed_loc2 = ace2.energy_distribution_locators
    
    # Compare number of secondary neutron reactions
    if ed_loc1.num_secondary_neutron_reactions != ed_loc2.num_secondary_neutron_reactions:
        if verbose:
            print(f"Energy dist locators mismatch: Number of secondary neutron reactions differs "
                  f"({ed_loc1.num_secondary_neutron_reactions} vs {ed_loc2.num_secondary_neutron_reactions})")
        return False
    
    # Compare number of photon production reactions
    if ed_loc1.num_photon_production_reactions != ed_loc2.num_photon_production_reactions:
        if verbose:
            print(f"Energy dist locators mismatch: Number of photon production reactions differs "
                  f"({ed_loc1.num_photon_production_reactions} vs {ed_loc2.num_photon_production_reactions})")
        return False
    
    # Compare number of particle types
    if ed_loc1.num_particle_types != ed_loc2.num_particle_types:
        if verbose:
            print(f"Energy dist locators mismatch: Number of particle types differs "
                  f"({ed_loc1.num_particle_types} vs {ed_loc2.num_particle_types})")
        return False
    
    # Compare number of delayed neutron precursors
    if ed_loc1.num_delayed_neutron_precursors != ed_loc2.num_delayed_neutron_precursors:
        if verbose:
            print(f"Energy dist locators mismatch: Number of delayed neutron precursors differs "
                  f"({ed_loc1.num_delayed_neutron_precursors} vs {ed_loc2.num_delayed_neutron_precursors})")
        return False
    
    return True

def compare_neutron_locators(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare neutron reaction energy distribution locators."""
    ed_loc1 = ace1.energy_distribution_locators
    ed_loc2 = ace2.energy_distribution_locators
    
    has_neutron1 = ed_loc1.has_neutron_data
    has_neutron2 = ed_loc2.has_neutron_data
    
    if not has_neutron1 and not has_neutron2:
        return True
    
    if has_neutron1 != has_neutron2:
        if verbose:
            print("Neutron energy dist locators mismatch: Presence differs")
        return False
    
    # Compare the number of locators
    n_locators1 = len(ed_loc1.incident_neutron)
    n_locators2 = len(ed_loc2.incident_neutron)
    
    if n_locators1 != n_locators2:
        if verbose:
            print(f"Neutron energy dist locators mismatch: Count differs ({n_locators1} vs {n_locators2})")
        return False
    
    # Compare locator values
    locators1 = ed_loc1.get_neutron_locator_values()
    locators2 = ed_loc2.get_neutron_locator_values()
    
    return compare_arrays(locators1, locators2, tolerance, "Neutron energy dist locators", verbose)

def compare_photon_locators(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare photon production energy distribution locators."""
    ed_loc1 = ace1.energy_distribution_locators
    ed_loc2 = ace2.energy_distribution_locators
    
    has_photon1 = ed_loc1.has_photon_production_data
    has_photon2 = ed_loc2.has_photon_production_data
    
    if not has_photon1 and not has_photon2:
        return True
    
    if has_photon1 != has_photon2:
        if verbose:
            print("Photon energy dist locators mismatch: Presence differs")
        return False
    
    # Compare the number of locators
    n_locators1 = len(ed_loc1.photon_production)
    n_locators2 = len(ed_loc2.photon_production)
    
    if n_locators1 != n_locators2:
        if verbose:
            print(f"Photon energy dist locators mismatch: Count differs ({n_locators1} vs {n_locators2})")
        return False
    
    # Compare locator values
    locators1 = ed_loc1.get_photon_locator_values()
    locators2 = ed_loc2.get_photon_locator_values()
    
    return compare_arrays(locators1, locators2, tolerance, "Photon energy dist locators", verbose)

def compare_particle_locators(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare particle production energy distribution locators."""
    ed_loc1 = ace1.energy_distribution_locators
    ed_loc2 = ace2.energy_distribution_locators
    
    has_particle1 = ed_loc1.has_particle_production_data
    has_particle2 = ed_loc2.has_particle_production_data
    
    if not has_particle1 and not has_particle2:
        return True
    
    if has_particle1 != has_particle2:
        if verbose:
            print("Particle energy dist locators mismatch: Presence differs")
        return False
    
    # Compare number of particle types
    n_types1 = len(ed_loc1.particle_production)
    n_types2 = len(ed_loc2.particle_production)
    
    if n_types1 != n_types2:
        if verbose:
            print(f"Particle energy dist locators mismatch: Number of particle types differs ({n_types1} vs {n_types2})")
        return False
    
    # Compare each particle type's locators
    for i in range(n_types1):
        # Get locator values for this particle type
        locators1 = ed_loc1.get_particle_locator_values(i)
        locators2 = ed_loc2.get_particle_locator_values(i)
        
        # Check if both are None or both exist
        if (locators1 is None) != (locators2 is None):
            if verbose:
                print(f"Particle type {i} energy dist locators mismatch: One has locators, the other doesn't")
            return False
        
        if locators1 is None and locators2 is None:
            continue
        
        # Compare locator counts
        if len(locators1) != len(locators2):
            if verbose:
                print(f"Particle type {i} energy dist locators mismatch: Count differs ({len(locators1)} vs {len(locators2)})")
            return False
        
        # Compare locator values
        if not compare_arrays(locators1, locators2, tolerance, f"Particle type {i} energy dist locators", verbose):
            return False
    
    return True

def compare_delayed_locators(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare delayed neutron energy distribution locators."""
    ed_loc1 = ace1.energy_distribution_locators
    ed_loc2 = ace2.energy_distribution_locators
    
    has_delayed1 = ed_loc1.has_delayed_neutron_data
    has_delayed2 = ed_loc2.has_delayed_neutron_data
    
    if not has_delayed1 and not has_delayed2:
        return True
    
    if has_delayed1 != has_delayed2:
        if verbose:
            print("Delayed neutron energy dist locators mismatch: Presence differs")
        return False
    
    # Compare the number of locators
    n_locators1 = len(ed_loc1.delayed_neutron)
    n_locators2 = len(ed_loc2.delayed_neutron)
    
    if n_locators1 != n_locators2:
        if verbose:
            print(f"Delayed neutron energy dist locators mismatch: Count differs ({n_locators1} vs {n_locators2})")
        return False
    
    # Compare locator values
    locators1 = ed_loc1.get_delayed_locator_values()
    locators2 = ed_loc2.get_delayed_locator_values()
    
    return compare_arrays(locators1, locators2, tolerance, "Delayed neutron energy dist locators", verbose)
