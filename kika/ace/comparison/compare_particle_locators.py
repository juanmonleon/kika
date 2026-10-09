"""
Module for comparing particle production locators in ACE format.
"""

from kika.ace.classes.ace import Ace

def compare_particle_production_locators(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare particle production locators between two ACE objects."""
    # Check if both objects have particle production locators
    has_locators1 = (ace1.secondary_particle_data_locations is not None and 
                    ace1.secondary_particle_data_locations.has_data)
    
    has_locators2 = (ace2.secondary_particle_data_locations is not None and 
                    ace2.secondary_particle_data_locations.has_data)
    
    if not has_locators1 and not has_locators2:
        return True
    
    if has_locators1 != has_locators2:
        if verbose:
            print("Particle production locators mismatch: Presence differs")
        return False
    
    # Compare number of particle types
    n_types1 = len(ace1.secondary_particle_data_locations.locator_sets)
    n_types2 = len(ace2.secondary_particle_data_locations.locator_sets)
    
    if n_types1 != n_types2:
        if verbose:
            print(f"Particle production locators mismatch: Number of particle types differs "
                  f"({n_types1} vs {n_types2})")
        return False
    
    # Compare each particle's locator set
    for i, (loc_set1, loc_set2) in enumerate(zip(ace1.secondary_particle_data_locations.locator_sets,
                                                ace2.secondary_particle_data_locations.locator_sets)):
        particle_idx = i + 1  # 1-based indexing for reporting
        
        # Compare each individual locator
        locator_attrs = ["hpd", "mtrh", "tyrh", "lsigh", "sigh", "landh", "andh", "ldlwh", "dlwh", "yh"]
        
        for attr in locator_attrs:
            val1 = getattr(loc_set1, attr)
            val2 = getattr(loc_set2, attr)
            
            if val1 != val2:
                if verbose:
                    print(f"Particle type {particle_idx} locator mismatch: {attr.upper()} differs "
                          f"({val1} vs {val2})")
                # This might not necessarily be an error - locators can differ 
                # between ACE files while the underlying data is still identical
                # Just issue a warning but continue checking other locators
                if verbose:
                    print(f"  Note: Different locator values may be acceptable if data content is identical")
    
    # For locators, we don't fail the comparison if locator values differ
    # since that can happen with equivalent data
    return True
