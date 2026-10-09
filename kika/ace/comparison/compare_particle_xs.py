"""
Module for comparing particle production cross section data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_utils import compare_arrays

def compare_particle_production_xs(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare particle production cross section data between two ACE objects."""
    # Check if both objects have particle production cross section data
    has_particle_xs1 = (ace1.secondary_particle_cross_sections is not None and 
                       ace1.secondary_particle_cross_sections.has_data)
    
    has_particle_xs2 = (ace2.secondary_particle_cross_sections is not None and 
                       ace2.secondary_particle_cross_sections.has_data)
    
    if not has_particle_xs1 and not has_particle_xs2:
        return True
    
    if has_particle_xs1 != has_particle_xs2:
        if verbose:
            print("Particle production cross section data mismatch: Presence differs")
        return False
    
    # Compare particle indices
    particle_indices1 = set(ace1.secondary_particle_cross_sections.particle_data.keys())
    particle_indices2 = set(ace2.secondary_particle_cross_sections.particle_data.keys())
    
    if particle_indices1 != particle_indices2:
        if verbose:
            print("Particle production cross section data mismatch: Different particle indices")
            print(f"Particle indices only in first: {sorted(particle_indices1 - particle_indices2)}")
            print(f"Particle indices only in second: {sorted(particle_indices2 - particle_indices1)}")
        return False
    
    # Compare each particle's cross section data
    for particle_idx in sorted(particle_indices1):
        xs_data1 = ace1.secondary_particle_cross_sections.particle_data[particle_idx]
        xs_data2 = ace2.secondary_particle_cross_sections.particle_data[particle_idx]
        
        # Compare energy grid indices
        # Note: The energy grid index can differ between files as long as the referenced energy points are the same
        if xs_data1.energy_grid_index != xs_data2.energy_grid_index and verbose:
            print(f"Particle type {particle_idx} cross section data: Energy grid index differs "
                  f"({xs_data1.energy_grid_index} vs {xs_data2.energy_grid_index})")
            print("  Note: Different grid indices may be acceptable if energy points are identical")
        
        # Compare number of energy points
        if xs_data1.num_energies != xs_data2.num_energies:
            if verbose:
                print(f"Particle type {particle_idx} cross section data mismatch: Number of energy points differs "
                      f"({xs_data1.num_energies} vs {xs_data2.num_energies})")
            return False
        
        # Compare cross section values
        xs_values1 = [float(xs) for xs in xs_data1.xs_values]
        xs_values2 = [float(xs) for xs in xs_data2.xs_values]
        
        if not compare_arrays(xs_values1, xs_values2, tolerance, 
                             f"Particle type {particle_idx} cross section values", verbose):
            return False
        
        # Compare heating numbers
        heating1 = [float(h) for h in xs_data1.heating_numbers]
        heating2 = [float(h) for h in xs_data2.heating_numbers]
        
        if not compare_arrays(heating1, heating2, tolerance, 
                             f"Particle type {particle_idx} heating numbers", verbose):
            return False
    
    return True

def compare_particle_production(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare all particle production data between two ACE objects."""
    # Import specific comparison functions
    from kika.ace.comparison.compare_particle_types import compare_particle_types
    from kika.ace.comparison.compare_particle_locators import compare_particle_production_locators
    
    # Compare particle types (PTYPE block)
    if not compare_particle_types(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle production locators (IXS block)
    if not compare_particle_production_locators(ace1, ace2, tolerance, verbose):
        # We don't fail the comparison just for different locator values
        pass
    
    # Compare particle production cross section data (HPD blocks)
    if not compare_particle_production_xs(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle production MT numbers (MTRH blocks)
    if hasattr(ace1, 'reaction_mt_data') and hasattr(ace2, 'reaction_mt_data'):
        from kika.ace.comparison.compare_mtr import compare_particle_mt_data
        if not compare_particle_mt_data(ace1, ace2, tolerance, verbose):
            return False
    
    # Compare particle release data (TYRH blocks)
    if hasattr(ace1, 'particle_release') and hasattr(ace2, 'particle_release'):
        from kika.ace.comparison.compare_particle_release import compare_particle_production_release
        if not compare_particle_production_release(ace1, ace2, tolerance, verbose):
            return False
    
    # Compare particle production cross sections (SIGH blocks)
    if hasattr(ace1, 'particle_production_xs') and hasattr(ace2, 'particle_production_xs'):
        from kika.ace.comparison.compare_photon_xs import compare_particle_production_xs as compare_particle_xs_blocks
        if not compare_particle_xs_blocks(ace1, ace2, tolerance, verbose):
            return False
    
    # Compare particle production angular distributions (ANDH blocks)
    if hasattr(ace1, 'angular_distributions') and hasattr(ace2, 'angular_distributions'):
        from kika.ace.comparison.compare_angular import compare_particle_angular
        if not compare_particle_angular(ace1, ace2, tolerance, verbose):
            return False
    
    # Compare particle production energy distributions (DLWH blocks)
    if hasattr(ace1, 'energy_distributions') and hasattr(ace2, 'energy_distributions'):
        from kika.ace.comparison.compare_energy_dist import compare_particle_energy_dist
        if not compare_particle_energy_dist(ace1, ace2, tolerance, verbose):
            return False
    
    return True
