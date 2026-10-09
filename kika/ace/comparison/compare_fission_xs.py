"""
Module for comparing fission cross section data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_utils import compare_arrays

def compare_fission_xs(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare fission cross section data between two ACE objects."""
    has_fission1 = (ace1.fission_xs is not None and ace1.fission_xs.has_data)
    has_fission2 = (ace2.fission_xs is not None and ace2.fission_xs.has_data)
    
    if not has_fission1 and not has_fission2:
        return True
    
    if has_fission1 != has_fission2:
        if verbose:
            print("Fission cross section mismatch: Presence differs")
        return False
    
    # Compare energy grid index
    # Note: This can differ between ACE files if the data is arranged differently
    if ace1.fission_xs.energy_grid_index != ace2.fission_xs.energy_grid_index and verbose:
        print(f"Fission cross section note: Energy grid index differs "
              f"({ace1.fission_xs.energy_grid_index} vs {ace2.fission_xs.energy_grid_index})")
        print("  Note: Different indices may be acceptable if energy points are identical")
    
    # Compare number of entries
    if ace1.fission_xs.num_entries != ace2.fission_xs.num_entries:
        if verbose:
            print(f"Fission cross section mismatch: Number of entries differs "
                  f"({ace1.fission_xs.num_entries} vs {ace2.fission_xs.num_entries})")
        return False
    
    # Compare cross section values
    xs_values1 = [float(xs) for xs in ace1.fission_xs.cross_sections]
    xs_values2 = [float(xs) for xs in ace2.fission_xs.cross_sections]
    
    # Note: We are directly comparing the cross sections here, assuming they're defined 
    # on the same energy grid. In a more sophisticated comparison, we might want to 
    # interpolate to a common grid.
    return compare_arrays(xs_values1, xs_values2, tolerance, "Fission cross section values", verbose)
