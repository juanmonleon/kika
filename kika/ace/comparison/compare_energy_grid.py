"""
Module for comparing energy grid data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_utils import compare_arrays

def compare_energy_grid(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare energy grids between two ACE objects."""
    # First check if both objects have energy grids
    if (not hasattr(ace1, 'esz_block') or ace1.esz_block is None or 
        not hasattr(ace1.esz_block, 'energies') or ace1.esz_block.energies is None):
        if (not hasattr(ace2, 'esz_block') or ace2.esz_block is None or 
            not hasattr(ace2.esz_block, 'energies') or ace2.esz_block.energies is None):
            # Both have no energy grid, that's a match
            return True
        else:
            if verbose:
                print("Energy grid mismatch: First ACE object has no energy grid")
            return False
    
    if (not hasattr(ace2, 'esz_block') or ace2.esz_block is None or 
        not hasattr(ace2.esz_block, 'energies') or ace2.esz_block.energies is None):
        if verbose:
            print("Energy grid mismatch: Second ACE object has no energy grid")
        return False
    
    # Compare has_data flag
    if ace1.esz_block.has_data != ace2.esz_block.has_data:
        if verbose:
            print(f"ESZ block mismatch: has_data flag differs ({ace1.esz_block.has_data} vs {ace2.esz_block.has_data})")
        return False
    
    # Compare energy grid values
    energies1 = [float(e) for e in ace1.esz_block.energies]
    energies2 = [float(e) for e in ace2.esz_block.energies]
    
    if not compare_arrays(energies1, energies2, tolerance, "Energy grid", verbose):
        return False
    
    # Compare the number of energy points
    if ace1.esz_block.num_energies != ace2.esz_block.num_energies:
        if verbose:
            print(f"ESZ block mismatch: Number of energy points differs ({ace1.esz_block.num_energies} vs {ace2.esz_block.num_energies})")
        return False
    
    return True
