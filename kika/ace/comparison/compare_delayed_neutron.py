"""
Module for comparing delayed neutron data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.classes.delayed_neutron.delayed_neutron import DelayedNeutronPrecursor
from kika.ace.comparison.compare_utils import compare_arrays, compare_floats

def compare_delayed_neutron(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare delayed neutron data between two ACE objects."""
    # Check if both objects have delayed neutron data
    has_delayed1 = (ace1.delayed_neutron_data is not None and 
                    hasattr(ace1.delayed_neutron_data, 'has_delayed_neutron_data') and 
                    ace1.delayed_neutron_data.has_delayed_neutron_data)
    
    has_delayed2 = (ace2.delayed_neutron_data is not None and 
                    hasattr(ace2.delayed_neutron_data, 'has_delayed_neutron_data') and 
                    ace2.delayed_neutron_data.has_delayed_neutron_data)
    
    if not has_delayed1 and not has_delayed2:
        return True
    
    if has_delayed1 != has_delayed2:
        if verbose:
            print("Delayed neutron mismatch: Presence differs")
        return False
    
    # Compare number of precursor groups
    n_groups1 = len(ace1.delayed_neutron_data.precursors)
    n_groups2 = len(ace2.delayed_neutron_data.precursors)
    
    if n_groups1 != n_groups2:
        if verbose:
            print(f"Delayed neutron mismatch: Number of precursor groups differs ({n_groups1} vs {n_groups2})")
        return False
    
    # Compare each precursor group
    for i, (precursor1, precursor2) in enumerate(zip(ace1.delayed_neutron_data.precursors, 
                                                     ace2.delayed_neutron_data.precursors)):
        if not compare_precursor_group(precursor1, precursor2, i, tolerance, verbose):
            return False
    
    return True

def compare_precursor_group(precursor1: DelayedNeutronPrecursor, precursor2: DelayedNeutronPrecursor, 
                           group_idx: int, tolerance: float, verbose: bool) -> bool:
    """
    Compare two delayed neutron precursor groups.
    
    Parameters
    ----------
    precursor1 : DelayedNeutronPrecursor
        First precursor group
    precursor2 : DelayedNeutronPrecursor
        Second precursor group
    group_idx : int
        Group index (for reporting)
    tolerance : float
        Tolerance for floating-point comparisons
    verbose : bool
        If True, print detailed information about any differences
        
    Returns
    -------
    bool
        True if groups are equivalent, False otherwise
    """
    # Compare decay constants
    if (precursor1.decay_constant is None) != (precursor2.decay_constant is None):
        if verbose:
            print(f"Delayed neutron group {group_idx} mismatch: One has decay constant, the other doesn't")
        return False
    
    if precursor1.decay_constant is not None and precursor2.decay_constant is not None:
        decay1 = precursor1.decay_constant
        decay2 = precursor2.decay_constant
        
        if not compare_floats(decay1, decay2, tolerance, f"Delayed neutron group {group_idx} decay constant", verbose):
            return False
    
    # Compare interpolation regions
    if len(precursor1.interpolation_regions) != len(precursor2.interpolation_regions):
        if verbose:
            print(f"Delayed neutron group {group_idx} mismatch: Different number of interpolation regions "
                  f"({len(precursor1.interpolation_regions)} vs {len(precursor2.interpolation_regions)})")
        return False
    
    for i, ((nbt1, int1), (nbt2, int2)) in enumerate(zip(precursor1.interpolation_regions, 
                                                         precursor2.interpolation_regions)):
        if nbt1 != nbt2 or int1 != int2:
            if verbose:
                print(f"Delayed neutron group {group_idx} mismatch: Interpolation region {i} differs "
                      f"({nbt1},{int1}) vs ({nbt2},{int2})")
            return False
    
    # Compare energy grid
    energy_values1 = [float(e) for e in precursor1.energies]
    energy_values2 = [float(e) for e in precursor2.energies]
    
    if not compare_arrays(energy_values1, energy_values2, tolerance, 
                         f"Delayed neutron group {group_idx} energy grid", verbose):
        return False
    
    # Compare probabilities
    prob_values1 = [float(p) for p in precursor1.probabilities]
    prob_values2 = [float(p) for p in precursor2.probabilities]
    
    if not compare_arrays(prob_values1, prob_values2, tolerance, 
                         f"Delayed neutron group {group_idx} probabilities", verbose):
        return False
    
    return True
