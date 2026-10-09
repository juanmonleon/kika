"""
Module for comparing unresolved resonance probability tables in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_utils import compare_arrays, compare_floats

def compare_unresolved_resonance(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare unresolved resonance probability tables between two ACE objects."""
    has_unr1 = (ace1.unresolved_resonance is not None and ace1.unresolved_resonance.has_data)
    has_unr2 = (ace2.unresolved_resonance is not None and ace2.unresolved_resonance.has_data)
    
    if not has_unr1 and not has_unr2:
        return True
    
    if has_unr1 != has_unr2:
        if verbose:
            print("Unresolved resonance mismatch: Presence differs")
        return False
    
    # Compare header values
    if not compare_unresolved_resonance_header(ace1, ace2, verbose):
        return False
    
    # Compare energy grid
    energy_values1 = [float(e) for e in ace1.unresolved_resonance.energies]
    energy_values2 = [float(e) for e in ace2.unresolved_resonance.energies]
    
    if not compare_arrays(energy_values1, energy_values2, tolerance, "Unresolved resonance energies", verbose):
        return False
    
    # Compare probability tables
    if not compare_probability_tables(ace1, ace2, tolerance, verbose):
        return False
    
    return True

def compare_unresolved_resonance_header(ace1: Ace, ace2: Ace, verbose: bool) -> bool:
    """Compare unresolved resonance header data."""
    unr1 = ace1.unresolved_resonance
    unr2 = ace2.unresolved_resonance
    
    # Compare number of energies
    if unr1.num_energies != unr2.num_energies:
        if verbose:
            print(f"Unresolved resonance mismatch: Number of energies differs ({unr1.num_energies} vs {unr2.num_energies})")
        return False
    
    # Compare table length
    if unr1.table_length != unr2.table_length:
        if verbose:
            print(f"Unresolved resonance mismatch: Table length differs ({unr1.table_length} vs {unr2.table_length})")
        return False
    
    # Compare interpolation method
    if unr1.interpolation != unr2.interpolation:
        if verbose:
            print(f"Unresolved resonance mismatch: Interpolation method differs ({unr1.interpolation} vs {unr2.interpolation})")
        return False
    
    # Compare flags
    if unr1.inelastic_flag != unr2.inelastic_flag:
        if verbose:
            print(f"Unresolved resonance mismatch: Inelastic flag differs ({unr1.inelastic_flag} vs {unr2.inelastic_flag})")
        return False
    
    if unr1.other_absorption_flag != unr2.other_absorption_flag:
        if verbose:
            print(f"Unresolved resonance mismatch: Other absorption flag differs "
                  f"({unr1.other_absorption_flag} vs {unr2.other_absorption_flag})")
        return False
    
    if unr1.factors_flag != unr2.factors_flag:
        if verbose:
            print(f"Unresolved resonance mismatch: Factors flag differs ({unr1.factors_flag} vs {unr2.factors_flag})")
        return False
    
    return True

def compare_probability_tables(ace1: Ace, ace2: Ace, tolerance: float, verbose: bool) -> bool:
    """Compare unresolved resonance probability tables."""
    unr1 = ace1.unresolved_resonance
    unr2 = ace2.unresolved_resonance
    
    # Compare number of tables
    if len(unr1.tables) != len(unr2.tables):
        if verbose:
            print(f"Unresolved resonance mismatch: Number of tables differs ({len(unr1.tables)} vs {len(unr2.tables)})")
        return False
    
    # Compare each table
    for i, (table1, table2) in enumerate(zip(unr1.tables, unr2.tables)):
        # Compare table energy
        if not compare_floats(table1.energy, table2.energy, tolerance, f"Probability table {i} energy", verbose):
            return False
        
        # Compare each component of the table
        for component in ["cumulative_probabilities", "total_xs", "elastic_xs", "fission_xs", "capture_xs", "heating_numbers"]:
            values1 = [float(entry) for entry in getattr(table1, component)]
            values2 = [float(entry) for entry in getattr(table2, component)]
            
            if not compare_arrays(values1, values2, tolerance, f"Probability table {i} {component}", verbose):
                return False
    
    return True
