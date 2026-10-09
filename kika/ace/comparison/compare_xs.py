"""
Module for comparing cross section data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_utils import compare_arrays

def compare_cross_sections(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare cross sections between two ACE objects."""
    # Compare standard cross sections (total, elastic, absorption)
    if not compare_standard_xs(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare reaction cross sections
    if not compare_reaction_xs(ace1, ace2, tolerance, verbose):
        return False
    
    return True

def compare_standard_xs(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare standard cross sections (total, elastic, absorption)."""
    # Check if both objects have ESZ block
    if ace1.esz_block is None and ace2.esz_block is None:
        return True
    
    if ace1.esz_block is None or ace2.esz_block is None:
        if verbose:
            print("Standard cross section mismatch: One ACE object has no ESZ block")
        return False
    
    # Compare total cross section
    total_xs1 = [float(x) for x in ace1.esz_block.total_xs] if ace1.esz_block.total_xs is not None else []
    total_xs2 = [float(x) for x in ace2.esz_block.total_xs] if ace2.esz_block.total_xs is not None else []
    
    if not compare_arrays(total_xs1, total_xs2, tolerance, "Total cross section", verbose):
        return False
    
    # Compare elastic cross section
    elastic_xs1 = [float(x) for x in ace1.esz_block.elastic_xs] if ace1.esz_block.elastic_xs is not None else []
    elastic_xs2 = [float(x) for x in ace2.esz_block.elastic_xs] if ace2.esz_block.elastic_xs is not None else []
    
    if not compare_arrays(elastic_xs1, elastic_xs2, tolerance, "Elastic cross section", verbose):
        return False
    
    # Compare absorption cross section
    abs_xs1 = [float(x) for x in ace1.esz_block.absorption_xs] if ace1.esz_block.absorption_xs is not None else []
    abs_xs2 = [float(x) for x in ace2.esz_block.absorption_xs] if ace2.esz_block.absorption_xs is not None else []
    
    if not compare_arrays(abs_xs1, abs_xs2, tolerance, "Absorption cross section", verbose):
        return False
    
    # Compare heating numbers
    heating1 = [float(x) for x in ace1.esz_block.heating_numbers] if ace1.esz_block.heating_numbers is not None else []
    heating2 = [float(x) for x in ace2.esz_block.heating_numbers] if ace2.esz_block.heating_numbers is not None else []
    
    if not compare_arrays(heating1, heating2, tolerance, "Heating numbers", verbose):
        return False
    
    return True

def compare_reaction_xs(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare reaction cross sections."""
    # Check if both objects have reaction cross section data
    if ace1.xs_data is None and ace2.xs_data is None:
        return True
    
    if ace1.xs_data is None or ace2.xs_data is None:
        if verbose:
            print("Reaction cross section mismatch: One ACE object has no XS data")
        return False
    
    # Get MT numbers from both objects
    mt_numbers1 = set(ace1.mt_numbers)
    mt_numbers2 = set(ace2.mt_numbers)
    
    # Check if MT sets are identical
    if mt_numbers1 != mt_numbers2:
        if verbose:
            print(f"Reaction cross section mismatch: Different MT numbers")
            print(f"MT numbers in first ACE: {sorted(mt_numbers1)}")
            print(f"MT numbers in second ACE: {sorted(mt_numbers2)}")
            print(f"MT numbers only in first: {sorted(mt_numbers1 - mt_numbers2)}")
            print(f"MT numbers only in second: {sorted(mt_numbers2 - mt_numbers1)}")
        return False
    
    # For each MT number, compare cross sections
    for mt in sorted(mt_numbers1):
        # Skip standard MT numbers (1, 2, 101) as they were already compared
        if mt in [1, 2, 101]:
            continue
        
        # Get cross section dataframes
        df1 = ace1.get_cross_section(mt)
        df2 = ace2.get_cross_section(mt)
        
        # Compare energy points
        if not compare_arrays(df1["Energy"].values, df2["Energy"].values, tolerance, f"MT={mt} energy points", verbose):
            return False
        
        # Compare cross section values
        col_name = f"MT={mt}"
        if not compare_arrays(df1[col_name].values, df2[col_name].values, tolerance, f"MT={mt} cross section", verbose):
            return False
    
    return True
