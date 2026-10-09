"""
Module for comparing nu-bar data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.classes.nubar.nubar import NuData
from kika.ace.comparison.compare_utils import compare_arrays

def compare_nubar(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare nu-bar data between two ACE objects."""
    # Check if both objects have nu-bar data
    has_nubar1 = (ace1.nubar is not None and hasattr(ace1.nubar, 'has_nubar') and ace1.nubar.has_nubar)
    has_nubar2 = (ace2.nubar is not None and hasattr(ace2.nubar, 'has_nubar') and ace2.nubar.has_nubar)
    
    if not has_nubar1 and not has_nubar2:
        return True
    
    if has_nubar1 != has_nubar2:
        if verbose:
            print("Nu-bar mismatch: Presence differs")
        return False
    
    # Check if both have the same types of nu-bar data
    has_both1 = hasattr(ace1.nubar, 'has_both_nu_types') and ace1.nubar.has_both_nu_types
    has_both2 = hasattr(ace2.nubar, 'has_both_nu_types') and ace2.nubar.has_both_nu_types
    
    if has_both1 != has_both2:
        if verbose:
            print("Nu-bar mismatch: One has both prompt and total, the other doesn't")
        return False
    
    # Compare presence of delayed nubar
    has_delayed1 = ace1.nubar.has_delayed
    has_delayed2 = ace2.nubar.has_delayed
    
    if has_delayed1 != has_delayed2:
        if verbose:
            print("Nu-bar mismatch: Delayed nubar presence differs")
        return False
    
    # Compare prompt nubar if present
    if ace1.nubar.prompt is not None and ace2.nubar.prompt is not None:
        if not compare_nudata(ace1.nubar.prompt, ace2.nubar.prompt, tolerance, "Prompt nubar", verbose):
            return False
    elif ace1.nubar.prompt is not None or ace2.nubar.prompt is not None:
        if verbose:
            print("Nu-bar mismatch: Prompt nubar present in only one object")
        return False
    
    # Compare total nubar if present
    if ace1.nubar.total is not None and ace2.nubar.total is not None:
        if not compare_nudata(ace1.nubar.total, ace2.nubar.total, tolerance, "Total nubar", verbose):
            return False
    elif ace1.nubar.total is not None or ace2.nubar.total is not None:
        if verbose:
            print("Nu-bar mismatch: Total nubar present in only one object")
        return False
    
    # Compare delayed nubar if present
    if has_delayed1 and has_delayed2:
        if not compare_nudata(ace1.nubar.delayed, ace2.nubar.delayed, tolerance, "Delayed nubar", verbose):
            return False
    
    return True

def compare_nudata(nudata1: NuData, nudata2: NuData, tolerance: float, name: str, verbose: bool) -> bool:
    """
    Compare two NuData objects for equality within tolerance.
    
    Parameters
    ----------
    nudata1 : NuData
        First NuData object
    nudata2 : NuData
        Second NuData object
    tolerance : float
        Tolerance for floating-point comparisons
    name : str
        Name of the data (for reporting)
    verbose : bool
        If True, print detailed information about any differences
        
    Returns
    -------
    bool
        True if objects are equivalent, False otherwise
    """
    # Compare format first
    if nudata1.format != nudata2.format:
        if verbose:
            print(f"{name} mismatch: Format differs ({nudata1.format} vs {nudata2.format})")
        return False
    
    # Compare polynomial data
    if nudata1.format == "polynomial":
        if nudata1.polynomial is None or nudata2.polynomial is None:
            if verbose:
                print(f"{name} mismatch: Polynomial data missing in one object")
            return False
            
        # Compare polynomial coefficients
        coef1 = [float(c) for c in nudata1.polynomial.coefficients]
        coef2 = [float(c) for c in nudata2.polynomial.coefficients]
        
        if not compare_arrays(coef1, coef2, tolerance, f"{name} polynomial coefficients", verbose):
            return False
    
    # Compare tabulated data
    elif nudata1.format == "tabulated":
        if nudata1.tabulated is None or nudata2.tabulated is None:
            if verbose:
                print(f"{name} mismatch: Tabulated data missing in one object")
            return False
        
        # Compare interpolation regions
        interp1 = nudata1.tabulated.interpolation_regions
        interp2 = nudata2.tabulated.interpolation_regions
        
        if len(interp1) != len(interp2):
            if verbose:
                print(f"{name} mismatch: Number of interpolation regions differs ({len(interp1)} vs {len(interp2)})")
            return False
        
        for i, ((nbt1, int1), (nbt2, int2)) in enumerate(zip(interp1, interp2)):
            if nbt1 != nbt2 or int1 != int2:
                if verbose:
                    print(f"{name} mismatch: Interpolation region {i} differs: ({nbt1},{int1}) vs ({nbt2},{int2})")
                return False
        
        # Compare energy grid
        energy1 = [float(e) for e in nudata1.tabulated.energies]
        energy2 = [float(e) for e in nudata2.tabulated.energies]
        
        if not compare_arrays(energy1, energy2, tolerance, f"{name} energy grid", verbose):
            return False
        
        # Compare nubar values
        values1 = [float(n) for n in nudata1.tabulated.nubar_values]
        values2 = [float(n) for n in nudata2.tabulated.nubar_values]
        
        if not compare_arrays(values1, values2, tolerance, f"{name} values", verbose):
            return False
    
    return True
