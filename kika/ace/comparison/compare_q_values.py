"""
Module for comparing Q-values data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_ace import compare_arrays

def compare_q_values(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare Q-values between two ACE objects."""
    # Check if both objects have Q-values data
    has_q_values1 = (ace1.q_values is not None and ace1.q_values.has_q_values)
    has_q_values2 = (ace2.q_values is not None and ace2.q_values.has_q_values)
    
    if not has_q_values1 and not has_q_values2:
        return True
    
    if has_q_values1 != has_q_values2:
        if verbose:
            print("Q-values mismatch: One object has Q-values, the other doesn't")
        return False
    
    # Compare the number of Q-values
    n_qvalues1 = len(ace1.q_values.q_values)
    n_qvalues2 = len(ace2.q_values.q_values)
    
    if n_qvalues1 != n_qvalues2:
        if verbose:
            print(f"Q-values mismatch: Number of Q-values differs ({n_qvalues1} vs {n_qvalues2})")
        return False
    
    # Compare Q-values
    qvalues1 = [float(q) for q in ace1.q_values.q_values]
    qvalues2 = [float(q) for q in ace2.q_values.q_values]
    
    return compare_arrays(qvalues1, qvalues2, tolerance, "Q-values", verbose)
