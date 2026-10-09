"""
Module for comparing ACE objects to verify equality.
This is useful for validating read/write operations to ensure data integrity.
"""

from typing import Dict, Tuple
from kika.ace.classes.ace import Ace
from kika.ace.parsers.parse_ace import read_ace

# Import utility functions
from kika.ace.comparison.compare_utils import compare_arrays

# Import comparison modules
from kika.ace.comparison.compare_header import compare_header
from kika.ace.comparison.compare_energy_grid import compare_energy_grid
from kika.ace.comparison.compare_xs import compare_cross_sections
from kika.ace.comparison.compare_angular import compare_angular_distributions
from kika.ace.comparison.compare_energy_dist import compare_energy_distributions
from kika.ace.comparison.compare_nubar import compare_nubar
from kika.ace.comparison.compare_photon_xs import compare_photon_production_xs
from kika.ace.comparison.compare_particle_xs import compare_particle_production_xs
from kika.ace.comparison.compare_delayed_neutron import compare_delayed_neutron
from kika.ace.comparison.compare_q_values import compare_q_values
from kika.ace.comparison.compare_xs_locators import compare_xs_locators
from kika.ace.comparison.compare_angular_locators import compare_angular_locators
from kika.ace.comparison.compare_mtr import compare_mt_data
from kika.ace.comparison.compare_gpd import compare_gpd
from kika.ace.comparison.compare_energy_dist_locators import compare_energy_dist_locators
from kika.ace.comparison.compare_particle_types import compare_particle_types
from kika.ace.comparison.compare_particle_locators import compare_particle_production_locators
from kika.ace.comparison.compare_particle_release import compare_particle_release
from kika.ace.comparison.compare_yield_multipliers import compare_yield_multipliers
from kika.ace.comparison.compare_unresolved_resonance import compare_unresolved_resonance
from kika.ace.comparison.compare_fission_xs import compare_fission_xs
from kika.ace.comparison.compare_reaction_counts import compare_reaction_counts

def compare_ace_files(filepath1: str, filepath2: str, tolerance: float = 1e-6, 
                     verbose: bool = True) -> bool:
    """
    Compare two ACE files by reading them and comparing the resulting ACE objects.
    
    Parameters
    ----------
    filepath1 : str
        Path to the first ACE file
    filepath2 : str
        Path to the second ACE file
    tolerance : float, optional
        Tolerance for floating-point comparisons, defaults to 1e-6
    verbose : bool, optional
        If True, print detailed information about any differences found
        
    Returns
    -------
    bool
        True if the ACE objects are equivalent, False otherwise
    """

    try:
        ace1 = read_ace(filepath1)
        ace2 = read_ace(filepath2)
    except Exception as e:
        print(f"Error reading ACE files: {e}")
        return False
        
    return compare_ace_objects(ace1, ace2, tolerance, verbose)

def compare_ace_objects(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, 
                       verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Compare two ACE objects to verify they are equivalent.
    
    Parameters
    ----------
    ace1 : Ace
        First ACE object
    ace2 : Ace
        Second ACE object
    tolerance : float, optional
        Tolerance for floating-point comparisons, defaults to 1e-6
    verbose : bool, optional
        If True, print detailed information about any differences found
        
    Returns
    -------
    Tuple[bool, Dict]
        A tuple containing:
        - Boolean indicating if the ACE objects are equivalent
        - Dictionary with detailed results for each component
    """
    if not all([ace1, ace2]):
        if verbose:
            print("One or both ACE objects are None or empty")
        return False, {}
    
    # Structure to track comparison results
    results = {
        "header": False,
        "energy_grid": False,
        "cross_sections": False,
        "angular_distributions": False,
        "angular_locators": False,
        "energy_distributions": False,
        "energy_dist_locators": False,
        "nubar": False,
        "photon_production": False,
        "photon_production_xs": False,
        "particle_types": False,
        "particle_production_xs": False,
        "particle_production_locators": False,
        "particle_release": False,
        "q_values": False,
        "xs_locators": False,
        "mt_data": False,
        "delayed_neutron": False,
        "unresolved_resonance": False,
        "fission_xs": False,
        "yield_multipliers": False,
        "reaction_counts": False,
        "xss_data": False,  
        "detailed_differences": []
    }
    
    # Compare header
    if verbose:
        print("\nComparing Header data...")
    results["header"] = compare_header(ace1, ace2, tolerance, verbose)
    
    # Compare energy grid
    if verbose:
        print("\nComparing Energy Grid...")
    results["energy_grid"] = compare_energy_grid(ace1, ace2, tolerance, verbose)
    
    # Compare cross sections
    if verbose:
        print("\nComparing Cross Sections...")
    results["cross_sections"] = compare_cross_sections(ace1, ace2, tolerance, verbose)
    
    # Compare q values
    if verbose:
        print("\nComparing Q-Values...")
    results["q_values"] = compare_q_values(ace1, ace2, tolerance, verbose)
    
    # Compare MT data
    if verbose:
        print("\nComparing MT Reaction Numbers...")
    results["mt_data"] = compare_mt_data(ace1, ace2, tolerance, verbose)
    
    # Compare angular distributions
    if verbose:
        print("\nComparing Angular Distributions...")
    results["angular_distributions"] = compare_angular_distributions(ace1, ace2, tolerance, verbose)
    
    # Compare angular locators
    if verbose:
        print("\nComparing Angular Distribution Locators...")
    results["angular_locators"] = compare_angular_locators(ace1, ace2, tolerance, verbose)
    
    # Compare energy distributions
    if verbose:
        print("\nComparing Energy Distributions...")
    results["energy_distributions"] = compare_energy_distributions(ace1, ace2, tolerance, verbose)
    
    # Compare energy distribution locators
    if verbose:
        print("\nComparing Energy Distribution Locators...")
    results["energy_dist_locators"] = compare_energy_dist_locators(ace1, ace2, tolerance, verbose)
    
    # Compare nu-bar data
    if verbose:
        print("\nComparing Nubar Data...")
    results["nubar"] = compare_nubar(ace1, ace2, tolerance, verbose)
    
    # Compare delayed neutron data
    if verbose:
        print("\nComparing Delayed Neutron Data...")
    results["delayed_neutron"] = compare_delayed_neutron(ace1, ace2, tolerance, verbose)
    
    # Compare photon production data
    if verbose:
        print("\nComparing Photon Production Data...")
    results["photon_production"] = compare_gpd(ace1, ace2, tolerance, verbose)
    
    # Compare photon production cross sections
    if verbose:
        print("\nComparing Photon Production Cross Sections...")
    results["photon_production_xs"] = compare_photon_production_xs(ace1, ace2, tolerance, verbose)
    
    # Compare cross section locators
    if verbose:
        print("\nComparing Cross Section Locators...")
    results["xs_locators"] = compare_xs_locators(ace1, ace2, tolerance, verbose)
    
    # Compare particle types
    if verbose:
        print("\nComparing Particle Types...")
    results["particle_types"] = compare_particle_types(ace1, ace2, tolerance, verbose)
    
    # Compare particle production cross sections
    if verbose:
        print("\nComparing Particle Production Cross Sections...")
    results["particle_production_xs"] = compare_particle_production_xs(ace1, ace2, tolerance, verbose)
    
    # Compare particle production locators
    if verbose:
        print("\nComparing Particle Production Locators...")
    results["particle_production_locators"] = compare_particle_production_locators(ace1, ace2, tolerance, verbose)
    
    # Compare particle release data
    if verbose:
        print("\nComparing Particle Release Data...")
    results["particle_release"] = compare_particle_release(ace1, ace2, tolerance, verbose)
    
    # Compare unresolved resonance data
    if verbose:
        print("\nComparing Unresolved Resonance Data...")
    results["unresolved_resonance"] = compare_unresolved_resonance(ace1, ace2, tolerance, verbose)
    
    # Compare fission cross sections
    if verbose:
        print("\nComparing Fission Cross Sections...")
    results["fission_xs"] = compare_fission_xs(ace1, ace2, tolerance, verbose)
    
    # Compare yield multipliers
    if verbose:
        print("\nComparing Yield Multipliers...")
    results["yield_multipliers"] = compare_yield_multipliers(ace1, ace2, tolerance, verbose)
    
    # Compare reaction counts
    if verbose:
        print("\nComparing Particle Reaction Counts...")
    results["reaction_counts"] = compare_reaction_counts(ace1, ace2, tolerance, verbose)
    
    # Add XSS data comparison
    if verbose:
        print("\nComparing Raw XSS Data...")
    results["xss_data"] = compare_xss_data(ace1, ace2, tolerance, verbose)
    
    # Calculate overall result
    overall_result = all([
        results["header"],
        results["energy_grid"],
        results["cross_sections"],
        results["angular_distributions"],
        results["angular_locators"],
        results["energy_distributions"],
        results["energy_dist_locators"],
        results["nubar"],
        results["photon_production"],
        results["photon_production_xs"],
        results["particle_types"],
        results["particle_production_xs"],
        results["particle_production_locators"],
        results["particle_release"],
        results["q_values"],
        results["xs_locators"],
        results["mt_data"],
        results["delayed_neutron"],
        results["unresolved_resonance"],
        results["fission_xs"],
        results["yield_multipliers"],
        results["reaction_counts"],
        results["xss_data"]  # Include XSS data in overall result calculation
    ])
    
    if verbose:
        print_comparison_summary(results, overall_result)
    
    return overall_result, results

def print_comparison_summary(results: Dict, overall_result: bool) -> None:
    """Print a summary of the comparison results."""
    print("\n=== ACE Object Comparison Summary ===")
    print(f"Overall result: {'MATCH' if overall_result else 'MISMATCH'}")
    print("\nComponent-level results:")
    
    # Group results by category for more organized output
    categories = {
        "Basic data": ["header", "energy_grid", "cross_sections", "q_values", "mt_data", "fission_xs"],
        "Angular data": ["angular_distributions", "angular_locators"],
        "Energy data": ["energy_distributions", "energy_dist_locators"],
        "Fission data": ["nubar", "delayed_neutron", "unresolved_resonance"],
        "Photon data": ["photon_production", "photon_production_xs"],
        "Particle data": ["particle_types", "particle_production_xs", "particle_production_locators", 
                          "particle_release", "reaction_counts"],
        "Additional data": ["xs_locators", "yield_multipliers", "xss_data"]  # Add XSS data to this category
    }
    
    # Tracking match counts
    match_count = 0
    total_count = 0
    
    # Print results by category
    for category, components in categories.items():
        print(f"\n  {category}:")
        for component in components:
            if component in results and component != "detailed_differences":
                result = results[component]
                match_status = "MATCH" if result else "MISMATCH"
                print(f"    {component.replace('_', ' ').title()}: {match_status}")
                
                # Update counters
                total_count += 1
                if result:
                    match_count += 1
    
    # Print match statistics
    match_percentage = (match_count / total_count * 100) if total_count > 0 else 0
    print(f"\nMatched components: {match_count}/{total_count} ({match_percentage:.1f}%)")
    
    if not overall_result:
        print("\nDetailed differences were found in the components marked as MISMATCH above.")
        print("For more detailed information, check the output above or set verbose=True.")

# Add this new function after the other comparison functions
def compare_xss_data(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare raw XSS data arrays between two ACE objects."""
    # Check if both objects have XSS data
    if ace1.xss_data is None and ace2.xss_data is None:
        return True
    
    if ace1.xss_data is None or ace2.xss_data is None:
        if verbose:
            print("XSS data mismatch: One ACE object has no XSS data")
        return False
    
    # Compare XSS data length
    if len(ace1.xss_data) != len(ace2.xss_data):
        if verbose:
            print(f"XSS data mismatch: Length differs ({len(ace1.xss_data)} vs {len(ace2.xss_data)})")
        return False
    
    # Skip the 0th element for 1-indexed arrays
    # Only compare values from index 1 onwards if arrays have more than 1 element
    start_idx = 1 if len(ace1.xss_data) > 1 else 0
    
    # Compare the XSS values position by position. xss_data is a numpy array
    # whose position IS the XSS index, so there is no separate index to compare.
    xss_values1 = ace1.xss_data[start_idx:]
    xss_values2 = ace2.xss_data[start_idx:]
    
    if not compare_arrays(xss_values1, xss_values2, tolerance, "XSS data values", verbose):
        return False
    
    return True
