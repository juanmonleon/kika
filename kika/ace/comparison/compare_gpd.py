"""
Module for comparing photon production data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_ace import compare_arrays

def compare_gpd(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare photon production data between two ACE objects."""
    # Check if both objects have photon production data
    has_gpd1 = (ace1.photon_production_data is not None and 
               ace1.photon_production_data.has_data)
    
    has_gpd2 = (ace2.photon_production_data is not None and 
               ace2.photon_production_data.has_data)
    
    if not has_gpd1 and not has_gpd2:
        return True
    
    if has_gpd1 != has_gpd2:
        if verbose:
            print("Photon production data mismatch: Presence differs")
        return False
    
    # Compare total photon production cross section
    xs_values1 = [float(xs) for xs in ace1.photon_production_data.total_xs]
    xs_values2 = [float(xs) for xs in ace2.photon_production_data.total_xs]
    
    if not compare_arrays(xs_values1, xs_values2, tolerance, "Total photon production cross section", verbose):
        return False
    
    # Compare outgoing photon energy data
    has_outgoing1 = ace1.photon_production_data.has_outgoing_energies
    has_outgoing2 = ace2.photon_production_data.has_outgoing_energies
    
    if has_outgoing1 != has_outgoing2:
        if verbose:
            print("Photon production data mismatch: Outgoing energy data presence differs")
        return False
    
    if has_outgoing1 and has_outgoing2:
        # Compare neutron energy boundaries
        if len(ace1.photon_production_data.neutron_energy_boundaries) != len(ace2.photon_production_data.neutron_energy_boundaries):
            if verbose:
                print("Photon production data mismatch: Number of neutron energy boundaries differs "
                      f"({len(ace1.photon_production_data.neutron_energy_boundaries)} vs "
                      f"{len(ace2.photon_production_data.neutron_energy_boundaries)})")
            return False
        
        boundaries1 = ace1.photon_production_data.neutron_energy_boundaries
        boundaries2 = ace2.photon_production_data.neutron_energy_boundaries
        
        if not compare_arrays(boundaries1, boundaries2, tolerance, "Neutron energy boundaries", verbose):
            return False
        
        # Compare outgoing photon energy matrix
        if len(ace1.photon_production_data.outgoing_energies) != len(ace2.photon_production_data.outgoing_energies):
            if verbose:
                print("Photon production data mismatch: Number of outgoing energy groups differs "
                      f"({len(ace1.photon_production_data.outgoing_energies)} vs "
                      f"{len(ace2.photon_production_data.outgoing_energies)})")
            return False
        
        # Compare each neutron energy group's outgoing photon energies
        for i, (energy_group1, energy_group2) in enumerate(zip(ace1.photon_production_data.outgoing_energies,
                                                              ace2.photon_production_data.outgoing_energies)):
            if len(energy_group1) != len(energy_group2):
                if verbose:
                    print(f"Photon production data mismatch: Number of outgoing photon energies for neutron group {i} differs "
                          f"({len(energy_group1)} vs {len(energy_group2)})")
                return False
            
            # Compare energy values for this group
            group_values1 = [float(e) for e in energy_group1]
            group_values2 = [float(e) for e in energy_group2]
            
            if not compare_arrays(group_values1, group_values2, tolerance, 
                                 f"Outgoing photon energies for neutron group {i}", verbose):
                return False
    
    return True

def compare_photon_production(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare all photon production data between two ACE objects."""
    # Compare GPD block
    if not compare_gpd(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare photon production cross sections (SIGP block)
    if hasattr(ace1, 'photon_production_xs') and hasattr(ace2, 'photon_production_xs'):
        # Import the function from compare_photon_xs
        from kika.ace.comparison.compare_photon_xs import compare_photon_production_xs
        if not compare_photon_production_xs(ace1, ace2, tolerance, verbose):
            return False
    
    # Compare photon yield multipliers if available
    if (hasattr(ace1, 'photon_yield_multipliers') and ace1.photon_yield_multipliers is not None and
        hasattr(ace2, 'photon_yield_multipliers') and ace2.photon_yield_multipliers is not None):
        
        has_data1 = hasattr(ace1.photon_yield_multipliers, 'has_data') and ace1.photon_yield_multipliers.has_data
        has_data2 = hasattr(ace2.photon_yield_multipliers, 'has_data') and ace2.photon_yield_multipliers.has_data
        
        if has_data1 != has_data2:
            if verbose:
                print("Photon yield multipliers mismatch: Presence differs")
            return False
        
        if has_data1 and has_data2:
            # We might need a more detailed comparison here depending on the structure of the yield multipliers
            if verbose:
                print("Note: Detailed comparison of photon yield multipliers is not implemented")
    
    return True
