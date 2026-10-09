"""
Module for comparing photon production cross section data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.classes.photon_production_xs import YieldBasedCrossSection, DirectCrossSection
from kika.ace.comparison.compare_utils import compare_arrays

def compare_photon_production_xs(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare photon production cross section data between two ACE objects."""
    # Check if both objects have photon production cross section data
    has_photon_xs1 = (ace1.photon_production_xs is not None and 
                      ace1.photon_production_xs.has_data)
    
    has_photon_xs2 = (ace2.photon_production_xs is not None and 
                      ace2.photon_production_xs.has_data)
    
    if not has_photon_xs1 and not has_photon_xs2:
        return True
    
    if has_photon_xs1 != has_photon_xs2:
        if verbose:
            print("Photon production cross section mismatch: Presence differs")
        return False
    
    # Compare reaction MT numbers
    # Keys are int MT numbers
    mt_numbers1 = set(ace1.photon_production_xs.cross_sections.keys())
    mt_numbers2 = set(ace2.photon_production_xs.cross_sections.keys())
    
    if mt_numbers1 != mt_numbers2:
        if verbose:
            print("Photon production cross section mismatch: Different MT numbers")
            print(f"MT numbers only in first: {sorted(mt_numbers1 - mt_numbers2)}")
            print(f"MT numbers only in second: {sorted(mt_numbers2 - mt_numbers1)}")
        return False
    
    # Compare each reaction's cross section data
    for mt_value in sorted(mt_numbers1):
        xs1 = ace1.photon_production_xs.cross_sections[mt_value]
        xs2 = ace2.photon_production_xs.cross_sections[mt_value]
        
        # Compare cross section types
        if type(xs1) != type(xs2):
            if verbose:
                print(f"Photon production MT={mt_value} cross section mismatch: Types differ "
                      f"({type(xs1).__name__} vs {type(xs2).__name__})")
            return False
        
        # Compare MF format type
        if xs1.mftype != xs2.mftype:
            if verbose:
                print(f"Photon production MT={mt_value} cross section mismatch: MF types differ "
                      f"({xs1.mftype} vs {xs2.mftype})")
            return False
        
        # Compare appropriate specific data based on cross section type
        if isinstance(xs1, YieldBasedCrossSection):
            if not compare_yield_based_xs(xs1, xs2, mt_value, tolerance, verbose):
                return False
        elif isinstance(xs1, DirectCrossSection):
            if not compare_direct_xs(xs1, xs2, mt_value, tolerance, verbose):
                return False
    
    return True

def compare_yield_based_xs(xs1: YieldBasedCrossSection, xs2: YieldBasedCrossSection, 
                          mt: int, tolerance: float, verbose: bool) -> bool:
    """Compare yield-based cross section data."""
    # Compare MTMULT
    if xs1.mtmult != xs2.mtmult:
        if verbose:
            print(f"Photon production MT={mt} cross section mismatch: MTMULT differs "
                  f"({xs1.mtmult} vs {xs2.mtmult})")
        return False
    
    # Compare number of interpolation regions
    if xs1.num_regions != xs2.num_regions:
        if verbose:
            print(f"Photon production MT={mt} cross section mismatch: Number of interpolation regions differs "
                  f"({xs1.num_regions} vs {xs2.num_regions})")
        return False
    
    # Compare interpolation bounds
    if len(xs1.interpolation_bounds) != len(xs2.interpolation_bounds):
        if verbose:
            print(f"Photon production MT={mt} cross section mismatch: Interpolation bounds length differs "
                  f"({len(xs1.interpolation_bounds)} vs {len(xs2.interpolation_bounds)})")
        return False
    
    bounds1 = [float(b) for b in xs1.interpolation_bounds]
    bounds2 = [float(b) for b in xs2.interpolation_bounds]
    
    if not compare_arrays(bounds1, bounds2, tolerance, f"Photon production MT={mt} interpolation bounds", verbose):
        return False
    
    # Compare interpolation schemes
    if len(xs1.interpolation_schemes) != len(xs2.interpolation_schemes):
        if verbose:
            print(f"Photon production MT={mt} cross section mismatch: Interpolation schemes length differs "
                  f"({len(xs1.interpolation_schemes)} vs {len(xs2.interpolation_schemes)})")
        return False
    
    schemes1 = [float(s) for s in xs1.interpolation_schemes]
    schemes2 = [float(s) for s in xs2.interpolation_schemes]
    
    if not compare_arrays(schemes1, schemes2, tolerance, f"Photon production MT={mt} interpolation schemes", verbose):
        return False
    
    # Compare number of energy points
    if xs1.num_energies != xs2.num_energies:
        if verbose:
            print(f"Photon production MT={mt} cross section mismatch: Number of energy points differs "
                  f"({xs1.num_energies} vs {xs2.num_energies})")
        return False
    
    # Compare energy grid
    energies1 = [float(e) for e in xs1.energies]
    energies2 = [float(e) for e in xs2.energies]
    
    if not compare_arrays(energies1, energies2, tolerance, f"Photon production MT={mt} energy grid", verbose):
        return False
    
    # Compare yield values
    yields1 = [float(y) for y in xs1.yields]
    yields2 = [float(y) for y in xs2.yields]
    
    if not compare_arrays(yields1, yields2, tolerance, f"Photon production MT={mt} yields", verbose):
        return False
    
    return True

def compare_direct_xs(xs1: DirectCrossSection, xs2: DirectCrossSection,
                     mt: int, tolerance: float, verbose: bool) -> bool:
    """Compare direct cross section data."""
    # Compare energy grid index
    if xs1.energy_grid_index != xs2.energy_grid_index:
        if verbose:
            print(f"Photon production MT={mt} cross section mismatch: Energy grid index differs "
                  f"({xs1.energy_grid_index} vs {xs2.energy_grid_index})")
        # This is not necessarily an error, as the index might be different but the data the same
        # Just issue a warning but continue checking the data
        
    # Compare number of entries
    if xs1.num_entries != xs2.num_entries:
        if verbose:
            print(f"Photon production MT={mt} cross section mismatch: Number of entries differs "
                  f"({xs1.num_entries} vs {xs2.num_entries})")
        return False
    
    # Compare cross section values
    xs_values1 = [float(x) for x in xs1.cross_sections]
    xs_values2 = [float(x) for x in xs2.cross_sections]
    
    if not compare_arrays(xs_values1, xs_values2, tolerance, f"Photon production MT={mt} cross sections", verbose):
        return False
    
    return True

def compare_particle_yield_xs(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare the SIGH blocks: per particle type, the yield-based production xs of each MT."""
    sigh1, sigh2 = ace1.particle_production_xs, ace2.particle_production_xs
    has1 = sigh1 is not None and sigh1.has_data
    has2 = sigh2 is not None and sigh2.has_data
    if not has1 and not has2:
        return True
    if has1 != has2:
        if verbose:
            print("Particle production (SIGH) mismatch: Presence differs")
        return False
    if sigh1.particle_types != sigh2.particle_types:
        if verbose:
            print("Particle production (SIGH) mismatch: Different particle types or MT lists")
        return False
    by_type1, by_type2 = sigh1.particle_cross_sections, sigh2.particle_cross_sections
    if {p: sorted(v) for p, v in by_type1.items()} != {p: sorted(v) for p, v in by_type2.items()}:
        if verbose:
            print("Particle production (SIGH) mismatch: Different (particle, MT) pairs")
        return False
    for ptype in sorted(by_type1):
        for mt in sorted(by_type1[ptype]):
            xs1, xs2 = by_type1[ptype][mt], by_type2[ptype][mt]
            if type(xs1) != type(xs2):
                if verbose:
                    print(f"Particle {ptype} MT={mt} production xs mismatch: types differ")
                return False
            if not compare_yield_based_xs(xs1, xs2, mt, tolerance, verbose):
                if verbose:
                    print(f"  (particle type {ptype})")
                return False
    return True
