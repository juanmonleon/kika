from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from kika.ace.classes.energy_distribution import tabular_math
from kika.ace.classes.photon_production_xs_repr import (
    yield_based_cross_section_repr,
    direct_cross_section_repr,
    photon_production_cross_sections_repr,
    particle_production_cross_sections_repr
)

@dataclass
class ProductionCrossSection:
    """Base class for production cross section data (SIGP/SIGH blocks)."""
    mt: int = 0
    mftype: int = 0
    
    def get_description(self) -> str:
        """Return a human-readable description of the reaction."""
        return f"MT={self.mt}"

@dataclass
class YieldBasedCrossSection(ProductionCrossSection):
    """
    Cross section data for MF=12 or MF=16 (yield-based) format.
    
    For this format, the production cross section is calculated as:
    σ_prod(E) = Y(E) * σ_MTMULT(E)
    
    where Y(E) is the yield as a function of energy and σ_MTMULT(E) is the
    cross section for the reaction identified by MTMULT.
    """
    mtmult: int = 0  # MT number of the cross section to multiply by yield
    num_regions: int = 0  # Number of interpolation regions
    interpolation_bounds: np.ndarray = field(default_factory=lambda: np.empty(0))  # NBT array (view of xss_data)
    interpolation_schemes: np.ndarray = field(default_factory=lambda: np.empty(0))  # INT array (view of xss_data)
    num_energies: int = 0  # Number of energy points
    energies: np.ndarray = field(default_factory=lambda: np.empty(0))  # Energy grid (view of xss_data)
    yields: np.ndarray = field(default_factory=lambda: np.empty(0))  # Yield values (view of xss_data)
    
    def get_energy_values(self) -> List[float]:
        """Get the energy grid values as a list of floats."""
        return self.energies.tolist()
    
    def get_yield_values(self) -> List[float]:
        """Get the yield values as a list of floats."""
        return self.yields.tolist()
    
    def get_interpolated_yield(self, energy: float) -> float:
        """
        Get the interpolated yield value at a specific energy.
        
        Parameters
        ----------
        energy : float
            The energy at which to interpolate
            
        Returns
        -------
        float
            The interpolated yield value
        """
        return tabular_math.tab1(self.energies, self.yields, self.interpolation_bounds,
                                 self.interpolation_schemes, energy)
    def reconstruct_xs(self, energy: float, mt_xs_function) -> float:
        """
        Reconstruct the production cross section at a specific energy using equation 20.
        
        Parameters
        ----------
        energy : float
            The energy at which to calculate the cross section
        mt_xs_function : callable
            A function that returns the cross section value for a given MT and energy
            
        Returns
        -------
        float
            The reconstructed production cross section value
        """
        # Get the yield at this energy
        yield_value = self.get_interpolated_yield(energy)
        
        # Get the cross section for MTMULT at this energy
        xs_value = mt_xs_function(self.mtmult, energy)
        
        # Apply equation 20: σ_prod(E) = Y(E) * σ_MTMULT(E)
        return yield_value * xs_value
    
    def get_description(self) -> str:
        """Return a human-readable description of the reaction."""
        mf_desc = "MF=12 (photon production)" if self.mftype == 12 else "MF=16 (particle production)" 
        return f"MT={self.mt}, {mf_desc}, based on MT={self.mtmult} with yield scaling"
    
    # Set repr method
    __repr__ = yield_based_cross_section_repr

@dataclass
class DirectCrossSection(ProductionCrossSection):
    """
    Cross section data for MF=13 format (direct cross section).
    
    This format is only valid for photon production and provides cross section
    values directly without requiring additional calculations.
    """
    energy_grid_index: int = 0  # IE - Starting index in the energy grid
    num_entries: int = 0        # NE - Number of consecutive entries
    cross_sections: np.ndarray = field(default_factory=lambda: np.empty(0))  # Cross section values (view of xss_data)
    
    def get_xs_values(self) -> List[float]:
        """Get the cross section values as a list of floats."""
        return self.cross_sections.tolist()
    
    def get_value(self, energy: float, energy_grid: List[float]) -> float:
        """
        Get the cross section value at a specific energy.
        
        Parameters
        ----------
        energy : float
            The energy at which to get the cross section
        energy_grid : List[float]
            The complete energy grid
            
        Returns
        -------
        float
            The cross section value
        """
        # Determine applicable energy range from energy_grid_index
        idx_start = self.energy_grid_index - 1  # Convert from 1-based to 0-based
        if idx_start < 0:
            idx_start = 0
            
        idx_end = idx_start + self.num_entries
        
        # Get relevant energy range
        energy_range = energy_grid[idx_start:idx_end]
        xs_values = self.get_xs_values()
        
        # Below its first grid point (IE) the reaction has no cross section
        # (manual Table 51); the last point is the end of the energy grid
        if energy < energy_range[0]:
            return 0.0
        if energy >= energy_range[-1]:
            return xs_values[-1]
        
        # Use numpy for fast linear interpolation
        return np.interp(energy, energy_range, xs_values)
    
    def get_description(self) -> str:
        """Return a human-readable description of the reaction."""
        return f"MT={self.mt}, MF=13 (direct photon production)"
    
    # Set repr method
    __repr__ = direct_cross_section_repr

@dataclass
class ProductionCrossSectionContainer:
    """
    Container for production cross section data.
    """
    has_data: bool = False
    cross_sections: Dict[int, Union[YieldBasedCrossSection, DirectCrossSection]] = field(default_factory=dict)
    
    def get_reaction_xs(self, mt: int) -> Optional[Union[YieldBasedCrossSection, DirectCrossSection]]:
        """
        Get cross section data for a specific MT number.
        
        Parameters
        ----------
        mt : int
            The MT number of the reaction
            
        Returns
        -------
        YieldBasedCrossSection or DirectCrossSection or None
            The cross section data for the specified MT, or None if not found
        """
        return self.cross_sections.get(mt)
    
    def get_available_mts(self) -> List[int]:
        """
        Get a list of all available MT numbers.
        
        Returns
        -------
        List[int]
            The list of available MT numbers in ascending order
        """
        return sorted(self.cross_sections.keys())
    
    def get_xs_descriptions(self) -> Dict[int, str]:
        """
        Get human-readable descriptions of all available cross sections.
        
        Returns
        -------
        Dict[int, str]
            Dictionary mapping MT numbers to descriptions
        """
        return {mt: xs.get_description() for mt, xs in self.cross_sections.items()}
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No production cross section data available"
        
        output = f"Production Cross Section Data ({len(self.cross_sections)} reactions)\n"
        output += "=" * 50 + "\n"
        
        for mt, xs in sorted(self.cross_sections.items()):
            output += f"{xs.get_description()}\n"
        
        return output

@dataclass
class PhotonProductionCrossSections(ProductionCrossSectionContainer):
    """
    Container for photon production cross section data (SIGP block).
    
    This container holds cross section data for photon-producing reactions.
    There are two types of cross section data:
    1. Yield-based (MFTYPE = 12 or 16): Cross section is calculated as Y(E) * σ_MT(E)
    2. Direct (MFTYPE = 13): Cross section is provided directly
    """
    
    def get_photon_production_xs(self, mt: int, energy: float, energy_grid: List[float], 
                               mt_xs_function) -> Optional[float]:
        """
        Get the photon production cross section for a specific MT at a specific energy.
        
        Parameters
        ----------
        mt : int
            The MT number of the reaction
        energy : float
            The energy at which to calculate the cross section
        energy_grid : List[float]
            The complete energy grid (needed for direct cross sections)
        mt_xs_function : callable
            A function that returns the cross section for a given MT and energy
            (needed for yield-based cross sections)
            
        Returns
        -------
        float or None
            The photon production cross section, or None if not available
        """
        xs_data = self.get_reaction_xs(mt)
        if not xs_data:
            return None
            
        if isinstance(xs_data, YieldBasedCrossSection):
            return xs_data.reconstruct_xs(energy, mt_xs_function)
        else:  # DirectCrossSection
            return xs_data.get_value(energy, energy_grid)
    
    # Set repr method
    __repr__ = photon_production_cross_sections_repr

@dataclass
class ParticleProductionCrossSections(ProductionCrossSectionContainer):
    """
    Container for particle production cross section data (SIGH block).
    
    This container holds cross section data for reactions that produce
    secondary particles (neutrons, protons, alphas, etc.). The cross section
    data is organized by particle type.
    
    IMPORTANT DISTINCTION:
    - This class (SIGH block) contains yield-based cross sections for specific reactions
      calculated using the formula: σ_prod(E) = Y(E) * σ_MTMULT(E)
    - This is different from the HPD block (SecondaryParticleCrossSections), which 
      contains the total production cross section for each particle type.
    """
    particle_types: Dict[int, List[int]] = field(default_factory=dict)
    # particle type (1-based, PTYPE order) -> MT -> cross section. The
    # inherited ``cross_sections`` stays empty: an MT alone is ambiguous here.
    particle_cross_sections: Dict[int, Dict[int, YieldBasedCrossSection]] = field(default_factory=dict)
    
    def get_reaction_xs(self, mt: int, particle_type: int) -> Optional[YieldBasedCrossSection]:
        """
        Cross section of reaction ``mt`` for one particle type.
        
        Parameters
        ----------
        mt : int
            The MT number of the reaction
        particle_type : int
            The particle type index (1-based)
        """
        return self.particle_cross_sections.get(particle_type, {}).get(mt)
    
    def get_available_mts(self, particle_type: Optional[int] = None) -> List[int]:
        """MTs with a cross section, for one particle type or for any."""
        if particle_type is not None:
            return sorted(self.particle_cross_sections.get(particle_type, {}))
        return sorted({mt for by_mt in self.particle_cross_sections.values() for mt in by_mt})
    
    def get_particle_mts(self, particle_type: int) -> List[int]:
        """
        Get the list of MT numbers for a specific particle type.
        
        Parameters
        ----------
        particle_type : int
            The particle type index (1-based)
            
        Returns
        -------
        List[int]
            The list of MT numbers for the specified particle type
        """
        return self.particle_types.get(particle_type, [])
    
    def get_particle_production_xs(self, mt: int, energy: float, 
                                 mt_xs_function, particle_type: int) -> Optional[float]:
        """
        Get the particle production cross section for a specific MT at a specific energy.
        
        Parameters
        ----------
        mt : int
            The MT number of the reaction
        energy : float
            The energy at which to calculate the cross section
        mt_xs_function : callable
            A function that returns the cross section for a given MT and energy
            
        Returns
        -------
        float or None
            The particle production cross section, or None if not available
        """
        xs_data = self.get_reaction_xs(mt, particle_type)
        if not xs_data:
            return None
            
        # Particle production cross sections are always yield-based
        if isinstance(xs_data, YieldBasedCrossSection):
            return xs_data.reconstruct_xs(energy, mt_xs_function)
        
        return None
    
    def __repr__(self) -> str:
        if not self.has_data or not self.cross_sections:
            return "No particle production cross section data available"
        
        output = f"Secondary Particle Yield-Based Cross Sections (SIGH Block)\n"
        output += "=" * 60 + "\n"
        output += "This data contains cross sections for producing secondary particles\n"
        output += "in specific reactions using yields (Y) and multiplier cross sections.\n"
        output += "Formula: σ_prod(E) = Y(E) * σ_MTMULT(E)\n\n"
        
        for particle_type, mts in sorted(self.particle_types.items()):
            output += f"Particle Type {particle_type}: {len(mts)} reactions\n"
            for mt in mts:
                xs = self.get_reaction_xs(mt, particle_type)
                if xs:
                    output += f"  {xs.get_description()}\n"
        
        return output
    
    # Set repr method
    __repr__ = particle_production_cross_sections_repr
