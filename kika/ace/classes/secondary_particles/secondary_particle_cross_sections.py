from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from kika.ace.classes.secondary_particles.secondary_particle_cross_section_repr import (
    particle_production_cross_section_repr,
    secondary_particle_cross_sections_repr
)

@dataclass
class ParticleProductionCrossSection:
    """
    Container for a single particle's production cross section and heating data.
    
    This holds the cross section for producing a specific secondary particle type
    (like neutrons, protons, etc.) and the associated heating (energy deposition).
    
    The cross section represents the probability of producing this type of particle
    in a nuclear reaction at different incident neutron energies.
    """
    energy_grid_index: int = 0  # Starting index in the energy grid
    num_energies: int = 0       # Number of energy points
    xs_values: np.ndarray = field(default_factory=lambda: np.empty(0))  # Cross section values (view of xss_data)
    heating_numbers: np.ndarray = field(default_factory=lambda: np.empty(0))  # Heating numbers (view of xss_data)
    
    def get_xs_values(self) -> List[float]:
        """
        Get the cross section values as a list of floats.
        
        Returns
        -------
        List[float]
            The cross section values in barns
        """
        return [float(entry) for entry in self.xs_values]
    
    def get_heating_values(self) -> List[float]:
        """
        Get the heating numbers as a list of floats.
        
        The heating numbers represent energy deposition in the material.
        
        Returns
        -------
        List[float]
            The heating number values
        """
        return [float(entry) for entry in self.heating_numbers]
    
    def get_xs_at_energy(self, energy: float, energy_grid: List[float]) -> float:
        """
        Get the interpolated cross section at a specific energy.
        
        Parameters
        ----------
        energy : float
            The energy at which to get the cross section (MeV)
        energy_grid : List[float]
            The complete energy grid from the ACE file
            
        Returns
        -------
        float
            The interpolated cross section value (barns)
        """
        # Determine applicable energy range from energy_grid_index
        idx_start = self.energy_grid_index - 1  # Convert from 1-based to 0-based
        if idx_start < 0:
            idx_start = 0
            
        idx_end = idx_start + self.num_energies
        
        # Get relevant energy range and cross section values
        energy_range = energy_grid[idx_start:idx_end]
        xs_values = self.get_xs_values()
        
        # Basic bounds checking
        if energy <= energy_range[0]:
            return xs_values[0]
        if energy >= energy_range[-1]:
            return xs_values[-1]
        
        # Use numpy for fast linear interpolation
        return np.interp(energy, energy_range, xs_values)
    
    def get_heating_at_energy(self, energy: float, energy_grid: List[float]) -> float:
        """
        Get the interpolated heating number at a specific energy.
        
        Parameters
        ----------
        energy : float
            The energy at which to get the heating number (MeV)
        energy_grid : List[float]
            The complete energy grid from the ACE file
            
        Returns
        -------
        float
            The interpolated heating number
        """
        # Determine applicable energy range from energy_grid_index
        idx_start = self.energy_grid_index - 1  # Convert from 1-based to 0-based
        if idx_start < 0:
            idx_start = 0
            
        idx_end = idx_start + self.num_energies
        
        # Get relevant energy range and heating values
        energy_range = energy_grid[idx_start:idx_end]
        heating_values = self.get_heating_values()
        
        # Basic bounds checking
        if energy <= energy_range[0]:
            return heating_values[0]
        if energy >= energy_range[-1]:
            return heating_values[-1]
        
        # Use numpy for fast linear interpolation
        return np.interp(energy, energy_range, heating_values)
    
    def get_energy_xs_pairs(self, energy_grid: List[float]) -> Tuple[List[float], List[float]]:
        """
        Get the energy-cross section pairs as a tuple of lists.
        
        This is useful for plotting or further analysis.
        
        Parameters
        ----------
        energy_grid : List[float]
            The complete energy grid from the ACE file
            
        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing (energies, cross_sections)
        """
        # Determine applicable energy range from energy_grid_index
        idx_start = self.energy_grid_index - 1  # Convert from 1-based to 0-based
        if idx_start < 0:
            idx_start = 0
            
        idx_end = idx_start + self.num_energies
        
        # Get relevant energy range and cross section values
        energies = energy_grid[idx_start:idx_end]
        xs_values = self.get_xs_values()
        
        return energies, xs_values
    
    def get_energy_heating_pairs(self, energy_grid: List[float]) -> Tuple[List[float], List[float]]:
        """
        Get the energy-heating pairs as a tuple of lists.
        
        This is useful for plotting or analyzing energy deposition.
        
        Parameters
        ----------
        energy_grid : List[float]
            The complete energy grid from the ACE file
            
        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing (energies, heating_numbers)
        """
        # Determine applicable energy range from energy_grid_index
        idx_start = self.energy_grid_index - 1  # Convert from 1-based to 0-based
        if idx_start < 0:
            idx_start = 0
            
        idx_end = idx_start + self.num_energies
        
        # Get relevant energy range and heating values
        energies = energy_grid[idx_start:idx_end]
        heating_values = self.get_heating_values()
        
        return energies, heating_values
    
    # Set repr method
    __repr__ = particle_production_cross_section_repr

@dataclass
class SecondaryParticleCrossSections:
    """
    Container for total production cross sections for all secondary particle types (HPD block).
    
    This holds the cross sections for producing each type of secondary particle,
    such as neutrons, protons, alphas, etc. For example, it includes the total 
    cross section for (n,p) reactions, (n,α) reactions, etc.
    
    IMPORTANT DISTINCTION:
    - This class (HPD block) contains the total production cross section for each particle type.
    - This is different from the SIGH block (ParticleProductionCrossSections), which 
      contains yield-based cross sections for specific reactions.
    """
    particle_data: Dict[int, ParticleProductionCrossSection] = field(default_factory=dict)
    has_data: bool = False
    
    def create_cross_section_data(self, **kwargs) -> ParticleProductionCrossSection:
        """
        Create a new cross section data object with the provided values.
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for ParticleProductionCrossSection constructor
            
        Returns
        -------
        ParticleProductionCrossSection
            The created cross section data object
        """
        return ParticleProductionCrossSection(**kwargs)
    
    def get_particle_types(self) -> List[int]:
        """
        Get a list of all available particle types.
        
        Returns
        -------
        List[int]
            The list of particle type indices in ascending order
        """
        return sorted(self.particle_data.keys())
    
    def get_particle_cross_section(self, particle_type: int) -> Optional[ParticleProductionCrossSection]:
        """
        Get the cross section data for a specific particle type.
        
        Parameters
        ----------
        particle_type : int
            The particle type index (1-based)
            
        Returns
        -------
        ParticleProductionCrossSection or None
            The cross section data for the specified particle type, or None if not found
        """
        return self.particle_data.get(particle_type)
    
    def get_xs_at_energy(self, particle_type: int, energy: float, energy_grid: List[float]) -> Optional[float]:
        """
        Get the production cross section for a specific particle at a specific energy.
        
        Parameters
        ----------
        particle_type : int
            The particle type index (1-based)
        energy : float
            The energy at which to get the cross section (MeV)
        energy_grid : List[float]
            The complete energy grid from the ACE file
            
        Returns
        -------
        float or None
            The production cross section value (barns), or None if not available
        """
        particle_xs = self.get_particle_cross_section(particle_type)
        if not particle_xs:
            return None
            
        return particle_xs.get_xs_at_energy(energy, energy_grid)
    
    def get_heating_at_energy(self, particle_type: int, energy: float, energy_grid: List[float]) -> Optional[float]:
        """
        Get the heating number for a specific particle at a specific energy.
        
        The heating number represents energy deposition in the material.
        
        Parameters
        ----------
        particle_type : int
            The particle type index (1-based)
        energy : float
            The energy at which to get the heating number (MeV)
        energy_grid : List[float]
            The complete energy grid from the ACE file
            
        Returns
        -------
        float or None
            The heating number, or None if not available
        """
        particle_xs = self.get_particle_cross_section(particle_type)
        if not particle_xs:
            return None
            
        return particle_xs.get_heating_at_energy(energy, energy_grid)
    
    def get_energy_xs_pairs(self, particle_type: int, energy_grid: List[float]) -> Optional[Tuple[List[float], List[float]]]:
        """
        Get energy-cross section pairs for a specific particle type.
        
        This is useful for plotting or further analysis.
        
        Parameters
        ----------
        particle_type : int
            The particle type index (1-based)
        energy_grid : List[float]
            The complete energy grid from the ACE file
            
        Returns
        -------
        Tuple[List[float], List[float]] or None
            A tuple containing (energies, cross_sections), or None if not available
        """
        particle_xs = self.get_particle_cross_section(particle_type)
        if not particle_xs:
            return None
            
        return particle_xs.get_energy_xs_pairs(energy_grid)
    
    # Set repr method
    __repr__ = secondary_particle_cross_sections_repr
