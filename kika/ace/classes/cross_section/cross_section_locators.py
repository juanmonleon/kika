from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class CrossSectionLocators:
    """Container for cross section locators from LSIG, LSIGP, and LSIGH blocks."""
    incident_neutron: np.ndarray = field(default_factory=lambda: np.empty(0))  # LSIG Block - neutron reaction xs locators
    photon_production: np.ndarray = field(default_factory=lambda: np.empty(0))  # LSIGP Block - photon production xs locators
    particle_production: List[np.ndarray] = field(default_factory=list)  # LSIGH Block - particle production xs locators
    
    @property
    def has_neutron_data(self) -> bool:
        """Check if neutron reaction cross section locators are available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_photon_production_data(self) -> bool:
        """Check if photon production cross section locators are available."""
        return len(self.photon_production) > 0
    
    @property
    def has_particle_production_data(self) -> bool:
        """Check if particle production cross section locators are available."""
        return len(self.particle_production) > 0
    
    def get_particle_production_locators(self, particle_idx: int = 0) -> Optional[np.ndarray]:
        """
        Get the list of particle production cross section locators for a specific particle type.
        
        Parameters
        ----------
        particle_idx : int
            Index of the particle type (0-based)
            
        Returns
        -------
        numpy.ndarray or None
            The list of locators, or None if the particle type doesn't exist
        """
        if particle_idx < 0 or particle_idx >= len(self.particle_production):
            return None
        return self.particle_production[particle_idx]
    
    def get_locator_value(self, locator: float) -> int:
        """
        Get the integer value of a locator.
        
        Parameters
        ----------
        locator : float
            The locator entry
            
        Returns
        -------
        int
            The integer value of the locator
        """
        return int(locator)
