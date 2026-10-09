from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
import numpy as np
from kika.ace.classes.mt_reaction.mtr_repr import mtr_repr

@dataclass
class ReactionMTData:
    """Container for MT reaction numbers from MTR, MTRP, and MTRH blocks."""
    incident_neutron: np.ndarray = field(default_factory=lambda: np.empty(0))  # MTR Block - neutron reaction MT numbers
    photon_production: np.ndarray = field(default_factory=lambda: np.empty(0))  # MTRP Block - photon production MT numbers
    particle_production: List[np.ndarray] = field(default_factory=list)  # MTRH Block - particle production MT numbers
    secondary_neutron_mt: np.ndarray = field(default_factory=lambda: np.empty(0))  # MT numbers for reactions with secondary neutrons
    
    @property
    def has_neutron_mt_data(self) -> bool:
        """Check if neutron reaction MT numbers are available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_photon_production_mt_data(self) -> bool:
        """Check if photon production MT numbers are available."""
        return len(self.photon_production) > 0
    
    @property
    def has_particle_production_mt_data(self) -> bool:
        """Check if particle production MT numbers are available."""
        return len(self.particle_production) > 0
    
    @property
    def has_secondary_neutron_data(self) -> bool:
        """Check if secondary neutron MT numbers are available."""
        return len(self.secondary_neutron_mt) > 0
    
    def get_mt_values(self, data_type: str, idx: int = 0) -> List[int]:
        """
        Get MT numbers for a specific reaction type as integer values.
        
        Parameters
        ----------
        data_type : str
            Type of MT data to retrieve: 'neutron', 'photon', 'secondary', or 'particle'
        idx : int, optional
            Index of particle type (only used when data_type='particle'), defaults to 0
            
        Returns
        -------
        List[int]
            List of MT numbers as integers
            
        Raises
        ------
        ValueError
            If data_type is not valid or the requested data is not available
        """
        if data_type == 'neutron':
            if not self.has_neutron_mt_data:
                return []
            return [int(mt) for mt in self.incident_neutron]
        elif data_type == 'photon':
            if not self.has_photon_production_mt_data:
                return []
            return [int(mt) for mt in self.photon_production]
        elif data_type == 'secondary':
            if not self.has_secondary_neutron_data:
                return []
            return [int(mt) for mt in self.secondary_neutron_mt]
        elif data_type == 'particle':
            if not self.has_particle_production_mt_data or idx < 0 or idx >= len(self.particle_production):
                return []
            return [int(mt) for mt in self.particle_production[idx]]
        else:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'neutron', 'photon', 'secondary', or 'particle'")

    def get_particle_production_mt_dict(self) -> Dict[int, List[int]]:
        """
        Get a dictionary of MT numbers for all particle types.
        
        Returns
        -------
        Dict[int, List[int]]
            Dictionary where keys are particle type indices (1-indexed) and 
            values are lists of MT numbers as integers
            
        Example
        -------
        {
            1: [22, 28, 103, 104],  # MT numbers for particle type 1
            2: [16, 17, 22, 28],    # MT numbers for particle type 2
            ...
        }
        """
        result = {}
        for i, mt_list in enumerate(self.particle_production):
            # Use 1-indexed for particle types to match ACE convention
            result[i+1] = [int(mt) for mt in mt_list]
        return result
    
    def get_particle_production_mt_numbers(self, particle_idx: int = 0) -> Optional[np.ndarray]:
        """
        Get the list of particle production MT numbers for a specific particle type.
        
        Parameters
        ----------
        particle_idx : int
            Index of the particle type (0-based)
            
        Returns
        -------
        numpy.ndarray or None
            The list of MT numbers, or None if the particle type doesn't exist
        """
        if particle_idx < 0 or particle_idx >= len(self.particle_production):
            return None
        return self.particle_production[particle_idx]
    
    def get_num_particle_types(self) -> int:
        """Get the number of particle types with MT data."""
        return len(self.particle_production)
        
    def __repr__(self) -> str:
        """Returns a formatted string representation of the ReactionMTData object."""
        return mtr_repr(self)
