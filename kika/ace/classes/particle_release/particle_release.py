from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np
from kika.ace.classes.particle_release.particle_release_repr import particle_release_repr

@dataclass
class ParticleRelease:
    """
    Container for particle release data from TYR and TYRH blocks.
    
    TY values in the TYR and TYRH blocks indicate how many particles are released
    in a reaction and in what reference frame. Values with absolute value > 100
    indicate energy-dependent yields, which are stored in the DLW, DLWP, or DLWH blocks.
    """
    incident_neutron: np.ndarray = field(default_factory=lambda: np.empty(0))  # TYR Block - neutron reaction particle release (view of xss_data)
    particle_production: List[np.ndarray] = field(default_factory=list)  # TYRH Block - particle production particle release
    
    @property
    def has_neutron_data(self) -> bool:
        """Check if neutron reaction particle release data is available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_particle_production_data(self) -> bool:
        """Check if particle production particle release data is available."""
        return len(self.particle_production) > 0
    
    def get_reaction_frame(self, ty_entry: float) -> str:
        """
        Get the reference frame for a given TY value.
        
        Parameters
        ----------
        ty_entry : float
            TY value from TYR or TYRH block
            
        Returns
        -------
        str
            'center-of-mass' or 'laboratory' depending on the sign
        """
        ty_value = int(ty_entry)
        if ty_value < 0:
            return 'center-of-mass'
        elif ty_value > 0:
            return 'laboratory'
        else:  # ty_value == 0
            return 'none'  # Absorption, no particles released
    
    def get_num_particles(self, ty_entry: Optional[float] = None) -> str:
        """
        Get the number of particles released for a given TY entry.
        If ty_entry is not provided, returns a summary of all entries.
        
        Parameters
        ----------
        ty_entry : float, optional
            TY value from TYR or TYRH block
            
        Returns
        -------
        str
            Description of the number of particles released or a summary
        """
        if ty_entry is None:
            # Return a summary if no specific TY entry provided
            if self.has_neutron_data:
                ty_types = {}
                for ty in self.incident_neutron:
                    ty_str = self._get_ty_description(ty)
                    ty_types[ty_str] = ty_types.get(ty_str, 0) + 1
                
                summary = "Neutron reaction particle release types:\n"
                for ty_str, count in ty_types.items():
                    summary += f"  {ty_str}: {count} reactions\n"
                return summary
            return "No particle release data available"
        
        return self._get_ty_description(ty_entry)
    
    def _get_ty_description(self, ty_entry: float) -> str:
        """
        Get a description for a given TY entry.
        
        Parameters
        ----------
        ty_entry : float
            TY value from TYR or TYRH block
            
        Returns
        -------
        str
            Description of what the TY value means
        """
        ty_value = int(ty_entry)
        return self._get_ty_description_value(ty_value)

    def has_energy_dependent_yield(self, ty_entry: float) -> bool:
        """
        Check if a TY entry indicates an energy-dependent yield.
        
        According to the ACE format specification, TY values with absolute
        value > 100 indicate energy-dependent yields, which are stored in
        the DLW, DLWP, or DLWH blocks.
        
        Parameters
        ----------
        ty_entry : float
            TY value from TYR or TYRH block
            
        Returns
        -------
        bool
            True if the TY value indicates an energy-dependent yield
        """
        ty_value = int(ty_entry)
        return abs(ty_value) > 100
    
    def get_energy_yield_offset(self, ty_entry: float, jed: int) -> int:
        """
        Calculate the offset for energy-dependent yield data.
        
        According to Table 52, for TY values > 100 in absolute value,
        the offset is calculated as KY = JED + |TY_i| - 101.
        
        Parameters
        ----------
        ty_entry : float
            TY value from TYR or TYRH block
        jed : int
            JED value (starting index of the corresponding energy distribution block)
            
        Returns
        -------
        int
            KY offset where the energy-dependent yield data begins
        """
        ty_value = int(ty_entry)
        return jed + abs(ty_value) - 101
    
    def _get_ty_description_value(self, ty_value: int) -> str:
        """
        Get a description for a given TY value.
        
        Parameters
        ----------
        ty_value : int
            TY value from TYR or TYRH block
            
        Returns
        -------
        str
            Description of what the TY value means
        """
        if ty_value == 0:
            return "0 (absorption)"
        elif abs(ty_value) == 19:
            return "fission (see NU block)"
        elif abs(ty_value) == 5:
            return f"5 particles ({self.get_reaction_frame(ty_value)} frame)"
        elif abs(ty_value) > 100:
            return f"energy-dependent yield (KY = JED + {abs(ty_value) - 101})"
        else:
            frame = "COM" if ty_value < 0 else "LAB"
            return f"{abs(ty_value)} particles ({frame} frame)"
    
    def get_summary(self) -> str:
        """
        Get a summary of the particle release data.
        
        Returns
        -------
        str
            Summary of the particle release data
        """
        summary = []
        
        if self.has_neutron_data:
            summary.append(f"Neutron reaction particle release: {len(self.incident_neutron)} reactions")
            
            # Group and count by TY value
            ty_counts = {}
            for ty in self.incident_neutron:
                ty_value = int(ty)
                ty_counts[ty_value] = ty_counts.get(ty_value, 0) + 1
            
            # Show the breakdown
            summary.append("TY values distribution:")
            for ty_value, count in sorted(ty_counts.items()):
                # Use the description function directly with the ty_value
                description = self._get_ty_description_value(ty_value)
                summary.append(f"  TY={ty_value} ({description}): {count} reactions")
        
        if self.has_particle_production_data:
            summary.append(f"Particle production data: {len(self.particle_production)} particle types")
            
            for i, ty_values in enumerate(self.particle_production):
                if len(ty_values) > 0:
                    summary.append(f"  Particle type {i+1}: {len(ty_values)} reactions")
        
        return "\n".join(summary) if summary else "No particle release data available"
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the ParticleRelease object."""
        return particle_release_repr(self)
