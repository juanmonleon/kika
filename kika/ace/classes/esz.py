from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from kika.ace.classes.xss import xss_position

@dataclass
class EszBlock:
    """Container for ESZ block data (energy grid and cross sections)."""
    has_data: bool = False  # True if ESZ block is present
    energies: np.ndarray = field(default_factory=lambda: np.empty(0))  # Energy grid
    total_xs: np.ndarray = field(default_factory=lambda: np.empty(0))  # Total cross section
    absorption_xs: np.ndarray = field(default_factory=lambda: np.empty(0))  # Absorption cross section
    elastic_xs: np.ndarray = field(default_factory=lambda: np.empty(0))  # Elastic cross section
    heating_numbers: np.ndarray = field(default_factory=lambda: np.empty(0))  # Average heating numbers
    
    @property
    def num_energies(self) -> int:
        """Get the number of energy points in the grid."""
        return len(self.energies)
    
    def get_energy_grid(self) -> List[float]:
        """Get the energy grid as a list of float values."""
        return self.energies.tolist()
    
    def get_total_xs(self) -> List[float]:
        """Get the total cross section as a list of float values."""
        return self.total_xs.tolist()
    
    def get_absorption_xs(self) -> List[float]:
        """Get the absorption cross section as a list of float values."""
        return self.absorption_xs.tolist()
    
    def get_elastic_xs(self) -> List[float]:
        """Get the elastic cross section as a list of float values."""
        return self.elastic_xs.tolist()
    
    def get_heating_numbers(self) -> List[float]:
        """Get the heating numbers as a list of float values."""
        return self.heating_numbers.tolist()
    
    def print_indices(self):
        """Print the original XSS indices for debugging purposes."""
        print("ESZ Block Indices:")
        for name in ("energies", "total_xs", "absorption_xs", "elastic_xs", "heating_numbers"):
            block = getattr(self, name)
            start = xss_position(block)
            if len(block) and start is not None:
                print(f"  {name}: {start} to {start + len(block) - 1}")
