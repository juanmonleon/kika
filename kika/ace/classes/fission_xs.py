from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

@dataclass
class FissionCrossSection:
    """
    Class for total fission cross section data (FIS block).
    
    The total fission cross section is the sum of first-, second-, third-, 
    and fourth-chance fission (MT = 19, 20, 21, and 38).
    """
    has_data: bool = False
    energy_grid_index: int = 0    # IE - Starting index in the energy grid
    num_entries: int = 0          # NE - Number of consecutive entries
    cross_sections: np.ndarray = field(default_factory=lambda: np.empty(0))  # Cross section values (view of xss_data)
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No fission cross section data available"
        
        output = f"Total Fission Cross Section Data\n"
        output += "=" * 50 + "\n"
        output += f"Energy grid index: {self.energy_grid_index}\n"
        output += f"Number of entries: {self.num_entries}\n"
        
        if len(self.cross_sections) > 0:
            # Extract values for min/max calculation
            xs_values = [float(xs) for xs in self.cross_sections]
            min_xs = min(xs_values)
            max_xs = max(xs_values)
            output += f"Cross section range: {min_xs:.6e} to {max_xs:.6e} barns\n"
        
        return output
