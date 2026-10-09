# EnergyDependentYield class

from dataclasses import dataclass, field
from typing import List
import numpy as np
from kika._utils import create_repr_section

@dataclass
class EnergyDependentYield:
    """
    Energy-dependent neutron yield data for reactions with |TY| > 100.
    
    Data format (Table 52):
    - N_R: Number of interpolation regions
    - NBT(l), l = 1,...,N_R: ENDF interpolation parameters
    - INT(l), l = 1,...,N_R: ENDF interpolation scheme
    - N_E: Number of energies
    - E(l), l = 1,...,N_E: Tabular energy points
    - Y(l), l = 1,...,N_E: Corresponding energy-dependent yields
    """
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: np.ndarray = field(default_factory=lambda: np.empty(0))  # ENDF interpolation parameters (view of xss_data)
    interp: np.ndarray = field(default_factory=lambda: np.empty(0))  # ENDF interpolation scheme (view of xss_data)
    n_energies: int = 0  # Number of energy points
    energies: np.ndarray = field(default_factory=lambda: np.empty(0))  # Tabular energy points (view of xss_data)
    yields: np.ndarray = field(default_factory=lambda: np.empty(0))  # Corresponding yields (view of xss_data)
    
    def get_yield(self, energy: float) -> float:
        """
        Get the yield value for a given incident energy.
        
        Parameters
        ----------
        energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The interpolated yield value
        """
        if len(self.energies) == 0 or len(self.yields) == 0:
            return 0.0
            
        # If energy is outside the tabulated range, use the closest value
        if energy <= self.energies[0]:
            return self.yields[0]
        if energy >= self.energies[-1]:
            return self.yields[-1]
            
        # Use linear interpolation to get yield value
        # In a full implementation, we would use the interpolation scheme from nbt and interp
        return np.interp(energy, self.energies, self.yields)
    
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the EnergyDependentYield object.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Energy-Dependent Yield Data':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the data
        description = (
            "This object contains energy-dependent yield data for reactions with |TY| > 100.\n"
            "It provides the number of neutrons produced as a function of incident energy.\n"
            "The yield data is used for reactions like (n,2n), (n,3n), etc., where the number\n"
            "of produced particles varies with incident energy.\n\n"
        )
        
        # Create a summary table of data information
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        info_table = "Data Information:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Energy Points", self.n_energies, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Interpolation Regions", self.n_interp_regions, 
            width1=property_col_width, width2=value_col_width)
        
        # Add information about energy range and yield range if data exists
        if len(self.energies) > 0:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Energy Range", f"{min(self.energies):.6g} - {max(self.energies):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        if len(self.yields) > 0:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Yield Range", f"{min(self.yields):.6g} - {max(self.yields):.6g}", 
                width1=property_col_width, width2=value_col_width)
            
            # Add yield properties like threshold energy and max yield
            threshold_idx = next((i for i, y in enumerate(self.yields) if y > 0), None)
            if threshold_idx is not None and threshold_idx > 0:
                threshold_energy = self.energies[threshold_idx-1] + (self.energies[threshold_idx] - self.energies[threshold_idx-1]) * (0 - self.yields[threshold_idx-1]) / (self.yields[threshold_idx] - self.yields[threshold_idx-1])
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Estimated Threshold Energy", f"{threshold_energy:.6g} MeV", 
                    width1=property_col_width, width2=value_col_width)
            
            max_yield_idx = list(self.yields).index(max(self.yields))
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Maximum Yield", f"{self.yields[max_yield_idx]:.6g} at {self.energies[max_yield_idx]:.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        # Interpolation information
        if len(self.interp) > 0:
            interp_schemes = {1: "Histogram", 2: "Linear-linear", 3: "Linear-log", 4: "Log-linear", 5: "Log-log"}
            scheme_strs = [interp_schemes.get(int(i), f"Unknown ({int(i)})") for i in self.interp]
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Interpolation Schemes", ", ".join(scheme_strs), 
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_yield(energy)": "Get the interpolated yield value for a given incident energy"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Add example section
        example = (
            "Example:\n"
            "--------\n"
            "# Get the yield at 14 MeV incident energy\n"
            "yield_value = yield_data.get_yield(energy=14.0)\n"
            "\n"
            "# For a typical (n,2n) reaction, this would return a value close to 2.0\n"
            "# For a typical (n,3n) reaction, this would return a value close to 3.0\n"
            "# The yield can vary with energy, especially near threshold energies\n"
        )
        
        return header + description + info_table + methods_section + "\n" + example
