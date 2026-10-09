# Law 7: Maxwell fission spectrum

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika._utils import create_repr_section


@dataclass
class MaxwellFissionSpectrum(EnergyDistribution):
    """
    Law 7: Maxwell fission spectrum.
    
    From ENDF-6, MF=5, LF=7, this represents a Maxwellian fission spectrum:
    f(E→E') = (sqrt(E_out) / I) * exp(-E_out / θ(E))
    
    where I is the normalization constant and θ(E) is the temperature parameter.
    
    Data format (Table 38):
    - N_R: Interpolation scheme between temperatures
    - NBT, INT: Interpolation parameters for temperatures
    - N_E: Number of incident energies
    - E(l): Incident energy table
    - θ(l): Effective temperature tabulated on incident energies
    - U: Restriction energy (upper limit constraint)
    """
    law: int = 7
    n_temp_interp_regions: int = 0  # Number of interpolation regions for temperature
    temp_nbt: List[int] = field(default_factory=list)  # Temperature interpolation region boundaries
    temp_interp: List[int] = field(default_factory=list)  # Temperature interpolation schemes
    n_incident_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy values
    temperatures: List[float] = field(default_factory=list)  # θ parameter values for each incident energy
    restriction_energy: float = 0.0  # Restriction energy U (upper limit constraint)
    
    def get_temperature(self, incident_energy: float) -> float:
        """
        Get the θ parameter for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The θ parameter value (temperature)
        """
        if len(self.incident_energies) == 0:
            return 0.0
            
        # If energy is outside the tabulated range, use the closest value
        if incident_energy <= self.incident_energies[0]:
            return self.temperatures[0]
        if incident_energy >= self.incident_energies[-1]:
            return self.temperatures[-1]
            
        # Use linear interpolation to get temperature
        # In a full implementation, we would use the interpolation scheme from temp_nbt and temp_interp
        idx = np.searchsorted(self.incident_energies, incident_energy) - 1
        e_low = self.incident_energies[idx]
        e_high = self.incident_energies[idx + 1]
        t_low = self.temperatures[idx]
        t_high = self.temperatures[idx + 1]
        
        # Linear interpolation
        t = t_low + (t_high - t_low) * (incident_energy - e_low) / (e_high - e_low)
        return t
    
    def calculate_normalization_constant(self, incident_energy: float, temperature: float) -> float:
        """
        Calculate the normalization constant I.
        
        I = (θ^3 * sqrt(π) / 2) * [ erf(√((E - U)/θ)) - √((E - U)/θ) * exp(−(E - U)/θ) ]
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        temperature : float
            The effective temperature
            
        Returns
        -------
        float
            The normalization constant I
        """
        from scipy import special
        
        # Calculate (E - U)/θ
        arg = (incident_energy - self.restriction_energy) / temperature
        if arg <= 0:
            return 1.0  # Default value if restriction exceeds incident energy
        
        # Calculate I
        sqrt_arg = np.sqrt(arg)
        erf_term = special.erf(sqrt_arg)
        exp_term = sqrt_arg * np.exp(-arg)
        
        # I = (θ^3 * sqrt(π) / 2) * [ erf(√((E - U)/θ)) - √((E - U)/θ) * exp(−(E - U)/θ) ]
        normalization = (temperature**3 * np.sqrt(np.pi) / 2.0) * (erf_term - exp_term)
        
        return max(normalization, 1.0e-30)  # Prevent division by zero
    
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the MaxwellFissionSpectrum distribution.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Maxwell Fission Spectrum (Law 7)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = (
            "This distribution represents a Maxwellian fission spectrum of the form:\n"
            "f(E→E') = (sqrt(E_out) / I) * exp(-E_out / θ(E))\n\n"
            "where I is the normalization constant and θ(E) is the temperature parameter\n"
            "that varies with incident energy.\n\n"
            "This spectrum is commonly used for prompt neutron emission in fission reactions.\n\n"
        )
        
        # Create a summary table of data information
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        info_table = "Distribution Information:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Law Number", self.law, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Incident Energies", self.n_incident_energies, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Restriction Energy (U)", f"{self.restriction_energy:.6g} MeV", 
            width1=property_col_width, width2=value_col_width)
        
        # If we have incident energies, show the range
        if len(self.incident_energies) > 0:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Incident Energy Range", 
                f"{min(self.incident_energies):.6g} - {max(self.incident_energies)::.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        # If we have temperatures, show the range
        if len(self.temperatures) > 0:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Temperature Parameter Range", 
                f"{min(self.temperatures):.6g} - {max(self.temperatures)::.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_temperature(incident_energy)": "Get the temperature parameter for a given incident energy",
            ".calculate_normalization_constant(...)": "Calculate the normalization constant I"
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
            "# Get the temperature parameter for an incident energy of 2 MeV\n"
            "temperature = distribution.get_temperature(incident_energy=2.0)\n\n"
            "# Calculate the normalization constant\n"
            "norm_const = distribution.calculate_normalization_constant(\n"
            "    incident_energy=2.0, temperature=temperature)\n\n"
            "# Access the raw data\n"
            "energy_points = distribution.incident_energies\n"
            "temp_values = distribution.temperatures\n"
        )
        
        return header + description + info_table + methods_section + "\n" + example
