# Law 5, 9: Evaporation spectra

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika._utils import create_repr_section

@dataclass
class GeneralEvaporationSpectrum(EnergyDistribution):
    """
    Law 5: General Evaporation Spectrum.
    
    From ENDF-6, MF=5, LF=5, this represents an evaporation spectrum where:
    E_out = X(ξ) * θ(E)
    
    where X(ξ) is a randomly sampled value from equiprobable bins and
    θ(E) is the effective temperature tabulated on incident energy.
    
    Data format (Table 37):
    - N_R: Interpolation scheme between temperatures
    - NBT, INT: Interpolation parameters for temperatures
    - N_E: Number of incident energies
    - E(l): Incident energy table
    - θ(l): Effective temperature tabulated on incident energies
    - NET: Number of X's tabulated
    - X(l): Equiprobable bins
    """
    law: int = 5
    n_temp_interp_regions: int = 0  # Number of interpolation regions for temperature
    temp_nbt: List[int] = field(default_factory=list)  # Temperature interpolation region boundaries
    temp_interp: List[int] = field(default_factory=list)  # Temperature interpolation schemes
    n_incident_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy table
    temperatures: List[float] = field(default_factory=list)  # Effective temperature table
    n_equiprob_bins: int = 0  # Number of equiprobable bin boundaries
    equiprob_values: List[float] = field(default_factory=list)  # Equiprobable bin boundaries (X values)
    
    def get_temperature(self, incident_energy: float) -> float:
        """
        Get the effective temperature for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The effective temperature value
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
    
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the GeneralEvaporationSpectrum.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'General Evaporation Spectrum (Law 5)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = (
            "This distribution represents a general evaporation spectrum where the outgoing\n"
            "energy E_out = X(ξ) * θ(E), where X(ξ) is a randomly sampled value from equiprobable\n"
            "bins and θ(E) is the effective temperature tabulated on incident energy.\n\n"
            "The general evaporation spectrum is used for neutron emission in fission and certain\n"
            "compound nuclear reactions.\n\n"
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
            "Number of Equiprobable Bins", self.n_equiprob_bins, 
            width1=property_col_width, width2=value_col_width)
        
        # If we have incident energies, show the range
        if len(self.incident_energies) > 0:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Incident Energy Range", 
                f"{min(self.incident_energies):.6g} - {max(self.incident_energies):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        # If we have temperatures, show the range
        if len(self.temperatures) > 0:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Temperature Range", 
                f"{min(self.temperatures):.6g} - {max(self.temperatures):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_temperature(incident_energy)": "Get the effective temperature for a given incident energy"
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
            "# Get the effective temperature at 1 MeV incident energy\n"
            "temperature = distribution.get_temperature(incident_energy=1.0)\n"
            "\n"
            "# Access the equiprobable bin values directly\n"
            "x_values = distribution.equiprob_values\n"
        )
        
        return header + description + info_table + methods_section + "\n" + example


@dataclass
class EvaporationSpectrum(EnergyDistribution):
    """
    Law 9: Evaporation Spectrum.
    
    From ENDF-6, MF=5, LF=9, this represents an evaporation spectrum where:
    f(E→E') = (sqrt(E_out) / I) * exp(-E_out / θ(E))
    
    where I is the normalization constant:
    I = θ^2 * [1 - exp(-(E - U) / θ) * (1 + (E - U) / θ)]
    
    Data format (Table 39):
    - N_R: Interpolation scheme between temperatures
    - NBT, INT: Interpolation parameters for temperatures
    - N_E: Number of incident energies
    - E(l): Incident energy table
    - θ(l): Effective temperature table
    - U: Restriction energy (upper limit constraint)
    """
    law: int = 9
    n_temp_interp_regions: int = 0  # Number of interpolation regions for temperature
    temp_nbt: List[int] = field(default_factory=list)  # Temperature interpolation region boundaries
    temp_interp: List[int] = field(default_factory=list)  # Temperature interpolation schemes
    n_incident_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy table
    temperatures: List[float] = field(default_factory=list)  # Effective temperature table
    restriction_energy: float = 0.0  # Restriction energy U (upper limit constraint)
    
    def get_temperature(self, incident_energy: float) -> float:
        """
        Get the effective temperature for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The effective temperature value
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
        
        I = θ^2 * [1 - exp(-(E - U) / θ) * (1 + (E - U) / θ)]
        
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
        # Calculate (E - U)/θ
        arg = (incident_energy - self.restriction_energy) / temperature
        if arg <= 0:
            return temperature * temperature  # Limiting case
        
        # Calculate I using equation 9
        exp_term = np.exp(-arg) * (1.0 + arg)
        normalization = temperature * temperature * (1.0 - exp_term)
        
        return max(normalization, 1.0e-30)  # Prevent division by zero
    
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the EvaporationSpectrum.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Evaporation Spectrum (Law 9)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = (
            "This distribution represents an evaporation spectrum of form:\n"
            "f(E→E') = (sqrt(E_out) / I) * exp(-E_out / θ(E))\n\n"
            "where I is the normalization constant:\n"
            "I = θ^2 * [1 - exp(-(E - U) / θ) * (1 + (E - U) / θ)]\n\n"
            "This spectrum is commonly used for neutron emission in compound nuclear reactions.\n\n"
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
                f"{min(self.incident_energies):.6g} - {max(self.incident_energies):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        # If we have temperatures, show the range
        if len(self.temperatures) > 0:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Temperature Range", 
                f"{min(self.temperatures):.6g} - {max(self.temperatures):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_temperature(incident_energy)": "Get the effective temperature for a given incident energy",
            ".calculate_normalization_constant(incident_energy, temperature)": "Calculate the normalization constant I"
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
            "# Get the effective temperature at 2 MeV incident energy\n"
            "temperature = distribution.get_temperature(incident_energy=2.0)\n"
            "\n"
            "# Calculate the normalization constant\n"
            "norm_const = distribution.calculate_normalization_constant(incident_energy=2.0, temperature=temperature)\n"
        )
        
        return header + description + info_table + methods_section + "\n" + example
