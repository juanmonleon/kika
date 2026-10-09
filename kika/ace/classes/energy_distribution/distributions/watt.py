# Law 11: Watt spectrum

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika._utils import create_repr_section

@dataclass
class EnergyDependentWattSpectrum(EnergyDistribution):
    """
    Law 11: Energy Dependent Watt Spectrum.
    
    From ENDF-6, MF=5, LF=11, this represents a Watt spectrum:
    f(E→E') = (1/I) * exp(-E_out / a) * sinh(sqrt(b * E_out))
    
    where I is the normalization constant and a(E), b(E) are energy-dependent parameters.
    
    Data format (Table 40):
    - N_Ra, NBT_a, INT_a: Interpolation scheme for parameter a
    - N_Ea, E_a, a: Tabulated values of parameter a(E)
    - N_Rb, NBT_b, INT_b: Interpolation scheme for parameter b
    - N_Eb, E_b, b: Tabulated values of parameter b(E)
    - U: Restriction energy (upper limit constraint)
    """
    law: int = 11
    
    # Parameter a interpolation data
    n_a_interp_regions: int = 0  # Number of interpolation regions for parameter a
    a_nbt: List[int] = field(default_factory=list)  # Parameter a interpolation region boundaries
    a_interp: List[int] = field(default_factory=list)  # Parameter a interpolation schemes
    n_a_energies: int = 0  # Number of incident energies for parameter a
    a_incident_energies: List[float] = field(default_factory=list)  # Incident energy table for a
    a_values: List[float] = field(default_factory=list)  # Parameter a values
    
    # Parameter b interpolation data
    n_b_interp_regions: int = 0  # Number of interpolation regions for parameter b
    b_nbt: List[int] = field(default_factory=list)  # Parameter b interpolation region boundaries
    b_interp: List[int] = field(default_factory=list)  # Parameter b interpolation schemes
    n_b_energies: int = 0  # Number of incident energies for parameter b
    b_incident_energies: List[float] = field(default_factory=list)  # Incident energy table for b
    b_values: List[float] = field(default_factory=list)  # Parameter b values
    
    restriction_energy: float = 0.0  # Restriction energy U (upper limit constraint)
    
    def get_a_parameter(self, incident_energy: float) -> float:
        """
        Get the a parameter for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The a parameter value
        """
        if len(self.a_incident_energies) == 0:
            return 0.0
            
        # If energy is outside the tabulated range, use the closest value
        if incident_energy <= self.a_incident_energies[0]:
            return self.a_values[0]
        if incident_energy >= self.a_incident_energies[-1]:
            return self.a_values[-1]
            
        # Use linear interpolation to get parameter a
        # In a full implementation, we would use the interpolation scheme from a_nbt and a_interp
        idx = np.searchsorted(self.a_incident_energies, incident_energy) - 1
        e_low = self.a_incident_energies[idx]
        e_high = self.a_incident_energies[idx + 1]
        a_low = self.a_values[idx]
        a_high = self.a_values[idx + 1]
        
        # Linear interpolation
        a = a_low + (a_high - a_low) * (incident_energy - e_low) / (e_high - e_low)
        return a
    
    def get_b_parameter(self, incident_energy: float) -> float:
        """
        Get the b parameter for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            The b parameter value
        """
        if len(self.b_incident_energies) == 0:
            return 0.0
            
        # If energy is outside the tabulated range, use the closest value
        if incident_energy <= self.b_incident_energies[0]:
            return self.b_values[0]
        if incident_energy >= self.b_incident_energies[-1]:
            return self.b_values[-1]
            
        # Use linear interpolation to get parameter b
        # In a full implementation, we would use the interpolation scheme from b_nbt and b_interp
        idx = np.searchsorted(self.b_incident_energies, incident_energy) - 1
        e_low = self.b_incident_energies[idx]
        e_high = self.b_incident_energies[idx + 1]
        b_low = self.b_values[idx]
        b_high = self.b_values[idx + 1]
        
        # Linear interpolation
        b = b_low + (b_high - b_low) * (incident_energy - e_low) / (e_high - e_low)
        return b
    
    def calculate_normalization_constant(self, incident_energy: float, a: float, b: float) -> float:
        """
        Calculate the normalization constant I for the Watt spectrum.
        
        I = (1/2) * sqrt(π * a^3 * b / 4) * exp(b * a / 4) *
            [erf((E − U)/a + sqrt(ab)/2) + erf(sqrt((E − U)/a − sqrt(ab)/2))]
            − a * exp(−(E − U)/a) * sinh(sqrt(b * (E − U)))
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        a : float
            The a parameter value
        b : float
            The b parameter value
            
        Returns
        -------
        float
            The normalization constant I
        """
        from scipy import special
        
        # Check for valid parameters
        if a <= 0.0 or b <= 0.0:
            return 1.0
        
        # Calculate (E − U)
        e_minus_u = incident_energy - self.restriction_energy
        if e_minus_u <= 0:
            return 1.0  # Default value if restriction exceeds incident energy
            
        # Calculate common terms
        sqrt_ab = np.sqrt(a * b)
        sqrt_e_minus_u = np.sqrt(e_minus_u)
        e_minus_u_over_a = e_minus_u / a
        
        # First term: (1/2) * sqrt(π * a^3 * b / 4) * exp(b*a/4)
        term1 = 0.5 * np.sqrt(np.pi * a**3 * b / 4.0) * np.exp(b * a / 4.0)
        
        # Error function terms
        arg1 = e_minus_u_over_a + sqrt_ab / 2.0
        arg2 = np.sqrt(e_minus_u_over_a) - sqrt_ab / 2.0
        
        if arg2 < 0:
            # Handle negative argument in second erf term
            erf_term = special.erf(arg1) - special.erf(-arg2)
        else:
            erf_term = special.erf(arg1) + special.erf(arg2)
            
        # Sinh term
        sinh_term = a * np.exp(-e_minus_u_over_a) * np.sinh(sqrt_ab * sqrt_e_minus_u / a)
        
        # Calculate I using equation 11
        normalization = term1 * erf_term - sinh_term
        
        return max(normalization, 1.0e-30)  # Prevent division by zero
    
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the EnergyDependentWattSpectrum.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Energy-Dependent Watt Spectrum (Law 11)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = (
            "This distribution represents a Watt fission spectrum with energy-dependent parameters a and b.\n"
            "The probability density function has the form:\n"
            "f(E→E') = (1/I) * exp(-E_out / a) * sinh(sqrt(b * E_out))\n\n"
            "where I is the normalization constant and the parameters a(E) and b(E) vary with incident\n"
            "energy. This spectrum is commonly used for prompt fission neutron emission.\n\n"
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
            "Restriction Energy (U)", f"{self.restriction_energy:.6g} MeV", 
            width1=property_col_width, width2=value_col_width)
        
        # Parameter a information
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Parameter a - Data Points", self.n_a_energies, 
            width1=property_col_width, width2=value_col_width)
        
        if len(self.a_incident_energies) > 0 and len(self.a_values) > 0:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Parameter a - Energy Range", 
                f"{min(self.a_incident_energies):.6g} - {max(self.a_incident_energies):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Parameter a - Value Range", 
                f"{min(self.a_values):.6g} - {max(self.a_values):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        # Parameter b information
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Parameter b - Data Points", self.n_b_energies, 
            width1=property_col_width, width2=value_col_width)
        
        if len(self.b_incident_energies) > 0 and len(self.b_values) > 0:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Parameter b - Energy Range", 
                f"{min(self.b_incident_energies):.6g} - {max(self.b_incident_energies):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Parameter b - Value Range", 
                f"{min(self.b_values):.6g} - {max(self.b_values):.6g} MeV^-1", 
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_a_parameter(incident_energy)": "Get parameter a value for a given incident energy",
            ".get_b_parameter(incident_energy)": "Get parameter b value for a given incident energy",
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
            "# Get parameters for a Watt spectrum at 2 MeV incident energy\n"
            "a = distribution.get_a_parameter(incident_energy=2.0)\n"
            "b = distribution.get_b_parameter(incident_energy=2.0)\n"
            "\n"
            "# Calculate the normalization constant\n"
            "norm_const = distribution.calculate_normalization_constant(\n"
            "    incident_energy=2.0, a=a, b=b)\n"
            "\n"
            "# Watt spectrum typical values:\n"
            "# a ≈ 0.8-1.0 MeV (average energy per fragment)\n"
            "# b ≈ 2.0-4.0 MeV^-1 (related to temperature)"
        )
        
        return header + description + info_table + methods_section + "\n" + example
