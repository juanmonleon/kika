# Law 3: Level scattering

from dataclasses import dataclass
import numpy as np
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika._utils import create_repr_section

@dataclass
class LevelScattering(EnergyDistribution):
    """
    Law 3: Level scattering.
    
    This represents discrete two-body scattering.
    
    Data format (Table 34):
    - LDAT(1): (A + 1)/A|Q|
    - LDAT(2): (A / (A + 1))^2
    """
    law: int = 3
    aplusoaabsq: float = 0.0  # (A + 1)/A|Q| parameter
    asquare: float = 0.0      # (A / (A + 1))^2 parameter
    
    def get_cm_energy(self, incident_energy: float) -> float:
        """
        Calculate the outgoing center-of-mass energy.
        
        E_out^CM = LDAT(2) * (E - LDAT(1))
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
            
        Returns
        -------
        float
            Outgoing center-of-mass energy
        """
        return self.asquare * (incident_energy - self.aplusoaabsq)
    
    def get_lab_energy(self, incident_energy: float, cm_cosine: float) -> float:
        """
        Calculate the outgoing energy in the laboratory system.
        
        E_out^LAB = E_out^CM + {E + 2μ_CM(A+1)(E*E_out^CM)^0.5} / (A+1)^2
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        cm_cosine : float
            Cosine of the center-of-mass scattering angle
            
        Returns
        -------
        float
            Outgoing energy in the laboratory system
        """
        # First get the CM energy
        e_cm = self.get_cm_energy(incident_energy)
        
        # If negative or zero, return 0
        if e_cm <= 0.0:
            return 0.0
        
        # LDAT(2) = (A/(A+1))^2, so sqrt(LDAT(2)) = A/(A+1) and A+1 = 1/(1 - sqrt(LDAT(2)))
        a_plus_1 = 1.0 / (1.0 - np.sqrt(self.asquare))
        
        # Manual eq. 5: E_lab = E_cm + [E + 2 mu (A+1) sqrt(E E_cm)] / (A+1)^2
        term1 = incident_energy
        term2 = 2.0 * cm_cosine * a_plus_1 * np.sqrt(incident_energy * e_cm)
        second_term = (term1 + term2) / (a_plus_1 * a_plus_1)
        
        # Calculate the lab energy
        e_lab = e_cm + second_term
        
        return max(0.0, e_lab)
        
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the LevelScattering distribution.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Level Scattering Distribution (Law 3)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = (
            "This distribution represents discrete two-body scattering reactions where the\n"
            "target nucleus is excited to a discrete energy level. The distribution uses\n"
            "parameters derived from the atomic weight ratio (A) and the energy level (Q).\n\n"
            "The outgoing energy in the center-of-mass frame is calculated as:\n"
            "    E_out^CM = (A/(A+1))^2 * (E - (A+1)/A*|Q|)\n\n"
            "Then in the laboratory frame:\n"
            "    E_out^LAB = E_out^CM + {E + 2μ_CM*(A+1)*(E*E_out^CM)^0.5} / (A+1)^2\n\n"
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
            "(A+1)/A|Q| Parameter", f"{self.aplusoaabsq:.6g}", 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "(A/(A+1))^2 Parameter", f"{self.asquare:.6g}", 
            width1=property_col_width, width2=value_col_width)
        
        # Calculate A from the parameters
        if self.asquare > 0:
            a_value = np.sqrt(self.asquare) / (1 - np.sqrt(self.asquare))
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Atomic Weight Ratio (A)", f"{a_value:.6g}", 
                width1=property_col_width, width2=value_col_width)
        
        # Calculate Q from the parameters
        if self.aplusoaabsq > 0 and self.asquare > 0:
            a_value = np.sqrt(self.asquare) / (1 - np.sqrt(self.asquare))
            q_value = -self.aplusoaabsq * a_value / (a_value + 1)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Energy Level (Q)", f"{q_value:.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_cm_energy(incident_energy)": "Calculate the outgoing center-of-mass energy",
            ".get_lab_energy(incident_energy, cm_cosine)": "Calculate the laboratory energy given CM cosine"
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
            "# Calculate the CM energy for an incident neutron at 2 MeV\n"
            "e_cm = distribution.get_cm_energy(incident_energy=2.0)\n\n"
            "# Calculate the lab energy for a cosine of 0.5 in the CM frame\n"
            "e_lab = distribution.get_lab_energy(incident_energy=2.0, cm_cosine=0.5)\n"
        )
        
        return header + description + info_table + methods_section + "\n" + example