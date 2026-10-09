# Law 66: N-body phase space

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika._utils import create_repr_section

@dataclass
class NBodyPhaseSpaceDistribution(EnergyDistribution):
    """
    Law 66: N-Body Phase Space Distribution.
    
    From ENDF-6 MF=6 LAW=6, this represents a phase space energy distribution
    for N bodies in the final state.
    
    Data format (Table 48):
    - LDAT(1): NPSX - Number of bodies in the phase space
    - LDAT(2): A_P - Total mass ratio for the NPSX particles
    - LDAT(3): INTT - Interpolation parameter
    - LDAT(4): N_P - Number of points in the distribution
    - LDAT(5) to LDAT(4+N_P): ξ_out(j) - ξ grid (between 0 and 1)
    - LDAT(5+N_P) to LDAT(4+2*N_P): PDF(j) - Probability density function
    - LDAT(5+2*N_P) to LDAT(4+3*N_P): CDF(j) - Cumulative density function
    """
    law: int = 66
    npsx: int = 0  # Number of bodies in the phase space
    ap: float = 0.0  # Total mass ratio for the NPSX particles
    intt: int = 0  # Interpolation parameter (1=histogram, 2=linear-linear)
    n_points: int = 0  # Number of points in the distribution
    xi_grid: List[float] = field(default_factory=list)  # ξ grid (between 0 and 1)
    pdf: List[float] = field(default_factory=list)  # Probability density function
    cdf: List[float] = field(default_factory=list)  # Cumulative density function
    
    def get_max_energy(self, incident_energy: float, awr: float, q_value: float = 0.0) -> float:
        """
        Calculate the maximum energy available to the outgoing particles.
        
        E_i^max = (A_P - 1)/A_P * (A / (A + 1) * E_in + Q)
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        awr : float
            Atomic weight ratio
        q_value : float, optional
            Q-value for the reaction
            
        Returns
        -------
        float
            Maximum available energy
        """
        if self.ap <= 1.0:
            return 0.0
            
        term1 = (self.ap - 1.0) / self.ap
        term2 = (awr / (awr + 1.0)) * incident_energy + q_value
        return term1 * term2

    def angular_pdf(self, incident_energy: float, cosines) -> np.ndarray:
        """
        Angular density at an incident energy: isotropic, 1/2 at every cosine.

        LAW=66 carries no angle. NJOY writes such a reaction with LOCB=-1 and
        a negative TY, and MCNP and OpenMC sample its cosine uniformly in the
        centre of mass, so the density in that frame is flat.

        Parameters
        ----------
        incident_energy : float
            Incident energy in MeV (the density does not depend on it)
        cosines : array_like
            Cosines at which to evaluate the density

        Returns
        -------
        numpy.ndarray
            0.5 at each cosine
        """
        return np.full(np.shape(cosines), 0.5)
    
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the NBodyPhaseSpaceDistribution.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'N-Body Phase Space Distribution (Law 66)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = (
            "This distribution represents a phase space energy distribution for N bodies in the\n"
            "final state. It is used for reactions where multiple particles share the available\n"
            "energy according to phase space considerations.\n\n"
            "The outgoing energy distribution follows the form:\n"
            "    E_out = T(ξ) * E_i^max\n"
            "where T(ξ) is a tabulated function sampled from a probability distribution and\n"
            "E_i^max is the maximum energy available for the outgoing particles.\n\n"
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
            "Number of Bodies (NPSX)", self.npsx, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Total Mass Ratio (AP)", f"{self.ap:.6g}", 
            width1=property_col_width, width2=value_col_width)
        
        # Interpolation type
        interp_desc = {1: "Histogram", 2: "Linear-Linear"}
        interp_str = interp_desc.get(self.intt, f"Unknown ({self.intt})")
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Interpolation Type", interp_str,
            width1=property_col_width, width2=value_col_width)
        
        # Distribution table information
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Points", self.n_points,
            width1=property_col_width, width2=value_col_width)
        
        # Show ξ grid range if available
        if len(self.xi_grid) > 0:
            xi_range = f"{min(self.xi_grid):.6g} - {max(self.xi_grid):.6g}"
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "ξ Grid Range", xi_range,
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_max_energy(incident_energy, awr, q_value)": "Calculate the maximum available energy"
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
            "# Calculate the maximum energy available for a reaction\n"
            "e_max = distribution.get_max_energy(incident_energy=14.0, awr=238.0, q_value=-2.5)\n\n"
            "# Access the tabulated probability distribution\n"
            "xi_values = distribution.xi_grid\n"
            "pdf_values = distribution.pdf\n"
            "cdf_values = distribution.cdf\n"
        )
        
        return header + description + info_table + methods_section + "\n" + example