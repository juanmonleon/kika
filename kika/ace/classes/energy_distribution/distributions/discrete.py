# Law 2: Discrete distributions

from dataclasses import dataclass
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika._utils import create_repr_section

@dataclass
class DiscreteEnergyDistribution(EnergyDistribution):
    """
    Law 2: one discrete photon line (manual Table 33).

    ``lp`` and ``eg`` are kept as the file states them. With LP = 0 or 1 the
    photon energy is EG; with LP = 2 (a primary photon) EG is the binding
    energy and the photon energy depends on the incident energy,
    EG + AWR/(AWR+1)·E, so it is not a number of the law alone: ask
    :meth:`get_photon_energy` for it.
    """
    law: int = 2
    lp: int = 0  # 0 or 1: the line is EG; 2: primary photon, EG is the binding energy
    eg: float = 0.0  # Photon energy (LP 0/1) or binding energy (LP 2), MeV
    
    def get_photon_energy(self, incident_energy: float, awr: float) -> float:
        """
        Calculate the photon energy based on LP indicator.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        awr : float
            Atomic weight ratio
            
        Returns
        -------
        float
            The photon energy
        """
        if self.lp == 0 or self.lp == 1:
            # For LP=0 or LP=1, the photon energy is simply EG
            return self.eg
        elif self.lp == 2:
            # For LP=2, the photon energy is EG + (AWR/(AWR+1))*E_N
            return self.eg + (awr / (awr + 1.0)) * incident_energy
        else:
            # Invalid LP, return EG
            return self.eg
    
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the DiscreteEnergyDistribution.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Discrete Energy Distribution (Law 2)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = (
            "This distribution represents discrete energy lines for outgoing particles.\n"
            "For nuclear reactions, this corresponds to discrete energy levels of the residual nucleus.\n"
            "For photon production, this represents discrete gamma ray energies.\n\n"
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
        # Add photon-specific data if applicable
        if self.lp > 0:
            lp_desc = {
                1: "Photon energy = EG",
                2: "Primary photon (energy = EG + (AWR/(AWR+1))*E_incident)"
            }
            lp_description = lp_desc.get(self.lp, f"Unknown LP={self.lp}")
            
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Photon Type (LP)", lp_description, 
                width1=property_col_width, width2=value_col_width)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Photon Energy (EG)", f"{self.eg:.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_photon_energy(incident_energy, awr)": "Calculate photon energy for given incident energy (photon data only)"
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
            "# For photon data with LP=2, calculate the photon energy at 1 MeV\n"
            "photon_energy = distribution.get_photon_energy(incident_energy=1.0, awr=238.0)\n"
        )
        
        return header + description + info_table + methods_section + "\n" + example
