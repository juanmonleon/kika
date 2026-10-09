from dataclasses import dataclass, field
from typing import List
import numpy as np
from kika.ace.classes.energy_distribution import tabular_math
from kika.ace.classes.energy_distribution.types import EnergyDistributionType
from kika._utils import create_repr_section


@dataclass
class EnergyDistribution:
    """Base class for energy distributions of secondary particles."""
    law: int = 0  # Energy distribution law number
    idat: int = 0  # Offset to distribution data in XSS array (relative to JED)
    
    # Law applicability parameters
    applicability_energies: np.ndarray = field(default_factory=lambda: np.empty(0))  # Energies at which to check if law applies (view of xss_data)
    applicability_probabilities: np.ndarray = field(default_factory=lambda: np.empty(0))  # Probability of law validity at each energy (view of xss_data)
    # NBT/INT of the law-validity table P(E) (manual Table 25). Kept apart from
    # ``nbt``/``interp``, which the tabular laws use for the interpolation
    # between their incident-energy tables (LDAT(1..)).
    applicability_nbt: List[int] = field(default_factory=list)
    applicability_interp: List[int] = field(default_factory=list)
    nbt: List[int] = field(default_factory=list)  # NBT interpolation parameters
    interp: List[int] = field(default_factory=list)  # INT interpolation scheme
    
    @property
    def law_name(self) -> str:
        """Get a descriptive name for the energy distribution law."""
        law_names = {
            1: "Tabular Energy Distribution",
            2: "Discrete Energy Distribution",
            3: "Level Scattering",
            4: "Continuous Energy-Angle Distribution",
            5: "General Evaporation Spectrum",
            7: "Maxwell Fission Spectrum",
            9: "Evaporation Spectrum",
            11: "Energy-dependent Watt Spectrum",
            22: "Tabular Linear Functions of Incident Energy Out",
            24: "Tabular Energy Multipliers",
            44: "Kalbach-Mann Correlated Energy-Angle Distribution",
            61: "Correlated Energy-Angle Distribution",
            66: "N-body Phase Space Distribution"
        }
        return law_names.get(self.law, f"Unknown Law {self.law}")
    
    def get_applicability_probability(self, energy: float) -> float:
        """
        Get the probability that this law applies at the given incident energy.
        
        If the law is the only law for a reaction, this returns 1.0.
        Otherwise, it interpolates between probability values.
        
        Parameters
        ----------
        energy : float
            The incident particle energy
            
        Returns
        -------
        float
            Probability between 0 and 1 that this law applies
        """
        # A single law has no table: it always applies
        if len(self.applicability_energies) == 0 or len(self.applicability_probabilities) == 0:
            return 1.0
        return tabular_math.tab1(self.applicability_energies, self.applicability_probabilities,
                                 self.applicability_nbt, self.applicability_interp, energy)
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of an EnergyDistribution object.
        
        This representation provides an overview of the energy distribution law and its parameters.
        
        Returns
        -------
        str
            Formatted string representation of the EnergyDistribution
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Energy Distribution Law ' + str(self.law):^{header_width}}\n"
        header += f"{self.law_name:^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = f"This object represents energy distribution Law {self.law} ({self.law_name}).\n"
        description += "Energy distributions determine the outgoing energy of secondary particles in nuclear reactions.\n\n"
        
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
            "Law Name", self.law_name, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "IDAT (Offset to data)", self.idat, 
            width1=property_col_width, width2=value_col_width)
        
        # Show applicability information if available
        has_applicability = (len(self.applicability_energies) > 0 and 
                            len(self.applicability_probabilities) > 0)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Has Applicability Data", "Yes" if has_applicability else "No", 
            width1=property_col_width, width2=value_col_width)
        
        if has_applicability:
            n_points = len(self.applicability_energies)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Applicability Data Points", n_points, 
                width1=property_col_width, width2=value_col_width)
            
            if n_points > 0:
                e_min = float(self.applicability_energies[0])
                e_max = float(self.applicability_energies[-1])
                energy_range = f"{e_min:.6g} - {e_max:.6g} MeV"
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Energy Range", energy_range,
                    width1=property_col_width, width2=value_col_width)
        
        # Add law-specific information based on type
        class_name = self.__class__.__name__
        if class_name != "EnergyDistribution":
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Specific Law Type", class_name, 
                width1=property_col_width, width2=value_col_width)
            
        # Add any specific fields based on the law type
        if class_name == "TabularEnergyDistribution":
            n_energies = getattr(self, 'n_incident_energies', 0)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Number of Incident Energies", n_energies, 
                width1=property_col_width, width2=value_col_width)
        
        elif class_name == "MaxwellFissionSpectrum":
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Restriction Energy (U)", getattr(self, 'restriction_energy', 'N/A'), 
                width1=property_col_width, width2=value_col_width)
            
        elif class_name == "EnergyDependentWattSpectrum":
            n_a_energies = getattr(self, 'n_a_energies', 0)
            n_b_energies = getattr(self, 'n_b_energies', 0)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Parameter a Energy Points", n_a_energies, 
                width1=property_col_width, width2=value_col_width)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Parameter b Energy Points", n_b_energies, 
                width1=property_col_width, width2=value_col_width)
                
        elif class_name == "NBodyPhaseSpaceDistribution":
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Number of Bodies (NPSX)", getattr(self, 'npsx', 'N/A'), 
                width1=property_col_width, width2=value_col_width)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Total Mass Ratio (AP)", getattr(self, 'ap', 'N/A'), 
                width1=property_col_width, width2=value_col_width)
            
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods - UPDATED to remove sampling methods
        methods = {
            ".get_applicability_probability(energy)": "Get probability this law applies at given incident energy"
        }
        
        # Add law-specific methods that aren't sampling-related
        if class_name == "LevelScattering":
            methods[".get_cm_energy(energy)"] = "Calculate center-of-mass energy for level scattering"
            methods[".get_lab_energy(energy, cosine)"] = "Calculate laboratory energy given CM cosine"
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        return header + description + info_table + methods_section