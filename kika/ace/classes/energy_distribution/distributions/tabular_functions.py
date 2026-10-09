# Law 22, 24: Tabular functions

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika._utils import create_repr_section


@dataclass
class TabularLinearFunctions(EnergyDistribution):
    """
    Law 22: Tabular Linear Functions of Incident Energy Out.
    
    From UK Law 2, this represents a tabular function form where the outgoing 
    energy is a linear function of the incident energy: E_out = C_ik * (E − T_ik).
    
    Data format (Table 41):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E_in(l): Tabulated incident energies
    - LOCE(l): Locators of E_out tables
    
    For each incident energy E_in(i):
    - NF_i: Number of functions for this energy
    - P_ik: Probability for each function
    - T_ik: Origin parameter for each function
    - C_ik: Slope parameter for each function
    """
    law: int = 22
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    table_locators: List[int] = field(default_factory=list)  # Locators of E_out tables
    
    # Store the function data for each incident energy
    # Each entry is a dictionary with 'nf', 'p', 't', 'c' keys
    function_data: List[Dict] = field(default_factory=list)
    
    def get_function_data(self, energy_idx: int) -> Dict:
        """
        Get the function data for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the function data with keys:
            - 'nf': Number of functions
            - 'p': Probability for each function
            - 't': Origin parameter for each function
            - 'c': Slope parameter for each function
            or None if index is invalid
        """
        if 0 <= energy_idx < len(self.function_data):
            return self.function_data[energy_idx]
        return None
    
    def get_interpolated_function_data(self, incident_energy: float) -> Dict:
        """
        Get interpolated function data for a specific incident energy.
        
        For Law 22, we don't interpolate between function data sets.
        Instead, we find the bracket incident energies and use the lower one's function data.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the function data or None if not available
        """
        # Find the bracketing incident energies
        if len(self.incident_energies) == 0 or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first function data
            return self.get_function_data(0) if self.function_data else None
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last function data
            return self.get_function_data(len(self.incident_energies) - 1) if self.function_data else None
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Return the function data for the lower incident energy
        return self.get_function_data(idx)
    
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the TabularLinearFunctions distribution.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Tabular Linear Functions (Law 22)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = (
            "This distribution represents outgoing energy as a linear function of the incident\n"
            "energy, using the formula: E_out = C_ik * (E - T_ik)\n\n"
            "For each incident energy, there are multiple functions with associated probabilities.\n"
            "This law is also known as UK Law 2 in some nuclear data formats.\n\n"
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
            "Number of Incident Energies", self.n_energies, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Interpolation Regions", self.n_interp_regions, 
            width1=property_col_width, width2=value_col_width)
        
        # If we have incident energies, show the range
        if len(self.incident_energies) > 0:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Incident Energy Range", 
                f"{min(self.incident_energies):.6g} - {max(self.incident_energies):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        # Count function data sets
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Function Data Sets", len(self.function_data),
            width1=property_col_width, width2=value_col_width)
        
        # Add information about the first energy point if available
        if self.function_data:
            first_data = self.function_data[0]
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Functions at First Energy Point", first_data.get('nf', 0),
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_function_data(energy_idx)": "Get function data for a specific energy index",
            ".get_interpolated_function_data(incident_energy)": "Get function data for any incident energy"
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
            "# Get the function data for a specific incident energy\n"
            "function_data = distribution.get_interpolated_function_data(incident_energy=2.0)\n\n"
            "# Extract parameters from the function data\n"
            "if function_data:\n"
            "    num_functions = function_data['nf']\n"
            "    probabilities = function_data['p']\n"
            "    origin_params = function_data['t']\n"
            "    slope_params = function_data['c']\n"
        )
        
        return header + description + info_table + methods_section + "\n" + example


@dataclass
class TabularEnergyMultipliers(EnergyDistribution):
    """
    Law 24: Tabular Energy Multipliers.
    
    From UK Law 6, this represents a tabular function where the outgoing 
    energy is a multiplier of the incident energy: E_out = T_k(l) * E.
    
    Data format (Table 42):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E_in(l): Tabulated incident energies
    - NET: Number of outgoing values in each table
    - T_i(l): Tables of energy multipliers for each incident energy
    """
    law: int = 24
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    n_mult_values: int = 0  # Number of multiplier values in each table (NET)
    
    # Store the multiplier tables for each incident energy
    # Each row corresponds to one incident energy
    multiplier_tables: List[List[float]] = field(default_factory=list)
    
    def get_multiplier_table(self, energy_idx: int) -> List[float]:
        """
        Get the multiplier table for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        List[float]
            List of multiplier values or None if index is invalid
        """
        if 0 <= energy_idx < len(self.multiplier_tables):
            return self.multiplier_tables[energy_idx]
        return None
    
    def get_interpolated_multiplier_table(self, incident_energy: float) -> List[float]:
        """
        Get interpolated multiplier table for a specific incident energy.
        
        For tabular energy multipliers, we interpolate between the
        multiplier tables of adjacent incident energies.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        List[float]
            Interpolated multiplier table or None if not available
        """
        # Find the bracketing incident energies
        if len(self.incident_energies) == 0 or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first table
            return self.get_multiplier_table(0)
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last table
            return self.get_multiplier_table(len(self.incident_energies) - 1)
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Get the multiplier tables for the bracketing energies
        table_low = self.get_multiplier_table(idx)
        table_high = self.get_multiplier_table(idx + 1)
        
        if (table_low is None or table_high is None or len(table_low) == 0
                or len(table_high) == 0 or len(table_low) != len(table_high)):
            return table_low if table_low is not None and len(table_low) > 0 else table_high
            
        # Calculate interpolation factor
        energy_low = self.incident_energies[idx]
        energy_high = self.incident_energies[idx + 1]
        factor = (incident_energy - energy_low) / (energy_high - energy_low)
        
        # Linearly interpolate between tables
        interp_table = [table_low[i] + factor * (table_high[i] - table_low[i]) for i in range(len(table_low))]
        
        return interp_table
    
    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the TabularEnergyMultipliers distribution.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Tabular Energy Multipliers (Law 24)':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the energy distribution
        description = (
            "This distribution represents outgoing energy as a multiplier of the incident energy,\n"
            "using the formula: E_out = T * E, where T is a multiplier sampled from a table.\n\n"
            "For each incident energy, there is a table of energy multipliers. This law is also\n"
            "known as UK Law 6 in some nuclear data formats.\n\n"
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
            "Number of Incident Energies", self.n_energies, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Interpolation Regions", self.n_interp_regions, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Multiplier Points per Table", self.n_mult_values, 
            width1=property_col_width, width2=value_col_width)
        
        # If we have incident energies, show the range
        if len(self.incident_energies) > 0:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Incident Energy Range", 
                f"{min(self.incident_energies):.6g} - {max(self.incident_energies):.6g} MeV", 
                width1=property_col_width, width2=value_col_width)
        
        # Information about multiplier tables
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Multiplier Tables", len(self.multiplier_tables),
            width1=property_col_width, width2=value_col_width)
        
        # Add information about the first table if available
        if self.multiplier_tables and len(self.multiplier_tables[0]) > 0:
            first_table = self.multiplier_tables[0]
            mult_range = f"{min(first_table):.6g} - {max(first_table):.6g}"
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "First Table Multiplier Range", mult_range,
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_multiplier_table(energy_idx)": "Get multiplier table for a specific energy index",
            ".get_interpolated_multiplier_table(...)": "Get interpolated table for any incident energy"
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
            "# Get the multiplier table for a specific incident energy\n"
            "multipliers = distribution.get_interpolated_multiplier_table(incident_energy=2.0)\n\n"
            "# Analyze the multiplier distribution\n"
            "if multipliers:\n"
            "    min_mult = min(multipliers)\n"
            "    max_mult = max(multipliers)\n"
            "    avg_mult = sum(multipliers) / len(multipliers)\n"
            "    print(f\"Multiplier range: {min_mult:.6g} - {max_mult:.6g}, Average: {avg_mult:.6g}\")\n"
        )
        
        return header + description + info_table + methods_section + "\n" + example
