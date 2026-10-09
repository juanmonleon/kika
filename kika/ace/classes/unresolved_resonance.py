from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from kika._utils import create_repr_section

@dataclass
class ProbabilityTable:
    """
    Class representing a single probability table at a given energy.
    """
    energy: float = 0.0
    # Views of xss_data for parsed tables; plain arrays for interpolated ones
    cumulative_probabilities: np.ndarray = field(default_factory=lambda: np.empty(0))
    total_xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    elastic_xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    fission_xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    capture_xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    heating_numbers: np.ndarray = field(default_factory=lambda: np.empty(0))
    
    @property
    def num_entries(self) -> int:
        """Return the number of entries in the probability table."""
        return len(self.cumulative_probabilities)
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the ProbabilityTable object."""
        header_width = 80
        header = "=" * header_width + "\n"
        header += f"{'Probability Table Details':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        output = header
        output += f"Probability table at energy: {self.energy:.6e} MeV\n"
        output += f"Number of entries: {self.num_entries}\n\n"
        
        # Create a summary of cross section data
        property_col_width = 15
        
        output += "Cross Section Summary:\n"
        output += "-" * header_width + "\n"
        output += f"{'Type':<{property_col_width}} {'Present':<10} {'Min Value':<20} {'Max Value':<20}\n"
        output += "-" * header_width + "\n"
        
        # Function to add cross section summary
        def add_xs_summary(name, xs_list):
            if len(xs_list) == 0:
                return f"{name:<{property_col_width}} {'No':<10} {'-':<20} {'-':<20}\n"
            
            values = [float(xs) for xs in xs_list]
            return f"{name:<{property_col_width}} {'Yes':<10} {min(values):<20.6e} {max(values):<20.6e}\n"
        
        # Add summaries for each cross section type
        output += add_xs_summary("Total", self.total_xs)
        output += add_xs_summary("Elastic", self.elastic_xs)
        output += add_xs_summary("Fission", self.fission_xs)
        output += add_xs_summary("Capture", self.capture_xs)
        output += add_xs_summary("Heating", self.heating_numbers)
        output += "-" * header_width + "\n"
        
        # Add probability distribution info
        if len(self.cumulative_probabilities) > 0:
            prob_values = [float(p) for p in self.cumulative_probabilities]
            output += "\nCumulative Probability Range: "
            output += f"{min(prob_values):.6f} to {max(prob_values):.6f}\n"
        
        return output

@dataclass
class UnresolvedResonanceTables:
    """
    Container for unresolved resonance probability tables (UNR block).
    
    These tables are used to sample cross sections in the unresolved resonance range.
    """
    has_data: bool = False
    num_energies: int = 0          # N - Number of incident energies where probability tables exist
    table_length: int = 0          # M - Length of each probability table
    interpolation: int = 0         # INT - Interpolation method between tables
    inelastic_flag: int = 0        # ILF - Inelastic competition flag
    other_absorption_flag: int = 0 # IOA - Other absorption flag
    factors_flag: int = 0          # IFF - Factors flag (0=cross sections, 1=factors)
    energies: np.ndarray = field(default_factory=lambda: np.empty(0))  # Incident energies (view of xss_data)
    tables: List[ProbabilityTable] = field(default_factory=list)  # Probability tables for each energy
    
    def __repr__(self) -> str:
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Unresolved Resonance Probability Tables':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        if not self.has_data:
            return header + "No unresolved resonance probability tables available"
        
        output = header
        
        # Description section
        description = (
            "Unresolved resonance probability tables contain cross section data for the unresolved\n"
            "resonance energy range. These tables allow for stochastic treatment of self-shielding\n"
            "effects in the unresolved resonance range during transport calculations.\n\n"
        )
        output += description
        
        # Basic information section
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        output += "Summary Information:\n"
        output += "-" * header_width + "\n"
        output += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        output += "-" * header_width + "\n"
        
        output += "{:<{width1}} {:<{width2}}\n".format(
            "Number of energy points", self.num_energies, 
            width1=property_col_width, width2=value_col_width)
        
        output += "{:<{width1}} {:<{width2}}\n".format(
            "Table length", self.table_length,
            width1=property_col_width, width2=value_col_width)
        
        # Interpolation method
        interp_methods = {2: "linear-linear", 5: "log-log"}
        interp_method = interp_methods.get(self.interpolation, f"unknown ({self.interpolation})")
        output += "{:<{width1}} {:<{width2}}\n".format(
            "Interpolation method", interp_method,
            width1=property_col_width, width2=value_col_width)
        
        # Inelastic competition flag
        inelastic_status = ""
        if self.inelastic_flag < 0:
            inelastic_status = "Zero in the unresolved range"
        elif self.inelastic_flag > 0:
            inelastic_status = f"Special MT={self.inelastic_flag}"
        else:
            inelastic_status = "Calculated from balance relationship"
        
        output += "{:<{width1}} {:<{width2}}\n".format(
            "Inelastic cross section", inelastic_status,
            width1=property_col_width, width2=value_col_width)
        
        # Other absorption flag
        other_abs_status = ""
        if self.other_absorption_flag < 0:
            other_abs_status = "Zero in the unresolved range"
        elif self.other_absorption_flag > 0:
            other_abs_status = f"Special MT={self.other_absorption_flag}"
        else:
            other_abs_status = "Calculated from balance relationship"
        
        output += "{:<{width1}} {:<{width2}}\n".format(
            "Other absorption cross section", other_abs_status,
            width1=property_col_width, width2=value_col_width)
        
        # Factors flag
        factors_status = "Cross sections" if self.factors_flag == 0 else "Factors to multiply smooth XS"
        output += "{:<{width1}} {:<{width2}}\n".format(
            "Table values represent", factors_status,
            width1=property_col_width, width2=value_col_width)
        
        # Energy range
        if len(self.energies) > 0:
            energy_values = [float(e) for e in self.energies]
            energy_range = f"{min(energy_values):.6e} to {max(energy_values):.6e} MeV"
            output += "{:<{width1}} {:<{width2}}\n".format(
                "Energy range", energy_range,
                width1=property_col_width, width2=value_col_width)
        
        output += "-" * header_width + "\n\n"
        
        # Cross section range summary
        if self.tables:
            output += "Cross Section Range Summary:\n"
            output += "-" * header_width + "\n"
            output += "{:<{width1}} {:<{width2}}\n".format(
                "Cross Section Type", "Range (min - max)",
                width1=property_col_width, width2=value_col_width)
            output += "-" * header_width + "\n"
            
            total_min, total_max = float('inf'), float('-inf')
            elastic_min, elastic_max = float('inf'), float('-inf')
            fission_min, fission_max = float('inf'), float('-inf')
            capture_min, capture_max = float('inf'), float('-inf')
            
            for table in self.tables:
                # Process total XS
                if len(table.total_xs) > 0:
                    table_values = [float(xs) for xs in table.total_xs]
                    table_min = min(table_values)
                    table_max = max(table_values)
                    total_min = min(total_min, table_min)
                    total_max = max(total_max, table_max)
                
                # Process elastic XS
                if len(table.elastic_xs) > 0:
                    table_values = [float(xs) for xs in table.elastic_xs]
                    table_min = min(table_values)
                    table_max = max(table_values)
                    elastic_min = min(elastic_min, table_min)
                    elastic_max = max(elastic_max, table_max)
                
                # Process fission XS
                if len(table.fission_xs) > 0:
                    table_values = [float(xs) for xs in table.fission_xs]
                    table_min = min(table_values)
                    table_max = max(table_values)
                    fission_min = min(fission_min, table_min)
                    fission_max = max(fission_max, table_max)
                
                # Process capture XS
                if len(table.capture_xs) > 0:
                    table_values = [float(xs) for xs in table.capture_xs]
                    table_min = min(table_values)
                    table_max = max(table_values)
                    capture_min = min(capture_min, table_min)
                    capture_max = max(capture_max, table_max)
            
            # Format for printing XS ranges with unit notation
            suffix = " (factors)" if self.factors_flag == 1 else ""
            
            if total_min != float('inf') and total_max != float('-inf'):
                range_str = f"{total_min:.6e} - {total_max:.6e}{suffix}"
                output += "{:<{width1}} {:<{width2}}\n".format(
                    "Total", range_str,
                    width1=property_col_width, width2=value_col_width)
                
            if elastic_min != float('inf') and elastic_max != float('-inf'):
                range_str = f"{elastic_min:.6e} - {elastic_max:.6e}{suffix}"
                output += "{:<{width1}} {:<{width2}}\n".format(
                    "Elastic", range_str,
                    width1=property_col_width, width2=value_col_width)
                
            if fission_min != float('inf') and fission_max != float('-inf'):
                range_str = f"{fission_min:.6e} - {fission_max:.6e}{suffix}"
                output += "{:<{width1}} {:<{width2}}\n".format(
                    "Fission", range_str,
                    width1=property_col_width, width2=value_col_width)
                
            if capture_min != float('inf') and capture_max != float('-inf'):
                range_str = f"{capture_min:.6e} - {capture_max:.6e}{suffix}"
                output += "{:<{width1}} {:<{width2}}\n".format(
                    "Capture", range_str,
                    width1=property_col_width, width2=value_col_width)
            
            output += "-" * header_width + "\n\n"
        
        # Add section for available methods using the utility function
        methods = {
            "get_probability_table(energy)": "Get probability table for specified energy, with interpolation if needed"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        output += methods_section
        
        return output
    
    def get_probability_table(self, energy: float) -> Optional[ProbabilityTable]:
        """
        Get the probability table for a specific energy, with interpolation if needed.
        
        Parameters
        ----------
        energy : float
            The incident energy
            
        Returns
        -------
        Optional[ProbabilityTable]
            The probability table for the specified energy, or None if out of range
        """
        if not self.has_data or len(self.energies) == 0 or not self.tables:
            return None
        
        # Extract energy values for comparison
        energy_values = [float(e) for e in self.energies]
        
        # Check if energy is outside the range
        if energy < energy_values[0] or energy > energy_values[-1]:
            return None
        
        # Find the nearest energy points for interpolation
        idx = np.searchsorted(energy_values, energy)
        if idx == 0:
            return self.tables[0]
        elif idx == len(energy_values):
            return self.tables[-1]
        
        # If the energy matches exactly, return that table
        if energy == energy_values[idx]:
            return self.tables[idx]
        
        # Perform interpolation between tables
        lower_idx = idx - 1
        upper_idx = idx
        
        lower_table = self.tables[lower_idx]
        upper_table = self.tables[upper_idx]
        
        lower_energy = energy_values[lower_idx]
        upper_energy = energy_values[upper_idx]
        
        # Create a new table for the interpolated result
        interp_table = ProbabilityTable(energy=energy)
        
        if self.interpolation == 2:  # Linear-linear
            factor = (energy - lower_energy) / (upper_energy - lower_energy)
            
            # Interpolate each array
            for attr in ['cumulative_probabilities', 'total_xs', 'elastic_xs', 
                         'fission_xs', 'capture_xs', 'heating_numbers']:
                lower_vals = getattr(lower_table, attr)
                upper_vals = getattr(upper_table, attr)
                
                if len(lower_vals) > 0 and len(upper_vals) > 0:
                    # Extract values for interpolation
                    lower_values = [float(entry) for entry in lower_vals]
                    upper_values = [float(entry) for entry in upper_vals]
                    
                    # Interpolated values are new data, not views of xss_data
                    interp_vals = np.array([lower_values[i] + factor * (upper_values[i] - lower_values[i]) 
                                  for i in range(len(lower_vals))], dtype=float)
                    setattr(interp_table, attr, interp_vals)
                
        elif self.interpolation == 5:  # Log-log
            log_factor = np.log(energy / lower_energy) / np.log(upper_energy / lower_energy)
            
            # Interpolate each array
            for attr in ['cumulative_probabilities', 'total_xs', 'elastic_xs', 
                         'fission_xs', 'capture_xs', 'heating_numbers']:
                lower_vals = getattr(lower_table, attr)
                upper_vals = getattr(upper_table, attr)
                
                if len(lower_vals) > 0 and len(upper_vals) > 0:
                    # Extract values for interpolation
                    lower_values = [float(entry) for entry in lower_vals]
                    upper_values = [float(entry) for entry in upper_vals]
                    
                    # Handle zero values in log-log interpolation
                    interp_vals = []
                    for i in range(len(lower_vals)):
                        if lower_values[i] <= 0 or upper_values[i] <= 0:
                            # Fall back to linear interpolation for zero/negative values
                            val = lower_values[i] + (energy - lower_energy) * (upper_values[i] - lower_values[i]) / (upper_energy - lower_energy)
                        else:
                            val = lower_values[i] * (upper_values[i] / lower_values[i]) ** log_factor
                        interp_vals.append(val)
                    
                    setattr(interp_table, attr, np.array(interp_vals, dtype=float))
        else:
            # Unknown interpolation method, default to nearest neighbor
            return self.tables[lower_idx] if (energy - lower_energy) < (upper_energy - energy) else self.tables[upper_idx]
        
        return interp_table
