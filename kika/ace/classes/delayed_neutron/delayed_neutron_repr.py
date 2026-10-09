import logging
from kika._utils import create_repr_section

def precursor_repr(self) -> str:
    """Returns a formatted string representation of the DelayedNeutronPrecursor object."""
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Delayed Neutron Precursor Group Details':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of the precursor group
    description = "This object contains data for a delayed neutron precursor group.\n"
    description += "Delayed neutrons are emitted following beta decay of fission fragments.\n\n"
    
    # Create a summary table of data information
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    info_table = "Precursor Data Information:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # Decay constant
    decay_constant = self.decay_constant if self.decay_constant is not None else "Not specified"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Decay Constant", f"{decay_constant:.6g} 1/s" if isinstance(decay_constant, float) else decay_constant,
        width1=property_col_width, width2=value_col_width)
    
    # Number of energy points
    num_points = len(self.energies)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Energy Points", num_points,
        width1=property_col_width, width2=value_col_width)
    
    # Energy range if available
    if num_points > 0:
        energy_range = f"{self.energies[0]:.6g} - {self.energies[-1]:.6g} MeV"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Energy Range", energy_range,
            width1=property_col_width, width2=value_col_width)
    
    # Interpolation information
    num_regions = len(self.interpolation_regions)
    if num_regions > 0:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Interpolation Regions", num_regions,
            width1=property_col_width, width2=value_col_width)
        
        if num_regions <= 3:  # Show details if there aren't too many regions
            regions_str = ", ".join([f"({r[0]}, {r[1]})" for r in self.interpolation_regions])
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Interpolation Regions", regions_str,
                width1=property_col_width, width2=value_col_width)
    else:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Interpolation", "Linear-Linear (default)",
            width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for available methods
    methods = {
        ".evaluate(energy)": "Get the delayed neutron probability at a specific energy",
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + methods_section


def delayed_neutron_data_repr(self) -> str:
    """Returns a formatted string representation of the DelayedNeutronData object."""
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Delayed Neutron Data Information':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of delayed neutron data
    description = (
        "Delayed neutrons are emitted following the beta decay of certain fission fragments\n"
        "(called precursors). These neutrons can be categorized into groups based on the\n"
        "half-life of the precursor. Each group has a decay constant and energy-dependent\n"
        "emission probability.\n\n"
    )
    
    # Create a summary table of available data
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    info_table = "Available Delayed Neutron Data:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # BDD Block presence
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "BDD Block Present", "Yes" if self.has_delayed_neutron_data else "No",
        width1=property_col_width, width2=value_col_width)
    
    # Number of precursor groups
    num_groups = len(self.precursors)
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Precursor Groups", num_groups,
        width1=property_col_width, width2=value_col_width)
    
    # Decay constants if available
    if num_groups > 0:
        decay_constants = [
            precursor.decay_constant if precursor.decay_constant is not None else 0.0
            for precursor in self.precursors
        ]
        
        if num_groups <= 8:  # Show decay constants if not too many
            constants_str = ", ".join([f"{dc:.6g}" for dc in decay_constants])
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Decay Constants (1/s)", constants_str,
                width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for available methods
    methods = {
        ".get_precursor_probability(...)": "Get probability for specific group and energy",
        ".get_decay_constant(...)": "Get decay constant for specific precursor group"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + methods_section
