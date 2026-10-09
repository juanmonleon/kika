from kika._utils import create_repr_section

def particle_production_cross_section_repr(self) -> str:
    """
    Returns a formatted string representation of the ParticleProductionCrossSection object.
    
    This representation provides an overview of the production cross section data for a single
    secondary particle type.
    
    :returns: Formatted string representation of the production cross section
    :rtype: str
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Secondary Particle Production Cross Section Details':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of the cross section data
    description = (
        "This object contains the cross section for producing a specific secondary particle type\n"
        "during nuclear reactions. It includes both the production cross section values and\n"
        "the corresponding heating numbers (energy deposition values).\n\n"
    )
    
    # Create a summary table of data information
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    info_table = "Cross Section Data Information:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Energy Grid Index", self.energy_grid_index,
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Energy Points", self.num_energies,
        width1=property_col_width, width2=value_col_width)
    
    # Add cross section range if data is available
    if len(self.xs_values) > 0:
        xs_float_values = [float(entry) for entry in self.xs_values]
        min_xs = min(xs_float_values)
        max_xs = max(xs_float_values)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Cross Section Range", f"{min_xs:.6e} to {max_xs:.6e} barns",
            width1=property_col_width, width2=value_col_width)
    
    # Add heating number range if data is available
    if len(self.heating_numbers) > 0:
        heating_float_values = [float(entry) for entry in self.heating_numbers]
        min_heat = min(heating_float_values)
        max_heat = max(heating_float_values)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Heating Number Range", f"{min_heat:.6e} to {max_heat:.6e}",
            width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for available methods
    methods = {
        ".get_xs_values()": "Get cross section values as a list of floats",
        ".get_heating_values()": "Get heating numbers as a list of floats",
        ".get_xs_at_energy(...)": "Get interpolated cross section at a specific energy",
        ".get_heating_at_energy(...)": "Get interpolated heating number at a specific energy",
        ".get_energy_xs_pairs(...)": "Get energy-cross section pairs for plotting",
        ".get_energy_heating_pairs(...)": "Get energy-heating pairs for plotting"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + methods_section


def secondary_particle_cross_sections_repr(self) -> str:
    """
    Returns a formatted string representation of the SecondaryParticleCrossSections object.
    
    This representation provides an overview of the production cross sections for all
    secondary particle types.
    
    :returns: Formatted string representation of the secondary particle cross sections
    :rtype: str
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Secondary Particle Total Cross Sections (HPD Block)':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of the cross section data
    description = (
        "This container holds the total production cross sections for each type of secondary particle\n"
        "that can be produced in nuclear reactions (neutrons, protons, alphas, etc.).\n\n"
        "IMPORTANT DISTINCTION:\n"
        "- This data (from the HPD block) contains the total production cross section for each particle type.\n"
        "- This is different from the SIGH block data, which contains yield-based cross sections\n"
        "  for specific reactions calculated as: σ_prod(E) = Y(E) * σ_MTMULT(E).\n\n"
    )
    
    # Create a summary table of available data
    property_col_width = 35
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    info_table = "Available Cross Section Data:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Property", "Value", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Data Present", "Yes" if self.has_data else "No",
        width1=property_col_width, width2=value_col_width)
    
    if self.has_data:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Particle Types", len(self.particle_data),
            width1=property_col_width, width2=value_col_width)
        
        # List particle types if there aren't too many
        if len(self.particle_data) <= 8:
            particle_types = ", ".join(str(p) for p in sorted(self.particle_data.keys()))
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Particle Type Indices", particle_types,
                width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for particle-specific information
    if self.has_data and self.particle_data:
        particle_info = "Particle Type Details:\n"
        particle_info += "-" * header_width + "\n"
        
        for idx, data in sorted(self.particle_data.items()):
            particle_info += f"Particle Type {idx}:\n"
            
            # Get energy range if possible
            if data.num_energies > 0 and len(data.xs_values) > 0:
                xs_float_values = [float(entry) for entry in data.xs_values]
                min_xs = min(xs_float_values)
                max_xs = max(xs_float_values)
                particle_info += f"  Cross Section Range: {min_xs:.6e} to {max_xs:.6e} barns\n"
                particle_info += f"  Number of Energy Points: {data.num_energies}\n"
            
            particle_info += "\n"
        
        particle_info += "-" * header_width + "\n\n"
    else:
        particle_info = ""
    
    # Create a section for available methods
    methods = {
        ".get_particle_types()": "Get a list of all available particle types",
        ".get_particle_cross_section(...)": "Get cross section data for a specific particle",
        ".get_xs_at_energy(...)": "Get production cross section at a specific energy",
        ".get_heating_at_energy(...)": "Get heating number at a specific energy",
        ".get_energy_xs_pairs(...)": "Get energy-cross section pairs for plotting"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + particle_info + methods_section
