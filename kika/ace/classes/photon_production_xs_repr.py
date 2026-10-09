from kika._utils import create_repr_section

def yield_based_cross_section_repr(self) -> str:
    """
    Returns a formatted string representation of the YieldBasedCrossSection object.
    
    This representation provides an overview of the yield-based cross section data
    for a specific reaction.
    
    :returns: Formatted string representation of the yield-based cross section
    :rtype: str
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Yield-Based Production Cross Section Details (MT={self.mt})':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of the cross section data
    description = (
        "This object contains yield-based production cross section data for a specific reaction.\n"
        f"The yield data is from {'photon production (MF=12)' if self.mftype == 12 else 'particle production (MF=16)'}\n"
        "and is used with the following formula to calculate the production cross section:\n\n"
        f"    σ_production(E) = Y(E) × σ_MT={self.mtmult}(E)\n\n"
        "where Y(E) is the yield function and σ_MTMULT(E) is the cross section for another reaction.\n\n"
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
        "MT Number", self.mt,
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "MFTYPE", f"{self.mftype} ({'MF=12 (photon production)' if self.mftype == 12 else 'MF=16 (particle production)'})",
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "MTMULT", f"{self.mtmult} (Multiplier MT number)",
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Energy Points", self.num_energies,
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Interpolation Regions", self.num_regions,
        width1=property_col_width, width2=value_col_width)
    
    # Add energy range if data is available
    if len(self.energies) > 0:
        energy_values = [float(entry) for entry in self.energies]
        min_e = min(energy_values)
        max_e = max(energy_values)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Energy Range", f"{min_e:.6e} to {max_e:.6e} MeV",
            width1=property_col_width, width2=value_col_width)
    
    # Add yield range if data is available
    if len(self.yields) > 0:
        yield_values = [float(entry) for entry in self.yields]
        min_y = min(yield_values)
        max_y = max(yield_values)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Yield Range", f"{min_y:.6e} to {max_y:.6e}",
            width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for available methods
    methods = {
        ".get_energy_values()": "Get energy grid as a list of floats",
        ".get_yield_values()": "Get yield values as a list of floats",
        ".get_interpolated_yield(...)": "Get interpolated yield at a specific energy",
        ".reconstruct_xs(...)": "Calculate production cross section at a specific energy"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + methods_section


def direct_cross_section_repr(self) -> str:
    """
    Returns a formatted string representation of the DirectCrossSection object.
    
    This representation provides an overview of the direct cross section data
    for a specific reaction.
    
    :returns: Formatted string representation of the direct cross section
    :rtype: str
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Direct Production Cross Section Details (MT={self.mt})':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of the cross section data
    description = (
        "This object contains direct production cross section data for a specific reaction.\n"
        "Unlike yield-based cross sections, direct cross sections (MFTYPE=13) provide the\n"
        "production cross section values explicitly without requiring additional calculations.\n"
        "This format is only used for photon production cross sections.\n\n"
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
        "MT Number", self.mt,
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "MFTYPE", "13 (direct cross section)",
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Energy Grid Index (IE)", self.energy_grid_index,
        width1=property_col_width, width2=value_col_width)
    
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Number of Entries (NE)", self.num_entries,
        width1=property_col_width, width2=value_col_width)
    
    # Add cross section range if data is available
    if len(self.cross_sections) > 0:
        xs_values = [float(entry) for entry in self.cross_sections]
        min_xs = min(xs_values)
        max_xs = max(xs_values)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Cross Section Range", f"{min_xs:.6e} to {max_xs:.6e} barns",
            width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for available methods
    methods = {
        ".get_xs_values()": "Get cross section values as a list of floats",
        ".get_value(...)": "Get interpolated cross section at a specific energy"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + methods_section


def photon_production_cross_sections_repr(self) -> str:
    """
    Returns a formatted string representation of the PhotonProductionCrossSections object.
    
    This representation provides an overview of all photon production cross sections.
    
    :returns: Formatted string representation of the photon production cross sections
    :rtype: str
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Photon Production Cross Sections (SIGP Block)':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of the cross section data
    description = (
        "This container holds cross section data for photon-producing reactions.\n"
        "There are two types of cross section data:\n"
        "1. Yield-based (MFTYPE = 12 or 16): Cross section is calculated as Y(E) * σ_MT(E)\n"
        "2. Direct (MFTYPE = 13): Cross section is provided directly\n\n"
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
            "Number of Reactions", len(self.cross_sections),
            width1=property_col_width, width2=value_col_width)
        
        # Count yield-based and direct cross sections
        yield_based = sum(1 for xs in self.cross_sections.values() if hasattr(xs, 'mtmult'))
        direct = sum(1 for xs in self.cross_sections.values() if not hasattr(xs, 'mtmult'))
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Yield-based Cross Sections", yield_based,
            width1=property_col_width, width2=value_col_width)
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Direct Cross Sections", direct,
            width1=property_col_width, width2=value_col_width)
        
        # List MT numbers if there aren't too many
        if len(self.cross_sections) <= 10:
            mt_numbers = ", ".join(str(mt) for mt in sorted(self.cross_sections.keys()))
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "MT Numbers", mt_numbers,
                width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for available methods
    methods = {
        ".get_reaction_xs(...)": "Get cross section data for a specific MT",
        ".get_available_mts()": "Get a list of all available MT numbers",
        ".get_xs_descriptions()": "Get descriptions of all available cross sections",
        ".get_photon_production_xs(...)": "Get production cross section at a specific energy"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + methods_section


def particle_production_cross_sections_repr(self) -> str:
    """
    Returns a formatted string representation of the ParticleProductionCrossSections object.
    
    This representation provides an overview of all particle production cross sections.
    
    :returns: Formatted string representation of the particle production cross sections
    :rtype: str
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Secondary Particle Yield-Based Cross Sections (SIGH Block)':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of the cross section data
    description = (
        "This container holds yield-based cross section data for reactions that produce\n"
        "secondary particles (neutrons, protons, alphas, etc.).\n\n"
        "IMPORTANT DISTINCTION:\n"
        "- This data (from the SIGH block) contains yield-based cross sections for specific reactions\n"
        "  calculated using the formula: σ_prod(E) = Y(E) * σ_MTMULT(E)\n"
        "- This is different from the HPD block data, which contains the total production\n"
        "  cross section for each particle type.\n\n"
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
        "Data Present", "Yes" if self.has_data and self.particle_cross_sections else "No",
        width1=property_col_width, width2=value_col_width)
    
    if self.has_data and self.particle_cross_sections:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Reactions", sum(len(v) for v in self.particle_cross_sections.values()),
            width1=property_col_width, width2=value_col_width)
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Particle Types", len(self.particle_types),
            width1=property_col_width, width2=value_col_width)
        
        # List particle types if there aren't too many
        if self.particle_types and len(self.particle_types) <= 8:
            particle_types = ", ".join(str(p) for p in sorted(self.particle_types.keys()))
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Particle Type Indices", particle_types,
                width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for particle-specific information
    if self.has_data and self.particle_types:
        particle_info = "Particle Type Details:\n"
        particle_info += "-" * header_width + "\n"
        
        for i, mts in sorted(self.particle_types.items()):
            particle_info += f"Particle Type {i}:\n"
            particle_info += f"  Number of Reactions: {len(mts)}\n"
            
            # List MT numbers if there aren't too many
            if len(mts) <= 8:
                mt_list = ", ".join(str(mt) for mt in sorted(mts))
                particle_info += f"  MT Numbers: {mt_list}\n"
            
            particle_info += "\n"
        
        particle_info += "-" * header_width + "\n\n"
    else:
        particle_info = ""
    
    # Create a section for available methods
    methods = {
        ".get_reaction_xs(...)": "Get cross section data for a specific MT",
        ".get_available_mts()": "Get a list of all available MT numbers",
        ".get_particle_mts(...)": "Get MT numbers for a specific particle type",
        ".get_particle_production_xs(...)": "Get production cross section at a specific energy"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + particle_info + methods_section
