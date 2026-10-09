from kika._utils import create_repr_section

def particle_release_repr(self) -> str:
    """
    Returns a formatted string representation of the ParticleRelease object.
    
    This representation provides an overview of the particle release data,
    explaining the TY values and how to access the data.
    
    Returns
    -------
    str
        Formatted string representation
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Particle Release Data (TYR/TYRH Blocks)':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of the TY values - update to clarify where energy-dependent yields are stored
    description = (
        "The TYR and TYRH blocks contain TY values that indicate how many particles\n"
        "are released in nuclear reactions and in which reference frame:\n\n"
        "• TY = 0: Absorption (no particles released)\n"
        "• TY = ±1, ±2, ±3, ±4, ±5: Fixed number of particles released\n"
        "• TY = ±19: Fission reaction (see NU block for neutron yields)\n"
        "• |TY| > 100: Energy-dependent yield (available in DLW/DLWP blocks only)\n\n"
        "The sign indicates the reference frame: positive for laboratory frame,\n"
        "negative for center-of-mass frame.\n\n"
    )
    
    # Create a summary table of available data
    property_col_width = 40
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    info_table = "Available Particle Release Data:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Data Type", "Information", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # Neutron reaction data
    if self.has_neutron_data:
        num_reactions = len(self.incident_neutron)
        
        # Calculate distribution of TY values
        ty_counts = {}
        energy_dependent_count = 0
        
        for ty in self.incident_neutron:
            ty_value = int(ty)
            if abs(ty_value) > 100:
                energy_dependent_count += 1
            else:
                ty_counts[ty_value] = ty_counts.get(ty_value, 0) + 1
        
        ty_info = f"Available - {num_reactions} reactions"
        
        # Add details if there aren't too many different TY values
        if len(ty_counts) <= 6:
            ty_summary = []
            for ty_value, count in sorted(ty_counts.items()):
                ty_summary.append(f"{ty_value}:{count}")
            
            if energy_dependent_count > 0:
                ty_summary.append(f"energy-dependent:{energy_dependent_count}")
                
            ty_info += f" (TY distribution: {', '.join(ty_summary)})"
        else:
            if energy_dependent_count > 0:
                ty_info += f" ({energy_dependent_count} with energy-dependent yields)"
                
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Neutron Reactions (TYR)", ty_info, 
            width1=property_col_width, width2=value_col_width)
    else:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Neutron Reactions (TYR)", "Not available", 
            width1=property_col_width, width2=value_col_width)
    
    # Particle production data
    if self.has_particle_production_data:
        num_particle_types = len(self.particle_production)
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Particle Production (TYRH)", f"Available - {num_particle_types} particle types", 
            width1=property_col_width, width2=value_col_width)
        
        # Add details for each particle type if there aren't too many
        if num_particle_types <= 5:
            for i, ty_list in enumerate(self.particle_production):
                if len(ty_list) > 0:
                    # Count energy-dependent yields
                    energy_dep_count = sum(1 for ty in ty_list if abs(int(ty)) > 100)
                    
                    particle_info = f"{len(ty_list)} reactions"
                    if energy_dep_count > 0:
                        particle_info += f" ({energy_dep_count} with energy-dependent yields)"
                        
                    info_table += "{:<{width1}} {:<{width2}}\n".format(
                        f"  Particle Type {i+1}", particle_info, 
                        width1=property_col_width, width2=value_col_width)
    else:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Particle Production (TYRH)", "Not available", 
            width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create sections for methods
    methods = {
        ".get_reaction_frame(ty_entry)": "Get reference frame ('laboratory' or 'center-of-mass')",
        ".get_num_particles(ty_entry=None)": "Get number of particles released for a TY value",
        ".has_energy_dependent_yield(ty_entry)": "Check if TY value indicates energy-dependent yield",
        ".get_energy_yield_offset(ty_entry, jed)": "Calculate offset for energy-dependent yield data",
        ".get_summary()": "Get a summary of all particle release data"
    }
    
    methods_section = create_repr_section(
        "Available Methods:", 
        methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    # Combine all sections
    return header + description + info_table + methods_section
