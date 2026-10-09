import logging
from kika._utils import create_repr_section


def mtr_repr(self) -> str:
    """Returns a formatted string representation of the ReactionMTData object."""
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'MT Reaction Number Data':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of MT numbers
    description = (
        "MT numbers are reaction identifiers in ENDF format that describe the type of\n"
        "nuclear reaction. This object contains MT numbers for different reaction types:\n"
        "- Incident neutron reactions (MTR block)\n"
        "- Photon production reactions (MTRP block)\n"
        "- Particle production reactions (MTRH block)\n"
        "- Secondary neutron production reactions\n\n"
    )
    
    # Create a summary table of available data
    property_col_width = 40
    value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
    
    info_table = "Available MT Reaction Data:\n"
    info_table += "-" * header_width + "\n"
    info_table += "{:<{width1}} {:<{width2}}\n".format(
        "Data Type", "Information", width1=property_col_width, width2=value_col_width)
    info_table += "-" * header_width + "\n"
    
    # Neutron MT numbers
    neutron_mt_count = len(self.incident_neutron)
    if neutron_mt_count > 0:
        mt_str = f"Available - {neutron_mt_count} reactions"
        # Show a sample of MT numbers if there aren't too many
        if neutron_mt_count <= 10:
            mt_values = [str(int(mt)) for mt in self.incident_neutron]
            mt_str += f" (MT: {', '.join(mt_values)})"
        elif neutron_mt_count > 10:
            # Show first 5 and last 5 with ellipsis in between
            first_mt = [str(int(mt)) for mt in self.incident_neutron[:5]]
            last_mt = [str(int(mt)) for mt in self.incident_neutron[-5:]]
            mt_str += f" (MT: {', '.join(first_mt)}, ..., {', '.join(last_mt)})"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Neutron Reactions (MTR)", mt_str, width1=property_col_width, width2=value_col_width)
    else:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Neutron Reactions (MTR)", "Not available", width1=property_col_width, width2=value_col_width)
    
    # Photon production MT numbers
    photon_mt_count = len(self.photon_production)
    if photon_mt_count > 0:
        mt_str = f"Available - {photon_mt_count} reactions"
        # Show a sample of MT numbers if there aren't too many
        if photon_mt_count <= 10:
            mt_values = [str(int(mt)) for mt in self.photon_production]
            mt_str += f" (MT: {', '.join(mt_values)})"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Photon Production (MTRP)", mt_str, width1=property_col_width, width2=value_col_width)
    else:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Photon Production (MTRP)", "Not available", width1=property_col_width, width2=value_col_width)
    
    # Secondary neutron MT numbers
    sec_neutron_count = len(self.secondary_neutron_mt)
    if sec_neutron_count > 0:
        mt_str = f"Available - {sec_neutron_count} reactions"
        # Show a sample of MT numbers if there aren't too many
        if sec_neutron_count <= 10:
            mt_values = [str(int(mt)) for mt in self.secondary_neutron_mt]
            mt_str += f" (MT: {', '.join(mt_values)})"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Secondary Neutron Reactions", mt_str, width1=property_col_width, width2=value_col_width)
    else:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Secondary Neutron Reactions", "Not available", width1=property_col_width, width2=value_col_width)
    
    # Particle production MT numbers
    num_particle_types = len(self.particle_production)
    if num_particle_types > 0:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Particle Production (MTRH)", f"Available - {num_particle_types} particle types", 
            width1=property_col_width, width2=value_col_width)
        
        # Add details for each particle type if there aren't too many
        if num_particle_types <= 5:
            for i, mt_list in enumerate(self.particle_production):
                mt_count = len(mt_list)
                if mt_count > 0:
                    mt_str = f"{mt_count} reactions"
                    if mt_count <= 8:
                        mt_values = [str(int(mt)) for mt in mt_list]
                        mt_str += f" (MT: {', '.join(mt_values)})"
                    info_table += "{:<{width1}} {:<{width2}}\n".format(
                        f"  Particle Type {i+1}", mt_str, width1=property_col_width, width2=value_col_width)
    else:
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Particle Production (MTRH)", "Not available", width1=property_col_width, width2=value_col_width)
    
    info_table += "-" * header_width + "\n\n"
    
    # Create a section for how to access data
    access_methods = {
        ".get_mt_values('neutron')": "Get list of MT numbers for incident neutron reactions",
        ".get_mt_values('photon')": "Get list of MT numbers for photon production reactions",
        ".get_mt_values('secondary')": "Get list of MT numbers for secondary neutron reactions",
        ".get_particle_production_mt_dict()": "Get dictionary of MT numbers for all particle types",
        ".get_mt_values('particle', idx=0)": "Get list of MT numbers for specific particle type (0-indexed)"
    }
    
    methods_section = create_repr_section(
        "How to Access MT Numbers:", 
        access_methods, 
        total_width=header_width, 
        method_col_width=property_col_width
    )
    
    return header + description + info_table + methods_section
