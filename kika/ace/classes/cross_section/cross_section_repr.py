from kika._utils import create_repr_section

def reaction_xs_repr(self):
    """
    Create a string representation of a ReactionCrossSection object.
    
    Parameters
    ----------
    self : ReactionCrossSection
        The reaction cross section object
        
    Returns
    -------
    str
        String representation of the object
    """
    # Create a centered header
    header_text = "REACTION CROSS SECTION DATA"
    header_width = 60
    header = "\n" + "=" * header_width + "\n"
    header += " " * ((header_width - len(header_text)) // 2) + header_text + "\n"
    header += "=" * header_width + "\n\n"
    
    # Description of the reaction cross section
    mt_value = self.mt if self.mt else 'Unknown'
    
    description = f"This object contains cross section data for reaction MT={mt_value}.\n\n"
    
    # Create a summary table of data information
    property_col_width = 35
    value_col_width = 25
    divider = "-" * (property_col_width + value_col_width) + "\n"
    
    summary = divider
    summary += f"{'Property':<{property_col_width}}{'Value':<{value_col_width}}\n"
    summary += divider
    
    summary += f"{'MT number':<{property_col_width}}{mt_value}\n"
    summary += f"{'Energy grid start index':<{property_col_width}}{self.energy_idx}\n"
    summary += f"{'Number of energy points':<{property_col_width}}{self.num_energies}\n"
    
    # Get number of XS values
    num_xs_values = len(self._xs_entries)
    summary += f"{'Number of XS values':<{property_col_width}}{num_xs_values}\n"
    
    if num_xs_values > 0:
        xs_values = self.xs_values
        summary += f"{'XS value range':<{property_col_width}}{min(xs_values):.6e} to {max(xs_values):.6e}\n"
    
    summary += divider
    
    # Add methods section
    methods_header = "\nMethods to access data:\n" + divider
    methods_content = f"{'Property':<{property_col_width}}{'Description':<{value_col_width}}\n" + divider
    
    methods = [
        (".xs_values", "Get cross section values as a list"),
        (".energies", "Get energy values as a list"),
        (".plot()", "Plot cross section"),
        (".to_dataframe()", "Get cross section data as DataFrame")
    ]
    
    for method, desc in methods:
        methods_content += f"{method:<{property_col_width}}{desc:<{value_col_width}}\n"
    
    methods_content += divider
    
    # Combine all parts
    return header + description + summary + methods_header + methods_content


def xs_data_repr(self) -> str:
    """
    Returns a formatted string representation of the CrossSectionData object.
    
    This representation provides an overview of all reaction cross sections and how to access them.
    
    Returns
    -------
    str
        Formatted string representation of the CrossSectionData
    """
    header_width = 85
    header = "=" * header_width + "\n"
    header += f"{'Cross Section Data Information':^{header_width}}\n"
    header += "=" * header_width + "\n\n"
    
    # Description of cross section data
    description = (
        "This object contains cross section data for multiple reactions.\n"
        "Each reaction is identified by its MT number according to ENDF/B standards.\n\n"
    )
    
    # Simple count of reactions
    num_reactions = len(self.reaction)
    basic_info = f"Number of incident neutron reactions: {num_reactions}\n\n"
    
    # Create a section for data access
    access_table = "How to Access Cross Section Data:\n"
    access_table += "-" * header_width + "\n"
    access_table += "{:<{width1}} {:<{width2}}\n".format(
        "Code", "Description", width1=35, width2=header_width-35-3)
    access_table += "-" * header_width + "\n"
    
    access_examples = [
        (".mt_numbers", "Get list of available MT numbers"),
        (".reaction[mt]", "Access a specific cross section by MT number"),
    ]
    
    for example, desc in access_examples:
        access_table += "{:<{width1}} {:<{width2}}\n".format(
            example, desc, width1=35, width2=header_width-35-3)
    
    access_table += "-" * header_width + "\n\n"
    
    # Methods section
    methods_table = "Available Methods:\n"
    methods_table += "-" * header_width + "\n"
    methods_table += "{:<{width1}} {:<{width2}}\n".format(
        "Method", "Description", width1=35, width2=header_width-35-3)
    methods_table += "-" * header_width + "\n"
    
    methods = [
        (".plot(mt)", "Plot cross section for specified MT (single or list)"),
        (".to_dataframe(mt_list)", "Get cross sections as DataFrame (single or list)"),
    ]
    
    for method, desc in methods:
        methods_table += "{:<{width1}} {:<{width2}}\n".format(
            method, desc, width1=35, width2=header_width-35-3)
    
    methods_table += "-" * header_width + "\n"
    
    return header + description + basic_info + access_table + methods_table
