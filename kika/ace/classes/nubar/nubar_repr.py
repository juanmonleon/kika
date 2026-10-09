from kika._utils import create_repr_section

def nudata_repr(self) -> str:
        """Returns a formatted string representation of the NuData object.
        
        This representation provides an overview of the nubar data format and content.
        
        :returns: Formatted string representation of the NuData
        :rtype: str
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Nubar Data Details':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of the nubar data
        description = "This object contains nubar data "
        if self.format == "polynomial":
            description += "in polynomial form.\n\n"
        elif self.format == "tabulated":
            description += "in tabulated form.\n\n"
        else:
            description += "but the format is not specified.\n\n"
        
        # Create a summary table of data information
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        info_table = "Data Information:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Format", self.format.capitalize() if self.format else "Not specified", 
            width1=property_col_width, width2=value_col_width)
        
        # Format-specific information
        if self.format == "polynomial" and self.polynomial is not None:
            num_coefficients = len(self.polynomial.coefficients)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Number of Coefficients", num_coefficients,
                width1=property_col_width, width2=value_col_width)
            
            # Show coefficients if there aren't too many
            if num_coefficients <= 5:
                coef_str = ", ".join(f"{c:.6g}" for c in self.polynomial.coefficients)
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Coefficients", coef_str,
                    width1=property_col_width, width2=value_col_width)
            
        elif self.format == "tabulated" and self.tabulated is not None:
            num_points = len(self.tabulated.energies)
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Number of Energy Points", num_points,
                width1=property_col_width, width2=value_col_width)
            
            if num_points > 0:
                energy_range = f"{self.tabulated.energies[0]:.6g} - {self.tabulated.energies[-1]:.6g} MeV"
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Energy Range", energy_range,
                    width1=property_col_width, width2=value_col_width)
            
            num_regions = len(self.tabulated.interpolation_regions)
            if num_regions > 0:
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Number of Interpolation Regions", num_regions,
                    width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".energies": "Get the energy grid (for tabulated data only)",
            ".to_dataframe()": "Convert nubar data to a pandas DataFrame",
            ".plot()": "Create a plot of the nubar data"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        return header + description + info_table + methods_section



def nucontainer_repr(self) -> str:
        """Returns a formatted string representation of the NuContainer object.
        
        This representation provides an overview of the available nubar data and how to access it.
        
        :returns: Formatted string representation of the NuContainer
        :rtype: str
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Nubar Data Information':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of nubar data
        description = (
            "Nubar data contains information about the number of neutrons released per fission.\n"
            "It can include prompt, total (prompt+delayed), and delayed neutron data.\n"
            "Data can be stored in polynomial or tabulated form.\n\n"
        )
        
        # Create a summary table of available data
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        info_table = "Available Nubar Data:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Data Type", "Status", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        # Add available nubar data types
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "NU Block Present", "Yes" if self.has_nubar else "No", 
            width1=property_col_width, width2=value_col_width)
        
        if self.has_nubar:
            if self.has_both_nu_types:
                prompt_format = self.prompt.format.capitalize() if self.prompt else "N/A"
                total_format = self.total.format.capitalize() if self.total else "N/A"
                
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Prompt Nubar", f"Available ({prompt_format})", 
                    width1=property_col_width, width2=value_col_width)
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    "Total Nubar", f"Available ({total_format})", 
                    width1=property_col_width, width2=value_col_width)
            else:
                # In this case, we typically have only total nubar
                if self.total:
                    total_format = self.total.format.capitalize()
                    info_table += "{:<{width1}} {:<{width2}}\n".format(
                        "Total Nubar", f"Available ({total_format})", 
                        width1=property_col_width, width2=value_col_width)
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "DNU Block Present", "Yes" if self.has_delayed else "No", 
            width1=property_col_width, width2=value_col_width)
        
        if self.has_delayed and self.delayed:
            delayed_format = self.delayed.format.capitalize()
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Delayed Nubar", f"Available ({delayed_format})", 
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"  # Add extra blank lines for readability
        
        # Create a section for data access using the utility function
        data_access = {
            ".prompt": "Access the prompt nubar data object",
            ".total": "Access the total nubar data object",
            ".delayed": "Access the delayed nubar data object"
        }
        
        data_access_section = create_repr_section(
            "Data Access Properties:", 
            data_access, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Add extra space for readability
        data_access_section += "\n"
        
        # Update methods dictionary to include all methods
        methods = {
            ".get_nubar(...)": "Get nubar value for specific energy and type",
            ".to_dataframe()": "Convert nubar data to DataFrame",
            ".plot()": "Create a plot of nubar data"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Add extra space after methods section
        methods_section += "\n"
        
        # Combine all sections
        return header + description + info_table + data_access_section + methods_section