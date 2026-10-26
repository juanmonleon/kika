from dataclasses import dataclass
from kika.materials import MaterialCollection
from .perturbations import Perturbation
from kika.energy_grids.utils import _identify_energy_grid


@dataclass
class Input:
    """Main class for storing MCNP input data.

    :ivar perturbation: Container for all perturbation cards in the input
    :type perturbation: Perturbation
    :ivar materials: Container for all material cards in the input
    :type materials: MaterialCollection
    """
    perturbation: 'Perturbation' = None
    materials: 'MaterialCollection' = None

    def __repr__(self):
        """Returns a formatted string representation of the MCNP input data.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the input data
        :rtype: str
        """
        header_width = 60
        header = "=" * header_width + "\n"
        header += f"{'MCNP Input Data':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        label_width = 25  # Width for labels
        sections = []
        
        # ====================== MATERIALS SECTION ======================
        if self.materials is not None and hasattr(self.materials, 'by_id'):
            mat_section = "-" * header_width + "\n"
            mat_section += f"{'MATERIALS':^{header_width}}\n"
            mat_section += "-" * header_width + "\n"
            
            mat_count = len(self.materials.by_id)
            mat_section += f"{'Number of materials:':{label_width}} {mat_count}\n"
            
            if mat_count > 0:
                # Get material IDs
                mat_ids = sorted(self.materials.by_id.keys())
                
                # Format material IDs in rows of 10
                if mat_count <= 10:
                    # Simple listing if few materials
                    mat_list = ", ".join(map(str, mat_ids))
                    mat_section += f"{'Material IDs:':{label_width}} {mat_list}\n"
                else:
                    # Multi-row listing for many materials
                    mat_section += f"{'Material IDs:':{label_width}}\n"
                    
                    # Format in rows with proper indentation
                    items_per_row = 10
                    for i in range(0, len(mat_ids), items_per_row):
                        row_ids = mat_ids[i:i+items_per_row]
                        row_str = ", ".join(map(str, row_ids))
                        if i == 0:
                            # First row has the label
                            mat_section += f"{'':{label_width}} {row_str}\n"
                        else:
                            # Subsequent rows have proper indentation
                            mat_section += f"{'':{label_width}} {row_str}\n"
                
                # Check if using weight or atomic fractions
                weight_count = 0
                atomic_count = 0
                for mat in self.materials.by_id.values():
                    if getattr(mat, "is_weight", False):
                        weight_count += 1
                    else:
                        atomic_count += 1
                
                if weight_count > 0:
                    mat_section += f"{'Weight fraction mats:':{label_width}} {weight_count}\n"
                if atomic_count > 0:
                    mat_section += f"{'Atomic fraction mats:':{label_width}} {atomic_count}\n"
            else:
                mat_section += f"{'Number of materials:':{label_width}} 0 (empty collection)\n"
            
            # Add access hint at the end of the section
            mat_section += "\nUse .materials to access material data.\n"
            sections.append(mat_section)
        
        # ====================== PERTURBATIONS SECTION ======================
        if self.perturbation is not None:
            pert_section = "-" * header_width + "\n"
            pert_section += f"{'PERTURBATIONS':^{header_width}}\n"
            pert_section += "-" * header_width + "\n"
            
            pert_count = len(self.perturbation.pert)
            pert_section += f"{'Number of perturbations:':{label_width}} {pert_count}\n"
            
            if pert_count > 0:
                # Get sorted perturbation numbers and display them in ranges
                pert_nums = sorted(self.perturbation.pert.keys())
                
                # Find ranges of consecutive perturbation numbers
                ranges = []
                start = pert_nums[0]
                prev = pert_nums[0]
                
                for i in range(1, len(pert_nums)):
                    # If there's a gap in sequence
                    if pert_nums[i] > prev + 1:
                        # Add the completed range
                        if start == prev:
                            ranges.append(str(start))
                        else:
                            ranges.append(f"{start}-{prev}")
                        # Start a new range
                        start = pert_nums[i]
                    prev = pert_nums[i]
                
                # Add the last range
                if start == prev:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{prev}")
                
                pert_section += f"{'Perturbation IDs:':{label_width}} {', '.join(ranges)}\n"
                
                # Add details about reactions, methods, and energy ranges
                reactions = self.perturbation.reactions
                if reactions:
                    pert_section += f"{'Reactions available:':{label_width}} {', '.join(map(str, reactions))}\n"
                
                methods = set(pert.method for pert in self.perturbation.pert.values() if pert.method is not None)
                if methods:
                    pert_section += f"{'Methods available:':{label_width}} {', '.join(map(str, sorted(methods)))}\n"
                
                # Get perturbed material IDs
                mat_ids = set(pert.material for pert in self.perturbation.pert.values() if pert.material is not None)
                if mat_ids:
                    pert_section += f"{'Perturbed materials:':{label_width}} {', '.join(map(str, sorted(mat_ids)))}\n"
                
                # Show energy ranges if available
                energies = self.perturbation.pert_energies
                if energies and len(energies) > 0:
                    pert_section += f"{'Energy range:':{label_width}} {min(energies):.2e} - {max(energies):.2e} MeV\n"
                    pert_section += f"{'Energy bins:':{label_width}} {len(energies)-1}\n"
                    
                    # Add grid structure identification
                    grid_name = _identify_energy_grid(energies)
                    if grid_name:
                        pert_section += f"{'Energy structure:':{label_width}} {grid_name}\n"
            else:
                pert_section += f"{'Number of perturbations:':{label_width}} 0 (empty collection)\n"
            
            # Add access hint at the end of the section
            pert_section += "\nUse .perturbation to access perturbation data.\n"
            sections.append(pert_section)
        
        # Join all sections
        content = "\n".join(sections)
        
        return header + content

