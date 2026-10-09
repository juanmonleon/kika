from typing import Optional, Tuple


class ErrorMessageDict(dict):
    """Dictionary that provides helpful error messages when accessing non-existent keys."""
    
    def __init__(self, *args, dict_name="Dictionary", **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_name = dict_name
    
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            # Create a user-friendly error message with available keys
            available_keys = sorted(self.keys_as_int())
            
            # Format the error message without newlines to avoid raw \n in output
            if (available_keys):
                error_msg = f"Error: MT={key} not found in {self.dict_name}. Available MT numbers: {available_keys}"
            else:
                error_msg = f"Error: MT={key} not found in {self.dict_name}. No MT numbers available in this collection."
            
            raise KeyError(error_msg)
    
    def keys_as_int(self):
        """Get keys as integers for better display."""
        return [int(key) for key in self.keys()]


class ErrorMessageList(list):
    """List that provides helpful error messages when accessing non-existent indices."""
    
    def __init__(self, *args, list_name="List", **kwargs):
        super().__init__(*args, **kwargs)
        self.list_name = list_name
    
    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except IndexError:
            # Create a user-friendly error message with available indices
            if len(self) > 0:
                error_msg = f"Error: Particle index {idx} not found in {self.list_name}. Available particle indices: {list(range(len(self)))}"
                
                # Add particle counts information in a single line if available
                counts_info = []
                for i, particle_data in enumerate(self):
                    counts_info.append(f"Index {i}: {len(particle_data)} reactions")
                
                if counts_info:
                    error_msg += f" (Particle counts: {', '.join(counts_info)})"
            else:
                error_msg = f"Error: Particle index {idx} not found in {self.list_name}. No particle production data available."
            
            raise IndexError(error_msg)




class Law44DataError(Exception):
    """Exception raised when Law 44 data is required but not available."""
    pass