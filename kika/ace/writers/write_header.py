def write_header(header):
    """Generate a string with the ACE header section from a Header object.
    
    This function formats the header exactly as it would appear in an ACE file,
    handling both legacy and 2.0.1 formats.
    
    :param header: The Header object containing all header information
    :type header: Header
    
    :returns: A formatted string representing the complete header section
    :rtype: str
    
    :raises ValueError: If required header fields are missing
    """
    if header is None:
        raise ValueError("Header object is None")
        
    if header.format_version not in ["legacy", "2.0.1"]:
        raise ValueError(f"Unsupported header format: {header.format_version}")
    
    result = []
    
    # Format the header opening based on format version
    if header.format_version == "legacy":
        # Check required fields
        if header.zaid is None or header.atomic_weight_ratio is None or header.temperature is None:
            raise ValueError("Required header fields missing for legacy format")
            
        # Line 1: ZAID, AWR, Temperature, Date
        date = header.date or ""
        # Format ZAID with extension if present
        zaid_str = f"{header.zaid}"
        if header.extension:
            zaid_str += header.extension
        line1 = f"{zaid_str:>10}{header.atomic_weight_ratio:12.6f}{header.temperature:12.4E}   {date:8}"
        result.append(line1)
        
        # Line 2: Comment (70 chars) + Material ID (starting at position 74)
        comment = header.comment or ""
        comment = comment[:70].ljust(70)  # Limit to 70 chars and pad
        
        mat_str = ""
        if header.matid is not None:
            # Position 74 is index 73, so need 73 characters before mat
            line2 = f"{comment}   "  # Add 3 spaces after comment to reach position 74
            mat_str = f"mat{header.matid:4d}"
            line2 += mat_str
        else:
            line2 = comment
            
        result.append(line2)
    else:  # 2.0.1 format
        # Check required fields
        if (header.zaid is None or header.atomic_weight_ratio is None or 
            header.temperature is None or header.comment_line_count is None):
            raise ValueError("Required header fields missing for 2.0.1 format")
            
        # Manual Table 2. Line 1: VERS A10, SZAID A24, SRC A24
        version = header.ace_version or ""
        zaid_str = f"{header.zaid}"
        if header.extension:
            zaid_str += header.extension
        source1 = header.source or ""
        line1 = f"{version:<10}{zaid_str:>24}{source1:>24}"
        result.append(line1)
        
        # Line 2: AW E12.0, TZ E12.0, 1X, HD A10, N I10
        date = header.date or ""
        line2 = (f"{header.atomic_weight_ratio:12.6f}{header.temperature:12.4E} "
                 f"{date:<10}{header.comment_line_count:10d}")
        result.append(line2)
        
        # Add comment lines
        if header.comment:
            # Split comment into lines and add each line
            comment_lines = header.comment.split('\n')
            for i in range(min(header.comment_line_count, len(comment_lines))):
                result.append(comment_lines[i])
            
            # If there are fewer comment lines than specified, add empty lines
            for i in range(len(comment_lines), header.comment_line_count):
                result.append("")
    
    # Format IZAW array (16 pairs over 4 lines)
    # Format for each line: 4(I7,F11.0) - right-justified integers and floats
    if header.izaw_array:
        # Skip index 0 and process 16 elements starting from index 1
        for i in range(0, 16, 4):
            line = ""
            for j in range(4):
                # Get index from original 1-indexed array but account for 0th element
                idx = i + j + 1  # Add 1 to start from index 1
                if idx < len(header.izaw_array):
                    za, awr = header.izaw_array[idx]
                else:
                    za, awr = 0, 0.0
                # Format exactly as Fortran 4(I7,F11.0) - right-justified
                line += f"{za:7d}{awr:11.0f}"
            result.append(line)
    
    # Format NXS array (16 integers over 2 lines)
    # Format for each line: 8I9
    if header.nxs_array:
        for i in range(0, 16, 8):
            line_parts = []
            for j in range(8):
                # Get index from original 1-indexed array
                idx = i + j + 1  # Add 1 to start from index 1
                if idx < len(header.nxs_array):
                    value = header.nxs_array[idx]
                else:
                    value = 0
                line_parts.append(f"{value:9d}")
            result.append("".join(line_parts))
    
    # Format JXS array (32 integers over 4 lines)
    # Format for each line: 8I9
    if header.jxs_array:
        for i in range(0, 32, 8):
            line_parts = []
            for j in range(8):
                # Get index from original 1-indexed array
                idx = i + j + 1  # Add 1 to start from index 1
                if idx < len(header.jxs_array):
                    value = header.jxs_array[idx]
                else:
                    value = 0
                line_parts.append(f"{value:9d}")
            result.append("".join(line_parts))
    
    # Join all lines with newline characters
    return "\n".join(result)
