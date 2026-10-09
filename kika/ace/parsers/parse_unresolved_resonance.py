import logging
from typing import List, Optional
from kika.ace.classes.ace import Ace
from kika.ace.classes.unresolved_resonance import UnresolvedResonanceTables, ProbabilityTable

# Setup logger
logger = logging.getLogger(__name__)

def read_unresolved_resonance_block(ace: Ace, debug=False) -> Optional[UnresolvedResonanceTables]:
    """
    Read the UNR block containing unresolved resonance probability tables.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    Optional[UnresolvedResonanceTables]
        The unresolved resonance tables if data exists, None otherwise
    """
    if debug:
        logger.debug("\n===== UNR BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Create a new UnresolvedResonanceTables object
    unr_data = UnresolvedResonanceTables()
    unr_data.has_data = False
    
    # Check if we have the necessary data
    if not ace.header or not ace.header.jxs_array or ace.xss_data is None or len(ace.xss_data) == 0:
        if debug:
            logger.debug("Skipping UNR block: required data missing")
        return unr_data
    
    # Get UNR block index (JXS(23))
    unr_idx = ace.header.jxs_array[23]  # JXS(23)
    
    if debug:
        logger.debug(f"JXS(23) = {unr_idx} → Locator for UNR block")
    
    # Check if UNR block exists (JXS(23) ≠ 0)
    if unr_idx <= 0:
        if debug:
            logger.debug(f"No UNR block present: JXS(23)={unr_idx}")
        return unr_data
    
    # Validate index is within bounds
    if unr_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"Invalid UNR block index: {unr_idx} >= {len(ace.xss_data)}")
        return unr_data
    
    if debug:
        logger.debug(f"UNR block starts at index {unr_idx} (FORTRAN 1-indexed)")
    
    # Read the number of incident energies (N)
    num_energies = int(ace.xss_data[unr_idx])
    
    if debug:
        logger.debug(f"Number of incident energies (N): {num_energies}")
    
    if num_energies <= 0:
        if debug:
            logger.debug(f"Invalid number of energies: {num_energies}")
        return unr_data
    
    # Read the table length (M)
    if unr_idx + 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read table length")
        return unr_data
    
    table_length = int(ace.xss_data[unr_idx + 1])
    
    if debug:
        logger.debug(f"Table length (M): {table_length}")
    
    if table_length <= 0:
        if debug:
            logger.debug(f"Invalid table length: {table_length}")
        return unr_data
    
    # Read the interpolation parameter (INT)
    if unr_idx + 2 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read interpolation parameter")
        return unr_data
    
    interpolation = int(ace.xss_data[unr_idx + 2])
    
    if debug:
        logger.debug(f"Interpolation parameter (INT): {interpolation} → {interpolation==2 and 'linear-linear' or interpolation==5 and 'log-log' or 'unknown'}")
    
    # Read the inelastic competition flag (ILF)
    if unr_idx + 3 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read inelastic flag")
        return unr_data
    
    inelastic_flag = int(ace.xss_data[unr_idx + 3])
    
    if debug:
        if inelastic_flag < 0:
            logger.debug(f"Inelastic competition flag (ILF): {inelastic_flag} → Inelastic cross section is zero")
        elif inelastic_flag > 0:
            logger.debug(f"Inelastic competition flag (ILF): {inelastic_flag} → Special MT number for inelastic sum")
        else:
            logger.debug(f"Inelastic competition flag (ILF): {inelastic_flag} → Using balance relationship")
    
    # Read the other absorption flag (IOA)
    if unr_idx + 4 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read absorption flag")
        return unr_data
    
    other_absorption_flag = int(ace.xss_data[unr_idx + 4])
    
    if debug:
        if other_absorption_flag < 0:
            logger.debug(f"Other absorption flag (IOA): {other_absorption_flag} → Other absorption cross section is zero")
        elif other_absorption_flag > 0:
            logger.debug(f"Other absorption flag (IOA): {other_absorption_flag} → Special MT number for other absorption sum")
        else:
            logger.debug(f"Other absorption flag (IOA): {other_absorption_flag} → Using balance relationship")
    
    # Read the factors flag (IFF)
    if unr_idx + 5 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read factors flag")
        return unr_data
    
    factors_flag = int(ace.xss_data[unr_idx + 5])
    
    if debug:
        if factors_flag == 0:
            logger.debug(f"Factors flag (IFF): {factors_flag} → Tabulations are cross sections")
        elif factors_flag == 1:
            logger.debug(f"Factors flag (IFF): {factors_flag} → Tabulations are factors to multiply by smooth cross sections")
        else:
            logger.debug(f"Factors flag (IFF): {factors_flag} → Unknown value")
    
    # Check if we have enough data for the energy grid
    if unr_idx + 6 + num_energies > len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data for energy grid")
        return unr_data
    
    # Read the energy grid (view of xss_data)
    energies = ace.xss_data[unr_idx + 6:unr_idx + 6 + num_energies]
    
    if debug:
        logger.debug(f"Read {len(energies)} energy points for UNR grid")
    
    # Calculate the start of the probability tables (PTABLE)
    ptable_idx = unr_idx + 6 + num_energies
    
    if debug:
        logger.debug(f"Probability tables start at index {ptable_idx}")
        logger.debug(f"Expected table size: 6 × {table_length} × {num_energies} = {6 * table_length * num_energies}")
    
    # Check if we have enough data for the tables
    total_table_size = 6 * table_length * num_energies
    if ptable_idx + total_table_size > len(ace.xss_data):
        if debug:
            logger.debug(f"WARNING: Not enough data for complete probability tables")
            logger.debug(f"Available: {len(ace.xss_data) - ptable_idx}, Required: {total_table_size}")
        # Don't proceed if we don't have enough data
        return unr_data
    
    # Store the basic data
    unr_data.num_energies = num_energies
    unr_data.table_length = table_length
    unr_data.interpolation = interpolation
    unr_data.inelastic_flag = inelastic_flag
    unr_data.other_absorption_flag = other_absorption_flag
    unr_data.factors_flag = factors_flag
    unr_data.energies = energies
    unr_data.tables = []
    
    # Read each probability table
    if debug:
        logger.debug(f"Reading {num_energies} probability tables")
    
    # Loop over each energy point
    for i in range(num_energies):
        energy_value = float(energies[i])
        table = ProbabilityTable(energy=energy_value)
        
        if debug and i == 0:
            logger.debug(f"  Reading table for energy {energy_value}")
        
        # Read the 6 components of the probability table according to Table 59
        # j=1: cumulative probability, j=2: total XS, j=3: elastic XS, 
        # j=4: fission XS, j=5: (n,γ) XS, j=6: heating number
        component_names = ["cumulative prob", "total XS", "elastic XS", "fission XS", "(n,γ) XS", "heating"]
        
        for j in range(6):
            # Calculate starting index for this component at this energy
            # PTABLE + (i-1)*6M + (j-1)*M
            start_idx = ptable_idx + i * 6 * table_length + j * table_length
            end_idx = start_idx + table_length
            
            if debug and i == 0:
                logger.debug(f"    Component {j+1}: {component_names[j]} - indices {start_idx}:{end_idx}")
            
            # Get the values for this component
            values = ace.xss_data[start_idx:end_idx]
            
            # Assign the values to the appropriate array
            if j == 0:
                table.cumulative_probabilities = values
            elif j == 1:
                table.total_xs = values
            elif j == 2:
                table.elastic_xs = values
            elif j == 3:
                table.fission_xs = values
            elif j == 4:
                table.capture_xs = values
            elif j == 5:
                table.heating_numbers = values
        
        unr_data.tables.append(table)
    
    unr_data.has_data = True
    
    if debug:
        logger.debug(f"Successfully read {len(unr_data.tables)} probability tables")
    
    return unr_data
