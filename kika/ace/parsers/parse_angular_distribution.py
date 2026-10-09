from typing import List, Optional
import numpy as np
from kika.ace.classes.ace import Ace
from kika.ace.classes.angular_distribution.base import AngularDistribution
from kika.ace.classes.angular_distribution.container import AngularDistributionContainer
from kika.ace.classes.angular_distribution.distributions.isotropic import IsotropicAngularDistribution
from kika.ace.classes.angular_distribution.distributions.equiprobable import EquiprobableAngularDistribution
from kika.ace.classes.angular_distribution.distributions.tabulated import TabulatedAngularDistribution
from kika.ace.classes.angular_distribution.distributions.kalbach_mann import KalbachMannAngularDistribution
import logging

# Setup logger
logger = logging.getLogger(__name__)

def read_angular_distribution_blocks(ace: Ace, debug: bool = False) -> AngularDistributionContainer:
    """
    Read the AND, ANDP, and ANDH blocks from the ACE file.
    
    These blocks contain angular distribution data:
    - AND: For incident neutron reactions (JXS[9])
    - ANDP: For photon production reactions (JXS[17])
    - ANDH: For other particle production reactions (accessed through particle pointers)
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    AngularDistributionContainer
        The angular distribution container
    
    Raises
    ------
    ValueError
        If required data is missing or indices are invalid
    """
    if debug:
        logger.debug("Reading angular distribution blocks")
        
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        raise ValueError("Cannot read angular distribution blocks: header or XSS data missing")
    
    # Make sure we have angular locators
    if ace.angular_locators is None:
        raise ValueError("Angular distribution locators missing. Call read_angular_locator_blocks first.")
    
    # Create a new container if not present
    if ace.angular_distributions is None:
        ace.angular_distributions = AngularDistributionContainer()
    
    # Get the JXS pointers for each block
    and_idx = ace.header.jxs_array[9]  # JXS(9)
    andp_idx = ace.header.jxs_array[17]  # JXS(17)
    
    if debug:
        logger.debug(f"AND block index: {and_idx}, ANDP block index: {andp_idx}")
    
    # Read AND block (incident neutron reactions)
    read_and_block(ace, and_idx, debug)
    
    # Read ANDP block (photon production)
    read_andp_block(ace, andp_idx, debug)
    
    # Read ANDH blocks (other particle production)
    read_andh_blocks(ace, debug)
    
    return ace.angular_distributions

def read_and_block(ace: Ace, and_idx: int, debug: bool = False) -> None:
    """
    Read the AND block containing angular distributions for incident neutron reactions.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    and_idx : int
        Starting index of the AND block in the XSS array
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the AND block data is missing or invalid
    """
    if debug:
        logger.debug(f"Reading AND block at index {and_idx}")
        
    # Check if AND block is present
    if and_idx <= 0:
        return  # No AND block
    
    locators = ace.angular_locators
    mts = []
    if ace.reaction_mt_data and ace.reaction_mt_data.has_neutron_mt_data:
        mts = ace.reaction_mt_data.incident_neutron

    # Elastic: LOCB(1), always present in LAND. LOCB=0 means isotropic in the
    # centre of mass, and a NE=0 or all-LC=0 array reads back as isotropic too.
    if locators.elastic_scattering is not None:
        elastic_locb = int(locators.elastic_scattering)
        if elastic_locb > 0:
            elastic_data_idx = and_idx + elastic_locb - 1
            if debug:
                logger.debug(f"Elastic scattering data index: and_idx + locb - 1 = {and_idx} + {elastic_locb} - 1 = {elastic_data_idx}")
            try:
                ace.angular_distributions.elastic = read_angular_distribution(
                    ace, elastic_data_idx, 2, and_idx, debug)
            except ValueError as e:
                raise ValueError(f"Error reading elastic scattering distribution: {e}")
        elif elastic_locb == 0:
            ace.angular_distributions.elastic = IsotropicAngularDistribution(mt=2)

    # LOCB(2..NXS(5)+1) belong to the first NXS(5) reactions of the MTR block,
    # the ones that emit neutrons (manual Tables 16-17).
    if len(locators.incident_neutron) > len(mts):
        logger.warning(
            f"LAND has {len(locators.incident_neutron)} reaction locators but MTR only "
            f"{len(mts)} MTs; the angular distributions past MTR are not read")
    for i, locb_entry in enumerate(locators.incident_neutron[:len(mts)]):
        locb_value = int(locb_entry)
        mt_value = int(mts[i])

        if locb_value == 0:
            # No data: isotropic, in the frame the TYR sign gives (manual 4.3.9)
            ace.angular_distributions.incident_neutron[mt_value] = IsotropicAngularDistribution(mt=mt_value)
            continue
        if locb_value == -1:
            # The angle is correlated with the energy and lives in the DLW
            # block (LAW=44, 61 or 67)
            if debug:
                logger.debug(f"Neutron reaction MT={mt_value}: LOCB=-1 → angular data in the DLW block")
            ace.angular_distributions.incident_neutron[mt_value] = KalbachMannAngularDistribution(
                mt=mt_value,
                reaction_index=i,
                is_particle_production=False,
                requires_law44_data=True,
            )
            continue
        if locb_value < -1:
            logger.warning(f"Invalid LAND locator {locb_value} for MT={mt_value}; angular distribution not read")
            continue

        data_idx = and_idx + locb_value - 1
        if debug:
            logger.debug(f"Angular distribution data index: and_idx + locb - 1 = {and_idx} + {locb_value} - 1 = {data_idx}")
        if data_idx >= len(ace.xss_data):
            logger.warning(f"AND locator for MT={mt_value} points past the XSS array ({data_idx}); not read")
            continue
        try:
            dist = read_angular_distribution(ace, data_idx, mt_value, and_idx, debug)
        except ValueError as e:
            logger.warning(f"Angular distribution for MT={mt_value} could not be read: {e}")
            continue
        ace.angular_distributions.incident_neutron[mt_value] = dist

def read_andp_block(ace: Ace, andp_idx: int, debug: bool = False) -> None:
    """
    Read the ANDP block containing angular distributions for photon production reactions.
    
    According to the ACE format specification, the angular distribution for the i-th 
    photon-producing reaction begins at JXS(17) + LOCBᵢ - 1, where LOCBᵢ comes from the 
    LANDP block. Each distribution contains:
    1. Number of energies (N_E)
    2. Energy grid (N_E values)
    3. Locators (L_C) for each energy (N_E values), relative to JXS(17)
    4. 32 equiprobable cosine bins (33 values) for each energy
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    andp_idx : int
        Starting index of the ANDP block in the XSS array (JXS(17))
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the ANDP block data is missing or invalid
    """
    if debug:
        logger.debug(f"Reading ANDP block at index {andp_idx}")
        
    # Check if ANDP block is present
    if andp_idx <= 0:
        if debug:
            logger.debug("No ANDP block found (andp_idx <= 0)")
        return  # No ANDP block
    
    try:
        # Get the number of photon production reactions from the locators
        num_photon_reactions = len(ace.angular_locators.photon_production)
        
        if debug:
            logger.debug(f"Number of photon production reactions with locators: {num_photon_reactions}")
            if num_photon_reactions == 0:
                logger.debug("No photon production reactions to process")
        
        if num_photon_reactions == 0:
            return
            
        # Count of isotropic distributions (specifically from LOCB=0)
        isotropic_count = 0
        non_isotropic_count = 0
        error_count = 0
        
        # Process photon production angular distributions
        for i, locb_entry in enumerate(ace.angular_locators.photon_production):
            locb_value = int(locb_entry)
            
            # Get the corresponding MT number
            mt_entry = None
            if ace.reaction_mt_data and ace.reaction_mt_data.has_photon_production_mt_data:
                if i < len(ace.reaction_mt_data.photon_production):
                    mt_entry = int(ace.reaction_mt_data.photon_production[i])
            
            if mt_entry is None:
                if debug:
                    logger.debug(f"Skipping photon reaction at index {i}: No MT number available")
                error_count += 1
                continue  # Skip if MT number not available
            
            mt_value = int(mt_entry)
            
            if locb_value == 0:
                # This is a legitimate isotropic distribution as per documentation
                dist = IsotropicAngularDistribution(mt=mt_entry)
                
                # Store using the MT value as the key
                ace.angular_distributions.photon_production[mt_value] = dist
                isotropic_count += 1
                
                if debug and i < 5:  # Limit debug output to first few reactions
                    logger.debug(f"Photon reaction MT={mt_value}: LOCB=0 → Created isotropic distribution")
                continue
                
            elif locb_value < 0:  # Invalid negative value
                logger.error(f"ERROR: Invalid negative locator value {locb_value} for photon production angular distribution MT={mt_value}")
                error_count += 1
                continue
            
            # Calculate the data index according to the documentation
            data_idx = andp_idx + locb_value - 1
            
            if debug and i < 5:  # Limit debug output
                logger.debug(f"Photon production distribution for MT={mt_value}: LOCB={locb_value}, index={data_idx}")
            
            # Check if the index is valid before trying to read
            if data_idx < 0 or data_idx >= len(ace.xss_data):
                logger.error(f"ERROR: Invalid angular distribution index for photon production MT={mt_value}: {data_idx} out of bounds")
                error_count += 1
                continue
            
            try:
                # Read the angular distribution with the ANDP block base index
                dist = read_angular_distribution_photon(ace, data_idx, andp_idx, mt_entry, debug)
                if dist:
                    # Store using the MT value as the key
                    ace.angular_distributions.photon_production[mt_value] = dist
                    non_isotropic_count += 1
                    
                    if debug and i < 5:  # Limit debug output
                        logger.debug(f"Read photon production distribution for MT={mt_value}")
                        
                        if isinstance(dist, EquiprobableAngularDistribution) and len(dist.cosine_bins) > 0:
                            # Access the cosine bins directly since they're already float values
                            sample = [dist.cosine_bins[0][j] for j in range(min(3, len(dist.cosine_bins[0])))]
                            logger.debug(f"  First 3 cosine values: {sample}")
            except ValueError as e:
                error_msg = f"ERROR reading photon production distribution for MT={mt_value}: {e}"
                logger.error(error_msg)
                error_count += 1
                continue
        
        if debug:
            logger.debug(f"ANDP processing summary: {isotropic_count} legitimate isotropic distributions (LOCB=0), "
                         f"{non_isotropic_count} non-isotropic distributions, {error_count} errors/skipped")
            logger.debug(f"Total photon production angular distributions: {len(ace.angular_distributions.photon_production)}")
    except Exception as e:
        raise ValueError(f"Error reading ANDP block: {e}")

def read_angular_distribution_photon(ace: Ace, data_idx: int, andp_idx: int, mt_entry: int, debug: bool = False) -> Optional[AngularDistribution]:
    """
    Read a photon production angular distribution from the XSS array.
    
    The distribution structure follows Table 54 from the ACE format specification:
    1. Number of energies (N_E)
    2. Energy grid (N_E values)
    3. Locators (L_C) for each energy (N_E values), relative to JXS(17)
    4. 32 equiprobable cosine bins (33 values) for each energy
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data
    data_idx : int
        Starting index of the angular distribution data in the XSS array
    andp_idx : int
        Base index of the ANDP block (JXS(17)) for relative locators
    mt_entry : int
        MT number for this reaction
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    AngularDistribution
        The parsed angular distribution
        
    Raises
    ------
    ValueError
        If data is invalid or inconsistent
    """
    if debug:
        logger.debug(f"Reading photon production angular distribution at index {data_idx} for MT={mt_entry}")
    
    if data_idx < 0 or data_idx >= len(ace.xss_data):
        raise ValueError(f"Angular distribution index out of bounds: {data_idx}")
    
    # First value is the number of energies (N_E)
    loc = data_idx
    num_energies = int(ace.xss_data[loc])
    
    if debug:
        logger.debug(f"Number of energy points: {num_energies}")
    
    if num_energies <= 0:
        # If N_E = 0, we have isotropic scattering for all energies
        return IsotropicAngularDistribution(mt=mt_entry)
    
    # Check if we have enough data
    if loc + 1 + 2*num_energies > len(ace.xss_data):
        raise ValueError(f"Photon angular distribution data truncated: need at least {1 + 2*num_energies} entries, but only {len(ace.xss_data) - loc} available")
    
    # Read the energy grid (N_E values)
    energies = ace.xss_data[loc + 1:loc + 1 + num_energies]
    
    # Read the locators (L_C) for each energy (N_E values) - relative to JXS(17)
    lc_start = loc + 1 + num_energies
    locc_entries = ace.xss_data[lc_start:lc_start + num_energies]
    
    # Create an equiprobable angular distribution
    distribution = EquiprobableAngularDistribution(mt=mt_entry, _energies=energies)
    
    # For each energy point with a non-zero locator
    for i, locc_entry in enumerate(locc_entries):
        locc_value = int(locc_entry)
        
        if locc_value == 0:
            # Isotropic distribution at this energy
            # Add 33 values from -1 to 1 (uniformly spaced)
            cosines = np.array([-1.0 + j * (2.0 / 32) for j in range(33)])
            distribution._cosine_bins.append(cosines)
            if debug:
                logger.debug(f"Using isotropic distribution for energy point {i} ({energies[i]})")
            continue
        
        # L_C is relative to JXS(17), so calculate the absolute index
        data_loc = andp_idx + locc_value - 1
        
        if debug:
            logger.debug(f"Reading cosine bins for energy point {i}: L_C={locc_value}, index={data_loc}")
        
        # Check if we have enough data
        if data_loc + 33 > len(ace.xss_data):
            raise ValueError(f"Equiprobable bin data truncated at energy {energies[i]}: need 33 entries, but only {len(ace.xss_data) - data_loc} available")
        
        # Read the 33 cosine values for 32 equiprobable bins
        cosines = ace.xss_data[data_loc:data_loc + 33]
        distribution._cosine_bins.append(cosines)
        
        if debug and i == 0:  # Print details for first energy only to avoid verbose output
            logger.debug(f"Cosine bins for first energy: {cosines[:5].tolist()}... (showing first 5)")
    
    return distribution

def read_andh_blocks(ace: Ace, debug: bool = False) -> None:
    """
    Read the ANDH blocks containing angular distributions for other particle production reactions.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the ANDH block data is missing or invalid
    """
    if debug:
        logger.debug("Reading ANDH blocks")
        logger.debug(f"Number of particle types: {ace.angular_locators.num_particle_types}")
    
    # Get the number of particle types
    num_particle_types = ace.angular_locators.num_particle_types
    if num_particle_types <= 0:
        return  # No particle production data
    
    # Initialize particle production dictionaries
    while len(ace.angular_distributions.particle_production) < num_particle_types:
        ace.angular_distributions.particle_production.append({})
    
    # Process each particle type
    for particle_idx in range(num_particle_types):
        if debug:
            logger.debug(f"Processing particle type {particle_idx}")
            
        # Get locators for this particle type
        locators = ace.angular_locators.get_particle_production_locators(particle_idx)
        if locators is None or len(locators) == 0:
            continue  # No locators for this particle type
        
        # Get MT numbers for this particle type
        mt_numbers = []
        if ace.reaction_mt_data and ace.reaction_mt_data.has_particle_production_mt_data:
            mt_numbers = ace.reaction_mt_data.get_particle_production_mt_numbers(particle_idx)
            if mt_numbers is None:
                mt_numbers = []
        
        if len(mt_numbers) == 0 or len(mt_numbers) != len(locators):
            continue  # MT numbers missing or count mismatch
        
        # Get the ANDH pointer for this particle
        andh_ptr = 0
        if len(ace.header.jxs_array) > 32:  # Need JXS(32)
            jxs32_idx = ace.header.jxs_array[32]  # JXS(32)
            if jxs32_idx > 0 and particle_idx < num_particle_types:
                # ANDH pointer is at XSS(JXS(32)+10*(i-1)+6)
                i = particle_idx + 1  # Convert to 1-based for formula
                andh_pointer_idx = jxs32_idx + 10 * (i - 1) + 6
                
                if debug:
                    logger.debug(f"ANDH pointer index calculation: jxs32 + 10*(i-1) + 6 = {jxs32_idx} + 10*({i}-1) + 6 = {andh_pointer_idx}")
                
                if andh_pointer_idx < len(ace.xss_data):
                    andh_ptr = int(ace.xss_data[andh_pointer_idx])
                    
                    if debug:
                        logger.debug(f"ANDH pointer = {andh_ptr}")
                    
                    # Validate the pointer value
                    if andh_ptr <= 0:
                        if debug:
                            logger.debug("ANDH pointer <= 0, skipping particle")
                        continue
                    
                    if andh_ptr > len(ace.xss_data):
                        if debug:
                            logger.debug(f"ANDH pointer {andh_ptr} is out of bounds, skipping particle")
                        continue
                    
                else:
                    continue
            else:
                continue
        else:
            continue
        
        if andh_ptr <= 0:
            continue  # No ANDH block for this particle type
            
        # Process angular distributions for this particle type
        success_count = 0
        error_count = 0
        kalbach_mann_count = 0  # Count of Law=44 (Kalbach-Mann) distributions
        for i, (mt, locator) in enumerate(zip(mt_numbers, locators)):
            locator_value = int(locator)
            mt_value = int(mt)
            mt = mt_value
            
            if locator_value == 0:
                # Isotropic distribution according to documentation
                dist = IsotropicAngularDistribution(mt=mt)
                ace.angular_distributions.particle_production[particle_idx][mt_value] = dist
                success_count += 1
                
                if debug and i < 5:  # Limit debug output
                    logger.debug(f"Particle reaction MT={mt_value}: LOCB=0 → Created isotropic distribution")
                continue
            elif locator_value == -1:
                # Angular distribution is in the DLWH block using Law=44
                if debug:
                    logger.debug(f"Particle production MT={mt_value}: LOCB=-1 → Kalbach-Mann (Law=44) angular distribution")
                    logger.debug(f"  NOTE: This distribution requires data from the energy distribution Law=44 in the DLWH block")
                
                # Create a Kalbach-Mann distribution object with the reaction and particle indices
                # This will be used to lookup the appropriate Law=44 distribution in the DLWH block
                dist = KalbachMannAngularDistribution(
                    mt=mt,
                    reaction_index=i,
                    is_particle_production=True,
                    particle_idx=particle_idx,
                    requires_law44_data=True  # Explicitly flag that this needs Law=44 data
                )
                
                # Store using the MT value as the key
                ace.angular_distributions.particle_production[particle_idx][mt_value] = dist
                kalbach_mann_count += 1
                success_count += 1
                
                if debug:
                    logger.debug(f"Read particle production distribution for particle {particle_idx}, MT={mt_value}")
                continue
            elif locator_value < -1:  # Invalid negative value
                logger.error(f"ERROR: Invalid negative locator value {locator_value} for particle {particle_idx}, MT={mt_value}")
                error_count += 1
                continue
            
            # Calculate the actual data index using the locator
            # The locator points to the angular distribution data for this MT
            # according to Table 23: each distribution is at ANDH + LOCB - 1
            data_idx = andh_ptr + locator_value - 1
            
            if debug:
                logger.debug(f"Particle production MT={mt_value}: LOCB={locator_value}")
                logger.debug(f"Data location: ANDH + LOCB - 1 = {andh_ptr} + {locator_value} - 1 = {data_idx}")
            
            # Check if the index is valid before trying to read
            if data_idx < 0 or data_idx >= len(ace.xss_data):
                logger.error(f"ERROR: Invalid angular distribution index for particle {particle_idx}, MT={mt_value}: {data_idx} out of bounds")
                error_count += 1
                continue
            
            try:
                # Read the angular distribution from the ANDH address
                # The angular distribution at ANDH follows the same format as at AND
                dist = read_angular_distribution(ace, data_idx, mt, andh_ptr, debug)  # Pass andh_ptr as base_idx
                if dist:
                    ace.angular_distributions.particle_production[particle_idx][mt_value] = dist
                    success_count += 1
                    if debug:
                        logger.debug(f"Read particle production distribution for particle {particle_idx}, MT={mt_value}")
            except ValueError as e:
                logger.error(f"ERROR reading particle {particle_idx} distribution for MT={mt_value}: {e}")
                error_count += 1
                continue
        
        if debug and (success_count > 0 or error_count > 0):
            logger.debug(f"Particle {particle_idx} summary: {success_count} distributions successfully read "
                         f"({kalbach_mann_count} Kalbach-Mann), {error_count} errors/skipped")

def read_angular_distribution(ace: Ace, data_idx: int, mt_entry: int, base_idx: int, debug: bool = False) -> Optional[AngularDistribution]:
    """
    Read a single angular distribution from the XSS array.
    
    The distribution structure starts with:
    1. Number of energies (NE)
    2. Energy grid (NE values)
    3. Locators (LC) for each energy (NE values)
    4. Distribution data pointed to by each LC
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data
    data_idx : int
        Starting index of the angular distribution data in the XSS array
    mt_entry : int
        MT number for this reaction
    base_idx : int
        Base index (AND, ANDP, or ANDH) for calculating locator positions
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    AngularDistribution
        The parsed angular distribution
        
    Raises
    ------
    ValueError
        If data is invalid or inconsistent
    """
    if debug:
        logger.debug(f"Reading angular distribution at index {data_idx} for MT={mt_entry}")
        logger.debug(f"Using base index {base_idx} for locator calculations")
    
    if data_idx < 0 or data_idx >= len(ace.xss_data):
        raise ValueError(f"Angular distribution index out of bounds: {data_idx}")
    
    # First value is the number of energies (NE)
    loc = data_idx
    num_energies = int(ace.xss_data[loc])
    
    if debug:
        logger.debug(f"Number of energy points (NE): {num_energies}")
    
    if num_energies <= 0:
        # If NE = 0, we have isotropic scattering for all energies
        return IsotropicAngularDistribution(mt=mt_entry)
    
    # Check if we have enough data
    if loc + 1 + 2*num_energies > len(ace.xss_data):
        raise ValueError(f"Angular distribution data truncated: need at least {1 + 2*num_energies} entries, but only {len(ace.xss_data) - loc} available")
    
    # Read the energy grid (NE values)
    energy_start = loc + 1
    energies = ace.xss_data[energy_start:energy_start + num_energies]
    
    # Read the locators (LC/LOCC) for each energy (NE values)
    locc_start = energy_start + num_energies
    locc_entries = ace.xss_data[locc_start:locc_start + num_energies]
    locc_values = [int(entry) for entry in locc_entries]
    
    if debug:
        # Display the first few energy points and their corresponding LOCC values
        display_count = min(5, num_energies)
        
        logger.debug(f"First {display_count} energy points and their LOCC values:")
        for i in range(display_count):
            e_val = energies[i]
            locc_val = locc_values[i]
            locc_type = "isotropic" if locc_val == 0 else ("equiprobable" if locc_val > 0 else "tabulated")
            logger.debug(f"  Energy[{i}] = {e_val} MeV, LOCC = {locc_val} → {locc_type}")
    
    # Check distribution type based on the locators
    # Handle special case: if all locators are 0, create isotropic distribution
    if all(lc_val == 0 for lc_val in locc_values):
        if debug:
            logger.debug("All LOCC values are 0 → isotropic distribution for all energies")
        # All locators are 0, meaning isotropic for all energies
        return IsotropicAngularDistribution(mt=mt_entry, _energies=energies)
    
    # Check if we have equiprobable bin or tabulated distributions
    if all(lc_val > 0 for lc_val in locc_values if lc_val != 0):
        if debug:
            logger.debug("All non-zero LOCC values are positive → equiprobable bin distribution")
        # All non-zero locators are positive - equiprobable bin distribution
        return read_equiprobable_distribution(ace, base_idx, mt_entry, num_energies, energies, locc_entries, debug)
    
    # Negative locators, or a mix of both signs: the sign is a per-energy flag
    # (manual 4.3.9), so a mixed array is legal. It is read as tabulated, with
    # each 32-bin table carried as the histogram it is.
    if debug:
        logger.debug("Negative or mixed LOCC values → tabulated distribution")
    return read_tabulated_distribution(ace, base_idx, mt_entry, num_energies, energies, locc_entries, debug)

def read_equiprobable_distribution(ace: Ace, base_idx: int, mt_entry: int, 
                                   num_energies: int, energies: np.ndarray, 
                                   locators: np.ndarray, debug: bool = False) -> EquiprobableAngularDistribution:
    """
    Read a 32 equiprobable cosine bin angular distribution.
    
    Each distribution consists of 33 cosine values defining 32 equiprobable bins.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data
    base_idx : int
        Base index of the AND block in the XSS array (not the distribution data!)
    mt_entry : int
        MT number for this reaction
    num_energies : int
        Number of energy points
    energies : np.ndarray
        Energy grid for the angular distribution (view of xss_data)
    locators : np.ndarray
        Locators (LC) for each energy (view of xss_data)
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    EquiprobableAngularDistribution
        The parsed equiprobable bin distribution
        
    Raises
    ------
    ValueError
        If data is invalid or inconsistent
    """
    if debug:
        logger.debug(f"Reading equiprobable distribution for MT={mt_entry} with {num_energies} energies")
    
    # Create an equiprobable angular distribution
    distribution = EquiprobableAngularDistribution(mt=mt_entry, _energies=energies)
    
    # For each energy point with a non-zero locator
    for i, locc_entry in enumerate(locators):
        locc_value = int(locc_entry)
        
        if locc_value == 0:
            # Isotropic distribution at this energy
            # Add 33 values from -1 to 1 (uniformly spaced)
            cosines = np.array([-1.0 + j * (2.0 / 32) for j in range(33)])
            distribution._cosine_bins.append(cosines)
            
            if debug and i < 3:  # Show first 3 energy points only
                logger.debug(f"Energy point {i} ({energies[i]} MeV): LOCC=0 → Using isotropic distribution")
                cosine_values = cosines[:5].tolist()  # Show first 5 cosine values
                logger.debug(f"  First 5 cosine values: {cosine_values}...")
            
            continue
        
        # Positive locator points to 32 equiprobable bin boundaries
        data_loc = base_idx + locc_value - 1
        
        if debug and i < 3:  # Show first 3 energy points only
            logger.debug(f"Energy point {i} ({energies[i]} MeV): LOCC={locc_value} → Equiprobable bins at {data_loc}")
        
        # Check if we have enough data
        if data_loc + 33 > len(ace.xss_data):
            raise ValueError(f"Equiprobable bin data truncated at energy {energies[i]}: need 33 entries, but only {len(ace.xss_data) - data_loc} available")
        
        # Read the 33 cosine values
        cosines = ace.xss_data[data_loc:data_loc + 33]
        distribution._cosine_bins.append(cosines)
        
        if debug and i < 3:  # Show first 3 energy points only
            cosine_values = cosines[:5].tolist()  # Show first 5 cosine values
            logger.debug(f"  First 5 cosine values: {cosine_values}...")
    
    return distribution

def read_tabulated_distribution(ace: Ace, base_idx: int, mt_entry: int, 
                                num_energies: int, energies: np.ndarray, 
                                locators: np.ndarray, debug: bool = False) -> TabulatedAngularDistribution:
    """
    Read a tabulated angular distribution.
    
    Each distribution consists of:
    1. Interpolation flag (0=histogram, 1=linear-linear per ACE Table 20)
    2. Number of points (Np)
    3. Cosine grid (Np values)
    4. PDF values (Np values)
    5. CDF values (Np values)
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data
    base_idx : int
        Base index of the AND block in the XSS array (not the distribution data!)
    mt_entry : int
        MT number for this reaction
    num_energies : int
        Number of energy points
    energies : np.ndarray
        Energy grid for the angular distribution (view of xss_data)
    locators : np.ndarray
        Locators (LC) for each energy (view of xss_data)
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    TabulatedAngularDistribution
        The parsed tabulated distribution
        
    Raises
    ------
    ValueError
        If data is invalid or inconsistent
    """
    if debug:
        logger.debug(f"Reading tabulated distribution for MT={mt_entry} with {num_energies} energies")
    
    # Create a tabulated angular distribution
    distribution = TabulatedAngularDistribution(mt=mt_entry, _energies=energies)
    
    # For each energy point
    for i, locc_entry in enumerate(locators):
        locc_value = int(locc_entry)
        
        if locc_value == 0:
            # Isotropic distribution at this energy
            # Add a simple two-point distribution: μ=[-1,1], PDF=[0.5,0.5], CDF=[0,1]
            # JJ follows ENDF (1 histogram, 2 lin-lin), which is what NJOY
            # writes; manual Table 20 says 0/1, but no processed table uses that.
            distribution.interpolation.append(2)
            
            # The simple two-point distribution (not part of xss_data)
            cosines = np.array([-1.0, 1.0])
            pdfs = np.array([0.5, 0.5])
            cdfs = np.array([0.0, 1.0])
            
            distribution._cosine_grid.append(cosines)
            distribution._pdf.append(pdfs)
            distribution._cdf.append(cdfs)
            
            if debug and i < 3:  # Show first 3 energy points only
                logger.debug(f"Energy point {i} ({energies[i]} MeV): LOCC=0 → Using isotropic distribution")
                logger.debug(f"  Created 2-point distribution with cosines=[-1,1], PDF=[0.5,0.5], CDF=[0,1]")
            
            continue
        
        if locc_value > 0:
            # 32 equiprobable bins in an otherwise tabulated array: the 33
            # edges become a histogram (JJ=1) with 1/32 in each bin.
            data_loc = base_idx + locc_value - 1
            if data_loc + 33 > len(ace.xss_data):
                raise ValueError(f"Equiprobable bin data truncated at energy {energies[i]}")
            edges = np.array(ace.xss_data[data_loc:data_loc + 33])
            widths = np.diff(edges)
            density = np.divide(1.0 / 32.0, widths, out=np.zeros_like(widths), where=widths > 0)
            distribution.interpolation.append(1)
            distribution._cosine_grid.append(edges)
            distribution._pdf.append(np.append(density, density[-1]))
            distribution._cdf.append(np.arange(33) / 32.0)
            continue

        # Negative locator points to tabulated distribution
        lc_abs = abs(locc_value)
        # LC is relative to the base block, not the specific distribution data
        data_loc = base_idx + lc_abs - 1
        
        if debug and i < 3:  # Show first 3 energy points only
            logger.debug(f"Energy point {i} ({energies[i]} MeV): LOCC={locc_value} → Tabulated distribution at {data_loc}")
        
        # Check if we have enough data for the header (interp + num_points)
        if data_loc + 2 > len(ace.xss_data):
            error_msg = f"Tabulated distribution data truncated at energy {energies[i]}: header missing"
            if debug:
                logger.error(f"  ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Read interpolation flag
        interp_flag = int(ace.xss_data[data_loc])
        distribution.interpolation.append(interp_flag)
        
        # Read number of points (Np)
        num_points = int(ace.xss_data[data_loc + 1])
        
        if debug and i < 3:  # Show first 3 energy points only
            interp_type = "histogram" if interp_flag == 0 else "linear-linear" if interp_flag == 1 else f"unknown ({interp_flag})"
            logger.debug(f"  Interpolation: {interp_flag} ({interp_type}), Number of points: {num_points}")
        
        # Validate num_points
        if num_points <= 0:
            error_msg = f"Invalid number of points ({num_points}) in tabulated distribution at energy {energies[i]}"
            if debug:
                logger.error(f"  ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Check if we have enough data for the full distribution
        # Need: interp(1) + num_points(1) + cosine(Np) + PDF(Np) + CDF(Np) = 2 + 3*Np
        if data_loc + 2 + 3*num_points > len(ace.xss_data):
            error_msg = f"Tabulated distribution data truncated at energy {energies[i]}: need {2 + 3*num_points} entries, but only {len(ace.xss_data) - data_loc} available"
            if debug:
                logger.error(f"  ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Read cosine grid (Np values)
        cosine_start = data_loc + 2
        cosines = ace.xss_data[cosine_start:cosine_start + num_points]
        distribution._cosine_grid.append(cosines)
        
        # Read PDF values (Np values)
        pdf_start = cosine_start + num_points
        pdfs = ace.xss_data[pdf_start:pdf_start + num_points]
        distribution._pdf.append(pdfs)
        
        # Read CDF values (Np values)
        cdf_start = pdf_start + num_points
        cdfs = ace.xss_data[cdf_start:cdf_start + num_points]
        distribution._cdf.append(cdfs)
        
        if debug and i < 3:  # Show first 3 energy points only
            display_count = min(5, num_points)
            cosine_values = cosines[:display_count].tolist()
            pdf_values = pdfs[:display_count].tolist()
            cdf_values = cdfs[:display_count].tolist()
            logger.debug(f"  First {display_count} points of the tabulated distribution:")
            logger.debug(f"    Cosines: {cosine_values}...")
            logger.debug(f"    PDFs: {pdf_values}...")
            logger.debug(f"    CDFs: {cdf_values}...")
    
    return distribution