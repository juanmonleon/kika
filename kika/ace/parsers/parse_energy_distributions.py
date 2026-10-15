from typing import List, Optional, Dict
import numpy as np
from kika.ace.classes.ace import Ace
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.container import EnergyDistributionContainer
from kika.ace.parsers.laws import (
    parse_tabular_energy_distribution,
    parse_discrete_energy_distribution,
    parse_level_scattering,
    parse_particle_level_scattering,
    parse_continuous_energy_angle_distribution,
    parse_general_evaporation_spectrum,
    parse_maxwell_fission_spectrum,
    parse_evaporation_spectrum,
    parse_energy_dependent_watt_spectrum,
    parse_tabular_linear_functions,
    parse_tabular_energy_multipliers,
    parse_kalbach_mann_distribution,
    parse_tabulated_angle_energy_distribution,
    parse_nbody_phase_space_distribution,
    parse_laboratory_angle_energy_distribution,
    parse_energy_dependent_yield
)
from kika.ace.classes.read_report import AceReadError, lost, note
from kika.ace.classes.xss import xss_ints
import logging

# Setup logger
logger = logging.getLogger(__name__)

class EnergyDistributionParseError(ValueError):
    """Exception raised for errors in parsing energy distribution data."""
    pass

def read_energy_distribution_blocks(ace: Ace, debug: bool = False) -> EnergyDistributionContainer:
    """
    Read the DLW, DLWP, DLWH, and DNED blocks from the ACE file.
    
    These blocks contain energy distributions for secondary particles:
    - DLW: For incident neutron reactions (secondary neutrons)
    - DLWP: For photon production reactions
    - DLWH: For other particle production reactions
    - DNED: For delayed neutron precursor groups
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS data and headers
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    EnergyDistributionContainer
        Container with the energy distribution data
        
    Raises
    ------
    ValueError
        If required data is missing or indices are invalid
    """

    if debug: 
        logger.debug("\n===== ENERGY DISTRIBUTION BLOCKS PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
        
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None or
        ace.energy_distribution_locators is None):
        raise ValueError("Cannot read energy distribution blocks: header, XSS data, or locators missing")
    
    # Create a new container for the return value
    result = EnergyDistributionContainer()
    
    # Get the JXS values for the energy distribution blocks
    jxs_dlw = ace.header.jxs_array[11]    # JXS(11)
    jxs_dlwp = ace.header.jxs_array[19]   # JXS(19)
    jxs_dned = ace.header.jxs_array[27]   # JXS(27)
    
    if debug:
        logger.debug(f"JXS(11) = {jxs_dlw} → DLW block for incident neutron reactions")
        logger.debug(f"JXS(19) = {jxs_dlwp} → DLWP block for photon production reactions")
        logger.debug(f"JXS(27) = {jxs_dned} → DNED block for delayed neutron precursors")
    
    # Store the container reference to be used by the helper functions
    _process_energy_distributions(ace, result, jxs_dlw, jxs_dlwp, jxs_dned, debug)
    
    return result


def _process_energy_distributions(ace: Ace, result: EnergyDistributionContainer, 
                                 jxs_dlw: int, jxs_dlwp: int, jxs_dned: int, debug: bool = False) -> None:
    """
    Process all energy distribution blocks and fill the result container.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    jxs_dlw : int
        JXS(11) - Starting index of the DLW block 
    jxs_dlwp : int
        JXS(19) - Starting index of the DLWP block
    jxs_dned : int
        JXS(27) - Starting index of the DNED block
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    # 1. Read DLW block (neutron energy distributions)
    if ace.energy_distribution_locators.has_neutron_data and jxs_dlw > 0:
        if debug: logger.debug(f"Reading DLW block starting at index {jxs_dlw}")
        read_dlw_block(ace, result, jxs_dlw, debug=debug)
    elif debug: logger.debug("Skipping DLW block (no neutron data or invalid JXS index)")
    
    # 2. Read DLWP block (photon production energy distributions)
    if ace.energy_distribution_locators.has_photon_production_data and jxs_dlwp > 0:
        if debug: logger.debug(f"Reading DLWP block starting at index {jxs_dlwp}")
        read_dlwp_block(ace, result, jxs_dlwp, debug=debug)
    elif debug: logger.debug("Skipping DLWP block (no photon production data or invalid JXS index)")
    
    # 3. Read DLWH block (other particle production energy distributions)
    if ace.energy_distribution_locators.has_particle_production_data:
        if debug: logger.debug("Reading DLWH block")
        read_dlwh_block(ace, result, debug=debug)
    elif debug: logger.debug("Skipping DLWH block (no particle production data)")
    
    # 4. Read DNED block (delayed neutron energy distributions)
    if ace.energy_distribution_locators.has_delayed_neutron_data and jxs_dned > 0:
        if debug: logger.debug(f"Reading DNED block starting at index {jxs_dned}")
        read_dned_block(ace, result, jxs_dned, debug=debug)
    elif debug: logger.debug("Skipping DNED block (no delayed neutron data or invalid JXS index)")
    
    if debug: logger.debug("Finished read_energy_distribution_blocks")


def _read_laws(ace: Ace, block: str, where: str, jed: int, locator, debug: bool = False) -> List[EnergyDistribution]:
    """The law chain at JED + LOCC - 1 (Table 30), or [] after reporting why not."""
    loc = int(locator)
    if loc <= 0:
        lost(ace, block, where, f"locator {loc} does not point into the block")
        return []
    offset = jed + loc - 1
    if offset >= len(ace.xss_data):
        lost(ace, block, where, f"JED + LOCC - 1 = {offset} is past the XSS array ({len(ace.xss_data)})")
        return []
    report = getattr(ace, "read_report", None)
    if report is not None:
        report.context = (block, where)
    try:
        laws = read_energy_distribution(ace, offset, jed, debug=debug)
    except AceReadError:
        raise
    except (ValueError, IndexError) as e:
        lost(ace, block, where, str(e))
        return []
    finally:
        if report is not None:
            report.context = None
    for law in laws:
        other = sorted({int(i) for i in getattr(law, "interp", ()) or ()} - {1, 2, 22})
        if other and law.law in (4, 22, 24, 44, 61, 67) and len(getattr(law, "incident_energies", ())) > 1:
            note(ace, block, where,
                 f"LAW={law.law} interpolates between incident energies with INT={other}; "
                 f"kika samples it unit-base, as MCNP and OpenMC do")
        grid = np.asarray(getattr(law, "incident_energies", ()), dtype=float)
        repeated = grid[1:][np.diff(grid) == 0]
        if repeated.size:
            note(ace, block, where,
                 f"LAW={law.law} repeats the incident energies {repeated.tolist()} MeV "
                 f"(a jump: the first table of each pair applies at the energy and below it)")
    return laws


def read_dlw_block(ace: Ace, result: EnergyDistributionContainer, jxs_dlw: int, debug: bool = False) -> None:
    """
    Read the DLW block: the secondary-neutron energy laws of each reaction,
    and the energy-dependent yields that TYR points into it (TY > 100).

    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    jxs_dlw : int
        JXS(11) - Starting index of the DLW block in the XSS array
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if not ace.reaction_mt_data or not ace.reaction_mt_data.has_neutron_mt_data:
        return

    neutron_mts = [int(mt) for mt in ace.reaction_mt_data.secondary_neutron_mt]
    ldlw = ace.energy_distribution_locators.incident_neutron
    mt_index = {int(mt): i for i, mt in enumerate(ace.reaction_mt_data.incident_neutron)}

    for mt in neutron_mts:
        i = mt_index.get(mt)
        if i is None or i >= len(ldlw):
            lost(ace, "DLW", f"MT={mt}", f"no LDLW locator (LDLW has {len(ldlw)} entries)")
            continue
        distributions = _read_laws(ace, "DLW", f"MT={mt}", jxs_dlw, ldlw[i], debug)
        if distributions:
            result.incident_neutron[mt] = distributions

    _padding_after_chains(ace, result.incident_neutron, ldlw, jxs_dlw)

    # Energy-dependent yields: KY = JED + |TY| - 101 (Table 52)
    if ace.particle_release and ace.particle_release.has_neutron_data:
        for i, ty in enumerate(ace.particle_release.incident_neutron):
            ty_value = int(ty)
            if abs(ty_value) <= 100 or i >= len(neutron_mts):
                continue
            mt = neutron_mts[i]
            ky = jxs_dlw + abs(ty_value) - 101
            if ky >= len(ace.xss_data):
                lost(ace, "DLW", f"MT={mt} yield", f"KY = {ky} is past the XSS array ({len(ace.xss_data)})")
                continue
            try:
                result.neutron_yields[mt] = parse_energy_dependent_yield(ace, ky)
            except (ValueError, IndexError) as e:
                lost(ace, "DLW", f"MT={mt} yield", str(e))


def _padding_after_chains(ace, chains: dict, locators, jed: int) -> None:
    """Zero words between the end of a chain's last table and the next record of the block.

    NJOY 2016.78 leaves such room after the MT5 of JEFF-4.0. Kept on the last
    law as ``padding_after`` so that the block is written back word for word;
    only the tabulated laws, which know where their data end (``data_end``).
    """
    starts = sorted({int(v) for v in locators if int(v) > 0})
    tys = getattr(getattr(ace, "particle_release", None), "incident_neutron", [])
    starts = sorted(set(starts) | {abs(int(t)) - 100 for t in tys if abs(int(t)) > 100})
    for chain in chains.values():
        last = chain[-1]
        end = getattr(last, "data_end", None)
        if end is None:
            continue
        later = [s for s in starts if s >= end]
        if not later or later[0] == end:
            continue
        gap = later[0] - end
        if not np.any(ace.xss_data[jed + end - 1:jed + end - 1 + gap]):
            last.padding_after = gap


def read_dlwp_block(ace: Ace, result: EnergyDistributionContainer, jxs_dlwp: int, debug: bool = False) -> None:
    """
    Read the DLWP block: the energy laws of each photon production entry of
    MTRP, at JED + LOCC - 1 with JED = JXS(19) and LOCC from LDLWP (Table 30).
    """
    if not ace.reaction_mt_data or not ace.reaction_mt_data.has_photon_production_mt_data:
        return
    photon_mts = ace.reaction_mt_data.photon_production
    locators = ace.energy_distribution_locators.photon_production
    if len(photon_mts) != len(locators):
        lost(ace, "DLWP", "", f"MTRP has {len(photon_mts)} entries but LDLWP {len(locators)}")
    for mt, locator in zip(photon_mts, locators):
        mt = int(mt)
        distributions = _read_laws(ace, "DLWP", f"MT={mt}", jxs_dlwp, locator, debug)
        if distributions:
            result.photon_production[mt] = distributions


def read_dlwh_block(ace: Ace, result: EnergyDistributionContainer, debug: bool = False) -> None:
    """
    Read the DLWH blocks: the energy laws of each produced particle type and
    reaction, at JED + LOCC - 1 with JED = XSS(JXS(32) + 10(i-1) + 8) and LOCC
    from LDLWH (Table 30).

    Energy-dependent yields live only in DLW and DLWP, not here.
    """
    if not ace.reaction_mt_data or not ace.reaction_mt_data.has_particle_production_mt_data:
        return
    jxs31 = ace.header.jxs_array[31]
    jxs32 = ace.header.jxs_array[32]
    if jxs31 <= 0 or jxs32 <= 0:
        lost(ace, "DLWH", "", f"particle production MTs but JXS(31)={jxs31}, JXS(32)={jxs32}")
        return

    n_types = ace.header.num_particle_types
    result.particle_production = [{} for _ in range(n_types)]
    ldlwh = ace.energy_distribution_locators.particle_production
    mtrh = ace.reaction_mt_data.particle_production
    for i in range(n_types):
        locators = ldlwh[i] if i < len(ldlwh) else []
        particle_mts = mtrh[i] if i < len(mtrh) else []
        if len(particle_mts) != len(locators):
            lost(ace, "DLWH", f"particle {i + 1}",
                 f"MTRH has {len(particle_mts)} entries but LDLWH {len(locators)}")
            continue
        if len(particle_mts) == 0:
            continue
        jed_idx = jxs32 + 10 * i + 8
        if jed_idx >= len(ace.xss_data):
            lost(ace, "DLWH", f"particle {i + 1}", f"IXS word {jed_idx} is past the XSS array")
            continue
        jed = int(ace.xss_data[jed_idx])
        for mt, locator in zip(particle_mts, locators):
            mt = int(mt)
            distributions = _read_laws(ace, "DLWH", f"particle {i + 1} MT={mt}", jed, locator, debug)
            if distributions:
                result.particle_production[i][mt] = distributions


def read_dned_block(ace: Ace, result: EnergyDistributionContainer, jxs_dned: int, debug: bool = False) -> None:
    """
    Read the DNED block: the energy laws of each delayed-neutron precursor
    group, at JED + LOCC - 1 with JED = JXS(27) and LOCC from DNEDL (Table 30).

    ``result.delayed_neutron[g]`` is the whole law chain of group g + 1, in the
    BDD order of the groups; a group that cannot be read (with
    ``strict=False``) is an empty list, so the numbering of the others holds.
    """
    if not ace.energy_distribution_locators.has_delayed_neutron_data:
        return
    for g, locator in enumerate(ace.energy_distribution_locators.delayed_neutron):
        result.delayed_neutron.append(
            _read_laws(ace, "DNED", f"group {g + 1}", jxs_dned, locator, debug))


def read_energy_distribution(ace: Ace, offset: int, base_jed: int, debug: bool = False) -> List[EnergyDistribution]:
    """
    Read energy distribution data starting at the given offset in the XSS array.
    
    According to Table 31, each energy distribution consists of:
    - LNW: Locator for the next law (0 if this is the last one)
    - LAW: Law number for the distribution
    - IDAT: Locator for the law data (relative to JED)
    - NR: Number of interpolation regions to define law applicability
    - NBT, INT arrays: ENDF interpolation parameters
    - NE: Number of energies for law applicability
    - E, P arrays: Energy points and probability for law validity
    
    For multiple laws:
    - If LNW=0, this is the last law (always used regardless of other conditions)
    - For a given incident energy E, LAW is used only if random_number < P(E)
    - If E < E(1), use P(1)
    - If E > E(NE), use P(NE)
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    offset : int
        Starting index in the XSS array (JED + LOCC - 1)
    base_jed : int
        Base JED index to be used for absolute position calculations
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    List[EnergyDistribution]
        List of energy distribution objects
        
    Raises
    ------
    EnergyDistributionParseError
        If there are issues with data format or indexing
    """
    distributions = []
    
    if debug:
        logger.debug(f"\n=== READING ENERGY DISTRIBUTION ===")
        logger.debug(f"Starting at offset={offset}, base_jed={base_jed}")
    
    if offset >= len(ace.xss_data):
        raise EnergyDistributionParseError(f"Initial offset {offset} out of bounds for XSS data with length {len(ace.xss_data)}")
    
    current_offset = offset
    law_count = 0
    
    while current_offset < len(ace.xss_data):
        law_count += 1
        if debug:
            logger.debug(f"\nParsing law #{law_count} at offset {current_offset}")
        
        try:
            # Read LNW, LAW, IDAT values
            lnw_entry = ace.xss_data[current_offset]
            law_entry = ace.xss_data[current_offset + 1]
            idat_entry = ace.xss_data[current_offset + 2]
            
            lnw = int(lnw_entry)
            law = int(law_entry)
            idat = int(idat_entry)
            
            if debug:
                logger.debug(f"Table 31 values: LNW={lnw}, LAW={law}, IDAT={idat}")
                logger.debug(f"Read from indices: {current_offset}, {current_offset+1}, {current_offset+2}")
        except (IndexError, AttributeError, ValueError) as e:
            raise EnergyDistributionParseError(f"Failed to read LNW/LAW/IDAT at offset {current_offset}: {str(e)}")
        
        if law not in _LAW_PARSERS:
            raise EnergyDistributionParseError(
                f"LAW={law} at {current_offset + 1} is not an ACE energy law "
                f"(known: {sorted(_LAW_PARSERS)})")

        try:
            # Read NR (number of interpolation regions)
            n_r_entry = ace.xss_data[current_offset + 3]
            n_r = int(n_r_entry)
            if debug:
                logger.debug(f"NR={n_r} (number of interpolation regions)")
                
            idx = current_offset + 4
        except (IndexError, AttributeError, ValueError) as e:
            raise EnergyDistributionParseError(f"Failed to read NR at offset {current_offset + 3}: {str(e)}")
        
        # Read NBT, INT arrays if present
        nbt = []
        interp = []
        
        if n_r > 0:
            if idx + 2*n_r - 1 >= len(ace.xss_data):
                raise EnergyDistributionParseError(
                    f"Not enough data for NBT/INT arrays. Need indices up to {idx + 2*n_r - 1}, have {len(ace.xss_data)}."
                )
            
            nbt = xss_ints(ace.xss_data, idx, n_r)
            idx += n_r
            interp = xss_ints(ace.xss_data, idx, n_r)
            idx += n_r
        
        # Read NE (number of energies for applicability)
        try:
            if idx >= len(ace.xss_data):
                raise EnergyDistributionParseError(f"Index {idx} out of bounds for XSS data when reading NE")
                
            n_e_entry = ace.xss_data[idx]
            n_e = int(n_e_entry)
            idx += 1
            
            if debug:
                logger.debug(f"NE={n_e} (number of energies for applicability)")
        except (IndexError, AttributeError, ValueError) as e:
            raise EnergyDistributionParseError(f"Failed to read NE at index {idx-1}: {str(e)}")
        
        # Read E, P arrays
        energies = []
        probabilities = []
        
        if n_e > 0:
            if idx + 2*n_e - 1 >= len(ace.xss_data):
                raise EnergyDistributionParseError(
                    f"Not enough data for E/P arrays. Need indices up to {idx + 2*n_e - 1}, have {len(ace.xss_data)}."
                )
            
            try:
                energies = ace.xss_data[idx:idx + n_e]
                idx += n_e
                probabilities = ace.xss_data[idx:idx + n_e]
                idx += n_e
                
                if debug and n_e > 0:
                    energy_values = energies.tolist()
                    prob_values = probabilities.tolist()
                    logger.debug(f"E range: [{energy_values[0]:.5e} - {energy_values[-1]:.5e}]")
                    logger.debug(f"P range: [{prob_values[0]:.5e} - {prob_values[-1]:.5e}]")
            except (IndexError, AttributeError) as e:
                raise EnergyDistributionParseError(f"Failed to read E/P arrays: {str(e)}")
        
        # Calculate absolute position using the base JED
        idat_absolute = base_jed + idat - 1
        
        if debug: 
            logger.debug(f"IDAT_absolute = base_jed + idat - 1 = {base_jed} + {idat} - 1 = {idat_absolute}")
            if idat_absolute < len(ace.xss_data):
                logger.debug(f"Value at IDAT_absolute: {ace.xss_data[idat_absolute]}")
        
        if idat_absolute >= len(ace.xss_data):
            raise EnergyDistributionParseError(
                f"IDAT_absolute={idat_absolute} out of bounds for XSS data with length {len(ace.xss_data)}"
            )
        
        # Create the energy distribution object
        distribution = create_energy_distribution(
            ace, law, idat, idat_absolute, 
            energies, probabilities, nbt, interp,
            debug=debug
        )
        
        if distribution:
            # Add relevant JXS values to help Law 61 parser
            if hasattr(distribution, 'jxs_dlw'):
                distribution.jxs_dlw = base_jed
            
            distributions.append(distribution)
            if debug: logger.debug(f"Successfully created distribution for law {law}")
        else:
            if debug: logger.debug(f"No distribution created for law {law}")
        
        # Check if this is the last law
        if lnw == 0:
            if debug: logger.debug("Reached last law (LNW=0)")
            break
            
        # Calculate position of next law using base_jed
        next_offset = base_jed + lnw - 1
        
        if debug:  
            logger.debug(f"Next law at: base_jed + lnw - 1 = {base_jed} + {lnw} - 1 = {next_offset}")
        
        if next_offset <= current_offset:
            raise EnergyDistributionParseError(
                f"Invalid next law offset {next_offset} <= current offset {current_offset}. Possible infinite loop."
            )
            
        if next_offset >= len(ace.xss_data):
            raise EnergyDistributionParseError(
                f"Next law offset {next_offset} out of bounds for XSS data with length {len(ace.xss_data)}"
            )
        
        current_offset = next_offset
    
    if debug: 
        logger.debug(f"\nFinished reading distributions. Found {len(distributions)} laws")
    
    return distributions

def create_energy_distribution(
    ace: Ace, law: int, idat: int, idat_absolute: int, 
    applicability_energies: np.ndarray, 
    applicability_probabilities: np.ndarray,
    nbt: List[int], interp: List[int],
    debug: bool = False
) -> Optional[EnergyDistribution]:
    """
    Create an energy distribution object based on the law number.
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    law : int
        Law number for the distribution
    idat : int
        Locator for the distribution data (relative to JED)
    idat_absolute : int
        Absolute index in XSS array for the law data 
    applicability_energies : np.ndarray
        Energy points for law applicability (view of xss_data)
    applicability_probabilities : np.ndarray
        Probability of law validity at each energy point
    nbt : List[int]
        NBT interpolation parameters
    interp : List[int]
        INT interpolation scheme parameters
    debug: bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    Optional[EnergyDistribution]
        Energy distribution object, or None if the law is not supported
        
    Raises
    ------
    EnergyDistributionParseError
        If there are issues with the parsing process
    """
    if debug: 
        logger.debug(f"\n--- Creating energy distribution for law {law} ---")
        logger.debug(f"IDAT={idat}, IDAT_absolute={idat_absolute}")
    
    # Create base distribution object
    base_distribution = EnergyDistribution(
        law=law, 
        idat=idat
    )
    base_distribution.applicability_energies = applicability_energies
    base_distribution.applicability_probabilities = applicability_probabilities
    base_distribution.applicability_nbt = nbt
    base_distribution.applicability_interp = interp
    

    # Store the actual JED value - this should always be used
    base_distribution.jed = idat_absolute - idat + 1  # This is the actual JED value

    ## Store the JXS values for Law 61
    #try:
    #    base_distribution.jxs_dlw = ace.header.jxs_array[11]  # JXS(11) for neutron reactions
    #    base_distribution.jxs_dlwp = ace.header.jxs_array[19]  # JXS(19) for photon-producing reactions
    #except (IndexError, AttributeError) as e:
    #    if debug:
    #        logger.debug(f"Warning: Could not read JXS values for distribution: {str(e)}")
    
    parser = _LAW_PARSERS.get(law)
    if parser is None:
        raise EnergyDistributionParseError(f"LAW={law} is not an ACE energy law")
    try:
        return parser(ace, base_distribution, idat_absolute, debug=debug)
    except Exception as e:
        raise EnergyDistributionParseError(f"Error parsing law {law} at absolute index {idat_absolute}: {str(e)}")


#: ACE energy laws (manual Table 32) and their parsers.
_LAW_PARSERS = {
    1: parse_tabular_energy_distribution,
    2: parse_discrete_energy_distribution,
    3: parse_level_scattering,
    4: parse_continuous_energy_angle_distribution,
    5: parse_general_evaporation_spectrum,
    7: parse_maxwell_fission_spectrum,
    9: parse_evaporation_spectrum,
    11: parse_energy_dependent_watt_spectrum,
    22: parse_tabular_linear_functions,
    24: parse_tabular_energy_multipliers,
    33: parse_particle_level_scattering,
    44: parse_kalbach_mann_distribution,
    61: parse_tabulated_angle_energy_distribution,
    66: parse_nbody_phase_space_distribution,
    67: parse_laboratory_angle_energy_distribution,
}
