"""
High-level functions for reading ENDF files with selective parsing.
"""
import os
from typing import List, Optional, Union

from .classes.endf import ENDF
from .parsers.parse_endf import (
    parse_endf_file,
    parse_mf_from_file,
    scan_mat_number,
    scan_tape_id,
    MF_PARSERS,
)
from .classes.mf1.mf1mt451 import MF1MT451
from .classes.mf2.mf2mt151 import MF2MT151
from .classes.mf3.mf3mt import MF3MT
from .classes.mf4.base import MF4MT


def read_endf(filepath: str, mf_numbers: Optional[Union[int, List[int]]] = None) -> ENDF:
    """Read and parse an ENDF file, optionally selecting specific MF sections.

    Parameters
    ----------
    filepath : str
        Path to the ENDF-6 tape.
    mf_numbers : int or list of int, optional
        The MF numbers to parse. None parses every MF that has a registered
        parser and skips the others with a warning. A listed MF that the tape
        does not contain is left out of the result, with a warning.

    Returns
    -------
    ENDF
        The parsed tape, with ``source_path`` set to the absolute path of
        ``filepath`` and, for a selective read as for a full one, the MAT and
        tape identification of the tape.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    ValueError
        If a listed MF has no registered parser.

    Notes
    -----
    Parsers are available for MF1-10, MF12-15, MF23, MF26, MF27, MF28, MF31-35
    and MF40; the registry is ``kika.endf.parsers.parse_endf.MF_PARSERS``.

    A full read parses all of them, whatever the caller then uses. To parse each
    MF only when it is first accessed, use :func:`kika.endf.open_endf`, which
    returns the same ``ENDF`` with the same MF objects (JENDL-5 U-238: 4.35 s
    here, 0.08 s to open there and 0.015 s more for MF3).

    Examples
    --------
    # Parse all MF sections with registered parsers
    endf = read_endf("path/to/file")

    # Parse only MF1 and MF4
    endf = read_endf("path/to/file", mf_numbers=[1, 4])

    # Parse only MF4
    endf = read_endf("path/to/file", mf_numbers=4)
    """
    # Normalize mf_numbers input
    if mf_numbers is not None and isinstance(mf_numbers, int):
        mf_numbers = [mf_numbers]
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ENDF file not found: {filepath}")
        
    # If no filtering, parse entire file with all available parsers
    if mf_numbers is None:
        endf = parse_endf_file(filepath)
        endf.source_path = os.path.abspath(filepath)
        return endf
    
    # Create empty ENDF object
    endf = ENDF()
    endf.source_path = os.path.abspath(filepath)

    # MAT is a property of the tape, not of the sections asked for, so a
    # targeted parse must report the same one a full parse does. It did not:
    # only parse_endf_file set it, so read_endf(f, mf_numbers=[1]).zaid was
    # None while read_endf(f).zaid was 26056 — and kika/sampling built its
    # output directory from the former, writing perturbed samples under
    # endf/unknown/ where the pairing stage looked under endf/26056/.
    with open(filepath, 'r') as f:
        lines = f.readlines()
    endf.mat = scan_mat_number(lines)
    endf.tape_id = scan_tape_id(lines)

    # Parse each requested MF section
    for mf_number in mf_numbers:
        mf = parse_mf_from_file(filepath, mf_number)
        if mf is None:
            continue

        # Add the MF to the ENDF object
        endf.add_file(mf)

    return endf


def read_mt451(filepath: str) -> Optional['MF1MT451']:
    """Read only the MF1/MT451 (general information) section of an ENDF file.

    The whole of MF1 is parsed (the nubar sections included); only MF1 is.

    Parameters
    ----------
    filepath : str
        Path to the ENDF-6 tape.

    Returns
    -------
    MF1MT451 or None
        The section, or None when the tape has no MF1 (a warning is issued) or
        no MT451 in it.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    """
    # Parse MF1
    mf1 = parse_mf_from_file(filepath, 1)
    if mf1 and 451 in mf1.sections:
        return mf1.sections[451]
    return None


def read_mf3_mt(filepath: str, mt_number: int) -> Optional['MF3MT']:
    """Read one MT section of MF3 (reaction cross sections) of an ENDF file.

    The whole of MF3 is parsed; only MF3 is.

    Parameters
    ----------
    filepath : str
        Path to the ENDF-6 tape.
    mt_number : int
        MT of the section (1 total, 2 elastic, 18 fission, ...).

    Returns
    -------
    MF3MT or None
        The section, or None when the tape has no MF3 (a warning is issued) or
        no such MT in it.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    """
    mf3 = parse_mf_from_file(filepath, 3)
    if mf3 and mt_number in mf3.sections:
        return mf3.sections[mt_number]
    return None


def read_mf2(filepath: str) -> Optional['MF2MT151']:
    """Read the MF2/MT151 (resonance parameters) section of an ENDF file.

    Parameters
    ----------
    filepath : str
        Path to the ENDF-6 tape.

    Returns
    -------
    MF2MT151 or None
        The section, or None when the tape has no MF2 (a warning is issued) or
        no MT151 in it.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    """
    mf2 = parse_mf_from_file(filepath, 2)
    if mf2 and 151 in mf2.sections:
        return mf2.sections[151]
    return None


def read_mf4_mt(filepath: str, mt_number: int) -> Optional['MF4MT']:
    """Read one MT section of MF4 (angular distributions) of an ENDF file.

    The whole of MF4 is parsed; only MF4 is.

    Parameters
    ----------
    filepath : str
        Path to the ENDF-6 tape.
    mt_number : int
        MT of the section (2 elastic, 51 to 91 inelastic levels and continuum).

    Returns
    -------
    MF4MT or None
        The section, or None when the tape has no MF4 (a warning is issued) or
        no such MT in it.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    """
    # Parse MF4
    mf4 = parse_mf_from_file(filepath, 4)
    if mf4 and mt_number in mf4.sections:
        return mf4.sections[mt_number]
    return None


def read_mf7_mt(filepath: str, mt_number: int):
    """Read one MT section of MF7 (thermal scattering law) of an ENDF file.

    The whole of MF7 is parsed; only MF7 is.

    Parameters
    ----------
    filepath : str
        Path to the ENDF-6 tape, a thermal-scattering one.
    mt_number : int
        2 (elastic), 4 (incoherent inelastic) or 451 (composition).

    Returns
    -------
    MF7MT2, MF7MT4, MF7MT451 or None
        The section, or None when the tape has no MF7 (a warning is issued) or
        no such MT in it.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.

    Notes
    -----
    MF7 lives only in the thermal scattering sublibrary (NSUB=12): no neutron
    evaluation in ENDF/B-VIII.1 carries one. These tapes are large: the MF7/MT4
    of ``tsl-HinH2O.endf`` is 1.14 million records and takes about 35 s to
    decode.
    """
    mf7 = parse_mf_from_file(filepath, 7)
    if mf7 and mt_number in mf7.sections:
        return mf7.sections[mt_number]
    return None
