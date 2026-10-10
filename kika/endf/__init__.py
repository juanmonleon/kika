"""
ENDF module for reading and working with Evaluated Nuclear Data Files.

This module provides functionality for:
- Reading local ENDF files
- Downloading ENDF files from IAEA Nuclear Data Service
- Caching downloaded files locally
"""
from .read_endf import (
    read_endf,
    read_mt451,
    read_mf2,
    read_mf3_mt,
    read_mf4_mt,
    read_mf7_mt,
)
from .lazy import LazyFiles, TapeChangedError, open_endf
from .classes.mf7.scatterer import ThermalScatterer, thermal_scatterer
from . import dcs
from .checks import check_covariance_library, check_covariances
from .inventory import TapeInventory, tape_inventory
from .reconstruction import EndfReconstruction, ReconstructedTable, reconstruct_endf
from .remote import (
    fetch_endf,
    download_endf,
    list_available_libraries,
    get_cache_info,
    clear_cache,
    ENDFRemoteError,
    IsotopeNotFoundError,
    LibraryNotFoundError,
    NetworkError,
    CacheError,
)

__all__ = [
    # Differential cross sections from MF4 + MF3 (angular reconstruction, the
    # elastic frame transform, and the three readings of sigma(E))
    "dcs",
    # Layer-1 checks of the covariance files as written
    "check_covariances",
    "check_covariance_library",
    # The sections of a tape from its MT451 directory, without parsing it
    "tape_inventory",
    "TapeInventory",
    # Resonance reconstruction with kika's engine (MF2 -> pointwise sigma, PENDF)
    "reconstruct_endf",
    "EndfReconstruction",
    "ReconstructedTable",
    # Local file reading
    "read_endf",
    "open_endf",
    "LazyFiles",
    "TapeChangedError",
    "read_mt451",
    "read_mf2",
    "read_mf3_mt",
    "read_mf4_mt",
    "read_mf7_mt",
    # Thermal scattering identity
    "ThermalScatterer",
    "thermal_scatterer",
    # Remote download
    "fetch_endf",
    "download_endf",
    "list_available_libraries",
    "get_cache_info",
    "clear_cache",
    # Exceptions
    "ENDFRemoteError",
    "IsotopeNotFoundError",
    "LibraryNotFoundError",
    "NetworkError",
    "CacheError",
]
