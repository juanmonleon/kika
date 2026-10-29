"""kika.tsl — thermal-scattering (TSL) libraries: materials by compound, in every format.

* :func:`index_tsl` lists a directory of TSL materials by compound, from headers:
  ENDF-6 tapes (NSUB=12), GNDS TNSL files, ACE thermal tables, OpenMC thermal
  HDF5 files and a G4NDL ``ThermalScattering`` folder.
* :func:`read_tsl` reads one material into the model (a ``reactionSuite``
  with the TNSL interaction) and answers σ_el, σ_inel and the outgoing tables
  the same way for every format.
* :func:`read_tsl_header` is what the index reads per ENDF-6 tape;
  :func:`tsl_naming` is how a tape is placed in its compound.
* :func:`write_g4ndl_thermal` writes a processed material (G4NDL, ACE, OpenMC)
  as the G4NDL ``ThermalScattering`` files Geant4 reads.

The same files open through :func:`kika.read`, and the phonon spectra an
evaluation was made from through :mod:`kika.njoy.phonon_spectra`.
"""
from .library import (TSLCompound, TSLLibraryIndex, TSLTape, companion_kind, index_tsl,
                      read_tsl_header)
from .naming import TSLNaming, tsl_naming, tsl_stem
from .g4ndl import write_g4ndl_thermal
from .read import PROCESSED_FORMATS, TSL_FORMATS, TSLMaterial, detect_tsl_format, read_tsl

__all__ = ["TSLCompound", "TSLLibraryIndex", "TSLTape", "TSLNaming", "TSLMaterial",
           "TSL_FORMATS", "PROCESSED_FORMATS", "companion_kind", "detect_tsl_format",
           "index_tsl", "read_tsl", "read_tsl_header", "tsl_naming", "tsl_stem",
           "write_g4ndl_thermal"]
