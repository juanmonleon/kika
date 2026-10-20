kika.processing
===============

kika's own processing of an evaluation: resonance reconstruction, Doppler
broadening (its own page), unresolved-resonance probability tables, heating,
thermal scattering tables, and the tabulation of secondary distributions for
Monte Carlo. The staged entry point for an ENDF-6 tape is
:func:`kika.endf.process_endf`.

.. automodule:: kika.processing
   :members:
   :exclude-members: ReconstructionOptions, UnsupportedResonanceError, ReconstructionConvergenceError, REJECTION_CATEGORIES, ENGINE_VERSION, native_available, resonance_group_average, resonance_window_average, njoy_reconstruct, njoy_reconstruct_stream, NjoyReconstructError, DEFAULT_PENDF_CACHE_DIR, mf33_needs_pendf, get_or_create_pendf, read_pendf_mf3_sections, NjoyNotFoundError, attach_pendf, find_njoy_executable

Resonance reconstruction
------------------------

The resonance-reconstruction API has its own page,
:doc:`processing-resonances`.

Unresolved-resonance probability tables
---------------------------------------

.. automodule:: kika.processing.resonances.probability_tables
   :members:

Tolerances of the processing chain
----------------------------------

NJOY's thermal rule and the shares of a heated table's tolerance; the method is
in :doc:`/drafts/methods/processing/processing-chain`.

.. autofunction:: kika.processing.tolerances.thermalFactor

.. autofunction:: kika.processing.tolerances.heatedShares

Heating and damage
------------------

.. automodule:: kika.processing.heating
   :members:

Thermal scattering tables
-------------------------

.. automodule:: kika.processing.thermal
   :members:

Tabulation of secondary distributions
-------------------------------------

.. automodule:: kika.processing.tabulation
   :members:

Derived covariances
-------------------

.. automodule:: kika.processing.derived_covariance
   :members: resolve_derived_covariance

Width-fluctuation integrals
---------------------------

.. automodule:: kika.processing.resonances.fluctuations
   :members: width_products

Group averages of reconstructed resonances
------------------------------------------

.. automodule:: kika.processing.group_averaging
   :members: resonance_group_average, resonance_window_average

Reconstruction and PENDF through NJOY
-------------------------------------

.. automodule:: kika.processing.njoy_reconstruct
   :members: njoy_reconstruct, njoy_reconstruct_stream, NjoyReconstructError, NjoyDecimalLocaleReconstructError

.. automodule:: kika.processing.njoy_pendf_cache
   :members: get_or_create_pendf, mf33_needs_pendf, read_pendf_mf3_sections, find_njoy_executable, attach_pendf, NjoyNotFoundError

Photo-atomic tables
-------------------

.. automodule:: kika.ace.processing.photoatomic
   :members: photoatomicTable, PhotoatomicNotes, comptonShells, ComptonShell, bmmProfiles, unionGrid, refinedGrid
   :exclude-members: PhotoatomicProcessing, process_photoatomic, ELECTRON_REST_ENERGY

.. automodule:: kika.ace.processing.photoatomic.tables
   :members: LOGLOG_RTOL, ELECTRON_REST_ENERGY
   :no-index:

.. automodule:: kika.ace.processing.photoatomic.compton
   :no-members:

.. automodule:: kika.ace.processing.photoatomic.electrons
   :members: electronBlocks

