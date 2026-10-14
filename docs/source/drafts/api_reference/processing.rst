kika.processing
===============

kika's own processing of an evaluation: resonance reconstruction, Doppler
broadening (its own page), unresolved-resonance probability tables, heating,
thermal scattering tables, and the tabulation of secondary distributions for
Monte Carlo. The staged entry point for an ENDF-6 tape is
:func:`kika.endf.process_endf`.

.. automodule:: kika.processing
   :members:
   :exclude-members: ReconstructionOptions, UnsupportedResonanceError, ReconstructionConvergenceError, REJECTION_CATEGORIES, ENGINE_VERSION, native_available

Resonance reconstruction
------------------------

.. automodule:: kika.processing.resonances
   :members:
   :exclude-members: ReactionKey

Unresolved-resonance probability tables
---------------------------------------

.. automodule:: kika.processing.resonances.probability_tables
   :members:

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
