kika.g4ndl
==========

G4NDL, the neutron data format that the ParticleHP models of Geant4 read. A library is
read into the model and written from it through :func:`kika.read` and :func:`kika.write`
with ``format="g4ndl"``. This package adds the library as an object, the replacement of
one isotope in a copy of a library, and summaries of what a suite read from one holds.

.. currentmodule:: kika.g4ndl

Opening a library
-----------------

.. autofunction:: open

.. autoclass:: G4NDLLibrary

.. autoclass:: IsotopeKey

Replacing an isotope
--------------------

.. autofunction:: patch_isotope

.. autofunction:: patch_elastic

.. autoclass:: PatchResult

Summaries of a suite
--------------------

.. autofunction:: isotopeSummary

.. autofunction:: crossSections

.. autofunction:: angularMTs

.. autofunction:: angularBulk

.. autofunction:: inelasticTotal

.. autofunction:: fissionSummary

Errors
------

.. autoexception:: G4NDLError

.. autoexception:: G4NDLFormatError

.. autoexception:: G4NDLUnsupportedError

.. autoexception:: IsotopeNotFoundError
