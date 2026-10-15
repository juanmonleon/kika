kika.hdf5
=========

OpenMC's HDF5 nuclear data: reading, writing, the processing of an evaluation
into the processed suite OpenMC's files hold, the contract check, the
comparison with another file, and ``cross_sections.xml``.

Reading and writing
-------------------

.. automodule:: kika.hdf5.api
   :members: read, write, filetype

Libraries and cross_sections.xml
--------------------------------

.. automodule:: kika.hdf5.library
   :members:

Processing an evaluation
------------------------

.. automodule:: kika.hdf5.processing
   :members:

.. automodule:: kika.hdf5.thermal_processing
   :members:

Fission energy release
----------------------

.. automodule:: kika.hdf5.fission_energy
   :members:

Checking and comparing files
----------------------------

.. automodule:: kika.hdf5.contract
   :members:

.. automodule:: kika.hdf5.compare
   :members:

The HDF5 tree
-------------

.. automodule:: kika.hdf5.records
   :members:
