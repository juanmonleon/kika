kika.materials
==============

Material compositions that do not belong to one transport code: nuclides with their
fractions and libraries, read from an MCNP or a Serpent input or an OpenMC ``materials.xml``
and written for any of the three.
The same classes are exported as ``kika.Material``, ``kika.MaterialCollection``,
``kika.Nuclide`` and ``kika.NuclideAccessor``, and by ``kika.mcnp``.

.. currentmodule:: kika.materials

Materials
---------

.. autoclass:: Material

.. autoclass:: MaterialCollection

.. autoclass:: Nuclide

.. autoclass:: NuclideAccessor

Reading them from an input
--------------------------

.. autofunction:: read_material

.. autofunction:: read_serpent_materials

.. autofunction:: read_openmc_materials

Writing them for a code
-----------------------

.. autoclass:: kika.materials.report.EmissionReport

.. autofunction:: kika.materials.report.read_library_tables
