kika.materials
==============

Material compositions that do not belong to one transport code: nuclides with their
fractions and libraries, read from an MCNP or a Serpent input and written back for either.
The same classes are exported as ``kika.Material``, ``kika.MaterialCollection``,
``kika.Nuclide`` and ``kika.NuclideAccessor``, and by ``kika.mcnp``. Their index entries are
on the page of the MCNP input, which lists the same classes.

.. currentmodule:: kika.materials

Materials
---------

.. autoclass:: Material
   :no-index:

.. autoclass:: MaterialCollection
   :no-index:

.. autoclass:: Nuclide
   :no-index:

.. autoclass:: NuclideAccessor
   :no-index:

Reading them from an input
--------------------------

.. autofunction:: read_material

.. autofunction:: read_serpent_materials
