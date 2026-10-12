Working with Mesh Tallies (FMESH)
=================================

MCNP writes FMESH results outside the MCTAL file, in one of two places:

* the **MESHTAL text file** (``meshtal``, ``*.msht``), for every MCNP version up
  to 6.3 with ``out=col`` (the default), ``cf``, ``colsci`` or ``cfsci``;
* the **HDF5 runtape** (``runtpe.h5``, ``<name>r.h5``), under
  ``/results/mesh_tally``, for MCNP 6.3 and later with ``out=xdmf``. MCNP 6.3
  deprecates every other output format, so this is the one to use from 6.3 on.

:func:`kika.read_meshtal` reads both into the same objects:

.. code-block:: python

   import kika

   meshtal = kika.read_meshtal('meshtal')        # or 'runtpe.h5'
   meshtal                                        # one line per mesh tally

   tally = meshtal[4]
   tally.geometry                                 # 'rectangular' or 'cylindrical'
   tally.particle, tally.comment
   tally.bounds                                   # edges of x y z, or r z theta

The data
--------

``tally.result`` and ``tally.error`` are arrays over ``(energy, time, i, j, k)``,
where ``i j k`` are ``x y z`` or ``r z theta`` as on the FMESH card. When there
is more than one energy (or time) bin MCNP also writes the total over them; it
is kept as the last index and read with :meth:`~kika.mcnp.meshtal.MeshTally.total`:

.. code-block:: python

   ds = tally.to_xarray()                         # bins only, labelled
   ds.result.sel(r=6.25, z=7.0, theta=0.625)      # spatial coordinates = bin centres
   ds.r_low, ds.r_high                            # bin edges

   tally.total("energy")                          # energy-integrated mesh
   tally.total("energy", "time")

   tally.volumes                                  # mesh cell volumes (cm3)

The ``energy`` and ``time`` dimensions appear only when there is more than one
bin. Theta is in revolutions (0 to 1) whichever file it came from.

What is not read
----------------

The matrix layouts (``out=ij``/``ik``/``jk``) repeat the same numbers with the
relative error cut to five decimals; use a column format or ``out=xdmf``
instead. Cell-under-voxel and nuclide-binned (D1SUNED) meshes are not read
either. Each of these raises ``NotImplementedError`` naming the layout.

The MESHTAL header prints spatial bounds with two decimals and energy/time
bounds with three significant digits, so ``bounds`` read from text carry that
rounding; the runtape keeps them in full precision.
