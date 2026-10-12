Working with MCTAL Files
========================

This tutorial covers reading MCNP MCTAL files with KIKA: the file header, the
tallies and their bins, totals, perturbations, the tally fluctuation chart and
the KCODE records.

Loading a MCTAL file
--------------------

.. code-block:: python

   import kika

   mctal = kika.read_mctal('path/to/mctalfile.m')
   mctal            # one-line summary per tally

   print(mctal.code, mctal.version, mctal.probid)
   print(f"histories: {mctal.nps:.2e}, title: {mctal.title}")
   print(mctal.tally_numbers)

Every tally type is read: cell and surface tallies, point and ring detectors
(``F5``, which list no cells), radiographs (``FIR``/``FIC``/``FIP``), TMESH mesh
tallies, any particle, and every bin card (``FS``, ``FM``, ``FT``/``FU``, ``C``,
``E``, ``T``, ``CF``/``SF``, with or without total bins).

Tallies
-------

A tally is indexed by its number:

.. code-block:: python

   tally = mctal[4]
   tally            # <Tally 4 (cell flux, n) 'shell Detector': cell=1, energy=3+total>

   tally.kind       # 'cell flux'
   tally.particles  # ['n']
   tally.comment    # the FC card
   tally.energies   # upper energy bounds, or None

MCNP writes a tally as an eight-dimensional array over the bins ``f d u s m c e
t`` (cells/surfaces/detectors, flagged/direct, user, segment, multiplier, cosine,
energy, time). ``tally.result`` and ``tally.error`` hold that array exactly as
written, **including** the total bins. ``tally.axes`` describes each axis.

The labelled view is an xarray Dataset with one named dimension per axis the
tally card binned, without the total bins:

.. code-block:: python

   ds = tally.to_xarray()
   ds.result.sel(energy=1.0)          # energy coordinate = upper bin bound
   ds.energy_low                      # lower bounds (first one is 0)

   tally.to_dataframe()               # one row per bin

Dimension names are ``cell``/``surface``/``detector`` (by tally type), ``flag``,
``user``, ``segment``, ``multiplier``, ``cosine``, ``energy`` and ``time``. An
axis with a single unlabelled bin (the flag axis without ``CF``/``SF``, a single
``FM`` bin) is not a dimension. Radiograph grids appear as ``image_s`` and
``image_t``; a TMESH tally is unfolded into ``cora``, ``corb`` and ``corc``.

Totals
------

The total bins MCNP appends (``et``, ``tt``, ``st``, ...) are read with
:meth:`~kika.mcnp.mctal.Tally.total`, which returns the same kind of Dataset
over the remaining axes:

.. code-block:: python

   tally.total("energy")              # energy-integrated value per cell, segment...
   tally.total("energy", "time")

Perturbations
-------------

PERT results (``vals pert``) have the shape of the unperturbed tally. Pass the
PERT card number to the same views:

.. code-block:: python

   len(tally.perturbations)
   tally.to_xarray(perturbation=174)
   tally.total("energy", perturbation=174)

Tally fluctuation chart
-----------------------

.. code-block:: python

   tally.tfc.to_dataframe()           # nps, result, error, fom
   tally.tfc.plot()                   # result, error and FOM against NPS

KCODE
-----

A criticality run carries the cycle records:

.. code-block:: python

   mctal.kcode.keff                   # final combined keff and its standard deviation
   mctal.kcode.to_dataframe()         # one row per cycle, named columns

Mesh tallies
------------

FMESH results are not written to MCTAL; see :doc:`mesh_tallies`.
