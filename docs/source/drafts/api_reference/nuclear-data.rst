kika.nuclear_data: photons, spectra and the forward operator
============================================================

Computations on the model that answer physical questions about an evaluation,
whatever format it was read from.

Photon production
-----------------

.. automodule:: kika.nuclear_data.photon_production
   :members:

Energy spectra laws
-------------------

.. automodule:: kika.nuclear_data.spectrum_laws
   :members:

The forward operator
--------------------

.. automodule:: kika.nuclear_data.forward
   :members:

Deprecated classes
------------------

The classes that ``kika.nuclear_data`` exports are deprecated and will be removed in
kika-nd 1.0. The canonical model is ``kika.nuclear_data.model``; each of these classes
is a façade over it. The model does not yet cover reconstruction, group averaging or
covariance collapsing, so there is no drop-in replacement for every use yet.

.. currentmodule:: kika.nuclear_data

.. autoclass:: CrossSection

.. autoclass:: AngularDistribution

.. autoclass:: ResonanceParameters

.. autoclass:: ResonanceRecord

.. autoclass:: LGroup

.. autoclass:: UnresolvedResonanceParameters

.. autoclass:: URR_LGroup

.. autoclass:: URR_JGroup

.. autoclass:: NuclideInfo
