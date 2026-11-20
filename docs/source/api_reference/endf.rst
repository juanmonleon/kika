kika.endf
=========

The endf module provides functionality for reading and working with Evaluated Nuclear Data Files (ENDF).

.. automodule:: kika.endf
   :members:
   :undoc-members:
   :show-inheritance:

Submodules
----------

kika.endf.read_endf
--------------------

The functions of this module are listed with the package above.

.. automodule:: kika.endf.read_endf
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

kika.endf.remote
----------------

Evaluations downloaded from the IAEA Nuclear Data Services, and cached.
:func:`kika.endf.download_endf` and :func:`kika.endf.fetch_endf` are listed with the
package above.

.. autofunction:: kika.endf.remote.get_catalog

.. autofunction:: kika.endf.remote.check_access

kika.endf.utils
----------------

.. automodule:: kika.endf.utils
   :members:
   :undoc-members:
   :show-inheritance:

What the readers return
-----------------------

.. autoclass:: kika.endf.classes.endf.ENDF
   :no-members:

.. autoclass:: kika.endf.classes.mf1.mf1mt451.MF1MT451
   :no-members:

.. autoclass:: kika.endf.classes.mf2.mf2mt151.MF2MT151
   :no-members:

.. autoclass:: kika.endf.classes.mf3.mf3mt.MF3MT
   :no-members:

.. autoclass:: kika.endf.classes.mf4.base.MF4MT
   :no-members:

.. autoclass:: kika.endf.classes.mf7.composition.MF7MT451
   :no-members:

.. autoclass:: kika.endf.classes.mf7.elastic.MF7MT2
   :no-members:

.. autoclass:: kika.endf.classes.mf7.inelastic.MF7MT4
   :no-members:

Errors of the remote download
-----------------------------

.. autoexception:: kika.endf.remote.exceptions.ENDFRemoteError

.. autoexception:: kika.endf.remote.exceptions.LibraryNotFoundError

.. autoexception:: kika.endf.remote.exceptions.NetworkError

.. autoexception:: kika.endf.remote.exceptions.AccessBlockedError

.. autoexception:: kika.endf.remote.exceptions.CacheError
