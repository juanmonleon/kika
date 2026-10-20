kika.ace
========

The ace module provides functionality for parsing ACE (A Compact ENDF) format nuclear data files.

.. automodule:: kika.ace
   :members:
   :undoc-members:
   :show-inheritance:

Submodules
----------

kika.ace.parsers.parse_ace
---------------------------

.. automodule:: kika.ace.parsers.parse_ace
   :members:
   :undoc-members:
   :show-inheritance:

kika.ace.model_adapter
----------------------

.. automodule:: kika.ace.model_adapter
   :members: decodeAce, encodeAce, AceEncodeError, aceLabels, aceStyles

kika.ace.xsdir
---------------

.. automodule:: kika.ace.xsdir
   :members:
   :undoc-members:
   :show-inheritance:

kika.ace.writers.write_ace
--------------------------

.. automodule:: kika.ace.writers.write_ace
   :members: write_ace, write_ace_checked

Dosimetry tables
----------------

.. automodule:: kika.ace.classes.dosimetry
   :members: DosimetryAce, DosimetryReaction, production_mt, split_production_mt

.. automodule:: kika.ace.parsers.parse_dosimetry
   :members: parse_dosimetry, is_dosimetry_extension

.. automodule:: kika.ace.encoders.dosimetry
   :members: encode_dosimetry

.. automodule:: kika.ace.model_adapter.dosimetry
   :members: dosimetryReactions, pointwiseDosimetryReactions
