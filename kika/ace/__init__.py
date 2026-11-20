"""ACE files: reading, decoding into the model, encoding back, and writing.

:func:`read_ace` reads an ACE table into an object that keeps its blocks and the XSS
array. The model adapter turns a table into a reaction suite and back, the writers
and the xsdir functions put a table on disk and in an MCNP xsdir, and the
dosimetry, photonuclear and photo-atomic tables have their own readers and builders.
"""
from .parsers.parse_ace import read_ace

__all__ = [
    'read_ace',
]
