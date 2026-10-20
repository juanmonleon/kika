"""Photonuclear tables: an evaluation of photons on a nucleus → the contents of a ``.u`` ACE table.

:func:`photonuclearTable` takes the model of an ENDF photonuclear evaluation
(sublibrary ``g``) and returns a :class:`~kika.ace.classes.photonuclear.PhotonuclearAce`
ready for :func:`kika.ace.encoders.photonuclear.encode_photonuclear`. The method
and its choices are described in :mod:`.tables`.
"""
from .tables import photonuclearTable, photonTwoBodyEnergy

__all__ = ["photonuclearTable", "photonTwoBodyEnergy"]
