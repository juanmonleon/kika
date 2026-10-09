"""The ENDF writer door for an evaluation that is a standalone PoPs.

A radioactive decay (NSUB=4) or fission yield (NSUB=5, 11) evaluation is a
nuclide, not a reactionSuite (roadmap E7). ``kika.write`` reaches its tape
through here, as it reaches a reactionSuite's through :mod:`.assemble`, so
that only the front door imports the model adapter.
"""
from __future__ import annotations

__all__ = ["writeSublibraryTape"]


def writeSublibraryTape(pops, path, mat=None, tapeId=None, report=None):
    """Write a decay or fission-yield PoPs as its ENDF-6 sublibrary tape."""
    from kika.endf.model_adapter.decay_sublibrary import writeSublibraryTape as write

    return write(pops, path, mat=mat, tapeId=tapeId, report=report)
