"""
ENDF-side helpers for resonance processing.

Public API
----------
detect_resonance_bounds
    The resolved/unresolved energy bounds declared in MF2.

Resonance reconstruction lives in :mod:`kika.processing.resonances` (on the
model) and :mod:`kika.processing.njoy_reconstruct` (NJOY RECONR).
"""

from .resonance_bounds import detect_resonance_bounds

__all__ = ["detect_resonance_bounds"]
