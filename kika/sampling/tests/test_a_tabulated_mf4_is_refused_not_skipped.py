"""The ENDF path refuses a tabulated MF4 rather than skipping it.

``apply_perturbation_factors_to_endf`` only knows how to scale Legendre
coefficients. It used to filter MF4 sections on ``MF4MTLegendre`` /
``MF4MTMixed`` and walk past anything else, so a request with MF34 components
for an LTT=2 reaction wrote a sample with that MT exactly as evaluated and
reported success. The applier for tables lives on the model path; here the only
change is that the miss is loud.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf.classes.endf import ENDF
from kika.endf.classes.mf import MF
from kika.endf.classes.mf4.polynomial import MF4MTLegendre
from kika.endf.classes.mf4.tabulated import MF4MTTabulated
from kika.sampling.endf_perturbation import apply_perturbation_factors_to_endf

ISO = 26056


def _tabulated(mt: int) -> MF4MTTabulated:
    mu = [-1.0, 0.0, 1.0]
    return MF4MTTabulated(number=mt, _energies=[1.0e6, 2.0e6],
                          _cosines=[mu, mu],
                          _probabilities=[[0.4, 0.5, 0.6], [0.3, 0.5, 0.7]],
                          _angular_interpolation=[[(3, 2)], [(3, 2)]],
                          _interpolation=[(2, 2)])


def _endf(*sections) -> ENDF:
    mf4 = MF(number=4)
    for section in sections:
        mf4.add_section(section)
    return ENDF(files={4: mf4})


def test_a_tabulated_section_with_mf34_components_is_refused():
    endf = _endf(_tabulated(2))
    with pytest.raises(ValueError, match="LTT=2"):
        apply_perturbation_factors_to_endf(
            endf, np.array([1.1]), 0, {(ISO, 2, 1): [1.0e5, 1.0e7]},
            [(ISO, 2, 1, 0)], verbose=False)


def test_a_tabulated_section_the_request_does_not_name_is_left_alone():
    """Only a requested MT is a miss; a table nobody asked about is not."""
    legendre = MF4MTLegendre(number=2, _energies=[1.0e6, 2.0e6],
                             _legendre_coeffs=[[0.5, 0.2], [0.4, 0.1]])
    endf = _endf(legendre, _tabulated(51))
    perturbed, _events = apply_perturbation_factors_to_endf(
        endf, np.array([1.1]), 0, {(ISO, 2, 1): [1.0e5, 1.0e7]},
        [(ISO, 2, 1, 0)], verbose=False)
    assert perturbed == [(ISO, 2, 1, 0)]
    assert endf.get_file(4).sections[51].probabilities[0] == [0.4, 0.5, 0.6]
