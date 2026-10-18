"""A product an MF4/MF5 section leaves without a multiplicity takes the count its MT fixes.

ENDF writes no multiplicity for the neutron of an MF4/MF5 section: (n,2n) emits two
because it is MT16. The processing wrote 1 for every such neutron, so Nb-93 of
ENDF/B-VIII.1 (MF4/MF5) had MT16's yield 1 where OpenMC's file has 2, and MT17's 1
for 3: one neutron per (n,2n) in transport, and the KERMA's balance one neutron short.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from kika.hdf5.processing import _tabulatedProduct
from kika.nuclear_data import model as m
from kika.processing.tabulation import TabulationOptions


class _Report:
    def lost(self, message):
        raise AssertionError(message)

    def approximated(self, message):
        pass


def _neutron():
    tables = [m.XYs1d(xs=[0.0, 4e6], ys=[2.5e-7, 2.5e-7], outerDomainValue=E) for E in (1e7, 2e7)]
    product = m.Product(pid="n", label="n")
    product.distribution = m.Distribution()
    product.distribution["eval"] = m.Uncorrelated(angular=m.Isotropic2d(), energy=m.XYs2d(function1ds=tables),
                                                 productFrame=m.Frame.lab)
    return product


@pytest.mark.parametrize("mt, count", [(16, 2.0), (17, 3.0), (37, 4.0), (91, 1.0), (24, 2.0)])
def test_an_unstated_neutron_count_is_the_mts(mt, count):
    reaction = NS(provenance=None, outputChannel=NS(Q=NS(value=-8e6)))
    new = _tabulatedProduct(_neutron(), reaction, mt, 92.1, 41093, TabulationOptions(), _Report(), qi=-8e6)
    assert float(new.multiplicity.form.evaluate([1.4e7])[0]) == count
