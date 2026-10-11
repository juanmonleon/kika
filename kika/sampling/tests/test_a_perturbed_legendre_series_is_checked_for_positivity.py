"""The model route checks a perturbed Legendre series for f(mu) < 0, and repairs it on request.

``angularPositivity="repair"`` used to reach only tabulated distributions; a
Legendre series that went negative was neither counted nor repaired.
"""
import numpy as np

from kika.nuclear_data.model.functions.higher import XYs2d
from kika.nuclear_data.model.functions.simple import Legendre
from kika.sampling.mf4_positivity import check_mf4_positivity
from kika.sampling.perturbation_set import _legendrePositivity


def _angular():
    return XYs2d(function1ds=[
        Legendre(coefficients=np.array([1.0, 0.9, 0.05]), outerDomainValue=1.0e6),
        Legendre(coefficients=np.array([1.0, 0.2, 0.0]), outerDomainValue=2.0e6),
    ])


def test_report_counts_and_leaves_the_series():
    angular = _angular()
    info = _legendrePositivity(angular, 1, repair=False)
    assert info["n_negative_series"] == 1 and info["n_repaired"] == 0
    assert info["min_sum"] < 0
    np.testing.assert_array_equal(angular.function1ds[0].coefficients, [1.0, 0.9, 0.05])


def test_repair_projects_within_the_perturbed_band():
    angular = _angular()
    info = _legendrePositivity(angular, 1, repair=True)
    assert info["n_repaired"] == 1
    a = angular.function1ds[0].coefficients
    assert a[0] == 1.0 and a[2] == 0.05          # a_0 and the unperturbed order pinned
    assert a[1] < 0.9
    assert check_mf4_positivity(a)[1] >= -1e-9
    np.testing.assert_array_equal(angular.function1ds[1].coefficients, [1.0, 0.2, 0.0])
