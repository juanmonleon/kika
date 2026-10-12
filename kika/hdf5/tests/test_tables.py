"""``kika.hdf5.tables``: the arrays a viewer draws from a nuclide read from HDF5."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from kika.hdf5 import read, tables as T  # noqa: E402

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def fe56():
    return read(DATA / "Fe56.h5")


def test_temperatures_are_sorted_and_294_is_the_reference(fe56):
    temps = T.temperatures(fe56)
    kelvins = [t["kelvin"] for t in temps]
    assert kelvins == sorted(kelvins) and kelvins
    assert all(t["label"] == f"heated_{t['kelvin']:g}K" for t in temps)
    reference = T.referenceTemperature(fe56)
    assert reference in kelvins
    assert abs(reference - 294) == min(abs(k - 294) for k in kelvins)


def test_cross_sections_per_temperature_and_the_0K_elastic(fe56):
    reference = T.referenceTemperature(fe56)
    xs = T.crossSections(fe56, reference)
    assert 2 in xs and 102 in xs
    e, s = xs[2]
    assert e.shape == s.shape and np.all(np.diff(e) >= 0)
    assert list(T.crossSections(fe56, 0)) == [2]
    with pytest.raises(KeyError, match="no data at 1 K"):
        T.crossSections(fe56, 1)


def test_angular_tables_are_normalised_and_moments_match_the_table(fe56):
    assert 2 in T.angularMTs(fe56)
    bulk = T.angularBulk(fe56, maxOrder=4, mt=2)
    assert bulk["representation"] == 2 and bulk["frame"] == "CM"
    a0 = np.asarray(bulk["coefficients_by_order"]["0"])
    assert np.allclose(a0, 1.0, atol=1e-5)
    # a_1 is the mean cosine of the table sent (lin-lin), exact on each segment.
    mu = np.asarray(bulk["pdf_mu_by_energy"][-1])
    p = np.asarray(bulk["pdf_by_energy"][-1])
    h = np.diff(mu)
    mean = np.sum(h * (mu[:-1] * (2 * p[:-1] + p[1:]) + mu[1:] * (p[:-1] + 2 * p[1:])) / 6)
    assert bulk["coefficients_by_order"]["1"][-1] == pytest.approx(mean, abs=1e-10)


def test_histogram_table_goes_out_as_steps():
    mu, p = T._linear(np.array([-1.0, 0.0, 1.0]), np.array([0.25, 0.75, 0.0]), True)
    assert mu.tolist() == [-1.0, 0.0, 0.0, 1.0]
    assert p.tolist() == [0.25, 0.25, 0.75, 0.75]
    a = T._moments(np.array([-1.0, 0.0, 1.0]), np.array([0.25, 0.75, 0.0]), True, 1)
    assert a[0] == pytest.approx(1.0) and a[1] == pytest.approx(0.25)


def test_summary_rows(fe56):
    summary = T.isotopeSummary(fe56)
    assert summary["target"] == "Fe56"
    mts = [r["mt"] for r in summary["reactions"]]
    assert mts == sorted(mts) and 2 in mts
    elastic = next(r for r in summary["reactions"] if r["mt"] == 2)
    assert elastic["angular"] and not elastic["sum"]
    assert summary["elastic_0k"]["count"] > 0


def test_a_correlated_law_is_not_angular(fe56):
    for mt in T.angularMTs(fe56):
        T.angularBulk(fe56, mt=mt, maxOrder=2)
    others = [r.ENDF_MT for r in fe56.reactions if r.ENDF_MT not in T.angularMTs(fe56)]
    if others:
        with pytest.raises(ValueError, match="no angular distribution"):
            T.angularBulk(fe56, mt=int(others[0]))
