"""The manifest's per-point columns must follow the points, not the table.

The database loader groups a dataset's points into energy blocks sorted by angle, while the
``dy`` columns it stashes for the manifest keep the original EXFOR table order. Until 2026-09-07
``apply_manifest_to_exfor`` applied the columns positionally, so an angle-major table (all
energies of one detector, then the next: Pirovano 23365004/5, Barnard 30076004, Tsukada
20304002) gave every point another row's DATA-ERR -- Pirovano's 16 degree rows ended at the 1 %
floor where EXFOR says 9-17 %. Invariance test: the same measurements in two table orders must
resolve to the same per-point sigma, and each must equal what its own DATA-ERR gives.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kika.exfor.database import X4ProDatabase, X4ProDataset
from scripts.uncertainty_manifest import apply_manifest_to_exfor

MICRO_MANIFEST = Path(__file__).parent / "data" / "uncertainty_manifest_micro.yaml"


@pytest.fixture
def micro_manifest(monkeypatch):
    monkeypatch.setenv("KIKA_UNCERTAINTY_MANIFEST_PATH", str(MICRO_MANIFEST))   # the cache is keyed by path
    yield


def _dataset(order: str) -> X4ProDataset:
    """Two energies x three angles; ``order`` = 'angle_major' or 'energy_major'."""
    energies = np.array([2.0e6, 3.0e6])
    angles = np.array([20.0, 90.0, 160.0])
    xs = {(2.0e6, 20.0): 1.0, (2.0e6, 90.0): 0.10, (2.0e6, 160.0): 0.20,
          (3.0e6, 20.0): 1.2, (3.0e6, 90.0): 0.08, (3.0e6, 160.0): 0.15}
    rel = {(2.0e6, 20.0): 0.12, (2.0e6, 90.0): 0.25, (2.0e6, 160.0): 0.20,
           (3.0e6, 20.0): 0.15, (3.0e6, 90.0): 0.30, (3.0e6, 160.0): 0.22}
    if order == "angle_major":
        keys = [(e, a) for a in angles[::-1] for e in energies]      # 160 deg first, like Pirovano
    else:
        keys = [(e, a) for e in energies for a in angles]
    e_arr = np.array([k[0] for k in keys]); a_arr = np.array([k[1] for k in keys])
    y = np.array([xs[k] for k in keys]); dy = np.array([rel[k] * xs[k] for k in keys])
    return X4ProDataset(
        dataset_id="99999002", year=2019, author="Test", target="26-FE-56", projectile="N",
        mf=4, mt=2, quant="DA", ndat=len(keys), reacode="26-FE-56(N,EL)26-FE-56,,DA",
        energies_ev=e_arr, angles_deg=a_arr, cross_sections=y, uncertainties=dy,
        energy_unit="EV", angle_unit="ADEG", xs_unit="B/SR",
        uncertainty_components=[{"header": "DATA-ERR", "kind": "per_point", "values": dy.tolist(), "unit": "B/SR"}],
    ), xs, rel


def _per_point(ad):
    out = {}
    for blk in ad._data_blocks:
        for pt in blk["data"]:
            out[(round(float(blk["value"]) * 1e6), round(float(pt["angle"]), 3))] = (
                float(pt["cross_section"]), float(pt["uncertainty_stat"]), float(pt.get("uncertainty_sys", 0.0)))
    return out


def test_points_carry_their_table_row():
    ds, _, _ = _dataset("angle_major")
    ad = X4ProDatabase.__new__(X4ProDatabase)._convert_to_angular_distribution(ds)
    seen = set()
    for blk in ad._data_blocks:
        for pt in blk["data"]:
            i = pt["table_index"]
            assert np.isclose(ds.cross_sections[i], pt["cross_section"])
            assert np.isclose(ds.angles_deg[i], pt["angle"])
            seen.add(i)
    assert seen == set(range(len(ds.cross_sections)))


def test_sigma_follows_the_point_not_the_table(micro_manifest):
    res = {}
    for order in ("angle_major", "energy_major"):
        ds, xs, rel = _dataset(order)
        ad = X4ProDatabase.__new__(X4ProDatabase)._convert_to_angular_distribution(ds)
        apply_manifest_to_exfor(ad, uncertainty_components=ad._raw_uncertainty_components)
        res[order] = _per_point(ad)
    assert res["angle_major"].keys() == res["energy_major"].keys()
    for k in res["angle_major"]:
        ya, sa, za = res["angle_major"][k]; yb, sb, zb = res["energy_major"][k]
        assert np.isclose(ya, yb)
        assert np.isclose(sa, sb), "{}: sigma_stat {} (angle-major table) vs {} (energy-major)".format(k, sa, sb)
        assert np.isclose(za, zb)
    # and each point's total is its OWN DATA-ERR: stat (+) sys == rel * y, up to the 1 % stat floor
    ds, xs, rel = _dataset("angle_major")
    for (e, a), (y, s, z) in res["angle_major"].items():
        own = rel[(float(e), a)] * xs[(float(e), a)]
        assert np.isclose(np.hypot(s, z), own, rtol=1e-6) or s <= 0.0100001 * y, (e, a, s, z, own)
