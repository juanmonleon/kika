"""The contract checker passes good files and catches each broken assumption."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.hdf5 import records as R
from kika.hdf5.contract import checkNeutronTree

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("name", ["Fe56", "U235", "Cu63", "H2"])
def test_the_fixtures_keep_the_contract(name):
    assert checkNeutronTree(R.readFile(DATA / f"{name}.h5")) == []


def _broken(name="U235"):
    return R.readFile(DATA / f"{name}.h5")


def test_a_misnamed_temperature_is_caught():
    tree = _broken()
    tree["U235/kTs/294K"].data = np.float64(0.03)
    assert any("kTs/294K" in v for v in checkNeutronTree(tree))


def test_a_threshold_index_off_by_one_is_caught():
    tree = _broken()
    xs = tree["U235/reactions/reaction_102/294K/xs"]
    xs.attrs["threshold_idx"] = R.Attribute.of(np.int64(1))
    assert any("threshold_idx" in v for v in checkNeutronTree(tree))


def test_reaction_001_and_a_missing_temperature_are_caught():
    tree = _broken()
    tree["U235/reactions"].children["reaction_001"] = tree["U235/reactions/reaction_102"]
    del tree["U235/reactions/reaction_018"].children["294K"]
    found = checkNeutronTree(tree)
    assert any("reaction_001" in v for v in found)
    assert any("reaction_018 has no 294K" in v for v in found)


def test_a_non_redundant_response_is_caught():
    tree = _broken()
    tree["U235/reactions/reaction_301"].setAttr("redundant", 0)
    assert any("reaction_301 must be redundant" in v for v in checkNeutronTree(tree))


def test_delayed_groups_out_of_place_are_caught():
    tree = _broken()
    r = tree["U235/reactions/reaction_018"]
    r.children["product_0"], r.children["product_1"] = r.children["product_1"], r.children["product_0"]
    assert any("product_0 is not" in v for v in checkNeutronTree(tree))


def test_a_bad_cdf_is_caught():
    tree = _broken("Fe56")
    mu = tree["Fe56/reactions/reaction_002/product_0/distribution_0/angle/mu"]
    data = mu.data.copy()
    data[2, -1] = 0.9
    mu.data = data
    assert any("CDF ends" in v for v in checkNeutronTree(tree))


def test_urr_naming_an_absent_reaction_is_caught():
    tree = _broken()
    del tree["U235/reactions"].children["reaction_004"]
    assert any("urr names MT4" in v for v in checkNeutronTree(tree))
