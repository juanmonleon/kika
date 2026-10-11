"""The record layer gives back what it read: datasets, dtypes and attributes.

Roadmap H1's gate on the committed fixtures; the same check over every file of
the three official libraries is ``test_full_libraries.py`` (marker ``tape``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.hdf5 import records as R

DATA = Path(__file__).parent / "data"
FIXTURES = sorted(DATA.glob("*.h5"))


def test_there_are_fixtures_of_every_kind():
    names = {p.stem for p in FIXTURES}
    assert {"Fe56", "U235", "Cu63", "H2", "c_Graphite", "c_H_in_ZrH",
            "c_H_in_H2O", "c_Zr_in_ZrH2", "Fe"} <= names


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_read_then_write_is_identical(path, tmp_path):
    tree = R.readFile(path)
    out = R.writeFile(tree, tmp_path / path.name)
    assert R.compareFiles(path, out) == []


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_lazy_read_then_write_is_identical(path, tmp_path):
    with R.openFile(path) as tree:
        R.writeFile(tree, tmp_path / path.name)
    assert R.compareFiles(path, tmp_path / path.name) == []


def test_strings_stay_fixed_length_bytes():
    tree = R.readFile(DATA / "U235.h5")
    filetype = tree.attrs["filetype"]
    assert filetype.value == b"data_neutron"
    assert filetype.dtype == np.dtype("S12")
    assert R.text(filetype) == "data_neutron"
    label = tree["U235/reactions/reaction_018"].attrs["label"]
    assert label.value == b"(n,fission)"


def test_a_change_is_reported():
    a = R.readFile(DATA / "H2.h5")
    b = R.readFile(DATA / "H2.h5")
    xs = b["H2/reactions/reaction_002/294K/xs"]
    xs.data = xs.data * (1 + 1e-15)
    b["H2"].attrs["Z"] = R.Attribute.of(np.int32(1))
    del b["H2/reactions/reaction_102"].children["product_0"]
    diffs = R.compareTrees(a, b)
    assert any("reaction_002/294K/xs: values differ" in d for d in diffs)
    assert any("@Z" in d and "int32" in d for d in diffs)
    assert any("product_0: only in first" in d for d in diffs)


def test_attribute_of_uses_openmcs_dtypes():
    assert R.Attribute.of("prompt").dtype == np.dtype("S6")
    assert R.Attribute.of(3).dtype == np.dtype("int64")
    assert R.Attribute.of(1.5).dtype == np.dtype("float64")
    assert R.Attribute.of([3, 0]).shape == (2,)
