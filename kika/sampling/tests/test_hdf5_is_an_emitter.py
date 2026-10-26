"""``formats=("hdf5",)``: each realisation processed by kika and written as OpenMC HDF5.

Two samples are two ``.h5`` files of the same nuclide whose cross sections
differ, ``RunResult.paths("hdf5")`` lists those files, and a named base
library gives each sample an overlay index that sends the nuclide to the
sample's file and every other material to the base library.
"""
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pytest

from kika.hdf5.library import CrossSectionsXml
from kika.sampling.model_perturbation import HDF5Options, perturbFromModel

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
TAPE = str(DATA / "micro_fe56_xs_and_angular.endf")
REQUEST = {33: None}

perturbFromModel = partial(perturbFromModel, resonanceRegion="evaluated")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """A base library index naming Fe56 and one other material; its files are never opened."""
    folder = tmp_path_factory.mktemp("base")
    (folder / "cross_sections.xml").write_text(
        "<?xml version='1.0' encoding='UTF-8'?>\n<cross_sections>\n"
        '  <library materials="H1" path="H1.h5" type="neutron"/>\n'
        '  <library materials="Fe56" path="Fe56.h5" type="neutron"/>\n'
        "</cross_sections>\n")
    return folder


@pytest.fixture(scope="module")
def run(tmp_path_factory, base):
    out = tmp_path_factory.mktemp("hdf5")
    return perturbFromModel(TAPE, REQUEST, 2, seed=20261026, outputDir=out, formats=("hdf5",),
                            hdf5=HDF5Options(temperatures=(293.6,), baseLibrary=str(base)))


def test_paths_lists_one_h5_per_sample(run):
    paths = run.paths("hdf5")
    assert [p.relative_to(run.outputDir).as_posix() for p in paths] == [
        "0000/hdf5/Fe56.h5", "0001/hdf5/Fe56.h5"]
    assert all(p.is_file() for p in paths)


def test_two_samples_are_two_different_files_of_one_nuclide(run):
    elastic = []
    for path in run.paths("hdf5"):
        with h5py.File(path, "r") as handle:
            elastic.append(np.asarray(handle["Fe56/reactions/reaction_002/294K/xs"]))
    assert all(np.all(np.isfinite(e)) for e in elastic)
    assert not np.array_equal(*elastic)


def test_the_overlay_sends_the_nuclide_to_the_sample_and_the_rest_to_the_base(run, base):
    for sample, h5 in zip(run.samples, run.paths("hdf5")):
        index = CrossSectionsXml.read(sample["files"]["hdf5"]["cross_sections"])
        where = {entry.materials[0]: Path(index.resolve(entry)).resolve() for entry in index.entries}
        assert where["Fe56"] == h5.resolve()
        assert where["H1"] == (base / "H1.h5").resolve()


def test_the_options_are_checked_before_anything_is_drawn(tmp_path):
    with pytest.raises(ValueError, match="at least one temperature"):
        perturbFromModel(TAPE, REQUEST, 1, formats=("hdf5",), hdf5=HDF5Options(temperatures=()))
    with pytest.raises(ValueError, match="urr, kerma and damage"):
        perturbFromModel(TAPE, REQUEST, 1, formats=("hdf5",), hdf5=HDF5Options(omit=("heating",)))
    with pytest.raises(ValueError, match="does not exist"):
        perturbFromModel(TAPE, REQUEST, 1, formats=("hdf5",),
                         hdf5=HDF5Options(baseLibrary=str(tmp_path / "nowhere")))
