"""HDF5 → model → HDF5 is identical, and what the model holds.

The same over every file of the three official libraries is checked
outside the test suite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kika
from kika.hdf5 import records as R
from kika.hdf5.neutron_decode import decodeNeutron
from kika.hdf5.neutron_encode import encodeNeutron
from kika.hdf5.thermal_decode import decodeThermal
from kika.hdf5.thermal_encode import encodeThermal

DATA = Path(__file__).parent / "data"
NEUTRON = ["Fe56", "U235", "Cu63", "H2"]
THERMAL = ["c_Graphite", "c_H_in_ZrH", "c_H_in_H2O", "c_Zr_in_ZrH2"]


@pytest.mark.parametrize("name", NEUTRON)
def test_neutron_round_trip_is_identical(name):
    tree = R.readFile(DATA / f"{name}.h5")
    suite, _ = decodeNeutron(tree)
    assert R.compareTrees(tree, encodeNeutron(suite)) == []


@pytest.mark.parametrize("name", THERMAL)
def test_thermal_round_trip_is_identical(name):
    tree = R.readFile(DATA / f"{name}.h5")
    suite, _ = decodeThermal(tree)
    assert R.compareTrees(tree, encodeThermal(suite)) == []


def test_the_door_reads_and_writes_hdf5(tmp_path):
    assert kika.sniff_format(DATA / "Fe56.h5") == "hdf5"
    suite = kika.read(DATA / "Fe56.h5")
    report = kika.write(suite, tmp_path, format="hdf5")
    assert R.compareFiles(DATA / "Fe56.h5", tmp_path / "Fe56.h5") == []
    assert (tmp_path / "cross_sections.xml").is_file()
    assert report is not None
    thermal = kika.read(DATA / "c_H_in_H2O.h5")
    kika.write(thermal, tmp_path, format="hdf5")
    assert R.compareFiles(DATA / "c_H_in_H2O.h5", tmp_path / "c_H_in_H2O.h5") == []
    index = (tmp_path / "cross_sections.xml").read_text()
    assert 'materials="Fe56"' in index and 'type="thermal"' in index


def test_what_the_model_holds_for_u235():
    suite = kika.read(DATA / "U235.h5")
    assert suite.styles.labels == ["eval", "heated_294K", "recon", "MonteCarlo", "URR_294K"]
    heated = suite.styles["heated_294K"]
    assert heated.temperature.unit == "eV/k"
    assert round(heated.temperature.value / 8.617333262e-05) == 294
    fission = suite.reactions[18]
    neutron = fission.outputChannel.products[0]
    assert neutron.pid == "n" and neutron.provenance.attributes["emission_mode"] == "prompt"
    families = fission.outputChannel.fissionFragmentData.delayedNeutrons
    assert len(families) == 6
    assert families[0].rate.unit == "1/s" and 0.01 < families[0].rate.value < 0.02
    assert suite.sums.multiplicitySums.byENDF_MT(452) is not None
    release = fission.outputChannel.fissionFragmentData.fissionEnergyReleases[0]
    assert release.promptProductKE is not None and "q_prompt" in release.provenance.attributes
    assert 4 in [r.ENDF_MT for r in suite.sums]
    assert sorted(r.ENDF_MT for r in suite.productions) == [301, 444, 901]
    urr = suite.reactions[2].crossSection["URR_294K"].table
    assert urr.table.shape[1] == 6 and urr.inelastic == 4
    assert suite.reactions[102].crossSection["URR_294K"].table is urr
    E, xs = suite.cross_section(2, form="recon")
    assert E[0] > 0 and len(E) == len(xs)


def test_what_the_model_holds_for_fe56():
    m = pytest.importorskip("kika.nuclear_data.model")
    suite = kika.read(DATA / "Fe56.h5")
    level = suite.reactions[51]
    form = level.outputChannel.products[0].distribution["MonteCarlo"]
    assert isinstance(form, m.DistributionMixture)
    assert isinstance(form.weighted[0].functional.energy, m.LevelEnergy)
    photon = [p for p in level.outputChannel.products if p.pid == "photon"][0]
    photonForm = photon.distribution["MonteCarlo"]
    if isinstance(photonForm, m.DistributionMixture):
        photonForm = photonForm.weighted[0].functional
    energy = photonForm.energy
    # ENDF/B-VIII.1 states the level's gamma as a line (JENDL-5 as a table).
    assert isinstance(energy, m.DiscretePhotonEnergy) and energy.primaryFlag in (0, 1)
    correlated = suite.reactions[16].outputChannel.products[0].distribution["MonteCarlo"]
    assert isinstance(correlated.weighted[0].functional, m.EnergyAngularMC)
    # Each table's cumulative is kept as stored, not recomputed.
    table = suite.reactions[2].outputChannel.products[0].distribution["MonteCarlo"].angular[0]
    assert isinstance(table, m.Xs_pdf_cdf1d)
    assert table.cdf[0] == 0.0 and table.cdf[-1] == pytest.approx(1.0)
    assert 103 in [r.ENDF_MT for r in suite.sums]
    summed = next(r for r in suite.sums if r.ENDF_MT == 103)
    assert len(summed.summands) == 1  # only MT600 was kept in the fixture


def test_the_total_the_cpp_rebuilds_is_the_sum_of_the_partials():
    """Gate 2's first half: nothing redundant sits in suite.reactions."""
    for name in NEUTRON:
        suite = kika.read(DATA / f"{name}.h5")
        for reaction in suite.reactions:
            assert reaction.provenance.attributes["redundant"] == 0
        for container in (suite.sums, suite.productions):
            for reaction in container:
                assert reaction.provenance.attributes["redundant"] == 1


def test_an_edited_model_is_written_edited(tmp_path):
    suite = kika.read(DATA / "H2.h5")
    form = suite.reactions[102].crossSection["heated_294K"]
    form.ys = form.ys * 2.0
    out = R.Group()
    tree = encodeNeutron(suite)
    xs = tree["H2/reactions/reaction_102/294K/xs"].data
    assert np.allclose(xs, form.ys)
