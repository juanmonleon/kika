"""``energyCovarianceTape`` gives the MF35 and MF5 a written tape gives back, bit for bit."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import kika
from kika.endf.read_endf import read_endf
from kika.endf.writers.assemble import energyCovarianceTape, writeEndfTape
from kika.gnds.covariances import readCovarianceSuite
from kika.gnds.xpath import Document


@pytest.fixture(scope="module")
def gnds(micro_pfns_tape, tmp_path_factory):
    folder = tmp_path_factory.mktemp("gnds")
    kika.write(kika.read(micro_pfns_tape), folder / "cf252.xml", format="gnds")
    return folder / "cf252.xml"


def _same(a, b, where):
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            assert np.array_equal(x, y, equal_nan=True), f"{where}: {field.name}"
        else:
            assert x == y, f"{where}: {field.name}"


def _assertSameMF35(written, built):
    assert built.mat == written.mat
    for mt, section in written.files[35].mt.items():
        other, mf5 = built.files[35].mt[mt], built.files[5].mt[mt]
        bands = section.subsections
        assert len(other.subsections) == len(bands)
        for index, (band, twin) in enumerate(zip(bands, other.subsections)):
            assert np.array_equal(band.energy_grid(), twin.energy_grid())
            assert np.array_equal(band.matrix(), twin.matrix())
            assert band.row_sum_residual() == twin.row_sum_residual()
            assert band.normalisation_drift() == twin.normalisation_drift()
            energies = [None, band.e1 + 0.3719 * (band.e2 - band.e1)]
            if index == len(bands) - 1:
                energies.append(band.e2)
            for energy in energies:
                expected = section.relative_uncertainty(
                    written.files[5].mt[mt], incident_energy=energy, band_index=index)
                assert expected.relative is not None
                _same(expected, other.relative_uncertainty(
                    mf5, incident_energy=energy, band_index=index), f"MT{mt} band {index} at {energy}")


def test_a_suite_read_from_gnds_with_its_covariances_apart(gnds, tmp_path):
    """The app's case: suite and covariance file read separately, provenance derived."""
    suite = kika.read(gnds, covariances=False)
    covariances, _ = readCovarianceSuite(Document.parse(gnds.parent / "Covariances" / "cf252-covar.xml"))
    built, _ = energyCovarianceTape(suite, covariances)

    full = kika.read(gnds)
    writeEndfTape(full, tmp_path / "derived.endf", files=((1, 451), 5, 35))
    _assertSameMF35(read_endf(str(tmp_path / "derived.endf"), mf_numbers=[1, 5, 35]), built)

    assert suite.covarianceSuite is None
    assert all(section.provenance is None for section in covariances.covarianceSections)


def test_a_suite_read_from_endf(micro_pfns_tape):
    built, _ = energyCovarianceTape(kika.read(micro_pfns_tape))
    assert sorted(built.files) == [5, 35]
    _assertSameMF35(read_endf(str(micro_pfns_tape), mf_numbers=[1, 5, 35]), built)
