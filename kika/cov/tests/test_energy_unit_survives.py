"""The energy unit and the file metadata survive every filter, and lookups by energy use the unit."""

import numpy as np
import pytest

from kika.cov import CrossSectionCovariance
from kika.cov.parse_covmat import read_boxer, write_boxer

FE56 = 26056
O16 = 8016
GRID_MEV = [1.0e-11, 0.1, 1.0, 20.0]


def _mev_covariance() -> CrossSectionCovariance:
    covariance = CrossSectionCovariance(
        num_groups=3,
        energy_grid=list(GRID_MEV),
        energy_unit="MeV",
        metadata={"awr": 55.45443, "temperature": 293.6},
    )
    for zaid in (FE56, O16):
        for mt in (2, 102):
            covariance.add_matrix(zaid, mt, zaid, mt, np.diag([0.01, 0.04, 0.09]), is_relative=True)
            covariance.cross_sections[(zaid, mt)] = np.array([1.0, 2.0, 3.0])
    return covariance


FILTERS = {
    "remove_matrix": lambda c: c.remove_matrix(FE56, [(102, 102)]),
    "clean_cov": lambda c: c.clean_cov(FE56),
    "filter_by_isotope": lambda c: c.filter_by_isotope(FE56),
    "filter_by_isotopes": lambda c: c.filter_by_isotopes([FE56]),
    "filter_by_reactions": lambda c: c.filter_by_reactions([2]),
}


@pytest.mark.parametrize("name", sorted(FILTERS))
def test_a_filter_keeps_the_energy_unit_and_the_metadata(name):
    filtered = FILTERS[name](_mev_covariance())

    assert filtered.energy_unit == "MeV"
    assert filtered.energy_grid == GRID_MEV
    assert filtered.metadata == {"awr": 55.45443, "temperature": 293.6}


def test_a_filtered_mev_covariance_is_written_in_ev(tmp_path):
    path = tmp_path / "fe56.boxer"
    write_boxer(_mev_covariance().filter_by_isotope(FE56), str(path))

    assert np.allclose(read_boxer(str(path)).energy_grid, [1.0e-5, 1.0e5, 1.0e6, 2.0e7])


@pytest.mark.parametrize("unit, scale", [("eV", 1.0e6), ("MeV", 1.0)])
def test_the_group_of_an_energy_follows_the_energy_unit(unit, scale):
    covariance = CrossSectionCovariance(
        num_groups=3,
        energy_grid=[e * scale for e in GRID_MEV],
        energy_unit=unit,
    )
    covariance.add_matrix(FE56, 2, FE56, 2, np.diag([0.01, 0.04, 0.09]), is_relative=True)

    assert covariance.get_uncertainty(FE56, 2, energy_mev=1.5) == pytest.approx(0.3)
    assert covariance.get_uncertainty(FE56, 2, energy_mev=0.05) == pytest.approx(0.1)


def test_an_absolute_block_without_its_cross_section_has_no_relative_uncertainty():
    covariance = CrossSectionCovariance(num_groups=3, energy_grid=[e * 1.0e6 for e in GRID_MEV])
    covariance.add_matrix(FE56, 2, FE56, 2, np.diag([0.01, 0.04, 0.09]), is_relative=False)

    with pytest.raises(ValueError, match="absolute"):
        covariance.get_uncertainty(FE56, 2)
