"""COVFIL, BOXER and COVERX hold relative covariances: an absolute block is converted or refused."""

import numpy as np
import pytest

from kika.cov import CrossSectionCovariance
from kika.cov.parse_covmat import (
    read_boxer,
    read_covfil,
    read_coverx,
    write_boxer,
    write_covfil,
    write_coverx,
)

FE56 = 26056
GRID = [1.0e-5, 1.0e5, 1.0e6, 2.0e7]
RELATIVE = np.array([[0.01, 0.002, 0.0], [0.002, 0.04, 0.003], [0.0, 0.003, 0.09]])
XS = np.array([2.0, 4.0, 0.5])

FORMATS = {
    "covfil": (lambda c, p: write_covfil(c, p), read_covfil),
    "boxer": (lambda c, p: write_boxer(c, p), read_boxer),
    "coverx": (lambda c, p: write_coverx(c, p, fmt="text"), read_coverx),
}


def _absolute(with_cross_section: bool = True) -> CrossSectionCovariance:
    covariance = CrossSectionCovariance(num_groups=3, energy_grid=list(GRID))
    covariance.add_matrix(FE56, 2, FE56, 2, RELATIVE * np.outer(XS, XS), is_relative=False)
    if with_cross_section:
        covariance.cross_sections[(FE56, 2)] = XS.copy()
    return covariance


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_an_absolute_block_is_written_relative(tmp_path, fmt):
    write, read = FORMATS[fmt]
    path = str(tmp_path / f"fe56.{fmt}")
    write(_absolute(), path)

    back = read(path)
    assert np.allclose(back.matrices[0], RELATIVE, rtol=1e-5)


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_an_absolute_block_without_its_cross_section_is_refused(tmp_path, fmt):
    write, _ = FORMATS[fmt]

    with pytest.raises(ValueError, match="absolute"):
        write(_absolute(with_cross_section=False), str(tmp_path / f"fe56.{fmt}"))


def test_the_writer_does_not_change_the_object_it_writes(tmp_path):
    covariance = _absolute()
    write_covfil(covariance, str(tmp_path / "fe56.gendf"))

    assert covariance.is_relative == [False]
    assert np.allclose(covariance.matrices[0], RELATIVE * np.outer(XS, XS))
