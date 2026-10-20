"""Automated coverage for the supported multigroup covariance formats."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from kika.cov import (
    read_boxer,
    read_covfil,
    read_coverx,
    write_boxer,
    write_covfil,
    write_coverx,
)
from kika.cov.cross_section_covariance import CrossSectionCovariance
from kika.cov.legendre_covariance import LegendreCovariance
from kika.cov.parse_covmat import _read_fortran_record


FE56 = 26056
GRID_EV = np.array([1.0e-5, 1.0, 1.0e3, 1.0e6, 2.0e7])


def make_covariance() -> CrossSectionCovariance:
    covariance = CrossSectionCovariance(
        num_groups=4,
        energy_grid=GRID_EV.tolist(),
        energy_unit="eV",
        metadata={"awr": 55.45443, "temperature": 293.6},
    )
    covariance.add_matrix(
        FE56,
        2,
        FE56,
        2,
        np.diag([0.01, 0.02, 0.03, 0.04]),
        is_relative=True,
    )
    covariance.add_matrix(
        FE56,
        18,
        FE56,
        18,
        np.array(
            [
                [0.10, 0.01, 0.00, 0.00],
                [0.01, 0.20, 0.02, 0.00],
                [0.00, 0.02, 0.30, 0.03],
                [0.00, 0.00, 0.03, 0.40],
            ]
        ),
        is_relative=True,
    )
    covariance.cross_sections[(FE56, 2)] = np.array([1.0, 2.0, 3.0, 4.0])
    covariance.cross_sections[(FE56, 18)] = np.array([0.5, 0.6, 0.7, 0.8])
    return covariance


def matrix_map(covariance: CrossSectionCovariance) -> dict:
    keys = zip(
        covariance.isotope_rows,
        covariance.reaction_rows,
        covariance.isotope_cols,
        covariance.reaction_cols,
    )
    return {key: matrix for key, matrix in zip(keys, covariance.matrices)}


def assert_covariance_equal(
    expected: CrossSectionCovariance,
    actual: CrossSectionCovariance,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
    cross_sections: bool = True,
) -> None:
    assert actual.num_groups == expected.num_groups
    np.testing.assert_allclose(actual.energy_grid, expected.energy_grid, rtol=rtol)

    expected_matrices = matrix_map(expected)
    actual_matrices = matrix_map(actual)
    assert actual_matrices.keys() == expected_matrices.keys()
    for key in expected_matrices:
        np.testing.assert_allclose(
            actual_matrices[key],
            expected_matrices[key],
            rtol=rtol,
            atol=atol,
        )

    if cross_sections:
        assert actual.cross_sections.keys() == expected.cross_sections.keys()
        for key in expected.cross_sections:
            np.testing.assert_allclose(
                actual.cross_sections[key],
                expected.cross_sections[key],
                rtol=rtol,
                atol=atol,
            )


@pytest.mark.parametrize(
    "fmt,endian",
    [("text", ">"), ("binary", ">"), ("binary", "<")],
)
def test_coverx_reader_roundtrip_and_unit_conversion(tmp_path, fmt, endian):
    source = make_covariance()
    source.add_matrix(
        FE56,
        2,
        FE56,
        18,
        np.array(
            [
                [0.001, 0.002, 0.003, 0.004],
                [0.005, 0.006, 0.007, 0.008],
                [0.009, 0.010, 0.011, 0.012],
                [0.013, 0.014, 0.015, 0.016],
            ]
        ),
        is_relative=True,
    )
    path = tmp_path / f"known.{fmt}"

    with pytest.warns(UserWarning):
        write_coverx(source, path, fmt=fmt, endian=endian, title="known values")

    parsed = read_coverx(path)
    assert_covariance_equal(
        source,
        parsed,
        rtol=2.0e-7,
        atol=2.0e-8,
        cross_sections=fmt == "binary",
    )
    if fmt == "text":
        assert parsed.cross_sections == {}

    parsed_mev = read_coverx(path, energy_unit="MeV")
    assert parsed_mev.energy_unit == "MeV"
    np.testing.assert_allclose(parsed_mev.energy_grid, GRID_EV * 1.0e-6, rtol=2.0e-7)

    native_order = read_coverx(path, ascending=False)
    np.testing.assert_allclose(native_order.energy_grid, GRID_EV[::-1], rtol=2.0e-7)
    for key, matrix in matrix_map(source).items():
        np.testing.assert_allclose(
            matrix_map(native_order)[key],
            np.flipud(np.fliplr(matrix)),
            rtol=2.0e-7,
            atol=2.0e-8,
        )


def test_coverx_binary_uses_standard_hollerith_and_xs_record_lengths(tmp_path):
    source = make_covariance()
    path = tmp_path / "records.coverx"
    with pytest.warns(UserWarning, match="float32"):
        write_coverx(source, path, fmt="binary", title="1234567", endian=">")

    with path.open("rb") as stream:
        _read_fortran_record(stream, ">")
        control = _read_fortran_record(stream, ">")
        ngroup, _, _, _, _, _, nholl = struct.unpack(">7i", control)
        description = _read_fortran_record(stream, ">")
        _read_fortran_record(stream, ">")
        _read_fortran_record(stream, ">")
        first_xs = _read_fortran_record(stream, ">")

    assert nholl == 2
    assert len(description) == nholl * 6
    assert len(first_xs) == 2 * ngroup * 4


REAL_SCALE44 = Path(
    os.environ.get(
        "KIKA_TEST_SCALE44_COVERX",
        "/share_snc/projets/INSIDER/SENSIBILITE/"
        "Covariances_scale_6.2.3_binary/scale.rev05.44groupcov",
    )
)


@pytest.mark.skipif(not REAL_SCALE44.is_file(), reason="real SCALE-44 COVERX unavailable")
def test_real_scale44_coverx_cross_block_orientation():
    covariance = read_coverx(REAL_SCALE44, energy_unit="MeV")

    # The file header identifies MT=4 horizontally and MT=102 vertically;
    # CrossSectionCovariance exposes conventional matrix row/column semantics.
    block = matrix_map(covariance)[(26057, 102, 26057, 4)]
    assert block[30, 31] == pytest.approx(-1.389399985782802e-3)
    assert block[31, 30] == pytest.approx(-4.971199855208397e-2)


def test_boxer_reader_roundtrip(tmp_path):
    source = make_covariance()
    path = tmp_path / "known.boxer"

    write_boxer(source, path, hlibid="TST", hdescr="known values", nvf=14)
    parsed = read_boxer(path)

    assert_covariance_equal(source, parsed)


def test_covfil_gendf_reader_roundtrip_and_metadata(tmp_path):
    source = make_covariance()
    path = tmp_path / "known.gendf"

    write_covfil(source, path, tape_label="known values")
    parsed = read_covfil(path)

    assert isinstance(parsed, CrossSectionCovariance)
    assert_covariance_equal(source, parsed)
    assert parsed.metadata["awr"] == pytest.approx(55.45443)
    assert parsed.metadata["temperature"] == pytest.approx(293.6)
    assert parsed.is_relative == [True, True]


def test_covfil_coverx_boxer_cross_format_chain(tmp_path):
    source = make_covariance()
    covfil_path = tmp_path / "source.gendf"
    coverx_path = tmp_path / "middle.coverx"
    boxer_path = tmp_path / "final.boxer"

    write_covfil(source, covfil_path)
    from_covfil = read_covfil(covfil_path)
    with pytest.warns(UserWarning, match="float32"):
        write_coverx(from_covfil, coverx_path, fmt="binary")
    from_coverx = read_coverx(coverx_path)
    write_boxer(from_coverx, boxer_path, nvf=14)
    from_boxer = read_boxer(boxer_path)

    assert_covariance_equal(
        source,
        from_boxer,
        rtol=2.0e-7,
        atol=2.0e-8,
    )


REAL_MF33 = Path(
    os.environ.get(
        "KIKA_TEST_FE56_MF33_GENDF",
        "/share_snc/snc/JuanMonleon/COV/cov/50/600/260560_50.06.xs.gendf",
    )
)
REAL_MF34 = Path(
    os.environ.get(
        "KIKA_TEST_FE56_MF34_GENDF",
        "/share_snc/snc/JuanMonleon/COV/mf34_processing/JEFF-4.0/tape21",
    )
)


@pytest.mark.skipif(not REAL_MF33.is_file(), reason="real Fe-56 MF33 GENDF unavailable")
def test_real_fe56_mf33_gendf_against_raw_reference_values():
    covariance = read_covfil(REAL_MF33)

    assert isinstance(covariance, CrossSectionCovariance)
    assert covariance.num_groups == 56
    # 150 blocks, plus MT 2 x MT 106 and MT 2 x MT 115, which state one entry each
    # (about 1e-11): small, not zero, so kept.
    assert covariance.num_matrices == 152
    assert covariance.metadata["awr"] == pytest.approx(55.45443)
    assert covariance.metadata["temperature"] == pytest.approx(600.0)
    np.testing.assert_allclose(
        [covariance.energy_grid[0], covariance.energy_grid[-1]],
        [1.0e-5, 2.0e7],
    )
    assert covariance.cross_sections[(FE56, 1)][0] == pytest.approx(23.58371)
    assert covariance.cross_sections[(FE56, 2)][0] == pytest.approx(14.79416)

    mt103 = matrix_map(covariance)[(FE56, 103, FE56, 103)]
    assert mt103[0, 0] == 0.0
    assert mt103[53, 53] == pytest.approx(4.583290e-3)
    assert mt103[55, 55] == pytest.approx(2.555789e-3)


@pytest.mark.skipif(not REAL_MF34.is_file(), reason="real Fe-56 MF34 GENDF unavailable")
def test_real_fe56_mf34_gendf_against_raw_reference_values():
    covariance = read_covfil(REAL_MF34)

    assert isinstance(covariance, LegendreCovariance)
    assert covariance.num_matrices == 1
    assert covariance.matrices[0].shape == (56, 56)
    assert covariance.matrices[0][0, 0] == pytest.approx(9.072000e-2)
    assert covariance.matrices[0][53, 53] == pytest.approx(9.374141e-5)
    assert covariance.matrices[0][55, 53] == pytest.approx(6.947195e-6)
    assert covariance.matrices[0][55, 55] == pytest.approx(2.321248e-4)


# ---------------------------------------------------------------------------
# Defects found while documenting the formats (11-oct-2026)
# ---------------------------------------------------------------------------

def _tiny_covariance(scale):
    """A block whose entries sum to about zero without being zero."""
    covariance = make_covariance()
    covariance.add_matrix(FE56, 2, FE56, 18, scale * np.array(
        [[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, -1.0, 1.0]]), is_relative=True)
    return covariance


@pytest.mark.parametrize("fmt", ["boxer", "covfil", "coverx_binary"])
def test_a_block_that_sums_to_nearly_zero_is_kept(tmp_path, fmt):
    source = _tiny_covariance(1.0e-9)
    path = tmp_path / f"tiny.{fmt}"
    if fmt == "boxer":
        write_boxer(source, str(path), nvf=14)
        back = read_boxer(str(path))
    elif fmt == "covfil":
        write_covfil(source, str(path))
        back = read_covfil(str(path))
    else:
        write_coverx(source, str(path), fmt="binary")
        back = read_coverx(str(path))
    assert (FE56, 2, FE56, 18) in matrix_map(back) or (FE56, 18, FE56, 2) in matrix_map(back)


@pytest.mark.parametrize("fmt", ["boxer", "covfil"])
def test_mev_converts_the_values_not_just_the_label(tmp_path, fmt):
    path = tmp_path / f"unit.{fmt}"
    if fmt == "boxer":
        write_boxer(make_covariance(), str(path), nvf=14)
        back = read_boxer(str(path), energy_unit="MeV")
    else:
        write_covfil(make_covariance(), str(path))
        back = read_covfil(str(path), energy_unit="MeV")
    assert back.energy_unit == "MeV"
    np.testing.assert_allclose(back.energy_grid, GRID_EV * 1e-6, rtol=1e-6)


def test_covfil_refuses_two_materials(tmp_path):
    source = make_covariance()
    source.add_matrix(92235, 18, 92235, 18, np.diag([0.01] * 4), is_relative=True)
    with pytest.raises(ValueError, match="one material"):
        write_covfil(source, str(tmp_path / "two.gendf"))


def test_to_covfil_keeps_the_temperature_it_read(tmp_path):
    first, second = tmp_path / "a.gendf", tmp_path / "b.gendf"
    write_covfil(make_covariance(), str(first))
    read_covfil(str(first)).to_covfil(str(second))
    assert read_covfil(str(second)).metadata["temperature"] == pytest.approx(293.6)


def test_a_mat_too_wide_for_its_field_is_refused(tmp_path):
    source = CrossSectionCovariance(num_groups=4, energy_grid=GRID_EV.tolist())
    source.add_matrix(1001001, 2, 1001001, 2, np.diag([0.01] * 4), is_relative=True)
    with pytest.raises(ValueError, match="MAT field"):
        write_covfil(source, str(tmp_path / "wide.gendf"))
    with pytest.raises(ValueError, match="MAT field"):
        write_boxer(source, str(tmp_path / "wide.boxer"), nvf=14)


def test_a_boxer_correlation_without_deviations_is_refused(tmp_path):
    from kika.cov.parse_covmat import InvalidDataFormatError

    path = tmp_path / "corr.boxer"
    write_boxer(make_covariance(), str(path), nvf=14)
    text = path.read_text().splitlines(keepends=True)
    out, skip = [], False
    for line in text:
        if line[:1].isdigit() and line[0] in "0123" and len(line) > 60:
            skip = line[0] == "2"
            line = ("4" + line[1:]) if line[0] == "3" else line
        if not skip:
            out.append(line)
    path.write_text("".join(out))
    with pytest.raises(InvalidDataFormatError, match="ITYPE 2"):
        read_boxer(str(path))


def test_project_to_grid_reads_an_unstated_class_as_relative():
    from types import SimpleNamespace

    covariance = CrossSectionCovariance()
    covariance.add_matrix(FE56, 2, FE56, 2, np.array([[0.01]]), energy_grid=[1.0, 10.0])
    sigma = SimpleNamespace(energies=np.array([0.1, 100.0]), values=np.array([2.0, 2.0]))
    projected = covariance.project_to_grid(np.array([1.0, 10.0]), xs_source=sigma)
    assert projected[0, 0] == pytest.approx(0.04)  # 0.01 relative x (2 b)^2


def test_a_written_covfil_carries_errorrs_tape_flag(tmp_path):
    """MF1/MT451 HEAD as ERRORR writes it: L1 = 6 (the ENDF format), N1 = -11
    (the flag of an MF33/MF34 covariance tape, which COVR checks)."""
    path = tmp_path / "flag.gendf"
    write_covfil(make_covariance(), path)
    head = path.read_text().splitlines()[1]
    assert head[70:75] == " 1451"
    assert [int(head[22 + 11 * i:33 + 11 * i]) for i in range(4)] == [6, 0, -11, 0]
    assert read_covfil(path).num_matrices == make_covariance().num_matrices
