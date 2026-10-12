import os
import textwrap

import numpy as np
import pandas as pd
import pytest

from kika.mcnp._fortran import fields, fortran_float, numbers
from kika.mcnp.parse_mctal import read_mctal

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Fortran reals
# ---------------------------------------------------------------------------

def test_fortran_reals_without_e_and_run_together():
    assert numbers("8.15942-132  1.00+002") == [8.15942e-132, 100.0]
    assert numbers("1.00E+00-2.00E+00") == [1.0, -2.0]
    assert numbers("1.0-2.0") == [1.0, -2.0]
    assert numbers("1.0D+00 .5 -3") == [1.0, 0.5, -3.0]
    assert fortran_float(" 4.53000-101 ") == 4.53e-101
    with pytest.raises(ValueError):
        fortran_float("1.0 2.0")
    out = fields("    Total       -8.000    -8.000 3.62090E-05")
    assert np.isnan(out[0]) and out[1:] == [-8.0, -8.0, 3.6209e-05]


# ---------------------------------------------------------------------------
# Real MCTAL files
# ---------------------------------------------------------------------------

def test_header_and_simple_tallies():
    mctal = read_mctal(_DATA + "/mctal/mctalfile_test_1.m")

    assert (mctal.code, mctal.version, mctal.probid) == ("mcnp6", "6", "02/22/25 15:50:25")
    assert (mctal.dump, mctal.nps, mctal.random_numbers) == (11, 100000, 140497964)
    assert mctal.title == "attenuation Sphere for PWR"
    assert mctal.npert == 0
    assert mctal.tally_numbers == [4, 14, 24]
    assert mctal.kcode is None

    tally = mctal[4]
    assert tally.id == 4 and tally.comment == "shell Detector"
    assert tally.particles == ["n"] and tally.kind == "cell flux"
    ds = tally.to_xarray()
    assert set(ds.result.dims) == {"cell", "energy"}
    np.testing.assert_array_equal(ds.energy, [0.1, 1.0, 3.0])
    np.testing.assert_array_equal(ds.energy_low, [0.0, 0.1, 1.0])
    np.testing.assert_array_equal(ds.result.values.ravel(), [2.95887e-09, 8.25405e-09, 8.62196e-10])
    np.testing.assert_array_equal(ds.error.values.ravel(), [0.1038, 0.0765, 0.1932])
    total = tally.total("energy")
    assert total.result.item() == 1.20751e-08 and total.error.item() == 0.0597

    assert list(tally.tfc.nps) == [8000, 16000, 24000, 32000, 40000, 48000, 56000, 64000,
                                   72000, 80000, 88000, 96000, 100000]
    assert tally.tfc.result[-1] == 1.20751e-08
    assert tally.tfc.error[-1] == 0.059727
    assert tally.tfc.fom[-1] == 782.933
    assert tally.perturbations == {}

    scalar = mctal[14]
    assert scalar.comment == "test Tally"
    assert scalar.energies is None
    ds = scalar.to_xarray()
    assert ds.result.item() == 1.35263e-05 and ds.error.item() == 0.0023
    with pytest.raises(ValueError, match="no total bin"):
        scalar.total("energy")


def test_perturbations():
    mctal = read_mctal(_DATA + "/pert/mctalfile_PERT_test_1.m")
    assert mctal.npert == 1760
    tally = mctal[4]
    assert len(tally.perturbations) == 1760

    pert = tally.perturbations[174]
    assert pert.number == 174 and pert.result.shape == tally.result.shape
    ds = tally.to_xarray(perturbation=174)
    np.testing.assert_array_equal(ds.result.values.ravel(), [2.65504e-09, 9.16002e-09, 1.06652e-09])
    np.testing.assert_array_equal(ds.error.values.ravel(), [0.0226, 0.0306, 0.0173])
    total = tally.total("energy", perturbation=174)
    assert total.result.item() == 1.28816e-08 and total.error.item() == 0.0223
    assert pert.tfc.nps[-1] == 10000000 and pert.tfc.fom[-1] == 1.57561
    with pytest.raises(KeyError):
        tally.to_xarray(perturbation=99999)


def test_segments_energy_totals_and_kcode():
    mctal = read_mctal(_DATA + "/mctal/mctalfile_test_2.m")
    assert mctal.title == "Example 3 MODEL"
    assert len(mctal) == 33 and mctal.tally_numbers[-1] == 20234
    with pytest.raises(KeyError):
        mctal[4]

    tally = mctal[10014]
    assert tally.comment == "Reaction rate 58Ni(n,p)  (Vol.) in (-1,+2)"
    ds = tally.to_xarray()
    # A single multiplier bin with no listed values is not a dimension.
    assert ds.result.dims == ("cell", "segment", "energy")
    assert dict(ds.sizes) == {"cell": 1, "segment": 7, "energy": 4}
    assert list(ds.cell.values) == [22002]
    np.testing.assert_array_equal(ds.energy, [0.1, 1.0, 3.0, 20.0])
    np.testing.assert_array_equal(
        ds.result.sel(segment=3).values.ravel(), [0.0, 214811.0, 26666300.0, 104354000.0])
    assert ds.result.sel(segment=2, energy=1.0).item() == 111063.0
    assert ds.error.sel(segment=2, energy=1.0).item() == 0.7287

    total = tally.total("energy")
    assert total.result.dims == ("cell", "segment")
    np.testing.assert_array_equal(
        total.result.values.ravel(),
        [6.06103e07, 6.62893e07, 1.31235e08, 8.36857e07, 2.48358e08, 9.33757e07, 9.92002e07])
    np.testing.assert_array_equal(
        total.error.values.ravel(), [0.6156, 0.6146, 0.283, 0.4887, 0.2721, 0.6997, 0.4681])

    df = tally.to_dataframe()
    assert len(df) == 28
    row = df[(df["segment"] == 1) & (df["energy"] == 1.0)]
    assert row["result"].item() == 150578.0 and row["error"].item() == 0.7105

    kcode = mctal.kcode
    assert (kcode.cycles, kcode.settle) == (110, 10)
    assert kcode.data.shape == (110, 19)
    assert kcode.keff == (0.994321, 0.00328746)
    frame = kcode.to_dataframe()
    assert frame.index[0] == 1 and frame["keff_collision"].iloc[0] == 1.01177


# ---------------------------------------------------------------------------
# Layouts written from the MCTAL format description (MCNP 6.3.1 manual,
# section 6.3.4): detectors, cosine/time bins, radiographs, TMESH.
# ---------------------------------------------------------------------------

_PARTICLES_N = " 1" + " 0" * 36
_PARTICLES_P = " 0 1" + " 0" * 35


def _write(tmp_path, body):
    path = tmp_path / "mctal"
    path.write_text(textwrap.dedent(body).lstrip("\n"))
    return path


def test_detector_without_cell_list_and_time_total(tmp_path):
    path = _write(tmp_path, f"""
        mcnp6      6     01/01/25 00:00:00     2          1000           12345
         detector test
        ntal     1
            5
        tally    5                   -1    1    0
        {_PARTICLES_N}
        f        1
        d        2
        u        0
        s        0
        m        0
        c        0
        e        0
        tt       3
          1.00000E+01  2.00000E+01
        vals
          1.00000E-01 0.1000  2.00000E-01 0.2000  3.00000E-01 0.3000  4.00000E-01 0.4000
          5.00000E-01 0.5000  6.00000E-01 0.6000
        tfc    1       1       1       1       1       1       1       1       3
                   1000  3.00000E-01  3.00000E-01  1.00000E+00
        """)
    tally = read_mctal(path)[5]
    assert tally.kind == "point detector" and tally.detector_type == 1
    ds = tally.to_xarray()
    assert ds.result.dims == ("detector", "flag", "time")
    assert list(ds.detector.values) == [1] and list(ds.flag.values) == [1, 2]
    np.testing.assert_array_equal(ds.time, [10.0, 20.0])
    assert np.isneginf(ds.time_low.values[0])
    np.testing.assert_array_equal(ds.result.values.ravel(), [0.1, 0.2, 0.4, 0.5])
    np.testing.assert_array_equal(tally.total("time").result.values.ravel(), [0.3, 0.6])
    assert tally.tfc.bins["t"] == 2


def test_cosine_bins_mcnp5_particles_and_three_digit_exponents(tmp_path):
    path = _write(tmp_path, """
        mcnp5      5     01/01/25 00:00:00     1          1000           12345
         cosine test
        ntal     1
            1
        tally    1                    2    0
        f        2
              10       20
        d        1
        u        0
        s        0
        m        0
        ct       3
         -5.00000E-01  1.00000E+00
        e        0
        t        0
        vals
          1.00000-101 0.1000  2.00000E-01 0.2000  3.00000E-01 0.3000  4.00000E-01 0.4000
          5.00000E-01 0.5000  6.00000E-01 0.6000
        tfc    1       1       1       1       1       1       1       1       1
                   1000  1.00000E-01  1.00000E-01  1.00000E+00
        """)
    mctal = read_mctal(path)
    assert (mctal.code, mctal.version) == ("mcnp5", "5")
    tally = mctal[1]
    assert tally.particles == ["p"] and tally.region_name == "surface"
    ds = tally.to_xarray()
    assert ds.result.dims == ("surface", "cosine")
    np.testing.assert_array_equal(ds.cosine, [-0.5, 1.0])
    np.testing.assert_array_equal(ds.cosine_low, [-1.0, -0.5])
    assert ds.result.values[0, 0] == 1e-101
    np.testing.assert_array_equal(tally.total("cosine").result.values, [0.3, 0.6])


def test_radiograph_grid_edges(tmp_path):
    path = _write(tmp_path, f"""
        mcnp6      6     01/01/25 00:00:00     2             1               2
         radiograph test
        ntal     1
            5
        tally    5                   -1    4    0
        {_PARTICLES_P}
             Sample Radiograph
        f        1
        d        2
        u        0
        s        2
         -1.00000E+00  0.00000E+00  1.00000E+00
        m        0
        c        3
         -3.00000E+00 -1.00000E+00  1.00000E+00  3.00000E+00
        e        0
        t        0
        vals
          1.0E+00 0.0000  2.0E+00 0.0000  3.0E+00 0.0000  4.0E+00 0.0000
          5.0E+00 0.0000  6.0E+00 0.0000  7.0E+00 0.0000  8.0E+00 0.0000
          9.0E+00 0.0000  1.0E+01 0.0000  1.1E+01 0.0000  1.2E+01 0.0000
        tfc    1       1       1       1       1       1       1       1       1
                      1  1.00000E+00  0.00000E+00  0.00000E+00
        """)
    tally = read_mctal(path)[5]
    assert tally.is_radiograph and tally.comment == "Sample Radiograph"
    ds = tally.to_xarray()
    assert ds.result.dims == ("detector", "flag", "image_s", "image_t")
    np.testing.assert_array_equal(ds.image_s_low, [-1.0, 0.0])
    np.testing.assert_array_equal(ds.image_s, [0.0, 1.0])
    np.testing.assert_array_equal(ds.image_t_low, [-3.0, -1.0, 1.0])
    np.testing.assert_array_equal(ds.image_t, [-1.0, 1.0, 3.0])
    assert ds.result.isel(flag=1, image_s=0, image_t=2).item() == 9.0


def test_tmesh_cora_runs_fastest(tmp_path):
    # 3 x 2 x 1 rectangular mesh; values numbered in file order.
    path = _write(tmp_path, f"""
        mcnp6      6     01/01/25 00:00:00     2          1000           12345
         tmesh test
        ntal     1
            1
        tally    1   -1   -1
        {_PARTICLES_N}
        f        6    0    3    2    1
          0.00000E+00  1.00000E+00  2.00000E+00  3.00000E+00
          0.00000E+00  1.00000E+00  2.00000E+00
          0.00000E+00  5.00000E+00
        d        1
        u        1
        s        1
        m        1
        c        1
        e        1
        t        1
        vals
          1.0E+00 0.1000  2.0E+00 0.1000  3.0E+00 0.1000  4.0E+00 0.1000
          5.0E+00 0.1000  6.0E+00 0.1000
        """)
    tally = read_mctal(path)[1]
    assert tally.kind == "mesh" and tally.mesh_geometry == "rectangular"
    ds = tally.to_xarray()
    assert ds.result.dims == ("cora", "corb", "corc")
    assert dict(ds.sizes) == {"cora": 3, "corb": 2, "corc": 1}
    np.testing.assert_array_equal(ds.cora, [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(ds.result.values[:, :, 0], [[1, 4], [2, 5], [3, 6]])
    assert tally.tfc is None


def test_value_count_is_checked(tmp_path):
    path = _write(tmp_path, f"""
        mcnp6      6     01/01/25 00:00:00     2          1000           12345
         short
        ntal     1
            4
        tally    4                   -1    0    0
        {_PARTICLES_N}
        f        1
              10
        d        1
        u        0
        s        0
        m        0
        c        0
        e        2
          1.00000E+00  2.00000E+00
        t        0
        vals
          1.0E+00 0.1000
        """)
    with pytest.raises(ValueError, match="result/error pairs"):
        read_mctal(path)


def test_tally_listed_but_missing(tmp_path):
    path = _write(tmp_path, """
        mcnp6      6     01/01/25 00:00:00     2          1000           12345
         truncated
        ntal     2
            4   14
        """)
    with pytest.raises(ValueError, match=r"not found: \[4, 14\]"):
        read_mctal(path)


def test_tfc_plot_and_frame():
    tally = read_mctal(_DATA + "/mctal/mctalfile_test_1.m")[4]
    frame = tally.tfc.to_dataframe()
    assert list(frame.columns) == ["nps", "result", "error", "fom"] and len(frame) == 13
    import matplotlib
    matplotlib.use("Agg")
    fig = tally.tfc.plot()
    assert len(fig.axes) == 3


def test_dataframe_of_scalar_tally():
    df = read_mctal(_DATA + "/mctal/mctalfile_test_1.m")[14].to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.to_dict("list") == {"cell": [4], "result": [1.35263e-05], "error": [0.0023]}


def test_mcnp63_union_and_total_cell_bins():
    """F4 1 2 (1 2) T: MCNP writes 0 for the union bin and for the total, so
    the bins are indexed and the numbers kept in cell_id (values checked
    against the MCNP 6.3.0 output listing)."""
    mctal = read_mctal(_DATA + "/mctal/mcnp630_check.m")
    tally = mctal[4]
    ds = tally.to_xarray()
    assert list(ds.cell.values) == [1, 2, 3, 4] and list(ds.cell_id.values) == [1, 2, 0, 0]
    np.testing.assert_array_equal(ds.multiplier, [1, 2])
    np.testing.assert_array_equal(
        ds.result.sel(cell=1, multiplier=1).values, [4.17322e-03, 7.72611e-04, 9.86075e-04])
    np.testing.assert_array_equal(
        ds.result.sel(cell=3, multiplier=1).values, [7.62492e-04, 2.40679e-04, 2.21262e-04])
    assert tally.total("energy").result.sel(cell=1, multiplier=1).item() == 5.93191e-03

    user = mctal[24].to_xarray()           # FT INC + FU 0 1 2 10 T
    np.testing.assert_array_equal(user.user, [0, 1, 2, 10])
    assert user.result.sel(user=0).item() == 1.69313e-05
    assert mctal[21].to_xarray().result.dims == ("surface", "segment")
    assert mctal[5].to_xarray().result.dims == ("detector", "flag", "energy")
