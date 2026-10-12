import os
import textwrap

import h5py
import numpy as np
import pytest

from kika import identify
from kika.mcnp import detect_mcnp_file_type, read_meshtal

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "meshtal")
TEXT = os.path.join(_DATA, "mcnptools_col_cf.msht")
RUNTAPE = os.path.join(_DATA, "mcnp631_xdmf_crop.h5")


# ---------------------------------------------------------------------------
# MESHTAL text (MCNP6, out=col and out=cf)
# ---------------------------------------------------------------------------

def test_text_header_and_tallies():
    meshtal = read_meshtal(TEXT)
    assert meshtal.source == "text"
    assert (meshtal.code, meshtal.version, meshtal.build_date) == ("mcnp", "6", "06/01/17")
    assert meshtal.probid == "06/07/17 09:54:35"
    assert meshtal.title.startswith("Input file to generate Mctal")
    assert meshtal.nps == 225000.0
    assert list(meshtal.tallies) == [14, 24, 34, 44, 54, 64, 74, 84]
    assert [t.geometry for t in meshtal] == ["rectangular"] * 4 + ["cylindrical"] * 4
    assert all(t.particle == "neutron" for t in meshtal)


def test_text_rectangular_energy_time_cf():
    tally = read_meshtal(TEXT)[44]
    np.testing.assert_array_equal(tally.bounds["x"], [-10, -6, -2, 2, 6, 10])
    np.testing.assert_array_equal(tally.energy_bounds, [0.0, 1e-9, 1e-3, 1e3])
    np.testing.assert_array_equal(tally.time_bounds, [-1e36, 0.0, 10.0, 20.0])
    assert tally.has_energy_total and tally.has_time_total
    assert tally.result.shape == (4, 4, 5, 5, 5)

    ds = tally.to_xarray()
    assert ds.result.dims == ("energy", "time", "x", "y", "z")
    np.testing.assert_array_equal(ds.x, [-8, -4, 0, 4, 8])
    np.testing.assert_array_equal(ds.x_low, [-10, -6, -2, 2, 6])
    np.testing.assert_array_equal(ds.energy, [1e-9, 1e-3, 1e3])
    np.testing.assert_array_equal(ds.time_low, [-1e36, 0.0, 10.0])
    cell = ds.sel(energy=1e-3, time=10.0, x=-4, y=0, z=4)
    assert cell.result.item() == 1.14306e-04 and cell.error.item() == 2.56109e-02

    # "Total  1.000E+01 -8 -8 -4" and "Total Total 0 0 0" rows of the file
    by_time = tally.total("energy").sel(time=10.0, x=-8, y=-8, z=-4)
    assert by_time.result.item() == 2.27838e-05 and by_time.error.item() == 4.96904e-02
    both = tally.total("energy", "time").sel(x=0, y=0, z=0)
    assert both.result.item() == 1.01068e-02 and both.error.item() == 4.13491e-03

    # The CF Volume column agrees with the volumes computed from the bounds.
    np.testing.assert_allclose(tally.attrs["volume"], tally.volumes, rtol=1e-5)


def test_text_cylindrical():
    meshtal = read_meshtal(TEXT)
    plain = meshtal[54]
    np.testing.assert_array_equal(plain.origin, [0.0, -10.0, 0.0])
    np.testing.assert_array_equal(plain.axis, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(plain.vec, [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(plain.bounds["theta"], [0, 0.25, 0.5, 0.75, 1.0])
    ds = plain.to_xarray()
    assert ds.result.dims == ("r", "z", "theta")
    assert ds.sel(r=1.25, z=1.0, theta=0.875).result.item() == 1.33402e-03
    assert ds.sel(r=1.25, z=3.0, theta=0.125).result.item() == 9.33638e-04

    full = meshtal[84]
    cell = full.to_xarray().sel(energy=1e3, time=20.0, r=6.25, z=7.0, theta=0.625)
    assert cell.result.item() == 0.0
    total = full.total("energy", "time").sel(r=1.25, z=1.0, theta=0.125)
    assert total.result.item() == 8.14945e-04 and total.error.item() == 3.06133e-02
    np.testing.assert_allclose(full.attrs["volume"], full.volumes, rtol=1e-5)


def test_total_of_an_unbinned_axis_is_the_axis():
    tally = read_meshtal(TEXT)[14]
    assert tally.n_energy == 1 and not tally.has_energy_total
    a, b = tally.total("energy"), tally.to_xarray()
    np.testing.assert_array_equal(a.result.values, b.result.values)
    with pytest.raises(ValueError, match="totals exist for energy and time"):
        tally.total("x")


def _write(tmp_path, body, name="meshtal"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body).lstrip("\n"))
    return path


def test_text_colsci_and_mcnp5_particle_line(tmp_path):
    # out=colsci (MCNP 6.3): coordinates in scientific notation, MCNP5-style
    # particle line, an FC comment and a three-digit exponent without E.
    path = _write(tmp_path, """
        mcnp   version 6.3   ld=01/26/23  probid =  01/01/25 00:00:00
         colsci test
         Number of histories used for normalizing tallies =             10.00

         Mesh Tally Number         4
             my comment
         This is a photon mesh tally.

         Tally bin boundaries:
            X direction:  0.00000E+00  1.00000E+00  3.00000E+00
            Y direction:  0.00000E+00  1.00000E+00
            Z direction:  0.00000E+00  1.00000E+00
            Energy bin boundaries: 0.00000E+00 1.00000E+36

                   X            Y            Z         Result      Rel Error
          5.00000E-01  5.00000E-01  5.00000E-01  1.00000E+00  1.00000E-01
          2.00000E+00  5.00000E-01  5.00000E-01  2.50000-120  2.00000E-01
        """)
    tally = read_meshtal(path)[4]
    assert tally.particle == "photon" and tally.comment == "my comment"
    ds = tally.to_xarray()
    np.testing.assert_array_equal(ds.x, [0.5, 2.0])
    np.testing.assert_array_equal(ds.result.values.ravel(), [1.0, 2.5e-120])


def test_text_row_order_is_checked(tmp_path):
    path = _write(tmp_path, """
         wrong order
         Number of histories used for normalizing tallies =             10.00

         Mesh Tally Number         4
         neutron  mesh tally.

         Tally bin boundaries:
            X direction:      0.00      1.00      2.00
            Y direction:      0.00      1.00      2.00
            Z direction:      0.00      1.00
            Energy bin boundaries: 0.00E+00 1.00E+36

                X         Y         Z     Result     Rel Error
              0.500     0.500     0.500 1.00000E+00 1.00000E-01
              1.500     0.500     0.500 2.00000E+00 1.00000E-01
              0.500     1.500     0.500 3.00000E+00 1.00000E-01
              1.500     1.500     0.500 4.00000E+00 1.00000E-01
        """)
    with pytest.raises(ValueError, match="unexpected row order"):
        read_meshtal(path)


def test_text_matrix_layout_is_refused(tmp_path):
    path = _write(tmp_path, """
         matrix
         Number of histories used for normalizing tallies =             10.00

         Mesh Tally Number         4
         neutron  mesh tally.

         Tally bin boundaries:
            X direction:      0.00      1.00
            Y direction:      0.00      1.00
            Z direction:      0.00      1.00
            Energy bin boundaries: 0.00E+00 1.00E+36

        Energy Bin:  0.00E+00 - 1.00E+36 MeV
        """)
    with pytest.raises(NotImplementedError, match="out=ij/ik/jk"):
        read_meshtal(path)


# ---------------------------------------------------------------------------
# HDF5 runtape (MCNP 6.3.1, out=xdmf)
# ---------------------------------------------------------------------------

def test_runtape_rectangular():
    meshtal = read_meshtal(RUNTAPE)
    assert meshtal.source == "hdf5"
    assert (meshtal.code, meshtal.version, meshtal.build_date) == ("mcnp6", "6.3.1", "02/08/25")
    tally = meshtal[4]
    assert tally.geometry == "rectangular" and tally.particle == "neutron"
    assert tally.comment == "Rectangular mesh tally"
    assert tally.attrs["tally_quantity"] == "track_length_particle_flux"
    assert tally.shape == (5, 4, 3)
    np.testing.assert_array_equal(tally.bounds["x"], [350, 360, 370, 380, 390, 400])

    with h5py.File(RUNTAPE) as h5:
        raw = h5["results/mesh_tally/mesh_tally_4/mean"][()]
    ds = tally.to_xarray()
    assert ds.result.dims == ("x", "y", "z")
    # stored (time, energy, z, y, x)
    np.testing.assert_array_equal(ds.result.values, raw[0, 0].transpose(2, 1, 0))
    assert ds.result.sel(x=355.0, y=-15.0, z=5.0).item() == raw[0, 0, 0, 0, 0]
    assert ds.result.sel(x=395.0, y=15.0, z=25.0).item() == raw[0, 0, 2, 3, 4]


def _runtape(tmp_path, n_time, n_energy, totals):
    """A runtape laid out as MCNP 6.3 documents it (appendix D.4): cylindrical,
    theta in radians, data (time, energy, k, j, i). Values encode their index."""
    r, z, t = np.array([0.0, 1, 2]), np.array([0.0, 5, 10, 15]), np.linspace(0, 2 * np.pi, 5)
    nt, ne = n_time + totals, n_energy + totals
    idx = np.indices((nt, ne, len(t) - 1, len(z) - 1, len(r) - 1))
    mean = (idx[0] * 10000 + idx[1] * 1000 + idx[2] * 100 + idx[3] * 10 + idx[4]).astype(float)
    path = tmp_path / "runtpe.h5"
    with h5py.File(path, "w") as h5:
        g = h5.create_group("results/mesh_tally/mesh_tally_24")
        g.attrs["coordinate_system"] = np.bytes_(b"cylindrical")
        g.attrs["particle_number"] = 2
        g.attrs["number_of_normalizing_histories"] = 1000
        g["grid_r"], g["grid_z"], g["grid_t"] = r, z, t
        g["grid_energy"] = np.linspace(0, 20, n_energy + 1)
        g["grid_time"] = np.linspace(0, 30, n_time + 1)
        g["basis_axis"], g["basis_vector"] = [0.0, 0, 1], [1.0, 0, 0]
        g["mean"] = mean
        g["relative_standard_error"] = mean / 1e5
    return path


def test_runtape_cylindrical_layout_and_totals(tmp_path):
    meshtal = read_meshtal(_runtape(tmp_path, n_time=2, n_energy=3, totals=True))
    assert meshtal.nps == 1000.0
    tally = meshtal[24]
    assert tally.geometry == "cylindrical" and tally.particle == "photon"
    np.testing.assert_allclose(tally.bounds["theta"], [0, 0.25, 0.5, 0.75, 1.0])
    assert tally.has_energy_total and tally.has_time_total
    ds = tally.to_xarray()
    assert ds.result.dims == ("energy", "time", "r", "z", "theta")
    # energy bin 2, time bin 1, r bin 1, z bin 2, theta bin 3
    value = ds.result.isel(energy=2, time=1, r=1, z=2, theta=3).item()
    assert value == 1 * 10000 + 2 * 1000 + 3 * 100 + 2 * 10 + 1
    assert tally.total("energy", "time").result.isel(r=0, z=0, theta=0).item() == 2 * 10000 + 3 * 1000


def test_runtape_without_mesh_results(tmp_path):
    path = tmp_path / "runtpe.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("variable")
    with pytest.raises(ValueError, match="out=xdmf"):
        read_meshtal(path)


# ---------------------------------------------------------------------------
# File-type detection
# ---------------------------------------------------------------------------

def test_detection():
    assert identify(TEXT) == "mcnp-meshtal"
    assert identify(RUNTAPE) == "mcnp-meshtal"
    assert detect_mcnp_file_type(TEXT) == "mcnp-meshtal"
