"""TSL materials in every format kika reads: GNDS, ACE thermal, OpenMC HDF5, G4NDL.

Each format is built small here, the way its writer lays it out, and read
back through :func:`kika.tsl.read_tsl`, :func:`kika.read` and the library
index. ENDF/B-VIII.1 in all five formats is reached under ``--deep``, where
Y in YH₂ (coherent, incoherent and inelastic) must give the same elastic σ
from every one of them.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.tsl import index_tsl
from kika.tsl.ace import read_ace_thermal, read_ace_thermal_header
from kika.tsl.g4ndl import g4ndl_thermal_naming, read_g4ndl_thermal
from kika.tsl.naming import tsl_naming_from_openmc, tsl_naming_from_title
from kika.tsl.read import detect_tsl_format, read_tsl

ENDF_DATA = Path(__file__).parents[2] / "endf" / "tests" / "data"

EDGES = np.array([0.004, 0.006])          # eV
S_CUM = np.array([0.006, 0.010])          # eV*b


# ---------------------------------------------------------------- ACE

def _ace_lines(values, per_line=4):
    return ["".join(f"{v:20.11E}" for v in values[i:i + per_line]) for i in range(0, len(values), per_line)]


def write_ace_thermal(path: Path, kt_mev: float = 2.53e-8, title: str = "Al metal - TEST - 293.6 K"):
    """A coherent-elastic table with a continuous inelastic block (IFENG = 2)."""
    incident = [1e-11, 1e-6]                  # MeV
    sigma_in = [5.0, 1.0]
    n_mu = 2
    xss = [len(incident)] + incident + sigma_in           # ITIE = 1
    itxe = len(xss) + 1
    records = []
    for e in incident:
        records.append([e * 0.5, 1.0e6, 0.5, -0.5, 0.5])
        records.append([e * 2.0, 1.0e6, 1.0, -0.2, 0.7])
    head = len(incident) * 2
    locators, start = [], itxe + head
    for rec in records[0::2]:
        locators.append(start - 1)
        start += 2 * (n_mu + 3)
    xss += locators + [2, 2]
    for rec in records:
        xss += rec
    itce = len(xss) + 1
    xss += [2] + list(EDGES / 1e6) + list(S_CUM / 1e6)
    nxs = [len(xss), 3, n_mu + 1, 2, 4, -1, 2, 0] + [0] * 8
    jxs = [1, 1 + 1 + len(incident), itxe, itce, itce + 1 + 2, 0] + [0] * 26
    lines = [f"  al-27.70t   26.749750  {kt_mev:10.4E}   01/01/26",
             f"{title:<70}   mat1325",
             "  13027         0.      0         0.      0         0.      0         0."]
    lines += ["      0         0.      0         0.      0         0.      0         0."] * 3
    lines += ["".join(f"{v:9d}" for v in nxs[i:i + 8]) for i in (0, 8)]
    lines += ["".join(f"{v:9d}" for v in jxs[i:i + 8]) for i in (0, 8, 16, 24)]
    lines += _ace_lines(xss)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_an_ace_thermal_table_is_read_into_the_model(tmp_path):
    path = write_ace_thermal(tmp_path / "al-27.70t")
    header = read_ace_thermal_header(path)
    assert (header.name, header.library, header.temperature) == ("Al metal", "TEST", 293.6)
    assert (header.elastic_mode, header.outgoing_kind) == ("coherent", "continuous")

    material = read_tsl(path)
    assert material.format == "ace" and material.is_processed
    bragg = material.bragg(293.6)
    np.testing.assert_allclose(bragg.edges, EDGES)
    np.testing.assert_allclose(bragg.s, S_CUM)
    sigma = material.elastic(293.6, [0.003, 0.005, 0.02])["coherent"]
    np.testing.assert_allclose(sigma, [0.0, 0.006 / 0.005, 0.010 / 0.02])
    energies, xs = material.inelastic(293.6)
    np.testing.assert_allclose(energies, [1e-5, 1.0])
    np.testing.assert_allclose(xs, [5.0, 1.0])
    table = material.outgoing(293.6)
    assert table.kind == "continuous" and table.cosines[1].shape == (2, 2)
    np.testing.assert_allclose(table.energies[1], [0.5, 2.0])
    assert table.probabilities(1).sum() == pytest.approx(1.0)

    suite = kika.read(str(path))
    assert suite.interaction == "thermalNeutronScatteringLaw"
    assert [r.label for r in suite.reactions] == ["coherent-elastic", "incoherent-inelastic"]
    assert sorted(list(suite.reactions)[1].crossSection) == ["heated_000"]


def test_ace_tables_of_one_material_are_one_entry_with_every_temperature(tmp_path):
    write_ace_thermal(tmp_path / "al-27.70t", 2.53e-8, "Al metal - TEST - 293.6 K")
    write_ace_thermal(tmp_path / "al-27.71t", 5.17e-8, "Al metal - TEST - 600.0 K")
    (tmp_path / "xsdir").write_text("al-27.70t 26.7 al-27.70t 0 1 1 10 0 0 2.53E-08\n")
    index = index_tsl(tmp_path)
    assert len(index.tapes) == 1 and index.unreadable == []
    tape = index.tapes[0]
    assert (tape.format, tape.compound, tape.temperatures) == ("ace", "Al metal", [293.6, 600.0])
    assert tape.key == "ace:al-27" and len(tape.paths) == 2
    material = read_tsl(tape.paths)
    assert material.temperatures()["coherent"] == [293.6, 600.0]


# ---------------------------------------------------------------- OpenMC

def write_openmc_thermal(path: Path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        f.attrs["filetype"] = np.bytes_(b"data_thermal")
        f.attrs["version"] = np.array([3, 0])
        g = f.create_group("c_Y_in_YH2")
        g.attrs["atomic_weight_ratio"] = 88.1421
        g.attrs["energy_max"] = 4.9
        g.attrs["nuclides"] = np.array([b"Y89"])
        g.create_group("kTs").create_dataset("294K", data=293.6 * 8.617333262e-5)
        t = g.create_group("294K")
        el = t.create_group("elastic")
        xs = el.create_group("xs")
        xs.attrs["type"] = np.bytes_(b"Sum")
        xs.create_dataset("func_1", data=np.vstack([EDGES, S_CUM])).attrs["type"] = np.bytes_(b"CoherentElastic")
        xs.create_dataset("func_2", data=[0.15, 2.5]).attrs["type"] = np.bytes_(b"IncoherentElastic")
        el.create_group("distribution").attrs["type"] = np.bytes_(b"mixed_elastic")
        inel = t.create_group("inelastic")
        inel.create_dataset("xs", data=np.array([[1e-5, 1.0], [5.0, 1.0]]))
        d = inel.create_group("distribution")
        d.attrs["type"] = np.bytes_(b"incoherent_inelastic")
        d.create_dataset("energy", data=[1e-5, 1.0])
        eout = np.array([[0.5e-5, 2e-5, 0.5, 2.0],      # E'
                         [1.0, 1.0, 1.0, 1.0],          # p
                         [0.5, 1.0, 0.5, 1.0],          # c
                         [0, 0, 0, 0],                  # mu interpolation
                         [0, 2, 4, 6]])                 # mu offsets
        ds = d.create_dataset("energy_out", data=eout)
        ds.attrs["offsets"] = np.array([0, 2])
        ds.attrs["interpolation"] = np.array([2, 2])
        ds.attrs["n_discrete_lines"] = np.array([0, 0])
        d.create_dataset("mu", data=np.tile([[-0.5, 0.5], [0.5, 0.5], [0.5, 1.0]], (1, 4)))
    return path


def test_an_openmc_thermal_file_keeps_the_exact_incoherent_parameters(tmp_path):
    path = write_openmc_thermal(tmp_path / "c_Y_in_YH2.h5")
    assert detect_tsl_format(path) == "openmc"
    material = read_tsl(path)
    block = material.processed.temperatures[0]
    assert block.temperature == 293.6 and block.incoherent.exact
    sigma = material.elastic(293.6, [1e-8])["incoherent"]
    assert sigma[0] == pytest.approx(0.15, rel=1e-6)          # σ_b at E → 0
    table = material.outgoing(293.6)
    assert [len(e) for e in table.energies] == [2, 2]
    np.testing.assert_allclose(table.cosines[1][0], [-0.5, 0.5])
    # kika.read opens OpenMC files on the feat/hdf5 branch (kika.hdf5); here the
    # model comes through read_tsl.
    suite = material.suite
    labels = [r.label for r in suite.reactions]
    assert labels == ["coherent-elastic", "incoherent-elastic", "incoherent-inelastic"]
    form = list(suite.reactions)[1].doubleDifferentialCrossSection["heated_000"]
    assert form.boundAtomCrossSection.value == pytest.approx(0.15)
    tape = index_tsl(path).tapes[0]
    assert (tape.compound, tape.principal, tape.elastic, tape.bragg_edges) == ("YH2", "Y", "both", 2)


# ---------------------------------------------------------------- G4NDL

def _g4(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zlib.compress(("        G4NDL\nTEST_SOURCE\n" + body).encode()))


def write_g4ndl_thermal(root: Path) -> Path:
    thermal = root / "ThermalScattering"
    for t in (293.6, 600.0):
        pass
    _g4(thermal / "Inelastic" / "CrossSection" / "al_metal.z",
        "3\n248\n293.6\n2\n1e-5 5.0 1.0 1.0\n3\n248\n600.0\n2\n1e-5 6.0 1.0 1.5\n")
    record = "0 1e-5 0 0 8 4\n0 0 0 0\n2e-5 1e5 -0.5 0.5\n"
    record2 = "0 1.0 0 0 8 4\n0 0 0 0\n2.0 1.0 -0.2 0.6\n"
    _g4(thermal / "Inelastic" / "FS" / "al_metal.z",
        "6\n248\n293.6\n2\n" + record + record2 + "6\n248\n600.0\n2\n" + record + record2)
    _g4(thermal / "Coherent" / "CrossSection" / "al_metal.z", "3\n244\n293.6\n1\n1e-5 0.0\n")
    _g4(thermal / "Coherent" / "FS" / "al_metal.z",
        "7\n2\n293.6\n2\n0.004 0.006 0.006 0.010\n7\n2\n600.0\n2\n0.005 0.009\n")
    return thermal


def test_a_g4ndl_thermal_folder_is_a_library_of_materials(tmp_path):
    root = tmp_path / "G4NDL"
    write_g4ndl_thermal(root)
    (root / "Inelastic" / "CrossSection").mkdir(parents=True)   # the neutron data beside it
    assert detect_tsl_format(root / "ThermalScattering") == "g4ndl"
    index = index_tsl(root)
    assert [t.key for t in index.tapes] == ["g4ndl:al_metal"]
    tape = index.tapes[0]
    assert (tape.compound, tape.elastic, tape.temperatures, tape.bragg_edges) == \
        ("Al metal", "coherent", [293.6, 600.0], 2)
    material = read_tsl(root, member="al_metal")
    np.testing.assert_allclose(material.bragg(600.0).edges, EDGES)
    np.testing.assert_allclose(material.bragg(600.0).s, [0.005, 0.009])
    table = material.outgoing(600.0)
    # Every row is kept, E' = 0 too: Geant4's cdf is the trapezoid rule over all of them.
    assert [len(e) for e in table.energies] == [2, 2]
    np.testing.assert_allclose(table.cdf[0], [0.0, 1.0])
    suite = kika.read(str(root), format="g4ndl", target="al_metal")
    assert len(suite.reactions) == 2


@pytest.mark.parametrize("name, principal, compound, variant", [
    ("h_water", "H", "H2O", None),
    ("si28_sio2_alpha", "Si28", "SiO2", "alpha"),
    ("graphite_10p", "C", "Graphite", "10P"),
    ("al_metal", "Al", "Al metal", None),
    ("mg24_magnesium", "Mg24", "Mg", None),
    ("zr90_zrh", "Zr90", "ZrH", None),
])
def test_g4ndl_names(name, principal, compound, variant):
    assert g4ndl_thermal_naming(name)[1:] == (principal, compound, variant)


@pytest.mark.parametrize("title, principal, compound, variant", [
    ("H in H2O", "H", "H2O", None),
    ("U in UO2 (19.75% U235, HALEU)", "U", "UO2", "HALEU"),
    ("U metal (5% U235)", "U", "U metal", "5P"),
    ("C in reactor graphite - 10% porosity", "C", "Graphite", "reactor 10P"),
    ("Be metal with Sd effects", "Be", "Be metal", "Sd"),
    ("ortho D", "D", "D2", "ortho"),
])
def test_ace_titles(title, principal, compound, variant):
    naming = tsl_naming_from_title(title, "?")
    assert (naming.principal, naming.compound, naming.variant) == (principal, compound, variant)


def test_openmc_group_names():
    naming = tsl_naming_from_openmc("c_O_in_UO2_5p")
    assert (naming.principal, naming.compound, naming.variant) == ("O", "UO2", "5P")
    assert tsl_naming_from_openmc("c_Al27").compound == "Al metal"


# ---------------------------------------------------------------- GNDS

def test_a_gnds_tsl_file_is_listed_from_its_head(tmp_path):
    suite = kika.read(str(ENDF_DATA / "micro_tsl_un_elastic.endf"))
    path = tmp_path / "tsl-NinUN.endf.gnds.xml"
    kika.write(suite, str(path), format="gnds")
    assert detect_tsl_format(path) == "gnds"
    tape = index_tsl(tmp_path).tapes[0]
    assert (tape.format, tape.principal, tape.compound, tape.elastic) == ("gnds", "N", "UN", "both")
    material = read_tsl(path)
    assert material.temperatures()["coherent"] == tape.elastic_temperatures


# ---------------------------------------------------------------- full size

def _nuclear_data(tape: Path) -> Path:
    for parent in Path(tape).parents:
        if (parent / "openmc").is_dir() and (parent / "gnds").is_dir():
            return parent
    pytest.skip("NuclearData (openmc/ and gnds/) is not next to the TSL tapes")


def test_y_in_yh2_is_the_same_in_every_format(tsl_be_metal_tape):
    data = _nuclear_data(tsl_be_metal_tape)
    sources = [
        (Path(tsl_be_metal_tape).parent / "tsl-YinYH2.endf", {}),
        (data / "gnds" / "endfb81" / "thermal_scatt" / "tsl-YinYH2.endf.gnds.xml", {}),
        (data / "ace" / "endf81sab" / "y-yh2.70t", {}),
        (data / "openmc" / "endfb81" / "thermal" / "c_Y_in_YH2.h5", {}),
    ]
    energies = np.array([1e-3, 1e-2, 0.0253, 0.1, 1.0])
    reference = None
    for source, kwargs in sources:
        if not source.exists():
            pytest.skip(f"{source} is not on this machine")
        elastic = read_tsl(source, **kwargs).elastic(293.6, energies)
        total = elastic["coherent"] + elastic["incoherent"]
        if reference is None:
            reference = total
        np.testing.assert_allclose(total, reference, rtol=2e-3)


def test_every_format_of_endf_b_viii_1_is_indexed(tsl_be_metal_tape):
    data = _nuclear_data(tsl_be_metal_tape)
    expected = {data / "gnds" / "endfb81" / "thermal_scatt": 114,
                data / "openmc" / "endfb81": 111,
                data / "g4ndl" / "g4ndl471": 58}
    for root, count in expected.items():
        if not root.exists():
            continue
        index = index_tsl(root)
        assert len(index.tapes) == count and index.unreadable == []
        assert len({t.key for t in index.tapes}) == count
