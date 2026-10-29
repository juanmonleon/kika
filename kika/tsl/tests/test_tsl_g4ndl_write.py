"""Writing G4NDL ``ThermalScattering``: read → write → the numbers Geant4 reads.

The full-library check against a C++ copy of Geant4's reading loops lives in
``kika-workspace/myworkspace/G4NDL/thermal/`` (58/58 materials of G4NDL 4.7.1
stored identically, 2026-10-14); here the same contract is tested token by
token, plus what a material from another format gets built.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest

from kika.g4ndl.exceptions import G4NDLUnsupportedError
from kika.tsl import read_tsl
from kika.tsl.g4ndl import (BRAGG_RATIO, DEFAULT_MT, TAIL_END, bragg_cross_section_table,
                            list_g4ndl_thermal, read_g4ndl_thermal, union_bragg_s,
                            write_g4ndl_thermal)
from kika.tsl.processed import (BraggTable, IncoherentElasticTable, OutgoingTable,
                                ProcessedTemperature, ProcessedTSL)
from kika.tsl.tests.test_tsl_formats import write_ace_thermal


def _g4(path: Path, body: str, header: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = "        G4NDL\nTEST_SOURCE\n" if header else ""
    path.write_bytes(zlib.compress((head + body).encode()))


def _material(root: Path, name: str = "zr90_zrh") -> Path:
    """Every process, two temperatures, the zero tails and E' = 0 rows of real files."""
    th = root / "ThermalScattering"
    cs = lambda mt, blocks: "".join(f"3\n{mt}\n{t}\n{len(p.split()) // 2}\n{p}\n" for t, p in blocks)
    _g4(th / "Coherent" / "CrossSection" / f"{name}.z", cs(230, [
        ("293.6", "1e-05 0 0.0039 0 0.0041 1.4634146 4.0 0.0015 4.000001 0 20000000.0 0"),
        ("600.0", "1e-05 0 0.0039 0 0.0041 1.2195122 4.0 0.00125 4.00004 0 2e7 0")]))
    _g4(th / "Coherent" / "FS" / f"{name}.z",
        "7\n2\n293.6\n2\n0.004 0.006 0.006 0.010\n7\n2\n600.0\n2\n0.005 0.009\n")
    _g4(th / "Incoherent" / "CrossSection" / f"{name}.z", cs(226, [
        ("293.6", "1e-05 8.19 1.0 7.5 4.0 6.0 4.000001 0 2e7 0")]))
    _g4(th / "Incoherent" / "FS" / f"{name}.z",
        "6\n226\n293.6\n2\n"
        "0 1e-05 0 0 6 6\n1.000000e-05 1.0 -0.75 -0.25 0.25 0.75\n"
        "0 1.23457 0 0 6 6\n1.234567e+00 1.0 -0.5 0.1 0.6 0.9\n")
    _g4(th / "Inelastic" / "CrossSection" / f"{name}.z", cs(225, [
        ("293.6", "1e-05 33.7 1.0 3.0 4.0 2.5 4.00004 0 2e7 0"),
        ("600.0", "1e-05 35.0 1.0 3.2 4.0 2.6 4.00004 0 2e7 0")]))
    rec = ("0 1e-05 0 0 20 4\n0 0 0 0\n1e-05 0.5 -0.5 0.5\n2e-05 0 0 0\n3e-05 0.25 -0.2 0.6\n"
           "4e-05 0 0 0\n")
    rec2 = "0 1.0 0 0 12 4\n0 0 0 0\n0.9 2.0 -0.1 0.4\n1.1 0 0 0\n"
    _g4(th / "Inelastic" / "FS" / f"{name}.z",
        "".join(f"6\n225\n{t}\n2\n" + rec + rec2 for t in ("293.6", "600.0")))
    return th


def _numbers(path: Path):
    tok = zlib.decompress(path.read_bytes()).decode().split()
    if tok[0] == "G4NDL":
        tok = tok[2:]
    return [float(t) for t in tok], tok


def _geant4_tokens(path: Path, kind: str):
    """Every token Geant4 keeps, as floats; Incoherent/FS rows open with two it skips."""
    values, _tok = _numbers(path)
    if kind != "Incoherent/FS":
        return values
    out, i = [], 0
    while i < len(values):
        n = int(values[i + 3])
        out += values[i:i + 4]
        i += 4
        for _ in range(n):
            nw = int(values[i + 4])
            out += values[i:i + 6] + values[i + 8:i + 6 + nw]
            i += 6 + nw
    return out


def _same_for_geant4(original: Path, rewrite: Path, name: str):
    for process in ("Coherent", "Incoherent", "Inelastic"):
        for kind in ("CrossSection", "FS"):
            a, b = original / process / kind / f"{name}.z", rewrite / process / kind / f"{name}.z"
            assert a.exists() == b.exists(), (process, kind)
            if a.exists():
                key = f"{process}/{kind}"
                assert _geant4_tokens(a, key) == _geant4_tokens(b, key), key


def test_a_g4ndl_material_is_written_back_as_geant4_read_it(tmp_path):
    th = _material(tmp_path / "in")
    material = read_tsl(th, member="zr90_zrh")
    written = write_g4ndl_thermal(material, tmp_path / "out")
    assert len(written) == 6
    out = tmp_path / "out" / "ThermalScattering"
    _same_for_geant4(th, out, "zr90_zrh")
    # Integers stay integers (Geant4 reads them with >> G4int), the header is carried.
    _values, tok = _numbers(out / "Inelastic" / "FS" / "zr90_zrh.z")
    assert tok[:4] == ["6", "225", "293.6", "2"] and tok[4:10] == ["0", "1e-05", "0", "0", "20", "4"]
    assert zlib.decompress((out / "Coherent" / "FS" / "zr90_zrh.z").read_bytes()).split()[:2] == \
        [b"G4NDL", b"TEST_SOURCE"]


def test_the_reader_keeps_what_the_writer_needs(tmp_path):
    p = read_g4ndl_thermal(_material(tmp_path), "zr90_zrh")
    assert p.details["mf_mt"] == {"Coherent/CrossSection": [3, 230], "Coherent/FS": [7, 2],
                                  "Incoherent/CrossSection": [3, 226], "Incoherent/FS": [6, 226],
                                  "Inelastic/CrossSection": [3, 225], "Inelastic/FS": [6, 225]}
    assert p.details["tails"]["Inelastic"][293.6] == [4.00004, 2e7]
    hot = p.at(600.0)
    np.testing.assert_array_equal(hot.coherent.energies, [1e-5, 0.0039, 0.0041, 4.0])
    np.testing.assert_array_equal(hot.inelastic_energies, [1e-5, 1.0, 4.0])   # cut at E max
    table = p.at(293.6).outgoing()
    # Every row stays: the cdf is Geant4's trapezoids over all of them, zeros included.
    np.testing.assert_array_equal(table.energies[0], [0, 1e-5, 2e-5, 3e-5, 4e-5])
    areas = np.array([0.25, 0.25, 0.125, 0.125]) * 1e-5
    np.testing.assert_allclose(table.cdf[0], np.concatenate([[0], np.cumsum(areas)]) / areas.sum())


def test_the_header_is_optional_as_in_geant4(tmp_path):
    th = tmp_path / "ThermalScattering"
    _g4(th / "Coherent" / "CrossSection" / "al_metal.z", "3\n244\n293.6\n2\n1e-05 0 4.0 0\n",
        header=False)
    _g4(th / "Coherent" / "FS" / "al_metal.z", "7\n2\n293.6\n1\n0.004 0.006\n", header=False)
    p = read_g4ndl_thermal(th, "al_metal")
    assert p.library is None and p.at(293.6).coherent.s.tolist() == [0.006]


def test_left_over_tokens_are_an_error(tmp_path):
    th = tmp_path / "ThermalScattering"
    _g4(th / "Coherent" / "CrossSection" / "al_metal.z", "3\n244\n293.6\n1\n1e-05 0 7\n")
    _g4(th / "Coherent" / "FS" / "al_metal.z", "7\n2\n293.6\n1\n0.004 0.006\n")
    with pytest.raises(ValueError, match="cut short|left over"):
        read_g4ndl_thermal(th, "al_metal")


def _geant4_xsec(x, y, e):
    """``G4ParticleHPVector::GetXsec``: lin-lin, the 1e-6 doubled-x guard, held past the end."""
    i = np.searchsorted(x, e, side="right")
    low = np.where(i == 0, 0, np.clip(i - 1, 0, x.size - 2))
    high = low + 1
    dup = (x[high] - x[low]) / x[high] < 1e-6
    lin = y[low] + (y[high] - y[low]) * (e - x[low]) / (x[high] - x[low])
    return np.where(e >= x[-1], y[-1], np.where(dup, y[low], lin))


def test_an_ace_material_gets_its_g4ndl_tables_built(tmp_path):
    ace = read_tsl(write_ace_thermal(tmp_path / "al-27.70t"))
    written = write_g4ndl_thermal(ace, tmp_path / "lib", name="al_metal")
    assert sorted(p.parent.parent.name for p in written) == ["Coherent"] * 2 + ["Inelastic"] * 2
    back = read_g4ndl_thermal(tmp_path / "lib", "al_metal")
    assert back.library == "kika"
    assert back.details["mf_mt"]["Inelastic/CrossSection"] == [3, DEFAULT_MT["Inelastic"]]
    src, out = ace.processed.temperatures[0], back.temperatures[0]
    np.testing.assert_array_equal(out.coherent.edges, src.coherent.edges)
    np.testing.assert_array_equal(out.coherent.s, src.coherent.s)
    np.testing.assert_array_equal(out.inelastic_xs, src.inelastic_xs)
    # The zeros that switch the process off above E max, as in every distributed file.
    tail = back.details["tails"]["Inelastic"][round(src.temperature, 3)]
    assert tail[-1] == TAIL_END and tail[0] == pytest.approx(src.inelastic_energies[-1], rel=2e-7)
    # σ_coh as Geant4 interpolates the built table: the exact S/E away from the edges.
    e = np.geomspace(1.1e-5, 0.999 * out.coherent.energies[-1], 5000)
    exact = src.coherent.cross_section(e)
    away = np.min(np.abs(np.log(e[:, None] / src.coherent.edges[None, :])), axis=1) > 1e-6
    got = _geant4_xsec(out.coherent.energies, out.coherent.xs, e)
    np.testing.assert_allclose(got[away], exact[away], rtol=(BRAGG_RATIO - 1) ** 2 / 4,
                               atol=0)
    a, b = src.outgoing(), out.outgoing()
    for k in range(a.incident.size):
        np.testing.assert_array_equal(b.energies[k], a.energies[k])
        np.testing.assert_array_equal(b.pdf[k], a.pdf[k])
        np.testing.assert_array_equal(b.cosines[k], a.cosines[k])


def test_the_built_bragg_table_jumps_at_each_edge():
    table = BraggTable(edges=np.array([0.004, 0.006]), s=np.array([0.006, 0.010]))
    e, y = bragg_cross_section_table(table, emax=1.0)
    assert e[0] == 1e-5 and y[0] == 0 and e[-1] == 1.0
    for edge, before, after in ((0.004, 0.0, 0.006), (0.006, 0.006, 0.010)):
        below = edge * (1 - 1e-7)
        assert _geant4_xsec(e, y, np.array([below * (1 + 5e-8)]))[0] == pytest.approx(before / below)
        assert _geant4_xsec(e, y, np.array([edge]))[0] == pytest.approx(after / edge)
    assert np.all(np.diff(e) > 0)


def test_edges_that_change_with_temperature_go_on_their_union():
    edges, s = np.array([1.0, 3.0]), np.array([2.0, 5.0])
    union = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    held = union_bragg_s(edges, s, union)
    np.testing.assert_array_equal(held, [0.0, 2.0, 2.0, 5.0, 5.0])
    e = np.linspace(0.1, 5.0, 200)
    np.testing.assert_array_equal(BraggTable(union, held).cross_section(e),
                                  BraggTable(edges, s).cross_section(e))


def _tsl(**block) -> ProcessedTSL:
    return ProcessedTSL(format="ace", name="X", awr=1.0,
                        temperatures=[ProcessedTemperature(temperature=293.6, **block)])


def test_what_geant4_would_read_as_something_else_is_refused(tmp_path):
    skewed = OutgoingTable(kind="skewed", incident=np.array([1.0]), energies=[np.array([0.5])],
                           cosines=[np.zeros((1, 2))])
    t = _tsl(inelastic_energies=np.array([1e-5, 1.0]), inelastic_xs=np.array([2.0, 1.0]),
             _outgoing=skewed)
    with pytest.raises(G4NDLUnsupportedError, match="skewed"):
        write_g4ndl_thermal(t, tmp_path, name="x")
    exact = _tsl(incoherent=IncoherentElasticTable(bound_xs=80.0, debye_waller=0.1))
    with pytest.raises(G4NDLUnsupportedError, match="Debye-Waller"):
        write_g4ndl_thermal(exact, tmp_path, name="x")
    bare = _tsl(incoherent=IncoherentElasticTable(energies=np.array([1e-5, 1.0]),
                                                  xs=np.array([80.0, 70.0])))
    with pytest.raises(G4NDLUnsupportedError, match="cosines"):
        write_g4ndl_thermal(bare, tmp_path, name="x")
    bragg = _tsl(coherent=BraggTable(edges=np.array([0.004]), s=np.array([0.006])))
    with pytest.raises(G4NDLUnsupportedError, match="one token"):
        write_g4ndl_thermal(bragg, tmp_path, name="x", source="two words")
    with pytest.raises(G4NDLUnsupportedError, match="Inelastic"):
        write_g4ndl_thermal(bragg, tmp_path, name="x", processes=["Inelastic"])
    with pytest.raises(ValueError, match="file stem"):
        write_g4ndl_thermal(bragg, tmp_path)
    assert not (tmp_path / "ThermalScattering").exists()


def test_overwrite_and_one_process_at_a_time(tmp_path):
    th = _material(tmp_path)
    material = read_tsl(th, member="zr90_zrh")
    with pytest.raises(FileExistsError):
        write_g4ndl_thermal(material, tmp_path)
    # Replacing one process leaves the other four files as they were.
    before = {p: p.read_bytes() for p in th.glob("*/*/zr90_zrh.z")}
    material.processed.temperatures[0].inelastic_xs = material.processed.temperatures[0].inelastic_xs * 0.97
    write_g4ndl_thermal(material, tmp_path, processes=["Inelastic"], overwrite=True)
    changed = {p for p, data in before.items() if p.read_bytes() != data}
    assert {p.parent.parent.name for p in changed} == {"Inelastic"}
    assert read_g4ndl_thermal(th, "zr90_zrh").temperatures[0].inelastic_xs[1] == pytest.approx(3.0 * 0.97)
    # A whole rewrite of a material without incoherent elastic removes those files.
    for b in material.processed.temperatures:
        b.incoherent = None
    write_g4ndl_thermal(material, tmp_path, overwrite=True)
    assert sorted(list_g4ndl_thermal(th)["zr90_zrh"]) == ["Coherent", "Inelastic"]
    assert not (th / "Incoherent" / "FS" / "zr90_zrh.z").exists()


@pytest.mark.tape
def test_every_thermal_material_of_g4ndl_471_is_written_back(tmp_path, g4ndl_g4ndl471_library):
    th = Path(g4ndl_g4ndl471_library) / "ThermalScattering"
    names = sorted(list_g4ndl_thermal(th))
    assert len(names) == 58
    for name in names:
        out = tmp_path / name
        write_g4ndl_thermal(read_tsl(th, member=name), out)
        _same_for_geant4(th, out / "ThermalScattering", name)
        for p in (out / "ThermalScattering").glob("*/*/*.z"):
            p.unlink()
