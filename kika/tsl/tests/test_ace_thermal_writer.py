"""ACE thermal tables written by kika: every elastic mode, and real tables recomposed byte for byte.

A table read with :func:`kika.tsl.ace.read_ace_thermal` and written back with
:func:`kika.tsl.ace_writer.write_ace_thermal` is the same file: ITIE, the
continuous outgoing blocks behind their locators (one word before each block),
the coherent Bragg edges, the incoherent σ and cosines, and a mixed elastic's
two halves at JXS(4-6) and JXS(7-9).
"""
import os
from pathlib import Path

import numpy as np
import pytest

from kika.tsl.ace import read_ace_thermal, read_ace_thermal_header
from kika.tsl.ace_writer import encode_ace_thermal, write_ace_thermal
from kika.tsl.processed import BraggTable, IncoherentElasticTable, OutgoingTable, ProcessedTemperature


def _block(elastic):
    incident = np.array([1e-5, 1e-3, 1.0])
    energies = [np.array([1e-6, 2e-3, 0.1]), np.array([1e-4, 1e-3]), np.array([0.5, 1.0, 2.0, 3.0])]
    pdf = [np.array([10.0, 5.0, 0.1]), np.array([300.0, 200.0]), np.array([0.2, 0.5, 0.3, 0.0])]
    cdf = [np.array([0.2, 0.7, 1.0]), np.array([0.4, 1.0]), np.array([0.1, 0.5, 0.9, 1.0])]
    cosines = [np.sort(np.random.default_rng(i).uniform(-1, 1, (len(e), 4)), axis=1) for i, e in enumerate(energies)]
    block = ProcessedTemperature(temperature=293.6, inelastic_energies=incident,
                                 inelastic_xs=np.array([20.0, 12.0, 4.0]), outgoing_kind="continuous")
    block._outgoing = OutgoingTable(kind="continuous", incident=incident, energies=energies, cosines=cosines,
                                    pdf=pdf, cdf=cdf)
    if elastic in ("coherent", "mixed"):
        block.coherent = BraggTable(edges=np.array([2e-3, 5e-3, 9e-3]), s=np.array([1e-3, 4e-3, 6e-3]))
    if elastic in ("incoherent", "mixed"):
        e = np.array([1e-5, 0.1, 2.0])
        block.incoherent = IncoherentElasticTable(energies=e, xs=np.array([40.0, 30.0, 5.0]), cosine_energies=e,
                                                  cosines=np.tile(np.linspace(-0.9, 0.9, 6), (3, 1)))
    return block


@pytest.mark.parametrize("elastic, idpnc", [(None, 0), ("coherent", 4), ("incoherent", 3), ("mixed", 5)])
def test_every_elastic_mode_writes_and_reads_back(tmp_path, elastic, idpnc):
    block = _block(elastic)
    path = write_ace_thermal(block, tmp_path / "x.20t", zaid="test.20t", awr=0.999167, izaw=[1001],
                             title="a test table", mat=125, date="10/15/26", nieb=64)
    header = read_ace_thermal_header(path)
    assert header.nxs[5] == idpnc and header.nxs[3] == 5 and header.nxs[7] == 2 and header.nxs[4] == 64
    back = read_ace_thermal(path).temperatures[0]
    np.testing.assert_allclose(back.inelastic_energies, block.inelastic_energies, rtol=1e-11)
    table, original = back.outgoing(), block.outgoing()
    for a, b in zip(table.energies + table.pdf + table.cdf, original.energies + original.pdf + original.cdf):
        np.testing.assert_allclose(a, b, rtol=1e-11)
    if elastic in ("coherent", "mixed"):
        np.testing.assert_allclose(back.coherent.s, block.coherent.s, rtol=1e-11)
    if elastic in ("incoherent", "mixed"):
        np.testing.assert_allclose(back.incoherent.cosines, block.incoherent.cosines, rtol=1e-11)
    again = write_ace_thermal(read_ace_thermal(path), tmp_path / "y.20t", zaid="test.20t", kt=header.kt,
                              izaw=[1001])
    assert again.read_bytes() == path.read_bytes()


def test_a_locator_points_one_word_before_its_block():
    nxs, jxs, xss, is_int = encode_ace_thermal(_block(None), nieb=64)
    n = int(xss[jxs[1]])
    locators = xss[jxs[3]:jxs[3] + n].astype(int)
    assert locators[0] + 1 == jxs[3] + 2 * n
    assert xss[locators[0] + 1] == pytest.approx(1e-6 / 1e6)
    assert is_int[jxs[3] - 1:jxs[3] - 1 + 2 * n].all()


SAB = os.environ.get("KIKA_ACE_SAB")


@pytest.mark.skipif(not SAB, reason="set KIKA_ACE_SAB to a directory of ACE thermal tables (*t)")
def test_real_tables_are_recomposed_byte_for_byte(tmp_path):
    tables = sorted(p for p in Path(SAB).iterdir() if p.name.endswith("t") and p.is_file())
    assert tables
    for path in tables:
        header = read_ace_thermal_header(path)
        out = write_ace_thermal(read_ace_thermal(path), tmp_path / path.name, zaid=header.zaid,
                                kt=header.kt, izaw=header.izaw)
        assert out.read_bytes() == path.read_bytes(), path.name
