"""Photonuclear ACE tables (``.u``): read, encode back, write.

The references are the tables NJOY2016 writes for its own photonuclear tests
(56, 57, 58, 64, 66, 78: U-235 and Bi-209 of ENDF/B-VIII.0, Co-59, He-3 and
Pu-239 of the IAEA library, Ra-226 of TENDL-2019) and an LA150u table (C-12,
2000), which writes NXS(10) = NXS(11) = 0 and its photons as law 4. Each is
read strictly, encoded from what was read, and written: words, integer
spellings and bytes must come back unchanged. The tests are skipped where those
tables are not on disk.
"""
from pathlib import Path

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.classes.photonuclear import PhotonuclearAce
from kika.ace.classes.xss import unpack_int_words
from kika.ace.encoders.photonuclear import encode_photonuclear
from kika.ace.writers.write_ace import write_ace, write_ace_checked

CODES = Path(__file__).resolve().parents[5] / "Codes"
TABLES = [CODES / f"NJOY2016/tests/{t}/referenceTape31" for t in (78, 56, 57, 58, 64, 66)]
TABLES.append(CODES / "dryad/test/resources/6012.24u")


def _encode(table):
    h = table.header
    nxs = list(h.nxs_array)
    return encode_photonuclear(table, zaid=h.zaid, extension=h.extension, awr=h.atomic_weight_ratio,
                               temperature_mev=h.temperature, comment=h.comment, date=h.date, matid=h.matid,
                               z=nxs[10], a=nxs[11], izaw=h.izaw_array)


@pytest.mark.parametrize("path", TABLES, ids=lambda p: p.parent.name)
def test_a_table_reads_and_writes_back_byte_for_byte(path, tmp_path):
    if not path.exists():
        pytest.skip(f"{path} is not on disk")
    table = read_ace(str(path), strict=True)
    assert isinstance(table, PhotonuclearAce)
    again = _encode(table)
    assert list(again.header.nxs_array) == list(table.header.nxs_array)
    assert list(again.header.jxs_array) == list(table.header.jxs_array)
    a, b = np.asarray(table.xss_data, float), np.asarray(again.xss_data, float)
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=5e-12, atol=0)
    n = a.size - 1
    assert np.array_equal(np.asarray(unpack_int_words(table.xss_int_words, n), bool),
                          np.asarray(unpack_int_words(again.xss_int_words, n), bool))
    out = tmp_path / "table.u"
    write_ace(again, str(out), True)
    assert out.read_bytes().replace(b"\r\n", b"\n") == path.read_bytes().replace(b"\r\n", b"\n")
    write_ace_checked(again, tmp_path / "checked.u")


def test_what_cobalt_59_says():
    """IAEA Co-59 (NJOY test 58): seven particle kinds, levels as law 33, the rest law 61."""
    path = TABLES[3]
    if not path.exists():
        pytest.skip(f"{path} is not on disk")
    t = read_ace(str(path), strict=True)
    assert t.za == 27059 and len(t.reactions) == 127 and t.elastic is None
    assert [p.particle for p in t.particles] == ["n", "photon", "H1", "H2", "H3", "He3", "He4"]
    n = t.particle("n")
    assert {r.mt for r in n.reactions} >= {5, 16, 50, 89}
    assert {type(r.laws[0]).__name__ for r in n.reactions if 50 <= r.mt <= 89} == {"ParticleLevelScattering"}
    # σ on the grid is zero below a reaction's first point and its own values from there
    r = t.reaction(16)
    on = r.on_grid(t.energies.size)
    assert not on[:r.first_index - 1].any() and np.array_equal(on[r.first_index - 1:], r.cross_section)
    assert t.energies[0] > 0.0 and np.all(np.diff(t.energies) >= 0.0)


def test_a_particle_kind_that_is_not_one_of_the_seven_is_refused():
    path = TABLES[0]
    if not path.exists():
        pytest.skip(f"{path} is not on disk")
    t = read_ace(str(path), strict=True)
    from kika.ace.parsers.parse_photonuclear import parse_photonuclear

    xss = np.array(t.xss_data, dtype=float)
    xss[int(t.header.jxs_array[10])] = 5.0             # IPT of the first entry
    with pytest.raises(ValueError, match="IPT = 5"):
        parse_photonuclear(t.header, xss)
