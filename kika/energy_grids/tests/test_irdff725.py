"""The IRDFF-II 725-group structure against the IAEA file it was taken from."""
import numpy as np

from kika.energy_grids import IRDFF725, SANDIIA640


def _iaea_spelling_hash(values_ev):
    text = " ".join(f"{m}e{int(e):+d}" for m, e in (f"{v:.4e}".split("e") for v in values_ev))
    h = 5381
    for c in text:
        h = ((h * 33) ^ ord(c)) & 0xFFFFFFFF
    return h, len(text)


def test_the_boundaries_are_those_of_the_iaea_file():
    # djb2 of SAND_725.egb's 726 values (eV, 5 digits), read from www-nds.iaea.org/IRDFF on 2026-11-21.
    assert _iaea_spelling_hash(np.array(IRDFF725) * 1e6) == (2034323496, 7259)


def test_the_structure_is_sand_iia_extended_down_and_up():
    g = np.array(IRDFF725)
    assert g.size == 726 and np.all(np.diff(g) > 0)
    assert g[0] == 1e-11 and g[45] == 1e-10 and g[-1] == 60.0
    np.testing.assert_array_equal(g[45:686], SANDIIA640)
    np.testing.assert_allclose(np.diff(g[685:]), [0.5] * 20 + [1.0] * 10 + [2.0] * 10)
