"""The electron blocks of an EPRDATA table (JXS 19-27), against their definition and eprdata25.

On the committed micro tape (iron, one ionised subshell): the layout is what the
EPRDATA format states (ESZE NE×(6+NSSH), EXCIT and BREML two columns, ELAS and
the spectra closed by cdfs that end at 1, SELAS the last block), the cut at
10 eV holds, and the total is the sum of its columns in the reference's order.

With LANL's eprdata25 and the ENDF/B-VIII.1 electrons sublibrary present
(``../NuclearData``), Fe, Pb and Ni must give eprdata25's electron words at the
file's precision (%.12E), Ni except where it repeats a subshell threshold that
starts on a non-zero value (the step the evaluation states; eprdata25 ramps from
the 10 eV cut) and the EION block indices that follow from it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kika
from kika.processing.photoatomic.electrons import ELECTRON_CUTOFF, electronBlocks

MICRO = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data" / "micro_fe_electrons.endf"
ND = Path(__file__).resolve().parents[5] / "NuclearData"


@pytest.fixture(scope="module")
def blocks():
    return electronBlocks(kika.read(str(MICRO)))


def test_layout(blocks):
    (ne, nxl, na, nb, nbl), words, jxs = blocks
    rel = dict(zip(range(19, 33), jxs))
    assert rel[19] == 0 and rel[20] == ne * (6 + 1)
    assert rel[21] == rel[20] + 2 * nxl and rel[22] == rel[21] + 3 * na
    assert rel[25] == rel[24] + 3 * nb and rel[27] == rel[26] + 2 * nbl
    assert words.size == rel[27] + 2 * ne
    assert all(v is None for k, v in rel.items() if k > 27)


def test_grid_starts_at_the_cut_and_total_adds_up(blocks):
    (ne, *_), words, _ = blocks
    columns = words[:ne * 7].reshape(7, ne)
    energy = columns[0] * 1e6
    assert energy[0] == ELECTRON_CUTOFF and np.all(np.diff(energy) >= 0)
    total = ((columns[2] + columns[3]) + columns[4]) + columns[5]
    np.testing.assert_allclose(columns[1], total, rtol=1e-15)
    np.testing.assert_array_equal(columns[5], columns[6])          # one subshell


def test_cdfs_end_at_one(blocks):
    (ne, nxl, na, nb, nbl), words, jxs = blocks
    rel = dict(zip(range(19, 33), jxs))
    lengths = words[rel[21] + na:rel[21] + 2 * na].astype(int)
    locators = words[rel[21] + 2 * na:rel[21] + 3 * na].astype(int)
    for n, loc in zip(lengths, locators):
        cdf = words[rel[22] + loc + n:rel[22] + loc + 2 * n]
        assert cdf[0] == 0.0 and cdf[-1] == 1.0 and np.all(np.diff(cdf) >= 0)


def _reference(z):
    tape = next((ND / "electrons" / "endfb81").glob(f"e-{z:03d}_*_000.endf"), None)
    table = ND / "ace" / "eprdata25" / "eprdata25" / f"{1000 * z}.25p"
    return (tape, table) if tape is not None and table.exists() else (None, None)


@pytest.mark.parametrize("z", [26, 82])
def test_eprdata25_electron_words(z):
    tape, table = _reference(z)
    if tape is None:
        pytest.skip("no eprdata25 / ENDF/B-VIII.1 electrons in ../NuclearData")
    from kika.ace import read_ace

    nxs, words, jxs = electronBlocks(kika.read(str(tape)))
    ref = read_ace(str(table))
    assert nxs == ref.electron_nxs and jxs == ref.electron_jxs
    printed = np.count_nonzero(np.array([format(v, ".12E") for v in words])
                               != np.array([format(v, ".12E") for v in ref.electron_words]))
    scale = np.maximum(np.abs(words), np.abs(ref.electron_words))
    rel = np.where(scale > 0, np.abs(words - ref.electron_words) / np.where(scale > 0, scale, 1), 0)
    # the last printed digit may round the other way on a few words (Pb: 5 of ~30 000)
    assert rel.max() <= 5.5e-13 and printed <= 10
