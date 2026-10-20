"""``process_photoatomic``: three evaluations of an element → its EPRDATA table, written.

With the committed ENDF/B-VIII.1 hydrogen tapes (photo-atomic, atomic
relaxation and electro-atomic, whole): the table is written as ``1000.00p``
with its xsdir line, reads
back as an EPRDATA table with every block (electrons included), and the photon
blocks are those :func:`photoatomicTable` builds. A missing evaluation, a tape of
the wrong sublibrary or of another element is refused.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.ace import read_ace
from kika.endf import process_photoatomic

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
PHOTOAT = DATA / "b81_h_photoatomic.endf"
RELAX = DATA / "b81_h_relaxation.endf"
ELECTRONS = DATA / "b81_h_electrons.endf"
EPRDATA25 = Path(__file__).resolve().parents[5] / "NuclearData" / "ace" / "eprdata25" / "eprdata25" / "1000.25p"


@pytest.fixture(scope="module")
def run():
    return process_photoatomic(PHOTOAT, RELAX, ELECTRONS)


def test_the_table_is_written_and_reads_back(run, tmp_path):
    path = run.write_ace(tmp_path)
    assert path.name == "1000.00p"
    assert "1000.00p" in (tmp_path / "xsdir").read_text()
    t = read_ace(str(path))
    nxs, jxs = t.header.nxs_array, t.header.jxs_array
    assert (nxs[2], nxs[6]) == (1, 3) and nxs[7] == len(t.subshell_designators)
    assert all(jxs[k] > 0 for k in range(19, 28))              # every electron block, SELAS included
    np.testing.assert_allclose(t.energies, run.table.energies, rtol=1e-11)
    np.testing.assert_allclose(t.heating, run.table.heating, rtol=1e-11)
    assert sum(t.shell_electrons) == pytest.approx(1.0)
    assert t.compton_profiles[0].cdf[-1] == 1.0


def test_every_word_is_spelled_as_eprdata(run, tmp_path):
    path = run.write_ace(tmp_path, extension=".01p", xsdir=False)
    data = path.read_text().splitlines()[12:]
    assert all(len(line) <= 80 for line in data)
    assert data[0][:20].strip().endswith(("E+00", "E-01", "E-02", "E-03", "E-04", "E-05", "E-06", "E-07", "E+01"))


def test_the_xsdir_refuses_a_second_table_of_the_same_name(run, tmp_path):
    from kika.endf import ProcessingOutputError

    run.write_ace(tmp_path)
    with pytest.raises(ProcessingOutputError) as error:
        run.write_ace(tmp_path)
    assert error.value.category == "ace-table-exists"
    run.write_ace(tmp_path, replace=True)


def test_all_three_evaluations_are_required():
    with pytest.raises(ValueError, match="electro-atomic"):
        process_photoatomic(PHOTOAT, RELAX, None)
    with pytest.raises(ValueError, match="photoat is not"):
        process_photoatomic(RELAX, RELAX, ELECTRONS)


def test_hydrogen_agrees_with_eprdata25(run, tmp_path):
    """On eprdata25's nodes, every block but the heating numbers to print precision; the
    electron words to print precision; the heating to its quadrature (<= 3e-5)."""
    if not EPRDATA25.exists():
        pytest.skip("no eprdata25 in ../NuclearData")
    mine = read_ace(str(run.write_ace(tmp_path)))
    ref = read_ace(str(EPRDATA25))
    at = np.searchsorted(mine.energies, ref.energies * (1 - 1e-11))
    at = at + np.concatenate(([0], (ref.energies[1:] == ref.energies[:-1]).astype(int)))
    np.testing.assert_allclose(mine.energies[at], ref.energies, rtol=1e-11)
    for name in ("incoherent", "coherent", "photoelectric", "pair_production"):
        np.testing.assert_allclose(getattr(mine, name)[at], getattr(ref, name), rtol=1e-11)
    np.testing.assert_allclose(mine.heating[at], ref.heating, rtol=3e-5)
    np.testing.assert_allclose(mine.electron_words, ref.electron_words, rtol=1e-11, atol=0)
