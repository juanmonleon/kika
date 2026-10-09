"""RECONR and MT3: a tape kika writes makes NJOY rebuild MT3 as the original does.

``docs/library/njoy_reconr_redundant_sums.md`` measured that RECONR rebuilds MT3
**only when MF12/MT3 exists** (``reconr.f90`` ``anlyzd``) and drops a stated
MF3/MT3 otherwise. Before roadmap E5 a tape kika wrote carried no MF12, so the
same evaluation gave a PENDF with MT3 from the original and without it from
kika's copy. Since E5b/E5d the photons of MT3 are an ``orphanProduct`` that the
writer emits as MF12/MT3; this is the measurement that closes the note (roadmap
E5f.1).

JENDL-5 Hf-182 is the witness: MF12/13-15 on MT3, a small tape, an RRR.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.model_adapter.decode import decodeReactionSuite
from kika.endf.writers import remove_sections, update_mf1_directory
from kika.endf.writers.assemble import writeEndfTape

pytestmark = pytest.mark.njoy


@pytest.fixture(scope="module")
def hf182(neutron_libraries):
    path = neutron_libraries["jendl5"] / "n_072-Hf-182.dat"
    if not path.is_file():
        pytest.skip(f"{path} is not here")
    return path


def _same(a, b) -> bool:
    return (np.array_equal(np.asarray(a.energies), np.asarray(b.energies))
            and np.array_equal(np.asarray(a.values), np.asarray(b.values)))


def test_reconr_rebuilds_the_same_mt3_from_kikas_tape(hf182, njoy_exe, tmp_path):
    from kika.processing.njoy_reconstruct import njoy_reconstruct

    suite, _ = decodeReactionSuite(read_endf(str(hf182)))
    written = tmp_path / "kika.endf"
    writeEndfTape(suite, written)
    assert 3 in read_endf(str(written), mf_numbers=[12]).mf[12].mt

    original = njoy_reconstruct(hf182, njoy_exe)
    mine = njoy_reconstruct(written, njoy_exe)
    assert 3 in original and 3 in mine
    for mt in (1, 3, 4):
        assert _same(original[mt], mine[mt]), f"PENDF MT{mt} differs"


def test_without_mf12_mt3_reconr_writes_no_mt3(hf182, njoy_exe, tmp_path):
    """The control: the same tape with MF12-15/MT3 cut out gives no MT3 at all,
    which is what every kika tape gave before E5 (risk R7 of the E5 plan)."""
    from kika.processing.njoy_reconstruct import njoy_reconstruct

    text = hf182.read_text(encoding="latin-1")
    stripped, removed = remove_sections(text, [(12, 3), (13, 3), (14, 3), (15, 3)])
    assert removed
    cut = tmp_path / "no_mf12_mt3.endf"
    cut.write_text(stripped, encoding="latin-1", newline="")
    update_mf1_directory(str(cut))
    assert 3 not in njoy_reconstruct(cut, njoy_exe)
