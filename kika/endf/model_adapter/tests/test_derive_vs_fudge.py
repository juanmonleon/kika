"""G3/G4: kika's ENDF from NNDC's GNDS is FUDGE's ENDF from the same GNDS.

The strongest gate the GNDS->ENDF derivation has: the distributed ENDF/B-VIII.1
GNDS files were written by FUDGE from the ENDF tapes, and FUDGE's ``toENDF6``
writes them back. Where kika and FUDGE agree, kika derives what FUDGE derives;
where the file the GNDS came from says something else (the fission Q, an AWR
the GNDS mass no longer gives), FUDGE says it too, so the comparison is not
polluted by what GNDS lost.

Measured 2026-10-09, every MT of MF1/452-458, MF3, MF4, MF5, MF6 and MF12-15
identical in its numbers (blank and zero read alike) on Be-9, Li-6, Fe-56 and
U-235, and MF12-15 on N-14 (MF13 on its sums' orphanProducts included); MF1/451
differs only in its directory, which lists MF2 (G5, not this line).

``KIKA_GNDS_LIB`` points at the folder of NNDC's ``n-*.endf.gnds.xml`` files;
by default ``NuclearData/gnds/endfb81`` beside ``KIKA_LIB_TAPES``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import kika
from kika.endf import read_endf
from kika.endf.writers.assemble import writeEndfTape

pytestmark = [pytest.mark.fudge, pytest.mark.slow]

CASES = {
    "n-004_Be_009": (1, 3, 4, 5, 6, 12, 13, 14, 15),
    "n-003_Li_006": (1, 3, 4, 5, 6, 12, 13, 14, 15),
    "n-026_Fe_056": (1, 3, 4, 5, 6, 12, 13, 14, 15),
    "n-092_U_235": (1, 3, 4, 5, 6, 12, 13, 14, 15),
    "n-007_N_014": (12, 13, 14, 15),
}


def _library() -> Path:
    stated = os.environ.get("KIKA_GNDS_LIB")
    if stated:
        return Path(stated)
    tapes = os.environ.get("KIKA_LIB_TAPES")
    return Path(tapes).parent / "gnds" / "endfb81" if tapes else Path("missing")


def _fields(line):
    text = line[:66].ljust(66)
    return [text[k:k + 11].strip() or "0" for k in range(0, 66, 11)]


@pytest.mark.parametrize("name", sorted(CASES))
def test_kika_writes_from_nndcs_gnds_what_fudge_writes(name, tmp_path):
    import sys

    command = os.environ.get("KIKA_FUDGE_PYTHON", "").split()
    if not command:
        pytest.skip("KIKA_FUDGE_PYTHON is not set")
    source = _library() / f"{name}.endf.gnds.xml"
    if not source.is_file():
        pytest.skip(f"{source} is not here")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_fudge_in_the_loop import _runFudge

    fudge = _runFudge(command, source, name=f"{name}.xml")
    assert "endf" in fudge, fudge.get("endfError")
    theirs_path = tmp_path / "fudge.endf"
    theirs_path.write_text(fudge["endf"])
    mine_path = tmp_path / "kika.endf"
    writeEndfTape(kika.read(source, covariances=False), mine_path)
    theirs, mine = read_endf(str(theirs_path)), read_endf(str(mine_path))

    for mf in CASES[name]:
        a = dict(getattr(theirs.mf.get(mf), "mt", {}))
        b = dict(getattr(mine.mf.get(mf), "mt", {}))
        if mf == 1:
            # The directory of MF1/451 lists sections kika does not derive yet.
            a.pop(451, None)
            b.pop(451, None)
        assert sorted(a) == sorted(b), f"MF{mf}: {sorted(set(a) ^ set(b))}"
        for mt in a:
            ours = [_fields(l) for l in str(b[mt]).splitlines()]
            fudges = [_fields(l) for l in str(a[mt]).splitlines()]
            assert ours == fudges, f"{name} MF{mf}/MT{mt}"
