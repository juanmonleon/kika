"""The evaluations the full-library sweep refused, processed whole.

Each tape carries one defect the processing now handles as NJOY and FUDGE do:

* JEFF-4.0 Ni-59: BW LRX=1 with no competitive width beyond rounding;
* JEFF-4.0 Mo-100, Cs-134: MF3 (0, 0) pairs that are blank fields;
* JEFF-4.0 B-11: MT16's incident grid steps at 20 MeV (unit-base CM mean);
* B-VIII.1 W-183: an MF8 pointer at an MF10 kept as read;
* B-VIII.1 Pb-208: MT4 zeros past EMAX;
* B-VIII.1 Th-229: a URR grid node stated twice;
* B-VIII.1 U-241: one URR group with J=I at L=0;
* B-VIII.1 Ra-226: an MF1/MT452 domain approximation;
* B-VIII.1 Sm-153: the URR range's SPI (3/2) differs from the resolved range's (5/2);
* JEFF-4.0 Pu-242: QX stated with LRX=0;
* JENDL-5 Pr-141: URR parameters tabulated to 100 keV in a region to 160 keV.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# the repository's conftest, by path: a bare `from conftest import` takes whichever
# conftest a full collection put first on sys.path (kika/plotting/tests has one)
_spec = importlib.util.spec_from_file_location(
    "_kika_root_conftest", Path(__file__).resolve().parents[3] / "conftest.py")
_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root)
resolve_tape = _root.resolve_tape

pytestmark = pytest.mark.tape

TAPES = ("t3_ni59_jeff40", "t3_mo100_jeff40", "t3_cs134_jeff40", "t3_b11_jeff40", "t3_w183_b81",
         "t3_pb208_b81", "t3_th229_b81", "t3_u241_b81", "t3_ra226_b81",
         "t3_sm153_b81", "t3_pu242_jeff40", "t3_pr141_jendl5")


@pytest.mark.parametrize("name", TAPES)
def test_the_tape_t3_refused_is_processed(name):
    path = resolve_tape(name)
    if path is None:
        pytest.skip(f"{name} not reachable")
    import kika
    from kika.hdf5.processing import processSuite
    from kika.nuclear_data import model as m

    suite = kika.read(str(path), covariances=False)
    out = processSuite(suite, temperatures=[293.6], report=m.ConversionReport(), endf=str(path), workers=2)
    assert out.reactions
