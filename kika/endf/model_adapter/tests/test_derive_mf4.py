"""G3: MF4 from a suite that never saw ENDF (``derive/products.py``).

Three gates, as ``gnds_to_endf_plan.md`` §3.3 asks:

1. the rules on their own (``angularHeader``) for every LTT and the refusals;
2. **strip and derive** on a committed cut: the MF4 written from the stripped
   suite is the original's, with the one named difference (NM);
3. **a GNDS kika did not write**: NNDC's S-36 → ENDF gives MF4 identical, in
   columns 1-66, to the ENDF/B-VIII.1 tape it was made from; and ``-m fudge``,
   the same GNDS through FUDGE's ``toENDF6`` gives the same MF4 as kika's.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf import read_endf
from kika.endf.model_adapter.decode import decodeReactionSuite
from kika.endf.model_adapter.derive.oracle import strip
from kika.endf.model_adapter.derive.products import angularHeader
from kika.endf.writers.assemble import writeEndfTape
from kika.nuclear_data.model import (AngularTwoBody, Frame, Isotropic2d, Legendre,
                                     Regions2d, XYs1d, XYs2d, angularAxes)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
GNDS = Path(__file__).resolve().parents[3] / "gnds" / "tests" / "data"


def _legendre(e, *coefficients):
    f = Legendre(coefficients=np.asarray((1.0,) + coefficients))
    f.outerDomainValue = e
    return f


def _table(e):
    f = XYs1d(xs=np.array([-1.0, 1.0]), ys=np.array([0.5, 0.5]))
    f.outerDomainValue = e
    return f


def _body(functions):
    return XYs2d(function1ds=functions, axes=angularAxes())


@pytest.mark.parametrize("form, ltt, li, nm", [
    (Isotropic2d(productFrame=Frame.lab), 0, 1, None),
    (AngularTwoBody(angular=_body([_legendre(1.0, 0.1), _legendre(2.0, 0.2, 0.0)]),
                    productFrame=Frame.centerOfMass), 1, 0, None),
    (AngularTwoBody(angular=_body([_table(1.0), _table(2.0)]),
                    productFrame=Frame.centerOfMass), 2, 0, None),
    (AngularTwoBody(angular=Regions2d(function2ds=[
        _body([_legendre(1.0, 0.1, 0.2, 0.3)]), _body([_table(2.0)])]),
        productFrame=Frame.centerOfMass), 3, 0, 3),
])
def test_each_shape_gets_its_ltt(form, ltt, li, nm):
    header = angularHeader(form, 2)
    assert (header["ltt"], header["li"], header["nm"]) == (ltt, li, nm)
    assert header["lct"] == (1 if form.productFrame == Frame.lab else 2)


def test_a_mixed_table_is_refused_by_name():
    form = AngularTwoBody(angular=_body([_legendre(1.0, 0.1), _table(2.0)]),
                          productFrame=Frame.centerOfMass)
    with pytest.raises(ValueError, match="mixes Legendre and tabulated"):
        angularHeader(form, 2)


def _mf4Lines(path):
    endf = read_endf(str(path), mf_numbers=[4])
    return {mt: [line[:66] for line in str(section).splitlines()]
            for mt, section in endf.mf[4].mt.items()}


def test_stripped_and_derived_mf4_is_the_tapes(tmp_path):
    """JEFF-4.0 Fe-56 MT2 is LTT=3; its NM is the one named difference."""
    tape = DATA / "micro_fe56_xs_and_angular.endf"
    suite, _ = decodeReactionSuite(read_endf(str(tape)))
    bare = strip(suite)
    out = tmp_path / "derived.endf"
    writeEndfTape(bare, out)

    original, derived = _mf4Lines(tape), _mf4Lines(out)
    assert set(derived) == set(original)
    for mt, lines in original.items():
        assert len(derived[mt]) == len(lines)
        different = [k for k, (a, b) in enumerate(zip(lines, derived[mt])) if a != b]
        if mt == 2:
            # Line 2 is the LTT=3 CONT carrying NM: 31 in the file, 32 derived.
            assert different == [1]
            assert int(lines[1][55:66]) == 31 and int(derived[mt][1][55:66]) == 32
        else:
            assert different == [], mt


@pytest.fixture(scope="module")
def s36Tape(neutron_libraries):
    path = neutron_libraries["endfb81"] / "n-016_S_036.endf"
    if not path.is_file():
        pytest.skip(f"{path} is not here")
    return path


def test_nndcs_gnds_writes_the_mf4_of_the_tape_it_came_from(s36Tape, tmp_path):
    suite = kika.read(GNDS / "n-016_S_036.endf.gnds.xml", covariances=False)
    out = tmp_path / "from_gnds.endf"
    writeEndfTape(suite, out)
    assert _mf4Lines(out) == _mf4Lines(s36Tape)


@pytest.mark.fudge
def test_fudge_and_kika_write_the_same_mf4_from_the_same_gnds(tmp_path):
    import os
    import sys

    command = os.environ.get("KIKA_FUDGE_PYTHON", "").split()
    if not command:
        pytest.skip("KIKA_FUDGE_PYTHON is not set")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_fudge_in_the_loop import _runFudge

    source = GNDS / "n-016_S_036.endf.gnds.xml"
    fudge = _runFudge(command, source, name="s36.xml")
    assert "endf" in fudge, fudge.get("endfError")
    theirs = tmp_path / "fudge.endf"
    theirs.write_text(fudge["endf"])
    mine = tmp_path / "kika.endf"
    writeEndfTape(kika.read(source, covariances=False), mine)
    assert _mf4Lines(mine) == _mf4Lines(theirs)
