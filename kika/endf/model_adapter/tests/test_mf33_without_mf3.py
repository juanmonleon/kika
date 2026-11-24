"""A summed MT with MF33 and no MF3 gets a crossSectionSum for its covariance to name.

ENDF/B-VIII.1 Fe-56 states MF33/MT103 and no MF3/MT103: the (n,p) sum is
MT600-649's. FUDGE writes it as a ``crossSectionSum`` of those (NNDC's
``sum(z,p)``) and, where the tape has no MF3 for it, flags it ``omit`` so that
toENDF6 does not write one (``toENDF6/sums.py``). kika does both.
"""
from __future__ import annotations

import pytest

import kika
from kika.endf import read_endf
from kika.nuclear_data.model import CrossSectionSum
from kika.nuclear_data.model.endf_conversion import EndfConversionFlags, isOmitted

HREF = "/reactionSuite/sums/crossSectionSums/crossSectionSum[@label='MT103']"


@pytest.fixture(scope="module")
def fe56(fe56_b81_tape):
    return kika.read(str(fe56_b81_tape))


def test_mt103_is_the_sum_of_mt600_to_649_and_is_omitted(fe56):
    (sum103,) = [s for s in fe56.sums if s.ENDF_MT == 103]
    assert isinstance(sum103, CrossSectionSum)
    members = sorted(int(r.ENDF_MT) for r in fe56.reactions if 600 <= int(r.ENDF_MT) <= 649)
    assert [a.href for a in sum103.summands] == [
        f"/reactionSuite/reactions/reaction[@label='MT{mt}']/crossSection" for mt in members]
    assert sum103.outputChannel.Q.value == -2912600.0
    assert isOmitted(sum103, fe56)
    assert EndfConversionFlags.of(fe56).flagsFor(HREF) == {"omit": None}


def test_the_covariance_names_it_and_the_tape_gets_no_mf3(fe56, tmp_path):
    (section,) = [s for s in fe56.covarianceSuite.covarianceSections if s.label == "MF33-MT103"]
    assert section.rowData.href == HREF + "/crossSection"
    gnds = tmp_path / "fe56.gnds.xml"
    kika.write(fe56, str(gnds), format="gnds")
    back = kika.read(str(gnds))
    assert not [m for m in back.report.losses if "MT103" in m]
    tape = tmp_path / "fe56.endf"
    kika.write(back, str(tape), format="endf")
    written = read_endf(str(tape), mf_numbers=[3, 33])
    assert 103 not in written.mf[3].mt
    assert 103 in written.mf[33].mt
