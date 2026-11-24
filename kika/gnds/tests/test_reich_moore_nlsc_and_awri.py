"""A Reich-Moore range keeps NLSC through GNDS and does not declare its AWRI lost.

NLSC has no GNDS node below 2.2. FUDGE keeps it as ``LvaluesNeededForConvergence=``
in the range's ``ENDFconversionFlags`` (beside ``LRF3``), and from 2.2 as
``LValuesNeededForAngularConvergence`` when it exceeds NLS; kika writes the flag
and reads both. The per-L AWRI is the formalism's target mass in its own PoPs, so
it is not lost where that mass states it.
"""
from __future__ import annotations

import re
from pathlib import Path

import kika
from kika.endf import read_endf

TAPE = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data" / "micro_mn55_mf2.endf"
HREF = "/reactionSuite/resonances/resolved/RMatrix[@label='eval']"


def _gnds(tmp_path):
    path = tmp_path / "mn55.gnds.xml"
    report = kika.write(kika.read(str(TAPE)), str(path), format="gnds")
    return path, report


def test_the_flag_is_fudges_and_nothing_is_declared_lost(tmp_path):
    path, report = _gnds(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert f'flags="LRF3,LvaluesNeededForConvergence=4" href="{HREF}"' in text
    assert not [m for m in report.losses if "NLSC" in m or "atomicWeightRatio" in m]


def test_nlsc_comes_back_to_the_model_and_to_mf2(tmp_path):
    path, _ = _gnds(tmp_path)
    suite = kika.read(str(path))
    assert suite.resonances.resolved[0].formalism.angularLCount == 4
    out = tmp_path / "mn55.endf"
    report = kika.write(suite, str(out), format="endf")
    (isotope,) = read_endf(str(out), mf_numbers=[2]).mf[2].mt[151].isotopes
    assert isotope.energy_ranges[0].parameters.nlsc == 4
    assert not [m for m in report.approximations if "NLSC" in m]


def test_rewriting_the_gnds_keeps_one_flag(tmp_path):
    path, _ = _gnds(tmp_path)
    again = tmp_path / "again.gnds.xml"
    kika.write(kika.read(str(path)), str(again), format="gnds")
    assert again.read_text(encoding="utf-8").count(f'href="{HREF}"') == 1


def test_the_gnds_2_2_attribute_is_read(tmp_path):
    path, _ = _gnds(tmp_path)
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'\s*<conversion flags="LRF3[^>]*/>', "", text)
    text = text.replace('<RMatrix label="eval"',
                        '<RMatrix label="eval" LValuesNeededForAngularConvergence="7"', 1)
    edited = tmp_path / "attribute.gnds.xml"
    edited.write_text(text, encoding="utf-8")
    assert kika.read(str(edited)).resonances.resolved[0].formalism.angularLCount == 7
