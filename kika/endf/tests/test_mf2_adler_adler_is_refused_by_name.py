"""MF2: an Adler-Adler range (LRF=4) is refused by name (ENDF-coverage roadmap T3).

Decided by Juan on 2026-10-08. GNDS has no node for Adler-Adler, and the census
of that day found no LRF=4 range in the 1 950 evaluations of ENDF/B-VIII.1,
JEFF-4.0 and JENDL-5, so there is no witness to build a reader against.

What was wrong before: the parser said "unrecognized LRU=1/LRF=4" and **read
on**, so the records of the Adler-Adler body were decoded as the next range's
CONT and everything after it in MF2/151 was garbage that still looked like a
section. Now the read stops at the range, says how many ranges it could not
reach, and each layer names the format: the parser, the model decoder and the
encoder.

The section here is synthetic, since no real one exists: a CONT that declares
an Adler-Adler range, a body kika does not read, and a second, ordinary range
after it that must **not** be read as anything.
"""
from __future__ import annotations

import pytest

from kika.endf.model_adapter import decodeMF2MT151
from kika.endf.parsers.parse_mf2 import ADLER_ADLER_REFUSAL, parse_mf2_mt151
from kika.endf.utils import ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, format_endf_data_line

_MAT = 2625
_FLOATS = [ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT] + [ENDF_FORMAT_INT] * 4


def _line(values, number):
    return format_endf_data_line(values, _MAT, 2, 151, number, formats=_FLOATS)


@pytest.fixture
def adlerSection():
    rows = [
        [26056.0, 55.4544, 0, 0, 1, 0],          # HEAD: NIS=1
        [26056.0, 1.0, 0, 0, 2, 0],              # isotope: NER=2
        [1.0e-5, 1.0e3, 1, 4, 0, 0],             # range 1: LRU=1, LRF=4
        [0.0, 0.5, 0, 0, 1, 0],                  # body kika does not read
        [55.4544, 0.0, 0, 0, 12, 2],
        [1.0e3, 2.0e4, 1, 2, 0, 0],              # range 2: must not be read
    ]
    return [_line(row, i + 1) for i, row in enumerate(rows)]


def test_the_parser_names_adler_adler_and_stops(adlerSection):
    with pytest.warns(UserWarning, match=r"LRF=4 \(Adler-Adler\)") as caught:
        section = parse_mf2_mt151(adlerSection, 151)
    said = " ".join(str(w.message) for w in caught)
    assert "the 1 range(s) after it in this isotope" in said

    ranges = section._isotopes[0].energy_ranges
    assert len(ranges) == 1, "the range after the Adler-Adler one was read"
    assert (ranges[0].lru, ranges[0].lrf) == (1, 4)
    assert ranges[0].parameters is None


def test_the_model_decoder_names_it_too(adlerSection):
    with pytest.warns(UserWarning):
        section = parse_mf2_mt151(adlerSection, 151)
    _resonances, _provenance, report = decodeMF2MT151(section)
    said = "\n".join(report.unsupported)
    assert ADLER_ADLER_REFUSAL in said
    assert "the ranges after it were not read" in said


def test_the_census_is_what_the_refusal_says():
    """The refusal quotes a census; this keeps the sentence and the number together."""
    assert "ENDF/B-VIII.1, JEFF-4.0 or JENDL-5" in ADLER_ADLER_REFUSAL
