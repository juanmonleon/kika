"""Numeric acceleration must preserve decimal rounding and record semantics."""
import math
import random

import pytest

from kika._records import parse_line, parse_number, parse_record_values
from kika.endf.utils import parse_data_pairs, parse_data_values, parse_interp_pairs
from kika.endf.parsers.parse_mf33 import _read_list_values
from kika.endf.parsers.parse_mf34 import _read_floats


def record(fields, ids="2631 33102    1"):
    return "".join(str(f).rjust(11) for f in fields) + ids


def test_implicit_exponents_are_one_correctly_rounded_decimal_conversion():
    rng = random.Random(1701)
    for _ in range(10000):
        mantissa = f"{rng.randint(-9999999, 9999999) / 1000000:.6f}"
        exponent = rng.randint(-320, 310)
        expected = float(f"{mantissa}e{exponent:+d}")
        actual = parse_number(f"{mantissa}{exponent:+d}")
        assert actual == expected
        assert type(actual) is (int if expected.is_integer() else float)
    assert parse_number("2.427894+7") == 24278940


@pytest.mark.parametrize("text, expected", [
    ("", None), ("   ", None), ("word", None), ("12", 12), (".+3", None),
    ("1_000.0", 1000), ("-1.25E-3", -0.00125),
    ("+.25+2", 25), ("1.+2", 100), ("-.25-2", -0.0025),
    ("1.0+9999", math.inf), ("prefix1.25+2suffix", 125),
    ("1_0.2+3", 200), ("1.2+3garbage", 1200),
])
def test_standard_syntax_and_legacy_fallback(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["nan", "NaN", "+nan", "-nan"])
def test_nan_remains_nan(text):
    assert math.isnan(parse_number(text))


def test_positional_fields_keep_padding_truncation_and_zero_policies():
    lines = [record(["1.000000+0", "", "invalid", 2, 3, 4]),
             record([5, 6, "", "", "", ""])]
    assert parse_record_values(lines[0]) == (1, None, None, 2, 3, 4)
    # The generic LIST reader treats blanks as zeros, not invalid text.
    assert parse_data_values(lines, 0, 7) == ([1, 0.0, 2, 3, 4, 5, 6], 2)
    # MF33/MF34 keep their historical skip-blank policy and consume another
    # line when the requested number of actual values has not been reached.
    assert _read_list_values(lines, 0, 6) == ([1, 2, 3, 4, 5, 6], 2)
    assert _read_floats(lines, 0, 6) == ([1, 2, 3, 4, 5, 6], 2)
    assert parse_data_pairs(lines, 0, 2) == ([3], [4], 1)
    assert parse_interp_pairs(lines, 0, 2) == ([], 1)
    assert parse_record_values("short") == (None,) * 6


@pytest.mark.parametrize("column", [(66, 70), (70, 72), (72, 75), (75, 80)])
def test_invalid_identification_columns_still_raise(column):
    line = record([1, 2, 3, 4, 5, 6])
    a, b = column
    line = line[:a] + "x".rjust(b-a) + line[b:]
    for read in (parse_line, parse_record_values):
        with pytest.raises(ValueError):
            read(line)


@pytest.mark.parametrize("width", [65, 66, 74, 75, 79, 80])
def test_short_records_keep_the_same_numeric_fields(width):
    line = record([1, "2.500000-1", "", -3, 4, 5])[:width]
    expected = tuple(parse_line(line).get(f"C{i}") for i in range(1, 7))
    assert parse_record_values(line) == expected
