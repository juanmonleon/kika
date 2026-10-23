"""The C field reader and writer give what the Python definitions give.

``kika._records`` keeps the Python functions as the definition and hands them to
``kika._records_native``, which decides the common case itself and calls them
for the rest. These tests compare the two on the cases that decide the grammar
and on random values; the whole-library sweep (every line of ENDF/B-VIII.1,
JEFF-4.0 and JENDL-5) is too long for the suite.
"""
import math
import random

import pytest

native = pytest.importorskip("kika._records_native")
import kika._records as records  # noqa: E402

py_parse_number, py_parse_record_values, py_format_fields, py_format_number = records._python_definitions


def outcome(function, *args, **kwargs):
    try:
        value = function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - the error is part of the result
        return ("raise", type(error).__name__, str(error))
    if isinstance(value, (tuple, list)):
        return type(value).__name__, tuple((type(v).__name__, repr(v)) for v in value)
    return type(value).__name__, repr(value)


def test_the_module_functions_are_the_native_ones():
    assert records.parse_number is native.parse_number
    assert records.parse_record_values is native.parse_record_values
    assert records.format_endf_fields is native.format_endf_fields
    assert records.format_endf_number is native.format_endf_number
    assert records.format_endf_number_precise is native.format_endf_number


FIELDS = [
    "", "           ", " 1.000000-5", "-1.234567+5", " 1.23456+10", " 1.5963-100", ".025870388",
    "-.061916773", " 20000000.1", " 1680612.75", "         12", "        -12", "          0",
    "  -0.000000", " -0.0", " 1.", "1.e+3", " 1.0E+05", " 1.0e-5", " 1+5", " 1.5+", " .+3", "-.-3",
    " +.+3", " 1.35462+ 7", "1 .5", "  1_000.5", " inf", " nan", " -Infinity", " 1.0+999",
    " 1.0-999", "+1.5", "+.5", "-.5", ".", "+", "-", "1.2.3", "1.2+3+4", " 12345678901",
    " 9.99999999+9", "\t1.5", "1.5\t", " 0x10", "١.٥", " 2.427894+7", " 7.0760435-4",
]


@pytest.mark.parametrize("text", FIELDS)
def test_a_field_reads_as_the_python_reader_reads_it(text):
    assert outcome(records.parse_number, text) == outcome(py_parse_number, text)


def line(fields, ident=" 125 3  1    5"):
    return "".join(f[:11].rjust(11) for f in fields).ljust(66) + ident


LINES = [
    line([" 1.000000-5", " 2.345678+6", ".025870388", "  -1.23-12", "12", "0"]),
    line(["", "", "", "", "", ""]),
    line([" 1.0E+05", "", "", "", "", ""]),                     # explicit exponent: Python path
    line([" 1.35462+ 7", "", "", "", "", ""]),                  # blank in the exponent
    line(["1", "2", "3"], ident=" 125 3  1"),                    # no sequence number
    line(["1", "2", "3"], ident=""),                            # no identification at all
    line(["1"], ident="  12 3  1    5") + "\n",
    line(["1"], ident=" 1x5 3  1    5"),                        # MAT not a number
    line(["1"], ident=" 125 3  1   x5"),                        # sequence not a number
    line(["1"], ident=" 125 3  1 1 5"),
    line(["1"], ident="-125 3  1    5"),
    line(["1"], ident=" 125\t3  1    5"),
    "short", "", " 1.000000-5" * 6,
    " 1.000000-5" * 5 + " 2.0",                                # 61 columns
]


@pytest.mark.parametrize("text", LINES)
def test_a_record_reads_as_the_python_reader_reads_it(text):
    assert outcome(records.parse_record_values, text) == outcome(py_parse_record_values, text)


def test_a_non_string_goes_to_the_python_reader():
    for value in (None, 3, b" 1.0+5"):
        assert outcome(records.parse_number, value) == outcome(py_parse_number, value)
        assert outcome(records.parse_record_values, value) == outcome(py_parse_record_values, value)


VALUES = [
    1.0e-5, 0.12640975, -0.061916773, 20000000.1, 1680612.75, 9.999995000000001e-10,
    9.9999995e-100, 9.99999949e9, 9.9999995e9, 1.5963e-100, 7.0760435e-4, 3, -3, 0.5, -0.5,
    1e10, 1e11, 99999999999.0, 1e-10, 1e-300, 5e-324, 1.7976931348623157e308, 2.427894e7,
    0.025870388, 0.279110747, 1.0000005, 1.00000049999, 2.5e-7, 10 ** 15, 2 ** 70, -2 ** 70,
    0.1, 0.9999999995, 0.99999995, 999999.95, 9999999.95, -9999999.95, 12345678.5, 123456789.5,
]


@pytest.mark.parametrize("value", VALUES)
def test_a_value_writes_as_the_python_writer_writes_it(value):
    assert outcome(records.format_endf_number, value) == outcome(py_format_number, value)
    assert outcome(records.format_endf_fields, [value]) == outcome(py_format_fields, [value])


def test_what_the_c_writer_does_not_decide_goes_to_python():
    import numpy as np
    for value in (0, 0.0, -0.0, None, math.inf, -math.inf, math.nan, True, np.float64(2.5e-3),
                  np.int64(7), "1.5", 10 ** 400):
        assert outcome(records.format_endf_number, value) == outcome(py_format_number, value)
    assert outcome(records.format_endf_number, 1.5, 12) == outcome(py_format_number, 1.5, 12)
    assert outcome(records.format_endf_number, None, 11) == outcome(py_format_number, None, 11)
    for values in ([1.0, math.nan], [np.float64(1.5), np.int64(2), "3.25", 0], (1.5,), [],
                   np.array([1.5, -2.5e-7]), [None], 3.0):
        assert outcome(records.format_endf_fields, values) == outcome(py_format_fields, values)
    assert outcome(records.format_endf_fields, [1.5], "float") == outcome(py_format_fields, [1.5], "float")
    assert (outcome(records.format_endf_fields, [1.5], data_format="float")
            == outcome(py_format_fields, [1.5], data_format="float"))


def test_random_values_write_and_read_as_python_does():
    rng = random.Random(20261023)
    values = []
    for _ in range(20000):
        exponent = rng.uniform(-320, 308)
        values.append(rng.choice((-1, 1)) * 10 ** exponent)
    for _ in range(20000):
        # Near a decimal tie of the seven-digit field, and plain-decimal ranges.
        mantissa = rng.randrange(1_000_000, 10_000_000) + 0.5
        values.append(mantissa * 10.0 ** rng.randint(-12, 4) * (1 + rng.choice((-1, 0, 1)) * 1e-16))
        values.append(rng.uniform(-1e10, 1e10))
    values = [v for v in values if math.isfinite(v) and v != 0]
    assert records.format_endf_fields(values) == py_format_fields(values)
    for v in values[::7]:
        assert records.format_endf_number(v) == py_format_number(v)
    texts = records.format_endf_fields(values)
    for text in texts:
        assert outcome(records.parse_number, text) == outcome(py_parse_number, text)
