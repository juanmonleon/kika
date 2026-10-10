"""The batch field formatter writes what the per-value formatters write."""
import numpy as np
import pytest

from kika._records import (ENDF_FORMAT_FLOAT, ENDF_FORMAT_PRECISE, format_endf_fields,
                           format_endf_number, format_endf_number_precise, parse_number)
from kika.endf.utils import PAD_BLANK, PAD_ZERO, format_data_pairs


def _values():
    rng = np.random.default_rng(3)
    values = list(10 ** rng.uniform(-300, 300, 3000) * rng.choice([-1, 1], 3000))
    values += list(10 ** rng.uniform(-12, 12, 20000) * rng.choice([-1, 1], 20000))
    values += [round(float(x), int(d)) for x, d in zip(10 ** rng.uniform(-3, 10, 5000), rng.integers(0, 8, 5000))]
    edge = 10.0 ** np.arange(-300, 300)
    values += list(edge) + list(np.nextafter(edge, 0)) + list(np.nextafter(edge, np.inf))
    values += list(9.999995 * 10.0 ** np.arange(-30, 30)) + list(9.99995 * 10.0 ** np.arange(-300, 300))
    values += [0.0, -0.0, 1e-5, 20000000.1, 0.5, 9.9999995, 99999999.95, 1e10, -1e10, 99999999999.5]
    return [float(v) for v in values]


@pytest.mark.parametrize("fmt, one", [(ENDF_FORMAT_PRECISE, format_endf_number_precise),
                                      (ENDF_FORMAT_FLOAT, format_endf_number)])
def test_batch_fields_match_the_per_value_formatter(fmt, one):
    values = _values()
    assert format_endf_fields(values, fmt) == [one(v) for v in values]


@pytest.mark.parametrize("value, field", [(9.999995000000001e-10, " 1.000000-9"),
                                          (-9.999995000000001e-10, "-1.000000-9"),
                                          (9.999950000999995e-100, " 1.00000-99")])
def test_a_value_just_below_an_exponent_width_change_terminates(value, field):
    # Five decimals carry it to the narrower exponent, six bring it back:
    # this used to loop forever.
    assert format_endf_number(value) == field
    assert format_endf_number_precise(value) == field
    assert len(field) == 11 and abs(parse_number(field) - value) <= 1e-5 * abs(value)


@pytest.mark.parametrize("pad", [PAD_BLANK, PAD_ZERO])
@pytest.mark.parametrize("fmt", [ENDF_FORMAT_PRECISE, ENDF_FORMAT_FLOAT])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_batch_pairs_match_the_line_by_line_body(pad, fmt, n, monkeypatch):
    import kika.endf.utils as utils
    rng = np.random.default_rng(n)
    x = np.sort(10 ** rng.uniform(-5, 7, n))
    y = 10 ** rng.uniform(-9, 3, n)
    # Start near the 99999 wrap so the sequence numbers are exercised too.
    fast = format_data_pairs(x, y, 9237, 3, 102, 99998, pad=pad, data_format=fmt)
    monkeypatch.setattr(utils, "_format_float_pairs", lambda *_: None)
    general = format_data_pairs(x.tolist(), y.tolist(), 9237, 3, 102, 99998, pad=pad, data_format=fmt)
    assert fast == general
    assert all(len(line) == 80 for line in fast[0])
