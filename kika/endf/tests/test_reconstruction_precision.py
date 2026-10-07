"""Opt-in ENDF decimal precision, preserving legacy evaluated serialization."""
import numpy as np
import pytest
from kika._records import (format_endf_number,format_endf_number_precise,
    parse_number,format_endf_data_line,ENDF_FORMAT_PRECISE)


@pytest.mark.parametrize('value',[0.,100.44989981925573,988832.1445758647,-988832.1445758647,
    1e-100,1e100,1e-12,1e12,9.9999999,9999999999.,-9999999999.])
def test_best_precision_is_never_worse_than_legacy(value):
    precise=format_endf_number_precise(value)
    assert len(precise)==11
    assert abs(parse_number(precise)-value)<=abs(parse_number(format_endf_number(value))-value)
    line=format_endf_data_line([value],2631,3,2,formats=[ENDF_FORMAT_PRECISE])
    assert len(line)==80
    assert line[:11]==precise


def test_fixed_notation_retains_the_digits_lost_in_normalized_exponent():
    assert format_endf_number_precise(100.44989981925573)=='100.4498998'
    assert format_endf_number(100.44989981925573)==' 1.004499+2'


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite_values_are_not_serialized(value):
    with pytest.raises(ValueError):format_endf_number_precise(value)


def test_precise_choice_matches_exhaustive_legal_fixed_decimal_search():
    import numpy as np
    from kika._records import format_endf_number,format_endf_number_precise,parse_number
    rng=np.random.default_rng(3941)
    values=np.r_[np.sign(rng.normal(size=401))*10.**rng.uniform(-100.,100.,401),999.99999999,-999.99999999,1e10,1e11,9.999999999e-10]
    for value in values:
        candidates=[format_endf_number(value)]
        for places in range(11):
            fixed=f'{value:.{places}f}'
            if len(fixed)<=11:candidates.append(fixed)
        best=min(abs(parse_number(field)-value) for field in candidates)
        assert abs(parse_number(format_endf_number_precise(value))-value)==best
