"""An MT listed twice in a particle's MTRH keeps both entries, tape-free.

NJOY writes n + He-3 -> d + d (MT650) as two deuteron entries of MTRH, each
with its own SIGH yield, ANDH and DLWH record. The per-particle containers
key the first by the MT and the second by ``RepeatedMT(mt, 2)``
(``kika/ace/classes/ixs_keys.py``); before that the second overwrote the
first on read and the writers wrote one deuteron instead of two. The table is
``_synthetic``'s with ``particles=True``: MT51 produces a deuteron twice, with
different yields, angular records and LAW=33 constants.
"""
from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.classes.ixs_keys import RepeatedMT, ixs_keys
from kika.ace.contract import check
from kika.ace.encoders.layout import encode_xss, rebuild
from kika.ace.tests._synthetic import DEUTERON_ENTRIES, write_table

KEYS = [51, RepeatedMT(51, 2)]


@pytest.fixture
def ace(tmp_path):
    return read_ace(str(write_table(tmp_path / "d.ace", particles=True)), strict=True)


def _entries(ace):
    """(yields, angular class, LAW=33 constants) of each deuteron entry, in MTRH order."""
    sigh = ace.particle_production_xs.particle_cross_sections[1]
    andh = ace.angular_distributions.particle_production[0]
    dlwh = ace.energy_distributions.particle_production[0]
    out = []
    for key in ixs_keys(ace.reaction_mt_data.particle_production[0]):
        (law,) = dlwh[key]
        out.append((list(sigh[key].yields), type(andh[key]).__name__,
                    (law.law, law.threshold, law.energy_factor)))
    return out


def _stated():
    return [(e["yields"], "IsotropicAngularDistribution" if e["isotropic"] else "TabulatedAngularDistribution",
             (33,) + tuple(e["law33"])) for e in DEUTERON_ENTRIES]


def test_the_keys_of_an_mtrh_list():
    assert ixs_keys([600, 650, 650, 102, 650]) == [600, 650, RepeatedMT(650, 2), 102, RepeatedMT(650, 3)]
    assert all(type(k) is int for k in ixs_keys([5, 16, 103]))


def test_a_repeated_key_is_its_mt_as_a_number_and_its_own_key_in_a_dict():
    key = RepeatedMT(650, 2)
    assert int(key) == 650 and type(int(key)) is int and float(key) == 650.0
    assert key != 650 and 650 != key and key == RepeatedMT(650, 2) and key != RepeatedMT(650, 3)
    assert {650: "first", key: "second"}[key] == "second"
    assert sorted([key, 651, 650, 600]) == [600, 650, key, 651]
    assert str(key) == "650#2" and f"MT{key}" == "MT650#2"
    for again in (pickle.loads(pickle.dumps(key)), copy.deepcopy(key)):
        assert again == key and again.occurrence == 2
    with pytest.raises(ValueError):
        RepeatedMT(650, 1)


def test_both_entries_are_read_with_their_own_records(ace):
    assert [int(m) for m in ace.reaction_mt_data.particle_production[0]] == [51, 51]
    assert list(ace.particle_production_xs.particle_cross_sections[1]) == KEYS
    assert list(ace.angular_distributions.particle_production[0]) == KEYS
    assert list(ace.energy_distributions.particle_production[0]) == KEYS
    assert _entries(ace) == _stated()
    assert ace.angular_distributions.get_particle_production_mt_numbers(0) == KEYS
    assert ace.energy_distributions.get_particle_production_mt_numbers(0) == KEYS
    assert not check(ace)


def test_the_encoders_write_both_entries(ace):
    # The synthetic MT51 shares the elastic's angular table (11 words NJOY
    # would write twice); every IXS word is otherwise the table's own.
    enc = encode_xss(ace)
    assert enc.nxs[1] == ace.header.nxs_array[1] + 11
    n = enc.nxs[1] - enc.jxs[30] + 1
    ixs, ixs0 = enc.xss[enc.jxs[30]:], ace.xss_data[ace.header.jxs_array[30]:]
    assert len(ixs) == len(ixs0) == n
    locators = np.zeros(n, dtype=bool)
    locators[2:12] = True                                  # IXS: absolute positions
    np.testing.assert_array_equal(ixs[~locators], ixs0[~locators])
    np.testing.assert_array_equal(ixs[locators], ixs0[locators] + 11)
    again = rebuild(ace)
    assert _entries(again) == _stated()
    np.testing.assert_array_equal(encode_xss(again).xss, enc.xss)
