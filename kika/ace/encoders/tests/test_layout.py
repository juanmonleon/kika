"""The layout puts an Ace's blocks back into an XSS, and a changed object stays consistent.

Tape-free, on the synthetic table of ``kika/ace/tests/_synthetic.py``. The real
gate (every word of every table of two libraries) runs outside
the test suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.contract import AceContractError, check
from kika.ace.encoders.layout import encode_xss, rebuild
from kika.ace.tests._synthetic import write_table
from kika.ace.writers.write_ace import write_ace


@pytest.fixture
def ace(tmp_path):
    return read_ace(str(write_table(tmp_path / "t.ace")))


def test_the_layout_is_njoys_and_a_shared_table_is_written_twice(ace):
    # The synthetic MT51 points at the elastic's first angular table, which the
    # manual allows and NJOY never writes: the encoder gives each reaction its
    # own copy (11 words), and everything else word for word.
    enc = encode_xss(ace)
    assert enc.nxs[1] == ace.header.nxs_array[1] + 11
    jxs0, jxs = ace.header.jxs_array, enc.jxs
    n = jxs0[9] + 29  # up to MT51's NE and E, before its LC
    np.testing.assert_array_equal(enc.xss[:n], ace.xss_data[:n])
    np.testing.assert_array_equal(enc.xss[jxs[10]:], ace.xss_data[jxs0[10]:])
    new = rebuild(ace)
    np.testing.assert_array_equal(new.angular_distributions.incident_neutron[51].cosine_grid[0], [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(encode_xss(new).xss, enc.xss)


def test_a_table_without_secondary_neutrons_is_recomposed(tmp_path):
    ace = read_ace(str(write_table(tmp_path / "h.ace", secondaries=False)))
    np.testing.assert_array_equal(encode_xss(ace).xss, ace.xss_data)


def test_counts_and_locators_are_spelled_as_integers(ace):
    enc = encode_xss(ace)
    jxs = enc.jxs
    assert enc.is_int[jxs[3] - 1:jxs[3] - 1 + 3].all()          # MTR
    assert not enc.is_int[jxs[4] - 1:jxs[4] - 1 + 3].any()      # LQR
    assert not enc.is_int[jxs[1] - 1:jxs[1] - 1 + 3].any()      # energies


def test_a_point_added_to_an_angular_table_moves_everything_after_it(ace, tmp_path):
    ace = rebuild(ace)  # the NJOY layout, without the shared table
    before = ace.header
    el = ace.angular_distributions.elastic
    el._cosine_grid[0] = np.array([-1.0, -0.5, 0.0, 1.0])
    el._pdf[0] = np.array([0.25, 0.375, 0.5, 0.75])
    el._cdf[0] = np.array([0.0, 0.15625, 0.375, 1.0])
    new = rebuild(ace)
    assert new.header.nxs_array[1] == before.nxs_array[1] + 3
    np.testing.assert_array_equal(new.angular_distributions.elastic.cosine_grid[0], [-1.0, -0.5, 0.0, 1.0])
    # the blocks after AND were moved, and still read as before
    assert new.header.jxs_array[11] == before.jxs_array[11] + 3
    assert [d.law for d in new.energy_distributions.incident_neutron[16]] == [44]
    np.testing.assert_array_equal(new.energy_distributions.incident_neutron[16][0].distributions[1]["e_out"],
                                  [0.0, 2.0, 8.0])
    assert check(new) == []
    # and it survives the file
    path = tmp_path / "moved.ace"
    write_ace(new, str(path))
    np.testing.assert_array_equal(read_ace(str(path)).xss_data, new.xss_data)


def _drop_reaction(ace, mt):
    rmt = ace.reaction_mt_data
    keep = [k for k, m in enumerate(rmt.incident_neutron) if int(m) != mt]
    rmt.incident_neutron = rmt.incident_neutron[keep]
    rmt.secondary_neutron_mt = rmt.secondary_neutron_mt[rmt.secondary_neutron_mt != mt]
    ace.q_values.q_values = ace.q_values.q_values[keep]
    ace.particle_release.incident_neutron = ace.particle_release.incident_neutron[keep]
    ace.cross_section.reaction.pop(mt)
    ace.angular_distributions.incident_neutron.pop(mt, None)
    ace.energy_distributions.incident_neutron.pop(mt, None)


def test_a_reaction_can_be_removed(ace):
    _drop_reaction(ace, 51)
    new = rebuild(ace)
    assert [int(m) for m in new.reaction_mt_data.incident_neutron] == [16, 102]
    assert new.header.nxs_array[4] == 2 and new.header.nxs_array[5] == 1
    assert sorted(new.energy_distributions.incident_neutron) == [16]
    assert new.cross_section.reaction[102]._xs_entries.tolist() == [0.3, 0.3, 0.3]
    assert check(new) == []


def test_blocks_that_disagree_are_refused_with_the_reason(ace):
    ace.q_values.q_values = ace.q_values.q_values[:2]
    with pytest.raises(AceContractError, match="LQR: 2 Q values for 3 MTR reactions"):
        encode_xss(ace)


def test_a_reaction_with_secondary_neutrons_needs_its_laws(ace):
    ace.energy_distributions.incident_neutron.pop(51)
    with pytest.raises(AceContractError, match="DLW MT=51: no energy distribution"):
        encode_xss(ace)
