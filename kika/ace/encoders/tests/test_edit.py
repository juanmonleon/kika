"""Adding or removing a reaction gives a table whose counts and locators all agree.

Tape-free, on the synthetic table. ``kika_dev/ace/audit_2026-10-14/edit_check.py``
in kika-workspace does the same on Fe-56 B-VIII.0 and has the result read by a
decoder that does not use kika and by OpenMC.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.contract import check
from kika.ace.encoders.edit import add_reaction, remove_reaction
from kika.ace.tests._synthetic import write_table
from kika.ace.writers.write_ace import write_ace


@pytest.fixture
def ace(tmp_path):
    return read_ace(str(write_table(tmp_path / "t.ace")))


def _through_a_file(table, tmp_path):
    path = tmp_path / "edited.ace"
    write_ace(table, str(path), overwrite=True)
    return read_ace(str(path))


def test_a_reaction_without_neutrons_goes_last_and_moves_the_total(ace, tmp_path):
    total = np.array(ace.esz_block.total_xs)
    absorption = np.array(ace.esz_block.absorption_xs)
    new = _through_a_file(add_reaction(ace, 108, [0.0, 0.25], energy_idx=1, q_value=-6.0), tmp_path)
    assert [int(m) for m in new.reaction_mt_data.incident_neutron] == [16, 51, 102, 108]
    assert new.header.nxs_array[4] == 4 and new.header.nxs_array[5] == 2
    np.testing.assert_array_equal(new.cross_section.reaction[108].xs_values, [0.0, 0.25])
    np.testing.assert_array_equal(new.esz_block.total_xs, total + [0.0, 0.0, 0.25])
    np.testing.assert_array_equal(new.esz_block.absorption_xs, absorption + [0.0, 0.0, 0.25])
    assert sorted(new.energy_distributions.incident_neutron) == [16, 51]
    assert check(new) == []
    # the input table is untouched
    np.testing.assert_array_equal(ace.esz_block.total_xs, total)


def test_a_reaction_with_neutrons_goes_after_the_others_that_have_them(ace, tmp_path):
    new = _through_a_file(add_reaction(
        ace, 90, [0.0, 0.1, 0.1], energy_idx=0, q_value=-1.0, ty=-1,
        angular=ace.angular_distributions.incident_neutron[51],
        energy_laws=ace.energy_distributions.incident_neutron[51]), tmp_path)
    assert [int(m) for m in new.reaction_mt_data.incident_neutron] == [16, 51, 90, 102]
    assert new.header.nxs_array[5] == 3
    assert [int(m) for m in new.reaction_mt_data.secondary_neutron_mt] == [16, 51, 90]
    (law,) = new.energy_distributions.incident_neutron[90]
    assert law.law == 3 and law.asquare == ace.energy_distributions.incident_neutron[51][0].asquare
    np.testing.assert_array_equal(new.angular_distributions.incident_neutron[90].cosine_grid[0], [-1.0, 0.0, 1.0])
    assert check(new) == []


def test_a_reaction_can_be_removed_and_the_total_follows(ace, tmp_path):
    total = np.array(ace.esz_block.total_xs)
    new = _through_a_file(remove_reaction(ace, 51), tmp_path)
    assert [int(m) for m in new.reaction_mt_data.incident_neutron] == [16, 102]
    np.testing.assert_allclose(new.esz_block.total_xs, total - [0.2, 0.2, 0.2])
    assert 51 not in new.angular_distributions.incident_neutron
    assert check(new) == []


def test_what_cannot_be_added_is_refused(ace):
    with pytest.raises(ValueError, match="already in the table"):
        add_reaction(ace, 51, [0.1, 0.1, 0.1], energy_idx=0, q_value=0.0)
    with pytest.raises(ValueError, match="do not end at the last"):
        add_reaction(ace, 108, [0.1], energy_idx=0, q_value=0.0)
    with pytest.raises(ValueError, match="needs its angular distribution"):
        add_reaction(ace, 90, [0.1, 0.1, 0.1], energy_idx=0, q_value=0.0, ty=1)
