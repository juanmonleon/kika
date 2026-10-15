"""Light charged particles in a table kika processed: PTYPE, NTRO, IXS and their blocks.

O-16 of ENDF/B-VIII.1 emits protons and alphas (MT 103-107 and their levels,
the MF6 continua). Each particle's blocks are checked against the table's own
reactions: the production cross section is Σ y·σ of the reactions MTRH names,
the yields are MFTYPE 12 on their own reaction, and a level reaction's
particle follows law 33 with the constants its Q fixes.
"""
import numpy as np
import pytest

from kika.ace import read_ace
from kika.endf import ProcessingPlan, process_endf

pytestmark = pytest.mark.tape

AWP = {9: 0.99862347, 31: 1.99626, 32: 2.98960, 33: 2.98903, 34: 3.96713}


@pytest.fixture(scope="module")
def table(o16_b81_tape, tmp_path_factory):
    out = tmp_path_factory.mktemp("o16")
    (path,) = process_endf(o16_b81_tape, ProcessingPlan(temperatures=(293.6,), workers=1)).write_ace(out)
    return read_ace(str(path), strict=True)


def _sigma(ace, mt):
    E = np.asarray(ace.esz_block.energies)
    r = ace.cross_section.reaction[mt]
    sigma = np.zeros(E.size)
    sigma[int(r.energy_idx):int(r.energy_idx) + len(r._xs_entries)] = r._xs_entries
    return sigma


def test_protons_and_alphas_are_written(table):
    ptypes = list(table.secondary_particle_types.particle_ids)
    assert 9 in ptypes and 34 in ptypes and ptypes == sorted(ptypes)


def test_each_particle_is_produced_by_the_reactions_mtrh_names(table):
    E = np.asarray(table.esz_block.energies)
    for i, ipt in enumerate(table.secondary_particle_types.particle_ids):
        mts = [int(m) for m in table.reaction_mt_data.particle_production[i]]
        assert mts == sorted(mts)
        production = np.zeros(E.size)
        for mt in mts:
            record = table.particle_production_xs.particle_cross_sections[i + 1][mt]
            assert record.mftype == 12 and record.mtmult == mt
            production += np.interp(E, record.energies, record.yields, left=0.0, right=0.0) * _sigma(table, mt)
        hpd = table.secondary_particle_cross_sections.particle_data[i + 1]
        start = int(hpd.energy_grid_index) - 1
        np.testing.assert_allclose(hpd.xs_values, production[start:start + len(hpd.xs_values)],
                                   rtol=1e-10, atol=1e-14 * production.max())
        assert np.all(np.asarray(hpd.heating_numbers) >= 0)


def test_a_level_reactions_particle_is_law_33(table):
    A = float(table.header.atomic_weight_ratio)
    q = dict(zip((int(m) for m in table.reaction_mt_data.incident_neutron),
                 np.asarray(table.q_values.q_values, dtype=float)))
    seen = 0
    for i, ipt in enumerate(table.secondary_particle_types.particle_ids):
        for mt in (int(m) for m in table.reaction_mt_data.particle_production[i]):
            if not (600 <= mt <= 648 or 800 <= mt <= 848):
                continue
            (law,) = table.energy_distributions.particle_production[i][mt]
            assert law.law == 33
            assert law.threshold == pytest.approx(-(A + 1) / A * q[mt], rel=1e-10)
            assert law.energy_factor == pytest.approx(A * (A + 1 - AWP[ipt]) / (A + 1) ** 2, rel=1e-10)
            seen += 1
    assert seen
