"""The ACE block parsers read what the ACE manual lays out, and nothing else.

One test per defect found in the 2026-10-09 review of the parser against the
manual (LA-UR-19-29016). Tape-free: the table comes from ``_synthetic``, which
writes each block word by word from the manual, or the law parsers and classes
are fed their arrays directly.
"""
from __future__ import annotations

import logging
import types

import numpy as np
import pytest
from scipy.integrate import quad

from kika.ace.classes.energy_distribution import tabular_math
from kika.ace.classes.energy_distribution.base import EnergyDistribution
from kika.ace.classes.energy_distribution.distributions.evaporation import EvaporationSpectrum
from kika.ace.classes.energy_distribution.distributions.kalbach_mann import KalbachMannDistribution
from kika.ace.classes.energy_distribution.distributions.level_scattering import LevelScattering
from kika.ace.classes.energy_distribution.distributions.maxwell import MaxwellFissionSpectrum
from kika.ace.classes.energy_distribution.distributions.tabular import ContinuousTabularDistribution
from kika.ace.classes.energy_distribution.distributions.watt import EnergyDependentWattSpectrum
from kika.ace.classes.header import Header
from kika.ace.parsers.laws.law_1 import parse_tabular_energy_distribution
from kika.ace.parsers.parse_ace import read_ace
from kika.ace.parsers.parse_gpd import read_gpd_block
from kika.ace.parsers.parse_yield_multipliers import read_photon_yield_multipliers
from kika.ace.tests._synthetic import AWR, write_table
from kika.ace.writers.write_ace import write_ace


@pytest.fixture
def table(tmp_path):
    return read_ace(str(write_table(tmp_path / "t.ace")))


def _stub(xss, nxs=None, jxs=None):
    header = Header()
    header.nxs_array = nxs or [0] * 17
    header.jxs_array = jxs or [0] * 33
    return types.SimpleNamespace(header=header, xss_data=np.asarray([0.0] + list(xss), dtype=float))


# --- angular distributions --------------------------------------------------

def test_every_land_reaction_gets_its_angular_distribution(table):
    # The store sat inside a debug sample guard (i < 3), so only the first
    # three LAND reactions were kept. LOCB=-1 is the DLW-correlated one.
    ad = table.angular_distributions
    assert sorted(ad.incident_neutron) == [16, 51]
    assert type(ad.incident_neutron[16]).__name__ == "KalbachMannAngularDistribution"
    assert ad.incident_neutron[51].cosine_grid == [[-1.0, 0.0, 1.0]]


def test_the_elastic_locator_is_read_without_secondary_neutrons(tmp_path):
    # LAND always exists (manual Table 16): H-1 has NXS(5)=0 and an elastic AND.
    ace = read_ace(str(write_table(tmp_path / "h.ace", secondaries=False)))
    assert ace.angular_distributions.elastic is not None
    assert ace.angular_distributions.elastic.energies == [1.0e-11, 20.0]


def test_tabulated_angular_mixes_the_two_tables_on_their_union_grid(table):
    el = table.angular_distributions.elastic
    df = el.to_dataframe(10.0, interpolate=False)       # half-way in energy... by lin-lin, f ~ 0.5
    f = (10.0 - 1e-11) / (20.0 - 1e-11)
    expected = (1 - f) * np.array([0.25, 0.5, 0.75]) + f * np.array([0.75, 0.5, 0.25])
    np.testing.assert_allclose(df["pdf"].to_numpy(), expected)
    # outside the grid the end table is used, not an isotropic stand-in
    np.testing.assert_allclose(el.to_dataframe(30.0)["pdf"].to_numpy(), [0.75, 0.5, 0.25])


def test_kalbach_angular_is_the_marginal_over_outgoing_energy(table):
    km = table.angular_distributions.incident_neutron[16]
    mu = np.linspace(-1, 1, 4001)
    pdf = km.angular_pdf(20.0, table, mu)
    assert np.trapezoid(pdf, mu) == pytest.approx(1.0, abs=1e-6)
    # energy=None used to call a method that does not exist
    df = km.to_dataframe(None, table)
    assert sorted(df["energy"].unique()) == [11.3, 20.0]


# --- energy distributions ---------------------------------------------------

def test_law44_tables_are_located_from_jed_not_idat(table):
    law = table.energy_distributions.incident_neutron[16][0]
    assert [t["n_points"] for t in law.distributions] == [2, 3]
    np.testing.assert_array_equal(law.distributions[1]["a"], [1.0, 2.0, 3.0])


def test_interpolated_tables_stay_normalised_and_scale_their_range(table):
    law = table.energy_distributions.incident_neutron[16][0]
    d = law.get_interpolated_distribution(15.65)        # f = 0.5
    assert d["e_out"][0] == 0.0 and d["e_out"][-1] == pytest.approx(0.5 * 0.1 + 0.5 * 8.0)
    assert np.trapezoid(d["pdf"], d["e_out"]) == pytest.approx(1.0)
    assert d["cdf"][-1] == pytest.approx(1.0)


def test_law4_interpolation_runs_and_keeps_discrete_lines():
    law = ContinuousTabularDistribution()
    law.incident_energies = np.array([1.0, 3.0])
    law.distributions = [
        {"intt": 2, "n_discrete": 1, "n_points": 3, "e_out": [0.5, 0.0, 1.0],
         "pdf": [0.2, 0.8, 0.8], "cdf": [0.2, 0.2, 1.0]},
        {"intt": 2, "n_discrete": 0, "n_points": 2, "e_out": [0.0, 2.0],
         "pdf": [0.5, 0.5], "cdf": [0.0, 1.0]},
    ]
    d = law.get_interpolated_distribution(2.0)
    np.testing.assert_allclose(d["discrete_energies"], [0.5])
    np.testing.assert_allclose(d["discrete_probabilities"], [0.1])
    total = np.trapezoid(d["pdf"], d["e_out"]) + d["discrete_probabilities"].sum()
    assert total == pytest.approx(1.0)


def test_histogram_between_incident_energies_takes_the_lower_table():
    law = KalbachMannDistribution()
    law.incident_energies = np.array([1.0, 3.0])
    law.nbt, law.interp = [2], [1]
    t = {"intt": 1, "n_discrete": 0, "e_out": [0.0, 1.0], "pdf": [1.0, 0.0], "cdf": [0.0, 1.0],
         "r": [0.0, 0.0], "a": [0.0, 0.0]}
    law.distributions = [t, dict(t, e_out=[0.0, 4.0], pdf=[0.25, 0.0])]
    assert law.get_interpolated_distribution(2.9)["e_out"][-1] == 1.0


def test_law1_tables_are_equiprobable_bins_and_interpolate_their_bounds():
    # NR=0, NE=2, E=(1,3), NET=3, two Eout tables (manual Table 26)
    ace = _stub([0, 2, 1.0, 3.0, 3, 0.0, 1.0, 2.0, 0.0, 2.0, 4.0])
    base = EnergyDistribution(law=1, idat=1)
    law = parse_tabular_energy_distribution(ace, base, 1)
    assert law.distribution_data[0]["cdf"].tolist() == [0.0, 0.5, 1.0]
    bounds, pdf = law.get_outgoing_energy_distribution(2.0)
    assert bounds == [0.0, 1.5, 3.0]
    assert pdf == pytest.approx([1 / 3, 1 / 3, 0.0])


def test_only_neutron_reactions_carry_tyr_yields(table):
    assert not hasattr(table.energy_distributions, "photon_yields")


# --- the analytic laws --------------------------------------------------------

@pytest.mark.parametrize("theta", [0.5, 1.3])
def test_maxwell_and_evaporation_normalisations(theta):
    e, u = 5.0, -1.0
    i7 = MaxwellFissionSpectrum(restriction_energy=u).calculate_normalization_constant(e, theta)
    i9 = EvaporationSpectrum(restriction_energy=u).calculate_normalization_constant(e, theta)
    assert i7 == pytest.approx(quad(lambda x: np.sqrt(x) * np.exp(-x / theta), 0, e - u)[0], rel=1e-10)
    assert i9 == pytest.approx(quad(lambda x: x * np.exp(-x / theta), 0, e - u)[0], rel=1e-10)


@pytest.mark.parametrize("a,b", [(0.988, 2.249), (1.2, 4.0)])
def test_watt_normalisation(a, b):
    e, u = 5.0, -1.0
    i11 = EnergyDependentWattSpectrum(restriction_energy=u).calculate_normalization_constant(e, a, b)
    assert i11 == pytest.approx(quad(lambda x: np.exp(-x / a) * np.sinh(np.sqrt(b * x)), 0, e - u)[0], rel=1e-10)


@pytest.mark.parametrize("mu", [-1.0, 0.0, 1.0])
def test_level_scattering_lab_energy(mu):
    q, en = -0.8467, 3.0
    lvl = LevelScattering(aplusoaabsq=(AWR + 1) / AWR * abs(q), asquare=(AWR / (AWR + 1)) ** 2)
    ecm = lvl.get_cm_energy(en)
    ref = ecm + en / (AWR + 1) ** 2 + 2 * mu * np.sqrt(en * ecm) / (AWR + 1)
    assert lvl.get_lab_energy(en, mu) == pytest.approx(ref)


def test_kalbach_pdf_is_normalised_and_does_not_overflow():
    mu = np.linspace(-1, 1, 20001)
    for a, r in [(0.0, 0.3), (1e-9, 0.5), (2.0, 0.7), (900.0, 0.2)]:
        pdf = tabular_math.kalbach_pdf(mu, a, r)
        assert np.all(np.isfinite(pdf))
        if a < 100:
            assert np.trapezoid(pdf, mu) == pytest.approx(1.0, abs=1e-6)


# --- the other blocks -----------------------------------------------------------

def test_tyr_has_one_entry_per_mtr_reaction(table, tmp_path):
    # NXS(4) already excludes elastic; TYR used to drop the last reaction, and
    # a table with a single reaction lost the whole block.
    np.testing.assert_array_equal(table.particle_release.incident_neutron, [-2, -1, 0])
    single = read_ace(str(write_table(tmp_path / "h.ace", secondaries=False)))
    np.testing.assert_array_equal(single.particle_release.incident_neutron, [0])


def test_yp_is_read_from_jxs20():
    jxs = [0] * 33
    jxs[20] = 3
    ace = _stub([9.0, 9.0, 2, 102, 103], nxs=[0, 5, 0, 0, 0, 0, 7] + [0] * 10, jxs=jxs)
    assert read_photon_yield_multipliers(ace).multiplier_mts == [102, 103]


def test_gpd_total_photon_xs_is_read_whenever_jxs12_is_set():
    jxs = [0] * 33
    jxs[12], jxs[13] = 1, 4
    nxs = [0] * 17
    nxs[3] = 3
    ace = _stub([1.0, 2.0, 3.0, 102.0], nxs=nxs, jxs=jxs)
    gpd = read_gpd_block(ace)
    np.testing.assert_array_equal(gpd.total_xs, [1.0, 2.0, 3.0])
    assert not gpd.outgoing_energies


def test_a_201_header_reads_and_writes_back(tmp_path):
    ace = read_ace(str(write_table(tmp_path / "v2.ace", header="2.0.1")))
    assert ace.header.atomic_weight_ratio == pytest.approx(AWR)
    assert ace.header.date == "2018-05-02"
    assert ace.header.comment_line_count == 2
    out = tmp_path / "v2_out.ace"
    write_ace(ace, str(out), overwrite=True)
    again = read_ace(str(out))
    np.testing.assert_array_equal(again.xss_data, ace.xss_data)
    assert again.header.zaid == ace.header.zaid and again.header.date == ace.header.date


def test_the_comparison_package_imports_and_compares(tmp_path):
    from kika.ace.comparison import compare_ace_files
    path = str(write_table(tmp_path / "t.ace"))
    assert compare_ace_files(path, path, verbose=False)


def test_a_step_in_a_table_survives_interpolation():
    # NJOY closes a range with a repeated node (a step down to zero); the union
    # grid used to collapse it and lose the mass on one side (Zr-90 MT5: 0.5)
    low = {"intt": 2, "n_discrete": 0, "e_out": [0.0, 1e-11], "pdf": [2e11, 0.0], "cdf": [0.0, 1.0]}
    high = {"intt": 2, "n_discrete": 0, "e_out": [0.0, 0.3307, 0.3307],
            "pdf": [0.0, 2 / 0.3307, 0.0], "cdf": [0.0, 1.0, 1.0]}
    d = tabular_math.interpolate_tables(low, high, 0.5, 2)
    assert np.trapezoid(d["pdf"], d["e_out"]) == pytest.approx(1.0)
    assert d["cdf"][-1] == pytest.approx(1.0)


def test_a_histogram_and_a_linlin_table_mix_exactly():
    hist = {"intt": 1, "n_discrete": 0, "e_out": [0.0, 1.0, 2.0], "pdf": [0.25, 0.75, 0.0], "cdf": [0.0, 0.25, 1.0]}
    lin = {"intt": 2, "n_discrete": 0, "e_out": [0.0, 2.0], "pdf": [0.5, 0.5], "cdf": [0.0, 1.0]}
    d = tabular_math.interpolate_tables(hist, lin, 0.5, 2)
    assert np.trapezoid(d["pdf"], d["e_out"]) == pytest.approx(1.0)
    probe = np.array([0.5, 1.5])
    assert np.interp(probe, d["e_out"], d["pdf"]) == pytest.approx([0.375, 0.625])


@pytest.mark.parametrize("ty", [-6, 6, -19, 0, 101, -105])
def test_any_neutron_multiplicity_is_a_valid_ty(ty):
    from kika.ace.parsers.parse_tyr import _is_valid_ty_value
    assert _is_valid_ty_value(ty)


# --- the extended particle blocks, on a real table (shared tree, ``tape``) -------

@pytest.mark.tape
def test_particle_blocks_are_read_per_particle_type(fe56_ace):
    # NTRO/IXS/HPD looked for an attribute the Ace class does not have, YH was
    # read from the DLWH word, and SIGH kept one particle per MT.
    ace = read_ace(str(fe56_ace))
    x, j = ace.xss_data, ace.header.jxs_array
    n_types = ace.header.nxs_array[7]
    assert n_types == 5
    assert len(ace.secondary_particle_reactions.reaction_counts) == n_types
    assert ace.secondary_particle_cross_sections.has_data
    for i in range(1, n_types + 1):
        ly = int(x[j[32] + 10 * (i - 1) + 9])
        nyh = int(x[ly])
        assert ace.particle_yield_multipliers.particle_multipliers[i] == x[ly + 1:ly + 1 + nyh].astype(int).tolist()
    sigh = ace.particle_production_xs
    assert sigh.has_data
    assert sorted(sigh.particle_cross_sections) == list(range(1, n_types + 1))
    # MT5 produces every particle type; each keeps its own yield
    assert all(sigh.get_reaction_xs(5, i) is not None for i in range(1, n_types + 1))
