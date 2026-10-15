"""The energy-law encoders are the exact inverse of the law parsers, tape-free.

Each law object is encoded into a DLW-like block (LA-UR-19-29016, Tables
30-52), read back with the reader's own chain walker, and encoded again: the
words and their integer flags must come back identical, and the decoded
fields equal to what went in. The chain starts a few words into the block,
so every JED-relative locator (IDAT, LNW, L, LC, LMU) is exercised away from
position 1.

Laws 1, 5, 22, 24 and 67 have no witness in any processed library on the
share; for them these tests are the only check. The other laws were also
compared word for word, flags included, against NJOY tables.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.ace.classes.energy_distribution.distributions.angle_energy import (
    LaboratoryAngleEnergyDistribution, TabulatedAngleEnergyDistribution)
from kika.ace.classes.energy_distribution.distributions.discrete import DiscreteEnergyDistribution
from kika.ace.classes.energy_distribution.distributions.evaporation import (
    EvaporationSpectrum, GeneralEvaporationSpectrum)
from kika.ace.classes.energy_distribution.distributions.kalbach_mann import KalbachMannDistribution
from kika.ace.classes.energy_distribution.distributions.level_scattering import (
    LevelScattering, ParticleLevelScattering)
from kika.ace.classes.energy_distribution.distributions.maxwell import MaxwellFissionSpectrum
from kika.ace.classes.energy_distribution.distributions.phase_space import NBodyPhaseSpaceDistribution
from kika.ace.classes.energy_distribution.distributions.tabular import (
    ContinuousTabularDistribution, TabularEnergyDistribution)
from kika.ace.classes.energy_distribution.distributions.tabular_functions import (
    TabularEnergyMultipliers, TabularLinearFunctions)
from kika.ace.classes.energy_distribution.distributions.watt import EnergyDependentWattSpectrum
from kika.ace.classes.energy_distribution.yields.energy_dependent import EnergyDependentYield
from kika.ace.encoders._words import Words
from kika.ace.encoders.laws import encode_energy_dependent_yield, encode_law_chain, encode_law_data
from kika.ace.parsers.laws import parse_energy_dependent_yield
from kika.ace.parsers.parse_energy_distributions import _read_laws
from kika.ace.tests.test_reader_laws_and_report import _ace, _chain, _law4, _law22, _record

PAD = 3  # words of the block before the chain: LOCC = 4


def _app(dist, energies=(1.0e-5, 20.0), probs=(1.0, 1.0), nbt=(), interp=()):
    dist.applicability_energies = np.asarray(energies, dtype=float)
    dist.applicability_probabilities = np.asarray(probs, dtype=float)
    dist.applicability_nbt = list(nbt)
    dist.applicability_interp = list(interp)
    return dist


def _encode(laws):
    w = Words()
    w.real(*([0.0] * PAD))
    locc = w.next
    encode_law_chain(laws, w)
    return w, locc


def _round_trip(*laws):
    """Encode the chain, read it back as the DLW reader does, re-encode; returns the read laws."""
    w, locc = _encode(list(laws))
    values, flags = w.arrays()
    ace = _ace(values)
    back = _read_laws(ace, "DLW", "MT=16", 1, locc)
    assert not ace.read_report.lost
    assert len(back) == len(laws)
    w2, _ = _encode(back)
    v2, f2 = w2.arrays()
    np.testing.assert_array_equal(v2, values)
    np.testing.assert_array_equal(f2, flags)
    for a, b in zip(laws, back):
        assert b.law == a.law
        np.testing.assert_array_equal(b.applicability_energies, a.applicability_energies)
        np.testing.assert_array_equal(b.applicability_probabilities, a.applicability_probabilities)
        assert list(b.applicability_nbt) == list(a.applicability_nbt)
        assert list(b.applicability_interp) == list(a.applicability_interp)
    return back if len(back) > 1 else back[0]


def _eq(a, b):
    np.testing.assert_array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def _words_of(rec):
    """The XSS words of a hand-laid record of the reader tests (chain at XSS(1))."""
    return np.asarray(rec, dtype=float)


def _reencode_hand_laid(rec):
    """Read a record laid out word by word from the manual, encode it back: same words."""
    ace = _ace(rec)
    laws = _read_laws(ace, "DLW", "MT=16", 1, 1)
    w = Words()
    encode_law_chain(laws, w)
    np.testing.assert_array_equal(w.arrays()[0], _words_of(rec))
    return w


# --- the record and the chain (Table 30) -------------------------------------

def test_a_chain_puts_each_ldat_after_its_record_and_lnw_on_the_next_record():
    first = _app(ParticleLevelScattering(threshold=2.0, energy_factor=0.5), probs=(0.25, 0.25))
    second = _app(LevelScattering(aplusoaabsq=1.0, asquare=0.9), probs=(0.75, 0.75, 0.75),
                  energies=(1.0e-5, 1.0, 20.0), nbt=[3], interp=[2])
    w, locc = _encode([first, second])
    v, f = w.arrays()
    # record 1: LNW, LAW, IDAT, NR=0, NE=2, E, E, P, P, then Ec, C
    assert v[locc - 1:locc + 2].tolist() == [locc + 11, 33, locc + 9]
    assert f[locc - 1:locc + 4].all() and not f[locc + 4:locc + 8].any()
    rec2 = locc + 11
    assert v[rec2 - 1:rec2 + 2].tolist() == [0, 3, rec2 + 13]   # last law: LNW = 0
    assert v[rec2 + 2:rec2 + 6].tolist() == [1, 3, 2, 3]       # NR, NBT, INT, NE
    laws = _round_trip(first, second)
    assert [law.law for law in laws] == [33, 3]


def test_the_hand_laid_chain_of_the_reader_tests_encodes_to_the_same_words():
    _reencode_hand_laid(_chain())


def test_an_empty_chain_or_an_unknown_law_is_refused():
    with pytest.raises(ValueError, match="at least one law"):
        encode_law_chain([], Words())
    bad = _app(LevelScattering())
    bad.law = 8
    with pytest.raises(ValueError, match="LAW=8"):
        encode_law_data(bad, Words())


def test_mismatched_applicability_arrays_are_refused():
    law = _app(LevelScattering(), energies=(1.0, 2.0), probs=(1.0,))
    with pytest.raises(ValueError, match="LAW=3"):
        encode_law_chain([law], Words())


# --- two-word laws: 2, 3, 33 -------------------------------------------------

@pytest.mark.parametrize("lp,eg", [(0, 0.8467), (2, 7.6)])
def test_law2_lp_is_an_integer_and_eg_a_real(lp, eg):
    law = _round_trip(_app(DiscreteEnergyDistribution(lp=lp, eg=eg)))
    assert (law.lp, law.eg) == (lp, eg)
    w = Words()
    encode_law_data(DiscreteEnergyDistribution(lp=lp, eg=eg), w)
    assert w.is_int == [True, False]
    _reencode_hand_laid(_record(2, [lp, eg]))


def test_law3_and_law33_are_two_reals():
    law = _round_trip(_app(LevelScattering(aplusoaabsq=0.123456789012345, asquare=0.98)))
    assert (law.aplusoaabsq, law.asquare) == (0.123456789012345, 0.98)
    law = _round_trip(_app(ParticleLevelScattering(threshold=2.5, energy_factor=0.25)))
    assert (law.threshold, law.energy_factor) == (2.5, 0.25)
    for d in (LevelScattering(), ParticleLevelScattering()):
        w = Words()
        encode_law_data(d, w)
        assert w.is_int == [False, False]
    _reencode_hand_laid(_record(33, [2.5, 0.25]))


# --- LAW=1 (Table 32), no witness ---------------------------------------------

def _law1(nbt=(), interp=()):
    d = TabularEnergyDistribution(law=1, nbt=list(nbt), interp=list(interp))
    d.incident_energies = np.array([1.0, 5.0, 20.0])
    d.distribution_data = [{'e_out': np.array(e)} for e in
                           ([0.0, 0.1, 0.4, 1.0], [0.0, 0.5, 2.0, 5.0], [0.0, 1.0, 4.0, 20.0])]
    return _app(d)


@pytest.mark.parametrize("nbt,interp", [((), ()), ((1, 3), (1, 2))])
def test_law1_round_trips_its_equiprobable_tables(nbt, interp):
    src = _law1(nbt, interp)
    law = _round_trip(src)
    assert (law.nbt, law.interp) == (list(nbt), list(interp))
    _eq(law.incident_energies, src.incident_energies)
    assert len(law.distribution_data) == 3
    for a, b in zip(src.distribution_data, law.distribution_data):
        _eq(b['e_out'], a['e_out'])
        assert b['n_points'] == 4
    w = Words()
    encode_law_data(src, w)
    nr = len(nbt)
    # NR, NBT, INT, NE: ints; E: reals; NET: int; the 3 x 4 boundaries: reals
    assert w.is_int == [True] * (1 + 2 * nr + 1) + [False] * 3 + [True] + [False] * 12
    assert w.values[1 + 2 * nr + 1 + 3] == 4.0


def test_law1_refuses_tables_of_different_lengths():
    src = _law1()
    src.distribution_data[1] = {'e_out': np.array([0.0, 1.0])}
    with pytest.raises(ValueError, match="LAW=1"):
        encode_law_data(src, Words())


# --- LAW=4 and LAW=44 (Tables 35-36, 43-44) ----------------------------------

def _tables(extra=()):
    rng = np.random.default_rng(4)
    out = []
    for n_d, intt, n in ((0, 2, 3), (2, 1, 4), (0, 2, 2)):
        e = np.sort(rng.random(n))
        d = {'intt': intt, 'n_discrete': n_d, 'n_points': n, 'e_out': e,
             'pdf': rng.random(n), 'cdf': np.linspace(0.0, 1.0, n)}
        for k in extra:
            d[k] = rng.random(n)
        out.append(d)
    return out


def test_law4_round_trips_with_l_relative_to_jed_and_discrete_lines():
    d = _app(ContinuousTabularDistribution(nbt=[3], interp=[2], incident_energies=np.array([1.0, 2.0, 3.0]),
                                           distribution_locations=[999, 999, 999], distributions=_tables()))
    law = _round_trip(d)
    for a, b in zip(d.distributions, law.distributions):
        assert (b['intt'], b['n_discrete'], b['n_points']) == (a['intt'], a['n_discrete'], a['n_points'])
        for k in ('e_out', 'pdf', 'cdf'):
            _eq(b[k], a[k])
    # L(l) are recomputed: the first table right after the header, the others in order
    header = 1 + 2 + 1 + 3 + 3
    first = PAD + 1 + 9 + header          # after PAD, the 9-word record and the header
    assert law.distribution_locations == [first, first + 2 + 9, first + 2 + 9 + 2 + 12]


def test_law4_hand_laid_record_encodes_to_the_same_words():
    t1 = (21, [0.5, 1.0], [0.0, 0.0], [0.25, 1.0])
    t2 = (2, [0.0, 1.0], [1.0, 1.0], [0.0, 1.0])
    _reencode_hand_laid(_record(4, _law4([t1, t2], nbt=[2], interp=[1])))


def test_law44_round_trips_r_and_a():
    d = _app(KalbachMannDistribution(incident_energies=np.array([1.0, 2.0, 3.0]),
                                     distributions=_tables(extra=('r', 'a'))))
    law = _round_trip(d)
    for a, b in zip(d.distributions, law.distributions):
        for k in ('e_out', 'pdf', 'cdf', 'r', 'a'):
            _eq(b[k], a[k])


def test_law4_refuses_a_table_with_arrays_of_different_lengths():
    tables = _tables()
    tables[1]['pdf'] = tables[1]['pdf'][:-1]
    d = _app(ContinuousTabularDistribution(incident_energies=np.array([1.0, 2.0, 3.0]), distributions=tables))
    with pytest.raises(ValueError, match="LAW=4"):
        encode_law_data(d, Words())


# --- LAW=5 (Table 37), no witness ---------------------------------------------

@pytest.mark.parametrize("nbt,interp", [((), ()), ((2, 4), (1, 5))])
def test_law5_round_trips_temperatures_and_equiprobable_x(nbt, interp):
    d = _app(GeneralEvaporationSpectrum(
        temp_nbt=list(nbt), temp_interp=list(interp),
        incident_energies=np.array([1.0e-5, 1.0, 5.0, 20.0]),
        temperatures=np.array([0.3, 0.4, 0.6, 0.9]),
        equiprob_values=np.array([0.0, 0.2, 0.7, 1.5, 3.1, 8.0])))
    law = _round_trip(d)
    assert (law.temp_nbt, law.temp_interp) == (list(nbt), list(interp))
    _eq(law.incident_energies, d.incident_energies)
    _eq(law.temperatures, d.temperatures)
    _eq(law.equiprob_values, d.equiprob_values)
    assert law.n_equiprob_bins == 6
    w = Words()
    encode_law_data(d, w)
    nr = len(nbt)
    assert w.is_int == [True] * (2 + 2 * nr) + [False] * 8 + [True] + [False] * 6


# --- LAW=7, 9, 11 (Tables 38-40) ----------------------------------------------

@pytest.mark.parametrize("cls", [MaxwellFissionSpectrum, EvaporationSpectrum])
def test_law7_and_law9_round_trip_theta_and_u(cls):
    d = _app(cls(temp_nbt=[3], temp_interp=[2], incident_energies=np.array([1.0, 2.0, 3.0]),
                 temperatures=np.array([1.3, 1.31, 1.4]), restriction_energy=-0.5))
    law = _round_trip(d)
    assert law.law == d.law
    _eq(law.temperatures, d.temperatures)
    assert law.restriction_energy == -0.5


def test_law11_round_trips_both_parameter_tables():
    d = _app(EnergyDependentWattSpectrum(
        a_nbt=[2], a_interp=[2], a_incident_energies=np.array([1.0, 20.0]), a_values=np.array([0.98, 1.02]),
        b_nbt=[], b_interp=[], b_incident_energies=np.array([1.0, 5.0, 20.0]),
        b_values=np.array([2.5, 2.6, 2.7]), restriction_energy=-1.0e-7))
    law = _round_trip(d)
    assert (law.a_nbt, law.b_nbt) == ([2], [])
    _eq(law.a_values, d.a_values)
    _eq(law.b_incident_energies, d.b_incident_energies)
    _eq(law.b_values, d.b_values)
    assert law.restriction_energy == -1.0e-7


# --- LAW=22 (Table 41), no witness ---------------------------------------------

def _law22_obj():
    sets = [([1.0], [0.1], [0.5]), ([0.25, 0.75], [0.2, 0.3], [0.7, 0.8]), ([0.5, 0.25, 0.25], [0.0] * 3, [1.0] * 3)]
    d = TabularLinearFunctions(nbt=[1, 3], interp=[1, 2], incident_energies=np.array([1.0, 3.0, 9.0]),
                               table_locators=[1, 1, 1],
                               function_data=[{'nf': len(p), 'p': np.array(p), 't': np.array(t),
                                               'c': np.array(c)} for p, t, c in sets])
    return _app(d)


def test_law22_round_trips_its_function_sets_with_loce_relative_to_jed():
    d = _law22_obj()
    law = _round_trip(d)
    assert (law.nbt, law.interp) == ([1, 3], [1, 2])
    for a, b in zip(d.function_data, law.function_data):
        assert b['nf'] == a['nf']
        for k in ('p', 't', 'c'):
            _eq(b[k], a[k])
    header = 1 + 4 + 1 + 3 + 3
    first = PAD + 1 + 9 + header
    assert law.table_locators == [first, first + 4, first + 4 + 7]
    w = Words()
    encode_law_data(d, w)
    assert w.is_int[:6] == [True] * 6 and not any(w.is_int[6:9]) and all(w.is_int[9:13])
    assert w.is_int[12] and not any(w.is_int[13:16])     # NF int, P/T/C reals


def test_law22_hand_laid_record_encodes_to_the_same_words():
    sets = [([1.0], [0.1], [0.5]), ([0.4, 0.6], [0.2, 0.3], [0.7, 0.9])]
    _reencode_hand_laid(_record(22, _law22(sets, nbt=[2], interp=[2])))


def test_law22_refuses_p_t_c_of_different_lengths():
    d = _law22_obj()
    d.function_data[1]['c'] = np.array([0.7])
    with pytest.raises(ValueError, match="LAW=22"):
        encode_law_data(d, Words())


# --- LAW=24 (Table 42), no witness ---------------------------------------------

def test_law24_round_trips_its_multiplier_tables():
    d = _app(TabularEnergyMultipliers(nbt=[2], interp=[2], incident_energies=np.array([1.0, 3.0]),
                                      multiplier_tables=[np.array([0.1, 0.5, 0.9]), np.array([0.3, 0.7, 1.1])]))
    law = _round_trip(d)
    assert law.n_mult_values == 3
    for a, b in zip(d.multiplier_tables, law.multiplier_tables):
        _eq(b, a)
    w = Words()
    encode_law_data(d, w)
    assert w.is_int == [True] * 4 + [False] * 2 + [True] + [False] * 6
    _reencode_hand_laid(_record(24, [1, 2, 2, 2, 1.0, 3.0, 3, 0.1, 0.5, 0.9, 0.3, 0.7, 1.1]))


def test_law24_refuses_tables_of_different_lengths():
    d = _app(TabularEnergyMultipliers(incident_energies=np.array([1.0, 3.0]),
                                      multiplier_tables=[np.array([0.1, 0.5]), np.array([0.3])]))
    with pytest.raises(ValueError, match="LAW=24"):
        encode_law_data(d, Words())


# --- LAW=61 (Tables 45-47) ----------------------------------------------------

def _angular(lc, n):
    return {'lc': lc, 'jj': 2, 'n_points': n, 'cosines': np.linspace(-1.0, 1.0, n),
            'pdf': np.full(n, 0.5), 'cdf': np.linspace(0.0, 1.0, n)}


def _law61_obj():
    # Stored LCs are only keys: -900 is shared by both E' tables, -901 is
    # first used by the second one, 0 is isotropic.
    dists = [
        {'intt': 2, 'n_discrete': 0, 'n_points': 2, 'e_out': np.array([0.0, 1.0]),
         'pdf': np.array([1.0, 1.0]), 'cdf': np.array([0.0, 1.0]), 'lc': [-900, -900]},
        {'intt': 2, 'n_discrete': 0, 'n_points': 3, 'e_out': np.array([0.0, 0.5, 2.0]),
         'pdf': np.array([0.5, 0.5, 0.5]), 'cdf': np.array([0.0, 0.25, 1.0]), 'lc': [0, -901, -900]},
    ]
    d = TabulatedAngleEnergyDistribution(incident_energies=np.array([1.0, 2.0]), distributions=dists,
                                         angular_tables=[_angular(-900, 3), _angular(-901, 2)])
    return _app(d)


def test_law61_writes_each_angular_table_once_right_after_the_e_table_that_first_uses_it():
    d = _law61_obj()
    w, locc = _encode([d])
    v, f = w.arrays()
    law = _round_trip(d)
    l1, l2 = law.distribution_locations
    a900 = l1 + 2 + 4 * 2                      # right after the first E' table
    assert law.distributions[0]['lc'] == [-a900, -a900]
    a901 = l2 + 2 + 4 * 3                      # right after the second E' table
    assert l2 == a900 + 2 + 3 * 3
    assert law.distributions[1]['lc'] == [0, -a901, -a900]
    assert [t['lc'] for t in law.angular_tables] == [-a900, -a901]
    for src, back in zip(d.angular_tables, law.angular_tables):
        assert back['jj'] == src['jj']
        for k in ('cosines', 'pdf', 'cdf'):
            _eq(back[k], src[k])
    # LC and JJ/NP are integer words, the angular arrays reals
    assert f[l1 + 2 + 3 * 2 - 1:l1 + 2 + 4 * 2 - 1].all()
    assert f[a900 - 1:a900 + 1].all() and not f[a900 + 1:a900 + 1 + 9].any()


def test_law61_keeps_a_positive_lc_positive():
    d = _law61_obj()
    d.distributions[1]['lc'] = [0, 901, -900]
    d.angular_tables[1]['lc'] = 901
    law = _round_trip(d)
    assert law.distributions[1]['lc'][1] > 0


def test_law61_refuses_an_lc_with_no_table_and_a_table_no_lc_uses():
    d = _law61_obj()
    d.distributions[1]['lc'] = [0, -902, -900]
    with pytest.raises(ValueError, match="LAW=61: LC=-902"):
        encode_law_data(d, Words())
    d = _law61_obj()
    d.angular_tables.append(_angular(-903, 2))
    with pytest.raises(ValueError, match="referenced by no LC"):
        encode_law_data(d, Words())


# --- LAW=66 (Table 48) --------------------------------------------------------

def test_law66_round_trips_with_and_without_its_xi_table():
    xi = np.array([0.0, 0.5, 1.0])
    d = _app(NBodyPhaseSpaceDistribution(npsx=3, ap=2.0, intt=2, xi_grid=xi,
                                         pdf=np.array([0.0, 2.0, 0.0]), cdf=np.array([0.0, 0.5, 1.0])))
    law = _round_trip(d)
    assert (law.npsx, law.ap, law.intt, law.n_points) == (3, 2.0, 2, 3)
    _eq(law.xi_grid, xi)
    _eq(law.cdf, d.cdf)
    w = Words()
    encode_law_data(d, w)
    assert w.is_int == [True, False, True, True] + [False] * 9
    old = _app(NBodyPhaseSpaceDistribution(npsx=3, ap=2.0))      # written before NJOY 2016
    w = Words()
    encode_law_data(old, w)
    assert w.values == [3.0, 2.0] and w.is_int == [True, False]


# --- LAW=67 (Tables 49-51), no witness -----------------------------------------

def _law67_obj():
    rng = np.random.default_rng(67)

    def energy(n, intep=2):
        return {'intep': intep, 'npep': n, 'e_out': np.sort(rng.random(n)), 'pdf': rng.random(n),
                'cdf': np.linspace(0.0, 1.0, n)}
    aes = [
        {'intmu': 2, 'n_cosines': 2, 'cosines': np.array([-1.0, 1.0]),
         'energy_dist_locations': [1, 1], 'energy_distributions': [energy(2), energy(3, 1)]},
        {'intmu': 1, 'n_cosines': 3, 'cosines': np.array([-1.0, 0.0, 1.0]),
         'energy_dist_locations': [1, 1, 1], 'energy_distributions': [energy(2), energy(4), energy(2)]},
    ]
    d = LaboratoryAngleEnergyDistribution(nbt=[2], interp=[2], incident_energies=np.array([2.0, 14.0]),
                                          distribution_locations=[1, 1], angle_energy_distributions=aes)
    return _app(d)


def test_law67_round_trips_cosines_and_their_energy_tables():
    d = _law67_obj()
    law = _round_trip(d)
    assert (law.nbt, law.interp) == ([2], [2])
    for a, b in zip(d.angle_energy_distributions, law.angle_energy_distributions):
        assert (b['intmu'], b['n_cosines']) == (a['intmu'], a['n_cosines'])
        _eq(b['cosines'], a['cosines'])
        for ea, eb in zip(a['energy_distributions'], b['energy_distributions']):
            assert (eb['intep'], eb['npep']) == (ea['intep'], ea['npep'])
            for k in ('e_out', 'pdf', 'cdf'):
                _eq(eb[k], ea[k])


def test_law67_lays_each_cosine_table_after_its_lmu_in_order():
    law = _round_trip(_law67_obj())
    l1, l2 = law.distribution_locations
    first = law.angle_energy_distributions[0]
    lmu = first['energy_dist_locations']
    assert lmu[0] == l1 + 2 + 2 * 2                 # after INTMU, NMU, XMU(2), LMU(2)
    assert lmu[1] == lmu[0] + 2 + 3 * 2
    assert l2 == lmu[1] + 2 + 3 * 3
    second = law.angle_energy_distributions[1]['energy_dist_locations']
    assert second[0] == l2 + 2 + 2 * 3


def test_law67_refuses_a_missing_energy_table():
    d = _law67_obj()
    d.angle_energy_distributions[1]['energy_distributions'][1] = None
    with pytest.raises(ValueError, match="LAW=67: incident energy 2, cosine 2"):
        encode_law_data(d, Words())


# --- TY > 100 yields (Table 52) ------------------------------------------------

@pytest.mark.parametrize("nbt,interp", [((), ()), ((2, 3), (1, 2))])
def test_an_energy_dependent_yield_round_trips(nbt, interp):
    y = EnergyDependentYield(nbt=np.array(nbt, dtype=float), interp=np.array(interp, dtype=float),
                             energies=np.array([1.0e-5, 1.0, 20.0]), yields=np.array([2.0, 2.0, 1.7]))
    w = Words()
    w.real(0.0, 0.0)
    ky = w.next
    encode_energy_dependent_yield(y, w)
    values, flags = w.arrays()
    back = parse_energy_dependent_yield(_ace(values), ky)
    _eq(back.nbt, nbt)
    _eq(back.interp, interp)
    _eq(back.energies, y.energies)
    _eq(back.yields, y.yields)
    nr = len(nbt)
    assert flags[ky - 1:].tolist() == [True] * (2 + 2 * nr) + [False] * 6
