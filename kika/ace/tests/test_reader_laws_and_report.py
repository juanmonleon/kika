"""The ACE reader fixes, one test each, tape-free.

The 2026-10-14 audit
read two whole libraries against an independent decoder. What it found wrong
or untested is pinned here on law records laid out word by word from the
manual (LA-UR-19-29016, Tables 30-49): LAW=33 was not decoded, a lines-only
table interpolated to empty arrays, DNED kept only the first law of a group,
LAW=2 hid LP=2, LAW=66/22/24/67 ignored lengths or NBT/INT, and a block that
could not be read was a warning in the log.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from kika.ace.classes.energy_distribution import tabular_math
from kika.ace.classes.header import Header
from kika.ace.classes.read_report import AceReadError, ReadReport
from kika.ace.parsers.parse_ace import read_ace
from kika.ace.parsers.parse_energy_distributions import _read_laws, read_dned_block
from kika.ace.tests._synthetic import build_table, write_table


def _ace(words, strict=True):
    """An ace holding ``words`` as XSS(1..), with a read report."""
    return types.SimpleNamespace(
        header=Header(), xss_data=np.asarray([0.0] + list(words), dtype=float),
        read_report=ReadReport(strict=strict))


def _record(law, ldat, energies=(1.0e-5, 20.0), probs=(1.0, 1.0), lnw=0, start=1):
    """One law record of Table 30 at XSS(start), JED = 1, LDAT right after it.

    Locators inside ``ldat`` given as ``("L", k)`` resolve to the XSS index of
    ``ldat[k]`` (they are relative to JED = 1, so the index itself), and
    ``("-L", k)`` to minus that (a tabulated LC).
    """
    head = [lnw, law, None, 0, len(energies), *energies, *probs]
    idat = start + len(head)
    head[2] = idat

    def resolve(w):
        if not isinstance(w, tuple):
            return w
        return (idat + w[1]) * (-1 if w[0] == "-L" else 1)
    return head + [resolve(w) for w in ldat]


def _laws(words, strict=True):
    """Read the law chain at XSS(1) as the DLW reader does for MT=16."""
    ace = _ace(words, strict)
    return ace, _read_laws(ace, "DLW", "MT=16", 1, 1)


# --- LAW=33 (B1) -------------------------------------------------------------

def test_law33_is_read_as_the_particle_level_scattering_njoy_writes():
    # NJOY acelcp: LDAT = Ec, C and E_out = C (E - Ec); H-1 B-VIII.0 MT102
    # recoil and 9 150 more distributions in B-VIII.0 + JEFF-4.0 fell to an
    # empty EnergyDistribution.
    _, (law,) = _laws(_record(33, [2.5, 0.25]))
    assert type(law).__name__ == "ParticleLevelScattering"
    assert (law.law, law.threshold, law.energy_factor) == (33, 2.5, 0.25)
    assert law.get_energy(10.5) == pytest.approx(2.0)
    assert law.get_energy(1.0) == 0.0


def test_an_unknown_law_is_refused_by_number():
    with pytest.raises(ValueError, match="LAW=8 .* is not an ACE energy law"):
        _laws(_record(8, [0.0, 0.0]))


# --- LAW=2 with LP=2 (S2, D-A2) ---------------------------------------------

def test_a_primary_photon_keeps_eg_and_lp_and_the_energy_needs_e_and_awr():
    _, (law,) = _laws(_record(2, [2, 7.6]))
    assert (law.lp, law.eg) == (2, 7.6)
    assert not hasattr(law, "discrete_energies")
    assert law.get_photon_energy(1.0, 55.0) == pytest.approx(7.6 + 55.0 / 56.0)
    _, (line,) = _laws(_record(2, [0, 0.8467]))
    assert line.get_photon_energy(1.0, 55.0) == 0.8467


# --- LAW=4 lines only (B2) ---------------------------------------------------

def _law4(tables, energies=(1.0, 2.0), nbt=(), interp=()):
    """LAW=4 LDAT (Table 34) with one (INTT', E', pdf, cdf) table per energy."""
    nr = len(nbt)
    ldat = [nr, *nbt, *interp, len(energies), *energies] + [None] * len(energies)
    for k, (intt, e, p, c) in enumerate(tables):
        ldat[1 + 2 * nr + 1 + len(energies) + k] = ("L", len(ldat))
        ldat += [intt, len(e), *e, *p, *c]
    return ldat


def test_a_lines_only_table_interpolates_to_its_lines_not_to_empty_arrays():
    # Photon spectra with NP = ND (INTT' = 10 ND + INTT): 315 tables in the
    # audit sample, and cdf[-1] of the empty result crashed the comparator.
    t1 = (21, [0.5, 1.0], [0.0, 0.0], [0.25, 1.0])
    t2 = (21, [0.5, 1.0], [0.0, 0.0], [0.75, 1.0])
    _, (law,) = _laws(_record(4, _law4([t1, t2])))
    d = law.get_interpolated_distribution(1.5)
    assert d["discrete_only"]
    assert len(d["e_out"]) == 0
    np.testing.assert_allclose(sorted(d["discrete_probabilities"]), [0.125, 0.125, 0.375, 0.375])
    assert d["discrete_probabilities"].sum() == pytest.approx(1.0)
    assert law.get_interpolated_distribution(1.0)["discrete_only"]


# --- repeated incident energy (S7, D-A3) and INT=12 (S5, D-A4) ----------------

def test_a_repeated_incident_energy_is_a_jump_and_is_noted():
    grid = [0.001, 0.001, 1.0]
    assert tabular_math.bracket(grid, 0.001) == (0, 0.0)      # exactly at it: the first
    i, f = tabular_math.bracket(grid, 0.5)
    assert i == 1 and f == pytest.approx((0.5 - 0.001) / (1.0 - 0.001))  # above: from the second
    tables = [(2, [0.0, 1.0], [1.0, 1.0], [0.0, 1.0])] * 3
    ace, _ = _laws(_record(4, _law4(tables, energies=grid)))
    (note,) = ace.read_report.notes
    assert "repeats the incident energies [0.001]" in note.message
    assert not ace.read_report.lost


def test_corresponding_points_interpolation_is_noted():
    tables = [(2, [0.0, 1.0], [1.0, 1.0], [0.0, 1.0])] * 2
    ace, _ = _laws(_record(4, _law4(tables, nbt=[2], interp=[12])))
    assert any("INT=[12]" in n.message and "unit-base" in n.message for n in ace.read_report.notes)


# --- LAW=66 length (S3) ------------------------------------------------------

def test_law66_reads_its_tabulated_shape_and_refuses_a_truncated_one():
    xi = [0.0, 0.5, 1.0]
    full = [3, 2.0, 2, 3, *xi, 0.0, 2.0, 0.0, 0.0, 0.5, 1.0]   # NPSX, AP, INTT, NP, ξ, pdf, cdf
    _, (law,) = _laws(_record(66, full))
    assert (law.npsx, law.ap) == (3, 2.0)
    # NJOY before 2016 wrote NPSX and Ap alone: no table is taken, and it is noted
    ace, (old,) = _laws(_record(66, full[:-2]))
    assert old.npsx == 3 and len(old.xi_grid) == 0
    assert "no tabulated" in ace.read_report.notes[0].message


# --- LAW=22 and LAW=24 between incident energies (S4) -------------------------

def _law22(sets, energies=(1.0, 3.0), nbt=(), interp=()):
    nr = len(nbt)
    ldat = [nr, *nbt, *interp, len(energies), *energies] + [None] * len(energies)
    for k, (p, t, c) in enumerate(sets):
        ldat[1 + 2 * nr + 1 + len(energies) + k] = ("L", len(ldat))
        ldat += [len(p), *p, *t, *c]
    return ldat


@pytest.mark.parametrize("interp,expected_nf", [(2, 2), (1, 1)])
def test_law22_mixes_the_two_function_sets_unless_the_interval_is_histogram(interp, expected_nf):
    sets = [([1.0], [0.1], [0.5]), ([1.0], [0.2], [0.7])]
    _, (law,) = _laws(_record(22, _law22(sets, nbt=[2], interp=[interp])))
    d = law.get_interpolated_function_data(2.5)
    assert d["nf"] == expected_nf
    if expected_nf == 2:
        np.testing.assert_allclose(d["p"], [0.25, 0.75])
        np.testing.assert_allclose(d["c"], [0.5, 0.7])
    else:
        np.testing.assert_allclose(d["c"], [0.5])


def test_law24_interpolates_point_by_point_and_keeps_the_lower_table_in_a_histogram():
    def ldat(interp):
        return [1, 2, interp, 2, 1.0, 3.0, 3, 0.1, 0.5, 0.9, 0.3, 0.7, 1.1]
    _, (law,) = _laws(_record(24, ldat(2)))
    np.testing.assert_allclose(law.get_interpolated_multiplier_table(2.0), [0.2, 0.6, 1.0])
    _, (law,) = _laws(_record(24, ldat(1)))
    np.testing.assert_allclose(law.get_interpolated_multiplier_table(2.0), [0.1, 0.5, 0.9])


def test_table_weights_follow_nbt_int():
    # NBT = (2, 3), INT = (1, 2): the first interval is histogram, the second lin-lin
    assert tabular_math.table_weights([1.0, 2.0, 4.0], [2, 3], [1, 2], 1.5) == [(0, 1.0)]
    assert tabular_math.table_weights([1.0, 2.0, 4.0], [2, 3], [1, 2], 3.0) == [(1, 0.5), (2, 0.5)]


# --- LAW=61 from the XSS -----------------------------------------------------

def test_law61_reads_each_angular_table_once_in_order_of_first_use():
    # NE = 2; both E' tables point at the same angular table (LC = -a) and the
    # second also has an isotropic point (LC = 0).
    ldat = [0, 2, 1.0, 2.0, None, None]
    for k in range(2):
        ldat[4 + k] = ("L", len(ldat))
        ldat += [2, 2, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, "LC", 0 if k else "LC"]
    ang = len(ldat)
    ldat += [2, 3, -1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 1.0]
    ldat = [("-L", ang) if w == "LC" else w for w in ldat]
    rec = _record(61, ldat)
    idat = rec[2]
    _, (law,) = _laws(rec)
    assert [d["lc"] for d in law.distributions] == [[-(idat + ang)] * 2, [-(idat + ang), 0]]
    (table,) = law.angular_tables
    assert table["lc"] == -(idat + ang)
    np.testing.assert_array_equal(table["cosines"], [-1.0, 0.0, 1.0])


# --- chains and DNED (S1) ----------------------------------------------------

def _chain():
    """Two laws: LAW=33 with P = 0.25, then LAW=3 (LNW = 0) with P = 0.75."""
    second = _record(3, [1.0, 0.9], probs=(0.75, 0.75))
    first = _record(33, [2.0, 0.5], probs=(0.25, 0.25))
    start2 = 1 + len(first)
    first[0] = start2
    second = _record(3, [1.0, 0.9], probs=(0.75, 0.75), start=start2)
    return first + second


def test_a_chain_keeps_every_law_and_its_probability():
    _, laws = _laws(_chain())
    assert [law.law for law in laws] == [33, 3]
    np.testing.assert_array_equal(laws[0].applicability_probabilities, [0.25, 0.25])
    np.testing.assert_array_equal(laws[1].applicability_probabilities, [0.75, 0.75])


def _dned_ace(strict):
    chain = _chain()
    bad = [0, 4, 99999, 0, 2, 1.0e-5, 20.0, 1.0, 1.0]          # IDAT past the array
    ace = _ace(chain + bad, strict)
    ace.energy_distribution_locators = types.SimpleNamespace(
        has_delayed_neutron_data=True, delayed_neutron=[len(chain) + 1, 1])
    out = types.SimpleNamespace(delayed_neutron=[])
    return ace, out


def test_dned_keeps_whole_chains_and_a_lost_group_keeps_its_place():
    ace, out = _dned_ace(strict=False)
    read_dned_block(ace, out, 1)
    assert out.delayed_neutron[0] == []                       # group 1 lost, still group 1
    assert [law.law for law in out.delayed_neutron[1]] == [33, 3]
    (issue,) = ace.read_report.lost
    assert (issue.block, issue.where) == ("DNED", "group 1")


def test_dned_raises_naming_the_group_when_strict():
    ace, out = _dned_ace(strict=True)
    with pytest.raises(AceReadError, match="DNED group 1"):
        read_dned_block(ace, out, 1)


# --- strict reading (S6, D-A1) ------------------------------------------------

def _broken_table(tmp_path):
    """The synthetic table with the LDLW locator of MT16 pointing past the XSS."""
    nxs, jxs, xss = build_table()
    xss[jxs[10]] = 10 ** 7
    path = tmp_path / "broken.ace"
    write_table(path)
    lines = path.read_text().splitlines()
    words = [w for line in lines[12:] for w in (line[i:i + 20] for i in range(0, len(line), 20))]
    words[jxs[10] - 1] = f"{1.0e7:20.11E}"
    body = ["".join(words[k:k + 4]) for k in range(0, len(words), 4)]
    path.write_text("\n".join(lines[:12] + body) + "\n")
    return path


def test_a_block_that_cannot_be_read_raises_by_default(tmp_path):
    with pytest.raises(AceReadError, match="DLW MT=16"):
        read_ace(str(_broken_table(tmp_path)))


def test_strict_false_keeps_the_rest_and_reports_the_gap(tmp_path):
    ace = read_ace(str(_broken_table(tmp_path)), strict=False)
    assert [str(i) for i in ace.read_report.lost] == [
        f"DLW MT=16: JED + LOCC - 1 = {ace.header.jxs_array[11] + 10 ** 7 - 1} is past the XSS array "
        f"({len(ace.xss_data)})"]
    assert 51 in ace.energy_distributions.incident_neutron
    assert 16 not in ace.energy_distributions.incident_neutron


def test_a_clean_table_has_an_empty_report(tmp_path):
    ace = read_ace(str(write_table(tmp_path / "t.ace")))
    assert not ace.read_report
    assert ace.read_report.strict
