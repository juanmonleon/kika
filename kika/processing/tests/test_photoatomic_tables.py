"""The photon and relaxation blocks of an EPRDATA table (kika.processing.photoatomic).

* The union grid repeats an edge, holds the value below then above it, and puts
  a table that opens on a non-zero value inside the union on a step.
* The incoherent mean outgoing energy equals an independent adaptive quadrature
  to 1e-10.
* The Compton shells follow BMM's list with the evaluation's electrons: Z < 36 by
  (n, l), a j-partner BMM omits joins its sibling (Pb 6p3/2), a BMM shell the
  evaluation lacks is dropped (Ir 6s1/2), and the Eu-63 2p1/2 value at
  p_z = 30 a.u. is the paper's 4.61e-3.
* The grid reads log-log within 1e-3 of the lin-lin data, keeping every node.
* The x-ray energy of a relaxation cascade, by hand on three subshells.
* With ENDF/B-VIII.1 and LANL's eprdata25 beside the repo (``../NuclearData``),
  Fe and Pb are built from the model and every block agrees with eprdata25 on
  eprdata25's nodes to print precision, except Pb's Compton shell arrays (out of
  step in eprdata25) and the heating numbers, which in kika follow the whole
  relaxation cascade: they lie between 0 and E, and below every edge with
  radiative transitions they are eprdata25's.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from kika.processing.photoatomic import comptonShells, unionGrid
from kika.processing.photoatomic.tables import ELECTRON_REST_ENERGY, _X_PER_EV, _incoherentMean, _onGrid

ND = Path(__file__).resolve().parents[5] / "NuclearData"


def test_union_repeats_edges_and_opens_tables_on_steps():
    total = (np.array([1.0, 2.0, 5.0, 5.0, 10.0]), np.array([9.0, 8.0, 7.0, 12.0, 3.0]))
    shell = (np.array([5.0, 10.0]), np.array([4.0, 2.0]))           # opens at 5 on 4 b, no zero point
    grid = unionGrid([total, shell])
    assert grid.tolist() == [1.0, 2.0, 5.0, 5.0, 10.0]
    on = _onGrid(*shell, grid)
    assert on.tolist() == [0.0, 0.0, 0.0, 4.0, 2.0]                   # below, then above, the edge
    late = (np.array([3.0, 10.0]), np.array([1.0, 1.0]))
    assert unionGrid([total, late]).tolist() == [1.0, 2.0, 3.0, 3.0, 5.0, 5.0, 10.0]
    np.testing.assert_allclose(_onGrid(*total, np.array([1.5, 7.5])), [8.5, 7.5])


def _reference_mean(e, sx, sy):
    from scipy import integrate

    mc2 = ELECTRON_REST_ENERGY
    k = e / mc2

    def f(ep):
        kp = ep / mc2
        mu = 1 + 1 / k - 1 / kp
        x = _X_PER_EV * e * np.sqrt(max(0.5 * (1 - mu), 0.0))
        return np.interp(min(x, sx[-1]), sx, sy) * (1 + mu * mu + k * kp * (1 - mu) ** 2)

    a = e / (1 + 2 * k)
    pts = [p for p in sorted(1 / (1 / e + 2 * (xx / (_X_PER_EV * e)) ** 2 / mc2) for xx in sx if xx > 0)
           if a < p < e][:200]
    num = integrate.quad(lambda t: f(t) * t, a, e, points=pts, limit=2000, epsabs=0, epsrel=1e-12)[0]
    den = integrate.quad(f, a, e, points=pts, limit=2000, epsabs=0, epsrel=1e-12)[0]
    return num / den


def test_incoherent_mean_agrees_with_an_adaptive_quadrature():
    sx = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0])
    sy = np.array([0.0, 2.0, 9.0, 20.0, 24.0, 26.0, 26.0, 26.0])
    energies = np.array([10.0, 1.0e3, 1.0e5, 2.0e6, 5.0e7])
    mine = _incoherentMean(energies, sx, sy)
    for e, m in zip(energies, mine):
        ref = _reference_mean(e, sx, sy)
        assert abs((e - m) / (e - ref) - 1.0) < 1e-9


def _shells(z, labels):
    return [(label, float(n), float(b)) for label, n, b in labels]


def test_compton_shells_light_element_groups_by_n_and_l():
    shells, notes = comptonShells(26, _shells(26, [
        ("1s1/2", 2, 7117), ("2s1/2", 2, 851), ("2p1/2", 2, 726), ("2p3/2", 4, 713), ("3s1/2", 2, 98),
        ("3p1/2", 2, 61), ("3p3/2", 4, 59), ("3d3/2", 2.4, 8.5), ("3d5/2", 3.6, 8.3), ("4s1/2", 2, 7.1)]))
    assert [s.label for s in shells] == ["1s", "2s", "2p", "3s", "3p", "3d", "4s"]
    assert shells[2].electrons == 6.0 and shells[2].bindingEnergy == pytest.approx((2 * 726 + 4 * 713) / 6)
    assert sum(s.electrons for s in shells) == pytest.approx(26.0) and not notes
    assert shells[0].cdf[-1] == 1.0 and np.trapezoid(shells[0].pdf, shells[0].pz) == pytest.approx(1.0)


def test_compton_shells_missing_j_partner_joins_its_sibling():
    from kika.processing.photoatomic.compton import bmmProfiles

    _, labels, _ = bmmProfiles(82)
    assert "6p1/2" in labels and "6p3/2" not in labels
    subs = [(lab, 2.0, 100.0) for lab in labels if lab != "6p1/2"] + [("6p1/2", 0.67, 3.0), ("6p3/2", 1.33, 2.0)]
    shells, notes = comptonShells(82, subs)
    p = next(s for s in shells if s.label == "6p1/2")
    assert p.electrons == pytest.approx(2.0) and p.subshells == ("6p1/2", "6p3/2")
    assert any("6p3/2" in n for n in notes)


def test_compton_shells_drop_a_shell_the_evaluation_lacks():
    from kika.processing.photoatomic.compton import bmmProfiles

    _, labels, _ = bmmProfiles(77)
    assert labels[-1] == "6s1/2"
    shells, notes = comptonShells(77, [(lab, 2.0, 10.0) for lab in labels[:-1]])
    assert [s.label for s in shells] == labels[:-1]
    assert any("6s1/2" in n for n in notes)


def test_europium_profile_has_the_paper_value():
    from kika.processing.photoatomic.compton import bmmProfiles

    pz, labels, j = bmmProfiles(63)
    assert labels[2] == "2p1/2" and j[2][list(pz).index(30.0)] == pytest.approx(4.61e-3)


def _library(z):
    photo = glob.glob(str(ND / "photoat" / "endfb81" / f"photoat-{z:03d}_*_000.endf"))
    relax = glob.glob(str(ND / "atomic_relax" / "endfb81" / f"atom-{z:03d}_*_000.endf"))
    ace = ND / "ace" / "eprdata25" / "eprdata25" / f"{1000 * z}.25p"
    return (photo[0], relax[0], ace) if photo and relax and ace.exists() else None


def _nodes(fine, coarse):
    """The index in ``fine`` of every node of ``coarse`` (a repeated node to its own copy)."""
    first = np.searchsorted(fine, coarse * (1 - 1e-11), side="left")
    copy = np.zeros(coarse.size, dtype=int)
    copy[1:] = coarse[1:] == coarse[:-1]
    return first + copy


def test_the_grid_reads_log_log_within_the_tolerance():
    from kika.processing.photoatomic.tables import LOGLOG_RTOL, refinedGrid
    x = np.array([1.0, 10.0, 100.0, 1.0e3])
    y = np.array([1.0e4, 50.0, 30.0, 1.0e-2])            # lin-lin data, far from log-log between nodes
    grid = refinedGrid([(x, y)], lambda g: [np.interp(g, x, y)])
    assert set(x) <= set(grid.tolist())
    true = np.interp(grid, x, y)
    for t in np.linspace(0.05, 0.95, 19):
        u = grid[:-1] + t * (grid[1:] - grid[:-1])
        log = np.exp(np.log(true[:-1]) + (np.log(true[1:]) - np.log(true[:-1])) * np.log(u / grid[:-1])
                     / np.log(grid[1:] / grid[:-1]))
        assert np.max(np.abs(log / np.interp(u, x, y) - 1)) <= LOGLOG_RTOL * 1.01


def test_cascade_xray_energy_by_hand():
    from kika.processing.photoatomic.tables import _Subshell, _Transition, cascadeXrayEnergies

    shells = [
        _Subshell(1, "1s1/2", 2.0, 1000.0, [_Transition(2, 0, 900.0, 0.6), _Transition(2, 2, 800.0, 0.4)]),
        _Subshell(2, "2s1/2", 2.0, 100.0, [_Transition(3, 0, 90.0, 0.1), _Transition(3, 3, 80.0, 0.9)]),
        _Subshell(3, "2p1/2", 2.0, 10.0, []),                       # filled without radiation
    ]
    x = cascadeXrayEnergies(shells)
    assert x[3] == 0.0
    assert x[2] == pytest.approx(0.1 * 90.0)
    assert x[1] == pytest.approx(0.6 * (900.0 + 9.0) + 0.4 * (9.0 + 9.0))


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(s > 0, np.abs(a - b) / np.where(s > 0, s, 1), 0.0))) if a.size else 0.0


@pytest.mark.parametrize("z", [26, 82])
def test_blocks_agree_with_eprdata25(z):
    import kika
    from kika.ace import read_ace
    from kika.processing.photoatomic import photoatomicTable

    found = _library(z)
    if found is None:
        pytest.skip("no ENDF/B-VIII.1 photoat/atomic_relax or eprdata25 in ../NuclearData")
    k, _ = photoatomicTable(kika.read(found[0]), kika.read(found[1]))
    e = read_ace(str(found[2]))
    # kika's grid is eprdata25's plus the nodes the log-log reading needs: compare on eprdata25's
    at = _nodes(k.energies, e.energies)
    assert _rel(k.energies[at], e.energies) < 1e-11
    for name in ("incoherent", "coherent", "photoelectric", "pair_production"):
        a, b = getattr(k, name)[at], getattr(e, name)
        assert np.array_equal(a == 0, b == 0)
        assert np.max(np.abs(np.log(a[a > 0]) - np.log(b[b > 0]))) < 1e-11
    assert _rel(k.coherent_integrated_form_factor, e.coherent_integrated_form_factor) < 1e-12
    assert _rel(k.fluorescence_yields, e.fluorescence_yields) < 1e-12
    assert _rel(k.subshell_binding_energies, e.subshell_binding_energies) < 1e-12
    for d, rows in e.relaxation.items():
        assert [(r.designator_1, r.designator_2) for r in k.relaxation[d]] == \
               [(r.designator_1, r.designator_2) for r in rows]
        assert _rel([r.energy for r in k.relaxation[d]], [r.energy for r in rows]) < 1e-12
    # heating: the cascade's x-rays leave; below the first subshell that radiates, nothing does
    assert np.all((k.heating >= 0.0) & (k.heating <= k.energies * (1 + 1e-12)))
    from kika.processing.photoatomic.tables import _relaxation, PhotoatomicNotes
    _, shells = _relaxation(kika.read(found[1]), PhotoatomicNotes())
    first = min(s.binding for s in shells if s.radiative())
    below = (e.energies > 20.0) & (e.energies < first)
    d = np.abs(k.heating[at] - e.heating) / np.abs(e.heating)
    assert below.sum() > 100 and np.mean(d[below] < 1e-6) > 0.99
    if z == 26:
        assert _rel(k.shell_electrons, e.shell_electrons) < 1e-12
        assert _rel(k.compton_profiles[0].pdf, e.compton_profiles[0].pdf) < 1e-12
    else:   # Pb: eprdata25's arrays are out of step with its 23 profiles; kika's sum to Z
        assert k.shell_electrons.sum() == pytest.approx(82.0)
        assert e.shell_electrons.sum() == pytest.approx(80.67)
