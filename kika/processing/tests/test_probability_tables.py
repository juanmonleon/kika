"""kika's own URR probability tables (roadmap H8): the C kernel against the
exact NumPy reference, and PURR's conventions."""
import numpy as np
import pytest

from kika.processing.resonances.probability_tables import (CHISQ, _nativeKernel, _psiChi,
                                                           bandBounds, competitionFlags)


def test_competition_flags_follow_purr():
    assert competitionFlags([]) == (-1, -1)
    assert competitionFlags([51]) == (51, -1)
    assert competitionFlags([51, 52]) == (4, -1)
    assert competitionFlags([51, 16, 17]) == (51, 0)
    assert competitionFlags([2, 4, 18, 102]) == (-1, -1)


def test_band_bounds_are_increasing_and_open_at_the_top():
    rng = np.random.default_rng(3)
    bounds = bandBounds(rng.lognormal(size=10000), 20)
    assert len(bounds) == 20 and bounds[-1] == np.inf
    assert np.all(np.diff(bounds[:-1]) > 0)


def test_chi_square_quantiles_have_the_right_means():
    # 20 equiprobable quantiles of a chi-square with nu degrees average to ~nu.
    for nu in range(1, 5):
        assert CHISQ[nu - 1].mean() == pytest.approx(nu, rel=0.02)


def test_native_ladder_matches_the_exact_reference():
    native = _nativeKernel()
    if native is None:
        pytest.skip("native kernels not built")
    rng = np.random.default_rng(7)
    es = np.sort(rng.uniform(0, 200, 4000))
    er = np.sort(rng.uniform(0, 200, 300))
    gt = rng.uniform(0.02, 0.2, 300)
    gn, gg, gf = gt * 0.2, gt * 0.5, gt * 0.3
    s0 = rng.uniform(100, 1000, 300)
    for delta in (0.0, 0.05, 1.0):
        out = [np.zeros_like(es) for _ in range(3)]
        native.urr_ladder(len(es), len(er), 0.9, 0.1, delta, 1e9, 1e9, es, er, gt, gn, gg, gf, s0, *out)
        x = 2 * (es[:, None] - er[None, :]) / gt[None, :]
        if delta:
            psi, chi = _psiChi(x, np.broadcast_to((gt / delta)[None, :], x.shape))
        else:
            psi, chi = 1 / (1 + x * x), x / (1 + x * x)
        ratio = gn / gt
        el = np.sum(s0 * ((0.9 - (1 - ratio)) * psi + 0.1 * chi), axis=1)
        ca = np.sum(s0 * gg / gt * psi, axis=1)
        assert np.max(np.abs(out[1] - ca) / np.max(np.abs(ca))) < 2e-4
        assert np.max(np.abs(out[0] - el) / np.max(np.abs(el))) < 2e-4


def test_sigfig_is_njoys():
    from kika.processing.resonances.probability_tables import sigfig

    assert sigfig(2250.0, 7, +1) == pytest.approx(2250.001, rel=1e-12)
    assert sigfig(25000.0, 7, -1) == pytest.approx(24999.99, rel=1e-12)
    assert sigfig(1.0, 7, 0) == pytest.approx(1.0, rel=1e-12)


def test_purr_grid_shades_the_ends_and_fills_wide_gaps():
    from types import SimpleNamespace

    from kika.processing.resonances.probability_tables import purrGrid

    group = SimpleNamespace(levelSpacingEnergies=np.array([2250.0, 3000.0, 10000.0, 25000.0]))
    region = SimpleNamespace(domainMin=2250.0, domainMax=25000.0,
                             tabulatedWidths=SimpleNamespace(spinGroups=[group], energyGrid=None))
    grid = purrGrid(region)
    assert grid[0] == pytest.approx(2250.001) and grid[-1] == pytest.approx(24999.99)
    # 3000 -> 10000 is wider than x1.26: the standard grid fills it.
    assert {3500.0, 4000.0, 5000.0, 6000.0, 7200.0, 8500.0} <= set(grid)
    # 2250.001 -> 3000 is too: 2500 goes in.
    assert 2500.0 in grid
    assert np.all(np.diff(grid) > 0)


def test_ladder_special_functions():
    """The URR kernel's own exp/sin/cos (to the ulp) and Humlicek's W4 (to its 1e-4)."""
    import math
    from scipy.special import wofz
    native = pytest.importorskip("kika.processing.resonances._rm_native")
    for a in np.r_[np.linspace(-60., 60., 2401), [1e-10, 1e-3, 0.785398, 1e3]]:
        e, s, c, _, _ = native.special(float(a), 1.0)
        assert abs(s - math.sin(a)) <= 2.3e-16 and abs(c - math.cos(a)) <= 2.3e-16
        ref = math.exp(-abs(a))
        assert (e == 0.0) if ref == 0.0 else abs(e / ref - 1) <= 4.5e-16
    for x in np.linspace(-20., 20., 161):
        for y in np.r_[np.geomspace(1e-6, 1., 13), [3., 10., 20.]]:
            *_, wr, wi = native.special(float(x), float(y))
            ref = wofz(x + 1j * y)
            assert abs(complex(wr, wi) - ref) <= 1e-4 * abs(ref)
