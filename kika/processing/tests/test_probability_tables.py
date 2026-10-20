"""kika's own URR probability tables: the C kernel against the
exact NumPy reference, and PURR's conventions."""
import numpy as np
import pytest

from kika.processing.resonances.probability_tables import (NAVOID, NERMAX, REACH_DOPPLER,
                                                           _ladder, _ladderSpan, _nativeKernel,
                                                           _psiChi, bandBounds, competitionFlags)


def test_competition_flags_follow_purr():
    assert competitionFlags([]) == (-1, -1)
    assert competitionFlags([51]) == (51, -1)
    assert competitionFlags([51, 52]) == (4, -1)
    assert competitionFlags([51, 16, 17]) == (51, 0)
    assert competitionFlags([2, 4, 18, 102]) == (-1, -1)


def test_a_reaction_tabulated_with_leading_zeros_starts_at_its_threshold():
    # ENDF/B-VIII.1 Pu-240 MT20: zeros from 1e-5 eV to 4 MeV. PURR reads the
    # PENDF, which starts it one point before its first nonzero value, so it
    # does not compete in a URR that ends at 40 keV (NJOY: IOA = -1).
    from kika.nuclear_data.model import XYs1d
    from kika.processing.resonances.probability_tables import _threshold

    assert _threshold(XYs1d(xs=[1e-5, 4e6, 5e6, 2e7], ys=[0.0, 0.0, 0.1, 0.2])) == 4e6
    assert _threshold(XYs1d(xs=[4.3e4, 4.4e4], ys=[0.0, 0.017])) == 4.3e4
    assert _threshold(XYs1d(xs=[1e-5, 2e7], ys=[0.0, 0.0])) == np.inf


def test_band_bounds_are_increasing_and_open_at_the_top():
    rng = np.random.default_rng(3)
    bounds = bandBounds(rng.lognormal(size=10000), 20)
    assert len(bounds) == 20 and bounds[-1] == np.inf
    assert np.all(np.diff(bounds[:-1]) > 0)


def test_widths_carry_the_whole_porter_thomas_spread():
    # PURR's 20 slice means keep <x> but lose 5.6 % of <x^2> for one degree of
    # freedom (2.83 for 3); the exact chi-square has <x^2> = 1 + 2/nu, and
    # its tail beyond PURR's largest value, 5.58, is not empty.
    rng = np.random.default_rng(11)
    for nu in (1, 2, 3, 4, 1.7):
        seq = dict(D=1.0, means=np.array([2.0, 0.0, 0.0, 0.0]), degrees=np.array([nu, 0, 0, 0]))
        x = np.concatenate([_ladder(rng, seq, 0.0, 2.0e5)[1][0] for _ in range(4)]) / 2.0
        assert x.mean() == pytest.approx(1.0, rel=0.01)
        assert np.mean(x * x) == pytest.approx(1.0 + 2.0 / nu, rel=0.02)
    assert np.mean(x > 5.58) > 0


def test_the_samples_keep_a_resonance_reach_from_the_ladder_ends():
    # Six sequences of D = 1 eV: PURR's margin is 300/6 = 50 eV. Cold, the
    # span is PURR's; at a 10 eV Doppler width the reach is 200 eV, and the
    # margins grow to it while the sampled span stays PURR's.
    seqs = [dict(D=1.0, means=np.array([0.001, 0.04, 0.3, 0.0]))] * 6
    cold = _ladderSpan(seqs, 0.1)
    assert cold == (10.0, 10.0 + 0.9 * NERMAX, 10.0 + NAVOID / 6, 10.0 + 0.9 * NERMAX - NAVOID / 6)
    elow, ehigh, emin, emax = _ladderSpan(seqs, 10.0)
    assert emin - elow >= REACH_DOPPLER * 10.0 and ehigh - emax >= REACH_DOPPLER * 10.0
    assert emax - emin == pytest.approx(cold[3] - cold[2])


def test_the_competition_opens_at_the_lowest_competing_threshold():
    from types import SimpleNamespace as NS
    from kika.nuclear_data.model import XYs1d
    from kika.processing.resonances.probability_tables import competitionThreshold

    def reaction(mt, x, y):
        return NS(ENDF_MT=mt, crossSection=NS(forms={"eval": XYs1d(xs=x, ys=y)}))
    suite = NS(reactions=[reaction(2, [1e-5, 2e7], [10.0, 10.0]), reaction(102, [1e-5, 2e7], [1.0, 1.0]),
                          reaction(51, [1e-5, 22746.2, 3e4], [0.0, 0.0, 0.5]),
                          reaction(52, [48109.2, 6e4], [0.0, 0.2])],
               sums=[reaction(4, [22746.2, 3e4], [0.0, 0.5])], fissionComponents=[])
    assert competitionThreshold(suite) == 22746.2
    assert competitionThreshold(NS(reactions=suite.reactions[:2], sums=[], fissionComponents=[])) == np.inf


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


# ----------------------------------------------------------------------
# Statistical error (NR-10): the ladder is the unit, jackknife over ladders
# ----------------------------------------------------------------------

def _fakePrepared():
    """One s-wave spin group of a U-238-like nucleus at constant average parameters."""
    from types import SimpleNamespace

    class Const:
        def __init__(self, v):
            self.v = v

        def evaluate(self, e):
            return np.full(len(e), self.v)

    from kika.processing.resonances.context import NeutronContext

    group = SimpleNamespace(context=NeutronContext(236.0, 0.0), l=0, spin=0.5,
                            channel_radius=Const(9.4), phase_radius=Const(9.4), reduced_neutron=False,
                            averages=[Const(0.05), Const(0.023), Const(0.0), Const(0.0)],
                            degrees=[1.0, 0.0, 0.0, 0.0], spacing=Const(20.0))
    return SimpleNamespace(groups=[group])


def _job(**opts):
    from kika.processing.resonances.probability_tables import ProbabilityTableOptions

    options = ProbabilityTableOptions(samples=2000, **opts)
    return (_fakePrepared(), 236.0, 20000.0, 0, [293.6], {2: 0.0, 18: 0.0, 102: 0.0}, 0.0, options)


def test_jackknife_of_a_mean_is_the_clt_error():
    from kika.processing.resonances.probability_tables import _ladderStatistics

    rng = np.random.default_rng(5)
    L, M = 40, 3
    lc = np.full((L, M), 100.0)
    ls = np.zeros((L, 4, M))
    ls[:, 0] = rng.normal(10.0, 1.0, (L, M)) * 100
    ls[:, 1] = ls[:, 0]
    lt = ls.sum(-1)
    st = _ladderStatistics(lc, ls, lt, (1.0,))
    # Equal counts: each band's share p·factor is linear in the ladder sums up
    # to the common normalisation, and the probabilities have no error at all.
    assert np.all(st["probabilitySigma"] < 1e-12)
    assert st["ladders"] == L
    assert np.all(np.isfinite(st["contributionSigma"]))


def test_statistics_match_the_spread_of_independent_replicas():
    """Out of sample: the jackknife σ of each replica against the other replicas."""
    from kika.processing.resonances.probability_tables import _tablesAt, replace

    est, sig = [], []
    for seed in range(12):
        job = list(_job(ladders=16))
        job[-1] = replace(job[-1], seed=100 + seed)
        _, stats = _tablesAt(tuple(job))
        est.append(stats[0]["shielding"][1:2, 0])        # elastic f(1 b)
        sig.append(stats[0]["shieldingSigma"][1:2, 0])
    est, sig = np.array(est)[:, 0], np.array(sig)[:, 0]
    ratio = est.std(ddof=1) / np.sqrt((sig ** 2).mean())
    assert 0.5 < ratio < 1.6


def test_the_default_stops_at_5e3_after_purrs_64_ladders_capped_at_4096():
    from kika.processing.resonances.probability_tables import ProbabilityTableOptions

    o = ProbabilityTableOptions()
    assert (o.errorTarget, o.minimumLadders, o.ladders) == (5.0e-3, 64, 4096)


def test_without_a_target_every_ladder_is_drawn():
    from kika.processing.resonances.probability_tables import _tablesAt

    tables, stats = _tablesAt(_job(ladders=12, errorTarget=None))
    assert tables.shape == (1, 6, 20)
    assert stats[0]["ladders"] == 12 and stats[0]["converged"] is None
    assert stats[0]["shielding"].shape == (4, 5) and stats[0]["contributionSigma"].shape == (4, 20)


def test_an_error_target_stops_early_with_the_same_table_as_that_many_ladders():
    from kika.processing.resonances.probability_tables import _tablesAt

    tables, stats = _tablesAt(_job(ladders=40, errorTarget=0.5, checkEvery=4, minimumLadders=4))
    assert stats[0]["ladders"] == 4 and stats[0]["converged"] is True
    # The stream is the same: stopping after n ladders is drawing n ladders.
    fixed, _ = _tablesAt(_job(ladders=4))
    assert np.array_equal(tables, fixed)
    # An unreachable target runs to the cap and says so.
    _, capped = _tablesAt(_job(ladders=8, errorTarget=1e-9, checkEvery=4, minimumLadders=4))
    assert capped[0]["ladders"] == 8 and capped[0]["converged"] is False
    # By default the target is first checked at PURR's 64 ladders, never before.
    _, purr = _tablesAt(_job(ladders=100, errorTarget=0.5, checkEvery=4))
    assert purr[0]["ladders"] == 64 and purr[0]["converged"] is True


def test_error_target_must_be_positive():
    from kika.processing.resonances.probability_tables import ProbabilityTableOptions

    with pytest.raises(ValueError):
        ProbabilityTableOptions(errorTarget=0.0)
    with pytest.raises(ValueError):
        ProbabilityTableOptions(checkEvery=1)
    with pytest.raises(ValueError):
        ProbabilityTableOptions(minimumLadders=1)


def test_report_line_names_only_reactions_with_an_average():
    from kika.hdf5.processing import _urrErrorLine
    from kika.nuclear_data.model import ProbabilityTableStatistics
    from kika.processing.resonances.probability_tables import _tablesAt

    _, stats = _tablesAt(_job(ladders=8))
    st = stats[0]
    statistics = ProbabilityTableStatistics(
        ladders=np.array([8]), sigma0=np.array([1.0, 10.0, 100.0, 1000.0, 10000.0]),
        **{k: st[k][None] for k in ("probabilitySigma", "factorSigma", "contributionSigma",
                                    "shielding", "shieldingSigma")})
    worst = statistics.worst()
    assert worst["fission"]["contribution"] == 0.0 and worst["capture"]["contribution"] > 0
    line = _urrErrorLine(293.6, statistics)
    assert "capture f(1 b)" in line and "fission" not in line and "8 ladders" in line


def test_transport_shielding_uses_partial_factors_and_ignores_the_total_row():
    from kika.processing.resonances.probability_tables import _estimates

    counts = np.array([25., 75.])
    means = np.array([[100., 200.], [2., 6.], [0., 0.], [8., 2.]])
    sums = counts * means
    totals = sums.sum(-1)
    smooth = (10., 0., 3., 4.)
    result = _estimates(counts, sums, totals, (1.,), smooth)
    # Partial factors restore [4, 12] b elastic and [48/7, 12/7] b
    # non-fission absorption; add the constant 4 b inelastic competition.
    p = counts / 100.
    elastic = np.array([4., 12.])
    capture = np.array([48./7., 12./7.])
    weight = p / (elastic + capture + 4. + 1.)
    expected = (weight * elastic).sum() / weight.sum() / 10.
    assert result['shielding'][1, 0] == pytest.approx(expected)
    # A corrupt/otherwise-normalised unused total cannot alter transport f.
    sums[0] *= np.array([.01, 100.])
    totals[0] = sums[0].sum()
    changed = _estimates(counts, sums, totals, (1.,), smooth)
    np.testing.assert_array_equal(changed['shielding'], result['shielding'])


def test_transport_diagnostics_do_not_change_fixed_count_tables():
    from kika.processing.resonances.probability_tables import _tablesAt

    job = _job(ladders=8)
    old, old_stats = _tablesAt(job)
    new, new_stats = _tablesAt(job + ([(4., 0., 2., 1.)],))
    np.testing.assert_array_equal(old, new)
    assert not np.array_equal(old_stats[0]['shielding'], new_stats[0]['shielding'])
    assert np.isfinite(new_stats[0]['shieldingSigma']).all()


def test_processed_transport_absorption_excludes_redundant_sums():
    from types import SimpleNamespace
    from kika.nuclear_data import model as m
    from kika.hdf5.processing import _urrTransportXS
    from kika.hdf5.names import heatedLabel

    label = heatedLabel('294K')
    def reaction(mt, xs, redundant=0):
        return SimpleNamespace(ENDF_MT=mt,
            crossSection=SimpleNamespace(forms={label: m.XYs1d(xs=[1., 2.], ys=[xs, xs])}),
            provenance=SimpleNamespace(attributes={'redundant': redundant}))
    suite = SimpleNamespace(reactions=[reaction(2, 10.), reaction(102, 2.),
                                      reaction(600, 3.), reaction(51, 4.), reaction(52, 5.)],
                            fissionComponents=[reaction(19, 6.)],
                            sums=[reaction(103, 3., 1), reaction(18, 6., 1)])
    values = _urrTransportXS(suite, '294K', np.array([.5, 1.5]), 4)
    np.testing.assert_array_equal(values, [[0., 0., 0., 0.], [10., 6., 5., 9.]])
