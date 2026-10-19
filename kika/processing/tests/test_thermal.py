"""kika's own thermal inelastic tables (roadmap H9c): the kernel's pieces.

The whole tables are checked against the official ENDF/B-VIII.1 files in
``kika-workspace/kika_dev/processing/thermal`` (H in H2O: same outgoing points
at every incident energy, cosines to 1e-8, densities to 1e-6).
"""
import numpy as np
import pytest

native = pytest.importorskip("kika.processing._thermal_native")

from kika.processing.thermal import _acer  # noqa: E402


def test_endf11_reads_back_as_njoy_writes():
    a = np.array([4.9011135e-10, 0.321876663, 8.6793154e-6, -0.94172753, 1234.56789012, 0.0])
    native.endf11(a)
    assert a.tolist() == [4.90111e-10, 0.32187666, 8.679315e-06, -0.9417275, 1234.56789, 0.0]


def test_sigfig_is_njoys():
    assert native.sigfig(2250.0, 7, 1) == pytest.approx(2250.001, rel=1e-12)
    assert native.sigfig(0.0253, 8, 0) == pytest.approx(0.0253, rel=1e-12)


def test_sig_at_a_table_node_is_the_tabulated_s():
    alpha = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
    beta = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    S = np.exp(-alpha[None, :] - beta[:, None])          # [ib, ia]
    sab = np.log(S).ravel()
    tev, az, sb = 0.0253, 1.0, 20.0
    e = 0.05
    b, a = 1.0, 0.4                                       # nodes beta[2], alpha[2]
    ep = e + b * tev
    u = (e + ep - a * az * tev) / (2 * np.sqrt(e * ep))
    got = native.sig(e, ep, u, tev, tev, az, sb, 0.0, 0.0, tev, tev, 0.0, 0, 0, 5, 5,
                     alpha, beta, sab)
    expect = sb / (2 * tev) * np.sqrt(ep / e) * np.exp(np.log(S[2, 2]) - b / 2)
    assert got == pytest.approx(expect, rel=1e-12)


def test_acer_merges_intervals_below_one_millionth_of_the_area():
    x = np.array([0.0, 1e-9, 2e-9, 1.0, 2.0])
    y = np.zeros((5, 3))
    y[:, 0] = [0.0, 1e-4, 1e-4, 1.0, 0.0]                # the first intervals carry < 1e-6
    y[:, 1:] = [[0.1, 0.2]] * 5
    ex, pdf, cdf, cos = _acer(x, y, 2)
    assert ex.tolist() == [1.0, 2.0]                     # 1e-9, 2e-9 merged into 1.0
    assert cdf[-1] == pytest.approx(1.0)


# -- free gas (THERMR iinc=1), roadmap NR-2 ------------------------------------------------
from math import erf, exp, pi, sqrt  # noqa: E402

from kika.processing import thermal as th  # noqa: E402


def _free_gas_total(A, kT, E):
    """sigma(E)/sigma_f of a free gas (target-at-rest sigma_f constant), closed form."""
    y = sqrt(A * E / kT)
    return (1 + 1 / (2 * y * y)) * erf(y) + exp(-y * y) / (y * sqrt(pi))


def test_sig_free_gas_is_the_closed_form_kernel():
    A, T = 15.858, 293.6
    inputs = th.freeGasInputs(A, T, 10.0)
    tev = inputs.tev
    e, ep, u = 0.0253, 0.031, 0.3
    got = native.sig(e, ep, u, tev, th.KT0, inputs.az, inputs.sb, 0.0, 0.0, tev, 0.0, 0.0, 0, 0,
                     len(inputs.alpha), len(inputs.beta), inputs.alpha, inputs.beta, inputs.sab, 1)
    a = (e + ep - 2 * u * sqrt(e * ep)) / (A * tev)
    b = (ep - e) / tev
    S = exp(-(a + b) ** 2 / (4 * a)) / sqrt(4 * pi * a)
    expect = ((A + 1) / A) ** 2 / (2 * tev) * sqrt(ep / e) * S
    assert got == pytest.approx(expect, rel=1e-13)


@pytest.mark.parametrize("A", [0.99917, 15.858, pytest.param(236.006, id="236.006")])
@pytest.mark.parametrize("E", [1e-5, 0.0253, 1.55])
def test_calcem_free_gas_integral_within_the_declared_budget(A, E):
    # PXB-T01: calcem's integral (sigma_f = 1) within 2e-3 of the closed form.
    options = th.ThermalInelasticOptions(iinc=1)
    inputs = th.freeGasInputs(A, 293.6, 10.0)
    _, _, xsi = th._calcem(inputs, native.sigfig(E, 8, 0), options)
    assert xsi == pytest.approx(_free_gas_total(A, inputs.tev, E), rel=2e-3)


def test_free_gas_tables_carry_the_elastic_and_normalised_spectra():
    E = np.array([1e-6, 1.0, 20.0])
    sigma = np.array([30.0, 20.0, 20.0])
    xs, form, emax = th.freeGasTables(15.858, 293.6, (E, sigma), emax=1.0,
                                      options=th.ThermalInelasticOptions(iinc=1, workers=1))
    assert emax < 1.0
    assert np.allclose(xs.ys, np.interp(xs.xs, E, sigma), rtol=1e-9)
    for spectrum in form.energy.function1ds:
        assert spectrum.cdf[-1] == pytest.approx(1.0, abs=1e-12)


def test_free_gas_options_are_checked():
    elastic = ([1e-5, 10.0], [20.0, 20.0])
    with pytest.raises(ValueError, match="iinc"):
        th.freeGasTables(1.0, 293.6, elastic, options=th.ThermalInelasticOptions())
    with pytest.raises(ValueError, match="iinc"):
        th.inelasticTables(None, 293.6, options=th.ThermalInelasticOptions(iinc=1))
    with pytest.raises(ValueError, match="3000"):
        th.freeGasInputs(1.0, 3500.0, 10.0)
    with pytest.raises(ValueError, match="does not reach"):
        th.freeGasTables(1.0, 293.6, ([1e-3, 10.0], [20.0, 20.0]), emax=0.1)


def test_corrected_free_gas_pdf_and_cdf_describe_the_same_tail_mass():
    # ACER can join a nonzero tail PDF to a distant zero after dropping tiny
    # CDF increments. Normalising CDF[-1] alone does not catch that defect.
    _, form, _ = th.freeGasTables(236.006,293.6,([1e-6,1.],[1.,1.]),emax=3e-5,
                                  options=th.ThermalInelasticOptions(iinc=1,workers=2))
    for spectrum in form.energy.function1ds:
        x=np.asarray(spectrum.xs);p=np.asarray(spectrum.pdf);c=np.asarray(spectrum.cdf)
        areas=np.diff(x)*(p[:-1]+p[1:])/2
        np.testing.assert_allclose(np.diff(c),areas,rtol=1e-9,atol=2e-15)
        assert areas.sum()==pytest.approx(1.,abs=1e-12)
        assert np.all(p>=0)


def test_the_incident_grid_keeps_thermrs_nodes_and_adds_what_sigma_needs():
    # sigma ~ 1/sqrt(E) read lin-lin between THERMR's nodes misses by up to ~1 % near 1e-5 eV
    # (ND-8); the refined grid keeps every node and reads within SIGMA_TOL at its midpoints.
    from kika.processing import thermal as T

    def entry(e):
        return dict(enow=e, sigma=1 / np.sqrt(e), moments=(2 * e, .1))

    def compute(points):
        return [entry(e) for e in points]

    grid = [native.sigfig(float(e), 8, 0) for e in T.EGRID[:20]]
    refined = [x["enow"] for x in T._refine(compute(grid), compute)]
    assert set(grid) <= set(refined) and len(refined) > len(grid)
    for a, b in zip(refined[:-1], refined[1:]):
        middle = entry(native.sigfig(float(np.sqrt(a * b)), 8, 0))
        if a < middle["enow"] < b:
            assert not T._misses(entry(a), entry(b), middle)
    linear = lambda points: [dict(enow=e, sigma=3 * e, moments=(2 * e, .1)) for e in points]  # noqa: E731
    assert [x["enow"] for x in T._refine(linear(grid), linear)] == grid      # lin-lin is exact
