"""kika's own thermal inelastic tables (roadmap H9c): the kernel's pieces.

The whole tables are checked against the official ENDF/B-VIII.1 files in
``kika-workspace/kika_dev/processing/thermal`` (H in H2O: same outgoing points
at every incident energy, cosines to 1e-8, densities to 1e-6).
"""
import numpy as np
import pytest

native = pytest.importorskip("kika.processing._thermal_native")

from kika.processing.thermal import _aceTables  # noqa: E402


# -- the Python loop the native tpend + ACER kernel reproduces bit for bit ----------------
def _reference_tpend(x, y, xsi):
    """What THERMR's ``tpend`` leaves on the tape ACER reads: ``(E', y)``.

    The density is σ(E→E')/σ(E), to 7 figures (6 below 1e-9) -- so ACER's 1e-6
    merge threshold is on a unit area, not in barns -- and every number then
    goes through NJOY's 11-column ENDF writer (``endf.f90`` ``a11``): 7
    significant figures in exponent form, 9 in the fixed form it uses for
    positive values in (0.1, 1e7). Two outgoing energies that differ beyond
    the 7th figure read back equal, and ACER skips the second.
    """
    out = y.copy()
    rx = 1.0 / xsi
    out[:, 0] = [native.sigfig(v * rx, 7, 0) if v * rx >= 1e-9 else native.sigfig(v * rx, 6, 0)
                 for v in y[:, 0]]
    x = np.ascontiguousarray(x, dtype=float).copy()
    out = np.ascontiguousarray(out, dtype=float)
    native.endf11(x)
    native.endf11(out.reshape(-1))
    return x, out


def _reference_acer(x, y, nang):
    """ACER's ``acesix`` with ``iwt = 2`` on one incident energy: ``(E', pdf, cdf, cosines)``.

    ``y`` is what THERMR's ``tpend`` leaves on the tape (:func:`_reference_tpend`).
    """
    nep = len(x)
    cos = y[:, 1:1 + nang].copy()
    if nep >= 3:
        cos[0] = cos[1]
        cos[-1] = cos[-2]
    sig = y[:, 0]
    eps = 1e-5
    out_x, out_y, out_f, out_c = [], [], [], []
    xl, yl = x[0], sig[0]
    total = 0.0
    fract = 1.0
    i = 0
    while i < nep:
        xi, yi = x[i], sig[i]
        i += 1
        while True:
            add = (yi + yl) * (xi - xl) / 2
            if xi == xl:
                break
            if i == nep:
                xn = xi
                fract = total + add
            else:
                if total + add > eps / 10:
                    fract = total + add
                if not total + add >= fract - fract / 10000:
                    total += add
                    break
                if total + add < fract + fract / 10000:
                    xn = xi
                elif abs(yi - yl) > (yi + yl) / 100000:
                    f = (yi - yl) / (xi - xl)
                    disc = abs((yl / f) ** 2 + 2 * (fract - total) / f)
                    xn = xl - yl / f + np.sign(f) * np.sqrt(disc)
                else:
                    xn = min(xl + (fract - total) / yl, xi)
            yn = yl + (yi - yl) * (xn - xl) / (xi - xl)
            # The cosines at xn, found as ACER finds them: the first THERMR
            # point at or above xn (aceth.f90:432-443), then that point's own
            # cosines when xn is within eps of it (iwt=2: it always is).
            k = 1
            while k < nep - 1 and x[k] < xn:
                k += 1
            lo, hi = x[k - 1], x[k]
            if abs(xn - hi) < eps:
                c = cos[k]
            elif abs(xn - lo) < eps:
                c = cos[k - 1]
            else:
                c = cos[k - 1] + (cos[k] - cos[k - 1]) * (xn - lo) / (hi - lo)
            out_x.append(xn)
            out_y.append(yn)
            out_f.append(fract)
            out_c.append(np.clip(c, -1.0, 1.0))
            xl, yl = xn, yn
            fract = 1.0
            total = 0.0
            if xl < xi:
                continue
            break
        xl, yl = xi, yi
    area = np.cumsum(out_f)
    total = area[-1] if len(area) else 0.0
    return (np.asarray(out_x), np.asarray(out_y) / total, area / total, np.asarray(out_c))



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
    ex, pdf, cdf, cos = _aceTables(x, y, 1.0, 2)
    assert ex.tolist() == [1.0, 2.0]                     # 1e-9, 2e-9 merged into 1.0
    assert cdf[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(6))
def test_the_native_ace_tables_are_the_python_loop_bit_for_bit(seed):
    rng = np.random.default_rng(seed)
    n, nang = 400, 8
    x = np.sort(rng.uniform(0, 2, n)) * 10.0 ** rng.integers(-3, 1)
    x[0] = 0.0
    y = np.empty((n, nang + 1))
    y[:, 0] = rng.lognormal(0, 3, n) * (rng.random(n) > 0.05)    # some zeros, a wide range
    y[:, 1:] = np.sort(rng.uniform(-1.05, 1.05, (n, nang)), axis=1)
    xsi = float(rng.uniform(0.5, 50))
    expect = _reference_acer(*_reference_tpend(x, y, xsi), nang)
    got = _aceTables(x, y, xsi, nang)
    for a, b in zip(expect, got):
        assert a.shape == b.shape and np.array_equal(a, b)


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


def test_the_tables_do_not_depend_on_the_workers():
    def flat(result):
        xs, form, _ = result
        out = [xs.xs, xs.ys]
        for spectrum, angular in zip(form.energy.function1ds, form.energyAngular.function2ds):
            out += [spectrum.xs, spectrum.pdf, spectrum.cdf] + [t.xs for t in angular.function1ds]
        return out
    elastic = ([1e-6, 1.0], [20.0, 20.0])
    one, many = (flat(th.freeGasTables(15.858, 293.6, elastic, emax=0.5,
                                       options=th.ThermalInelasticOptions(iinc=1, workers=w)))
                 for w in (1, 4))
    assert len(one) == len(many)
    assert all(np.array_equal(a, b) for a, b in zip(one, many))


def test_process_thermal_takes_the_workers_as_process_suite_does(monkeypatch):
    from kika.hdf5 import thermal_processing

    seen = {}
    monkeypatch.setattr(thermal_processing, "_processFreeGas",
                        lambda path, **kwargs: seen.update(kwargs["options"].__dict__))
    thermal_processing.processThermal("n-001_H_001.endf", name="c_H_free", temperatures=[293.6],
                                      options=th.ThermalInelasticOptions(iinc=1), workers=3)
    assert seen["workers"] == 3 and seen["iinc"] == 1


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


def test_thermrs_grid_misses_a_one_over_v_cross_section_by_three_percent():
    # THERMR's 118 incident energies, read lin-lin at their geometric midpoints, on
    # sigma ~ 1/sqrt(E): what the refinement of the incident grid is for, and its cost.
    from kika.processing import thermal as T

    grid = np.asarray(T.EGRID, dtype=float)
    middle = np.sqrt(grid[:-1] * grid[1:])
    error = np.interp(middle, grid, 1 / np.sqrt(grid)) * np.sqrt(middle) - 1
    assert len(grid) == 118
    assert 0.031 < error.max() < 0.032 and middle[error.argmax()] < grid[1]
    assert (error > T.SIGMA_TOL).sum() == 45

    def compute(points):
        return [dict(enow=e, sigma=1 / np.sqrt(e), moments=(2 * e, .1)) for e in points]

    assert len(T._refine(compute(list(grid)), compute)) == 187
