"""The native damage kernel is the NumPy reference: bit for bit where it can be.

The kernel replays the NumPy expressions operation for operation (order, np.interp's
branches, np.sum's pairwise summation, no fused operations) and takes ``pow`` and
``cbrt`` from the C runtime, because that is where NumPy takes them on Windows (the
UCRT); ``tanh`` is the C runtime's on both sides (the reference calls math.tanh).
There the two paths are bit-identical and these tests assert it.

They cannot be everywhere. NumPy's x86-64 Linux wheels evaluate ``cbrt`` and
``power`` with their own SIMD code (SVML) whenever the CPU has AVX-512, scalars
included, and those are not glibc's: 1 to 3 ulp apart on about half of all
arguments for ``cbrt`` (measured 2026-10-25 on NumPy 2.2.3 under WSL, which is also
what made the GitHub runners red from 2026-10-10). That code is not reachable from C,
so on such a machine the contract is "equal to a few ulp": :func:`_agree` is exact
when :data:`EXACT` (np.cbrt and np.power are this C runtime's, probed below), else
a relative bound of :data:`RTOL`. Comparisons with no transcendental in them
(the binning and summation kernels, native against native) stay exact everywhere.

Nothing downstream relies on native == NumPy to the bit: ``_Batched`` and the
refinement compare the native batch with the native chunk (``test_heating_batches``),
and a build without the extension uses the NumPy path for both.
"""
import ctypes
import ctypes.util
import sys

import numpy as np
import pytest

from kika.processing import heating as H

native = pytest.importorskip("kika.processing._heating_native")


def _numpy_transcendentals_are_the_c_runtime() -> bool:
    """Whether np.cbrt and np.power give the C runtime's bits on this machine."""
    try:
        if sys.platform == "win32":
            crt = ctypes.CDLL("ucrtbase")
        else:
            crt = ctypes.CDLL(ctypes.util.find_library("m") or None)
        crt.cbrt.restype = ctypes.c_double
        crt.cbrt.argtypes = [ctypes.c_double]
        crt.pow.restype = ctypes.c_double
        crt.pow.argtypes = [ctypes.c_double, ctypes.c_double]
    except (OSError, AttributeError):
        return False
    rng = np.random.default_rng(20261025)
    x = 10.0 ** rng.uniform(-6, 8, 2048)
    b = rng.uniform(-2, 2, 2048)
    return (np.array_equal(np.cbrt(x), [crt.cbrt(v) for v in x])
            and np.array_equal(np.power(x, b), [crt.pow(v, e) for v, e in zip(x, b)]))


EXACT = _numpy_transcendentals_are_the_c_runtime()
#: A few ulp at the result: SVML's cbrt is within 3 ulp of glibc's, and the quadratures
#: sum positive terms, so the per-term relative error is also the sum's.
RTOL = 16 * np.finfo(float).eps


def _agree(actual, expected):
    """Native against NumPy: to the bit if :data:`EXACT`, else within :data:`RTOL`."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if EXACT:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0, equal_nan=True)


def _case(rng, hist, lab, Zr):
    nx = int(rng.integers(2, 40))
    x = np.cumsum(rng.uniform(0.1, 3.0, nx)) * 10.0 ** rng.integers(2, 6)
    x -= x[0] * rng.choice([0.0, 0.5])
    p = rng.uniform(0.0, 2.0, nx)
    p /= float(np.sum(np.diff(x) * (p[:-1] if hist else (p[:-1] + p[1:]) / 2)))
    oldwidth = x[-1] - x[0]
    width = oldwidth * rng.uniform(0.5, 1.5)
    low = x[0] * rng.uniform(0.5, 1.5)
    mapped = low + (x - x[0]) * width / oldwidth
    E = 10.0 ** rng.uniform(3, 7.3)
    A = float(rng.choice([1.0, 15.86, 55.4, 235.0]))
    Z = 26 if A > 2 else 1
    ed = H.displacementEnergy(Z)
    denominator = 1.0 if lab or Zr != Z else A
    incident = 0.0 if lab else E
    extra = [(np.sqrt(incident) - np.sqrt(denominator * ed)) ** 2, (np.sqrt(incident) + np.sqrt(denominator * ed)) ** 2]
    cuts = np.unique(np.r_[mapped, extra, rng.uniform(mapped[0], mapped[-1], int(rng.integers(0, 300)))])
    cuts = cuts[(cuts >= mapped[0]) & (cuts <= mapped[-1])]
    order = int(rng.choice([8, 16, 32, 64]))
    g, w = np.polynomial.legendre.leggauss(order)
    mu, wm = np.polynomial.legendre.leggauss(2 * order)
    partition = (Zr, 4.0 if Zr == 2 else A, Z, A)
    return (cuts[:257], g, w, mu, wm, x, p, low, width, oldwidth, incident, denominator, ed, partition, hist, lab)


@pytest.mark.parametrize("hist", [False, True])
@pytest.mark.parametrize("lab", [False, True])
@pytest.mark.parametrize("Zr", [0, 2, 26])
def test_native_emission_chunk_equals_numpy(hist, lab, Zr):
    rng = np.random.default_rng(1000 * hist + 100 * lab + Zr)
    for _ in range(60):
        args = _case(rng, hist, lab, Zr)
        if len(args[0]) < 2:
            continue
        _agree(H._emissionChunk(*args), H._emissionChunkNumpy(*args))


def test_native_kernel_rejects_inconsistent_buffers():
    g, w = np.polynomial.legendre.leggauss(8)
    with pytest.raises(ValueError, match="inconsistent"):
        native.emission_chunk(np.array([1.0]), g, w, g, w, np.array([0.0, 1.0]), np.array([1.0, 1.0]),
                              0., 1., 1., 1e6, 56., 1400., 1., 1., 25., False, False, False)
    with pytest.raises(ValueError, match="float64"):
        native.emission_chunk(np.array([1, 2]), g, w, g, w, np.array([0.0, 1.0]), np.array([1.0, 1.0]),
                              0., 1., 1., 1e6, 56., 1400., 1., 1., 25., False, False, False)


@pytest.mark.parametrize("hist", [False, True])
def test_native_two_body_chunk_equals_numpy(hist):
    rng = np.random.default_rng(77 + hist)
    for _ in range(300):
        nx = int(rng.integers(2, 60))
        x = np.unique(np.r_[-1.0, 1.0, rng.uniform(-1, 1, nx)])
        p = rng.uniform(0.0, 3.0, len(x))
        A = float(rng.choice([1.0, 11.9, 55.4, 235.0]))
        Z = 26 if A > 2 else 1
        ed = H.displacementEnergy(Z)
        E = 10.0 ** rng.uniform(1, 7.5)
        c = E * A / (A + 1) ** 2
        r = rng.uniform(0.2, 1.0)
        offset, slope = c * (1 + r * r), 2 * c * r
        last = min(1.0, (offset - ed) / slope)
        if last <= -1:
            continue
        mcuts = (offset - ed * 10.0 ** np.arange(13)) / slope
        cuts = np.unique(np.r_[-1.0, x[(x > -1) & (x < last)], mcuts[(mcuts > -1) & (mcuts < last)], last])
        u, w = np.polynomial.legendre.leggauss(int(rng.choice([8, 16, 32, 64])))
        args = (cuts, u, w, x, p, offset, slope, (Z, A, Z, A), ed, hist)
        _agree(H._twoBodyChunk(*args), H._twoBodyChunkNumpy(*args))


def test_a_strided_table_reaches_the_kernel():
    """A model table's xs can be a strided view (an ENDF-decoded MF6 recoil
    spectrum); the kernel needs C-contiguous doubles. Fe-56 B-VIII.1's heating
    by the ENDF route stopped on it ("ndarray is not C-contiguous", 2026-10-16)."""
    args = list(_case(np.random.default_rng(7), False, False, 26))
    x = args[5]
    wide = np.empty(2 * x.size)
    wide[::2] = x
    args[5] = wide[::2]
    assert not args[5].flags.c_contiguous
    _agree(H._emissionChunk(*args), H._emissionChunkNumpy(*args))



@pytest.mark.parametrize("name", ["nrt", "arc-dpa"])
def test_native_kernels_count_displacements_as_numpy(name):
    # Under a displacement model both kernels return N_d(T_d) of each recoil;
    # still the NumPy reference to the bit (Fe: NRT with HEATR's 40 eV, arc-dpa with
    # Nordlund's constants).
    model = H.displacementModels(26, 40.0)[name]
    token = H._DISPLACEMENTS.set(model)
    try:
        rng = np.random.default_rng(5 + len(name))
        for lab in (False, True):
            for _ in range(30):
                args = _case(rng, False, lab, 26)
                if len(args[0]) < 2:
                    continue
                _agree(H._emissionChunk(*args), H._emissionChunkNumpy(*args))
        u, w = np.polynomial.legendre.leggauss(16)
        x = np.linspace(-1, 1, 9); p = np.ones(9) / 2
        for E in (1e3, 1e5, 1e7):
            c = E * 55.4 / 56.4 ** 2
            args = (np.linspace(-1, 1, 7), u, w, x, p, 2 * c, 2 * c * 0.9, (26, 55.4, 26, 55.4), 40.0, False)
            _agree(H._twoBodyChunk(*args), H._twoBodyChunkNumpy(*args))
    finally:
        H._DISPLACEMENTS.reset(token)


def test_joint_two_body_chunk_is_each_component_bit_for_bit():
    # One evaluation of T_d per recoil for MT444, NRT and arc-dpa: each component
    # is the separate kernel's sum to the bit, and the native kernel NumPy's.
    models = tuple(H.displacementModels(26, 40.0).values())
    rng = np.random.default_rng(11)
    u, w = np.polynomial.legendre.leggauss(16)
    for _ in range(40):
        x = np.unique(np.r_[-1.0, 1.0, rng.uniform(-1, 1, 7)]); p = rng.uniform(0.1, 2.0, len(x))
        E = 10.0 ** rng.uniform(3, 7); c = E * 55.4 / 56.4 ** 2; r = rng.uniform(0.3, 1.0)
        args = (np.linspace(-1, 1, 6), u, w, x, p, c * (1 + r * r), 2 * c * r, (26, 55.4, 26, 55.4), 40.0, False)
        joint = H._twoBodyChunkJoint(*args, models)
        _agree(joint, H._twoBodyChunkJointNumpy(*args, models))
        assert joint[0] == H._twoBodyChunk(*args)
        for k, model in enumerate(models):
            token = H._DISPLACEMENTS.set(model)
            try:
                assert joint[1 + k] == H._twoBodyChunk(*args)
            finally:
                H._DISPLACEMENTS.reset(token)


@pytest.mark.parametrize("lab", [False, True])
@pytest.mark.parametrize("Zr", [0, 2, 26])
def test_joint_emission_chunk_is_numpy(lab, Zr):
    # The joint kernel splits the recoil angle at every component's threshold; the
    # native kernel and the NumPy reference agree to the bit, component by component.
    models = tuple(H.displacementModels(74, 90.0).values())   # W: three distinct thresholds
    rng = np.random.default_rng(31 + 10 * lab + Zr)
    for _ in range(40):
        args = list(_case(rng, False, lab, Zr))
        if len(args[0]) < 2:
            continue
        partition = args[13]
        ed = args[12]
        thresholds = [ed] + ([H._thresholdOf(float(m.ed), *map(float, partition)) for m in models] if Zr else [])
        joint = (args[:12] + [thresholds, ed, partition, args[14], args[15], models])
        _agree(H._emissionChunkJoint(*joint), H._emissionChunkJointNumpy(*joint[:12],
               (thresholds + [thresholds[-1]] * 3)[:3], *joint[13:]))


def _numpy(monkeypatch_target, function, *args):
    """``function(*args)`` with the native kernel switched off: the NumPy definition."""
    saved = H._heating_native
    H._heating_native = None
    try:
        return function(*args)
    finally:
        H._heating_native = saved


@pytest.mark.parametrize("model", [None, "nrt", "arc-dpa"])
@pytest.mark.parametrize("cm", [False, True])
def test_native_recoil_rows_equal_numpy(cm, model):
    # _recoilDamage's rows: shared rule and one row per outgoing energy (Kalbach-Mann).
    token = H._DISPLACEMENTS.set(None if model is None else H.displacementModels(26, 40.0)[model])
    try:
        rng = np.random.default_rng(7 + cm + (0 if model is None else len(model)))
        for _ in range(40):
            n, m_ = int(rng.integers(1, 60)), int(rng.choice([2, 8, 16, 32, 100, 200]))
            E = 10.0 ** rng.uniform(3, 7.3)
            e = np.sort(rng.uniform(0.0, E, n))
            mus = np.sort(rng.uniform(-1.0, 1.0, m_))
            for weights in (rng.uniform(0.0, 1.0, m_), rng.uniform(0.0, 1.0, (n, m_))):
                args = (E, e, mus, weights, cm, 54.4, float(rng.choice([1.0, 0.9986, 3.97])), 55.4, 26, 26, 40.0)
                got = H._recoilRows(*args)
                want = _numpy(None, H._recoilRows, *args)
                _agree(got, want)
    finally:
        H._DISPLACEMENTS.reset(token)


def _momenta(rng, n):
    pz = rng.normal(0.0, 1e3, n)
    pp = np.abs(rng.normal(0.0, 1e3, n))
    w = rng.uniform(0.0, 1.0, n)
    return H._binMomenta(pz, pp, w / w.sum())


@pytest.mark.parametrize("model", [None, "nrt"])
def test_native_momenta_damage_equals_numpy(model):
    # _independentRecoilDamage's last addition: every pair of bins times the azimuths.
    token = H._DISPLACEMENTS.set(None if model is None else H.displacementModels(26, 40.0)[model])
    try:
        rng = np.random.default_rng(11)
        for na, nb in ((5, 7), (30, 600), (576, 576)):
            a, b = _momenta(rng, na), _momenta(rng, nb)
            P, ar, zr, Z, A, ed = 5.3e3, 53.4, 26, 26, 55.4, 40.0

            def damage(qz, vz, qp2, w):
                er = ((P - qz) ** 2 + vz + qp2) / (2.0 * ar)
                return float(np.sum(w * H.lindhard(er, zr, ar, Z, A, ed)))

            got = H._momentaDamage(a, b, P, ar, zr, Z, A, ed, damage)
            _agree(got, H._addMomenta(a, b, last=damage))
    finally:
        H._DISPLACEMENTS.reset(token)


def test_native_momenta_bins_equal_numpy_bit_for_bit():
    # The intermediate additions of three or more particles: the bins' five sums.
    rng = np.random.default_rng(13)
    for na, nb in ((5, 7), (30, 600), (576, 576)):
        a, b = _momenta(rng, na), _momenta(rng, nb)
        got = H._addMomenta(a, b)
        want = _numpy(None, H._addMomenta, a, b)
        for x, y in zip(got, want):
            np.testing.assert_array_equal(x, y)


def test_native_points_bins_equal_numpy_bit_for_bit():
    # Each particle's momenta binned before they are added (_binMomenta).
    rng = np.random.default_rng(17)
    for n in (30, 5000, 200000):
        pz, pp, w = rng.normal(0.0, 1e3, n), np.abs(rng.normal(0.0, 1e3, n)), rng.uniform(0.0, 1.0, n)
        got = H._binMomenta(pz, pp, w)
        want = _numpy(None, H._binMomenta, pz, pp, w)
        for x, y in zip(got, want):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("kalbach", [False, True])
def test_native_unit_base_integral_equals_numpy(flat, kalbach):
    # _unitBaseCMMean's cross moment, one quadrature: correlated (mu) and Kalbach-Mann
    # (r and a, the Langevin function on both sides of |a| = 0.01).
    rng = np.random.default_rng(19 + 2 * flat + kalbach)
    for _ in range(60):
        n = int(rng.integers(2, 80))
        x = np.cumsum(rng.uniform(0.1, 2.0, n)) * 10.0 ** rng.integers(3, 6)
        x -= x[0] * rng.choice([0.0, 0.9])
        p = rng.uniform(0.0, 2.0, n)
        a, width = float(x[0]), float(x[-1] - x[0])
        qa, qw = a * rng.uniform(0.5, 1.5), width * rng.uniform(0.5, 1.5)
        nodes, ww = np.polynomial.legendre.leggauss(int(rng.choice([4, 8, 16, 32])))
        if kalbach:
            angular = (rng.uniform(0.0, 1.0, n), rng.choice([1e-3, 5e-3, 0.5, 3.0], n) * rng.choice([-1, 1], n))
        else:
            angular = rng.uniform(-1.0, 1.0, n)
        args = (x, p, angular, nodes, ww, a, width, qa, qw, flat, kalbach)
        _agree(H._unitBaseIntegral(*args), H._unitBaseIntegralNumpy(*args))


def test_the_kernels_sum_a_whole_array_as_this_numpy_does():
    # NumPy 2.5 sums a contiguous array pairwise over all of it; older NumPy went
    # by buffers of 8192. The kernel takes whichever this NumPy does (heating.py
    # measures it on import), else every sum longer than a buffer is an ulp off.
    rng = np.random.default_rng(5)
    buffer = H._numpySumBuffer()
    for n in (100, 8191, 8193, 20000, 70001):
        a = rng.normal(size=n) * 10.0 ** rng.uniform(-8, 8, n)
        assert native.total(a, buffer) == float(np.sum(a))
