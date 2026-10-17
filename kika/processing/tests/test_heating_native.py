"""The native damage kernel is the NumPy reference, bit for bit."""
import numpy as np
import pytest

from kika.processing import heating as H

native = pytest.importorskip("kika.processing._heating_native")


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
def test_native_emission_chunk_equals_numpy_bit_for_bit(hist, lab, Zr):
    rng = np.random.default_rng(1000 * hist + 100 * lab + Zr)
    for _ in range(60):
        args = _case(rng, hist, lab, Zr)
        if len(args[0]) < 2:
            continue
        expected = H._emissionChunkNumpy(*args)
        actual = H._emissionChunk(*args)
        assert actual == expected or (np.isnan(actual) and np.isnan(expected)), (actual, expected)


def test_native_kernel_rejects_inconsistent_buffers():
    g, w = np.polynomial.legendre.leggauss(8)
    with pytest.raises(ValueError, match="inconsistent"):
        native.emission_chunk(np.array([1.0]), g, w, g, w, np.array([0.0, 1.0]), np.array([1.0, 1.0]),
                              0., 1., 1., 1e6, 56., 1400., 1., 1., 25., False, False, False)
    with pytest.raises(ValueError, match="float64"):
        native.emission_chunk(np.array([1, 2]), g, w, g, w, np.array([0.0, 1.0]), np.array([1.0, 1.0]),
                              0., 1., 1., 1e6, 56., 1400., 1., 1., 25., False, False, False)


@pytest.mark.parametrize("hist", [False, True])
def test_native_two_body_chunk_equals_numpy_bit_for_bit(hist):
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
        assert H._twoBodyChunk(*args) == H._twoBodyChunkNumpy(*args)


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
    assert H._emissionChunk(*args) == H._emissionChunkNumpy(*args)



@pytest.mark.parametrize("name", ["nrt", "arc-dpa"])
def test_native_kernels_count_displacements_as_numpy_bit_for_bit(name):
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
                assert H._emissionChunk(*args) == H._emissionChunkNumpy(*args)
        u, w = np.polynomial.legendre.leggauss(16)
        x = np.linspace(-1, 1, 9); p = np.ones(9) / 2
        for E in (1e3, 1e5, 1e7):
            c = E * 55.4 / 56.4 ** 2
            args = (np.linspace(-1, 1, 7), u, w, x, p, 2 * c, 2 * c * 0.9, (26, 55.4, 26, 55.4), 40.0, False)
            assert H._twoBodyChunk(*args) == H._twoBodyChunkNumpy(*args)
    finally:
        H._DISPLACEMENTS.reset(token)
