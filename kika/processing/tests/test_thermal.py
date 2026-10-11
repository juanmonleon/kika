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
