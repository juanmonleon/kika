"""The decompositions on the matrices that used to slip through.

Each case is one defect found while documenting :mod:`kika.cov.decomposition`
(11-oct-2026): a method that fell into another branch, a threshold that was
absolute where the matrix has no scale, and a non-finite entry the ``auto``
resolver cleaned for itself only.
"""
import numpy as np
import pytest

from kika.cov.decomposition import (cholesky_decomposition, eigen_decomposition,
                                    nearest_psd_higham, svd_decomposition)

INDEFINITE = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1


def test_cholesky_clip_rescale_is_its_own_branch():
    # It fell into the "none" branch, whose jitter cannot reach a -1 eigenvalue
    # within max_jitter_ratio, and raised LinAlgError.
    L = cholesky_decomposition(matrix=INDEFINITE, psd_method="clip_rescale", verbose=False)
    M = L @ L.T
    assert np.allclose(np.diag(M), np.diag(INDEFINITE), rtol=1e-6)
    assert np.linalg.eigvalsh(M)[0] >= -1e-12


def test_higham_judges_definiteness_relative_to_the_matrix():
    # Scaled to 1e-12, lambda_min = -1e-12 sat above the absolute -1e-10 and the
    # indefinite matrix came back as "already PSD".
    tiny = INDEFINITE * 1e-12
    out, info = nearest_psd_higham(tiny, verbose=False)
    assert info["iterations"] > 0 and info["n_negative_eigenvalues_before"] == 1
    assert np.linalg.eigvalsh(out)[0] >= -1e-10 * 3e-12
    # And the same rounding-level negative eigenvalue at any scale stays a no-op.
    w, V = np.linalg.eigh(np.array([[2.0, 1.0], [1.0, 2.0]]))
    almost = V @ np.diag([-1e-13, w[1]]) @ V.T
    for scale in (1.0, 1e-12):
        _, info = nearest_psd_higham(almost * scale, verbose=False)
        assert info["iterations"] == 0


@pytest.mark.parametrize("which", ["svd", "eigen", "cholesky"])
def test_auto_hands_the_cleaned_matrix_on(which):
    # auto measured the cleaned copy and, when it resolved to clip on a matrix
    # that is PSD once cleaned, passed the original (with its NaN) on.
    M = np.array([[2.0, 0.5, np.nan], [0.5, 1.0, 0.0], [np.nan, 0.0, 1.0]])
    if which == "svd":
        U, S, Vt = svd_decomposition(matrix=M, psd_method="auto", verbose=False)
        out = U @ np.diag(S) @ Vt
    elif which == "eigen":
        vals, vecs = eigen_decomposition(matrix=M, psd_method="auto", verbose=False)
        out = vecs @ np.diag(vals) @ vecs.T
    else:
        L = cholesky_decomposition(matrix=M, psd_method="auto", verbose=False)
        out = L @ L.T
    assert np.isfinite(out).all()
    assert np.allclose(out, np.nan_to_num(M, nan=0.0), atol=1e-8)
