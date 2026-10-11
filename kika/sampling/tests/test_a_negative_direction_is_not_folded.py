"""A direction stated with negative variance gets zero variance, not |λ|.

An SVD of a symmetric matrix returns |λ|, so drawing from its factors with
``psd_method="none"`` used to realise ``V|Λ|Vᵀ``: variance added in a direction
the evaluation does not state. The draw now zeroes it, as the eigen path does;
the frozen format-level samplers ask for the old folding explicitly.
"""

import numpy as np
import pytest

from kika.sampling.core import draw_samples


def _indefinite():
    rotation = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    return rotation @ np.diag([1.0, -0.5]) @ rotation.T, rotation[:, 1]


@pytest.mark.parametrize("method", ["svd", "eigen"])
def test_the_negative_direction_is_drawn_with_zero_variance(method):
    matrix, negative = _indefinite()
    samples, info = draw_samples(
        {"b": matrix}, 4096, decomposition_method=method, psd_method="none",
        null_tol=None, seed=7, verbose=False,
    )
    along = samples["b"] @ negative
    assert np.max(np.abs(along)) < 1e-12
    assert info["b"]["n_null"] == 1


def test_fold_keeps_the_magnitude_for_the_frozen_samplers():
    matrix, negative = _indefinite()
    samples, _ = draw_samples(
        {"b": matrix}, 4096, decomposition_method="svd", psd_method="none",
        null_tol=None, seed=7, verbose=False, negative_directions="fold",
    )
    assert np.var(samples["b"] @ negative) == pytest.approx(0.5, rel=0.05)


def test_an_unknown_choice_is_refused():
    with pytest.raises(ValueError, match="negative_directions"):
        draw_samples({"b": np.eye(2)}, 8, negative_directions="clip",
                     verbose=False)
