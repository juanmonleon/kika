"""A |ρ| > 1 that only log1p(Σ) produces is repaired, not refused.

log1p is concave, so two bins the file states fully correlated with different
σ come out of the log transform with a correlation above 1. The file is not at
fault; the definiteness repair (clip) restores |ρ| <= 1 and only raises the
variances.
"""

import numpy as np

from kika.cov.conditioning import apply_plan, inspect_blocks


def _fully_correlated():
    sigma = np.array([0.1, 1.0])
    return np.outer(sigma, sigma)


def test_the_log_transform_excess_is_a_note_and_clip_repairs_it():
    stated = _fully_correlated()
    log = np.log1p(stated)
    assert abs(log[0, 1]) > np.sqrt(log[0, 0] * log[1, 1])

    report = inspect_blocks({"b": log}, stated={"b": stated}, predict=False)
    (block,) = report.blocks
    (bound,) = [f for f in block.findings if f.check == "correlation_bound"]
    assert bound.severity == "note" and bound.evidence["log_transform"]
    assert report.samplable

    (repaired, _record), = [(m, r) for (_k, m), r in zip(*apply_plan(
        [("b", log)], report.recommended_plan()))]
    assert np.linalg.eigvalsh(repaired).min() >= -1e-15
    assert np.all(np.diag(repaired) >= np.diag(log) - 1e-15)


def test_a_correlation_above_one_the_file_states_still_blocks():
    stated = np.array([[1.0, 1.2], [1.2, 1.0]]) * 0.01
    report = inspect_blocks({"b": np.log1p(stated)}, stated={"b": stated},
                            predict=False)
    assert not report.samplable
