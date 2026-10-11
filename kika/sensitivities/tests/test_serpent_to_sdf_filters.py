"""SERPENT sensitivities to SDF: duplicate MT labels and filters that match nothing.

Found while documenting kika.sensitivities (11-oct-2026): an MT with two
SERPENT labels announced "Taking average" and produced a 2-D array the
validation rejected, so the profile was skipped; a material or nuclide filter
that matched nothing was widened to every material or nuclide of the file.
"""
import numpy as np
import pytest

from kika.sensitivities.sensitivity_processing import _process_single_serpent_file
from kika.serpent.sens import Response, SensitivityFile, SensitivitySet
from kika.serpent.utils import parse_perturbation_label

GRID = np.array([1e-11, 1e-6, 1.0, 20.0])  # MeV, three groups


def _file(labels, values):
    """One material, one nuclide (Fe-56), one response, the given perturbations."""
    n_p, n_e = len(labels), len(GRID) - 1
    ed = np.zeros((1, 1, 1, n_p, n_e, 2))
    ed[0, 0, 0, :, :, 0] = values
    ed[0, 0, 0, :, :, 1] = 0.01
    it = ed.sum(axis=4)
    perts = [parse_perturbation_label(lab, index=i) for i, lab in enumerate(labels)]
    sset = SensitivitySet([Response("sens_keff", None, "sens_keff")], ed, it)
    return SensitivityFile(
        n_materials=1, n_nuclides=1, n_perturbations=n_p, n_energy_bins=n_e, n_mu_bins=None,
        materials=["fuel"], nuclides=[260560], perturbations_raw=list(labels),
        perturbations=perts, energy_grid=GRID, lethargy_widths=np.log(GRID[1:] / GRID[:-1]),
        data={"sens_keff": sset},
    )


def test_two_labels_for_one_mt_are_averaged_into_one_profile():
    sens = _file(["mt 4 xs", "inl scatt xs", "mt 2 xs"],
                 [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.5, 0.5, 0.5]])
    reactions = {r.mt: r for r in _process_single_serpent_file(sens, "sens_keff")}
    assert set(reactions) == {2, 4}
    assert np.allclose(reactions[4].sensitivity, [0.2, 0.2, 0.2])


@pytest.mark.parametrize("kwargs", [{"material_filter": "reflector"},
                                    {"nuclide_filter": 922350}])
def test_a_filter_that_matches_nothing_selects_nothing(kwargs):
    sens = _file(["mt 2 xs"], [[0.5, 0.5, 0.5]])
    with pytest.warns(UserWarning, match="not in this SERPENT file"):
        assert _process_single_serpent_file(sens, "sens_keff", **kwargs) == []
