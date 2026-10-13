"""Independent accounting contracts for requested HEATR responses."""
import numpy as np
import pytest
from kika.processing.heating import HeatingResult


def result():
    E = np.array([1., 2.])
    # Distinct collision energies and cross sections expose double counting.
    per = {2: (E, E, 2*E), 51: (3*E, 5*E, 7*E),
           102: (11*E, 13*E, 17*E), 600: (19*E, 23*E, 29*E),
           19: (0*E, 0*E, 31*E)}
    photons = {2: 0*E, 51: 2*E, 102: 2*E, 600: 4*E, 4: 37*E}
    sigma = {2: 2*E, 51: 3*E, 102: 5*E, 600: 7*E, 19: 11*E, 4: 13*E}
    damage = sum(sigma[mt]*v[2] for mt, v in per.items())
    return HeatingResult(E, 0*E, 0*E, damage, perReaction=per, photonEnergy=photons), sigma


def test_exclusive_fission_levels_and_sum_photons_are_selected_once():
    r, xs = result()
    fission = {19: 41*r.grid}
    responses = r.partialsOf(xs, [301, 303, 304, 318, 401, 403, 442, 444, 445, 446, 447, 351],
                             fissionKerma=fission, fissionPhotonEnergy={19:43*r.grid})
    E2 = r.grid**2
    assert np.allclose(responses[301], (2+9+55+133+451-481)*E2)
    assert np.allclose(responses[303], responses[301]-2*E2)
    assert np.allclose(responses[304], (9-481)*E2)
    assert np.allclose(responses[318], 451*E2)
    assert np.allclose(responses[401], (55+133)*E2)
    assert np.allclose(responses[403], 133*E2)
    assert np.allclose(responses[442], (6+10+28+481+473)*E2)
    assert np.array_equal(responses[444], r.damage)
    assert np.allclose(responses[445], 4*E2)
    assert np.allclose(responses[446], 21*E2)
    assert np.allclose(responses[447], (85+203)*E2)
    assert np.allclose(responses[351], (9-481)*E2)


def test_local_partial_deposits_its_own_photons_and_omits_sum_subtraction():
    r, xs = result()
    assert np.allclose(r.partialsOf(xs, [304], local=True)[304], 15*r.grid**2)
    # A request for damage alone needs no fission heating input.
    assert np.array_equal(r.partialsOf(xs, [444])[444], r.damage)


def test_missing_fission_release_is_not_silently_zero():
    r, xs = result()
    with pytest.raises(ValueError, match='release energy for MT19'):
        r.partialsOf(xs, [318])
    with pytest.raises(ValueError, match='photon energy for MT19'):
        r.partialsOf(xs, [442])
    # Inactive fission does not require a fabricated release.
    xs[19] = np.zeros(2)
    assert np.array_equal(r.partialsOf(xs, [318])[318], np.zeros(2))


@pytest.mark.parametrize('ids', [[443], [444, 444], [True], [304.0], [999]])
def test_invalid_or_unimplemented_requests_fail_explicitly(ids):
    r, xs = result()
    with pytest.raises(ValueError):
        r.partialsOf(xs, ids)


@pytest.mark.parametrize('bad', [np.array([-1., 0.]), np.array([np.nan, 0.]), np.array([1.])])
def test_invalid_cross_sections_are_rejected(bad):
    r, xs = result()
    xs[2] = bad
    with pytest.raises(ValueError, match='cross section'):
        r.partialsOf(xs, [445])
