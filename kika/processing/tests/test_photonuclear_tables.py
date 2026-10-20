"""Photonuclear tables: the photon's kinematics, its Kalbach systematics, and tables of ENDF/B-VIII.1.

The first tests need no data. The last ones read the ENDF/B-VIII.1 photonuclear
sublibrary (NNDC, ``gammas-version.VIII.1``) where it is on disk, and are skipped
elsewhere.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from kika.processing.heating import NEUTRON_REST_ENERGY, _cmToLab, _Kinematics
from kika.processing.photonuclear.tables import nuclearRestEnergy, photonTwoBodyEnergy
from kika.processing.tabulation.kalbach import kalbachSlope, photonKalbachDensity

LIBRARY = (Path(__file__).resolve().parents[5] / "NuclearData" / "photonuclear" / "endfb81"
           / "gammas-version.VIII.1")


def _tape(name):
    path = LIBRARY / name
    if not path.exists():
        pytest.skip(f"{path} is not on disk")
    return path


# --- no data ---------------------------------------------------------------

@pytest.mark.parametrize("r, a", [(0.3, 1.5), (0.9, 0.05), (0.5, 40.0), (1.0, 600.0)])
def test_photon_kalbach_shape_is_normalised_with_its_mean_cosine(r, a):
    """ENDF-6 Eq. 6.6: integral 1 over mu, <mu> = r (coth a - 1/a) (integrating the equation)."""
    norm = quad(lambda m: photonKalbachDensity(m, r, a), -1, 1, limit=500)[0]
    mean = quad(lambda m: m * photonKalbachDensity(m, r, a), -1, 1, limit=500)[0]
    assert norm == pytest.approx(1.0, abs=1e-12)
    assert mean == pytest.approx(r * (1.0 / np.tanh(a) - 1.0 / a), rel=1e-10)


def test_photon_kalbach_slope_is_the_neutron_one_times_eq_6_5():
    """a_gamma = a_n(E_gamma) sqrt(E_gamma / 2 m_n) min(4, max(1, 9.3/sqrt(E_b))), MeV; 4 at E_b = 0."""
    E, eb = 20e6, np.array([0.0, 1e6, 5e6, 12e6])
    factor = np.sqrt(20.0 / (2 * 939.56542052)) * np.array([4.0, 4.0, 4.0, 9.3 / np.sqrt(12.0)])
    assert np.allclose(kalbachSlope(E, eb, 1, 26056, 0) / kalbachSlope(E, eb, 1, 26056, 1), factor,
                       rtol=1e-14, atol=0)


def test_ame_q_of_helium_3_break_up():
    """He-3 -> p + d and -> n + p + p from AME2020: -5.4935 and -7.7180 MeV."""
    he3, p, d, n = (nuclearRestEnergy(za) for za in (2003, 1001, 1002, 1))
    assert (he3 - p - d) / 1e6 == pytest.approx(-5.49348, abs=2e-5)
    assert (he3 - n - 2 * p) / 1e6 == pytest.approx(-7.71804, abs=2e-5)


@pytest.mark.parametrize("E", [6e6, 30e6, 140e6])
def test_two_body_photon_energy_conserves_energy_and_momentum(E):
    """gamma + He-3 -> p + d: T_p + T_d = sqrt(s) - m_p - m_d, equal momenta, in the centre of mass."""
    M, mp, md = nuclearRestEnergy(2003), nuclearRestEnergy(1001), nuclearRestEnergy(1002)
    Q = M - mp - md
    tp = photonTwoBodyEnergy(np.array([E]), targetMass=M, emittedMass=mp, Q=Q)[0]
    td = photonTwoBodyEnergy(np.array([E]), targetMass=M, emittedMass=md, Q=Q)[0]
    root = np.sqrt(M * M + 2 * E * M)
    assert tp + td == pytest.approx(root - mp - md, rel=1e-12)
    assert tp * (tp + 2 * mp) == pytest.approx(td * (td + 2 * md), rel=1e-10)


def test_photon_boost_is_the_centre_of_mass_velocity():
    """With a photon the centre of mass moves at beta = E/(E + M c^2), not at a neutron's speed."""
    k = _Kinematics(A=206.0, Q=0.0, awp=1.0, photon=True)
    E = np.array([30e6])
    beta = 30e6 / (30e6 + 206.0 * NEUTRON_REST_ENERGY)
    assert k.boost(E)[0] == pytest.approx(0.5 * NEUTRON_REST_ENERGY * beta * beta, rel=1e-14)
    assert _cmToLab(E, np.array([1e6]), np.array([0.0]), k)[0] == pytest.approx(1e6 + k.boost(E)[0])


# --- ENDF/B-VIII.1 ----------------------------------------------------------

def test_deuteron_photodisintegration_heats_by_the_exact_balance():
    """H-2 (gamma, n) p: the proton's energy, the only one left local, is E + Q - <E_n> to 1e-3."""
    import kika
    from kika.processing.photonuclear import photonuclearTable

    t = photonuclearTable(kika.read(str(_tape("g-001_H_002.endf"))))
    assert t.mt_numbers == [50] and [p.particle for p in t.particles] == ["n", "H1"]
    E, total = t.energies, t.total
    n = t.particle("n")
    carried = np.zeros(E.size)
    carried[n.heating_first_index - 1:] = n.heating * total[n.heating_first_index - 1:]
    sigma = t.reaction(50).on_grid(E.size)
    balance = sigma * (E + t.reaction(50).q_value) - carried
    use = sigma > 1e-3 * sigma.max()
    assert np.allclose(t.heating[use] * total[use], balance[use], rtol=1e-3)


def test_lumped_two_body_takes_the_channel_q():
    """He-3 MT 5 states Q = 0; its deuteron (p + d channel) gets what -5.49 MeV leaves.

    In the laboratory <E_d> = T*_d + E_c + 2 sqrt(E_c T*_d) <mu_cm>, so it lies within
    2 sqrt(E_c T*_d) of T*_d + E_c, T*_d the exact centre-of-mass energy at Q(p + d).
    """
    import kika
    from kika.processing.photonuclear import photonuclearTable

    t = photonuclearTable(kika.read(str(_tape("g-002_He_003.endf"))))
    d = t.particle("H2")
    E = t.energies
    i = int(np.searchsorted(E, 6e6))
    sigma = t.reaction(5).on_grid(E.size)
    heat = d.heating[i - d.heating_first_index + 1] * t.total[i]
    mean = heat / sigma[i]                                      # y_d = 1 at 6 MeV
    M, mp, md = nuclearRestEnergy(2003), nuclearRestEnergy(1001), nuclearRestEnergy(1002)
    cm = photonTwoBodyEnergy(E[i:i + 1], targetMass=M, emittedMass=md, Q=M - mp - md)[0]
    ec = _Kinematics(A=M / NEUTRON_REST_ENERGY, Q=0.0, awp=md / NEUTRON_REST_ENERGY, photon=True).boost(E[i])
    assert abs(mean - (cm + ec)) <= 2.0 * np.sqrt(ec * cm) * (1.0 + 1e-6)
    assert mean < E[i] + (M - mp - md)


def test_fission_neutrons_include_the_delayed_ones():
    """U-235 MT 18: the neutron yield is nu_p + nu_d and the law chain has the prompt law and one per family."""
    import kika
    from kika.processing.photonuclear import photonuclearTable

    path = _tape("g-092_U_235.endf")
    suite = kika.read(str(path))
    t = photonuclearTable(suite)
    entry = next(r for r in t.particle("n").reactions if r.mt == 18)
    data = suite.findReactionByENDF_MT(18).outputChannel.fissionFragmentData
    assert len(entry.laws) == 1 + len(data.delayedNeutrons)
    prompt = suite.findReactionByENDF_MT(18).outputChannel.products[0].multiplicity
    form = prompt[list(prompt)[0]]
    delayed = sum(f.product.multiplicity[list(f.product.multiplicity)[0]].evaluate(1e7)
                  for f in data.delayedNeutrons)
    y = np.interp(1e7, entry.yield_.energies, entry.yield_.values)
    assert y == pytest.approx(float(form.evaluate(1e7)) + delayed, rel=1e-6)


def test_photonuclear_ace_writes_a_table_that_reads_back(tmp_path):
    import kika
    from kika.ace import read_ace
    from kika.ace.classes.photonuclear import PhotonuclearAce

    path = kika.endf.photonuclear_ace(_tape("g-002_He_003.endf"), tmp_path)
    table = read_ace(str(path), strict=True)
    assert isinstance(table, PhotonuclearAce) and path.name == "2003.01u"
    assert (tmp_path / "xsdir").read_text().startswith("2003.01u")
    assert [p.particle for p in table.particles] == ["n", "H1", "H2"]


def test_a_reaction_is_listed_once_per_particle_and_lines_are_law_4():
    """N-14 states MF 12 lines as separate photons of one reaction: one entry each, law 4."""
    import collections

    import kika
    from kika.processing.photonuclear import photonuclearTable

    t = photonuclearTable(kika.read(str(_tape("g-007_N_014.endf"))))
    for p in t.particles:
        assert not [mt for mt, n in collections.Counter(p.mt_numbers).items() if n > 1], p.particle
    laws = {int(law.law) for r in t.particle("photon").reactions for law in r.laws}
    assert 2 not in laws and laws <= {4, 7, 9, 33, 44, 61}
