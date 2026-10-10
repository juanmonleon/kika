"""A level's gamma lines, the whole cascade down (``cascadeLines``).

MF12 LO=2 states only each level's own transitions; what a reaction to a level
emits is the cascade followed to the ground state. A hand-built scheme pins the
arithmetic, and the S-36 cut of ENDF/B-VIII.1 the reading of a real one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kika.nuclear_data.model import (ELECTROMAGNETIC, Decay, DecayData, DecayMode, DecayModes,
                                     DecayPath, Nuclide, PhotonEmissionProbabilities,
                                     PhysicalQuantity, PoPs, Product, Shell, cascadeLines)

DATA = Path(__file__).resolve().parents[3] / "endf" / "tests" / "data"


def _level(pid, index, energy, transitions):
    """``transitions``: ``[(final, TP, GP or None)]``."""
    modes = DecayModes([DecayMode(
        label=str(k), mode=ELECTROMAGNETIC, probability=tp,
        photonEmissionProbabilities=(None if gp is None
                                     else PhotonEmissionProbabilities([Shell("total", gp)])),
        decayPath=DecayPath([Decay(index=k, products=[Product(pid="photon", label="photon"),
                                                       Product(pid=final, label=final)])]))
        for k, (final, tp, gp) in enumerate(transitions)])
    return Nuclide(id=pid, Z=26, A=56, nuclearLevel=index,
                   energy=None if energy is None else PhysicalQuantity(energy, "eV"),
                   decayData=DecayData(decayModes=modes) if transitions else None)


def scheme():
    """e3 -> e2 (0.6) | e1 (0.4, GP 0.5); e2 -> e1 (1.0); e1 -> ground (1.0).

    The ground state states no energy, as the ENDF adapter writes it.
    """
    pops = PoPs()
    pops.add(_level("Fe56", 0, None, []))
    pops.add(_level("Fe56_e1", 1, 1.0e6, [("Fe56", 1.0, None)]))
    pops.add(_level("Fe56_e2", 2, 2.5e6, [("Fe56_e1", 1.0, None)]))
    pops.add(_level("Fe56_e3", 3, 4.0e6, [("Fe56_e2", 0.6, None), ("Fe56_e1", 0.4, 0.5)]))
    return pops


def test_each_line_counts_every_path_through_it():
    lines = {(l.initial, l.final): l for l in cascadeLines(scheme(), "Fe56_e3")}
    assert lines[("Fe56_e3", "Fe56_e1")].photons == pytest.approx(0.2)   # TP x GP
    assert lines[("Fe56_e3", "Fe56_e2")].photons == pytest.approx(0.6)
    assert lines[("Fe56_e2", "Fe56_e1")].photons == pytest.approx(0.6)
    # Reached by both paths, the converted one included: a conversion
    # electron still takes the nucleus down to e1.
    assert lines[("Fe56_e1", "Fe56")].photons == pytest.approx(1.0)
    assert lines[("Fe56_e3", "Fe56_e1")].energy == pytest.approx(3.0e6)


def test_lines_come_highest_energy_first_and_a_ground_state_has_none():
    lines = cascadeLines(scheme(), "Fe56_e3")
    assert [l.energy for l in lines] == sorted((l.energy for l in lines), reverse=True)
    assert cascadeLines(scheme(), "Fe56") == []


def test_with_no_conversion_the_lines_carry_the_level_energy():
    pops = scheme()
    pops.particles["Fe56_e3"].decayData.decayModes.decayModes[1].photonEmissionProbabilities = None
    lines = cascadeLines(pops, "Fe56_e3")
    assert sum(l.photons * l.energy for l in lines) == pytest.approx(4.0e6)


@pytest.mark.skipif(not (DATA / "micro_s36_b81_photons.endf").exists(), reason="fixture absent")
def test_the_s36_cascade_follows_its_tp():
    from kika.endf import read_endf
    from kika.endf.model_adapter import decodeReactionSuite

    suite, _ = decodeReactionSuite(read_endf(str(DATA / "micro_s36_b81_photons.endf")))
    lines = cascadeLines(suite.PoPs, "S36_e4")
    modes = list(suite.PoPs.particles["S36_e4"].decayData.decayModes)
    direct = {m.finalState(): m.probability for m in modes}
    byFinal = {(l.initial, l.final): l.photons for l in lines}
    # Every level reached from e4 decays to the ground state in one step
    # through e1, so e1's line carries everything that went through it.
    assert byFinal[("S36_e4", "S36")] == pytest.approx(direct["S36"] * (
        modes[[m.finalState() for m in modes].index("S36")].photonEmissionProbabilities.total()))
    assert byFinal[("S36_e1", "S36")] == pytest.approx(direct["S36_e1"], rel=1e-3)
