"""MF12 LO=2 through PoPs and back (roadmap E5c, decision J1 = A).

A level's gamma cascade is a property of the level, so it goes where GNDS puts
it: the residual's PoPs entry gets ``nucleus/energy`` and one electromagnetic
``decayMode`` per transition, and the residual's decay channel a
``branching1d``/``branching3d`` photon pointing at it. The gates:

* the same cascade NNDC distributes -- ``micro_s36_b81_photons`` is cut from
  the ENDF/B-VIII.1 tape that ``n-016_S_036.endf.gnds.xml`` was made from;
* ENDF -> model -> ENDF byte for byte while the cascade is unchanged
  (``test_photon_round_trip``'s gate runs on these cuts too), and rebuilt from
  PoPs, with LP and NS as read, once it changed;
* ENDF -> GNDS -> model keeps the decay.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import kika
from kika.endf import read_endf
from kika.endf.model_adapter.decode import decodeReactionSuite
from kika.endf.model_adapter.photons import encodePhotonSections
from kika.gnds.decode import readReactionSuite
from kika.gnds.xpath import Document
from kika.nuclear_data.model import Branching1d, Branching3d, EVAL_LABEL

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
GNDS_DATA = Path(__file__).resolve().parents[3] / "gnds" / "tests" / "data"


def decode(name):
    return decodeReactionSuite(read_endf(str(DATA / f"micro_{name}_photons.endf")))


def cascade(suite):
    """``{level: (energy, [(TP, final, GP)])}`` for every level with a decay."""
    out = {}
    for pid, particle in suite.PoPs.particles.items():
        data = getattr(particle, "decayData", None)
        if data is None:
            continue
        energy = particle.energy.value if particle.energy is not None else None
        out[pid] = (energy, [
            (mode.probability, mode.finalState(),
             mode.photonEmissionProbabilities.total()
             if mode.photonEmissionProbabilities is not None else None)
            for mode in data.decayModes])
    return out


def test_the_endf_cascade_is_the_one_nndc_distributes():
    """S-36 of ENDF/B-VIII.1, read from ENDF by kika and from GNDS as NNDC wrote it."""
    fromEndf, report = decode("s36_b81")
    fromGnds, _ = readReactionSuite(Document.parse(GNDS_DATA / "n-016_S_036.endf.gnds.xml"))

    assert cascade(fromEndf) == cascade(fromGnds)
    assert len(cascade(fromEndf)) == 5
    assert not [m for m in report.unsupported if "MF12" in m]


def test_the_residual_points_at_its_decay():
    suite, _ = decode("s36_b81")
    residual = next(p for p in suite.reactionByENDF_MT(54).outputChannel.products
                    if p.pid == "S36_e4")
    (photon,) = [p for p in residual.outputChannel.products if p.pid == "photon"]
    assert isinstance(photon.multiplicity.form, Branching1d)
    assert isinstance(photon.distribution[EVAL_LABEL], Branching3d)
    assert residual.outputChannel.Q.value == pytest.approx(4522990.0)


@pytest.mark.parametrize("name", ["s36", "cm243"])
def test_lg1_and_lg2_cascades_reach_pops(name):
    suite, report = decode(name)
    levels = cascade(suite)
    assert len(levels) == 3
    assert not [m for m in report.unsupported if "MF12" in m]
    gp = {g for _, modes in levels.values() for *_, g in modes}
    # Cm-243 of JEFF-4.0 is LG=1 (no GP); S-36 of JENDL-5 is LG=2.
    assert (gp == {None}) == (name == "cm243")


def test_an_edited_cascade_is_rebuilt_from_pops_with_lp_and_ns_as_read():
    suite, report = decode("s36_b81")
    mode = suite.PoPs["S36_e4"].decayData.decayModes.decayModes[0]
    mode.probability = 0.3

    sections = {(mf, mt): s for mf, mt, s in encodePhotonSections(suite, report=report)[0]}
    rebuilt = sections[(12, 54)]
    original = read_endf(str(DATA / "micro_s36_b81_photons.endf")).mf[12].mt[54]
    assert rebuilt.transitions[0][1] == 0.3
    assert rebuilt.transitions[1] == original.transitions[1]
    assert (rebuilt.lp, rebuilt._ns, rebuilt.lg) == (original.lp, original._ns, original.lg)
    assert any("S36_e4" in m and "rebuilt" in m for m in report.warnings)
    # The other levels did not change, so they are their own bytes.
    assert str(sections[(12, 53)]) == str(read_endf(
        str(DATA / "micro_s36_b81_photons.endf")).mf[12].mt[53])


def test_the_cascade_survives_a_gnds_round_trip(tmp_path):
    suite, _ = decode("s36_b81")
    out = tmp_path / "s36.gnds.xml"
    kika.write(suite, out)
    back, report = readReactionSuite(Document.parse(out))
    assert cascade(back) == cascade(suite)
    assert not [m for m in report.losses if "decay" in m]
