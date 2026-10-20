"""MF2's SPI is the target's spin in the suite's PoPs.

FUDGE writes it there, on the top-level target nucleus, and it is where the
resonance reconstruction of a GNDS file looks for it. Without it a GNDS file
kika wrote from a tape could not be processed again (``context-missing``).

Checked on every committed tape with MF2/151, against the file's own parse: one
isotope and one SPI across its ranges gives that spin; anything else leaves it
unstated rather than guessed.
"""
from pathlib import Path

import pytest

import kika
from kika.endf.read_endf import read_endf

DATA = Path(__file__).resolve().parents[2] / "tests/data"
TAPES = sorted(p for p in DATA.glob("*.endf") if not p.name.startswith("dec-"))


def _mf2(path):
    try:
        tape = read_endf(str(path))
    except Exception:
        return None
    mf2 = tape.mf.get(2)
    return mf2.mt.get(151) if mf2 is not None else None


@pytest.mark.parametrize("path", TAPES, ids=lambda p: p.stem)
def test_target_spin_is_mf2_spi(path):
    section = _mf2(path)
    if section is None or not getattr(section, "isotopes", None):
        pytest.skip("no MF2/151")
    spins = set()
    for isotope in section.isotopes:
        for rng in isotope.energy_ranges:
            if (rng.lru, rng.lrf) == (1, 7):
                # No SPI on an RML range (its CONT holds 0): the target is
                # particle b of the elastic pair, IB.
                spins |= {abs(float(p.ib)) for p in rng.parameters.particle_pairs if p.mt == 2}
            elif getattr(rng.parameters, "spi", None) is not None:
                spins.add(float(rng.parameters.spi))
    suite = kika.read(str(path), format="endf", covariances=False)
    target = suite.PoPs.particles[suite.target]
    if len(section.isotopes) == 1 and len(spins) == 1:
        assert target.spin is not None and target.spin.unit == "hbar"
        assert target.spin.value == spins.pop()
    else:
        assert target.spin is None


def test_rml_formalism_pops_states_the_elastic_pair_spin():
    """The formalism's own PoPs, which the preparation reads, takes an RML
    range's spin from the elastic pair and not from the 0 of its CONT."""
    suite = kika.read(str(DATA / "micro_cl35_mf32.endf"), format="endf", covariances=False)
    (region,) = suite.resonances.resolved
    (target,) = region.formalism.PoPs.particles.values()
    assert target.spin.value == 1.5


def test_fe56_spin_survives_a_gnds_write(tmp_path):
    suite = kika.read(str(DATA / "micro_fe56_structural.endf"), format="endf", covariances=False)
    kika.write(suite, tmp_path / "fe56.gnds.xml")
    back = kika.read(str(tmp_path / "fe56.gnds.xml"), format="gnds", covariances=False)
    assert back.PoPs.particles[back.target].spin.value == 0.0


def test_formalism_pops_states_mf2_awri_as_fudge_does(tmp_path):
    """The formalism's PoPs carries MF2's AWRI as the target mass, beside MF1's
    AWR on the top-level target, and it survives a GNDS write and read."""
    from kika._constants import NEUTRON_MASS_AMU
    tape = read_endf(str(DATA / "micro_fe56_structural.endf"))
    awri = next(b.awri for r in tape.mf[2].mt[151].isotopes[0].energy_ranges
                for b in r.parameters.l_values if b.awri)
    suite = kika.read(str(DATA / "micro_fe56_structural.endf"), format="endf", covariances=False)
    (region,) = suite.resonances.resolved
    local = region.formalism.PoPs.particles[suite.target].mass
    assert local.unit == "amu" and local.value == awri * NEUTRON_MASS_AMU
    assert suite.PoPs.particles[suite.target].mass.value != local.value
    kika.write(suite, tmp_path / "fe56.gnds.xml")
    back = kika.read(str(tmp_path / "fe56.gnds.xml"), format="gnds", covariances=False)
    (region,) = back.resonances.resolved
    assert region.formalism.PoPs.particles[back.target].mass.value == local.value


def test_gnds_reconstruction_takes_the_formalism_mass():
    """FUDGE's GNDS: the reconstruction context is the formalism's AWRI mass,
    not the top-level AWR."""
    from kika.processing.resonances.suite import _resonance_mass
    suite = kika.read(str(DATA.parents[2] / "gnds/tests/data/micro_fe56.gnds.xml"),
                      format="gnds", covariances=False)
    assert _resonance_mass(suite.resonances, suite.target).value == 55.9350085699
