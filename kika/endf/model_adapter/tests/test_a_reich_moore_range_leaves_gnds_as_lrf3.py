"""A Reich-Moore range read from GNDS is written as the LRF=3 range the tape had.

GNDS holds it as (L, J, channel spin) spin groups with no ENDF bookkeeping. The
tape kika writes from that GNDS carries the source's resonance rows in the
source's order, with AJ's sign, AP, APL, AWRI, NAPS and LAD, and its 0 K
resonance cross sections equal the source's bit for bit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf import read_endf

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
TAPES = ("micro_mn55_mf2.endf", "micro_fe56_xs_and_angular.endf")


def _viaGnds(source: Path, tmp_path):
    gnds = tmp_path / f"{source.stem}.gnds.xml"
    kika.write(kika.read(str(source)), str(gnds), format="gnds")
    suite = kika.read(str(gnds))
    out = tmp_path / f"{source.stem}.endf"
    report = kika.write(suite, str(out), format="endf")
    return suite, out, report


def _resolved(path):
    (isotope,) = read_endf(str(path), mf_numbers=[2]).mf[2].mt[151].isotopes
    return isotope, next(r for r in isotope.energy_ranges if r.lru == 1)


def _rows(energyRange):
    return [(block.l, r.energy, r.spin, r.c3, r.c4, r.c5, r.c6)
            for block in energyRange.parameters.l_values for r in block.resonances]


def _effectiveApl(energyRange):
    p = energyRange.parameters
    return [(b.l, b.awri, b.apl_or_qx or p.ap) for b in p.l_values]


@pytest.mark.parametrize("name", TAPES)
def test_the_range_comes_back_as_the_tape_states_it(name, tmp_path):
    _, out, report = _viaGnds(DATA / name, tmp_path)
    _, source = _resolved(DATA / name)
    _, written = _resolved(out)
    assert (written.el, written.eh, written.lru, written.lrf, written.nro, written.naps) == \
        (source.el, source.eh, source.lru, source.lrf, source.nro, source.naps)
    assert (written.parameters.spi, written.parameters.ap) == \
        (source.parameters.spi, source.parameters.ap)
    assert _effectiveApl(written) == _effectiveApl(source)
    assert [b.apl_or_qx for b in written.parameters.l_values] ==         [b.apl_or_qx for b in source.parameters.l_values]
    assert written.parameters.lad == source.parameters.lad
    assert _rows(written) == _rows(source)


def test_mn55_keeps_the_sign_of_aj_for_both_channel_spins(tmp_path):
    _, out, report = _viaGnds(DATA / "micro_mn55_mf2.endf", tmp_path)
    isotope, written = _resolved(out)
    spins = {r.spin for block in written.parameters.l_values for r in block.resonances}
    assert {-3.0, -2.0, 2.0, 3.0} <= spins
    assert [r.lru for r in isotope.energy_ranges] == [1, 2]
    assert not any("unresolved" in message for message in report.losses)


@pytest.mark.parametrize("name", TAPES)
def test_its_resonance_cross_sections_are_the_tapes(name, tmp_path):
    from kika.endf.reconstruction import context_from_endf
    from kika.processing.resonances.prepare import prepare_resonances

    _, out, _ = _viaGnds(DATA / name, tmp_path)
    sigma = []
    for path in (DATA / name, out):
        tape = read_endf(str(path))
        suite = kika.read(str(path))
        context, _ = context_from_endf(tape, suite)
        region = suite.resonances.resolved[0]
        # Not EH itself: there Mn-55's tape hands over to the unresolved range.
        energies = np.geomspace(max(region.domainMin, 1e-3), region.domainMax, 4001)[:-1]
        sigma.append(prepare_resonances(suite.resonances, context).evaluate(energies))
    assert sigma[0].keys() == sigma[1].keys()
    for mt in sigma[0]:
        assert np.array_equal(sigma[0][mt], sigma[1][mt]), mt


def test_spin_groups_of_one_l_that_disagree_on_the_radius_are_refused(tmp_path):
    gnds = tmp_path / "mn55.gnds.xml"
    kika.write(kika.read(str(DATA / "micro_mn55_mf2.endf")), str(gnds), format="gnds")
    suite = kika.read(str(gnds))
    formalism = suite.resonances.resolved[0].formalism
    elastic = {r.label for r in formalism.resonanceReactions if r.reactionMT == 2}
    entrances = [c for g in formalism.spinGroups for c in g.channels
                 if c.resonanceReaction in elastic and c.L == 1]
    entrances[0].scatteringRadius = 6.0
    out = tmp_path / "mn55.endf"
    report = kika.write(suite, str(out), format="endf")
    assert 2 not in read_endf(str(out)).mf
    assert any("disagree on the radius" in message for message in report.losses)


def _mn55Gnds(tmp_path):
    gnds = tmp_path / "mn55.gnds.xml"
    kika.write(kika.read(str(DATA / "micro_mn55_mf2.endf")), str(gnds), format="gnds")
    return kika.read(str(gnds))


def test_lad_crosses_gnds_as_supports_angular_reconstruction(tmp_path):
    suite = _mn55Gnds(tmp_path)
    suite.resonances.resolved[0].formalism.supportsAngularReconstruction = True
    gnds = tmp_path / "lad.gnds.xml"
    kika.write(suite, str(gnds), format="gnds")
    assert 'supportsAngularReconstruction="true"' in gnds.read_text(encoding="utf-8")
    out = tmp_path / "lad.endf"
    kika.write(kika.read(str(gnds)), str(out), format="endf")
    assert _resolved(out)[1].parameters.lad == 1


@pytest.mark.parametrize("explicit", [True, False])
def test_an_l_whose_channel_states_no_radius_gets_ap_only_under_explicit_apl(explicit, tmp_path):
    """NNDC's GNDS keeps no APL equal to AP; FUDGE's explicitAPL says the tape wrote it."""
    from kika.gnds.endf_conversion import rMatrixHref
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags

    suite = _mn55Gnds(tmp_path)
    for group in suite.resonances.resolved[0].formalism.spinGroups:
        for channel in group.channels:
            channel.scatteringRadius = channel.hardSphereRadius = None
    flags = EndfConversionFlags.of(suite)
    href = rMatrixHref(suite, 0)
    flags.conversions = [(h, f.replace(",explicitAPL", "")) if h == href else (h, f)
                         for h, f in flags.conversions]
    if explicit:
        flags.conversions = [(h, f + ",explicitAPL") if h == href else (h, f)
                             for h, f in flags.conversions]
    out = tmp_path / "mn55.endf"
    kika.write(suite, str(out), format="endf")
    written = _resolved(out)[1].parameters
    assert {b.apl_or_qx for b in written.l_values} == ({written.ap} if explicit else {0.0})
