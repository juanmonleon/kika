"""Conversions to ACE: an OpenMC file (neutron or thermal) through the model, and what the
NJOY-style conventions add to a suite that did not come from ACE.

An OpenMC neutron file is a processed suite like kika's own, so it writes as
ACE and reads back as the same cross sections. What OpenMC does not state
(MF1/455, which ENDF file a photon was in, ACER's heating convention) is
derived and said in the report.
"""
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

import kika
from kika.ace import read_ace

DATA = Path(__file__).resolve().parents[3] / "hdf5" / "tests" / "data"


def _read(path):
    suite = kika.read(str(path))
    return suite[0] if isinstance(suite, tuple) else suite


def _kelvin(suite):
    from kika.hdf5.names import K_BOLTZMANN
    from kika.nuclear_data.model import Heated

    return [float(s.temperature.value) / K_BOLTZMANN for s in suite.styles
            if isinstance(s, Heated) and s.temperature is not None]


def test_an_openmc_file_goes_to_ace_and_back_with_its_cross_sections(tmp_path):
    from kika.hdf5.compare import compareNeutron

    suite = _read(DATA / "Fe56.h5")
    T = _kelvin(suite)[0]
    kika.write(suite, tmp_path / "fe56.ace", format="ace", temperatures=T, extension=".80c")
    back = tmp_path / "back"
    kika.write(_read(tmp_path / "fe56.ace"), back, format="hdf5")
    comparison = compareNeutron(DATA / "Fe56.h5", next(back.glob("*.h5")))
    assert comparison.crossSections
    worst = max(d for d, _ in comparison.crossSections.values())
    assert worst < 1e-10, comparison.worst(5)


def test_an_openmc_fissile_states_its_delayed_nu_as_the_sum_of_its_groups(tmp_path):
    suite = _read(DATA / "U235.h5")
    T = _kelvin(suite)[0]
    report = kika.write(suite, tmp_path / "u235.ace", format="ace", temperatures=T, extension=".80c")
    assert any("sum of the delayed groups" in line for line in report.approximations)
    ace = read_ace(str(tmp_path / "u235.ace"), strict=True)
    assert ace.nubar.has_delayed
    groups = ace.delayed_neutron_data.precursors
    assert groups
    total = sum(np.asarray(g.probabilities, dtype=float)[0] for g in groups)
    assert total == pytest.approx(1.0, rel=1e-6)


def test_the_heating_number_is_heatrs_without_the_delayed_betas(tmp_path):
    """ESZ's heating is HEATR's local = 0 KERMA: OpenMC's MT301 less EB·σ_f."""
    from kika.ace.model_adapter.encode import encodeAce
    from kika.hdf5.fission_energy import evaluate

    suite = _read(DATA / "U235.h5")
    T = _kelvin(suite)[0]
    ace, report = encodeAce(suite, T, extension=".80c")
    assert any("delayed betas" in line for line in report.warnings)
    E = np.asarray(ace.esz_block.energies) * 1e6
    kerma = np.asarray(ace.esz_block.heating_numbers) * np.asarray(ace.esz_block.total_xs) * 1e6
    label = f"heated_{next(s.label.split('_', 1)[1] for s in suite.styles if s.label.startswith('heated_'))}"
    mt301 = next(r for r in suite.productions if int(r.ENDF_MT) == 301).crossSection.forms[label]
    fission = next(r for r in suite.reactions if int(r.ENDF_MT) == 18)
    sigmaF = fission.crossSection.forms[label]
    release = fission.outputChannel.fissionFragmentData.fissionEnergyReleases[0]
    expected = np.interp(E, mt301.xs, mt301.ys) - evaluate(release.delayedBetaEnergy, E) * \
        np.interp(E, sigmaF.xs, sigmaF.ys, left=0.0, right=0.0)
    np.testing.assert_allclose(kerma, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize("name, elastic", [("c_H_in_ZrH", "incoherent"), ("c_Graphite", "coherent")])
def test_an_openmc_thermal_file_goes_to_ace(tmp_path, name, elastic):
    from kika.tsl.ace import read_ace_thermal

    suite = _read(DATA / f"{name}.h5")
    T = _kelvin(suite)[0]
    path = tmp_path / f"{name}.20t"
    kika.write(suite, path, format="ace", temperatures=T, extension=".20t")
    block = read_ace_thermal(path).temperatures[0]
    table = block.outgoing()
    assert table.kind == "continuous" and table.incident.size == block.inelastic_energies.size
    inelastic = next(r for r in suite.reactions if int(r.ENDF_MT) == 4)
    label = next(iter(inelastic.crossSection.forms))
    np.testing.assert_allclose(block.inelastic_xs, inelastic.crossSection.forms[label].ys, rtol=1e-11)
    if elastic == "incoherent":
        form = next(r for r in suite.reactions if int(r.ENDF_MT) == 2).crossSection.forms[label]
        np.testing.assert_allclose(block.incoherent.xs, form.evaluate(block.incoherent.energies), rtol=1e-11)
        assert block.incoherent.cosines.shape == (block.incoherent.energies.size, 16)
        assert np.all(np.diff(block.incoherent.cosines, axis=1) > 0)
    else:
        form = next(r for r in suite.reactions if int(r.ENDF_MT) == 2).crossSection.forms[label]
        np.testing.assert_allclose(block.coherent.edges, form.braggEdges, rtol=1e-11)
        np.testing.assert_allclose(block.coherent.s, form.structureFactors, rtol=1e-11)


def test_thermrs_incoherent_cosines_cut_the_law_into_equal_parts():
    """iel: the mean cosine of each of n bins of p(μ) ∝ exp(−c(1−μ)) carrying 1/n each."""
    from kika.tsl.ace_writer import incoherent_elastic_table

    table = incoherent_elastic_table([0.01, 0.5, 4.0], bound_xs=80.0, debye_waller=0.03, bins=8)
    for e, row in zip(table.energies, table.cosines):
        c = 2.0 * e * 0.03
        edges = 1.0 + np.log(np.exp(-2 * c) + np.arange(9) / 8 * (1 - np.exp(-2 * c))) / c
        mu = np.linspace(edges[:-1], edges[1:], 2001)
        w = np.exp(-c * (1.0 - mu))
        np.testing.assert_allclose(row, np.trapezoid(w * mu, mu, axis=0) / np.trapezoid(w, mu, axis=0), atol=1e-6)
    np.testing.assert_allclose(table.xs, 0.5 * 80.0 * -np.expm1(-4 * table.energies * 0.03)
                               / (2 * table.energies * 0.03), rtol=1e-13)


def test_a_gnds_photon_keeps_the_file_fudge_flagged_it_from():
    from kika.hdf5.processing import _photonFile, _photonFlags
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags

    flags = EndfConversionFlags(conversions=[
        ("/reactionSuite/reactions/reaction[@label='n + O16_e1']/outputChannel/products/product[@label='photon']",
         "MF13"),
        ("/reactionSuite/reactions/reaction[@label='n + O16_e2']/outputChannel/products/product[@label='photon']",
         "MF6"),
        ("/reactionSuite/reactions/reaction[@label='n + O16_e3']/outputChannel/products/product[@label='photon']",
         "ESk=6.1e6")])
    suite = NS(applicationData=NS(entries=[flags]))
    found = _photonFlags(suite)
    product = NS(label="photon")
    files = [_photonFile(NS(label=f"n + O16_e{i}", provenance=None), None, product, found) for i in (1, 2, 3, 4)]
    assert files == [13, 16, 12, 12]
    assert _photonFile(NS(label="x", provenance=None), None, product, {}) == 16


def test_mt4_is_kept_last_when_the_urr_tables_compete_with_it():
    """ACER (unionx) keeps MT4 for the URR inelastic competition (ILF = 4), after the rest."""
    from kika.ace.model_adapter.encode import encodeAce

    suite = _read(DATA / "U235.h5")
    ace, _ = encodeAce(suite, _kelvin(suite)[0], extension=".80c")
    assert int(ace.unresolved_resonance.inelastic_flag) == 4
    assert [int(m) for m in ace.reaction_mt_data.incident_neutron][-1] == 4


def test_law_61_locators_are_positive_and_a_constant_yield_is_ty():
    """OpenMC reads a law 61 cosine table only behind a positive LC (any other is
    isotropic), as NJOY 2016 writes them; a constant integer yield is TY, not 101."""
    from kika.ace.model_adapter.encode import encodeAce

    suite = _read(DATA / "Fe56.h5")
    ace, _ = encodeAce(suite, _kelvin(suite)[0], extension=".80c")
    signs = {int(np.sign(lc)) for chain in ace.energy_distributions.incident_neutron.values()
             for law in chain if law.law == 61 for d in law.distributions for lc in d["lc"] if lc}
    assert signs == {1}
    ty = dict(zip((int(m) for m in ace.reaction_mt_data.incident_neutron),
                  (int(t) for t in ace.particle_release.incident_neutron)))
    assert abs(ty[16]) == 2


def test_a_table_read_back_from_gnds_is_put_back_on_the_union_grid():
    """GNDS spells a step as a regions1d and drops an energy repeated with the same
    value, so a processed suite read back from it is not on one grid; the encoder
    flattens the regions and reads each table on the union grid."""
    import copy

    from kika.ace.encoders.layout import encode_xss
    from kika.ace.model_adapter.encode import encodeAce
    from kika.nuclear_data.model import Regions1d, XYs1d

    suite = _read(DATA / "Fe56.h5")
    T = _kelvin(suite)[0]
    label = next(k for k in suite.reactionByENDF_MT(2).crossSection.forms if k.startswith("heated"))
    capture = suite.reactionByENDF_MT(102).crossSection.forms[label]
    i = len(capture.xs) // 2
    ys = np.array(capture.ys, dtype=float)
    x = np.asarray(capture.xs, dtype=float)
    ys[i] = ys[i - 1] + (ys[i + 1] - ys[i - 1]) * (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
    suite.reactionByENDF_MT(102).crossSection.forms[label] = XYs1d(xs=x, ys=ys, axes=capture.axes)
    expected, _ = encodeAce(suite, T, extension=".80c")

    gnds = copy.deepcopy(suite)
    gnds.reactionByENDF_MT(102).crossSection.forms[label] = XYs1d(
        xs=np.delete(x, i), ys=np.delete(ys, i), axes=capture.axes)
    elastic = gnds.reactionByENDF_MT(2).crossSection.forms[label]
    k = len(elastic.xs) // 3
    gnds.reactionByENDF_MT(2).crossSection.forms[label] = Regions1d(function1ds=[
        XYs1d(xs=elastic.xs[:k + 1], ys=elastic.ys[:k + 1]), XYs1d(xs=elastic.xs[k:], ys=elastic.ys[k:])],
        axes=elastic.axes)
    written, _ = encodeAce(gnds, T, extension=".80c")

    a, b = encode_xss(expected), encode_xss(written)
    assert list(a.nxs) == list(b.nxs) and list(a.jxs) == list(b.jxs)
    np.testing.assert_allclose(b.xss[1:], a.xss[1:], rtol=1e-14, atol=0)


def test_a_table_kika_wrote_from_openmc_decodes_and_encodes_back_the_same(tmp_path):
    """OpenMC's fission neutrons are isotropic in CM inside a lab-frame reaction,
    which kika's ACE states as an explicit record (LC = 0 at each energy). Read
    as a bare Isotropic2d it came back as LOCB = 0, five words shorter."""
    from kika.ace.encoders.layout import encode_xss
    from kika.ace.model_adapter.encode import encodeAce

    suite = _read(DATA / "U235.h5")
    kika.write(suite, tmp_path / "u235.ace", format="ace", temperatures=_kelvin(suite)[0],
               extension=".80c")
    original = read_ace(str(tmp_path / "u235.ace"))
    again, _ = encodeAce(_read(tmp_path / "u235.ace"))
    a, b = encode_xss(original), encode_xss(again)
    assert list(a.nxs) == list(b.nxs) and list(a.jxs) == list(b.jxs)
    np.testing.assert_allclose(b.xss[1:], a.xss[1:], rtol=1e-12, atol=0)
