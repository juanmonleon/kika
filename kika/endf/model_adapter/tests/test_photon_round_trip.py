"""MF12-15 through the model and back (roadmap E5b).

**The gate.** A tape read into a :class:`ReactionSuite` and written back carries
every photon section it came with, byte for byte under the stated exception of
``kika/endf/tests/test_photon_files.py`` (a field the source writes
non-canonically comes back canonical, its value bit for bit). That holds for
both halves of what this phase does:

* MF12 LO=1, MF14 and MF15 reach the model as ``photon`` products and are
  re-encoded **from** it — multiplicities, line energies, angular and energy
  tables — with only ENDF's own bookkeeping taken from the provenance;
* MF13 (E5d) is divided by σ on the way in and comes back as its own bytes
  while nothing changed, rebuilt by multiplication once something did;
* MF12 LO=2 (E5c) and the photons of an MT with no cross section are kept
  verbatim and declared, and come back from the kept text.

On the six committed cuts here, and — ``tape``/``slow`` — on every tape of the
three libraries whose photons the model now carries.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.model_adapter.decode import decodeReactionSuite
from kika.endf.model_adapter.photons import PHOTONS_VERBATIM_KEY, encodePhotonSections
from kika.endf.writers.assemble import writeEndfTape
from kika.endf.tests.test_photon_files import section_lines, textual_divergence
from kika.nuclear_data.model import (
    EVAL_LABEL,
    DiscreteGamma,
    Isotropic2d,
    Legendre,
    PrimaryGamma,
    Uncorrelated,
    XYs2d,
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
FIXTURES = {key: DATA / f"micro_{key}_photons.endf"
            for key in ("n14", "fe56", "u235", "s36", "hf182", "cm243", "li7")}


def decode(key):
    return decodeReactionSuite(read_endf(str(FIXTURES[key])))


def photons(channel):
    return [p for p in channel.products if p.pid == "photon"]


def form(product):
    return product.distribution.get(EVAL_LABEL)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_a_tape_comes_back_with_every_photon_section(key, tmp_path):
    suite, _ = decode(key)
    out = tmp_path / f"{key}.endf"
    writeEndfTape(suite, out)
    source = section_lines(FIXTURES[key].read_text())
    written = section_lines(out.read_text())
    assert set(written) == set(source)
    for where, lines in source.items():
        assert not textual_divergence(lines, written[where]), (key, where)


def test_the_tables_are_re_encoded_from_the_model(tmp_path):
    """Change a multiplicity, a line and an angular coefficient in the model,
    and the written sections change with them — they are not the kept text."""
    suite, _ = decode("n14")
    channel = suite.reactionByENDF_MT(102).outputChannel
    line = next(p for p in photons(channel) if isinstance(form(p).energy, DiscreteGamma))
    line.multiplicity.form.ys[0] = 0.123
    form(line).energy.value = 1.234e6
    anisotropic = next(p for p in photons(channel) if isinstance(form(p).angular, XYs2d))
    anisotropic_form = form(anisotropic).angular.function1ds[0]
    anisotropic_form.coefficients[1] = 0.25

    sections = {(mf, mt): s for mf, mt, s in encodePhotonSections(suite)[0]}
    mf12 = sections[(12, 102)]
    record = next(p for p in mf12.photons if p.eg == 1.234e6)
    assert record.y[0] == 0.123
    mf14 = sections[(14, 102)]
    assert mf14.anisotropic[0].nodes[0].coefficients[0] == 0.25


# ---------------------------------------------------------------------------
# What the model holds
# ---------------------------------------------------------------------------

def test_n14_capture_is_59_photons_and_their_total():
    """MT102: discrete lines, primaries (LP=2) and a continuum, with FUDGE's labels."""
    suite, _ = decode("n14")
    lines = photons(suite.reactionByENDF_MT(102).outputChannel)
    assert len(lines) == 59
    assert [p.label for p in lines[:3]] == ["photon", "photon__a", "photon__b"]
    assert lines[-1].label == "photon__bf"
    kinds = {type(form(p).energy) for p in lines}
    assert kinds == {DiscreteGamma, PrimaryGamma, XYs2d}
    assert all(isinstance(form(p), Uncorrelated) and form(p).isComplete for p in lines)

    total = suite.sums.multiplicitySums.byENDF_MT(102)
    assert total is not None and len(total.summands) == 59
    assert total.summands[0].href.endswith("product[@label='photon']/multiplicity")


def test_a_line_spans_the_domain_of_its_multiplicity():
    """Decision J6: the section's own domain, where FUDGE uses the cross section's."""
    suite, _ = decode("n14")
    for product in photons(suite.reactionByENDF_MT(102).outputChannel):
        energy = form(product).energy
        if isinstance(energy, (DiscreteGamma, PrimaryGamma)):
            xs = product.multiplicity.form.xs
            assert (energy.domainMin, energy.domainMax) == (xs[0], xs[-1])


def test_mf14_legendre_gets_its_a0_and_isotropic_photons_say_so():
    suite, _ = decode("n14")
    lines = photons(suite.reactionByENDF_MT(102).outputChannel)
    angular = [form(p).angular for p in lines]
    legendre = [a for a in angular if isinstance(a, XYs2d)]
    assert len(legendre) == 1
    first = legendre[0].function1ds[0]
    assert isinstance(first, Legendre) and first.coefficients[0] == 1.0
    assert sum(isinstance(a, Isotropic2d) for a in angular) == 58


def test_a_level_photon_is_the_decay_of_the_excited_residual():
    """Li-7 MT51: the photon goes into Li7_e1's own output channel, as in FUDGE,
    and replaces the placeholder the residual's decay was built with."""
    suite, _ = decode("li7")
    reaction = suite.reactionByENDF_MT(51)
    assert not photons(reaction.outputChannel)
    residual = next(p for p in reaction.outputChannel.products if p.outputChannel is not None)
    decay = photons(residual.outputChannel)
    assert len(decay) == 1 and decay[0].multiplicity is not None
    assert isinstance(form(decay[0]).energy, DiscreteGamma)


def test_mt18_has_one_photon_with_mf6_and_mf12_both_stating_it():
    """U-235 MT18: MF6 gives a LAW=0 photon and MF12/14/15 describe it; one product."""
    suite, report = decode("u235")
    fission = suite.reactionByENDF_MT(18)
    (photon,) = photons(fission.outputChannel)
    assert isinstance(form(photon).energy, XYs2d)
    assert any("MF6 already gave" in w for w in report.warnings)


def test_the_pfns_selector_still_finds_the_neutron_on_mt18():
    """Roadmap risk R1: with a photon carrying P(E'|E) beside it, MF35 used to
    meet two candidates and refuse every PFNS draw."""
    from kika.nuclear_data.model import Distribution, Frame
    from kika.sampling.perturbation_set import PerturbationSet

    suite, _ = decode("u235")
    fission = suite.reactionByENDF_MT(18)
    (photon,) = photons(fission.outputChannel)
    # The cut keeps no MF5 (4 892 lines), so the neutron is given a tabulated
    # spectrum here -- the photon's own table, which is all the selector looks at.
    neutron = fission.outputChannel.ensureProduct("n", "n")
    neutron.distribution = Distribution()
    neutron.distribution[EVAL_LABEL] = Uncorrelated(
        angular=Isotropic2d(productFrame=Frame.lab), energy=form(photon).energy,
        productFrame=Frame.lab)
    product, _ = PerturbationSet._energyOf(fission, 18)
    assert product is neutron


def test_what_the_model_does_not_carry_is_kept_and_declared():
    suite, report = decode("fe56")
    kept = suite.provenance.headerFields[PHOTONS_VERBATIM_KEY]
    assert {"12/52", "14/52"} <= set(kept)
    assert any("LO=2" in m and "E5c" in m for m in report.unsupported)
    suite, report = decode("n14")
    # N-14's MT28 and MT32 state MF13 photons and no MF3: nothing to divide by.
    assert {"13/28", "13/32"} <= set(suite.provenance.headerFields[PHOTONS_VERBATIM_KEY])
    assert "13/4" not in suite.provenance.headerFields[PHOTONS_VERBATIM_KEY]
    assert any("MT28" in m and "no cross section" in m for m in report.unsupported)


def test_the_photons_survive_a_gnds_round_trip(tmp_path):
    import kika

    suite, _ = decode("n14")
    out = tmp_path / "n14.gnds.xml"
    report = kika.write(suite, out)
    assert not [m for m in report.losses + report.unsupported if "photon" in m.lower()]
    back = kika.read(out)

    def shape(s):
        return [(p.label, type(form(p).energy).__name__)
                for p in photons(s.reactionByENDF_MT(102).outputChannel)]

    assert shape(back) == shape(suite)
    assert len(back.sums.multiplicitySums.byENDF_MT(102).summands) == 59


# ---------------------------------------------------------------------------
# The three libraries
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_every_modelled_photon_section_comes_back_from_the_model(neutron_libraries):
    """Every tape whose MF12 has an LO=1 section: decode, re-encode MF12-15.

    Only MF1, MF3, MF6 and MF12-15 are read — what the photons need (the
    header, the reactions to hang them on, MF6's own photon on MT18) — so the
    sweep stays affordable on the 12 GB box.
    """
    failures, modelled = [], 0
    for directory in neutron_libraries.values():
        for path in sorted(Path(directory).iterdir()):
            if path.suffix not in (".endf", ".jeff", ".dat"):
                continue
            text = path.read_text(encoding="latin-1")
            source = section_lines(text)
            if not any(mf in (12, 13) for mf, _ in source):
                continue
            endf = read_endf(str(path), mf_numbers=[1, 3, 6, 12, 13, 14, 15])
            suite, _ = decodeReactionSuite(endf)
            written = {(mf, mt): str(s).splitlines()[:-1]
                       for mf, mt, s in encodePhotonSections(suite)[0]}
            modelled += sum(1 for r in suite.reactions
                            if "mf12" in (getattr(r.provenance, "headerFields", None) or {}))
            if set(written) != set(source):
                failures.append((path.name, sorted(set(source) ^ set(written))[:4]))
                continue
            for where, lines in source.items():
                bad = textual_divergence(lines, written[where])
                if bad:
                    failures.append((path.name, where, bad[:2]))
                    break
            if len(failures) > 10:
                break
    assert not failures, failures
    assert modelled > 0


# ---------------------------------------------------------------------------
# MF13 (roadmap E5d)
# ---------------------------------------------------------------------------

def test_mf13_photons_are_products_with_sigma_gamma_over_sigma():
    """N-14 MT4: 43 lines; each multiplicity times σ gives back MF13's σ_γ."""
    from kika.endf.model_adapter.photons import _linlin

    endf = read_endf(str(FIXTURES["n14"]))
    suite, report = decodeReactionSuite(endf)
    reaction = suite.reactionByENDF_MT(4)
    lines = photons(reaction.outputChannel)
    assert len(lines) == 43
    assert any("MF13" in m and "multiplicity" in m for m in report.approximations)

    sigma = reaction.crossSection[EVAL_LABEL]
    table = endf.mf[13].mt[4].photons[0]
    product = next(p for p in lines if form(p).energy.value == table.eg)
    sx, sy = _linlin(sigma)
    at = np.asarray(table.x[1:-1], dtype=float)
    y = np.interp(at, product.multiplicity.form.xs, product.multiplicity.form.ys)
    np.testing.assert_allclose(y * np.interp(at, sx, sy), table.y[1:-1], rtol=1e-3)


def test_mf13_comes_back_as_its_own_bytes_while_untouched():
    suite, _ = decode("n14")
    sections = {(mf, mt): s for mf, mt, s in encodePhotonSections(suite)[0]}
    source = section_lines(FIXTURES["n14"].read_text())[(13, 4)]
    assert [l[:75] for l in str(sections[(13, 4)]).splitlines()[:-1]] ==         [l[:75] for l in source]


def test_a_changed_multiplicity_rebuilds_mf13_by_multiplication():
    """Decision J2's other half too: a changed σ rebuilds it the same way."""
    suite, _ = decode("n14")
    reaction = suite.reactionByENDF_MT(4)
    first = photons(reaction.outputChannel)[0]
    first.multiplicity.form.ys[:] = 2.0 * first.multiplicity.form.ys
    report_sections, report = encodePhotonSections(suite)
    mf13 = {mt: s for mf, mt, s in report_sections if mf == 13}[4]
    source = read_endf(str(FIXTURES["n14"])).mf[13].mt[4]
    np.testing.assert_allclose(mf13.photons[0].y[1:-1],
                               2.0 * np.asarray(source.photons[0].y[1:-1]), rtol=2e-3)
    np.testing.assert_allclose(mf13.photons[1].y[1:-1], source.photons[1].y[1:-1],
                               rtol=2e-3)
    assert any("rebuilt" in m for m in report.approximations)


def test_mf13_photons_carry_fudges_esk_flag():
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags

    suite, _ = decode("n14")
    flags = EndfConversionFlags.of(suite)
    assert flags is not None
    mf13 = [f for _, f in flags.conversions if f.startswith("MF13,ESk=")]
    assert len(mf13) == sum(len(photons(suite.reactionByENDF_MT(mt).outputChannel))
                            for mt in (4, 103, 104, 105, 107))


def test_a_sums_photons_become_an_orphan_product(n14_b81_tape, tmp_path):
    """The whole N-14: MT4 and MT103-107 are sums, so their MF13 photons go to
    five orphanProducts referencing them, as NNDC's GNDS of N-14 has them."""
    import kika
    from kika.nuclear_data.model.cross_section_forms import Reference

    suite, _ = decodeReactionSuite(read_endf(str(n14_b81_tape)))
    assert [o.ENDF_MT for o in suite.orphanProducts] == [4, 103, 104, 105, 107]
    assert all(isinstance(o.crossSection[EVAL_LABEL], Reference) for o in suite.orphanProducts)
    out = tmp_path / "n14.endf"
    writeEndfTape(suite, out)
    source = section_lines(Path(n14_b81_tape).read_text())
    written = section_lines(out.read_text())
    assert set(written) == set(source)
    for where, lines in source.items():
        assert not textual_divergence(lines, written[where]), where
    gnds = tmp_path / "n14.gnds.xml"
    kika.write(suite, gnds)
    back = kika.read(gnds)
    assert [len(o.outputChannel.products) for o in back.orphanProducts] == \
        [len(o.outputChannel.products) for o in suite.orphanProducts]
