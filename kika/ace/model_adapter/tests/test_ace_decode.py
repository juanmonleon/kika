"""ACE → ``reactionSuite``: the numbers, the style chain, and the Q values.

No fixture is committed — a real ACE file is 16 MB at best and truncating one
means recomputing the internal pointer blocks, which would make the fixture
kika's arithmetic rather than the library's. Everything here is ``tape``-marked
and runs against the shared tree.

**The sharpest test in this file is about Q values**, because it contradicts
an earlier conclusion, "ACE lacks ``qm``/``qi``", which made
``CrossSection.to_endf()`` raise for an ACE-sourced section on that basis. The
raise is right; the reason was not. ACE's LQR block carries one Q per reaction,
in MeV, positionally aligned with MTR — and reading it back gives Fe-56 capture
+7.646 MeV, which is the right answer. What ACE has no counterpart for is **QM**
and **LR**. So the defect was never the format's; it was ``from_ace`` not
reading a block that is present.
"""
from __future__ import annotations

import numpy as np
import pytest

from kika.ace import read_ace
from kika.ace.model_adapter import MONTE_CARLO_LABEL, aceLabels, decodeAce, qValuesByMT
from kika.nuclear_data import CrossSection
from kika.nuclear_data.model import (Evaluated, GriddedCrossSection, Heated, MonteCarlo_cdf,
                                     URR_probabilityTables1d, XYs1d)

pytestmark = pytest.mark.tape


@pytest.fixture(scope="module")
def ace(fe56_ace):
    return read_ace(str(fe56_ace))


@pytest.fixture(scope="module")
def decoded(ace):
    return decodeAce(ace)


@pytest.fixture(scope="module")
def labels(ace):
    return aceLabels(ace)


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------

def test_every_reaction_reproduces_the_flat_path_array_for_array(ace, decoded, labels):
    """``assert_array_equal``, not ``allclose``. The adapter re-expresses; the
    moment it starts recomputing, this is the test that says so."""
    suite, _ = decoded
    assert len(suite.reactions) > 0

    for reaction in suite.reactions:
        mt = reaction.id.ENDF_MT
        expected = CrossSection.from_ace(ace, mt)
        form = reaction.crossSection[labels[0]]

        np.testing.assert_array_equal(form.xs, expected.energies)
        np.testing.assert_array_equal(form.ys, expected.values)


def test_the_composite_mts_are_present(ace, decoded):
    """MT4, MT101 and friends are summed on demand, and are real cross sections.

    Losing them would make the suite quietly smaller than the file's own view of
    itself, which is the failure this whole increment is written against.
    """
    from kika._constants import MT_COMPOSITE_ORDER

    suite, _ = decoded
    # Since A3 a summed MT is a CrossSectionSum in suite.sums, not a reaction
    decodedMTs = {r.id.ENDF_MT for c in (suite.reactions, suite.sums, suite.fissionComponents,
                                         suite.productions) for r in c}
    assert not {r.id.ENDF_MT for r in suite.reactions} & {1, 3, 4, 27, 101}
    available = {mt for mt in MT_COMPOSITE_ORDER
                 if ace.cross_section._get_or_compute_reaction(mt) is not None}
    assert available <= decodedMTs, f"composites lost: {sorted(available - decodedMTs)}"


def test_energies_are_in_ev_not_mev(decoded, labels):
    """ACE is MeV throughout and the model is eV, as ENDF is. Getting this wrong
    is a factor of a million that still plots as a plausible curve."""
    suite, _ = decoded
    form = suite.reactionByENDF_MT(2).crossSection[labels[0]]
    assert form.domainMax > 1.0e6


def test_the_form_is_an_xys1d_and_not_a_single_region_regions1d(decoded, labels):
    """ACE has no ``(NBT, INT)`` regions: it is lin-lin on one union grid. A
    one-region ``regions1d`` would assert a structure the file does not have."""
    suite, _ = decoded
    form = suite.reactionByENDF_MT(2).crossSection[labels[0]]
    assert isinstance(form, XYs1d)
    assert form.interpolation.value == "lin-lin"


# ---------------------------------------------------------------------------
# The processed styles — what makes ACE not an evaluation
# ---------------------------------------------------------------------------

def test_the_derived_from_chain_resolves_to_an_evaluated_style(decoded, labels):
    suite, _ = decoded
    heated, _ = labels
    chain = [s.label for s in suite.styles.chain(heated)]
    assert chain == [heated, "eval"]
    assert isinstance(suite.styles.evaluatedFor(heated), Evaluated)


def test_the_labels_are_the_processed_suites(decoded, labels):
    """``heated_<T>``/``URR_<T>``/``MonteCarlo``, the labels the
    HDF5 decoder and kika's processing give, so an ACE suite needs no relabelling
    to be written to HDF5 or back to ACE. No ``gridded`` style any more."""
    suite, _ = decoded
    heated, urr = labels
    assert heated == "heated_294K" and urr == "URR_294K"
    assert isinstance(suite.styles[MONTE_CARLO_LABEL], MonteCarlo_cdf)
    assert not any(isinstance(s, GriddedCrossSection) for s in suite.styles)


def test_an_ace_suite_goes_to_hdf5_without_relabelling(ace, decoded):
    """What D-A11 buys: the HDF5 encoder takes the decoded suite as it is.

    Before it, ``a3_gate.py`` copied every ``gridded`` form to ``heated_<T>``
    and added the styles by hand before it could encode.
    """
    from kika.hdf5.neutron_encode import encodeNeutron

    suite, _ = decoded
    tree = encodeNeutron(suite, awr=float(ace.header.atomic_weight_ratio))
    (name, nuclide), = tree.groups()
    assert list(nuclide["kTs"].children) == ["294K"]
    assert "294K" in nuclide["reactions/reaction_002"].children


def test_the_cross_sections_are_heated_and_never_evaluated(decoded, labels):
    """The whole point. An ACE file contains no evaluated data, so nothing in it
    may be labelled ``eval`` — otherwise a processed representation starts
    claiming to be the evaluation it came from."""
    suite, _ = decoded
    for reaction in suite.reactions:
        assert "eval" not in reaction.crossSection.forms
        assert labels[0] in reaction.crossSection.forms


def test_the_heated_style_carries_kT_in_eV(ace, decoded):
    """ACE's header field is kT in **MeV**; 2.53e-08 is 293.6 K, not 2.53e-08 K.

    ``Header.temperature``'s inline comment says kelvin and its class docstring
    says MeV. They cannot both be right, and the docstring is. The style states
    kT in eV (``eV/k``), as the HDF5 decoder's does.
    """
    from kika.hdf5.names import K_BOLTZMANN

    suite, _ = decoded
    heated = next(s for s in suite.styles if isinstance(s, Heated))
    assert heated.derivedFrom == "eval"
    assert heated.temperature.unit == "eV/k"
    assert heated.temperature.value == pytest.approx(ace.header.temperature * 1e6)
    assert 250.0 < heated.temperature.value / K_BOLTZMANN < 350.0


def test_a_reconstructed_style_is_not_invented(decoded):
    """A continuous-energy ACE has almost certainly been through RECONR. The
    file does not say so, and a style chain is a claim about provenance."""
    from kika.nuclear_data.model import CrossSectionReconstructed

    suite, _ = decoded
    assert not any(isinstance(s, CrossSectionReconstructed) for s in suite.styles)


def test_the_urr_tables_are_converted_where_they_exist(ace, decoded, labels):
    suite, report = decoded
    if not getattr(ace.unresolved_resonance, "has_data", False):
        pytest.skip("this ACE file has no URR probability tables")

    heated, urr = labels
    assert any(s.label == urr for s in suite.styles)
    assert suite.styles.chain(urr)[1].label == heated

    form = suite.reactionByENDF_MT(102).crossSection[urr]
    assert isinstance(form, URR_probabilityTables1d)
    urr = ace.unresolved_resonance
    assert form.table.table.shape == (urr.num_energies, 6, urr.table_length)
    np.testing.assert_allclose(form.table.energies, np.asarray(urr.energies) * 1e6)
    np.testing.assert_allclose(form.table.table[:, 5], np.array([t.heating_numbers for t in urr.tables]) * 1e6)
    assert not any("URR probability tables" in entry for entry in report.unsupported)



# ---------------------------------------------------------------------------
# Q values — the finding
# ---------------------------------------------------------------------------

def test_ace_does_carry_q_values_and_they_are_right(ace):
    """Three reactions with independently known Q values, from the LQR block."""
    byMT = qValuesByMT(ace)
    assert byMT, "the LQR block was not read at all"

    # Fe-56: capture is exothermic, (n,2n) and (n,p) are not.
    assert byMT[102] == pytest.approx(7.64617e6, rel=1e-6)
    assert byMT[16] == pytest.approx(-11.19706e6, rel=1e-6)
    assert byMT[103] == pytest.approx(-2.91315e6, rel=1e-6)


def test_elastic_q_is_zero_because_it_is_known_and_not_because_it_defaulted(decoded):
    """MT2 is absent from MTR. Elastic Q is 0 by definition — that is knowledge,
    and ``isKnown`` has to distinguish it from a missing value filled with 0."""
    suite, _ = decoded
    q = suite.reactionByENDF_MT(2).outputChannel.Q
    assert q.value == 0.0
    assert q.isKnown is True


def test_a_composite_has_no_q_and_says_so(decoded):
    """MT4 is a sum over inelastic levels with different Q values, so it has
    none — which is a different statement from Q = 0."""
    suite, _ = decoded
    reaction = suite.findReactionByENDF_MT(4)
    if reaction is None:
        pytest.skip("this ACE file exposes no MT4")
    assert reaction.outputChannel.Q.value is None
    assert reaction.outputChannel.Q.isKnown is False


def test_the_flat_path_reads_the_same_q_values_this_decoder_does(ace):
    """The gap this decoder recorded, now closed on the flat side too.

    ``CrossSection.from_ace`` used to build six ACE metadata keys and no Q at
    all, with ``ace.q_values`` right there. It
    calls ``qValuesByMT`` now, so there is one alignment convention rather than
    two, and this asserts the two paths agree rather than merely that the flat
    one is non-empty. Agreement is the property that matters: a second copy of
    a positional convention is exactly what drifts.
    """
    assert ace.q_values.has_q_values, (
        "the premise of this test is gone: the file has no LQR block"
    )
    expected = qValuesByMT(ace)

    flat = CrossSection.from_ace(ace, 102)
    assert flat.metadata["qi"] == expected[102]
    # QM is the mass-difference Q and has no ACE counterpart. Still absent, and
    # that is the reason `to_endf` still refuses on an ACE-sourced section.
    assert "qm" not in flat.metadata


def test_the_flat_path_gives_elastic_its_defined_zero(ace):
    """MT 2 is absent from MTR because its Q is zero by definition.

    Filling it in is knowledge, not a default — the same call the decoder
    makes, so the flat path cannot disagree with it.
    """
    assert CrossSection.from_ace(ace, 2).metadata["qi"] == 0.0


def test_a_composite_has_no_qi_on_the_flat_path_either(ace):
    """MT 4 sums levels with different Q values, so it has none.

    Absent rather than zero: the same statement the model makes with
    ``Q.isKnown is False``, made in the only way an untyped dict can make it.
    """
    flat = CrossSection.from_ace(ace, 4)
    assert "qi" not in flat.metadata


def test_the_report_names_qm_and_lr_rather_than_claiming_ace_has_no_q(decoded):
    """The wording matters: the previous formulation was false."""
    _, report = decoded
    text = " ".join(report.unsupported)
    assert "QM" in text and "LR" in text
    assert "QI *is* present" in text


def test_the_report_declares_the_absent_evaluation(decoded):
    """The ``eval`` style is a declared placeholder with nothing behind it."""
    _, report = decoded
    assert any("evaluation it does not contain" in entry for entry in report.losses)


def test_the_report_declares_the_absence_of_covariances(decoded):
    _, report = decoded
    assert any("no covariances" in entry for entry in report.unsupported)


def test_the_photons_the_ace_carries_are_decoded_not_dropped(ace, decoded):
    """Roadmap E5f.2: every MTRP entry is a photon product (test_ace_photons.py
    has the detail)."""
    suite, _ = decoded
    if not ace.photon_production_xs:
        pytest.skip("this ACE table carries no photon production")
    photons = sum(1 for node in list(suite.reactions) + list(suite.orphanProducts)
                  for p in node.outputChannel.products if p.pid == "photon")
    assert photons == len(ace.photon_production_xs.cross_sections)
