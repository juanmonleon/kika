"""The thermal scattering law through GNDS (ENDF-coverage roadmap E4b).

ENDF MF7 → model is E4a (``kika/endf/model_adapter/thermal_scattering.py``);
this is model → GNDS → model on the four committed TSL micro-tapes, which
between them carry every form: coherent elastic (LTHR=1, 3), incoherent elastic
(LTHR=2, 3) and incoherent inelastic with an SCT secondary atom (s-CH4).

FUDGE in the loop -- FUDGE reading the file kika writes, and kika reading the
file FUDGE writes -- is in ``kika/endf/model_adapter/tests/
test_fudge_in_the_loop.py``, behind the ``fudge`` marker.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf.model_adapter import decodeReactionSuite
from kika.endf.model_adapter.thermal_scattering import encodeMF7Sections
from kika.endf.read_endf import read_endf
from kika.nuclear_data.model import (EVAL_LABEL, CoherentElastic, ConversionReport,
                                     Evaluated, IncoherentElastic, IncoherentInelastic,
                                     Nuclide, PhysicalQuantity, TargetInfo,
                                     TargetInfoElement, TargetInfoNuclide,
                                     ThermalNeutronScatteringLaw,
                                     ThermalNeutronScatteringLaw1d, Unorthodox, XYs1d)
from kika.nuclear_data.model.thermal_scattering import TNSL_INTERACTION


def _evaluated(suite):
    return next(style for style in suite.styles if isinstance(style, Evaluated))


def _form(reaction):
    return reaction.doubleDifferentialCrossSection[EVAL_LABEL]


def _same1d(a, b):
    """W'(T), T_effective: the same curve, a repeated identical point aside."""
    xa, ya, _ = a.toEndfRegions()
    xb, yb, _ = b.toEndfRegions()
    keep = np.ones(len(xa), dtype=bool)
    keep[1:] = ~((np.diff(xa) == 0) & (np.diff(ya) == 0))
    np.testing.assert_array_equal(np.asarray(xa)[keep], xb)
    np.testing.assert_array_equal(np.asarray(ya)[keep], yb)


def _sameGridded(a, b):
    np.testing.assert_array_equal(a.values, b.values)
    for ga, gb in zip(a.grids, b.grids):
        assert (ga.index, ga.label, ga.unit) == (gb.index, gb.label, gb.unit)
        assert str(ga.interpolation) == str(gb.interpolation)
        np.testing.assert_array_equal(ga.values, gb.values)
    assert a.dependentAxis.label == b.dependentAxis.label


def _sameForm(a, b):
    assert type(a) is type(b)
    assert (a.pid, a.productFrame) == (b.pid, b.productFrame)
    if isinstance(a, CoherentElastic):
        _sameGridded(a.S_table, b.S_table)
    elif isinstance(a, IncoherentElastic):
        assert a.boundAtomCrossSection == b.boundAtomCrossSection
        _same1d(a.DebyeWallerIntegral, b.DebyeWallerIntegral)
    else:
        assert (a.primaryScatterer, a.calculatedAtThermal, a.incoherentApproximation) == \
               (b.primaryScatterer, b.calculatedAtThermal, b.incoherentApproximation)
        assert len(a.scatteringAtoms) == len(b.scatteringAtoms)
        for x, y in zip(a.scatteringAtoms, b.scatteringAtoms):
            for name in ("pid", "numberPerMolecule", "primaryScatterer", "mass", "e_max",
                         "e_critical", "boundAtomCrossSection",
                         "boundAtomCrossSectionByNuclide"):
                assert getattr(x, name) == getattr(y, name), name
            assert x.selfScatteringKernel.symmetric == y.selfScatteringKernel.symmetric
            kx, ky = x.selfScatteringKernel.kernel, y.selfScatteringKernel.kernel
            assert type(kx) is type(ky)
            if hasattr(kx, "values"):
                _sameGridded(kx, ky)
            assert (x.T_effective is None) == (y.T_effective is None)
            if x.T_effective is not None:
                _same1d(x.T_effective, y.T_effective)


@pytest.fixture
def roundTrip(micro_tsl_tape, tmp_path):
    suite, _ = decodeReactionSuite(read_endf(str(micro_tsl_tape)))
    path = tmp_path / "tsl.gnds.xml"
    kika.write(suite, path)
    return suite, kika.read(path, covariances=False), path


def test_every_tsl_form_survives_gnds(roundTrip):
    suite, back, _ = roundTrip
    assert back.interaction == TNSL_INTERACTION
    assert [r.label for r in back.reactions] == [r.label for r in suite.reactions]
    for before, after in zip(suite.reactions, back.reactions):
        _sameForm(_form(before), _form(after))


def test_the_links_point_at_the_law(roundTrip):
    """``crossSection`` and the neutron's ``distribution`` are links, both ways."""
    _, back, path = roundTrip
    root = ET.parse(path).getroot()
    for reaction in back.reactions:
        link = reaction.crossSection[EVAL_LABEL]
        assert isinstance(link, ThermalNeutronScatteringLaw1d)
        neutron = next(p for p in reaction.outputChannel.products if p.pid == "n")
        law = neutron.distribution[EVAL_LABEL]
        assert isinstance(law, ThermalNeutronScatteringLaw)
        assert law.href == link.href
        target = root.find("." + link.href[len("/reactionSuite"):])
        assert target is not None and target.tag == _form(reaction).gndsNodeName
        assert reaction.outputChannel.Q.value == 0.0
        assert neutron.multiplicity.form.constant == 1.0


def test_the_target_is_an_unorthodox_particle_and_its_composition_goes_round(roundTrip):
    suite, back, _ = roundTrip
    assert back.target == suite.target and back.target.startswith("tnsl-")
    assert isinstance(back.PoPs[back.target], Unorthodox)
    assert back.PoPs[back.target].mass == suite.PoPs[suite.target].mass
    assert _evaluated(back).targetInfo == _evaluated(suite).targetInfo
    assert _evaluated(back).projectileEnergyDomain == _evaluated(suite).projectileEnergyDomain
    info = _evaluated(back).targetInfo
    for element in (info.chemicalElements if info else []):
        for nuclide in element.nuclides:
            assert isinstance(back.PoPs[nuclide.pid], Nuclide)


def test_writing_twice_gives_the_same_file(roundTrip, tmp_path):
    _, back, first = roundTrip
    second = tmp_path / "again.gnds.xml"
    kika.write(back, second)
    assert ET.canonicalize(from_file=str(first), strip_text=True) == \
           ET.canonicalize(from_file=str(second), strip_text=True)


def test_a_repeated_point_is_kept_in_the_model_and_dropped_from_the_file(
        micro_tsl_sch4_tape, tmp_path):
    """s-CH4's W'(T) is two identical points at 22 K. A GNDS XYs1d must ascend,
    and FUDGE refuses the file otherwise; the model keeps the tape's two."""
    suite, _ = decodeReactionSuite(read_endf(str(micro_tsl_sch4_tape)))
    form = next(_form(r) for r in suite.reactions if isinstance(_form(r), IncoherentElastic))
    assert list(form.DebyeWallerIntegral.toEndfRegions()[0]) == [22.0, 22.0]
    path = tmp_path / "sch4.xml"
    report = kika.write(suite, path)
    back = kika.read(path, covariances=False)
    again = next(_form(r) for r in back.reactions if isinstance(_form(r), IncoherentElastic))
    assert list(again.DebyeWallerIntegral.toEndfRegions()[0]) == [22.0]
    assert any("repeating the one before" in line for line in report.approximations)


def test_mf7_mt451_is_written_from_target_info_when_no_text_was_kept(micro_tsl_sch4_tape):
    """FUDGE's rule for the way back, on s-CH4 given a one-nuclide composition.

    s-CH4 has no MT451 of its own, so one is stated in the model: H-1, atom
    fraction 1, its bound cross section the principal atom's. The section must
    give back B(3) as AWRI and B(1)/B(6) as SFI.
    """
    suite, _ = decodeReactionSuite(read_endf(str(micro_tsl_sch4_tape)))
    inelastic = next(_form(r) for r in suite.reactions
                     if isinstance(_form(r), IncoherentInelastic))
    principal = inelastic.principal
    principal.pid = inelastic.primaryScatterer = "H1"
    suite.PoPs.add(Nuclide(id="H1", Z=1, A=1, mass=principal.mass))
    principal.boundAtomCrossSectionByNuclide = {"H1": principal.boundAtomCrossSection}
    _evaluated(suite).targetInfo = TargetInfo([TargetInfoElement(
        "H", [TargetInfoNuclide("H1", 1.0)])])

    report = ConversionReport()
    sections, _ = encodeMF7Sections(suite, suite.provenance.mat, report)
    mt451 = next(section for mf, mt, section in sections if mt == 451)
    (element,) = mt451.elements
    (isotope,) = element.isotopes
    b = next(r for r in suite.reactions if r.ENDF_MT == 4).provenance.headerFields["mf7"]["b"]
    assert (isotope.zai, isotope.lis, isotope.atom_fraction) == (1001, 0, 1.0)
    assert element.nas == principal.numberPerMolecule
    assert isotope.awr == pytest.approx(b[2], rel=1e-12)
    assert isotope.sigma_free == pytest.approx(b[0] / b[5], rel=1e-12)
    assert any("written from targetInfo" in line for line in report.approximations)


# ----------------------------------------------------------------------
# GNDS → ENDF (roadmap E4, with G1 of gnds_to_endf_plan.md)
# ----------------------------------------------------------------------

def _withoutFlags(suite):
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags

    suite.applicationData.entries[:] = [e for e in suite.applicationData.entries
                                        if not isinstance(e, EndfConversionFlags)]
    return suite


def _sch4WithAPrincipalNuclide(tape):
    """s-CH4 with its principal atom named H-1, as an MF7/MT451 would name it."""
    suite, _ = decodeReactionSuite(read_endf(str(tape)))
    inelastic = next(_form(r) for r in suite.reactions
                     if isinstance(_form(r), IncoherentInelastic))
    principal = inelastic.principal
    principal.pid = inelastic.primaryScatterer = "H1"
    suite.PoPs.add(Nuclide(id="H1", Z=1, A=1, mass=principal.mass))
    return suite


def _suiteNote(path):
    root = ET.parse(path).getroot()
    (conversion,) = [c for c in root.iter("conversion") if c.get("href") == "/reactionSuite"]
    return conversion.get("flags")


#: The note kika writes per micro-tape, and FUDGE's where it differs. The ZA is
#: MF7/MT451's principal element's most abundant nuclide (N14, Be9), else the
#: tape's header ZA when it is not MAT + 100 (JEFF-4.0's 4000; FUDGE's file-name
#: table says 4009). s-CH4 names no nuclide at all: the MAT alone.
_NOTES = {"bemetal_elastic": "MAT=26,ZA=4009", "jeff_be_elastic": "MAT=26,ZA=4000",
          "un_elastic": "MAT=71,ZA=7014", "sch4": "MAT=34"}


def test_the_gnds_carries_fudges_mat_note(roundTrip, micro_tsl_tape):
    """FUDGE writes ``MAT=…,ZA=…`` on every TSL suite and cannot get back to ENDF
    without both halves (roadmap T6)."""
    suite, _, path = roundTrip
    assert _suiteNote(path) == _NOTES[Path(micro_tsl_tape).stem.removeprefix("micro_tsl_")]


def test_the_note_names_the_principal_scatterer_as_fudge_does(micro_tsl_sch4_tape, tmp_path):
    """FUDGE's ZA is the principal atom's, not the header's pseudo-ZA: for
    s-CH4 FUDGE writes ``MAT=34,ZA=1001``, and so does kika once H-1 is named."""
    path = tmp_path / "sch4.xml"
    kika.write(_sch4WithAPrincipalNuclide(micro_tsl_sch4_tape), path)
    assert _suiteNote(path) == "MAT=34,ZA=1001"


def test_a_tsl_suite_read_from_gnds_writes_its_tape_back(roundTrip, tmp_path):
    """GNDS → ENDF → model gives the source's forms, MAT, AWR and header.

    The pseudo-ZA is MAT + 100, said in the report: GNDS does not carry the
    tape's own, so JEFF-4.0's Be metal (4000) comes back as 126.
    """
    from kika.endf.writers.assemble import writeEndfTape

    suite, back, _ = roundTrip
    tape = tmp_path / "back.endf"
    report = writeEndfTape(back, tape)
    again, _ = decodeReactionSuite(read_endf(str(tape)))
    assert again.target == suite.target
    assert [r.label for r in again.reactions] == [r.label for r in suite.reactions]
    for before, after in zip(suite.reactions, again.reactions):
        _sameForm(_form(before), _form(after))
    source, written = suite.provenance, again.provenance
    assert (written.mat, written.awr) == (source.mat, source.awr)
    assert written.za == source.mat + 100
    assert any("MAT + 100" in line for line in report.approximations)
    for name in ("nsub", "nlib", "nver", "lrel", "nmod", "emax", "temp", "lrp", "lfi"):
        assert written.headerFields.get(name) == source.headerFields.get(name), name


def test_the_principal_za_does_not_become_the_header_za(micro_tsl_sch4_tape, tmp_path):
    """A note's ZA=1001 is FUDGE's principal atom: the header still says 134."""
    from kika.endf.writers.assemble import writeEndfTape

    path = tmp_path / "sch4.xml"
    kika.write(_sch4WithAPrincipalNuclide(micro_tsl_sch4_tape), path)
    tape = tmp_path / "sch4.endf"
    writeEndfTape(kika.read(path, covariances=False), tape)
    again, _ = decodeReactionSuite(read_endf(str(tape)))
    assert (again.provenance.mat, again.provenance.za) == (34, 134)


def test_without_the_note_a_tsl_tape_needs_its_mat(roundTrip, tmp_path):
    """No note, no MAT: a TSL material is in no MAT table, so kika refuses by
    name, and a MAT passed by the caller is enough."""
    from kika.endf.writers.assemble import writeEndfTape

    suite, back, _ = roundTrip
    back = _withoutFlags(back)
    with pytest.raises(ValueError, match="thermal-scattering.*mat="):
        writeEndfTape(back, tmp_path / "x.endf")
    tape = tmp_path / "y.endf"
    writeEndfTape(back, tape, mat=suite.provenance.mat)
    again, _ = decodeReactionSuite(read_endf(str(tape)))
    assert (again.provenance.mat, again.provenance.za) == (suite.provenance.mat,
                                                         suite.provenance.mat + 100)
