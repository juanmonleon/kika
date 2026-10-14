"""MF6 LAW=1 photons with discrete lines: FUDGE's split, and the way back (D40).

A LAW=1 LANG=1 photon record may open each incident node with ``ND`` discrete
energies, each with its share of the yield, ahead of the continuum. The model
used to carry them as points of the spectrum -- an x that decreases, which no
GNDS reader takes -- so the GNDS writer left those distributions empty
(``docs/library/library-gaps.md`` D40). Now they are FUDGE's products: a
normalised continuum, then one ``discreteGamma`` per line and one
``primaryGamma`` per primary, each with ``P·y`` as its multiplicity.

Two committed cuts of ENDF/B-VIII.1, verbatim: Pt-190's capture (a continuum,
55 discrete lines and 26 primaries in one subsection) and B-11's MT103 (a node
whose yield is zero and still lists its lines).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from kika.endf import read_endf
from kika.endf.model_adapter import decodeMF6MT, encodeMF6MT
from kika.endf.model_adapter.decode import decodeReactionSuite
from kika.endf.model_adapter.photon_lines import (
    LINES_KEY,
    lineSpectrumLabels,
    mergedLineSpectra,
    restoreLineSpectrum,
)
from kika.nuclear_data.model import (
    EVAL_LABEL,
    DiscreteGamma,
    Isotropic2d,
    PrimaryGamma,
    Uncorrelated,
    XYs2d,
)
from kika.nuclear_data.model.functions.integration import integrateFunction1d

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
PT190 = DATA / "micro_pt190_photons.endf"
B11 = DATA / "micro_b11_photons.endf"


def _section(tape, mt):
    return read_endf(str(tape), mf_numbers=[6]).mf[6].mt[mt]


def _decode(tape, mt):
    entries, provenance, report = decodeMF6MT(_section(tape, mt))
    forms = {label: d for label, _p, _m, d in entries}
    multiplicities = {label: m.form for label, _p, m, _d in entries}
    return entries, provenance, report, forms, multiplicities


def _record(provenance):
    (record,) = [r for r in provenance.headerFields["mf6"]["products"]
                 if (r.get("law_fields") or {}).get(LINES_KEY)]
    return record


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------

def test_a_line_spectrum_becomes_a_continuum_and_one_product_per_line():
    entries, *_ = _decode(PT190, 102)
    labels = [label for label, *_ in entries]
    kinds = [type(d.energy).__name__ for *_, d in entries]
    # FUDGE's order and labels: the continuum, the lines by energy, the primaries.
    assert labels[:3] == ["photon", "photon__a", "photon__b"]
    assert kinds[0] == "XYs2d"
    assert kinds.count("DiscreteGamma") == 55 and kinds.count("PrimaryGamma") == 26
    assert kinds == ["XYs2d"] + ["DiscreteGamma"] * 55 + ["PrimaryGamma"] * 26
    lines = [d.energy.value for *_, d in entries if isinstance(d.energy, DiscreteGamma)]
    assert lines == sorted(lines)
    for *_, distribution in entries:
        assert isinstance(distribution, Uncorrelated)
        assert isinstance(distribution.angular, Isotropic2d)


def test_no_spectrum_of_the_split_runs_backwards():
    """The defect itself: a table whose x decreases."""
    entries, *_ = _decode(PT190, 102)
    continuum = entries[0][3].energy
    assert isinstance(continuum, XYs2d)
    for node in continuum.function1ds:
        assert np.all(np.diff(node.xs) >= 0)


def test_a_line_multiplicity_is_its_share_of_the_yield():
    """``P(E)·y(E)`` at each node that lists the line -- FUDGE's number."""
    section = _section(PT190, 102)
    product = section.products[0]
    body = product.law_data
    entries, *_ = _decode(PT190, 102)
    first = next(e for e in entries if isinstance(e[3].energy, DiscreteGamma))
    eg = first[3].energy.value
    y = np.interp(body.incident_energies, product.y_energies, product.y_values)
    expected = []
    for k in range(len(body.nd)):
        table = body.block(k)[: body.nd[k]]
        expected.append(table[table[:, 0] == eg, 1][0] * y[k])
    np.testing.assert_allclose(first[2].form.ys, expected, rtol=1e-15)


def test_the_continuum_is_normalised_and_its_multiplicity_carries_the_integral():
    section = _section(PT190, 102)
    product = section.products[0]
    body = product.law_data
    entries, *_ = _decode(PT190, 102)
    _, _, multiplicity, distribution = entries[0]
    y = np.interp(body.incident_energies, product.y_energies, product.y_values)
    for k, node in enumerate(distribution.energy.function1ds):
        assert integrateFunction1d(node) == pytest.approx(1.0, rel=1e-12)
        raw = body.block(k)[body.nd[k]:]
        integral = integrateFunction1d(type(node)(xs=raw[:, 0], ys=raw[:, 1],
                                                  interpolation=node.interpolation))
        assert multiplicity.form.ys[k] == pytest.approx(y[k] * integral, rel=1e-12)


def test_a_primary_is_named_by_its_binding_energy():
    """The file writes ``-(E_B + A/(A+1)·E)`` per node; the model keeps ``E_B``."""
    section = _section(PT190, 102)
    entries, *_ = _decode(PT190, 102)
    primaries = [d.energy for *_, d in entries if isinstance(d.energy, PrimaryGamma)]
    body = section.products[0].law_data
    first = body.block(0)[: body.nd[0], 0]
    # Below 0.01 eV FUDGE takes no projectile term out: the first node is E_B.
    assert sorted(p.value for p in primaries) == sorted(-first[first < 0])
    awr = section._awr
    last = body.block(len(body.nd) - 1)[: body.nd[-1], 0]
    shifted = -last[last < 0] - awr / (1 + awr) * body.incident_energies[-1]
    np.testing.assert_allclose(sorted(shifted), sorted(p.value for p in primaries),
                               rtol=1e-4)


def test_a_node_with_zero_yield_has_zero_multiplicity_and_still_comes_back():
    """B-11 MT103 lists lines at a node where y = 0: no photons, in the model."""
    section = _section(B11, 103)
    entries, provenance, report, forms, multiplicities = _decode(B11, 103)
    record = _record(provenance)
    lines = [label for label in lineSpectrumLabels(record)
             if isinstance(forms[label].energy, DiscreteGamma)]
    assert lines
    first = record["law_fields"][LINES_KEY]["energies"][0]
    for label in lines:
        function = multiplicities[label]
        if function.xs[0] == first:
            assert function.ys[0] == 0.0
    encoded, _ = encodeMF6MT(forms, provenance, 103, report, multiplicities)
    assert str(encoded) == str(section)


# ---------------------------------------------------------------------------
# The way back
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tape, mt", [(PT190, 102), (B11, 103)])
def test_the_section_encodes_byte_identically(tape, mt):
    section = _section(tape, mt)
    entries, provenance, report, forms, multiplicities = _decode(tape, mt)
    encoded, _ = encodeMF6MT(forms, provenance, mt, report, multiplicities)
    assert str(encoded) == str(section)


@pytest.mark.parametrize("tape", [PT190, B11])
def test_the_whole_tape_comes_back_with_its_mf6(tape, tmp_path):
    import kika

    suite, _ = decodeReactionSuite(read_endf(str(tape)))
    written = tmp_path / "written.endf"
    kika.write(suite, written, format="endf")
    before = read_endf(str(tape), mf_numbers=[6]).mf[6].mt
    after = read_endf(str(written), mf_numbers=[6]).mf[6].mt
    assert sorted(after) == sorted(before)
    for mt in before:
        assert str(after[mt]) == str(before[mt]), f"MT{mt}"


@pytest.mark.parametrize("tape, mt", [(PT190, 102), (B11, 103)])
def test_rebuilt_from_the_model_the_nodes_are_the_files(tape, mt):
    """The path a changed model takes, run on an unchanged one: the same numbers.

    Forced by spoiling the digest. Zero-yield nodes take the probabilities as
    read, the primaries the energies as read while their value is unchanged.
    """
    from kika.endf.utils import format_endf_number

    entries, provenance, _, forms, multiplicities = _decode(tape, mt)
    record = _record(provenance)
    exact = restoreLineSpectrum(record, forms, multiplicities, mt)
    forced = copy.deepcopy(record)
    forced["law_fields"][LINES_KEY]["digest"] = "spoiled"
    rebuilt = restoreLineSpectrum(forced, forms, multiplicities, mt)
    assert rebuilt.nd == exact.nd
    for a, b in zip(exact.values, rebuilt.values):
        assert list(map(format_endf_number, a)) == list(map(format_endf_number, b))


def test_a_changed_line_changes_only_its_own_probabilities():
    entries, provenance, report, forms, multiplicities = _decode(PT190, 102)
    record = _record(provenance)
    target = next(label for label in lineSpectrumLabels(record)
                  if isinstance(forms[label].energy, DiscreteGamma))
    changed = dict(multiplicities)
    function = copy.deepcopy(multiplicities[target])
    function.ys = np.asarray(function.ys) * 2.0
    changed[target] = function
    before = restoreLineSpectrum(record, forms, multiplicities, 102)
    after = restoreLineSpectrum(record, forms, changed, 102, report)
    eg = forms[target].energy.value
    moved = 0
    for k, (a, b) in enumerate(zip(before.values, after.values)):
        a = np.asarray(a).reshape(-1, 2)
        b = np.asarray(b).reshape(-1, 2)
        for j in range(len(a)):
            if j < before.nd[k] and a[j, 0] == eg:
                assert b[j, 1] == pytest.approx(2 * a[j, 1], rel=1e-12)
                moved += 1
            else:
                assert b[j, 1] == pytest.approx(a[j, 1], rel=1e-12, abs=0)
    assert moved == len(before.nd)


def test_encoding_a_split_section_without_the_multiplicities_is_refused():
    entries, provenance, report, forms, _ = _decode(PT190, 102)
    with pytest.raises(ValueError, match="does not hand back"):
        encodeMF6MT(forms, provenance, 102, report)


# ---------------------------------------------------------------------------
# The suite, GNDS, and the writers that want the lines in the table
# ---------------------------------------------------------------------------

def test_every_piece_is_a_product_of_the_reaction_flagged_mf6():
    from kika.nuclear_data.model.endf_conversion import EndfConversionFlags

    suite, _ = decodeReactionSuite(read_endf(str(PT190)))
    reaction = suite.findReactionByENDF_MT(102)
    record = _record(reaction.provenance)
    labels = lineSpectrumLabels(record)
    assert labels == [p.label for p in reaction.outputChannel.products if p.pid == "photon"]
    flagged = {href for href, flag in EndfConversionFlags.of(suite).conversions
               if flag == "MF6"}
    assert all(any(f"[@label='{label}']" in href for href in flagged) for label in labels)


def test_the_gnds_file_states_every_line(tmp_path):
    """No empty distribution, and the lines read back as the same products."""
    import kika

    suite, _ = decodeReactionSuite(read_endf(str(PT190)))
    path = tmp_path / "pt190.gnds.xml"
    kika.write(suite, path)
    text = path.read_text()
    assert "<distribution/>" not in text
    assert text.count("<discreteGamma") == 55 and text.count("<primaryGamma") == 26
    back = kika.read(str(path))
    back = back[0] if isinstance(back, tuple) else back
    photons = [p for p in back.findReactionByENDF_MT(102).outputChannel.products
               if p.pid == "photon"]
    original = [p for p in suite.findReactionByENDF_MT(102).outputChannel.products
                if p.pid == "photon"]
    assert [p.label for p in photons] == [p.label for p in original]
    for a, b in zip(photons, original):
        np.testing.assert_allclose(a.multiplicity.form.ys, b.multiplicity.form.ys,
                                   rtol=1e-12)


def test_merged_for_the_processed_writers_is_the_section_as_stated():
    """HDF5 and G4NDL put the lines first in each table, as ACER does: the
    merged product is the record's own nodes, lines then continuum."""
    section = _section(PT190, 102)
    suite, _ = decodeReactionSuite(read_endf(str(PT190)))
    reaction = suite.findReactionByENDF_MT(102)
    merged, hidden = mergedLineSpectra(reaction)
    assert list(merged) == ["photon"] and len(hidden) == 81
    energy = merged["photon"].distribution[EVAL_LABEL].energy
    body = section.products[0].law_data
    for k, node in enumerate(energy.function1ds):
        table = body.block(k)
        np.testing.assert_array_equal(node.xs, table[:, 0])
        np.testing.assert_array_equal(node.ys, table[:, 1])
    y = merged["photon"].multiplicity.form
    np.testing.assert_array_equal(y.xs, section.products[0].y_energies)


def test_a_repeated_incident_energy_takes_each_side_of_the_step():
    """JEFF-4.0 Fe-58 and Hf capture state one incident energy twice; the first
    node of the pair is the left limit, or the rebuild halves a line there."""
    from kika.endf.model_adapter.photon_lines import _yieldAtNodes
    from kika.nuclear_data.model import Regions1d

    y = Regions1d.fromEndfRegions(np.array([1.0, 2.0, 2.0, 3.0]),
                                  np.array([1.0, 1.0, 4.0, 4.0]), [(2, 2), (4, 2)])
    assert _yieldAtNodes(y, [1.0, 2.0, 2.0, 3.0]) == pytest.approx([1.0, 1.0, 4.0, 4.0])
