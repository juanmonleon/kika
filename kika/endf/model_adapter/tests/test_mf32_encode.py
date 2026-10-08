"""MF32 out of the model: ``encodeMF32MT`` (ENDF-coverage roadmap E1).

Until this encoder existed MF32 was the one file the model decoded and could
not write: a tape written from a suite with resonance-parameter covariances came
back without them, and only the report said so. The encoder writes into the
section the decoder kept, because MF32 restates File 2's layout and the model
holds only the matrix (see the notes above ``encodeMF32MT``). So the gates are:

1. **Untouched means byte for byte.** ENDF → model → ENDF gives the section back
   character for character, on each of the four committed sub-formats.
2. **Changed means written, and read back as the model said.** A covariance
   scaled in the model is the covariance the written section decodes to; the
   uncertainties are exact to ENDF's float format and an LCOMP=2 correlation to
   the section's own NDIGIT, which the report declares.
3. **Nowhere to write means no MF32, declared.** A suite whose covariances carry
   no MF32 text (built in memory, or read from GNDS) writes none and says why;
   so does one that lost a covariance the section states.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from kika.endf.model_adapter import decodeCovarianceSuite, encodeMF32MT
from kika.endf.model_adapter.parameter_covariances import (MF32MT151_KEY,
                                                           _packIntg)
from kika.endf.read_endf import read_endf
from kika.nuclear_data.model import ConversionReport


_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


@pytest.fixture
def micro_th232_mf32_tape():
    return _DATA / "micro_th232_mf32.endf"


@pytest.fixture
def micro_cm244_mf32_tape():
    return _DATA / "micro_cm244_mf32.endf"


#: ENDF's eleven-column float keeps six or seven significant digits (six once
#: the exponent takes two), so a rewritten number is exact to ~5e-6 relative.
#: Mn-55 reaches 4.9e-6; the micro-tapes happen to stay under 2e-6.
RTOL = 1e-5


def _suite(path):
    endf = read_endf(str(path))
    suite, _ = decodeCovarianceSuite(endf)
    return endf, suite


def _decodeWritten(section):
    """The written section, parsed from its text and decoded again."""
    from kika.endf.model_adapter import decodeMF32MT
    from kika.endf.parsers.parse_mf32 import parse_mf32_mt151

    reparsed = parse_mf32_mt151(str(section).split("\n")[:-1], 151)
    covariances, _ = decodeMF32MT(reparsed, ConversionReport())
    return {c.label: c.form for c in covariances}


def test_an_untouched_mf32_comes_back_byte_for_byte(micro_mf32_tape):
    endf, suite = _suite(micro_mf32_tape)
    section, report = encodeMF32MT(suite)
    assert section is not None
    assert str(section) == str(endf.mf[32].mt[151])
    assert not report.approximations and not report.losses


def test_every_covariance_shares_one_kept_text(micro_mf32_tape):
    """One list per section, not one copy per covariance: Ta-181 is 240 131 lines."""
    _endf, suite = _suite(micro_mf32_tape)
    texts = {id(c.provenance.headerFields[MF32MT151_KEY])
             for c in suite.parameterCovariances}
    assert len(texts) == 1


def test_a_scaled_covariance_is_the_one_written(micro_mf32_tape):
    """σ × 1.1 everywhere: the correlations are unchanged, so even INTG is exact."""
    _endf, suite = _suite(micro_mf32_tape)
    for covariance in suite.parameterCovariances:
        covariance.form.matrix = covariance.form.matrix * 1.21

    section, report = encodeMF32MT(suite)
    assert section is not None
    assert any("rewritten into the section's records" in a
               for a in report.approximations)

    written = _decodeWritten(section)
    for covariance in suite.parameterCovariances:
        got = written[covariance.label]
        np.testing.assert_allclose(got.matrix, covariance.form.matrix,
                                   rtol=RTOL, atol=0.0,
                                   err_msg=covariance.label)
        np.testing.assert_array_equal(got.parameterValues,
                                      covariance.form.parameterValues)


def test_changed_parameter_values_are_written(micro_mf32_tape):
    _endf, suite = _suite(micro_mf32_tape)
    for covariance in suite.parameterCovariances:
        values = np.asarray(covariance.form.parameterValues, dtype=float)
        covariance.form.parameterValues = values * (1.0 + 1e-3)

    section, _report = encodeMF32MT(suite)
    written = _decodeWritten(section)
    for covariance in suite.parameterCovariances:
        np.testing.assert_allclose(written[covariance.label].parameterValues,
                                   covariance.form.parameterValues,
                                   rtol=1e-6, err_msg=covariance.label)


def test_a_new_correlation_is_kept_to_ndigit(micro_th232_mf32_tape):
    """Th-232's LCOMP=2 range: one coefficient moved, packed at NDIGIT."""
    _endf, suite = _suite(micro_th232_mf32_tape)
    covariance = next(c for c in suite.parameterCovariances
                      if not c.form.isRelative)
    matrix = covariance.form.matrix.copy()
    sigma = np.sqrt(np.diag(matrix))
    matrix[1, 0] = matrix[0, 1] = 0.4321 * sigma[0] * sigma[1]
    covariance.form.matrix = matrix

    section, report = encodeMF32MT(suite)
    assert any("re-packed into INTG" in a for a in report.approximations)
    got = _decodeWritten(section)[covariance.label].matrix
    rho = got[1, 0] / np.sqrt(got[0, 0] * got[1, 1])
    ndigit = section.isotopes[0].energy_ranges[0].body.correlations.ndigit
    assert abs(rho - 0.4321) <= 10.0 ** -ndigit


def test_packing_a_read_correlation_gives_back_its_integers(micro_th232_mf32_tape):
    """The packer is the exact inverse of the reader, coefficient by coefficient."""
    endf = read_endf(str(micro_th232_mf32_tape))
    intg = endf.mf[32].mt[151].isotopes[0].energy_ranges[0].body.correlations
    read = intg.correlation_matrix()
    packed = _packIntg(read, intg.ndigit)

    def asDict(entries):
        out = {}
        for ii, jj, values in entries:
            for offset, value in enumerate(values):
                if value and jj + offset < ii:
                    out[(ii, jj + offset)] = value
        return out

    assert asDict(packed) == asDict(intg.entries)


def test_no_kept_text_writes_no_mf32_and_says_so(micro_mf32_tape):
    _endf, suite = _suite(micro_mf32_tape)
    for covariance in suite.parameterCovariances:
        covariance.provenance = copy.copy(covariance.provenance)
        covariance.provenance.headerFields = {}

    section, report = encodeMF32MT(suite)
    assert section is None
    assert any("carry no MF32 section to write into" in u
               for u in report.unsupported)


def test_a_dropped_covariance_writes_no_mf32(micro_th232_mf32_tape):
    """Writing the kept section would restate what the model no longer carries."""
    _endf, suite = _suite(micro_th232_mf32_tape)
    assert len(suite.parameterCovariances) > 1
    suite.parameterCovariances = suite.parameterCovariances[1:]

    section, report = encodeMF32MT(suite)
    assert section is None
    assert any("would restate or invent a covariance" in l for l in report.losses)


def test_a_reshaped_matrix_is_refused(micro_cm244_mf32_tape):
    _endf, suite = _suite(micro_cm244_mf32_tape)
    covariance = suite.parameterCovariances[0]
    covariance.form.matrix = covariance.form.matrix[:-1, :-1]
    with pytest.raises(ValueError, match="restates File 2's resonance list"):
        encodeMF32MT(suite)


def test_lcomp0_says_what_it_cannot_state(micro_cm244_mf32_tape):
    """An ER-GN covariance has no slot in §32.2.1's eighteen numbers."""
    _endf, suite = _suite(micro_cm244_mf32_tape)
    covariance = suite.parameterCovariances[0]
    matrix = covariance.form.matrix.copy()
    matrix[0, 1] = matrix[1, 0] = 1e-6
    covariance.form.matrix = matrix

    _section, report = encodeMF32MT(suite)
    assert any("LCOMP=0 states no covariance" in l for l in report.losses)


@pytest.mark.tape
def test_lcomp1_round_trips_through_the_model(mn55_b81_tape):
    """LCOMP=1 has no committed micro-tape (the smallest evaluation is too big)."""
    endf, suite = _suite(mn55_b81_tape)
    section, _report = encodeMF32MT(suite)
    assert str(section) == str(endf.mf[32].mt[151])

    for covariance in suite.parameterCovariances:
        covariance.form.matrix = covariance.form.matrix * 1.21
    section, _report = encodeMF32MT(suite)
    written = _decodeWritten(section)
    for covariance in suite.parameterCovariances:
        np.testing.assert_allclose(written[covariance.label].matrix,
                                   covariance.form.matrix, rtol=RTOL,
                                   err_msg=covariance.label)
