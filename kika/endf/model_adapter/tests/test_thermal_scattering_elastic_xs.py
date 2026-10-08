"""The thermal elastic cross sections, from the model (ENDF-coverage roadmap E4).

The closed forms are tested against their properties on the flat classes
(``kika/endf/tests/test_mf7_elastic_xs.py``). What is tested here is that the
model's ``CoherentElastic.crossSection`` and ``IncoherentElastic.crossSection``
give **the same numbers** as the flat ones on every committed TSL tape — the
model stores S_table and W'(T) in its own containers and units, so agreeing
with the flat path is the statement that nothing was lost or rescaled on the
way in — and that they still do after a GNDS round trip, where the units are
whatever the file says.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kika
from kika.endf.model_adapter import decodeReactionSuite
from kika.endf.read_endf import read_endf
from kika.nuclear_data.model import CoherentElastic, IncoherentElastic

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_TAPES = ("sch4", "un_elastic", "bemetal_elastic", "jeff_be_elastic")
_ENERGIES = np.concatenate([[0.0, 1e-5], np.geomspace(1e-4, 10.0, 400)])


def _forms(suite, kind):
    return [r.doubleDifferentialCrossSection["eval"] for r in suite.reactions
            if isinstance(r.doubleDifferentialCrossSection["eval"], kind)]


@pytest.fixture(params=_TAPES)
def tape(request):
    endf = read_endf(str(_DATA / f"micro_tsl_{request.param}.endf"))
    suite, _report = decodeReactionSuite(endf)
    return endf.mf[7].sections.get(2), suite


def _assert_same_as_flat(mt2, suite):
    for form in _forms(suite, CoherentElastic):
        assert list(form.temperatures) == list(mt2.coherent.temperatures)
        for t in form.temperatures:
            np.testing.assert_allclose(form.crossSection(_ENERGIES, t),
                                       mt2.coherent.cross_section(_ENERGIES, t), rtol=1e-12)
    for form in _forms(suite, IncoherentElastic):
        for t in sorted(set(mt2.incoherent.temperatures)):
            assert form.debyeWaller(t) == pytest.approx(mt2.incoherent.debye_waller(t), rel=1e-12)
            np.testing.assert_allclose(form.crossSection(_ENERGIES, t),
                                       mt2.incoherent.cross_section(_ENERGIES, t), rtol=1e-12)


def test_the_model_gives_the_flat_cross_sections(tape):
    mt2, suite = tape
    assert mt2 is not None
    _assert_same_as_flat(mt2, suite)


def test_and_still_does_after_a_gnds_round_trip(tape, tmp_path):
    mt2, suite = tape
    path = tmp_path / "tsl.gnds.xml"
    kika.write(suite, path)
    _assert_same_as_flat(mt2, kika.read(path, covariances=False))


def test_coherent_temperatures_are_exact_and_the_staircase_is_required(tape):
    _mt2, suite = tape
    for form in _forms(suite, CoherentElastic):
        with pytest.raises(KeyError, match="not tabulated"):
            form.crossSection(0.1, 351.0)


def test_debye_waller_is_refused_outside_its_table(tape):
    _mt2, suite = tape
    for form in _forms(suite, IncoherentElastic):
        with pytest.raises(KeyError, match="outside"):
            form.debyeWaller(float(form.temperatures.max()) + 1000.0)
