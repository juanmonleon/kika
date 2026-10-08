"""MF7 into the model and back (ENDF-coverage roadmap E4a).

A TSL tape used to decode to an empty ``nuclear`` suite that said MF7 was not
covered. Now it is a ``thermalNeutronScatteringLaw`` suite whose reactions hold
the law in ``doubleDifferentialCrossSection``, and the gates are the usual two:

1. **Untouched means byte for byte**, per MF7 section, on the four committed
   TSL micro-tapes. Between them they cover LTHR = 1, 2 and 3, MT4 with a
   free-gas secondary, MT451, and both libraries' column widths.
2. **Changed means written and read back as the model said**: a scaled kernel,
   a new bound cross section, a new Debye-Waller integral.

Plus what is refused by name: LLN = 1 and alpha grids that differ between betas.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from kika.endf.model_adapter import decodeReactionSuite
from kika.endf.model_adapter.thermal_scattering import LLN_REFUSAL
from kika.endf.read_endf import read_endf
from kika.endf.writers.assemble import encodeTapeSections, writeEndfTape
from kika.nuclear_data.model import (TNSL_INTERACTION, CoherentElastic,
                                     FreeGasApproximation, Gridded3d,
                                     IncoherentElastic, IncoherentInelastic,
                                     PhysicalQuantity,
                                     ThermalNeutronScatteringLaw,
                                     ThermalNeutronScatteringLaw1d)

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_TAPES = ("sch4", "un_elastic", "bemetal_elastic", "jeff_be_elastic")

#: ENDF's eleven-column float keeps six or seven significant digits.
RTOL = 1e-5


def _load(name):
    endf = read_endf(str(_DATA / f"micro_tsl_{name}.endf"))
    suite, report = decodeReactionSuite(endf)
    return endf, suite, report


def _form(suite, kind):
    for reaction in suite.reactions:
        form = reaction.doubleDifferentialCrossSection["eval"]
        if isinstance(form, kind):
            return reaction, form
    raise LookupError(kind.__name__)


def _mf7(suite):
    sections, report, _mat = encodeTapeSections(suite)
    return {mt: section for mf, mt, section in sections if mf == 7}, report


@pytest.fixture(params=_TAPES)
def tape(request):
    return request.param


def test_a_tsl_tape_is_a_tsl_suite(tape):
    _endf, suite, report = _load(tape)
    assert suite.interaction == TNSL_INTERACTION
    assert not any("MF7" in u for u in report.unsupported)
    assert not any("no MF3" in l for l in report.losses), "a TSL tape has no MF3 by construction"
    for reaction in suite.reactions:
        form = reaction.doubleDifferentialCrossSection["eval"]
        link = reaction.crossSection["eval"]
        assert isinstance(link, ThermalNeutronScatteringLaw1d)
        assert link.href.endswith(f"{form.gndsNodeName}[@label='eval']")
        distribution = reaction.outputChannel.products[0].distribution["eval"]
        assert isinstance(distribution, ThermalNeutronScatteringLaw)
        assert distribution.href == link.href


def test_every_mf7_section_comes_back_byte_for_byte(tape):
    endf, suite, _report = _load(tape)
    written, report = _mf7(suite)
    assert sorted(written) == sorted(endf.mf[7].mt)
    for mt, section in written.items():
        assert str(section) == str(endf.mf[7].mt[mt]), f"MF7/MT{mt}"
    assert not report.losses


def test_lthr3_is_two_reactions_on_mt2(tmp_path):
    _endf, suite, _report = _load("un_elastic")
    assert [(r.label, r.ENDF_MT) for r in suite.reactions] == [
        ("coherent-elastic", 2), ("incoherent-elastic", 2)]


def test_the_s_table_is_a_staircase_on_the_bragg_edges():
    endf, suite, _report = _load("bemetal_elastic")
    _reaction, form = _form(suite, CoherentElastic)
    temperature, energy = form.S_table.grids
    flat = endf.mf[7].mt[2].coherent.table
    assert energy.interpolation.value == "flat"
    np.testing.assert_array_equal(energy.values, flat.x)
    np.testing.assert_array_equal(temperature.values, flat.temperatures)
    np.testing.assert_array_equal(form.S_table.values, flat.values)
    assert form.S_table.dependentAxis.unit == "eV*b"


def test_mt4_reads_the_b_array_into_atoms():
    endf, suite, _report = _load("sch4")
    _reaction, form = _form(suite, IncoherentInelastic)
    b = endf.mf[7].mt[4].b
    principal = form.principal
    assert principal.primaryScatterer and principal.numberPerMolecule == round(b[5])
    awr = b[2]
    assert principal.boundAtomCrossSection.value == pytest.approx(
        b[0] / b[5] * ((awr + 1) / awr) ** 2, rel=1e-15)
    assert isinstance(principal.selfScatteringKernel.kernel, Gridded3d)
    assert principal.selfScatteringKernel.symmetric is True   # LASYM = 0
    assert principal.T_effective is not None
    (secondary,) = [a for a in form.scatteringAtoms if a is not principal]
    assert isinstance(secondary.selfScatteringKernel.kernel, FreeGasApproximation)
    assert secondary.T_effective is None, "a free-gas atom carries no Teff"


def test_the_kernel_is_transposed_to_temperature_beta_alpha():
    endf, suite, _report = _load("sch4")
    _reaction, form = _form(suite, IncoherentInelastic)
    kernel = form.principal.selfScatteringKernel.kernel
    mt4 = endf.mf[7].mt[4]
    for j in (0, 17, len(mt4.blocks) - 1):
        np.testing.assert_array_equal(kernel.values[0, j, :], mt4.blocks[j].table.values[0])


def test_a_scaled_kernel_is_the_one_written():
    _endf, suite, _report = _load("sch4")
    _reaction, form = _form(suite, IncoherentInelastic)
    kernel = form.principal.selfScatteringKernel.kernel
    kernel.values = kernel.values * 1.5
    written, _report = _mf7(suite)
    reread = written[4]
    got = np.asarray([[blk.table.values[t] for blk in reread.blocks]
                      for t in range(len(reread.blocks[0].table.temperatures))])
    np.testing.assert_allclose(got, kernel.values, rtol=RTOL)


def test_a_new_bound_cross_section_recomputes_the_b_array():
    endf, suite, _report = _load("sch4")
    _reaction, form = _form(suite, IncoherentInelastic)
    old = form.principal.boundAtomCrossSection
    form.principal.boundAtomCrossSection = PhysicalQuantity(old.value * 1.1, old.unit)
    written, report = _mf7(suite)
    b = written[4].b
    assert b[0] == pytest.approx(endf.mf[7].mt[4].b[0] * 1.1, rel=1e-12)
    assert b[4] == endf.mf[7].mt[4].b[4], "B(5) has no model node and is kept"
    assert any("B array is recomputed" in a for a in report.approximations)


def test_a_new_debye_waller_integral_is_written():
    _endf, suite, _report = _load("un_elastic")
    _reaction, form = _form(suite, IncoherentElastic)
    form.DebyeWallerIntegral.ys = form.DebyeWallerIntegral.ys * 0.9
    written, _report = _mf7(suite)
    np.testing.assert_allclose(written[2].incoherent.w, form.DebyeWallerIntegral.ys, rtol=RTOL)


def test_the_model_fixed_point_through_a_whole_tape(tape, tmp_path):
    _endf, suite, _report = _load(tape)
    out = tmp_path / "written.endf"
    writeEndfTape(suite, out)
    again, report = decodeReactionSuite(read_endf(str(out)))
    assert [r.label for r in again.reactions] == [r.label for r in suite.reactions]
    for left, right in zip(suite.reactions, again.reactions):
        a = left.doubleDifferentialCrossSection["eval"]
        b = right.doubleDifferentialCrossSection["eval"]
        assert type(a) is type(b)
        if isinstance(a, CoherentElastic):
            np.testing.assert_array_equal(a.S_table.values, b.S_table.values)
        elif isinstance(a, IncoherentInelastic):
            np.testing.assert_array_equal(a.principal.selfScatteringKernel.kernel.values,
                                          b.principal.selfScatteringKernel.kernel.values)
            assert [x.pid for x in a.scatteringAtoms] == [x.pid for x in b.scatteringAtoms]


def test_lln1_is_refused_by_name():
    endf = read_endf(str(_DATA / "micro_tsl_sch4.endf"))
    endf.mf[7].mt[4]._lln = 1
    suite, report = decodeReactionSuite(endf)
    assert any(LLN_REFUSAL in u for u in report.unsupported)
    assert [r.ENDF_MT for r in suite.reactions] == [2]


def test_alpha_grids_that_differ_between_betas_are_refused():
    endf = read_endf(str(_DATA / "micro_tsl_sch4.endf"))
    block = endf.mf[7].mt[4].blocks[3]
    block.table = copy.deepcopy(block.table)
    block.table.x = [x * 1.01 for x in block.table.x]
    _suite, report = decodeReactionSuite(endf)
    assert any("differ between beta" in u for u in report.unsupported)
