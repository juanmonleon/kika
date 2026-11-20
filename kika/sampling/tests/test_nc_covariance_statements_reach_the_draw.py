"""MF33 sections stated through others (NC LTY=0) do not keep a tape from being drawn.

The decoder keeps an NC LTY=0 sub-subsection as a ``Sum`` (beside NI, a
``Mixed`` of the NI matrix and the ``Sum``). The draw realises those statements
through :mod:`kika.sampling.cross_section_sums`, so the assembly takes the NI
matrix and leaves the ``Sum`` out. Until 2026-11-20 it refused the whole MF33
of any tape carrying one: 37 of the 77 ENDF/B-VIII.1 neutron tapes, U-235,
U-238, Pu-239 and O-16 among them (roadmap item 9).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

from kika.endf.model_adapter.covariances import decodeCovarianceSuite
from kika.endf import read_endf
from kika.sampling.joint_blocks import ComponentKey
from kika.sampling.model_blocks import _cross_section_entries
from kika.sampling import model_perturbation
from kika.sampling.model_perturbation import _statedCrossSectionMTs

# The micro-tape is cut from Fe-56 and NJOY cannot read it; see
# test_the_cross_section_sums_hold_on_a_realisation.py.
perturbFromModel = partial(model_perturbation.perturbFromModel,
                           resonanceRegion="evaluated")

DATA = Path(__file__).resolve().parents[2] / "endf" / "tests" / "data"
FE56_XS = DATA / "micro_fe56_xs_and_angular.endf"
ZA = 26056
MAT_MF = "263133"


def _line(fields, mt):
    return "".join(fields) + f"{MAT_MF}{mt:3d}     \n"


def _int(value):
    return f"{value:11d}"


def _cont(mt, c1, c2, l1, l2, n1, n2):
    return _line([c1, c2, _int(l1), _int(l2), _int(n1), _int(n2)], mt)


ZERO = " 0.000000+0"


def _ncDerived(mt, terms, lo=" 1.000000-5", hi=" 2.000000+7"):
    """One NC LTY=0 sub-subsection: an ENDF-6 §33.2 CONT and its LIST."""
    values = []
    for coefficient, target in terms:
        values += [coefficient, target]
    lines = [_cont(mt, ZERO, ZERO, 0, 0, 0, 0),
             _cont(mt, lo, hi, 0, 0, len(values), len(terms))]
    for start in range(0, len(values), 6):
        row = values[start:start + 6]
        lines.append(_line(row + [" " * 11] * (6 - len(row)), mt))
    return lines


def _mf33Lines():
    lines = FE56_XS.read_text(encoding="utf-8").splitlines(keepends=True)
    section = [i for i, line in enumerate(lines)
               if line[70:72] == "33" and line[72:75] == "  2"]
    return lines, section[0], section[-1]


def _write(tmp_path, name, lines):
    from kika.endf.writers.update_directory import update_mf1_directory

    out = tmp_path / name
    out.write_text("".join(lines), encoding="utf-8", newline="")
    update_mf1_directory(str(out))
    return out


def _send(mt):
    return _cont(mt, ZERO, ZERO, 0, 0, 0, 0).replace(f"{MAT_MF}{mt:3d}",
                                                     f"{MAT_MF}  0")


@pytest.fixture
def derivedTotal(tmp_path):
    """MF33 MT1 stated only as MT2 + MT102, beside MT2's own block.

    B-VIII.1 H-1's and U-235's MT1 in miniature.
    """
    lines, first, last = _mf33Lines()
    head = lines[first]
    mt1 = ([head[:33] + _int(0) + _int(0) + _int(1) + f"{MAT_MF}  1     \n",
            _cont(1, ZERO, ZERO, 0, 1, 1, 0)]
           + _ncDerived(1, [(" 1.000000+0", " 2.000000+0"),
                            (" 1.000000+0", " 1.020000+2")])
           + [_send(1)])
    return _write(tmp_path, "fe56_derived_total.endf",
                  lines[:first] + mt1 + lines[first:])


@pytest.fixture
def remainderBesideMatrix(tmp_path):
    """MT1 with a block, MT2 with its own NI and, above 2.5 MeV, MT1 - MT102.

    B-VIII.1 Pu-239's MT2 in miniature: a ``Mixed`` of a matrix and a ``Sum``.
    """
    lines, first, last = _mf33Lines()
    block = lines[first:last + 1]
    mt1 = [line[:72] + "  1" + line[75:] for line in block]
    sub = mt1[1]
    mt1[1] = sub[:33] + _int(1) + sub[44:]
    mt1.append(_send(1))
    mt2 = list(block)
    sub = mt2[1]
    mt2[1] = sub[:44] + _int(1) + sub[55:]
    mt2[2:2] = _ncDerived(2, [(" 1.000000+0", " 1.000000+0"),
                              ("-1.000000+0", " 1.020000+2")],
                          lo=" 2.500000+6", hi=" 2.000000+7")
    return _write(tmp_path, "fe56_remainder.endf",
                  lines[:first] + mt1 + mt2 + lines[last + 1:])


def _covariances(path):
    return decodeCovarianceSuite(read_endf(str(path)))[0]


def test_a_derived_total_contributes_no_block_and_no_claim(derivedTotal):
    covariances = _covariances(derivedTotal)
    entries = _cross_section_entries(covariances, mf=33)
    assert {(row[1], col[1]) for row, col, *_ in entries} == {(2, 2)}
    assert _statedCrossSectionMTs(covariances) == {2}


def test_a_derived_total_does_not_stop_the_pre_flight(derivedTotal):
    for onMissing in ("skip", "raise"):
        run = perturbFromModel(str(derivedTotal), {33: None}, 1, seed=3,
                               dryRun=True, onMissing=onMissing)
        assert {c.mt for c in run.samples[0]["set"].components()} == {2}
        assert any("re-derived from their moved partials: MT1" in note
                   for note in run.notes), run.notes


def test_naming_a_derived_total_is_its_partials(derivedTotal):
    run = perturbFromModel(str(derivedTotal), {33: [1]}, 1, seed=3, dryRun=True,
                           onMissing="raise")
    assert {c.mt for c in run.samples[0]["set"].components()} == {2}
    assert any("MT1 was asked for and states no covariance of its own" in note
               for note in run.notes), run.notes


def test_a_remainder_beside_a_matrix_draws_the_matrix_and_takes_the_difference(
        remainderBesideMatrix):
    covariances = _covariances(remainderBesideMatrix)
    entries = _cross_section_entries(covariances, mf=33)
    assert {(row[1], col[1]) for row, col, *_ in entries} == {(1, 1), (2, 2)}

    run = perturbFromModel(str(remainderBesideMatrix), {33: None}, 1, seed=3,
                           dryRun=True, onMissing="skip")
    sample = run.samples[0]
    assert {c.mt for c in sample["set"].components()} == {1, 2}
    assert sample["applied"][ComponentKey(ZA, 33, 2)]["remainder_of"] == 1
    assert any("MT1's own covariance is drawn although the file decomposes it"
               in note for note in run.notes), run.notes


class _Log:
    def warning(self, *args, **kwargs):
        pass


def test_a_lump_is_stated_through_its_members():
    """MT851 has no MF3 section; its members do, so its block is kept."""
    from kika.endf.model_adapter import decodeReactionSuite
    from kika.sampling.model_perturbation import _dropUnstatedReactions

    suite, _ = decodeReactionSuite(read_endf(str(FE56_XS)))
    lump = ComponentKey(ZA, 33, 851)
    entries = [(lump, lump, None, None, None)]
    skipped = []
    kept = _dropUnstatedReactions(suite, entries, "raise", skipped, _Log(),
                                  lumped={851: (2, 102)})
    assert kept == entries and not skipped
    with pytest.raises(ValueError, match="MT851"):
        _dropUnstatedReactions(suite, entries, "raise", skipped, _Log())


#: What each tape needs beyond the app's defaults, and what it must show.
#: U-235's MT102 states a relative covariance of -1.056, where log space's
#: ln(1 + C) does not exist. O-16's 44 reactions on one pooled grid are a
#: 68922-row matrix (35 GiB) under grouping="mf".
B81_TAPES = {
    "u235_b81_tape": ({"space": "linear"}, {851, 852}),
    "u238_b81_tape": ({}, {851, 852}),
    "pu239_b81_tape": ({}, {1, 2}),
    "o16_b81_tape": ({"grouping": "stated"}, {2}),
}


@pytest.mark.slow
@pytest.mark.parametrize("tape", sorted(B81_TAPES))
def test_endf_b81_mf33_passes_the_pre_flight(tape, request):
    """The four tapes roadmap item 9 names, as the app asks for them."""
    options, drawn = B81_TAPES[tape]
    path = request.getfixturevalue(tape)
    run = model_perturbation.perturbFromModel(str(path), {33: None}, 1, seed=1,
                                              dryRun=True, onMissing="skip",
                                              **options)
    components = {c.mt for c in run.samples[0]["set"].components()}
    assert drawn <= components, components
    assert not any("not perturbed" in note for note in run.notes), run.notes
