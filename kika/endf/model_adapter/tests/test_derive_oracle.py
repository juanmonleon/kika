"""G0 as a test: strip every micro-tape's provenance, derive it back, compare.

The oracle (``derive/oracle.py``) existed only as a command-line table, so the
figures quoted when G0-G2 landed were reproduced by nothing. This file pins the
table instead: **every kind of mismatch that remains is listed in ``PENDING``
with the phase that closes it**, and the test fails both ways --

* a mismatch of a kind not in the table is a regression of G1/G2 (or a new
  deriver that is wrong), and
* a kind in the table that no longer occurs means a phase landed and forgot to
  shrink the table, which is how a gate quietly stops measuring.

The phases are those of ``docs/library/gnds_to_endf_plan.md`` and
``endf_coverage_remaining_plan.md``. Measured 2026-10-09 on ``develop`` with the
projectile fix of this change (an alpha tape used to decode as neutron-incident,
so AWI and NSUB could not be derived back).
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import pytest

from kika.endf.model_adapter import decodeReactionSuite
from kika.endf.model_adapter.derive.oracle import COMPARED, oracle
from kika.endf.read_endf import read_endf

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
TAPES = sorted(p for p in DATA.glob("micro_*.endf") if "tsl" not in p.name)

#: (node kind, field) -> why it is still different. A refusal is keyed by the
#: start of its message.
PENDING = {
    # -- the files no deriver writes yet --------------------------------------
    ("Product", "<node>"): "G3 (MF4) and G4b/G4c (MF5, MF6)",
    ("Multiplicity", "<node>"): "G4a (MF1/452-456)",
    ("DelayedNeutrons", "<node>"): "G4a (MF1/455)",
    ("FissionEnergyRelease", "<node>"): "G4a (MF1/458)",
    ("Reaction", "headerFields.mf6"): "G4c (MF6)",
    ("Reaction", "headerFields.mf12"): "E5e (photons from the model)",
    ("Reaction", "headerFields.mf13"): "E5e",
    ("Reaction", "headerFields.mf14"): "E5e",
    ("Reaction", "headerFields.mf15"): "E5e",
    ("ReactionSuite", "headerFields.photonsVerbatim"): "E5c (MF12 LO=2 kept as bytes)",
    ("Resonances", "<node>"): "G5 (MF2), outside this line (D5)",
    # -- not derivable, by nature --------------------------------------------
    ("ReactionSuite", "headerFields.tpid"):
        "the tape's first line belongs to no material; a derived tape gets "
        "the default one (T1)",
    ("ReactionSuite", "headerFields.lrp"):
        "the micro-tapes cut MF2 out and keep the header's LRP; on a whole "
        "tape LRP follows MF2, which is G5",
    ("ReactionSuite", "headerFields.lfi"):
        "Cm-243's cut keeps MT51-53 only, so nothing in it says fissionable",
    ("ReactionSuite", "headerFields.awi"):
        "alpha on He-4: projectile and target share the id He4, and one PoPs "
        "entry carries one mass (AWR's, not AWI's)",
    # -- tapes that are not whole evaluations --------------------------------
    ("refused", "MF1/451 needs EMAX"):
        "covariance-only cuts (MF1 + MF31-35): no MF3, no energy domain; G6",
    ("refused", "no ENDF MAT for the target 'unknown'"):
        "synthetic covariance tapes with no MF1/451",
}


def _key(mismatch):
    if mismatch.kind == "refused":
        for kind, start in PENDING:
            if kind == "refused" and str(mismatch.derived).startswith(start):
                return kind, start
        return "refused", str(mismatch.derived)
    return mismatch.kind, mismatch.field


@pytest.fixture(scope="module")
def table():
    found = defaultdict(list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in TAPES:
            suite, _ = decodeReactionSuite(read_endf(str(path)))
            mismatches, _ = oracle(suite)
            for mismatch in mismatches:
                found[_key(mismatch)].append(f"{path.stem}:{mismatch.path}")
    return found


def test_the_corpus_is_the_micro_tapes():
    assert len(TAPES) > 40, f"{len(TAPES)} micro-tapes; the table was measured on 42"


def test_every_remaining_mismatch_is_a_named_pending_phase(table):
    unexpected = {key: where[:3] for key, where in table.items() if key not in PENDING}
    assert not unexpected, f"mismatches no phase accounts for: {unexpected}"


def test_g1_and_g2_fields_derive_exactly(table):
    """MAT, ZA, AWR, QM, LR and interpolation regions are G1-G2's, and done."""
    wrong = {key: where[:3] for key, where in table.items() if key[1] in COMPARED}
    assert not wrong, wrong


def test_a_pending_entry_that_no_longer_occurs_is_removed(table):
    stale = sorted(set(PENDING) - set(table))
    assert not stale, (
        f"{stale} no longer occur: the phase that closed them has to take "
        f"them out of PENDING"
    )


def test_the_derivers_run_suite_first_whatever_was_imported_first():
    """The order used to be import order, and a test importing
    ``derive.reactions`` first gave every reaction ZA and AWR of ``None``."""
    from kika.endf.model_adapter.derive import _inRunOrder
    from kika.endf.model_adapter.derive import reactions, suite  # noqa: F401

    assert [name for name, _, _ in _inRunOrder()][:2] == ["suite", "reactions"]
