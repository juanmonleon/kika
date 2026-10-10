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
so AWI and NSUB could not be derived back). G3 (MF4's LTT/LI/LCT/NM,
``derive/products.py``) closed the products' ``<node>`` row the same day; on
131 whole tapes sampled from the three libraries it derives LTT, LI and LCT
exactly. G4a (the nu-bars, MF1/458) and G4b (MF5: tables, the five §18.3 laws,
weighted sums and MT455's families) closed their rows next: every MF5 block of
the cuts, kept bytes included, is derived as the file states it.
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
    ("FissionEnergyRelease", "headerFields.replaced"):
        "MF1/458 LFC=1: the thermal value and σ the LIST repeats beside a "
        "tabulated term are not in the model (FUDGE writes σ=0)",
    ("Reaction", "headerFields.mf6"):
        "G4c derives every MF6 of the neutron cuts; what is left is what the "
        "model does not hold: LAW=5 (charged-particle elastic, refused by name) "
        "and the LAW=4 recoil of it, a particle the evaluation gives two masses "
        "(alpha AWP beside the He-4 target's AWR; C-12's MT5 residual) where "
        "PoPs holds one, ND>0 (Be-9 MT701's discrete photon point), and "
        "U-235's MT18 at JP=11 with its P(nu) subsections (no model node, J8)",
    ("Reaction", "headerFields.mf12"):
        "E5e derives MF12 from the model; what differs is what the model does "
        "not hold: an LO=2 section's own bytes and SHA-256 (the derived one is "
        "rebuilt from PoPs, in FUDGE's decreasing order and with its LP=0), and "
        "LP of a line with ES != 0 (FUDGE's rule writes 1; Li-7 and N-14 state 0)",
    ("Reaction", "headerFields.mf13"):
        "E5e derives MF13; the model holds the multiplicity on MF13's grid "
        "united with sigma's (decision J5), so the derived sigma_gamma is on that "
        "grid and not on the file's own, and LP follows FUDGE's rule as in MF12",
    ("ReactionSuite", "headerFields.photonsVerbatim"):
        "MF12 LO=2 cascades whose ES_i name no level of the cut (Fe-56 without "
        "MT51) and MF13 on an MT with no MF3 (N-14 MT28/32): kept as bytes",
    ("Product", "headerFields.nm"):
        "the libraries' NM disagrees with their data (JEFF-4.0 Fe-56 states 31 "
        "beside a 32nd-order row; 74 of 131 sampled tapes state 0); derived is "
        "the highest order, ENDF-102's and FUDGE's",
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
    ("Product", "headerFields.ltt"):
        "Cf-252's cut keeps MF5/MT18 and not its MF4: the decoder gives the "
        "neutron an isotropic angular half, which derives an MF4 the cut lacks",
    ("Product", "headerFields.li"): "see ltt",
    ("Product", "headerFields.lct"): "see ltt",
    ("ReactionSuite", "headerFields.awi"):
        "alpha on He-4, d on H-2: projectile and target share one id, and one "
        "PoPs entry carries one mass (AWR's, not AWI's)",
    # -- tapes that are not whole evaluations --------------------------------
    ("refused", "MF1/451 needs EMAX"):
        "covariance-only cuts (MF1 + MF31-35): no MF3, no energy domain; G6",
    ("refused", "no ENDF MAT for the target 'unknown'"):
        "synthetic covariance tapes with no MF1/451",
}


#: Kinds that are a cut's artefact may occur on that cut only: anywhere else
#: they would be a deriver's error, not the named exception.
ONLY_ON = {
    ("Product", "headerFields.ltt"): {"micro_cf252_pfns"},
    ("Product", "headerFields.li"): {"micro_cf252_pfns"},
    ("Product", "headerFields.lct"): {"micro_cf252_pfns"},
    # U-238 cut to MT2 keeps the fissile LFI=1 with no fission reaction left.
    ("ReactionSuite", "headerFields.lfi"): {"micro_cm243_photons",
                                            "micro_u238_ltt2_mf34",
                                            "micro_u238_mf34_l0"},
    ("ReactionSuite", "headerFields.awi"): {"micro_a_he4_mf6", "micro_d_h2_mf6"},
    ("Reaction", "headerFields.mf6"): {"micro_a_he4_mf6", "micro_be9_mf6",
                                       "micro_c12_mf6", "micro_d_h2_mf6",
                                       "micro_h3_he4_mf6", "micro_p_he3_mf6",
                                       "micro_t_li7_mf6", "micro_u235_mf6",
                                       "micro_u235_photons"},
    ("FissionEnergyRelease", "headerFields.replaced"): {"micro_u235_fission_energy"},
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


def test_a_cuts_artefact_occurs_on_that_cut_only(table):
    stray = {key: [w for w in table[key] if w.split(":")[0] not in tapes]
             for key, tapes in ONLY_ON.items() if key in table}
    assert not {k: v for k, v in stray.items() if v}, stray


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
